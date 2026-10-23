"""Qwen3.8-Flash-Next 缩小版"全套"(v2): 在 HF Qwen3_5MoeForCausalLM 的子模块上自己写前向。
部件: 4 路分支残差(mHC: H_pre/H_post/H_res + Sinkhorn) / QSA(块稀疏注意力 + lightning indexer, flex_attention) / N-gram Embedding / 1 层 MTP。
实现假设全文见 q38f/RESULT.md "实现假设"(收尾时写); 关键点在各类的 docstring 里。"""
import copy, math, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, BlockMask
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M

BLK = 64
import torch._inductor.kernel.flex_attention as _fa
_fa._get_default_config_bwd = lambda query: (32, 64, 4, 3)
_fa._get_default_config_fwd = lambda query: (64, 64, 4, 3)    # torch 2.7.1 的默认前向 tile 是 128, 与稀疏块 64 不整除; 改成 (BLOCK_M, BLOCK_N, warps, stages)=(64,64,4,3)
_flex = torch.compile(flex_attention, dynamic=False)
_KO = {}


def _causal(b, h, q, kv): return q >= kv


# ---------------------------------------------------------------- 分支残差 (Hyper-Connections / mHC)
def _sinkhorn(logits, iters=20):
    P = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))
    for _ in range(iters):
        P = P / P.sum(-1, keepdim=True).clamp(min=1e-9); P = P / P.sum(-2, keepdim=True).clamp(min=1e-9)
    return P


def _coeffs(Xf, A, B, b_pre, b_post, b_res, alphas, n, iters):
    """Xf: [B,S,n*h] 已 RMSNorm。dyn = tanh(B(A(Xf))) → (pre[B,S,n], post[B,S,n], res[B,S,n,n])。"""
    z = torch.tanh(F.linear(F.linear(Xf, A), B).float())
    zp, zq, zr = z[..., :n], z[..., n:2 * n], z[..., 2 * n:].reshape(*z.shape[:-1], n, n)
    pre = torch.sigmoid(b_pre + alphas[0] * zp)
    post = 2 * torch.sigmoid(b_post + alphas[1] * zq)
    res = _sinkhorn(b_res + alphas[2] * zr, iters)
    return pre, post, res


def _agg(X, pre): return (pre.unsqueeze(-1) * X).sum(2)          # X [B,S,n,h] → [B,S,h]


def _mix(X, res, post, y):                                        # X' = H_res X + H_post^T y
    return torch.einsum("bsmn,bsnh->bsmh", res, X) + post.unsqueeze(-1) * y.float().unsqueeze(2)


_agg_c = torch.compile(_agg); _mix_c = torch.compile(_mix); _coeffs_c = torch.compile(_coeffs)


class MHC(nn.Module):
    """一个子层(注意力/GDN 或 MoE)的分支残差系数。静态参数 + 动态部分(RMSNorm(flatten 4 路) → 低秩 r → 24 个系数, tanh 限幅)。
    初始化 = 普通残差: 动态部分输出 0(B 零初始化); H_res≈I(对角 0/非对角 −8 过 Sinkhorn); H_pre 和为 1(本层"主读"的那路 0.7、其余 0.1);
    H_post 全 1(每路都写子层输出)。embedding 复制 4 路时前向与普通残差一致(差异来自 Sinkhorn 的 ~1e-3 泄漏)。"""
    def __init__(self, h, n=4, r=None, slot=0, iters=20):
        super().__init__()
        self.n, self.iters = n, iters; r = r or round(h / 8)
        self.A = nn.Linear(n * h, r, bias=False); self.B = nn.Linear(r, 2 * n + n * n, bias=False)
        nn.init.normal_(self.A.weight, std=0.02); nn.init.zeros_(self.B.weight)
        pre = torch.full((n,), 0.1); pre[slot % n] = 0.7
        self.b_pre = nn.Parameter(torch.log(pre / (1 - pre))); self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(torch.full((n, n), -8.0) + 8.0 * torch.eye(n)); self.alphas = nn.Parameter(torch.full((3,), 0.1))

    def n_params(self): return sum(p.numel() for p in self.parameters())

    def pre(self, X):
        X = X.float(); Xf = X.flatten(2); Xf = Xf * torch.rsqrt(Xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        with torch.autocast("cuda", enabled=False):
            c = _coeffs_c(Xf, self.A.weight.float(), self.B.weight.float(), self.b_pre.float(), self.b_post.float(), self.b_res.float(), self.alphas.float(), self.n, self.iters)
        return _agg_c(X, c[0]), (X,) + c

    def post(self, X, y, c): return _mix_c(c[0], c[3], c[2], y)


# ---------------------------------------------------------------- QSA: lightning indexer + 块稀疏注意力
def _idx_score_fn(q, k, w, scale):
    s = torch.matmul(q, k.transpose(-1, -2))                                  # [B,nh,S,S] bf16
    return torch.einsum("bhts,bth->bts", F.relu(s.float()) * scale, w)


_idx_score = torch.compile(_idx_score_fn)


class Indexer(nn.Module):
    """DSA 风格: MQA, n_ih 个 q 头 + 1 个共享 k 头, head_dim d_ih; 打分 I[t,s] = Σ_j w_j(t)·ReLU(q_j(t)·k(s)) (RoPE 同主干的 partial rope)。
    只靠 KL 蒸馏训练(输入 detach, 不回传主干)。"""
    def __init__(self, h, n_ih=4, d_ih=64):
        super().__init__()
        self.n_ih, self.d_ih = n_ih, d_ih
        self.wq = nn.Linear(h, n_ih * d_ih, bias=False); self.wk = nn.Linear(h, d_ih, bias=False); self.ww = nn.Linear(h, n_ih, bias=False)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.n_ih, self.d_ih).transpose(1, 2); k = self.wk(x).view(B, S, 1, self.d_ih).transpose(1, 2)
        q, k = M.apply_rotary_pos_emb(q, k, cos, sin)                          # q [B,nh,S,d]  k [B,1,S,d]
        w = self.ww(x).float() * (self.n_ih ** -0.5)                            # [B,S,nh]
        return _idx_score(q, k, w, self.d_ih ** -0.5)                           # [B,S,S]


@torch.no_grad()
def block_mass(q, k, scale, chunk=256):
    """主注意力(稠密, 因果)在每个 64 块上的概率质量, 头平均: q [B,nq,S,d], k [B,nkv,S,d] → [B,S,nb] (fp32, detach)。"""
    B, nq, S, d = q.shape; nkv = k.shape[1]; rep = nq // nkv; nb = S // BLK
    qg = q.view(B, nkv, rep, S, d); out = []
    kk = k[:, :, None]
    cols = torch.arange(S, device=q.device)
    for t0 in range(0, S, chunk):
        sc = torch.matmul(qg[:, :, :, t0:t0 + chunk], kk.transpose(-1, -2)).float() * scale          # [B,nkv,rep,c,S]
        rows = torch.arange(t0, min(S, t0 + chunk), device=q.device)
        sc = sc.masked_fill(cols[None, :] > rows[:, None], float("-inf"))
        p = torch.softmax(sc, dim=-1).view(B, nkv, rep, -1, nb, BLK).sum(-1)
        out.append(p.mean(dim=(1, 2)))
    return torch.cat(out, dim=1)


def select_blocks(lp_blk, budget_blocks):
    """lp_blk: [B,S,nb] indexer 在块上的 log 概率(−inf=非因果)。按 query 块(64 个 token)平均概率 → 每个 query 块 top-k 块, 总含当前块。
    返回 BlockMask 需要的 (kv_num_blocks [B,1,nbq], kv_indices [B,1,nbq,k]) (int32)。"""
    B, S, nb = lp_blk.shape; nbq = S // BLK; k = min(budget_blocks, nb)
    p = lp_blk.detach().exp().view(B, nbq, BLK, nb).mean(2)
    qb = torch.arange(nbq, device=p.device)[:, None]; kb = torch.arange(nb, device=p.device)[None, :]
    p = torch.where(kb > qb, torch.full_like(p, -1.0), p)
    p = torch.where(kb == qb, torch.full_like(p, 2.0), p)
    idx = p.topk(k, dim=-1).indices
    num = torch.clamp(qb[:, 0] + 1, max=k).to(torch.int32)[None].expand(B, -1)
    full = torch.zeros(B, 1, nbq, nb, dtype=torch.int32, device=p.device); full[..., :k] = idx[:, None].to(torch.int32)   # from_kv_blocks 要求最后一维 = 块总数(尾部不读)
    return num[:, None].contiguous(), full


QSA_STAT = {}


def qsa_attention(attn, x, cos, sin, indexer, budget_blocks, dense, want_kl):
    """用 attn(Qwen3_5MoeAttention 的权重) 做注意力; dense=True 走全注意力(预热/测试), 否则 indexer 选块 + flex_attention。
    返回 (输出, kl 或 None)。KL = mean_t Σ_blk P_main(blk|t)·(log P_main − log P_idx), P_main 为稠密主注意力(头平均, detach)。"""
    B, S, _ = x.shape; hd = attn.head_dim
    q, gate = torch.chunk(attn.q_proj(x).view(B, S, -1, hd * 2), 2, dim=-1); gate = gate.reshape(B, S, -1)
    q = attn.q_norm(q).transpose(1, 2); k = attn.k_norm(attn.k_proj(x).view(B, S, -1, hd)).transpose(1, 2); v = attn.v_proj(x).view(B, S, -1, hd).transpose(1, 2)
    q, k = M.apply_rotary_pos_emb(q, k, cos, sin)
    kl = None; lp_blk = None
    if indexer is not None and (want_kl or not dense):
        I = indexer(x.detach(), cos, sin)
        ar = torch.arange(S, device=x.device)
        I = I.masked_fill(ar[None, :] > ar[:, None], -1e9)    # 有限大负数而非 -inf: 避免 logsumexp/log_softmax 反传出 NaN
        lp = torch.log_softmax(I, dim=-1)
        lp_blk = torch.logsumexp(lp.view(B, S, S // BLK, BLK), dim=-1)                  # [B,S,nb]
        if want_kl:
            P = block_mass(q.detach(), k.detach(), attn.scaling)
            kl = torch.where(P > 0, P * (torch.log(P.clamp(min=1e-12)) - lp_blk), torch.zeros_like(P)).sum(-1).mean()
            if QSA_STAT.get("on") and not dense:
              with torch.no_grad():   # 统计: 稠密主注意力概率质量落在 indexer 选中块里的比例(recall), 对照"随机选同样多块"的期望
                  num_, idx_ = select_blocks(lp_blk, budget_blocks); nbq_, nb_ = S // BLK, S // BLK
                  sel = torch.zeros(B, nbq_, nb_, device=x.device).scatter_(-1, idx_[:, 0, :, :min(budget_blocks, nb_)].long(), 1.0)
                  sel = sel * (torch.arange(nb_, device=x.device)[None, :] <= torch.arange(nbq_, device=x.device)[:, None])
                  rec = (P.view(B, nbq_, BLK, nb_) * sel[:, :, None, :]).sum(-1).mean()
                  QSA_STAT.setdefault("recall", []).append(rec.item())
    if dense or indexer is None:
        o = F.scaled_dot_product_attention(q, k, v, scale=attn.scaling, is_causal=True, enable_gqa=True)
    else:
        num, idx = select_blocks(lp_blk, budget_blocks)
        bm = BlockMask.from_kv_blocks(num, idx, BLOCK_SIZE=BLK, mask_mod=_causal, seq_lengths=(S, S))
        o = _flex(q, k, v, scale=attn.scaling, block_mask=bm, enable_gqa=True)
    o = o.transpose(1, 2).reshape(B, S, -1) * torch.sigmoid(gate)
    return attn.o_proj(o), kl


# ---------------------------------------------------------------- 整体模型
class Q38F(nn.Module):
    def __init__(self, lm, cfg, n_branch=4, ngram=None, mtp=True, qsa_budget_blocks=8, n_ih=4, d_ih=64, mhc_iters=20, use_mhc=True, use_qsa=True):
        super().__init__()
        self.lm, self.cfg, self.n, self.ng = lm, cfg, n_branch, ngram
        self.use_mhc, self.use_qsa, self.budget = use_mhc, use_qsa, qsa_budget_blocks
        L, h = cfg.num_hidden_layers, cfg.hidden_size
        self.types = list(cfg.layer_types[:L])
        if use_mhc: self.mhc = nn.ModuleList([MHC(h, n_branch, round(h / 8), slot=j, iters=mhc_iters) for j in range(2 * L)])
        if use_qsa: self.indexers = nn.ModuleDict({str(i): Indexer(h, n_ih, d_ih) for i, t in enumerate(self.types) if t == "full_attention"})
        self.mtp = None
        if mtp:
            c2 = copy.deepcopy(cfg); c2.layer_types = list(cfg.layer_types[:L]) + ["linear_attention"]
            blk = M.Qwen3_5MoeDecoderLayer(c2, L)
            self.mtp = nn.ModuleDict(dict(enorm=M.Qwen3_5MoeRMSNorm(h, eps=cfg.rms_norm_eps), hnorm=M.Qwen3_5MoeRMSNorm(h, eps=cfg.rms_norm_eps),
                                          proj=nn.Linear(2 * h, h, bias=False), block=blk, norm=M.Qwen3_5MoeRMSNorm(h, eps=cfg.rms_norm_eps)))
            nn.init.normal_(self.mtp["proj"].weight, std=cfg.initializer_range)
            blk.apply(lm._init_weights)
        self.qsa_dense = False; self.collect_router = False; self._router = []
        for lay in list(lm.model.layers) + ([self.mtp["block"]] if self.mtp else []):
            lay.mlp.gate.register_forward_hook(lambda m, i, o: self._router.append(o[0]) if self.collect_router else None)
        inv = lm.model.rotary_emb.inv_freq
        self.register_buffer("inv_freq", inv.clone(), persistent=False)

    # --- 参数口径
    def extra_param_counts(self):
        d = dict(mhc=sum(m.n_params() for m in self.mhc) if self.use_mhc else 0,
                 indexer=sum(p.numel() for m in self.indexers.values() for p in m.parameters()) if self.use_qsa else 0,
                 ngram_gate=self.ng.gate_params() if self.ng else 0, ngram_table=self.ng.table_params() if self.ng else 0,
                 mtp=sum(p.numel() for p in self.mtp.parameters()) if self.mtp else 0)
        return d

    def _rope(self, S, dev):
        f = torch.arange(S, device=dev, dtype=torch.float32)[:, None] * self.inv_freq.float().to(dev)[None]
        f = torch.cat([f, f], dim=-1); return f.cos()[None], f.sin()[None]

    def trunk(self, ids):
        """→ (X 4 路求和后的最终隐状态 [B,S,h], kl 列表)。"""
        lm = self.lm.model; B, S = ids.shape; dev = ids.device
        cos, sin = self._rope(S, dev); cos = cos.to(torch.bfloat16 if torch.is_autocast_enabled() else lm.embed_tokens.weight.dtype); sin = sin.to(cos.dtype)
        e = lm.embed_tokens(ids)
        X = e.float().unsqueeze(2).expand(-1, -1, self.n, -1).contiguous() if self.use_mhc else e
        kls = []
        for i, layer in enumerate(lm.layers):
            if self.use_mhc: xin, c = self.mhc[2 * i].pre(X)
            else: xin = X
            a = layer.input_layernorm(xin)
            if self.types[i] == "linear_attention": y = layer.linear_attn(hidden_states=a, cache_params=None, attention_mask=None)
            elif self.use_qsa:
                want = self.training and torch.is_grad_enabled()
                y, kl = qsa_attention(layer.self_attn, a, cos, sin, self.indexers[str(i)], self.budget, self.qsa_dense, want)
                if kl is not None: kls.append(kl)
            else:
                y, kl = qsa_attention(layer.self_attn, a, cos, sin, None, self.budget, True, False)
            X = self.mhc[2 * i].post(X, y, c) if self.use_mhc else X + y
            if self.use_mhc: xin, c = self.mhc[2 * i + 1].pre(X)
            else: xin = X
            y = layer.mlp(layer.post_attention_layernorm(xin))
            if isinstance(y, tuple): y = y[0]
            X = self.mhc[2 * i + 1].post(X, y, c) if self.use_mhc else X + y
            if i == 0 and self.ng is not None:      # N-gram: 第 2 层输入前注入; 门由 4 路均值算, 增量加到每一路
                hm = X.mean(2) if self.use_mhc else X
                d = self.ng.delta(hm, ids)
                X = X + (d.unsqueeze(2) if self.use_mhc else d)
        return (X.sum(2) if self.use_mhc else X), kls

    def forward(self, ids):
        hs, _ = self.trunk(ids); return self.lm.lm_head(self.lm.model.norm(hs))

    def mtp_hidden(self, hs, ids_next):
        """DeepSeek-V3 式 MTP: hs=主干最终隐状态(4 路求和后, 未过 final norm), ids_next=x_{t+1}。→ MTP 块输出(过 MTP 自己的 norm)。"""
        m = self.mtp
        z = m["proj"](torch.cat([m["enorm"](self.lm.model.embed_tokens(ids_next)), m["hnorm"](hs)], dim=-1))
        z = m["block"](z, position_embeddings=None, attention_mask=None)
        return m["norm"](z)

    def take_router(self):
        r = tuple(self._router); self._router = []; return r


def make_hf(vocab, layers, hidden, heads, kv_heads, head_dim, lin_k, lin_v, lin_hd, experts, topk, moe_ffn, shared_ffn, seq=2048, aux_coef=0.001,
            rope_theta=1e7, partial_rotary=0.25, eos=0, experts_impl="grouped_mm"):
    from transformers import Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM
    types = [("full_attention" if i % 4 == 3 else "linear_attention") for i in range(layers)]
    cfg = Qwen3_5MoeTextConfig(vocab_size=vocab, hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=head_dim,
                               linear_num_key_heads=lin_k, linear_num_value_heads=lin_v, linear_key_head_dim=lin_hd, linear_value_head_dim=lin_hd, linear_conv_kernel_dim=4,
                               num_experts=experts, num_experts_per_tok=topk, moe_intermediate_size=moe_ffn, shared_expert_intermediate_size=shared_ffn,
                               router_aux_loss_coef=aux_coef, output_router_logits=False, layer_types=types, max_position_embeddings=seq, rms_norm_eps=1e-6,
                               rope_parameters={"rope_type": "default", "rope_theta": rope_theta, "partial_rotary_factor": partial_rotary}, partial_rotary_factor=partial_rotary,
                               tie_word_embeddings=True, attention_bias=False, initializer_range=0.02, bos_token_id=eos, eos_token_id=eos, pad_token_id=eos)
    cfg._experts_implementation = experts_impl; cfg._attn_implementation = "sdpa"
    return cfg, Qwen3_5MoeForCausalLM(cfg)
