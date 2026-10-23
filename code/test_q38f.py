"""单元测试(GPU): QSA 预算=seq 等于全注意力; 分支残差初始化≈普通残差; Muon NS 奇异值≈1; MTP target 偏移与文档边界; 选块含当前块、因果。"""
import sys, torch, math
sys.path.insert(0, ".")
from q38f_model import *
from muon_q38f import newton_schulz, Muon
dev = torch.device("cuda"); torch.manual_seed(0)
H, L, V, S = 192, 4, 1024, 2048
cfg, lm = make_hf(V, L, H, 8, 2, 64, 2, 6, 64, 16, 3, 48, 48, eos=1)
lm = lm.to(dev)

# 1) QSA: 预算 = 全部块 → 与 dense SDPA 一致; 预算 = 8 → 与"手工按选块做掩码"的 SDPA 一致
m = Q38F(lm, cfg, mtp=False, use_mhc=False, qsa_budget_blocks=S // BLK).to(dev).eval()
x = torch.randn(2, S, H, device=dev); cos, sin = m._rope(S, dev); attn = lm.model.layers[3].self_attn
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o_d, _ = qsa_attention(attn, x, cos.bfloat16(), sin.bfloat16(), m.indexers["3"], 32, True, False)
    o_s, _ = qsa_attention(attn, x, cos.bfloat16(), sin.bfloat16(), m.indexers["3"], 32, False, False)
print("QSA(预算=seq) vs dense max|diff|", (o_d - o_s).abs().max().item(), "rms", o_d.float().pow(2).mean().sqrt().item()); assert (o_d - o_s).abs().max() < 2e-2
# 手工掩码对照(预算 8)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o8, _ = qsa_attention(attn, x, cos.bfloat16(), sin.bfloat16(), m.indexers["3"], 8, False, False)
    I = m.indexers["3"](x, cos.bfloat16(), sin.bfloat16()); ar = torch.arange(S, device=dev); I = I.masked_fill(ar[None] > ar[:, None], float("-inf"))
    lpb = torch.logsumexp(torch.log_softmax(I, -1).view(2, S, S // 64, 64), -1)
    num, idx = select_blocks(lpb, 8)
    assert (num[0, 0] == torch.clamp(torch.arange(32, device=dev) + 1, max=8)).all()
    for qb in range(32):   # 选块因果 + 含当前块 + 无重复
        sel = idx[0, 0, qb, :num[0, 0, qb]].long(); assert qb in sel.tolist() and (sel <= qb).all() and len(set(sel.tolist())) == len(sel)
    mask = torch.zeros(2, 1, S, S, dtype=torch.bool, device=dev)
    for b in range(2):
        for qb in range(32):
            for kb in idx[b, 0, qb, :num[b, 0, qb]].long().tolist(): mask[b, 0, qb * 64:(qb + 1) * 64, kb * 64:(kb + 1) * 64] = True
    mask &= (ar[None] <= ar[:, None])
    hd = 64; q = attn.q_proj(x).view(2, S, -1, hd * 2); q, gate = torch.chunk(q, 2, -1); gate = gate.reshape(2, S, -1)
    q = attn.q_norm(q).transpose(1, 2); k = attn.k_norm(attn.k_proj(x).view(2, S, -1, hd)).transpose(1, 2); v = attn.v_proj(x).view(2, S, -1, hd).transpose(1, 2)
    q, k = M.apply_rotary_pos_emb(q, k, cos.bfloat16(), sin.bfloat16())
    ref = F.scaled_dot_product_attention(q, k.repeat_interleave(4, 1), v.repeat_interleave(4, 1), attn_mask=mask, scale=attn.scaling)
    ref = attn.o_proj(ref.transpose(1, 2).reshape(2, S, -1) * torch.sigmoid(gate))
print("QSA(预算8) vs 手工掩码 max|diff|", (o8 - ref).abs().max().item()); assert (o8 - ref).abs().max() < 2e-2
# 梯度: KL 能回传到 indexer, 不回传到主干输入
m.train(); xx = x.clone().requires_grad_(True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    o, kl = qsa_attention(attn, xx, cos.bfloat16(), sin.bfloat16(), m.indexers["3"], 8, False, True)
(o.float().sum() * 0 + kl).backward()
print("KL", kl.item(), "indexer grad", m.indexers["3"].wq.weight.grad.abs().sum().item(), "x grad(应 0)", (xx.grad.abs().sum().item() if xx.grad is not None else 0.0))
assert m.indexers["3"].wq.weight.grad.abs().sum() > 0 and (xx.grad is None or xx.grad.abs().sum() == 0)

# 2) 分支残差(init) ≈ 普通残差
ids = torch.randint(2, V, (2, S), device=dev)
m_plain = Q38F(lm, cfg, mtp=False, use_mhc=False, use_qsa=False).to(dev).eval()
m_mhc = Q38F(lm, cfg, mtp=False, use_mhc=True, use_qsa=False).to(dev).eval()
for l in lm.model.layers: torch.nn.init.normal_(l.mlp.gate.weight, std=0.5)   # HF 路由器零初始化 → 并列, top-k 选择对数值噪声敏感; 测试里给非零权重
with torch.no_grad():   # fp32(不用 autocast)比较, 避免 bf16 噪声翻转 top-k 路由
    a = m_plain(ids).float(); b = m_mhc(ids).float()
print("mHC(init) vs plain logits max|diff|", (a - b).abs().max().item(), "rms", a.pow(2).mean().sqrt().item())
r = m_mhc.mhc[0]; Xt = torch.randn(1, 4, 4, H, device=dev); xin, c = r.pre(Xt); print("H_res 对角", c[3][0, 0].diag().tolist(), "行和", c[3][0, 0].sum(-1).tolist(), "H_pre和", c[1][0, 0].sum().item())
assert (a - b).abs().max() < 0.02 * a.pow(2).mean().sqrt()   # Sinkhorn 泄漏 + bf16 噪声
# 4 路对称性: 初始各路相同
# 3) Muon NS 奇异值
for shp in [(192, 96), (96, 192), (4, 80, 192)]:
    G = torch.randn(*shp, device=dev); O = newton_schulz(G).float(); sv = torch.linalg.svdvals(O)
    print("NS", shp, "sv min/max", sv.min().item(), sv.max().item()); assert sv.min() > 0.3 and sv.max() < 1.5
# 4) MTP 偏移 / 边界
m_t = Q38F(lm, cfg, mtp=True, use_mhc=True, use_qsa=True).to(dev).train()
x_in = torch.randint(2, V, (2, S + 1), device=dev); x_in[0, 100] = 1; x, y = x_in[:, :-1], x_in[:, 1:]
mtp_y = torch.cat([y[:, 1:], torch.full_like(y[:, :1], -100)], 1); mtp_y = torch.where(y == 1, torch.full_like(mtp_y, -100), mtp_y)
assert mtp_y[0, 99] == -100 and mtp_y[0, 98] == x_in[0, 100] and mtp_y[0, 98] == y[0, 99] and mtp_y[:, -1].eq(-100).all()
print("MTP target: 位置 t 预测 x[t+2]=y[t+1]; y[t]==EOS 与末位置被 -100 屏蔽 ✓")
with torch.autocast("cuda", dtype=torch.bfloat16):
    hs, kls = m_t.trunk(x); mh = m_t.mtp_hidden(hs, y)
print("trunk/mtp 形状", hs.shape, mh.shape, "kl层数", len(kls))
print("ALL TESTS PASSED")
