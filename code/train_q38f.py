"""Qwen3.8-Flash-Next 缩小版"全套"(q38f)预训练闭环: 由 train_q38ng.py 改来。DDP/ckpt/resume/cosine/token 预算/HF 导出同 v1。
新增: 4 路分支残差(mHC) / QSA(块稀疏注意力 + indexer KL 蒸馏) / Muon(隐层矩阵) + AdamW(其余) / 1 层 MTP / N-gram(第 2 层输入注入, 加到每一路)。
用法见 run_one.sh。"""
import contextlib, argparse, os, time, math, json, csv, numpy as np, torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
from count_flops_q38f import count as count_flops
from ngram_q38f import NGramEmbedding
from q38f_model import Q38F, make_hf, QSA_STAT
from muon_q38f import Muon
ap = argparse.ArgumentParser()
for k, v in dict(run_dir="../runs/q38f", data="", tokenizer="", layers=8, hidden=192, heads=8, kv_heads=2, head_dim=64, lin_k_heads=2, lin_v_heads=6, lin_head_dim=64,
                 experts=256, topk=5, moe_ffn=48, shared_ffn=48, aux_coef=0.001, rope_theta=1e7, partial_rotary=0.25, experts_impl="grouped_mm",
                 seq=2048, tokens=300e6, batch_tokens=524288, micro_bs=8, lr=3e-3, min_lr_ratio=0.1, warmup=0, wd=0.1, clip=1.0, seed=0, ngram_rows=0, ngram_lr_mult=1.0,
                 mhc=1, qsa=1, mtp=1, mtp_coef=0.3, qsa_blocks=8, kl_coef=0.05, muon=1, muon_lr_mult=1.0, adam_only_lr_mult=1.0, dense_warmup=1,
                 log_every=10, eval_every=500, eval_tokens=2e6, gpu=0, ckpt_every=200, resume=1, max_steps=0).items():
    ap.add_argument(f"--{k}", type=type(v), default=v)
a = ap.parse_args(); os.makedirs(a.run_dir, exist_ok=True)
import torch.distributed as dist
DDP_ON = int(os.environ.get("WORLD_SIZE", "1")) > 1
if DDP_ON:
    dist.init_process_group(os.environ.get("Q35_DIST_BACKEND", "nccl")); RANK = dist.get_rank(); WORLD = dist.get_world_size(); a.gpu = int(os.environ["LOCAL_RANK"]) % torch.cuda.device_count()
else: RANK, WORLD = 0, 1
torch.manual_seed(a.seed + RANK)
dev = torch.device("cuda", a.gpu); torch.cuda.set_device(dev)
def log0(*x, **k):
    if RANK == 0: print(*x, flush=True)
tok = AutoTokenizer.from_pretrained(a.tokenizer)
cfg, lm = make_hf(len(tok), a.layers, a.hidden, a.heads, a.kv_heads, a.head_dim, a.lin_k_heads, a.lin_v_heads, a.lin_head_dim, a.experts, a.topk, a.moe_ffn, a.shared_ffn,
                  a.seq, a.aux_coef, a.rope_theta, a.partial_rotary, tok.eos_token_id, a.experts_impl)
lm = lm.to(dev, dtype=torch.float32); types = cfg.layer_types
assert lm.lm_head.weight.data_ptr() == lm.model.embed_tokens.weight.data_ptr(), "嵌入未绑定"
ng = NGramEmbedding(a.hidden, len(tok), a.ngram_rows, tok.eos_token_id).to(dev, dtype=torch.float32) if a.ngram_rows else None
model = Q38F(lm, cfg, ngram=ng, mtp=bool(a.mtp), qsa_budget_blocks=a.qsa_blocks, use_mhc=bool(a.mhc), use_qsa=bool(a.qsa)).to(dev, dtype=torch.float32)
model.collect_router = True
import fla  # noqa
_impl = [c.cell_contents.__module__ for c in (M.torch_chunk_gated_delta_rule.__closure__ or []) if callable(c.cell_contents) and c.cell_contents.__module__.startswith("fla")]
log0("GDN kernel:", _impl or "torch 慢路径!"); assert _impl, "chunk_gated_delta_rule 没走 fla"
ex = model.extra_param_counts(); embed_n = lm.model.embed_tokens.weight.numel()
trunk_n = sum(p.numel() for n, p in lm.named_parameters()) - embed_n          # HF 部分(不含嵌入); lm_head 与嵌入绑定不重复计
n_total_nv = trunk_n + ex["mhc"] + ex["indexer"] + ex["ngram_gate"]
n_expert_each = 3 * a.hidden * a.moe_ffn; n_active_nv = n_total_nv - a.layers * (a.experts - a.topk) * n_expert_each
FC = count_flops(cfg.to_dict(), a.seq, a.ngram_rows, bool(a.mhc), bool(a.qsa), bool(a.mtp), a.qsa_blocks); flops_per_token = FC["train_flops_per_token"]
assert FC["total_nonvocab_params"] == n_total_nv and FC["active_nonvocab_params"] == n_active_nv, (FC["total_nonvocab_params"], n_total_nv, FC["active_nonvocab_params"], n_active_nv)
if a.mtp: assert FC["mtp_params"] == ex["mtp"], (FC["mtp_params"], ex["mtp"])
n_all = n_total_nv + embed_n + ex["ngram_table"] + ex["mtp"]
log0(f"层型 {types}\n参数 总(含词表+n-gram表+MTP)={n_all/1e6:.2f}M | n-gram表={ex['ngram_table']/1e6:.2f}M(每表 {a.ngram_rows} 行) MTP={ex['mtp']/1e6:.2f}M mHC={ex['mhc']/1e6:.2f}M indexer={ex['indexer']/1e6:.3f}M | 非词表 总={n_total_nv/1e6:.2f}M 激活={n_active_nv/1e6:.3f}M"
     f" | 每token训练 {flops_per_token/1e6:.1f} MFLOPs {({k: round(v/1e6,1) for k, v in FC['breakdown_train'].items()})} | 预算 {a.tokens:.3e} token ≈ {flops_per_token*a.tokens:.3e} FLOPs")
DT = np.uint32 if len(tok) > 65535 else np.uint16
data = np.fromfile(a.data, dtype=DT) if os.path.getsize(a.data) < 64e9 else np.memmap(a.data, dtype=DT, mode="r")
n_tok_data = len(data); log0(f"数据 {n_tok_data/1e9:.2f}B token 已载入内存 (world={WORLD})")
eval_n = int(a.eval_tokens); train_data = data[:-eval_n]; eval_data = data[-eval_n:]
per_micro = a.micro_bs * a.seq * WORLD
assert a.batch_tokens % per_micro == 0, f"batch_tokens {a.batch_tokens} 不能被 micro_bs×seq×world={per_micro} 整除"
steps = int(a.tokens // a.batch_tokens); accum = a.batch_tokens // per_micro; rng = np.random.default_rng(a.seed + RANK)
if a.max_steps: steps = min(steps, a.max_steps)
if not a.warmup: a.warmup = max(10, round(steps * 0.01))
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
EOS = tok.eos_token_id
class LossModel(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m; self.ce = LigerFusedLinearCrossEntropyLoss()
    def forward(self, x, y):
        m = self.m; hs, kls = m.trunk(x); W = m.lm.lm_head.weight
        h = m.lm.model.norm(hs); ce = self.ce(W, h.reshape(-1, h.size(-1)), y.reshape(-1))
        if m.mtp is not None:   # 位置 t 的 MTP 预测 x[t+2]=y[t+1]; 末位置、以及 y[t] 是 EOS(下一个 token 属于新文档)的位置不计
            ty = torch.cat([y[:, 1:], torch.full_like(y[:, :1], -100)], 1); ty = torch.where(y == EOS, torch.full_like(ty, -100), ty)
            mh = m.mtp_hidden(hs, y); mce = self.ce(W, mh.reshape(-1, mh.size(-1)), ty.reshape(-1))
        else: mce = ce.new_zeros(())
        kl = torch.stack(kls).mean() if kls else ce.new_zeros(())
        r = m.take_router(); aux = M.load_balancing_loss_func(r, a.experts, a.topk)
        return ce, mce, aux, kl
lm_loss = LossModel(model)
if DDP_ON: lm_loss = torch.nn.parallel.DistributedDataParallel(lm_loss, device_ids=[a.gpu], find_unused_parameters=True)
def batch(src, bs):
    ix = rng.integers(0, len(src) - a.seq - 1, bs)
    x = torch.from_numpy(np.stack([src[i:i+a.seq].astype(np.int64) for i in ix])); y = torch.from_numpy(np.stack([src[i+1:i+1+a.seq].astype(np.int64) for i in ix]))
    return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
# ---- 优化器分组: Muon 吃隐层矩阵(含 3 维专家张量, 逐片正交化); 其余 AdamW
MUON_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj",
             "experts.gate_up_proj", "experts.down_proj", "mhc.", ".A.weight", ".B.weight", "indexers.", "mtp.proj", "ngram.gate.weight")
ng_tab = {id(ng.emb2.weight), id(ng.emb3.weight)} if ng else set()
muon_p, adam_dec, adam_nodec, tabs = [], [], [], []; muon_names = []
for n, p in model.named_parameters():
    nn_ = n.replace("lm.", "", 1) if n.startswith("lm.") else n
    if n.startswith("ng."): nn_ = n.replace("ng.", "ngram.", 1)
    if id(p) in ng_tab: tabs.append(p); continue
    is_mat = (p.dim() >= 2) and any(k in nn_ for k in MUON_KEYS) and "embed" not in nn_ and not (nn_.startswith("mhc.") and not (nn_.endswith(".A.weight") or nn_.endswith(".B.weight")))
    if a.muon and is_mat: muon_p.append(p); muon_names.append(nn_)
    elif p.dim() >= 2 and not nn_.startswith("mhc."): adam_dec.append(p)
    else: adam_nodec.append(p)
groups = [{"params": adam_dec, "weight_decay": a.wd, "lr_mult": a.adam_only_lr_mult}, {"params": adam_nodec, "weight_decay": 0.0, "lr_mult": a.adam_only_lr_mult}]
if tabs: groups.append({"params": tabs, "weight_decay": 0.0, "lr_mult": a.ngram_lr_mult})
opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
opt_m = Muon([{"params": muon_p, "lr_mult": a.muon_lr_mult}], lr=a.lr, momentum=0.95, weight_decay=a.wd) if muon_p else None
for g in opt_m.param_groups if opt_m else []: g["lr_mult"] = a.muon_lr_mult
log0(f"优化器: Muon {len(muon_p)} 个张量/{sum(p.numel() for p in muon_p)/1e6:.2f}M 参数; AdamW {sum(p.numel() for g in groups for p in g['params'])/1e6:.2f}M 参数 (含词表嵌入 {embed_n/1e6:.2f}M, n-gram 表 {ex['ngram_table']/1e6:.2f}M)")
all_params = [p for p in model.parameters()]
def lr_at(s):
    if s < a.warmup: return a.lr * (s + 1) / a.warmup
    pr = (s - a.warmup) / max(1, steps - a.warmup); return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * pr)))
@torch.no_grad()
def evaluate():
    model.eval(); tot = 0.0; n = 0
    erng = np.random.default_rng(1234)
    for _ in range(eval_n // a.seq):
        ix = erng.integers(0, len(eval_data) - a.seq - 1, 1)
        x = torch.from_numpy(np.stack([eval_data[i:i+a.seq].astype(np.int64) for i in ix])).to(dev); y = torch.from_numpy(np.stack([eval_data[i+1:i+1+a.seq].astype(np.int64) for i in ix])).to(dev)
        model.collect_router = False
        with torch.autocast("cuda", dtype=torch.bfloat16): logits = model(x)
        fl = logits.view(-1, logits.size(-1)); yl = y.view(-1); CH = max(256, (8192 * 32768) // logits.size(-1))
        for o in range(0, yl.numel(), CH): tot += torch.nn.functional.cross_entropy(fl[o:o+CH].float(), yl[o:o+CH], reduction="sum").item()
        n += y.numel()
    model.train(); model.collect_router = True; model._router = []; return tot / n
@torch.no_grad()
def mhc_stats():
    if not a.mhc: return ""
    out = []
    for j in (0, len(model.mhc) // 2, len(model.mhc) - 1):
        m = model.mhc[j]; res = torch.softmax(m.b_res.float(), -1)   # 仅静态部分的近似显示; 动态部分幅度 α·tanh ≤ α
        from q38f_model import _sinkhorn
        R = _sinkhorn(m.b_res.float(), m.iters); out.append(f"s{j}:diag{R.diag().mean().item():.3f}/α{m.alphas.abs().max().item():.2f}")
    return " | mHC " + " ".join(out)
if RANK == 0: json.dump({**vars(a), "n_params": n_all, "n_nonvocab_total": n_total_nv, "n_nonvocab_active": n_active_nv, "n_ngram_params": ex["ngram_table"], "n_mtp_params": ex["mtp"],
                         "flops_per_token": flops_per_token, "flops_breakdown_train": FC["breakdown_train"], "steps": steps, "accum": accum, "config": cfg.to_dict()}, open(f"{a.run_dir}/run_config.json", "w"), indent=1, ensure_ascii=False)
CKPT = f"{a.run_dir}/ckpt.pt"; resuming = a.resume and os.path.exists(CKPT)
log = csv.writer(open(f"{a.run_dir}/training_metrics.csv", "a" if resuming else "w", newline="")) if RANK == 0 else None
if log and not resuming: log.writerow(["step", "tokens", "loss", "mtp_loss", "kl", "aux", "lr", "grad_norm", "tok_per_s", "elapsed_s", "val_loss", "active_nonvocab", "total_nonvocab"])
model.train(); t0 = time.time(); t_last = t0; tok_seen = 0; start_step = 0
if resuming:
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); (opt_m.load_state_dict(ck["opt_m"]) if opt_m else None); start_step = ck["step"] + 1; tok_seen = ck["tok_seen"]
    if RANK == 0: rng.bit_generator.state = ck["rng"]
    else: rng = np.random.default_rng([a.seed + RANK, start_step])
    log0(f"从 {CKPT} 恢复：step {start_step}, tok_seen {tok_seen}"); del ck
def save_ckpt(step):
    if RANK != 0: return
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "opt_m": opt_m.state_dict() if opt_m else None, "step": step, "tok_seen": tok_seen, "rng": rng.bit_generator.state}, CKPT + ".tmp"); os.replace(CKPT + ".tmp", CKPT)
log0(f"steps {steps} accum {accum} micro_bs {a.micro_bs} warmup {a.warmup} lr {a.lr} (dense warmup 前 {a.warmup if a.dense_warmup else 0} 步 QSA 层全注意力)")
for step in range(start_step, steps):
    for g in opt.param_groups: g["lr"] = lr_at(step) * g["lr_mult"]
    for g in (opt_m.param_groups if opt_m else []): g["lr"] = lr_at(step) * g["lr_mult"]
    model.qsa_dense = bool(a.dense_warmup and step < a.warmup)
    loss_acc = mtp_acc = kl_acc = aux_acc = 0.0; QSA_STAT.clear()
    for _mi in range(accum):
        x, y = batch(train_data, a.micro_bs)
        QSA_STAT['on'] = (_mi == accum - 1) and (step % a.log_every == 0 or step == steps - 1)
        if ng: ng.collect = (_mi == accum - 1) and (step % a.log_every == 0 or step == steps - 1)
        _ctx = lm_loss.no_sync() if (DDP_ON and _mi < accum - 1) else contextlib.nullcontext()
        with _ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            ce, mce, aux, kl = lm_loss(x, y)
            loss = (ce + a.mtp_coef * mce + a.aux_coef * aux + a.kl_coef * kl) / accum
        loss.backward(); loss_acc += ce.item() / accum; mtp_acc += float(mce) / accum; kl_acc += float(kl) / accum; aux_acc += float(aux) / accum
    gn = torch.nn.utils.clip_grad_norm_(all_params, a.clip).item(); opt.step(); (opt_m.step() if opt_m else None); opt.zero_grad(set_to_none=True); (opt_m.zero_grad(set_to_none=True) if opt_m else None); tok_seen += accum * per_micro
    if DDP_ON:
        la = torch.tensor(loss_acc, device=dev); dist.all_reduce(la); loss_acc = la.item() / WORLD
    if a.ckpt_every and step > 0 and step % a.ckpt_every == 0: save_ckpt(step)
    if (step % a.log_every == 0 or step == steps - 1) and RANK == 0:
        now = time.time(); tps = min(a.log_every, step - start_step + 1) * accum * per_micro / max(1e-6, now - t_last); t_last = now
        vl = evaluate() if (step % a.eval_every == 0 and step > 0) or step == steps - 1 else ""
        log.writerow([step, tok_seen, f"{loss_acc:.4f}", f"{mtp_acc:.4f}", f"{kl_acc:.4f}", f"{aux_acc:.4f}", f"{lr_at(step):.2e}", f"{gn:.2f}", f"{tps:.0f}", f"{now-t0:.0f}", f"{vl:.4f}" if vl != "" else "", n_active_nv, n_total_nv])
        print(f"step {step}/{steps} tok {tok_seen/1e6:.0f}M loss {loss_acc:.4f} mtp {mtp_acc:.3f} kl {kl_acc:.4f} aux {aux_acc:.3f} lr {lr_at(step):.2e} gn {gn:.2f} {tps:.0f} tok/s {(now-t0)/60:.1f}min mem {torch.cuda.max_memory_allocated(dev)/2**30:.1f}G"
              + (f" | val {vl:.4f}" if vl != "" else "") + (f" | ng gate {ng.stat['gate_mean']:.3f} <0.1:{ng.stat['gate_frac_lt01']:.2f} >0.9:{ng.stat['gate_frac_gt09']:.2f} |ge|/|h| {ng.stat['ratio']:.3f}" if ng and ng.stat else "") + (f" | QSA recall {np.mean(QSA_STAT['recall']):.3f}" if QSA_STAT.get("recall") else "") + mhc_stats(), flush=True)
        if not math.isfinite(loss_acc): raise SystemExit("loss 非有限, 终止")
if RANK == 0:
    lm.to(torch.bfloat16).save_pretrained(f"{a.run_dir}/hf"); tok.save_pretrained(f"{a.run_dir}/hf")
    ext = {k: v.float() for k, v in model.state_dict().items() if not k.startswith("lm.") and not k.startswith("mtp.") and k != "inv_freq"}
    torch.save(ext, f"{a.run_dir}/hf/extras.pt")
    json.dump({"hidden": a.hidden, "layers": a.layers, "vocab": len(tok), "ngram_rows": a.ngram_rows, "eos_id": tok.eos_token_id, "mhc": a.mhc, "qsa": a.qsa, "qsa_blocks": a.qsa_blocks, "mtp": 0}, open(f"{a.run_dir}/hf/q38f_config.json", "w"))
    json.dump({"tokens_trained": tok_seen, "train_flops": flops_per_token * tok_seen, "active_nonvocab": n_active_nv, "total_nonvocab": n_total_nv, "ngram_params": ex["ngram_table"], "mtp_params": ex["mtp"], "elapsed_s": time.time() - t0, "world": WORLD},
              open(f"{a.run_dir}/summary.json", "w"), indent=1)
    print("DONE 已导出 HF 目录", f"{a.run_dir}/hf", flush=True)
if DDP_ON: dist.barrier(); dist.destroy_process_group()
