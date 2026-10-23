"""q38f(全套)的 FLOPs/参数核算, 在 count_flops_q35 口径上加(前向每 token, 训练 = 3×前向, 蒸馏项另计 1×):
  QSA 实际注意力: 4·nq·hd·(平均被关注 token 数); query 块 i 选 min(i+1, B) 个 64 块, 对角块按因果半块 → avg = 64·(sel−1)+32.5 对 query 块平均 (稠密时 → seq/2, 与 count_flops_q35 的 2·seq·nq·hd 同口径)
  indexer: 投影记入参数(2N 项); 稠密打分 2·seq·n_ih·d_ih /token/QSA 层(全 seq×seq, 不因果对折), 反向 ×3
  蒸馏目标(detach 的稠密主注意力概率): 2·seq·nq·hd /token/QSA 层, 只有前向 → 训练 FLOPs 只记 1×
  分支残差: 每子层 A(n·h→r)+B(r→2n+n²) 记入参数; 聚合/混合/写回 2·h·(n+n²+n) /token/子层; Sinkhorn(20 次迭代 ~1.3k FLOPs/子层)忽略
  N-gram: 门矩阵 2·h² /token (同 v1), 查表不计
  MTP: 2·(2h·h 投影 + GDN + MoE 激活 + norm) + GDN 核 + 2·h·V 输出头, ×3; 参数单列, 不进激活参数
  Muon 的 Newton–Schulz、dense warmup(前 1% 步 QSA 层走全注意力)不计 (优化器开销/占比 <0.1%)"""
import argparse, json
from count_flops_q35 import count as count_q35, CHUNK

BLK = 64

def qsa_avg_attended(seq, budget_blocks):
    nb = seq // BLK
    return sum(BLK * (min(i + 1, budget_blocks) - 1) + (BLK + 1) / 2 for i in range(nb)) / nb

def count(c, seq, ngram_rows=0, mhc=True, qsa=True, mtp=True, budget_blocks=8, n_branch=4, n_ih=4, d_ih=64):
    r = count_q35(c, seq); c = c.get("text_config", c)
    h, L, V = r["hidden"], r["layers"], r["vocab"]; nq, hd = r["attn_heads"], r["head_dim"]
    types = r["layer_types"]; n_full = types.count("linear_attention") * 0 + types.count("full_attention")
    bd = dict(r["breakdown"]); fwd = r["fwd_flops_per_token"]; act, tot = r["active_nonvocab_params"], r["total_nonvocab_params"]
    ex = {}
    if qsa:
        avg = qsa_avg_attended(seq, budget_blocks)
        fwd += n_full * 4 * nq * hd * avg - bd["attn_score"]; bd["attn_score"] = n_full * 4 * nq * hd * avg
        ip = n_full * (h * n_ih * d_ih + h * d_ih + h * n_ih); act += ip; tot += ip; ex["indexer_params"] = ip
        bd["indexer_proj"] = 2 * ip; bd["indexer_score"] = n_full * 2 * seq * n_ih * d_ih; fwd += bd["indexer_score"]
    distill = n_full * 2 * seq * nq * hd if qsa else 0
    if mhc:
        rr = round(h / 8); mp = 2 * L * (n_branch * h * rr + rr * (2 * n_branch + n_branch ** 2) + 2 * n_branch + n_branch ** 2 + 3)
        act += mp; tot += mp; ex["mhc_params"] = mp; bd["mhc_proj"] = 2 * mp
        bd["mhc_mix"] = 2 * L * 2 * h * (2 * n_branch + n_branch ** 2); fwd += bd["mhc_mix"]
    if ngram_rows:
        gp = h * h + h + h; act += gp; tot += gp; bd["ngram_gate"] = 2 * h * h; fwd += 2 * h * h
    mtp_params = mtp_fwd = 0
    if mtp:
        E, k, fe, fs = r["experts"], r["topk"], r["expert_ffn"], r["shared_ffn"]
        nk, nv, dk, dv, K = r["lin_k_heads"], r["lin_v_heads"], c["linear_key_head_dim"], c["linear_value_head_dim"], c["linear_conv_kernel_dim"]
        moe_act = E * h + 3 * h * fs + h + k * 3 * h * fe; moe_tot = E * h + 3 * h * fs + h + E * 3 * h * fe
        gdn = h * (2 * nk * dk + nv * dv) + h * nv * dv + 2 * h * nv + (2 * nk * dk + nv * dv) * K + 2 * nv + dv + nv * dv * h
        proj = 2 * h * h; norms = 2 * h + 3 * h
        mtp_params = proj + gdn + moe_tot + norms; mtp_act = proj + gdn + moe_act + norms
        gdn_core = nv * (6 * dk * dv + 6 * CHUNK * dk + 4 * CHUNK * dv)
        mtp_fwd = 2 * mtp_act + gdn_core + 2 * h * V; bd["mtp_fwd"] = mtp_fwd
    fwd_total_main = fwd
    train = 3 * (fwd + mtp_fwd) + distill
    ex["mtp_params"] = mtp_params
    bd = {k: (3 * v if k not in ("mtp_fwd",) else 3 * v) for k, v in bd.items()}; bd["distill_target(1x)"] = distill
    r.update(active_nonvocab_params=act, total_nonvocab_params=tot, nonvocab_params=act, fwd_flops_per_token=fwd + mtp_fwd, train_flops_per_token=train,
             main_fwd_flops_per_token=fwd_total_main, breakdown_train=bd, ngram_rows=ngram_rows, ngram_params=2 * ngram_rows * h, **ex)
    r["total_params_all"] = tot + r["embed_params"] + r["ngram_params"] + mtp_params
    r.pop("breakdown", None); return r

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--tokens", type=float, default=0); ap.add_argument("--ngram_rows", type=int, default=0)
    ap.add_argument("--mhc", type=int, default=1); ap.add_argument("--qsa", type=int, default=1); ap.add_argument("--mtp", type=int, default=1); ap.add_argument("--budget_blocks", type=int, default=8)
    a = ap.parse_args(); c = json.load(open(a.model if a.model.endswith(".json") else f"{a.model}/config.json"))
    res = count(c, a.seq, a.ngram_rows, bool(a.mhc), bool(a.qsa), bool(a.mtp), a.budget_blocks)
    if a.tokens: res["train_tokens"] = a.tokens; res["train_flops"] = res["train_flops_per_token"] * a.tokens
    print(json.dumps(res, indent=1))
