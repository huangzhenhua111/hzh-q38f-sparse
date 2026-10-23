"""生成 registry 行与队列行: python3 gen_rows.py [h x L ...]  默认 6 点。输出两段 REGISTRY / QUEUE。"""
import sys; sys.path.insert(0, "."); from plan_q38f import cfg, rows, tokens
from count_flops_q38f import count
LR = {192: "4e-3", 256: "3.5e-3", 320: "3e-3"}; MB = {192: 16, 256: 16, 320: 16, 384: 8, 448: 8, 512: 4}
def row(h, L, nong=False):
    c = cfg(h, L); M = rows(c); r = count(c, 2048, 0 if nong else M); T = tokens(c, ng=not nong)
    name = f"q38f_{h}x{L}" + ("_nong" if nong else "")
    note = (f"Qwen3.8-Flash-Next 缩小版全套: 256专家选5+1共享 宽h/4 + N-gram({'无(对照点)' if nong else f'{M}行x2表,{2*M*h/1e6:.0f}M'}) + QSA(块64 预算512tok + indexer KL蒸馏) + 4路分支残差(r=h/8,Sinkhorn) + Muon + MTP1层(系数0.3,{r['mtp_params']/1e6:.1f}M不计激活) "
            f"bpe16k_v2 seq2048 单卡 lr{LR.get(h,'2.5e-3')}")
    reg = f"{name}\t4090a\t3e17\tQ38-FULL-16k-2048\t{r['active_nonvocab_params']/1e6:.2f}M/{r['total_nonvocab_params']/1e6:.1f}M\t{T}\t16k\t{note}"
    q = f"{h} {L} {LR.get(h,'2.5e-3')} {MB[h]} {0 if nong else M} 0 256 5"
    return reg, q, r, T
if __name__ == "__main__":
    pts = [tuple(map(int, a.split("x"))) for a in sys.argv[1:]] or [(192, 8), (256, 8), (320, 8), (384, 10), (448, 12), (512, 12)]
    for h, L in pts:
        reg, q, r, T = row(h, L); print("REGISTRY\t" + reg); print("QUEUE\t" + q); print(f"#  {h}x{L}: act={r['active_nonvocab_params']/1e6:.2f}M tot={r['total_nonvocab_params']/1e6:.1f}M all={r['total_params_all']/1e6:.0f}M tokens={T/1e9:.3f}B", file=sys.stderr)
