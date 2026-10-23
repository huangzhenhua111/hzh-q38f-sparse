import sys; sys.path.insert(0, '.')
from count_flops_q38f import count
def cfg(h, L, E=256, K=5):
    nk = max(2, round(0.8 * h / 64)); nv = 3 * nk; nq = 2 * max(1, round(1.2 * h / 64)); fe = max(8, round(h / 4 / 8) * 8)
    t = [("full_attention" if i % 4 == 3 else "linear_attention") for i in range(L)]
    return dict(num_hidden_layers=L, hidden_size=h, vocab_size=16384, num_experts=E, num_experts_per_tok=K, moe_intermediate_size=fe, shared_expert_intermediate_size=fe,
                num_attention_heads=nq, num_key_value_heads=2, head_dim=64, linear_num_key_heads=nk, linear_num_value_heads=nv, linear_key_head_dim=64, linear_value_head_dim=64,
                linear_conv_kernel_dim=4, layer_types=t)
def rows(c, **kw):   # M = 0.205 * N_total_nonvocab(主干, 不含 MTP/n-gram 表) / h, 取 1024 倍数
    r = count(c, 2048, 1, **kw); return int(round(0.205 * r["total_nonvocab_params"] / c["hidden_size"] / 1024) * 1024)
def tokens(c, ng=True, **kw):
    r = count(c, 2048, rows(c, **kw) if ng else 0, **kw); return round(3e17 / r["train_flops_per_token"] / 524288) * 524288
if __name__ == "__main__":
    E, K = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (256, 5)
    for h, L in [(160, 8), (192, 8), (256, 8), (320, 8), (384, 8), (384, 10), (448, 12), (512, 12), (576, 12)]:
        c = cfg(h, L, E, K); M = rows(c); r = count(c, 2048, M); T = tokens(c)
        print(f"{h}x{L} nk{c['linear_num_key_heads']} nv{c['linear_num_value_heads']} nq{c['num_attention_heads']} fe{c['moe_intermediate_size']} act={r['active_nonvocab_params']/1e6:.2f}M tot={r['total_nonvocab_params']/1e6:.1f}M ngram={r['ngram_params']/1e6:.1f}M(M={M}) mtp={r['mtp_params']/1e6:.1f}M all={r['total_params_all']/1e6:.0f}M tok={T/1e9:.3f}B fl/tok={r['train_flops_per_token']/1e6:.0f}M")
