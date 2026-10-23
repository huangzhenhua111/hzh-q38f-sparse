"""微基准: 各部件开关下 fwd+bwd 每 micro-step 耗时 (用法: bench_q38f.py h L mb); 不含优化器。"""
import sys, time, torch
sys.path.insert(0, "."); from q38f_model import *; from ngram_q38f import NGramEmbedding; from plan_q38f import cfg as pc
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as Mm
h, L, mb = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); c = pc(h, L)
dev = torch.device("cuda"); V = 16384; S = 2048; ce_fn = LigerFusedLinearCrossEntropyLoss()
def run(name, mhc, qsa, mtp, ng, dense=False, n=6):
    cfg, lm = make_hf(V, L, h, c["num_attention_heads"], 2, 64, c["linear_num_key_heads"], c["linear_num_value_heads"], 64, 256, 5, c["moe_intermediate_size"], c["moe_intermediate_size"], eos=1)
    lm = lm.to(dev); ngm = NGramEmbedding(h, V, 64512, 1).to(dev) if ng else None
    m = Q38F(lm, cfg, ngram=ngm, mtp=mtp, use_mhc=mhc, use_qsa=qsa).to(dev).train(); m.collect_router = True; m.qsa_dense = dense
    x = torch.randint(2, V, (mb, S), device=dev); y = torch.randint(2, V, (mb, S), device=dev)
    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs, kls = m.trunk(x); W = lm.lm_head.weight; hh = lm.model.norm(hs); loss = ce_fn(W, hh.reshape(-1, h), y.reshape(-1))
            if mtp: mh = m.mtp_hidden(hs, y); loss = loss + 0.3 * ce_fn(W, mh.reshape(-1, h), y.reshape(-1))
            if kls: loss = loss + 0.05 * torch.stack(kls).mean()
            loss = loss + 0.001 * Mm.load_balancing_loss_func(m.take_router(), 256, 5)
        loss.backward()
    for _ in range(3): step()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): step()
    torch.cuda.synchronize(); dt = (time.time() - t) / n
    print(f"{name:28s} {dt*1000:7.0f} ms/micro  {mb*S/dt/1e3:6.1f}k tok/s  mem {torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
    del m, lm; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
import os
if os.environ.get("ONLY"): run("全套", 1, 1, 1, 1)
else:
    run("plain (无 mhc/qsa/mtp)", 0, 0, 0, 0)
    run("+ngram", 0, 0, 0, 1)
    run("+mtp", 0, 0, 1, 1)
    run("+qsa(dense,含KL)", 0, 1, 1, 1, dense=True)
    run("+qsa(sparse flex)", 0, 1, 1, 1)
    run("+mhc (全套)", 1, 1, 1, 1)
