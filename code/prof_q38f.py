import sys, torch; sys.argv = [sys.argv[0]] + sys.argv[1:]
import os; os.environ["ONLY"] = "1"
h, L, mb = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
src = open("bench_q38f.py").read().split("import os\nif os.environ")[0]
exec(src)
from torch.profiler import profile, ProfilerActivity
cfg_, lm = make_hf(V, L, h, c["num_attention_heads"], 2, 64, c["linear_num_key_heads"], c["linear_num_value_heads"], 64, 256, 5, c["moe_intermediate_size"], c["moe_intermediate_size"], eos=1)
lm = lm.to(dev); ngm = NGramEmbedding(h, V, 64512, 1).to(dev)
m = Q38F(lm, cfg_, ngram=ngm, mtp=True).to(dev).train(); m.collect_router = True
x = torch.randint(2, V, (mb, S), device=dev); y = torch.randint(2, V, (mb, S), device=dev)
def step():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hs, kls = m.trunk(x); W = lm.lm_head.weight; hh = lm.model.norm(hs); loss = ce_fn(W, hh.reshape(-1, h), y.reshape(-1))
        mh = m.mtp_hidden(hs, y); loss = loss + 0.3 * ce_fn(W, mh.reshape(-1, h), y.reshape(-1)) + 0.05 * torch.stack(kls).mean()
        loss = loss + 0.001 * Mm.load_balancing_loss_func(m.take_router(), 256, 5)
    loss.backward()
for _ in range(3): step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
    step(); torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=28, max_name_column_width=70))
