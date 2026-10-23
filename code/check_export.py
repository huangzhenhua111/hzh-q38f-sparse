"""导出一致性: 载入 HF 目录 + extras.pt, 用与 eval_q38f.py 相同的包装对训练时 evaluate() 的同一批 val 序列算 loss, 应与训练日志 val 接近。"""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "."); from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig
from q38f_model import Q38F; from ngram_q38f import NGramEmbedding
d = Path(sys.argv[1]); dev = torch.device("cuda")
model = AutoModelForCausalLM.from_pretrained(d, config=AutoConfig.from_pretrained(d).get_text_config(), dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
c = json.load(open(d / "q38f_config.json")); ng = NGramEmbedding(c["hidden"], c["vocab"], c["ngram_rows"], c["eos_id"]) if c["ngram_rows"] else None
qm = Q38F(model, model.config, ngram=ng, mtp=False, qsa_budget_blocks=c["qsa_blocks"], use_mhc=bool(c["mhc"]), use_qsa=bool(c["qsa"]))
r = qm.load_state_dict(torch.load(d / "extras.pt"), strict=False); assert not r.unexpected_keys and all(k.startswith("lm.") for k in r.missing_keys), r
qm.to(dev).eval()
data = np.fromfile(sys.argv[2], dtype=np.uint16, count=int(sys.argv[3])) if False else np.memmap(sys.argv[2], dtype=np.uint16, mode="r")
eval_n = 2000000; ed = data[-eval_n:]; erng = np.random.default_rng(1234); S = 2048; tot = 0.0; n = 0
with torch.inference_mode():
    for _ in range(eval_n // S):
        ix = erng.integers(0, len(ed) - S - 1, 1)
        x = torch.from_numpy(np.stack([ed[i:i+S].astype(np.int64) for i in ix])).to(dev); y = torch.from_numpy(np.stack([ed[i+1:i+1+S].astype(np.int64) for i in ix])).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = qm(x)
        tot += F.cross_entropy(lg.view(-1, lg.size(-1)).float(), y.view(-1), reduction="sum").item(); n += y.numel()
print("export 路径 val loss", tot / n)
