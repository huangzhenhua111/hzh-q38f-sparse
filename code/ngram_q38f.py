"""N-gram Embedding (Qwen3.8-Flash-Next 的 N-gram Embedding 缩小版, 哈希 bigram/trigram 表, 第 2 层输入注入)。
位置 t: bigram=(x[t-1],x[t]), trigram=(x[t-2],x[t-1],x[t]); 乘法哈希 ((a*P1)^(b*P2)^(c*P3)) 后做一次 xorshift 再 mod M, 两张表各 M 行 × h 维;
n-gram 不跨文档: 以 EOS 为文档起点, 位置 t 往前数到 t 所在文档起点(含 EOS 本身)之前的 token 一律换成 PAD(=V), 序列开头不足同理。
融合: e = RMSNorm(E2[i2] + E3[i3]); h <- h + sigmoid(W_g h + b_g) * e (W_g: h→h 逐通道门); 通过 hook 挂在 layers[0] 输出上(= 第 2 层输入)。"""
import torch, torch.nn as nn

P2 = (1000003, 998244353); P3 = (1000033, 998244353, 1000000007)
MIX = 2654435761


class NGramEmbedding(nn.Module):
    def __init__(self, hidden, vocab, rows, eos_id, gate_bias=-1.0, init_std=0.02, norm_init=0.02):
        super().__init__()
        self.hidden, self.vocab, self.rows, self.eos_id = hidden, vocab, rows, eos_id
        self.emb2 = nn.Embedding(rows, hidden); self.emb3 = nn.Embedding(rows, hidden)
        nn.init.normal_(self.emb2.weight, std=init_std); nn.init.normal_(self.emb3.weight, std=init_std)
        self.norm = nn.RMSNorm(hidden, eps=1e-6); nn.init.constant_(self.norm.weight, norm_init)   # e 的 RMS 起步 ≈ 嵌入量级, 不压过残差流
        self.gate = nn.Linear(hidden, hidden, bias=True)
        nn.init.normal_(self.gate.weight, std=0.02); nn.init.constant_(self.gate.bias, gate_bias)
        self._ids = None; self.stat = {}

    def table_params(self): return self.emb2.weight.numel() + self.emb3.weight.numel()
    def gate_params(self): return sum(p.numel() for p in self.gate.parameters()) + self.norm.weight.numel()

    def indices(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).expand(B, T)
        last = torch.cummax(torch.where(x == self.eos_id, pos, torch.full_like(pos, -1)), dim=1).values.clamp(min=0)
        def prev(k):
            xs = torch.cat([torch.full((B, k), self.vocab, device=x.device, dtype=x.dtype), x[:, :-k]], dim=1)
            return torch.where((pos - k) >= last, xs, torch.full_like(xs, self.vocab))
        a, b = prev(1), x
        i2 = (a * P2[0]) ^ (b * P2[1])
        c = prev(2)
        i3 = (c * P3[0]) ^ (a * P3[1]) ^ (b * P3[2])
        i2 = (i2 ^ (i2 >> 17)) * MIX; i3 = (i3 ^ (i3 >> 17)) * MIX
        return (i2 % self.rows), (i3 % self.rows)

    def delta(self, h, x):
        """返回要加到残差流上的增量 g*e (h 只用来算门; 4 路分支残差下传各路的均值)。"""
        return self.forward(h, x) - h

    def forward(self, h, x):
        i2, i3 = self.indices(x)
        e = self.norm(self.emb2(i2) + self.emb3(i3)).to(h.dtype)
        g = torch.sigmoid(self.gate(h))
        with torch.no_grad():
            self.stat = dict(gate_mean=g.float().mean().item(), gate_frac_lt01=(g < 0.1).float().mean().item(), gate_frac_gt09=(g > 0.9).float().mean().item(),
                             ratio=((g * e).float().norm() / h.float().norm().clamp(min=1e-9)).item()) if getattr(self, "collect", False) else self.stat
        return h + g * e
