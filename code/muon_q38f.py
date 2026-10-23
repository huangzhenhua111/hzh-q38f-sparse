"""Muon (Keller Jordan 的 Newton–Schulz 正交化 + Moonlight 的 update-RMS 对齐)。
- Nesterov 动量 0.95; NS 5 步, 系数 (3.4445, -4.7750, 2.0315); 3 维张量(MoE 专家 [E, A, B])按第 0 维逐片正交化。
- update ×= 0.2·sqrt(max(A,B)) 对齐 AdamW 的 update RMS(≈0.2), 因此可沿用 AdamW 的 lr 与解耦 weight decay。
- NS 的 FLOPs 属于优化器开销, 不计入模型 FLOPs。梯度已在 DDP 里同步后才调用 step()。"""
import torch

NS_COEF = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def newton_schulz(G, steps=5, eps=1e-7):
    """G: [..., A, B] → 近似正交化(奇异值≈1), 在 bf16 里算。"""
    a, b, c = NS_COEF
    X = G.to(torch.bfloat16)
    tr = X.size(-2) > X.size(-1)
    if tr: X = X.transpose(-2, -1)
    X = X / (X.flatten(-2).norm(dim=-1).clamp(min=eps)[..., None, None])
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        Bm = b * A + c * (A @ A)
        X = a * X + Bm @ X
    if tr: X = X.transpose(-2, -1)
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.1, ns_steps=5, rms_scale=0.2):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps, rms_scale=rms_scale))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                st = self.state[p]
                if "buf" not in st: st["buf"] = torch.zeros_like(p)
                buf = st["buf"]; buf.mul_(g["momentum"]).add_(p.grad)
                u = p.grad.add(buf, alpha=g["momentum"])            # Nesterov
                o = newton_schulz(u, g["ns_steps"]).float()
                o.mul_(g["rms_scale"] * max(p.shape[-2], p.shape[-1]) ** 0.5)
                if g["weight_decay"]: p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(o, alpha=-g["lr"])
