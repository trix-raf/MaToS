"""
model.py - Conv1D stem + small Transformer encoder over a fixed window of landmark features.

Written with plain ops (Linear, slice/concat, MatMul, Softmax, ReduceMean/Max, SiLU) and fixed
shapes, so the exported ONNX graph stays inside what OpenCV's DNN importer handles well.
`set_export_mode(model, True)` swaps the fused training kernels for the explicit versions
(same weights, same maths) right before export.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LN(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.eps, self.export_mode = eps, False

    def forward(self, x):
        if not self.export_mode:
            return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        xc = x - x.mean(-1, keepdim=True)
        var = (xc * xc).mean(-1, keepdim=True)
        return xc / torch.sqrt(var + self.eps) * self.weight + self.bias


class TemporalConv(nn.Module):
    """A 1-D convolution over time, written as 'concatenate k shifted copies -> Linear' (identical maths)."""

    def __init__(self, cin: int, cout: int, k: int, T: int):
        super().__init__()
        self.k, self.T, self.pad = k, T, k // 2
        self.lin = nn.Linear(cin * k, cout)

    def forward(self, x):  # [B,T,C]
        xp = F.pad(x, (0, 0, self.pad, self.pad))
        return self.lin(torch.cat([xp[:, i:i + self.T] for i in range(self.k)], dim=-1))


class Attention(nn.Module):
    def __init__(self, d: int, heads: int, T: int, drop: float):
        super().__init__()
        self.h, self.hd, self.T, self.drop = heads, d // heads, T, drop
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.export_mode = False

    def forward(self, x):
        T, H, hd = self.T, self.h, self.hd
        q = self.q(x).reshape(-1, T, H, hd).transpose(1, 2)
        k = self.k(x).reshape(-1, T, H, hd).transpose(1, 2)
        v = self.v(x).reshape(-1, T, H, hd).transpose(1, 2)
        if self.export_mode or not hasattr(F, "scaled_dot_product_attention"):
            a = torch.matmul(q * (hd ** -0.5), k.transpose(-2, -1)).softmax(-1)  # MatMul -> Softmax -> MatMul
            y = torch.matmul(a, v)
        else:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(-1, T, H * hd))


class Block(nn.Module):
    def __init__(self, d, heads, T, ff, drop):
        super().__init__()
        self.n1, self.att, self.n2 = LN(d), Attention(d, heads, T, drop), LN(d)
        self.fc1, self.fc2, self.drop = nn.Linear(d, ff * d), nn.Linear(ff * d, d), nn.Dropout(drop)

    def forward(self, x):
        x = x + self.drop(self.att(self.n1(x)))
        return x + self.drop(self.fc2(self.drop(F.silu(self.fc1(self.n2(x))))))


class SignEncoder(nn.Module):
    def __init__(self, in_dim: int, n_out: int, T: int = 64, d: int = 256, layers: int = 4, heads: int = 4,
                 ff: int = 4, drop: float = 0.1, in_drop: float = 0.1):
        super().__init__()
        self.T, self.d = T, d
        self.in_drop = nn.Dropout(in_drop)
        self.c1, self.n1 = TemporalConv(in_dim, d, 5, T), LN(d)
        self.c2, self.n2 = TemporalConv(d, d, 3, T), LN(d)
        self.pos = nn.Parameter(torch.zeros(1, T, d))
        self.blocks = nn.ModuleList([Block(d, heads, T, ff, drop) for _ in range(layers)])
        self.norm = LN(d)
        self.head_drop = nn.Dropout(drop)
        self.head = nn.Linear(3 * d, n_out)
        self.apply(self._init)
        nn.init.trunc_normal_(self.pos, std=0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x):  # [B,T,F] -> logits [B,n_out]
        x = F.silu(self.n1(self.c1(self.in_drop(x))))
        x = x + F.silu(self.n2(self.c2(x)))
        x = x + self.pos
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        c = x.narrow(1, self.T // 2, 1).reshape(-1, self.d)  # window centre = where the label's sign sits
        return self.head(self.head_drop(torch.cat([x.mean(1), x.amax(1), c], dim=-1)))


def set_export_mode(model: nn.Module, flag: bool):
    for m in model.modules():
        if hasattr(m, "export_mode"):
            m.export_mode = flag
    return model


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())