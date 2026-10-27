"""Spatial multi-head self-attention adapter (experimental). P05-owned; attached INSIDE layer 10 (after the unchanged C2PSA) via
the C2PSAMHSA subclass so layer indices, concat sources and the 708 pretrained tensor keys are untouched.

  x (B,C,H,W) -> tokens (B,N,C) -> LayerNorm(C) + fixed 2D sin/cos PE -> qkv Linear(C, 3*dim) -> heads -> scaled dot-product attention
  -> Linear(dim, C) -> gamma * out + x      (gamma: learnable scalar, init 0.01 => near-identity at step zero)
Frozen design: C=512 (built pinned yolo26s layer 10), 20x20 tokens at 640 input, heads=8, dim=256 (head dim 32), dropout 0.0.
dim=256 (not C) keeps the added parameters near 5% of the model (full-width projections would add ~10.6%, over the 1.10x params limit).
Production path uses F.scaled_dot_product_attention; `reference()` is the transparent explicit softmax(QK^T/sqrt(d))V path for tests.
"""
from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import C2PSA


def sincos_2d(h: int, w: int, c: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Fixed 2D sin/cos positional encoding, shape (h*w, c); c must be divisible by 4. Deterministic, parameter-free."""
    if c % 4:
        raise ValueError("channels must be divisible by 4 for 2D sin/cos PE")
    q = c // 4
    freq = 1.0 / (10000 ** (torch.arange(q, dtype=torch.float32) / q))
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    ay, ax = ys.reshape(-1, 1) * freq, xs.reshape(-1, 1) * freq
    return torch.cat([ax.sin(), ax.cos(), ay.sin(), ay.cos()], 1).to(device=device, dtype=dtype)


class MHSA(nn.Module):
    def __init__(self, c: int, heads: int = 8, dim: int = 256, dropout: float = 0.0, gamma0: float = 0.01):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.c, self.heads, self.dim, self.hd, self.dropout = c, heads, dim, dim // heads, dropout
        self.norm = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * dim)
        self.proj = nn.Linear(dim, c)
        self.gamma = nn.Parameter(torch.tensor(float(gamma0)))
        init_mhsa_(self, torch.Generator().manual_seed(0))

    def _qkv(self, x):
        b, c, h, w = x.shape
        if c != self.c:
            raise ValueError(f"expected {self.c} channels, got {c}")
        t = x.flatten(2).transpose(1, 2)                                            # (B,N,C)
        t = self.norm(t) + sincos_2d(h, w, c, x.device, torch.float32).to(t.dtype)
        q, k, v = self.qkv(t).view(b, h * w, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)   # each (B,heads,N,hd)
        return q, k, v

    def _out(self, o, x):
        b, c, h, w = x.shape
        o = self.proj(o.transpose(1, 2).reshape(b, h * w, self.dim))
        return x + self.gamma * o.transpose(1, 2).reshape(b, c, h, w).to(x.dtype)

    def forward(self, x):
        q, k, v = self._qkv(x)
        return self._out(F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0), x)

    def reference(self, x):
        """Explicit attention (materialises N x N): softmax(QK^T / sqrt(hd)) V; half-precision inputs are upcast to float32, float32/64 stay native."""
        q, k, v = (t.float() if t.dtype in (torch.float16, torch.bfloat16) else t for t in self._qkv(x))
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.hd), dim=-1)
        return self._out((a @ v).to(x.dtype), x)


def mhsa_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(hashlib.sha256(f"mhsa|{seed}".encode()).hexdigest()[:15], 16))


def init_mhsa_(m: MHSA, g: torch.Generator):
    """Xavier-uniform Linear weights from an explicit generator, zero biases, LayerNorm identity, gamma = 0.01."""
    with torch.no_grad():
        for lin in (m.qkv, m.proj):
            a = math.sqrt(6.0 / (lin.in_features + lin.out_features))
            lin.weight.copy_((torch.rand(lin.weight.shape, generator=g) * 2 - 1) * a)
            lin.bias.zero_()
        m.norm.weight.fill_(1.0); m.norm.bias.zero_(); m.gamma.fill_(0.01)


class C2PSAMHSA(C2PSA):
    """Unchanged C2PSA followed by the MHSA adapter. State-dict keys of C2PSA are preserved; new keys live under `.mhsa.`."""
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5, heads: int = 8, dim: int = 256):
        super().__init__(c1, c2, n, e)
        self.mhsa = MHSA(c2, heads, dim)

    def forward(self, x):
        return self.mhsa(super().forward(x))
