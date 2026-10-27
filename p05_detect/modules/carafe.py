"""CARAFE x2 content-aware upsampling. Operator: Wang et al., "CARAFE: Content-Aware ReAssembly of
FEatures", ICCV 2019. P05-owned pure-PyTorch implementation (no CUDA extension); AGPL-3.0 with Ultralytics integration.

  comp  1x1 conv C->c_mid(64);  enc 3x3 conv c_mid -> scale^2 * k_up^2 (=100);  PixelShuffle(2) -> (B,25,2H,2W);
  softmax over the 25 taps (float32);  out[b,c,2h+p,2w+q] = sum_k w[b,k,2h+p,2w+q] * x_pad[b,c, tap_k(h,w)]  (zero padding)
Memory: the 5x5 neighbourhood is unfolded ONCE at low resolution and contracted with einsum against the kernels viewed as
(B,25,H,2,W,2): the scratch probe's 4x nearest-repeat of the unfolded tensor is never materialised. Optional row-chunking with
activation checkpointing (`chunk_rows`) further bounds autograd storage; it is mathematically identical (tested).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class CARAFE(nn.Module):
    def __init__(self, c1: int, k_up: int = 5, k_enc: int = 3, c_mid: int = 64, scale: int = 2):
        super().__init__()
        if not (isinstance(c1, int) and c1 > 0):
            raise ValueError("c1 must be a positive int")
        if k_up % 2 == 0 or k_up < 1 or k_enc % 2 == 0 or k_enc < 1:
            raise ValueError("k_up and k_enc must be odd positive")
        if scale != 2:
            raise ValueError("only scale=2 is supported")
        self.c1, self.k, self.scale, self.chunk_rows = c1, k_up, scale, 0
        # Finite-input validation forces a GPU sync per call and rejects Ultralytics' torch.empty() warmup tensors, so it is opt-in
        # (tests and trust boundaries enable it); shape/channel validation is always on.
        self.check_finite = False
        self.comp = nn.Conv2d(c1, c_mid, 1)
        self.enc = nn.Conv2d(c_mid, scale ** 2 * k_up ** 2, k_enc, padding=k_enc // 2)
        self.ps = nn.PixelShuffle(scale)
        init_carafe_(self, torch.Generator().manual_seed(0))      # deterministic default; campaigns overwrite with a paired generator

    def _check(self, x):
        if x.ndim != 4 or x.shape[1] != self.c1 or x.shape[0] < 1 or x.shape[2] < 1 or x.shape[3] < 1:
            raise ValueError(f"expected (B,{self.c1},H,W) with H,W>=1, got {tuple(x.shape)}")
        if self.check_finite and not torch.isfinite(x).all():
            raise ValueError("non-finite input")

    def _in(self, x):
        return x if torch.is_autocast_enabled(x.device.type) else x.to(self.comp.weight.dtype)

    def kernels(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised reassembly kernels (B,k*k,2H,2W) in float32; each output position's kernel sums to 1."""
        w = self.ps(self.enc(self.comp(self._in(x))))
        return F.softmax(w.float(), dim=1)

    def _reassemble(self, xp, w6, r0, r1):
        """xp: zero-padded input (B,C,H+4,W+4); w6: (B,25,H,2,W,2). Rows [r0,r1) of the source grid."""
        B, C = xp.shape[:2]
        pad = self.k // 2
        xu = F.unfold(xp[:, :, r0:r1 + 2 * pad, :].float(), self.k)                  # (B, C*25, (r1-r0)*W)
        xu = xu.view(B, C, self.k * self.k, r1 - r0, -1)
        return torch.einsum("bckhw,bkhpwq->bchpwq", xu, w6[:, :, r0:r1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        B, C, H, W = x.shape
        in_dtype = x.dtype
        w6 = self.kernels(x).view(B, self.k * self.k, H, 2, W, 2)
        pad = self.k // 2
        xp = F.pad(x, (pad, pad, pad, pad))
        rows = self.chunk_rows if (self.chunk_rows and self.chunk_rows < H) else H
        outs = []
        for r0 in range(0, H, rows):
            r1 = min(H, r0 + rows)
            if self.training and rows < H and torch.is_grad_enabled():
                outs.append(checkpoint(self._reassemble, xp, w6, r0, r1, use_reentrant=False))
            else:
                outs.append(self._reassemble(xp, w6, r0, r1))
        out = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]
        return out.reshape(B, C, 2 * H, 2 * W).to(in_dtype)


def carafe_reference(x: torch.Tensor, kernels: torch.Tensor, k: int = 5, scale: int = 2) -> torch.Tensor:
    """Naive loop reference of the CARAFE reassembly (slow; for equivalence tests only). x (B,C,H,W), kernels (B,k*k,2H,2W)."""
    B, C, H, W = x.shape
    xp = F.pad(x.float(), (k // 2,) * 4)
    out = torch.zeros(B, C, scale * H, scale * W, dtype=torch.float32, device=x.device)
    for i in range(scale * H):
        for j in range(scale * W):
            h, w = i // scale, j // scale
            patch = xp[:, :, h:h + k, w:w + k].reshape(B, C, k * k)
            out[:, :, i, j] = (patch * kernels[:, :, i, j].unsqueeze(1)).sum(-1)
    return out.to(x.dtype)


def init_carafe_(m: CARAFE, gen: torch.Generator) -> CARAFE:
    """Initialisation using an explicit generator (independent of global RNG): comp Kaiming-uniform(a=sqrt5), bias 0;
    enc ~ N(0, 1e-3), bias 0 except the four sub-pixel copies of the centre tap (channels 12*4+q) set to 8."""
    with torch.no_grad():
        fan_in = m.comp.in_channels
        bound = math.sqrt(6.0 / ((1 + 5) * fan_in))
        m.comp.weight.copy_((torch.rand(m.comp.weight.shape, generator=gen) * 2 - 1) * bound)
        m.comp.bias.zero_()
        m.enc.weight.copy_(torch.randn(m.enc.weight.shape, generator=gen) * 1e-3)
        m.enc.bias.zero_()
        centre = (m.k * m.k) // 2
        for q in range(m.scale ** 2):
            m.enc.bias[centre * m.scale ** 2 + q] = 8.0
    return m
