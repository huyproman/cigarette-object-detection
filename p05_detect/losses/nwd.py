"""NWD/CIoU hybrid box loss. Algorithm: Wang et al., "A Normalized Gaussian Wasserstein Distance for
Tiny Object Detection" (arXiv:2110.13389); hybrid use follows the YOLOv8-MNC idea. Implementation is P05-owned and
subclasses Ultralytics 8.4.174 (AGPL-3.0) classes; no upstream file is modified.

  L_box = (1-lambda) * L_CIoU + lambda * L_NWD,   NWD = exp(-sqrt(W2^2 + eps) / C)
  W2^2 = ||c1-c2||^2 + ||(w1/2,h1/2)-(w2/2,h2/2)||^2, boxes in NETWORK-INPUT PIXELS.
The L1/DFL term, assignment and Progressive Loss weighting are upstream, unchanged.
"""
from __future__ import annotations

import math

import torch
from ultralytics.utils.loss import BboxLoss, E2ELoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou

EPS = 1e-7


def _check_finite(*xs):
    for x in xs:
        if not torch.isfinite(x).all():
            raise ValueError("non-finite box coordinates")


def nwd_similarity(b1: torch.Tensor, b2: torch.Tensor, C: float, eps: float = EPS) -> torch.Tensor:
    """NWD between xyxy pixel boxes (...,4). Always computed in float32 (AMP safe)."""
    if not (isinstance(C, (int, float)) and math.isfinite(C) and C > 0):
        raise ValueError("C must be finite and > 0")
    if not (eps > 0):
        raise ValueError("eps must be > 0")
    b1, b2 = b1.float(), b2.float()
    _check_finite(b1, b2)
    c1, c2 = (b1[..., :2] + b1[..., 2:]) / 2, (b2[..., :2] + b2[..., 2:]) / 2
    s1, s2 = (b1[..., 2:] - b1[..., :2]).clamp_min(0) / 2, (b2[..., 2:] - b2[..., :2]).clamp_min(0) / 2
    w2 = ((c1 - c2) ** 2).sum(-1) + ((s1 - s2) ** 2).sum(-1)
    return torch.exp(-torch.sqrt(w2 + eps) / C)


def pixel_boxes(boxes: torch.Tensor, fg_mask: torch.Tensor, stride: torch.Tensor) -> torch.Tensor:
    """Select positives of (B,A,4) stride-unit boxes and scale to pixels with the per-ANCHOR stride (A,1).
    Uses (batch, anchor) indices: indexing stride by the flattened mask index is wrong."""
    bi, ai = fg_mask.nonzero(as_tuple=True)
    return boxes[bi, ai] * stride[ai]


class NWDHybridBboxLoss(BboxLoss):
    def __init__(self, reg_max: int = 1, lam: float = 0.5, C: float = 12.8):
        super().__init__(reg_max)
        if not (isinstance(lam, (int, float)) and math.isfinite(lam) and 0.0 <= lam <= 1.0):
            raise ValueError("lambda must be finite in [0,1]")
        if not (isinstance(C, (int, float)) and math.isfinite(C) and C > 0):
            raise ValueError("C must be finite and > 0")
        self.lam, self.C = float(lam), float(C)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride):
        # upstream computes CIoU + (DFL | L1); reg_max==1 path rescales pred_dist IN PLACE -> pass a clone (grads still flow)
        loss_ciou, loss_l1 = super().forward(pred_dist.clone() if self.dfl_loss is None else pred_dist, pred_bboxes, anchor_points,
                                             target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride)
        if self.lam == 0.0:
            return loss_ciou, loss_l1                               # exact upstream outputs
        bi, ai = fg_mask.nonzero(as_tuple=True)
        weight = target_scores[bi, ai].sum(-1)
        # target boxes carry no gradient (assignment output); predictions are scaled to pixels
        nwd = nwd_similarity(pred_bboxes[bi, ai] * stride[ai], target_bboxes[bi, ai].detach() * stride[ai], self.C)
        loss_nwd = ((1.0 - nwd) * weight).sum() / target_scores_sum
        return (1.0 - self.lam) * loss_ciou + self.lam * loss_nwd, loss_l1


class P05DetectionLoss(v8DetectionLoss):
    """v8DetectionLoss whose bbox_loss is the hybrid. Accepts/forwards tal_topk and tal_topk2 (E2ELoss builds both heads)."""
    nwd_lambda = 0.5
    nwd_C = 12.8

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.bbox_loss = NWDHybridBboxLoss(self.reg_max, type(self).nwd_lambda, type(self).nwd_C).to(self.device)


def make_p05_loss_class(lam: float, C: float):
    """Bind validated (lambda, C) into a loss class without global mutation."""
    NWDHybridBboxLoss(1, lam, C)
    return type("P05DetectionLossBound", (P05DetectionLoss,), {"nwd_lambda": float(lam), "nwd_C": float(C)})
