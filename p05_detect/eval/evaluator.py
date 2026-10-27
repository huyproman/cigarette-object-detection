"""P05 custom COCO-compatible detection evaluator. Pure numpy.

Matching (per IoU threshold, reset each time): predictions sorted by (-score, image_id, pred_index),
capped at max_det per image; greedy highest-IoU unmatched same-class valid GT; valid-GT duplicates
stay FP; only an otherwise-unmatched prediction with IoU>=0.5 to an ignore region and no valid-GT
overlap at the current threshold is neutralised. unknown_empty / missing_suspected images are
excluded wholly (GT and predictions). AP: 101-point COCO recall grid on the precision envelope.
This is CUSTOM_COCO_REPORTING_AP, NOT the upstream selection AP.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

COORD_SPACE = "original_pixel_xyxy"
IOUS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
EPS = 1e-12
EXCLUDED_STATUS = {"unknown_empty", "missing_suspected"}


@dataclass
class Image:
    image_id: str
    width: int
    height: int
    gt: list = field(default_factory=list)       # [(class_id, (x1,y1,x2,y2))]
    ignore: list = field(default_factory=list)   # [(class_id, box)]
    status: str = "positive"                     # positive | confirmed_negative | unknown_empty | missing_suspected


@dataclass
class Pred:
    image_id: str
    class_id: int
    score: float
    box: tuple
    index: int = 0
    space: str = COORD_SPACE


def validate_pred(p: Pred, num_classes: int, known_images) -> None:
    if p.space != COORD_SPACE:
        raise ValueError(f"coordinate space {p.space!r} != {COORD_SPACE}")
    if p.image_id not in known_images:
        raise ValueError(f"untraceable image id {p.image_id!r}")
    if not (isinstance(p.class_id, (int, np.integer)) and 0 <= p.class_id < num_classes):
        raise ValueError(f"invalid class id {p.class_id!r}")
    if not (isinstance(p.score, (int, float, np.floating)) and math.isfinite(p.score) and 0.0 <= p.score <= 1.0):
        raise ValueError(f"score outside [0,1]: {p.score!r}")
    if len(p.box) != 4 or not all(math.isfinite(float(v)) for v in p.box):
        raise ValueError("non-finite box")
    x1, y1, x2, y2 = p.box
    if not (x2 > x1 and y2 > y1):
        raise ValueError("non-positive box")


def size_bin(area: float) -> str:
    """Half-open P05 bins: small <1024, medium [1024, 9216), large >=9216."""
    return "small" if area < 1024 else ("medium" if area < 9216 else "large")


def area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (area(a) + area(b) - inter)


def clip_box(b, w, h):
    return (min(max(b[0], 0.0), w), min(max(b[1], 0.0), h), min(max(b[2], 0.0), w), min(max(b[3], 0.0), h))


def letterbox_params(w, h, size=640):
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    return r, (size - nw) / 2.0, (size - nh) / 2.0


def to_letterbox(box, w, h, size=640):
    r, px, py = letterbox_params(w, h, size)
    return (box[0] * r + px, box[1] * r + py, box[2] * r + px, box[3] * r + py)


def from_letterbox(box, w, h, size=640):
    r, px, py = letterbox_params(w, h, size)
    return ((box[0] - px) / r, (box[1] - py) / r, (box[2] - px) / r, (box[3] - py) / r)


def ap101(tp_flags, n_gt):
    """tp_flags: 1/0 for non-ignored detections in score order."""
    if n_gt == 0:
        return None, [], []
    if not len(tp_flags):
        return 0.0, [], []
    tp = np.cumsum(tp_flags)
    fp = np.cumsum(1 - np.asarray(tp_flags))
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-16)
    env = np.maximum.accumulate(prec[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, grid - 1e-12, side="left")
    q = np.array([env[i] if i < len(env) else 0.0 for i in idx])
    return float(q.mean()), rec.tolist(), prec.tolist()


def _match_threshold(images, preds_by_img, thr, max_det, cls_ok=None, in_bin=None, pred_in_bin=None):
    """Return ordered list of (score, image_id, index, class, flag, ...) with flag in {1 TP, 0 FP, -1 ignored}."""
    out = []
    for im in images:
        plist = sorted(preds_by_img.get(im.image_id, []), key=lambda p: (-p.score, p.image_id, p.index))[:max_det]
        used = set()
        for p in plist:
            cands = []
            overlaps_valid = False
            hit_out_bin = False
            for j, (c, g) in enumerate(im.gt):
                if c != p.class_id:
                    continue
                v = iou(p.box, g)
                if v >= thr - EPS:
                    overlaps_valid = True
                    if j in used:
                        continue
                    if in_bin is not None and not in_bin(im, j):
                        hit_out_bin = True
                        continue
                    cands.append((-v, j))
            if cands:
                cands.sort()
                j = cands[0][1]
                used.add(j)
                flag = 1
            elif hit_out_bin and not any(
                    c == p.class_id and iou(p.box, g) >= thr - EPS and j not in used and (in_bin is None or in_bin(im, j))
                    for j, (c, g) in enumerate(im.gt)):
                flag = -1
            elif overlaps_valid:
                flag = 0
            elif any(c == p.class_id and iou(p.box, g) >= 0.5 - EPS for c, g in im.ignore):
                flag = -1
            elif pred_in_bin is not None and not pred_in_bin(im, p):
                flag = -1
            else:
                flag = 0
            out.append((p.score, im.image_id, p.index, p.class_id, flag))
    out.sort(key=lambda t: (-t[0], t[1], t[2]))
    return out


def _ap_for(images, preds_by_img, thrs, max_det, num_classes, in_bin=None, pred_in_bin=None):
    per_thr_cls = {}
    for thr in thrs:
        res = _match_threshold(images, preds_by_img, thr, max_det, in_bin=in_bin, pred_in_bin=pred_in_bin)
        for c in range(num_classes):
            n_gt = sum(1 for im in images for j, (cc, _) in enumerate(im.gt)
                       if cc == c and (in_bin is None or in_bin(im, j)))
            flags = [1 if t[4] == 1 else 0 for t in res if t[3] == c and t[4] != -1]
            per_thr_cls[(thr, c)] = ap101(flags, n_gt)[0]
    return per_thr_cls


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def evaluate(images, preds, num_classes=1, iou_thresholds=None, max_det=300, op_conf=0.25, ap_floor=0.001):
    thrs = list(iou_thresholds) if iou_thresholds else IOUS
    known = {im.image_id for im in images}
    for p in preds:
        validate_pred(p, num_classes, known)
    kept = [im for im in images if im.status not in EXCLUDED_STATUS]
    excluded = len(images) - len(kept)
    kept_ids = {im.image_id for im in kept}
    by_img = {}
    for p in preds:
        if p.image_id in kept_ids and p.score >= ap_floor:
            by_img.setdefault(p.image_id, []).append(p)
    by_id = {im.image_id: im for im in kept}

    table = _ap_for(kept, by_img, thrs, max_det, num_classes)
    per_class_50 = {}
    per_class_5095 = {}
    for c in range(num_classes):
        v50 = table.get((0.5, c)) if 0.5 in thrs else table[(thrs[0], c)]
        per_class_50[c] = v50
        per_class_5095[c] = _mean([table[(t, c)] for t in thrs])
    ap50 = _mean(per_class_50.values())
    ap5095 = _mean(per_class_5095.values())

    t0 = thrs[0]
    res = _match_threshold(kept, by_img, t0, max_det)
    n_gt = sum(len(im.gt) for im in kept)
    tp = sum(1 for t in res if t[4] == 1)
    fp = sum(1 for t in res if t[4] == 0)
    ign = sum(1 for t in res if t[4] == -1)
    op_tp = sum(1 for t in res if t[4] == 1 and t[0] >= op_conf)
    op_fp = sum(1 for t in res if t[4] == 0 and t[0] >= op_conf)
    op_fn = n_gt - op_tp
    _, rec, prec = ap101([1 if t[4] == 1 else 0 for t in res if t[4] != -1], n_gt)
    fails = {}
    for s, iid, idx, c, f in res:
        if f == 0:
            fails.setdefault(iid, {"fp": 0, "fn": 0})["fp"] += 1
    tp_per = {}
    for s, iid, idx, c, f in res:
        if f == 1:
            tp_per[iid] = tp_per.get(iid, 0) + 1
    for im in kept:
        miss = len(im.gt) - tp_per.get(im.image_id, 0)
        if miss:
            fails.setdefault(im.image_id, {"fp": 0, "fn": 0})["fn"] = miss

    def size_block(area_fn):
        block = {}
        for name in ("small", "medium", "large"):
            def in_bin(im, j, name=name):
                return size_bin(area_fn(im, im.gt[j][1])) == name

            def pred_in(im, p, name=name):
                return size_bin(area_fn(im, p.box)) == name
            n = sum(1 for im in kept for j in range(len(im.gt)) if in_bin(im, j))
            t = _ap_for(kept, by_img, thrs, max_det, num_classes, in_bin=in_bin, pred_in_bin=pred_in)
            a50 = _mean([t[(0.5 if 0.5 in thrs else thrs[0], c)] for c in range(num_classes)])
            a95 = _mean([t[(th, c)] for th in thrs for c in range(num_classes)])
            block[name] = {"n_gt": n, "AP50": a50, "AP50:95": a95}
        return block

    def src_area(im, b):
        return area(b)

    def r640_area(im, b):
        r = letterbox_params(im.width, im.height)[0]
        return area(b) * r * r

    return {
        "AP50": ap50, "AP50:95": ap5095, "per_class_AP50": per_class_50, "per_class_AP50:95": per_class_5095,
        "iou_thresholds": thrs, "tp": tp, "fp": fp, "fn": n_gt - tp, "ignored_preds": ign, "n_gt": n_gt,
        "n_images": len(kept), "excluded_images": excluded, "n_preds": tp + fp,
        "precision": (op_tp / (op_tp + op_fp)) if op_tp + op_fp else 0.0,
        "recall": (op_tp / n_gt) if n_gt else 0.0,
        "op_tp": op_tp, "op_fp": op_fp, "op_fn": op_fn, "op_conf": op_conf,
        "confusion": {"tp": tp, "fp": fp, "fn": n_gt - tp},
        "pr_curve": {"recall": rec, "precision": prec},
        "size": size_block(src_area), "size_resize640": size_block(r640_area),
        "per_image_failures": fails,
        "tp_list": [(t[1], t[2]) for t in res if t[4] == 1],
        "evaluator": {"backend": "P05_CUSTOM_COCO_REPORTING_AP", "ap_grid": 101, "max_det": max_det,
                      "ap_conf_floor": ap_floor},
    }
