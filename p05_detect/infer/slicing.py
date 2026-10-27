"""Opt-in sliced inference: source-pixel tiling (SLICE640, SLICE960, FULL_PLUS_SLICE, FULL) with deterministic edge handling,
tile-to-original coordinate restoration, and class-aware global NMS merging at merge_iou=0.5."""
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from ultralytics.utils import nms as _nms

from p05_detect.infer import core

MERGE_IOU = 0.50


TILE_SIZE_BY_MODE = {"SLICE640": 640, "SLICE960": 960, "FULL_PLUS_SLICE": 640}  # FULL_PLUS_SLICE = FULL + SLICE640

def tile_size_for_mode(mode: str) -> int:
    """Explicit mode -> tile size (a substring test on the mode name would send FULL_PLUS_SLICE to 960)."""
    mode = mode.removeprefix("P3-")
    if mode not in TILE_SIZE_BY_MODE:
        raise ValueError(f"unknown tiled mode {mode!r}")
    return TILE_SIZE_BY_MODE[mode]

def generate_tiles(image_w: int, image_h: int, tile_size: int, overlap_ratio: float = 0.20) -> List[Tuple[int, int, int, int]]:
    """Generate (x1, y1, x2, y2) source-pixel tile coordinates covering the image.
    If image dimension < tile_size, returns single tile [0, 0, image_w, image_h].
    """
    if image_w <= tile_size and image_h <= tile_size:
        return [(0, 0, image_w, image_h)]

    step = int(tile_size * (1.0 - overlap_ratio))
    if step <= 0:
        step = tile_size

    if image_w <= tile_size:
        x_coords = [0]
    else:
        x_coords = list(range(0, image_w - tile_size, step))
        if not x_coords or x_coords[-1] + tile_size < image_w:
            x_coords.append(image_w - tile_size)

    if image_h <= tile_size:
        y_coords = [0]
    else:
        y_coords = list(range(0, image_h - tile_size, step))
        if not y_coords or y_coords[-1] + tile_size < image_h:
            y_coords.append(image_h - tile_size)

    tiles = []
    for y in y_coords:
        for x in x_coords:
            tiles.append((x, y, min(x + tile_size, image_w), min(y + tile_size, image_h)))

    return tiles


def global_merge(all_detections: List[Dict[str, Any]], image_w: int, image_h: int, merge_iou: float = MERGE_IOU, max_det: int = 300) -> List[Dict[str, Any]]:
    """Class-aware global NMS merger for candidate detections from multiple tiles and/or full image."""
    if not all_detections:
        return []

    boxes = torch.tensor([d["box_xyxy"] for d in all_detections], dtype=torch.float32)
    scores = torch.tensor([d["confidence"] for d in all_detections], dtype=torch.float32)
    classes = torch.tensor([d["class_id"] for d in all_detections], dtype=torch.float32)

    import torchvision.ops as tv_ops
    keep = tv_ops.batched_nms(boxes, scores, classes, iou_threshold=merge_iou)
    keep = keep[:max_det]

    merged = []
    for idx in keep:
        det = all_detections[idx]
        b = det["box_xyxy"]
        x1 = float(min(max(b[0], 0), image_w))
        y1 = float(min(max(b[1], 0), image_h))
        x2 = float(min(max(b[2], 0), image_w))
        y2 = float(min(max(b[3], 0), image_h))
        if x2 <= x1 or y2 <= y1:
            continue
        merged.append({
            "class_id": det["class_id"],
            "class_name": det["class_name"],
            "confidence": det["confidence"],
            "box_xyxy": [x1, y1, x2, y2],
            "box_xyxy_normalized": [x1 / image_w, y1 / image_h, x2 / image_w, y2 / image_h]
        })

    merged.sort(key=lambda d: -d["confidence"])
    return merged


def predict_sliced(
    image,
    model: core.Model,
    mode: str = "SLICE640",
    conf: float = 0.25,
    iou: float = 0.70,
    merge_iou: float = MERGE_IOU,
    max_det: int = 300,
    _loaded = None
) -> Dict[str, Any]:
    """Execute sliced detection on image.
    Modes:
      'FULL': Standard single-pass full image resized to model input size.
      'SLICE640': 640x640 source tiles, 20% overlap.
      'SLICE960': 960x960 source tiles, 20% overlap.
      'FULL_PLUS_SLICE': Full image + 640x640 tiles fused via global_merge.
    Legacy 'P3-' mode prefixes are accepted; results use readable mode names.
    """
    mode = mode.removeprefix("P3-")
    t_start = time.perf_counter()
    rgb, name = _loaded if _loaded is not None else core.load_image(image)
    decode_ms = (time.perf_counter() - t_start) * 1e3 if _loaded is None else 0.0
    h, w = rgb.shape[:2]

    if mode == "FULL":
        res = core.predict(None, model, conf=conf, iou=iou, max_det=max_det, _loaded=(rgb, name))
        res["strategy"] = "FULL"
        res["n_tiles"] = 1
        return res

    tile_size = tile_size_for_mode(mode)
    tile_coords = generate_tiles(w, h, tile_size, overlap_ratio=0.20)

    all_candidates = []
    forward_times = []

    if mode == "FULL_PLUS_SLICE":
        full_res = core.predict(None, model, conf=conf, iou=iou, max_det=max_det, _loaded=(rgb, name))
        forward_times.append(full_res["timing_ms"]["model_forward"])
        all_candidates.extend(full_res["detections"])

    for (tx1, ty1, tx2, ty2) in tile_coords:
        tile_rgb = np.ascontiguousarray(rgb[ty1:ty2, tx1:tx2])
        tw, th = tx2 - tx1, ty2 - ty1

        t_res = core.predict(None, model, conf=conf, iou=iou, max_det=max_det, _loaded=(tile_rgb, f"{name}_tile_{tx1}_{ty1}"))
        forward_times.append(t_res["timing_ms"]["model_forward"])

        for det in t_res["detections"]:
            bx1, by1, bx2, by2 = det["box_xyxy"]
            orig_x1 = bx1 + tx1
            orig_y1 = by1 + ty1
            orig_x2 = bx2 + tx1
            orig_y2 = by2 + ty1
            all_candidates.append({
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "box_xyxy": [orig_x1, orig_y1, orig_x2, orig_y2]
            })

    t0_merge = time.perf_counter()
    final_dets = global_merge(all_candidates, w, h, merge_iou=merge_iou, max_det=max_det)
    merge_ms = (time.perf_counter() - t0_merge) * 1e3

    total_ms = (time.perf_counter() - t_start) * 1e3

    return {
        "schema_version": core.SCHEMA_VERSION,
        "strategy": mode,
        "n_tiles": len(tile_coords),
        "n_full_passes": 1 if mode == "FULL_PLUS_SLICE" else 0,
        "image": {"name": name, "width": w, "height": h},
        "model": {
            "profile_id": model.profile["profile_id"],
            "seed": model.profile["seed"],
            "checkpoint_sha256": model.profile["checkpoint_sha256"],
            "arm": model.profile["arm"]
        },
        "settings": {
            "conf": conf,
            "iou": iou,
            "merge_iou": merge_iou,
            "max_det": max_det,
            "tile_size": tile_size,
            "overlap_ratio": 0.20
        },
        "detections": final_dets,
        "num_detections": len(final_dets),
        "raw_candidates_before_merge": len(all_candidates),
        "timing_ms": {
            "decode": decode_ms,
            "sum_model_forward": sum(forward_times),
            "merge_nms": merge_ms,
            "end_to_end": total_ms
        }
    }
