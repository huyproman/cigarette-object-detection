"""Prediction exporter: ultralytics Results -> P05 Pred records in original_pixel_xyxy, tagged with head/checkpoint."""
import hashlib
from pathlib import Path
from p05_detect.eval.evaluator import Pred, COORD_SPACE, clip_box

def clip_or_drop(box, w, h):
    b = clip_box(box, w, h)
    return b if b[2] > b[0] and b[3] > b[1] else None


HEADS = {"one2many": dict(nms=True), "one2one": dict(nms=False)}


def predict_records(weights_or_model, image_paths, head="one2many", conf=0.001, iou=0.7, max_det=300, imgsz=640, device=0):
    from ultralytics import YOLO
    m = weights_or_model if hasattr(weights_or_model, "predict") else YOLO(str(weights_or_model))
    recs, meta = [], []
    dropped = 0
    for k in range(0, len(image_paths), 32):
        chunk = [str(p) for p in image_paths[k:k + 32]]
        # Validator semantics: nms=True -> end2end False (one2many + NMS); nms=False -> end2end True (one2one).
        # Head is selected by the `nms` predict argument (-> AutoBackend(end2end=nms is False)); the predictor must be rebuilt
        # when it changes, so set it explicitly and drop any cached predictor.
        if m.predictor is not None and getattr(m.predictor.args, "nms", None) != HEADS[head]["nms"]:
            m.predictor = None
        for p, r in zip(chunk, m.predict(chunk, conf=conf, iou=iou, max_det=max_det, imgsz=imgsz, device=device, verbose=False, nms=HEADS[head]["nms"])):
            h, w = r.orig_shape
            meta.append((p, w, h))
            b = r.boxes
            for i in range(len(b)):
                x = [float(v) for v in b.xyxy[i].tolist()]
                cb = clip_or_drop(x, w, h)
                if cb is None:
                    dropped += 1
                    continue
                recs.append(Pred(p, int(b.cls[i]), float(min(max(b.conf[i], 0), 1)), cb, i, COORD_SPACE))
    predict_records.last_dropped_degenerate = dropped
    return recs, meta
