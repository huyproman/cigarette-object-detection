"""TRAIN-only NWD normalisation constant C: median of sqrt(w*h) of eligible TRAIN GT boxes after the deterministic
640 letterbox; C = 12.8 px if median/12.8 in [0.5, 2] else the median. No VAL/HOLDOUT label or metric enters."""
import hashlib, json, statistics
from pathlib import Path


def compute_c(boxes_xyxy, sizes_wh, default=12.8):
    if not boxes_xyxy:
        raise ValueError("empty TRAIN: C undefined (gate failure)")
    vals = []
    for (x1, y1, x2, y2), (w, h) in zip(boxes_xyxy, sizes_wh):
        r = min(640 / w, 640 / h)
        vals.append(((x2 - x1) * r * (y2 - y1) * r) ** 0.5)
    med = statistics.median(vals)
    inside = 0.5 <= med / default <= 2.0
    return dict(n_boxes=len(vals), s_med=med, C=default if inside else med, rule="default_12.8" if inside else "median_outside_band")


def from_manifest(dataset_dir):
    d = Path(dataset_dir); boxes, sizes = [], []
    man = [json.loads(l) for l in open(d / "split_manifest.jsonl")]
    for m in man:
        if m["split"] != "TRAIN":
            continue
        for ln in Path(m["path"].replace("/images/", "/labels/")).with_suffix(".txt").read_text().splitlines():
            _, cx, cy, bw, bh = map(float, ln.split()); W, H = m["w"], m["h"]
            boxes.append(((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H)); sizes.append((W, H))
    r = compute_c(boxes, sizes)
    r.update(split_manifest_sha256=hashlib.sha256((d / "split_manifest.jsonl").read_bytes()).hexdigest(), split="TRAIN",
             calculation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return r


if __name__ == "__main__":
    import sys
    r = from_manifest(sys.argv[1]); Path(sys.argv[2]).write_text(json.dumps(r, indent=1)); print(r)
