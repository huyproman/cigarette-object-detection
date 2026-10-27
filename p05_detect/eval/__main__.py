"""Score explicit original-pixel JSON records using the existing reporting engine."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from p05_detect.eval.evaluator import Image, Pred, validate_pred
from p05_detect.eval.paired_stats import prepare, point_metrics, weighted_ap


def score_records(data):
    """Single-class records; fixed resize640 cohorts, independent of inference size."""
    images = [Image(**row) for row in data["images"]]
    known = {im.image_id for im in images}
    if not images or len(known) != len(images):
        raise ValueError("images must be nonempty with unique image_id values")
    for im in images:
        if not isinstance(im.image_id, str) or not im.image_id:
            raise ValueError("image_id must be a nonempty string")
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in (im.width, im.height)):
            raise ValueError("image dimensions must be positive integers")
        if im.status not in ("positive", "confirmed_negative", "unknown_empty", "missing_suspected"):
            raise ValueError("unsupported image status")
        for cls, box in im.gt + im.ignore:
            validate_pred(Pred(im.image_id, cls, 1.0, box), 1, known)
            if not (0 <= box[0] < box[2] <= im.width and 0 <= box[1] < box[3] <= im.height):
                raise ValueError("ground truth/ignore boxes must be inside the original image")
    preds = [Pred(**row) for row in data["predictions"]]
    for pred in preds:
        validate_pred(pred, 1, known)
    p = prepare(images, preds)
    if not p["n_images"]:
        raise ValueError("no eligible images after status exclusions")
    result = point_metrics(p, np.ones(p["n_images"]))
    for cohort, row in result.items():
        _, flags75, _ = p["rows"][cohort][.75]
        row["AP75"] = weighted_ap(flags75, np.ones(len(flags75)), row["n_gt"])
        scores, flags, _ = p["rows"][cohort][.5]
        hit = scores >= .25
        tp, fp = float((flags * hit).sum()), float(((1 - flags) * hit).sum())
        row.update(tp=tp, fp=fp, fn=row["n_gt"] - tp,
                   precision=tp / (tp + fp) if tp + fp else 0.,
                   fp_per_image=fp / p["n_images"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="JSON with images and predictions arrays")
    args = parser.parse_args(argv)
    try:
        result = score_records(json.loads(args.records.read_text()))
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
