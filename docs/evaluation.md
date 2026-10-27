# Evaluation

## What can be evaluated publicly

The evaluator accepts explicit ground-truth and prediction records. Synthetic records need no dataset or checkpoint; real records require authorization. Inference JSON is not automatically ground truth, and a rendered overlay is not an accuracy evaluation.

```python
from p05_detect.eval.evaluator import Image, Pred, evaluate

images = [Image("synthetic", 640, 640, gt=[(0, (10, 10, 30, 30))])]
predictions = [Pred("synthetic", 0, 0.9, (10, 10, 30, 30))]
report = evaluate(images, predictions)
print(report)
```

This perfect synthetic match is a plumbing check, not model evidence. The CLI wraps the same matching/statistics engine; it does not run inference or load a private dataset:

```bash
p05-evaluate records.json
# Equivalent: python -m p05_detect.eval records.json
```

Minimal `records.json` (original-image pixel coordinates):

```json
{"images": [{"image_id": "synthetic", "width": 640, "height": 640, "gt": [[0, [10, 10, 30, 30]]]}], "predictions": [{"image_id": "synthetic", "class_id": 0, "score": 0.9, "box": [10, 10, 30, 30]}]}
```

This single-class interface validates identities, dimensions, boxes and scores before scoring. Output contains `all`, `small`, and `nonsmall` cohorts, AP50/AP75/AP50:95, operating recall/precision, TP/FP/FN and FP/image. Empty eligible cohorts have `null` AP. Input errors return exit code 2. Redirect standard output to retain the report.

## Record and metric contract

- `Image`: stable `image_id`, original width/height, `gt` and optional `ignore` lists of `(class_id, (x1, y1, x2, y2))`, and annotation status.
- `Pred`: matching `image_id`, integer `class_id`, finite score in `[0, 1]`, positive-area pixel `xyxy` box, stable tie-breaking `index`, and `space="original_pixel_xyxy"`.
- Use unique image IDs and consistent class mappings. Convert normalized or letterboxed boxes before scoring. Never score predictions against a different split or image identity.
- `unknown_empty` and `missing_suspected` images are excluded in full. Confirmed negatives remain eligible; missing annotations are not confirmed negatives.
- Greedy same-class matching resets at every IoU threshold. Duplicate detections of valid ground truth remain false positives; eligible ignore-region matches are neutralized.
- AP uses a 101-point precision envelope over IoU `0.50:0.05:0.95`, confidence floor `0.001`, and at most 300 detections/image by default. Operating recall uses confidence `0.25` at IoU `0.50`.
- This is `CUSTOM_COCO_REPORTING_AP`, not Ultralytics' upstream checkpoint-selection AP. Do not interchange them.

Size reports distinguish source-pixel and resized-to-640 area. Research small-object comparisons use **RESIZE640**, small area `<32²`; medium is `[32², 96²)`, large is `>=96²`. State the coordinate/size convention with every result.

## Canonical statistics and AP75

`p05_detect.eval.paired_stats` supplies evaluator-matched sufficient statistics, cohort metrics and paired source-group bootstrap comparisons. Preserve identical image order, group IDs, prediction settings, and seed pairing across arms. Corrective comparisons use 2,000 draws with bootstrap seed `20261025`; report uncertainty, not only mean differences. Resample source groups, not independent boxes; these intervals are conditional on training seed 101.

AP75 comes from the canonical matching/statistics path at **IoU 0.75**, using `weighted_ap` on that threshold's matches. Do not relabel AP50 or substitute AP50:95. The CLI supplies this field without changing historical `point_metrics` output. Synthetic regression tests distinguish AP75 from both AP50 and the threshold average; they do not validate model accuracy.

## Interpreting the research

The frozen internal validation split (VAL) contains 1,048 images, 462 source groups, 1,131 targets and 227 RESIZE640-small targets. Reference architecture comparisons used three seeds; key CARAFE/CARAFE+NWD gains had intervals including zero. Seed-101 demo outputs are not those averages. Corrective data comparisons used only seed 101, not a multi-seed replication.

Primary corrective checkpoints were selected by upstream whole-VAL fitness, not this reporting AP or small-object outcomes. The positive best-checkpoint small AP difference reverses for last checkpoints. Strict annotation verification remains blocked; the project was closed under an owner-accepted exception. Decision: **NO_PROMOTION**. Internal HOLDOUT was already inspected and is not independent confirmation; no external-generalization claim follows. See [research summary](research-summary.md) for corrected metrics and slicing replay.
