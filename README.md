# P05 - cigarette object detection

Single-image detection of **cigarette objects** (bounding boxes) with YOLO26s. It is object detection, not smoking-action recognition, person tracking, video analytics, or a production/real-time system.

**Problem.** Cigarettes are small, low-contrast objects, so small-object recall and localization quality are the weak point of a stock detector.

**Approach.** A YOLO26s baseline is compared with two variants: CARAFE content-aware upsampling in the neck, and CARAFE plus a Normalized Gaussian Wasserstein Distance (NWD) box-loss term used during training only. Optional tiled ("sliced") inference is also evaluated.

**Contribution.** Pure-PyTorch CARAFE and NWD implementations integrated with a pinned Ultralytics, a custom COCO-style evaluator with small/non-small cohorts and paired source-group bootstrap statistics, an inference CLI/UI with hash-verified checkpoints, and a hash-checked local training entry point.

**Evaluated.** On an internal validation split (1,048 images, 462 source groups) with custom COCO-style AP. Small objects are those under 32x32 px after resizing to 640.

**Principal result: NO_PROMOTION.** The enhanced variants were not promoted to default. The baseline at 640 px input (`b0`, plain YOLO26s, the "B0" configuration) stays the default.
- The CARAFE+NWD model run at 960 px input (optional) improves small-object AP by about +4.77 pp (recall +5.29 pp) but lowers non-small AP by about -1.267 pp, which breaks the -1 pp protection limit.
- A later corrected data comparison found +1.83 pp small AP at the best checkpoint (90% CI +0.68 to +2.86), but -1.11 pp at the last checkpoint, and both arms lose more than 1 pp whole-image AP against the earlier reference.

**Limitations.** Single training seed for the follow-up comparisons; the validation split was also used for checkpoint selection, so absolute APs are optimistic; no independent test or external-generalization claim. Strict annotation verification stayed blocked; the project was closed under an owner-accepted annotation-QA exception, which is not human QA acceptance.

**Rights boundary.** Code is AGPL-3.0-only. Datasets, labels, trained checkpoints and internal reports are not included; their redistribution rights are unresolved (see [Rights](#license-attribution-and-rights)).

| Profile | Architecture | Status |
|---|---|---|
| `b0` | YOLO26s | Reference and **default at 640** |
| `e2_carafe` | YOLO26s + CARAFE upsampling | Optional research comparison |
| `e3_combined` | Same CARAFE graph; NWD added to the training loss only | Optional research comparison |

Sliced inference is experimental and opt-in. Profile keys `e2`/`e3` below and in the docs refer to `e2_carafe` and `e3_combined`.

## Install

Python **>=3.12**; upstream integration is pinned to `ultralytics==8.4.174`. CPU supports smoke checks; CUDA is optional for inference. A fresh-install verification is separate from historical research-environment validation.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Install the optional local UI with `python -m pip install -e '.[ui]'`. HEIF/HEIC input is not supported; undecodable files raise `InferenceError` (`.[heif]` only adds the optional fallback decoder Ultralytics may try).
See [reproducibility](docs/reproducibility.md) for environment and private-data limits. Installation does not supply model weights.

## Authorized weights and inference

No checkpoint download is provided. Obtain the three authorized seed-101 reference checkpoints separately and set an **absolute** directory:

```bash
export P05_WEIGHTS_DIR=/absolute/path/to/authorized/checkpoints
python -m p05_detect.infer verify
python -m p05_detect.infer infer examples/synthetic_fixture.png --profile b0 --device cpu --out p05_out
python -m p05_detect.infer compare examples/synthetic_fixture.png --out p05_out
```

`verify` checks the filenames and SHA256 values in [`p05_detect/configs/registry/model_registry.json`](p05_detect/configs/registry/model_registry.json). Missing or mismatched files fail; nothing is downloaded, trained, or substituted. Possession, hash verification, and this documentation imply **no rights** to data or checkpoints.

`p05-detect` is the installed alias for this module CLI (`infer`, `compare`, `verify`). `p05-evaluate records.json` scores explicit ground-truth/prediction records without weights or private folders; see the [evaluation schema and Python API](docs/evaluation.md).

```python
from p05_detect.infer import load_model, predict, serialize

model = load_model("b0", device="cpu", precision="fp32")
result = predict("examples/synthetic_fixture.png", model)
print(serialize(result))
```

Defaults: 640 input, confidence `0.25`, NMS IoU `0.70`, FP32, one-to-many head + NMS. `--precision fp16` requires CUDA and does not silently fall back. JSON schema `"1"` records image dimensions, model/checkpoint identity, settings, detections in original-image pixel `xyxy` and normalized coordinates, and stage timings. `compare` writes three results and, unless disabled, overlays and a montage. The synthetic fixture checks plumbing, not accuracy.

## Training and evidence

For training on user-authorized assets, start with `python -m p05_detect.train --help`. Supply explicit configuration, initialization and hash identity; this public path does **not** reproduce the private corrective protocol exactly.

Internal validation found a small-object trade-off for the CARAFE+NWD model at 960 px input (+4.77 pp small AP, +5.29 pp recall, -1.267 pp non-small AP). A corrected, single-seed data comparison found +1.83 pp small AP (90% CI +0.68 to +2.86), but the last-checkpoint sensitivity reversed sign (-1.11 pp). Both come from one seed on a validation split that was also used for checkpoint selection, so absolute APs are optimistic. Neither result promotes a model. Demo checkpoints are not the corrective pair; no external-generalization claim is supported.

- [Architecture and public/internal boundaries](docs/architecture.md)
- [Evaluation conventions and synthetic example](docs/evaluation.md)
- [Reproducibility and evidence status](docs/reproducibility.md)
- [Research summary, corrected results and limitations](docs/research-summary.md)

## Local UI and security

```bash
python -m p05_detect.infer.app
```

The optional UI binds to `127.0.0.1:7860`, disables sharing/analytics, and has no authentication. Do not expose it publicly. Uploads are limited to JPG/PNG, 25 MiB and 40 MP. Temporary uploads/exports are removed on normal exit and handled termination; crashes or `kill -9` may leave files. Supply only authorized images and trusted checkpoints.

## License, attribution and rights

- Code: **AGPL-3.0-only** (see [`LICENSE`](LICENSE)). The package depends on Ultralytics (AGPL-3.0); four model YAMLs keep their notices: `yolo26.yaml` and `yolo26s-p2.yaml` are copies; `yolo26s-carafe.yaml` and `yolo26s-mhsa.yaml` are modifications. Network-served modified versions carry AGPL source-offer duties.
- CARAFE and NWD are local reimplementations of published algorithms (Wang et al., ICCV 2019; Wang et al., arXiv:2110.13389); no third-party implementation is vendored.
- Excluded: dataset images/labels, review kits, trained checkpoints, internal reports. The training data came from a third-party source (Roboflow Universe); its dataset name, license, and attribution terms are **not established in this repository**, so no redistribution or reuse right is implied. Checkpoint redistribution rights are likewise unresolved.
- Original project source holder: Nguyen Quang Huy, Copyright (c) 2026. Source-publication authority is confirmed by OWNER_ATTESTATION (2026-10-27), not independent ownership verification. The historical Roboflow key is revoked according to OWNER_ATTESTATION, not independently verified provider evidence. Neither statement clears dataset or checkpoint rights.

Full inventory: [`NOTICE.md`](NOTICE.md), [`THIRD_PARTY.md`](THIRD_PARTY.md), [`docs/licenses.md`](docs/licenses.md).
