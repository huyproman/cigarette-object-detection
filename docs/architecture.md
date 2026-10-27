# Architecture

## Public inference core

`p05_detect.infer` provides `load_model`, `predict`, `compare`, and `serialize`. Its module CLI exposes `infer`, `compare`, and `verify`; the optional Gradio app uses the same core.

| Component | Responsibility |
|---|---|
| `infer/registry.py` + `p05_detect/configs/registry/model_registry.json` | Profile identity, authorized checkpoint resolution, size/hash checks |
| `infer/core.py` | Image validation, preprocessing, model loading, prediction, coordinate restoration, JSON and overlays |
| `infer/__main__.py`, `infer/app.py` | CLI and optional loopback-only UI |
| `infer/montage.py` | Three-profile visual comparison |
| `infer/slicing.py` | Separate opt-in tiled inference and merge logic |
| `modules/carafe.py` | CARAFE graph component for `e2_carafe`/`e3_combined` |
| `eval/evaluator.py`, `eval/paired_stats.py` | Record-based scoring and paired statistics |

Registry and model YAMLs ship inside `p05_detect/configs/`. Default weight lookup is `./weights` in the caller's working directory; set `P05_WEIGHTS_DIR` explicitly for location-independent use. `p05-detect` aliases the module CLI; `p05-evaluate` wraps record matching/statistics without importing operational scripts.

The default pipeline decodes one JPG/PNG, letterboxes to 640, normalizes, transfers to the selected device, executes the one-to-many detection head, applies NMS, and restores/clips boxes to original-image pixels. Results include model identity and explicit settings. Empty detections are valid, not an inference error.

`timing_ms.end_to_end` covers the steady-state image pipeline through result construction. Model loading and warmup are separate one-time costs; overlay writing and UI interactions are not an end-to-end deployment benchmark.

## Model variants

- **`b0` at 640:** baseline YOLO26s (B0 = baseline configuration, 640 = input size in pixels) and the retained default.
- **`e2_carafe`:** replaces upsampling at layers 11 and 14 with CARAFE.
- **`e3_combined`:** uses the same CARAFE inference graph. NWD changes training loss only; it is not an additional inference module.
- **Slicing:** optional inference strategy, not a fourth selected model. It changes coverage, duplicate merging, and cost; compare it under explicit, matched settings rather than silently enabling it.

Research-only code that the public CLI does not expose: an MHSA attention adapter (`modules/mhsa.py`, `configs/models/yolo26s-mhsa.yaml`) and a P2 (stride-4 head) comparison model (`configs/models/yolo26s-p2.yaml`, `engine/trainer.py` arm `P2`). Both are covered by tests, were exploratory, and are not part of the three shipped profiles. Internal arm keys `E1`-`E4` in `experiments/paired_init.py` denote: NWD-only, CARAFE, CARAFE + NWD, and MHSA variants.

Checkpoint loading integrates custom model classes with pinned Ultralytics. Load only trusted, authorized checkpoints: a matching research hash identifies bytes, not ownership or universal safety.

## Training and research boundary

`python -m p05_detect.train --help` is the public entry point for user-authorized training assets. It requires explicit configuration and hash-identified initialization, rather than inferring private dataset locations or downloading replacement weights. CARAFE changes the graph; NWD changes training loss, not inference outputs or the metric definition.

The public candidate excludes internal reports, operational scripts and dataset-specific utilities. The private corrective experiment additionally required frozen manifests, annotation identities, paired initial tensors, an exact successful-update controller, exposure accounting and sealed checkpoint-selection evidence. The public trainer is not an exact replay of that protocol.

The project was closed under an owner-accepted annotation-QA exception, while strict annotation verification remains blocked. No new default or deployment model was selected. See [research summary](research-summary.md) for corrected evidence and [reproducibility](reproducibility.md) for asset requirements.
