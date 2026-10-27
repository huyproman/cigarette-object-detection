# Reproducibility

## Public checks without private assets

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m p05_detect.infer --help
python -m p05_detect.train --help
python -m pytest -q
```

Python >=3.12 is required. Ultralytics is pinned to `8.4.174`; fresh dependency-install validation is separate from the historical research environment. UI dependencies are separate: `python -m pip install -e '.[ui]'`. Weight-dependent checks skip when authorized weights are absent; skipped checks are not inference or accuracy validation. The synthetic evaluator example in [evaluation](evaluation.md) works without weights.

Install metadata defines the public dependencies. Historical CUDA research environments and shared-host timings are not reproduced by editable installation, nor are those timings a production SLA.

## Replaying authorized reference inference

### Migration from earlier layouts and Python names

The default checkpoint directory is now `weights`, not `weights/r4`. For an
existing authorized layout, pass `--weights-dir /absolute/path/to/weights/r4`
or set `P05_WEIGHTS_DIR` to that directory. No checkpoint is silently selected,
downloaded or substituted; registry identities still apply.

Update Python imports from `p05_detect.eval.r3_stats` to
`p05_detect.eval.paired_stats`. Registry profiles no longer expose `campaign_id`,
`training_code_revision`, or `registry_sha256`; external consumers must stop
requiring those fields and retain their own provenance when needed. Readable
slicing modes are primary; the earlier `P3-SLICE640`, `P3-SLICE960`,
`P3-FULL_PLUS_SLICE`, and `P3-FULL` Python arguments remain accepted.

Place these exact files in an authorized directory, then use an absolute path:

| Filename | Profile |
|---|---|
| `b0_seed101_best.pt` | `b0` |
| `e2_carafe_seed101_best.pt` | `e2_carafe` |
| `e3_combined_seed101_best.pt` | `e3_combined` |

```bash
export P05_WEIGHTS_DIR=/absolute/path/to/authorized/checkpoints
python -m p05_detect.infer verify
python -m p05_detect.infer infer examples/synthetic_fixture.png --profile b0 --device cpu --precision fp32 --out p05_out
```

The [packaged registry](../p05_detect/configs/registry/model_registry.json) is the source of filenames, sizes and SHA256 identities. There is no automatic download or replacement. Verification checks bytes, not redistribution rights. Code installation and synthetic examples confer no access to private weights or data.

For comparable runs, retain checkpoint hashes, profile, input identities, image size, head, confidence, NMS IoU, max detections, precision, device, package versions, hardware, and full output records. Record slicing separately if enabled; default inference remains **`b0` at 640 without slicing**.

## User-assets training versus exact research replay

Use `python -m p05_detect.train --help` for the public user-assets training interface. Supply explicit configuration, authorized dataset paths, initialization and its expected hash. Retain the complete invocation, seed, dataset/split identities, environment, output checkpoints and logs. This permits auditable training on supplied assets, not recovery of missing private data.

```bash
python -m p05_detect.train --profile e3_combined \
  --data /absolute/path/data.yaml --data-sha256 DATA_YAML_SHA256 \
  --init /absolute/path/baseline_init.pt --init-sha256 TENSOR_STATE_SHA256 \
  --recipe examples/training_recipe.json --seed 101 --device cpu \
  --out outputs/authorized-training --dry-run
```

Replace both digest placeholders. The data digest hashes YAML bytes, not every
image/label. The init digest uses `experiments.init_state.state_hash`, not the
checkpoint-file digest. Initialization must be a trusted stored B0 tensor
dictionary matching the pinned single-class model, not a serialized model object.
Use standard YOLO image/label layout, one class, local train/val directories or
image lists, and an absolute YAML `path`. Download directives are rejected.
The example recipe exposes optimizer/scheduler choices at fixed 640 input; its
CPU-safe worker/AMP settings do not recreate the original GPU environment.
Dry-run validates inputs and prints the plan without training. Remove it only
for separately authorized training on your own assets.
Dry-run verifies initialization integrity, not full model tensor compatibility;
the existing trainers enforce compatible tensors before any optimizer update.

Exact corrective replay additionally needs private images/labels, frozen split/group manifests, annotation status/ignore records, paired initial tensors, predeclared data exposures, a successful-update-indexed schedule and sealed validation/selection evidence. Both private arms completed 6,629 successful SGD updates with logical batch 64; AMP-skipped attempts did not consume that budget. The public trainer does **not** claim to implement this private protocol exactly. Internal reports, operational scripts and dataset-specific utilities are excluded from the public candidate.

## Evidence and release status

| Item | Status |
|---|---|
| Experiments | Completed |
| Closure | Closed under an owner-accepted annotation-QA exception |
| Strict annotation verification | Blocked (independent verification unmet) |
| Strict result on the revised dataset | Blocked pending new data |
| Model decision | `NO_PROMOTION`: enhanced variants not promoted; `b0` at 640 remains default |
| Human QA | Unmet; the owner exception is not human acceptance |

The [research summary](research-summary.md) records evidence identities without requiring private report links. Confidence intervals describe source-group uncertainty conditional on seed 101, not annotation correctness or training-seed variability.

Publication and data/checkpoint rights remain separate gates. No dataset download entitlement, checkpoint license, or deployment authorization is supplied here. Only use trusted checkpoints and images/labels for which you have permission.
