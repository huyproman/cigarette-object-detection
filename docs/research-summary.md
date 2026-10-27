# Research summary

## Scope and final decision

Single-image cigarette object localization, not smoking-behaviour recognition. Evidence is internal research, not an external benchmark or production claim.

Experiments completed and the project was closed under an owner-accepted annotation-QA exception. Strict annotation verification, and the strict result on the revised dataset, remain blocked. Independent annotation verification is unmet; the owner data-use exception is not human QA acceptance. Decision: **NO_PROMOTION** (the enhanced variants were not promoted to default). The `b0` baseline at 640 px input without slicing remains the default; publication/deployment rights are separate.

## Evaluation frame

Frozen VAL: 1,048 images, 462 source groups, 1,131 targets, 227 small targets. Small means area `<32²` after fixed RESIZE640, even when inference uses 960. AP below is custom COCO-reporting AP50:95; recall uses IoU 0.50 and confidence 0.25. Differences are percentage points (pp), except FP/image. See [evaluation](evaluation.md) for matching and exclusions.

Reference architecture evidence used three seeds and did not establish a CARAFE/NWD winner or synergy. Follow-up comparisons below use **seed 101 only**. Internal HOLDOUT was already inspected, so it is not independent confirmation.

## Resolution trade-off

The `e3_combined` checkpoint at 960 px input versus the same checkpoint at 640: **+4.77 pp small AP**, **+5.29 pp small recall**, **-0.29 pp whole AP**, **-1.267 pp non-small AP**, and **+0.0716 FP/image**. The non-small protection limit (no worse than -1 pp) fails. This is a small-object trade-off, not an accepted meaningful gain or promotion. It measures inference resolution, not a general benefit from high-resolution training.

## Corrective data comparison

The control arm and the data-change arm (the revised dataset) started from equal initialization tensors and each completed exactly **6,629 successful SGD updates**, logical batch 64 (16×4), with matched update-indexed schedules and validation opportunities. AMP-skipped attempts did not advance the budget.

Primary checkpoints were selected by upstream whole-VAL fitness, not by small-target reporting results. Their best updates were early: control **133**, data-change **399**. Both used seed 101; the demo registry does not distribute this corrective pair.

| Reporting metric | Control best | Data-change best | Difference |
|---|---:|---:|---:|
| Small AP50:95 | 32.45% | 34.28% | +1.83 pp |
| Small recall | 75.33% | 76.21% | +0.88 pp |
| Whole AP50:95 | 42.25% | 43.49% | +1.24 pp |
| Non-small AP50:95 | 44.62% | 45.84% | +1.22 pp |
| FP/image | 0.21660 | 0.20134 | -0.0153 |

Paired source-group bootstrap: 2,000 draws, seed `20261025`. Small AP difference 90% CI **[+0.68, +2.86] pp** (95%: [+0.49, +3.10]); small recall 90% CI **[-1.40, +3.13] pp**. These intervals are conditional on seed 101, not evidence across training seeds.

Last-checkpoint sensitivity (update 6,629 for both) gives **-1.11 pp small AP** and **-1.32 pp small recall**. Best remains primary; the sign flip cannot justify checkpoint cherry-picking. The pairwise data gain also does not establish a better deployed model: both best arms lose more than 1 pp whole/non-small AP against the original `e3_combined` reference. Strict QA remains blocked. **NO_PROMOTION** applies despite the positive best-checkpoint AP interval.

## Corrected slicing replay

The old mode-name heuristic accidentally selected 960 tiles for FULL+SLICE640. Explicit tile-size selection corrected the replay on all 1,048 VAL images using the same `e3_combined` checkpoint and ground truth.

| Strategy | Small AP | Small recall | Whole AP | Non-small AP | FP/image | p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| FULL | 34.01% | 74.45% | 44.35% | 46.79% | 0.207 | 11.4 / 15.6 |
| SLICE640 | 30.64% | 70.04% | 41.80% | 44.48% | 0.209 | 15.7 / 59.0 |
| SLICE960 | 33.08% | 73.57% | 43.02% | 45.42% | 0.168 | 11.2 / 44.7 |
| FULL+SLICE640 | 32.05% | 77.09% | 42.59% | 45.16% | 0.225 | 25.2 / 69.6 |

Corrected FULL+SLICE640: -1.96 pp small AP, +2.64 pp recall, about 2.2× median latency versus FULL. No slicing mode improves small AP; slicing stays opt-in. Timings cover all passes and merge, batch 1, three sweeps over 100 long-edge-stratified images; they are not a production SLA or directly comparable to full-VAL benchmarks.

## Evidence identity and limits

The private final reports are not public dependencies. The following SHA256 identities pin the read-only source artifacts used for this summary; labels are provenance descriptions, not download links.

| Source artifact | SHA256 |
|---|---|
| Final closeout report | `9d34a23392b756d9739e879c3312bf22f62cbcd3a195e9b6c8df7e33377c3bcb` |
| Final terminal status | `bfda278e6a890617e2fe5bf5ecd4be5201762ab8a93781f59f8b7a064ef144c3` |
| Corrected stage effects | `7a151421816f9ccede3d641f2566a1daf27427bf8863e839b87da456c9b8f6cd` |
| Corrective metrics | `7928f236c249e6ecc3d6de41d229379ab8fdd6260fbdb70fc8822126fa8ba020` |
| Best/last checkpoint sensitivity | `e07c096da9bdb9296b8b181f5a56873ad0e29f3162af4fc4f286334906da180a` |
| Slicing replay | `b05117abf31f7de030e81d5c4beb80812185b81fcc5b8fb83e012e5351d59920` |

Private data, review materials and corrective checkpoints are not supplied. The public user-assets trainer is not an exact replay of the private corrective protocol. See [reproducibility](reproducibility.md) for the supported boundary.
