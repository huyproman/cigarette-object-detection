# Third-party components and rights inventory

Facts only, from the files in this repository. Items marked `OWNER_OR_EXTERNAL_RIGHTS_DECISION_REQUIRED` are not established.

| Item | Relationship | License / status |
|---|---|---|
| Ultralytics 8.4.174 (commit ac5c3c0099109dff0a372b5ec80778deb1dd9057) | Runtime dependency, not vendored. Code imports and subclasses its classes (`DetectionModel`, `BboxLoss`, `E2ELoss`, `DetectionTrainer`, `AutoBackend`) and patches its model parser at runtime (`p05_detect/engine/model.py`). | AGPL-3.0. This repository is AGPL-3.0-only accordingly. |
| `p05_detect/configs/models/yolo26.yaml`, `yolo26s-p2.yaml` | Copies of Ultralytics 8.4.174 YAMLs (`yolo26s-p2.yaml` carries only the upstream header). | AGPL-3.0, upstream header retained. |
| `p05_detect/configs/models/yolo26s-carafe.yaml`, `yolo26s-mhsa.yaml` | Modified Ultralytics YAMLs; modification stated in the header. | AGPL-3.0, upstream header retained. |
| CARAFE (`modules/carafe.py`) | Pure-PyTorch reimplementation of Wang et al., ICCV 2019. File header states it is project-owned and no CUDA extension is used. No upstream source was found copied; no formal code-similarity audit was done. | Algorithm is published; implementation licensed with this repo. |
| NWD loss (`losses/nwd.py`) | Reimplementation of Wang et al., arXiv:2110.13389; subclasses Ultralytics classes, no upstream file modified. | As above. |
| pycocotools 2.0.11, PyTorch, torchvision, NumPy, Pillow, PyYAML, Gradio (optional), pi-heif (optional) | Installed separately, not vendored. | Own licenses; see `docs/licenses.md`. Not re-audited here beyond declared metadata. |
| `examples/synthetic_fixture.png` | Project-generated synthetic image. | No third-party content known. |
| Training dataset (Roboflow Universe) | Not included. Dataset name, license, and attribution requirements are not recorded in this repository. | `OWNER_OR_EXTERNAL_RIGHTS_DECISION_REQUIRED` |
| Trained checkpoints | Not included; only SHA256 values are in `configs/registry/model_registry.json`. Derived from the dataset above. | `OWNER_OR_EXTERNAL_RIGHTS_DECISION_REQUIRED` |
| Original project source holder and authority | Copyright (c) 2026 Nguyen Quang Huy; source-publication authority confirmed by OWNER_ATTESTATION (2026-10-27), not independent ownership verification. Upstream holders retain their rights. | Existing AGPL-3.0-only selection retained; no dataset/checkpoint clearance. |
| Historical Roboflow key | Revoked per OWNER_ATTESTATION (2026-10-27). | Owner statement only, not independently verified provider evidence. |
