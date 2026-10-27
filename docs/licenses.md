# Licensing and asset boundary

The source is distributed under **AGPL-3.0-only**; see `LICENSE` and `NOTICE.md`.
Ultralytics integrations and modified model YAMLs retain their upstream notices.
CARAFE and NWD source identify their algorithm publications and the local implementation.
No borrowed third-party implementation is silently relicensed as permissive code.

Installed dependency metadata was checked for the historical research environment (the package now declares `torch>=2.7`, `torchvision>=0.22`; later versions were tested):

| Dependency | Version | Declared license |
|---|---|---|
| Ultralytics | 8.4.174 | AGPL-3.0 |
| pycocotools | 2.0.11 | FreeBSD |
| PyTorch | 2.7.1 | BSD-3-Clause |
| torchvision | 0.22.1 | BSD |
| Gradio (optional) | 6.30.0 | Apache-2.0 |
| NumPy | 2.5.4 | BSD-3-Clause and bundled permissive licenses |
| Pillow | 12.3.0 | MIT-CMU |

Dependencies are installed separately, not vendored into this source tree. Their
distributions retain their own license/notice files, including bundled-library
licenses. AGPL obligations still apply to covered modifications, including source
availability when a modified covered program is offered over a network. This is
not a promise that every deployment or commercial integration meets those duties.

Training images, labels, annotation reviews and all trained checkpoints are
excluded. Their third-party redistribution permissions are unresolved; local
research authorization is not a redistribution license. The only included image
is project-generated synthetic artwork; it is not a dataset photograph or accuracy
example. Do not add private assets or publish a network demo without a separate
rights/security review. No model/data download entitlement is supplied.
