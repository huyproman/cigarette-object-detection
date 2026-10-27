"""P05DetectionModel: DetectionModel with (a) CARAFE registered with the pinned YAML parser and (b) an optional NWD-hybrid criterion.
No upstream file is modified. Importing this module is what makes CARAFE YAMLs build (do it before any trainer/model call)."""
from pathlib import Path

import ultralytics.nn.tasks as _T
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import E2ELoss

from p05_detect.losses.nwd import make_p05_loss_class
from p05_detect.modules.carafe import CARAFE
from p05_detect.modules.mhsa import C2PSAMHSA

_T.C2PSAMHSA = C2PSAMHSA                             # C2PSA + MHSA adapter inside layer 10 (experimental)
_T.CARAFE = CARAFE                                   # parse_model resolves custom names via tasks globals
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs/models"


class P05DetectionModel(DetectionModel):
    def __init__(self, cfg="yolo26s.yaml", ch=3, nc=None, verbose=True, nwd=None):
        self.nwd = dict(nwd) if nwd else None            # {'lam':..., 'C':...}; None => upstream loss
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        if not self.nwd:
            return super().init_criterion()
        fn = make_p05_loss_class(self.nwd["lam"], self.nwd["C"])
        if getattr(self.model[-1], "one2one_cv2", None) is None:
            return fn(self)
        return E2ELoss(self, loss_fn=fn)
