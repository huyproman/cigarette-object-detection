"""P05DetectionTrainer: stock DetectionTrainer building a P05DetectionModel from a paired, hash-verified initial state.
Step-zero equality of the 708 common tensors is asserted BEFORE the first optimizer step."""
import torch
from ultralytics.models.yolo.detect import DetectionTrainer

from p05_detect.engine.model import P05DetectionModel
from p05_detect.experiments.b0_trainer import sd_hash
from p05_detect.experiments.paired_init import build_paired_model, common_equal


class P05DetectionTrainer(DetectionTrainer):
    arm = None            # E1 | E2 | E3 | P2 (internal arm keys)
    seed = None
    p2_pretrained = None  # pretrained checkpoint path; used only by arm P2
    init_path = None
    init_hash = None
    nwd = None            # {'lam','C'} or None
    step_zero = None      # filled with evidence

    def get_model(self, cfg=None, weights=None, verbose=True):
        sd = torch.load(self.init_path, map_location="cpu")
        assert sd_hash(sd) == self.init_hash, "stored init state hash mismatch"
        if self.arm == "P2":                                  # descriptive comparator: transfer what fits; NOT paired
            m = P05DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=False)
            pre = torch.load(self.p2_pretrained, map_location="cpu", weights_only=False)
            src = (pre["model"] if isinstance(pre, dict) and "model" in pre else pre).float().state_dict()
            own = m.state_dict(); xfer = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
            m.load_state_dict(xfer, strict=False)
            type(self).step_zero = dict(arm="P2", transferred=len(xfer), total=len(own), paired=False)
            return m
        m = build_paired_model(self.arm, self.seed, self.init_path, nwd=self.nwd)
        n = common_equal(m, sd)
        type(self).step_zero = dict(arm=self.arm, common_tensors_equal=n, total_state=len(m.state_dict()), paired=True)
        assert n == 708
        return m
