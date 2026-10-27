"""B0 trainer: stock ultralytics DetectionTrainer, but the model is built from the stored per-seed init state
(otherwise the yaml path would silently rebuild a random model). No loss/arch changes."""
import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from p05_detect.experiments.init_state import state_hash as sd_hash




class B0Trainer(DetectionTrainer):
    init_state_path = None
    expected_hash = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        m = DetectionModel(cfg or "yolo26s.yaml", nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
        sd = torch.load(self.init_state_path, map_location="cpu")
        assert sd_hash(sd) == self.expected_hash, "init state hash mismatch"
        m.load_state_dict(sd, strict=True)
        assert sd_hash({k: v for k, v in m.state_dict().items()}) == self.expected_hash
        return m
