"""Paired initialisation: every arm starts from the stored per-seed B0 state (708 tensors); CARAFE tensors come from an explicit
per-seed generator shared by E2 and E3."""
import hashlib
import torch

from p05_detect.engine.model import P05DetectionModel, CONFIG_DIR
from p05_detect.modules.carafe import CARAFE, init_carafe_

CARAFE_LAYERS = (11, 14)


def carafe_generator(seed: int, layer: int) -> torch.Generator:
    h = int(hashlib.sha256(f"carafe|{seed}|{layer}".encode()).hexdigest()[:15], 16)
    return torch.Generator().manual_seed(h)


def build_paired_model(arm: str, seed: int, init_path, nwd=None, device="cpu"):
    """arm in {E1,E2,E3}. Returns a model whose 708 common tensors equal the stored B0 state exactly."""
    sd = torch.load(init_path, map_location="cpu")
    if arm == "E1":
        assert nwd
        m = P05DetectionModel("yolo26s.yaml", nc=1, verbose=False, nwd=nwd)
        m.load_state_dict(sd, strict=True)
    elif arm in ("E2", "E3"):
        if arm == "E3":
            assert nwd
        m = P05DetectionModel(str(CONFIG_DIR / "yolo26s-carafe.yaml"), nc=1, verbose=False, nwd=nwd if arm == "E3" else None)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected and sorted(missing) == sorted(f"model.{i}.{n}" for i in CARAFE_LAYERS for n in ("comp.weight", "comp.bias", "enc.weight", "enc.bias"))
        for i in CARAFE_LAYERS:
            init_carafe_(m.model[i], carafe_generator(seed, i))
    elif arm == "E4":
        from p05_detect.modules.mhsa import init_mhsa_, mhsa_generator
        m = P05DetectionModel(str(CONFIG_DIR / "yolo26s-mhsa.yaml"), nc=1, verbose=False)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected and sorted(missing) == sorted(k for k in m.state_dict() if k.startswith("model.10.mhsa.")) and len(missing) == 7
        init_mhsa_(m.model[10].mhsa, mhsa_generator(seed))
    else:
        raise ValueError(arm)
    m.args = None
    return m.to(device)


def common_equal(model, stored_sd) -> int:
    cur = model.state_dict()
    n = 0
    for k, v in stored_sd.items():
        assert k in cur and torch.equal(cur[k].cpu(), v), f"common tensor mismatch: {k}"
        n += 1
    return n


def carafe_state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if any(k.startswith(f"model.{i}.") for i in CARAFE_LAYERS)}
