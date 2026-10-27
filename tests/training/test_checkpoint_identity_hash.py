"""The checkpoint identity hash keeps its historical byte contract."""
import hashlib

import torch


def test_checkpoint_hash_alias_and_frozen_bytes():
    from p05_detect.experiments.b0_trainer import sd_hash
    from p05_detect.experiments.init_state import state_hash

    state = {"z": torch.tensor([2., 3.]), "a": torch.tensor([1.])}
    expected = hashlib.sha256(b"a" + state["a"].numpy().tobytes()
                              + b"z" + state["z"].numpy().tobytes()).hexdigest()
    assert sd_hash(state) == state_hash(state) == expected
    assert sd_hash is state_hash
