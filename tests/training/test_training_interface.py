"""Public training validation, without constructing a trainer or training."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


@pytest.fixture
def inputs(tmp_path):
    (tmp_path / "images").mkdir()
    data = tmp_path / "data.yaml"
    data.write_text(f"path: {tmp_path}\ntrain: images\nval: images\nnames: [object]\n")
    init = tmp_path / "init.pt"
    state = {"weight": torch.tensor([1.0])}
    torch.save(state, init)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"epochs": 1, "optimizer": "SGD", "lr0": 0.002, "cos_lr": True}))
    # Independent reproduction of the documented tensor-byte hash, not the file hash.
    tensor_hash = hashlib.sha256(b"weight" + state["weight"].numpy().tobytes()).hexdigest()
    return ["--profile", "b0", "--data", str(data), "--data-sha256",
            hashlib.sha256(data.read_bytes()).hexdigest(), "--init", str(init),
            "--init-sha256", tensor_hash, "--recipe", str(recipe), "--out", str(tmp_path / "out")]


def test_module_help():
    result = subprocess.run([sys.executable, "-m", "p05_detect.train", "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--init-sha256" in result.stdout
    assert "--dry-run" in result.stdout


@pytest.mark.parametrize("profile,arm,nwd", [("b0", None, None), ("e2_carafe", "E2", None),
                                              ("e3_combined", "E3", {"lam": 0.5, "C": 61})])
def test_dry_run_reports_verified_plan(inputs, capsys, monkeypatch, profile, arm, nwd):
    module = importlib.import_module("p05_detect.train")
    from p05_detect.experiments.b0_trainer import B0Trainer
    from p05_detect.engine.trainer import P05DetectionTrainer
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run must not construct a trainer")
    monkeypatch.setattr(B0Trainer, "__init__", forbidden)
    monkeypatch.setattr(P05DetectionTrainer, "__init__", forbidden)
    inputs[1] = profile
    assert module.main(inputs + ["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["profile"] == profile
    assert plan["arm"] == arm
    assert plan["nwd"] == nwd
    assert plan["overrides"]["seed"] == 101
    assert plan["overrides"]["device"] == "cpu"
    assert plan["overrides"]["imgsz"] == 640
    assert plan["overrides"]["optimizer"] == "SGD"
    assert plan["overrides"]["cos_lr"] is True
    assert plan["overrides"]["lr0"] == 0.002
    assert plan["overrides"]["pretrained"] is False
    assert not Path(inputs[-1]).exists()


@pytest.mark.parametrize("flag", ["--data-sha256", "--init-sha256"])
def test_bad_hash_fails_closed(inputs, flag, capsys):
    module = importlib.import_module("p05_detect.train")
    inputs[inputs.index(flag) + 1] = "0" * 64
    with pytest.raises(SystemExit) as exc:
        module.main(inputs + ["--dry-run"])
    assert exc.value.code == 2
    assert "hash mismatch" in capsys.readouterr().err


@pytest.mark.parametrize("key", ["profile", "arm", "model", "data", "pretrained", "resume", "project", "name", "seed", "device", "cfg", "task", "mode", "distill_model", "init", "out"])
def test_recipe_identity_override_rejected(inputs, key, capsys):
    module = importlib.import_module("p05_detect.train")
    Path(inputs[inputs.index("--recipe") + 1]).write_text(json.dumps({key: "override"}))
    with pytest.raises(SystemExit) as exc:
        module.main(inputs + ["--dry-run"])
    assert exc.value.code == 2
    assert key in capsys.readouterr().err


@pytest.mark.parametrize("recipe", [[], {"imgsz": 320}, {"multi_scale": 0.5}, {"epochs": "bad"}, {"epochs": 0}, {"unknown": 1}])
def test_invalid_recipe_rejected(inputs, recipe):
    module = importlib.import_module("p05_detect.train")
    Path(inputs[inputs.index("--recipe") + 1]).write_text(json.dumps(recipe))
    with pytest.raises(SystemExit) as exc:
        module.main(inputs + ["--dry-run"])
    assert exc.value.code == 2


@pytest.mark.parametrize("flag", ["--data", "--init", "--recipe"])
def test_missing_local_file_rejected(inputs, flag, capsys):
    module = importlib.import_module("p05_detect.train")
    inputs[inputs.index(flag) + 1] += ".missing"
    with pytest.raises(SystemExit) as exc:
        module.main(inputs + ["--dry-run"])
    assert exc.value.code == 2
    assert "local file" in capsys.readouterr().err


def test_b0_existing_trainer_defaults_unchanged():
    from p05_detect.experiments.b0_trainer import B0Trainer
    assert B0Trainer.init_state_path is None
    assert B0Trainer.expected_hash is None


def test_init_file_hash_is_not_tensor_hash(inputs, capsys):
    module = importlib.import_module("p05_detect.train")
    init = Path(inputs[inputs.index("--init") + 1])
    inputs[inputs.index("--init-sha256") + 1] = hashlib.sha256(init.read_bytes()).hexdigest()
    with pytest.raises(SystemExit) as exc:
        module.main(inputs + ["--dry-run"])
    assert exc.value.code == 2
    assert "init tensor state hash mismatch" in capsys.readouterr().err


def test_init_uses_safe_tensor_loader(inputs, monkeypatch, capsys):
    module = importlib.import_module("p05_detect.train")
    original = torch.load
    loads = []
    def load(*args, **kwargs):
        loads.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "load", load)
    assert module.main(inputs + ["--dry-run"]) == 0
    assert loads == [{"map_location": "cpu", "weights_only": True}]


@pytest.mark.parametrize("content", ["{", '{"lr0": 1e999}', '{"cos_lr": "yes"}'])
def test_malformed_recipe_is_actionable(inputs, content, capsys):
    module = importlib.import_module("p05_detect.train")
    Path(inputs[inputs.index("--recipe") + 1]).write_text(content)
    with pytest.raises(SystemExit) as exc:
        module.main(inputs + ["--dry-run"])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_dataset_download_directive_rejected(inputs, capsys):
    module = importlib.import_module("p05_detect.train")
    data = Path(inputs[inputs.index("--data") + 1])
    data.write_text(data.read_text() + "download: https://example.invalid/data.zip\n")
    inputs[inputs.index("--data-sha256") + 1] = hashlib.sha256(data.read_bytes()).hexdigest()
    with pytest.raises(SystemExit) as exc:
        module.main(inputs + ["--dry-run"])
    assert exc.value.code == 2
    assert "download" in capsys.readouterr().err


@pytest.mark.parametrize("profile", ["b0", "e2_carafe", "e3_combined"])
def test_dispatch_uses_existing_trainer_and_restores_class(inputs, monkeypatch, profile, capsys):
    # Only replace the training boundary: no real trainer construction, optimizer or GPU operation.
    module = importlib.import_module("p05_detect.train")
    from p05_detect.experiments.b0_trainer import B0Trainer
    from p05_detect.engine.trainer import P05DetectionTrainer
    inputs[1] = profile
    inputs += ["--seed", "202"]
    selected = B0Trainer if profile == "b0" else P05DetectionTrainer
    events = []
    def construct(self, *, overrides):
        assert type(self) is selected
        assert overrides["seed"] == 202
        assert overrides["pretrained"] is False
        assert overrides["resume"] is False
        assert overrides["model"].endswith("yolo26s.yaml")
        if profile == "b0":
            assert self.init_state_path == inputs[inputs.index("--init") + 1]
            assert self.expected_hash == inputs[inputs.index("--init-sha256") + 1]
        else:
            assert self.arm == ("E2" if profile == "e2_carafe" else "E3")
            assert self.seed == 202
            assert self.nwd == (None if profile == "e2_carafe" else {"lam": 0.5, "C": 61})
        events.append("construct")
    def no_training(self):
        events.append("train-boundary")
    monkeypatch.setattr(selected, "__init__", construct)
    monkeypatch.setattr(selected, "train", no_training)
    assert module.main(inputs) == 0
    assert events == ["construct", "train-boundary"]
    assert B0Trainer.init_state_path is None
    assert P05DetectionTrainer.arm is None
