"""Hash-verified, local-only research entry point using the existing trainers."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
from unittest.mock import patch


def _file(value, label):
    path = Path(value).expanduser()
    if "://" in value or not path.is_file():
        raise ValueError(f"{label}: provide an existing local file: {value}")
    return path.resolve()


def _hash(actual, expected, label):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError(f"{label}: expected a 64-character SHA-256 hex digest")
    if actual != expected.lower():
        raise ValueError(f"{label} hash mismatch: expected {expected.lower()}, got {actual}")


def _recipe(path):
    from ultralytics.cfg import get_cfg

    recipe = json.loads(path.read_text())
    if not isinstance(recipe, dict):
        raise ValueError("recipe must be a JSON object of training arguments")
    # Only training controls; identity, alternate models and external config are not recipe inputs.
    allowed = set("epochs time patience batch imgsz save save_period cache workers optimizer verbose deterministic "
                  "rect cos_lr close_mosaic amp fraction freeze multi_scale compile channels_last val split "
                  "save_json conf iou max_det plots lr0 lrf momentum weight_decay warmup_epochs warmup_momentum "
                  "warmup_bias_lr box cls dfl nbs hsv_h hsv_s hsv_v degrees translate scale shear perspective "
                  "flipud fliplr bgr mosaic mixup cutmix copy_paste copy_paste_mode auto_augment erasing".split())
    invalid = set(recipe) - allowed
    if invalid:
        raise ValueError(f"recipe cannot override identity or unsupported fields: {', '.join(sorted(invalid))}")
    if any(isinstance(v, float) and not math.isfinite(v) for v in recipe.values()):
        raise ValueError("recipe numbers must be finite")
    if recipe.get("imgsz", 640) != 640 or recipe.get("multi_scale", 0) != 0 or recipe.get("rect", False):
        raise ValueError("this interface requires fixed imgsz=640, multi_scale=0, rect=false")
    get_cfg(overrides=recipe)  # Upstream strict argument types and fractional ranges.
    for key in ("epochs", "batch", "time", "nbs"):
        if key in recipe and (isinstance(recipe[key], bool) or not isinstance(recipe[key], (int, float)) or recipe[key] <= 0):
            raise ValueError(f"recipe {key} must be positive")
    if "optimizer" in recipe and recipe["optimizer"] not in {
        "auto", "SGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "MuSGD"
    }:
        raise ValueError("recipe optimizer is not supported by the installed trainer")
    return recipe


def _dataset(path):
    import yaml

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("--data must be a local YAML file")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "download" in data:
        raise ValueError("data YAML must be an object without a download directive")
    names = data.get("names")
    if not isinstance(names, (list, dict)) or len(names) != 1 or data.get("nc", 1) != 1 or data.get("channels", 3) != 3:
        raise ValueError("stored B0 initialization requires data names for exactly one class and channels=3")
    root = data.get("path", str(path.parent))
    if not isinstance(root, str) or "://" in root:
        raise ValueError("data path must be local")
    root = Path(root).expanduser()
    if not root.is_absolute():
        raise ValueError("data YAML path must be absolute (or omitted); use local absolute split paths")
    for split in ("train", "val", "test", "minival"):
        values = data.get(split)
        if not values and split not in {"train", "val"}:
            continue
        values = values if isinstance(values, list) else [values]
        for value in values:
            if not isinstance(value, str) or "://" in value or not (root / value).exists():
                raise ValueError(f"data {split}: provide an existing local path, got {value!r}")
            target = root / value
            if target.is_file():
                if target.suffix.lower() != ".txt":
                    raise ValueError(f"data {split}: expected an image directory or local .txt image list")
                for image in target.read_text().splitlines():
                    image_path = target.parent / image[2:] if image.startswith("./") else Path(image)
                    if image and ("://" in image or not image_path.is_file()):
                        raise ValueError(f"data {split}: missing local image in {target}: {image}")


def _offline(*args, **kwargs):
    raise RuntimeError("network/downloads are disabled; provide all inputs locally")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("b0", "e2_carafe", "e3_combined"), required=True)
    parser.add_argument("--data", required=True, help="local single-class detection YAML")
    parser.add_argument("--data-sha256", required=True, help="SHA-256 of the YAML file bytes")
    parser.add_argument("--init", required=True, help="local stored B0 tensor state dictionary")
    parser.add_argument("--init-sha256", required=True, help="tensor state_hash digest, NOT the .pt file hash")
    parser.add_argument("--recipe", required=True, help="JSON object of Ultralytics training arguments")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cpu", help="cpu, mps, or one numeric CUDA device; no distributed launch")
    parser.add_argument("--out", required=True, help="new or empty local output directory")
    parser.add_argument("--dry-run", action="store_true", help="validate and print JSON; do not construct a trainer or train")
    args = parser.parse_args(argv)
    os.environ["YOLO_OFFLINE"] = "true"
    try:
        data = _file(args.data, "--data")
        init = _file(args.init, "--init")
        recipe_path = _file(args.recipe, "--recipe")
        _hash(hashlib.sha256(data.read_bytes()).hexdigest(), args.data_sha256, "data YAML")
        if args.seed < 0 or not re.fullmatch(r"cpu|mps|[0-9]+", args.device):
            raise ValueError("use a nonnegative --seed and --device cpu, mps, or one CUDA index")
        out = Path(args.out).expanduser().resolve()
        if "://" in args.out or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
            raise ValueError("--out must be a new or empty local directory")
        if any(p.exists() and not p.is_dir() for p in out.parents):
            raise ValueError("--out parent must be a directory")

        import torch
        from p05_detect.experiments.init_state import state_hash
        state = torch.load(init, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
            raise ValueError("--init must contain a nonempty tensor state dictionary, not a model/checkpoint object")
        _hash(state_hash(state), args.init_sha256, "init tensor state")
        del state
        recipe = _recipe(recipe_path)
        _dataset(data)
        from ultralytics.utils import ROOT
        model = ROOT / "cfg/models/26/yolo26s.yaml"
        if not (model.parent / "yolo26.yaml").is_file():
            raise ValueError("installed Ultralytics is missing the local YOLO26 model YAML")
        arm = {"b0": None, "e2_carafe": "E2", "e3_combined": "E3"}[args.profile]
        nwd = {"lam": 0.5, "C": 61} if arm == "E3" else None
        overrides = dict(recipe, model=str(model), data=str(data), pretrained=False, resume=False,
                         project=str(out.parent), name=out.name, exist_ok=True, seed=args.seed,
                         device=args.device, imgsz=640, task="detect", mode="train")
        plan = dict(profile=args.profile, arm=arm, nwd=nwd, init=str(init),
                    init_sha256=args.init_sha256.lower(), data_sha256=args.data_sha256.lower(),
                    recipe=str(recipe_path), overrides=overrides, dry_run=args.dry_run)
    except Exception as exc:
        parser.error(str(exc))
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0

    from p05_detect.experiments.b0_trainer import B0Trainer
    from p05_detect.engine.trainer import P05DetectionTrainer
    trainer = B0Trainer if arm is None else P05DetectionTrainer
    attrs = dict(init_state_path=str(init), expected_hash=args.init_sha256.lower()) if arm is None else dict(
        arm=arm, seed=args.seed, init_path=str(init), init_hash=args.init_sha256.lower(), nwd=nwd)
    load = torch.load
    def safe_load(*a, **kw):
        if a and isinstance(a[0], (str, Path)) and Path(a[0]).resolve() == init:
            kw["weights_only"] = True
        return load(*a, **kw)
    with ExitStack() as stack:
        for key, value in attrs.items():
            stack.enter_context(patch.object(trainer, key, value))
        stack.enter_context(patch("torch.load", safe_load))
        stack.enter_context(patch("socket.socket.connect", _offline))
        stack.enter_context(patch("socket.socket.connect_ex", _offline))
        stack.enter_context(patch("socket.getaddrinfo", _offline))
        stack.enter_context(patch("ultralytics.utils.downloads.safe_download", _offline))
        trainer(overrides=overrides).train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
