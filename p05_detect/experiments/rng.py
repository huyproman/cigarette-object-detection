"""RNG separation and augmentation provenance. Guarantees are stated by tests, not assumed.
- model init: stored per-seed state (init_state.py); its RNG stream id is derived here for the record.
- sampler: seed = run seed (upstream InfiniteDataLoader generator seeded from it; independent of global RNG).
- augmentation recipe key = SHA256(seed|epoch|image_id|slot): a PROVENANCE key. Upstream augmentations draw from
  worker-local numpy/random state seeded by the loader generator; they do NOT consume this key, so equality of
  recipe keys does not imply pixel-identical augmented batches. replay_probe measures what does replay."""
import hashlib, json
from pathlib import Path


def _h(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def streams(seed: int) -> dict:
    return dict(run_seed=seed, model_init=_h(seed, "model_init"), sampler=seed, augmentation=_h(seed, "augmentation"),
                worker_base=_h(seed, "worker"))


def aug_recipe_key(seed, epoch, image_id, slot) -> str:
    return hashlib.sha256(f"{seed}|{epoch}|{image_id}|{slot}".encode()).hexdigest()


def image_order(image_ids, seed, epoch):
    return sorted(image_ids, key=lambda i: hashlib.sha256(f"{seed}|{epoch}|{i}|order".encode()).hexdigest())


def resolved_aug_config():
    from ultralytics.cfg import get_cfg
    c = get_cfg()
    keys = ["mosaic", "mixup", "cutmix", "copy_paste", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
            "flipud", "fliplr", "bgr", "erasing", "auto_augment", "close_mosaic", "multi_scale"]
    d = {k: getattr(c, k) for k in keys if hasattr(c, k)}
    d["sha256"] = hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()
    return d


def replay_probe(data_yaml, seed, build_model_first=False, batches=3, workers=2):
    """Run the REAL upstream train dataloader and record image order + hashes of augmented batches."""
    import numpy as np, torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset, build_dataloader
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.torch_utils import init_seeds
    cfg = get_cfg(overrides=dict(data=str(data_yaml), imgsz=160, batch=4, workers=workers, seed=seed, deterministic=True))
    init_seeds(seed + 1, deterministic=True)
    if build_model_first:
        from ultralytics.nn.tasks import DetectionModel
        DetectionModel("yolo26s.yaml", nc=1, verbose=False)   # consumes global RNG, like real trainer
    data = check_det_dataset(str(data_yaml))
    ds = build_yolo_dataset(cfg, data["train"], 4, data, mode="train")
    dl = build_dataloader(ds, 4, workers, shuffle=True, rank=-1)
    it = iter(dl); order, hashes, base_px = [], [], []
    for _ in range(batches):
        b = next(it)
        order += [Path(f).stem for f in b["im_file"]]
        hashes.append(hashlib.sha256(b["img"].numpy().tobytes()).hexdigest())
    return dict(order=order, batch_hashes=hashes, pixels_equal_to_base=None)
