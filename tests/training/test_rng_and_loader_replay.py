import hashlib, numpy as np, pytest, torch
from pathlib import Path
from PIL import Image
from p05_detect.experiments.rng import streams, aug_recipe_key, image_order, resolved_aug_config, replay_probe


def test_streams_deterministic_and_distinct():
    a, b = streams(101), streams(101)
    assert a == b
    assert len({a["model_init"], a["sampler"], a["augmentation"], a["worker_base"]}) == 4
    assert streams(101)["sampler"] == 101                      # sampler seed = run seed
    for s1, s2 in ((101, 202), (202, 303)):
        assert all(streams(s1)[k] != streams(s2)[k] for k in ("model_init", "augmentation", "worker_base"))


def test_aug_recipe_key_formula():
    k = aug_recipe_key(101, 3, "img_a", 0)
    assert k == hashlib.sha256(b"101|3|img_a|0").hexdigest()
    assert k != aug_recipe_key(101, 4, "img_a", 0) != aug_recipe_key(202, 3, "img_a", 0)


def test_image_order_independent_of_other_rng_consumption():
    ids = [f"i{k}" for k in range(50)]
    o1 = image_order(ids, 101, 0)
    torch.manual_seed(5); torch.rand(1000); np.random.rand(100)
    assert image_order(ids, 101, 0) == o1 != image_order(ids, 101, 1)
    assert sorted(o1) == sorted(ids)


def test_resolved_aug_config_pins_defaults_and_hash():
    c = resolved_aug_config()
    assert c["mosaic"] == 1.0 and c["fliplr"] == 0.5 and c["hsv_h"] == 0.015 and c["close_mosaic"] == 10
    assert len(c["sha256"]) == 64 and c == resolved_aug_config()


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("fx"); (d / "images").mkdir(); (d / "labels").mkdir()
    g = np.random.default_rng(0)
    for i in range(12):
        Image.fromarray(g.integers(0, 255, (320, 400, 3), dtype=np.uint8)).save(d / "images" / f"{i}.jpg")
        (d / "labels" / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (d / "list.txt").write_text("\n".join(str(d / "images" / f"{i}.jpg") for i in range(12)))
    (d / "data.yaml").write_text(f"path: {d}\ntrain: list.txt\nval: list.txt\nnames:\n  0: cigarette\n")
    return d


def test_loader_replay_same_seed_identical_and_robust_to_model_construction(fixture_dir):
    base = replay_probe(fixture_dir / "data.yaml", seed=101, build_model_first=False)
    again = replay_probe(fixture_dir / "data.yaml", seed=101, build_model_first=False)
    after_model = replay_probe(fixture_dir / "data.yaml", seed=101, build_model_first=True)
    other = replay_probe(fixture_dir / "data.yaml", seed=202, build_model_first=False)
    assert base["order"] == again["order"] == after_model["order"] != other["order"]
    assert base["batch_hashes"] == again["batch_hashes"]
    # recorded, not assumed: whether augmented pixels replay across model construction order
    import json, os
    meas = dict(order_replays_across_model_construction=base["order"] == after_model["order"],
                augmented_pixels_replay_same_seed=base["batch_hashes"] == again["batch_hashes"],
                augmented_pixels_replay_after_model_construction=base["batch_hashes"] == after_model["batch_hashes"],
                different_seed_changes_pixels=base["batch_hashes"] != other["batch_hashes"])
    if os.environ.get("P05_RNG_EVIDENCE"):
        Path(os.environ["P05_RNG_EVIDENCE"]).write_text(json.dumps(meas, indent=1))
    assert meas["order_replays_across_model_construction"] and meas["different_seed_changes_pixels"]
