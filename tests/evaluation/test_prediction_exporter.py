import torch, pytest
from pathlib import Path
from PIL import Image
from p05_detect.eval.evaluator import to_letterbox, from_letterbox
from p05_detect.eval.exporter import predict_records

INIT = Path(__file__).parents[2] / "runtime/init_states/B0_seed101_init_state.pt"


@pytest.mark.parametrize("w,h", [(1280, 720), (390, 280), (640, 640), (173, 280), (1080, 1440)])
def test_letterbox_round_trip(w, h):
    b = (10.5, 20.25, 99.0, 77.5)
    r = from_letterbox(to_letterbox(b, w, h), w, h)
    assert all(abs(x - y) < 1e-9 for x, y in zip(b, r))


def test_letterbox_geometry_1280x720():
    # r=.5, resized 640x360, pad y=140
    assert to_letterbox((0, 0, 100, 100), 1280, 720) == (0, 140, 50, 190)


@pytest.mark.skipif(not INIT.exists(), reason="init state not built")
def test_head_routing_changes_output_path_and_records_valid(tmp_path):
    from ultralytics import YOLO
    y = YOLO("yolo26s.yaml"); y.model.load_state_dict(torch.load(INIT)) if False else None
    from ultralytics.nn.tasks import DetectionModel
    m = DetectionModel("yolo26s.yaml", nc=1, verbose=False); m.load_state_dict(torch.load(INIT)); y.model = m
    p = tmp_path / "a.jpg"; Image.new("RGB", (500, 300), (120, 90, 60)).save(p)
    a, meta = predict_records(y, [p], "one2many", conf=0.0001, device="cpu")
    assert y.predictor.model.end2end is False
    b, _ = predict_records(y, [p], "one2one", conf=0.0001, device="cpu")
    assert y.predictor.model.end2end is True
    for r in a + b:
        assert 0 <= r.score <= 1 and r.box[2] > r.box[0] and 0 <= r.box[0] and r.box[2] <= 500 and r.box[3] <= 300
    assert meta[0][1:] == (500, 300)


def test_degenerate_after_clip_is_dropped_and_counted():
    from p05_detect.eval.exporter import clip_or_drop
    assert clip_or_drop((600., 10., 700., 50.), 500, 300) is None          # fully outside -> zero width
    assert clip_or_drop((10., 10., 10.0, 50.), 500, 300) is None
    assert clip_or_drop((-5., 10., 50., 50.), 500, 300) == (0., 10., 50., 50.)


BEST = Path(__file__).parents[2] / "runtime/runs/B0-101/weights/best.pt"


@pytest.mark.skipif(not BEST.exists(), reason="no trained checkpoint")
def test_heads_actually_differ_on_trained_checkpoint():
    import json
    val = [json.loads(l)["path"] for l in open(Path(__file__).parents[2] / "runtime/dataset/split_manifest.jsonl") if '"split": "VAL"' in l][:24]
    a, _ = predict_records(BEST, val, "one2many", device=0)
    b, _ = predict_records(BEST, val, "one2one", device=0)
    assert len(a) != len(b), "one2many(NMS) and one2one must not produce identical output"
    assert len(b) <= 24 * 300
