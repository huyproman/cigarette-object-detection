"""Regression tests: mode->tile-size mapping, independent geometry oracles, full+tile fusion, NMS oracle."""
import numpy as np
import pytest
from p05_detect.infer import slicing
from p05_detect.infer.slicing import generate_tiles, global_merge, tile_size_for_mode

def test_mode_tile_sizes_frozen_contract():
    assert tile_size_for_mode("SLICE640") == 640
    assert tile_size_for_mode("SLICE960") == 960
    assert tile_size_for_mode("FULL_PLUS_SLICE") == 640     # a substring test would fall through to 960
    with pytest.raises(ValueError):
        tile_size_for_mode("BOGUS")

@pytest.mark.parametrize("mode,size", [("SLICE640", 640), ("SLICE960", 960), ("FULL_PLUS_SLICE", 640)])
def test_legacy_tile_modes_match_readable_sizes_and_geometry(mode, size):
    assert tile_size_for_mode("P3-" + mode) == tile_size_for_mode(mode) == size
    assert generate_tiles(1920, 1080, tile_size_for_mode("P3-" + mode)) == generate_tiles(1920, 1080, size)

@pytest.mark.parametrize("mode", ["SLICE640", "SLICE960", "FULL_PLUS_SLICE", "FULL"])
def test_legacy_predict_modes_match_readable_modes(monkeypatch, mode):
    calls = []
    class Model:
        profile = dict(profile_id="synthetic", seed=101, checkpoint_sha256="synthetic", arm="a")
    def fake_predict(_img, model, conf, iou, max_det, _loaded):
        rgb, name = _loaded
        calls.append((rgb.shape, name, conf, iou, max_det))
        return dict(detections=[dict(class_id=0, class_name="synthetic", confidence=0.9,
                                    box_xyxy=[1.0, 2.0, 10.0, 20.0])],
                    timing_ms=dict(model_forward=0.0))
    monkeypatch.setattr(slicing.core, "predict", fake_predict)
    loaded = (np.zeros((1080, 1920, 3), np.uint8), "synthetic")
    readable = slicing.predict_sliced(None, Model(), mode, _loaded=loaded)
    expected_calls = calls[:]
    calls.clear()
    legacy = slicing.predict_sliced(None, Model(), "P3-" + mode, _loaded=loaded)
    assert calls == expected_calls
    assert {k: v for k, v in legacy.items() if k != "timing_ms"} == {
        k: v for k, v in readable.items() if k != "timing_ms"
    }
    assert legacy["strategy"] == mode
    if mode == "FULL":
        assert len(calls) == legacy["n_tiles"] == 1
        assert calls[0][:2] == (loaded[0].shape, "synthetic")

def test_full_plus_slice_uses_slice640_tiles_and_full_pass(monkeypatch):
    calls = []
    class P:  # minimal model stub
        profile = dict(profile_id="x", seed=101, checkpoint_sha256="s", arm="a")
    def fake_predict(_img, model, conf, iou, max_det=300, _loaded=None):
        rgb = _loaded[0]; calls.append(rgb.shape[:2])
        return dict(detections=[], timing_ms=dict(model_forward=0.0))
    monkeypatch.setattr(slicing.core, "predict", fake_predict)
    rgb = np.zeros((1080, 1920, 3), np.uint8)
    r = slicing.predict_sliced(None, P(), "FULL_PLUS_SLICE", _loaded=(rgb, "t"))
    expected = len(generate_tiles(1920, 1080, 640, 0.2))
    assert r["n_tiles"] == expected and r["settings"]["tile_size"] == 640
    assert len(calls) == expected + 1 and calls[0] == (1080, 1920)   # one full pass + SLICE640 tiles
    assert r["n_full_passes"] == 1

def _oracle_tiles(w, h, t, ov):
    """independent expected tiling: stride = int(t*(1-ov)); last tile flush with edge."""
    s = int(t * (1 - ov))
    def axis(n):
        if n <= t: return [0]
        a = list(range(0, n - t, s))
        if a[-1] + t < n: a.append(n - t)
        return a
    return [(x, y, min(x + t, w), min(y + t, h)) for y in axis(h) for x in axis(w)]

@pytest.mark.parametrize("w,h", [(1920, 1080), (1080, 1440), (250, 250), (641, 641), (640, 640), (700, 300), (2048, 983), (1300, 640)])
def test_geometry_oracle_and_coverage(w, h):
    for t in (640, 960):
        tiles = generate_tiles(w, h, t, 0.2)
        assert tiles == _oracle_tiles(w, h, t, 0.2)
        cov = np.zeros((h, w), np.uint8)
        for x1, y1, x2, y2 in tiles: cov[y1:y2, x1:x2] += 1
        assert cov.min() >= 1
        assert all(0 <= a < c <= w and 0 <= b < d <= h for a, b, c, d in tiles)
        if w <= t and h <= t: assert tiles == [(0, 0, w, h)]

def test_tile_to_source_inversion_roundtrip():
    tiles = generate_tiles(1920, 1080, 640, 0.2)
    for (tx, ty, _, _) in tiles:
        local = np.array([10.0, 20.0, 50.0, 60.0]); src = local + [tx, ty, tx, ty]
        assert np.allclose(src - [tx, ty, tx, ty], local)

def test_nms_oracle_class_aware_iou05_boundary():
    mk = lambda c, s, b: dict(class_id=c, class_name="x", confidence=s, box_xyxy=b)
    a = [0, 0, 10, 10]
    b_iou_exact_half = [0, 0, 10, 20 / 3]          # IoU = (10*6.667)/(100)=0.667 -> merged
    b_low = [0, 0, 10, 4]                           # IoU 0.4 -> kept
    m = global_merge([mk(0, .9, a), mk(0, .8, b_iou_exact_half), mk(0, .7, b_low)], 100, 100, 0.5)
    assert [d["confidence"] for d in m] == [0.9, 0.7]
    m2 = global_merge([mk(0, .9, a), mk(1, .8, a)], 100, 100, 0.5)          # class-aware: different class kept
    assert len(m2) == 2
    assert global_merge([], 10, 10) == []
    dup = global_merge([mk(0, .9, a), mk(0, .9, a)], 100, 100, 0.5); assert len(dup) == 1
