import pytest
from p05_detect.losses.nwd_c import compute_c


def test_guarded_rule_keeps_12p8_inside_band():
    r = compute_c([(0, 0, 12, 12)] * 5, [(640, 640)] * 5)
    assert r["s_med"] == pytest.approx(12.0) and r["C"] == 12.8 and r["rule"] == "default_12.8"


def test_uses_median_geometric_side_after_resize640():
    # 1280x1280 -> r=.5 ; 100x100 box -> 50 px ; band [6.4, 25.6] exceeded -> C = s_med
    r = compute_c([(0, 0, 100, 100)] * 3, [(1280, 1280)] * 3)
    assert r["s_med"] == pytest.approx(50.0) and r["C"] == pytest.approx(50.0) and r["rule"] == "median_outside_band"


def test_band_edges_inclusive():
    assert compute_c([(0, 0, 6.4, 6.4)], [(640, 640)])["C"] == 12.8
    assert compute_c([(0, 0, 25.6, 25.6)], [(640, 640)])["C"] == 12.8
    assert compute_c([(0, 0, 6.39, 6.39)], [(640, 640)])["C"] == pytest.approx(6.39)


def test_empty_train_is_error():
    with pytest.raises(ValueError): compute_c([], [])
