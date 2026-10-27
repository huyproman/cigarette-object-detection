"""Analytic oracles for the P05 detection evaluator. Expected values are hand-derived."""
import math
import numpy as np
import pytest
from p05_detect.eval.evaluator import Image, Pred, evaluate, validate_pred, size_bin

B1 = (0., 0., 10., 10.)
B2 = (100., 100., 110., 110.)
FAR = (500., 500., 510., 510.)


def img(i, gt=(), ignore=(), status="positive", w=640, h=640, cls=None):
    gt = [(c, b) for c, b in enumerate_gt(gt, cls)]
    return Image(i, w, h, gt, [(0, b) for b in ignore], status)


def enumerate_gt(gt, cls):
    return [((cls[k] if cls else 0), b) for k, b in enumerate(gt)]


def P(i, s, b, c=0, k=0):
    return Pred(i, c, s, b, k)


def ev(images, preds, **kw):
    return evaluate(images, preds, num_classes=kw.pop("num_classes", 1), **kw)


def test_perfect_two_objects():
    r = ev([img("a", [B1, B2])], [P("a", .9, B1, k=0), P("a", .8, B2, k=1)])
    assert r["AP50"] == 1 and r["AP50:95"] == 1
    assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 0)


def test_no_predictions():
    r = ev([img("a", [B1, B2])], [])
    assert r["AP50"] == 0 and (r["tp"], r["fp"], r["fn"]) == (0, 0, 2)


def test_duplicate_is_fp_but_ap_one():
    r = ev([img("a", [B1])], [P("a", .9, B1), P("a", .8, B1, k=1)])
    assert r["AP50"] == 1 and (r["tp"], r["fp"], r["fn"]) == (1, 1, 0)
    assert r["precision"] == .5 and r["recall"] == 1


def test_fp_before_tp_ap_half():
    r = ev([img("a", [B1])], [P("a", .9, FAR), P("a", .8, B1, k=1)])
    assert r["AP50"] == pytest.approx(.5) and (r["tp"], r["fp"]) == (1, 1)


def test_two_gt_one_pred_51_over_101():
    r = ev([img("a", [B1, B2])], [P("a", .9, B1)])
    assert r["AP50"] == pytest.approx(51 / 101) and r["fn"] == 1


def test_wrong_class_and_wrong_location():
    r = ev([img("a", [B1], cls=[0])], [P("a", .9, B1, c=1)], num_classes=2)
    assert r["per_class_AP50"][0] == 0 and r["tp"] == 0
    r = ev([img("a", [B1])], [P("a", .9, FAR)])
    assert r["AP50"] == 0 and r["fp"] == 1 and r["fn"] == 1


def test_iou_thresholds_independent_matching():
    # IoU of (0,0,10,10) with (0,0,10,6) is .6 -> TP at .5/.55/.6 only (3 of 10 thresholds)
    r = ev([img("a", [B1])], [P("a", .9, (0., 0., 10., 6.))])
    assert r["AP50"] == 1
    assert r["AP50:95"] == pytest.approx(0.3, abs=1e-9)


def test_iou_exactly_at_threshold_counts():
    r = ev([img("a", [B1])], [P("a", .9, (0., 0., 10., 5.))])  # IoU exactly .5
    assert r["AP50"] == 1


def test_confirmed_negative_counts_fp_unknown_empty_excluded():
    r = ev([img("n", [], status="confirmed_negative")], [P("n", .9, B1)])
    assert r["fp"] == 1 and r["fn"] == 0
    r = ev([img("u", [], status="unknown_empty")], [P("u", .9, B1)])
    assert r["fp"] == 0 and r["excluded_images"] == 1
    r = ev([img("m", [B1], status="missing_suspected")], [P("m", .9, B1)])
    assert r["tp"] == 0 and r["fn"] == 0 and r["excluded_images"] == 1


def test_valid_gt_overlapping_ignore_keeps_tp():
    r = ev([img("a", [B1], ignore=[B1])], [P("a", .9, B1)])
    assert (r["tp"], r["fp"], r["ignored_preds"]) == (1, 0, 0)


def test_ignore_neutralizes_unmatched_pred_only():
    r = ev([img("a", [B1], ignore=[B2])], [P("a", .9, B1), P("a", .8, B2, k=1)])
    assert (r["tp"], r["fp"], r["ignored_preds"]) == (1, 0, 1)
    assert r["precision"] == 1


def test_duplicate_around_valid_gt_overlapping_ignore_stays_fp():
    r = ev([img("a", [B1], ignore=[B1])], [P("a", .9, B1), P("a", .8, B1, k=1)])
    assert (r["tp"], r["fp"], r["ignored_preds"]) == (1, 1, 0)


def test_highest_iou_gt_chosen_and_confidence_order():
    g1, g2 = (0., 0., 10., 10.), (0., 0., 10., 8.)
    r = ev([img("a", [g1, g2])], [P("a", .9, g2)])
    assert r["tp"] == 1  # matched to g2 (IoU 1), g1 left as FN
    assert r["fn"] == 1


def test_confidence_order_beats_iou_priority():
    # GT g; first (score .9) IoU .6, second (score .8) IoU 1.0. P05 matches the first one.
    g = (0., 0., 10., 10.)
    r = ev([img("a", [g])], [P("a", .9, (0., 0., 10., 6.), k=0), P("a", .8, g, k=1)], iou_thresholds=[.75])
    # at .75 the .9 pred (IoU .6) cannot match; greedy-by-score then lets the .8 pred match
    assert (r["tp"], r["fp"]) == (1, 1)
    r = ev([img("a", [g])], [P("a", .9, (0., 0., 10., 8.), k=0), P("a", .8, g, k=1)], iou_thresholds=[.5])
    assert (r["tp"], r["fp"]) == (1, 1)
    assert r["tp_list"][0] == ("a", 0)  # the high-score IoU .8 pred is the TP, not the IoU 1 pred


def test_equal_score_stable_order():
    imgs = [img("a", [B1])]
    a = ev(imgs, [P("a", .5, FAR, k=0), P("a", .5, B1, k=1)])
    b = ev(imgs, [P("a", .5, B1, k=1), P("a", .5, FAR, k=0)])
    assert a["AP50"] == b["AP50"] == pytest.approx(.5)


def test_max_det_cap_300():
    preds = [P("a", 1 - i * 1e-4, FAR, k=i) for i in range(300)] + [P("a", .001, B1, k=300)]
    r = ev([img("a", [B1])], preds)
    assert r["tp"] == 0 and r["fp"] == 300


def test_size_bins_half_open():
    assert size_bin(1023.99) == "small" and size_bin(1024) == "medium"
    assert size_bin(9215.99) == "medium" and size_bin(9216) == "large"


def test_size_ap_reports_denominator():
    small = (0., 0., 20., 20.)   # 400 -> small
    large = (0., 0., 200., 200.)  # 40000 -> large
    r = ev([img("a", [small]), img("b", [large])], [P("a", .9, small), P("b", .8, large)])
    assert r["size"]["small"]["n_gt"] == 1 and r["size"]["large"]["n_gt"] == 1
    assert r["size"]["small"]["AP50"] == 1 and r["size"]["medium"]["n_gt"] == 0


def test_resize640_bins_differ_from_source():
    # 20x20 box in a 1280x1280 image -> 10x10 after resize640 (small either way); 40x40 in 1280 -> 20x20
    g = (0., 0., 40., 40.)  # source area 1600 medium, resized 400 small
    r = ev([img("a", [g], w=1280, h=1280)], [P("a", .9, g)])
    assert r["size"]["medium"]["n_gt"] == 1
    assert r["size_resize640"]["small"]["n_gt"] == 1


@pytest.mark.parametrize("bad", [
    dict(box=(0, 0, float("nan"), 5)), dict(box=(5, 5, 5, 9)), dict(box=(0, 0, 1, 1), score=1.5),
    dict(box=(0, 0, 1, 1), cls=3), dict(box=(0, 0, 1, 1), space="norm_xywh"), dict(box=(0, 0, 1, 1), score=-.1),
])
def test_invalid_predictions_rejected(bad):
    p = Pred("a", bad.get("cls", 0), bad.get("score", .5), bad["box"], 0, bad.get("space", "original_pixel_xyxy"))
    with pytest.raises(ValueError):
        validate_pred(p, num_classes=1, known_images={"a"})
    with pytest.raises(ValueError):
        validate_pred(Pred("zzz", 0, .5, (0, 0, 1, 1), 0), num_classes=1, known_images={"a"})


def test_confusion_accounting():
    r = ev([img("a", [B1, B2])], [P("a", .9, B1), P("a", .8, FAR, k=1)])
    cm = r["confusion"]
    assert cm == {"tp": 1, "fp": 1, "fn": 1}
    assert r["tp"] + r["fn"] == 2 and r["tp"] + r["fp"] == 2


def test_operating_point_threshold_025():
    r = ev([img("a", [B1])], [P("a", .2, B1)])
    assert r["AP50"] == 1 and r["op_tp"] == 0 and r["op_fn"] == 1


def test_deterministic_repeat():
    imgs = [img("a", [B1, B2])]
    preds = [P("a", .9, B1), P("a", .5, FAR, k=1), P("a", .7, B2, k=2)]
    assert ev(imgs, preds) == ev(imgs, list(reversed(preds)))
