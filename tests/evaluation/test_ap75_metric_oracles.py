"""Regression: AP75 must be measured (not the IoU threshold constant); oracle cases for the evaluator."""
from p05_detect.eval.evaluator import Image, Pred, evaluate

def ap(images, preds, thr): return evaluate(images, preds, iou_thresholds=[thr])["AP50:95"]
def img(gt, ign=()): return Image("a", 100, 100, [(0, g) for g in gt], [(0, g) for g in ign], "positive")
G = (10, 10, 50, 50)
def P(box, s=0.9, i=0): return Pred("a", 0, s, box, i)

def test_perfect_and_empty():
    assert ap([img([G])], [P(G)], 0.75) == 1.0
    assert ap([img([G])], [], 0.75) == 0.0

def test_localization_below_and_above_075():
    below = (10, 10, 50, 50 - 40 * 0.30)   # IoU 0.70
    above = (10, 10, 50, 50 - 40 * 0.20)   # IoU 0.80
    assert ap([img([G])], [P(below)], 0.75) == 0.0 and ap([img([G])], [P(below)], 0.5) == 1.0
    assert ap([img([G])], [P(above)], 0.75) == 1.0

def test_duplicate_is_fp_and_ap_not_constant():
    d = ap([img([G])], [P(G, .9), P(G, .8, 1)], 0.75)       # second is FP after the TP -> envelope keeps AP 1.0
    assert d == 1.0
    d2 = ap([img([G])], [P(G, .8), P((60, 60, 90, 90), .9, 1)], 0.75)   # FP ranked first -> AP 0.5
    assert abs(d2 - 0.5) < 1e-9
    assert d2 != 0.75

def test_ignore_region_neutralises_fp():
    a = ap([img([G], ign=[(60, 60, 90, 90)])], [P(G, .8), P((60, 60, 90, 90), .9, 1)], 0.75)
    assert a == 1.0
