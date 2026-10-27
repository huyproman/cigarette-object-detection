import json, numpy as np, pytest
from pathlib import Path
from p05_detect.eval.evaluator import Image, Pred, evaluate
from p05_detect.eval.paired_stats import prepare, point_metrics, weighted_ap, bootstrap_paired, load_val_images, load_predictions

ROOT = Path(__file__).parents[2]
B0P = ROOT / "runtime/eval/B0-101/predictions_one2many.json"
B0M = ROOT / "runtime/eval/B0-101/metrics_val.json"


def toy():
    imgs = [Image(f"i{k}", 640, 640, [(0, (10., 10., 30., 30.)), (0, (100., 100., 300., 300.))] if k % 2 == 0 else [(0, (50., 50., 70., 70.))]) for k in range(8)]
    preds = []
    for k in range(8):
        for j, (b, s) in enumerate([((10., 10., 30., 31.), .9), ((100., 100., 280., 300.), .7), ((400., 400., 420., 420.), .6), ((50., 50., 70., 72.), .5)]):
            preds.append(Pred(f"i{k}", 0, s - .01 * k, b, j))
    return imgs, preds


def test_weighted_ap_matches_evaluator_exactly_when_unit_weights():
    imgs, preds = toy(); P = prepare(imgs, preds)
    full = evaluate(imgs, preds)
    m = point_metrics(P, np.ones(len(imgs)))
    assert m["all"]["AP50:95"] == pytest.approx(full["AP50:95"], abs=1e-12)
    assert m["all"]["AP50"] == pytest.approx(full["AP50"], abs=1e-12)
    assert m["small"]["AP50:95"] == pytest.approx(full["size_resize640"]["small"]["AP50:95"], abs=1e-12)
    assert m["small"]["n_gt"] == full["size_resize640"]["small"]["n_gt"]


def test_multiplicity_equals_duplicating_images():
    imgs, preds = toy(); P = prepare(imgs, preds)
    mult = np.array([2, 0, 1, 3, 0, 1, 1, 2.])
    dup_imgs, dup_preds = [], []
    for k, c in enumerate(mult.astype(int)):
        for r in range(c):
            dup_imgs.append(Image(f"i{k}#{r}", 640, 640, imgs[k].gt))
            dup_preds += [Pred(f"i{k}#{r}", 0, p.score, p.box, p.index) for p in preds if p.image_id == f"i{k}"]
    full = evaluate(dup_imgs, dup_preds); m = point_metrics(P, mult)
    assert m["all"]["AP50:95"] == pytest.approx(full["AP50:95"], abs=1e-12)
    assert m["small"]["AP50:95"] == pytest.approx(full["size_resize640"]["small"]["AP50:95"], abs=1e-12)


def test_small_recall_at_iou50_conf25():
    imgs = [Image("a", 640, 640, [(0, (0., 0., 20., 20.))]), Image("b", 640, 640, [(0, (0., 0., 20., 20.))])]
    preds = [Pred("a", 0, .9, (0., 0., 20., 20.)), Pred("b", 0, .2, (0., 0., 20., 20.))]
    m = point_metrics(prepare(imgs, preds), np.ones(2))
    assert m["small"]["recall_50_c25"] == pytest.approx(.5) and m["small"]["n_gt"] == 2


@pytest.mark.skipif(not B0P.exists(), reason="B0 records missing")
def test_matches_stored_B0_seed101_metrics():
    imgs = load_val_images(ROOT); preds = load_predictions(B0P)
    P = prepare(imgs, preds); m = point_metrics(P, np.ones(len(imgs))); ref = json.loads(B0M.read_text())["one2many"]
    assert m["all"]["AP50:95"] == pytest.approx(ref["AP50:95"], abs=1e-9)
    assert m["small"]["AP50:95"] == pytest.approx(ref["size_resize640"]["small"]["AP50:95"], abs=1e-9)


def test_bootstrap_is_paired_seeded_and_reproducible():
    imgs, preds = toy(); better = [Pred(p.image_id, 0, p.score, p.box, p.index) for p in preds if p.index != 2]   # drop FP
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    A = {101: prepare(imgs, preds), 202: prepare(imgs, preds)}; B = {101: prepare(imgs, better), 202: prepare(imgs, better)}
    r1 = bootstrap_paired(A, B, groups, draws=200, seed=20261025); r2 = bootstrap_paired(A, B, groups, draws=200, seed=20261025)
    assert r1 == r2 and r1["n_draws"] == 200 and r1["invalid_draws"] >= 0
    assert r1["metrics"]["all"]["AP50:95"]["ci90"][0] >= -1e-12                   # B dominates A on every draw
    assert r1["metrics"]["all"]["AP50:95"]["mean_diff"] > 0


def test_invalid_draw_zero_primary_positives_is_redrawn_and_counted():
    imgs = [Image("a", 640, 640, [(0, (0., 0., 20., 20.))]), Image("b", 640, 640, [(0, (0., 0., 200., 200.))])]   # only a has a SMALL gt
    preds = [Pred("a", 0, .9, (0., 0., 20., 20.)), Pred("b", 0, .9, (0., 0., 200., 200.))]
    P = {101: prepare(imgs, preds)}
    r = bootstrap_paired(P, P, np.array([0, 1]), draws=300, seed=1)
    assert r["invalid_draws"] > 0 and r["n_draws"] == 300
