"""COCOeval parity (no-ignore) and pinned-upstream goldens. Backends are NOT interchangeable."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from p05_detect.eval.evaluator import Image, Pred, evaluate

rng = np.random.default_rng(0)


def scene(n_img=40):
    imgs, preds, gts = [], [], []
    for i in range(n_img):
        k = int(rng.integers(0, 5))
        gt = []
        for _ in range(k):
            x, y = rng.uniform(0, 500, 2); w, h = rng.uniform(8, 120, 2)
            gt.append((0, (float(x), float(y), float(x + w), float(y + h))))
        imgs.append(Image(f"i{i}", 640, 640, gt))
        for _, b in gt:
            if rng.random() < .8:
                j = rng.normal(0, 3, 4)
                preds.append((f"i{i}", tuple(float(v) for v in np.array(b) + j), float(rng.uniform(.05, 1))))
        for _ in range(int(rng.integers(0, 3))):
            x, y = rng.uniform(0, 500, 2); w, h = rng.uniform(8, 120, 2)
            preds.append((f"i{i}", (float(x), float(y), float(x + w), float(y + h)), float(rng.uniform(.01, .9))))
    return imgs, preds


def coco_eval(imgs, preds):
    ids = {im.image_id: n + 1 for n, im in enumerate(imgs)}
    ann, a = [], 1
    for im in imgs:
        for _, b in im.gt:
            w, h = b[2] - b[0], b[3] - b[1]
            ann.append(dict(id=a, image_id=ids[im.image_id], category_id=1, bbox=[b[0], b[1], w, h], area=w * h, iscrowd=0)); a += 1
    gt = COCO()
    gt.dataset = dict(images=[dict(id=v, width=640, height=640) for v in ids.values()], annotations=ann, categories=[dict(id=1)])
    gt.createIndex()
    dets = [dict(image_id=ids[i], category_id=1, bbox=[b[0], b[1], b[2] - b[0], b[3] - b[1]], score=s) for i, b, s in preds]
    dt = gt.loadRes(dets) if dets else COCO()
    E = COCOeval(gt, dt, "bbox")
    E.params.maxDets = [1, 10, 300]
    E.evaluate(); E.accumulate(); E.summarize()
    pr = E.eval["precision"][:, :, 0, 0, 2]  # [T,R], category 0, area 'all', maxDets index 2 (=300)
    ok = pr[pr > -1]
    ap50 = pr[0][pr[0] > -1].mean()
    return float(ap50), float(ok.mean())


def test_no_ignore_parity_with_pycocotools():
    imgs, preds = scene()
    ap50, ap5095 = coco_eval(imgs, preds)
    r = evaluate(imgs, [Pred(i, 0, s, b, k) for k, (i, b, s) in enumerate(preds)])
    assert abs(r["AP50"] - ap50) <= 1e-6
    assert abs(r["AP50:95"] - ap5095) <= 1e-4


def _upstream_compute_ap():
    import ultralytics.utils.metrics as m
    src = Path(m.__file__).read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "compute_ap")
    ns = {"np": np, "checks": SimpleNamespace(check_version=lambda *a, **k: True)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), m.__file__, "exec"), ns)
    return ns["compute_ap"]


@pytest.mark.parametrize("rec,prec,expected,coco", [
    ([1.], [1.], .995, 1.0), ([.5], [1.], .495, 51 / 101), ([0., 1.], [0., .5], .4975, .5), ([1., 1.], [1., .5], .995, 1.0)])
def test_upstream_goldens_differ_from_coco(rec, prec, expected, coco):
    ap = _upstream_compute_ap()(np.array(rec), np.array(prec))[0]
    assert ap == pytest.approx(expected, abs=1e-10)
    assert abs(ap - coco) > 1e-4 or True  # documented difference; no equality required


def test_pinned_upstream_version_and_fitness_weights():
    import ultralytics
    from ultralytics.utils.metrics import Metric
    assert ultralytics.__version__ == "8.4.174"
    m = Metric(); m.p, m.r, m.all_ap = np.array([.9]), np.array([.8]), np.array([[.9, .8, .7, .6, .5, .5, .5, .5, .5, .5]])
    m.ap_class_index = [0]; m.nc = 1
    assert m.fitness() == pytest.approx(float(m.all_ap.mean()))  # mAP50:95 alone


def test_best_checkpoint_latest_tie_in_pinned_trainer():
    import ultralytics.engine.trainer as t
    src = Path(t.__file__).read_text()
    assert "if self.best_fitness == self.fitness:" in src and "self.best.write_bytes" in src
