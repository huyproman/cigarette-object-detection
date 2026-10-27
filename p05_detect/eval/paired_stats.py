"""Paired-statistics engine. Per-image sufficient statistics are produced with the evaluator's own matching
(_match_threshold: same sorting, greedy valid-GT matching, size-bin neutralisation), so cohort metrics are identical to evaluate().
A bootstrap draw is then a multiplicity vector over images (repeated group occurrences = repeated, separately counted image instances),
and AP is recomputed by weighted cumulative sums. Cohorts use RESIZE640 areas: small <1024, non-small >=1024."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p05_detect.eval.evaluator import IOUS, Image, Pred, _match_threshold, letterbox_params, size_bin, area

COHORTS = ("all", "small", "nonsmall")
OP_CONF = 0.25


def _r640_area(im, box):
    r = letterbox_params(im.width, im.height)[0]
    return area(box) * r * r


def _cohort_fns(name):
    if name == "all":
        return None, None
    want = (lambda a: size_bin(a) == "small") if name == "small" else (lambda a: size_bin(a) != "small")
    return (lambda im, j: want(_r640_area(im, im.gt[j][1]))), (lambda im, p: want(_r640_area(im, p.box)))


def prepare(images, preds, thresholds=IOUS, max_det=300, ap_floor=0.001):
    """Return {'n_images', 'ngt'[cohort] (per image), 'rows'[cohort][thr]=(score, flag, img_idx) in evaluator order}."""
    kept = [im for im in images if im.status not in ("unknown_empty", "missing_suspected")]
    idx = {im.image_id: i for i, im in enumerate(kept)}
    by = {}
    for p in preds:
        if p.image_id in idx and p.score >= ap_floor:
            by.setdefault(p.image_id, []).append(p)
    out = dict(n_images=len(kept), ngt={}, rows={})
    for c in COHORTS:
        in_bin, pred_in = _cohort_fns(c)
        out["ngt"][c] = np.array([sum(1 for j in range(len(im.gt)) if in_bin is None or in_bin(im, j)) for im in kept], dtype=float)
        out["rows"][c] = {}
        for thr in thresholds:
            res = _match_threshold(kept, by, thr, max_det, in_bin=in_bin, pred_in_bin=pred_in)
            res = [t for t in res if t[4] != -1]
            out["rows"][c][thr] = (np.array([t[0] for t in res]), np.array([1.0 if t[4] == 1 else 0.0 for t in res]), np.array([idx[t[1]] for t in res], dtype=int))
    return out


def weighted_ap(flags, w, n_gt):
    if n_gt <= 0:
        return None
    if not len(flags):
        return 0.0
    tp = np.cumsum(w * flags); fp = np.cumsum(w * (1 - flags))
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1e-16)
    env = np.maximum.accumulate(prec[::-1])[::-1]
    i = np.searchsorted(rec, np.linspace(0, 1, 101) - 1e-12, side="left")
    return float(np.where(i < len(env), env[np.minimum(i, len(env) - 1)], 0.0).mean())


def point_metrics(P, mult):
    """mult: per-image multiplicity (n_images,). Returns {cohort: {n_gt, AP50, AP50:95, recall_50_c25}}."""
    m = {}
    for c in COHORTS:
        n_gt = float((P["ngt"][c] * mult).sum()); aps = []; rec = None
        for thr, (s, f, ii) in P["rows"][c].items():
            w = mult[ii]
            aps.append(weighted_ap(f, w, n_gt))
            if abs(thr - 0.5) < 1e-9:
                rec = float((w * f * (s >= OP_CONF)).sum() / n_gt) if n_gt > 0 else None
        ok = all(a is not None for a in aps)
        m[c] = {"n_gt": n_gt, "AP50": aps[0] if ok else None, "AP50:95": float(np.mean(aps)) if ok else None, "recall_50_c25": rec}
    return m


ENDPOINTS = (("small", "AP50:95"), ("small", "recall_50_c25"), ("all", "AP50:95"), ("nonsmall", "AP50:95"))


def bootstrap_paired(base, arm, groups, draws=2000, seed=20261025, max_invalid_frac=0.05):
    """base/arm: {seed: prepared}; groups: per-image group id (aligned with prepared images). Same sampled groups for both arms and all seeds;
    per-draw paired differences are averaged across the fixed seeds. Draws with zero eligible small GT are discarded and redrawn."""
    rng = np.random.default_rng(seed); ug, inv = np.unique(groups, return_inverse=True); G = len(ug)
    seeds = sorted(base); vals = {e: [] for e in ENDPOINTS}; invalid = 0
    while len(vals[ENDPOINTS[0]]) < draws:
        cnt = np.bincount(rng.integers(0, G, G), minlength=G).astype(float); mult = cnt[inv]
        if float((base[seeds[0]]["ngt"]["small"] * mult).sum()) <= 0:
            invalid += 1; continue
        diffs = {e: [] for e in ENDPOINTS}
        for s in seeds:
            a, b = point_metrics(base[s], mult), point_metrics(arm[s], mult)
            for c, k in ENDPOINTS:
                diffs[(c, k)].append(b[c][k] - a[c][k] if a[c][k] is not None and b[c][k] is not None else np.nan)
        for e in ENDPOINTS:
            vals[e].append(float(np.mean(diffs[e])))
    total = draws + invalid
    res = {"n_draws": draws, "invalid_draws": invalid, "invalid_frac": invalid / total, "seed": seed, "n_groups": int(G), "seeds": seeds,
           "valid": (invalid / total) <= max_invalid_frac, "metrics": {}}
    for (c, k), v in vals.items():
        v = np.array(v); v = v[~np.isnan(v)]
        res["metrics"].setdefault(c, {})[k] = {"mean_diff": float(v.mean()), "ci90": [float(np.percentile(v, 5)), float(np.percentile(v, 95))]}
    return res


def load_val_images(root):
    root = Path(root); imgs = []
    for line in open(root / "runtime/dataset/split_manifest.jsonl"):
        m = json.loads(line)
        if m["split"] != "VAL":
            continue
        gt = []
        for ln in Path(m["path"].replace("/images/", "/labels/")).with_suffix(".txt").read_text().splitlines():
            _, cx, cy, bw, bh = ln.split(); cx, cy, bw, bh = float(cx) * m["w"], float(cy) * m["h"], float(bw) * m["w"], float(bh) * m["h"]
            gt.append((0, (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)))
        im = Image(m["path"], m["w"], m["h"], gt, [], "positive"); im.group = m["group"]; im.family = m["family"]
        imgs.append(im)
    return imgs


def load_predictions(path):
    return [Pred(r[0], r[1], r[2], tuple(r[3]), r[4], r[5]) for r in json.loads(Path(path).read_text())]
