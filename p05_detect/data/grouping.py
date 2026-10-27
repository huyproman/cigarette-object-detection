"""Canonical grouping, deterministic split, tiny-box policy and semantic-gate maths."""
from __future__ import annotations

import hashlib
import math
import re

import numpy as np

SPLITS = ("TRAIN", "VAL", "HOLDOUT")
RATIOS = (0.70, 0.15, 0.15)
DHASH_MAX = 6
TINY_MIN_SIDE = 8.0


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


IMG_RE = re.compile(r"^img_?(\d+)")


def root_key(root: str) -> str:
    return re.split(r"\.rf\.", root.lower())[0]


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def build_groups(records, dhash_max=DHASH_MAX):
    """records: dicts with image_id, sha256, dh (4 flip variants), root, w, h. Returns groups + edge counts.
    Edges: exact SHA256, dHash<=6 over flip variants, case-insensitive Roboflow root. All groups are PROBABLE."""
    recs = sorted(records, key=lambda r: r["image_id"])
    n = len(recs)
    dsu = _DSU(n)
    edges = {"sha256": 0, "dhash": 0, "root": 0}
    first = {}
    for i, r in enumerate(recs):
        for kind, key in (("sha256", r["sha256"]), ("root", root_key(r["root"]))):
            k = (kind, key)
            if k in first:
                if dsu.find(first[k]) != dsu.find(i):
                    edges[kind] += 1
                dsu.union(first[k], i)
            else:
                first[k] = i
    if n:
        base = np.array([r["dh"][0] & (2**64 - 1) for r in recs], dtype=np.uint64)
        variants = np.array([[v & (2**64 - 1) for v in r["dh"]] for r in recs], dtype=np.uint64)
        for i in range(n - 1):
            # symmetric over flips: min(base_i vs variants_j, variants_i vs base_j)
            d = np.minimum(np.bitwise_count(base[i] ^ variants[i + 1:]).min(axis=1),
                           np.bitwise_count(variants[i][None, :] ^ base[i + 1:, None]).min(axis=1))
            for j in np.nonzero(d <= dhash_max)[0]:
                if dsu.find(i) != dsu.find(i + 1 + int(j)):
                    edges["dhash"] += 1
                dsu.union(i, i + 1 + int(j))
    # within-family IMG_#### numeric adjacency <= 3 (sorted consecutive links are equivalent to all pairs <= 3)
    num = {}
    for i, r in enumerate(recs):
        m = IMG_RE.match(r["root"].lower())
        if m:
            num.setdefault(r["fam"], []).append((int(m.group(1)), i))
    for fam, lst in num.items():
        lst.sort()
        for (a, ia), (b, ib) in zip(lst, lst[1:]):
            if b - a <= 3:
                if dsu.find(ia) != dsu.find(ib):
                    edges["img_adjacency"] = edges.get("img_adjacency", 0) + 1
                dsu.union(ia, ib)
    comps = {}
    for i, r in enumerate(recs):
        comps.setdefault(dsu.find(i), []).append(r)
    groups, mapping = {}, {}
    for members in comps.values():
        ids = sorted(m["image_id"] for m in members)
        gid = hashlib.sha256("|".join(ids).encode()).hexdigest()[:16]
        canon = sorted(members, key=lambda m: (-m.get("pixels", m["w"] * m["h"]), m["image_id"]))[0]
        groups[gid] = dict(members=ids, canonical=canon["image_id"], size=len(ids),
                           families=sorted({m["fam"] for m in members}), status="PROBABLE")
        for m in ids:
            mapping[m] = gid
    return dict(groups=groups, image_to_group=mapping, edges=edges)


def largest_remainder(n: int, ratios=RATIOS):
    q = [n * r for r in ratios]
    base = [math.floor(v + 1e-9) for v in q]
    left = n - sum(base)
    order = sorted(range(len(ratios)), key=lambda i: (-(q[i] - base[i]), i))
    for i in order[:left]:
        base[i] += 1
    return base


def split_groups(groups: dict, seed: int = 20261024):
    """groups: gid -> {stratum}. Within each stratum: order by SHA256(seed|gid), allocate by largest remainder."""
    by = {}
    for gid, g in groups.items():
        by.setdefault(g["stratum"], []).append(gid)
    out = {}
    for stratum in sorted(by):
        ids = sorted(by[stratum], key=lambda g: hashlib.sha256(f"{seed}|{g}".encode()).hexdigest())
        counts = largest_remainder(len(ids))
        k = 0
        for name, c in zip(SPLITS, counts):
            for gid in ids[k:k + c]:
                out[gid] = name
            k += c
    return out


def resize640_min_side(box, w, h):
    r = min(640 / w, 640 / h)
    return r * min(box[2] - box[0], box[3] - box[1])


def tiny_ignored(box, w, h):
    return resize640_min_side(box, w, h) < TINY_MIN_SIDE


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    p = k / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def sem_decision(k, n, ambiguous, tau_pass=.85, tau_cond=.70, amb_max=.15, min_groups=30):
    if n < min_groups:
        return "INSUFFICIENT_EVIDENCE"
    L = wilson_lower(k, n)
    if L < tau_cond:
        return "EXCLUDED"
    if L < tau_pass or ambiguous / n > amb_max:
        return "CONDITIONAL_QUARANTINE"
    return "ELIGIBLE"
