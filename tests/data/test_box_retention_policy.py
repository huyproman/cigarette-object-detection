import json, hashlib
import pytest
from pathlib import Path
from p05_detect.data.box_policy import box_disposition as D
def test_boundaries_640_frame():            # 640x640 image: scale 1
    assert D([0, 0, 2.99, 20], 640, 640) == "DISCARDED_SUB_PIXEL"
    assert D([0, 0, 3.0, 20], 640, 640) == "RETAINED"
    assert D([0, 0, 2.5, 20], 640, 640) == "DISCARDED_SUB_PIXEL"      # between 2 and 3 px is dropped
    assert D([0, 0, 20, 2.99], 640, 640) == "DISCARDED_SUB_PIXEL"
def test_scale_uses_long_edge():
    assert D([0, 0, 5, 40], 1280, 640) == "DISCARDED_SUB_PIXEL"       # 5px*0.5=2.5
    assert D([0, 0, 6, 40], 1280, 640) == "RETAINED"
def test_oversized_heuristic():
    assert D([0, 0, 200, 50], 640, 640) == "DISCARDED_OVERSIZED_NON_CIGARETTE"   # nw .31 > .25
    assert D([0, 0, 50, 200], 640, 640) == "DISCARDED_OVERSIZED_NON_CIGARETTE"
    assert D([0, 0, 100, 100], 640, 640) == "RETAINED"
def test_regenerates_all_new_v3_labels():
    root = Path(__file__).resolve().parents[2]; man = root / "runtime/dataset_v3/split_manifest.jsonl"
    if not man.exists(): pytest.skip("private dataset split manifest absent")
    pool = {json.loads(l)["image_id"]: json.loads(l) for l in open(root / "runtime/pool/pool.jsonl")}
    n = 0
    for l in open(man):
        m = json.loads(l)
        if not m.get("is_new_tranche"): continue
        c = pool[m["image_id"][4:]]; w, h = m["w"], m["h"]; L = []
        for x1, y1, x2, y2 in c["boxes"]:
            if D([x1, y1, x2, y2], w, h) == "RETAINED":
                L.append(f"0 {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n")
        assert hashlib.sha256("".join(L).encode()).hexdigest() == m["label_sha256"]; n += 1
    assert n == 300
