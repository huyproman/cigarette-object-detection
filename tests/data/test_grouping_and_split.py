"""Grouping, split and tiny-box policy tests."""
import pytest
from p05_detect.data.grouping import build_groups, hamming, largest_remainder, split_groups, tiny_ignored, resize640_min_side, wilson_lower, sem_decision


def rec(i, fam="F", sha=None, dh=0, root=None, w=640, h=640):
    return dict(image_id=i, fam=fam, sha256=sha or f"s{i}", dh=[dh, dh, dh, dh], root=root or f"r{i}", w=w, h=h, pixels=w * h)


def test_hamming():
    assert hamming(0b1011, 0b0001) == 2


def test_exact_duplicate_and_dhash6_link_same_group():
    g = build_groups([rec("a", sha="x", dh=2**40 - 1), rec("b", sha="x", dh=2**40 - 1), rec("c", dh=0b111111), rec("d", dh=0), rec("e", dh=0b1111111)])
    gid = g["image_to_group"]
    assert gid["a"] == gid["b"]
    assert gid["c"] == gid["d"]            # hamming 6 -> linked
    assert gid["e"] == gid["d"]            # transitive via c (hamming 1): chaining is reported, not fragmented
    assert gid["e"] != gid["a"]


def test_dhash7_not_linked_but_flip_variant_linked():
    a = rec("a"); b = rec("b", dh=0b1111111)
    assert build_groups([a, b])["image_to_group"]["a"] != build_groups([a, b])["image_to_group"]["b"]
    f = rec("f"); f["dh"] = [2**64 - 1, 0, 0, 0]
    assert build_groups([rec("a"), f])["image_to_group"]["a"] == build_groups([rec("a"), f])["image_to_group"]["f"]


def test_same_roboflow_root_links_case_insensitive():
    g = build_groups([rec("a", root="IMG_1.rf.AAA", dh=0), rec("b", root="img_1.rf.bbb", dh=2**40 - 1)])
    assert g["image_to_group"]["a"] == g["image_to_group"]["b"]


def test_group_id_stable_under_order():
    r = [rec("a", sha="x"), rec("b", sha="x"), rec("c")]
    assert build_groups(r)["groups"].keys() == build_groups(r[::-1])["groups"].keys()


def test_canonical_is_largest_then_smallest_id():
    g = build_groups([rec("b", sha="x", w=100, h=100), rec("a", sha="x", w=200, h=200), rec("c", sha="x", w=200, h=200)])
    assert list(g["groups"].values())[0]["canonical"] == "a"


def test_largest_remainder_70_15_15():
    assert largest_remainder(10, (.7, .15, .15)) == [7, 2, 1]   # tie order TRAIN,VAL,HOLDOUT gets leftover
    assert sum(largest_remainder(101, (.7, .15, .15))) == 101
    assert largest_remainder(3, (.7, .15, .15)) == [2, 1, 0]


def test_split_deterministic_disjoint():
    groups = {f"g{i}": dict(stratum="A") for i in range(100)}
    s1, s2 = split_groups(groups, 20261024), split_groups(groups, 20261024)
    assert s1 == s2 and len(set(s1.values())) == 3
    assert sum(v == "TRAIN" for v in s1.values()) == 70
    assert sum(v == "VAL" for v in s1.values()) == 15
    assert split_groups(groups, 1) != s1


def test_tiny_boundary_resize640():
    # 1280x1280 -> r=.5 ; box 16px -> 8.0 : NOT ignored (<8 is ignored); 15.99 -> ignored
    assert resize640_min_side((0, 0, 16, 40), 1280, 1280) == pytest.approx(8.0)
    assert not tiny_ignored((0, 0, 16, 40), 1280, 1280)
    assert tiny_ignored((0, 0, 15.99, 40), 1280, 1280)
    # source-pixel size is not used: 7px box in a 320px image upscales to 14 -> kept
    assert not tiny_ignored((0, 0, 7, 7), 320, 320)


def test_wilson_lower_bound_examples():
    assert wilson_lower(76, 80) == pytest.approx(.878375, abs=1e-6)
    assert wilson_lower(68, 80) == pytest.approx(.755866, abs=1e-6)
    assert wilson_lower(50, 80) == pytest.approx(.515485, abs=1e-6)


def test_sem_decision():
    assert sem_decision(76, 80, 0) == "ELIGIBLE"
    assert sem_decision(68, 80, 0) == "CONDITIONAL_QUARANTINE"
    assert sem_decision(50, 80, 0) == "EXCLUDED"
    assert sem_decision(78, 80, 14) == "CONDITIONAL_QUARANTINE"   # ambiguity 17.5% > 15%
    assert sem_decision(20, 20, 0) == "INSUFFICIENT_EVIDENCE"


def test_dhash_link_is_symmetric_over_flip_variants():
    """Reviewer B1: base_i vs flip_j was checked but base_j vs flip_i was not. Order of ids must not matter."""
    x = rec("a", dh=0); y = rec("b", dh=0)
    y["dh"] = [2**40 - 1, 0, 0, 0]          # y's FLIP variant equals x's base; y's base is far from x's flips
    x["dh"] = [0, 2**50 - 1, 2**50 - 1, 2**50 - 1]
    g1 = build_groups([x, y])["image_to_group"]; assert g1["a"] == g1["b"]
    x2 = dict(x, image_id="z"); g2 = build_groups([x2, y])["image_to_group"]   # swap id order
    assert g2["z"] == g2["b"]


def test_img_number_adjacency_links_within_family_only():
    a = rec("a", fam="F", root="IMG_1000_jpg.rf.aaa", dh=0); b = rec("b", fam="F", root="IMG_1003_jpg.rf.bbb", dh=2**40 - 1)
    c = rec("c", fam="F", root="IMG_1004_jpg.rf.ccc", dh=2**30 - 1 << 20); d = rec("d", fam="G", root="IMG_1001_jpg.rf.ddd", dh=(2**64 - 1) ^ 2**33)
    g = build_groups([a, b, c, d])["image_to_group"]
    assert g["a"] == g["b"]            # diff 3 -> linked
    assert g["b"] == g["c"]            # diff 1 -> linked (transitive chain reported, not fragmented)
    assert g["d"] != g["a"]            # other family: not linked by number
    far = build_groups([a, rec("e", fam="F", root="IMG_1009_jpg.rf.eee", dh=2**20 - 1 << 30)])["image_to_group"]
    assert far["a"] != far["e"]        # diff 9 -> not linked
