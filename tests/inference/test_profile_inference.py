"""Inference functional tests. Weight-dependent tests use the REAL checkpoints and are skipped ONLY if the authorized files are absent (no-weights route)."""
import io, json, socket, subprocess, sys
from pathlib import Path
import numpy as np, pytest, torch
from PIL import Image

from p05_detect.infer import core
from p05_detect.infer.registry import PROFILES, InferenceError, get_profile, load_registry, resolve_checkpoint, sha256_file, weights_dir

ROOT = Path(__file__).parents[2]
HAVE = all((weights_dir() / p["checkpoint_file"]).is_file() for p in load_registry()["profiles"])
needs_w = pytest.mark.skipif(not HAVE, reason="authorized checkpoint files absent (no-weights route)")
VAL = None
MAN = ROOT / "runtime/dataset/split_manifest.jsonl"
if MAN.exists():
    VAL = [json.loads(l)["path"] for l in open(MAN) if json.loads(l)["split"] == "VAL"][:12]
needs_img = pytest.mark.skipif(not VAL, reason="local VAL images absent")


def synth(w, h, mode="RGB", seed=0):
    rng = np.random.default_rng(seed); a = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    return Image.fromarray(a).convert(mode)


# ---- registry ---------------------------------------------------------------
def test_registry_has_exactly_three_profiles():
    reg = load_registry()
    assert [p["profile_id"] for p in reg["profiles"]] == list(PROFILES) == ["b0", "e2_carafe", "e3_combined"]
    assert all(p["seed"] == 101 for p in reg["profiles"]) and reg["default_profile"] == "b0"
    assert reg["inference_policy"]["head"] == "one2many" and reg["inference_policy"]["iou"] == 0.7 and reg["inference_policy"]["max_det"] == 300


def test_unknown_profile_rejected():
    for bad in ("e1", "e4_mhsa", "p2", "x"):
        with pytest.raises(InferenceError, match="unknown profile"):
            get_profile(bad)


def test_missing_and_tampered_checkpoint_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("P05_WEIGHTS_DIR", str(tmp_path))
    with pytest.raises(InferenceError, match="not found"):
        resolve_checkpoint("b0")
    f = tmp_path / get_profile("b0")["checkpoint_file"]; f.write_bytes(b"x" * get_profile("b0")["checkpoint_bytes"])
    with pytest.raises(InferenceError, match="SHA256 mismatch"):
        resolve_checkpoint("b0")
    f.write_bytes(b"short")
    with pytest.raises(InferenceError, match="size mismatch"):
        resolve_checkpoint("b0")


@needs_w
def test_checkpoint_hashes_match_registry_and_frozen_eval_records():
    for p in load_registry()["profiles"]:
        path, _ = resolve_checkpoint(p["profile_id"]); assert sha256_file(path) == p["checkpoint_sha256"]
    # independent link to the frozen VAL records (hash recorded when those metrics were produced)
    for pid, d, arm in (("b0", "eval", "B0"), ("e2_carafe", "eval_r3", "E2"), ("e3_combined", "eval_add01", "E3")):
        f = ROOT / "runtime" / d / f"{arm}-101/metrics_val.json"
        if (ROOT / "runtime/dataset").exists():           # research environment: the link to the frozen record is mandatory
            assert f.exists(), f"frozen VAL record missing: {f}"
        if f.exists():
            assert json.loads(f.read_text())["checkpoint_sha256"] == get_profile(pid)["checkpoint_sha256"]


# ---- images ---------------------------------------------------------------
def test_image_modes_and_exif(tmp_path):
    for mode in ("L", "RGBA", "P", "RGB"):
        p = tmp_path / f"{mode}.png"; synth(40, 30, "RGB").convert(mode).save(p)
        a, _ = core.load_image(p); assert a.shape == (30, 40, 3) and a.dtype == np.uint8
    im = synth(50, 20); ex = im.getexif(); ex[274] = 6; p = tmp_path / "rot.jpg"; im.save(p, exif=ex)
    a, _ = core.load_image(p); assert a.shape == (50, 20, 3), "EXIF orientation 6 must rotate 50x20 -> 20 wide x 50 high"


def test_bad_inputs_rejected(tmp_path):
    (tmp_path / "bad.png").write_bytes(b"not an image"); (tmp_path / "a.txt").write_text("x")
    for p, msg in ((tmp_path / "bad.png", "cannot decode"), (tmp_path / "a.txt", "unsupported image type"), (tmp_path / "nope.png", "not found")):
        with pytest.raises(InferenceError, match=msg):
            core.load_image(p)
    good = tmp_path / "g.png"; synth(32, 32).save(good)
    data = good.read_bytes(); (tmp_path / "trunc.png").write_bytes(data[: len(data) // 2])
    with pytest.raises(InferenceError):
        core.load_image(tmp_path / "trunc.png")
    with pytest.raises(InferenceError, match="too large"):
        core.load_image(good, max_pixels=100)
    with pytest.raises(InferenceError, match="file too large"):
        core.load_image(good, max_bytes=10)
    with pytest.raises(InferenceError, match="too small"):
        core.load_image(np.zeros((1, 50, 3), np.uint8))


# ---- letterbox geometry ---------------------------------------------------------------
@pytest.mark.parametrize("w,h", [(640, 640), (1000, 300), (300, 1000), (333, 517), (641, 639), (40, 2000), (5, 7), (4000, 3000)])
def test_letterbox_inverse_mapping_independent(w, h):
    """Independent geometry: forward map = scale r=min(640/w,640/h) then centre padding; inverse written here from first principles
    (NOT ultralytics.ops.scale_boxes). A box painted on the original must come back within 1 original pixel + one letterbox pixel."""
    lb = core.LetterBox((640, 640), auto=False, stride=32)
    img = np.zeros((h, w, 3), np.uint8); x1, y1, x2, y2 = int(w * 0.2), int(h * 0.3), max(int(w * 0.2) + 1, int(w * 0.7)), max(int(h * 0.3) + 1, int(h * 0.8)); img[y1:y2, x1:x2] = 255
    out = lb(image=img); ys, xs = np.where(out[:, :, 0] > 127)
    r = min(640 / w, 640 / h); nw, nh = round(w * r), round(h * r); px, py = (640 - nw) / 2, (640 - nh) / 2
    assert out.shape == (640, 640, 3)
    if len(xs) == 0:
        pytest.skip("object vanishes at 640 scale (extreme aspect)")
    bx1, by1, bx2, by2 = (xs.min() - px) / r, (ys.min() - py) / r, (xs.max() + 1 - px) / r, (ys.max() + 1 - py) / r
    tol = 1.5 / r + 1.0
    assert abs(bx1 - x1) <= tol and abs(by1 - y1) <= tol and abs(bx2 - x2) <= tol and abs(by2 - y2) <= tol, (bx1, by1, bx2, by2, x1, y1, x2, y2, tol)


def test_core_scale_boxes_agrees_with_independent_inverse():
    from ultralytics.utils import ops
    for w, h in ((1000, 300), (300, 1000), (333, 517), (4000, 3000)):
        r = min(640 / w, 640 / h); nw, nh = round(w * r), round(h * r); px, py = (640 - nw) / 2, (640 - nh) / 2
        b = torch.tensor([[100.0, 120.0, 400.0, 500.0]]); got = ops.scale_boxes((640, 640), b.clone(), (h, w))[0].tolist()
        exp = [(100 - px) / r, (120 - py) / r, (400 - px) / r, (500 - py) / r]
        exp = [min(max(v, 0), w if i % 2 == 0 else h) for i, v in enumerate(exp)]
        assert np.allclose(got, exp, atol=1.0), (w, h, got, exp)


def test_detection_records_clipped_positive_area():
    W, H = 200, 80                                # NON-square so a swapped w/h in the normalisation is detected
    p = torch.tensor([[-5.0, -5.0, 40.0, 20.0, 0.9, 0],          # overhangs the top-left corner -> clipped to [0,0,40,20]
                      [150.0, 60.0, 260.0, 130.0, 0.8, 0],       # overhangs the bottom-right corner -> clipped to [150,60,200,80]
                      [300.0, 10.0, 310.0, 20.0, 0.7, 0],        # entirely outside -> dropped
                      [10.0, 10.0, 10.0, 30.0, 0.6, 0]])         # zero width -> dropped
    d = core.detections_from(p, W, H)
    assert [round(r["confidence"], 1) for r in d] == [0.9, 0.8]
    assert d[0]["box_xyxy"] == [0.0, 0.0, 40.0, 20.0] and d[1]["box_xyxy"] == [150.0, 60.0, 200.0, 80.0]
    for r in d:
        x1, y1, x2, y2 = r["box_xyxy"]; assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
        n = r["box_xyxy_normalized"]; assert np.allclose([n[0] * W, n[1] * H, n[2] * W, n[3] * H], r["box_xyxy"]) and all(0 <= v <= 1 for v in n)


# ---- real inference ---------------------------------------------------------------
@needs_w
@needs_img
@pytest.mark.parametrize("pid", PROFILES)
def test_single_image_path_agrees_with_independent_geometry(pid):
    """Single-image predict() must put boxes where an independent first-principles pipeline (own letterbox/inverse, raw backend + NMS) puts them."""
    import cv2
    from ultralytics.utils import nms as _n
    m = core.load_model(pid, "cuda:0" if torch.cuda.is_available() else "cpu", warmup=False)
    rgb, _ = core.load_image(VAL[4]); H, W = rgb.shape[:2]; r = min(640 / W, 640 / H); nw, nh = round(W * r), round(H * r)
    canvas = np.full((640, 640, 3), 114, np.uint8); top, left = (640 - nh) // 2, (640 - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(canvas).permute(2, 0, 1)[None].float().div(255).to(m.device)
    with torch.inference_mode(): raw = m.backend(x)
    ref = _n.non_max_suppression(raw, 0.25, 0.7, None, False, max_det=300, nc=0, end2end=False)[0].cpu().numpy()
    got = core.predict(VAL[4], m, conf=0.25)["detections"]
    assert len(got) == len(ref), (len(got), len(ref))
    for g, rr in zip(got, ref):
        bx = [(rr[0] - left) / r, (rr[1] - top) / r, (rr[2] - left) / r, (rr[3] - top) / r]; bx = [min(max(v, 0), W if i % 2 == 0 else H) for i, v in enumerate(bx)]
        assert max(abs(a - b) for a, b in zip(g["box_xyxy"], bx)) < 1.5, (g["box_xyxy"], bx)     # independent inverse: ~1 px (interpolation / rounding differences)


@needs_w
@needs_img
@pytest.mark.parametrize("pid", PROFILES)
def test_real_inference_schema_and_geometry(pid):
    m = core.load_model(pid, "cpu" if not torch.cuda.is_available() else "cuda:0")
    r = core.predict(VAL[0], m, conf=0.05)
    assert r["schema_version"] == "1" and r["model"]["profile_id"] == pid and r["model"]["seed"] == 101
    assert r["model"]["checkpoint_sha256"] == get_profile(pid)["checkpoint_sha256"]
    assert r["settings"]["head"] == "one-to-many" and r["settings"]["iou"] == 0.7 and r["settings"]["max_det"] == 300
    assert set(r["timing_ms"]) >= {"decode", "model_forward", "end_to_end", "model_load_cold", "warmup", "scope"}
    json.dumps(r)                                # plain-python types only
    W, H = r["image"]["width"], r["image"]["height"]
    confs = [d["confidence"] for d in r["detections"]]; assert confs == sorted(confs, reverse=True)
    for d in r["detections"]:
        x1, y1, x2, y2 = d["box_xyxy"]; assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H and d["class_name"] == "cigarette" and d["class_id"] == 0


@needs_w
def test_model_class_and_architecture_identity():
    ms = {p: core.load_model(p, "cpu", warmup=False) for p in PROFILES}
    names = {p: dict(m.backend.names) for p, m in ms.items()}; assert all(v == {0: "cigarette"} for v in names.values())
    kinds = {p: [type(m.backend.model.model[i]).__name__ for i in (11, 14)] for p, m in ms.items()}
    assert kinds["b0"] == ["Upsample", "Upsample"] and kinds["e2_carafe"] == kinds["e3_combined"] == ["CARAFE", "CARAFE"]
    n = {p: sum(x.numel() for x in m.backend.model.parameters()) for p, m in ms.items()}
    assert n["e2_carafe"] == n["e3_combined"] and n["e2_carafe"] > n["b0"]          # E3's NWD adds no inference parameters
    sd2, sd3 = ms["e2_carafe"].backend.model.state_dict(), ms["e3_combined"].backend.model.state_dict()
    assert sd2.keys() == sd3.keys() and any(not torch.equal(sd2[k], sd3[k]) for k in sd2)   # same graph, different weights


@needs_w
@needs_img
def test_thresholds_monotone_and_shared_policy():
    m = core.load_model("e2_carafe", "cpu", warmup=False)
    n = [core.predict(VAL[1], m, conf=c)["num_detections"] for c in (0.01, 0.1, 0.25, 0.5, 0.9)]
    assert n == sorted(n, reverse=True)
    best = None
    for v in VAL:                                # find an image where the IoU threshold visibly matters (many overlapping candidates)
        lo = core.predict(v, m, conf=0.001, iou=0.99)["num_detections"]; hi = core.predict(v, m, conf=0.001, iou=0.3)["num_detections"]
        if lo - hi >= 3: best = (v, lo, hi); break
    assert best is not None, "no VAL image with IoU-sensitive candidates among the first 12"
    assert best[2] < best[1]                     # strict: a wrong/ignored IoU (e.g. hard-coded 0.7) cannot satisfy both settings
    mid = core.predict(best[0], m, conf=0.001, iou=0.7)["num_detections"]; assert best[2] <= mid <= best[1]
    c = core.compare(VAL[2], conf=0.3, iou=0.6, device="cpu")
    assert {r["settings"]["conf"] for r in c.values()} == {0.3} and {r["settings"]["iou"] for r in c.values()} == {0.6}
    assert {(r["image"]["width"], r["image"]["height"]) for r in c.values()} == {(c["b0"]["image"]["width"], c["b0"]["image"]["height"])}


@needs_w
@needs_img
def test_profile_independence_and_order_invariance():
    a = core.compare(VAL[3], profiles=("b0", "e2_carafe", "e3_combined"), device="cpu")
    b = core.compare(VAL[3], profiles=("e3_combined", "e2_carafe", "b0"), device="cpu")
    for p in PROFILES:
        assert a[p]["detections"] == b[p]["detections"], p


@needs_w
def test_empty_detection_is_empty_list(tmp_path):
    p = tmp_path / "blank.png"; Image.new("RGB", (200, 150), (128, 128, 128)).save(p)
    r = core.predict(p, core.load_model("b0", "cpu", warmup=False), conf=0.9)
    assert r["detections"] == [] and r["num_detections"] == 0 and json.loads(core.serialize(r))["detections"] == []


@needs_w
def test_unsupported_device_and_precision_fail_intelligibly():
    with pytest.raises(InferenceError, match="fp16 requires a CUDA"):
        core.load_model("b0", "cpu", "fp16")
    with pytest.raises(InferenceError, match="unsupported precision"):
        core.load_model("b0", "cpu", "bf16")
    with pytest.raises(InferenceError, match="unsupported device"):
        core.load_model("b0", "tpu")
    with pytest.raises(InferenceError, match="CUDA is not available|CUDA device\\(s\\) exist"):
        core.load_model("b0", "cuda:9")


@needs_w
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@needs_img
def test_fp16_runs_and_is_labelled_fp16():
    r = core.predict(VAL[0], core.load_model("e3_combined", "cuda:0", "fp16"), conf=0.25)
    assert r["settings"]["precision"] == "fp16"


@needs_w
@needs_img
def test_cpu_gpu_agreement_at_operating_point():
    if not torch.cuda.is_available(): pytest.skip("no CUDA")
    c = core.predict(VAL[0], core.load_model("b0", "cpu", warmup=False)); g = core.predict(VAL[0], core.load_model("b0", "cuda:0", warmup=False))
    assert abs(c["num_detections"] - g["num_detections"]) <= 1
    if c["num_detections"] == g["num_detections"] and c["num_detections"]:
        assert max(abs(a - b) for x, y in zip(c["detections"], g["detections"]) for a, b in zip(x["box_xyxy"], y["box_xyxy"])) < 3.0


@needs_w
@needs_img
def test_inference_makes_no_network_connections(monkeypatch):
    def deny(*a, **k): raise AssertionError("network access attempted during inference")
    m = core.load_model("b0", "cpu", warmup=False)
    monkeypatch.setattr(socket.socket, "connect", deny); monkeypatch.setattr(socket, "create_connection", deny)
    core.predict(VAL[0], m)


# ---- CLI ---------------------------------------------------------------
def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "p05_detect.infer", *args], capture_output=True, text=True, cwd=ROOT)


def test_cli_help_lists_profiles():
    r = run_cli("--help"); assert r.returncode == 0
    r2 = run_cli("infer", "--help"); assert all(p in (r.stdout + r2.stdout) for p in PROFILES) and "--conf" in r2.stdout and "--precision" in r2.stdout


def test_cli_rejects_fourth_profile_and_missing_image(tmp_path):
    assert run_cli("infer", "x.png", "--profile", "e4").returncode != 0
    r = run_cli("infer", str(tmp_path / "none.png"), "--out", str(tmp_path)); assert r.returncode == 2 and "error:" in r.stderr


@needs_w
@needs_img
def test_cli_infer_and_compare(tmp_path):
    r = run_cli("infer", VAL[0], "--device", "cpu", "--out", str(tmp_path / "a")); assert r.returncode == 0, r.stderr
    stem = Path(VAL[0]).stem; j = json.loads((tmp_path / "a" / f"{stem}.b0.json").read_text()); assert j["model"]["profile_id"] == "b0" and (tmp_path / "a" / f"{stem}.b0.png").exists()
    r = run_cli("compare", VAL[0], "--device", "cpu", "--out", str(tmp_path / "c")); assert r.returncode == 0, r.stderr
    for p in PROFILES:
        assert (tmp_path / "c" / f"{stem}.{p}.json").exists() and (tmp_path / "c" / f"{stem}.{p}.png").exists()
    mont = Image.open(tmp_path / "c" / f"{stem}.compare.png"); assert mont.width > mont.height
    allj = json.loads((tmp_path / "c" / f"{stem}.compare.json").read_text())["results"]; assert list(allj) == list(PROFILES)


# ---- frozen-reference parity subset ---------------------------------------------------------------
@needs_w
@needs_img
@pytest.mark.skipif(not torch.cuda.is_available(), reason="parity reference was produced on CUDA")
@pytest.mark.parametrize("pid,d,arm", [("b0", "eval", "B0"), ("e2_carafe", "eval_r3", "E2"), ("e3_combined", "eval_add01", "E3")])
def test_batched_core_matches_frozen_predictions(pid, d, arm):
    f = ROOT / "runtime" / d / f"{arm}-101/predictions_one2many.json"
    if not f.exists(): pytest.skip("frozen predictions absent")
    ref = {}
    for p in json.loads(f.read_text()): ref.setdefault(p[0], []).append(p)
    paths = [json.loads(l)["path"] for l in open(MAN) if json.loads(l)["split"] == "VAL"][:64]
    m = core.load_model(pid, "cuda:0", warmup=False); got = {}
    for k in range(0, 64, 32):
        ch = paths[k:k + 32]
        for p, ds in zip(ch, core.predict_batch([core.load_image(q)[0] for q in ch], m, conf=0.001, iou=0.7)): got[p] = ds
    for p in paths:
        r = sorted(ref.get(p, []), key=lambda q: -q[2]); assert len(r) == len(got[p]), p
        for a, b in zip(got[p], r):
            assert abs(a["confidence"] - b[2]) < 1e-6 and max(abs(x - y) for x, y in zip(a["box_xyxy"], b[3])) < 1e-3


# ---- UI handler ---------------------------------------------------------------
@needs_w
def test_ui_handler_three_panels_controls_and_errors(tmp_path):
    gr = pytest.importorskip("gradio")
    from p05_detect.infer import app
    app.ARGS.update(device="cpu", precision="fp32")
    img = ROOT / "examples/synthetic_fixture.png"
    out = app.run(str(img), "Compare all profiles", "b0", 0.25, 0.7)
    panels, tables, notes, jfile, mfile = out[:3], out[3:6], out[6:9], out[9], out[10]
    assert all(p is not None for p in panels) and all("detection(s) at conf 0.25, IoU 0.7" in n for n in notes) and mfile
    assert "model-forward" in notes[0] and "cold load" in notes[0] and "end-to-end" in notes[0]       # labelled timing scopes
    res = json.loads(Path(jfile).read_text())["results"]; assert list(res) == list(PROFILES)
    single = app.run(str(img), "Single model", "e2_carafe", 0.25, 0.7)                                  # single-model mode runs only the selected profile
    assert single[0] is None and single[1] is not None and single[2] is None
    hi = app.run(str(img), "Compare all profiles", "b0", 0.95, 0.7)                                     # controls reach every predictor; empty result is not an error
    assert all(r is not None for r in hi[:3])
    assert json.loads(Path(hi[9]).read_text())["results"]["b0"]["settings"]["conf"] == 0.95
    with pytest.raises(gr.Error):
        app.run(None, "Single model", "b0", 0.25, 0.7)
    bad = tmp_path / "bad.png"; bad.write_bytes(b"nope")
    with pytest.raises(gr.Error, match="cannot decode"):
        app.run(str(bad), "Single model", "b0", 0.25, 0.7)
    import argparse
    ns = argparse.ArgumentParser()
    host_default = [a.default for a in app.argparse.ArgumentParser.__init__.__globals__.get("_a", [])] if False else None
    src = Path(app.__file__).read_text()
    assert '"--host", default="127.0.0.1"' in src and "share=False" in src and src.count("." .join(["0"] * 4)) == 0


# ---- review fixes: input edge cases, drift, temp-file hygiene ---------------------------------------------------------------
def test_rgba_composited_on_white_and_16bit_gray_scaled(tmp_path):
    rgba = Image.new("RGBA", (20, 20), (255, 0, 0, 0)); rgba.putpixel((5, 5), (0, 0, 255, 255)); p = tmp_path / "t.png"; rgba.save(p)
    a, _ = core.load_image(p); assert tuple(a[0, 0]) == (255, 255, 255) and tuple(a[5, 5]) == (0, 0, 255)    # transparent -> white, opaque kept
    g = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 16); q = tmp_path / "g16.png"; Image.fromarray(g).save(q)
    b, _ = core.load_image(q); assert b.shape == (64, 64, 3) and 100 < float(b.mean()) < 160, float(b.mean())   # mean ~127, not saturated at 255


def test_decompression_bomb_rejected_cleanly(tmp_path):
    p = tmp_path / "bomb.png"; Image.new("L", (14000, 14000)).save(p)           # ~196 MP of zeros, tiny on disk
    with pytest.raises(InferenceError):
        core.load_image(p)


def test_first_image_not_pathologically_slow_after_load():
    if not (HAVE and VAL):
        pytest.skip("needs weights and images")
    m = core.load_model("b0", "cuda:0" if torch.cuda.is_available() else "cpu")
    r = core.predict(VAL[0], m)
    assert r["timing_ms"]["postprocess_nms"] < 15.0, r["timing_ms"]               # NMS kernels are initialised during warm-up


@needs_w
@needs_img
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_fp16_drift_bounded_vs_fp32():
    for pid in PROFILES:
        a = core.predict(VAL[0], core.load_model(pid, "cuda:0", "fp32", warmup=False), conf=0.25); b = core.predict(VAL[0], core.load_model(pid, "cuda:0", "fp16", warmup=False), conf=0.25)
        assert abs(a["num_detections"] - b["num_detections"]) <= 1
        if a["num_detections"] == b["num_detections"] and a["num_detections"]:
            assert abs(a["detections"][0]["confidence"] - b["detections"][0]["confidence"]) < 0.05
            assert max(abs(x - y) for x, y in zip(a["detections"][0]["box_xyxy"], b["detections"][0]["box_xyxy"])) < 4.0


@needs_w
def test_ui_leaves_no_files_after_exit():
    """The UI process must keep uploads/exports in ONE private dir that is removed at exit (no p05_* or gradio residue we created)."""
    pytest.importorskip("gradio")
    import glob, os, tempfile, time
    code = ("from p05_detect.infer import app; from pathlib import Path; "
            "app.ARGS.update(device='cpu', precision='fp32'); "
            "o = app.run(str(Path('examples/synthetic_fixture.png')), 'Compare all profiles', 'b0', 0.25, 0.7); "
            "print('TMP=' + app.TMP); assert Path(o[9]).parent == Path(app.TMP) and Path(o[10]).parent == Path(app.TMP)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT); assert r.returncode == 0, r.stderr[-400:]
    tmp = [l for l in r.stdout.splitlines() if l.startswith("TMP=")][0][4:]
    assert not os.path.exists(tmp), "private UI temp dir must be removed at process exit"


@needs_w
def test_ui_temp_dir_removed_on_sigterm():
    pytest.importorskip("gradio")
    import os, signal, time
    code = ("from p05_detect.infer import app; import sys, time; app._init_tmp(); print('TMP=' + app.TMP, flush=True); time.sleep(60)")
    pr = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, cwd=ROOT)
    tmp = pr.stdout.readline().strip()[4:]
    assert os.path.isdir(tmp)
    pr.send_signal(signal.SIGTERM); pr.wait(timeout=20)
    assert not os.path.exists(tmp), "private UI temp dir must be removed on SIGTERM"


def test_importing_app_has_no_side_effects():
    pytest.importorskip("gradio")
    r = subprocess.run([sys.executable, "-c", "import signal; h = signal.getsignal(signal.SIGTERM); from p05_detect.infer import app; assert app.TMP is None and signal.getsignal(signal.SIGTERM) is h; print('ok')"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-300:]


def test_missing_optional_heif_decoder_gives_inference_error(tmp_path, monkeypatch):
    """Ultralytics' Image.open patch imports 'pi_heif' on any PIL failure; a missing module must surface as InferenceError."""
    def boom(*a, **k):
        try:
            raise Image.UnidentifiedImageError("cannot identify image file")
        except Exception:
            raise ModuleNotFoundError("No module named 'pi_heif'", name="pi_heif")
    monkeypatch.setattr(core.Image, "open", boom)
    p = tmp_path / "bad.png"; p.write_bytes(b"not an image")
    with pytest.raises(InferenceError, match=r"cannot decode.*UnidentifiedImageError.*pi_heif.*optional") as ei:
        core.load_image(p)
    assert isinstance(ei.value.__cause__, ModuleNotFoundError)


def test_standard_formats_still_load(tmp_path):
    """Standard PNG/JPEG still load."""
    for ext in ("png", "jpg"):
        p = tmp_path / f"g.{ext}"; synth(40, 30).save(p)
        a, _ = core.load_image(p); assert a.shape == (30, 40, 3)
