"""CLI: python -m p05_detect.infer {infer,compare,verify} ... (see --help)."""
import argparse, json, sys
from pathlib import Path

from p05_detect.infer import core
from p05_detect.infer.registry import PROFILES, InferenceError, load_registry, resolve_checkpoint

def _common(p):
    p.add_argument("image", help="local JPG/PNG image")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N (default auto)")
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16"], help="fp16 requires CUDA (no silent fallback)")
    p.add_argument("--conf", type=float, default=0.25, help="display/operating confidence threshold (default 0.25; AP evaluation uses 0.001)")
    p.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold (default 0.70)")
    p.add_argument("--out", default="p05_out", help="output directory (default ./p05_out)")
    p.add_argument("--no-overlay", action="store_true", help="skip annotated image output")
    p.add_argument("--weights-dir", help="directory with the authorized checkpoint files (default: $P05_WEIGHTS_DIR or weights)")

def build():
    ap = argparse.ArgumentParser(prog="python -m p05_detect.infer", description="Cigarette object detection (not smoking-action recognition). Profiles: " + ", ".join(PROFILES))
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("infer", help="one image, one profile"); _common(a); a.add_argument("--profile", default="b0", choices=PROFILES, help="default b0")
    c = sub.add_parser("compare", help="one image, all three profiles side by side"); _common(c)
    v = sub.add_parser("verify", help="verify checkpoint files against the registry SHA256"); v.add_argument("--weights-dir")
    return ap

def _save(res, img, stem, out, overlay):
    out.mkdir(parents=True, exist_ok=True)
    jp = out / f"{stem}.json"; jp.write_text(core.serialize(res))
    if overlay:
        core.draw_overlay(img, res).save(out / f"{stem}.png")
    return jp

def main(argv=None):
    ap = build(); a = ap.parse_args(argv)
    try:
        if getattr(a, "weights_dir", None):
            import os; os.environ["P05_WEIGHTS_DIR"] = a.weights_dir
        if a.cmd == "verify":
            for p in PROFILES:
                path, prof = resolve_checkpoint(p); print(f"OK {p}: {path.name} sha256={prof['checkpoint_sha256']}")
            return 0
        out = Path(a.out); img, name = core.load_image(a.image)
        profiles = [a.profile] if a.cmd == "infer" else list(PROFILES)
        results = {}
        for p in profiles:
            m = core.load_model(p, a.device, a.precision)
            res = core.predict(None, m, a.conf, a.iou, _loaded=(img, name)); results[p] = res
            jp = _save(res, img, f"{Path(name).stem}.{p}", out, not a.no_overlay)
            print(f"{p}: {res['num_detections']} detection(s), e2e {res['timing_ms']['end_to_end']:.1f} ms, forward {res['timing_ms']['model_forward']:.1f} ms -> {jp}")
        if a.cmd == "compare":
            from p05_detect.infer.montage import montage
            (out / f"{Path(name).stem}.compare.json").write_text(json.dumps(dict(schema_version=core.SCHEMA_VERSION, results=results), indent=2))
            if not a.no_overlay:
                mp = out / f"{Path(name).stem}.compare.png"; montage(img, results).save(mp); print(f"montage -> {mp}")
        return 0
    except InferenceError as e:
        print(f"error: {e}", file=sys.stderr); return 2

if __name__ == "__main__":
    sys.exit(main())
