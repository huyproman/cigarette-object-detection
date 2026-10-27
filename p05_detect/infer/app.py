"""Local three-profile comparison demo (Gradio). Binds 127.0.0.1, analytics/sharing disabled; uploads/exports live in a private temp dir removed at exit.
Start: python -m p05_detect.infer.app   (see README)"""
import argparse, json, os, sys, tempfile

import atexit, shutil
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
# All uploads Gradio copies and all exports we write live in ONE private directory that is deleted when the process exits.
# Created lazily by _init_tmp() (called from main() / run()), so importing this module has no side effects.
TMP = None
def _init_tmp():
    global TMP
    if TMP is not None: return TMP
    TMP = tempfile.mkdtemp(prefix="p05_ui_"); os.chmod(TMP, 0o700); os.environ["GRADIO_TEMP_DIR"] = TMP
    atexit.register(shutil.rmtree, TMP, ignore_errors=True)
    import signal
    def _bye(signum, frame):                  # SIGTERM / SIGHUP skip atexit by default; clean up, then exit with the conventional status
        shutil.rmtree(TMP, ignore_errors=True); os._exit(128 + signum)
    for _s in (signal.SIGTERM, signal.SIGHUP):
        try: signal.signal(_s, _bye)
        except (ValueError, OSError): pass
    return TMP
import gradio as gr
from PIL import Image

from p05_detect.infer import core
from p05_detect.infer.montage import montage
from p05_detect.infer.registry import PROFILES, InferenceError, load_registry

REG = load_registry()
NOTE = ("Detects **cigarette objects** (bounding-box detection). It does **not** recognise smoking actions, track people or run in real time. "
        "Demo panels use one checkpoint per profile (seed 101), not the three-seed research averages. All processing is local.")
CARDS = {
    "b0": "**YOLO26s baseline (`b0`)**: default reference. Simplest graph, lowest overhead.",
    "e2_carafe": "**YOLO26s + CARAFE (`e2_carafe`)**: optional comparison; content-aware upsampling at layers 11 and 14.",
    "e3_combined": "**YOLO26s + NWD + CARAFE (`e3_combined`)**: optional comparison; same CARAFE inference graph as `e2_carafe`, NWD is a training-loss change only (no extra inference compute).",
}
import re as _re
CARDS_HTML = {k: _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", v) for k, v in CARDS.items()}
_CACHE = {}
ARGS = {"device": "auto", "precision": "fp32"}


def _model(pid):
    k = (pid, ARGS["device"], ARGS["precision"])
    if k not in _CACHE:
        _CACHE[k] = core.load_model(pid, ARGS["device"], ARGS["precision"])
    return _CACHE[k]


def _rows(res):
    return [[i + 1, f"{d['confidence']:.3f}", "[" + ", ".join(f"{v:.1f}" for v in d["box_xyxy"]) + "]"] for i, d in enumerate(res["detections"])]


def _timing(res):
    t = res["timing_ms"]
    return (f"end-to-end {t['end_to_end']:.1f} ms (decode+letterbox+H2D+forward+NMS+output, steady state) | model-forward {t['model_forward']:.1f} ms | "
            f"cold load {t['model_load_cold']:.0f} ms + warmup {t['warmup']:.0f} ms (one-time, not in end-to-end) | {res['settings']['device']} {res['settings']['precision']}")


def run(image_path, mode, single_profile, conf, iou):
    if not image_path:
        raise gr.Error("Upload a JPG or PNG image first.")
    try:
        img, name = core.load_image(image_path)
        profiles = list(PROFILES) if mode.startswith("Compare") else [single_profile]
        results = {}
        for p in profiles:
            results[p] = core.predict(None, _model(p), float(conf), float(iou), _loaded=(img, name))
    except InferenceError as e:
        raise gr.Error(str(e))
    outs, tables, notes = [], [], []
    for p in PROFILES:
        if p in results:
            r = results[p]; outs.append(core.draw_overlay(img, r)); tables.append(gr.update(value=_rows(r), visible=True))
            notes.append(f"{r['model']['display_name']}: **{r['num_detections']}** detection(s) at conf {conf}, IoU {iou}" + ("" if r["num_detections"] else " (no cigarette found at this threshold)") + "\n\n" + _timing(r))
        else:
            outs.append(None); tables.append(gr.update(value=[], visible=False)); notes.append("(not run in single-model mode)")
    js = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, prefix="p05_", dir=_init_tmp()); json.dump(dict(schema_version=core.SCHEMA_VERSION, results=results), js, indent=2); js.close()
    mp = None
    if len(results) > 1:
        mp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="p05_", dir=_init_tmp()).name; montage(img, results).save(mp)
    return [*outs, *tables, *notes, js.name, mp]


def build():
    with gr.Blocks(title="P05 cigarette detection") as demo:
        gr.Markdown("# Cigarette object detection: three profiles\n" + NOTE)
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="filepath", label="Input image (JPG/PNG, local)", sources=["upload"], image_mode=None)
                mode = gr.Radio(["Single model", "Compare all profiles"], value="Single model", label="Mode")
                single = gr.Radio(list(PROFILES), value="b0", label="Single-model profile (default: b0)")
                conf = gr.Slider(0.01, 0.95, value=0.25, step=0.01, label="Confidence threshold (applied to all models)")
                iou = gr.Slider(0.1, 0.95, value=0.70, step=0.05, label="NMS IoU (applied to all models)")
                go = gr.Button("Run", variant="primary")
                gr.Markdown(f"Limits: ≤ {core.MAX_FILE_BYTES // 2**20} MiB, ≤ {core.MAX_PIXELS // 10**6} MP. Uploads and exports are kept only in a private temporary folder that is deleted when the app exits; nothing is sent anywhere.")
            with gr.Column(scale=3):
                panels, tabs, notes = [], [], []
                with gr.Row():
                    for p in PROFILES:
                        with gr.Column():
                            gr.HTML(f'<div style="min-height:11em">{CARDS_HTML[p]}</div>')
                            panels.append(gr.Image(label=p, interactive=False, type="pil"))
                            notes.append(gr.Markdown())
                            tabs.append(gr.Dataframe(headers=["#", "conf", "box xyxy (px)"], interactive=False, visible=False, wrap=True, column_widths=["10%", "20%", "70%"], label=f"{p} boxes (original pixels)"))
                with gr.Row():
                    jf = gr.File(label="Export JSON (local)"); mf = gr.File(label="Export montage PNG (local)")
        go.click(run, [inp, mode, single, conf, iou], [*panels, *tabs, *notes, jf, mf])
    return demo


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m p05_detect.infer.app"); ap.add_argument("--device", default="auto"); ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--port", type=int, default=7860); ap.add_argument("--host", default="127.0.0.1", help="default localhost; do not expose without review")
    ap.add_argument("--weights-dir"); a = ap.parse_args(argv)
    if a.weights_dir: os.environ["P05_WEIGHTS_DIR"] = a.weights_dir
    ARGS.update(device=a.device, precision=a.precision); _init_tmp()
    try:
        for p in PROFILES: core.resolve_checkpoint(p)
    except InferenceError as e:
        print(f"error: {e}", file=sys.stderr); return 2
    build().queue(max_size=4).launch(server_name=a.host, server_port=a.port, share=False, show_error=True, inbrowser=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
