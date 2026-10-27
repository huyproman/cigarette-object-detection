"""Shared inference core for the three profiles (b0, e2_carafe, e3_combined); the CLI and UI call this module.
Pipeline mirrors the Ultralytics 8.4.174 predictor used for evaluation: LetterBox(640) -> RGB/255 -> AutoBackend(fused)
-> non_max_suppression(one2many, NMS) -> scale_boxes to original pixels. Profiles differ only in graph/weights."""
import copy, time, warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

import p05_detect.engine.model  # noqa: F401  registers CARAFE with the pinned parser (required to load e2_carafe/e3_combined)
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import nms as _nms, ops

from p05_detect.infer.registry import InferenceError, PROFILES, load_registry, resolve_checkpoint

SCHEMA_VERSION = "1"
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 40_000_000
MIN_SIDE = 2
IMGSZ = 640
MAX_DET = 300
EXTS = {".jpg", ".jpeg", ".png"}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_device(device="auto"):
    if device in (None, "auto"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        d = torch.device(device if ":" in str(device) or str(device) == "cpu" else (f"cuda:{device}" if str(device).isdigit() else device))
    except (RuntimeError, ValueError):
        raise InferenceError(f"unsupported device '{device}' (use auto, cpu or cuda[:N])")
    if d.type == "cuda":
        if not torch.cuda.is_available():
            raise InferenceError(f"device '{device}' requested but CUDA is not available")
        if d.index is not None and d.index >= torch.cuda.device_count():
            raise InferenceError(f"device '{device}' requested but only {torch.cuda.device_count()} CUDA device(s) exist")
    elif d.type != "cpu":
        raise InferenceError(f"unsupported device '{device}' (use auto, cpu or cuda[:N])")
    return d


def load_image(source, max_bytes=MAX_FILE_BYTES, max_pixels=MAX_PIXELS):
    """Return (RGB uint8 HxWx3 ndarray, name). Applies EXIF orientation, converts gray/RGBA/palette to RGB, enforces limits."""
    if isinstance(source, np.ndarray):
        a = source
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        if a.ndim != 3 or a.shape[2] not in (3, 4) or a.dtype != np.uint8:
            raise InferenceError("array input must be uint8 HxWx3 (RGB) or HxWx4 (RGBA)")
        if a.shape[0] * a.shape[1] > max_pixels:
            raise InferenceError(f"image too large: {a.shape[1]}x{a.shape[0]} exceeds {max_pixels} pixels")
        return _check(np.ascontiguousarray(a[:, :, :3])), "array"
    if isinstance(source, Image.Image):
        im, name = source, "pil"
    else:
        p = Path(source)
        if not p.is_file():
            raise InferenceError(f"image not found: {p}")
        if p.suffix.lower() not in EXTS:
            raise InferenceError(f"unsupported image type '{p.suffix}' (JPG/PNG only)")
        if p.stat().st_size > max_bytes:
            raise InferenceError(f"file too large: {p.stat().st_size} bytes exceeds {max_bytes}")
        name = p.name
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                im = Image.open(p)
                im.verify()
                im = Image.open(p)
        except ImportError as e:
            # Ultralytics patches Image.open: on any PIL failure it lazily imports the optional 'pi-heif' plugin. Report the
            # original PIL failure (the ImportError's __context__) plus the missing optional dependency instead of leaking it.
            first = e.__context__
            raise InferenceError(f"cannot decode image '{p.name}': {type(first).__name__}: {first}; the HEIF fallback decoder "
                                 f"('{e.name or 'pi-heif'}') is optional and not installed") from e
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
            raise InferenceError(f"cannot decode image '{p.name}': {type(e).__name__}: {e}")
    if im.width * im.height > max_pixels:
        raise InferenceError(f"image too large: {im.width}x{im.height} exceeds {max_pixels} pixels")
    try:
        im.load()
        im = ImageOps.exif_transpose(im)
        if im.mode.startswith(("I", "F")):            # 16/32-bit gray: PIL's convert('RGB') would saturate; scale to 8 bit explicitly
            a16 = np.asarray(im).astype(np.float64)
            hi = 65535.0 if a16.max() > 255 else 255.0
            g = np.clip(a16 / hi * 255.0, 0, 255).astype(np.uint8)
            im = Image.fromarray(np.stack([g] * 3, -1))
        if im.mode in ("RGBA", "LA", "P"):
            rgba = im.convert("RGBA"); bg = Image.new("RGB", rgba.size, (255, 255, 255)); bg.paste(rgba, mask=rgba.split()[3]); im = bg
        else:
            im = im.convert("RGB")
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as e:
        raise InferenceError(f"cannot decode image '{name}': {type(e).__name__}: {e}")
    return _check(np.asarray(im, dtype=np.uint8)), name


def _check(a):
    if min(a.shape[:2]) < MIN_SIDE:
        raise InferenceError(f"image too small: {a.shape[1]}x{a.shape[0]}")
    return a


@dataclass
class Model:
    profile: dict
    backend: AutoBackend
    device: torch.device
    precision: str
    checkpoint_path: str
    load_ms: float
    warmup_ms: float = 0.0
    imgsz: int = IMGSZ
    letterbox: LetterBox = field(default=None)


def load_model(profile_id, device="auto", precision="fp32", checkpoint_override=None, warmup=True, imgsz=IMGSZ, verify_profile_hash=True):
    if precision not in ("fp32", "fp16"):
        raise InferenceError(f"unsupported precision '{precision}' (fp32 or fp16)")
    dev = resolve_device(device)
    if precision == "fp16" and dev.type != "cuda":
        raise InferenceError("fp16 requires a CUDA device; no CPU fallback is performed (use --precision fp32)")
    t0 = time.perf_counter()
    path, prof = resolve_checkpoint(profile_id, checkpoint_override, verify_profile_hash=verify_profile_hash)
    try:
        # Load route of the evaluation predictor (YOLO() -> deepcopy -> AutoBackend(module)). Passing the path instead fuses the
        # fp16-stored checkpoint weights differently (~1e-6 per tensor) and breaks prediction parity.
        net = copy.deepcopy(YOLO(str(path)).model)
        be = AutoBackend(model=net, device=dev, fp16=precision == "fp16", fuse=True, verbose=False, end2end=False)
    except Exception as e:  # incompatible/corrupt checkpoint
        raise InferenceError(f"failed to load checkpoint for '{profile_id}': {type(e).__name__}: {e}")
    be.eval()
    names = dict(be.names)
    if names != {0: "cigarette"}:
        raise InferenceError(f"unexpected class map {names} for '{profile_id}' (expected {{0: 'cigarette'}})")
    _sync(dev)
    m = Model(prof, be, dev, precision, str(path), (time.perf_counter() - t0) * 1e3, imgsz=imgsz, letterbox=LetterBox((imgsz, imgsz), auto=False, stride=32))
    if warmup:
        w0 = time.perf_counter()
        x = torch.full((1, 3, imgsz, imgsz), 0.45, device=dev, dtype=torch.half if precision == "fp16" else torch.float32)
        with torch.inference_mode():
            for _ in range(3):
                raw = be(x)
        # NMS has lazy one-time cost on its first NON-EMPTY call (~70 ms); trigger it with a synthetic head output holding two overlapping candidates.
        num_anchors = raw[0].shape[-1] if isinstance(raw, (list, tuple)) else raw.shape[-1]
        fake = torch.zeros(1, 5, num_anchors, device=dev, dtype=raw[0].dtype if isinstance(raw, (list, tuple)) else raw.dtype)
        fake[0, :4, 0] = torch.tensor([imgsz / 2.0, imgsz / 2.0, 40.0, 30.0]); fake[0, 4, 0] = 0.9
        fake[0, :4, 1] = torch.tensor([imgsz / 2.0 + 2.0, imgsz / 2.0 + 1.0, 40.0, 30.0]); fake[0, 4, 1] = 0.8
        for _ in range(3):
            _nms.non_max_suppression(fake, 0.25, 0.7, None, False, max_det=MAX_DET, nc=0, end2end=False)
        _sync(dev); m.warmup_ms = (time.perf_counter() - w0) * 1e3
    return m


def preprocess(m, rgb):
    """RGB uint8 HxWx3 -> (1,3,640,640) tensor on device (letterbox pad 114, /255). Returns (tensor, host_ms, h2d_ms)."""
    t0 = time.perf_counter()
    lb = m.letterbox(image=rgb)                      # channel-agnostic; pad value 114 on all channels
    t = torch.from_numpy(lb).unsqueeze(0)
    t1 = time.perf_counter()
    t = t.to(m.device).permute(0, 3, 1, 2).contiguous()
    t = (t.half() if m.precision == "fp16" else t.float()).div_(255)
    _sync(m.device)
    return t, (t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3


def forward(m, tensor):
    with torch.inference_mode():
        return m.backend(tensor)


def postprocess(m, raw, tensor, orig_shape, conf, iou, max_det=MAX_DET):
    p = _nms.non_max_suppression(raw, conf, iou, None, False, max_det=max_det, nc=0, end2end=getattr(m.backend, "end2end", False))[0]
    p = p.clone()
    p[:, :4] = ops.scale_boxes(tensor.shape[2:], p[:, :4], orig_shape)
    return p


def detections_from(p, w, h):
    """NMS output tensor (N,6, original pixels) -> validated, clipped, serializable detection records."""
    dets = []
    for r in p.float().cpu().numpy():
        x1, y1, x2, y2 = (float(min(max(r[0], 0), w)), float(min(max(r[1], 0), h)), float(min(max(r[2], 0), w)), float(min(max(r[3], 0), h)))
        if x2 <= x1 or y2 <= y1:
            continue
        dets.append(dict(class_id=int(r[5]), class_name="cigarette", confidence=float(min(max(r[4], 0.0), 1.0)),
                         box_xyxy=[x1, y1, x2, y2], box_xyxy_normalized=[x1 / w, y1 / h, x2 / w, y2 / h]))
    dets.sort(key=lambda d: -d["confidence"])
    return dets


def predict_batch(rgbs, model, conf=0.25, iou=0.70, max_det=MAX_DET):
    """Batched variant of the same preprocess/forward/postprocess functions (for parity with evaluation runs, which
    predict in 32-image chunks). Returns a list of detection lists."""
    xs = [preprocess(model, r)[0] for r in rgbs]
    X = torch.cat(xs)
    raw = forward(model, X)
    out = _nms.non_max_suppression(raw, conf, iou, None, False, max_det=max_det, nc=0, end2end=getattr(model.backend, "end2end", False))
    res = []
    for o, r in zip(out, rgbs):
        o = o.clone(); o[:, :4] = ops.scale_boxes(X.shape[2:], o[:, :4], r.shape); res.append(detections_from(o, r.shape[1], r.shape[0]))
    return res


def predict(image, model, conf=0.25, iou=0.70, head="one-to-many", max_det=MAX_DET, _loaded=None):
    """Run one image through one loaded model. `model` is a Model (or a profile id, which is loaded fp32/auto)."""
    if isinstance(model, str):
        model = load_model(model)
    if head not in ("one-to-many", "one2many"):
        raise InferenceError("only the primary one-to-many head (with NMS) is supported")
    if not (0.0 <= conf <= 1.0) or not (0.0 < iou <= 1.0):
        raise InferenceError("conf must be in [0,1] and iou in (0,1]")
    t_all = time.perf_counter()
    rgb, name = _loaded if _loaded is not None else load_image(image)
    decode_ms = (time.perf_counter() - t_all) * 1e3 if _loaded is None else 0.0
    h, w = rgb.shape[:2]
    x, host_ms, h2d_ms = preprocess(model, rgb)
    t = time.perf_counter(); raw = forward(model, x); _sync(model.device); fwd_ms = (time.perf_counter() - t) * 1e3
    t = time.perf_counter(); p = postprocess(model, raw, x, (h, w), conf, iou, max_det); post_ms = (time.perf_counter() - t) * 1e3
    dets = detections_from(p, w, h)
    out_t = time.perf_counter()
    res = dict(schema_version=SCHEMA_VERSION, image=dict(name=name, width=w, height=h),
               model=dict(profile_id=model.profile["profile_id"], display_name=model.profile["display_name"], seed=model.profile["seed"], run_id=model.profile["run_id"],
                          checkpoint_sha256=model.profile["checkpoint_sha256"], checkpoint_file=model.profile["checkpoint_file"], arm=model.profile["arm"]),
               settings=dict(device=str(model.device), precision=model.precision, conf=conf, iou=iou, head="one-to-many", nms=True, max_det=max_det, imgsz=model.imgsz,
                             box_convention="xyxy, original image pixels, origin top-left, x right / y down, clipped to [0,w]x[0,h]"),
               detections=dets, num_detections=len(dets))
    total = (time.perf_counter() - t_all) * 1e3
    res["timing_ms"] = dict(decode=decode_ms, letterbox_normalize_host=host_ms, host_to_device=h2d_ms, model_forward=fwd_ms, postprocess_nms=post_ms,
                            output_creation=(time.perf_counter() - out_t) * 1e3, end_to_end=total + (time.perf_counter() - out_t) * 1e3,
                            model_load_cold=model.load_ms, warmup=model.warmup_ms,
                            scope="end_to_end = decode+EXIF/RGB + letterbox + normalize + H2D + forward + NMS + result creation per image. Excludes model_load_cold/warmup. Lazy one-time GPU initialisation is triggered during warm-up, so the first image is close to steady state.")
    return res


def compare(image, profiles=PROFILES, conf=0.25, iou=0.70, device="auto", precision="fp32", models=None):
    """Same decoded image, same thresholds, shared code path; only the loaded model differs."""
    for p in profiles:
        if p not in PROFILES:
            raise InferenceError(f"unknown profile '{p}'; valid profiles: {', '.join(PROFILES)}")
    loaded = load_image(image)
    out = {}
    for p in profiles:
        m = (models or {}).get(p) or load_model(p, device, precision)
        out[p] = predict(None, m, conf, iou, _loaded=loaded)
    return out


def serialize(result, indent=2):
    import json
    return json.dumps(result, indent=indent)


def draw_overlay(image, result, color=(255, 64, 0)):
    """Render boxes from the returned coordinates only. Returns PIL image (original resolution)."""
    from PIL import ImageDraw
    rgb, _ = load_image(image) if not isinstance(image, np.ndarray) or image.shape[2] != 3 else (image, "")
    im = Image.fromarray(rgb); d = ImageDraw.Draw(im); lw = max(2, round(max(im.size) / 300))
    for det in result["detections"]:
        b = det["box_xyxy"]; d.rectangle(b, outline=color, width=lw)
        txt = f"{det['confidence']:.2f}"; ty = max(0, b[1] - 12); d.rectangle([b[0], ty, b[0] + 7 * len(txt) + 4, ty + 12], fill=color); d.text((b[0] + 2, ty), txt, fill=(255, 255, 255))
    return im
