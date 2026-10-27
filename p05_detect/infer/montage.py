from PIL import Image, ImageDraw
from p05_detect.infer import core

def montage(rgb, results, height=480):
    """Side-by-side panels of the SAME original image; boxes come from each result's returned coordinates."""
    panels = []
    for pid, res in results.items():
        im = core.draw_overlay(rgb, res); s = height / im.height; im = im.resize((max(1, round(im.width * s)), height))
        bar = Image.new("RGB", (im.width, 44), (30, 30, 30)); d = ImageDraw.Draw(bar)
        d.text((6, 4), f"{res['model']['display_name']}  [{pid}]", fill=(255, 255, 255))
        d.text((6, 22), f"{res['num_detections']} det @conf {res['settings']['conf']} iou {res['settings']['iou']}  seed {res['model']['seed']}  ck {res['model']['checkpoint_sha256'][:8]}", fill=(200, 200, 200))
        p = Image.new("RGB", (im.width, height + 44)); p.paste(bar, (0, 0)); p.paste(im, (0, 44)); panels.append(p)
    W = sum(p.width for p in panels) + 8 * (len(panels) - 1); M = Image.new("RGB", (W, panels[0].height), (255, 255, 255)); x = 0
    for p in panels: M.paste(p, (x, 0)); x += p.width + 8
    return M
