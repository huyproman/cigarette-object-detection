"""Box-level eligibility rule used to build the corrected dataset, extracted so it is testable: boxes failing the oversize
heuristic are dropped, as are boxes whose shorter side is below 3.0 px after the 640 resize."""
SUBPIXEL_MIN_SIDE_640 = 3.0

def box_disposition(box_xyxy, w, h):
    x1, y1, x2, y2 = box_xyxy; bw, bh = x2 - x1, y2 - y1; nw, nh = bw / w, bh / h; s = 640.0 / max(w, h)
    if nw > 0.25 or nh > 0.30 or nw * nh > 0.05:
        return "DISCARDED_OVERSIZED_NON_CIGARETTE"
    if min(bw, bh) * s < SUBPIXEL_MIN_SIDE_640:
        return "DISCARDED_SUB_PIXEL"
    return "RETAINED"
