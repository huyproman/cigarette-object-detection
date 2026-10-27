"""Tests for the NWD/CIoU hybrid box loss."""
import io, math, pytest, torch
from ultralytics.utils.loss import BboxLoss, E2ELoss
from p05_detect.losses.nwd import nwd_similarity, NWDHybridBboxLoss, P05DetectionLoss, pixel_boxes

B = torch.tensor([[10., 10., 30., 40.]])


def test_identical_boxes():
    assert abs(nwd_similarity(B, B, 12.8).item() - 1) < 3e-5


def test_monotone_in_centre_and_size():
    d = [nwd_similarity(B, B + torch.tensor([s, 0, s, 0.]), 12.8).item() for s in (0., 2., 4., 8., 16.)]
    assert all(a > b for a, b in zip(d, d[1:]))
    z = [nwd_similarity(B, B + torch.tensor([0, 0, s, s]), 12.8).item() for s in (0., 2., 4., 8.)]
    assert all(a > b for a, b in zip(z, z[1:]))


def test_known_value_shift4_c12p8():
    a = torch.tensor([[0., 0., 8., 8.]]); b = a + torch.tensor([4., 0, 4., 0])
    assert nwd_similarity(a, b, 12.8).item() == pytest.approx(math.exp(-math.sqrt(16 + 1e-7) / 12.8), rel=1e-6)
    assert nwd_similarity(a, b, 12.8).item() == pytest.approx(.7316, abs=1e-4)
    big = torch.tensor([[0., 0., 64., 64.]]); assert nwd_similarity(big, big + torch.tensor([4., 0, 4., 0]), 12.8).item() == pytest.approx(.7316, abs=1e-4)


def test_translation_invariance():
    t = torch.tensor([300., 200., 300., 200.])
    a = nwd_similarity(B, B + torch.tensor([3., 1, 5, 2]), 12.8); b = nwd_similarity(B + t, B + t + torch.tensor([3., 1, 5, 2]), 12.8)
    assert torch.allclose(a, b, atol=1e-6)


def test_zero_size_finite_grad():
    p = torch.tensor([[5., 5., 5., 5.]], requires_grad=True); t = torch.tensor([[5., 5., 5., 5.]])
    nwd_similarity(p, t, 12.8).sum().backward()
    assert torch.isfinite(p.grad).all()


def mk(reg_max=1, lam=.5, C=12.8, n_fg=3, A=20, nb=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    stride = torch.where(torch.arange(A) < 12, 8., 16.).view(A, 1)
    anchors = torch.rand(A, 2, generator=g) * 20
    ctr = anchors.unsqueeze(0).expand(nb, A, 2)
    wh = torch.rand(nb, A, 2, generator=g) * 6 + 1
    tgt = torch.cat([ctr - wh / 2, ctr + wh / 2], -1)
    pred = (tgt + (torch.rand(nb, A, 4, generator=g) - .5)).requires_grad_()
    dist = torch.rand(nb, A, 4 * reg_max, generator=g).requires_grad_()
    fg = torch.zeros(nb, A, dtype=torch.bool); fg[0, :n_fg] = True; fg[1, 5:5 + n_fg] = True
    scores = torch.rand(nb, A, 1, generator=g) * fg.unsqueeze(-1)
    return dict(pred_dist=dist, pred_bboxes=pred, anchor_points=anchors, target_bboxes=tgt, target_scores=scores,
                target_scores_sum=scores.sum().clamp(min=1), fg_mask=fg, imgsz=torch.tensor([640., 640.]), stride=stride), (lam, C, reg_max)


def test_empty_foreground_finite():
    kw, (lam, C, rm) = mk(n_fg=0)
    kw["fg_mask"] = torch.zeros_like(kw["fg_mask"]); kw["target_scores"] = torch.zeros_like(kw["target_scores"])
    loss = NWDHybridBboxLoss(rm, lam, C)(**kw)
    assert all(torch.isfinite(x) for x in loss) and loss[0].item() == 0
    (loss[0] + loss[1]).backward()
    assert torch.isfinite(kw["pred_bboxes"].grad).all()


def test_lambda0_exact_upstream_parity():
    kw, (_, C, rm) = mk()
    ref = BboxLoss(rm)(**{k: (v.detach().clone().requires_grad_() if k in ("pred_dist", "pred_bboxes") else v) for k, v in kw.items()})
    got = NWDHybridBboxLoss(rm, 0.0, C)(**kw)
    assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])


def test_hybrid_differs_and_l1_unchanged():
    kw, (lam, C, rm) = mk()
    ref = BboxLoss(rm)(**kw); got = NWDHybridBboxLoss(rm, .5, C)(**kw)
    assert not torch.allclose(ref[0], got[0]) and torch.equal(ref[1], got[1])
    only = NWDHybridBboxLoss(rm, 1.0, C)(**kw)
    nwd = nwd_similarity(pixel_boxes(kw["pred_bboxes"], kw["fg_mask"], kw["stride"]), pixel_boxes(kw["target_bboxes"], kw["fg_mask"], kw["stride"]), C)
    w = kw["target_scores"][kw["fg_mask"]].sum(-1)
    assert torch.allclose(only[0], ((1 - nwd) * w).sum() / kw["target_scores_sum"], atol=1e-6)
    assert torch.allclose(got[0], .5 * ref[0] + .5 * only[0], atol=1e-6)


def test_gradients_boxes_only_assignment_inputs_detached():
    kw, (lam, C, rm) = mk()
    kw["target_bboxes"].requires_grad_(False)
    loss = NWDHybridBboxLoss(rm, 1.0, C)(**kw)
    loss[0].backward()
    assert kw["pred_bboxes"].grad is not None and kw["pred_bboxes"].grad.abs().sum() > 0
    assert kw["pred_bboxes"].grad[~kw["fg_mask"]].abs().sum() == 0          # only positives receive NWD gradient
    assert kw["target_bboxes"].grad is None and kw["target_scores"].grad_fn is None


def test_pixel_recovery_uses_batch_anchor_indices():
    A = 6; stride = torch.tensor([8., 8., 8., 16., 16., 32.]).view(A, 1)
    box = torch.zeros(2, A, 4)
    box[0, 1] = torch.tensor([0, 0, 1.6, 1.6]); box[1, 3] = torch.tensor([0, 0, 8., 8.]); box[1, 5] = torch.tensor([0, 0, .3, .6])
    fg = torch.zeros(2, A, dtype=torch.bool); fg[0, 1] = fg[1, 3] = fg[1, 5] = True
    px = pixel_boxes(box, fg, stride)
    assert torch.allclose(px, torch.tensor([[0, 0, 12.8, 12.8], [0, 0, 128., 128.], [0, 0, 9.6, 19.2]]), atol=1e-5)
    # the flattened-index pitfall would pick stride[flat_idx % ...] wrongly; make sure it differs
    flat = fg.flatten().nonzero().squeeze(-1)
    assert flat.max() >= A                                           # flat index is out of range for stride (A,1): a known pitfall


def test_amp_vs_fp32():
    g = torch.Generator().manual_seed(1)
    a = torch.rand(256, 4, generator=g) * 100; a[:, 2:] += a[:, :2] + 4; b = a + torch.randn(256, 4, generator=g) * 3
    ref = nwd_similarity(a.half().float(), b.half().float(), 12.8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        amp = nwd_similarity(a.half(), b.half(), 12.8)
    assert amp.dtype == torch.float32 and torch.allclose(amp, ref, atol=1e-4, rtol=1e-3)


def test_invalid_config_and_coordinates():
    for lam in (-.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError): NWDHybridBboxLoss(1, lam, 12.8)
    for C in (0., -1., float("nan"), float("inf")):
        with pytest.raises(ValueError): NWDHybridBboxLoss(1, .5, C)
    with pytest.raises(ValueError): nwd_similarity(torch.tensor([[0., 0., float("nan"), 1.]]), B, 12.8)
    with pytest.raises(ValueError): nwd_similarity(B, B, 0.)
    with pytest.raises(ValueError): nwd_similarity(B, B, 12.8, eps=0)
