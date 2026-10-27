import io, pytest, torch
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from p05_detect.engine.model import P05DetectionModel
from p05_detect.losses.nwd import NWDHybridBboxLoss, P05DetectionLoss, make_p05_loss_class

DEV = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def model(device="cpu", lam=.5, C=12.8, cfg="yolo26s.yaml"):
    m = P05DetectionModel(cfg, nc=1, verbose=False, nwd=dict(lam=lam, C=C))
    m.args = get_cfg(); m.nc = 1
    return m.to(device).train()


def batch(device, b=4, empty=False):
    g = torch.Generator().manual_seed(0); n = 0 if empty else b * 2
    xy = torch.rand(n, 2, generator=g) * .7 + .15
    return dict(img=torch.rand(b, 3, 320, 320, generator=g).to(device), batch_idx=torch.arange(b).repeat_interleave(2)[:n].float().to(device),
                cls=torch.zeros(n, 1).to(device), bboxes=torch.cat([xy, torch.full((n, 2), .05)], 1).to(device))


def test_criterion_is_e2e_with_hybrid_in_both_heads_and_topk_forwarded():
    m = model(); c = m.init_criterion()
    assert isinstance(c, E2ELoss)
    for head, topk, topk2 in ((c.one2many, 10, 10), (c.one2one, 7, 1)):   # assigner stores topk2 = topk2 or topk
        assert isinstance(head, P05DetectionLoss) and isinstance(head.bbox_loss, NWDHybridBboxLoss)
        assert head.bbox_loss.lam == .5 and head.bbox_loss.C == 12.8
        assert head.assigner.topk == topk and head.assigner.topk2 == topk2
    c.update(); assert c.updates == 1 and c.o2m < .8             # Progressive weighting preserved


def test_both_heads_actually_call_nwd(monkeypatch):
    m = model(); calls = []
    orig = NWDHybridBboxLoss.forward
    monkeypatch.setattr(NWDHybridBboxLoss, "forward", lambda self, *a, **k: (calls.append(id(self)), orig(self, *a, **k))[1])
    m.loss(batch("cpu"))
    assert len(calls) == 2 and len(set(calls)) == 2


@pytest.mark.parametrize("device", DEV)
def test_full_forward_backward_finite(device):
    m = model(device); loss, items = m.loss(batch(device))
    assert torch.isfinite(loss).all() and all(torch.isfinite(v).all() for v in (items.values() if isinstance(items, dict) else [items]))
    loss.sum().backward()
    g = [p.grad for p in m.parameters() if p.requires_grad]
    assert sum(x is not None for x in g) > 300 and all(torch.isfinite(x).all() for x in g if x is not None)


def test_empty_batch_finite():
    m = model(); loss, _ = m.loss(batch("cpu", empty=True)); loss.sum().backward()
    assert torch.isfinite(loss).all()


def test_lambda0_whole_model_loss_equals_upstream():
    m = model(lam=0.0); up = P05DetectionModel.__mro__[1]("yolo26s.yaml", nc=1, verbose=False); up.args = get_cfg(); up.nc = 1
    up.load_state_dict(m.state_dict()); up.train(); m.train()
    b = batch("cpu"); a, _ = m.loss(b); r, _ = up.loss(b)
    assert torch.equal(a, r)


def test_hybrid_changes_loss_vs_upstream():
    m = model(); up = P05DetectionModel.__mro__[1]("yolo26s.yaml", nc=1, verbose=False); up.args = get_cfg(); up.nc = 1
    up.load_state_dict(m.state_dict()); up.train()
    b = batch("cpu"); assert not torch.equal(m.loss(b)[0], up.loss(b)[0])


def test_checkpoint_roundtrip_inference_without_criterion():
    m = model(); sd = m.state_dict(); buf = io.BytesIO(); torch.save({"model": sd}, buf); buf.seek(0)
    m2 = P05DetectionModel("yolo26s.yaml", nc=1, verbose=False, nwd=dict(lam=.5, C=12.8)); m2.load_state_dict(torch.load(buf)["model"]); m2.eval()
    assert getattr(m2, 'criterion', None) is None          # criterion is lazily built by loss(); inference never needs it
    x = torch.rand(1, 3, 320, 320)
    with torch.no_grad(): y = m2(x)
    assert torch.isfinite(y[0] if isinstance(y, (list, tuple)) else y).all()
    # stable inference graph: identical keys as upstream
    up = P05DetectionModel.__mro__[1]("yolo26s.yaml", nc=1, verbose=False)
    assert list(up.state_dict()) == list(sd)


def test_state_dict_identical_to_upstream_so_E1_shares_708_tensors():
    m = model(); up = P05DetectionModel.__mro__[1]("yolo26s.yaml", nc=1, verbose=False)
    assert [(k, v.shape) for k, v in m.state_dict().items()] == [(k, v.shape) for k, v in up.state_dict().items()] and len(m.state_dict()) == 708
