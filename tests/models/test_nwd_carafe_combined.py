"""CARAFE + NWD model correctness: build, placements, NWD in both heads, FP32/AMP forward-backward, gradients, checkpoint, paired init."""
import io, pytest, torch
from pathlib import Path
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import E2ELoss
from p05_detect.engine.model import P05DetectionModel, CONFIG_DIR
from p05_detect.modules.carafe import CARAFE
from p05_detect.experiments.paired_init import build_paired_model, common_equal, carafe_state
from p05_detect.losses import nwd as NWD

ROOT = Path(__file__).parents[2]
INIT = {s: ROOT / f"runtime/init_states/B0_seed{s}_init_state.pt" for s in (101, 202, 303)}
# NWD normalization constant C at the 640 input frame.
C = 61.0
NWDCFG = dict(lam=0.5, C=C)
CUDA = torch.cuda.is_available()
Y = str(CONFIG_DIR / "yolo26s-carafe.yaml")

def e3(dev="cpu"):
    m = P05DetectionModel(Y, nc=1, verbose=False, nwd=NWDCFG); m.args = get_cfg(); m.nc = 1; return m.to(dev)

def batch(dev, n=2):
    return dict(img=torch.rand(n, 3, 640, 640, device=dev), batch_idx=torch.arange(n, dtype=torch.float, device=dev), cls=torch.zeros(n, 1, device=dev),
                bboxes=torch.tensor([[.5, .5, .2, .2], [.4, .4, .05, .05]][:n], device=dev))

def test_carafe_placements_and_hyperparameters():
    m = e3(); assert isinstance(m.model[11], CARAFE) and isinstance(m.model[14], CARAFE)
    for i, c1 in ((11, 512), (14, 256)):
        L = m.model[i]; assert (L.c1, L.k, L.scale) == (c1, 5, 2) and L.comp.out_channels == 64 and L.enc.kernel_size == (3, 3) and L.enc.out_channels == 100
    assert not any(isinstance(l, torch.nn.Upsample) for l in m.model)

def test_nwd_active_in_both_heads(monkeypatch):
    m = e3(); crit = m.init_criterion(); assert isinstance(crit, E2ELoss)
    for head in (crit.one2many, crit.one2one):
        assert isinstance(head.bbox_loss, NWD.NWDHybridBboxLoss) and head.bbox_loss.lam == 0.5 and head.bbox_loss.C == 61.0
    calls = []; orig = NWD.nwd_similarity; monkeypatch.setattr(NWD, "nwd_similarity", lambda *a, **k: calls.append(1) or orig(*a, **k))
    m.train(); loss, _ = m.loss(batch("cpu")); assert len(calls) >= 2 and torch.isfinite(loss).all()      # one call per head

@pytest.mark.parametrize("dev", ["cpu"] + (["cuda:0"] if CUDA else []))
def test_fp32_forward_backward_and_grad_finite(dev):
    m = e3(dev).train(); loss, _ = m.loss(batch(dev)); loss.sum().backward()
    assert torch.isfinite(loss).all()
    for i in (11, 14):
        for n, p in m.model[i].named_parameters(): assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, (i, n)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)

@pytest.mark.skipif(not CUDA, reason="needs GPU")
def test_amp_forward_backward():
    m = e3("cuda:0").train(); opt = torch.optim.SGD(m.parameters(), lr=1e-4); sc = torch.amp.GradScaler(init_scale=1024.0)   # the initial 65536 scale legitimately overflows fp16 grads; the scaler then skips and halves
    for _ in range(3):
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16): loss, _ = m.loss(batch("cuda:0"))
        sc.scale(loss.sum()).backward(); sc.unscale_(opt)
        assert torch.isfinite(loss).all() and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        sc.step(opt); sc.update()

@pytest.mark.parametrize("seed", [101, 202, 303])
def test_paired_init_all_seeds_tensor_equal_and_carafe_shared_with_E2(seed):
    if not INIT[seed].exists(): pytest.skip("init state absent")
    sd = torch.load(INIT[seed], map_location="cpu")
    m3 = build_paired_model("E3", seed, INIT[seed], nwd=NWDCFG); m2 = build_paired_model("E2", seed, INIT[seed])
    assert common_equal(m3, sd) == 708 and len(m3.state_dict()) == 716 and len(set(m3.state_dict()) - set(sd)) == 8
    a, b = carafe_state(m3), carafe_state(m2); assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)      # identical to E2's CARAFE init
    assert m3.nwd == NWDCFG and m2.nwd is None

def test_no_missing_original_layers_and_no_upstream_patching():
    ref = P05DetectionModel("yolo26s.yaml", nc=1, verbose=False); m = e3()
    assert set(ref.state_dict()) <= set(m.state_dict())
    import ultralytics.nn.modules.block as B, ultralytics.utils.loss as L
    assert "p05_detect" not in (B.C2PSA.__module__ or "") and L.E2ELoss.__module__.startswith("ultralytics")

def test_checkpoint_roundtrip_inference_equal():
    m = e3().eval(); x = torch.rand(1, 3, 640, 640)
    with torch.no_grad(): a = m(x)
    buf = io.BytesIO(); torch.save(m.state_dict(), buf); buf.seek(0)
    m2 = e3().eval(); m2.load_state_dict(torch.load(buf))
    with torch.no_grad(): b = m2(x)
    fa = a[0] if isinstance(a, (list, tuple)) else a; fb = b[0] if isinstance(b, (list, tuple)) else b
    if isinstance(fa, dict): fa, fb = fa["one2many"]["boxes"], fb["one2many"]["boxes"]
    assert torch.equal(fa, fb)

def test_params_equal_e2_plus_zero():
    assert sum(p.numel() for p in e3().parameters()) == sum(p.numel() for p in P05DetectionModel(Y, nc=1, verbose=False).parameters())
