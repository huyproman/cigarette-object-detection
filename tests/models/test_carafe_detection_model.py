"""Parser, shapes, paired init, dual-head gradients, checkpoint."""
import io, pytest, torch, yaml
from pathlib import Path
from ultralytics.cfg import get_cfg
from p05_detect.engine.model import P05DetectionModel, CONFIG_DIR
from p05_detect.modules.carafe import CARAFE
from p05_detect.experiments.paired_init import build_paired_model, common_equal, carafe_state

Y = str(CONFIG_DIR / "yolo26s-carafe.yaml")
INIT = Path(__file__).parents[2] / "runtime/init_states/B0_seed101_init_state.pt"
DEV = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def mk(device="cpu", nwd=None, cfg=Y):
    m = P05DetectionModel(cfg, nc=1, verbose=False, nwd=nwd); m.args = get_cfg(); m.nc = 1
    return m.to(device).train()


def test_parser_builds_with_explicit_c1_and_shapes():
    m = mk(); assert isinstance(m.model[11], CARAFE) and isinstance(m.model[14], CARAFE)
    assert (m.model[11].c1, m.model[14].c1) == (512, 256)
    shapes = {}
    hooks = [m.model[i].register_forward_hook(lambda mod, i_, o, i=i: shapes.__setitem__(i, tuple(o.shape))) for i in (11, 14, 12, 15)]
    m.eval()
    with torch.no_grad(): m(torch.zeros(1, 3, 640, 640))
    assert shapes[11] == (1, 512, 40, 40) and shapes[14] == (1, 256, 80, 80)
    assert shapes[12] == (1, 768, 40, 40) and shapes[15] == (1, 512, 80, 80)       # B8: lateral concat unchanged
    assert list(m.stride.int()) == [8, 16, 32]


def test_wrong_c1_is_rejected_at_forward():
    cfg = yaml.safe_load(Path(Y).read_text()); cfg["head"][0][3] = [256]
    with pytest.raises(Exception):
        P05DetectionModel(cfg, nc=1, verbose=False)


def test_missing_c1_fails_loudly():
    cfg = yaml.safe_load(Path(Y).read_text()); cfg["head"][0][3] = []
    with pytest.raises(TypeError): P05DetectionModel(cfg, nc=1, verbose=False)


def test_param_delta_matches_r1():
    b = sum(p.numel() for p in P05DetectionModel("yolo26s.yaml", nc=1, verbose=False).parameters())
    c = sum(p.numel() for p in mk().parameters())
    assert c - b == 164_680                       # 8 CARAFE tensors (comp 512x64+64, enc 64x100x9+100, ...)


@pytest.mark.skipif(not INIT.exists(), reason="stored init states missing")
def test_common_state_exactly_equal_and_carafe_tensors_paired():
    sd = torch.load(INIT, map_location="cpu")
    m1 = build_paired_model("E2", 101, INIT); m2 = build_paired_model("E3", 101, INIT, nwd=dict(lam=.5, C=61.0))
    assert len(sd) == 708 and common_equal(m1, sd) == 708 and common_equal(m2, sd) == 708
    a, b = carafe_state(m1), carafe_state(m2)
    assert sorted(a) == sorted(b) and len(a) == 8 and all(torch.equal(a[k], b[k]) for k in a)      # E2/E3 share CARAFE tensors
    other = carafe_state(build_paired_model("E2", 202, Path(str(INIT).replace("101", "202"))))
    assert not all(torch.equal(a[k], other[k]) for k in a)
    e1 = build_paired_model("E1", 101, INIT, nwd=dict(lam=.5, C=61.0))
    assert common_equal(e1, sd) == 708 and len(e1.state_dict()) == 708


@pytest.mark.skipif(not INIT.exists(), reason="stored init states missing")
def test_pairing_does_not_depend_on_global_rng():
    torch.manual_seed(1); a = carafe_state(build_paired_model("E2", 101, INIT))
    torch.manual_seed(2); torch.rand(1000); b = carafe_state(build_paired_model("E2", 101, INIT))
    assert all(torch.equal(a[k], b[k]) for k in a)


def bt(device, n=4):
    g = torch.Generator().manual_seed(0); k = n * 2; xy = torch.rand(k, 2, generator=g) * .7 + .15
    return dict(img=torch.rand(n, 3, 320, 320, generator=g).to(device), batch_idx=torch.arange(n).repeat_interleave(2).float().to(device),
                cls=torch.zeros(k, 1).to(device), bboxes=torch.cat([xy, torch.full((k, 2), .05)], 1).to(device))


@pytest.mark.parametrize("device", DEV)
@pytest.mark.parametrize("nwd", [None, dict(lam=.5, C=61.0)])
def test_full_dual_head_forward_backward(device, nwd):
    m = mk(device, nwd); loss, _ = m.loss(bt(device)); loss.sum().backward()
    assert torch.isfinite(loss).all()
    for i in (11, 14):
        for p in m.model[i].parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    bad = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not bad, bad[:5]
    # both heads feed the loss: one2one head params receive gradients too
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if "one2one" in n)


def test_empty_targets():
    m = mk(); b = bt("cpu"); b = dict(b, batch_idx=b["batch_idx"][:0], cls=b["cls"][:0], bboxes=b["bboxes"][:0])
    loss, _ = m.loss(b); loss.sum().backward(); assert torch.isfinite(loss).all()


def test_checkpoint_roundtrip_new_process_importable(tmp_path):
    import subprocess, sys
    m = mk(); m.eval(); p = tmp_path / "m.pt"
    torch.save({"model": m}, p)
    code = ("import torch, p05_detect.engine.model\n"
            f"d = torch.load(r'{p}', weights_only=False)['model'].eval()\n"
            "y = d(torch.zeros(1,3,320,320)); y = y[0] if isinstance(y,(list,tuple)) else y\n"
            "assert torch.isfinite(y).all(); print('OK')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path(__file__).parents[2], env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).parents[2])})
    assert "OK" in r.stdout, r.stderr[-500:]
    assert type(m.model[11]).__module__ == "p05_detect.modules.carafe"


def test_p2_comparator_parameter_and_stride_counts():
    """yolo26s-p2 = 9,663,464 params, strides [4,8,16,32], 360/902 tensors transferable. Scale comes from the FILENAME."""
    pre = Path(__file__).parents[2] / "runtime/weights/yolo26s.pt"
    m = P05DetectionModel(str(CONFIG_DIR / "yolo26s-p2.yaml"), nc=1, verbose=False)
    assert sum(p.numel() for p in m.parameters()) == 9_663_464 and list(m.stride.int()) == [4, 8, 16, 32]
    if pre.exists():
        src = torch.load(pre, map_location="cpu", weights_only=False)["model"].float().state_dict(); own = m.state_dict()
        assert len(own) == 902 and sum(1 for k, v in src.items() if k in own and own[k].shape == v.shape) == 360
