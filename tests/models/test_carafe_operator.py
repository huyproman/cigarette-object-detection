"""CARAFE operator tests + independent naive-reference equivalence of the production operator."""
import io, pytest, torch, torch.nn.functional as F
from p05_detect.modules.carafe import CARAFE, carafe_reference, init_carafe_

DEV = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def rnd(*s, dev="cpu", seed=0):
    return torch.randn(*s, generator=torch.Generator().manual_seed(seed)).to(dev)


@pytest.mark.parametrize("c", [128, 256, 512])
def test_channels(c):
    m = CARAFE(c); y = m(rnd(2, c, 6, 6))
    assert y.shape == (2, c, 12, 12)


def test_odd_spatial():
    assert CARAFE(32)(rnd(1, 32, 15, 17)).shape == (1, 32, 30, 34)


def test_kernel_sums_to_one():
    m = CARAFE(32); k = m.kernels(rnd(2, 32, 7, 9))
    assert k.shape == (2, 25, 14, 18) and k.dtype == torch.float32
    assert torch.allclose(k.sum(1), torch.ones(2, 14, 18), atol=1e-6) and (k >= 0).all()


@pytest.mark.parametrize("hw", [(5, 5), (8, 6)])
def test_delta_kernel_equals_nearest(hw):
    m = CARAFE(16); x = rnd(2, 16, *hw)
    with torch.no_grad():
        m.enc.weight.zero_(); m.enc.bias.zero_()
        for q in range(4): m.enc.bias[12 * 4 + q] = 60.0              # centre tap at all 4 sub-pixels, saturated softmax
        y = m(x)
    assert torch.allclose(y, F.interpolate(x, scale_factor=2, mode="nearest"), atol=1e-6)


@pytest.mark.parametrize("device", DEV)
def test_equivalence_to_naive_reference_fp32(device):
    m = CARAFE(24).to(device); x = rnd(2, 24, 7, 6, dev=device, seed=3)
    with torch.no_grad():
        m.enc.weight.normal_(0, .2)
        assert torch.allclose(m(x), carafe_reference(x, m.kernels(x)), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("device", DEV)
def test_gradients_match_naive_reference(device):
    m = CARAFE(8).to(device); x0 = rnd(1, 8, 5, 5, dev=device, seed=4)
    with torch.no_grad(): m.enc.weight.normal_(0, .2)
    xa = x0.clone().requires_grad_(); xb = x0.clone().requires_grad_()
    m(xa).pow(2).sum().backward()
    carafe_reference(xb, m.kernels(xb)).pow(2).sum().backward()
    assert torch.allclose(xa.grad, xb.grad, atol=1e-4, rtol=1e-3)


def test_chunked_equals_unchunked_forward_and_backward():
    m = CARAFE(16); x = rnd(1, 16, 10, 10, seed=5)
    with torch.no_grad(): m.enc.weight.normal_(0, .3)
    a = x.clone().requires_grad_(); b = x.clone().requires_grad_()
    m.chunk_rows = 0; ya = m(a); ya.pow(2).sum().backward()
    m.chunk_rows = 3; yb = m(b); yb.pow(2).sum().backward()
    assert torch.allclose(ya, yb, atol=1e-6) and torch.allclose(a.grad, b.grad, atol=1e-5)


def test_half_precision_agrees_with_fp32():
    x = (torch.rand(1, 32, 10, 10, generator=torch.Generator().manual_seed(1)) * 2 - 1)
    m = CARAFE(32); ref = m(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = m(x.bfloat16())
    assert torch.allclose(y.float(), ref, atol=.02, rtol=.02)
    assert m.kernels(x.bfloat16()).dtype == torch.float32               # softmax kernel stays float32
    if torch.cuda.is_available():
        mc = CARAFE(32).cuda(); mc.load_state_dict(m.state_dict())
        with torch.autocast("cuda", dtype=torch.float16):
            yh = mc(x.cuda().half())
        assert yh.dtype == torch.float16 and torch.allclose(yh.float().cpu(), ref, atol=.02, rtol=.02)


def test_invalid_inputs_rejected():
    m = CARAFE(8)
    for bad in (rnd(8, 4, 4), rnd(1, 7, 4, 4), torch.zeros(1, 8, 0, 4)):
        with pytest.raises(ValueError): m(bad)
    nan = torch.full((1, 8, 4, 4), float("nan"))
    m(nan)                                           # default: no per-call sync, garbage-in/garbage-out (Ultralytics warmup uses torch.empty)
    m.check_finite = True
    with pytest.raises(ValueError): m(nan)
    for kw in (dict(k_up=4), dict(k_enc=2), dict(scale=3), dict(c1=0)):
        with pytest.raises(ValueError): CARAFE(**{"c1": 8, **kw})


def test_initialisation_policy():
    m = CARAFE(16); g = torch.Generator().manual_seed(7); init_carafe_(m, g)
    assert m.comp.bias.abs().max() == 0
    assert m.enc.weight.std().item() == pytest.approx(.001, rel=.1) and abs(m.enc.weight.mean().item()) < 1e-4
    cen = torch.tensor([12 * 4 + q for q in range(4)])
    assert (m.enc.bias[cen] == 8).all() and m.enc.bias.sum().item() == pytest.approx(32.0)
    bound = (6 / ((1 + 5) * 16)) ** .5                                  # kaiming_uniform(a=sqrt5): bound = sqrt(6/((1+a^2) fan_in))
    assert m.comp.weight.abs().max() <= bound + 1e-6
    k = m.kernels(rnd(1, 16, 4, 4)); assert k[:, 12].min() > .98       # near-nearest behaviour at start (centre weight ~.992)


def test_init_is_deterministic_and_independent_of_global_rng():
    a, b = CARAFE(16), CARAFE(16)
    init_carafe_(a, torch.Generator().manual_seed(5)); torch.manual_seed(999); torch.rand(100); init_carafe_(b, torch.Generator().manual_seed(5))
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))


def test_save_load_roundtrip():
    m = CARAFE(16); init_carafe_(m, torch.Generator().manual_seed(1)); buf = io.BytesIO(); torch.save(m.state_dict(), buf); buf.seek(0)
    m2 = CARAFE(16); m2.load_state_dict(torch.load(buf)); x = rnd(1, 16, 5, 5)
    assert torch.equal(m(x), m2(x)) and sorted(m.state_dict()) == ["comp.bias", "comp.weight", "enc.bias", "enc.weight"]


def test_deterministic_cpu():
    m = CARAFE(16); x = rnd(1, 16, 6, 6)
    assert torch.equal(m(x), m(x))
