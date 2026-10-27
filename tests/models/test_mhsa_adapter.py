"""MHSA adapter tests. References are independent of the production path (explicit softmax, hand-built loops, analytic PE)."""
import copy, io, math, pytest, torch
from pathlib import Path
from ultralytics.cfg import get_cfg
from p05_detect.engine.model import P05DetectionModel, CONFIG_DIR
from p05_detect.modules.mhsa import MHSA, C2PSAMHSA, sincos_2d, init_mhsa_, mhsa_generator
from p05_detect.experiments.paired_init import build_paired_model, common_equal
from p05_detect.experiments.b0_trainer import sd_hash

Y = str(CONFIG_DIR / "yolo26s-mhsa.yaml")
ROOT = Path(__file__).parents[2]
INIT = ROOT / "runtime/init_states/B0_seed101_init_state.pt"
CUDA = torch.cuda.is_available()
DEV = ["cpu"] + (["cuda:0"] if CUDA else [])

def mk(device="cpu"):
    m = P05DetectionModel(Y, nc=1, verbose=False); m.args = get_cfg(); m.nc = 1
    return m.to(device)

def rand_mhsa(c=64, heads=4, dim=32, gamma=0.5, seed=0):
    m = MHSA(c, heads, dim); init_mhsa_(m, torch.Generator().manual_seed(seed)); m.gamma.data.fill_(gamma); return m

@pytest.mark.parametrize("c,h,w,heads,dim", [(64, 5, 7, 4, 32), (512, 20, 20, 8, 256), (32, 1, 1, 2, 16), (128, 8, 3, 8, 64)])
def test_output_shape_and_dtype(c, h, w, heads, dim):
    m = MHSA(c, heads, dim); x = torch.randn(2, c, h, w); y = m(x)
    assert y.shape == x.shape and y.dtype == x.dtype

def test_token_head_dimensions():
    m = MHSA(512, 8, 256); x = torch.randn(3, 512, 20, 20); q, k, v = m._qkv(x)
    assert q.shape == k.shape == v.shape == (3, 8, 400, 32) and m.hd == 32 and m.dim == 8 * m.hd
    with pytest.raises(ValueError): MHSA(64, 5, 32)                     # dim not divisible by heads
    with pytest.raises(ValueError): m(torch.randn(1, 64, 4, 4))          # wrong channels

def test_qkv_and_scaling_against_handbuilt_loop():
    c, h, w, heads, dim = 16, 3, 2, 2, 8
    m = rand_mhsa(c, heads, dim, gamma=1.0).double(); x = torch.randn(1, c, h, w, dtype=torch.double)
    toks = x.flatten(2).transpose(1, 2)[0]                                # (N,C)
    n = torch.nn.functional.layer_norm(toks, (c,), m.norm.weight, m.norm.bias, 1e-5) + sincos_2d(h, w, c, dtype=torch.double)
    qkv = n @ m.qkv.weight.T + m.qkv.bias; hd = dim // heads; outs = []
    for hh in range(heads):                                               # slow explicit per-head, per-query loop
        q = qkv[:, 0 * dim + hh * hd:0 * dim + (hh + 1) * hd]; k = qkv[:, dim + hh * hd:dim + (hh + 1) * hd]; v = qkv[:, 2 * dim + hh * hd:2 * dim + (hh + 1) * hd]
        o = torch.stack([sum(torch.softmax(torch.stack([(q[i] * k[j]).sum() / math.sqrt(hd) for j in range(len(q))]), 0)[j] * v[j] for j in range(len(q))) for i in range(len(q))])
        outs.append(o)
    att = torch.cat(outs, 1) @ m.proj.weight.T + m.proj.bias
    want = x + m.gamma * att.T.reshape(1, c, h, w)
    assert torch.allclose(m(x), want, atol=1e-9)

def test_attention_rows_normalised_and_finite():
    m = rand_mhsa(); x = torch.randn(2, 64, 6, 6); q, k, _ = m._qkv(x)
    a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(m.hd), -1)
    assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5) and (a >= 0).all() and torch.isfinite(m(x)).all()
    assert torch.isfinite(m(x * 1e3)).all()

@pytest.mark.parametrize("dev", DEV)
def test_reference_parity(dev):
    m = rand_mhsa().to(dev).eval(); x = torch.randn(2, 64, 9, 9, device=dev)
    assert torch.allclose(m(x), m.reference(x), atol=1e-5, rtol=1e-4)
    if dev == "cpu":
        md = copy.deepcopy(m).double(); xd = x.double(); assert torch.allclose(md(xd), md.reference(xd), atol=1e-10)

def test_identity_friendly_init():
    m = MHSA(512, 8, 256); x = torch.randn(2, 512, 20, 20)
    assert float(m.gamma) == pytest.approx(0.01) and (m.proj.bias == 0).all() and (m.qkv.bias == 0).all()
    rel = (m(x) - x).norm() / x.norm(); assert rel < 0.05                  # near-identity at step zero
    m0 = copy.deepcopy(m); m0.gamma.data.zero_(); assert torch.equal(m0(x), x)   # exact identity when gamma = 0

def test_init_is_seed_deterministic_and_seed_specific():
    a, b, c = (MHSA(512, 8, 256) for _ in range(3))
    init_mhsa_(a, mhsa_generator(101)); init_mhsa_(b, mhsa_generator(101)); init_mhsa_(c, mhsa_generator(202))
    assert torch.equal(a.qkv.weight, b.qkv.weight) and not torch.equal(a.qkv.weight, c.qkv.weight)

def test_gradient_flow_to_all_trainable_params():
    m = rand_mhsa(); x = torch.randn(2, 64, 5, 5, requires_grad=True); m(x).pow(2).sum().backward()
    for n, p in m.named_parameters(): assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, n
    assert x.grad.abs().sum() > 0

def test_gradient_matches_reference_path():
    m = rand_mhsa().double(); x = torch.randn(2, 64, 5, 5, dtype=torch.double)
    g1 = torch.autograd.grad(m(x.clone().requires_grad_()).sum(), list(m.parameters()), allow_unused=False)
    g2 = torch.autograd.grad(m.reference(x.clone().requires_grad_()).sum(), list(m.parameters()), allow_unused=False)
    assert all(torch.allclose(a, b, atol=1e-8) for a, b in zip(g1, g2))

@pytest.mark.skipif(not CUDA, reason="needs GPU")
def test_fp32_and_amp_stability():
    m = rand_mhsa(512, 8, 256).cuda(); x = torch.randn(4, 512, 20, 20, device="cuda")
    y32 = m(x)
    with torch.autocast("cuda", dtype=torch.float16): y16 = m(x)
    assert torch.isfinite(y16).all() and (y16.float() - y32).abs().max() < 0.05
    with torch.autocast("cuda", dtype=torch.float16): l = m(x).float().pow(2).mean()
    l.backward(); assert all(torch.isfinite(p.grad).all() for p in m.parameters())

@pytest.mark.parametrize("dev", DEV)
def test_full_model_forward_backward(dev):
    m = mk(dev).train(); x = torch.randn(2, 3, 640, 640, device=dev)
    out = m(x); assert isinstance(out, dict) or isinstance(out, (list, tuple))
    batch = dict(img=x, batch_idx=torch.tensor([0., 1.], device=dev), cls=torch.zeros(2, 1, device=dev), bboxes=torch.tensor([[.5, .5, .2, .2], [.4, .4, .1, .1]], device=dev))
    loss, items = m.loss(batch); loss.sum().backward()
    g = [p.grad for n, p in m.named_parameters() if n.startswith("model.10.mhsa.")]
    assert torch.isfinite(loss).all() and len(g) == 7 and all(t is not None and torch.isfinite(t).all() for t in g)

def test_c2psa_preserved_and_composed():
    m = mk(); lyr = m.model[10]; assert isinstance(lyr, C2PSAMHSA)
    ref = P05DetectionModel("yolo26s.yaml", nc=1, verbose=False).model[10]
    assert {k for k in lyr.state_dict() if not k.startswith("mhsa.")} == set(ref.state_dict())
    assert [p.shape for p in lyr.cv1.parameters()] == [p.shape for p in ref.cv1.parameters()] and len(lyr.m) == len(ref.m)
    lyr.load_state_dict({k: v for k, v in ref.state_dict().items()}, strict=False); lyr.eval(); ref.eval(); x = torch.randn(1, 512, 20, 20)
    with torch.no_grad():
        base = ref(x); assert torch.equal(C2PSA_only(lyr, x), base) and torch.allclose(lyr(x), lyr.mhsa(base))

def C2PSA_only(lyr, x):
    from ultralytics.nn.modules.block import C2PSA
    return C2PSA.forward(lyr, x)

@pytest.mark.skipif(not INIT.exists(), reason="init state absent")
def test_pretrained_common_tensors_preserved_and_only_mhsa_new():
    sd = torch.load(INIT, map_location="cpu"); m = build_paired_model("E4", 101, INIT)
    assert common_equal(m, sd) == 708 and len(m.state_dict()) == 715
    new = [k for k in m.state_dict() if k not in sd]; assert len(new) == 7 and all(k.startswith("model.10.mhsa.") for k in new)
    assert sum(p.numel() for k, p in m.named_parameters() if k.startswith("model.10.mhsa.")) == 526593
    m2 = build_paired_model("E4", 101, INIT)
    assert sd_hash({k: v for k, v in m.state_dict().items() if k in new}) == sd_hash({k: v for k, v in m2.state_dict().items() if k in new})

def test_checkpoint_roundtrip_inference_equivalence():
    m = mk().eval(); x = torch.randn(1, 3, 640, 640)
    with torch.no_grad(): a = m(x)
    buf = io.BytesIO(); torch.save(m.state_dict(), buf); buf.seek(0)
    m2 = mk().eval(); m2.load_state_dict(torch.load(buf)); 
    with torch.no_grad(): b = m2(x)
    fa = a[0] if isinstance(a, (list, tuple)) else a; fb = b[0] if isinstance(b, (list, tuple)) else b
    if isinstance(fa, dict): fa, fb = fa["one2many"]["boxes"], fb["one2many"]["boxes"]
    assert torch.equal(fa, fb)

def test_topology_and_head_unchanged():
    a = P05DetectionModel("yolo26s.yaml", nc=1, verbose=False); b = mk()
    assert [(l.f, l.i, l.type.split(".")[-1]) for l in a.model if l.i != 10] == [(l.f, l.i, l.type.split(".")[-1]) for l in b.model if l.i != 10]
    assert list(a.save) == list(b.save) and torch.equal(a.stride, b.stride)
    assert [tuple(p.shape) for k, p in a.model[-1].named_parameters()] == [tuple(p.shape) for k, p in b.model[-1].named_parameters()]
    assert sum(p.numel() for p in b.parameters()) / sum(p.numel() for p in a.parameters()) <= 1.10

@pytest.mark.skipif(not CUDA, reason="needs GPU")
def test_gpu_memory_and_latency_smoke():
    m = mk("cuda:0").train(); x = torch.randn(16, 3, 640, 640, device="cuda:0"); torch.cuda.reset_peak_memory_stats()
    batch = dict(img=x, batch_idx=torch.arange(16., device="cuda:0"), cls=torch.zeros(16, 1, device="cuda:0"), bboxes=torch.full((16, 4), .2, device="cuda:0"))
    with torch.autocast("cuda", dtype=torch.float16): loss, _ = m.loss(batch)
    loss.sum().backward(); torch.cuda.synchronize(); peak = torch.cuda.max_memory_allocated() // 2**20
    assert peak < 8000 and torch.isfinite(loss).all()

def test_determinism_same_seed_same_output():
    outs = []
    for _ in range(2):
        m = MHSA(64, 4, 32); init_mhsa_(m, mhsa_generator(7)); torch.manual_seed(0); outs.append(m(torch.randn(1, 64, 4, 4)))
    assert torch.equal(outs[0], outs[1])
    p1, p2 = sincos_2d(20, 20, 512), sincos_2d(20, 20, 512); assert torch.equal(p1, p2) and p1.shape == (400, 512)

def test_sincos_pe_analytic_values():
    pe = sincos_2d(4, 5, 8); q = 2                                          # layout [sin x | cos x | sin y | cos y], freq_k = 1/10000^(k/q)
    for (y, x) in [(0, 0), (2, 3), (3, 4)]:
        row = pe[y * 5 + x]
        for k in range(q):
            f = 1.0 / 10000 ** (k / q)
            assert row[k] == pytest.approx(math.sin(x * f), abs=1e-6) and row[q + k] == pytest.approx(math.cos(x * f), abs=1e-6)
            assert row[2 * q + k] == pytest.approx(math.sin(y * f), abs=1e-6) and row[3 * q + k] == pytest.approx(math.cos(y * f), abs=1e-6)

@pytest.mark.skipif(not CUDA, reason="needs GPU")
def test_loss_finite_over_initial_real_steps():
    import json
    from PIL import Image
    import numpy as np
    man = [json.loads(l) for l in open(ROOT / "runtime/dataset/split_manifest.jsonl")]; tr = [m for m in man if m["split"] == "TRAIN"][:16]
    imgs = []
    for m in tr:
        im = Image.open(m["path"]).convert("RGB").resize((640, 640)); imgs.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255)
    x = torch.stack(imgs).cuda(); bi, cls, bb = [], [], []
    for i, m in enumerate(tr):
        for l in Path(m["path"].replace("/images/", "/labels/")).with_suffix(".txt").read_text().splitlines():
            _, cx, cy, w, h = map(float, l.split()); bi.append(i); cls.append(0.); bb.append([cx, cy, w, h])
    batch = dict(img=x, batch_idx=torch.tensor(bi, device="cuda"), cls=torch.tensor(cls, device="cuda").view(-1, 1), bboxes=torch.tensor(bb, device="cuda"))
    m = mk("cuda:0").train(); opt = torch.optim.SGD(m.parameters(), lr=1e-3, momentum=0.9); sc = torch.amp.GradScaler()
    for _ in range(5):
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16): loss, _ = m.loss(batch)
        assert torch.isfinite(loss).all(); sc.scale(loss.sum()).backward(); sc.step(opt); sc.update()
