"""NWD C scaling 61*(R/640) is scale-equivariant between the 640 and 960 frames."""
import torch
from p05_detect.losses.nwd import nwd_similarity
def test_c_scales_with_resolution():
    g = torch.Generator().manual_seed(0)
    xy = torch.rand(200, 2, generator=g) * 500 + 20; wh = torch.rand(200, 2, generator=g) * 60 + 6
    b1 = torch.cat([xy, xy + wh], 1); d = torch.randn(200, 4, generator=g) * 2; b2 = b1 + d
    a = nwd_similarity(b1, b2, 61.0); b = nwd_similarity(b1 * 1.5, b2 * 1.5, 91.5)
    assert torch.allclose(a, b, atol=1e-5)
    assert not torch.allclose(a, nwd_similarity(b1 * 1.5, b2 * 1.5, 61.0), atol=1e-3)   # unscaled C would differ
