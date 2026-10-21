"""The neural completed fill (product 4) on a small wall: a regular window
pattern with a rectangle no frame saw. The hole is filled and marked NEURAL,
every seen texel is left as it was, and the fill follows the windows better
than the plain fill it replaces. Skipped without PyTorch (pip install .[neural])."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kyfromabove_3d.facade_fill import NEURAL, PLAIN, SEEN, _highpass, _luma   # noqa: E402
from kyfromabove_3d.mesh.textures import Texture                               # noqa: E402
from kyfromabove_3d.neural_fill import neural_fill, train_building_model         # noqa: E402


def _wall(h=120, w=200, texel_m=0.15):
    """Brick-coloured wall, dark windows 1.2 x 1.5 m every 3 m across and 3.5 m up."""
    img = np.full((h, w, 3), (150, 90, 70), np.uint8)
    for y0 in range(4, h, int(3.5 / texel_m)):
        for x0 in range(5, w, int(3.0 / texel_m)):
            img[y0:y0 + int(1.5 / texel_m), x0:x0 + int(1.2 / texel_m)] = (40, 45, 55)
    return img


def test_a_hidden_rectangle_is_filled_by_the_windows_and_seen_texels_are_kept():
    t = 0.15
    truth = _wall()
    src = np.full(truth.shape[:2], SEEN, np.uint8)
    box = (slice(40, 70), slice(80, 120))                          # 4.5 x 6 m no frame saw
    img = truth.copy()
    img[box] = np.median(truth.reshape(-1, 3), axis=0)
    src[box] = PLAIN
    cpu = torch.device("cpu")
    torch.manual_seed(0)
    _, refs, model = train_building_model([Texture(img, src)], [{"kind": "tile", "face": 1}], t, steps=8, device=cpu)
    out, s2 = neural_fill(img, src, t, face=1, kind="tile", ref_sources=refs, model=model, device=cpu)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert (s2[box] == NEURAL).all()                                # every hidden texel invented, and marked
    keep = src == SEEN
    assert np.array_equal(out[keep], img[keep]) and (s2[keep] == SEEN).all()   # seen texels untouched
    hp_truth = _highpass(_luma(truth), np.ones(src.shape, bool), 2.0)[box].ravel()
    hp_out = _highpass(_luma(out), np.ones(src.shape, bool), 2.0)[box].ravel()
    agree = float(np.corrcoef(hp_truth, hp_out)[0, 1])
    assert agree > 0.3, agree                                       # the plain fill has no pattern: 0
