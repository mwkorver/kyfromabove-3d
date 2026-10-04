"""The completed product's fills on synthetic facades: a window lattice with
a block nothing saw is refilled from its own rhythm; texture with no
rhythm is not; a hidden face takes its twin's texels mirrored; measured
texels never change."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kyfromabove_3d.facade_fill import (KNOWN, PATCH, PLAIN, RHYTHM, SEEN, TWIN,  # noqa: E402
                                        complete_textures, holdout, patch_fill, period, rhythm_fill,
                                        shares, twin_fill)

TEXEL = 0.1


def lattice(H=120, W=300, bay_px=30, storey_px=40, seed=0):
    """A facade: light wall, a dark window in every bay of every storey,
    with a little noise. 3 m bays, 4 m storeys at 0.1 m texels."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 190.0)
    for y0 in range(8, H, storey_px):
        for x0 in range(6, W, bay_px):
            img[y0:y0 + 22, x0:x0 + 16] = (60.0, 70.0, 85.0)
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_the_facade_rhythm_is_found():
    img = lattice()
    known = np.ones(img.shape[:2], dtype=bool)
    assert period(img, known, TEXEL, 1, 1.0, 8.0) == 30          # 3 m bays
    assert period(img, known, TEXEL, 0, 2.5, 6.0) == 40          # 4 m storeys
    noise = np.random.default_rng(1).integers(0, 255, img.shape).astype(np.uint8)
    assert period(noise, known, TEXEL, 1, 1.0, 8.0) is None


def test_a_hidden_block_is_refilled_from_the_rhythm_and_measured_texels_never_change():
    truth = lattice()
    source = np.full(truth.shape[:2], SEEN, dtype=np.uint8)
    source[40:80, 120:200] = PLAIN                               # a stretch nothing saw
    img = truth.copy(); img[source == PLAIN] = np.median(truth.reshape(-1, 3), axis=0)
    out, s = rhythm_fill(img, source, TEXEL)
    hidden = source == PLAIN
    assert (s[hidden] == RHYTHM).all()
    assert np.abs(out[hidden].astype(float) - truth[hidden].astype(float)).mean() < 8.0
    assert (out[~hidden] == img[~hidden]).all() and (s[~hidden] == SEEN).all()


def test_texture_with_no_rhythm_is_left_plain_by_the_rhythm_fill():
    noise = np.random.default_rng(2).integers(0, 255, (120, 300, 3)).astype(np.uint8)
    source = np.full(noise.shape[:2], SEEN, dtype=np.uint8); source[40:80, 120:200] = PLAIN
    out, s = rhythm_fill(noise, source, TEXEL)
    assert (s == source).all()


def test_patch_fill_reaches_only_a_few_metres_in():
    truth = lattice()
    source = np.full(truth.shape[:2], SEEN, dtype=np.uint8); source[:, 150:] = PLAIN   # 15 m unseen
    out, s = patch_fill(truth, source, TEXEL, max_depth_m=3.0)
    assert (s[:, 150:175] == PATCH).mean() > 0.9                 # the first 2.5 m filled
    assert (s[:, 200:] == PLAIN).all()                            # 5 m in and beyond: still plain


def test_a_hidden_face_takes_its_twin_mirrored_at_the_same_heights():
    twin = lattice()
    twin[:, :40] = (200, 30, 30)                                  # the twin's west end is red
    tsrc = np.full(twin.shape[:2], SEEN, dtype=np.uint8)
    face = np.full_like(twin, 128); fsrc = np.full(twin.shape[:2], PLAIN, dtype=np.uint8)
    meta = {"kind": "profile", "face": 2, "z": (0.0, 12.0)}
    tmeta = {"kind": "profile", "face": 0, "z": (0.0, 12.0)}
    out, s = twin_fill(face, fsrc, meta, twin, tsrc, tmeta)
    assert (s == TWIN).all()
    assert (out[:, -40:] == (200, 30, 30)).all()                  # mirrored: its east end is red


def test_complete_textures_keeps_order_and_shares_add_up():
    class Tex(np.ndarray):
        pass
    a = lattice().view(Tex); a.source = np.full(a.shape[:2], SEEN, dtype=np.uint8); a.source[40:80, 120:200] = PLAIN
    b = lattice(seed=3).view(Tex); b.source = np.full(b.shape[:2], SEEN, dtype=np.uint8)
    out = complete_textures([a, b], [{"kind": "profile", "face": 0, "z": (0, 12)}, {"kind": "tile"}], TEXEL)
    assert len(out) == 2 and (out[1][1] == SEEN).all()
    sh = shares([o[1] for o in out])
    assert abs(sum(v for k, v in sh.items() if k != "n") - 1.0) < 1e-3 and sh["plain"] == 0.0


def test_holdout_scores_the_rhythm_fill_above_plain():
    img = lattice(W=400)
    source = np.full(img.shape[:2], SEEN, dtype=np.uint8)
    res = holdout(img, source, TEXEL, lambda i, s: rhythm_fill(i, s, TEXEL), trials=3)
    assert len(res) == 3
    assert all(r["share_filled"] > 0.9 and r["ncc"] > 0.8 and r["colour_err"] < r["plain_colour_err"] for r in res)
