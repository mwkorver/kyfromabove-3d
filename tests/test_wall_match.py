"""The wall matcher's two judgements, on errors made up on purpose: the
confidence cut drops matches that failed (off by metres) and keeps the rest,
and taking out a building's lean uses only that building's *other* bands."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import kyfromabove_3d.wall_match as wm  # noqa: E402


def band(depth, margin, pattern=20.0, thirds=(0.0, 0.1, -0.1)):
    return dict(depth_m=depth, margin=margin, pattern=pattern, thirds_m=[depth + t for t in thirds])


def test_the_cut_drops_failed_matches_and_keeps_good_ones():
    rng = np.random.default_rng(0)
    good = [band(-1.0, float(m)) for m in rng.uniform(0.1, 0.5, 80)]
    bad = [band(-1.0, float(m)) for m in rng.uniform(0.0, 0.03, 20)]
    err = np.r_[rng.normal(0, 0.2, 80), rng.choice([-4.0, 3.0], 20)]
    cut = wm.choose_cut(good + bad, err, max_gross=0.03)
    acc = np.array([wm.is_accepted(r, cut) for r in good + bad])
    assert acc[:80].mean() > 0.9 and acc[80:].mean() < 0.1


def test_the_fixed_rules_reject_the_search_limit_and_disagreeing_thirds():
    assert wm.passes_rules(band(-1.0, 0.2))
    assert not wm.passes_rules(band(-7.8, 0.2))                              # at the deep end of the search
    assert not wm.passes_rules(band(2.8, 0.2))                               # at the outer end
    assert not wm.passes_rules(band(-1.0, 0.2, thirds=(0.0, 1.5, -0.2)))     # one third two metres off
    assert not wm.passes_rules(dict(depth_m=-1.0, margin=0.2, pattern=20.0, thirds_m=[-1.0, None, None]))


def test_the_lean_comes_from_the_other_bands_of_the_building():
    err = np.array([-0.5, -0.5, -0.5, -0.5, 0.2, 0.2, 0.2])
    bld = np.array(["a", "a", "a", "a", "b", "b", "b"])
    out = wm.lean_removed(err, bld, min_others=3)
    assert np.allclose(out[:4], 0.0)
    assert np.all(np.isnan(out[4:]))                                         # only two others each in "b"
