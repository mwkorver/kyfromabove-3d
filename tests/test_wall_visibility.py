"""The wall-visibility measure's rule for what is wall: a point counts only
where the building's roof, just inside it, stands higher -- a podium's edge
runs on up past its roof as air, not as hidden wall."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kyfromabove_3d.stages import wall_visibility as wv  # noqa: E402


def podium_and_tower():
    """Vertices of a 40 x 40 m podium 15 m tall with a 10 x 10 m tower 60 m
    tall on its north-west corner, roofs as 0.75 m height-field points."""
    xs = np.arange(0.0, 40.01, 0.75)
    X, Y = np.meshgrid(xs, xs)
    Z = np.where((X <= 10) & (Y >= 30), 60.0, 15.0)
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def test_wall_below_the_roof_and_air_above_it():
    grid = wv.roof_grid(podium_and_tower())
    south = np.array([0.0, -1.0, 0.0])                                       # the south face, y = 0, faces south
    pts = np.array([[20.0, 0.0, 5.0], [20.0, 0.0, 14.0], [20.0, 0.0, 16.0], [20.0, 0.0, 40.0]])
    assert wv.wall_exists(pts, np.tile(south, (4, 1)), grid).tolist() == [True, True, False, False]


def test_the_tower_wall_stands_above_the_podium():
    grid = wv.roof_grid(podium_and_tower())
    north = np.array([0.0, 1.0, 0.0])                                        # the north face, y = 40
    pts = np.array([[5.0, 40.0, 40.0], [25.0, 40.0, 40.0], [25.0, 40.0, 10.0]])
    assert wv.wall_exists(pts, np.tile(north, (3, 1)), grid).tolist() == [True, False, True]
