"""The atlas's pattern reading: the horizontal lines a laid-flat facade shows,
and the heights where the frames say the wall steps."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import face_match as fm  # noqa: E402


def test_image_lines_finds_panel_joints_and_their_spacing():
    rows = np.full(190, 0.1)
    rows[[3, 29, 55, 81, 107, 133, 159]] = 5.0                              # a joint every 2.6 m
    got = fm.image_lines(rows, 0.1)
    assert got["lines"] == [0.3, 2.9, 5.5, 8.1, 10.7, 13.3, 15.9]
    assert got["spacing_m"] == 2.6 and got["regular"] == 1.0


def test_image_lines_gives_no_spacing_when_lines_are_uneven():
    rows = np.full(190, 0.1)
    rows[[10, 30, 100, 110, 180]] = 5.0
    got = fm.image_lines(rows, 0.1)
    assert len(got["lines"]) == 5 and got["spacing_m"] is None


def test_find_steps_marks_one_step_between_two_depths():
    curve = [[z + 0.5, -1.5 if z < 8 else -0.25, 0.4, 0.1] for z in range(5, 20)]
    steps = fm.find_steps(curve)
    assert len(steps) == 1 and abs(steps[0]["z"] - 8.0) <= 1.0
    assert steps[0]["below"] == -1.5 and steps[0]["above"] == -0.25
