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


def _scores(depths, clear=0.3):
    """Agreement for each trial depth (rows) and piece (columns): a peak at each piece's depth."""
    return np.array([[clear * np.exp(-((s - d) / 0.3) ** 2) for d in depths] for s in fm.SHIFTS])


def test_runs_put_the_step_on_the_line_the_frames_prefer():
    edges = [4.8, 5.9, 7.2, 8.2, 10.8, 16.0, 22.6, 24.5]                     # Hampton 1006's lines
    S = _scores([-1.5, -1.5, -0.25, -0.25, -0.25, -0.25, -0.25])             # the 7.2-8.2 piece belongs above
    runs = fm.runs_from_scores(S, np.ones(7), edges)
    assert [r["z_lo"] for r in runs] == [4.8, 7.2] and [r["shift"] for r in runs] == [-1.5, -0.25]
    assert fm.steps_of(runs) == [dict(z=7.2, below=-1.5, above=-0.25)]


def test_one_unclear_piece_makes_no_step():
    edges = [4.8, 7.5, 10.1, 12.7, 15.4, 17.9, 20.8, 22.1, 23.5]
    S = _scores([0.5] * 8)
    S[:, 6] = 0.05 + 0.02 * np.cos(fm.SHIFTS)                               # a piece with two surfaces: no clear depth
    runs = fm.runs_from_scores(S, np.ones(8), edges)
    assert len(runs) == 1 and runs[0]["shift"] == 0.5


def test_a_run_shorter_than_two_metres_is_not_split_off():
    edges = [0.0, 1.0, 10.0]
    runs = fm.runs_from_scores(_scores([-2.0, 0.0]), np.array([1.0, 3.0]), edges)
    assert len(runs) == 1


def test_trim_keeps_the_rows_and_columns_that_are_mostly_wall():
    wall = np.ones((20, 10), bool)
    wall[:4] = False                                                     # buried at the bottom
    wall[18:] = False                                                    # roof at the top
    wall[:, 0] = False                                                   # a column with no wall
    rs, cs = fm.trim_extent(wall)
    assert (rs.start, rs.stop) == (4, 18) and (cs.start, cs.stop) == (1, 10)
    assert fm.trim_extent(np.zeros((5, 5), bool)) is None
