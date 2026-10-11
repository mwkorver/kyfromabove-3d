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


def _deck_scene(deck: bool):
    """A textured wall 60 m long and 20 m high, and (if ``deck``) a textured
    deck at 4.7 m in front of it, 15 m deep; three frames looking at it from
    70 m out and 35 m up. Returns the face's plane, grid, views and a frame
    stand-in with the identity transform."""
    import types

    import render_flyaround as rf

    from kyfromabove_3d.features import wall_planes
    pl = wall_planes(np.array([[0.0, 0.0], [60.0, 0.0], [60.0, 30.0], [0.0, 30.0]]), z_lo=0.0, z_hi=20.0)[0]
    d = (pl.p1 - pl.p0) / np.linalg.norm(pl.p1 - pl.p0)
    n = np.asarray(pl.normal, float)

    def patch(seed, along, out, z):
        t = rf.ground_scene(np.array([0.0, 0.0]), 30.0, 0.0, seed=seed)
        x, y = t.tris[..., 0] + 30.0, t.tris[..., 1] + 30.0              # 0..60 each way
        a_, o_, z_ = along(x, y), out(x, y), z(x, y)
        t.tris = np.concatenate([pl.p0[None, None, :] + a_[..., None] * d + o_[..., None] * n, z_[..., None]], axis=-1)
        return t
    scene = patch(4, lambda x, y: x, lambda x, y: 0 * x, lambda x, y: y / 3.0)            # the wall, 0..20 m up
    if deck:
        scene = scene + patch(9, lambda x, y: x, lambda x, y: y / 4.0, lambda x, y: 0 * x + 4.7)   # 0..15 m out
    W, H = 900, 600
    K = rf.intrinsics(W, H, 55.0)
    centre = np.r_[pl.p0 + 30.0 * d, 8.0]
    views = {}
    for k, side in enumerate((-22.0, 0.0, 22.0)):
        C = centre + np.r_[70.0 * n + side * d, 27.0]
        R = rf.look_rotation(centre - C)
        img, _ = rf.render(scene, K, R, -R @ C, W, H)
        views[f"cam{k}"] = dict(K=K, R_cw=R, C=C, gray=img.mean(axis=2).astype(np.float32), x0=0, y0=0)
    A, Z = np.meshgrid(np.arange(4.0, 56.0, fm.GRID_M), np.arange(0.0, 20.0, fm.GRID_M))
    for w in views.values():
        w["mask"] = np.ones(A.shape, bool)
    return types.SimpleNamespace(M=np.eye(4)), pl, A, Z, views


def test_a_deck_in_front_of_the_wall_is_found_level_and_cut_away():
    b, pl, A, Z, views = _deck_scene(deck=True)
    got = fm.level_ends(b, pl, A, Z, 4.0, 56.0, 0.0, 0.0, views, [0.0, 4.7, 9.0, 14.0, 20.0])
    assert got["bottom"] == 4.7 and got["top"] == 20.0, got
    first = got["tests"][0]
    assert first["is_level"] and abs(first["level_h"] - 4.7) <= 0.25, first


def test_a_plain_wall_is_left_whole():
    b, pl, A, Z, views = _deck_scene(deck=False)
    got = fm.level_ends(b, pl, A, Z, 4.0, 56.0, 0.0, 0.0, views, [0.0, 4.7, 9.0, 14.0, 20.0])
    assert got["bottom"] == 0.0 and got["top"] == 20.0, got
