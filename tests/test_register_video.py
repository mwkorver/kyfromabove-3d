"""Registering a real video to the meshes, checked where the answer is known:
a camera solved from points projected through a known one, a pixel taken
back to the surface it shows, a track through frames rendered along a known
path, and the blend of two tracks that disagree by a known amount."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kyfromabove_3d.stages import register_video as rv  # noqa: E402
from kyfromabove_3d.stages import render_flyaround as rf  # noqa: E402

cv2 = pytest.importorskip("cv2")


def a_camera(C=(40.0, -300.0, 25.0), target=(0.0, 0.0, 30.0), hfov=50.0, W=1280, H=720):
    return rv.intrinsics(W, H, hfov), rf.look_rotation(np.subtract(target, C)), np.array(C, float)


def test_pnp_finds_the_camera_and_its_lens():
    K, R, C = a_camera(hfov=47.0)
    rng = np.random.default_rng(3)
    X = rng.uniform([-60, -40, 0], [60, 40, 120], size=(9, 3))               # spread in depth as well as across
    x = rv.project(K, R, C, X)
    K2, R2, C2, e = rv.solve_pnp(X, x, 1280, 720)
    assert rv.hfov(K2, 1280) == pytest.approx(47.0, abs=0.25)
    assert np.linalg.norm(C2 - C) < 2.0
    assert e.max() < 1.0


def test_pnp_with_the_lens_fixed_is_exact():
    K, R, C = a_camera()
    X = np.array([[0, 0, 50], [30, 10, 80], [-25, -5, 60], [10, 20, 5], [-40, 15, 100]], float)
    K2, R2, C2, e = rv.solve_pnp(X, rv.project(K, R, C, X), 1280, 720, hfov_deg=50.0)
    assert np.allclose(C2, C, atol=1e-3) and np.allclose(R2, R, atol=1e-6)


def wall_scene():
    """A textured wall 60 m wide and 40 m high, facing south at y = 0, and textured ground in front of it."""
    g = rf.ground_scene(np.array([0.0, -40.0]), 60.0, 0.0, seed=11)
    wall = rf.ground_scene(np.array([0.0, 0.0]), 30.0, 0.0, seed=5)
    t = wall.tris.copy()
    wall.tris = np.stack([t[..., 0], np.zeros_like(t[..., 1]), t[..., 1] + 30.0], axis=-1)   # stand it up
    return g + wall


def test_backproject_lands_on_the_surface():
    scene = wall_scene()
    K, R, C = a_camera(C=(5.0, -80.0, 12.0), target=(0.0, 0.0, 20.0), W=640, H=360)
    _, depth = rv.render(scene, R, C, K, 640, 360)
    px = np.array([[320.0, 120.0], [200.0, 90.0], [450.0, 150.0]])
    X = rv.backproject(K, R, C, px, depth)
    assert np.all(np.abs(X[:, 1]) < 0.15)                                    # on the wall, y = 0
    assert np.allclose(rv.project(K, R, C, X), px, atol=0.5)


def test_track_follows_a_known_path(tmp_path):
    scene = wall_scene()
    W, H = 640, 360
    K = rv.intrinsics(W, H, 55.0)
    poses = [rf.Pose(rf.look_rotation(np.array([0.0, 0.0, 18.0]) - c), c)
             for c in [np.array([-6.0 + 0.6 * i, -70.0 + 0.8 * i, 10.0 + 0.3 * i]) for i in range(16)]]
    frames = []
    for i, p in enumerate(poses):
        img, _ = rf.render(scene, K, p.R_cw, p.t_cw, W, H)
        frames.append(str(tmp_path / f"{i:03d}.png"))
        cv2.imwrite(frames[-1], img[:, :, ::-1])
    cam = rv.camera_json(W, H, K, poses[0].R_cw, poses[0].C)
    out = rv.track(scene, frames, cam, 0, 1, total=300, keep=150, log=lambda *_: None)
    err = [np.linalg.norm(np.array(q["C"]) - p.C) for q, p in zip(out["poses"], poses)]
    assert max(err) < 0.3, err
    back = rv.track(scene, frames, rv.camera_json(W, H, K, poses[-1].R_cw, poses[-1].C), len(frames) - 1, -1,
                    total=300, keep=150, log=lambda *_: None)
    assert np.linalg.norm(np.array(back["poses"][0]["C"]) - poses[0].C) < 0.3


def test_blend_leans_on_the_nearer_end():
    n = 11
    R = np.eye(3).tolist()
    truth = np.c_[np.linspace(0, 100, n), np.zeros(n), np.linspace(10, 60, n)]
    fw = {"poses": [dict(i=i, R_cw=R, C=(truth[i] + [0, 0, 0.5 * i]).tolist()) for i in range(n)]}           # drifts up
    bw = {"poses": [dict(i=i, R_cw=R, C=(truth[i] - [0, 0, 0.5 * (n - 1 - i)]).tolist()) for i in range(n)]}  # drifts down
    idx, Rwc, C, gap_m, gap_deg = rv.blend(fw, bw)
    assert np.allclose(C[0], fw["poses"][0]["C"]) and np.allclose(C[-1], bw["poses"][-1]["C"])
    assert np.allclose(gap_m, 0.5 * (n - 1)) and np.allclose(gap_deg, 0.0)
    assert np.allclose(C[:, :2], truth[:, :2])


def test_tum_round_trip_and_track_cameras(tmp_path):
    K, R, C = a_camera()
    rv.write_tum(tmp_path / "t.txt", [0.0, 0.5], np.stack([R.T, R.T]), np.stack([C, C + 1]), "test")
    t, Rwc, C2 = rv.read_tum(tmp_path / "t.txt")
    assert np.allclose(Rwc[0], R.T, atol=1e-7) and np.allclose(C2[1], C + 1) and t[1] == 0.5
    track = dict(width=1280, height=720, K=K.tolist(), poses=[dict(i=7, R_cw=R.tolist(), C=C.tolist())])
    (tmp_path / "track.json").write_text(json.dumps(track))
    cam = rv.read_camera(f"{tmp_path / 'track.json'}@7")
    assert np.allclose(cam["C"], C) and cam["hfov_deg"] == pytest.approx(50.0)
