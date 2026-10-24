"""The fly-around renderer against geometry worked out by hand: a textured
wall written the way build_mesh writes one, seen through cameras whose rays
can be intersected with it directly. If the frames and the poses beside them
disagree, every score made from them is wrong, so the poses are checked by
what the frames show."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kyfromabove_3d.stages.build_mesh import GlbWriter                                            # noqa: E402
from kyfromabove_3d.stages import render_flyaround as rf                                                # noqa: E402

WALL_L, WALL_H = 20.0, 10.0


def gradient_texture(w=256, h=64):
    """Red rises along the wall, green up it: a pixel's colour says where on the wall it is."""
    x = (np.arange(w) + 0.5) / w
    y = (np.arange(h) + 0.5) / h
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = np.round(255 * x)[None, :]
    img[..., 1] = np.round(255 * (1 - y))[:, None]                          # row 0 is the top of the wall
    img[..., 2] = 40
    return img


def wall_glb(tmp_path: Path) -> Path:
    """A 20 m x 10 m wall in the plane y = 0, facing -y, plus a flat-coloured roof slab."""
    w = GlbWriter("png")
    pos = np.array([[0, 0, 0], [WALL_L, 0, 0], [WALL_L, 0, WALL_H], [0, 0, WALL_H]], float)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)                  # build_mesh's uv: v = 0 at the bottom
    nrm = np.tile([0.0, -1.0, 0.0], (4, 1))
    w.primitive(pos, nrm, uv, np.array([0, 1, 2, 0, 2, 3]), w.material_texture(gradient_texture(), "wall"))
    roof = np.array([[0, 0, WALL_H], [WALL_L, 0, WALL_H], [WALL_L, 8, WALL_H], [0, 8, WALL_H]], float)
    w.primitive(roof, np.tile([0.0, 0.0, 1.0], (4, 1)), None, np.array([0, 1, 2, 0, 2, 3]),
                w.material_flat((200, 30, 30), "roof"))
    p = tmp_path / "building.glb"
    w.write(p, {"name": "test wall", "gers": "00000000-test"})
    return p


@pytest.fixture
def scene(tmp_path):
    return rf.load_glb(wall_glb(tmp_path))


def test_the_glb_comes_back_in_the_local_frame(scene):
    V = scene.tris.reshape(-1, 3)
    assert np.allclose(V.min(0), [0, 0, 0], atol=1e-5) and np.allclose(V.max(0), [WALL_L, 8, WALL_H], atol=1e-5)
    assert sorted(set(scene.tex.tolist())) == [-1, 0]
    # build_mesh writes a flat colour's bytes as glTF's linear factor; drawn as a
    # viewer draws it, it comes out encoded to sRGB
    roof = scene.colour[scene.tex < 0]
    assert np.allclose(roof, rf._linear_to_srgb(np.array([200, 30, 30]) / 255), atol=1e-6)


def ray_hits_wall(K, pose, w, h):
    """Each pixel's ray against the plane y = 0: (hit xyz (H, W, 3), depth (H, W), on the wall (H, W))."""
    u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    d = d_cam @ pose.R_cw                                                    # to the world: R_cw.T @ d
    s = -pose.C[1] / d[..., 1]
    X = pose.C + s[..., None] * d
    on = (s > 0) & (X[..., 0] > 0.05) & (X[..., 0] < WALL_L - 0.05) & (X[..., 2] > 0.05) & (X[..., 2] < WALL_H - 0.05)
    return X, s, on                                                          # d_cam has z = 1, so s is the depth


def test_depth_and_colour_match_the_rays_through_each_pixel(scene):
    w, h = 160, 120
    K = rf.intrinsics(w, h, 70.0)
    C = np.array([-6.0, -18.0, 4.0])                                          # well off to one side: strong perspective
    pose = rf.Pose(rf.look_rotation(np.array([10.0, 0.0, 5.0]) - C), C)
    img, depth = rf.render(scene, K, pose.R_cw, pose.t_cw, w, h)
    X, s, on = ray_hits_wall(K, pose, w, h)
    on &= X[..., 2] < WALL_H - 0.3                                           # the roof slab's edge is not the wall's
    assert on.sum() > 2000
    assert np.abs(depth[on] - s[on]).max() < 1e-6
    # perspective-correct texture: the colour names the point the ray hit
    red = img[..., 0][on].astype(float) / 255 * WALL_L
    green = img[..., 1][on].astype(float) / 255 * WALL_H
    assert np.abs(red - X[..., 0][on]).max() < 0.2
    assert np.abs(green - X[..., 2][on]).max() < 0.4
    assert (depth[~on & (img == rf.BACKGROUND).all(-1)] == np.inf).all()


def test_a_wall_that_runs_past_the_camera_is_drawn_to_the_edge(scene):
    """Standing 2 m off the wall near its middle, looking along it: the wall's
    two triangles both have corners behind the camera, and must be cut, not dropped."""
    w, h = 160, 120
    K = rf.intrinsics(w, h, 90.0)
    C = np.array([8.0, -2.0, 1.6])
    pose = rf.Pose(rf.look_rotation(np.array([1.0, 0.25, 0.1])), C)
    assert (np.einsum("ij,tkj->tk", pose.R_cw, scene.tris[scene.tex >= 0]) + pose.t_cw)[..., 2].min() < 0
    img, depth = rf.render(scene, K, pose.R_cw, pose.t_cw, w, h)
    X, s, on = ray_hits_wall(K, pose, w, h)
    on &= X[..., 2] < WALL_H - 0.3
    assert on.sum() > 0.25 * w * h
    assert np.abs(depth[on] - s[on]).max() < 1e-6
    red = img[..., 0][on].astype(float) / 255 * WALL_L
    assert np.abs(red - X[..., 0][on]).max() < 0.2


def test_clipping_keeps_what_is_in_front():
    """A triangle cut at the near plane: the pieces cover exactly its part in front, and carry its attributes."""
    P = np.array([[[0.0, 0.0, -1.0], [2.0, 0.0, 3.0], [0.0, 2.0, 3.0]],      # one corner behind
                  [[0.0, 0.0, 5.0], [2.0, 0.0, -3.0], [0.0, 2.0, -3.0]],      # two corners behind
                  [[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -2.0]]])    # all behind
    A = P[..., :1] * 10.0                                                     # an attribute: 10 x
    Q, B, src = rf.clip_near(P, A, 1.0)
    assert sorted(src.tolist()) == [0, 0, 1]
    assert (Q[..., 2] >= 1.0 - 1e-12).all()
    assert np.allclose(B[..., 0], Q[..., 0] * 10.0)                            # attributes follow the cut
    normal = lambda T: np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    for i in (0, 1):                                                          # same side up as the input
        assert (normal(Q[src == i]) @ normal(P[i:i + 1])[0] > 0).all()


def test_the_nearer_surface_wins(scene):
    w, h = 120, 90
    K = rf.intrinsics(w, h, 60.0)
    C = np.array([10.0, -20.0, 25.0])                                          # above, looking down onto the roof
    pose = rf.Pose(rf.look_rotation(np.array([10.0, 4.0, WALL_H]) - C), C)
    img, depth = rf.render(scene, K, pose.R_cw, pose.t_cw, w, h)
    roof_rgb = 255 * rf._linear_to_srgb(np.array([200, 30, 30]) / 255)
    roof = np.abs(img.astype(float) - roof_rgb).sum(-1) < 3
    assert roof[h // 2, w // 2]                                               # the roof, not the wall behind it
    # the middle pixel's ray, against the roof's plane
    u, v = w // 2, h // 2
    d = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0]) @ pose.R_cw
    assert np.isclose(depth[v, u], (WALL_H - C[2]) / d[2], atol=1e-6)


def test_supersampling_keeps_the_pixel_grid(scene):
    w, h = 80, 60
    K = rf.intrinsics(w, h, 60.0)
    C = np.array([10.0, -25.0, 5.0])
    pose = rf.Pose(rf.look_rotation(np.array([10.0, 0.0, 5.0]) - C), C)
    one = rf.render_frame(scene, K, pose, w, h, 1).astype(float)
    four = rf.render_frame(scene, K, pose, w, h, 4).astype(float)
    inner = (one != rf.BACKGROUND).all(-1)
    inner[[0, -1], :] = inner[:, [0, -1]] = False
    from scipy.ndimage import binary_erosion
    inner = binary_erosion(inner, iterations=2)
    assert np.abs(one[inner] - four[inner]).mean() < 2.0                      # the same picture, smoother edges only


def test_the_orbit_keeps_the_building_in_the_middle():
    poses = rf.orbit_path(np.array([5.0, -3.0]), 12.0, 60.0, 20.0, 40.0, 2.0, 24)
    assert len(poses) == 48
    target = np.array([5.0, -3.0, 12.0])
    for p in poses:
        x = p.R_cw @ target + p.t_cw
        assert x[2] > 0 and np.allclose(x[:2] / x[2], 0, atol=1e-9)
        assert np.isclose(np.hypot(*(p.C[:2] - target[:2])), 60.0)
        assert np.allclose(p.R_cw @ p.R_cw.T, np.eye(3), atol=1e-12)
    assert np.isclose(poses[0].C[2], 20.0) and np.isclose(poses[-1].C[2], 40.0)
    assert np.allclose(poses[0].C[:2], [5.0, -63.0])                         # starts due south


def test_the_walk_stays_off_the_wall_and_faces_it():
    xy = np.array([[0, 0], [30, 0], [30, 20], [0, 20]], float)
    poses = rf.walk_path(xy, 8.0, 1.0, 1.6, 20.0, 0.0)
    from shapely.geometry import Point, Polygon
    hull = Polygon(xy)
    dist = np.array([hull.exterior.distance(Point(p.C[:2])) for p in poses])
    assert np.allclose(dist, 8.0, atol=0.05)
    assert all(np.isclose(p.C[2], 1.6) for p in poses)
    centre = np.array([15.0, 10.0, 1.6])
    for p in poses:
        fwd = p.R_cw[2]
        assert np.isclose(np.degrees(np.arcsin(fwd[2])), 20.0)
        to_centre = centre[:2] - p.C[:2]
        assert np.dot(fwd[:2], to_centre) > 0                                # looking in, not out
    P = np.array([p.C[:2] for p in poses])
    signed_area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    assert signed_area > 0                                                   # counter-clockwise


def test_shake_moves_the_camera_and_records_where_it_went():
    poses = rf.orbit_path(np.zeros(2), 10.0, 50.0, 20.0, 20.0, 1.0, 60)
    shaken = rf.shake(poses, 0.3, 1.0)
    dC = np.array([s.C - p.C for s, p in zip(shaken, poses)])
    assert 0.1 < dC.std() < 0.6
    for s in shaken:
        assert np.allclose(s.R_cw @ s.R_cw.T, np.eye(3), atol=1e-9)
    assert rf.shake(poses, 0.0, 0.0)[5] is poses[5]


def test_a_tum_line_is_the_camera_to_world_pose():
    from scipy.spatial.transform import Rotation
    C = np.array([3.0, -40.0, 12.0])
    pose = rf.Pose(rf.look_rotation(np.array([0.0, 0.0, 5.0]) - C), C)
    t, x, y, z, *q = map(float, rf.tum_line(1.5, pose).split())
    assert t == 1.5 and np.allclose([x, y, z], C)
    R_wc = Rotation.from_quat(q).as_matrix()
    assert np.allclose(R_wc, pose.R_cw.T, atol=1e-6)


def test_the_script_writes_frames_and_the_poses_that_made_them(tmp_path):
    glb = wall_glb(tmp_path)
    out = tmp_path / "fly"
    assert rf.main(["--glb", str(glb), "--out", str(out), "--width", "96", "--height", "64",
                    "--supersample", "1", "--turns", "1", "--frames-per-turn", "6"]) == 0
    rec = json.loads((out / "cameras.json").read_text())
    assert rec["source"]["gers"] == "00000000-test" and len(rec["frames"]) == 6
    assert rec["path"]["kind"] == "orbit"
    K = np.array(rec["camera"]["K"])
    tum = [l.split() for l in (out / "trajectory_tum.txt").read_text().splitlines() if not l.startswith("#")]
    assert len(tum) == 6
    from PIL import Image
    for f, line in zip(rec["frames"], tum):
        R, t, C = np.array(f["R_cw"]), np.array(f["t_cw"]), np.array(f["C"])
        assert np.allclose(R @ C + t, 0, atol=1e-5)
        assert np.allclose([float(v) for v in line[1:4]], C, atol=1e-5)
        img = np.asarray(Image.open(out / f["file"]))
        assert img.shape == (64, 96, 3)
        # the building's middle projects to the middle of every frame, and something is drawn there
        x = R @ np.array([WALL_L / 2, 4.0, WALL_H / 2]) + t
        u, v = (K @ (x / x[2]))[:2]
        assert abs(u - 47.5) < 1e-3 and abs(v - 31.5) < 1e-3
        assert (img[31, 47] != rf.BACKGROUND).any()


def test_a_stored_product_zip_is_read_like_the_glb(tmp_path):
    import zipfile
    glb = wall_glb(tmp_path)
    z = tmp_path / "3dtiles_mesh.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("tileset.json", "{}")
        f.write(glb, "building.glb")
    (tmp_path / "x").mkdir()
    got = rf.glb_path(z, tmp_path / "x")
    assert got.read_bytes() == glb.read_bytes()
