"""The texture check on a synthetic wall: a facade pattern, four cameras
that photograph it, and a mesh textured the way build_mesh textures it --
from the most face-on camera, sampled on the mesh's own wall. Where the
mesh's wall is where the facade is, the other cameras agree with it and
need no shift; where it stands 1 m in front, they see the pattern slide."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from kyfromabove_3d.stages.build_mesh import GlbWriter                                            # noqa: E402
from kyfromabove_3d.stages.colorize_building import project_visible                               # noqa: E402
from kyfromabove_3d.features import WallPlane                               # noqa: E402
from kyfromabove_3d.texture_check import (face_grids, read_glb, score_face,  # noqa: E402
                                          summarise_face)

WALL_L, WALL_H = 20.0, 10.0
RES = 0.05                                                  # the facade pattern's cell (m)


def _facade():
    """A non-repeating facade: smoothed noise, 0.05 m cells, x in [0, 20], z in [0, 10]."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(11)
    p = gaussian_filter(rng.normal(size=(int(WALL_H / RES), int(WALL_L / RES))), 4.0)
    return 128 + 90 * p / p.std()


FACADE = _facade()


def facade_at(x, z):
    i = np.clip((z / RES).astype(int), 0, FACADE.shape[0] - 1)
    j = np.clip((x / RES).astype(int), 0, FACADE.shape[1] - 1)
    inside = (x >= 0) & (x <= WALL_L) & (z >= 0) & (z <= WALL_H)
    return np.where(inside, FACADE[i, j], 60.0)


def camera(x, head, f=1400.0, wh=(640, 480)):
    """A camera 60 m in front of the wall (the wall faces -y), 25 m up, looking at its middle."""
    C = np.array([x, -60.0, 25.0])
    z = np.array([10.0, 0.0, 5.0]) - C; z /= np.linalg.norm(z)
    xa = np.cross(z, [0.0, 0.0, 1.0]); xa /= np.linalg.norm(xa)
    y = np.cross(z, xa)
    R = np.stack([xa, y, z])
    K = np.array([[f, 0, wh[0] / 2], [0, f, wh[1] / 2], [0, 0, 1.0]])
    # render: every pixel's ray meets the plane y = 0
    u, v = np.meshgrid(np.arange(wh[0]) + 0.5, np.arange(wh[1]) + 0.5)
    rays = (R.T @ (np.linalg.inv(K) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)]))).T
    t = -C[1] / rays[:, 1]
    hit = C[None, :] + t[:, None] * rays
    g = facade_at(hit[:, 0], hit[:, 2]).reshape(wh[1], wh[0])
    img = np.repeat(np.clip(g, 0, 255)[..., None], 3, axis=2).astype(np.uint8)
    return {"id": f"KY/{head}_7000_{int(x)}", "head": head, "K": K, "R_cw": R, "C": C, "wh": wh,
            "img": img, "mask": None, "depth": None}


CAMS = [camera(-25.0, "Left"), camera(10.0, "Fwd"), camera(45.0, "Right"), camera(-45.0, "Bwd")]
PLANE = WallPlane(face=0, p0=np.array([0.0, 0.0]), p1=np.array([WALL_L, 0.0]),
                  normal=np.array([0.0, -1.0]), z_lo=0.0, z_hi=WALL_H)


def textured_wall_glb(tmp_path: Path, y_wall: float) -> Path:
    """The wall quad at ``y_wall``, textured the way build_mesh does: from the
    most face-on camera (Fwd), sampled on the quad's own plane."""
    texel = 0.1
    W, H = int(WALL_L / texel), int(WALL_H / texel)
    xs = (np.arange(W) + 0.5) * texel
    zs = WALL_H - (np.arange(H) + 0.5) * texel                  # row 0 is the top
    X, Z = np.meshgrid(xs, zs)
    pts = np.column_stack([X.ravel(), np.full(X.size, y_wall), Z.ravel()])
    nrm = np.tile([0.0, -1.0, 0.0], (len(pts), 1))
    src = CAMS[1]
    idx, u, v, _ = project_visible(pts, nrm, src, 0.15, 1.5)
    tex = np.zeros((len(pts), 3), dtype=np.uint8)
    tex[idx] = src["img"][v, u]
    tex = tex.reshape(H, W, 3)
    verts = np.array([[0, y_wall, 0], [WALL_L, y_wall, 0], [WALL_L, y_wall, WALL_H], [0, y_wall, WALL_H]], float)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    glb = GlbWriter("png")
    glb.primitive(verts, np.tile([0.0, -1.0, 0.0], (4, 1)), uv, np.array([0, 1, 2, 0, 2, 3]),
                  glb.material_texture(tex, "walls_0"))
    path = tmp_path / "building.glb"
    glb.write(path, {"name": "test"})
    return path


def _check(tmp_path, y_wall):
    grids = face_grids(read_glb(textured_wall_glb(tmp_path, y_wall)), [PLANE], spacing_m=0.2)
    assert [g.key for g in grids] == ["face 0"]
    g = grids[0]
    scores = score_face(g, CAMS, lambda p, n, c: project_visible(p, n, c, 0.15, 1.5))
    return g, scores, summarise_face(g, scores)


def test_the_glb_reads_back_as_the_wall_it_was_written_as(tmp_path):
    prims = read_glb(textured_wall_glb(tmp_path, 0.0))
    assert len(prims) == 1
    p = prims[0]
    assert np.allclose(p.positions[:, 1], 0.0) and p.positions[:, 2].max() == pytest.approx(WALL_H)
    g = face_grids(prims, [PLANE], spacing_m=0.2)[0]
    assert g.area_m2 == pytest.approx(WALL_L * WALL_H, rel=0.05)
    # the model's colour at a grid point is the facade's colour there
    pts = g.xyz[g.valid]
    truth = facade_at(pts[:, 0], pts[:, 2])
    assert np.corrcoef(g.model[g.valid][:, 0], truth)[0, 1] > 0.9


def test_a_wall_in_the_right_place_agrees_with_the_other_frames(tmp_path):
    _, scores, face = _check(tmp_path, 0.0)
    held = [s for s in scores if not s.source]
    assert len(held) >= 2
    assert all(s.ncc > 0.8 for s in held)
    assert all(np.hypot(*s.shift_m) <= 0.2 for s in held)
    assert face["shift_m"] is None                              # no frame needed a shift to agree


def test_a_wall_a_metre_in_front_shows_as_a_shift_in_the_other_frames(tmp_path):
    _, scores, face = _check(tmp_path, -1.0)
    held = [s for s in scores if not s.source]
    left, right = next(s for s in held if s.head == "Left"), next(s for s in held if s.head == "Right")
    # seen from 35 m either side of the texturing camera at 60 m, a 1 m depth
    # error slides the pattern about 0.6 m along the wall, in opposite senses
    assert left.ncc < 0.6 and right.ncc < 0.6
    assert left.ncc_best - left.ncc > 0.2
    assert 0.4 <= abs(left.shift_m[0]) <= 0.9 and 0.4 <= abs(right.shift_m[0]) <= 0.9
    assert np.sign(left.shift_m[0]) != np.sign(right.shift_m[0])
    assert face["clear_shifts"] >= 2


def test_the_source_frame_is_never_scored_against_itself(tmp_path):
    g, scores, _ = _check(tmp_path, -1.0)
    fwd = [s for s in scores if s.head == "Fwd"]
    # Fwd textured every point it sees, so it appears only as the ceiling
    assert fwd and all(s.source for s in fwd)
    assert fwd[0].ncc > 0.9                                     # it agrees with itself even on the wrong wall


def test_an_occluded_point_is_not_scored(tmp_path):
    """Something 20 m in front of the wall hides its left half from the Right camera."""
    cams = [dict(c) for c in CAMS]
    right = cams[2]
    w, h = right["wh"]
    ds = 8
    depth = np.full((h // ds + 1, w // ds + 1), np.inf)
    depth[:, : (w // 2) // ds] = 20.0                           # the left half of the image: an occluder
    right.update(depth=depth, downscale=ds)
    grids = face_grids(read_glb(textured_wall_glb(tmp_path, 0.0)), [PLANE], spacing_m=0.2)
    project = lambda p, n, c: project_visible(p, n, c, 0.15, 1.5)          # noqa: E731
    open_n = next(s.n for s in score_face(grids[0], CAMS, project) if s.head == "Right" and not s.source)
    hidden_n = next(s.n for s in score_face(grids[0], cams, project) if s.head == "Right" and not s.source)
    assert hidden_n < 0.7 * open_n


def test_a_neighbour_prism_hides_the_wall_behind_it(tmp_path):
    """A 40 m block standing 20 m in front of the left half of the wall, as
    the Right camera sees it: with the block as a prism in the depth maps,
    those points are not seen -- with LiDAR points alone (none here) they
    would be."""
    from kyfromabove_3d.stages.colorize_building import add_prism_occluders
    cams = [dict(c) for c in CAMS]
    for c in cams:
        w, h = c["wh"]
        c.update(depth=np.full((h // 8 + 1, w // 8 + 1), np.inf), downscale=8)
    ring = np.array([[-5.0, -25.0], [8.0, -25.0], [8.0, -20.0], [-5.0, -20.0], [-5.0, -25.0]])
    add_prism_occluders(cams, [(ring, 0.0, 40.0)])
    g = face_grids(read_glb(textured_wall_glb(tmp_path, 0.0)), [PLANE], spacing_m=0.2)[0]
    pts, nrm = g.xyz[g.valid], g.normal[g.valid]
    idx_open, *_ = project_visible(pts, nrm, CAMS[1], 0.15, 1.5)
    idx_hid, *_ = project_visible(pts, nrm, cams[1], 0.15, 1.5)
    hidden = np.setdiff1d(idx_open, idx_hid)
    assert len(hidden) > 0.2 * len(idx_open)
    assert pts[hidden, 0].max() < 12.0                     # only the part behind the block is lost
