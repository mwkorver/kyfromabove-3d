"""Matched wall bands in the profile: one is drawn where the frames put it,
is never the source of a mirror, and ``match_profiles`` finds a wall drawn
1.5 m behind its footprint edge, in two frames rendered of it."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from test_features import _profiles_for_square  # noqa: E402


def test_a_matched_band_is_extruded_at_its_own_depth():
    from kyfromabove_3d.features import FaceBand, extrude_profile, wall_planes

    pl = wall_planes(np.array([[0.0, 0.0], [60.0, 0.0], [60.0, 20.0], [0.0, 20.0]]), z_lo=0.0, z_hi=20.0)[0]
    bands = [FaceBand(0.0, 10.0, 0.0, 100, 0.1, 2.0, 58.0, "observed"),
             FaceBand(10.0, 20.0, -1.5, 0, 0.0, 1.5, 58.5, "matched")]
    pts = extrude_profile(pl, bands, spacing_m=0.5)
    up = pts[pts[:, 2] > 10.5]
    assert len(up) and np.allclose((up[:, :2] - pl.p0) @ pl.normal, -1.5, atol=1e-6)


def test_a_matched_band_is_never_mirrored():
    from kyfromabove_3d.features import mirror_hidden_bands

    planes, west, east, profiles = _profiles_for_square({}, {})
    for k, b in enumerate(profiles[west.face]):                              # the west face known only from the frames
        profiles[west.face][k] = type(b)(b.z_lo, b.z_hi, -2.0, 0, 0.0, 5.0, 95.0, "matched")
    mirror_hidden_bands(planes, profiles)
    assert all(b.status == "unknown" for b in profiles[east.face])


def test_match_profiles_finds_a_wall_behind_its_footprint_edge():
    from kyfromabove_3d.stages import render_flyaround as rf

    from kyfromabove_3d.features import FaceBand, wall_planes
    from kyfromabove_3d.wall_match import match_profiles

    pl = wall_planes(np.array([[0.0, 0.0], [60.0, 0.0], [60.0, 30.0], [0.0, 30.0]]), z_lo=0.0, z_hi=30.0)[0]
    d = (pl.p1 - pl.p0) / np.linalg.norm(pl.p1 - pl.p0)
    true_off = -1.5
    # a textured wall, 60 m along and 30 m up, standing 1.5 m behind the edge
    tex = rf.ground_scene(np.array([0.0, 0.0]), 30.0, 0.0, seed=4)
    t = tex.tris
    along, up = t[..., 0] + 30.0, (t[..., 1] + 30.0) * 0.5
    xy = pl.p0[None, None, :] + along[..., None] * d[None, None, :] + true_off * pl.normal[None, None, :]
    tex.tris = np.concatenate([xy, up[..., None]], axis=-1)
    W, H = 900, 600
    K = rf.intrinsics(W, H, 55.0)
    centre = np.r_[pl.p0 + 30.0 * d, 15.0]
    cams = []
    for k, side in enumerate((-22.0, 22.0)):
        C = centre + np.r_[70.0 * pl.normal + side * d, 25.0]
        R = rf.look_rotation(centre - C)
        img, _ = rf.render(tex, K, R, -R @ C, W, H)
        cams.append(dict(id=f"cam{k}", K=K, R_cw=R, C=C, img=img))

    def visible(xyz, nrm, c):
        pc = (xyz - c["C"]) @ c["R_cw"].T
        u = c["K"][0, 0] * pc[:, 0] / pc[:, 2] + c["K"][0, 2]
        v = c["K"][1, 1] * pc[:, 1] / pc[:, 2] + c["K"][1, 2]
        return np.flatnonzero((pc[:, 2] > 0) & (u > 2) & (u < W - 3) & (v > 2) & (v < H - 3))

    profiles = {pl.face: [FaceBand(z, z + 5.0, None, 0, 0.0, 0.0, 0.0, "unknown") for z in (5.0, 10.0, 15.0, 20.0)]}
    report = match_profiles([pl], profiles, cams, visible, log=lambda *_: None)
    got = [b for b in profiles[pl.face] if b.status == "matched"]
    assert report["matched"] >= 3 and len(got) == report["matched"], report
    assert all(abs(b.offset_m - true_off) <= 0.15 for b in got), [b.offset_m for b in got]
    assert report["lean_m"] == 0.0                                           # no measured band to lean on


def test_a_matched_band_moved_in_is_closed_at_its_corners():
    from kyfromabove_3d.stages import build_mesh as bm

    from kyfromabove_3d.features import FaceBand, wall_planes

    pl = wall_planes(np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]]), z_lo=0.0, z_hi=10.0)[0]
    bands = [FaceBand(0.0, 5.0, -1.0, 0, 0.0, 0.0, float(pl.length_m), "matched")]
    # both neighbours are prisms on the footprint line: their walls meet this face's ends at the corners
    quads = bm.corner_returns(pl, bands, lambda end, z: pl.p0 if end == 0 else pl.p1)
    assert len(quads) == 2 and all(k == "return" for *_, k in quads)
    for corners, want, _, _ in quads:
        gap = corners[1] - corners[0]
        assert abs(np.hypot(gap[0], gap[1]) - 1.0) < 1e-6                   # 1 m, from the moved wall to the corner
        assert np.isclose(corners[:, 2].min(), 0.0) and np.isclose(corners[:, 2].max(), 5.0)
    # a band on the footprint line needs nothing
    flat = [FaceBand(0.0, 5.0, 0.0, 0, 0.0, 0.0, float(pl.length_m), "matched")]
    assert bm.corner_returns(pl, flat, lambda end, z: pl.p0 if end == 0 else pl.p1) == []
