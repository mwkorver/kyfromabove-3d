"""Unit tests for walls painted from whole photographs (``kyfromabove_3d.mesh.paint``)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kyfromabove_3d.facade_fill import PLAIN, SEEN  # noqa: E402
from kyfromabove_3d.features import FaceBand, WallPlane  # noqa: E402
from kyfromabove_3d.mesh.paint import paint_section, section_uv, wall_sections  # noqa: E402
from kyfromabove_3d.mesh.walls import plane_bands, wall_geometry  # noqa: E402


def _walls(plane, offsets):
    """The measured walls of ``plane`` for one 5 m band per offset, from the ground up."""
    bands = [FaceBand(5.0 * i, 5.0 * (i + 1), off, 200, 0.1, 0.0, plane.length_m, "observed")
             for i, off in enumerate(offsets)]
    quads, _ = wall_geometry(plane, plane_bands(plane, bands), None, edge_face=plane.face)
    return [q for q in quads if q[3] == "wall"], [q for q in quads if q[3] != "wall"]


def test_a_wall_stepping_back_is_two_sections():
    # 400 West Market's sides: straight, then 5 m back for the crown
    plane = WallPlane(0, np.array([0.0, 0.0]), np.array([20.0, 0.0]), np.array([0.0, -1.0]), 0.0, 30.0)
    walls, other = _walls(plane, [0.0, 0.1, 0.0, 0.1, -5.0, -5.1])
    secs = wall_sections(plane, walls)
    assert len(secs) == 2
    low, high = sorted(secs, key=lambda s: s["z_lo"])
    assert abs(low["off"]) < 0.2 and (low["z_lo"], low["z_hi"]) == (0.0, 20.0)
    assert abs(high["off"] + 5.0) < 0.2 and (high["z_lo"], high["z_hi"]) == (20.0, 30.0)
    # the step's roof between them is a ledge, not part of either picture
    assert any(abs(q[0][:, 2].mean() - 20.0) < 1e-6 and np.ptp(q[0][:, 1]) > 4.9 for q in other)
    # together the section's pieces fill its picture from the bottom to the top
    for sec in secs:
        uv = np.vstack([section_uv(plane, sec, np.asarray(q[0], dtype=np.float64)) for q in sec["quads"]])
        assert np.isclose(uv.min(axis=0), 0.0).all() and np.isclose(uv.max(axis=0), 1.0).all()


def test_a_flat_wall_is_one_section():
    plane = WallPlane(0, np.array([0.0, 0.0]), np.array([20.0, 0.0]), np.array([0.0, -1.0]), 0.0, 30.0)
    walls, _ = _walls(plane, [0.0, 0.2, -0.1, 0.1, 0.0, 0.2])
    secs = wall_sections(plane, walls)
    assert len(secs) == 1 and (secs[0]["z_lo"], secs[0]["z_hi"]) == (0.0, 30.0)


# a 12 m x 8 m wall along y = 0, facing -y, seen by frames 30 m in front of it
PLANE = WallPlane(0, np.array([0.0, 0.0]), np.array([12.0, 0.0]), np.array([0.0, -1.0]), 0.0, 8.0)
SEC = {"off": 0.0, "a_lo": 0.0, "a_hi": 12.0, "z_lo": 0.0, "z_hi": 8.0}


def _cam(x: float, shade: int = 0):
    """A frame looking along +y from (x, -30, 4) at a wall of 1 m checks (y = 0)."""
    K = np.array([[500.0, 0, 160.0], [0, 500.0, 120.0], [0, 0, 1.0]])
    R = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    C = np.array([x, -30.0, 4.0])
    yy, xx = np.meshgrid(np.arange(240) + 0.5, np.arange(320) + 0.5, indexing="ij")
    rays = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ np.linalg.inv(K).T @ R      # world directions
    t = -C[1] / rays[..., 1]
    wx, wz = C[0] + t * rays[..., 0], C[2] + t * rays[..., 2]
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[..., 0] = (np.floor(wx) % 2) * 180 + 40 + shade
    img[..., 1] = (np.floor(wz) % 2) * 180 + 40
    img[..., 2] = 140
    return {"id": f"Front_{x}", "head": "Front", "img": img, "mask": None, "wh": (320, 240),
            "K": K, "R_cw": R, "C": C}


def test_a_section_is_painted_whole_from_one_frame():
    tex, info = paint_section(PLANE, SEC, [_cam(5.5), _cam(6.5)], texel_m=0.2)
    assert tex.shape == (40, 60, 3)
    assert (tex.source == SEEN).mean() > 0.95
    assert len(info["frames"]) >= 1 and info["seen"] > 0.95


def test_a_tree_in_front_of_the_wall_blocks_the_frame():
    # a tree 10 m in front of the wall's left half, in the first frame's depth map
    tree = _cam(5.5)
    ds = 8
    depth = np.full((240 // ds + 1, 320 // ds + 1), np.inf)
    depth[:, : 160 // ds] = 15.0                       # the left half of the view: something 15 m away
    tree.update(depth=depth, downscale=ds)
    tex, info = paint_section(PLANE, SEC, [tree], texel_m=0.2)
    src = tex.source[::-1]                                 # rows bottom-up, as the wall
    assert (src[:, :25] != SEEN).all()                    # behind the tree: not seen
    assert (src[:, 35:] == SEEN).all()                    # beside it: seen
    # a second frame, from the other side, sees past the tree and paints it
    tex2, info2 = paint_section(PLANE, SEC, [tree, _cam(0.0, shade=20)], texel_m=0.2)
    assert (tex2.source[::-1][:, :25] == SEEN).all()
    assert len(info2["frames"]) == 2


def test_a_section_no_frame_sees_is_not_painted():
    behind = _cam(6.0)
    behind["C"] = np.array([6.0, 30.0, 4.0])              # behind the wall: it faces away
    tex, info = paint_section(PLANE, SEC, [behind], texel_m=0.2)
    assert tex is None and info["frames"] == []
    assert PLAIN != SEEN


def test_the_frames_that_painted_a_section_are_found_on_it():
    from kyfromabove_3d.mesh.paint import paint_record, painted_by
    rec = paint_record(PLANE, 0, {"off": 0.0, "a_lo": 0.0, "a_hi": 6.0, "z_lo": 0.0, "z_hi": 8.0}, ["Front_1"])
    rec2 = paint_record(PLANE, 0, {"off": 0.0, "a_lo": 6.0, "a_hi": 12.0, "z_lo": 0.0, "z_hi": 8.0}, ["Front_2", "Back_3"])
    pts = np.array([[3.0, 0.0, 4.0], [9.0, 0.1, 2.0], [9.0, -3.0, 2.0], [20.0, 0.0, 2.0]])
    assert painted_by([rec, rec2], pts) == [frozenset({"Front_1"}), frozenset({"Front_2", "Back_3"}),
                                             frozenset(), frozenset()]
