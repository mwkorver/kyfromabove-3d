"""Snapping footprint corners to the edges the frames show."""

from __future__ import annotations

import numpy as np

from kyfromabove_3d.corners import edge_candidates, edge_strip, flight_line_of, snap_corners
from kyfromabove_3d.features import WallPlane, wall_planes


def _strip_with_edge(edge_col: int, w: int = 60, h: int = 40, noise: float = 2.0, sky: bool = True, seed: int = 0):
    """Brick-like wall (horizontal courses) up to ``edge_col``; past it sky
    (flat) or, when ``sky`` is False, the same wall in shadow."""
    rng = np.random.default_rng(seed)
    rows = 120 + 40 * ((np.arange(h) // 4) % 2)                       # courses every 4 rows
    wall = np.repeat(rows[:, None], w, axis=1).astype(float)
    gray = wall.copy()
    if sky:
        gray[:, edge_col:] = 220.0
    else:
        gray[:, edge_col:] = wall[:, edge_col:] * 0.5
    gray += rng.normal(0, noise, gray.shape)
    seen = np.ones_like(gray, dtype=bool)
    along = -3.0 + 0.1 * np.arange(w)
    return gray, seen, along


def test_edge_finder_lands_on_the_silhouette_and_reads_the_structure_drop():
    gray, seen, along = _strip_with_edge(35)
    cands = edge_candidates(gray, seen, along)
    assert cands, "no candidate"
    a, strength, far = cands[0]
    assert abs(a - along[35]) <= 0.15                                  # within a texel or two
    assert far < 0.6                                                   # the courses stop past the edge: sky
    assert strength > 2.0


def test_a_shadow_keeps_the_structure_past_the_edge():
    gray, seen, along = _strip_with_edge(35, sky=False)
    cands = edge_candidates(gray, seen, along)
    assert cands and abs(cands[0][0] - along[35]) <= 0.15
    assert cands[0][2] > 0.6                                           # the courses continue, darker


def test_no_edge_or_unseen_strip_gives_nothing():
    gray, seen, along = _strip_with_edge(35)
    seen[:, :] = False
    assert edge_candidates(gray, seen, along) == []
    flat = np.full((40, 60), 150.0); flat += np.random.default_rng(1).normal(0, 2.0, flat.shape)
    assert edge_candidates(flat, np.ones_like(flat, dtype=bool), along) == []


def test_flight_line_is_the_middle_token():
    assert flight_line_of("KY_KYAPED_2024_Season1_3IN/Bwd_7021_47003") == "7021"


def test_edge_strip_samples_the_plane_through_the_sampler():
    plane = WallPlane(face=0, p0=np.array([0.0, 0.0]), p1=np.array([20.0, 0.0]), normal=np.array([0.0, -1.0]), z_lo=0.0, z_hi=12.0)
    got = {}

    def sampler(pts, normals):
        got["pts"] = pts
        return np.full((len(pts), 3), 100.0), np.ones(len(pts))

    gray, seen, along = edge_strip(plane, sampler, 20.0, 1.0, 9.0, half_m=3.0, texel_m=0.5)
    assert gray.shape == (17, 13) and seen.all()
    assert abs(along[0] - 17.0) < 1e-9 and abs(along[-1] - 23.0) < 1e-9
    assert np.allclose(got["pts"][:, 1], 0.0) and got["pts"][:, 2].min() == 1.0    # on the plane, from 1 m up


def test_snap_moves_the_unverified_line_onto_the_observed_corners():
    """A 20 x 10 m box; the east edge (face 1) is drawn 1 m too far east.
    The north and south faces are verified; their east ends are observed at
    20 m (the truth). The east line must slide west by 1 m, the other three
    corners stay, and the loop settles in a pass or two."""
    ring = np.array([[0.0, 0.0], [21.0, 0.0], [21.0, 10.0], [0.0, 10.0]])
    planes = wall_planes(ring, 0.0, 10.0, min_length_m=0.3)
    # faces: 0 south (0,0)->(21,0); 1 east; 2 north (21,10)->(0,10); 3 west
    verified = {0, 2, 3}
    obs = {(0, "end"): 20.0, (2, "start"): 1.0, (1, "start"): None, (1, "end"): None}
    new_ring, new_planes, recs = snap_corners(ring, planes, verified, obs)
    assert np.allclose(new_ring[1], [20.0, 0.0], atol=0.05) and np.allclose(new_ring[2], [20.0, 10.0], atol=0.05)
    assert np.allclose(new_ring[0], [0.0, 0.0]) and np.allclose(new_ring[3], [0.0, 10.0])
    east = [p for p in new_planes if p.face == 1][0]
    assert abs(east.p0[0] - 20.0) < 0.05 and abs(east.p1[0] - 20.0) < 0.05
    assert recs[1].status == "snapped" and recs[2].status == "snapped"
    assert recs[1].gap_before_m > 0.9 and recs[1].gap_after_m < 0.1


def test_snap_refuses_a_big_move_and_a_disagreement_and_skips_party_corners():
    ring = np.array([[0.0, 0.0], [21.0, 0.0], [21.0, 10.0], [0.0, 10.0]])
    planes = wall_planes(ring, 0.0, 10.0, min_length_m=0.3)
    # a 5 m move: refused
    _, _, recs = snap_corners(ring, planes, {0, 2, 3}, {(0, "end"): 16.0})
    assert recs[1].status == "refused"
    # neither face verified, the two observations 3 m apart: refused
    _, _, recs = snap_corners(ring, planes, set(), {(0, "end"): 20.0, (1, "start"): 3.0})
    assert recs[1].status == "refused"
    # a party corner is never touched
    new_ring, _, recs = snap_corners(ring, planes, {0, 2, 3}, {(0, "end"): 20.0}, skip={1: "skipped-party"})
    assert recs[1].status == "skipped-party" and np.allclose(new_ring[1], [21.0, 0.0])
