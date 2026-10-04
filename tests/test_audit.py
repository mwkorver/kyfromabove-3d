"""The audit on made-up buildings with known answers: a box over its own
solid is clean; a wall with a 10 m x 5 m piece missing shows a 50 m2 hole on
that face; a 3 m recess is not a hole; a panel on a flat roof is flagged and
a penthouse on its own step is not; dots and lines in a texture's sources
are counted, a clean texture has neither."""

from __future__ import annotations

import numpy as np

from kyfromabove_3d import audit as A
from kyfromabove_3d.features import wall_planes


def _quad(p0, p1, p2, p3):
    return [p0, p1, p2, p3]


def _mesh(quads):
    V, T = [], []
    for q in quads:
        k = len(V)
        V.extend(q)
        T.extend([[k, k + 1, k + 2], [k, k + 2, k + 3]])
    return np.array(V, dtype=float), np.array(T, dtype=np.int64)


def _box_quads(x0, x1, y0, y1, z0, z1, south=True):
    q = [
        _quad((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)),      # north
        _quad((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)),      # east
        _quad((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)),      # west
        _quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)),      # roof
    ]
    if south:
        q.append(_quad((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)))
    return q


def _south_with_gap(x0, x1, z0, z1, gx0, gx1, gz0, gz1, y=0.0):
    """The south wall (y = 0) with the rectangle [gx0, gx1] x [gz0, gz1] cut out."""
    return [_quad((x0, y, z0), (x1, y, z0), (x1, y, gz0), (x0, y, gz0)),
            _quad((x0, y, gz1), (x1, y, gz1), (x1, y, z1), (x0, y, z1)),
            _quad((x0, y, gz0), (gx0, y, gz0), (gx0, y, gz1), (x0, y, gz1)),
            _quad((gx1, y, gz0), (x1, y, gz0), (x1, y, gz1), (gx1, y, gz1))]


def _field(fn, x0=0.0, x1=20.0, y0=0.0, y1=10.0, cell=0.5):
    xs = np.arange(x0 + cell / 2, x1, cell); ys = np.arange(y0 + cell / 2, y1, cell)
    X, Y = np.meshgrid(xs, ys)
    Z = fn(X, Y).astype(float)
    return xs, ys, Z, np.isfinite(Z)


RING = np.array([[0, 0], [20, 0], [20, 10], [0, 10], [0, 0]], dtype=float)


def _solid(field):
    pts, top = A.lidar_solid(*field, floor_z=1.5)
    face = np.full(len(pts), -1, dtype=np.int64)
    face[~top] = A.nearest_face(pts[~top], wall_planes(RING, 0.0, 15.0, min_length_m=0.3))
    return A.Solid(pts, face)


def _south_face():
    return [e.face for e in wall_planes(RING, 0.0, 15.0, min_length_m=0.3) if e.normal[1] < -0.9][0]


def test_a_whole_box_over_its_own_solid_has_no_hole():
    V, T = _mesh(_box_quads(0, 20, 0, 10, 0, 15))
    pts, _ = A.sample_mesh(V, T)
    h = A.holes(pts, _solid(_field(lambda X, Y: np.full(X.shape, 15.0))))
    assert h["total_m2"] < 2.0, h["views"]


def test_a_missing_piece_of_wall_is_a_hole_of_its_size_on_its_face():
    quads = _box_quads(0, 20, 0, 10, 0, 15, south=False) + _south_with_gap(0, 20, 0, 15, 5, 15, 5, 10)
    V, T = _mesh(quads)
    pts, _ = A.sample_mesh(V, T)
    h = A.holes(pts, _solid(_field(lambda X, Y: np.full(X.shape, 15.0))))
    from_s = [v for v in h["views"] if v["view"] == "from S"][0]
    assert 40.0 <= from_s["hole_m2"] <= 60.0                          # 10 m x 5 m, give or take a pixel ring
    assert from_s["holes"][0]["face"] == _south_face()
    assert 4.0 <= from_s["holes"][0]["z_lo"] <= 5.5 and 9.5 <= from_s["holes"][0]["z_hi"] <= 11.0
    assert h["by_face"][str(_south_face())] >= 40.0


def test_a_shallow_recess_is_not_a_hole():
    # the same piece of wall set 3 m back, with its returns: a loggia
    quads = _box_quads(0, 20, 0, 10, 0, 15, south=False) + _south_with_gap(0, 20, 0, 15, 5, 15, 5, 10)
    quads += [_quad((5, 3, 5), (15, 3, 5), (15, 3, 10), (5, 3, 10)),
              _quad((5, 0, 5), (5, 3, 5), (5, 3, 10), (5, 0, 10)),
              _quad((15, 0, 5), (15, 3, 5), (15, 3, 10), (15, 0, 10))]
    V, T = _mesh(quads)
    pts, _ = A.sample_mesh(V, T)
    h = A.holes(pts, _solid(_field(lambda X, Y: np.full(X.shape, 15.0))))
    assert [v for v in h["views"] if v["view"] == "from S"][0]["hole_m2"] < 2.0


def test_a_panel_on_a_flat_roof_is_flagged_and_a_penthouse_on_its_step_is_not():
    box = _box_quads(0, 20, 0, 10, 0, 15)
    panel = [_quad((8, 5, 15), (12, 5, 15), (12, 5, 18), (8, 5, 18))]           # 4 m x 3 m, mid-roof
    V, T = _mesh(box + panel)
    flat = _field(lambda X, Y: np.full(X.shape, 15.0))
    r = A.roof_walls(V, T, *flat, RING)
    assert 10.0 <= r["m2"] <= 13.0
    pent = _box_quads(8, 12, 3, 7, 15, 18)
    V, T = _mesh(box + pent)
    stepped = _field(lambda X, Y: np.where((X > 8) & (X < 12) & (Y > 3) & (Y < 7), 18.0, 15.0))
    assert A.roof_walls(V, T, *stepped, RING)["m2"] < 1.0
    V, T = _mesh(box)
    assert A.roof_walls(V, T, *flat, RING)["m2"] == 0.0                         # the outer walls are not on the roof


def test_dots_and_lines_are_counted_and_a_clean_texture_has_neither():
    texel = 0.15
    s = np.full((200, 300), A.PLAIN, dtype=np.uint8)
    yy, xx = np.mgrid[:200, :300]
    for cy, cx in ((40, 40), (40, 120), (120, 60), (150, 200)):                 # discs 1.8 m across round a speck
        s[(yy - cy) ** 2 + (xx - cx) ** 2 <= 36] = A.NEAR
        s[cy, cx] = A.SEEN
    assert A.texture_artefacts([s], texel)["dots"] == 4
    s = np.full((200, 300), A.PLAIN, dtype=np.uint8)
    for r in (20, 60, 100):                                                      # 3 m bridges across blank wall
        s[r, 50:70] = A.NEAR
        s[r, 49] = s[r, 70] = A.SEEN
    t = A.texture_artefacts([s], texel)
    assert t["lines"] == 3 and t["dots"] == 0
    s = np.full((200, 300), A.PLAIN, dtype=np.uint8)
    s[:120, :] = A.SEEN
    s[120:130, :] = A.NEAR                                                      # a solid patch grown 1.5 m
    t = A.texture_artefacts([s], texel)
    assert t["dots"] == 0 and t["lines"] == 0 and abs(t["blank_share"] - 70 / 200) < 1e-9


def test_the_thinnest_width_of_an_outline():
    from shapely import Polygon
    assert abs(A.min_width_m(Polygon([(0, 0), (25, 0), (25, 4), (0, 4)])) - 4.0) < 0.3
