"""Three shapes that stopped the trial's builds (trial1, 10 of 26,709
buildings), each in miniature: a roof with no cell of two returns, a roof
whose cells with returns lie on one line, and a wall shared with two
neighbours along the same stretch, one of unknown height."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, box

from kyfromabove_3d.mesh.party import party_edges
from kyfromabove_3d.mesh.roof import roof_heightfield, tidy_roof_field


def test_a_roof_with_no_cell_of_two_returns_has_no_valid_cells():
    """37393d1b: every cell held one return; its "valid" cells were NaN."""
    poly = box(0, 0, 6, 6)
    xs = np.arange(0.375, 6, 0.75)
    pts = np.array([(x, y, 5.0) for x in xs[::2] for y in xs[::2]])         # one return in every other cell
    _, _, Z, valid = roof_heightfield(pts, poly, ground_z=0.0)
    assert np.isfinite(Z[valid]).all()
    assert not valid.any()                                                  # no roof from the returns


def test_a_roof_whose_cells_with_returns_lie_on_one_line_is_tidied_without_a_triangulation():
    """62e822b0, b6c453f9: the cells to interpolate between were collinear.
    The roof is level at a whole metre, which also left the level histogram
    with no bins."""
    poly = box(0, 0, 6, 6)
    xs = ys = np.arange(0.375, 6, 0.75)
    Z = np.full((len(ys), len(xs)), 5.0)
    valid = np.ones_like(Z, bool)
    row = [(x + dx, ys[3], 5.0) for x in xs for dx in (-0.1, 0.1)]         # two returns a cell, one row only
    out = tidy_roof_field(np.array(row), poly, 0.0, xs, ys, Z, valid)
    assert np.isfinite(out[valid]).all()


class _Face:
    face = 7

    def world_line(self):
        return [(0.0, 0.0), (10.0, 0.0)]

    def local(self, pts):
        return pts[:, 0], pts[:, 1]


def test_a_wall_shared_with_two_neighbours_on_one_stretch_one_of_unknown_height():
    """ce8b50e8: (0, 10, None) and (0, 10, 6.0) could not be sorted."""
    nb = Polygon([(0, -5), (10, -5), (10, 0), (0, 0)])
    out = party_edges(np.zeros((4, 2)), [nb, nb], [_Face()], tops=[6.0, None])
    assert [t for _, _, t in out[7]] == [None, 6.0]
