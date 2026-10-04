"""Product 1's measures, shared by the command line and the cloud stage.

Every Overture footprint's ground and LiDAR height (the roof measured more
than 1.5 m inside the outline, where a neighbour's wall cannot spill), and
the cut of an outline that holds two or three buildings side by side into
its parts, ``<gers>-a``, ``-b``, ``-c``. ``scripts/overture_boxes.py`` runs
them over an area from files; ``kyfromabove_3d.pieces`` runs them over one
LiDAR tile in memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .coordinates import METERS_PER_US_SURVEY_FOOT


class PointGrid:
    """Points sorted into square cells once, so the points inside a box are
    found by reading only the cells the box covers. Scanning every point of
    the tile for every building made the tile step grow as buildings x
    points: 7 minutes in Lambda for 1,121 buildings over 12 million returns."""

    def __init__(self, xy: np.ndarray, cell_m: float = 10.0):
        self.xy = np.asarray(xy)[:, :2]
        self.cell = float(cell_m)
        if not len(self.xy):
            self.nx = self.ny = 0
            return
        self.x0, self.y0 = float(self.xy[:, 0].min()), float(self.xy[:, 1].min())
        ix = ((self.xy[:, 0] - self.x0) // self.cell).astype(np.int64)
        iy = ((self.xy[:, 1] - self.y0) // self.cell).astype(np.int64)
        self.nx, self.ny = int(ix.max()) + 1, int(iy.max()) + 1
        key = iy * self.nx + ix
        self.order = np.argsort(key, kind="stable")
        self.starts = np.searchsorted(key[self.order], np.arange(self.nx * self.ny + 1))

    def box(self, minx: float, miny: float, maxx: float, maxy: float) -> np.ndarray:
        """Indices, in their original order, of the points strictly inside
        the box -- what ``(x > minx) & (x < maxx) & (y > miny) & (y < maxy)``
        selects, without looking at the rest."""
        if not self.nx:
            return np.zeros(0, dtype=np.int64)
        cx0 = max(int((minx - self.x0) // self.cell), 0)
        cx1 = min(int((maxx - self.x0) // self.cell), self.nx - 1)
        cy0 = max(int((miny - self.y0) // self.cell), 0)
        cy1 = min(int((maxy - self.y0) // self.cell), self.ny - 1)
        if cx0 > cx1 or cy0 > cy1:
            return np.zeros(0, dtype=np.int64)
        idx = np.concatenate([self.order[self.starts[r * self.nx + cx0]:self.starts[r * self.nx + cx1 + 1]]
                              for r in range(cy0, cy1 + 1)])
        p = self.xy[idx]
        keep = (p[:, 0] > minx) & (p[:, 0] < maxx) & (p[:, 1] > miny) & (p[:, 1] < maxy)
        return np.sort(idx[keep])


def rounded_wkt(poly, decimals: int = 7) -> str:
    """A polygon's WKT with its coordinates rounded: 7 decimals of a degree
    is about 1 cm, Overture's own precision -- the noise digits of a
    projection round trip made the same outline differ between machines.
    Falls back to 8 decimals, then to full precision, if rounding would make
    the polygon invalid."""
    import shapely

    for d in (decimals, decimals + 1):
        w = shapely.to_wkt(poly, rounding_precision=d, trim=True)
        if shapely.from_wkt(w).is_valid:
            return w
    return poly.wkt


def local_polygons_about(recs, ox: float, oy: float):
    """Every record's footprint (its largest ring) in local metres about
    ``ox, oy`` (EPSG:3089 US survey feet)."""
    from pyproj import Transformer
    from shapely.geometry import Polygon
    from shapely.wkt import loads as wkt_loads

    to_ft = Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True).transform
    polys = []
    for r in recs:
        g = wkt_loads(r["wkt"])
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda q: q.area)
        x, y = to_ft(*np.array(g.exterior.coords).T)
        polys.append(Polygon(np.column_stack([(np.asarray(x) - ox) * METERS_PER_US_SURVEY_FOOT,
                                              (np.asarray(y) - oy) * METERS_PER_US_SURVEY_FOOT])))
    return polys


def read_boxes(allb: np.ndarray, pad: float, lo: np.ndarray, hi: np.ndarray, cell: float = 150.0) -> list:
    """Disjoint boxes (local metres) covering every point within ``pad`` of a
    footprint's bounds: the ``cell``-metre squares any padded footprint
    touches, joined along each row where they touch. One box around all the
    footprints read 39 million returns for a rural Hazard tile's 24 houses
    (the tile's forest between them) and ran a 3 GB Lambda out of memory."""
    nx = max(int(np.ceil((hi[0] - lo[0]) / cell)), 1)
    ny = max(int(np.ceil((hi[1] - lo[1]) / cell)), 1)
    occ = np.zeros((ny, nx), bool)
    for x0, y0, x1, y1 in allb:
        i0, i1 = int((y0 - pad - lo[1]) // cell), int((y1 + pad - lo[1]) // cell)
        j0, j1 = int((x0 - pad - lo[0]) // cell), int((x1 + pad - lo[0]) // cell)
        occ[max(i0, 0):min(i1, ny - 1) + 1, max(j0, 0):min(j1, nx - 1) + 1] = True
    boxes = []
    for i in range(ny):
        j = 0
        while j < nx:
            if not occ[i, j]:
                j += 1
                continue
            k = j
            while k + 1 < nx and occ[i, k + 1]:
                k += 1
            boxes.append((lo[0] + j * cell, lo[1] + i * cell,
                          min(lo[0] + (k + 1) * cell, hi[0] + 1e-6), min(lo[1] + (i + 1) * cell, hi[1] + 1e-6)))
            j = k + 1
    return boxes


def area_cloud_local(polys, ox: float, oy: float, resolution_m=None, index=None, log=print):
    """The LiDAR over ``polys`` (local metres about ``ox, oy``, EPSG:3089 ft)
    plus 20 m, by COPC range read: (xyz local metres, classes), vegetation,
    noise and water left out. ``index`` is a LidarTileIndex (default: the
    grids in data/raw).

    Full resolution unless ``resolution_m`` is given: a COPC query's
    resolution caps how deep into the file's octree it reads. The old 0.5 m
    cap returned every point of Phase 2 downtown (it has no finer level) and
    identical heights and splits, but would silently thin denser LiDAR."""
    import laspy

    from kyfromabove_3d.lidar import LidarTileIndex, copc_query, copc_reader

    allb = np.array([p.bounds for p in polys])
    pad = 20.0
    lo = allb[:, :2].min(axis=0) - pad
    hi = allb[:, 2:].max(axis=0) + pad
    ft = METERS_PER_US_SURVEY_FOOT
    bounds_ft = (lo[0] / ft + ox, lo[1] / ft + oy, hi[0] / ft + ox, hi[1] / ft + oy)
    # the index holds one phase (sources_choice.indexes): the tile's chosen one
    tiles = (index or LidarTileIndex(Path("data/raw"), phases=("phase2",))).tiles_for_bounds(bounds_ft)
    boxes = read_boxes(allb, pad, lo, hi)
    log(f"{len(polys)} footprints over {hi[0] - lo[0]:.0f} x {hi[1] - lo[1]:.0f} m in {len(boxes)} box(es); "
        f"{len(tiles)} LiDAR tile(s)")
    xyz, cls = [], []
    for t in tiles:
        h = copc_reader(t.url).header
        for bx0, by0, bx1, by1 in boxes:
            box = laspy.copc.Bounds(mins=np.array([bx0 / ft + ox, by0 / ft + oy, h.z_min]),
                                    maxs=np.array([bx1 / ft + ox, by1 / ft + oy, h.z_max]))
            pts = copc_query(t.url, box, None if resolution_m is None else resolution_m / ft)
            c = np.asarray(pts.classification)
            x, y = (np.asarray(pts.x) - ox) * ft, (np.asarray(pts.y) - oy) * ft
            # drop noise (7), water (9) and vegetation (3-5): a tree over a low
            # roof would otherwise be the roof. And trim to the box on every
            # side: a COPC query can return a point a hair past its bounds
            # (1 in 22,031 on a downtown test), which the next box, sharing
            # that edge, returns too -- counted twice, it moved 11 buildings'
            # return counts by one or two
            keep = (~np.isin(c, (3, 4, 5, 7, 9)) & (x >= bx0) & (x < bx1) & (y >= by0) & (y < by1))
            xyz.append(np.column_stack([x[keep], y[keep], np.asarray(pts.z)[keep] * ft]))
            cls.append(c[keep])
    if not xyz:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.uint8)
    xyz = np.vstack(xyz); cls = np.concatenate(cls)
    log(f"{len(xyz):,} returns ({'full resolution' if resolution_m is None else f'{resolution_m} m resolution'}); "
        f"{int((cls == 2).sum()):,} ground")
    return xyz, cls


def roof_tiers(poly, above: np.ndarray, n_tiers: int = 3, cell_m: float = 1.0, min_area_m2: float = 40.0,
               min_share: float = 0.15, max_spread_m: float = 1.5, min_step_m: float = 3.0,
               max_line_m: float = 1.0, max_enclosed: float = 0.6):
    """Where one Overture outline holds ``n_tiers`` buildings side by side --
    39a33bd6 is three, stepping down 12, 8 and 4.5 m along its length: the
    roof cells (the top of the returns above the ground in each ``cell_m``
    cell) fall into as many heights -- tiers -- each flat to
    ``max_spread_m``, each at least ``min_area_m2`` and ``min_share`` of the
    roof and one patch, each ``min_step_m`` from the next, chosen as the
    natural breaks (the least summed spread); and where two tiers meet
    there is a seam (:func:`_fit_seam`): the wall between them.

    A tower *on* a podium is not this:
    the tallest tier enclosed by lower roof on more than ``max_enclosed`` of
    its boundary is left to the podium/tower rule.

    Returns ``None`` or ``{"tiers": [{"z", "share", "cells"}, ...]`` (lowest
    first; ``cells`` the (x, y) centres of the tier's roof cells) ``,
    "seams": [{"between": (i, j), "line": ((x0, y0), (x1, y1)), "rms_m"}]``
    in the cloud's frame.
    """
    from itertools import combinations
    from scipy.ndimage import binary_dilation, binary_erosion, label
    from scipy.stats import binned_statistic_2d
    from shapely import contains_xy

    if len(above) < 200:
        return None
    minx, miny, maxx, maxy = poly.bounds
    xe = np.arange(minx, maxx + cell_m, cell_m); ye = np.arange(miny, maxy + cell_m, cell_m)
    if len(xe) < 6 or len(ye) < 6:
        return None
    Z, _, _, _ = binned_statistic_2d(above[:, 0], above[:, 1], above[:, 2],
                                     statistic=lambda v: np.percentile(v, 80), bins=[xe, ye])
    Z = Z.T
    cx, cy = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(cx, cy)
    valid = np.isfinite(Z) & contains_xy(poly, X.ravel(), Y.ravel()).reshape(X.shape)
    h = Z[valid]
    if len(h) < 60:
        return None
    min_cells = max(min_area_m2 / (cell_m * cell_m), min_share * len(h))
    # thresholds every 0.25 m: one falls in the empty band between two roofs,
    # where a percentile of the heights would land inside a roof and hand its
    # top few cells to the next tier up, scattering the seam between them
    lo_h, hi_h = np.percentile(h, 1), np.percentile(h, 99)
    candidates = np.arange(np.floor(lo_h * 4) / 4 + 0.125, hi_h, 0.25)

    hs = np.sort(h)
    s1 = np.concatenate([[0.0], np.cumsum(hs)]); s2 = np.concatenate([[0.0], np.cumsum(hs * hs)])

    def tiers_for(cuts):
        """The tiers the thresholds ``cuts`` make -- (their summed squared
        spread, cuts): the natural breaks, smallest is best -- or None when
        one fails; means and spreads from running sums."""
        idx = [0, *np.searchsorted(hs, cuts).tolist(), len(hs)]
        means, sse = [], 0.0
        for a, b in zip(idx[:-1], idx[1:]):
            n = b - a
            if n < min_cells:
                return None
            m = (s1[b] - s1[a]) / n
            var = (s2[b] - s2[a]) / n - m * m
            if var >= max_spread_m ** 2:
                return None
            means.append(m); sse += n * var
        gap = min(b - a for a, b in zip(means[:-1], means[1:]))
        return (sse, cuts) if gap >= min_step_m else None

    k = n_tiers
    best = None
    for cuts in combinations(candidates, k - 1):
        got = tiers_for(list(cuts))
        if got is not None and (best is None or got[0] < best[0]):
            best = got
    if best is None:
        return None
    edges = [-np.inf, *best[1], np.inf]
    masks = [valid & (Z >= lo) & (Z < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    if any(not _one_patch(m, label) for m in masks):
        return None
    top = masks[-1]
    rim = top & ~binary_erosion(top)
    if rim.sum() == 0 or (top & binary_dilation(valid & ~top)).sum() / rim.sum() > max_enclosed:
        return None                                           # enclosed: a tower on a podium
    seams = []
    for i, j in combinations(range(k), 2):
        meet = masks[i] & binary_dilation(masks[j])
        if meet.sum() < 4:
            continue                                          # apart, or touching at a corner: no wall
        line = _fit_seam(np.column_stack([X[meet], Y[meet]]), poly, max_line_m)
        if line is None:
            return None
        seams.append({"between": (i, j), "line": line})
    if len(seams) < k - 1:
        return None
    return {"tiers": [{"z": float(np.median(Z[m])), "share": float(m.sum() / valid.sum()),
                       "cells": np.column_stack([X[m], Y[m]])} for m in masks],
            "seams": seams}


def _fit_seam(pts: np.ndarray, poly, max_line_m: float = 1.0,
              notch_m: float = 2.0, notch_len_m: float = 8.0, notch_deg: float = 15.0):
    """The wall between two tiers, from the cells where they meet: a line
    through ``pts`` along one of the outline's own axes where it fits (walls
    between buildings run with the block), placed
    at the median of the cells, and accepted when four in five cells lie
    within ``max_line_m`` of it -- a roof with few returns scatters a few
    boundary cells, which the median and the 80th percentile ride out; the
    outline's axes are tried before a free fit.
    Where a short step in the outline (under ``notch_len_m``, within
    ``notch_deg`` of the line and ``notch_m`` of it) marks the wall's foot,
    the line runs through that step instead: 39a33bd6's two seams run
    through its two notches. Returns ((x0, y0), (x1, y1)) far past the
    outline, or None."""
    if len(pts) < 4:
        return None
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    rect = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)
    axes = [e / max(np.linalg.norm(e), 1e-9) for e in (rect[1] - rect[0], rect[2] - rect[1])]

    def fit(d):
        n = np.array([-d[1], d[0]])
        off = float(np.median(pts @ n))
        return float(np.percentile(np.abs(pts @ n - off), 80)), d, n, off

    # the outline's two axes first -- one stray cell can swing a free fit
    # far off the wall (39a33bd6: seven cells on the seam, one 6 m away)
    best = min(fit(d) for d in axes)
    if best[0] > max_line_m:
        free = fit(vt[0])
        best = free if free[0] < best[0] else best
    spread, d, n, off = best
    if spread > max_line_m:
        return None
    centre = c + (off - float(c @ n)) * n
    ring = np.asarray(poly.exterior.coords)
    best = None
    for a, b in zip(ring[:-1], ring[1:]):
        e = b - a
        L = float(np.hypot(*e))
        if L < 0.5 or L > notch_len_m or abs(float(e @ d)) / L < np.cos(np.radians(notch_deg)):
            continue
        dist = abs(float((0.5 * (a + b)) @ n) - off)
        if dist <= notch_m and (best is None or dist < best[0]):
            best = (dist, 0.5 * (a + b), e / L)
    if best is not None:
        centre, d = best[1], best[2]
    minx, miny, maxx, maxy = poly.bounds
    reach = 2.0 * max(maxx - minx, maxy - miny)
    return (tuple((centre - reach * d).tolist()), tuple((centre + reach * d).tolist()))


def _one_patch(mask, label) -> bool:
    lab, n = label(mask)
    return n == 1 or np.bincount(lab.ravel())[1:].max() >= 0.9 * mask.sum()


def roof_step_split(poly, above: np.ndarray, cell_m: float = 1.0, min_share: float = 0.25, min_area_m2: float = 40.0,
                    max_spread_m: float = 1.5, min_step_m: float = 3.0, max_line_rms_m: float = 1.0,
                    max_enclosed: float = 0.6):
    """Where one Overture outline holds two buildings side by side: the roof
    cells (the top of the returns above the ground in each ``cell_m`` cell)
    fall into two heights, each holding at least ``min_share`` of the cells
    and ``min_area_m2``, each spread under ``max_spread_m``, ``min_step_m``
    apart, each one patch, and the cells where they meet lie on a straight
    line within ``max_line_rms_m`` -- the wall between the two buildings.

    A tower *on* a podium is not this: its high patch is enclosed by low
    cells on most of its boundary (more than ``max_enclosed``), and the
    podium/tower rule handles it. Here the high patch meets the low one
    along one side and the outline everywhere else.

    Returns ``None`` or ``{"line": ((x0, y0), (x1, y1)), "threshold_z",
    "step_m", "low_z", "high_z", "share_low"}`` in the cloud's frame.
    """
    from scipy.ndimage import binary_dilation, binary_erosion, label
    from scipy.stats import binned_statistic_2d
    from shapely import contains_xy

    if len(above) < 200:
        return None
    minx, miny, maxx, maxy = poly.bounds
    xe = np.arange(minx, maxx + cell_m, cell_m); ye = np.arange(miny, maxy + cell_m, cell_m)
    if len(xe) < 6 or len(ye) < 6:
        return None
    Z, _, _, _ = binned_statistic_2d(above[:, 0], above[:, 1], above[:, 2],
                                     statistic=lambda v: np.percentile(v, 80), bins=[xe, ye])
    Z = Z.T
    cx, cy = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(cx, cy)
    valid = np.isfinite(Z) & contains_xy(poly, X.ravel(), Y.ravel()).reshape(X.shape)
    h = Z[valid]
    if len(h) < 60:
        return None
    hs = np.sort(h); N = len(hs); best = None
    for k in range(int(min_share * N), int((1 - min_share) * N)):
        lo, hi = hs[:k], hs[k:]
        gap = hi.mean() - lo.mean()
        if lo.std() < max_spread_m and hi.std() < max_spread_m and gap >= min_step_m and (best is None or gap > best[0]):
            best = (gap, hs[k])
    if best is None:
        return None
    gap, thr = best
    high = valid & (Z >= thr); low = valid & (Z < thr)
    if min(high.sum(), low.sum()) * cell_m * cell_m < min_area_m2:
        return None
    for m in (high, low):
        lab, n = label(m)
        if n != 1 and np.bincount(lab.ravel())[1:].max() < 0.9 * m.sum():
            return None
    ring = high & ~binary_erosion(high)                       # the high patch's boundary cells
    meets_low = high & binary_dilation(low)
    if ring.sum() == 0 or meets_low.sum() / ring.sum() > max_enclosed:
        return None                                           # enclosed: a tower on a podium
    pts = np.column_stack([X[meets_low], Y[meets_low]])
    if len(pts) < 4:
        return None
    c = pts.mean(axis=0)
    _, sv, vt = np.linalg.svd(pts - c)
    rms = float(np.sqrt(((pts - c) @ vt[1]) ** 2).mean()) if len(pts) > 2 else 0.0
    if rms > max_line_rms_m:
        return None
    d = vt[0]
    reach = 2.0 * max(maxx - minx, maxy - miny)
    line = (tuple((c - reach * d).tolist()), tuple((c + reach * d).tolist()))
    return {"line": line, "threshold_z": float(thr),
            "step_m": float(gap), "low_z": float(np.median(Z[low])), "high_z": float(np.median(Z[high])),
            "share_low": float(low.sum() / valid.sum()), "line_rms_m": rms,
            # the same answer in roof_tiers' form, for cut_into_tiers
            "tiers": [{"z": float(np.median(Z[m])), "share": float(m.sum() / valid.sum()),
                       "cells": np.column_stack([X[m], Y[m]])} for m in (low, high)],
            "seams": [{"between": (0, 1), "line": line}]}


def min_corner_deg(poly, merge_m: float = 0.5) -> float:
    """The sharpest corner of ``poly``'s outline, in degrees, after merging
    vertices closer than ``merge_m`` -- a 13 cm jog at the end of a wedge is
    the wedge's tip, not two right angles."""
    pts = [np.asarray(p, dtype=float) for p in poly.exterior.coords[:-1]]
    keep = [pts[0]]
    for q in pts[1:]:
        if np.hypot(*(q - keep[-1])) >= merge_m:
            keep.append(q)
    if len(keep) > 1 and np.hypot(*(keep[0] - keep[-1])) < merge_m:
        keep.pop()
    if len(keep) < 3:
        return 0.0
    angs = []
    for i in range(len(keep)):
        u, v = keep[i - 1] - keep[i], keep[(i + 1) % len(keep)] - keep[i]
        c = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
        angs.append(float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
    return min(angs)


def cut_into_tiers(poly, found: dict, min_corner_deg_limit: float = 30.0, min_area_m2: float = 40.0):
    """Cut ``poly`` into the tiers :func:`roof_tiers` found: split it with
    every seam, then give each piece to the tier whose roof cells it holds
    (a seam extended across a third tier cuts it in two, and the halves go
    back together). Returns ``(parts, None)`` -- ``[(tier index, polygon)]``,
    the tallest first -- or ``([], reason)`` when a tier falls apart into
    separate pieces, a part is under ``min_area_m2``, or a part has a corner
    sharper than ``min_corner_deg_limit``: a wall between two buildings
    leaves square-ish parts, and a sliver means the cut is wrong (39a33bd6
    steps down in three tiers along its length; one fitted line ran along
    it and left an 11 degree wedge)."""
    from shapely import contains_xy
    from shapely.geometry import LineString
    from shapely.ops import split as shp_split, unary_union

    pieces = [poly]
    for seam in found["seams"]:
        pieces = [q for piece in pieces for q in shp_split(piece, LineString(seam["line"])).geoms]
    tiers = found["tiers"]
    cells = np.vstack([t["cells"] for t in tiers])
    cell_tier = np.concatenate([np.full(len(t["cells"]), i) for i, t in enumerate(tiers)])
    by_tier = {i: [] for i in range(len(tiers))}
    for q in pieces:
        inq = contains_xy(q, cells[:, 0], cells[:, 1])
        if inq.any():
            by_tier[int(np.bincount(cell_tier[inq]).argmax())].append(q)
        else:                                                  # no roof cell in it: the nearest tier's
            c = np.asarray(q.representative_point().coords[0])
            by_tier[int(cell_tier[np.argmin(np.hypot(*(cells - c).T))])].append(q)
    parts = []
    for i in range(len(tiers) - 1, -1, -1):
        g = unary_union(by_tier[i]) if by_tier[i] else None
        if g is not None and g.geom_type == "Polygon" and min_corner_deg(g) < min_corner_deg_limit:
            # a seam laid along a step of the outline runs on past it and back:
            # a zero-width spike. In 1 cm and back out, square-cornered, and the
            # part is its real shape; a part with no sharp corner is left as cut
            g = g.buffer(-0.01, join_style=2).buffer(0.01, join_style=2)
            if g.geom_type == "MultiPolygon":
                g = max(g.geoms, key=lambda q: q.area)
        if g is None or g.geom_type != "Polygon":
            return [], "a tier falls apart into separate pieces"
        parts.append((i, g))
    if min(g.area for _, g in parts) < min_area_m2:
        return [], f"a part under {min_area_m2:.0f} m2"
    sharp = min(min_corner_deg(g) for _, g in parts)
    if sharp < min_corner_deg_limit:
        return [], f"the cut would leave a {sharp:.0f} degree corner"
    return parts, None


def split_records(recs, polys, xyz: np.ndarray, ox: float, oy: float, frame_label: str,
                  min_returns: int = 20, min_corner_deg_limit: float = 30.0, log=print):
    """The records with each footprint that holds two or three buildings
    replaced by its parts (see :func:`split`), and the list of splits.
    ``polys`` are the footprints in local metres about ``ox, oy`` (EPSG:3089
    ft), ``xyz`` the area's LiDAR in the same frame; each record needs its
    ``lidar_ground_z`` (:func:`measure_heights`)."""
    from pyproj import Transformer
    from shapely import contains_xy
    from shapely.geometry import Polygon

    ft = METERS_PER_US_SURVEY_FOOT
    to_wgs = Transformer.from_crs("EPSG:3089", "EPSG:4326", always_xy=True).transform

    grid = PointGrid(xyz)

    def wkt_of(poly_m):
        # canonical: a corner within 1 cm of the straight line through its
        # neighbours is no corner -- such a point, 1 mm off an edge, came and
        # went between Linux and macOS (their maths libraries differ in the
        # last bit), even with every Python library pinned alike
        poly_m = poly_m.simplify(0.01, preserve_topology=True)
        x, y = np.array(poly_m.exterior.coords).T
        lon, lat = to_wgs(x / ft + ox, y / ft + oy)
        return rounded_wkt(Polygon(np.column_stack([lon, lat])))

    def points_in(poly_m):
        pts = xyz[grid.box(*poly_m.bounds)]
        return pts[contains_xy(poly_m, pts[:, 0], pts[:, 1])]

    def measure(poly_m, gz):
        inside = points_in(poly_m)
        above = inside[inside[:, 2] > gz + 2.0]
        if len(above) < min_returns:
            return {"lidar_height_m": 0.0, "lidar_median_m": 0.0, "lidar_n": int(len(inside))}
        roof = roof_returns(poly_m, above, min_returns=min_returns)
        return {"lidar_height_m": round(float(np.percentile(roof[:, 2], 99)) - gz, 2),
                "lidar_median_m": round(float(np.median(roof[:, 2])) - gz, 2), "lidar_n": int(len(above))}

    out_recs, splits = [], []
    for r, p in zip(recs, polys):
        gz = r.get("lidar_ground_z")
        if gz is None or (r.get("lidar_n") or 0) < 200:
            out_recs.append(r); continue
        inside = points_in(p)
        above = inside[inside[:, 2] > gz + 2.0]
        # three buildings in a row when the roof says so and the cuts come out
        # clean; otherwise the two-building rule, as before
        found = roof_tiers(p, above, n_tiers=3)
        parts, why = cut_into_tiers(p, found, min_corner_deg_limit) if found else ([], "no three tiers")
        if why is not None:
            found = roof_step_split(p, above)
            if found is None:
                out_recs.append(r); continue
            parts, why = cut_into_tiers(p, found, min_corner_deg_limit)
        tiers = found["tiers"]
        heights = " / ".join(f"{t['z'] - gz:.1f}" for t in tiers[::-1])
        if why is not None:
            log(f"  {r['id'][:8]} {r.get('name') or '':<32.32} {len(tiers)} roof heights ({heights} m): refused, {why}")
            out_recs.append(r); continue
        for letter, (i, g) in zip("abc", parts):
            rec = dict(r)
            rec.update({"id": f"{r['id']}-{letter}", "parent_gers": r["id"], "wkt": wkt_of(g),
                        "name": f"{r['name']} ({letter})" if r.get("name") else None,
                        "split": {"part": letter, "of": len(parts),
                                  "tier_heights_m": [round(t["z"] - gz, 1) for t in tiers[::-1]],
                                  "seams_local_m": [[[round(v, 2) for v in pt] for pt in sm["line"]]
                                                    for sm in found["seams"]],
                                  "frame": frame_label}})
            rec.update(measure(g, gz))
            out_recs.append(rec)
        splits.append({"gers": r["id"], "name": r.get("name"), "heights_m": [round(t["z"] - gz, 1) for t in tiers[::-1]],
                       "areas_m2": [int(g.area) for _, g in parts]})
        log(f"  {r['id'][:8]} {r.get('name') or '':<32.32} {len(tiers)} roof heights ({heights} m) "
            f"-> {len(parts)} parts of {', '.join(str(int(g.area)) for _, g in parts)} m2")
    return out_recs, splits


def roof_returns(poly, above: np.ndarray, edge_m: float = 1.5, min_returns: int = 20) -> np.ndarray:
    """The returns that speak for a footprint's roof: those more than
    ``edge_m`` inside its outline, when there are ``min_returns`` of them;
    all of ``above`` otherwise (a footprint too small for an inside). An
    Overture outline overlaps its neighbour's wall by a few decimetres, and a
    tall neighbour's facade returns land along the edge: 9947f2b0, an 18 m
    building beside 400 West Market, measured 115 m on its whole footprint
    and 17.7 m inside it; 35 downtown footprints drop by more than 3 m."""
    from shapely import contains_xy
    inner = poly.buffer(-edge_m)
    if inner.is_empty or not len(above):
        return above
    ins = above[contains_xy(inner, above[:, 0], above[:, 1])]
    return ins if len(ins) >= min_returns else above


def measure_heights(recs, polys, xyz: np.ndarray, cls: np.ndarray, min_returns: int = 20) -> int:
    """Each record's ground and LiDAR height (see :func:`lidar_heights`),
    written onto the record; ``polys`` and ``xyz`` in one local frame.
    Returns how many were measured."""
    from shapely import contains_xy

    ground = xyz[cls == 2]
    all_grid, ground_grid = PointGrid(xyz), PointGrid(ground)
    n_ok = 0
    for r, p in zip(recs, polys):
        minx, miny, maxx, maxy = p.bounds
        near = ground[ground_grid.box(minx - 15, miny - 15, maxx + 15, maxy + 15)]
        gz = float(np.median(near[:, 2])) if len(near) >= 5 else None
        cand = xyz[all_grid.box(minx, miny, maxx, maxy)]
        inside = cand[contains_xy(p, cand[:, 0], cand[:, 1])]
        r["lidar_ground_z"] = gz
        if gz is None or len(inside) < min_returns:
            r["lidar_height_m"] = None; r["lidar_n"] = int(len(inside)); continue
        above = inside[inside[:, 2] > gz + 2.0]
        if len(above) < min_returns:
            r["lidar_height_m"] = 0.0; r["lidar_n"] = int(len(inside)); r["lidar_median_m"] = 0.0
            n_ok += 1
            continue
        roof = roof_returns(p, above, min_returns=min_returns)
        r["lidar_height_m"] = round(float(np.percentile(roof[:, 2], 99)) - gz, 2)
        r["lidar_median_m"] = round(float(np.median(roof[:, 2])) - gz, 2)
        r["lidar_n"] = int(len(above))
        n_ok += 1
    return n_ok
