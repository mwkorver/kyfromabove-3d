"""A building's levels: the outlines its roof heights step through."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

# --------------------------------------------------------------- levels
#
# A footprint from Overture is the ground floor's outline. A tower set on a
# podium -- RIVUE Tower, Humana, the Galt House -- is two bodies with two
# outlines, and only the lower one is drawn. The LiDAR knows the other: the
# roof returns fall on two levels, and the upper one has a shape of its own.

LEVEL_FACE_STRIDE = 1000     # faces of level k are numbered k * 1000 + i


def building_levels(xyz: np.ndarray, ring_m: np.ndarray, ground_z: float, cell_m: float = 1.0,
                    upper_fraction: float = 0.4, min_cover: float = 0.05, max_cover: float = 0.7,
                    min_step_m: float = 12.0, min_area_m2: float = 150.0,
                    max_vertices: int = 16, from_tiers: bool = False,
                    min_tier_share: float = 0.10, below_main: bool = False, straighten: bool = False,
                    fine_cell_m: float = 0.5, smooth_m: float = 1.0) -> List[Dict[str, Any]]:
    """The bodies a footprint holds, from the roof returns inside it.

    A height field of the returns inside the footprint (the top of each
    ``cell_m`` cell). When the cells at or above ``upper_fraction`` of the
    top height cover between ``min_cover`` and ``max_cover`` of the
    footprint and stand at least ``min_step_m`` above the median of the
    rest, the upper cells are a second body: each connected patch of them
    over ``min_area_m2`` is traced to a polygon, smoothed by half a cell
    and simplified to at most ``max_vertices`` corners (a rotated
    rectangle when the trace will not simplify). The podium's top is the
    median height of the cells outside the towers.

    Returns a list of levels ``{"level", "ring" (local metres, closed),
    "z_lo", "z_hi"}``: one entry (the footprint, ground to top) when the
    roof is one level; the footprint to the podium top and one entry per
    tower above it otherwise. Level ``k``'s faces are numbered from
    ``k * LEVEL_FACE_STRIDE`` so every face of the building keeps a unique
    index across the profiles.

    With ``from_tiers`` the upper body is what stands above the midpoint
    between the roof's two highest tiers (flat levels holding
    ``min_tier_share`` of the roof each, ``min_step_m`` or more apart),
    not above ``upper_fraction`` of the top: a 25 m building with roofs at 17
    and 25 m has both above 40% of its top, and was one level. The settings
    made for towers (a 12 m step, 150 m2) are a class's to change
    (configs/class_strategies.yaml).

    With ``straighten`` an upper body's outline is traced as the LiDAR
    towers are (scripts/lidar_outline.py): on a ``fine_cell_m`` grid of the
    highest returns, closed and opened by ``smooth_m``, straightened to the
    footprint's directions (kyfromabove_3d.outlines.regularize) and clipped
    to the footprint -- the 1 m trace cut to 16 corners wobbled, and on
    e5214ab7 ran onto the neighbour.
    """
    from scipy.ndimage import label
    from scipy.stats import binned_statistic_2d
    from shapely import contains_xy
    from shapely.geometry import Polygon

    ring = np.asarray(ring_m, dtype=np.float64)[:, :2]
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[:1]])
    poly = Polygon(ring)
    top = float(np.percentile(xyz[:, 2], 99)) if len(xyz) else ground_z
    whole = [{"level": 0, "ring": ring, "z_lo": float(ground_z), "z_hi": top}]
    inside = contains_xy(poly, xyz[:, 0], xyz[:, 1]) & (xyz[:, 2] > ground_z + 2.0)
    p = xyz[inside]
    minx, miny, maxx, maxy = poly.bounds
    xe = np.arange(minx, maxx + cell_m, cell_m)
    ye = np.arange(miny, maxy + cell_m, cell_m)
    if len(p) < 50 or len(xe) < 4 or len(ye) < 4:
        return whole
    Z, _, _, _ = binned_statistic_2d(p[:, 0], p[:, 1], p[:, 2],
                                     statistic=lambda v: np.percentile(v, 80), bins=[xe, ye])
    Z = Z.T                                                      # [iy, ix]
    cx, cy = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(cx, cy)
    infp = contains_xy(poly, X.ravel(), Y.ravel()).reshape(X.shape)
    valid = np.isfinite(Z) & infp
    if valid.sum() < 20:
        return whole
    h = Z - ground_z
    top_h = float(np.percentile(h[valid], 99))
    upper = valid & (h >= upper_fraction * top_h)
    split_h = upper_fraction * top_h
    if below_main:
        # the upper body is everything near the roof's main height: lower wings
        # spread over several heights (the Galt House's at 12-24 m beside its
        # 50 m block) make no tier of their own in 1 m steps, and the upper body
        # may cover most of the footprint (max_cover)
        hv = h[valid]
        hist, edges = np.histogram(hv, bins=np.arange(np.floor(hv.min()), np.ceil(hv.max()) + 1.01, 1.0))
        main_h = float(edges[int(np.argmax(hist))] + 0.5)
        upper = valid & (h >= main_h - max(min_step_m, 0.2 * top_h))
    elif from_tiers:
        hv = h[valid]
        hist, edges = np.histogram(hv, bins=np.arange(np.floor(hv.min()), np.ceil(hv.max()) + 1.01, 1.0))
        share = hist / max(hist.sum(), 1)
        tiers = []
        for k in np.argsort(-share):
            if share[k] < min_tier_share:
                break
            lvl = float(edges[k] + 0.5)
            if all(abs(lvl - t) >= min_step_m for t in tiers):
                tiers.append(lvl)
        if len(tiers) < 2:
            return whole
        tiers.sort()
        split_h = 0.5 * (tiers[-1] + tiers[-2])
        upper = valid & (h >= split_h)
    cover = upper.sum() / max(valid.sum(), 1)
    # Returns from a taller neighbour's wall spill a metre or two over the
    # footprint's edge and read as a sliver of "upper" roof; the building's
    # top is then its own lower roof. A spire or a small tower stands
    # inside the footprint and is kept: only upper cells that lie along the
    # edge are taken for spill.
    def capped_at_lower() -> Optional[List[Dict[str, Any]]]:
        lower_cells = valid & ~upper
        if lower_cells.sum() < 10 or not _along_edge(upper, X, Y, poly):
            return None
        return [{"level": 0, "ring": ring, "z_lo": float(ground_z),
                 "z_hi": float(ground_z + np.percentile(h[lower_cells], 99))}]

    if not (min_cover <= cover <= max_cover):
        return (capped_at_lower() or whole) if cover < min_cover else whole
    lower = valid & ~upper
    if lower.sum() < 10:
        return whole
    podium_top = float(np.median(Z[lower]))
    if float(np.median(Z[upper])) - podium_top < min_step_m:
        return whole
    lab, n = label(upper)
    towers = []
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() * cell_m * cell_m < min_area_m2:
            continue
        towers.append((m, float(np.percentile(Z[m], 95))))
    if not towers:
        return capped_at_lower() or whole
    levels = [{"level": 0, "ring": ring, "z_lo": float(ground_z), "z_hi": podium_top}]
    for j, (m, z_top) in enumerate(sorted(towers, key=lambda t: -t[0].sum()), start=1):
        tower_ring = _straight_upper_ring(p, poly, ground_z + split_h, m, cx, cy, cell_m, fine_cell_m, smooth_m) \
            if straighten else None
        if tower_ring is None:
            tower_ring = _mask_to_ring(m, cx, cy, cell_m, max_vertices)
        if tower_ring is None:
            continue
        levels.append({"level": j, "ring": tower_ring, "z_lo": podium_top, "z_hi": z_top})
    return levels if len(levels) > 1 else whole


def _straight_upper_ring(p: np.ndarray, poly, z_min: float, patch: np.ndarray, cx: np.ndarray, cy: np.ndarray,
                         cell_m: float, fine_cell_m: float, smooth_m: float) -> Optional[np.ndarray]:
    """An upper body's outline traced finely and straightened: the cells of a
    ``fine_cell_m`` grid of the highest returns ``p`` (inside the footprint)
    at ``z_min`` or above, closed and opened by ``smooth_m``, the piece most
    overlapping the coarse ``patch`` (on the ``cell_m`` grid at ``cx``,
    ``cy``) traced, straightened to the footprint's directions and clipped to
    the footprint. None when it cannot be made."""
    import rasterio.features
    from affine import Affine
    from scipy.ndimage import binary_closing, binary_opening, label
    from shapely.geometry import Polygon, shape

    from ..outlines import main_directions, regularize

    minx, miny, maxx, maxy = poly.bounds
    nx, ny = int(np.ceil((maxx - minx) / fine_cell_m)) + 1, int(np.ceil((maxy - miny) / fine_cell_m)) + 1
    ix = np.clip(((p[:, 0] - minx) / fine_cell_m).astype(int), 0, nx - 1)
    iy = np.clip(((p[:, 1] - miny) / fine_cell_m).astype(int), 0, ny - 1)
    g = np.full((ny, nx), -np.inf)
    np.maximum.at(g, (iy, ix), p[:, 2])
    mask = g >= z_min
    k = max(1, int(round(smooth_m / fine_cell_m)))
    st = np.ones((2 * k + 1, 2 * k + 1), dtype=bool)
    mask = binary_opening(binary_closing(mask, st), st)
    lab, n = label(mask)
    if n == 0:
        return None
    # the piece the coarse patch is: count its cells' centres in each piece
    py, px = np.nonzero(patch)
    fx = np.clip(((cx[px] - minx) / fine_cell_m).astype(int), 0, nx - 1)
    fy = np.clip(((cy[py] - miny) / fine_cell_m).astype(int), 0, ny - 1)
    votes = np.bincount(lab[fy, fx], minlength=n + 1)
    votes[0] = 0
    if votes.max() == 0:
        return None
    piece = lab == int(np.argmax(votes))
    tr = Affine(fine_cell_m, 0, minx, 0, fine_cell_m, miny)
    polys = [shape(geom) for geom, v in rasterio.features.shapes(piece.astype(np.uint8), mask=piece, transform=tr) if v == 1]
    if not polys:
        return None
    traced = max(polys, key=lambda q: q.area)
    ring = regularize(traced, main_directions(np.asarray(poly.exterior.coords)))
    if ring is None:
        return None
    clipped = Polygon(ring).buffer(0).intersection(poly)
    if clipped.geom_type == "MultiPolygon":
        clipped = max(clipped.geoms, key=lambda q: q.area)
    if clipped.geom_type != "Polygon" or clipped.area < 0.5 * traced.area:
        return None
    return np.asarray(clipped.exterior.coords)


def _along_edge(mask: np.ndarray, X: np.ndarray, Y: np.ndarray, poly, within_m: float = 2.0,
                min_share: float = 0.8) -> bool:
    """Whether at least ``min_share`` of the cells in ``mask`` lie within
    ``within_m`` of the footprint's outline."""
    import shapely
    if not mask.any():
        return False
    d = shapely.distance(shapely.points(X[mask], Y[mask]), poly.exterior)
    return float((d <= within_m).mean()) >= min_share


def _mask_to_ring(mask: np.ndarray, cx: np.ndarray, cy: np.ndarray, cell_m: float,
                  max_vertices: int) -> Optional[np.ndarray]:
    """A raster patch -> a simple polygon in local metres."""
    from shapely.geometry import Polygon, shape
    from shapely.ops import unary_union

    try:
        from rasterio import features as rfeatures
        from rasterio.transform import from_origin
        # rows run north-down in a raster; our rows run south-up, so flip
        transform = from_origin(cx[0] - cell_m / 2, cy[-1] + cell_m / 2, cell_m, cell_m)
        arr = mask[::-1].astype(np.uint8)
        geoms = [shape(g) for g, v in rfeatures.shapes(arr, mask=arr.astype(bool), transform=transform) if v == 1]
        g = unary_union(geoms)
    except Exception:                                            # noqa: BLE001
        g = None
    if g is None or g.is_empty:
        return None
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda q: q.area)
    g = Polygon(g.exterior.coords).buffer(cell_m / 2).buffer(-cell_m / 2)
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda q: q.area)
    if g.is_empty or g.geom_type != "Polygon":
        return None
    simple = None
    for tol in [0.8 * cell_m, 1.2 * cell_m, 1.6 * cell_m, 2.0 * cell_m, 2.5 * cell_m, 3.0 * cell_m]:
        cand = g.simplify(tol, preserve_topology=True)
        if not cand.is_empty and cand.geom_type == "Polygon" and 3 <= len(cand.exterior.coords) - 1 <= max_vertices:
            simple = cand
            break
    if simple is None:
        simple = g.minimum_rotated_rectangle
    ring = np.array(simple.exterior.coords, dtype=np.float64)
    return ring if len(ring) >= 4 else None
