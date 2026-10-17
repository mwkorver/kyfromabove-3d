"""The roof of the mesh: the height field from the LiDAR, tidied, cut along the levels'
outlines, and its texture from the frames."""

from __future__ import annotations

from typing import Optional

import numpy as np

from kyfromabove_3d.features import (
    WALL_STATUSES,
)

from .textures import MAX_TEX_PX, TEXEL_M, _fill_holes, _mult4

ROOF_CELL_M = 0.75        # roof height-field cell


# ---------------------------------------------------------------------------- roof

def roof_heightfield(xyz: np.ndarray, poly, ground_z: float, cell: float = ROOF_CELL_M,
                     max_z: Optional[float] = None, edge_m: float = 2.0):
    """Top height of the returns in each cell inside the footprint.

    Returns (xs, ys, Z, valid): cell-centre coordinates, heights, and which
    cells hold a value (their own returns, or the nearest cell's within 3 m).
    """
    from scipy.ndimage import distance_transform_edt
    from scipy.stats import binned_statistic_2d
    from shapely import contains_xy

    inside = contains_xy(poly.buffer(0.3), xyz[:, 0], xyz[:, 1]) & (xyz[:, 2] > ground_z + 2.0)
    if max_z is not None:
        # a taller neighbour's wall spills a metre or two over the outline;
        # above max_z only that strip is dropped. Anything standing on the
        # roof inside it -- a penthouse, a lantern, a water tank -- stays: the
        # Kentucky Home Life Building's 20 m2 rooftop structure to 94 m was
        # cut at 77 m and left as a ring of vertical panels
        import shapely
        high = inside & (xyz[:, 2] > max_z)
        if high.any():
            near_edge = shapely.distance(shapely.points(xyz[high, 0], xyz[high, 1]), poly.exterior) <= edge_m
            drop = np.flatnonzero(high)[near_edge]
            inside[drop] = False
    p = xyz[inside]
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx + cell / 2, maxx, cell)
    ys = np.arange(miny + cell / 2, maxy, cell)
    if len(xs) < 2 or len(ys) < 2 or len(p) < 3:
        # nothing above the ground here (a low structure, a footprint the
        # scanner missed): no roof from the returns
        xs = xs if len(xs) >= 2 else np.array([minx + cell / 2, minx + 1.5 * cell])
        ys = ys if len(ys) >= 2 else np.array([miny + cell / 2, miny + 1.5 * cell])
        Z = np.full((len(ys), len(xs)), np.nan)
        return xs, ys, Z, np.zeros(Z.shape, dtype=bool)
    xe = np.concatenate([xs - cell / 2, [xs[-1] + cell / 2]])
    ye = np.concatenate([ys - cell / 2, [ys[-1] + cell / 2]])
    # The top surface of each cell, not its median: a cell under an overhang
    # holds returns from the roof above and the roof below (Humana's top
    # floors stand out over the Main Street podium), and the median hangs a
    # curtain between the two. What the nadir sees is the upper one.
    med, _, _, _ = binned_statistic_2d(p[:, 0], p[:, 1], p[:, 2],
                                       statistic=lambda v: np.percentile(v, 80), bins=[xe, ye])
    cnt, _, _, _ = binned_statistic_2d(p[:, 0], p[:, 1], None, statistic="count", bins=[xe, ye])
    Z = med.T                                       # [iy, ix]
    have = (cnt.T >= 2) & np.isfinite(Z)
    X, Y = np.meshgrid(xs, ys)
    infp = contains_xy(poly, X.ravel(), Y.ravel()).reshape(X.shape)
    dist, (iy, ix) = distance_transform_edt(~have, return_indices=True)
    Zf = Z[iy, ix]
    valid = infp & (have | (dist * cell <= 3.0))
    Zf[~valid] = np.nan
    # A 3x3 median where at least five neighbours are valid: a one-cell
    # spike (a lattice dome's rib, a mast) is not a wall, and left alone it
    # became a fringe of small vertical triangles round every crown.
    from scipy.ndimage import generic_filter
    def _nanmedian(v):
        v = v[np.isfinite(v)]
        return np.median(v) if len(v) >= 5 else np.nan
    Zs = generic_filter(Zf, _nanmedian, size=3, mode="constant", cval=np.nan)
    smooth = valid & np.isfinite(Zs)
    Zf[smooth] = Zs[smooth]
    return xs, ys, Zf, valid


def _lower_rooftop_boxes(Z, valid, cell, level, max_box_m2=40.0, max_box_h_m=2.5,
                         min_h_m=0.9, ring_m=0.8) -> np.ndarray:
    """Small boxes on a flat roof lowered to it: a patch of cells standing
    ``min_h_m`` to ``max_box_h_m`` above the roof's main ``level``, up to
    ``max_box_m2``, whose neighbouring cells are near that level (their
    median within ``ring_m``), is set to the level. Its skirt, the cells the
    field's smoothing left half-way up, is then near enough for the level
    snap. Below ``min_h_m`` the skirts join box to box and to anything else
    raised. A chimney or a lift room taller than ``max_box_h_m`` stays, with
    the skirt round it, and so does a raised part bigger than ``max_box_m2``
    (2b18f52d's 22.7 m strip along its north side)."""
    from scipy.ndimage import binary_dilation, label
    cand = valid & (Z >= level + min_h_m) & (Z <= level + max_box_h_m)
    tall = valid & (Z > level + max_box_h_m)
    # not at the roof's edge: a stretch of parapet is no box
    rim = valid & binary_dilation(~valid)
    Z = Z.copy()
    regions, n = label(cand)
    for r in range(1, n + 1):
        m = regions == r
        if m.sum() * cell * cell > max_box_m2 or (m & rim).any():
            continue
        near = binary_dilation(m)
        ring = near & ~m & valid
        if (near & tall).any() or not ring.any() or abs(float(np.median(Z[ring])) - level) > ring_m:
            continue
        Z[m] = level
    return Z


def tidy_roof_field(xyz: np.ndarray, poly, ground_z: float, xs, ys, Z, valid, level_bin_m: float = 0.25,
                    min_level_share: float = 0.08, level_sep_m: float = 1.0, snap_m: float = 0.8,
                    min_region_m2: float = 4.0, max_flat_deg: float = 15.0,
                    max_box_m2: float = 40.0, max_box_h_m: float = 2.5, max_ramp_deg: float = 60.0) -> np.ndarray:
    """The mesh's roof field made calm where the returns are sparse.

    1. A cell with no returns of its own is filled by linear interpolation
       between the cells that have some, not copied from the nearest one: a
       slope the laser saw little of (2b18f52d's west side) was a flat shelf
       at the eave's height and then a jump to the top; it is a ramp --
       where it is no steeper than ``max_ramp_deg``.
    2. A mostly flat roof is snapped to its levels: heights that hold
       ``min_level_share`` of the roof each, ``level_sep_m`` apart; a cell
       within ``snap_m`` of one takes it (by the majority of its 3 x 3
       neighbours, so cells do not flicker), in patches of ``min_region_m2``
       or more, where the roof around it slopes under ``max_flat_deg``, and
       not on the roof's outermost cells, which hold its parapet. On
       a 1.5 m grid over 3 returns per square metre, a top seen at 20.2 and
       21.7 m jumped between the two from cell to cell, and the
       roof's picture smeared over the waves. A pitched roof has no levels
       and is left alone.

    Between the two, a box on a flat roof -- an air-handling unit, a hatch,
    under ``max_box_m2`` and ``max_box_h_m`` tall on the main level --
    is lowered to it (_lower_rooftop_boxes): on a 1.5 m grid
    each of 2b18f52d's seven 3 m units was a pyramid, and the slopes round
    them were the lumps. The frames still paint them on the roof.

    Only the mesh's copy: the audit's LiDAR solid is built from the field as
    roof_heightfield returns it. Returns the new heights."""
    from scipy.interpolate import griddata
    from scipy.ndimage import generic_filter, label
    from shapely import contains_xy

    Z = np.array(Z, dtype=float, copy=True)
    if valid.sum() < 8:
        return Z
    cell = float(xs[1] - xs[0])
    inside = contains_xy(poly.buffer(0.3), xyz[:, 0], xyz[:, 1]) & (xyz[:, 2] > ground_z + 2.0)
    p = xyz[inside]
    ix = np.floor((p[:, 0] - (xs[0] - cell / 2)) / cell).astype(int)
    iy = np.floor((p[:, 1] - (ys[0] - cell / 2)) / cell).astype(int)
    ok = (ix >= 0) & (iy >= 0) & (ix < len(xs)) & (iy < len(ys))
    cnt = np.zeros(Z.shape, dtype=int)
    np.add.at(cnt, (iy[ok], ix[ok]), 1)
    own = valid & (cnt >= 2)
    X, Y = np.meshgrid(xs, ys)
    fill = valid & ~own
    if fill.any() and own.sum() >= 4:
        zi = griddata((X[own], Y[own]), Z[own], (X[fill], Y[fill]), method="linear")
        Zl = Z.copy()
        Zl[fill] = np.where(np.isfinite(zi), zi, Z[fill])
        # A ramp only where it is no steeper than ``max_ramp_deg``: beside a
        # tall wall the strip the laser could not see is a shadow, not a
        # slope, and a ramp down it from the tower's top to the roof below
        # stood walls and slopes where there are none (9947f2b0, the Omni).
        # (the steepest step to a neighbouring cell, so a cell on the
        # outline is judged by the cells it has)
        Zv = np.where(valid, Zl, np.nan)
        P = np.pad(Zv, 1, constant_values=np.nan)
        rise = np.zeros(Z.shape)
        for dy, dx in ((0, 1), (1, 0), (2, 1), (1, 2)):
            d = np.abs(P[dy:dy + Z.shape[0], dx:dx + Z.shape[1]] - Zv)
            rise = np.fmax(rise, np.nan_to_num(d, nan=0.0))
        steep = rise > cell * np.tan(np.radians(max_ramp_deg))
        Z = np.where(fill & ~steep, Zl, Z)
    zv = Z[valid]
    hist, edges = np.histogram(zv, bins=np.arange(np.floor(zv.min()), np.ceil(zv.max()) + level_bin_m, level_bin_m))
    # a level's share: the cells within a bin either side of a local peak
    share = np.convolve(hist, np.ones(3), mode="same") / max(len(zv), 1)
    levels = []
    for k in np.argsort(-share):
        if share[k] < min_level_share:
            break
        lvl = float(edges[k] + level_bin_m / 2)
        if all(abs(lvl - q) >= level_sep_m for q in levels):
            levels.append(lvl)
    if not levels:
        return Z
    Z = _lower_rooftop_boxes(Z, valid, cell, levels[0], max_box_m2, max_box_h_m)
    L = np.array(levels)
    near = np.argmin(np.abs(Z[..., None] - L[None, None, :]), axis=-1)
    d = np.abs(Z - L[near])
    # only where the roof around the cell is nearly flat: a slope the laser
    # saw (2b18f52d's east side, 20.5 down to 17 m over 4 m) is not a level
    from scipy.ndimage import maximum_filter, minimum_filter
    Zv = np.where(valid, Z, np.nan)
    hi = maximum_filter(np.nan_to_num(Zv, nan=-1e9), size=3)
    lo = minimum_filter(np.nan_to_num(Zv, nan=1e9), size=3)
    flat = valid & ((hi - lo) <= 2.0 * cell * np.tan(np.radians(max_flat_deg)))
    lab = np.where(valid & flat & (d <= snap_m), near + 1, 0)

    def majority(v):
        c = int(v[len(v) // 2])
        if c == 0:
            return 0
        vals = v[v > 0].astype(int)
        return int(np.bincount(vals).argmax()) if len(vals) else c
    lab = generic_filter(lab, majority, size=3, mode="constant", cval=0).astype(int)
    # the roof's outermost cells keep their heights: a parapet 0.5-1 m over
    # the roof was snapped into it, and every roofline lost its lip
    from scipy.ndimage import binary_dilation
    lab[~valid | binary_dilation(~valid)] = 0
    for k in range(1, len(L) + 1):
        regions, n = label(lab == k)
        for r in range(1, n + 1):
            m = regions == r
            if m.sum() * cell * cell >= min_region_m2:
                Z[m] = L[k - 1]
    return Z


def roof_sloped_fraction(Z: np.ndarray, valid: np.ndarray, cell: float,
                         lo_deg: float = 10.0, hi_deg: float = 60.0, patch_m: float = 3.0) -> float:
    """Share of the roof cells that slope like a pitched roof: between
    ``lo_deg`` (flatter is a flat roof's drainage fall) and ``hi_deg``
    (steeper is a wall), in a sloping patch at least ``patch_m`` across.

    The patch rule matters downtown: the height field is a per-cell
    percentile, smoothed, so a sheer step between two flat roofs becomes a
    ramp one or two cells wide -- on a sample of 45 flat-plated roofs a
    fifth of the cells read as sloped that way, from steps, plant rooms and
    parapets. A pitched roof slopes over metres. Only cells whose
    neighbours are all valid count, so the footprint's edge is not a slope.
    """
    from scipy.ndimage import binary_opening
    if valid.sum() < 4:
        return 0.0
    gy, gx = np.gradient(np.where(valid, Z, np.nan), cell)
    ok = valid & np.isfinite(gx) & np.isfinite(gy)
    if not ok.any():
        return 0.0
    slope = np.degrees(np.arctan(np.hypot(np.nan_to_num(gx), np.nan_to_num(gy))))
    sloped = ok & (slope >= lo_deg) & (slope <= hi_deg)
    k = max(int(round(patch_m / cell)), 1)
    sloped = binary_opening(sloped, structure=np.ones((k, k), dtype=bool))
    return float(sloped.sum() / ok.sum())


def outside_modelled_wall(cen: np.ndarray, planes, profiles, slack_m: float = 1.0) -> np.ndarray:
    """Which of ``cen`` stand outside the wall the profile puts at their
    height: between a face's modelled wall and its footprint edge. A steep
    roof cell there is the underside of an overhang that the 2.5-D height
    field has filled in as wall -- Humana's porch cantilevers 24 m over its
    podium -- and not a wall the LiDAR saw."""
    out = np.zeros(len(cen), dtype=bool)
    for plane in planes:
        L = plane.length_m
        along, off = plane.local(cen[:, :2])
        on_face = (along >= -1.0) & (along <= L + 1.0) & (off > -0.6 * L) & (off < 1.0)
        known = [(0.5 * (b.z_lo + b.z_hi), b.offset_m) for b in profiles.get(plane.face, [])
                 if b.status in WALL_STATUSES and b.offset_m is not None]
        if not known or not on_face.any():
            continue
        # at a height with no band of its own, the nearest known band
        # within 30 m speaks for the wall
        kz = np.array([z for z, _ in known]); ko = np.array([o for _, o in known])
        i = np.abs(cen[:, 2][:, None] - kz[None, :]).argmin(axis=1)
        has = np.abs(cen[:, 2] - kz[i]) <= 30.0
        out[on_face & has & (off > ko[i] + slack_m)] = True
    return out


def roof_from_outline(xs, ys, Z, valid, rings):
    """The roof as triangles cut along the outlines: points every half a cell
    along each level's outline -- the ground level's at the height of the
    field's cell nearest it; an upper level's twice, a few centimetres inside at its
    own roof's height and outside at the one below, the step between them
    left to the steep triangles -- and the field's cells inside it,
    triangulated and kept within the ground level's outline.
    Two triangles per cell of a north-south grid stepped along every wall set
    at an angle (8c47a2dd's roof left 57% of the strip inside its walls open).
    Returns (verts (V, 3), tris (T, 3))."""
    import shapely
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import Delaunay

    cell = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    Zf = Z[iy, ix]

    def z_at(xy):
        gx = np.clip(np.round((xy[:, 0] - xs[0]) / cell).astype(int), 0, len(xs) - 1)
        gy = np.clip(np.round((xy[:, 1] - ys[0]) / cell).astype(int), 0, len(ys) - 1)
        return Zf[gy, gx]

    def ccw(r):
        r = np.asarray(r, dtype=float)[:, :2]
        if not np.allclose(r[0], r[-1]):
            r = np.vstack([r, r[:1]])
        a2 = np.sum(r[:-1, 0] * r[1:, 1] - r[1:, 0] * r[:-1, 1])
        return r if a2 > 0 else r[::-1]

    pts, zs = [], []
    for k, ring in enumerate(rings):
        R = ccw(ring)
        for p0, p1 in zip(R[:-1], R[1:]):
            L = float(np.hypot(*(p1 - p0)))
            if L < 1e-6:
                continue
            n = max(1, int(np.ceil(L / (0.5 * cell))))
            P = p0[None, :] + (p1 - p0)[None, :] * (np.arange(n) / n)[:, None]
            inward = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / L          # left of travel on a counter-clockwise ring
            if k == 0:
                # the nearest cell's height to the point itself: taken half a cell in,
                # RIVUE's edge points stood at its podium's 9 m over a 4.4 m canopy at
                # the edge, a sliver of roof standing up as wall
                pts.append(P); zs.append(z_at(P))
            else:
                pts.append(P + 0.05 * inward); zs.append(z_at(P + 0.5 * cell * inward))
                pts.append(P - 0.05 * inward); zs.append(z_at(P - 0.5 * cell * inward))
    outer = shapely.Polygon(ccw(rings[0]))
    if not outer.is_valid:
        outer = outer.buffer(0)
    X, Y = np.meshgrid(xs, ys)
    # every cell inside the outline, bar those almost on it: leaving out the
    # half-cell strip, a canopy one cell wide lost its cells and the roof
    # sloped from its edge straight up to the podium's (RIVUE)
    keep = valid & shapely.contains_xy(outer.buffer(-0.15 * cell), X, Y)
    for ring in rings[1:]:
        keep &= shapely.distance(shapely.LinearRing(ccw(ring)), shapely.points(X, Y)) > 0.5 * cell
    pts.append(np.column_stack([X[keep], Y[keep]])); zs.append(Z[keep])
    P, z = np.vstack(pts), np.concatenate(zs)
    P, first = np.unique(np.round(P, 3), axis=0, return_index=True)
    z = z[first]
    if len(P) < 3:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    tri = Delaunay(P).simplices
    polys = shapely.polygons(np.concatenate([P[tri], P[tri][:, :1]], axis=1))
    ok = shapely.within(polys, outer.buffer(0.05))
    # no slivers: along a straight edge with no cell close inside, three of its
    # points made a triangle 105 m long and centimetres wide that stood up as a
    # wall (RIVUE); a roof triangle has some width from above
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    plan = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    longest = np.max(np.stack([np.hypot(*(b - a).T), np.hypot(*(c - b).T), np.hypot(*(a - c).T)]), axis=0)
    ok &= plan >= 0.05 * cell * longest
    return np.column_stack([P, z]), tri[ok].astype(np.int64)


def roof_mesh(xs, ys, Z, valid, wall_quads, planes=(), profiles=None, max_slope_deg: float = 75.0,
              min_step_m: float = 2.0, levels=None):
    """The roof's triangles, split into the flat ones (the roof, textured
    from the plan view) and the steep ones (a step or a wall the LiDAR saw
    that no band models -- Humana's bulging north side); a steep triangle
    standing where a wall quad already is gets dropped. The triangles are
    cut along the levels' outlines (roof_from_outline); with no levels, two
    per cell of four valid corners."""
    if levels:
        verts, tris = roof_from_outline(xs, ys, Z, valid, [lv["ring"] for lv in levels])
    else:
        ny, nx = Z.shape
        vid = -np.ones((ny, nx), dtype=int)
        vid[valid] = np.arange(valid.sum())
        X, Y = np.meshgrid(xs, ys)
        verts = np.column_stack([X[valid], Y[valid], Z[valid]])
        tris = []
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                a, b, c, d = vid[iy, ix], vid[iy, ix + 1], vid[iy + 1, ix], vid[iy + 1, ix + 1]
                if min(a, b, c, d) < 0:
                    continue
                tris.append((a, b, d))
                tris.append((a, d, c))
        tris = np.array(tris, dtype=np.int64).reshape(-1, 3)
    if not len(tris):
        return verts, tris, tris
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    nz = n[:, 2] / (np.linalg.norm(n, axis=1) + 1e-9)
    drop = np.max(np.stack([v0[:, 2], v1[:, 2], v2[:, 2]]), axis=0) - np.min(np.stack([v0[:, 2], v1[:, 2], v2[:, 2]]), axis=0)
    steep = (nz < np.cos(np.radians(max_slope_deg))) & (drop >= min_step_m)
    if steep.any() and wall_quads:
        cen = (v0 + v1 + v2) / 3
        near = np.zeros(len(tris), dtype=bool)
        # a wall quad's plan segment, and its height range
        segs = [(q[0][0][:2], q[0][1][:2], q[0][:, 2].min(), q[0][:, 2].max()) for q in wall_quads if q[3] == "wall"]
        for s_idx in np.flatnonzero(steep):
            cx, cy, cz = cen[s_idx]
            for pa, pb, zlo, zhi in segs:
                if cz < zlo - 1 or cz > zhi + 1:
                    continue
                d = pb - pa
                t = np.clip(np.dot([cx - pa[0], cy - pa[1]], d) / (np.dot(d, d) + 1e-9), 0, 1)
                if np.hypot(*(pa + t * d - [cx, cy])) < 1.5:
                    near[s_idx] = True
                    break
        if profiles:
            near |= outside_modelled_wall(cen, planes, profiles)
        if levels and len(levels) > 1:
            for lv in levels[1:]:
                ring_lv = np.asarray(lv["ring"], dtype=float)
                for pa, pb in zip(ring_lv[:-1], ring_lv[1:]):
                    d = pb - pa
                    L2 = float(np.dot(d, d))
                    if L2 < 1e-6:
                        continue
                    t = np.clip(((cen[:, 0] - pa[0]) * d[0] + (cen[:, 1] - pa[1]) * d[1]) / L2, 0, 1)
                    dist = np.hypot(cen[:, 0] - (pa[0] + t * d[0]), cen[:, 1] - (pa[1] + t * d[1]))
                    near |= steep & (dist < 2.5)
        tris, steep = tris[~(steep & near)], steep[~(steep & near)]
    return verts, tris[~steep], tris[steep]


def roof_texture(xs, ys, Z, valid, poly, cams, texel_m: float = TEXEL_M, outline=None):
    """Plan-view texture of the roof sampled through the camera models."""
    from scipy.ndimage import distance_transform_edt
    from shapely import contains_xy

    from kyfromabove_3d.colourise.colour import sample_colour

    if outline is not None:
        poly = outline.union(poly)                      # the texture's extent covers both
    minx, miny, maxx, maxy = poly.bounds
    W = _mult4(min(MAX_TEX_PX, int(np.ceil((maxx - minx) / texel_m))))
    H = _mult4(min(MAX_TEX_PX, int(np.ceil((maxy - miny) / texel_m))))
    tx = minx + (np.arange(W) + 0.5) * (maxx - minx) / W
    ty = miny + (np.arange(H) + 0.5) * (maxy - miny) / H
    TX, TY = np.meshgrid(tx, ty)
    # height from the nearest valid cell
    cell = xs[1] - xs[0]
    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    Zf = Z[iy, ix]
    gx = np.clip(((TX - xs[0]) / cell + 0.5).astype(int), 0, len(xs) - 1)
    gy = np.clip(((TY - ys[0]) / cell + 0.5).astype(int), 0, len(ys) - 1)
    tz = Zf[gy, gx]
    # with the roof cut along the outline, all of it inside the outline is
    # painted (heights from the nearest cell), not only the field's cells
    inside = (contains_xy(outline, TX.ravel(), TY.ravel()) if outline is not None
              else contains_xy(poly, TX.ravel(), TY.ravel()) & valid[gy, gx].ravel())
    pts = np.column_stack([TX.ravel(), TY.ravel(), tz.ravel()])[inside]
    up = np.repeat(np.array([[0.0, 0.0, 1.0]]), len(pts), axis=0)
    rgb, seen, _, _ = sample_colour(pts, up, cams, min_facing_cos=0.3, occlusion_slack_m=1.5)
    img = np.zeros((H * W, 3)); cnt = np.zeros(H * W)
    ii = np.flatnonzero(inside)[seen > 0]
    img[ii] = rgb[seen > 0]; cnt[ii] = 1
    tex = _fill_holes(img.reshape(H, W, 3), cnt.reshape(H, W))
    return tex, (minx, miny, maxx, maxy), int((seen > 0).sum()), int(inside.sum())


def roof_texture_from_cloud(xs, ys, Z, valid, poly, cxyz: np.ndarray, crgb: np.ndarray,
                            texel_m: float = TEXEL_M, flat_z: float | None = None):
    """Plan-view texture of the roof rasterized from the colorized aerial point cloud.

    Used when camera crop PNGs are not on disk (which occurs after run_downtown
    cleans intermediate crops to save disk space).
    """
    from scipy.ndimage import distance_transform_edt
    from shapely import contains_xy

    minx, miny, maxx, maxy = poly.bounds
    W = _mult4(min(MAX_TEX_PX, int(np.ceil((maxx - minx) / texel_m))))
    H = _mult4(min(MAX_TEX_PX, int(np.ceil((maxy - miny) / texel_m))))

    inside = contains_xy(poly, cxyz[:, 0], cxyz[:, 1])
    has_color = crgb.sum(axis=1) > 15
    m = inside & has_color
    if not m.any():
        return None, (minx, miny, maxx, maxy), 0, 0

    if flat_z is not None:
        on_roof = m & (cxyz[:, 2] >= flat_z - 3.0)
        if not on_roof.any():
            on_roof = m
    elif valid.any():
        cell = xs[1] - xs[0]
        _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
        Zf = Z[iy, ix]
        gx = np.clip(((cxyz[:, 0] - xs[0]) / cell + 0.5).astype(int), 0, len(xs) - 1)
        gy = np.clip(((cxyz[:, 1] - ys[0]) / cell + 0.5).astype(int), 0, len(ys) - 1)
        on_roof = m & (np.abs(cxyz[:, 2] - Zf[gy, gx]) < 2.0)
        if not on_roof.any():
            on_roof = m
    else:
        top_z = cxyz[m, 2].max()
        on_roof = m & (cxyz[:, 2] >= top_z - 3.0)
        if not on_roof.any():
            on_roof = m

    pts_roof = cxyz[on_roof]
    rgb_roof = crgb[on_roof]

    u = np.clip(((pts_roof[:, 0] - minx) / max(maxx - minx, 1e-6) * W).astype(int), 0, W - 1)
    v = np.clip(((pts_roof[:, 1] - miny) / max(maxy - miny, 1e-6) * H).astype(int), 0, H - 1)

    img = np.zeros((H, W, 3), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.float64)
    np.add.at(img, (v, u), rgb_roof.astype(np.float64))
    np.add.at(cnt, (v, u), 1.0)

    tex = _fill_holes(img, cnt, max_fill_px=max(4, int(2.0 / texel_m)))
    return tex, (minx, miny, maxx, maxy), int(len(pts_roof)), int(inside.sum())
