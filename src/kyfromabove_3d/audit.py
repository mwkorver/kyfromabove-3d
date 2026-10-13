"""Audit a building's mesh against evidence it was not built from.

The pipeline has broken in ways only a person looking at the viewer caught:
a staircase-shaped hole 60 m wide in the Omni's west tower wall, the Omni's
57 m south wing with almost no south wall, wall panels standing on Kentucky
Home Life's roof, walls covered in round dots and then in straight lines.
Each measure here puts a number on one such failure, so a code change can
be checked by comparing numbers instead of by eye.

The main reference is the LiDAR as a solid: every roof cell the returns
give, a column from just above the ground to its height. A building's mesh
should cover that solid from every side.

Measures (all in plain units, per building and where it applies per face):

* ``holes`` -- the mesh and the solid seen straight on from eight sides and
  from above, 0.5 m pixels: where the solid is there and the mesh is not, or
  is more than ``see_through_m`` behind the solid's front (looking through a
  missing wall at the back of the building), is a hole;
* ``roof_walls`` -- wall standing on a flat part of the roof, well inside
  the outline (a real rooftop structure stands on a step in the LiDAR roof);
* ``heights`` -- the mesh's top against the LiDAR's and product 1's;
* ``texture`` -- the share of wall left blank, and two artefacts of the
  small-gap fill: round *dots* (colour grown round a lone speck) and thin
  *lines* (colour bridged between specks across blank wall);
* ``split_shape`` -- the thinnest width of the outline (its sharpest corner
  is ``overture_boxes.min_corner_deg``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PIXEL_M = 0.5

SEEN, NEAR, PLAIN = 1, 2, 3


# ------------------------------------------------------------------ sampling

def triangle_normals_areas(V: np.ndarray, T: np.ndarray):
    """Unit normals (T, 3) and areas (T,) of the triangles."""
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / (2.0 * area[:, None])
    return np.nan_to_num(n), area


def sample_mesh(V: np.ndarray, T: np.ndarray, per_m2: float = 48.0, seed: int = 0):
    """Points spread over the mesh's surface, ``per_m2`` per square metre on
    average (dense enough that a 0.5 m pixel facing the view is all but
    never missed): (points (P, 3), triangle index of each (P,))."""
    if not len(T):
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    _, area = triangle_normals_areas(V, T)
    rng = np.random.default_rng(seed)
    n = np.floor(area * per_m2 + rng.random(len(T))).astype(np.int64)
    tri = np.repeat(np.arange(len(T)), n)
    r1, r2 = rng.random(len(tri)), rng.random(len(tri))
    flip = r1 + r2 > 1.0
    r1[flip], r2[flip] = 1.0 - r1[flip], 1.0 - r2[flip]
    a, b, c = V[T[tri, 0]], V[T[tri, 1]], V[T[tri, 2]]
    return a + r1[:, None] * (b - a) + r2[:, None] * (c - a), tri


@dataclass
class Solid:
    """The LiDAR as a solid: surface points and, for each, the face it lies
    on (-1 for the roof)."""
    points: np.ndarray
    face: np.ndarray


def lidar_solid(xs: np.ndarray, ys: np.ndarray, Z: np.ndarray, valid: np.ndarray, floor_z,
                step_m: float = 0.25):
    """Surface points of the columns the roof field describes (each valid
    cell a column from its floor to its height): the tops, and the sides
    where a column stands where its neighbour's does not (above it, or
    below the neighbour's floor, or beside the field's edge). Only the outer
    surface -- the inside of a solid is never what a view sees first.
    ``floor_z`` is one height, or one per cell: a cell under an overhang
    (the Hyatt's drum over its shaft, ``overhangs.json``) has its column
    start at the overhang's underside. Returns (points (P, 3), is_top (P,))."""
    cell = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    H, W = Z.shape
    F = np.broadcast_to(np.asarray(floor_z, dtype=float), Z.shape)
    per = max(2, int(round(cell / step_m)))
    offs = (np.arange(per) + 0.5) / per - 0.5                     # across a cell, in cells
    iy, ix = np.nonzero(valid & (Z > F))
    pts = []
    # tops
    gx, gy = np.meshgrid(offs, offs)
    tx = (xs[ix][:, None] + gx.ravel()[None, :] * cell).ravel()
    ty = (ys[iy][:, None] + gy.ravel()[None, :] * cell).ravel()
    pts.append(np.column_stack([tx, ty, np.repeat(Z[iy, ix], per * per)]))
    # sides, towards each of the four neighbours: where this column stands and
    # the neighbour's does not -- above the neighbour's top, and below its floor
    has = valid & (Z > F)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        ny, nx = iy + dy, ix + dx
        inside = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        nb_has = np.zeros(len(iy), dtype=bool)
        nb_has[inside] = has[ny[inside], nx[inside]]
        top, bot = Z[iy, ix], F[iy, ix]
        nb_top = np.where(nb_has, Z[np.clip(ny, 0, H - 1), np.clip(nx, 0, W - 1)], -np.inf)
        nb_bot = np.where(nb_has, F[np.clip(ny, 0, H - 1), np.clip(nx, 0, W - 1)], -np.inf)   # no neighbour: all of it is above
        for lo, hi in ((np.maximum(bot, nb_top), top),                   # above the neighbour (or all of it)
                       (bot, np.minimum(top, nb_bot))):                  # below the neighbour's floor
            k = hi > lo + step_m
            if not k.any():
                continue
            cy, cx, l0, h0 = iy[k], ix[k], lo[k], hi[k]
            nz = np.ceil((h0 - l0) / step_m).astype(np.int64)
            rep = np.repeat(np.arange(len(cy)), nz * per)
            start = np.repeat(np.cumsum(nz * per) - nz * per, nz * per)
            j = np.arange(len(rep)) - start
            zi, ai = j // per, j % per
            z = np.minimum(l0[rep] + (zi + 0.5) * step_m, h0[rep])
            x = xs[cx[rep]] + (0.5 * dx + (offs[ai] if dx == 0 else 0.0)) * cell
            y = ys[cy[rep]] + (0.5 * dy + (offs[ai] if dy == 0 else 0.0)) * cell
            pts.append(np.column_stack([x, y, z]))
    P = np.vstack(pts)
    top = np.zeros(len(P), dtype=bool)
    top[:len(pts[0])] = True
    return P, top


def nearest_face(points: np.ndarray, edges: Sequence[Any], step_m: float = 1.0) -> np.ndarray:
    """For each point, the face of the nearest footprint edge whose height
    range holds it (-1 when none does): ``edges`` are WallPlane / ArcWall
    with ``face``, ``z_lo``, ``z_hi``."""
    out = np.full(len(points), -1, dtype=np.int64)
    best = np.full(len(points), np.inf)
    xy, z = points[:, :2], points[:, 2]
    for e in edges:
        line = e.world_line(step_m) if hasattr(e, "world_line") else np.array([e.p0, e.p1])
        in_z = (z >= e.z_lo - 0.5) & (z <= e.z_hi + 0.5)
        if not in_z.any():
            continue
        d = np.full(len(points), np.inf)
        for p, q in zip(line[:-1], line[1:]):
            seg = q - p
            L2 = float(seg @ seg)
            t = np.clip(((xy - p) @ seg) / max(L2, 1e-12), 0.0, 1.0)
            d = np.minimum(d, np.hypot(*(xy - (p + t[:, None] * seg)).T))
        d[~in_z] = np.inf
        closer = d < best
        best[closer], out[closer] = d[closer], e.face
    return out


# ------------------------------------------------------------------- views

VIEWS = [(f"from {n}", az) for n, az in
         (("S", 0.0), ("SW", 45.0), ("W", 90.0), ("NW", 135.0), ("N", 180.0), ("NE", 225.0), ("E", 270.0), ("SE", 315.0))]
TOP = ("from above", None)


def _view_axes(az_deg: Optional[float]):
    """(right, up, forward) unit vectors of a straight-on view. ``az_deg``
    is the bearing the viewer looks along, clockwise from north: 0 looks
    north (a view "from S"), 90 looks east ("from W"). None looks down."""
    if az_deg is None:
        return np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, -1.0])
    a = np.radians(az_deg)
    fwd = np.array([np.sin(a), np.cos(a), 0.0])                     # from S (az 0) looks north
    right = np.array([fwd[1], -fwd[0], 0.0])
    return right, np.array([0, 0, 1.0]), fwd


def project(points: np.ndarray, view_az: Optional[float]):
    """(u, v, depth) of points in a straight-on view."""
    r, u, f = _view_axes(view_az)
    return points @ r, points @ u, points @ f


def depth_buffer(u, v, depth, u0, v0, W, H, pixel_m: float = PIXEL_M):
    """The nearest depth per pixel (inf where nothing) and the index of the
    point that gave it (-1)."""
    iu = np.floor((u - u0) / pixel_m).astype(np.int64)
    iv = np.floor((v - v0) / pixel_m).astype(np.int64)
    ok = (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(-depth[idx], kind="stable")]              # farthest first: the nearest writes last
    buf = np.full(H * W, np.inf)
    who = np.full(H * W, -1, dtype=np.int64)
    flat = iv[order] * W + iu[order]
    buf[flat] = depth[order]
    who[flat] = order
    return buf.reshape(H, W), who.reshape(H, W)


def holes(mesh_pts: np.ndarray, solid: Solid, see_through_m: float = 6.0, pixel_m: float = PIXEL_M,
          min_hole_m2: float = 4.0, views=None, face_normals: Optional[Dict[int, np.ndarray]] = None,
          min_facing: float = 0.3) -> Dict[str, Any]:
    """Where the solid shows and the mesh does not, view by view.

    A hole is put down to the face most of its pixels lie on, among the
    faces the view looks at (``face_normals``, plan normals pointing out, at
    least ``min_facing`` towards the viewer): a face seen edge-on shows only
    a sliver at a corner, and a sliver is not that face's hole. A hole on no
    such face is a "corner".

    Returns {"views": [{view, hole_m2, holes: [{m2, z_lo, z_hi, face}]}],
    "by_face": {face: largest hole area over the views, m2}, "total_m2":
    the largest view's hole area, "images": per view (mesh depth, solid
    depth, hole mask, extent) for the picture}."""
    from scipy.ndimage import binary_erosion, binary_opening, label

    views = views if views is not None else VIEWS + [TOP]
    out_views, by_face, images = [], {}, {}
    min_px = max(1, int(round(min_hole_m2 / pixel_m ** 2)))
    for name, az in views:
        mu, mv, md = project(mesh_pts, az)
        su, sv, sd = project(solid.points, az)
        if not len(su):
            continue
        allu, allv = np.concatenate([mu, su]), np.concatenate([mv, sv])
        u0, v0 = allu.min() - pixel_m, allv.min() - pixel_m
        W = int(np.ceil((allu.max() - u0) / pixel_m)) + 2
        H = int(np.ceil((allv.max() - v0) / pixel_m)) + 2
        mbuf, _ = depth_buffer(mu, mv, md, u0, v0, W, H, pixel_m)
        sbuf, swho = depth_buffer(su, sv, sd, u0, v0, W, H, pixel_m)
        have_solid = np.isfinite(sbuf)
        # the solid's own outline is a pixel wide either way: judge its inside
        core = binary_erosion(have_solid, iterations=1)
        hole = core & (mbuf > sbuf + see_through_m)
        hole = binary_opening(hole, structure=np.ones((2, 2)))
        lab, n = label(hole)
        recs = []
        for k in range(1, n + 1):
            m = lab == k
            if m.sum() < min_px:
                hole[m] = False
                continue
            faces = solid.face[swho[m]]
            faces = faces[faces >= 0]
            if face_normals is not None and az is not None and len(faces):
                to_viewer = -_view_axes(az)[2][:2]
                facing = np.array([float(np.asarray(face_normals.get(int(f), to_viewer))[:2] @ to_viewer) >= min_facing
                                   for f in faces])
                faces = faces[facing]
            face = int(np.bincount(faces).argmax()) if len(faces) else "corner"
            rows = np.nonzero(m.any(axis=1))[0]
            rec = {"m2": round(float(m.sum()) * pixel_m ** 2, 1), "face": face if az is not None else "roof"}
            if az is not None:
                rec["z_lo"], rec["z_hi"] = round(v0 + rows.min() * pixel_m, 1), round(v0 + (rows.max() + 1) * pixel_m, 1)
            recs.append(rec)
            key = str(rec["face"])
            by_face[key] = max(by_face.get(key, 0.0), rec["m2"])
        recs.sort(key=lambda r: -r["m2"])
        area = round(float(hole.sum()) * pixel_m ** 2, 1)
        out_views.append({"view": name, "hole_m2": area, "holes": recs[:10]})
        images[name] = (mbuf, sbuf, hole, (u0, v0, W, H))
    return {"views": out_views, "by_face": {k: round(v, 1) for k, v in sorted(by_face.items(), key=lambda kv: -kv[1])},
            "total_m2": max((v["hole_m2"] for v in out_views), default=0.0), "images": images}


# ------------------------------------------------------- walls on the roof

def roof_walls(V: np.ndarray, T: np.ndarray, xs, ys, Z, valid, outline, inside_m: float = 2.0,
               step_tol_m: float = 1.5, radius_m: float = 2.0, per_m2: float = 8.0) -> Dict[str, Any]:
    """Wall standing on a flat part of the roof: near-vertical triangles
    more than ``inside_m`` inside ``outline`` whose surroundings in the
    LiDAR roof (within ``radius_m``) do not step by more than ``step_tol_m``,
    and which stand on that roof rather than below it. A tower rising from
    its podium, a penthouse, a lantern: each stands on a step, and is not
    counted. Returns {"m2", "n_triangles"}."""
    import shapely
    from scipy.ndimage import maximum_filter, minimum_filter

    n, area = triangle_normals_areas(V, T)
    vert = (np.abs(n[:, 2]) < 0.2) & (area > 0.01)
    if not vert.any() or not valid.any():
        return {"m2": 0.0, "n_triangles": 0}
    pts, tri = sample_mesh(V, T[vert], per_m2=per_m2, seed=1)
    if not len(pts):
        return {"m2": 0.0, "n_triangles": 0}
    cell = float(xs[1] - xs[0])
    r = max(1, int(np.ceil(radius_m / cell)))
    size = 2 * r + 1
    hi = maximum_filter(np.where(valid, Z, -np.inf), size=size, mode="nearest")
    lo = minimum_filter(np.where(valid, Z, np.inf), size=size, mode="nearest")
    ix = np.clip(np.round((pts[:, 0] - xs[0]) / cell).astype(int), 0, len(xs) - 1)
    iy = np.clip(np.round((pts[:, 1] - ys[0]) / cell).astype(int), 0, len(ys) - 1)
    flat = valid[iy, ix] & np.isfinite(hi[iy, ix]) & np.isfinite(lo[iy, ix]) & (hi[iy, ix] - lo[iy, ix] <= step_tol_m)
    above = pts[:, 2] >= Z[iy, ix] - 0.5
    poly = shapely.Polygon(outline)
    deep = shapely.contains_xy(poly.buffer(-inside_m), pts[:, 0], pts[:, 1]) if not poly.buffer(-inside_m).is_empty \
        else np.zeros(len(pts), dtype=bool)
    bad = flat & above & deep
    share = np.bincount(tri, weights=bad, minlength=int(vert.sum())) / np.maximum(np.bincount(tri, minlength=int(vert.sum())), 1)
    flagged = share >= 0.5
    return {"m2": round(float(area[vert][flagged].sum()), 1), "n_triangles": int(flagged.sum())}


def stray_walls(V: np.ndarray, T: np.ndarray, xs, ys, Z, valid, near_m: float = 1.0, above_m: float = 1.5,
                per_m2: float = 8.0) -> Dict[str, Any]:
    """Wall drawn where the LiDAR has no building: near-vertical triangles
    standing more than ``near_m`` outside every roof cell of the solid
    (*outside*), or more than ``above_m`` above the highest roof within
    ``near_m`` of them (*above* -- a wall carried up past a lower roof). The
    holes measure what the mesh leaves out; this measures what it invents.
    Returns {"m2", "outside_m2", "above_m2"}."""
    from scipy.ndimage import maximum_filter

    n, area = triangle_normals_areas(V, T)
    vert = (np.abs(n[:, 2]) < 0.2) & (area > 0.01)
    none = {"m2": 0.0, "outside_m2": 0.0, "above_m2": 0.0}
    if not vert.any() or not valid.any():
        return none
    pts, tri = sample_mesh(V, T[vert], per_m2=per_m2, seed=2)
    if not len(pts):
        return none
    cell = float(xs[1] - xs[0])
    r = max(1, int(np.ceil(near_m / cell)))
    hi = maximum_filter(np.where(valid, Z, -np.inf), size=2 * r + 1, mode="constant", cval=-np.inf)
    fx, fy = (pts[:, 0] - xs[0]) / cell, (pts[:, 1] - ys[0]) / cell
    ix = np.clip(np.round(fx).astype(int), 0, len(xs) - 1)
    iy = np.clip(np.round(fy).astype(int), 0, len(ys) - 1)
    # past the field's edge by more than near_m: nothing of the solid is near
    beyond = (np.maximum(np.abs(fx - ix), np.abs(fy - iy)) * cell) > near_m
    top = np.where(beyond, -np.inf, hi[iy, ix])
    outside = ~np.isfinite(top)
    above = ~outside & (pts[:, 2] > top + above_m)
    counts = np.maximum(np.bincount(tri, minlength=int(vert.sum())), 1)
    a_vert = area[vert]
    res = {}
    for k, bad in (("outside_m2", outside), ("above_m2", above)):
        share = np.bincount(tri, weights=bad, minlength=int(vert.sum())) / counts
        res[k] = round(float(a_vert[share >= 0.5].sum()), 1)
    res["m2"] = round(res["outside_m2"] + res["above_m2"], 1)
    return res


def inside_walls(V: np.ndarray, T: np.ndarray, xs, ys, Z, valid, outline, inside_m: float = 2.0,
                 step_tol_m: float = 1.5, radius_m: float = 2.0, per_m2: float = 8.0) -> Dict[str, Any]:
    """Wall drawn inside the building, under its roof: near-vertical
    triangles more than ``inside_m`` inside ``outline`` and below the LiDAR
    roof there, where that roof does not step (within ``step_tol_m`` over
    ``radius_m``) -- a wall standing under an unbroken roof is not a wall
    anyone sees. fb099a5d, one storey, had its walls drawn 4-7 m inside its
    edges this way; the holes never showed it, the mesh being less than 6 m
    behind the solid's front. A tower's wall stands on a step and is not
    counted. Returns {"m2", "n_triangles"}."""
    import shapely
    from scipy.ndimage import maximum_filter, minimum_filter

    n, area = triangle_normals_areas(V, T)
    vert = (np.abs(n[:, 2]) < 0.2) & (area > 0.01)
    if not vert.any() or not valid.any():
        return {"m2": 0.0, "n_triangles": 0}
    pts, tri = sample_mesh(V, T[vert], per_m2=per_m2, seed=3)
    if not len(pts):
        return {"m2": 0.0, "n_triangles": 0}
    cell = float(xs[1] - xs[0])
    r = max(1, int(np.ceil(radius_m / cell)))
    hi = maximum_filter(np.where(valid, Z, -np.inf), size=2 * r + 1, mode="nearest")
    lo = minimum_filter(np.where(valid, Z, np.inf), size=2 * r + 1, mode="nearest")
    ix = np.clip(np.round((pts[:, 0] - xs[0]) / cell).astype(int), 0, len(xs) - 1)
    iy = np.clip(np.round((pts[:, 1] - ys[0]) / cell).astype(int), 0, len(ys) - 1)
    flat = valid[iy, ix] & np.isfinite(hi[iy, ix]) & np.isfinite(lo[iy, ix]) & (hi[iy, ix] - lo[iy, ix] <= step_tol_m)
    under = pts[:, 2] < Z[iy, ix] - 0.5
    poly = shapely.Polygon(outline)
    core = poly.buffer(-inside_m)
    deep = shapely.contains_xy(core, pts[:, 0], pts[:, 1]) if not core.is_empty else np.zeros(len(pts), dtype=bool)
    bad = flat & under & deep
    nt = int(vert.sum())
    share = np.bincount(tri, weights=bad, minlength=nt) / np.maximum(np.bincount(tri, minlength=nt), 1)
    flagged = share >= 0.5
    return {"m2": round(float(area[vert][flagged].sum()), 1), "n_triangles": int(flagged.sum())}


# ------------------------------------------------------------------ texture

def texture_artefacts(sources: Sequence[np.ndarray], texel_m: float) -> Dict[str, Any]:
    """From product 2's per-texel sources (SEEN / NEAR / PLAIN) of the wall
    textures: the share of wall left blank, and counts of two artefacts of
    the small-gap fill.

    * a *dot*: a patch of seen-or-filled wall between about 0.5 and 4 m
      across, roughly round, standing in blank wall, whose seen part is under
      a tenth of it -- colour grown round a lone speck;
    * a *line*: a strip of filled texels at most two texels wide and at least
      1 m long, with blank wall on both sides -- colour bridged between specks.
    """
    from scipy.ndimage import binary_dilation, label

    n = blank = 0
    dots = lines = 0
    for s in sources:
        s = np.asarray(s)
        n += s.size
        blank += int((s == PLAIN).sum())
        known = (s == SEEN) | (s == NEAR)
        lab, k = label(known, structure=np.ones((3, 3)))
        if not k:
            continue
        sizes = np.bincount(lab.ravel(), minlength=k + 1)
        seen_in = np.bincount(lab.ravel(), weights=(s == SEEN).ravel(), minlength=k + 1)
        lo_px, hi_px = (0.25 / texel_m) ** 2 * np.pi, (2.0 / texel_m) ** 2 * np.pi
        for c in np.flatnonzero((sizes >= lo_px) & (sizes <= hi_px) & (seen_in < 0.1 * sizes)):
            if c == 0:
                continue
            m = lab == c
            rows, cols = np.nonzero(m)
            h, w = np.ptp(rows) + 1, np.ptp(cols) + 1
            ring = binary_dilation(m) & ~m
            if 0.6 <= h / w <= 1.6 and sizes[c] >= 0.55 * h * w and (s[ring] == PLAIN).mean() > 0.6:
                dots += 1
        near = s == NEAR
        lab2, k2 = label(near, structure=np.ones((3, 3)))
        if not k2:
            continue
        sizes2 = np.bincount(lab2.ravel(), minlength=k2 + 1)
        for c in np.flatnonzero(sizes2 >= 1.0 / texel_m):
            if c == 0:
                continue
            m = lab2 == c
            rows, cols = np.nonzero(m)
            length = max(np.ptp(rows), np.ptp(cols), 1) + 1
            if length * texel_m < 1.0 or sizes2[c] > 2.0 * length:
                continue
            ring = binary_dilation(m, structure=np.ones((3, 3))) & ~m
            if (s[ring] == PLAIN).mean() > 0.5:
                lines += 1
    area_m2 = n * texel_m ** 2
    return {"blank_share": round(blank / n, 4) if n else None,
            "dots": dots, "lines": lines,
            "dots_per_1000m2": round(1000.0 * dots / area_m2, 2) if area_m2 else None,
            "lines_per_1000m2": round(1000.0 * lines / area_m2, 2) if area_m2 else None}


# ------------------------------------------------------------- split shape

def min_width_m(poly) -> float:
    """The thinnest width of an outline: twice the radius of the largest
    circle that fits inside it, metres."""
    import shapely

    return round(2.0 * float(shapely.maximum_inscribed_circle(poly, tolerance=0.1).length), 1)


# ------------------------------------------------------------ the picture

def draw(images: Dict[str, Any], path, title: str = "", pixel_m: float = PIXEL_M) -> None:
    """One panel per view: the mesh shaded by depth, the solid's outline,
    holes in magenta."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(images)
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows))
    for ax in np.ravel(axs):
        ax.axis("off")
    for ax, name in zip(np.ravel(axs), names):
        mbuf, sbuf, hole, (u0, v0, W, H) = images[name]
        rgb = np.ones((H, W, 3))
        m = np.isfinite(mbuf)
        if m.any():
            d = mbuf[m]
            shade = 0.35 + 0.5 * (d - d.min()) / max(float(np.ptp(d)), 1e-6)
            rgb[m] = shade[:, None]
        edge = np.isfinite(sbuf) & ~np.isfinite(np.roll(sbuf, 1, axis=1))
        rgb[edge] = (0.16, 0.47, 0.84)
        rgb[hole] = (0.9, 0.16, 0.78)
        ax.imshow(rgb[::-1], extent=(u0, u0 + W * pixel_m, v0, v0 + H * pixel_m), interpolation="nearest")
        area = float(hole.sum()) * pixel_m ** 2
        ax.set_title(f"{name}: hole {area:,.0f} m²", fontsize=10)
    fig.suptitle(title + "  (grey: the mesh, nearer is darker; magenta: the LiDAR solid shows and the mesh does not)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=80)
    plt.close(fig)
