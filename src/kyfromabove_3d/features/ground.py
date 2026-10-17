"""The ground under and around a building, rays to the wall planes, and the shadow an
overhang casts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .walls import WallPlane


@dataclass
class GroundField:
    """Continuous ground elevation field along a footprint perimeter.

    Interpolates ground elevation z_ground(s) along each footprint edge
    from LiDAR ground returns, enabling sloped sidewalk draping.
    """

    vertex_z: np.ndarray      # (N+1,) float elevations at ring vertices
    ring_m: np.ndarray        # (N+1, 2) footprint polygon

    def elevation_at_vertex(self, index: int) -> float:
        """Ground elevation at vertex index."""
        idx = min(max(index, 0), len(self.vertex_z) - 1)
        return float(self.vertex_z[idx])

    def elevation_along_edge(self, edge_idx: int, s: float, L: float) -> float:
        """Ground elevation at distance s along edge edge_idx of length L."""
        i = edge_idx % (len(self.vertex_z) - 1)
        z0 = float(self.vertex_z[i])
        z1 = float(self.vertex_z[i + 1])
        if L < 1e-6:
            return z0
        t = float(np.clip(s / L, 0.0, 1.0))
        return z0 + t * (z1 - z0)


def ground_field_for_building(
    xyz: np.ndarray,
    ring_m: np.ndarray,
    classification: Optional[np.ndarray] = None,
    search_radius_m: float = 8.0,
    collar_scale: float = 1.35,
    fallback_ground_z: float = 0.0,
) -> GroundField:
    """Compute per-vertex ground elevations around a footprint from LiDAR ground returns.

    Samples ASPRS class 2 ground returns (or returns in an outer collar below the roof line)
    near each footprint vertex. Returns a GroundField that continuously interpolates
    ground elevation along each edge so wall baselines hug the terrain.
    """
    from matplotlib.path import Path as MplPath
    from scipy.spatial import cKDTree

    ring = np.asarray(ring_m, dtype=np.float64)[:, :2]
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[:1]])
    N = len(ring) - 1
    if N < 3 or not len(xyz):
        return GroundField(vertex_z=np.full(len(ring), fallback_ground_z), ring_m=ring)

    cls = np.asarray(classification) if classification is not None else None
    has_cls2 = (cls is not None) and ((cls == 2).sum() >= 20)

    if has_cls2:
        ground_pts = xyz[cls == 2]
    else:
        centre = ring[:-1].mean(axis=0)
        inner = MplPath(ring[:-1])
        outer = MplPath(centre + (ring[:-1] - centre) * collar_scale)
        in_collar = outer.contains_points(xyz[:, :2]) & ~inner.contains_points(xyz[:, :2])
        if in_collar.sum() >= 20:
            c_z = xyz[in_collar, 2]
            thresh = float(np.percentile(c_z, 40))
            ground_pts = xyz[in_collar & (xyz[:, 2] <= thresh + 1.0)]
        else:
            ground_pts = xyz[xyz[:, 2] <= fallback_ground_z + 2.5]

    if not len(ground_pts):
        return GroundField(vertex_z=np.full(len(ring), fallback_ground_z), ring_m=ring)

    tree = cKDTree(ground_pts[:, :2])
    dists, idxs = tree.query(ring[:-1], k=20, distance_upper_bound=search_radius_m)

    vertex_z = np.full(N, np.nan)
    for i in range(N):
        valid = dists[i] < search_radius_m
        if valid.any():
            near_z = ground_pts[idxs[i][valid], 2]
            vertex_z[i] = float(np.median(near_z))

    nan_mask = np.isnan(vertex_z)
    if nan_mask.all():
        vertex_z[:] = fallback_ground_z
    elif nan_mask.any():
        valid_indices = np.flatnonzero(~nan_mask)
        valid_vals = vertex_z[valid_indices]
        all_indices = np.arange(N)
        vertex_z = np.interp(
            all_indices,
            valid_indices,
            valid_vals,
            period=N
        )

    vertex_z_closed = np.append(vertex_z, vertex_z[0])
    return GroundField(vertex_z=vertex_z_closed, ring_m=ring)


def ray_to_wall_planes(C: np.ndarray, ray_dirs: np.ndarray,
                       planes: List[WallPlane]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where each ray from ``C`` first meets a wall, approaching from outside.

    Returns ``(points (N, 3), hit (N,), face (N,))``; ``points`` is NaN and
    ``face`` is -1 where a ray misses every wall. A ray meets a wall only where
    it crosses the plane inside the edge's extent and between ``z_lo`` and
    ``z_hi``, travelling against the wall's outward normal -- a ray leaving the
    building through its far wall is not a hit. Of the walls a ray does meet,
    the nearest is the one it sees.
    """
    C = np.asarray(C, dtype=np.float64)
    d = np.asarray(ray_dirs, dtype=np.float64)
    n = len(d)
    best_t = np.full(n, np.inf)
    best_face = np.full(n, -1, dtype=int)
    for pl in planes:
        # Signed distance of C from the plane, and the ray's speed towards it.
        denom = d[:, 0] * pl.normal[0] + d[:, 1] * pl.normal[1]
        dist = (C[0] - pl.p0[0]) * pl.normal[0] + (C[1] - pl.p0[1]) * pl.normal[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(denom < -1e-12, -dist / denom, np.inf)
        ok = (t > 0) & np.isfinite(t) & (dist > 0)          # outside, moving in
        x = C[0] + t * d[:, 0]
        y = C[1] + t * d[:, 1]
        z = C[2] + t * d[:, 2]
        seg = pl.p1 - pl.p0
        along = ((x - pl.p0[0]) * seg[0] + (y - pl.p0[1]) * seg[1]) / max(float(seg @ seg), 1e-12)
        ok &= (along >= 0.0) & (along <= 1.0) & (z >= pl.z_lo) & (z <= pl.z_hi)
        better = ok & (t < best_t)
        best_t[better] = t[better]
        best_face[better] = pl.face
    hit = np.isfinite(best_t)
    pts = np.full((n, 3), np.nan)
    pts[hit] = C[None, :] + best_t[hit, None] * d[hit]
    return pts, hit, best_face


@dataclass
class OverhangShadow:
    """Where a wall can never be reached from the air, predicted from above.

    An airborne scanner looks down. Anything set back beneath a projection is
    in that projection's shadow and returns nothing -- not because the flight
    missed it, but because no flight can reach it. A crown wider than the shaft
    below it, a cornice, piers standing proud of a recessed curtain: each casts
    a band of wall beneath it that stays dark from every airborne angle.

    The scanner is not exactly vertical, so the band is finite. A ray coming
    down ``max_off_nadir_deg`` from vertical clears an overhang of depth ``d``
    and lands ``d / tan(angle)`` below it; the wall between is shadowed. At 20
    degrees that is 2.75 m of wall per metre of overhang.

    The point is that the overhang itself is visible from above -- it is roof
    and crown, the surfaces the scanner sees best -- so the walls it hides are
    predictable **before** anything is attempted on them, from returns the
    survey already has.

    ``cells`` is ``(n_up, n_along)`` over the face; ``shadow_depth_m`` is how
    far the nearest occluder above each cell projects past the wall, or 0.
    """

    face: int
    along_edges_m: np.ndarray
    up_edges_m: np.ndarray
    shadowed: np.ndarray            # bool (n_up, n_along)
    shadow_depth_m: np.ndarray      # float (n_up, n_along)
    n_returns: np.ndarray           # int (n_up, n_along), wall returns observed

    @property
    def shadowed_frac(self) -> float:
        return float(self.shadowed.mean()) if self.shadowed.size else 0.0


def predict_overhang_shadow(
    xyz: np.ndarray,
    wall_mask: np.ndarray,
    vertices_m: np.ndarray,
    height_m: float,
    max_off_nadir_deg: float = 20.0,
    cell_m: float = 3.0,
    min_overhang_m: float = 1.5,
    min_length_m: float = 3.0,
    max_offset_m: float = 2.0,
    min_occluder_height_m: float = 3.0,
) -> List[OverhangShadow]:
    """Predict overhang-shadowed wall per face, from the building's own returns.

    ``xyz`` is every return on the building (roof, crown, wall); ``wall_mask``
    marks the wall zone from :func:`split_surface_zones` and is used only to
    record what was actually observed against the prediction.

    A cell at height ``u`` on a face is shadowed if some return above it, at
    height ``u'`` and standing ``o'`` proud of the wall plane, satisfies
    ``u' - u <= o' / tan(max_off_nadir_deg)`` -- the steepest usable ray that
    clears that return lands at or below the cell. Returns behind the plane
    (``o' <= min_overhang_m``) shadow nothing: a setback hides no wall.

    ``min_overhang_m`` is 1.5 m, not the 0.5 m first tried. The wall's own
    depth is taken per column, and on a plain curtain wall those medians
    scatter by half a metre with mullions and window reveals -- at 0.5 m every
    column read as overhung by its neighbours, and 400 West Market came out a
    quarter shadowed with its walls fully captured. A cornice, balcony, podium
    or pier is a metre and a half or more. ``min_occluder_height_m`` keeps
    street clutter from shadowing the base.
    """
    v = np.asarray(vertices_m, dtype=np.float64)[:, :2]
    if len(v) > 1 and np.allclose(v[0], v[-1]):
        v = v[:-1]
    if len(v) < 3 or height_m <= 0:
        return []
    if np.sum((np.roll(v[:, 0], -1) - v[:, 0]) * (np.roll(v[:, 1], -1) + v[:, 1])) > 0:
        v = v[::-1]
    xyz = np.asarray(xyz, dtype=np.float64)
    z0 = float(xyz[:, 2].min())
    reach = 1.0 / np.tan(np.radians(max_off_nadir_deg))

    out: List[OverhangShadow] = []
    for i in range(len(v)):
        p0, p1 = v[i], v[(i + 1) % len(v)]
        seg = p1 - p0
        L = float(np.hypot(*seg))
        if L < min_length_m:
            continue
        e1 = seg / L
        n_out = np.array([e1[1], -e1[0]])            # CCW ring: outward is right
        rel = xyz[:, :2] - p0
        along = rel @ e1
        o = rel @ n_out                              # signed, + is outward
        up = xyz[:, 2] - z0

        # Only returns within this face's along-span matter to it.
        on_face = (along >= -max_offset_m) & (along <= L + max_offset_m)
        a_edges = np.arange(0.0, L + cell_m, cell_m)
        u_edges = np.arange(0.0, height_m + cell_m, cell_m)
        na, nu = len(a_edges) - 1, len(u_edges) - 1
        shadowed = np.zeros((nu, na), dtype=bool)
        depth = np.zeros((nu, na))
        n_ret = np.zeros((nu, na), dtype=int)

        # Observed wall returns per cell, for the comparison.
        w = on_face & wall_mask & (np.abs(o) <= max_offset_m)
        if w.any():
            ai = np.clip(((along[w]) // cell_m).astype(int), 0, na - 1)
            ui = np.clip(((up[w]) // cell_m).astype(int), 0, nu - 1)
            np.add.at(n_ret, (ui, ai), 1)

        # The wall's own depth, per column along the face. Measured against
        # the footprint plane, a recess is invisible: the piers ARE the plane
        # and the curtain between them sits behind it, so nothing reads as
        # proud. Measured against where that column's wall actually is, the
        # piers stand out from the curtain by exactly the recess depth, and
        # they shadow it the way a cornice shadows a wall. Columns with no wall
        # returns fall back to the plane, which is the footprint's claim.
        o_ref = np.zeros(na)
        wall_here = on_face & wall_mask & (np.abs(o) <= 4.0 * max_offset_m)
        if wall_here.any():
            wa = np.clip((along[wall_here] // cell_m).astype(int), 0, na - 1)
            for j in range(na):
                sel = wa == j
                if sel.sum() >= 3:
                    o_ref[j] = float(np.median(o[wall_here][sel]))

        # Candidate occluders: anything standing proud of this column's wall.
        oa_all = np.clip((along // cell_m).astype(int), 0, na - 1)
        # Not the ends: within a cell of a corner the adjacent face's returns
        # sit well proud of this plane and read as an overhang running the
        # full height of the corner column.
        interior = (along >= cell_m) & (along <= L - cell_m)
        occ = (on_face & interior & ((o - o_ref[oa_all]) > min_overhang_m)
               & (up >= min_occluder_height_m))
        if occ.any():
            oa = oa_all[occ]
            ou, oo = up[occ], (o - o_ref[oa_all])[occ]
            u_mid = (u_edges[:-1] + u_edges[1:]) / 2.0
            for j in range(na):
                sel = oa == j
                if not sel.any():
                    continue
                # For each cell height, the deepest occluder above it that is
                # close enough above to shadow it.
                du = ou[sel][None, :] - u_mid[:, None]          # (nu, k)
                hits = (du > 0) & (du <= oo[sel][None, :] * reach)
                if hits.any():
                    d = np.where(hits, oo[sel][None, :], 0.0).max(axis=1)
                    shadowed[:, j] = d > 0
                    depth[:, j] = d
        out.append(OverhangShadow(face=i, along_edges_m=a_edges, up_edges_m=u_edges,
                                  shadowed=shadowed, shadow_depth_m=depth,
                                  n_returns=n_ret))
    return out
