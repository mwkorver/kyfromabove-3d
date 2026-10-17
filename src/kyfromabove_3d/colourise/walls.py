"""Wall points made where the LiDAR has few: walls resampled on the outline, and faces
extruded from their verified planes or their profiles."""

from __future__ import annotations

from typing import Optional

import numpy as np

from kyfromabove_3d.features import (
    VerifiedWallPlane,
    WallPlane,
    facade_faces,
    infer_prism_faces,
    verify_face_plane,
)


def resample_walls(xyz, wall_mask, ring_m, height_m, spacing_m, min_face_m,
                   max_offset_m, tile_m=12.0, max_rms_m=0.35, min_points=40, skip_faces=()):
    """New points on locally fitted wall planes, at imagery resolution.

    Depth comes from the planes the LiDAR defines, never from the imagery.

    **Fitted per tile, not per face.** One plane across a whole 60 m wall is too
    coarse a model for a real building: measured here, all four of the Humana
    Building's faces fit a single plane at 0.58-1.22 m, because Graves put
    setbacks, a portico and a cantilevered belvedere on them, and one of PNC
    Tower's two large faces missed a 0.5 m cut by 0.064 m. Fitting ``tile_m``
    squares instead lets the surface step where the building steps, and the
    residual then measures the wall's own texture rather than its shape.

    A tile whose returns still do not lie on a plane is skipped rather than
    extrapolated. Where a facade is genuinely not planar, this method has
    nothing to offer and should decline.
    """
    faces = facade_faces(ring_m, height_m=height_m, wall_xy=xyz[wall_mask][:, :2],
                         min_length_m=min_face_m, max_offset_m=max_offset_m)
    # an edge of a curved wall is profiled with its arc, not fitted as a plane
    faces = [f for f in faces if f.index not in set(skip_faces)]
    wall_xyz = xyz[wall_mask]
    v = np.asarray(ring_m, dtype=np.float64)[:, :2]
    if np.allclose(v[0], v[-1]):
        v = v[:-1]
    if np.sum((np.roll(v[:, 0], -1) - v[:, 0]) * (np.roll(v[:, 1], -1) + v[:, 1])) > 0:
        v = v[::-1]

    out_xyz, out_n, out_face, stats = [], [], [], []
    tiles_by_face: dict[int, list[dict]] = {}
    for f in faces:
        if not f.captured:
            continue
        p0, p1 = v[f.index], v[(f.index + 1) % len(v)]
        seg = p1 - p0
        L = np.hypot(*seg)
        t = np.clip(((wall_xyz[:, :2] - p0) @ seg) / max(L ** 2, 1e-9), 0.0, 1.0)
        closest = p0 + t[:, None] * seg
        pts = wall_xyz[np.linalg.norm(wall_xyz[:, :2] - closest, axis=1) <= max_offset_m]
        if height_m is not None and height_m > 0:
            pts = pts[pts[:, 2] <= (pts[:, 2].min() + height_m + 1.0)]
        if len(pts) < min_points:
            continue

        # Nominal frame for the face: along the wall, and up it.
        e1 = np.array([seg[0] / L, seg[1] / L, 0.0])
        e2 = np.array([0.0, 0.0, 1.0])
        origin = np.array([p0[0], p0[1], pts[:, 2].min()])
        rel = pts - origin
        a, b = rel @ e1, rel @ e2
        kept = skipped = 0
        for ai in np.arange(a.min(), a.max(), tile_m):
            for bi in np.arange(b.min(), b.max(), tile_m):
                m = (a >= ai) & (a < ai + tile_m) & (b >= bi) & (b < bi + tile_m)
                if m.sum() < min_points:
                    continue
                tp = pts[m]
                centroid = tp.mean(axis=0)
                normal = np.linalg.svd(tp - centroid, full_matrices=False)[2][2]
                rms = float(np.sqrt(np.mean(((tp - centroid) @ normal) ** 2)))
                if rms > max_rms_m:
                    skipped += 1
                    continue
                outward = np.array([np.sin(np.radians(f.bearing_deg)),
                                    np.cos(np.radians(f.bearing_deg)), 0.0])
                if normal @ outward < 0:
                    normal = -normal
                t1 = e1 - (e1 @ normal) * normal
                t1 /= np.linalg.norm(t1)
                t2 = np.cross(normal, t1)
                if t2[2] < 0:
                    t2 = -t2
                r = tp - centroid
                ga = np.arange((r @ t1).min(), (r @ t1).max() + spacing_m, spacing_m)
                gb = np.arange((r @ t2).min(), (r @ t2).max() + spacing_m, spacing_m)
                if len(ga) < 2 or len(gb) < 2 or len(ga) * len(gb) > 1_000_000:
                    continue
                # The tile as a rectangle in its own plane, for a mesh to use.
                tiles_by_face.setdefault(f.index, []).append({
                    "centroid": centroid, "normal": normal, "rms": rms,
                    "t1": t1, "t2": t2,
                    "a_range": (float(ga[0]), float(ga[-1])), "b_range": (float(gb[0]), float(gb[-1])),
                })
                A, B = np.meshgrid(ga, gb)
                grid = centroid + A.ravel()[:, None] * t1 + B.ravel()[:, None] * t2
                out_xyz.append(grid)
                out_n.append(np.repeat(normal[None, :], len(grid), axis=0))
                out_face.append(np.full(len(grid), f.index))
                kept += 1
        stats.append((f.index, f.length_m, len(pts), kept, skipped))

    if not out_xyz:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, dtype=int), faces, stats, tiles_by_face
    return (np.vstack(out_xyz), np.vstack(out_n),
            np.concatenate(out_face), faces, stats, tiles_by_face)


def extrude_verified_faces(
    nominal_planes: list[WallPlane],
    tiles_by_face: dict[int, list[dict]],
    spacing_m: float,
    tile_m: float = 12.0,
    max_offset_rms_m: float = 0.30,
    max_normal_dev_deg: float = 4.0,
    roof_z: Optional[float] = None,
    prism_completion: bool = True,
    min_prism_ratio: float = 0.60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, VerifiedWallPlane]]:
    """Extrude verified vertical wall planes and de-duplicate against fitted tiles.

    If ``prism_completion`` is True and verified faces account for at least
    ``min_prism_ratio`` of the nominal perimeter, unverified connecting walls
    are completed as planar prism faces bridging the verified walls.

    Returns:
        (ext_xyz, ext_n, ext_face, verified_planes)
    """
    ext_xyz, ext_n, ext_face = [], [], []
    verified_planes: dict[int, VerifiedWallPlane] = {}

    for plane in nominal_planes:
        face_idx = plane.face
        raw_tiles = tiles_by_face.get(face_idx, [])
        # Only consider tiles facing in the nominal outward wall direction,
        # filtering out horizontal ledges, canopies, or floor slabs captured near the facade.
        n_out_3d = np.array([plane.normal[0], plane.normal[1], 0.0])
        tiles = [t for t in raw_tiles if t["normal"] @ n_out_3d > 0.7]
        if len(tiles) < 2:
            continue
        centroids = np.array([t["centroid"] for t in tiles], dtype=np.float64)
        normals = np.array([t["normal"] for t in tiles], dtype=np.float64)

        verified = verify_face_plane(
            plane=plane,
            tile_centroids=centroids,
            tile_normals=normals,
            max_offset_rms_m=max_offset_rms_m,
            max_normal_dev_deg=max_normal_dev_deg,
            min_tiles=2,
            tile_m=tile_m,
            roof_z=roof_z,
        )
        if verified is not None:
            verified_planes[face_idx] = verified

    if prism_completion:
        inferred = infer_prism_faces(
            nominal_planes=nominal_planes,
            verified_planes=verified_planes,
            min_perimeter_ratio=min_prism_ratio,
            roof_z=roof_z,
        )
        verified_planes.update(inferred)

    for face_idx, verified in verified_planes.items():
        raw_tiles = tiles_by_face.get(face_idx, [])
        n_out_3d = np.array([verified.normal[0], verified.normal[1], 0.0])
        tiles = [t for t in raw_tiles if t["normal"] @ n_out_3d > 0.7]

        p0, p1 = verified.p0, verified.p1
        seg = p1 - p0
        L = float(np.hypot(seg[0], seg[1]))
        if L < 1e-6:
            continue
        e1 = seg / L
        n_out = verified.normal

        s_vals = np.arange(0.0, L, spacing_m)
        z_vals = np.arange(verified.z_lo, verified.z_hi, spacing_m)
        if len(s_vals) == 0 or len(z_vals) == 0:
            continue

        S, Z = np.meshgrid(s_vals, z_vals)
        S_flat = S.ravel()
        Z_flat = Z.ravel()

        keep = np.ones(len(S_flat), dtype=bool)
        half_tile = tile_m / 2.0
        for t in tiles:
            c_t = t["centroid"]
            s_t = float((c_t[:2] - p0) @ e1)
            z_t = float(c_t[2])
            in_tile = (np.abs(S_flat - s_t) < half_tile) & (np.abs(Z_flat - z_t) < half_tile)
            keep[in_tile] = False

        if not keep.any():
            continue

        S_keep = S_flat[keep]
        Z_keep = Z_flat[keep]

        pts_xy = p0[None, :] + S_keep[:, None] * e1[None, :]
        pts_3d = np.column_stack([pts_xy, Z_keep])
        pts_norm = np.tile(np.array([n_out[0], n_out[1], 0.0]), (len(pts_3d), 1))
        pts_face = np.full(len(pts_3d), face_idx, dtype=int)

        ext_xyz.append(pts_3d)
        ext_n.append(pts_norm)
        ext_face.append(pts_face)

    if not ext_xyz:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, dtype=int), verified_planes
    return (np.vstack(ext_xyz), np.vstack(ext_n),
            np.concatenate(ext_face), verified_planes)


def under_measured_edge(planes, is_arc, level_of, tower_levels, measured_edges, max_off_m: float = 1.5,
                        max_angle_deg: float = 10.0, min_cover: float = 0.6) -> set:
    """The faces standing under a roof edge the LiDAR measured: every face of
    a level whose outline is the LiDAR's tower edge, and the ground level's
    faces lying along (within ``max_off_m`` and ``max_angle_deg``) outline
    edges whose roof end it found, over at least ``min_cover`` of their length."""
    out = set()
    for pl in planes:
        if is_arc[pl.face]:
            continue
        lv = level_of[pl.face]
        if lv in tower_levels:
            out.add(pl.face)
            continue
        if lv != 0 or not measured_edges:
            continue
        L = pl.length_m
        e1 = (pl.p1 - pl.p0) / max(L, 1e-9)
        covered = []
        for q0, q1 in measured_edges:
            d = q1[:2] - q0[:2]
            Lq = float(np.hypot(*d))
            if Lq < 1e-6 or abs(float(np.dot(d / Lq, e1))) < np.cos(np.radians(max_angle_deg)):
                continue
            mid = 0.5 * (q0[:2] + q1[:2])
            if abs(float(np.dot(mid - pl.p0, pl.normal))) > max_off_m:
                continue
            a, b = sorted((float(np.dot(q0[:2] - pl.p0, e1)), float(np.dot(q1[:2] - pl.p0, e1))))
            if min(b, L) > max(a, 0.0):
                covered.append((max(a, 0.0), min(b, L)))
        covered.sort()
        tot, end = 0.0, 0.0
        for a, b in covered:
            if b > end:
                tot += b - max(a, end)
                end = b
        if tot >= min_cover * L:
            out.add(pl.face)
    return out
