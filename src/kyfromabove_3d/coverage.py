"""Facade coverage grids: measuring fillable gaps and photo illumination.

Decouples coverage measurement from visualization. Computes per-face 2D cell
grids tracking which parts of a building's envelope have LiDAR returns, which
have been coloured by aerial imagery or photos, and which resampled wall cells
remain uncoloured (the fillable deficit).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FaceCoverage:
    """Coverage state of a single planar wall face."""

    face: int
    bearing_deg: float
    along_edges_m: np.ndarray
    up_edges_m: np.ndarray
    n_lidar: np.ndarray        # (n_up, n_along) count of physical LiDAR returns
    n_coloured: np.ndarray     # (n_up, n_along) count of coloured resampled points
    n_uncoloured: np.ndarray   # (n_up, n_along) count of uncoloured resampled points
    n_from_photo: np.ndarray   # (n_up, n_along) count of points taking colour from photos
    n_extruded: np.ndarray | None = None  # (n_up, n_along) count of extruded points
    n_tile: np.ndarray | None = None      # (n_up, n_along) count of fitted tile points
    plane_verified: bool = False
    plane_inferred: bool = False
    verified_rms_m: float | None = None
    verified_yaw_deg: float | None = None
    verified_offset_m: float | None = None

    @property
    def fillable(self) -> np.ndarray:
        """Boolean mask of cells with resampled points no camera has coloured yet."""
        return (self.n_coloured + self.n_uncoloured) > 0

    @property
    def unfillable(self) -> np.ndarray:
        """Boolean mask of cells with no resampled point of any kind."""
        return (self.n_coloured + self.n_uncoloured) == 0

    @property
    def n_fillable_cells(self) -> int:
        return int(self.fillable.sum())

    @property
    def n_unfillable_cells(self) -> int:
        return int(self.unfillable.sum())

    @property
    def n_photo_cells(self) -> int:
        return int((self.n_from_photo > 0).sum())

    @property
    def n_extruded_cells(self) -> int:
        if self.n_extruded is None:
            return 0
        return int((self.n_extruded > 0).sum())

    @property
    def n_tile_cells(self) -> int:
        if self.n_tile is None:
            return 0
        return int((self.n_tile > 0).sum())

    def to_dict(self) -> dict[str, Any]:
        d = {
            "face": self.face,
            "bearing_deg": float(self.bearing_deg),
            "n_along": int(len(self.along_edges_m) - 1),
            "n_up": int(len(self.up_edges_m) - 1),
            "n_lidar_total": int(self.n_lidar.sum()),
            "n_coloured_total": int(self.n_coloured.sum()),
            "n_uncoloured_total": int(self.n_uncoloured.sum()),
            "n_from_photo_total": int(self.n_from_photo.sum()),
            "n_fillable_cells": self.n_fillable_cells,
            "n_unfillable_cells": self.n_unfillable_cells,
            "n_photo_cells": self.n_photo_cells,
            "n_extruded_cells": self.n_extruded_cells,
            "n_tile_cells": self.n_tile_cells,
            "plane_verified": bool(self.plane_verified),
            "plane_inferred": bool(self.plane_inferred),
        }
        if self.verified_rms_m is not None:
            d["verified_rms_m"] = float(self.verified_rms_m)
        if self.verified_yaw_deg is not None:
            d["verified_yaw_deg"] = float(self.verified_yaw_deg)
        if self.verified_offset_m is not None:
            d["verified_offset_m"] = float(self.verified_offset_m)
        return d


def facade_coverage(
    resampled_xyz: np.ndarray,
    coloured_mask: np.ndarray,
    from_photo_mask: np.ndarray,
    lidar_wall_xyz: np.ndarray,
    ring_m: np.ndarray,
    height_m: float,
    cell_m: float = 3.0,
    min_face_length_m: float = 8.0,
    max_offset_m: float = 4.0,
    surface_mask: Optional[np.ndarray] = None,
    verified_planes: Optional[dict[int, Any]] = None,
) -> list[FaceCoverage]:
    """Measure cell-by-cell coverage grids for each facade of a building.

    Args:
        resampled_xyz: (N, 3) resampled wall point cloud.
        coloured_mask: (N,) bool mask of resampled points coloured by any camera.
        from_photo_mask: (N,) bool mask of resampled points coloured by photos (source=2).
        lidar_wall_xyz: (M, 3) raw physical LiDAR wall returns.
        ring_m: (K, 2) building footprint polygon in local metres.
        height_m: building height in metres.
        cell_m: grid cell side in metres.
        min_face_length_m: minimum edge length to treat as a distinct facade face.
        max_offset_m: perpendicular distance from face plane to assign points.
        surface_mask: (N,) optional uint8 array of surface types (1=tile, 3=extruded).
        verified_planes: optional dict mapping face index to VerifiedWallPlane.

    Returns:
        List of FaceCoverage instances, one per qualifying footprint face.
    """
    v = np.asarray(ring_m, dtype=np.float64)[:, :2]
    if len(v) > 1 and np.allclose(v[0], v[-1]):
        v = v[:-1]
    if len(v) < 3 or height_m <= 0:
        return []

    # Ensure counter-clockwise orientation so outward normal is to the right
    signed_area = np.sum((np.roll(v[:, 0], -1) - v[:, 0]) * (np.roll(v[:, 1], -1) + v[:, 1]))
    if signed_area > 0:
        v = v[::-1]

    n_vertices = len(v)
    ground_z = float(resampled_xyz[:, 2].min()) if len(resampled_xyz) else 0.0
    if len(lidar_wall_xyz):
        ground_z = min(ground_z, float(lidar_wall_xyz[:, 2].min()))

    resampled_uncoloured_mask = ~coloured_mask

    coverages: list[FaceCoverage] = []

    for i in range(n_vertices):
        p0 = v[i]
        p1 = v[(i + 1) % n_vertices]
        seg = p1 - p0
        L = float(np.hypot(seg[0], seg[1]))
        if L < min_face_length_m:
            continue

        e1 = seg / L
        n_out = np.array([e1[1], -e1[0]])
        bearing = float(np.degrees(np.arctan2(n_out[0], n_out[1])) % 360.0)

        along_edges = np.arange(0.0, L + cell_m, cell_m)
        up_edges = np.arange(0.0, height_m + cell_m, cell_m)
        na = len(along_edges) - 1
        nu = len(up_edges) - 1
        if na <= 0 or nu <= 0:
            continue

        grid_lidar = np.zeros((nu, na), dtype=int)
        grid_coloured = np.zeros((nu, na), dtype=int)
        grid_uncoloured = np.zeros((nu, na), dtype=int)
        grid_photo = np.zeros((nu, na), dtype=int)
        grid_extruded = np.zeros((nu, na), dtype=int) if surface_mask is not None else None
        grid_tile = np.zeros((nu, na), dtype=int) if surface_mask is not None else None

        # 1. Bin LiDAR returns
        if len(lidar_wall_xyz):
            rel_l = lidar_wall_xyz[:, :2] - p0
            along_l = rel_l @ e1
            off_l = rel_l @ n_out
            up_l = lidar_wall_xyz[:, 2] - ground_z
            on_l = (along_l >= 0) & (along_l <= L) & (np.abs(off_l) <= max_offset_m) & (up_l >= 0) & (up_l <= height_m)
            if on_l.any():
                ai = np.clip(np.searchsorted(along_edges, along_l[on_l], side="right") - 1, 0, na - 1)
                ui = np.clip(np.searchsorted(up_edges, up_l[on_l], side="right") - 1, 0, nu - 1)
                np.add.at(grid_lidar, (ui, ai), 1)

        # 2. Bin resampled points (coloured, uncoloured, from photo, extruded, tile)
        if len(resampled_xyz):
            rel_r = resampled_xyz[:, :2] - p0
            along_r = rel_r @ e1
            off_r = rel_r @ n_out
            up_r = resampled_xyz[:, 2] - ground_z
            on_r = (along_r >= 0) & (along_r <= L) & (np.abs(off_r) <= max_offset_m) & (up_r >= 0) & (up_r <= height_m)

            if on_r.any():
                idx_on = np.flatnonzero(on_r)
                ai_r = np.clip(np.searchsorted(along_edges, along_r[on_r], side="right") - 1, 0, na - 1)
                ui_r = np.clip(np.searchsorted(up_edges, up_r[on_r], side="right") - 1, 0, nu - 1)

                is_col = coloured_mask[idx_on]
                is_uncol = resampled_uncoloured_mask[idx_on]
                is_photo = from_photo_mask[idx_on]

                if is_col.any():
                    np.add.at(grid_coloured, (ui_r[is_col], ai_r[is_col]), 1)
                if is_uncol.any():
                    np.add.at(grid_uncoloured, (ui_r[is_uncol], ai_r[is_uncol]), 1)
                if is_photo.any():
                    np.add.at(grid_photo, (ui_r[is_photo], ai_r[is_photo]), 1)

                if surface_mask is not None:
                    surf = surface_mask[idx_on]
                    is_ext = (surf == 3)
                    is_til = (surf == 1)
                    if is_ext.any():
                        np.add.at(grid_extruded, (ui_r[is_ext], ai_r[is_ext]), 1)
                    if is_til.any():
                        np.add.at(grid_tile, (ui_r[is_til], ai_r[is_til]), 1)

        vp = verified_planes.get(i) if verified_planes else None
        coverages.append(FaceCoverage(
            face=i,
            bearing_deg=bearing,
            along_edges_m=along_edges,
            up_edges_m=up_edges,
            n_lidar=grid_lidar,
            n_coloured=grid_coloured,
            n_uncoloured=grid_uncoloured,
            n_from_photo=grid_photo,
            n_extruded=grid_extruded,
            n_tile=grid_tile,
            plane_verified=(vp is not None),
            plane_inferred=(vp.inferred if vp else False),
            verified_rms_m=vp.rms_m if vp else None,
            verified_yaw_deg=vp.yaw_dev_deg if vp else None,
            verified_offset_m=vp.offset_m if vp else None,
        ))

    return coverages
