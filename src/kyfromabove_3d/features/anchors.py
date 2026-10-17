"""Planar anchors and check features: points on surfaces the LiDAR shows, for the
reprojection checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .planes import (
    _outward_normal,
    facade_tangent_basis,
    fit_local_planes,
    tangent_basis,
)


# -------------------------------------------------------------------- anchors
@dataclass
class PlanarAnchor:
    """A LiDAR point on a locally planar surface, with its tangent frame."""

    xyz_local_m: np.ndarray
    normal: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    planarity_rms_m: float
    height_above_ground_m: float
    label: str = "surface"
    anchor_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "label": self.label,
            "xyz_local_m": [float(v) for v in self.xyz_local_m],
            "normal": [float(v) for v in self.normal],
            "planarity_rms_m": float(self.planarity_rms_m),
            "height_above_ground_m": float(self.height_above_ground_m),
        }


def ground_surface(
    xyz: np.ndarray, classification: np.ndarray, cell_m: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Coarse gridded ground model from ASPRS class-2 returns.

    Returns ``(grid, x_edges, y_edges, fallback_z)``; empty cells hold NaN.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    ground = xyz[np.asarray(classification) == 2]
    if len(ground) == 0:
        ground = xyz

    x_edges = np.arange(xyz[:, 0].min(), xyz[:, 0].max() + cell_m, cell_m)
    y_edges = np.arange(xyz[:, 1].min(), xyz[:, 1].max() + cell_m, cell_m)
    grid = np.full((len(y_edges) - 1, len(x_edges) - 1), np.nan)

    ix = np.clip(np.digitize(ground[:, 0], x_edges) - 1, 0, grid.shape[1] - 1)
    iy = np.clip(np.digitize(ground[:, 1], y_edges) - 1, 0, grid.shape[0] - 1)
    for gy in range(grid.shape[0]):
        for gx in range(grid.shape[1]):
            sel = (ix == gx) & (iy == gy)
            if sel.any():
                grid[gy, gx] = np.median(ground[sel, 2])

    return grid, x_edges, y_edges, float(np.median(ground[:, 2]))


def height_above_ground(
    xyz: np.ndarray, grid: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, fallback_z: float
) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    ix = np.clip(np.digitize(xyz[:, 0], x_edges) - 1, 0, grid.shape[1] - 1)
    iy = np.clip(np.digitize(xyz[:, 1], y_edges) - 1, 0, grid.shape[0] - 1)
    base = grid[iy, ix]
    base = np.where(np.isnan(base), fallback_z, base)
    return xyz[:, 2] - base


def build_planar_anchors(
    lidar,
    cell_m: float = 8.0,
    max_planarity_rms_m: float = 0.08,
    min_normal_z: float = 0.80,
    per_cell: int = 2,
    k: int = 16,
    include_facades: bool = False,
    facade_max_normal_z: float = 0.35,
    facade_max_planarity_rms_m: float = 0.20,
    facade_k: int = 8,
) -> List[PlanarAnchor]:
    """Select well-conditioned, spatially spread anchors from a LiDAR cloud.

    Anchors are the 3D input to cross-view matching, so they must sit on
    surfaces that are genuinely planar (a patch rectified through a wrong plane
    would decorrelate for reasons unrelated to camera geometry). Selection is
    gridded so anchors cover the building rather than clustering on whichever
    area happens to be densest.

    The defaults describe a **horizontal** building: ``min_normal_z = 0.80``
    admits only surfaces within ~37 deg of level, and cells are formed on
    ``(x, y)``. On a tower that yields almost nothing -- 24 anchors inside
    400 West Market's volume, all on its crown, none on 165 m of facade -- so
    the reported residuals describe its neighbours rather than the building.

    ``include_facades`` adds a second pass for near-vertical surfaces. It needs
    its own parameters, all measured on the tower's shaft rather than guessed:

    * ``facade_max_planarity_rms_m`` 0.20 m, against 0.08 for roofs. A facade is
      locally flat but is sampled at grazing incidence, so its returns are
      noisier.
    * ``facade_k`` 8, against 16. A facade curves around the building, so a wide
      neighbourhood straddles corners: median local RMS is 0.104 m at k=8 and
      0.211 m at k=32.
    * **3D cell keys.** Every facade point up a tower lands in the same
      ``(x, y)`` cell, so a planimetric grid would return ``per_cell`` anchors
      for a whole column regardless of its height.
    """
    xyz = np.asarray(lidar.xyz_local_m, dtype=np.float64)
    cls = np.asarray(lidar.classification)

    grid, x_edges, y_edges, fallback_z = ground_surface(xyz, cls)
    hag = height_above_ground(xyz, grid, x_edges, y_edges, fallback_z)

    normals, residuals, n_used = fit_local_planes(xyz, k=k)

    eligible = (
        (residuals <= max_planarity_rms_m)
        & (normals[:, 2] >= min_normal_z)
        & (n_used >= 8)
        & (cls != 7)  # drop low-noise returns
    )

    anchors: List[PlanarAnchor] = []
    if not eligible.any():
        return anchors

    gx = np.floor((xyz[:, 0] - xyz[:, 0].min()) / cell_m).astype(int)
    gy = np.floor((xyz[:, 1] - xyz[:, 1].min()) / cell_m).astype(int)
    keys = gx.astype(np.int64) * 100000 + gy

    for key in np.unique(keys[eligible]):
        sel = np.where(eligible & (keys == key))[0]
        # Flattest first.
        sel = sel[np.argsort(residuals[sel])][:per_cell]
        for i in sel:
            e1, e2 = tangent_basis(normals[i])
            label = "roof" if hag[i] > 2.0 else "ground"
            anchors.append(
                PlanarAnchor(
                    xyz_local_m=xyz[i].copy(),
                    normal=normals[i].copy(),
                    e1=e1,
                    e2=e2,
                    planarity_rms_m=float(residuals[i]),
                    height_above_ground_m=float(hag[i]),
                    label=label,
                    anchor_id=f"A{len(anchors):04d}",
                )
            )

    if include_facades:
        f_normals, f_residuals, f_used = fit_local_planes(xyz, k=facade_k)
        vertical = (
            (np.abs(f_normals[:, 2]) < facade_max_normal_z)
            & (f_residuals <= facade_max_planarity_rms_m)
            & (f_used >= 8)
            & (cls != 7)
        )
        if vertical.any():
            gz = np.floor((xyz[:, 2] - xyz[:, 2].min()) / cell_m).astype(int)
            vkeys = keys * 1000 + gz          # (x, y, z), so a column stratifies
            for key in np.unique(vkeys[vertical]):
                sel = np.where(vertical & (vkeys == key))[0]
                sel = sel[np.argsort(f_residuals[sel])][:per_cell]
                for i in sel:
                    # Orient the normal outward, away from the local mass of
                    # points. fit_local_planes returns an arbitrary sign, and a
                    # facade normal's sign is what says which side of the wall a
                    # camera has to be on to see it -- 70% of anchor/camera
                    # pairs on this tower are behind the wall, and attempting
                    # them produces guaranteed low_ncc rejections that read as
                    # failures of the method.
                    n = _outward_normal(xyz, i, f_normals[i])
                    e1, e2 = facade_tangent_basis(n)
                    anchors.append(
                        PlanarAnchor(
                            xyz_local_m=xyz[i].copy(),
                            normal=n,
                            e1=e1,
                            e2=e2,
                            planarity_rms_m=float(f_residuals[i]),
                            height_above_ground_m=float(hag[i]),
                            label="facade",
                            anchor_id=f"A{len(anchors):04d}",
                        )
                    )
    return anchors


# ------------------------------------------------------------- check features
@dataclass
class CheckFeature:
    """A named 3D point used for reporting and visual inspection.

    ``xyz_local_m`` must come from measured data (LiDAR), never from a literal
    typed into a script.
    """

    feature_id: str
    description: str
    xyz_local_m: Tuple[float, float, float]
    provenance: str = "lidar"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "description": self.description,
            "xyz_local_m": [float(v) for v in self.xyz_local_m],
            "provenance": self.provenance,
        }


def extract_roof_points(
    lidar,
    min_height_m: float = 3.0,
    max_height_m: Optional[float] = 15.0,
    max_planarity_rms_m: float = 0.15,
    min_normal_z: float = 0.90,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(roof_xyz, height_above_ground)`` for flat elevated surfaces.

    The tile carries no ASPRS class-6 (building) returns, so the roof is
    isolated geometrically: elevated, flat, and near-horizontal. That also
    rejects the tree canopy along the north edge of the AOI.

    ``max_height_m`` defaults to 15 m, which suits the single-storey pilot and
    **silently excludes anything taller**. A 165 m tower has no roof at all
    under that ceiling, and the function then returns whatever low structure
    shares the AOI -- a podium, or a neighbour -- with no indication that it
    picked the wrong building. Pass ``None`` for no ceiling, or set it from the
    config to the AOI's real vertical extent.
    """
    xyz = np.asarray(lidar.xyz_local_m, dtype=np.float64)
    cls = np.asarray(lidar.classification)
    grid, x_edges, y_edges, fallback_z = ground_surface(xyz, cls)
    hag = height_above_ground(xyz, grid, x_edges, y_edges, fallback_z)
    normals, residuals, n_used = fit_local_planes(xyz)

    sel = (
        (hag >= min_height_m)
        & (residuals <= max_planarity_rms_m)
        & (normals[:, 2] >= min_normal_z)
        & (n_used >= 8)
        & (cls != 7)
    )
    if max_height_m is not None:
        sel &= hag <= max_height_m
    return xyz[sel], hag[sel]


def _plane_fit_rms(xyz: np.ndarray) -> float:
    """RMS distance from a least-squares plane, in metres.

    The test that decides whether a rectangle is an honest description of this
    roof. Measured: the pilot's flat warehouse roof gives 0.192 m, 400 West
    Market's dome gives 1.496 m. The 0.5 m default sits well clear of both.
    """
    pts = np.asarray(xyz, dtype=np.float64)
    if len(pts) < 4:
        return 0.0
    centred = pts - pts.mean(axis=0)
    normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
    return float(np.sqrt(np.mean((centred @ normal) ** 2)))


def _local_surface_z(xy, roof_xyz: np.ndarray, radius_m: float) -> Optional[float]:
    """Median elevation of roof points near ``xy``, or None if none are near.

    Returning None rather than an interpolated guess is the point: a check
    feature has to sit on measured surface, so a position with no support is
    not emitted at all.
    """
    d = np.hypot(roof_xyz[:, 0] - xy[0], roof_xyz[:, 1] - xy[1])
    near = d <= radius_m
    if near.sum() < 5:
        return None
    return float(np.median(roof_xyz[near, 2]))


def _curved_roof_features(roof_xyz: np.ndarray, plane_rms: float) -> List[CheckFeature]:
    """Check features for a roof that is not a plane.

    Every feature is snapped to the **local** surface elevation rather than to
    one median, and any position without nearby returns is dropped. So these are
    measured points on the roof, which the rectangle corners would not be.
    """
    cx = float(np.median(roof_xyz[:, 0]))
    cy = float(np.median(roof_xyz[:, 1]))
    radius = float(np.percentile(np.hypot(roof_xyz[:, 0] - cx, roof_xyz[:, 1] - cy), 95))
    # Far enough out to span the curvature -- at 0.6 the samples sit near the
    # apex and barely differ from it -- while staying inside the measured
    # surface rather than off its edge.
    sample_r = 0.8 * radius
    support = max(0.15 * radius, 1.0)

    features: List[CheckFeature] = []
    apex_i = int(np.argmax(roof_xyz[:, 2]))
    apex_z = _local_surface_z(roof_xyz[apex_i, :2], roof_xyz, support)
    if apex_z is not None:
        features.append(CheckFeature(
            feature_id="RoofApex",
            description=f"Highest point of a curved roof "
                        f"(plane RMS {plane_rms:.2f} m over {radius:.1f} m radius)",
            xyz_local_m=(float(roof_xyz[apex_i, 0]), float(roof_xyz[apex_i, 1]), apex_z),
            provenance="lidar:apex",
        ))

    for name, (dx, dy) in {"N": (0.0, 1.0), "E": (1.0, 0.0),
                           "S": (0.0, -1.0), "W": (-1.0, 0.0)}.items():
        xy = (cx + dx * sample_r, cy + dy * sample_r)
        z = _local_surface_z(xy, roof_xyz, support)
        if z is None:
            continue
        features.append(CheckFeature(
            feature_id=f"RoofSurface_{name}",
            description=f"Curved roof surface {sample_r:.1f} m {name} of centre, "
                        f"at the local median elevation",
            xyz_local_m=(float(xy[0]), float(xy[1]), z),
            provenance="lidar:surface_sample",
        ))
    return features


def building_surface(
    lidar,
    link_m: float = 5.0,
    band_m: float = 20.0,
    max_extent_growth: float = 1.5,
    facade_max_normal_z: float = 0.35,
    facade_max_planarity_rms_m: float = 0.20,
    facade_k: int = 8,
) -> np.ndarray:
    """All surface points of one building: every roof level plus its facades.

    :func:`roof_check_features` selects a *single* cluster, which on a stepped
    tower is its crown. 400 West Market's crown is 15.5 x 15.8 m while the
    building is 49 x 49 m, so a footprint taken from it excludes the shaft --
    and with it 1,003 of 1,041 facade anchors, the crops' true width, and the
    region a building-restricted score should cover.

    A building is instead taken as one **3D connected component** over roof-
    and facade-eligible points, seeded from the highest point. Connectivity has
    to be 3D and it has to reach across the step: at a 3 m link radius the crown
    stays isolated at 473 points, and at 5 m the component runs the full height,
    z 1 to 165 m.

    The component does not stop at the building. Near the ground it merges
    through the plaza into neighbouring structures -- measured on the tower, the
    span goes from 49 m above z=20 to 79 m below it. So bands are accumulated
    from the top down and the walk **stops when the horizontal extent grows by
    more than** ``max_extent_growth``, which is where the building ends and its
    surroundings begin.
    """
    from scipy.spatial import cKDTree

    xyz = np.asarray(lidar.xyz_local_m, dtype=np.float64)
    cls = np.asarray(lidar.classification)

    roof_xyz, _ = extract_roof_points(lidar, max_height_m=None)
    normals, residuals, n_used = fit_local_planes(xyz, k=facade_k)
    facade = (
        (np.abs(normals[:, 2]) < facade_max_normal_z)
        & (residuals <= facade_max_planarity_rms_m)
        & (n_used >= 8)
        & (cls != 7)
    )
    roof_keys = {tuple(v) for v in np.round(roof_xyz, 3)}
    is_roof = np.array([tuple(v) in roof_keys for v in np.round(xyz, 3)])
    pts = xyz[is_roof | facade]
    if len(pts) < 8:
        return roof_xyz

    parent = np.arange(len(pts))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in cKDTree(pts).query_pairs(r=link_m, output_type="ndarray"):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    labels = np.array([find(i) for i in range(len(pts))])
    component = pts[labels == labels[int(np.argmax(pts[:, 2]))]]

    # Walk down in bands, stopping where the footprint suddenly widens.
    top = component[:, 2].max()
    kept, previous = [], None
    edge = top
    while edge > component[:, 2].min():
        band = component[(component[:, 2] < edge) & (component[:, 2] >= edge - band_m)]
        edge -= band_m
        if len(band) < 5:
            continue
        span = max(band[:, 0].ptp() if hasattr(band[:, 0], "ptp")
                   else band[:, 0].max() - band[:, 0].min(),
                   band[:, 1].max() - band[:, 1].min())
        if previous is not None and span > previous * max_extent_growth:
            break
        kept.append(band)
        previous = max(span, previous or span)
    return np.vstack(kept) if kept else component


def roof_footprint(roof_xyz: np.ndarray) -> np.ndarray:
    """The roof's planimetric extent as four corners, ordered NW, SW, SE, NE.

    Deliberately separate from :func:`roof_check_features`. A minimum-area
    rectangle is an honest *extent* for any roof, domed or flat -- it answers
    "how much ground does this cover", which is what crop sizing and the
    bounding volume need. It is not an honest set of *measured surface points*,
    because for a curved roof its corners sit at radius x sqrt(2), outside the
    surface entirely. Conflating the two is how four positions that touch no
    roof at all end up labelled ``provenance="lidar:minAreaRect"``.
    """
    import cv2

    pts = np.asarray(roof_xyz, dtype=np.float64)
    corners = cv2.boxPoints(cv2.minAreaRect(pts[:, :2].astype(np.float32))).astype(np.float64)
    centre = corners.mean(axis=0)
    ang = np.arctan2(corners[:, 1] - centre[1], corners[:, 0] - centre[0])
    return corners[np.argsort(-ang)]


def roof_check_features(
    lidar,
    min_cluster_points: int = 200,
    max_height_m: Optional[float] = 15.0,
    select: str = "largest",
    max_plane_rms_m: float = 0.5,
) -> Tuple[List[CheckFeature], np.ndarray]:
    """Derive roof corner/centroid check features from the LiDAR roof surface.

    Returns ``(features, roof_xyz)``. Corners come from the minimum-area
    rectangle of the roof point footprint, lifted to the median roof elevation,
    so each one is a measured position rather than an asserted one.

    ``select`` decides which cluster is the building when several qualify:

    * ``"largest"`` -- most points. Correct for an isolated structure, and what
      the pilot uses.
    * ``"tallest"`` -- highest median elevation. Correct in a dense downtown,
      where a tower's roof is small and its podium and neighbours are large, so
      "largest" reliably picks the wrong one.
    """
    import cv2

    roof_xyz, _ = extract_roof_points(lidar, max_height_m=max_height_m)
    if len(roof_xyz) < min_cluster_points:
        return [], roof_xyz

    # Keep one connected cluster so outbuildings/canopies do not stretch the
    # rectangle across the whole AOI.
    #
    # Connectivity is 3D, not 2D. In plan view a tower's roof overlaps the
    # podium beneath it and everything else in its footprint, so a 2D graph
    # merges surfaces 144 m apart into one cluster whose median elevation is
    # whichever surface has more points -- always the lower, larger one. Adding
    # z separates them. For an isolated single-storey building nothing is
    # stacked, so this changes nothing.
    tree = cKDTree(roof_xyz[:, :3])
    pairs = tree.query_pairs(r=3.0, output_type="ndarray")
    parent = np.arange(len(roof_xyz))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    labels = np.array([find(i) for i in range(len(roof_xyz))])
    counts = np.bincount(labels)
    if select == "tallest":
        eligible = [lab for lab in np.unique(labels)
                    if counts[lab] >= min_cluster_points]
        if not eligible:
            eligible = [int(counts.argmax())]
        chosen = max(eligible, key=lambda lab: float(np.median(roof_xyz[labels == lab, 2])))
    elif select == "largest":
        chosen = int(counts.argmax())
    else:
        raise ValueError(f"select must be 'largest' or 'tallest', got {select!r}")
    roof_xyz = roof_xyz[labels == chosen]

    # Is this roof actually a plane? The min-area-rect construction below puts
    # all four corners at one elevation, which is only a measured position if
    # the surface is flat there. 400 West Market is domed: its crown falls 2.71 m
    # from centre to edge over an 8.3 m radius, so rectangle corners at the
    # median elevation float roughly 3 m above the roof -- and the rect corners
    # sit further out still, where the dome has fallen away further. Asserting
    # a square on a dome produces four "measured" points that touch nothing.
    plane_rms = _plane_fit_rms(roof_xyz)
    if plane_rms > max_plane_rms_m:
        return _curved_roof_features(roof_xyz, plane_rms), roof_xyz

    pts_xy = roof_xyz[:, :2].astype(np.float32)
    rect = cv2.minAreaRect(pts_xy)
    corners = cv2.boxPoints(rect).astype(np.float64)
    roof_z = float(np.median(roof_xyz[:, 2]))

    # Order corners NW, SW, SE, NE for stable naming.
    centre = corners.mean(axis=0)
    ang = np.arctan2(corners[:, 1] - centre[1], corners[:, 0] - centre[0])
    corners = corners[np.argsort(-ang)]

    names = ["RoofCorner_A", "RoofCorner_B", "RoofCorner_C", "RoofCorner_D"]
    features = [
        CheckFeature(
            feature_id=name,
            description=f"LiDAR min-area-rect roof corner at z={roof_z:.2f} m "
                        f"(plane RMS {plane_rms:.3f} m)",
            xyz_local_m=(float(c[0]), float(c[1]), roof_z),
            provenance="lidar:minAreaRect",
        )
        for name, c in zip(names, corners)
    ]
    features.append(
        CheckFeature(
            feature_id="RoofCentroid",
            description="Centroid of LiDAR roof surface",
            xyz_local_m=(
                float(roof_xyz[:, 0].mean()),
                float(roof_xyz[:, 1].mean()),
                roof_z,
            ),
            provenance="lidar:centroid",
        )
    )
    return features, roof_xyz


def ground_elevation_under(
    lidar, footprint_xy: np.ndarray, collar_scale: float = 1.35
) -> float:
    """Median ground elevation on the apron immediately around a footprint.

    Ground returns *inside* a building footprint are the roof, so the elevation
    is taken from a collar just outside it: the polygon scaled about its centre,
    minus the polygon itself, restricted to ASPRS class 2.
    """
    from matplotlib.path import Path as MplPath

    footprint_xy = np.asarray(footprint_xy, dtype=np.float64)[:, :2]
    xyz = np.asarray(lidar.xyz_local_m, dtype=np.float64)
    cls = np.asarray(lidar.classification)

    centre = footprint_xy.mean(axis=0)
    inner = MplPath(footprint_xy)
    outer = MplPath(centre + (footprint_xy - centre) * collar_scale)

    collar = (
        outer.contains_points(xyz[:, :2])
        & ~inner.contains_points(xyz[:, :2])
        & (cls == 2)
    )
    if not collar.any():
        return float(np.median(xyz[cls == 2, 2])) if (cls == 2).any() else float(np.median(xyz[:, 2]))
    return float(np.median(xyz[collar, 2]))
