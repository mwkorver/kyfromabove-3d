"""Check features and match anchors derived from LiDAR.

Phase 1 originally hardcoded seven "check features" as literal coordinates with
no stated provenance; plotted on the imagery they land in the parking lot. This
module derives them from the LiDAR instead, so a check feature refers to a
surface that is actually there.

Two products:

``PlanarAnchor``
    A 3D point plus a locally-fitted surface plane. The plane is what makes
    cross-view patch rectification possible, which is what turns reprojection
    QA into a measurement instead of an assertion.

``CheckFeature``
    A named, human-recognisable point (roof corners, roof centroid) for visual
    inspection and reporting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------- planes
def tangent_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors spanning the plane with the given normal."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    # Cross with whichever world axis is least parallel to n.
    axis = np.zeros(3)
    axis[int(np.argmin(np.abs(n)))] = 1.0
    e1 = np.cross(n, axis)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def _outward_normal(xyz: np.ndarray, index: int, normal: np.ndarray,
                    slab_m: float = 3.0, radius_m: float = 40.0) -> np.ndarray:
    """Flip a plane normal so it points out of the building, not into it.

    ``fit_local_planes`` returns an arbitrary sign, and for a facade the sign is
    load-bearing: it says which side of the wall a camera must be on to see the
    surface. On 400 West Market, 68% of anchor/camera pairs sit behind the wall,
    and attempting them yields guaranteed ``low_ncc`` rejections that read as a
    failure of the method rather than of the geometry.

    Orientation is against the centroid of the building's **cross-section at
    that height** -- points in a thin z-slab, within ``radius_m`` horizontally.
    A closed cross-section has its centroid inside it, so outward is away.

    Two simpler tests were tried and are wrong. A centroid of the local
    neighbourhood sits *in* the wall plane, because on a hollow cloud the
    neighbourhood is the wall. An occupancy probe -- "LiDAR never sees inside a
    building, so the emptier side is outside" -- is false: the outside of a wall
    is air and is equally empty. On a clean hollow box it scored 60%, barely
    above chance.
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    p = xyz[index]
    slab = xyz[(np.abs(xyz[:, 2] - p[2]) <= slab_m)
               & (np.linalg.norm(xyz[:, :2] - p[:2], axis=1) <= radius_m)]
    if len(slab) < 8:
        return n
    inward = np.median(slab[:, :2], axis=0) - p[:2]
    if np.linalg.norm(inward) < 1e-9:
        return n
    return -n if np.dot(n[:2], inward) > 0 else n


def facade_tangent_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Tangent basis for a near-vertical plane: e1 along the wall, e2 up it.

    :func:`tangent_basis` picks whichever world axis is least parallel to the
    normal, which is numerically fine but leaves the in-plane axes arbitrary.
    NCC does not care, but the reported ``offset_e1_m`` and ``offset_e2_m`` do:
    on a facade they should read as "along the wall" and "up the wall", not as
    two unnamed directions. Falls back when the plane is not actually vertical.
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    e1 = np.cross(np.array([0.0, 0.0, 1.0]), n)
    norm = np.linalg.norm(e1)
    if norm < 1e-6:                      # horizontal plane: no "along the wall"
        return tangent_basis(n)
    e1 /= norm
    e2 = np.cross(n, e1)
    return e1, e2 / np.linalg.norm(e2)


def fit_local_planes(
    xyz: np.ndarray, k: int = 16, max_radius_m: float = 4.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a plane to each point's k nearest neighbours.

    Returns ``(normals, rms_residual_m, n_used)``. Points whose neighbourhood is
    sparser than ``max_radius_m`` get a large residual so they are filtered out
    downstream rather than silently producing a garbage normal.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    tree = cKDTree(xyz)
    dists, idx = tree.query(xyz, k=min(k, len(xyz)))

    normals = np.zeros_like(xyz)
    residuals = np.full(len(xyz), np.inf)
    n_used = np.zeros(len(xyz), dtype=int)

    for i in range(len(xyz)):
        sel = idx[i][dists[i] <= max_radius_m]
        if len(sel) < 5:
            continue
        pts = xyz[sel]
        centroid = pts.mean(axis=0)
        centred = pts - centroid
        # Smallest singular vector is the plane normal.
        _, sv, vh = np.linalg.svd(centred, full_matrices=False)
        normal = vh[2]
        if normal[2] < 0:
            normal = -normal
        normals[i] = normal
        residuals[i] = float(np.sqrt(np.mean((centred @ normal) ** 2)))
        n_used[i] = len(sel)

    return normals, residuals, n_used


def surface_coherence(
    xyz: np.ndarray,
    outer_k: int = 48,
    inner_k: int = 8,
    max_radius_m: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Separate real structure from range noise in a rough surface.

    Returns ``(coherence, offset_m)``. A high local plane residual means a point
    does not sit on a plane, but says nothing about *why*: a roof vent and a
    noisy grazing-incidence facade return look identical to
    :func:`fit_local_planes`. On 400 West Market two thirds of the surface has a
    residual above 0.08 m, and densifying noise would resolve nothing while
    looking like progress.

    The distinguishing property is **spatial coherence**. Real structure makes a
    point's immediate neighbours deviate from the surrounding surface *together*
    and in the same direction -- a step, a ridge, a parapet, a mechanical unit.
    Range noise makes them deviate independently.

    So a plane is fitted to a wide neighbourhood (``outer_k``) and the signed
    deviations of a tight one (``inner_k``) are measured against it:

        coherence = |mean(d)| / mean(|d|)


    which is 1.0 when the inner cluster sits wholly to one side of the
    surrounding surface, and near 0 when it scatters. ``offset_m`` is
    ``mean(d)`` -- how far, and which way.

    Deliberately not a smoothness measure. A rough *but coherent* surface is
    detail worth resolving; a rough incoherent one is measurement error.

    **``outer_k`` sets the largest structure this can see.** The wide
    neighbourhood has to reach *past* a feature for the feature to deviate from
    it; anything bigger becomes the local surface and reports no offset. At the
    pilot's in-volume density of 2.82 pts/m^2 the default spans roughly a 2.3 m
    radius, which covers vents, parapets, ledges and roof steps. A whole roof
    level is invisible to it, and should be — that is what
    :func:`building_surface` is for.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    coherence = np.zeros(n)
    offset = np.zeros(n)
    if n < outer_k:
        return coherence, offset

    tree = cKDTree(xyz)
    dists, idx = tree.query(xyz, k=min(outer_k, n))

    for i in range(n):
        near = idx[i][dists[i] <= max_radius_m]
        if len(near) < max(inner_k + 4, 12):
            continue
        pts = xyz[near]
        centroid = pts.mean(axis=0)
        centred = pts - centroid
        normal = np.linalg.svd(centred, full_matrices=False)[2][2]
        # SVD returns an arbitrary sign, which would make the offset's direction
        # meaningless. Orient upward so positive reads as "above the local
        # surface" -- a vent rather than a light well.
        if normal[2] < 0:
            normal = -normal

        # Signed deviation of the tight neighbourhood from the wide plane.
        inner = idx[i][:inner_k]
        d = (xyz[inner] - centroid) @ normal
        denom = float(np.mean(np.abs(d)))
        if denom < 1e-9:
            continue
        coherence[i] = abs(float(np.mean(d))) / denom
        offset[i] = float(np.mean(d))

    return coherence, offset


# ------------------------------------------------------------- complexity
#: Local plane residual above which a point is not sitting on a plane. At the
#: 0.08 m level a genuinely flat roof reads as flat and a parapet does not.
ROUGH_RESIDUAL_M = 0.08
#: Coherence above which a rough point's neighbours deviate together: structure.
STRUCTURE_COHERENCE = 0.7
#: Coherence below which they scatter independently: range noise.
NOISE_COHERENCE = 0.4


@dataclass
class SurfaceComplexity:
    """How much of a surface is genuinely under-sampled, and how much is noise.

    ``structure_frac`` is the number this exists to produce: the share of the
    surface that is both rough *and* coherent, which is the only share where
    adding points resolves something. ``noise_frac`` is reported alongside it
    because the two together say whether a low structure fraction means "this
    roof is flat" or "this measurement is drowning".
    """

    n_points: int
    rough_frac: float
    structure_frac: float
    noise_frac: float
    structure_offset_p50_m: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_points": self.n_points,
            "rough_frac": self.rough_frac,
            "structure_frac": self.structure_frac,
            "noise_frac": self.noise_frac,
            "structure_offset_p50_m": self.structure_offset_p50_m,
        }


@dataclass
class SurfacePointClasses:
    """Per-point verdicts behind a :class:`SurfaceComplexity`.

    Four boolean masks over the input points plus the signed offset that
    produced them. :func:`assess_surface_complexity` reduces these to fractions;
    they are exposed because a whole-building fraction hides *where* on the
    building the structure is, and on a tall one that is the whole question -- a
    wall and a roof can reach the same fraction for entirely different reasons.
    """

    assessable: np.ndarray
    rough: np.ndarray
    structure: np.ndarray
    noise: np.ndarray
    offset_m: np.ndarray


def classify_surface_points(
    xyz: np.ndarray,
    k: int = 16,
    rough_residual_m: float = ROUGH_RESIDUAL_M,
    structure_coherence: float = STRUCTURE_COHERENCE,
    noise_coherence: float = NOISE_COHERENCE,
) -> SurfacePointClasses:
    """Classify every point as assessable / rough / structure / noise."""
    xyz = np.asarray(xyz, dtype=np.float64)
    _, residual, used = fit_local_planes(xyz, k=k)
    coherence, offset = surface_coherence(xyz)
    assessable = np.isfinite(residual) & (used >= 8)
    rough = assessable & (residual > rough_residual_m)
    return SurfacePointClasses(
        assessable=assessable,
        rough=rough,
        structure=rough & (coherence > structure_coherence),
        noise=rough & (coherence < noise_coherence),
        offset_m=offset,
    )


def assess_surface_complexity(
    xyz: np.ndarray,
    k: int = 16,
    rough_residual_m: float = ROUGH_RESIDUAL_M,
    structure_coherence: float = STRUCTURE_COHERENCE,
    noise_coherence: float = NOISE_COHERENCE,
    min_points: int = 200,
    min_assessed: int = 100,
    classes: Optional[SurfacePointClasses] = None,
) -> Optional[SurfaceComplexity]:
    """Rough / structure / noise fractions for one building's points.

    Pass ``classes`` when the caller already has them from
    :func:`classify_surface_points` and wants the per-point masks too. The two
    passes underneath are the expensive part of a survey, and computing them
    twice per building doubles a run for nothing.

    Returns ``None`` rather than a fabricated zero when there is not enough
    surface to judge -- fewer than ``min_points`` returns, or fewer than
    ``min_assessed`` of them with a usable local plane. A building that returns
    ``None`` is missing from the sample, which is honest; one that returned 0.0
    would be counted as flat, which is not.

    The same function serves the statewide survey and the named case studies, so
    that their numbers are comparable. They are not otherwise: a survey building
    is clipped to an Overture footprint, and both are only as good as that
    polygon.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) < min_points:
        return None
    c = classes if classes is not None else classify_surface_points(
        xyz, k=k, rough_residual_m=rough_residual_m,
        structure_coherence=structure_coherence, noise_coherence=noise_coherence)
    if c.assessable.sum() < min_assessed:
        return None
    n = int(c.assessable.sum())
    return SurfaceComplexity(
        n_points=n,
        rough_frac=float(c.rough.sum() / n),
        structure_frac=float(c.structure.sum() / n),
        noise_frac=float(c.noise.sum() / n),
        structure_offset_p50_m=(
            float(np.percentile(np.abs(c.offset_m[c.structure]), 50))
            if c.structure.any() else 0.0),
    )


@dataclass
class SurfaceZones:
    """A building's returns split into ground, wall and roof.

    A single whole-building complexity number hides which part of the building
    produced it, and the two parts do not behave alike. Measured on the case
    buildings, wall returns score about a third "structure" whether the wall
    carries anything or not -- an aerial scanner sees a vertical face nearly
    edge-on, so the returns are sparse and noisy. Roof returns discriminate.

    ``plate_z_m`` is the modal elevation: the most populated ``bin_m`` band,
    which on a flat-roofed building is its roof. Deliberately not a high
    percentile -- PNC Tower's roof is at 153 m under a mast reaching 163 m, and
    the 98th percentile lands between the two, on neither.
    """

    plate_z_m: float
    plateau_frac: float
    plate_rms_m: Optional[float]
    ground: np.ndarray
    wall: np.ndarray
    roof: np.ndarray
    roof_plateau_frac: float = 0.0      # share of the roof returns on the plate

    def zone_rates(self, mask: np.ndarray, min_points: int = 200
                   ) -> Dict[str, Optional[float]]:
        """Share of each zone's returns inside ``mask``.

        ``None`` where a zone holds fewer than ``min_points``, which is the
        common case for walls on a low building -- the pilot has 22 of them, and
        a rate over 22 points would invite a comparison the data cannot carry.
        """
        out: Dict[str, Optional[float]] = {}
        for name, zone in (("ground", self.ground), ("wall", self.wall),
                           ("roof", self.roof)):
            out[f"{name}_frac"] = float(zone.mean())
            out[f"{name}_rate"] = (float(mask[zone].mean())
                                   if zone.sum() >= min_points else None)
        return out


def split_surface_zones(
    xyz: np.ndarray,
    ground_m: float = 3.0,
    roof_slab_m: float = 2.0,
    band_m: float = 0.5,
    bin_m: float = 0.5,
) -> Optional[SurfaceZones]:
    """Find the roof plate and split the cloud around it.

    Elevations are taken relative to the lowest return in ``xyz``, so the input
    may be in any vertical frame; ``plate_z_m`` comes back in that same frame.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) < 50:
        return None
    z = xyz[:, 2] - xyz[:, 2].min()
    edges = np.arange(0.0, z.max() + bin_m, bin_m)
    if len(edges) < 2:
        return None
    counts, _ = np.histogram(z, edges)
    plate = float((edges[counts.argmax()] + edges[counts.argmax() + 1]) / 2)

    on_plate = np.abs(z - plate) <= band_m
    rms = None
    if on_plate.sum() >= 50:
        centred = xyz[on_plate] - xyz[on_plate].mean(axis=0)
        normal = np.linalg.svd(centred, full_matrices=False)[2][2]
        rms = float(np.sqrt(np.mean((centred @ normal) ** 2)))

    # Assigned in order so the three are disjoint whatever the building's
    # height. On anything under about 5 m the roof band reaches below
    # ``ground_m`` and an independent "z < ground_m" test would put the same
    # points in two zones -- most of Kentucky's large buildings are that low,
    # so this is the common case, not the edge one. A low building simply has
    # no separable wall, and everything under its roof reads as ground.
    roof = z >= plate - roof_slab_m
    wall = (z >= ground_m) & ~roof
    roof_plateau = on_plate & roof
    roof_plateau_frac = float(roof_plateau.sum() / max(roof.sum(), 1))
    return SurfaceZones(
        plate_z_m=plate + float(xyz[:, 2].min()),
        plateau_frac=float(on_plate.mean()),
        plate_rms_m=rms,
        ground=~roof & ~wall,
        wall=wall,
        roof=roof,
        roof_plateau_frac=roof_plateau_frac,
    )


#: Wall returns per square metre of computed facade, below which the LiDAR did
#: not meaningfully reach the wall. Calibrated on the three case buildings:
#: Tractor Supply 0.014 (22 returns on a 1,593 m^2 facade -- nothing), PNC Tower
#: 0.069, 400 West Market 0.233. A judgement call, not a derived constant; it
#: separates "the wall was flown" from "the wall exists on paper".
FACADE_CAPTURE_PER_M2 = 0.05


@dataclass
class FacadeExtent:
    """How much wall a building has, and how much of it the LiDAR reached.

    A facade is not judged the way a roof is. Roofs are scored on roughness,
    because the question there is whether 0.60 m sampling misses structure. A
    facade is already flat -- 0.104 m median local-plane RMS on the tower shaft,
    better than the roof filter's own threshold assumes -- so roughness says
    nothing useful about it. What a facade lacks is **colour**, which the LiDAR
    has nowhere, and the imagery supplies at 7.8x finer resolution. So the
    question is extent and capture, not detail.

    ``area_m2`` is what exists; ``returns_per_m2`` is what was reached. The pair
    matters more than either: a large facade with no returns is occluded or was
    flown from the wrong side, and only the two together tell that apart from a
    genuinely small facade.
    """

    area_m2: float
    to_roof_ratio: float
    returns_per_m2: float
    captured: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facade_area_m2": self.area_m2,
            "facade_to_roof": self.to_roof_ratio,
            "facade_returns_per_m2": self.returns_per_m2,
            "facade_captured": self.captured,
        }


def facade_extent(
    perimeter_m: float,
    height_m: float,
    footprint_area_m2: float,
    n_wall_returns: int,
    capture_per_m2: float = FACADE_CAPTURE_PER_M2,
) -> FacadeExtent:
    """Facade area from the footprint's perimeter and the building's height.

    Takes floats rather than a polygon so this module stays free of the shapely
    dependency it does not otherwise carry; callers pass
    ``poly.length * METERS_PER_US_SURVEY_FOOT``.

    ``height_m`` should come from the LiDAR -- :func:`split_surface_zones`
    reports the roof plate -- not from Overture, which has a height for 79% of
    Kentucky's large buildings and is modelled rather than measured on many of
    those.

    A vertical extrusion is assumed. That understates a building with setbacks
    and overstates one whose footprint includes an attached single-storey wing,
    which is the commoner error on the large-footprint population.
    """
    area = max(0.0, float(perimeter_m) * float(height_m))
    return FacadeExtent(
        area_m2=area,
        to_roof_ratio=(area / footprint_area_m2) if footprint_area_m2 > 0 else 0.0,
        returns_per_m2=(n_wall_returns / area) if area > 0 else 0.0,
        captured=bool(area > 0 and n_wall_returns / area >= capture_per_m2),
    )


@dataclass
class FacadeFace:
    """One wall of a building, and whether the LiDAR reached it.

    A facade is not one surface. **A wall faces one way**, and an aerial scanner
    flying one direction over it reaches some faces and not others -- measured on
    400 West Market, only 30% of camera-anchor pairs have the camera on the
    visible side at all. A single whole-building return density averages the
    face the aircraft flew along with the face it never saw, and reports a
    building as captured when half of it is missing.

    ``bearing_deg`` is the compass direction the wall looks towards, so a run of
    uncaptured faces sharing a bearing is a flight-direction gap rather than an
    occlusion.
    """

    index: int
    length_m: float
    area_m2: float
    bearing_deg: float
    n_returns: int
    returns_per_m2: float
    captured: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "face": self.index, "length_m": self.length_m,
            "area_m2": self.area_m2, "bearing_deg": self.bearing_deg,
            "n_returns": self.n_returns,
            "returns_per_m2": self.returns_per_m2, "captured": self.captured,
        }


def facade_faces(
    vertices_m: np.ndarray,
    height_m: float,
    wall_xy: Optional[np.ndarray] = None,
    min_length_m: float = 3.0,
    max_offset_m: float = 2.0,
    capture_per_m2: float = FACADE_CAPTURE_PER_M2,
) -> List[FacadeFace]:
    """Split a footprint into walls and count the returns that reached each.

    ``vertices_m`` is the footprint's exterior ring, ``(N, 2)`` in metres, closed
    or not; winding is fixed internally so the bearings come out pointing away
    from the building whichever way the ring was written. ``wall_xy`` is the
    ``(M, 2)`` plan position of the wall-zone returns, from
    :func:`split_surface_zones`.

    Each return is assigned to its nearest wall segment and dropped if it is
    further than ``max_offset_m`` from every one -- an Overture footprint is
    ML-derived and sits a metre or two off, so a return from the real wall lands
    just inside or outside the drawn one.

    Segments under ``min_length_m`` are skipped. They are mostly the small steps
    a traced footprint puts on a corner, and a face two metres wide gets a
    return density dominated by which side of the corner a point was assigned
    to.
    """
    v = np.asarray(vertices_m, dtype=np.float64)[:, :2]
    if len(v) > 1 and np.allclose(v[0], v[-1]):
        v = v[:-1]
    if len(v) < 3 or height_m <= 0:
        return []

    # Force counter-clockwise, so the outward side of a segment is consistently
    # to its right and the bearings mean something.
    if np.sum((np.roll(v[:, 0], -1) - v[:, 0]) * (np.roll(v[:, 1], -1) + v[:, 1])) > 0:
        v = v[::-1]

    p0, p1 = v, np.roll(v, -1, axis=0)
    seg = p1 - p0
    length = np.hypot(seg[:, 0], seg[:, 1])
    keep = length >= min_length_m
    if not keep.any():
        return []

    nearest = None
    if wall_xy is not None and len(wall_xy):
        w = np.asarray(wall_xy, dtype=np.float64)[:, :2]
        # Perpendicular distance to each segment, clamped to its extent so a
        # point beyond an end is measured to the end rather than to the
        # infinite line.
        d = w[:, None, :] - p0[None, :, :]
        denom = np.maximum((seg ** 2).sum(axis=1), 1e-9)
        t = np.clip((d * seg[None, :, :]).sum(axis=2) / denom, 0.0, 1.0)
        closest = p0[None, :, :] + t[:, :, None] * seg[None, :, :]
        dist = np.linalg.norm(w[:, None, :] - closest, axis=2)
        dist[:, ~keep] = np.inf
        nearest = np.where(dist.min(axis=1) <= max_offset_m, dist.argmin(axis=1), -1)

    faces: List[FacadeFace] = []
    for i in np.flatnonzero(keep):
        # Outward normal of a CCW ring is the segment turned clockwise.
        nx, ny = seg[i, 1] / length[i], -seg[i, 0] / length[i]
        area = float(length[i] * height_m)
        n = int((nearest == i).sum()) if nearest is not None else 0
        per_m2 = n / area if area > 0 else 0.0
        faces.append(FacadeFace(
            index=int(i),
            length_m=float(length[i]),
            area_m2=area,
            bearing_deg=float(np.degrees(np.arctan2(nx, ny)) % 360.0),
            n_returns=n,
            returns_per_m2=per_m2,
            captured=bool(per_m2 >= capture_per_m2),
        ))
    return faces


@dataclass
class WallPlane:
    """One wall of a building as a vertical plane: the footprint edge from
    ``p0`` to ``p1`` extruded from ``z_lo`` to ``z_hi``, facing ``normal``.

    On a curtain-wall tower the LiDAR reaches the wall thinly -- PNC Tower's
    walls hold 0.57 returns per square metre, one every 1.3 m -- so a camera
    ray looking for the nearest return within a metre usually finds none, and
    the plane-fit tiles that the aerial pass colours cannot be fitted at all
    (2,710 of its 2,727 wall cells). But the wall *is* a plane: the footprint
    says where, the roof plate says how high. This is that plane, so a ray can
    be lifted onto it exactly, every ray, with no return needed.
    """

    face: int
    p0: np.ndarray            # (2,) metres, footprint edge start
    p1: np.ndarray            # (2,) metres, footprint edge end
    normal: np.ndarray        # (2,) unit, pointing away from the building
    z_lo: float
    z_hi: float

    @property
    def length_m(self) -> float:
        return float(np.hypot(*(self.p1 - self.p0)))

    def local(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(along from p0, offset outward) of plan positions ``xy``."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        seg = self.p1 - self.p0
        rel = xy - self.p0[None, :]
        return rel @ (seg / max(self.length_m, 1e-9)), rel @ self.normal

    def world(self, along, off) -> np.ndarray:
        """Plan positions from (along, offset outward)."""
        along = np.asarray(along, dtype=np.float64).reshape(-1)
        off = np.asarray(off, dtype=np.float64).reshape(-1)
        e1 = (self.p1 - self.p0) / max(self.length_m, 1e-9)
        return self.p0[None, :] + along[:, None] * e1[None, :] + off[:, None] * self.normal[None, :]

    def world_line(self, step_m: float = 1.0) -> np.ndarray:
        """The wall's line in plan, as points from p0 to p1."""
        return np.vstack([self.p0, self.p1])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "face": int(self.face),
            "p0": self.p0.tolist(),
            "p1": self.p1.tolist(),
            "normal": self.normal.tolist(),
            "z_lo": float(self.z_lo),
            "z_hi": float(self.z_hi),
            "verified": False,
        }


@dataclass
class VerifiedWallPlane(WallPlane):
    """A WallPlane whose location and planarity are verified by fitted LiDAR tiles,
    or inferred via prism completion from adjacent verified planar walls.

    Extends WallPlane with tile statistics, evidence-based z_top, and inference flag.
    """

    n_tiles: int = 0
    rms_m: float = 0.0
    max_normal_dev_deg: float = 0.0
    z_top: float = 0.0
    offset_m: float = 0.0
    yaw_dev_deg: float = 0.0
    inferred: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "verified": True,
            "inferred": bool(self.inferred),
            "n_tiles": int(self.n_tiles),
            "rms_m": float(self.rms_m),
            "max_normal_dev_deg": float(self.max_normal_dev_deg),
            "z_top": float(self.z_top),
            "offset_m": float(self.offset_m),
            "yaw_dev_deg": float(self.yaw_dev_deg),
        })
        return d


@dataclass
class ArcWall(WallPlane):
    """A curved wall as one face: a circular arc through a run of the
    footprint's short edges, handled in an **unrolled** frame.

    Overture draws a curve as a chain of edges under 3 m, each too short
    to hold a profile of its own, so a curved wall came out as a fan of
    flat slats with a seam at each. Here the run is one face. Its plane
    frame is the arc unrolled flat: ``p0 = (0, 0)``, ``p1 = (L, 0)`` with
    ``L`` the arc length, ``normal = (0, -1)`` (outward is to the right of
    travel, as on every other face), so the profile, the bands, the
    texture and the wall rectangles are computed by the same code as a
    straight face. :meth:`roll` puts finished points back on the arc and
    :meth:`unroll` brings returns into the frame; ``x`` is distance along
    the arc from its first vertex, ``y`` is ``-`` the outward offset.

    ``turn`` is the signed angle the arc sweeps (radians; positive when
    the angle about the centre grows along the wall), ``convex`` whether
    outward points away from the centre (a bulge) or towards it (a dent).
    """

    centre: np.ndarray = field(default_factory=lambda: np.zeros(2))
    radius: float = 1.0
    theta0: float = 0.0
    turn: float = 0.0
    convex: bool = True
    first_edge: int = 0
    last_edge: int = 0
    rms_m: float = 0.0

    @classmethod
    def make(cls, face: int, centre, radius: float, theta0: float, turn: float, convex: bool,
             first_edge: int, last_edge: int, z_lo: float, z_hi: float, rms_m: float = 0.0) -> "ArcWall":
        L = float(abs(turn) * radius)
        return cls(face=int(face), p0=np.zeros(2), p1=np.array([L, 0.0]), normal=np.array([0.0, -1.0]),
                   z_lo=float(z_lo), z_hi=float(z_hi), centre=np.asarray(centre, dtype=np.float64),
                   radius=float(radius), theta0=float(theta0), turn=float(turn), convex=bool(convex),
                   first_edge=int(first_edge), last_edge=int(last_edge), rms_m=float(rms_m))

    @property
    def direction(self) -> float:
        return 1.0 if self.turn >= 0 else -1.0

    @property
    def out_sign(self) -> float:
        return 1.0 if self.convex else -1.0

    def theta(self, along) -> np.ndarray:
        return self.theta0 + self.direction * np.asarray(along, dtype=np.float64) / self.radius

    def basis(self, along):
        """Unit tangent (the unrolled +x) and outward radial at ``along``."""
        th = self.theta(along).reshape(-1)
        radial = np.column_stack([np.cos(th), np.sin(th)])
        tangent = self.direction * np.column_stack([-np.sin(th), np.cos(th)])
        return tangent, self.out_sign * radial

    def unroll(self, xy: np.ndarray) -> np.ndarray:
        """Plan positions -> the unrolled frame ``(x = along, y = -outward)``."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        rel = xy - self.centre[None, :]
        r = np.hypot(rel[:, 0], rel[:, 1])
        th = np.arctan2(rel[:, 1], rel[:, 0])
        # the angle from the arc's start, wound so the arc itself is [0, |turn|]
        # and a little before the start reads negative rather than a lap later
        margin = min(np.pi / 4, 0.5 * (2 * np.pi - abs(self.turn)))
        a = (self.direction * (th - self.theta0) + margin) % (2 * np.pi) - margin
        d = self.out_sign * (r - self.radius)
        return np.column_stack([a * self.radius, -d])

    def roll(self, sy: np.ndarray) -> np.ndarray:
        """The unrolled frame ``(x, y)`` -> plan positions on the arc."""
        sy = np.asarray(sy, dtype=np.float64).reshape(-1, 2)
        _, radial_out = self.basis(sy[:, 0])
        return self.centre[None, :] + self.radius * self.out_sign * radial_out + (-sy[:, 1])[:, None] * radial_out

    def local(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u = self.unroll(xy)
        return u[:, 0], -u[:, 1]

    def world(self, along, off) -> np.ndarray:
        along = np.asarray(along, dtype=np.float64).reshape(-1)
        off = np.asarray(off, dtype=np.float64).reshape(-1)
        return self.roll(np.column_stack([along, -off]))

    def world_line(self, step_m: float = 1.0) -> np.ndarray:
        n = max(2, int(np.ceil(self.length_m / step_m)) + 1)
        return self.world(np.linspace(0.0, self.length_m, n), np.zeros(n))

    def to_world_3d(self, pts: np.ndarray, normals: Optional[np.ndarray] = None):
        """Unrolled ``(x, y, z)`` points (and normals) -> world."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        xy = self.roll(pts[:, :2])
        out = np.column_stack([xy, pts[:, 2]])
        if normals is None:
            return out
        t, r_out = self.basis(pts[:, 0])
        nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        nxy = nrm[:, :1] * t + nrm[:, 1:2] * (-r_out)
        return out, np.column_stack([nxy, nrm[:, 2]])

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({"kind": "arc", "centre": self.centre.tolist(), "radius": float(self.radius),
                  "theta0": float(self.theta0), "turn": float(self.turn), "convex": bool(self.convex),
                  "first_edge": int(self.first_edge), "last_edge": int(self.last_edge), "rms_m": float(self.rms_m)})
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArcWall":
        return cls.make(d["face"], d["centre"], d["radius"], d["theta0"], d["turn"], d["convex"],
                        d["first_edge"], d["last_edge"], d["z_lo"], d["z_hi"], d.get("rms_m", 0.0))


def _ccw_ring(vertices_m: np.ndarray) -> np.ndarray:
    """The ring open and counter-clockwise, as ``facade_faces`` and
    ``wall_planes`` see it, so edge indices agree everywhere."""
    v = np.asarray(vertices_m, dtype=np.float64)[:, :2]
    if len(v) > 1 and np.allclose(v[0], v[-1]):
        v = v[:-1]
    if len(v) >= 3 and np.sum((np.roll(v[:, 0], -1) - v[:, 0]) * (np.roll(v[:, 1], -1) + v[:, 1])) > 0:
        v = v[::-1]
    return v


def _fit_circle(pts: np.ndarray):
    """Algebraic least-squares circle: (centre, radius, rms of the radial residual)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    bvec = -(x * x + y * y)
    (a, b, c), *_ = np.linalg.lstsq(A, bvec, rcond=None)
    cx, cy = -a / 2, -b / 2
    r2 = cx * cx + cy * cy - c
    if r2 <= 0:
        return None
    R = float(np.sqrt(r2))
    res = np.hypot(x - cx, y - cy) - R
    return np.array([cx, cy]), R, float(np.sqrt((res ** 2).mean()))


def arc_faces(vertices_m: np.ndarray, z_lo: float, z_hi: float, min_edges: int = 4, max_edge_m: float = 4.0,
              turn_deg: Tuple[float, float] = (3.0, 35.0), min_total_deg: float = 35.0,
              max_rms_m: float = 0.3, max_radius_m: float = 200.0, min_length_m: float = 6.0) -> List[ArcWall]:
    """The curved walls of a footprint: runs of at least ``min_edges``
    consecutive edges no longer than ``max_edge_m`` that turn the same way
    by ``turn_deg`` at every vertex and by ``min_total_deg`` in all, whose
    vertices fit a circle within ``max_rms_m`` and whose arc is at least
    ``min_length_m`` long (a rounded corner of a metre or two stays a
    corner). A run of every edge is one full circle. Each arc's ``face`` is
    its first edge's index."""
    v = _ccw_ring(vertices_m)
    n = len(v)
    if n < min_edges + 1 or z_hi <= z_lo:
        return []
    seg = np.roll(v, -1, axis=0) - v
    length = np.hypot(seg[:, 0], seg[:, 1])
    heading = np.arctan2(seg[:, 1], seg[:, 0])
    # turn[i]: at the vertex between edge i and edge i+1
    turn = (np.roll(heading, -1) - heading + np.pi) % (2 * np.pi) - np.pi
    lo, hi = np.radians(turn_deg[0]), np.radians(turn_deg[1])
    short = length <= max_edge_m
    # joint i: edges i and i+1 may belong to one arc
    joint = short & np.roll(short, -1) & (np.abs(turn) >= lo) & (np.abs(turn) <= hi)
    same = np.sign(turn) == np.sign(np.roll(turn, 1))            # turn[i] agrees with turn[i-1]
    runs: List[List[int]] = []
    if joint.all() and same.all():
        runs = [list(range(n))]                                    # a full circle
    else:
        start = int(np.flatnonzero(~joint)[0]) + 1 if (~joint).any() else 0
        cur = [start % n]
        for k in range(1, n):
            i = (start + k) % n
            prev = (i - 1) % n
            if joint[prev] and (len(cur) == 1 or same[prev]):
                cur.append(i)
            else:
                runs.append(cur); cur = [i]
        runs.append(cur)
    arcs: List[ArcWall] = []
    for run in runs:
        if len(run) < min_edges:
            continue
        total = float(sum(turn[i] for i in run[:-1]))
        if abs(total) < np.radians(min_total_deg):
            continue
        verts = np.array([v[i] for i in run] + [v[(run[-1] + 1) % n]])
        full = len(run) == n
        fit = _fit_circle(verts[:-1] if full else verts)
        if fit is None:
            continue
        centre, R, rms = fit
        if rms > max_rms_m or R > max_radius_m or R < 0.5 or R * abs(total) < min_length_m:
            continue
        rel = verts - centre[None, :]
        th = np.arctan2(rel[:, 1], rel[:, 0])
        dth = (np.diff(th) + np.pi) % (2 * np.pi) - np.pi
        sweep = float(dth.sum())
        if full:
            sweep = float(np.sign(sweep) * 2 * np.pi)
        mid = run[len(run) // 2]
        out_n = np.array([seg[mid, 1], -seg[mid, 0]])              # right of travel: outward on a CCW ring
        convex = bool(out_n @ (v[mid] - centre) > 0)
        arcs.append(ArcWall.make(face=run[0], centre=centre, radius=R, theta0=float(th[0]), turn=sweep,
                                 convex=convex, first_edge=run[0], last_edge=run[-1], z_lo=z_lo, z_hi=z_hi, rms_m=rms))
    return arcs


def verify_face_plane(
    plane: WallPlane,
    tile_centroids: np.ndarray,
    tile_normals: np.ndarray,
    max_offset_rms_m: float = 0.30,
    max_normal_dev_deg: float = 3.0,
    min_tiles: int = 2,
    tile_m: float = 12.0,
    roof_z: Optional[float] = None,
) -> Optional[VerifiedWallPlane]:
    """Verify whether a face's fitted LiDAR tiles define a single vertical plane.

    Fits a vertical plane (offset and yaw relative to the nominal footprint edge)
    to the tile centroids via least squares. If at least ``min_tiles`` exist,
    every tile centroid is within ``max_offset_rms_m`` (both RMS and max residual),
    and every tile normal is within ``max_normal_dev_deg`` of the fitted plane's
    outward normal, the face is declared planar-verified.

    Returns:
        VerifiedWallPlane with refined p0, p1, and normal, extruded from
        z_lo to z_top (highest tile top, capped at roof_z), or None if unverified.
    """
    c = np.asarray(tile_centroids, dtype=np.float64)
    n = np.asarray(tile_normals, dtype=np.float64)
    if len(c) < min_tiles or len(c) != len(n):
        return None

    p0, p1 = plane.p0, plane.p1
    seg = p1 - p0
    L = float(np.hypot(seg[0], seg[1]))
    if L < 1e-6:
        return None
    e1 = seg / L
    n_out = plane.normal

    # Project centroids into nominal edge frame: s along edge, d along normal
    rel = c[:, :2] - p0
    s = rel @ e1
    d = rel @ n_out

    # Fit line d(s) = d_0 + k * s
    if np.ptp(s) < 1.0:
        # Tiles stacked vertically at roughly the same along-wall position; no yaw
        k = 0.0
        d_0 = float(np.mean(d))
    else:
        k, d_0 = np.polyfit(s, d, deg=1)
        k = float(k)
        d_0 = float(d_0)

    yaw_rad = np.arctan(k)
    yaw_deg = float(np.degrees(np.abs(yaw_rad)))
    if yaw_deg > max_normal_dev_deg:
        return None

    # Centroid residuals to the fitted line
    d_fit = d_0 + k * s
    residuals = d - d_fit
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    max_res = float(np.max(np.abs(residuals)))
    if rms > max_offset_rms_m or max_res > max_offset_rms_m:
        return None

    # Fitted outward normal in 2D
    dir_line = e1 + k * n_out
    dir_line /= np.linalg.norm(dir_line)
    n_fitted_2d = np.array([dir_line[1], -dir_line[0]], dtype=np.float64)
    if n_fitted_2d @ n_out < 0:
        n_fitted_2d = -n_fitted_2d
    n_fitted_2d /= np.linalg.norm(n_fitted_2d)

    # Check tile normal deviations (in 3D). One tile among many leaning a
    # degree past the limit is a mullion or a canopy in that tile, not a
    # different wall: up to a quarter of the tiles may miss the limit as
    # long as the rest hold it. PNC Tower's west face lost its verification
    # -- and its two short faces their prism completion -- when dropping
    # 185 noise returns nudged one tile of twelve from 3.4 to 4.4 degrees.
    n_plane_3d = np.array([n_fitted_2d[0], n_fitted_2d[1], 0.0], dtype=np.float64)
    dots = np.clip(n @ n_plane_3d, -1.0, 1.0)
    normal_devs_deg = np.degrees(np.arccos(dots))
    within = normal_devs_deg <= max_normal_dev_deg
    if within.sum() < max(min_tiles, int(np.ceil(0.75 * len(n)))):
        return None
    max_normal_dev = float(np.max(normal_devs_deg[within]))

    # Refine p0 and p1 along the fitted line
    p0_ref = p0 + d_0 * n_out
    p1_ref = p0 + L * e1 + (d_0 + k * L) * n_out

    highest_z = float(np.max(c[:, 2]))
    top_cand = highest_z + tile_m / 2.0
    z_top = float(min(roof_z if roof_z is not None else plane.z_hi, top_cand))

    return VerifiedWallPlane(
        face=plane.face,
        p0=p0_ref,
        p1=p1_ref,
        normal=n_fitted_2d,
        z_lo=plane.z_lo,
        z_hi=z_top,
        n_tiles=len(c),
        rms_m=rms,
        max_normal_dev_deg=max_normal_dev,
        z_top=z_top,
        offset_m=d_0,
        yaw_dev_deg=yaw_deg,
    )


def infer_prism_faces(
    nominal_planes: List[WallPlane],
    verified_planes: Dict[int, VerifiedWallPlane],
    min_perimeter_ratio: float = 0.60,
    roof_z: Optional[float] = None,
) -> Dict[int, VerifiedWallPlane]:
    """Infer unverified connecting facade faces for buildings with verified planar curtain walls.

    When major facade faces of a building are verified planar curtain walls
    (accounting for at least ``min_perimeter_ratio`` of the total nominal footprint perimeter),
    the building is inferred to be a vertical prism. The remaining unverified faces
    (typically narrow end walls where LiDAR returns are depleted by specular grazing reflections)
    are completed as planar prism faces bridging between adjacent verified planes,
    extruded up to the minimum z_top of the adjacent verified faces (capped at roof_z).

    Non-planar buildings (such as post-modern architecture with setbacks, arches, or loggias,
    e.g. Humana Building with 0% verified perimeter) will fail the perimeter evidence gate
    and return an empty dictionary, ensuring zero geometry is invented.

    Args:
        nominal_planes: list of nominal footprint WallPlanes in perimeter order.
        verified_planes: dict of directly verified WallPlanes (e.g. from verify_face_plane).
        min_perimeter_ratio: fraction of total perimeter that must be verified (default 0.60).
        roof_z: optional roof elevation cap.

    Returns:
        Dict mapping face index to inferred VerifiedWallPlane instances (with inferred=True).
    """
    if not verified_planes or not nominal_planes:
        return {}

    lengths = [float(np.hypot(p.p1[0] - p.p0[0], p.p1[1] - p.p0[1])) for p in nominal_planes]
    total_perimeter = sum(lengths)
    if total_perimeter <= 0:
        return {}

    verified_perimeter = sum(
        lengths[idx] for idx, p in enumerate(nominal_planes) if p.face in verified_planes
    )
    ratio = verified_perimeter / total_perimeter
    if ratio < min_perimeter_ratio:
        return {}

    n_faces = len(nominal_planes)
    face_to_idx = {p.face: idx for idx, p in enumerate(nominal_planes)}
    inferred: Dict[int, VerifiedWallPlane] = {}

    for p in nominal_planes:
        if p.face in verified_planes:
            continue

        curr_idx = face_to_idx[p.face]
        prev_p = nominal_planes[(curr_idx - 1) % n_faces]
        next_p = nominal_planes[(curr_idx + 1) % n_faces]

        # Snap endpoints to adjacent verified walls if available, closing the polygon
        if prev_p.face in verified_planes:
            p0 = verified_planes[prev_p.face].p1.copy()
        else:
            p0 = p.p0.copy()

        if next_p.face in verified_planes:
            p1 = verified_planes[next_p.face].p0.copy()
        else:
            p1 = p.p1.copy()

        # Compute outward normal perpendicular to the refined edge segment
        seg = p1 - p0
        L = float(np.hypot(seg[0], seg[1]))
        if L > 1e-6:
            norm_cand = np.array([seg[1], -seg[0]]) / L
            if norm_cand @ p.normal < 0:
                norm_cand = -norm_cand
            normal = norm_cand
        else:
            normal = p.normal.copy()

        # Extrusion height: min z_top of adjacent verified faces (or all verified if neither)
        has_prev = prev_p.face in verified_planes
        has_next = next_p.face in verified_planes
        if has_prev and has_next:
            z_cand = min(verified_planes[prev_p.face].z_top, verified_planes[next_p.face].z_top)
        elif has_prev:
            z_cand = verified_planes[prev_p.face].z_top
        elif has_next:
            z_cand = verified_planes[next_p.face].z_top
        else:
            z_cand = min(v.z_top for v in verified_planes.values())

        if roof_z is not None:
            z_top = float(min(roof_z, z_cand))
        else:
            z_top = float(min(p.z_hi, z_cand))

        inferred[p.face] = VerifiedWallPlane(
            face=p.face,
            p0=p0,
            p1=p1,
            normal=normal,
            z_lo=p.z_lo,
            z_hi=z_top,
            n_tiles=0,
            rms_m=0.0,
            max_normal_dev_deg=0.0,
            z_top=z_top,
            offset_m=0.0,
            yaw_dev_deg=0.0,
            inferred=True,
        )

    return inferred


def wall_planes(vertices_m: np.ndarray, z_lo: float, z_hi: float,
                min_length_m: float = 3.0, arcs: bool = False) -> List[WallPlane]:
    """The footprint's edges as vertical planes between ``z_lo`` and ``z_hi``.

    Uses the same winding fix and outward-normal convention as
    :func:`facade_faces`, so ``WallPlane.face`` matches ``FacadeFace.index``
    for the same ring, and the same ``min_length_m`` drops the small corner
    steps a traced footprint carries. With ``arcs`` the runs of short edges
    that :func:`arc_faces` reads as curved walls become one :class:`ArcWall`
    each (its ``face`` the first edge's index) in place of their edges.
    """
    v = _ccw_ring(vertices_m)
    if len(v) < 3 or z_hi <= z_lo:
        return []
    p0, p1 = v, np.roll(v, -1, axis=0)
    seg = p1 - p0
    length = np.hypot(seg[:, 0], seg[:, 1])
    planes: List[WallPlane] = []
    in_arc = set()
    arcs = arc_faces(v, z_lo, z_hi) if arcs else []
    for a in arcs:
        i = a.first_edge
        while True:
            in_arc.add(i)
            if i == a.last_edge:
                break
            i = (i + 1) % len(v)
    for i in np.flatnonzero(length >= min_length_m):
        if int(i) in in_arc:
            continue
        normal = np.array([seg[i, 1], -seg[i, 0]]) / length[i]
        planes.append(WallPlane(face=int(i), p0=p0[i].copy(), p1=p1[i].copy(),
                                normal=normal, z_lo=float(z_lo), z_hi=float(z_hi)))
    planes.extend(arcs)
    return sorted(planes, key=lambda pl: pl.face)


def save_wall_planes(path: Union[Path, str], planes: List[WallPlane]) -> None:
    """Save a list of WallPlane or VerifiedWallPlane instances to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {"planes": [pl.to_dict() for pl in planes]}
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_wall_planes(path: Union[Path, str]) -> List[WallPlane]:
    """Load a list of WallPlane / VerifiedWallPlane instances from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("planes", data if isinstance(data, list) else [])
    planes: List[WallPlane] = []
    for item in items:
        if item.get("kind") == "arc":
            planes.append(ArcWall.from_dict(item))
        elif item.get("verified", False):
            planes.append(VerifiedWallPlane(
                face=int(item["face"]),
                p0=np.array(item["p0"], dtype=np.float64),
                p1=np.array(item["p1"], dtype=np.float64),
                normal=np.array(item["normal"], dtype=np.float64),
                z_lo=float(item["z_lo"]),
                z_hi=float(item["z_hi"]),
                n_tiles=int(item.get("n_tiles", 0)),
                rms_m=float(item.get("rms_m", 0.0)),
                max_normal_dev_deg=float(item.get("max_normal_dev_deg", 0.0)),
                z_top=float(item.get("z_top", item["z_hi"])),
                offset_m=float(item.get("offset_m", 0.0)),
                yaw_dev_deg=float(item.get("yaw_dev_deg", 0.0)),
                inferred=bool(item.get("inferred", False)),
            ))
        else:
            planes.append(WallPlane(
                face=int(item["face"]),
                p0=np.array(item["p0"], dtype=np.float64),
                p1=np.array(item["p1"], dtype=np.float64),
                normal=np.array(item["normal"], dtype=np.float64),
                z_lo=float(item["z_lo"]),
                z_hi=float(item["z_hi"]),
            ))
    return planes


def rectify_footprint_corners(
    ring_m: np.ndarray,
    planes: List[WallPlane],
    max_corner_shift_m: float = 2.0,
    min_angle_deg: float = 15.0,
) -> Tuple[np.ndarray, List[WallPlane]]:
    """Rectify footprint corners by mathematically intersecting adjacent verified wall lines.

    For each corner between adjacent footprint edges i and (i+1) % N:
    If adjacent faces have verified or inferred planes, their 2D line equations
    n_i @ x = n_i @ p_0,i and n_{i+1} @ x = n_{i+1} @ p_0,{i+1} are solved for their
    exact intersection point P. If the angle between their outward normals is at
    least min_angle_deg and the solved intersection point is within max_corner_shift_m
    of the nominal corner vertex, the corner is updated to P.

    This aligns the footprint polygon with the physical LiDAR curtain-wall planes to
    millimeter accuracy, removing gaps and overhang offsets.

    Returns:
        (rectified_ring_m, updated_planes)
    """
    ring = np.asarray(ring_m, dtype=np.float64)[:, :2]
    if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    N = len(ring)
    if N < 3:
        return ring_m, planes

    face_to_plane = {p.face: p for p in planes}
    min_sin = float(np.sin(np.radians(min_angle_deg)))

    new_corners = ring.copy()
    for i in range(N):
        next_i = (i + 1) % N
        pA = face_to_plane.get(i)
        pB = face_to_plane.get(next_i)

        is_verified_A = isinstance(pA, VerifiedWallPlane) if pA else False
        is_verified_B = isinstance(pB, VerifiedWallPlane) if pB else False
        if not (is_verified_A or is_verified_B):
            continue

        # Line A
        if pA is not None:
            nA = np.asarray(pA.normal, dtype=np.float64)[:2]
            nA_len = np.linalg.norm(nA)
            if nA_len < 1e-6:
                continue
            nA = nA / nA_len
            cA = float(nA @ pA.p0[:2])
        else:
            segA = ring[(i + 1) % N] - ring[i]
            lenA = np.hypot(segA[0], segA[1])
            if lenA < 1e-6:
                continue
            nA = np.array([segA[1], -segA[0]]) / lenA
            cA = float(nA @ ring[i])

        # Line B
        if pB is not None:
            nB = np.asarray(pB.normal, dtype=np.float64)[:2]
            nB_len = np.linalg.norm(nB)
            if nB_len < 1e-6:
                continue
            nB = nB / nB_len
            cB = float(nB @ pB.p0[:2])
        else:
            segB = ring[(next_i + 1) % N] - ring[next_i]
            lenB = np.hypot(segB[0], segB[1])
            if lenB < 1e-6:
                continue
            nB = np.array([segB[1], -segB[0]]) / lenB
            cB = float(nB @ ring[next_i])

        M = np.array([nA, nB], dtype=np.float64)
        det = float(np.linalg.det(M))
        if abs(det) < min_sin:
            nominal_corner = ring[next_i]
            if is_verified_B:
                pt = nominal_corner + (cB - float(nB @ nominal_corner)) * nB
            elif is_verified_A:
                pt = nominal_corner + (cA - float(nA @ nominal_corner)) * nA
            else:
                continue
            shift = float(np.hypot(pt[0] - nominal_corner[0], pt[1] - nominal_corner[1]))
            if shift <= max_corner_shift_m:
                new_corners[next_i] = pt
            continue

        try:
            pt = np.linalg.solve(M, np.array([cA, cB], dtype=np.float64))
        except np.linalg.LinAlgError:
            continue

        nominal_corner = ring[next_i]
        shift = float(np.hypot(pt[0] - nominal_corner[0], pt[1] - nominal_corner[1]))
        if shift <= max_corner_shift_m:
            new_corners[next_i] = pt

    rectified_ring = np.vstack([new_corners, new_corners[:1]])

    updated_planes: List[WallPlane] = []
    for p in planes:
        face = p.face
        if face < N:
            c0 = new_corners[face]
            c1 = new_corners[(face + 1) % N]
            if isinstance(p, VerifiedWallPlane):
                updated_p = VerifiedWallPlane(
                    face=p.face,
                    p0=c0.copy(),
                    p1=c1.copy(),
                    normal=p.normal.copy(),
                    z_lo=p.z_lo,
                    z_hi=p.z_hi,
                    n_tiles=p.n_tiles,
                    rms_m=p.rms_m,
                    max_normal_dev_deg=p.max_normal_dev_deg,
                    z_top=p.z_top,
                    offset_m=p.offset_m,
                    yaw_dev_deg=p.yaw_dev_deg,
                    inferred=p.inferred,
                )
            else:
                updated_p = WallPlane(
                    face=p.face,
                    p0=c0.copy(),
                    p1=c1.copy(),
                    normal=p.normal.copy(),
                    z_lo=p.z_lo,
                    z_hi=p.z_hi,
                )
            updated_planes.append(updated_p)
        else:
            updated_planes.append(p)

    return rectified_ring, updated_planes


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
    from scipy.spatial import cKDTree
    from matplotlib.path import Path as MplPath

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


# --- a face as a profile of bands, for buildings that are not prisms --------

@dataclass
class FaceBand:
    """One horizontal band of a wall face and where the LiDAR says the wall is.

    ``offset_m`` is the wall's distance outward from the footprint edge (a
    negative offset is inside the drawn footprint); ``status`` says how the
    band is known: ``observed`` from its own returns, ``bridged`` between two
    observed bands that agree, or ``unknown``.
    """

    z_lo: float
    z_hi: float
    offset_m: Optional[float]
    n_returns: int
    spread_m: float                 # median absolute deviation of the returns about the offset
    along_lo: float                 # extent along the edge the returns cover, metres from p0
    along_hi: float
    status: str
    # Where the wall departs from the band's offset along the edge: runs of
    # (along_lo, along_hi, offset) at their own offset -- a recessed bay, a
    # projecting stair tower. Empty when the band is one flat wall.
    segments: List[Tuple[float, float, float]] = field(default_factory=list)
    # Where the neighbouring walls stand at this height, measured along this
    # edge from p0: the wall cannot run past them. None = unclipped.
    clip_lo: Optional[float] = None
    clip_hi: Optional[float] = None
    # An open band (a loggia, a colonnade) has two surfaces; this is the
    # deeper one, where the wall behind the piers stands. None otherwise.
    back_offset_m: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"z_lo": self.z_lo, "z_hi": self.z_hi,
                "clip_lo": None if self.clip_lo is None else round(float(self.clip_lo), 1),
                "clip_hi": None if self.clip_hi is None else round(float(self.clip_hi), 1),
                "back_offset_m": None if self.back_offset_m is None else round(float(self.back_offset_m), 2),
                "offset_m": None if self.offset_m is None else round(float(self.offset_m), 2),
                "n_returns": int(self.n_returns), "spread_m": round(float(self.spread_m), 2),
                "along_lo": round(float(self.along_lo), 1), "along_hi": round(float(self.along_hi), 1),
                "segments": [[round(a, 1), round(b, 1), round(c, 2)] for a, b, c in self.segments],
                "status": self.status}


def _band_segments(a: np.ndarray, o: np.ndarray, offset: float, L: float,
                   cell_m: float = 3.0, step_m: float = 0.8, min_cell: int = 6) -> List[Tuple[float, float, float]]:
    """Runs along the edge where the wall stands at a different offset from
    the band's. The profile is otherwise one offset per band, and that
    filled the recessed bay down the middle of Humana's south face -- 13 m
    wide, 65 m tall, 3-5 m deep -- flat. Each ``cell_m`` cell along the edge
    gets the modal offset of its own returns; cells that differ from the
    band's offset by more than ``step_m``, and agree with their neighbour,
    become a segment. Cells with too few returns keep the band's offset."""
    edges = np.arange(0.0, L + cell_m, cell_m)
    cells = []
    for c0, c1 in zip(edges[:-1], edges[1:]):
        m = (a >= c0) & (a < c1)
        if m.sum() < min_cell:
            cells.append(None)
            continue
        oc = o[m]
        hist, be = np.histogram(oc, bins=np.arange(np.floor(oc.min()), oc.max() + 0.5, 0.5))
        mode = be[int(np.argmax(hist))] + 0.25
        near = np.abs(oc - mode) <= 0.6
        cells.append(float(np.median(oc[near])) if near.sum() >= min_cell else None)
    segs: List[Tuple[float, float, float]] = []
    run_start = None
    run_vals: List[float] = []
    for i, v in enumerate(cells + [None]):
        differs = v is not None and abs(v - offset) > step_m
        if differs and (run_start is None or abs(v - run_vals[-1]) <= step_m / 2):
            if run_start is None:
                run_start = i
            run_vals.append(v)
            continue
        if run_start is not None and len(run_vals) >= 2:            # two cells: 6 m, not a stray
            segs.append((float(edges[run_start]), float(edges[i]), float(np.median(run_vals))))
        run_start, run_vals = (i, [v]) if differs else (None, [])
    return segs


def face_profile(plane: WallPlane, xyz: np.ndarray, band_m: float = 5.0,
                 offset_window_m: Optional[Tuple[float, float]] = None,
                 min_returns: int = 30, max_spread_m: float = 0.45,
                 min_along_fraction: float = 0.25,
                 max_bridge_m: float = 80.0, max_step_m: float = 0.8) -> List[FaceBand]:
    """Where a face's wall stands, band by band, from the returns near it.

    A single plane per face fits a slab tower and fails a building like the
    Humana Building, whose east wall stands 2 m outside its footprint line
    for a hundred metres and then steps 6 m back at the crown, and whose
    south wall the flight saw from the ground to 40 m and again from 90 m
    up but never between. Measured band by band, each of those is simple:
    the returns in a band pile up at one offset from the footprint edge, and
    the wall is there.

    A band is *observed* when at least ``min_returns`` returns sit within
    0.6 m of the band's modal offset, spread no more than ``max_spread_m``
    about it, and reach along at least ``min_along_fraction`` of the edge.
    Unobserved bands between two observed bands whose offsets agree within
    ``max_step_m`` are *bridged* at that offset, up to ``max_bridge_m`` of
    gap -- the LiDAR saw the wall above and below and the same wall is
    between. A step larger than that is kept as a step, and nothing is
    bridged across it. Nothing is extruded above the highest observed band
    or below the lowest: a crown or a portico the LiDAR never reached stays
    unknown.
    """
    seg = plane.p1 - plane.p0
    L = float(np.hypot(*seg))
    if L < 1e-6 or len(xyz) == 0:
        return []
    if offset_window_m is None:
        # 8 m inside the footprint edge, 3 m outside. A tower standing
        # further back than that from the edge its podium was traced along
        # -- Humana's shaft is 25 m inside its north edge -- is out of this
        # window on purpose: widening it to the far side let returns from
        # deep inside the block turn solid walls "open" on the other faces,
        # and the set-back shaft was seen too thinly (about 20 returns per
        # band) to pass the evidence rule anyway. Such a face is reported
        # as unmodelled, and is a job for a photograph.
        offset_window_m = (-8.0, 3.0)
    rel = xyz[:, :2] - plane.p0
    along = (rel @ seg) / (L * L)
    off = rel @ plane.normal
    # The corners are left out: the neighbouring wall's returns sit there at
    # every depth and belong to it.
    corner = 1.5 / L
    in_face = (along >= corner) & (along <= 1.0 - corner) & (off >= offset_window_m[0]) & (off <= offset_window_m[1])
    z = xyz[:, 2]
    bands: List[FaceBand] = []
    returns_of: List[Tuple[np.ndarray, np.ndarray]] = []   # (along m, offset m) per band, for segments later
    z0 = plane.z_lo
    while z0 < plane.z_hi - 1e-6:
        z1 = min(z0 + band_m, plane.z_hi)
        m = in_face & (z >= z0) & (z < z1)
        returns_of.append((along[m] * L, off[m]))
        band = FaceBand(z0, z1, None, int(m.sum()), 0.0, 0.0, 0.0, "unknown")
        if m.sum() >= min_returns:
            o = off[m]
            hist, edges = np.histogram(o, bins=np.arange(np.floor(offset_window_m[0]), offset_window_m[1] + 0.5, 0.5))
            mode = edges[int(np.argmax(hist))] + 0.25
            near = np.abs(o - mode) <= 0.6
            # Two walls in one band -- piers at the face and a recess metres
            # behind them, each with a good share of the returns -- is a
            # colonnade or a loggia, not a solid wall. Humana's crown is one:
            # a plane through the piers gets painted with the sky between
            # them. Such a band is "open" and nothing is extruded on it.
            # Measured, not assumed: the second-highest bin at least 2 m
            # from the mode must hold at least half what the mode's bin
            # holds. Through-glass returns behind a window wall are spread
            # thin and do not reach that; a second wall does.
            # Through a glass wall the laser also returns from the floor slab
            # just inside, 2-3 m behind: that is one wall, not two. A recess
            # 4 m or more behind the piers is a loggia.
            h2 = hist.copy()
            k = int(np.argmax(hist))
            h2[max(0, k - 8):k + 9] = 0
            if h2.max() >= 0.5 * hist[k] and h2.max() >= min_returns / 2:
                # the deeper of the two surfaces is the wall behind the
                # piers; its extent along the edge is where its returns are
                second = edges[int(np.argmax(h2))] + 0.25
                back = float(min(mode, second))
                on_back = np.abs(o - back) <= 0.6
                a_back = along[m][on_back] * L
                bands.append(FaceBand(z0, z1, float(mode), int(m.sum()), 0.0,
                                      float(a_back.min()) if len(a_back) else 0.0,
                                      float(a_back.max()) if len(a_back) else 0.0, "open",
                                      back_offset_m=back))
                z0 = z1
                continue
            if near.sum() >= min_returns:
                offset = float(np.median(o[near]))
                spread = float(np.median(np.abs(o[near] - offset)))
                a = along[m][near] * L
                lo, hi = float(np.percentile(a, 2)), float(np.percentile(a, 98))
                # A wall's returns run up through the band; a roof slab or a
                # ledge caught at one offset sits at one height. Humana's
                # porch roof, 25 m inside the north edge, read as a wall band
                # until this: its returns all lie within a metre of 101 m.
                z_near = z[m][near]
                tall = (np.percentile(z_near, 90) - np.percentile(z_near, 10)) >= 0.5 * (z1 - z0)
                if spread <= max_spread_m and (hi - lo) >= min_along_fraction * L and tall:
                    band = FaceBand(z0, z1, offset, int(near.sum()), spread, lo, hi, "observed",
                                    segments=_band_segments(along[m] * L, o, offset, L, step_m=max_step_m))
        bands.append(band)
        z0 = z1

    # A wall is vertical: its offset holds from one band to the next. A band
    # whose offset agrees with neither neighbour (within ``max_step_m`` / 2)
    # is a surface that leans -- a dome, a sloping crown, a cornice seen
    # edge-on -- and a vertical strip through it would be a pane on a dome.
    # Such bands are ``sloped``: recorded, never extruded. The returns that
    # found them carry the shape themselves.
    seen = list(bands)                                      # judged against what was observed, not what was just relabelled
    for i, b in enumerate(seen):
        if b.status != "observed":
            continue
        neighbours = [seen[j] for j in (i - 1, i + 1)
                      if 0 <= j < len(seen) and seen[j].status == "observed" and seen[j].offset_m is not None]
        if not neighbours:
            # If immediate neighbours are unobserved (a gap across a terrace or setback),
            # check the nearest observed band in each direction within 15 m (3 bands).
            for j in range(i - 1, max(-1, i - 4), -1):
                if seen[j].status == "observed" and seen[j].offset_m is not None:
                    if (b.z_lo - seen[j].z_hi) <= 15.0:
                        neighbours.append(seen[j])
                    break
            for j in range(i + 1, min(len(seen), i + 4)):
                if seen[j].status == "observed" and seen[j].offset_m is not None:
                    if (seen[j].z_lo - b.z_hi) <= 15.0:
                        neighbours.append(seen[j])
                    break
        if not neighbours:
            # An isolated band far above the ground (> 15 m) with no observed wall within 15 m
            # cannot be a vertical wall holding its offset across bands; it is a crown/dome feature.
            if b.z_lo - plane.z_lo > 15.0:
                bands[i] = FaceBand(b.z_lo, b.z_hi, b.offset_m, b.n_returns, b.spread_m, b.along_lo, b.along_hi, "sloped")
            continue
        if not any(abs(n.offset_m - b.offset_m) <= max_step_m / 2 for n in neighbours):
            bands[i] = FaceBand(b.z_lo, b.z_hi, b.offset_m, b.n_returns, b.spread_m, b.along_lo, b.along_hi, "sloped")

    # Extend a wall through thin bands at its ends. Below the lowest observed
    # band the returns are few -- the street is cluttered, the podium is
    # under trees and canopies -- but if what there is sits at the same
    # offset, the wall continues to the ground; without this, Humana's east
    # wall stopped at 12 m and hung over its own ground floor like a panel.
    # The same at the top.
    for step in (-1, 1):
        obs_idx = [i for i, b in enumerate(bands) if b.status == "observed"]
        if not obs_idx:
            break
        i = min(obs_idx) if step < 0 else max(obs_idx)
        anchor = bands[i]
        j = i + step
        while 0 <= j < len(bands) and bands[j].status == "unknown":
            b = bands[j]
            m = in_face & (z >= b.z_lo) & (z < b.z_hi)
            o = off[m]
            # Looser than an observed band, because the wall beside it is
            # already established: within a metre of it, a quarter of the
            # returns an observed band needs.
            near = np.abs(o - anchor.offset_m) <= 1.0
            if near.sum() < max(min_returns // 4, 6):
                break
            a = along[m][near] * L
            lo, hi = float(np.percentile(a, 2)), float(np.percentile(a, 98))
            bands[j] = FaceBand(b.z_lo, b.z_hi, float(np.median(o[near])), int(near.sum()),
                                float(np.median(np.abs(o[near] - np.median(o[near])))),
                                min(lo, anchor.along_lo), max(hi, anchor.along_hi), "extended")
            j += step

    # Bridge between observed bands that agree.
    obs = [i for i, b in enumerate(bands) if b.status == "observed"]
    for i, j in zip(obs, obs[1:]):
        if j - i <= 1:
            continue
        a, c = bands[i], bands[j]
        gap = c.z_lo - a.z_hi
        if abs(a.offset_m - c.offset_m) > max_step_m or gap > max_bridge_m:
            continue
        # An open band between them stays open (a terrace, a loggia); the
        # unknown bands are bridged when the wall above and below agree.
        lo, hi = min(a.along_lo, c.along_lo), max(a.along_hi, c.along_hi)
        # A bridged band was not observed as a wall, but its returns are
        # still there -- too few or too spread for the wall itself, yet
        # often plenty on a recessed bay. Segments come from its own returns
        # where it has them, else from a recess both ends share.
        shared = [sa for sa in a.segments if any(abs(sa[0] - sc[0]) < 3.0 and abs(sa[1] - sc[1]) < 3.0
                                                  and abs(sa[2] - sc[2]) <= max_step_m for sc in c.segments)]
        for k in range(i + 1, j):
            b = bands[k]
            if b.status == "open":
                continue
            frac = (b.z_lo - a.z_hi) / max(gap, 1e-6)
            offset_k = a.offset_m + frac * (c.offset_m - a.offset_m)
            a_k, o_k = returns_of[k]
            own = _band_segments(a_k, o_k, offset_k, L, step_m=max_step_m) if len(o_k) >= 12 else []
            bands[k] = FaceBand(b.z_lo, b.z_hi, offset_k, b.n_returns, 0.0, lo, hi, "bridged",
                                segments=own or list(shared))
    return bands


def band_extent(plane: WallPlane, b: FaceBand, full_width_fraction: float = 0.6,
                margin_m: float = 3.0) -> Tuple[float, float]:
    """How far along the edge a band's wall is taken to run: the whole edge
    when the returns reached most of it, else their own extent plus a margin;
    and never past where the neighbouring walls stand at this height."""
    L = float(np.hypot(*(plane.p1 - plane.p0)))
    if (b.along_hi - b.along_lo) >= full_width_fraction * L:
        s0, s1 = 0.0, L
    else:
        s0, s1 = max(0.0, b.along_lo - margin_m), min(L, b.along_hi + margin_m)
    if b.clip_lo is not None:
        s0 = max(s0, b.clip_lo)
    if b.clip_hi is not None:
        s1 = min(s1, b.clip_hi)
    return s0, s1


# The kinds of band that are wall. ``matched`` has its depth from the frames
# (kyfromabove_3d.wall_match), not the LiDAR: a wall to draw, but never an
# anchor -- bridging, extending and mirroring read ``observed`` bands only.
WALL_STATUSES = ("observed", "bridged", "extended", "mirrored", "matched")


def crown_bands(bands: List[FaceBand], reference_offset_m: Optional[float] = None,
                max_step_m: float = 2.0, top_z: Optional[float] = None,
                top_of: Optional[Callable[[FaceBand], Optional[float]]] = None) -> int:
    """An "open" band at the top of a face -- returns at two depths, with no
    wall above it -- is the wall's crown or parapet in front and the roof
    behind it. Where its front surface continues the wall directly below
    (within ``max_step_m``; or, on a verified plane with nothing measured
    below, the plane itself: ``reference_offset_m``), it is wall, at its front
    offset. Left open, PNC Tower's walls stopped at 152 m under a roof at
    155-160 m, a floating lid; drawn at its *back* offset it had stood panels
    3-7 m inboard on Kentucky Home Life's roof. Bottom-up, so a crown two
    bands tall continues itself. In place; returns how many were made wall."""
    # What stands below must be drawn as wall: on a verified plane every band
    # but an open one is (a sloped band too); elsewhere only the wall
    # statuses. A crown over nothing drawn would turn a face the prism path
    # walls to its roof edge into a lone band at the top -- Truist Tower's
    # face 1007 lost 12-71 m of wall that way.
    order = sorted(range(len(bands)), key=lambda i: bands[i].z_lo)
    wallish = WALL_STATUSES + (("sloped",) if reference_offset_m is not None else ())
    # the crown is above the face's highest wall band: an open band lower
    # down, with wall above it, is a colonnade or a loggia. Each open band up
    # there counts only over a band that is itself drawn -- on a face open all
    # the way (4a668624's face 1) its open ground band made wall left the open
    # bands above it bare, where the prism path had walled the whole face;
    # Chase Bank Plaza's face 1005 has wall to 19 m, then open, unmeasured and
    # open bands, and only the first continues the wall
    walls = [k for k, i in enumerate(order) if bands[i].status in WALL_STATUSES and bands[i].offset_m is not None]
    n = 0
    for k in range(walls[-1] + 1 if walls else 0, len(order)):
        b = bands[order[k]]
        if b.status != "open" or b.offset_m is None or k == 0:
            continue
        below = bands[order[k - 1]]
        if abs(below.z_hi - b.z_lo) > 0.05:
            continue
        if below.status in wallish and below.offset_m is not None:
            ref = below.offset_m
        elif reference_offset_m is not None and below.status != "open":
            ref = reference_offset_m                        # a verified plane: the plane is drawn over it
        else:
            continue                                        # nothing drawn below it to continue
        # a crown stands at the building's edge: within max_step_m of the wall
        # below it *and* of the footprint's edge -- Waterfront Plaza's open
        # band 6 m inboard, over a wall already set back, stood a panel on its
        # roof like Kentucky Home Life's
        if abs(b.offset_m - ref) > max_step_m or abs(b.offset_m) > max_step_m:
            continue
        # and no higher than the face, nor than its own returns reach (``top_of``):
        # a 5 m band over a roof 2 m above its foot put Fifth Third's top 3 m
        # above the LiDAR's
        z_hi = min(b.z_hi, top_z) if top_z is not None else b.z_hi
        own = top_of(b) if top_of is not None else None
        if own is not None:
            z_hi = min(z_hi, own)
        if z_hi - b.z_lo < 0.3:
            continue
        bands[order[k]] = FaceBand(b.z_lo, z_hi, b.offset_m, b.n_returns, b.spread_m, b.along_lo, b.along_hi,
                                   "observed", segments=list(b.segments), clip_lo=b.clip_lo, clip_hi=b.clip_hi)
        n += 1
    return n


def carry_extents_down(bands: List[FaceBand], max_offset_step_m: float = 1.0) -> int:
    """A wall stands on the wall below it. Where a band's returns stop short
    of the band directly above it at nearly the same offset -- a neighbour
    hid part of the wall from the scanner, in a staircase down the face (the
    Omni's west tower wall, 47-67 m up) -- the band runs at least as far as
    the one above. Only downward: a tower set back above a wider podium is
    untouched, and an open band (a colonnade) breaks the chain. In place;
    returns how many bands were widened."""
    order = sorted(range(len(bands)), key=lambda i: bands[i].z_lo)
    n = 0
    for lo_i, hi_i in zip(order[-2::-1], order[:0:-1]):
        b, above = bands[lo_i], bands[hi_i]
        if (b.status not in WALL_STATUSES or above.status not in WALL_STATUSES
                or b.offset_m is None or above.offset_m is None
                or abs(b.offset_m - above.offset_m) > max_offset_step_m or abs(b.z_hi - above.z_lo) > 0.05):
            continue
        lo, hi = min(b.along_lo, above.along_lo), max(b.along_hi, above.along_hi)
        if lo < b.along_lo - 0.05 or hi > b.along_hi + 0.05:
            b.along_lo, b.along_hi = lo, hi
            n += 1
    return n


def clip_profiles_by_neighbours(planes: List[WallPlane], profiles: Dict[int, List[FaceBand]],
                                min_setback_m: float = 1.0) -> int:
    """Where a neighbouring face's wall stands back from the footprint at
    some height, this face's wall cannot reach the corner at that height.

    Seen from Main Street, Humana's east wall ran the full 56 m of its edge
    at every height, while the north wall it meets stands 24.5 m back from
    the north corner above the podium: the east wall was drawn 24.5 m too
    wide up there, matching the base. Faces share corners, so each band is
    clipped at each end by the neighbour's offset at the same height --
    when the neighbour has a wall there; an unknown neighbour clips nothing.
    Returns how many bands were clipped.
    """
    by_face = {pl.face: pl for pl in planes}
    def neighbour_at(pl: WallPlane, corner: np.ndarray, own_end: str):
        for other in planes:
            if other.face == pl.face:
                continue
            if np.linalg.norm(other.p1 - corner) < 0.5 and own_end == "p0":
                return other
            if np.linalg.norm(other.p0 - corner) < 0.5 and own_end == "p1":
                return other
        return None
    n_clipped = 0
    for face, bands in profiles.items():
        pl = by_face.get(face)
        if pl is None:
            continue
        L = float(np.hypot(*(pl.p1 - pl.p0)))
        prev_pl = neighbour_at(pl, pl.p0, "p0")
        next_pl = neighbour_at(pl, pl.p1, "p1")
        for i, b in enumerate(bands):
            if b.status not in WALL_STATUSES:
                continue
            clip_lo = clip_hi = None
            for nb_pl, end in ((prev_pl, "lo"), (next_pl, "hi")):
                if nb_pl is None or nb_pl.face not in profiles:
                    continue
                nb = next((x for x in profiles[nb_pl.face] if x.z_lo <= b.z_lo < x.z_hi), None)
                if nb is None or nb.status not in WALL_STATUSES or nb.offset_m is None:
                    # The neighbour has no wall at this height (Humana's north
                    # wall is unseen under its porch): take its nearest known
                    # band within 25 m of height, so the clip does not jump
                    # back to the corner across a gap in the LiDAR.
                    known = [x for x in profiles[nb_pl.face] if x.status in WALL_STATUSES and x.offset_m is not None
                             and abs((x.z_lo + x.z_hi) / 2 - (b.z_lo + b.z_hi) / 2) <= 25.0]
                    if not known:
                        continue
                    nb = min(known, key=lambda x: abs((x.z_lo + x.z_hi) / 2 - (b.z_lo + b.z_hi) / 2))
                setback = -nb.offset_m
                if setback < min_setback_m:
                    continue
                if end == "lo":
                    clip_lo = setback
                else:
                    clip_hi = L - setback
            if clip_lo is not None or clip_hi is not None:
                bands[i] = FaceBand(b.z_lo, b.z_hi, b.offset_m, b.n_returns, b.spread_m, b.along_lo, b.along_hi,
                                    b.status, segments=b.segments, clip_lo=clip_lo, clip_hi=clip_hi)
                n_clipped += 1
    return n_clipped


def profile_keeps(plane: WallPlane, bands: List[FaceBand], xyz: np.ndarray,
                  dense_per_m2: float = 0.5) -> np.ndarray:
    """Which of ``xyz`` (points on a plane extruded over the whole face) the
    profile lets stand. A verified plane has the tiles' evidence behind it,
    so the profile removes only what it has evidence *against*: bands that
    are open (a loggia: two surfaces, one 4 m or more behind the other), and
    the part of a band beyond the returns' extent when the returns are dense
    enough (``dense_per_m2`` over the extent they cover) for their edge to
    mean something. Humana's north face above 40 m is only its east 30 m --
    the shaft is narrower than the podium -- and the returns there, two per
    square metre, stop sharply at the step; the plane painted sky on the
    rest. On a glass tower the returns are thin everywhere, a tenth of that,
    and thin is not absent: such bands are left alone."""
    seg = plane.p1 - plane.p0
    L = float(np.hypot(*seg))
    along = ((xyz[:, :2] - plane.p0) @ seg) / max(L, 1e-9)
    keep = np.ones(len(xyz), dtype=bool)
    for b in bands:
        in_band = (xyz[:, 2] >= b.z_lo) & (xyz[:, 2] < b.z_hi)
        if b.clip_lo is not None:
            keep &= ~(in_band & (along < b.clip_lo))
        if b.clip_hi is not None:
            keep &= ~(in_band & (along > b.clip_hi))
        if b.status == "open":
            keep &= ~in_band
        elif b.status == "observed" and (b.along_hi - b.along_lo) < 0.8 * L:
            area = max(b.along_hi - b.along_lo, 1.0) * (b.z_hi - b.z_lo)
            if b.n_returns / area >= dense_per_m2:
                s0, s1 = band_extent(plane, b, full_width_fraction=0.8)
                keep &= ~(in_band & ((along < s0) | (along > s1)))
    return keep


def extrude_profile(plane: WallPlane, bands: List[FaceBand], spacing_m: float,
                    full_width_fraction: float = 0.6, margin_m: float = 3.0) -> np.ndarray:
    """Grid points on the wall as the profile describes it: every observed
    or bridged band, at its own offset, across the full edge when the returns
    reached most of it and across their own extent plus a margin otherwise."""
    seg = plane.p1 - plane.p0
    L = float(np.hypot(*seg))
    if L < 1e-6:
        return np.zeros((0, 3))
    e1 = seg / L
    out = []
    for b in bands:
        if b.status == "open" and b.back_offset_m is not None and (b.along_hi - b.along_lo) >= 3.0:
            # the wall behind the piers of a colonnade
            b = FaceBand(b.z_lo, b.z_hi, b.back_offset_m, b.n_returns, 0.0, b.along_lo, b.along_hi, "observed",
                         clip_lo=b.clip_lo, clip_hi=b.clip_hi)
        if b.status not in WALL_STATUSES or b.offset_m is None:
            continue
        s0, s1 = band_extent(plane, b, full_width_fraction, margin_m)
        s = np.arange(s0, s1, spacing_m)
        zz = np.arange(b.z_lo, b.z_hi, spacing_m)
        if len(s) == 0 or len(zz) == 0:
            continue
        # Each point along the edge takes the segment's offset where one
        # covers it, and the band's elsewhere.
        off = np.full(len(s), b.offset_m)
        for a0, a1, oseg in b.segments:
            off[(s >= a0) & (s < a1)] = oseg
        S, Z = np.meshgrid(s, zz)
        O = np.broadcast_to(off[None, :], S.shape)
        xy = plane.p0[None, :] + S.ravel()[:, None] * e1[None, :] + O.ravel()[:, None] * plane.normal[None, :]
        out.append(np.column_stack([xy, Z.ravel()]))
    return np.vstack(out) if out else np.zeros((0, 3))


def opposite_face(plane: WallPlane, planes: List[WallPlane]) -> Optional[WallPlane]:
    """The face across the building from ``plane``: normals within 25 deg of
    opposite, and the largest overlap when projected onto this edge."""
    seg = plane.p1 - plane.p0
    L = float(np.hypot(*seg))
    if L < 1e-6:
        return None
    e1 = seg / L
    best, best_overlap = None, 0.0
    for other in planes:
        if other.face == plane.face or float(other.normal @ plane.normal) > -0.9:
            continue
        a0, a1 = sorted([float((other.p0 - plane.p0) @ e1), float((other.p1 - plane.p0) @ e1)])
        overlap = max(0.0, min(a1, L) - max(a0, 0.0))
        if overlap > best_overlap:
            best, best_overlap = other, overlap
    return best if best_overlap >= 0.5 * L else None


def mirror_hidden_bands(planes: List[WallPlane], profiles: Dict[int, List[FaceBand]],
                        max_shape_diff_m: float = 0.6, min_joint_bands: int = 3) -> Dict[int, Dict[str, Any]]:
    """Fill a face's unseen bands from the face opposite, when the two
    faces have been seen to have the same shape.

    Buildings are mostly symmetric across their middle: a setback at the
    twentieth floor on the west wall is there on the east wall too. Where a
    face is hidden -- Humana's east wall in PNC Tower's shadow, a tower's
    short end the flight never looked at -- the opposite face is usually
    the best evidence there is. It is used only as a *shape*: the offsets
    are compared band by band where both faces were observed, relative to
    each face's own lowest observed wall (so a footprint drawn a metre off
    on one side does not break the match), and the pair is taken as
    symmetric when they agree within ``max_shape_diff_m`` on at least
    ``min_joint_bands`` bands. Then each unknown band on the hidden face
    takes the opposite band's offset (shifted by the difference of the two
    anchors) and its extent mirrored along the edge, with status
    ``mirrored``. A bridged band is replaced too where the twin was
    *observed* at that height: a bridge interpolates between its anchors and
    cannot know about a step between them; the twin can. A face with no
    observed band at all takes the opposite face's offsets as they are.
    Observed, extended, open and sloped bands are never replaced: a face's
    own measurement outranks the prior.

    Returns, per face that received bands, a small record of what was done.
    """
    by_face = {pl.face: pl for pl in planes}
    report: Dict[int, Dict[str, Any]] = {}
    for face, bands in profiles.items():
        pl = by_face.get(face)
        if pl is None:
            continue
        opp = opposite_face(pl, planes)
        if opp is None or opp.face not in profiles:
            continue
        other = profiles[opp.face]
        L_f = float(np.hypot(*(pl.p1 - pl.p0)))
        L_g = float(np.hypot(*(opp.p1 - opp.p0)))
        def band_at(bs: List[FaceBand], z: float) -> Optional[FaceBand]:
            return next((b for b in bs if b.z_lo <= z < b.z_hi), None)
        own_obs = [b for b in bands if b.status == "observed"]
        opp_obs = [b for b in other if b.status == "observed"]
        if not opp_obs:
            continue
        # Anchors: each face's lowest observed wall, the reference its shape is measured from.
        anchor_g = float(np.median([b.offset_m for b in sorted(opp_obs, key=lambda b: b.z_lo)[:3]]))
        if own_obs:
            anchor_f = float(np.median([b.offset_m for b in sorted(own_obs, key=lambda b: b.z_lo)[:3]]))
            joint = [(b, band_at(other, (b.z_lo + b.z_hi) / 2)) for b in own_obs]
            joint = [(b, g) for b, g in joint if g is not None and g.status == "observed"]
            if len(joint) < min_joint_bands:
                continue
            diffs = [abs((b.offset_m - anchor_f) - (g.offset_m - anchor_g)) for b, g in joint]
            shape_diff = float(np.percentile(diffs, 75))     # one odd band is tolerated, a second is not
            if shape_diff > max_shape_diff_m:
                report[face] = {"from_face": opp.face, "symmetric": False, "shape_diff_m": round(shape_diff, 2), "n_joint": len(joint), "mirrored": 0}
                continue
            shift = anchor_f - anchor_g
        else:
            joint, shape_diff, shift = [], None, 0.0
        n_mirrored = 0
        for i, b in enumerate(bands):
            # Unknown bands take the twin's word; so do bridged ones where
            # the twin was actually observed at that height -- a bridge
            # interpolates between anchors and cannot know about a step
            # between them, the twin can.
            if b.status not in ("unknown", "bridged"):
                continue
            g = band_at(other, (b.z_lo + b.z_hi) / 2)
            if g is None or g.offset_m is None:
                continue
            if b.status == "unknown" and g.status not in ("observed", "extended", "bridged"):
                continue
            if b.status == "bridged" and g.status != "observed":
                continue
            # The twin's band must itself be a wall that holds from one band
            # to the next -- the same test that keeps a vertical strip off a
            # dome. A twin band on a receding crown is not copied across.
            gi = other.index(g)
            steady = any(0 <= j < len(other) and other[j].status in WALL_STATUSES and other[j].offset_m is not None
                         and abs(other[j].offset_m - g.offset_m) <= 0.4 for j in (gi - 1, gi + 1))
            if not steady:
                continue
            lo = L_f * (1.0 - g.along_hi / L_g)
            hi = L_f * (1.0 - g.along_lo / L_g)
            bands[i] = FaceBand(b.z_lo, b.z_hi, g.offset_m + shift, b.n_returns, g.spread_m,
                                max(0.0, lo), min(L_f, hi), "mirrored")
            n_mirrored += 1
        report[face] = {"from_face": opp.face, "symmetric": True, "shape_diff_m": None if shape_diff is None else round(shape_diff, 2),
                        "n_joint": len(joint), "mirrored": n_mirrored, "anchored": bool(own_obs)}
    return report


def mirror_colour_from_twin(planes: List[WallPlane], profiles: Dict[int, List[FaceBand]],
                            mirror_report: Dict[int, Dict[str, Any]],
                            xyz: np.ndarray, face_of: np.ndarray, rgb: np.ndarray, seen: np.ndarray,
                            tolerance_m: float = 0.4) -> np.ndarray:
    """Colour a hidden face's inferred wall from the face opposite.

    Where a face's shape was taken from its twin, its colour can be too: a
    point on a mirrored or bridged band, or any point no camera coloured,
    takes the colour of the twin's point at the same height and the
    mirrored position along the edge, when the twin has a coloured point
    within ``tolerance_m`` there. Humana's east wall stands in PNC Tower's
    shadow and the obliques that reach it do so at a grazing angle, dark;
    its west wall was photographed. Observed bands keep their own colour.

    Modifies ``rgb`` and ``seen`` in place; returns a mask of the points
    whose colour came from the twin.
    """
    from scipy.spatial import cKDTree

    by_face = {pl.face: pl for pl in planes}
    took = np.zeros(len(xyz), dtype=bool)
    for face, rec in mirror_report.items():
        if not rec.get("symmetric"):
            continue
        pl, twin = by_face.get(face), by_face.get(rec["from_face"])
        if pl is None or twin is None:
            continue
        L_f = float(np.hypot(*(pl.p1 - pl.p0)))
        L_g = float(np.hypot(*(twin.p1 - twin.p0)))
        e_f = (pl.p1 - pl.p0) / L_f
        e_g = (twin.p1 - twin.p0) / L_g
        g_idx = np.flatnonzero((face_of == twin.face) & (seen > 0))
        if len(g_idx) < 100:
            continue
        # The twin's coloured points, laid out as they would sit on this face.
        a_g = (xyz[g_idx, :2] - twin.p0) @ e_g
        key_g = np.column_stack([L_f * (1.0 - a_g / L_g), xyz[g_idx, 2]])
        tree = cKDTree(key_g)
        f_idx = np.flatnonzero(face_of == face)
        if len(f_idx) == 0:
            continue
        z_f = xyz[f_idx, 2]
        inferred = seen[f_idx] == 0
        for b in profiles.get(face, []):
            if b.status in ("mirrored", "bridged"):
                inferred |= (z_f >= b.z_lo) & (z_f < b.z_hi)
        cand = f_idx[inferred]
        if len(cand) == 0:
            continue
        a_f = (xyz[cand, :2] - pl.p0) @ e_f
        d, j = tree.query(np.column_stack([a_f, xyz[cand, 2]]), distance_upper_bound=tolerance_m)
        hit = np.isfinite(d)
        src = g_idx[j[hit]]
        tgt = cand[hit]
        # Colour flows towards the darker side. A face in a neighbour's
        # shadow is coloured by grazing, shadowed oblique pixels; its twin
        # was lit or photographed. A point that already has colour keeps it
        # unless the twin's is clearly brighter; an uncoloured point takes
        # whatever the twin has.
        own_bright = rgb[tgt].astype(np.float64).mean(axis=1)
        twin_bright = rgb[src].astype(np.float64).mean(axis=1)
        take = (seen[tgt] == 0) | (twin_bright >= 1.2 * own_bright + 10.0)
        tgt, src = tgt[take], src[take]
        rgb[tgt] = rgb[src]
        seen[tgt] = np.maximum(seen[tgt], 1)
        took[tgt] = True
    return took


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
                    max_vertices: int = 16) -> List[Dict[str, Any]]:
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
        tower_ring = _mask_to_ring(m, cx, cy, cell_m, max_vertices)
        if tower_ring is None:
            continue
        levels.append({"level": j, "ring": tower_ring, "z_lo": podium_top, "z_hi": z_top})
    return levels if len(levels) > 1 else whole


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


# --------------------------------------------------- structural classification

@dataclass
class BuildingClassification:
    """A building's structural type, as a label for the viewer and the records.

    The label does not steer the mesh: every building is built from its own
    returns. A whole-building type is too coarse for a downtown (one building
    has a flat roof, a curved corner and a party wall at once), and where the
    LiDAR is dense the height field and the wall profiles already say what is
    there. Choosing a model by type belongs where the returns are too thin to
    show the shape -- and then per roof region or face group, checked on a
    before/after sheet.

    Types:
    - ``simple_box``: one level, flat roof, planar walls.
    - ``tiered_box``: several levels, flat roofs, planar walls.
    - ``complex_roof``: planar walls under a pitched, domed or faceted roof.
    - ``sculpted_facade``: curved walls, colonnades, or walls stepping in and out.
    """

    archetype: str
    roof_type: str
    facade_type: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"archetype": self.archetype, "roof_type": self.roof_type,
                "facade_type": self.facade_type, "metrics": self.metrics}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BuildingClassification":
        return cls(archetype=d["archetype"], roof_type=d.get("roof_type", "unknown"),
                   facade_type=d.get("facade_type", "unknown"), metrics=d.get("metrics", {}))


#: A roof is flat when the returns sit within this RMS of one plate (m) ...
FLAT_PLATE_RMS_M = 0.35
#: ... most roof returns are on that plate (the rest is parapets and plant) ...
FLAT_PLATEAU_FRAC = 0.50
#: ... and under this share of roof cells slope like a pitched roof.
FLAT_MAX_SLOPED_FRAC = 0.05
#: A street wall stepping in or out by more than this is not planar (m).
PLANAR_MAX_OFFSET_M = 0.45


def classify_building_structure(
    levels: Sequence[Dict[str, Any]],
    zones: Optional[SurfaceZones],
    profiles: Optional[Dict[int, Sequence[Any]]] = None,
    arc_edges: Optional[Union[Set[int], Sequence[int]]] = None,
    *,
    sloped_fraction: float,
    party_faces: Optional[Union[Set[int], Sequence[int]]] = None,
) -> BuildingClassification:
    """Label a building's structure from what the colourise and the roof field measured.

    ``levels`` from :func:`building_levels`; ``zones`` from
    :func:`split_surface_zones`; ``profiles`` maps a face to its bands;
    ``arc_edges`` are the faces on curved walls; ``sloped_fraction`` is the
    share of roof cells that slope like a pitched roof (see
    ``roof_sloped_fraction`` in build_mesh); ``party_faces`` are the faces
    against a neighbour, whose returns are the neighbour's and are ignored.
    """
    profiles = profiles or {}
    party = set(party_faces or [])
    has_arcs = bool(arc_edges)
    street = [b for f, bands in profiles.items() if f not in party for b in bands]
    max_wall_offset = max((abs(float(b.offset_m)) for b in street
                           if b.status in ("observed", "bridged", "extended") and b.offset_m is not None),
                          default=0.0)
    has_open_bands = any(b.status == "open" for b in street)

    if zones is not None:
        plate_rms, roof_frac = zones.plate_rms_m, zones.roof_plateau_frac
        is_flat_roof = (plate_rms is not None and plate_rms <= FLAT_PLATE_RMS_M
                        and roof_frac >= FLAT_PLATEAU_FRAC and sloped_fraction < FLAT_MAX_SLOPED_FRAC)
    else:
        plate_rms = roof_frac = None
        is_flat_roof = False
    is_planar_facade = not has_arcs and not has_open_bands and max_wall_offset < PLANAR_MAX_OFFSET_M

    num_levels = max(len(levels), 1)
    if is_flat_roof and is_planar_facade:
        archetype = "simple_box" if num_levels == 1 else "tiered_box"
    elif is_planar_facade and num_levels == 1:
        archetype = "complex_roof"
    else:
        archetype = "sculpted_facade"

    return BuildingClassification(
        archetype=archetype,
        roof_type="flat" if is_flat_roof else "complex",
        facade_type="planar" if is_planar_facade else "sculpted",
        metrics={
            "num_levels": num_levels,
            "plate_rms_m": round(plate_rms, 4) if plate_rms is not None else None,
            "roof_plateau_frac": round(roof_frac, 4) if roof_frac is not None else None,
            "sloped_fraction": round(float(sloped_fraction), 4),
            "max_wall_offset_m": round(max_wall_offset, 3),
            "has_arcs": has_arcs,
            "has_open_bands": has_open_bands,
            "party_faces": len(party),
        },
    )
