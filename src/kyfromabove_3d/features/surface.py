"""A building's surface sorted into structure, rough and noise; its zones; how much of a
facade the LiDAR captured."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .planes import fit_local_planes, surface_coherence

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
