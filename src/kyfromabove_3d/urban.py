"""Finding buildings worth reconstructing, and the imagery that covers them.

The pilot building was chosen by hand. Scaling past one needs the choice to be
made from data, and two facts make that awkward:

* **The oblique imagery is not statewide.** KyFromAbove flew four separate
  oblique collections with barely-overlapping footprints, and only one of them
  covers Kentucky's urban cores. Picking a building before checking which
  collection sees it is how you pick a building with no imagery.
* **The frame indexes are enormous.** ``KY_KYAPED_2024_Season1_3IN``'s index is
  1.07 GB. It never has to be downloaded: a GeoPackage is SQLite with an RTree,
  so GDAL's ``/vsicurl`` reads it by range request. A downtown query returns in
  seconds against a file that stays on S3.

Building height comes from the Phase 2 COPC point cloud rather than from any
footprint database, because height is what decides whether a building is an
interesting test. The pilot is 16 m and effectively all roof; a 164 m tower is
mostly facade, which is exactly where aerial LiDAR is thinnest and the obliques
are the only real source.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: The four oblique collections, with the extents measured from their published
#: ``ImageFrameBoundary`` files. Only 2024_Season1 reaches the urban cores --
#: 2022_Season2, which the pilot uses, is a 19 km-wide rural corridor.
OBLIQUE_COLLECTIONS: Dict[str, Dict[str, Any]] = {
    "2022_Season2": {"lon": (-83.06, -82.84), "lat": (37.66, 38.80),
                     "note": "narrow eastern corridor; contains the pilot, no cities"},
    "2023_Season1": {"lon": (-87.11, -81.92), "lat": (36.54, 38.87),
                     "note": "Ashland, Richmond, Somerset, Pikeville"},
    "2023_Season2": {"lon": (-87.48, -83.26), "lat": (36.59, 38.20),
                     "note": "Owensboro, Bowling Green"},
    "2024_Season1": {"lon": (-89.61, -83.68), "lat": (36.46, 39.19),
                     "note": "Louisville, Lexington, Covington, Frankfort, Paducah"},
}

S3_BASE = ("https://kyfromabove.s3.us-west-2.amazonaws.com/imagery/obliques/Phase3")


def collection_key(collection: str) -> str:
    """Normalise ``KY_KYAPED_2024_Season1_3IN`` to the ``2024_Season1`` key.

    Configs name the collection the way S3 does; this module keys on the short
    form. Accepting either avoids making every caller remember which is which.
    """
    key = collection
    if key.startswith("KY_KYAPED_"):
        key = key[len("KY_KYAPED_"):]
    if key.endswith("_3IN"):
        key = key[: -len("_3IN")]
    if key not in OBLIQUE_COLLECTIONS:
        raise KeyError(f"unknown collection {collection!r}; "
                       f"expected one of {sorted(OBLIQUE_COLLECTIONS)}")
    return key


def collection_full_name(collection: str) -> str:
    """The S3 / filename form, e.g. ``KY_KYAPED_2024_Season1_3IN``."""
    return f"KY_KYAPED_{collection_key(collection)}_3IN"


def frames_index_url(collection: str) -> str:
    """The ``/vsicurl`` path to a collection's frame index.

    Returned as a GDAL virtual path rather than a plain URL: the point is to
    range-read the index in place, never to download it.
    """
    return (f"/vsicurl/{S3_BASE}/FlightInformationData_GPKG/"
            f"{collection_full_name(collection)}_ImageFrames.gpkg")


def eo_index_url(collection: str) -> str:
    """The ``/vsicurl`` path to a collection's exterior-orientation table."""
    return (f"/vsicurl/{S3_BASE}/FlightInformationData_GPKG/"
            f"{collection_full_name(collection)}_ImageFrameEO.gpkg")


def eo_table_name(collection: str) -> str:
    return f"{collection_full_name(collection)}_ImageFrameEO"


def collections_covering(lon: float, lat: float) -> List[str]:
    """Collections whose published extent contains a point.

    A bounding-box test, so it over-reports: the real footprints have 47 holes
    in the 2024 case. Use it to shortlist, then confirm against the frame index.
    """
    hits = []
    for name, meta in OBLIQUE_COLLECTIONS.items():
        (lo_x, hi_x), (lo_y, hi_y) = meta["lon"], meta["lat"]
        if lo_x <= lon <= hi_x and lo_y <= lat <= hi_y:
            hits.append(name)
    return hits


def heights_above_ground(
    xyz: np.ndarray,
    classification: np.ndarray,
    grid_ft: float = 100.0,
) -> np.ndarray:
    """Height of every point above a local ground surface, in the input's units.

    The ground surface is the median of class-2 returns on a coarse grid, not a
    single elevation, because a downtown spans real terrain -- Louisville's
    riverfront falls away from the core, and a flat datum would report the slope
    as building height. Cells with no ground return yield NaN rather than a
    guess; the caller drops them.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    classification = np.asarray(classification)
    gi = np.floor(xyz[:, 0] / grid_ft).astype(np.int64)
    gj = np.floor(xyz[:, 1] / grid_ft).astype(np.int64)

    ground = classification == 2
    cells: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for i, j, z in zip(gi[ground], gj[ground], xyz[ground, 2]):
        cells[(int(i), int(j))].append(float(z))
    medians = {k: float(np.median(v)) for k, v in cells.items()}

    base = np.array([medians.get((int(i), int(j)), np.nan) for i, j in zip(gi, gj)])
    return xyz[:, 2] - base


@dataclass
class TallCluster:
    """A candidate structure, ranked by height."""

    height_ft: float
    height_m: float
    n_points: int
    x_ft: float
    y_ft: float
    lon: float
    lat: float
    frames: Optional[int] = None
    frames_by_head: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __repr__(self) -> str:
        cov = "" if self.frames is None else f", {self.frames} frames"
        return (f"TallCluster({self.height_ft:.0f} ft / {self.height_m:.0f} m "
                f"@ {self.lon:.5f},{self.lat:.5f}{cov})")


def find_tall_clusters(
    xyz: np.ndarray,
    classification: np.ndarray,
    to_lonlat,
    cell_ft: float = 60.0,
    min_points: int = 40,
    min_separation_ft: float = 400.0,
    percentile: float = 95.0,
    max_results: int = 10,
    exclude_classes: Sequence[int] = (2, 7, 18),
) -> List[TallCluster]:
    """Rank distinct tall structures in a point cloud.

    Height per cell is a **percentile, not the maximum**: a single bird or
    antenna return would otherwise outrank an entire tower. Class 7 and 18
    (noise, high noise) are dropped for the same reason.

    ``min_separation_ft`` suppresses neighbouring cells of one building, so the
    result is a list of distinct structures rather than a list of roof pixels
    from the tallest one.
    """
    heights = heights_above_ground(xyz, classification)
    keep = np.isfinite(heights) & ~np.isin(classification, list(exclude_classes))
    if not keep.any():
        return []

    pts, hts = np.asarray(xyz)[keep], heights[keep]
    ci = np.floor(pts[:, 0] / cell_ft).astype(np.int64)
    cj = np.floor(pts[:, 1] / cell_ft).astype(np.int64)
    cells: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for i, j, h in zip(ci, cj, hts):
        cells[(int(i), int(j))].append(float(h))

    ranked = sorted(
        ((float(np.percentile(v, percentile)), len(v), i, j)
         for (i, j), v in cells.items() if len(v) >= min_points),
        reverse=True,
    )

    out: List[TallCluster] = []
    for height, n, i, j in ranked:
        x, y = (i + 0.5) * cell_ft, (j + 0.5) * cell_ft
        if any(abs(x - c.x_ft) < min_separation_ft and abs(y - c.y_ft) < min_separation_ft
               for c in out):
            continue
        lon, lat = to_lonlat(x, y)
        out.append(TallCluster(height_ft=height, height_m=height * 0.3048006096,
                               n_points=n, x_ft=x, y_ft=y, lon=lon, lat=lat))
        if len(out) >= max_results:
            break
    return out
