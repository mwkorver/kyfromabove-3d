"""Where a building's footprint comes from, in one place.

Three scripts used to carry their own copy of the same lookup -- a case name
in ``outputs/survey/case_footprints.json``, or a GERS id in the facade CSV --
and a fourth source arrived with the downtown run: a ``footprint.json``
written into the building's own directory by ``prepare``, which is the only
one that knows the building's neighbours and its LiDAR ground. This module
resolves any of them to the same record, and never falls back silently: the
caller says which source it means, and a miss is an error.
"""

from __future__ import annotations

import csv
import functools
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from kyfromabove_3d.coordinates import METERS_PER_US_SURVEY_FOOT

CASE_FOOTPRINTS = Path("outputs/survey/case_footprints.json")
FACADE_CSV = Path("data/derived/ky_facade_tiled_wkt.csv")


def largest_ring(geom):
    """One exterior ring for a footprint: the largest polygon of a
    MultiPolygon, holes dropped, an invalid polygon repaired."""
    from shapely.geometry import Polygon

    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type != "Polygon":
        raise ValueError(f"footprint is a {geom.geom_type}, not a polygon")
    return Polygon(geom.exterior.coords)


def ring_local_m(ring_ft: np.ndarray, ox: float, oy: float) -> np.ndarray:
    """State-plane feet -> metres from the local origin."""
    ring_ft = np.asarray(ring_ft, dtype=np.float64)
    return np.column_stack([(ring_ft[:, 0] - ox) * METERS_PER_US_SURVEY_FOOT,
                            (ring_ft[:, 1] - oy) * METERS_PER_US_SURVEY_FOOT])


@functools.lru_cache(maxsize=None)
def wgs84_to_state_plane():
    """The WGS84 -> EPSG:3089 transformer, made once per process. Making one
    searches PROJ's database (1.6 ms on the laptop, several times that on a
    Graviton core); the neighbour prisms converted 2,433 outlines a script,
    each with a new one -- most of a small building's colourise, mesh and
    check (September 2026)."""
    from pyproj import Transformer

    return Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True)


def footprint_wgs84_to_local(wkt: str, ox: float, oy: float):
    """A WKT footprint in WGS84 -> shapely Polygon in the local metre frame."""
    from shapely.geometry import Polygon
    from shapely.ops import transform as shp_transform
    from shapely.wkt import loads as wkt_loads

    poly_ft = largest_ring(shp_transform(wgs84_to_state_plane().transform, wkt_loads(wkt)))
    return Polygon(ring_local_m(np.array(poly_ft.exterior.coords), ox, oy))


def load_footprint_entry(case: Optional[str] = None, gers: Optional[str] = None,
                         footprint_file: Optional[Path] = None,
                         footprints_json: Path = CASE_FOOTPRINTS,
                         footprint_csv: Path = FACADE_CSV) -> Dict[str, Any]:
    """The footprint record from exactly one source.

    Returns a dict with at least ``gers``, ``name``, ``wkt`` (WGS84), and
    whatever else the source carries (``height_m``, ``ground_z_m``,
    ``neighbours`` from a ``footprint.json``).
    """
    given = [x for x in (case, gers, footprint_file) if x]
    if len(given) != 1:
        raise ValueError("give exactly one of case, gers, footprint_file")
    if footprint_file is not None:
        entry = json.loads(Path(footprint_file).read_text(encoding="utf-8"))
        entry.setdefault("name", entry.get("gers"))
        return entry
    if case is not None:
        fp = json.loads(Path(footprints_json).read_text(encoding="utf-8"))
        if case not in fp:
            raise KeyError(f"'{case}' not in {footprints_json}. Have: {', '.join(sorted(fp))}")
        e = dict(fp[case])
        e.setdefault("name", e.get("overture_name", case))
        e.setdefault("gers", e.get("id", case))
        return e
    csv.field_size_limit(1 << 30)
    with open(footprint_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["id"] == gers:
                return {"gers": gers, "name": gers, "wkt": r["wkt"],
                        "height_m": float(r["est_height_m"]) if r.get("est_height_m") else None}
    raise KeyError(f"GERS {gers} not in {footprint_csv}")


def resolve_footprint(building: Path, ox: float, oy: float, case: Optional[str] = None,
                      gers: Optional[str] = None, footprint: Optional[Path] = None,
                      footprints_json: Path = CASE_FOOTPRINTS,
                      footprint_csv: Path = FACADE_CSV) -> Tuple[Any, Dict[str, Any]]:
    """(polygon in the local frame, record). With no source named, the
    building's own ``footprint.json`` is used when it exists."""
    if not any((case, gers, footprint)):
        own = Path(building) / "footprint.json"
        if own.exists():
            footprint = own
        else:
            raise ValueError("give --case, --gers or --footprint (no footprint.json in the building dir)")
    entry = load_footprint_entry(case=case, gers=gers, footprint_file=footprint,
                                 footprints_json=footprints_json, footprint_csv=footprint_csv)
    return footprint_wgs84_to_local(entry["wkt"], ox, oy), entry
