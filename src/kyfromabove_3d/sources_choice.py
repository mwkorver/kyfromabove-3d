"""Which LiDAR and which oblique photos a LiDAR tile's buildings are made from.

Geometry comes from the LiDAR, colour from the oblique frames; a building
built or torn down between the two captures gets a shape from one moment
and a picture from another. KyFromAbove has two COPC LiDAR phases (Phase 2
statewide, flown 2019 to early 2024 by region; Phase 3 a re-flight of the
east, 2023-2025) and four oblique collections (2022 Season 2, 2023 Seasons 1
and 2, 2024 Season 1), and half of Kentucky's buildings would pair Phase 2
with obliques more than 18 months later (measured September 2026).

The rule (the user's): per tile, the LiDAR phase and oblique collection whose
captures are **closest together in time**; a tie goes to the newer LiDAR.

* A LiDAR tile is dated by its COPC header's GPS times (the midpoint). They
  are adjusted standard GPS time -- seconds since the GPS epoch less 10^9 --
  although the header's flag says week time; values above 10^8 are read as
  adjusted. Its density (returns per square metre) is recorded: Phase 3 can
  be half as dense as Phase 2 over the same ground.
* A collection is a candidate where its published outline
  (``ImageFrameBoundary.gpkg``) meets the tile; it is dated by the median
  ``FlightDate`` of its frames over the tile.

Each phase gets its own single-phase :class:`LidarTileIndex`: an index
holding two phases returns both phases' tiles for a box, and a clip read
through it would merge two captures of the same ground.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .coordinates import METERS_PER_US_SURVEY_FOOT

PHASES = ("phase2", "phase3")
GPS_EPOCH = dt.datetime(1980, 1, 6)
OBLIQUE_BOUNDARY = ("https://kyfromabove.s3.us-west-2.amazonaws.com/imagery/obliques/Phase3/"
                    "FlightInformationData_GPKG/KY_KYAPED_{c}_3IN_ImageFrameBoundary.gpkg")
_OUTLINES: Dict[str, Any] = {}


def indexes(raw_dir: Path, phases: Sequence[str] = PHASES) -> Dict[str, Any]:
    """One single-phase LiDAR index per phase that has a grid in ``raw_dir``."""
    from .lidar import LidarTileIndex

    out = {}
    for p in phases:
        idx = LidarTileIndex(raw_dir, phases=(p,))
        if idx.phases:
            out[p] = idx
    return out


def index_for(indexes_or_index, phase: Optional[str]):
    """The index for ``phase`` from a dict of single-phase indexes (a plain
    index is returned as it is: the steps before the choice existed)."""
    if isinstance(indexes_or_index, dict):
        return indexes_or_index[phase or "phase2"]
    return indexes_or_index


def gps_to_datetime(gps: float) -> Optional[dt.datetime]:
    """A LAS GPS time as a date: adjusted standard (seconds since the GPS
    epoch less 10^9) when it is large enough to be one; None for week time."""
    if gps is None or not np.isfinite(gps) or gps < 1e8:
        return None
    return GPS_EPOCH + dt.timedelta(seconds=float(gps) + 1e9)


def lidar_candidates(tile: str, idx: Dict[str, Any], tile_area_m2: float) -> List[dict]:
    """Each phase that has the tile: its URL, flight date and density."""
    from .lidar import copc_reader

    out = []
    for phase, index in idx.items():
        grid = index.grids[phase]
        row = grid[grid["tilename"] == tile]
        if not len(row):
            continue
        url = row.iloc[0]["aws_url"]
        r = copc_reader(url)
        ci, h = r.copc_info, r.header
        when = gps_to_datetime((ci.gps_min + ci.gps_max) / 2.0)
        out.append({"phase": phase, "url": url, "date": when,
                    "points_per_m2": round(float(h.point_count) / max(tile_area_m2, 1.0), 2)})
    return out


def _outline(collection: str):
    """A collection's published outline, in EPSG:3089 (read once per process)."""
    if collection not in _OUTLINES:
        import geopandas as gpd

        g = gpd.read_file("/vsicurl/" + OBLIQUE_BOUNDARY.format(c=collection))
        _OUTLINES[collection] = g.to_crs("EPSG:3089").geometry.union_all()
    return _OUTLINES[collection]


def oblique_candidates(bounds_ft: Tuple[float, float, float, float], work: Path,
                       collections: Optional[Sequence[str]] = None, tile: str = "tile",
                       log: Callable = print) -> List[dict]:
    """Each collection with frames over the tile: its frames (a GeoDataFrame,
    cached as ``work/frames_<tile>_<collection>.gpkg``) and their median
    flight date."""
    import pandas as pd
    from shapely.geometry import box

    from .prepare import COLLECTIONS, load_frames_index

    out = []
    b = box(*bounds_ft)
    for c in (collections or COLLECTIONS):
        try:
            if not _outline(c).intersects(b):
                continue
        except Exception as exc:                                    # noqa: BLE001 (an outline we cannot read)
            log(f"[warn] no outline for {c} ({exc!r}); trying its frames")
        try:
            gdf = load_frames_index(Path(work) / f"frames_{tile}_{c}.gpkg", bounds_ft, collection=c)
        except RuntimeError:
            continue                                                # outline meets, no frame does
        dates = pd.to_datetime(gdf["FlightDate"], errors="coerce").dropna()
        out.append({"collection": c, "frames": gdf, "frames_path": Path(work) / f"frames_{tile}_{c}.gpkg",
                    "n_frames": int(len(gdf)), "date": dates.median().to_pydatetime() if len(dates) else None})
    return out


def choose(lidar: Sequence[dict], obliques: Sequence[dict]) -> Tuple[dict, dict]:
    """The pair closest together in time; a tie goes to the newer LiDAR. A
    candidate with no date loses to any dated one. No oblique collection over
    the tile: the newest dated LiDAR, and None -- the tile's buildings still get
    their LiDAR heights (product 1), as boxes, with nothing to colour a mesh."""
    if not lidar:
        raise RuntimeError("no LiDAR for this tile")
    if not obliques:
        return max(lidar, key=lambda c: (c["date"] is not None, c["date"] or dt.datetime.min)), None
    far = dt.timedelta(days=10 ** 5)

    def key(pair):
        l, o = pair
        gap = abs(o["date"] - l["date"]) if (l["date"] and o["date"]) else far
        return (gap, -l["date"].timestamp() if l["date"] else float("inf"))   # ties: the newer LiDAR
    return min(((l, o) for l in lidar for o in obliques), key=key)


def provenance(l: dict, o: Optional[dict]) -> Dict[str, Any]:
    """What goes on every building of the tile, and in its summary (``o``
    None: no photos; its fields None)."""
    gap = ((o["date"] - l["date"]).days / 30.44) if (o and l["date"] and o["date"]) else None
    return {"lidar_phase": l["phase"], "lidar_date": l["date"].date().isoformat() if l["date"] else None,
            "lidar_points_per_m2": l["points_per_m2"], "oblique_collection": o["collection"] if o else None,
            "oblique_date": o["date"].date().isoformat() if (o and o["date"]) else None,
            "gap_months": None if gap is None else round(gap, 1) + 0.0}          # no "-0.0"


def choose_sources(tile: str, grid, idx: Dict[str, Any], work: Path, log: Callable = print) -> Dict[str, Any]:
    """The tile's choice: {"provenance": fields for every building, "frames":
    the chosen collection's frames, "frames_path", "phase", "candidates": what
    was considered}."""
    from .pieces import tile_bounds_ft

    b = tile_bounds_ft(grid, tile)
    area_m2 = (b[2] - b[0]) * (b[3] - b[1]) * METERS_PER_US_SURVEY_FOOT ** 2
    lidar = lidar_candidates(tile, idx, area_m2)
    obliques = oblique_candidates(tile_bounds_ft(grid, tile, pad_ft=300.0), work, tile=tile, log=log)
    l, o = choose(lidar, obliques)
    prov = provenance(l, o)
    cands = {"lidar": [{k: (v.date().isoformat() if isinstance(v, dt.datetime) else v) for k, v in c.items()
                        if k != "url"} for c in lidar],
             "obliques": [{"collection": c["collection"], "n_frames": c["n_frames"],
                           "date": c["date"].date().isoformat() if c["date"] else None} for c in obliques]}
    log(f"{tile}: {prov['lidar_phase']} ({prov['lidar_date']}) with {prov['oblique_collection']} "
        f"({prov['oblique_date']}), {prov['gap_months']} months apart; considered "
        f"{[c['phase'] for c in lidar]} x {[c['collection'] for c in obliques]}")
    return {"provenance": prov, "frames": o["frames"] if o else [], "frames_path": o["frames_path"] if o else None,
            "phase": l["phase"], "candidates": cands}
