#!/usr/bin/env python3
"""When each part of Kentucky was flown, and which pair a statewide run would build from.

For every Phase 2 LiDAR tile (the 5,000 ft grid the queue run works in):

* **the LiDAR's flight dates**, Phase 2 and, where it has re-flown the tile,
  Phase 3: from each COPC file's own header (the GPS times of its first and
  last return, adjusted standard GPS time; ``sources_choice.gps_to_datetime``)
  -- one 600-byte range read a file, not the file;
* **each oblique collection's flight dates** over the tile: the
  ``FlightDate`` of every photo whose centre falls in it, from the
  collection's ``ImageFrameCentroids.gpkg`` (the median, the first and the
  last day);
* **how many buildings** stand on it, and their footprint areas: Overture's
  outlines (the release the pipeline pins), each counted on the tile its
  bounding box's centre falls in;

then the pair the pipeline would take there (``sources_choice``): the LiDAR
phase and the collection flown closest together, a tie to the newer LiDAR,
and the gap between them. A collection counts on a tile from
``--min-frames`` photos up.

Each part is cached under ``--out-dir`` and read back on a rerun; delete a
file to fetch it again. Reads only KyFromAbove's and Overture's public
buckets.

Usage::

    python scripts/survey_statewide_dates.py
    python scripts/survey_statewide_dates.py --out-dir outputs/survey/statewide
"""

from __future__ import annotations

import argparse
import struct
import sys
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from kyfromabove_3d.lidar import LidarTileIndex
from kyfromabove_3d.prepare import COLLECTIONS
from kyfromabove_3d.sources_choice import gps_to_datetime

CENTROIDS = ("https://kyfromabove.s3.us-west-2.amazonaws.com/imagery/obliques/Phase3/"
             "FlightInformationData_GPKG/KY_KYAPED_{c}_3IN_ImageFrameCentroids.gpkg")
#: the release scripts/fetch_overture_buildings.py and the tile step pin
OVERTURE_RELEASE = "2026-10-21.0"
OVERTURE = f"s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}/theme=buildings/type=building/*"
KY_BBOX = (-89.6, 36.4, -81.9, 39.2)
FT2 = 0.3048006096012192 ** 2


def copc_dates(url: str) -> dict:
    """A COPC file's GPS time span and point count, from its first 600 bytes
    (the LAS 1.4 header, 375 bytes, then the COPC info record). A tile the
    grid spells ``.copz.laz`` may be stored as ``.copc.laz``, and the other
    way round (``lidar.py`` reads them the same way)."""
    other = url.replace(".copz.", ".copc.") if ".copz." in url else url.replace(".copc.", ".copz.")
    b = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url if attempt % 2 == 0 else other, headers={"Range": "bytes=0-599"})
            b = urllib.request.urlopen(req, timeout=30).read()
            break
        except Exception:                                           # noqa: BLE001
            continue
    if b is None:
        return {"url": url, "error": True}
    hdr = struct.unpack_from("<H", b, 94)[0]                         # header size
    n_points = struct.unpack_from("<Q", b, 247)[0]
    info = hdr + 54                                                  # the first VLR's payload: copc info
    gps_min, gps_max = struct.unpack_from("<dd", b, info + 56)
    lo, hi = gps_to_datetime(gps_min), gps_to_datetime(gps_max)
    return {"url": url, "n_points": int(n_points),
            "first": lo.date() if lo else None, "last": hi.date() if hi else None}


def lidar_dates(raw_dir: Path, out: Path, workers: int) -> pd.DataFrame:
    if out.exists():
        return pd.read_pickle(out)
    import geopandas as gpd
    rows = []
    for phase in ("phase2", "phase3"):
        g = gpd.read_file(raw_dir / f"kyfromabove_{phase}_pointcloud_5k_grid.gpkg")
        urls = list(g["aws_url"])
        print(f"  {phase}: {len(urls):,} headers ...", flush=True)
        with ThreadPoolExecutor(workers) as ex:
            got = list(ex.map(copc_dates, urls))
        for name, r in zip(g["tilename"], got):
            rows.append({"tile": name, "phase": phase, **r})
    df = pd.DataFrame(rows)
    df.to_pickle(out)
    return df


def oblique_dates(grid, out: Path) -> pd.DataFrame:
    """Per tile and collection: how many photos, and their median, first and last day."""
    if out.exists():
        return pd.read_pickle(out)
    import geopandas as gpd
    import pyogrio
    parts = []
    for c in COLLECTIONS:
        print(f"  {c}: reading every photo's date ...", flush=True)
        df = pyogrio.read_dataframe("/vsicurl/" + CENTROIDS.format(c=c), columns=["FlightDate", "CameraID"])
        df = df.to_crs(grid.crs)
        j = gpd.sjoin(df[["FlightDate", "geometry"]], grid[["tilename", "geometry"]], predicate="within")
        d = pd.to_datetime(j["FlightDate"])
        agg = (pd.DataFrame({"tile": j["tilename"].values, "d": d.values})
               .groupby("tile")["d"].agg(n="count", median="median", first="min", last="max").reset_index())
        agg["collection"] = c
        parts.append(agg)
        print(f"    {len(df):,} photos, {d.min().date()} to {d.max().date()}, on {len(agg):,} tiles", flush=True)
    out_df = pd.concat(parts, ignore_index=True)
    out_df.to_pickle(out)
    return out_df


def buildings(grid, out: Path) -> pd.DataFrame:
    """Every Overture building in the Kentucky box: its tile and footprint area."""
    if out.exists():
        return pd.read_pickle(out)
    import duckdb
    import geopandas as gpd
    lon0, lat0, lon1, lat1 = KY_BBOX
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial; SET s3_region='us-west-2';")
    print(f"  Overture {OVERTURE_RELEASE}: every building in the Kentucky box ...", flush=True)
    df = con.execute(f"""
        SELECT (bbox.xmin + bbox.xmax) / 2 AS lon, (bbox.ymin + bbox.ymax) / 2 AS lat,
               ST_Area(ST_Transform(geometry, 'EPSG:4326', 'EPSG:3089', true)) * {FT2} AS area_m2
        FROM read_parquet('{OVERTURE}', filename=false, hive_partitioning=1)
        WHERE bbox.xmin BETWEEN {lon0} AND {lon1} AND bbox.ymin BETWEEN {lat0} AND {lat1}
    """).df()
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326").to_crs(grid.crs)
    j = gpd.sjoin(pts, grid[["tilename", "geometry"]], predicate="within")
    b = pd.DataFrame({"tile": j["tilename"].values, "area_m2": j["area_m2"].values})
    b.to_pickle(out)
    print(f"    {len(df):,} in the box, {len(b):,} on the LiDAR grid", flush=True)
    return b


def pair(lid: pd.DataFrame, obl: pd.DataFrame, min_frames: int) -> pd.DataFrame:
    """Per tile, the pipeline's choice: the LiDAR and collection flown closest together."""
    lid = lid[lid["first"].notna()].copy()
    lid["date"] = pd.to_datetime(lid["first"]) + (pd.to_datetime(lid["last"]) - pd.to_datetime(lid["first"])) / 2
    obl = obl[obl["n"] >= min_frames]
    by_tile_obl = {t: g for t, g in obl.groupby("tile")}
    out = []
    for tile, g in lid.groupby("tile"):
        best = None
        o = by_tile_obl.get(tile)
        for _, l in g.iterrows():
            if o is None or not len(o):
                cand = (np.inf, -l["date"].value, l, None)
            else:
                for _, c in o.iterrows():
                    gap = abs((c["median"] - l["date"]).days)
                    cand = (gap, -l["date"].value, l, c)
                    if best is None or cand[:2] < best[:2]:
                        best = cand
                continue
            if best is None or cand[:2] < best[:2]:
                best = cand
        gap, _, l, c = best
        p2 = g[g["phase"] == "phase2"]
        out.append({"tile": tile, "phase": l["phase"], "lidar_date": l["date"].date(),
                    "lidar_first": l["first"], "lidar_last": l["last"], "n_points": l["n_points"],
                    "phase2_date": p2["date"].iloc[0].date() if len(p2) else None,
                    "phases": len(g),
                    "collection": None if c is None else c["collection"],
                    "oblique_date": None if c is None else c["median"].date(),
                    "oblique_first": None if c is None else c["first"].date(),
                    "oblique_last": None if c is None else c["last"].date(),
                    "n_collections": 0 if o is None else len(o),
                    "gap_days": None if c is None else int((c["median"] - l["date"]).days)})
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/survey/statewide"))
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--min-frames", type=int, default=5,
                    help="photos of a collection over a tile for it to count there")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    import geopandas as gpd

    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid = gpd.read_file(args.raw_dir / "kyfromabove_phase2_pointcloud_5k_grid.gpkg")
    if grid.crs is None:
        grid = grid.set_crs(LidarTileIndex.GRID_CRS)
    print("LiDAR flight dates", flush=True)
    lid = lidar_dates(args.raw_dir, args.out_dir / "lidar_dates.pkl", args.workers)
    print(f"  {int(lid['error'].fillna(False).sum()) if 'error' in lid else 0} headers unreadable", flush=True)
    print("oblique flight dates", flush=True)
    obl = oblique_dates(grid, args.out_dir / "oblique_dates.pkl")
    print("buildings", flush=True)
    bld = buildings(grid, args.out_dir / "buildings.pkl")
    print("pairing", flush=True)
    p = pair(lid, obl, args.min_frames)
    counts = bld.groupby("tile")["area_m2"].agg(buildings="count", area_m2="sum").reset_index()
    p = p.merge(counts, on="tile", how="left").fillna({"buildings": 0, "area_m2": 0.0})
    p.to_pickle(args.out_dir / "pairing.pkl")
    print(f"{len(p):,} tiles, {int(p['buildings'].sum()):,} buildings -> {args.out_dir / 'pairing.pkl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
