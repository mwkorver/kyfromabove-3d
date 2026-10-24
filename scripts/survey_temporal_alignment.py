#!/usr/bin/env python3
"""Map where Kentucky's LiDAR and oblique imagery were flown closest in time.

Colorizing a point cloud from oblique imagery joins two separately contracted
acquisitions. They are not the same flight: measured at Louisville the LiDAR
leads the obliques by two months, at Lexington it *follows* them by two weeks,
and at the Grayson pilot the gap is over two years. Where the gap is weeks that
is immaterial. Where it is years, anything that changed between them is wrong,
and the product would be 2024 geometry wearing 2022 colour.

So the place to start is wherever the two line up closest. This samples points
across the state and reports, for each, the LiDAR acquisition date, the nearest
oblique flight date, and the gap.

**The LiDAR date is the LAS header's file-creation field**, which is nominally
processing rather than acquisition. A gap of years is far too large to be a
processing artefact; a gap of weeks may partly be one. Read it as an upper
bound on how well aligned a site is, not as a flight log.

Usage::

    python scripts/survey_temporal_alignment.py --samples 120
    python scripts/survey_temporal_alignment.py --samples 40 --out outputs/survey/temporal.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from kyfromabove_3d.lidar import LidarTileIndex, read_las_header
from kyfromabove_3d.urban import collections_covering, frames_index_url


def parse_date(text: str):
    try:
        return dt.date.fromisoformat(str(text)[:10])
    except (ValueError, TypeError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pad-ft", type=float, default=1500.0,
                    help="half-width of the oblique frame query around each sample")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/survey/temporal_alignment.json"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    import geopandas as gpd
    from pyproj import Transformer

    grid = gpd.read_file(args.raw_dir / "kyfromabove_phase2_pointcloud_5k_grid.gpkg")
    if grid.crs is None:
        grid = grid.set_crs(LidarTileIndex.GRID_CRS)
    grid = grid.to_crs(LidarTileIndex.GRID_CRS)

    rng = np.random.default_rng(args.seed)
    picks = grid.iloc[rng.choice(len(grid), size=min(args.samples, len(grid)),
                                 replace=False)]
    to_ll = Transformer.from_crs(LidarTileIndex.GRID_CRS, "EPSG:4326", always_xy=True).transform
    to_ft = Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True).transform

    rows = []
    for i, (_, tile) in enumerate(picks.iterrows(), 1):
        c = tile.geometry.centroid
        lon, lat = to_ll(c.x, c.y)
        try:
            header = read_las_header(str(tile["aws_url"]))
        except Exception:                                     # noqa: BLE001
            try:
                header = read_las_header(str(tile["aws_url"]).replace(".copz.", ".copc."))
            except Exception:                                 # noqa: BLE001
                continue
        lidar_date = parse_date(header.created)
        if lidar_date is None:
            continue

        x, y = to_ft(lon, lat)
        best = None
        for coll in collections_covering(lon, lat):
            try:
                frames = gpd.read_file(
                    frames_index_url(coll),
                    bbox=(x - args.pad_ft, y - args.pad_ft,
                          x + args.pad_ft, y + args.pad_ft))
            except Exception:                                 # noqa: BLE001
                continue
            if frames.empty:
                continue
            for value in frames["FlightDate"]:
                d = parse_date(value)
                if d is None:
                    continue
                gap = abs((d - lidar_date).days)
                if best is None or gap < best["gap_days"]:
                    best = {"collection": coll, "oblique_date": d.isoformat(),
                            "gap_days": gap, "n_frames": int(len(frames))}
        row = {"tilename": str(tile["tilename"]), "lon": lon, "lat": lat,
               "lidar_date": lidar_date.isoformat()}
        row.update(best or {"collection": None, "oblique_date": None,
                            "gap_days": None, "n_frames": 0})
        rows.append(row)
        if i % 10 == 0 or best is None:
            print(f"[{i}/{len(picks)}] {row['tilename']}: lidar {row['lidar_date']}"
                  f" oblique {row['oblique_date']} gap {row['gap_days']}")

    covered = [r for r in rows if r["gap_days"] is not None]
    if not covered:
        print("no tile had both LiDAR and oblique coverage")
        return 1

    gaps = np.array([r["gap_days"] for r in covered])
    print("\n" + "=" * 66)
    print(f"{len(rows)} tiles sampled; {len(covered)} have both "
          f"({100 * len(covered) / len(rows):.0f}%)")
    print(f"gap in days: p10 {np.percentile(gaps, 10):.0f}  p50 {np.percentile(gaps, 50):.0f}"
          f"  p90 {np.percentile(gaps, 90):.0f}  max {gaps.max():.0f}")
    for thr in (30, 90, 180, 365):
        print(f"  within {thr:>3} days: {100 * (gaps <= thr).mean():5.1f}% of tiles")
    print("\nBest-aligned samples:")
    for r in sorted(covered, key=lambda r: r["gap_days"])[:8]:
        print(f"  {r['gap_days']:>4} d  {r['tilename']:<10} {r['lidar_date']} vs "
              f"{r['oblique_date']} ({r['collection']})  {r['lon']:.3f},{r['lat']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": rows}, indent=2), encoding="utf-8")
    print(f"\nresults: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
