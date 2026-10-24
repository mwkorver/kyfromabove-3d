#!/usr/bin/env python3
"""Compare the KyFromAbove LiDAR phases: format, coverage, density, classification.

Which phase to read is not a detail. It decides whether a building can be
fetched with an HTTP range request or only by pulling a whole 50 MB tile, and
that ratio is the entire scaling argument for this project -- one building's AOI
is well under 1% of a tile.

Everything printed here is measured, not quoted from documentation. Tile
inventory and coverage come from the published 5k grid GeoPackages; density and
flight date come from LAS public headers fetched one range request at a time
(``--headers``); classification comes from COPC queries over a small window
(``--classes``), which is the slow part and is off by default.

Usage::

    python scripts/compare_lidar_phases.py                 # inventory + overlap
    python scripts/compare_lidar_phases.py --headers       # + density and dates
    python scripts/compare_lidar_phases.py --headers --classes
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import numpy as np

from kyfromabove_3d.lidar import LidarTileIndex, read_las_header

if TYPE_CHECKING:
    import geopandas as gpd

# Built-up ground inside the Phase 3 re-flight footprint. Sampling a tile centre
# blindly is how you end up comparing two phases over open water, which says
# nothing about how either reconstructs a building.
DEFAULT_PLACES = {
    "Ashland": (-82.638, 38.478),
    "Pikeville": (-82.519, 37.479),
}

ASPRS = {
    0: "never classified", 1: "unclassified", 2: "ground", 3: "low veg",
    4: "med veg", 5: "high veg", 6: "BUILDING", 7: "noise", 9: "water",
    10: "rail", 11: "road", 17: "bridge deck", 18: "high noise",
}


def load_grids(raw_dir: Path, phases) -> Dict[str, "gpd.GeoDataFrame"]:
    import geopandas as gpd

    grids = {}
    for phase in phases:
        path = raw_dir / f"kyfromabove_{phase}_pointcloud_5k_grid.gpkg"
        if not path.exists():
            print(f"  [skip] {path.name} not present")
            continue
        grid = gpd.read_file(path)
        # The Phase 2 grid ships with no CRS at all. Comparing an EPSG:3089 AOI
        # against it directly returns zero tiles, silently.
        declared = str(grid.crs) if grid.crs else "NONE (assumed EPSG:3857)"
        if grid.crs is None:
            grid = grid.set_crs(LidarTileIndex.GRID_CRS)
        grids[phase] = grid.to_crs(LidarTileIndex.GRID_CRS)
        grids[phase].attrs["declared_crs"] = declared
    return grids


def report_inventory(grids) -> List[dict]:
    print("=" * 78)
    print("Inventory")
    print("=" * 78)
    print(f"{'phase':<9} {'tiles':>8}  {'declared CRS':<26} formats")
    rows = []
    for phase, grid in grids.items():
        exts = Counter()
        for key in grid["key"]:
            name = str(key).rsplit("/", 1)[-1]
            exts[".".join(name.split(".")[1:]) or "?"] += 1
        copc = sum(v for k, v in exts.items() if "copc" in k or "copz" in k)
        summary = ", ".join(f"{k} x{v:,}" for k, v in exts.most_common())
        print(f"{phase:<9} {len(grid):>8,}  {grid.attrs['declared_crs']:<26} {summary}")
        rows.append({"phase": phase, "tiles": len(grid),
                     "declared_crs": grid.attrs["declared_crs"],
                     "formats": dict(exts), "copc_tiles": copc,
                     "all_copc": copc == len(grid)})
    print("\nCOPC supports HTTP range reads; plain .laz must be fetched whole.")
    return rows


def report_overlap(grids) -> dict:
    print("\n" + "=" * 78)
    print("Coverage overlap (by tilename)")
    print("=" * 78)
    names = {p: set(g["tilename"]) for p, g in grids.items()}
    out = {}
    for a in names:
        for b in names:
            if a >= b:
                continue
            only_a, only_b = names[a] - names[b], names[b] - names[a]
            print(f"  {a} only: {len(only_a):>6,}    {b} only: {len(only_b):>6,}    "
                  f"shared: {len(names[a] & names[b]):>6,}")
            out[f"{a}_vs_{b}"] = {"only_" + a: sorted(only_a)[:20],
                                  "n_only_" + a: len(only_a),
                                  "only_" + b: sorted(only_b)[:20],
                                  "n_only_" + b: len(only_b),
                                  "shared": len(names[a] & names[b])}
    # Where does the fallback order actually change the answer?
    if {"phase1", "phase2", "phase3"} <= set(names):
        gap = names["phase1"] - names["phase2"]
        contested = gap & names["phase3"]
        print(f"\n  Phase 2 absent but Phase 1 present: {len(gap)} tile(s)")
        print(f"  ...of which Phase 3 (COPC) also covers: {len(contested)} -> "
              f"{sorted(contested)}")
        print("  Those are the only tiles where the phase preference order matters.")
        out["order_sensitive_tiles"] = sorted(contested)
    return out


def report_headers(grids, places) -> List[dict]:
    from shapely.geometry import Point
    from pyproj import Transformer

    to_grid = Transformer.from_crs("EPSG:4326", LidarTileIndex.GRID_CRS,
                                   always_xy=True).transform
    print("\n" + "=" * 78)
    print("LAS headers (one HTTP range request each -- no point data read)")
    print("=" * 78)
    rows = []
    for name, (lon, lat) in places.items():
        pt = Point(*to_grid(lon, lat))
        print(f"\n{name} ({lon}, {lat})")
        print(f"  {'phase':<8} {'tile':<11} {'created':<12} {'points':>12} "
              f"{'pts/m2':>7} {'z span':>8}")
        for phase, grid in grids.items():
            hit = grid[grid.contains(pt)]
            if hit.empty:
                print(f"  {phase:<8} -- no coverage --")
                continue
            row = hit.iloc[0]
            try:
                h = read_las_header(str(row["aws_url"]))
            except Exception as exc:                      # noqa: BLE001
                print(f"  {phase:<8} {row['tilename']:<11} header read failed: {exc}")
                continue
            print(f"  {phase:<8} {row['tilename']:<11} {h.created:<12} "
                  f"{h.n_points:>12,} {h.density_per_m2():>7.2f} {h.z_span:>7.0f}'")
            rows.append({"place": name, "phase": phase,
                         "tilename": str(row["tilename"]), **h.to_dict()})
    print("\n  z span is a blunder detector: real terrain in a 5000 ft Kentucky tile")
    print("  spans a few hundred feet, not thousands.")
    return rows


def report_classes(grids, places, window_ft: float = 400.0) -> List[dict]:
    import laspy
    from shapely.geometry import Point
    from pyproj import Transformer

    to_grid = Transformer.from_crs("EPSG:4326", LidarTileIndex.GRID_CRS,
                                   always_xy=True).transform
    print("\n" + "=" * 78)
    print(f"Classification, COPC query over {2 * window_ft:.0f} x {2 * window_ft:.0f} ft")
    print("=" * 78)
    rows = []
    for name, (lon, lat) in places.items():
        pt = Point(*to_grid(lon, lat))
        print(f"\n{name}")
        for phase, grid in grids.items():
            hit = grid[grid.contains(pt)]
            if hit.empty:
                continue
            row = hit.iloc[0]
            key = str(row["key"])
            if ".copc." not in key and ".copz." not in key:
                print(f"  {phase:<8} {row['tilename']:<11} plain LAZ -- skipped "
                      f"(no range queries; would need the whole tile)")
                continue
            try:
                reader = laspy.CopcReader.open(str(row["aws_url"]))
                hdr = reader.header
                cx = (hdr.x_min + hdr.x_max) / 2
                cy = (hdr.y_min + hdr.y_max) / 2
                pts = reader.query(laspy.copc.Bounds(
                    mins=np.array([cx - window_ft, cy - window_ft, hdr.z_min]),
                    maxs=np.array([cx + window_ft, cy + window_ft, hdr.z_max])))
            except Exception as exc:                      # noqa: BLE001
                print(f"  {phase:<8} query failed: {exc}")
                continue
            cls = np.asarray(pts.classification)
            n = len(cls)
            if not n:
                print(f"  {phase:<8} {row['tilename']:<11} no points in window")
                continue
            area = (2 * window_ft * 0.3048006096) ** 2
            counts = Counter(cls.tolist())
            top = ", ".join(f"{c}={100 * k / n:.0f}%"
                            for c, k in counts.most_common(4))
            print(f"  {phase:<8} {row['tilename']:<11} {n:>8,} pts "
                  f"{n / area:5.2f} pts/m2   {top}")
            rows.append({"place": name, "phase": phase,
                         "tilename": str(row["tilename"]), "n_points": n,
                         "density_per_m2": n / area,
                         "classes": {int(c): int(k) for c, k in counts.items()}})
    has_building = any(6 in r["classes"] for r in rows)
    print(f"\n  class 6 (BUILDING) present in any sample: {has_building}")
    if not has_building:
        print("  KyFromAbove does not classify buildings. Footprints cannot come")
        print("  from the point cloud's own classification -- they need an external")
        print("  source (Overture) or segmentation on the imagery.")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--phases", nargs="*", default=list(LidarTileIndex.PHASE_PREFERENCE))
    ap.add_argument("--headers", action="store_true",
                    help="fetch LAS headers for sample locations")
    ap.add_argument("--classes", action="store_true",
                    help="COPC-query sample windows and tabulate classification")
    ap.add_argument("--out", type=Path, default=None, help="write the report as JSON")
    args = ap.parse_args()

    grids = load_grids(args.raw_dir, args.phases)
    if not grids:
        print(f"[FAIL] no phase grids under {args.raw_dir}")
        return 1

    report = {"inventory": report_inventory(grids), "overlap": report_overlap(grids)}
    if args.headers:
        report["headers"] = report_headers(grids, DEFAULT_PLACES)
    if args.classes:
        report["classes"] = report_classes(grids, DEFAULT_PLACES)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
