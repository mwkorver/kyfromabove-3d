#!/usr/bin/env python3
"""Measure how much of each building's surface the LiDAR under-samples.

Densification only pays where a surface is genuinely complex. A flat roof at
0.60 m point spacing is already fully described, and adding points to it
interpolates a plane. :func:`features.surface_coherence` separates real
structure from range noise, so it can answer, per building, whether there is
anything there worth resolving.

This runs that measurement over Overture's large-footprint buildings, as built
by :mod:`fetch_overture_buildings`. Work is **batched by LiDAR tile**, because a
COPC range query per building would spend all its time opening readers -- large
buildings sit a couple per 5k tile, so the tile read is the unit of cost, not the
building.

.. note::
   Area is recomputed here from the footprint polygon rather than taken from the
   building list. The original list's ``area_m2`` column was the axis-aligned
   *bounding box*, so "over 5,000 m^2" described a population whose median true
   footprint was 2,180 m^2, and every document repeated the column's name rather
   than its meaning.

Usage::

    python scripts/survey_building_complexity.py --buildings ky_large_tiled.csv --tiles 10
    python scripts/survey_building_complexity.py --buildings ... --tiles 40 --sample random
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from kyfromabove_3d.coordinates import METERS_PER_US_SURVEY_FOOT
from kyfromabove_3d.features import (FACADE_CAPTURE_PER_M2,
                                      assess_surface_complexity,
                                      classify_surface_points,
                                      facade_extent,
                                      split_surface_zones)
from kyfromabove_3d.lidar import LidarTileIndex, open_copc


def load_buildings(path: Path):
    """Read the building list, keeping the footprint polygon where present.

    A COPC reader can only be queried by bounding box, so the bbox is still how
    points are fetched -- but a bbox around a building takes in whatever sits
    beside it. On a rectangular warehouse that is a little apron; on an L-shaped
    or angled building it can be mostly car park. The polygon is what decides
    which of the fetched points actually belong to the building.
    """
    csv.field_size_limit(1 << 30)      # WKT for a large footprint is long
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({"id": r["id"], "tilename": r["tilename"],
                         "area_m2": float(r["area_m2"]),
                         "wkt": (r.get("wkt") or "").strip(),
                         "bbox": (float(r["xmin"]), float(r["ymin"]),
                                  float(r["xmax"]), float(r["ymax"]))})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buildings", type=Path, required=True,
                    help="CSV with id, tilename, area_m2, xmin/ymin/xmax/ymax (EPSG:4326) "
                         "and optionally wkt, the Overture footprint polygon")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--tiles", type=int, default=10, help="how many tiles to process")
    ap.add_argument("--sample", choices=["dense", "random"], default="dense",
                    help="'dense' takes the tiles holding most buildings (fastest per "
                         "building, urban-biased); 'random' is representative but "
                         "pays a tile read for ~2 buildings")
    ap.add_argument("--seed", type=int, default=0,
                    help="which 45 tiles the random sample draws. Recorded in the "
                         "output; changing it changes the answer, so quote it "
                         "alongside any number taken from a random run")
    ap.add_argument("--use-bbox", action="store_true",
                    help="measure the whole bounding box rather than clipping to "
                         "the footprint polygon, for comparison with earlier runs")
    ap.add_argument("--out", type=Path, default=Path("outputs/survey/complexity.json"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    import laspy
    from pyproj import Transformer

    rows = load_buildings(args.buildings)
    by_tile = defaultdict(list)
    for r in rows:
        by_tile[r["tilename"]].append(r)
    order = sorted(by_tile, key=lambda t: -len(by_tile[t]))
    if args.sample == "random":
        rng = np.random.default_rng(args.seed)
        order = list(rng.permutation(sorted(by_tile)))
    chosen = order[: args.tiles]
    total = sum(len(by_tile[t]) for t in chosen)
    print(f"{len(rows):,} buildings over {len(by_tile):,} tiles; "
          f"processing {len(chosen)} tiles ({args.sample}) = {total:,} buildings")

    from shapely import contains_xy
    from shapely.ops import transform as shp_transform
    from shapely.wkt import loads as wkt_loads

    index = LidarTileIndex(args.raw_dir)
    # Resolve by tilename, not by the first building's position. A building near
    # a tile edge resolves to its neighbour, and then every other building in
    # the group is queried against the wrong tile and silently returns nothing.
    grid = index.grids.get("phase2")
    if grid is None:
        print("[FAIL] no phase2 grid")
        return 1
    url_for = dict(zip(grid["tilename"].astype(str), grid["aws_url"].astype(str)))
    to_ft = Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True).transform
    results, started = [], time.time()

    for ti, tile in enumerate(chosen, 1):
        blds = by_tile[tile]
        url = url_for.get(tile)
        if not url or (".copc." not in url and ".copz." not in url):
            print(f"[{ti}/{len(chosen)}] {tile}: no COPC tile; skipped")
            continue
        try:
            reader = open_copc(url)
        except Exception as exc:                        # noqa: BLE001
            # One unreadable tile must not end a survey of thousands.
            print(f"[{ti}/{len(chosen)}] {tile}: unreadable ({str(exc)[:70]}); skipped")
            continue
        header = reader.header
        done = 0
        for b in blds:
            bx0, by0 = to_ft(b["bbox"][0], b["bbox"][1])
            bx1, by1 = to_ft(b["bbox"][2], b["bbox"][3])
            try:
                pts = reader.query(laspy.copc.Bounds(
                    mins=np.array([min(bx0, bx1), min(by0, by1), header.z_min]),
                    maxs=np.array([max(bx0, bx1), max(by0, by1), header.z_max])))
            except Exception:                                  # noqa: BLE001
                continue
            if len(pts.x) < 200:
                continue
            cls = np.asarray(pts.classification)
            keep = cls != 7
            xyz = np.stack([pts.x, pts.y, pts.z], axis=1)[keep]

            # Clip to the footprint. Without this the measurement includes the
            # apron the bbox caught, which is flat and drags the structure
            # fraction down on compact buildings and up wherever the box
            # clipped a neighbour.
            poly = None
            if b["wkt"]:
                try:
                    poly = shp_transform(to_ft, wkt_loads(b["wkt"]))
                except Exception:                           # noqa: BLE001
                    poly = None
            if poly is not None and not args.use_bbox:
                inside = contains_xy(poly, xyz[:, 0], xyz[:, 1])
                if inside.sum() < 200:
                    continue
                xyz = xyz[inside]
            # Metres, so residuals and offsets are comparable across buildings.
            xyz = np.column_stack([
                (xyz[:, 0] - xyz[:, 0].mean()) * METERS_PER_US_SURVEY_FOOT,
                (xyz[:, 1] - xyz[:, 1].mean()) * METERS_PER_US_SURVEY_FOOT,
                (xyz[:, 2] - xyz[:, 2].mean()) * METERS_PER_US_SURVEY_FOOT])
            # Classify once: the plane fit and the coherence pass are the
            # expensive part of the run, and both the summary and the zone
            # split are derived from the same masks.
            classes = classify_surface_points(xyz)
            got = assess_surface_complexity(xyz, classes=classes)
            if got is None:
                continue
            row = got.to_dict()
            row.update({"id": b["id"], "tilename": tile,
                        "listed_area_m2": b["area_m2"],
                        "clipped": bool(b["wkt"]) and not args.use_bbox})
            if poly is not None:
                # From the geometry, not from the list's column.
                row["area_m2"] = poly.area * METERS_PER_US_SURVEY_FOOT ** 2
            # Whole-building fractions mix roof with wall, and wall returns
            # score about a third "structure" on a blank wall. Record the split
            # so the mix can be undone afterwards rather than re-measured.
            zones = split_surface_zones(xyz)
            if zones is not None:
                height_m = zones.plate_z_m - float(xyz[:, 2].min())
                row["plate_z_m"] = height_m
                row["plateau_frac"] = zones.plateau_frac
                row.update(zones.zone_rates(classes.structure))
                # A roof is scored on roughness; a facade is already flat, and
                # what it lacks is colour. So the facade term measures how much
                # wall there is and how much of it the LiDAR reached -- a
                # different question, deliberately not folded into one score.
                if poly is not None:
                    row["perimeter_m"] = poly.length * METERS_PER_US_SURVEY_FOOT
                    row["height_m"] = height_m
                    row.update(facade_extent(
                        perimeter_m=row["perimeter_m"],
                        height_m=height_m,
                        footprint_area_m2=row["area_m2"],
                        n_wall_returns=int(zones.wall.sum()),
                    ).to_dict())
            results.append(row)
            done += 1
        print(f"[{ti}/{len(chosen)}] {tile}: {done}/{len(blds)} buildings "
              f"({time.time() - started:.0f}s elapsed, {len(results)} total)")

    if not results:
        print("no buildings measured")
        return 1

    sf = np.array([r["structure_frac"] for r in results])
    nf = np.array([r["noise_frac"] for r in results])
    print("\n" + "=" * 66)
    print(f"{len(results):,} buildings measured")
    print(f"{'structure fraction':<24}" + "".join(f"p{q:<5}" for q in (10, 25, 50, 75, 90)))
    print(f"{'':<24}" + "".join(f"{np.percentile(sf, q):<6.1%}"[:6] for q in (10, 25, 50, 75, 90)))
    print(f"median noise fraction   : {np.median(nf):.1%}")
    for thr in (0.05, 0.10, 0.20):
        print(f"  buildings with >{thr:.0%} real structure: "
              f"{100 * (sf > thr).mean():5.1f}%")

    # The same thing again on roof returns only. Where the two disagree, the
    # difference is wall, and wall does not discriminate.
    rf = np.array([r.get("roof_rate", np.nan) for r in results], dtype=float)
    has_roof = np.isfinite(rf)
    wf = np.array([r.get("wall_rate", np.nan) for r in results], dtype=float)
    has_wall = np.isfinite(wf)
    print(f"\nroof returns only ({has_roof.sum():,} buildings with a "
          f"measurable roof)")
    print(f"{'structure fraction':<24}" + "".join(f"p{q:<5}" for q in (10, 25, 50, 75, 90)))
    print(f"{'':<24}" + "".join(f"{np.percentile(rf[has_roof], q):<6.1%}"[:6]
                                for q in (10, 25, 50, 75, 90)))
    for thr in (0.05, 0.10, 0.20):
        print(f"  buildings with >{thr:.0%} real structure: "
              f"{100 * (rf[has_roof] > thr).mean():5.1f}%")
    if has_wall.sum():
        print(f"\nwall returns ({has_wall.sum():,} buildings have enough wall to "
              f"rate): median {np.median(wf[has_wall]):.1%}")
        both = has_roof & has_wall
        if both.sum():
            print(f"  of those, roof median {np.median(rf[both]):.1%} "
                  f"vs wall median {np.median(wf[both]):.1%}")

    # The facade term. Reported beside the roof score rather than merged into
    # it, because the two answer different questions and pick out different
    # buildings -- which is the whole reason for having both.
    ratio = np.array([r.get("facade_to_roof", np.nan) for r in results], dtype=float)
    caught = np.array([bool(r.get("facade_captured")) for r in results])
    has_f = np.isfinite(ratio)
    if has_f.sum():
        print(f"\nfacade extent ({has_f.sum():,} buildings with a footprint polygon)")
        print(f"{'facade area / roof area':<24}"
              + "".join(f"p{q:<5}" for q in (10, 25, 50, 75, 90)))
        print(f"{'':<24}" + "".join(f"{np.percentile(ratio[has_f], q):<6.1f}"[:6]
                                    for q in (10, 25, 50, 75, 90)))
        print(f"  the LiDAR reached the wall on: {100 * caught[has_f].mean():5.1f}% "
              f"(>= {FACADE_CAPTURE_PER_M2} returns/m2)")

        # The point of two scores is that they disagree. Show by how much.
        worth_roof = has_roof & (rf > 0.10)
        worth_facade = has_f & caught & (ratio > 2.0)
        print(f"\nwhat each score selects, of {len(results):,} buildings")
        print(f"  roof worth densifying (>10% structure) : {worth_roof.sum():>4} "
              f"({100 * worth_roof.mean():4.1f}%)")
        print(f"  facade worth imaging (>2x roof, reached): {worth_facade.sum():>4} "
              f"({100 * worth_facade.mean():4.1f}%)")
        print(f"  both                                    : "
              f"{(worth_roof & worth_facade).sum():>4}")
        print(f"  either                                  : "
              f"{(worth_roof | worth_facade).sum():>4} "
              f"({100 * (worth_roof | worth_facade).mean():4.1f}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Record what produced these numbers. The seed is not decoration: a survey
    # of 120 tiles out of the 1,805 holding a building is a *sample*, and a
    # different seed is a different set of buildings with a different answer:
    # on the earlier population seed 0 and seed 1 gave medians a full point
    # apart. Without this block a re-run silently measures somewhere else.
    args.out.write_text(json.dumps({
        "params": {
            "buildings": str(args.buildings),
            "tiles_requested": args.tiles,
            "tiles_measured": len({r["tilename"] for r in results}),
            "sample": args.sample,
            "seed": args.seed,
            "clipped_to_footprint": not args.use_bbox,
        },
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\nresults: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
