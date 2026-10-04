#!/usr/bin/env python3
"""Measure surface complexity on the named case buildings, footprint-clipped.

The statewide survey (:mod:`survey_building_complexity`) reports a distribution;
this reports the three buildings the rest of the project actually reconstructs,
through the *same* :func:`features.assess_surface_complexity`, so their numbers
sit on the same axis as that distribution.

Earlier case numbers do not. They were measured over the fitted building volume
with a 6 m pad, which on the pilot is nearly three times the Overture footprint
area -- Tractor Supply's canopy and yard came in with it. Every number here is
clipped to the Overture polygon instead.

**The third case exists to separate height from complexity.** 400 West Market
reads as the most complex surface in the project, and it is also by far the
tallest, so the two explanations are confounded: a 165 m building is mostly
facade, seen at grazing incidence, and grazing-incidence returns are both sparse
and noisy. PNC Tower is 300 m away, 156 m tall (94% of 400 West Market), on the
same LiDAR tile and in the same oblique collection -- and is a plain rectangular
slab with a flat roof where 400 West Market carries a stepped pyramidal crown.
If the measure tracks structure, they separate. If it tracks height, they don't.

Simplicity is asserted from evidence that never touches the coherence measure:
Overture's footprint vertex count and rectangularity (2D), and the roof's
plane-fit RMS (3D). Selecting a "simple" building by its low structure fraction
would prove nothing.

Usage::

    python scripts/measure_case_complexity.py
    python scripts/measure_case_complexity.py --cases building_002 pnc_tower
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from kyfromabove_3d.coordinates import METERS_PER_US_SURVEY_FOOT
from kyfromabove_3d.features import (assess_surface_complexity,
                                      classify_surface_points,
                                      facade_extent,
                                      split_surface_zones)
from kyfromabove_3d.lidar import LidarTileIndex, open_copc

#: Pinned so a re-run measures the same polygons. Overture revises footprints
#: between releases, and the clip is the dominant error term here.
OVERTURE_RELEASE = "2026-08-19.0"
OVERTURE_BUILDINGS = (
    f"s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}"
    "/theme=buildings/type=building/*"
)

#: The GERS ids are the identity of record; lon/lat is only a bbox hint so the
#: parquet predicate can be pushed down instead of scanning the theme.
CASES = {
    "building_001": {
        "gers": "63e05b9a-1f9b-41d1-b58a-81b67f569fa0",
        "label": "Tractor Supply, Grayson",
        "hint": (-82.946350, 38.345474),
        "role": "wide, low, flat roof",
    },
    "pnc_tower": {
        "gers": "1614a997-3d6f-475f-9bc5-bb5a9dbc878d",
        "label": "PNC Tower, Louisville",
        "hint": (-85.757940, 38.256210),
        "role": "tall, plain slab",
    },
    "building_002": {
        "gers": "df2fccca-d6bd-4d4d-b4bc-673cfb5245aa",
        "label": "400 West Market, Louisville",
        "hint": (-85.757197, 38.254916),
        "role": "tall, stepped crown",
    },
}


def fetch_footprints(names, cache: Path) -> dict:
    """Overture footprint, height and class for each case, cached to JSON."""
    have = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    want = [n for n in names if n not in have]
    if want:
        import duckdb

        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;"
                    " SET s3_region='us-west-2';")
        for name in want:
            lon, lat = CASES[name]["hint"]
            d = 0.002
            row = con.execute(f"""
                SELECT names.primary, height, num_floors, class, roof_shape,
                       ST_AsText(geometry)
                FROM read_parquet('{OVERTURE_BUILDINGS}', filename=false,
                                  hive_partitioning=1)
                WHERE bbox.xmin BETWEEN {lon - d} AND {lon + d}
                  AND bbox.ymin BETWEEN {lat - d} AND {lat + d}
                  AND id = '{CASES[name]["gers"]}'
            """).fetchone()
            if row is None:
                raise SystemExit(f"{name}: GERS {CASES[name]['gers']} not in "
                                 f"release {OVERTURE_RELEASE}")
            have[name] = {"gers": CASES[name]["gers"],
                          "overture_name": row[0], "height_m": row[1],
                          "num_floors": row[2], "class": row[3],
                          "roof_shape": row[4], "wkt": row[5],
                          "release": OVERTURE_RELEASE}
            print(f"  fetched {name}: {row[0]} h={row[1]}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(have, indent=2), encoding="utf-8")
    return {n: have[n] for n in names}


def footprint_shape(poly) -> dict:
    """2D simplicity, from the polygon alone -- no LiDAR involved."""
    rect = poly.minimum_rotated_rectangle
    xs, ys = rect.exterior.xy
    sides = sorted(math.dist((xs[i], ys[i]), (xs[i + 1], ys[i + 1]))
                   * METERS_PER_US_SURVEY_FOOT for i in range(4))
    return {
        "area_m2": poly.area * METERS_PER_US_SURVEY_FOOT ** 2,
        "short_side_m": sides[0], "long_side_m": sides[2],
        "n_vertices": len(poly.exterior.coords) - 1,
        # 1.0 means the footprint *is* its own bounding rectangle.
        "rectangularity": poly.area / rect.area,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=list(CASES),
                    choices=list(CASES), metavar="NAME")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--buffers", type=float, nargs="*", default=[-1.0, 0.0, 1.0],
                    help="metres to buffer the footprint before clipping. A tall "
                         "building's facade returns land *on* the polygon edge, so "
                         "a footprint off by a metre keeps or drops the whole wall; "
                         "running several says how much that matters")
    ap.add_argument("--plot", type=Path,
                    default=Path("outputs/survey/case_complexity.png"),
                    help="where to write the figure; the numbers alone do not say "
                         "whether a building's roughness is on its roof or its walls")
    ap.add_argument("--footprint-cache", type=Path,
                    default=Path("outputs/survey/case_footprints.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/survey/case_complexity.json"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    import laspy
    from pyproj import Transformer
    from shapely import contains_xy
    from shapely.ops import transform as shp_transform
    from shapely.wkt import loads as wkt_loads

    print(f"footprints from Overture {OVERTURE_RELEASE}")
    meta = fetch_footprints(args.cases, args.footprint_cache)
    to_ft = Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True).transform
    index = LidarTileIndex(args.raw_dir)
    # US survey feet: the footprint buffer is specified in metres.
    per_ft = 1.0 / METERS_PER_US_SURVEY_FOOT

    results, panels, started = [], [], time.time()
    for name in args.cases:
        info, spec = meta[name], CASES[name]
        poly = shp_transform(to_ft, wkt_loads(info["wkt"]))
        shape = footprint_shape(poly)
        print(f"\n=== {name}: {info['overture_name']} ({spec['role']})")
        print(f"    {shape['short_side_m']:.0f} x {shape['long_side_m']:.0f} m, "
              f"{shape['area_m2']:.0f} m2, {shape['n_vertices']} vertices, "
              f"rectangularity {shape['rectangularity']:.3f}, "
              f"Overture height {info['height_m']} m")

        # The bbox is how a COPC reader is queried; buffer it so a +N m clip
        # still has points to keep.
        pad = (max(args.buffers) + 2.0) * per_ft
        minx, miny, maxx, maxy = poly.bounds
        tiles = index.tiles_for_bounds((minx, miny, maxx, maxy), phase="phase2")
        if not tiles:
            print("    [skip] no phase2 tile")
            continue
        tile = tiles[0]
        try:
            reader = open_copc(tile.url)
        except Exception as exc:                                # noqa: BLE001
            print(f"    [skip] {tile.tilename} unreadable ({str(exc)[:70]})")
            continue
        header = reader.header
        pts = reader.query(laspy.copc.Bounds(
            mins=np.array([minx - pad, miny - pad, header.z_min]),
            maxs=np.array([maxx + pad, maxy + pad, header.z_max])))
        cls = np.asarray(pts.classification)
        raw = np.stack([pts.x, pts.y, pts.z], axis=1)[cls != 7]
        print(f"    tile {tile.tilename}: {len(raw):,} points in the padded bbox")

        # "bbox" is the pre-clip measurement, kept so the change the polygon
        # makes is visible rather than asserted.
        variants = [("bbox", None)] + [(f"poly{b:+.0f}m", b) for b in args.buffers]
        for label, buf in variants:
            xyz = raw
            if buf is not None:
                clip = poly if buf == 0 else poly.buffer(buf * per_ft)
                xyz = raw[contains_xy(clip, raw[:, 0], raw[:, 1])]
            if len(xyz) < 200:
                print(f"    {label:<10} too few points ({len(xyz)})")
                continue
            local = np.column_stack([
                (xyz[:, 0] - xyz[:, 0].mean()) * METERS_PER_US_SURVEY_FOOT,
                (xyz[:, 1] - xyz[:, 1].mean()) * METERS_PER_US_SURVEY_FOOT,
                (xyz[:, 2] - xyz[:, 2].mean()) * METERS_PER_US_SURVEY_FOOT])
            classes = classify_surface_points(local)
            got = assess_surface_complexity(local, classes=classes)
            if got is None:
                print(f"    {label:<10} not assessable")
                continue
            zones = split_surface_zones(local)
            rms = None if zones is None else zones.plate_rms_m
            row = got.to_dict()
            row.update({
                "case": name, "clip": label, "buffer_m": buf,
                "gers": spec["gers"], "label": info["overture_name"],
                "overture_height_m": info["height_m"],
                "tilename": tile.tilename,
                "roof_plate_rms_m": rms,
                "roof_plateau_frac": None if zones is None else zones.plateau_frac,
                "z_span_m": float(np.ptp(local[:, 2])), **shape,
            })
            print(f"    {label:<10} n={got.n_points:>7,}  rough {got.rough_frac:6.1%}"
                  f"  structure {got.structure_frac:6.1%}"
                  f"  noise {got.noise_frac:6.1%}"
                  f"  plate RMS {'n/a' if rms is None else f'{rms:.3f} m'}"
                  f"  plateau {zones.plateau_frac:5.1%}"
                  f"  ({time.time() - started:.0f}s)")

            if buf != 0.0 or zones is None:
                results.append(row)
                continue
            # On the polygon clip, rate walls and roof separately. A
            # whole-building fraction hides which of the two produced it, and on
            # a tall building they behave differently enough that the summary
            # number means little on its own.
            structure = classes.structure
            row.update(zones.zone_rates(structure))
            row["perimeter_m"] = poly.length * METERS_PER_US_SURVEY_FOOT
            row["height_m"] = zones.plate_z_m - float(local[:, 2].min())
            row.update(facade_extent(
                perimeter_m=row["perimeter_m"],
                height_m=row["height_m"],
                footprint_area_m2=shape["area_m2"],
                n_wall_returns=int(zones.wall.sum()),
            ).to_dict())
            results.append(row)
            panels.append({"label": f"{info['overture_name']}\n{spec['role']}",
                           "xyz": np.column_stack([local[:, 0], local[:, 1],
                                                   local[:, 2] - local[:, 2].min()]),
                           "structure": structure,
                           "plate_z_m": zones.plate_z_m - float(local[:, 2].min())})
            print(f"    {'':<10} of which:  "
                  + "   ".join(
                      f"{zone} {row[f'{zone}_frac']:.0%} of points, "
                      + ("-" if row[f'{zone}_rate'] is None
                         else f"{row[f'{zone}_rate']:.0%} structure")
                      for zone in ("wall", "roof")))

    if not results:
        print("nothing measured")
        return 1
    # Merge, never replace. A --cases run naming a subset used to overwrite the
    # whole file, and a single-case smoke test silently cut the committed
    # three-case results down to one (commit 7065049). Keyed on (case, clip) so
    # a re-run of one building updates its own rows and leaves the rest.
    kept = []
    if args.out.exists():
        try:
            prior = json.loads(args.out.read_text(encoding="utf-8")).get("results", [])
        except Exception:                                   # noqa: BLE001
            prior = []
        fresh = {(r["case"], r["clip"]) for r in results}
        kept = [r for r in prior if (r.get("case"), r.get("clip")) not in fresh]
        if kept:
            print(f"\nkeeping {len(kept)} row(s) for cases this run did not cover: "
                  f"{', '.join(sorted({r['case'] for r in kept}))}")
    merged = sorted(kept + results, key=lambda r: (list(CASES).index(r["case"])
                                                   if r["case"] in CASES else 99,
                                                   r["clip"]))

    print("\n" + "=" * 78)
    print(f"{'case':<14}{'clip':<10}{'height':>7}{'verts':>6}{'rect':>6}"
          f"{'rough':>8}{'structure':>11}{'noise':>8}{'plateRMS':>9}"
          f"{'plateau':>9}{'fac/roof':>10}{'reached':>9}")
    for r in merged:
        h, rms = r["overture_height_m"], r["roof_plate_rms_m"]
        ratio, per_m2 = r.get("facade_to_roof"), r.get("facade_returns_per_m2")
        height_s = "-" if h is None else f"{h:.0f}"
        rms_s = "-" if rms is None else f"{rms:.2f}"
        ratio_s = "-" if ratio is None else f"{ratio:.1f}"
        # Marked when the LiDAR did not reach the wall: a facade that exists on
        # paper and not in the data is not a candidate.
        per_s = "-" if per_m2 is None else (
            f"{per_m2:.3f}" + ("" if r.get("facade_captured") else "!"))
        print(f"{r['case']:<14}{r['clip']:<10}{height_s:>7}{r['n_vertices']:>6}"
              f"{r['rectangularity']:>6.2f}{r['rough_frac']:>8.1%}"
              f"{r['structure_frac']:>11.1%}"
              f"{r['noise_frac']:>8.1%}{rms_s:>9}"
              f"{r['roof_plateau_frac']:>9.1%}{ratio_s:>10}{per_s:>9}")
    print("   ! = the LiDAR did not meaningfully reach that wall")

    # A figure covering only some of the cases is worse than no figure: it looks
    # complete and is not. The comparison is the whole point of the plot.
    if args.plot and len(panels) == len(CASES):
        from kyfromabove_3d.plots import plot_case_complexity
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot_case_complexity(panels, save_to=args.plot)
        print(f"\nfigure:  {args.plot}")
    elif args.plot:
        print(f"\nfigure:  skipped -- {len(panels)}/{len(CASES)} cases in this run; "
              f"re-run without --cases to redraw it")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"overture_release": OVERTURE_RELEASE,
                                    "results": merged}, indent=2), encoding="utf-8")
    print(f"\nresults: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
