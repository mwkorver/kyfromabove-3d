#!/usr/bin/env python3
"""Did the LiDAR actually reach these walls, and is the height they were picked on right?

The complexity survey asks whether a **roof** is under-sampled, and its engine is
a roughness detector. That engine is useless on a facade: measured on two
Louisville towers 300 m apart, a plain flat slab and an articulated
crown-and-setback wall score 32% and 36%, indistinguishable. Roughness cannot
read a wall, because an aerial scanner sees a vertical surface nearly edge-on and
roughly a third of the returns read as structure whether or not there is any.

So this pass does not measure roughness. It measures two things a facade
candidate is actually selected on, and both come from the LiDAR:

**Was the wall reached** -- per wall face, not per building. A wall faces one
way, and an aircraft flying one direction reaches some faces and not others; a
whole-building return density averages the face it flew along with the face it
never saw and calls the building captured.

**Is the height right** -- the candidate list is built on Overture's ``height``,
which is modelled on many buildings, and both the facade area and the
wall-to-roof ratio that selected it are proportional to that height. If the
height is wrong the selection is wrong.

Not measured here: whether the **obliques** can see a face. That is the real
question for facade value and it needs the EO index and per-face view geometry,
which is a separate pass. The bearings this one reports are what it will join on.

An uncaptured face is not always a missed one. Downtown buildings abut, and a
**party wall has no facade to reach** -- nothing sees it, so nothing returns from
it, and it scores exactly like a wall the aircraft flew past. Separating the two
needs the neighbouring footprints, which the candidate list does not carry. Read
"not reached" as "not available from the air", which is the operative question
anyway.

**Every row is keyed on the GERS id**, Overture's stable building identifier, so
these results join to the candidate list, the complexity survey and any later
imagery pass without a spatial match.

Usage::

    python scripts/survey_facade_capture.py --buildings data/derived/ky_facade_tiled_wkt.csv --tiles 10
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
from kyfromabove_3d.features import (FACADE_CAPTURE_PER_M2, facade_faces,
                                      split_surface_zones)
from kyfromabove_3d.lidar import LidarTileIndex, open_copc

#: Matches scripts/fetch_overture_buildings.py. GERS ids are stable across
#: releases but the geometry behind them is not, and every area and bearing here
#: comes from that geometry.
OVERTURE_RELEASE = "2026-10-21.0"


def load_candidates(path: Path):
    """Read the facade candidate list, keyed on GERS id."""
    csv.field_size_limit(1 << 30)
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not (r.get("wkt") or "").strip():
                continue
            rows.append({
                "gers_id": r["id"],
                "tilename": r["tilename"],
                "wkt": r["wkt"].strip(),
                # Present on a --select facade list; absent on an area list,
                # which this pass can still run over.
                "est_height_m": float(r["est_height_m"]) if r.get("est_height_m") else None,
                "height_source": r.get("height_source") or "",
                "est_facade_m2": float(r["est_facade_m2"]) if r.get("est_facade_m2") else None,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buildings", type=Path,
                    default=Path("data/derived/ky_facade_tiled_wkt.csv"))
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--tiles", type=int, default=10)
    ap.add_argument("--sample", choices=["dense", "random"], default="dense")
    ap.add_argument("--seed", type=int, default=0,
                    help="which tiles a random sample draws; recorded in the output")
    ap.add_argument("--facade-pad-m", type=float, default=2.0,
                    help="how far outside the drawn footprint to keep returns. "
                         "Overture footprints are ML-derived and sit a metre or "
                         "two off, and a wall return lands ON the boundary -- "
                         "clipping strictly to the polygon drops the facade this "
                         "pass exists to measure")
    ap.add_argument("--min-face-m", type=float, default=3.0,
                    help="shortest wall worth rating separately")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/survey/facade_capture.json"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    import laspy
    from pyproj import Transformer
    from shapely import contains_xy
    from shapely.ops import transform as shp_transform
    from shapely.wkt import loads as wkt_loads

    rows = load_candidates(args.buildings)
    by_tile = defaultdict(list)
    for r in rows:
        by_tile[r["tilename"]].append(r)
    order = sorted(by_tile, key=lambda t: -len(by_tile[t]))
    if args.sample == "random":
        order = list(np.random.default_rng(args.seed).permutation(sorted(by_tile)))
    chosen = order[: args.tiles]
    print(f"{len(rows):,} facade candidates over {len(by_tile):,} tiles; "
          f"processing {len(chosen)} ({args.sample}) = "
          f"{sum(len(by_tile[t]) for t in chosen):,} buildings")

    index = LidarTileIndex(args.raw_dir)
    grid = index.grids.get("phase2")
    if grid is None:
        print("[FAIL] no phase2 grid")
        return 1
    url_for = dict(zip(grid["tilename"].astype(str), grid["aws_url"].astype(str)))
    to_ft = Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True).transform
    per_ft = 1.0 / METERS_PER_US_SURVEY_FOOT

    buildings, faces_out, started = [], [], time.time()
    for ti, tile in enumerate(chosen, 1):
        url = url_for.get(tile)
        if not url or (".copc." not in url and ".copz." not in url):
            print(f"[{ti}/{len(chosen)}] {tile}: no COPC tile; skipped")
            continue
        try:
            reader = open_copc(url)
        except Exception as exc:                            # noqa: BLE001
            print(f"[{ti}/{len(chosen)}] {tile}: unreadable ({str(exc)[:60]}); skipped")
            continue
        header, done = reader.header, 0
        for b in by_tile[tile]:
            try:
                poly = shp_transform(to_ft, wkt_loads(b["wkt"]))
            except Exception:                               # noqa: BLE001
                continue
            grown = poly.buffer(args.facade_pad_m * per_ft)
            minx, miny, maxx, maxy = grown.bounds
            try:
                pts = reader.query(laspy.copc.Bounds(
                    mins=np.array([minx, miny, header.z_min]),
                    maxs=np.array([maxx, maxy, header.z_max])))
            except Exception:                               # noqa: BLE001
                continue
            if len(pts.x) < 200:
                continue
            xyz = np.stack([pts.x, pts.y, pts.z], axis=1)[
                np.asarray(pts.classification) != 7]
            inside = contains_xy(grown, xyz[:, 0], xyz[:, 1])
            strict = contains_xy(poly, xyz[:, 0], xyz[:, 1])
            if inside.sum() < 200 or strict.sum() < 100:
                continue
            base = float(xyz[inside, 2].min())
            xyz, strict = xyz[inside], strict[inside]
            local = np.column_stack([
                (xyz[:, 0] - poly.centroid.x) * METERS_PER_US_SURVEY_FOOT,
                (xyz[:, 1] - poly.centroid.y) * METERS_PER_US_SURVEY_FOOT,
                (xyz[:, 2] - base) * METERS_PER_US_SURVEY_FOOT])
            zones = split_surface_zones(local)
            if zones is None:
                continue

            # Height from returns strictly INSIDE the footprint, not from the
            # padded set the faces use. Downtown buildings abut, so a 2 m pad
            # reaches onto the neighbour's roof, and on the first run that put
            # the LiDAR height at 2.17x the listed one at p90 -- measuring the
            # taller building next door.
            #
            # Two heights, because they disagree on a stepped building and the
            # disagreement is informative: the plate is the modal level, which
            # on 400 West Market is a setback 24 m below its crown.
            plate_h = float(zones.plate_z_m - local[:, 2].min())
            top_h = float(np.percentile(local[strict, 2], 99))
            listed = b["est_height_m"]

            ring = np.array(poly.exterior.coords, dtype=np.float64)
            ring = np.column_stack([(ring[:, 0] - poly.centroid.x) * METERS_PER_US_SURVEY_FOOT,
                                    (ring[:, 1] - poly.centroid.y) * METERS_PER_US_SURVEY_FOOT])
            faces = facade_faces(ring, height_m=top_h,
                                 wall_xy=local[zones.wall][:, :2],
                                 min_length_m=args.min_face_m,
                                 max_offset_m=args.facade_pad_m)
            if not faces:
                continue

            per_m2 = np.array([f.returns_per_m2 for f in faces])
            caught = np.array([f.captured for f in faces])
            area = np.array([f.area_m2 for f in faces])
            row = {
                "gers_id": b["gers_id"],                    # the join key
                "tilename": tile,
                "overture_release": OVERTURE_RELEASE,
                "n_faces": len(faces),
                "n_faces_captured": int(caught.sum()),
                "faces_captured_frac": float(caught.mean()),
                # Weighted by area: one captured face out of four says little if
                # it is the short one.
                "wall_area_captured_frac": float(area[caught].sum() / area.sum()),
                "wall_area_m2": float(area.sum()),
                "wall_returns": int(sum(f.n_returns for f in faces)),
                "returns_per_m2_median": float(np.median(per_m2)),
                "returns_per_m2_best": float(per_m2.max()),
                "returns_per_m2_worst": float(per_m2.min()),
                "lidar_height_plate_m": plate_h,
                "lidar_height_p99_m": top_h,
                "listed_height_m": listed,
                "height_source": b["height_source"],
                "height_error_m": (None if listed is None else float(top_h - listed)),
                "height_ratio": (None if not listed else float(top_h / listed)),
            }
            buildings.append(row)
            for f in faces:
                faces_out.append({"gers_id": b["gers_id"], **f.to_dict()})
            done += 1
        print(f"[{ti}/{len(chosen)}] {tile}: {done}/{len(by_tile[tile])} buildings "
              f"({time.time() - started:.0f}s, {len(buildings)} total)")

    if not buildings:
        print("nothing measured")
        return 1

    fc = np.array([r["wall_area_captured_frac"] for r in buildings])
    hr = np.array([r["height_ratio"] for r in buildings if r["height_ratio"]], dtype=float)
    print("\n" + "=" * 70)
    print(f"{len(buildings):,} buildings, {len(faces_out):,} wall faces")
    print(f"\nwall AREA the LiDAR reached (>= {FACADE_CAPTURE_PER_M2}/m2)")
    print(f"{'':<10}" + "".join(f"p{q:<5}" for q in (10, 25, 50, 75, 90)))
    print(f"{'':<10}" + "".join(f"{np.percentile(fc, q):<6.0%}"[:6] for q in (10, 25, 50, 75, 90)))
    for thr in (0.25, 0.50, 0.75):
        print(f"  buildings with >{thr:.0%} of their wall reached: "
              f"{100 * (fc > thr).mean():5.1f}%")
    print(f"  fully reached (every face): "
          f"{100 * np.mean([r['faces_captured_frac'] == 1.0 for r in buildings]):5.1f}%")
    print(f"  not reached at all        : {100 * (fc == 0).mean():5.1f}%")

    listed = np.array([r["listed_height_m"] for r in buildings
                       if r["height_ratio"]], dtype=float)
    if len(hr):
        # Broken out by height, because the aggregate hides the finding: the
        # heights that select the biggest facades are the reliable ones, and
        # the error is concentrated in the short buildings that make up the
        # bulk of the list.
        print(f"\nLiDAR height / listed height, by listed height "
              f"({len(hr):,} with a listed height)")
        print(f"{'listed':<12}{'n':>5}{'ratio p50':>12}{'excess m p50':>15}"
              f"{'within 20%':>12}")
        for lo, hi in ((0, 15), (15, 20), (20, 40), (40, 10000)):
            m = (listed >= lo) & (listed < hi)
            if m.sum() >= 3:
                lab = f"{lo}-{hi} m" if hi < 10000 else f"{lo}+ m"
                print(f"{lab:<12}{m.sum():>5}{np.median(hr[m]):>12.2f}"
                      f"{np.median(listed[m] * (hr[m] - 1)):>15.1f}"
                      f"{np.mean(np.abs(hr[m] - 1) < 0.2):>12.0%}")
        print(f"  overall: within 20% on {100 * np.mean(np.abs(hr - 1) < 0.2):.0f}%, "
              f"listed more than double the LiDAR on "
              f"{100 * np.mean(hr < 0.5):.0f}%")

    # Which way do the walls the scanner missed face? A run sharing a bearing is
    # a flight-direction gap; a scatter is occlusion by neighbours.
    fb = np.array([f["bearing_deg"] for f in faces_out])
    fcap = np.array([f["captured"] for f in faces_out])
    print("\nfaces reached, by the direction the wall looks")
    for lo, name in ((315, "N"), (45, "E"), (135, "S"), (225, "W")):
        m = ((fb - lo) % 360) < 90
        if m.sum():
            print(f"  {name}: {m.sum():>5,} faces, {100 * fcap[m].mean():5.1f}% reached")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "params": {"buildings": str(args.buildings), "tiles_requested": args.tiles,
                   "tiles_measured": len({r["tilename"] for r in buildings}),
                   "sample": args.sample, "seed": args.seed,
                   "facade_pad_m": args.facade_pad_m,
                   "overture_release": OVERTURE_RELEASE},
        "key": "gers_id",
        "buildings": buildings,
        "faces": faces_out,
    }, indent=2), encoding="utf-8")
    print(f"\nresults: {args.out}  (keyed on gers_id)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
