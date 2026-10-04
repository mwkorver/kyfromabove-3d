#!/usr/bin/env python3
"""Find tall buildings that actually have oblique coverage.

The pilot building was picked by hand, and it is 16 m of flat roof in a rural
corridor. Understanding how the pipeline behaves in a dense area needs a target
chosen from data, against two constraints that pull in opposite directions:

* **Height** comes from the Phase 2 COPC point cloud. A tower is mostly facade
  by surface area, which is where aerial LiDAR is thinnest -- so a tall
  candidate is a harder and more informative test than the pilot.
* **Coverage** comes from the oblique frame index. Only one of the four
  KyFromAbove oblique collections reaches the urban cores, so a tall building
  is worthless to this project if no collection flew it.

Nothing large is downloaded. The point cloud is read by COPC range query and the
1.07 GB frame index by ``/vsicurl`` range request against the RTree.

Usage::

    python scripts/find_urban_candidates.py --city Louisville
    python scripts/find_urban_candidates.py --lon -85.759 --lat 38.255 --radius-ft 2600
    python scripts/find_urban_candidates.py --city Lexington --no-coverage
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

from kyfromabove_3d.lidar import LidarTileIndex
from kyfromabove_3d.urban import (
    OBLIQUE_COLLECTIONS,
    collections_covering,
    find_tall_clusters,
    frames_index_url,
)

CITIES = {
    "Louisville": (-85.7590, 38.2550), "Lexington": (-84.4955, 38.0450),
    "Covington": (-84.5085, 39.0840), "Frankfort": (-84.8730, 38.2010),
    "Paducah": (-88.6000, 37.0840), "Owensboro": (-87.1130, 37.7740),
    "Bowling Green": (-86.4440, 36.9910), "Ashland": (-82.6380, 38.4780),
}


def load_cloud(index: LidarTileIndex, bounds):
    """Merge every Phase 2 COPC tile covering the window.

    Using only the first covering tile silently truncates at the tile boundary,
    which is how an early pass reported Lexington's core at 98 ft when it is
    221 ft. Downtowns sit on tile corners more often than not.
    """
    import laspy

    tiles = index.tiles_for_bounds(bounds)
    if not tiles:
        return None, None, []
    minx, miny, maxx, maxy = bounds
    chunks, classes = [], []
    for tile in tiles:
        if not tile.is_copc:
            print(f"  [skip] {tile.tilename} is plain LAZ; range queries need COPC")
            continue
        reader = laspy.CopcReader.open(tile.url)
        header = reader.header
        pts = reader.query(laspy.copc.Bounds(
            mins=np.array([minx, miny, header.z_min]),
            maxs=np.array([maxx, maxy, header.z_max])))
        if len(pts.x) == 0:
            continue
        chunks.append(np.stack([pts.x, pts.y, pts.z], axis=1))
        classes.append(np.asarray(pts.classification))
    if not chunks:
        return None, None, tiles
    return np.vstack(chunks), np.concatenate(classes), tiles


def frame_coverage(collection, cx, cy, half_ft):
    """Frames covering a box, by camera head, read from the index in place."""
    import geopandas as gpd

    # The frame footprints are "Measured 3D Polygon"; pyogrio warns on every
    # read that it is dropping the M values. We do not use them.
    warnings.filterwarnings("ignore", message=".*Measured.*", category=UserWarning)
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".gpkg")
    frames = gpd.read_file(frames_index_url(collection),
                           bbox=(cx - half_ft, cy - half_ft, cx + half_ft, cy + half_ft))
    if len(frames) == 0:
        return 0, {}, frames
    return len(frames), dict(sorted(Counter(frames["CameraID"]).items())), frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", choices=sorted(CITIES), default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--radius-ft", type=float, default=2600.0,
                    help="half-width of the search window")
    ap.add_argument("--box-ft", type=float, default=400.0,
                    help="half-width of the per-candidate coverage box")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--no-coverage", action="store_true",
                    help="skip the oblique index lookup (LiDAR heights only)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.city:
        lon, lat = CITIES[args.city]
        label = args.city
    elif args.lon is not None and args.lat is not None:
        lon, lat, label = args.lon, args.lat, f"{args.lon:.4f},{args.lat:.4f}"
    else:
        print("[FAIL] pass --city, or both --lon and --lat")
        return 1

    from pyproj import Transformer
    to_ft = Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True).transform
    to_ll = Transformer.from_crs("EPSG:3089", "EPSG:4326", always_xy=True).transform
    cx, cy = to_ft(lon, lat)
    R = args.radius_ft
    bounds = (cx - R, cy - R, cx + R, cy + R)

    shortlist = collections_covering(lon, lat)
    print("=" * 78)
    print(f"{label}  ({lon:.4f}, {lat:.4f})   window {2 * R:.0f} x {2 * R:.0f} ft")
    print("=" * 78)
    print(f"oblique collections whose extent contains this point: "
          f"{shortlist or 'NONE'}")
    for name in shortlist:
        print(f"    {name}: {OBLIQUE_COLLECTIONS[name]['note']}")
    print("  (bounding-box test -- the real footprints have holes, so this "
          "over-reports;\n   the per-candidate lookup below is the real check)")

    index = LidarTileIndex(args.raw_dir)
    xyz, cls, tiles = load_cloud(index, bounds)
    if xyz is None:
        print("\n[FAIL] no COPC points for this window")
        return 1
    print(f"\nPhase 2 COPC: {len(tiles)} tile(s) {[t.tilename for t in tiles]}, "
          f"{len(xyz):,} points")

    clusters = find_tall_clusters(xyz, cls, to_ll, max_results=args.top)
    if not clusters:
        print("no structures found")
        return 1

    if not args.no_coverage:
        for c in clusters:
            best = 0, {}
            for name in shortlist:
                try:
                    n, heads, _ = frame_coverage(name, c.x_ft, c.y_ft, args.box_ft)
                except Exception as exc:                    # noqa: BLE001
                    print(f"  [warn] {name} index read failed: {str(exc)[:120]}")
                    continue
                if n > best[0]:
                    best = n, heads
                    c.frames, c.frames_by_head = n, heads
            if c.frames is None:
                c.frames, c.frames_by_head = 0, {}

    print(f"\n{'#':<3}{'height':>12}{'pts':>7}{'frames':>8}  {'by head':<38}lon, lat")
    print("-" * 104)
    for i, c in enumerate(clusters, 1):
        heads = ("" if not c.frames_by_head
                 else " ".join(f"{k}:{v}" for k, v in c.frames_by_head.items()))
        fr = "-" if c.frames is None else str(c.frames)
        print(f"{i:<3}{c.height_ft:>8.0f} ft{c.height_m:>7.0f} m{c.n_points:>7}"
              f"{fr:>8}  {heads:<38}{c.lon:.5f}, {c.lat:.5f}")

    covered = [c for c in clusters if (c.frames or 0) > 0]
    if covered:
        best = max(covered, key=lambda c: c.height_ft)
        print(f"\nTallest candidate with oblique coverage: {best.height_ft:.0f} ft "
              f"({best.height_m:.0f} m) at {best.lon:.5f}, {best.lat:.5f}")
        heads = best.frames_by_head or {}
        expected = {"Bwd", "Color", "Fwd", "Left", "Right"}
        thin = sorted([k for k, v in heads.items() if v < 5]
                      + [k + ":0" for k in expected - set(heads)])
        if thin:
            print(f"  WATCH: thin coverage on {', '.join(thin)} -- ray separation "
                  f"depends on\n  balanced heads, and a lopsided set may not see "
                  f"every facade.")
    else:
        print("\nNo candidate here has oblique coverage.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"query": {"label": label, "lon": lon, "lat": lat,
                       "radius_ft": R, "collections": shortlist},
             "lidar_tiles": [t.tilename for t in tiles],
             "candidates": [c.to_dict() for c in clusters]}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
