#!/usr/bin/env python3
"""Stage 0: Overture's footprints grouped by the LiDAR tile that owns them.

Writes, to a store (a folder, or ``s3://bucket/prefix``):

* ``inputs/footprints/tile=<tilename>.json`` -- ``{"tile", "owned", "halo"}``:
  the footprints whose centroid lies in the tile, and those of other tiles
  within 30 m of it (measured with it, for party walls across the edge);
* ``inputs/tiles.json`` -- every tile with footprints, and their counts (the
  tiles written now added to those already listed).

The footprints come from a ``overture_boxes.py fetch`` JSON (``--overture``),
straight from the Overture release over a bounding box (``--bbox``), or over
the ``--tiles`` themselves and ``--margin-m`` around them (neither given): the
margin brings in the outlines on the far side of each edge, for the halo, and
the ones that start outside the box and reach in.

Usage::

    python scripts/stage0_index.py --overture data/derived/louisville_downtown_overture.json --store outputs/cloud
    python scripts/stage0_index.py --bbox -85.77 38.24 -85.74 38.27 --store s3://my-bucket/ky
    python scripts/stage0_index.py --store s3://my-bucket/ky --tiles N090E300 N090E301
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent / "src"))

from kyfromabove_3d.pieces import phase2_grid, records_by_tile              # noqa: E402
from kyfromabove_3d.store import store_from                               # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--overture", type=Path, help="a JSON written by overture_boxes.py fetch")
    src.add_argument("--bbox", type=float, nargs=4, metavar=("LON0", "LAT0", "LON1", "LAT1"))
    ap.add_argument("--store", required=True, help="a folder, or s3://bucket/prefix")
    ap.add_argument("--endpoint", default=None, help="an S3 endpoint (LocalStack: http://localhost:4566)")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"), help="where the LiDAR tile grids are")
    ap.add_argument("--tiles", nargs="+", default=None,
                    help="write only these tiles (a bbox also brings in its neighbours' outlines, for their halo)")
    ap.add_argument("--margin-m", type=float, default=400.0,
                    help="with --tiles and no --bbox or --overture: fetch this far around the tiles")
    args = ap.parse_args()

    grid = phase2_grid(args.raw)
    if not args.overture and not args.bbox:
        if not args.tiles:
            ap.error("give --overture, --bbox, or --tiles")
        args.bbox = tiles_bbox(grid, args.tiles, args.margin_m)
        print(f"fetching over {len(args.tiles)} tile(s) and {args.margin_m:.0f} m around them: "
              + " ".join(f"{v:.5f}" for v in args.bbox))
    if args.bbox:
        from overture_boxes import fetch
        tmp = Path(tempfile.mkdtemp()) / "overture.json"
        fetch(args.bbox, tmp)
        args.overture = tmp
    data = json.loads(args.overture.read_text(encoding="utf-8"))
    recs = data["buildings"]
    by = records_by_tile(recs, grid)
    if args.tiles:
        missing = [t for t in args.tiles if t not in by]
        if missing:
            print(f"[warn] no footprints on {', '.join(missing)}")
        by = {t: s for t, s in by.items() if t in set(args.tiles)}
    store = store_from(args.store, args.endpoint)
    for tile, sets in sorted(by.items()):
        store.put_json(f"inputs/footprints/tile={tile}.json", {"tile": tile, "release": data.get("release"), **sets})
    tiles = [{"tile": t, "owned": len(s["owned"]), "halo": len(s["halo"]), "release": data.get("release")}
             for t, s in sorted(by.items()) if s["owned"]]
    listed = store.read_json("inputs/tiles.json").get("tiles", []) if store.exists("inputs/tiles.json") else []
    new = {t["tile"] for t in tiles}
    every = sorted([t for t in listed if t["tile"] not in new] + tiles, key=lambda t: t["tile"])
    store.put_json("inputs/tiles.json", {"release": data.get("release"), "tiles": every})
    print(f"{len(recs)} footprints -> {len(tiles)} tile(s) written: "
          + ", ".join(f"{t['tile']} {t['owned']} (+{t['halo']} halo)" for t in tiles) + f" -> {args.store}")
    return 0


def tiles_bbox(grid, tiles, margin_m: float):
    """(lon0, lat0, lon1, lat1) around the named tiles of ``grid`` (EPSG:3857)
    and ``margin_m`` beyond them."""
    from pyproj import Transformer

    rows = grid[grid["tilename"].isin(tiles)]
    missing = sorted(set(tiles) - set(rows["tilename"]))
    if missing:
        raise SystemExit(f"not in the tile grid: {', '.join(missing)}")
    x0, y0, x1, y1 = rows.total_bounds
    lat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform((x0 + x1) / 2, (y0 + y1) / 2)[1]
    pad = margin_m / math.cos(math.radians(lat))           # Web Mercator metres are stretched by 1/cos(latitude)
    to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
    lon0, lat0 = to_ll(x0 - pad, y0 - pad)
    lon1, lat1 = to_ll(x1 + pad, y1 + pad)
    return [lon0, lat0, lon1, lat1]


if __name__ == "__main__":
    sys.exit(main())
