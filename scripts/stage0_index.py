#!/usr/bin/env python3
"""Stage 0: Overture's footprints grouped by the LiDAR tile that owns them.

Writes, to a store (a folder, or ``s3://bucket/prefix``):

* ``inputs/footprints/tile=<tilename>.json`` -- ``{"tile", "owned", "halo"}``:
  the footprints whose centroid lies in the tile, and those of other tiles
  within 30 m of it (measured with it, for party walls across the edge);
* ``inputs/tiles.json`` -- every tile with footprints, and their counts: the
  list stage 1 runs over.

The footprints come from a ``overture_boxes.py fetch`` JSON (``--overture``),
or straight from the Overture release over a bounding box (``--bbox``).

Usage::

    python scripts/stage0_index.py --overture data/derived/louisville_downtown_overture.json --store outputs/cloud
    python scripts/stage0_index.py --bbox -85.77 38.24 -85.74 38.27 --store s3://my-bucket/ky
"""

from __future__ import annotations

import argparse
import json
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
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--overture", type=Path, help="a JSON written by overture_boxes.py fetch")
    src.add_argument("--bbox", type=float, nargs=4, metavar=("LON0", "LAT0", "LON1", "LAT1"))
    ap.add_argument("--store", required=True, help="a folder, or s3://bucket/prefix")
    ap.add_argument("--endpoint", default=None, help="an S3 endpoint (LocalStack: http://localhost:4566)")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"), help="where the LiDAR tile grids are")
    ap.add_argument("--tiles", nargs="+", default=None,
                    help="write only these tiles (a bbox also brings in its neighbours' outlines, for their halo)")
    args = ap.parse_args()

    if args.bbox:
        from overture_boxes import fetch
        tmp = Path(tempfile.mkdtemp()) / "overture.json"
        fetch(args.bbox, tmp)
        args.overture = tmp
    data = json.loads(args.overture.read_text(encoding="utf-8"))
    recs = data["buildings"]
    grid = phase2_grid(args.raw)
    by = records_by_tile(recs, grid)
    if args.tiles:
        missing = [t for t in args.tiles if t not in by]
        if missing:
            print(f"[warn] no footprints on {', '.join(missing)}")
        by = {t: s for t, s in by.items() if t in set(args.tiles)}
    store = store_from(args.store, args.endpoint)
    for tile, sets in sorted(by.items()):
        store.put_json(f"inputs/footprints/tile={tile}.json", {"tile": tile, "release": data.get("release"), **sets})
    tiles = [{"tile": t, "owned": len(s["owned"]), "halo": len(s["halo"])} for t, s in sorted(by.items()) if s["owned"]]
    store.put_json("inputs/tiles.json", {"release": data.get("release"), "tiles": tiles})
    print(f"{len(recs)} footprints -> {len(tiles)} tile(s) written: "
          + ", ".join(f"{t['tile']} {t['owned']} (+{t['halo']} halo)" for t in tiles) + f" -> {args.store}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
