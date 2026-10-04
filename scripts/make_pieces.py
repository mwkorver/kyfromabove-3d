#!/usr/bin/env python3
"""Stage 1 on this machine: LiDAR tiles into per-building pieces.

For each tile: read its footprints (``inputs/footprints/tile=<name>.json``,
from ``stage0_index.py``), measure them from the LiDAR (product 1: ground,
height inside the edge, the split into parts), cut each building's piece
(``kyfromabove_3d.pieces``), and publish it -- files first, ``piece.json``
last -- to the store, a folder or ``s3://bucket/prefix``. A piece that has
not changed is not rewritten. The Lambda handler runs the same code.

Usage::

    python scripts/make_pieces.py --store outputs/cloud --all
    python scripts/make_pieces.py --store outputs/cloud --tiles N067E390 --frames-cache outputs/downtown/frames_2024s1.gpkg
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent / "src"))

from kyfromabove_3d.pieces import phase2_grid, run_tile, tile_bounds_ft   # noqa: E402
from kyfromabove_3d.store import store_from                               # noqa: E402


def frames_for(tile: str, grid, cache: Path | None, work: Path):
    """The oblique frames over a tile (read from the remote indexes once and
    cached), or the given cache when it covers the area already."""
    from kyfromabove_3d.prepare import load_frames_index
    return load_frames_index(cache or (work / f"frames_{tile}.gpkg"), tile_bounds_ft(grid, tile, pad_ft=300.0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="a folder, or s3://bucket/prefix")
    ap.add_argument("--endpoint", default=None, help="an S3 endpoint (LocalStack: http://localhost:4566)")
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--tiles", nargs="+")
    which.add_argument("--all", action="store_true", help="every tile in inputs/tiles.json")
    ap.add_argument("--work", type=Path, default=None, help="where pieces are cut before publishing (default: a temp dir)")
    ap.add_argument("--frames-cache", type=Path, default=None,
                    help="a frames index GeoPackage covering the tiles (load_frames_index writes one)")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"), help="where the LiDAR tile grids are")
    args = ap.parse_args()

    from kyfromabove_3d.lidar import LidarTileIndex

    store = store_from(args.store, args.endpoint)
    tiles = args.tiles or [t["tile"] for t in store.read_json("inputs/tiles.json")["tiles"]]
    grid = phase2_grid(args.raw)
    index = LidarTileIndex(args.raw)
    work = args.work or Path(tempfile.mkdtemp(prefix="pieces_"))
    t0 = time.time()
    total = written = 0
    for t in tiles:
        inputs = store.read_json(f"inputs/footprints/tile={t}.json")
        res = run_tile(t, inputs, work / t, store, grid, index, frames_for(t, grid, args.frames_cache, work))
        total += res["pieces"]; written += res["written"]
    print(f"{len(tiles)} tile(s): {total} pieces, {written} written, {total - written} unchanged "
          f"in {time.time() - t0:.0f} s -> {args.store}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
