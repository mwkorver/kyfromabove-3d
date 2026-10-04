#!/usr/bin/env python3
"""Step 2, the piece step, on this machine: what the Lambda
``kyfromabove-3d-piece-step`` does, for checking it before and after.

For each tile: read the tile step's ``tiles/<name>/buildings.json`` and
``frames.gpkg``, and for every building cut its piece -- the LiDAR clip at
full resolution, the frames that see it as pointers (``frames.json``), its
local frame and outline -- and publish it to ``pieces/<gers>/``, the marker
``piece.json`` last. Unchanged pieces are not rewritten.

Usage::

    python scripts/piece_step.py --store outputs/step2_local --inputs s3://kyfromabove-3d --tiles N076E229
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kyfromabove_3d.pieces import piece_step, publish_piece                 # noqa: E402
from kyfromabove_3d.store import store_from                                 # noqa: E402


def main() -> int:
    import geopandas as gpd

    from kyfromabove_3d.lidar import LidarTileIndex

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="where to write the pieces: a folder, or s3://bucket")
    ap.add_argument("--inputs", default=None, help="where the tile step wrote its results (default: --store)")
    ap.add_argument("--tiles", nargs="+", required=True)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    args = ap.parse_args()
    out = store_from(args.store)
    inp = store_from(args.inputs) if args.inputs else out
    from kyfromabove_3d.sources_choice import indexes
    index = indexes(args.raw)
    work = Path(tempfile.mkdtemp(prefix="piece_step_"))
    for t in args.tiles:
        t0 = time.time()
        tile = inp.read_json(f"tiles/{t}/buildings.json")
        frames = gpd.read_file(inp.get_file(f"tiles/{t}/frames.gpkg", work / t / "frames.gpkg"))
        made = piece_step(t, [r["id"] for r in tile["buildings"]], tile["buildings"], tile["halo"], frames, index,
                          work / t / "pieces")
        written = sum(publish_piece(f, out) for f in made["pieces"])
        n = len(made["pieces"])
        print(f"{t}: {n} pieces, {written} written, {n - written} unchanged, {len(made['failed'])} failed, "
              f"{time.time() - t0:.0f} s ({(time.time() - t0) / max(n, 1):.2f} s a building)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
