#!/usr/bin/env python3
"""Step 1, the tile step, on this machine: what the Lambda
``kyfromabove-3d-tile-step`` does, for checking it before and after.

For each tile: read its Overture outlines (``inputs/footprints/tile=<name>.json``,
from ``stage0_index.py``), read the LiDAR around them at full resolution,
measure every building (product 1: ground, height 1.5 m inside the edge),
cut outlines holding two or three buildings into parts, and list the oblique
frames over the tile. Writes ``tiles/<name>/buildings.json``, ``frames.gpkg``
and ``summary.json`` to the store (a folder or ``s3://bucket``). No pieces,
no crops.

Usage::

    python scripts/tile_step.py --store outputs/tile_step --inputs s3://kyfromabove-3d --tiles N076E228
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kyfromabove_3d.pieces import (phase2_grid, publish_tile_step, tile_bounds_ft, tile_ground_bytes,  # noqa: E402
                                   tile_step)
from kyfromabove_3d.store import store_from                                                  # noqa: E402


def run(tile: str, inputs_store, out_store, grid, index, work: Path, frames_cache: Path | None = None,
        log=print) -> dict:
    """One tile, as the Lambda runs it."""
    from kyfromabove_3d.sources_choice import choose_sources

    inputs = inputs_store.read_json(f"inputs/footprints/tile={tile}.json")
    # index: {phase: single-phase index}; the LiDAR and obliques closest in time
    choice = choose_sources(tile, grid, index, work, log=log)
    phase_index = index[choice["phase"]]
    res = tile_step(tile, inputs.get("owned", []), inputs.get("halo", []), grid, phase_index, log=log,
                    provenance=choice["provenance"])
    frames_path = choice["frames_path"]
    res["stats"]["frames"] = int(len(choice["frames"]))
    res["stats"]["sources_considered"] = choice["candidates"]
    index = phase_index
    ground = tile_ground_bytes(tile, grid, index, res["stats"], log=log)
    publish_tile_step(tile, res, frames_path, out_store, ground=ground)
    return res["stats"]


def main() -> int:
    from kyfromabove_3d.lidar import LidarTileIndex

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="where to write: a folder, or s3://bucket")
    ap.add_argument("--inputs", default=None, help="where stage 0 wrote the inputs (default: --store)")
    ap.add_argument("--tiles", nargs="+", required=True)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    args = ap.parse_args()
    out = store_from(args.store)
    inp = store_from(args.inputs) if args.inputs else out
    grid = phase2_grid(args.raw)
    from kyfromabove_3d.sources_choice import indexes
    index = indexes(args.raw)
    work = Path(tempfile.mkdtemp(prefix="tile_step_"))
    for t in args.tiles:
        s = run(t, inp, out, grid, index, work)
        print(f"{t}: {s['buildings']} buildings ({s['split']} outlines split), {s['measured']} measured, "
              f"{s['frames']} frames, {s['returns']:,} returns, {s['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
