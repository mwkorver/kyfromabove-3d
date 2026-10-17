"""The Lambda handler: :func:`tile_step`, step 1, invoked directly with
``{"tile": ...}``: product 1 and the frame list for one LiDAR tile, written to
the bucket. The pieces and the buildings are the queue run's
(``kyfromabove_3d.cloud_jobs``, refactor step 4).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

_CACHE: dict = {}


def _store():
    from kyfromabove_3d.store import S3Store

    return S3Store(os.environ["BUCKET"], os.environ.get("PREFIX", ""),
                   endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None)


def _grid_and_index():
    """The Phase 2 tile grid ships in the image; loaded once per container."""
    if "grid" not in _CACHE:
        from kyfromabove_3d.lidar import LidarTileIndex
        from kyfromabove_3d.pieces import phase2_grid

        raw = Path(os.environ.get("RAW_DIR", "/opt/data/raw"))
        _CACHE["grid"] = phase2_grid(raw)
        _CACHE["index"] = LidarTileIndex(raw, phases=("phase2",))
    return _CACHE["grid"], _CACHE["index"]


def _indexes():
    """One single-phase LiDAR index per phase (sources_choice), once per container."""
    if "indexes" not in _CACHE:
        from kyfromabove_3d.sources_choice import indexes

        _CACHE["indexes"] = indexes(Path(os.environ.get("RAW_DIR", "/opt/data/raw")))
    return _CACHE["indexes"]


def tile_step(event, context):
    """Step 1 for one tile, invoked directly: ``{"tile": "N076E228"}``.
    Reads the tile's Overture outlines from the bucket, measures them from
    the LiDAR, splits, lists the frames over the tile, and writes
    ``tiles/<name>/buildings.json``, ``frames.gpkg`` and ``summary.json``.
    Returns the summary."""
    from kyfromabove_3d.pieces import run_tile_step

    t0 = time.time()
    store = _store()
    cold = "grid" not in _CACHE
    grid, _ = _grid_and_index()
    idx = _indexes()
    t_setup = time.time() - t0
    tile = event["tile"]
    work = Path(tempfile.mkdtemp(dir=os.environ.get("WORK_DIR", "/tmp")))
    try:
        stats = run_tile_step(tile, store, store, grid, idx, work)
        stats.update({"seconds_setup": round(t_setup, 1), "cold_container": cold,
                      "seconds_handler": round(time.time() - t0, 1)})
        print(json.dumps(stats))
        return stats
    finally:
        shutil.rmtree(work, ignore_errors=True)
