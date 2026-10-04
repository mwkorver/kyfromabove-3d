"""The Lambda handlers: a few lines each around the package.

* :func:`tile_step` -- step 1, invoked directly with ``{"tile": ...}``:
  product 1 and the frame list for one LiDAR tile, written to the bucket.
* :func:`piece_step` -- step 2, invoked directly with ``{"tile", "gers"}``:
  a batch of one tile's buildings cut into pieces, written to the bucket.
* :func:`building_step` -- step 3, invoked directly with ``{"gers": [...]}``:
  a few nearby buildings from their pieces to their meshes, the check and
  the audit, written to ``buildings/<gers>/``.
* :func:`make_pieces` -- stage 1. Each SQS message is one LiDAR tile,
  ``{"tile": "N076E228"}``; its pieces are cut in /tmp and published to the
  bucket, each marker (``pieces/<gers>/piece.json``) last. A marker's arrival
  is the S3 event EventBridge routes to the next step.
* :func:`record_piece` -- the next step, this round: each SQS message is the
  EventBridge event of one marker; the piece is recorded in DynamoDB. The
  building step replaces it on the same queue in the next round.

Both report failed messages one by one (``batchItemFailures``) so a bad tile
is retried and, after that, parked in its dead-letter queue, while the rest
of its batch goes through.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from decimal import Decimal
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
    from kyfromabove_3d.pieces import publish_tile_step, tile_ground_bytes
    from kyfromabove_3d.pieces import tile_step as run_tile_step
    from kyfromabove_3d.sources_choice import choose_sources

    t0 = time.time()
    store = _store()
    cold = "grid" not in _CACHE
    grid, _ = _grid_and_index()
    idx = _indexes()
    t_setup = time.time() - t0
    tile = event["tile"]
    work = Path(tempfile.mkdtemp(dir=os.environ.get("WORK_DIR", "/tmp")))
    try:
        t = time.time()
        inputs = store.read_json(f"inputs/footprints/tile={tile}.json")
        t_inputs = time.time() - t
        # which LiDAR phase and oblique collection, closest in time (sources_choice)
        t = time.time()
        choice = choose_sources(tile, grid, idx, work)
        t_choice = time.time() - t
        index = idx[choice["phase"]]
        res = run_tile_step(tile, inputs.get("owned", []), inputs.get("halo", []), grid, index,
                            provenance=choice["provenance"])
        frames_path = choice["frames_path"]
        res["stats"].update({"frames": int(len(choice["frames"])), "seconds_choice": round(t_choice, 1),
                             "sources_considered": choice["candidates"],
                             "seconds_setup": round(t_setup, 1), "cold_container": cold,
                             "seconds_inputs": round(t_inputs, 1)})
        ground = tile_ground_bytes(tile, grid, index, res["stats"])
        publish_tile_step(tile, res, frames_path, store, ground=ground)
        res["stats"]["seconds_handler"] = round(time.time() - t0, 1)
        print(json.dumps(res["stats"]))
        return res["stats"]
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _tile_inputs(store, tile: str):
    """The tile step's results for one tile, read once per container."""
    key = ("tile", tile)
    if key not in _CACHE:
        import geopandas as gpd

        folder = Path(tempfile.mkdtemp(dir=os.environ.get("WORK_DIR", "/tmp")))
        rec = store.read_json(f"tiles/{tile}/buildings.json")
        frames = gpd.read_file(store.get_file(f"tiles/{tile}/frames.gpkg", folder / "frames.gpkg"))
        _CACHE[key] = (rec, frames)
    return _CACHE[key]


def piece_step(event, context):
    """Step 2 for a batch of one tile's buildings, invoked directly:
    ``{"tile": "N076E228", "gers": [...]}``. Cuts each building's piece
    (LiDAR clip, the frames that see it as pointers) and writes it to
    ``pieces/<gers>/``, ``piece.json`` last; an unchanged piece is not
    rewritten. Returns counts and timings."""
    from kyfromabove_3d.pieces import piece_step as run_piece_step
    from kyfromabove_3d.pieces import publish_piece

    t0 = time.time()
    store = _store()
    cold = "grid" not in _CACHE
    _grid_and_index()
    index = _indexes()                            # each building's clip from its tile's chosen phase
    tile, gers = event["tile"], list(event["gers"])
    rec, frames = _tile_inputs(store, tile)
    t_setup = time.time() - t0
    work = Path(tempfile.mkdtemp(dir=os.environ.get("WORK_DIR", "/tmp")))
    try:
        t = time.time()
        made = run_piece_step(tile, gers, rec["buildings"], rec["halo"], frames, index, work)
        t_cut = time.time() - t
        t = time.time()
        written = sum(publish_piece(f, store) for f in made["pieces"])
        out = {"tile": tile, "batch": len(gers), "pieces": len(made["pieces"]), "written": written,
               "unchanged": len(made["pieces"]) - written, "failed": made["failed"], "cold_container": cold,
               "seconds_setup": round(t_setup, 1), "seconds_cut": round(t_cut, 1),
               "seconds_publish": round(time.time() - t, 1),
               "seconds_per_building": round(t_cut / max(len(made["pieces"]), 1), 2)}
        print(json.dumps(out))
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def building_step(event, context):
    """Step 3 for a few nearby buildings, invoked directly:
    ``{"gers": [...], "force": false}``. Each building's piece is read from
    ``pieces/<gers>/``; its crops are cut in /tmp; its meshes, check and
    audit are written to ``buildings/<gers>/``, ``building.json`` last. A
    building already built from the same piece by the same code is skipped
    unless ``force``. Returns each building's seconds, peak memory and size."""
    from kyfromabove_3d.building import building_step as run_building_step

    t0 = time.time()
    store = _store()
    cold = "grid" not in _CACHE
    _grid_and_index()
    index = _indexes()
    work = Path(tempfile.mkdtemp(dir=os.environ.get("WORK_DIR", "/tmp")))
    # every script must end before Lambda does, with time left to report
    left_s = context.get_remaining_time_in_millis() / 1000.0 if context else 900.0
    try:
        res = run_building_step(list(event["gers"]), store, work, index, force=bool(event.get("force")),
                                timeout_s=max(60.0, left_s - 30.0))
        out = {"built": len(res["built"]), "skipped": len(res["skipped"]), "failed": res["failed"],
               "cold_container": cold, "seconds": res["seconds"], "peak_mb_scene": res["peak_mb_scene"],
               "buildings": [{k: b.get(k) for k in ("gers", "seconds", "peak_mb", "bytes", "triangles", "path")}
                             for b in res["built"]],
               "seconds_handler": round(time.time() - t0, 1)}
        print(json.dumps(out))
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def make_pieces(event, context):
    """Stage 1: one LiDAR tile per message."""
    from kyfromabove_3d.pieces import run_tile, tile_bounds_ft
    from kyfromabove_3d.prepare import load_frames_index

    store = _store()
    grid, index = _grid_and_index()
    failures = []
    for msg in event.get("Records", []):
        work = Path(tempfile.mkdtemp(dir=os.environ.get("WORK_DIR", "/tmp")))
        try:
            tile = json.loads(msg["body"])["tile"]
            inputs = store.read_json(f"inputs/footprints/tile={tile}.json")
            frames = load_frames_index(work / "frames.gpkg", tile_bounds_ft(grid, tile, pad_ft=300.0))
            res = run_tile(tile, inputs, work / "pieces", store, grid, index, frames)
            print(json.dumps({"tile": tile, **res}))
        except Exception as exc:                                    # noqa: BLE001
            print(json.dumps({"failed": msg.get("messageId"), "error": repr(exc)[:2000]}))
            failures.append({"itemIdentifier": msg["messageId"]})
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return {"batchItemFailures": failures}


def record_piece(event, context):
    """The next step, this round: record each piece whose marker arrived."""
    import boto3

    s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None)
    table = boto3.resource("dynamodb", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None
                           ).Table(os.environ["TABLE"])
    failures = []
    for msg in event.get("Records", []):
        try:
            ev = json.loads(msg["body"])
            bucket, key = ev["detail"]["bucket"]["name"], ev["detail"]["object"]["key"]
            marker = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read(), parse_float=Decimal)
            table.put_item(Item={
                **{k: v for k, v in marker.items() if v is not None and v != []},
                "gers": marker["gers"], "status": "ready", "key": key,
                "event_time": ev.get("time"), "recorded_at": int(time.time())})
        except Exception as exc:                                    # noqa: BLE001
            print(json.dumps({"failed": msg.get("messageId"), "error": repr(exc)[:2000]}))
            failures.append({"itemIdentifier": msg["messageId"]})
    return {"batchItemFailures": failures}
