"""The jobs a queue worker runs (``kyfromabove_3d.queue_worker``): the one
building pipeline (``kyfromabove_3d.pipeline``), in the cloud.

A run sends three kinds of message, each kind only once the one before is
done -- a tile's buildings are measured against its neighbouring tiles'
records, and built against their measured levels:

* ``{"kind": "tile", "run": ..., "tile": T}`` -- step 1 for one LiDAR tile
  (``pieces.run_tile_step``): its Overture outlines from
  ``inputs/footprints/tile=T.json`` measured from the LiDAR and split, the
  LiDAR and obliques taken closest together in time, the frames over it and
  its ground grid. Published to ``tiles/T/``, ``summary.json`` last.
* ``{"kind": "measure", "run": ..., "tile": T, "gers": [...]}`` -- the
  buildings of one LiDAR tile (from the tile step's ``tiles/T/buildings.json``)
  measured: each one's piece (``pieces.piece_step``, as the cloud has cut
  them since step 2), its class, its LiDAR outline and towers where the class
  builds on them, its levels as the LiDAR shows them. Published to
  ``pieces/<gers>/`` as one ``piece.zip`` and the marker ``piece.json`` last
  (``building.publish_piece``); the tile's buildings' frames and levels to
  ``tiles/T/measured.json``, which their neighbours' build batches read.
* ``{"kind": "build", "run": ..., "tile": T, "gers": [...]}`` -- a batch of
  8-16 nearby buildings built: their pieces and their neighbours' measured
  levels fetched, the LiDAR around them read once, then per building -- unless
  its fingerprint matches the one published -- the crops, the colourise, the
  mesh, the completed product and the borrowed fill (product 4), the texture
  check (on a sample), the audit;
  published to ``buildings/<gers>/``, ``building.json`` last (it holds the
  fingerprint: the checkpoint a later run, or another machine after a Spot
  interruption, skips by).

Then the viewer's two (step 4, ``scripts/publish_step.py``'s parts):

* ``{"kind": "publish", "run": ..., "tile": T}`` -- the tile's batched 3D
  Tiles for every view (``tiles/T/3dtiles/<product>/``), its buildings read
  from ``buildings/<gers>/`` beside it.
* ``{"kind": "tree", "run": ...}`` -- once the tiles are published: the
  statewide tree for each view (``kentucky/<product>/``) and the terrain
  (``terrain/``), over every tile published in the store, so tiles published
  before stay in the view. The viewer page and its settings are left as
  they are.

The store is ``STORE`` (``s3://bucket/prefix``, or a folder for a test), the
LiDAR tile indexes ``RAW_DIR``, the working disk ``WORK_DIR``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

#: the texture check runs on this share of the buildings (pipeline: the same ones every run)
CHECK_SAMPLE = float(os.environ.get("CHECK_SAMPLE", "0.1"))
NEIGHBOUR_REACH_M = 450.0               # neighbours fetched within this of a batch: the fingerprint looks 350 m from each building


def _store():
    from .store import store_from
    return store_from(os.environ["STORE"], endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None)


def _work() -> Path:
    return Path(tempfile.mkdtemp(dir=os.environ.get("WORK_DIR") or None))


def _indexes():
    from .sources_choice import indexes
    return indexes(Path(os.environ.get("RAW_DIR", "/opt/data/raw")))


def _record_failure(store, run: str, gers: str, error: str) -> None:
    store.put_json(f"runs/{run}/failed/{gers}.json", {"gers": gers, "error": error[-4000:],
                                                       "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


# ------------------------------------------------------------------ tile

def tile_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """One tile's step 1 (see the module)."""
    from .pieces import phase2_grid, run_tile_step

    store, tile = _store(), job["tile"]
    work = _work()
    try:
        grid = phase2_grid(Path(os.environ.get("RAW_DIR", "/opt/data/raw")))
        s = run_tile_step(tile, store, store, grid, _indexes(), work)
        return {"tile": tile, "buildings": s.get("buildings"), "measured": s.get("measured"),
                "frames": s.get("frames"), "seconds": s.get("seconds")}
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------------------ measure

def measure_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """One tile's buildings measured and published (see the module)."""
    import geopandas as gpd

    from .building import neighbour_records, publish_piece, publish_tile_measured
    from .pieces import piece_step
    from .pipeline import MEASURED_LEVELS, write_measured_levels
    from .classify import classify_building
    from .settings import resolve

    from .stages import lidar_outline
    from .colourise.levels import ColouriseFailure, footprint_and_levels

    store, run, tile = _store(), job.get("run", "run"), job["tile"]
    work = _work()
    try:
        rec = store.read_json(f"tiles/{tile}/buildings.json")
        if not store.exists(f"tiles/{tile}/frames.gpkg"):
            # no oblique photos over the tile (sources_choice): its buildings stay
            # LiDAR-height boxes; nothing to measure for a mesh
            return {"tile": tile, "measured": 0, "failed": 0, "no_photos": len(rec["buildings"])}
        frames = gpd.read_file(store.get_file(f"tiles/{tile}/frames.gpkg", work / "frames.gpkg"))
        gers = job.get("gers") or [r["id"] for r in rec["buildings"]]
        made = piece_step(tile, gers, rec["buildings"], rec["halo"], frames, _indexes(), work)
        records = neighbour_records(tile, store)
        done, failed = [], dict(made["failed"])
        for folder in made["pieces"]:
            g = folder.name
            try:
                classify_building(folder)
                s = resolve(folder)
                r = None
                if s.colourise.outline == "lidar":
                    r = lidar_outline.outline_for(folder, records)
                    if not lidar_outline.write_lidar_outline(folder, r):
                        r = None
                try:
                    F = footprint_and_levels(folder, s.colourise)
                    if r is not None and len(F.levels) > 1:
                        towers = lidar_outline.towers_for(folder, F.levels, r)
                        if towers:
                            lidar_outline.write_lidar_outline(folder, r, towers)
                            F = footprint_and_levels(folder, s.colourise)
                    write_measured_levels(folder, F.levels)
                except ColouriseFailure:
                    (folder / MEASURED_LEVELS).unlink(missing_ok=True)
                publish_piece(folder, store)                            # piece.zip, the marker last
                done.append(folder)
            except Exception as exc:                                    # noqa: BLE001
                failed[g] = repr(exc)
        if done:
            publish_tile_measured(tile, done, store)
        for g, err in failed.items():
            _record_failure(store, run, g, f"measure: {err}")
        return {"tile": tile, "measured": len(done), "failed": len(failed)}
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------------------ build

def _near(records: List[Dict[str, Any]], origins_ft, reach_m: float) -> List[str]:
    """The records whose outline lies within ``reach_m`` of any origin."""
    import numpy as np
    from shapely import wkt as _wkt
    from pyproj import Transformer

    to_ft = Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True)
    o = np.asarray(origins_ft, dtype=float)[:, :2]
    out = []
    for r in records:
        c = _wkt.loads(r["wkt"]).centroid
        x, y = to_ft.transform(c.x, c.y)
        if (np.hypot(o[:, 0] - x, o[:, 1] - y) * 0.3048006096012192).min() <= reach_m:
            out.append(r["id"])
    return out


def build_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """A batch of nearby buildings built and published (see the module)."""
    import hashlib

    import numpy as np

    from .building import (BUILDINGS_PREFIX, adjacent_tiles, neighbour_records, publish_building, pull_neighbours,
                           pull_piece, scene_cloud_for)
    from .coordinates import LocalCoordinateFrame
    from .pipeline import cut_crops_from_piece, fingerprint, run_main
    from .queue_worker import machine_type
    from .sources_choice import index_for

    store, run, tile, gers = _store(), job.get("run", "run"), job["tile"], list(job["gers"])
    work = _work()
    t0 = time.time()
    try:
        records = neighbour_records(tile, store)
        nb_path = work / "neighbours.json"
        nb_path.write_text(json.dumps({"buildings": records}), encoding="utf-8")
        markers = {}
        for g in gers:
            try:
                markers[g] = pull_piece(g, store, work / "buildings" / g)       # with what measuring added
            except Exception as exc:                                    # noqa: BLE001
                _record_failure(store, run, g, f"no piece: {exc!r}")
        if not markers:
            return {"built": 0, "skipped": 0, "failed": len(gers)}
        # the neighbours' measured levels (they hide each building's walls from the frames)
        origins = [LocalCoordinateFrame.load_json(work / "buildings" / g / "coordinates.json").origin_source_xyz
                   for g in markers]
        near = [nid for nid in _near(records, origins, NEIGHBOUR_REACH_M) if nid not in markers]
        pull_neighbours(near, [tile] + adjacent_tiles(tile), store, work / "buildings")
        # the buildings already published from the same inputs, settings, neighbours and code are done
        prints = {g: fingerprint(work / "buildings" / g, work / "buildings") for g in markers}
        skipped = 0
        for g in list(markers):
            done_key = f"{BUILDINGS_PREFIX}/{g}/building.json"
            if store.exists(done_key) and store.read_json(done_key).get("fingerprint") == prints[g]:
                markers.pop(g)
                skipped += 1
        if not markers:
            return {"built": 0, "skipped": skipped, "failed": 0, "seconds": round(time.time() - t0, 1)}
        origins = [LocalCoordinateFrame.load_json(work / "buildings" / g / "coordinates.json").origin_source_xyz
                   for g in markers]
        # the LiDAR around the batch, once (a batch is one tile's: one phase)
        phase = next(iter(markers.values())).get("lidar_phase")
        xyz = scene_cloud_for(origins, index_for(_indexes(), phase))
        scene = None
        if xyz is not None:
            scene = work / "scene.npz"
            np.savez(scene, xyz_ft=xyz)
            del xyz
        built, failed = 0, 0
        for g in markers:
            b = work / "buildings" / g
            try:
                fp = prints[g]
                t = time.time()
                cut_crops_from_piece(b, log=lambda *a, **k: None)
                cmd = ["--building", b, "--name", markers[g].get("name") or g, "--refresh", "--complete",
                       "--neural", "--dump-textures", "--neighbours", nb_path, "--neighbour-levels", work / "buildings"]
                if scene is not None:
                    cmd += ["--scene-cloud", scene]
                r = run_main("build_mesh", cmd)
                (b / "log.txt").write_text(r.stdout, encoding="utf-8")
                if r.returncode != 0 or not (b / "3dtiles_mesh" / "tileset.json").exists():
                    raise RuntimeError("build_mesh: " + "\n".join(r.stdout.splitlines()[-6:]))
                if int(hashlib.sha1(g.encode()).hexdigest()[:8], 16) % 1000 < CHECK_SAMPLE * 1000:
                    run_main("check_building", ["--building", b, "--footprints", nb_path, "--neighbour-levels",
                                                work / "buildings"] + (["--scene-cloud", scene] if scene else []))
                run_main("audit_building", ["--building", b])
                ex = json.loads((b / "3dtiles_mesh" / "tileset.json").read_text(encoding="utf-8")).get("extras", {})
                summary = {"gers": g, "tile": tile, "fingerprint": fp, "run": run, "seconds": round(time.time() - t, 1),
                           "machine": machine_type(),                   # seconds by machine type (c8g vs c7g)
                           **{k: ex.get(k) for k in ("path", "triangles", "n_frames")}}
                publish_building(b, summary, store)                     # building.json last: the checkpoint
                built += 1
            except Exception as exc:                                    # noqa: BLE001
                _record_failure(store, run, g, f"build: {exc!r}")
                failed += 1
            finally:
                for sub in ("images", "validity_masks", "3dtiles_mesh", "3dtiles_complete", "3dtiles_neural"):
                    shutil.rmtree(b / sub, ignore_errors=True)
        return {"built": built, "skipped": skipped, "failed": failed, "seconds": round(time.time() - t0, 1)}
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------------------ publish

def _publish_step():
    from .stages import publish_step
    return publish_step


def publish_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """One tile's 3D Tiles for every view (see the module)."""
    ps = _publish_step()
    store, raw = os.environ["STORE"], os.environ.get("RAW_DIR", "/opt/data/raw")
    res = ps.one_tile(job["tile"], store, store, raw, job.get("products") or ps.PRODUCTS)
    return {"tile": job["tile"], **{s["product"]: {"meshed": s.get("meshed", 0), "boxed": s.get("boxed", 0)}
                                    for s in res}}


def tree_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """The statewide tree and the terrain over every published tile (see the module)."""
    from .pieces import phase2_grid
    from .publish import publish_trees

    ps = _publish_step()
    store = _store()
    grid = phase2_grid(Path(os.environ.get("RAW_DIR", "/opt/data/raw")))
    products = job.get("products") or ps.PRODUCTS
    trees = publish_trees(store, grid, products)                        # every published tile, one listing
    some = trees.get("measured") or next(iter(trees.values()))
    tiles = some["tile_names"]
    if job.get("terrain", True):
        ps.terrain(tiles, store, store, grid, int(job.get("terrain_max_level", 15)))
    return {"tiles": len(tiles), "products": list(products)}


# ------------------------------------------------------------------ repack

def repack_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Pieces stored the old way, a file each, packed as ``piece.zip`` and
    checked before their loose files go (``building.repack_piece``); the
    tile's ``measured.json`` written from them. ``{"kind": "repack", "tile":
    T, "gers": [...]}``."""
    from .building import publish_tile_measured, repack_piece

    store, run, tile = _store(), job.get("run", "run"), job["tile"]
    work = _work()
    try:
        from concurrent.futures import ThreadPoolExecutor

        def one(g):
            try:
                return g, repack_piece(g, store, work), None
            except Exception as exc:                                    # noqa: BLE001
                return g, None, exc
        folders, repacked, removed, failed = [], 0, 0, 0
        # the requests wait on the network: 16 pieces at a time (one job a tile, so
        # the tile's measured.json is written by one job only)
        with ThreadPoolExecutor(16) as ex:
            for g, r, exc in ex.map(one, job["gers"]):
                if exc is not None:
                    _record_failure(store, run, g, f"repack: {exc!r}")
                    failed += 1
                else:
                    folders.append(r["folder"]); repacked += r["repacked"]; removed += r["removed"]
        if folders:
            publish_tile_measured(tile, folders, store)
        return {"tile": tile, "pieces": len(folders), "repacked": repacked, "loose_removed": removed, "failed": failed}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """The worker's handler: a job by its kind."""
    return {"tile": tile_job, "measure": measure_job, "build": build_job, "publish": publish_job,
            "tree": tree_job, "repack": repack_job}[job["kind"]](job)
