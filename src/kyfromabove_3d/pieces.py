"""Stage 1: a LiDAR tile into per-building *pieces*.

A piece is everything the building step needs that comes from the LiDAR and
the frame indexes, without the pixels (:func:`prepare.prepare_piece`):
``coordinates.json``, ``footprint.json`` (with product 1's LiDAR height and,
for a footprint holding several buildings, its part), the LiDAR clip
``lidar_local.laz`` and ``clip.json``, ``check_features.json`` and the frames
chosen, ``frame_coverage.csv``. Its marker, ``piece.json``, is written last:
in S3 its arrival is the event that the next step listens for.

The unit of work is one Phase 2 LiDAR tile (46,167 statewide, each a COPC
file read by range requests). A building belongs to the tile holding its
footprint's centroid, so it is cut exactly once. Footprints of other tiles
within :data:`HALO_M` of this one are measured too (the halo), so a party
wall across a tile edge still has its neighbour and the neighbour's height.

Everything here works on local folders; publishing goes through a
:class:`~kyfromabove_3d.store.Store`, a folder or S3.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Bump to rebuild every piece: a new marker is written, and its event fires.
#: stage1-2: the hash reads values rounded, not bytes; frames.json added.
CODE_VERSION = "stage1-2"

#: Footprints of neighbouring tiles measured with this one.
HALO_M = 30.0

#: The files of a piece, besides its marker.
PIECE_FILES = ("coordinates.json", "footprint.json", "lidar_local.laz", "clip.json", "check_features.json",
               "frame_coverage.csv", "frames.json", "cameras_normalized.json")

GRID_CRS = "EPSG:3857"

#: What sources_choice records on every building: which LiDAR and which obliques.
PROVENANCE = ("lidar_phase", "lidar_date", "lidar_points_per_m2", "oblique_collection", "oblique_date",
              "gap_months")


# ------------------------------------------------------------------- tiles

def phase2_grid(raw_dir: Path):
    """The Phase 2 5k tile grid (tilename, key, aws_url, geometry), EPSG:3857."""
    from .lidar import LidarTileIndex

    return LidarTileIndex(raw_dir, phases=("phase2",)).grids["phase2"]


def _centroids_3857(recs: Sequence[Dict[str, Any]]):
    from pyproj import Transformer
    from shapely.wkt import loads as wkt_loads

    to_grid = Transformer.from_crs("EPSG:4326", GRID_CRS, always_xy=True).transform
    out = []
    for r in recs:
        g = wkt_loads(r["wkt"])
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda q: q.area)
        c = g.centroid
        out.append(to_grid(c.x, c.y))
    return np.array(out, dtype=float).reshape(-1, 2)


def owner_tiles(recs: Sequence[Dict[str, Any]], grid) -> List[Optional[str]]:
    """The tile each footprint belongs to: the one holding its centroid
    (None outside the grid). A centroid exactly on a shared edge goes to one
    tile only -- the first in the grid's order."""
    import geopandas as gpd

    pts = gpd.GeoDataFrame({"i": np.arange(len(recs))},
                           geometry=gpd.points_from_xy(*_centroids_3857(recs).T), crs=GRID_CRS)
    hit = gpd.sjoin(pts, grid[["tilename", "geometry"]].to_crs(GRID_CRS), predicate="intersects", how="left")
    hit = hit.sort_index(kind="stable").groupby(level=0).first()
    return [None if (isinstance(t, float) and np.isnan(t)) or t is None else str(t) for t in hit["tilename"]]


def records_by_tile(recs: Sequence[Dict[str, Any]], grid, halo_m: float = HALO_M) -> Dict[str, Dict[str, list]]:
    """{tilename: {"owned": [...], "halo": [...]}}: each footprint owned by
    one tile, and listed as halo by every other tile within ``halo_m`` of
    it (in EPSG:3857 metres, close enough to ground metres at this scale)."""
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.ops import transform
    from shapely.wkt import loads as wkt_loads

    owners = owner_tiles(recs, grid)
    to_grid = Transformer.from_crs("EPSG:4326", GRID_CRS, always_xy=True).transform
    geoms = gpd.GeoSeries([transform(to_grid, wkt_loads(r["wkt"])) for r in recs], crs=GRID_CRS)
    near = gpd.sjoin(gpd.GeoDataFrame({"i": np.arange(len(recs))}, geometry=geoms.buffer(halo_m), crs=GRID_CRS),
                     grid[["tilename", "geometry"]].to_crs(GRID_CRS), predicate="intersects")
    out: Dict[str, Dict[str, list]] = {}
    for i, t in enumerate(owners):
        if t is not None:
            out.setdefault(t, {"owned": [], "halo": []})["owned"].append(recs[i])
    for i, t in zip(near["i"], near["tilename"]):
        if owners[i] is not None and owners[i] != t:
            out.setdefault(str(t), {"owned": [], "halo": []})["halo"].append(recs[i])
    return out


def tile_origin_ft(grid, tilename: str) -> Tuple[float, float]:
    """The tile's centre in EPSG:3089 US survey feet: the local frame the
    tile's area cloud is measured in."""
    from pyproj import Transformer

    row = grid[grid["tilename"] == tilename]
    if not len(row):
        raise KeyError(f"tile {tilename} is not in the grid")
    c = row.to_crs(GRID_CRS).geometry.iloc[0].centroid
    x, y = Transformer.from_crs(GRID_CRS, "EPSG:3089", always_xy=True).transform(c.x, c.y)
    return float(x), float(y)


def tile_bounds_ft(grid, tilename: str, pad_ft: float = 0.0) -> Tuple[float, float, float, float]:
    from pyproj import Transformer
    from shapely.ops import transform

    g = grid[grid["tilename"] == tilename].to_crs(GRID_CRS).geometry.iloc[0]
    b = transform(Transformer.from_crs(GRID_CRS, "EPSG:3089", always_xy=True).transform, g).bounds
    return (b[0] - pad_ft, b[1] - pad_ft, b[2] + pad_ft, b[3] + pad_ft)


# ------------------------------------------------------------------ pieces

def _canonical(name: str, data: bytes) -> bytes:
    """A file's content as the hash sees it: numbers rounded (JSON to 3
    decimals -- a millimetre, a thousandth of a foot or pixel; the frame
    table's to 2), so the last-bit differences between Linux's and macOS's
    maths libraries (a local origin's 16th digit) do not read as a change."""
    import csv
    import io

    def rnd(v, d):
        if isinstance(v, float):
            return round(v, d)
        if isinstance(v, list):
            return [rnd(x, d) for x in v]
        if isinstance(v, dict):
            return {k: rnd(x, d) for k, x in v.items()}
        return v

    if name.endswith(".json"):
        try:
            return json.dumps(rnd(json.loads(data), 3), sort_keys=True).encode()
        except ValueError:
            return data
    if name.endswith(".csv"):
        out = []
        for row in csv.reader(io.StringIO(data.decode("utf-8"))):
            cells = []
            for c in row:
                try:
                    cells.append(repr(round(float(c), 2)) if "." in c or "e" in c.lower() else c)
                except ValueError:
                    cells.append(c)
            out.append(",".join(cells))
        return "\n".join(out).encode()
    return data


def inputs_hash(folder: Path) -> str:
    """What the piece was cut from, and how: its footprint and product 1
    record, the clip's box and origin, the frames chosen, the clip's point
    count, and :data:`CODE_VERSION`. (Not the LAZ bytes: its header carries
    the day it was written.)"""
    h = hashlib.sha256(CODE_VERSION.encode())
    for name in ("footprint.json", "clip.json", "check_features.json", "frame_coverage.csv", "frames.json",
                 "cameras_normalized.json"):
        p = Path(folder) / name
        h.update(name.encode())
        h.update(_canonical(name, p.read_bytes()) if p.exists() else b"-")
    laz = Path(folder) / "lidar_local.laz"
    if laz.exists():
        import laspy
        with laspy.open(laz) as f:
            h.update(str(f.header.point_count).encode())
    return h.hexdigest()


def tile_step(tilename: str, owned: Sequence[Dict[str, Any]], halo: Sequence[Dict[str, Any]], grid, index,
              log=print, provenance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Step 1 for one tile: Overture's outlines measured from the LiDAR
    around them (product 1) -- each one's ground and its height 1.5 m inside
    the edge; the halo's too -- and the outlines that hold two or three
    buildings cut into their parts. Returns {"buildings": the owned records,
    split parts in place of their parent; "halo": the neighbours from other
    tiles, measured; "splits": [...]; "stats": {...}}."""
    import time

    from .product1 import area_cloud_local, local_polygons_about, measure_heights, split_records

    t0 = time.time()
    owned = [dict(r) for r in owned]
    halo = [dict(r) for r in halo]
    stats = {"tile": tilename, "code_version": CODE_VERSION, "owned": len(owned), "halo": len(halo)}
    if not owned:
        return {"buildings": [], "halo": halo, "splits": [], "stats": {**stats, "returns": 0, "measured": 0,
                                                                        "buildings": 0, "split": 0}}
    ox, oy = tile_origin_ft(grid, tilename)
    recs = owned + halo
    polys = local_polygons_about(recs, ox, oy)
    xyz, cls = area_cloud_local(polys, ox, oy, index=index, log=log)
    t_lidar = time.time() - t0
    n_ok = measure_heights(recs, polys, xyz, cls)
    log(f"tile {tilename}: {len(owned)} footprints (+{len(halo)} halo), {n_ok} measured")
    parts, splits = split_records(owned, polys[:len(owned)], xyz, ox, oy, f"tile:{tilename}", log=log)
    if provenance:
        # which LiDAR and which obliques this tile's buildings are made from
        # (sources_choice); the halo's belong to their own tiles
        for r in parts:
            r.update(provenance)
        stats.update(provenance)
    stats.update({"returns": int(len(xyz)), "ground_returns": int((cls == 2).sum()), "measured": n_ok,
                  "buildings": len(parts), "split": len(splits), "origin_ft": [round(ox, 3), round(oy, 3)],
                  "seconds_lidar": round(t_lidar, 1), "seconds": round(time.time() - t0, 1)})
    return {"buildings": parts, "halo": halo, "splits": splits, "stats": stats}


def tile_ground_bytes(tilename: str, grid, index, stats: Dict[str, Any], log=print) -> Optional[bytes]:
    """The tile's ground grid for the viewer's terrain (``terrain.tile_ground``),
    as npz bytes; None, with a warning, if it cannot be made -- the terrain
    is not worth failing a tile's buildings for."""
    import time

    from .terrain import save_ground, tile_ground

    t = time.time()
    try:
        g = tile_ground(tilename, grid, index, log=log)
        stats["ground_cells"] = int(g["Z_navd88_m"].size)
        return save_ground(g)
    except Exception as exc:                                        # noqa: BLE001
        log(f"[warn] {tilename}: no ground grid ({exc!r})")
        stats["ground_error"] = repr(exc)[:300]
        return None
    finally:
        stats["seconds_ground"] = round(time.time() - t, 1)


def publish_tile_step(tilename: str, result: Dict[str, Any], frames_path: Optional[Path], store,
                      prefix: str = "tiles", ground: Optional[bytes] = None) -> None:
    """The tile step's results: ``tiles/<name>/buildings.json`` (product 1),
    ``frames.gpkg`` (the oblique frames over the tile: pointers to
    KyFromAbove's GeoTIFFs, with each camera's position and model),
    ``ground.npz`` (the ground under it, for the viewer's terrain) and
    ``summary.json`` last."""
    import datetime
    import time

    t = time.time()
    store.put_json(f"{prefix}/{tilename}/buildings.json",
                   {"tile": tilename, "code_version": CODE_VERSION, "buildings": result["buildings"],
                    "halo": result["halo"], "split": result["splits"]})
    stats = dict(result["stats"])
    if frames_path is not None and Path(frames_path).exists():
        store.put_file(f"{prefix}/{tilename}/frames.gpkg", Path(frames_path))
    if ground is not None:
        store.put_bytes(f"{prefix}/{tilename}/ground.npz", ground)
    stats["seconds_publish"] = round(time.time() - t, 1)
    stats["written_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    store.put_json(f"{prefix}/{tilename}/summary.json", stats)


def run_tile_step(tilename: str, inputs_store, out_store, grid, indexes: Dict[str, Any], work: Path,
                  log=print) -> Dict[str, Any]:
    """Step 1 for one tile, from its inputs to its published results -- the
    one way it runs, in the queue's tile job, the Lambda and
    ``scripts/tile_step.py``: the tile's Overture outlines
    (``inputs/footprints/tile=<name>.json``, from stage 0), the LiDAR phase
    and oblique collection taken closest together in time
    (``sources_choice``), the outlines measured and split (:func:`tile_step`),
    the ground grid, and ``tiles/<name>/`` written (:func:`publish_tile_step`,
    ``summary.json`` last). ``indexes``: one single-phase LiDAR index per
    phase. Returns the tile's stats."""
    import time

    from .sources_choice import choose_sources

    t = time.time()
    inputs = inputs_store.read_json(f"inputs/footprints/tile={tilename}.json")
    t_inputs = time.time() - t
    t = time.time()
    choice = choose_sources(tilename, grid, indexes, work, log=log)
    t_choice = time.time() - t
    index = indexes[choice["phase"]]
    res = tile_step(tilename, inputs.get("owned", []), inputs.get("halo", []), grid, index, log=log,
                    provenance=choice["provenance"])
    res["stats"].update({"frames": int(len(choice["frames"])), "sources_considered": choice["candidates"],
                         "seconds_inputs": round(t_inputs, 1), "seconds_choice": round(t_choice, 1)})
    ground = tile_ground_bytes(tilename, grid, index, res["stats"], log=log)
    publish_tile_step(tilename, res, choice["frames_path"], out_store, ground=ground)
    return res["stats"]


def batches(buildings: Sequence[Dict[str, Any]], size: int) -> List[List[str]]:
    """The buildings' ids in batches of ``size``, nearby ones together
    (ordered by their centres, in rows about 200 m tall), each building in
    exactly one batch."""
    from shapely.wkt import loads as wkt_loads

    def key(r):
        c = wkt_loads(r["wkt"]).centroid
        return (round(c.y / 0.002), c.x, r["id"])

    ids = [r["id"] for r in sorted(buildings, key=key)]
    return [ids[i:i + size] for i in range(0, len(ids), max(1, int(size)))]


def _plain(v):
    """A value from a frame's index record, as JSON can hold it."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    if hasattr(v, "isoformat"):
        # as str() writes it, which is what the camera keeps (cameras.parse_camera_from_eo_record):
        # the frame rebuilt from frames.json is the frame the laptop chose, to the last character
        return str(v)
    return v


def frames_record(chosen: Optional[Dict[str, Any]], frames_gdf) -> List[Dict[str, Any]]:
    """The frames chosen for a building, each self-contained: its GeoTIFF's
    URL in KyFromAbove's bucket, the camera's direction, the building's
    window on it in sensor pixels, and the frame's full index record (the
    camera's position and angles when it fired, its model: sensor size,
    focal length, pixel size) -- everything the building step needs to crop
    it, with no join against the tile's frame list."""
    if not chosen:
        return []
    table = chosen["table"]
    rows = frames_gdf.drop(columns="geometry").set_index("ID", drop=False)
    out = []
    for cam in chosen["cams"]:
        t = table[table["frame_id"] == cam.frame_id].iloc[0]
        eo = {k: _plain(v) for k, v in rows.loc[cam.frame_id].items()}
        out.append({"frame_id": cam.frame_id, "url": eo.get("S3URL"), "direction": eo.get("CameraID"),
                    "window_px": {k: round(float(t[k]), 1) for k in ("u_min", "u_max", "v_min", "v_max",
                                                                      "span_u_px", "span_v_px")},
                    "corners_on_sensor": int(t["corners_on_sensor"]), "margin_px": round(float(t["margin_px"]), 1),
                    "eo": eo})
    return out


def piece_step(tilename: str, gers: Sequence[str], buildings: Sequence[Dict[str, Any]],
               halo: Sequence[Dict[str, Any]], frames_gdf, index, work: Path, pad_ft: float = 90.0,
               max_per_head: int = 4, min_frames: int = 3, log=print) -> Dict[str, Any]:
    """Step 2 for a batch of one tile's buildings (``gers``, from the tile
    step's ``buildings``): each building's piece into ``work/<gers>/`` --
    its LiDAR clip at full resolution, the frames that see it (as pointers,
    ``frames.json``; no pixels), and its marker ``piece.json``. Returns
    {"pieces": [folders], "failed": {gers: reason}}."""
    from shapely.geometry import box
    from shapely.wkt import loads as wkt_loads

    from .prepare import prepare_piece
    from .sources_choice import index_for

    work = Path(work)
    by_id = {r["id"]: r for r in buildings}
    recs = [by_id[g] for g in gers if g in by_id]
    failed = {g: "not in the tile's buildings" for g in gers if g not in by_id}
    everyone = list(buildings) + list(halo)
    grounds = [r["lidar_ground_z"] for r in everyone if r.get("lidar_ground_z") is not None]
    fallback = float(np.median(grounds)) if grounds else None
    # the neighbours a party wall could be shared with: the records within
    # about 50 m of the batch, not the whole tile
    if recs:
        b = np.array([wkt_loads(r["wkt"]).bounds for r in recs])
        area = box(b[:, 0].min() - 6e-4, b[:, 1].min() - 5e-4, b[:, 2].max() + 6e-4, b[:, 3].max() + 5e-4)
        pool = [r for r in everyone if wkt_loads(r["wkt"]).intersects(area)]
    else:
        pool = []
    folders = []
    for rec in recs:
        folder = work / rec["id"]
        try:
            # the LiDAR phase the tile step chose (index: a single index, or
            # one per phase from sources_choice.indexes)
            result, chosen = prepare_piece(rec, folder, frames_gdf, index_for(index, rec.get("lidar_phase")), pool,
                                           pad_ft, max_per_head, min_frames, fallback, log=lambda *a, **k: None)
            (folder / "frames.json").write_text(json.dumps(frames_record(chosen, frames_gdf), indent=1),
                                                encoding="utf-8")
        except Exception as exc:                                    # noqa: BLE001
            failed[rec["id"]] = repr(exc)[:500]
            log(f"  {rec['id']}: no piece ({exc})")
            continue
        marker = {"gers": rec["id"], "tile": tilename, "code_version": CODE_VERSION,
                  "parent_gers": rec.get("parent_gers"), "name": rec.get("name"),
                  "lidar_height_m": rec.get("lidar_height_m"), "lidar_ground_z": rec.get("lidar_ground_z"),
                  **{k: result.get(k) for k in ("n_returns_aoi", "n_on_footprint", "n_frames", "heads")},
                  **{k: rec.get(k) for k in PROVENANCE if rec.get(k) is not None},
                  "inputs_hash": inputs_hash(folder)}
        (folder / "piece.json").write_text(json.dumps(marker, indent=1, default=float), encoding="utf-8")
        folders.append(folder)
    return {"pieces": folders, "failed": failed}


def make_tile_pieces(tilename: str, owned: Sequence[Dict[str, Any]], halo: Sequence[Dict[str, Any]], work: Path,
                     grid, index, frames_gdf, pad_ft: float = 90.0, max_per_head: int = 4, min_frames: int = 3,
                     log=print) -> Dict[str, Any]:
    """Cut one tile's pieces into ``work/<gers>/`` (each with its marker
    ``piece.json``): :func:`tile_step`, then each building's piece. Returns
    {"pieces": [folders], "records": product 1 records of the owned
    footprints (split parts in place of their parent), "splits": [...]}."""
    step = tile_step(tilename, owned, halo, grid, index, log=log)
    parts, halo, splits = step["buildings"], step["halo"], step["splits"]
    if not parts:
        return {"pieces": [], "records": [], "splits": []}
    made = piece_step(tilename, [r["id"] for r in parts], parts, halo, frames_gdf, index, work, pad_ft,
                      max_per_head, min_frames, log=log)
    return {"pieces": made["pieces"], "records": parts, "splits": splits}


def publish_piece(folder: Path, store, prefix: str = "pieces") -> bool:
    """Write a piece's files to ``<prefix>/<gers>/`` and its marker
    ``piece.json`` last. A piece whose marker in the store already has the
    same ``inputs_hash`` is left alone -- no write, so no event. Returns
    whether it was written."""
    folder = Path(folder)
    marker = json.loads((folder / "piece.json").read_text(encoding="utf-8"))
    key = f"{prefix}/{marker['gers']}/piece.json"
    if store.exists(key):
        try:
            if store.read_json(key).get("inputs_hash") == marker["inputs_hash"]:
                return False
        except (ValueError, KeyError):
            pass
    for name in PIECE_FILES:
        p = folder / name
        if p.exists():
            store.put_file(f"{prefix}/{marker['gers']}/{name}", p)
    store.put_json(key, marker)                                     # last: its arrival is the event
    return True


def publish_tile(tilename: str, made: Dict[str, Any], written: Iterable[str], store,
                 prefix: str = "pieces", products: str = "products/footprints") -> None:
    """The tile's manifest (the pieces it owns, which were rewritten) and its
    product 1 records."""
    store.put_json(f"{prefix}/_tiles/{tilename}.json",
                   {"tile": tilename, "code_version": CODE_VERSION,
                    "pieces": [Path(f).name for f in made["pieces"]], "written": sorted(written)})
    store.put_json(f"{products}/tile={tilename}.json", {"tile": tilename, "buildings": made["records"],
                                                         "split": made["splits"]})


def run_tile(tilename: str, tile_inputs: Dict[str, list], work: Path, store, grid, index, frames_gdf,
             log=print) -> Dict[str, Any]:
    """One tile, end to end: cut its pieces, publish them (unchanged ones
    skipped), publish the tile's manifest and product 1 records."""
    made = make_tile_pieces(tilename, tile_inputs.get("owned", []), tile_inputs.get("halo", []), work, grid, index,
                            frames_gdf, log=log)
    written = [Path(f).name for f in made["pieces"] if publish_piece(f, store)]
    publish_tile(tilename, made, written, store)
    log(f"tile {tilename}: {len(made['pieces'])} pieces, {len(written)} written, "
        f"{len(made['pieces']) - len(written)} unchanged")
    return {"tile": tilename, "pieces": len(made["pieces"]), "written": len(written)}
