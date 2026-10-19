"""A building in the cloud: its piece fetched, its neighbours, the LiDAR
around a batch of buildings, and its results packed and published.

A piece (``pieces/<gers>/``, written by the measure job) holds everything
about a building that comes from the LiDAR and the frame indexes, without
pixels. The queue run's build job (``kyfromabove_3d.cloud_jobs``) builds it
by the one pipeline (``kyfromabove_3d.pipeline``) and publishes the results
to ``buildings/<gers>/`` (:data:`BUILDINGS_PREFIX`) as five objects
(:func:`pack_building`: a zip per mesh product, ``review.zip``,
``levels.json``) and ``building.json`` last, which holds the fingerprint a
later run skips by.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .coordinates import METERS_PER_US_SURVEY_FOOT, LocalCoordinateFrame
from .pieces import PIECE_FILES

#: The LiDAR within this of a building occludes its frames (the colourise's default).
SCENE_RADIUS_M = 250.0

#: What is published per building; the crops, the clip and the coloured
#: cloud stay behind (the piece holds the clip; the crops are cut again).
OUTPUT_FILES = ("audit.json", "audit.png", "texture_check.json", "classification.json", "levels.json",
                "wall_planes.json", "wall_profiles.json", "mirror_report.json", "colorized_aerial_coverage.json")
OUTPUT_DIRS = ("3dtiles_mesh", "3dtiles_complete")
#: Kept as its own object: small, and a neighbour will read it one by one.
LOOSE_FILES = ("levels.json",)
#: Everything else that is not a mesh product's tileset, glb or texture pages
#: (the reports, the tinted copy, the fill test's data, the log) goes here.
REVIEW_ZIP = "review.zip"
#: What ``building.json`` says about how the building is stored.
LAYOUT = "zips-1"
#: Where the buildings are kept: one folder per building, ``buildings/<gers>/``
#: (each build overwrites the last; the fingerprint says what it was built from).
BUILDINGS_PREFIX = "buildings"

# ------------------------------------------------------------------ the piece

#: What the measure phase adds to a piece.
MEASURED_FILES = ("building_class.json", "footprint_lidar.json", "levels_measured.json")
#: A piece stored as one zip (``piece.zip``) beside its marker: two uploads
#: and two downloads a building, not a dozen of each and an existence check
#: per file. Its marker says so (``layout``); a piece without it is the old
#: layout, a file each, and is still read.
PIECE_ZIP = "piece.zip"
PIECE_LAYOUT = "zip-1"
#: Each tile's buildings' local frames and measured levels, in one object
#: (``tiles/<tile>/measured.json``): a build batch reads its tile's and its
#: neighbours' instead of two small objects per neighbour -- 300 to 600
#: neighbours within reach of a suburban batch.
TILE_MEASURED = "measured.json"
NEIGHBOUR_FILES = ("coordinates.json", "levels_measured.json")


def publish_piece(folder: Path, store, prefix: str = "pieces") -> Dict[str, Any]:
    """A measured piece to ``<prefix>/<gers>/``: its files (the piece's and
    what the measure phase added) as ``piece.zip``, then its marker. Returns
    the marker as stored."""
    from .archive import zip_bytes

    folder = Path(folder)
    marker = json.loads((folder / "piece.json").read_text(encoding="utf-8"))
    members = {n: folder / n for n in PIECE_FILES + MEASURED_FILES if (folder / n).exists()}
    store.put_bytes(f"{prefix}/{marker['gers']}/{PIECE_ZIP}", zip_bytes(members), "application/zip")
    stored = {**marker, "layout": PIECE_LAYOUT}
    store.put_json(f"{prefix}/{marker['gers']}/piece.json", stored)               # last
    return stored


def pull_piece(gers: str, store, folder: Path, prefix: str = "pieces") -> Dict[str, Any]:
    """The piece's files, and what the measure phase added, into ``folder``
    (either layout); returns its marker. The folder's ``piece.json`` is the
    marker as the piece wrote it, without the storage's ``layout``."""
    from .archive import read_zip

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    marker = store.read_json(f"{prefix}/{gers}/piece.json")
    if marker.pop("layout", None) == PIECE_LAYOUT:
        z = read_zip(store, f"{prefix}/{gers}/{PIECE_ZIP}")
        for name in z.namelist():
            (folder / name).write_bytes(z.read(name))
    else:
        for name in PIECE_FILES + MEASURED_FILES:
            key = f"{prefix}/{gers}/{name}"
            if store.exists(key):
                store.get_file(key, folder / name)
    (folder / "piece.json").write_text(json.dumps(marker, indent=1), encoding="utf-8")
    return marker


def repack_piece(gers: str, store, work: Path, prefix: str = "pieces") -> Dict[str, Any]:
    """A piece stored the old way (a file each) as ``piece.zip`` and its
    marker; read back and checked file by file against what was there; and
    only then its loose files removed. A piece already packed is checked and
    any loose files left beside it removed, so a run cut short can be run
    again. Returns {"folder", "repacked", "removed"}; raises if the check
    fails, before anything is removed."""
    work = Path(work)
    marker = store.read_json(f"{prefix}/{gers}/piece.json")
    folder = work / gers
    pull_piece(gers, store, folder)                     # either layout
    repacked = marker.get("layout") != PIECE_LAYOUT
    if repacked:
        publish_piece(folder, store)
    check = work / f"{gers}.check"
    pull_piece(gers, store, check)                      # from the zip now
    a = {p.name: p.read_bytes() for p in folder.iterdir()}
    b = {p.name: p.read_bytes() for p in check.iterdir()}
    if a != b:
        raise ValueError(f"{gers}: the zip does not give back the piece's files ({sorted(set(a) ^ set(b))})")
    removed = 0
    for name in PIECE_FILES + MEASURED_FILES:
        key = f"{prefix}/{gers}/{name}"
        if store.exists(key):
            store.delete(key)
            removed += 1
    import shutil
    shutil.rmtree(check, ignore_errors=True)
    return {"folder": folder, "repacked": repacked, "removed": removed}


def publish_tile_measured(tile: str, folders: Sequence[Path], store, prefix: str = "tiles") -> int:
    """The measured buildings' local frames and levels into the tile's
    ``measured.json``, merged with what it holds (a job that re-measures some
    of a tile's buildings leaves the others). Each file is kept as its text,
    so a building's neighbours write them back byte for byte. Returns how
    many buildings it holds."""
    key = f"{prefix}/{tile}/{TILE_MEASURED}"
    have = store.read_json(key)["buildings"] if store.exists(key) else {}
    for f in folders:
        f = Path(f)
        have[f.name] = {n: (f / n).read_text(encoding="utf-8") for n in NEIGHBOUR_FILES if (f / n).exists()}
    store.put_json(key, {"tile": tile, "buildings": have})
    return len(have)


def pull_neighbours(ids: Sequence[str], tiles: Sequence[str], store, buildings_dir: Path,
                    prefix: str = "tiles", pieces_prefix: str = "pieces") -> Dict[str, int]:
    """The neighbours' local frames and measured levels into
    ``buildings_dir/<gers>/``: from the tiles' ``measured.json`` where they
    have one, else from each neighbour's piece (the old layout). Returns the
    counts by source."""
    want, n = set(ids), {"tile_files": 0, "from_tiles": 0, "files_from_pieces": 0}
    for t in tiles:
        key = f"{prefix}/{t}/{TILE_MEASURED}"
        if not store.exists(key):
            continue
        n["tile_files"] += 1
        for g, files in store.read_json(key)["buildings"].items():
            if g in want:
                d = Path(buildings_dir) / g
                d.mkdir(parents=True, exist_ok=True)
                for name, text in files.items():
                    (d / name).write_text(text, encoding="utf-8")
                want.discard(g)
                n["from_tiles"] += 1
    for g in sorted(want):
        d = Path(buildings_dir) / g
        for name in NEIGHBOUR_FILES:
            key = f"{pieces_prefix}/{g}/{name}"
            if store.exists(key):
                d.mkdir(parents=True, exist_ok=True)
                store.get_file(key, d / name)
                n["files_from_pieces"] += 1
    return n


def chosen_from_piece(folder: Path) -> Optional[Dict[str, Any]]:
    """What :func:`prepare.cut_crops` needs, rebuilt from the piece alone:
    each chosen frame's camera from its own index record (``frames.json``),
    the building's volume (``check_features.json``), its local frame
    (``coordinates.json``) and the coverage table (``frame_coverage.csv``),
    the chosen frames' rows made again from their cameras as
    :func:`prepare.select_cameras` made them: the CSV rounds their last digits,
    and the crops' windows come from them. None when no frame sees the building."""
    import pandas as pd

    from .cameras import parse_camera_from_eo_record
    from .crops import BoundingVolume3D
    from .sources import assess_frame_coverage

    folder = Path(folder)
    frames = json.loads((folder / "frames.json").read_text(encoding="utf-8")) \
        if (folder / "frames.json").exists() else []
    if not frames:
        return None
    frame = LocalCoordinateFrame.load_json(folder / "coordinates.json")
    volume = BoundingVolume3D(**json.loads((folder / "check_features.json").read_text(encoding="utf-8"))
                              ["volume_local_m"])
    cams = [parse_camera_from_eo_record(f["eo"], frame) for f in frames]
    table = pd.read_csv(folder / "frame_coverage.csv")
    again = assess_frame_coverage(cams, volume).set_index("frame_id")
    rows = table["frame_id"].isin(again.index)
    for col in [c for c in again.columns if c in table.columns]:
        table[col] = table[col].astype(again[col].dtype) if table[col].dtype != again[col].dtype else table[col]
        table.loc[rows, col] = again.loc[table.loc[rows, "frame_id"], col].to_numpy()
    return {"cams": cams, "volume": volume, "frame": frame, "table": table}


# -------------------------------------------------------------- the neighbours

def adjacent_tiles(tile: str) -> List[str]:
    """The eight tiles around ``tile`` (``N076E228`` style names)."""
    m = re.fullmatch(r"N(\d+)E(\d+)", tile)
    if not m:
        return []
    n, e, wn, we = int(m.group(1)), int(m.group(2)), len(m.group(1)), len(m.group(2))
    return [f"N{n + dn:0{wn}d}E{e + de:0{we}d}" for dn in (-1, 0, 1) for de in (-1, 0, 1)
            if (dn, de) != (0, 0) and n + dn >= 0 and e + de >= 0]


def neighbour_records(tile: str, store, prefix: str = "tiles") -> List[Dict[str, Any]]:
    """Every building the prisms can come from, once: the tile's own
    (``buildings``) and each measured neighbouring tile's; a neighbouring
    tile not measured yet contributes only what the tile measured of it
    (its ``halo``, 30 m deep). A halo footprint that its own tile split is
    left out when the parts are there (the parent would stand over them)."""
    tiles = [tile] + adjacent_tiles(tile)
    seen: Dict[str, Dict[str, Any]] = {}
    halo: List[Dict[str, Any]] = []
    for t in tiles:
        key = f"{prefix}/{t}/buildings.json"
        if not store.exists(key):
            continue
        d = store.read_json(key)
        for r in d.get("buildings", []):
            seen.setdefault(r["id"], r)
        if t == tile:
            halo = d.get("halo", [])
    parents = {r.get("parent_gers") for r in seen.values() if r.get("parent_gers")}
    for r in halo:
        if r["id"] not in seen and r["id"] not in parents:
            seen[r["id"]] = r
    keep = ("id", "wkt", "name", "lidar_ground_z", "lidar_height_m", "height", "parent_gers")
    return [{k: r.get(k) for k in keep} for r in seen.values()]


# ------------------------------------------------------------ the scene cloud

def scene_cloud_for(origins_ft: Sequence[Tuple[float, float, float]], index, radius_m: float = SCENE_RADIUS_M,
                    log: Callable = print) -> Optional[np.ndarray]:
    """The LiDAR (not noise, class 7) within ``radius_m`` of any of the
    buildings' origins, in source feet: read once for a batch of nearby
    buildings; each script windows it to its own building."""
    import laspy

    from .lidar import copc_query

    o = np.asarray(origins_ft, dtype=float)
    pad = radius_m / METERS_PER_US_SURVEY_FOOT
    bounds = (o[:, 0].min() - pad, o[:, 1].min() - pad, o[:, 0].max() + pad, o[:, 1].max() + pad)
    out = []
    for t in index.tiles_for_bounds(bounds):                       # a single-phase index
        pts = copc_query(t.url, laspy.copc.Bounds(mins=np.array([bounds[0], bounds[1], -1e5]),
                                                  maxs=np.array([bounds[2], bounds[3], 1e5])))
        cls = np.asarray(pts.classification)
        out.append(np.stack([pts.x, pts.y, pts.z], axis=1)[cls != 7])
    if not out:
        return None
    xyz = np.vstack(out)
    log(f"  scene: {len(xyz):,} returns within {radius_m:.0f} m of {len(o)} building(s)")
    return xyz


# ------------------------------------------------------------------ the steps

def output_files(b: Path) -> List[Path]:
    """The files of building folder ``b`` that are published."""
    b = Path(b)
    out = [b / n for n in OUTPUT_FILES if (b / n).exists()]
    for d in OUTPUT_DIRS:
        if (b / d).is_dir():
            out += sorted(p for p in (b / d).rglob("*") if p.is_file())
    if (b / "log.txt").exists():
        out.append(b / "log.txt")
    return out


def pack_building(b: Path) -> Tuple[Dict[str, bytes], Dict[str, int]]:
    """Building folder ``b``'s published files as the few objects stored for
    it, and each file's size: ``<product>.zip`` per mesh product (its
    tileset, the glb it names and its texture pages: what step 4 reads),
    ``levels.json`` as it is, and ``review.zip`` for the rest."""
    from .archive import zip_bytes

    b = Path(b)
    files = output_files(b)
    groups: Dict[str, Dict[str, Path]] = {}
    for d in OUTPUT_DIRS:
        ts = b / d / "tileset.json"
        uri = (json.loads(ts.read_text(encoding="utf-8")).get("root", {}).get("content", {}).get("uri")
               if ts.exists() else None)
        groups[d] = {}
        for p in files:
            rel = p.relative_to(b).as_posix()
            if rel.startswith(f"{d}/"):
                inner = rel[len(d) + 1:]
                if inner in ("tileset.json", uri) or inner.startswith("textures/"):
                    groups[d][inner] = p
    in_products = {p for g in groups.values() for p in g.values()}
    review = {p.relative_to(b).as_posix(): p for p in files
              if p not in in_products and p.relative_to(b).as_posix() not in LOOSE_FILES}
    objects: Dict[str, bytes] = {f"{d}.zip": zip_bytes(g) for d, g in groups.items() if g}
    if review:
        objects[REVIEW_ZIP] = zip_bytes(review)
    for name in LOOSE_FILES:
        if (b / name).exists():
            objects[name] = (b / name).read_bytes()
    return objects, {p.relative_to(b).as_posix(): p.stat().st_size for p in files}


def publish_building(b: Path, summary: Dict[str, Any], store, prefix: Optional[str] = None) -> Dict[str, Any]:
    """The building's outputs to ``<prefix>/<gers>/`` (default:
    :data:`BUILDINGS_PREFIX`) as a few objects (:func:`pack_building`),
    ``building.json`` last; the summary gains each file's size and each
    object's."""
    prefix = prefix or BUILDINGS_PREFIX
    objects, files = pack_building(b)
    summary["files"] = files
    summary["bytes"] = int(sum(files.values()))
    summary["objects"] = {k: len(v) for k, v in sorted(objects.items())}
    summary["layout"] = LAYOUT
    for name in sorted(objects):
        store.put_bytes(f"{prefix}/{summary['gers']}/{name}", objects[name],
                        "application/zip" if name.endswith(".zip") else "application/json")
    store.put_json(f"{prefix}/{summary['gers']}/building.json", summary)
    return summary
