"""Step 3, the building step: a piece in, a building's meshes out.

A piece (``pieces/<gers>/``, written by step 2) holds everything about a
building that comes from the LiDAR and the frame indexes, without pixels.
The building step turns it into the products, as ``run_downtown.py``'s
worker does on the laptop:

1. **crops** -- each frame in ``frames.json`` cut to the building by range
   reads of KyFromAbove's GeoTIFF (the camera rebuilt from the frame's own
   index record), into the working folder only, never to the bucket;
2. **colourise, mesh and the completed product** -- ``build_mesh.py
   --complete`` (which runs ``colorize_building.py``), occluded by the LiDAR
   within 250 m (read once for a batch of nearby buildings) and by the
   neighbours as prisms (product 1's outlines at their LiDAR heights, from
   the tile's ``buildings.json`` and its neighbouring tiles');
3. **the texture check** and **the audit** (``check_building.py``,
   ``audit_building.py``);
4. the results to ``buildings/<version>/<gers>/`` as five objects
   (:func:`pack_building`: a zip per mesh product, ``review.zip``,
   ``levels.json``) and ``building.json`` last: the piece it was built from
   (its ``inputs_hash``), the seconds and peak memory of each stage, the
   size of each file. A rebuild reads none of them: it starts again from
   the piece.

Each script runs in its own process, as on the laptop (GDAL and curl are not
fork-safe, and a crashed colourise must not take the batch with it); each
process's own peak memory is read back from the kernel.
"""

from __future__ import annotations

import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .coordinates import METERS_PER_US_SURVEY_FOOT, LocalCoordinateFrame
from .pieces import PIECE_FILES

#: Bump when the building step's method changes. Each version's buildings
#: are kept in their own folder, ``buildings/<version>/<gers>/``, so a new
#: method's results sit beside the old ones (``building.json`` records it).
BUILDING_CODE_VERSION = "stage3-2"          # 3-2: lossless texture pages kept, for step 4

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

SCRIPTS = Path(os.environ.get("SCRIPTS_DIR", Path(__file__).resolve().parents[2] / "scripts"))


# ------------------------------------------------------------------ the piece

def pull_piece(gers: str, store, folder: Path, prefix: str = "pieces") -> Dict[str, Any]:
    """The piece's files into ``folder``; returns its marker."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    marker = store.read_json(f"{prefix}/{gers}/piece.json")
    for name in PIECE_FILES:
        key = f"{prefix}/{gers}/{name}"
        if store.exists(key):
            store.get_file(key, folder / name)
    (folder / "piece.json").write_text(json.dumps(marker, indent=1), encoding="utf-8")
    return marker


def chosen_from_piece(folder: Path) -> Optional[Dict[str, Any]]:
    """What :func:`prepare.cut_crops` needs, rebuilt from the piece alone:
    each chosen frame's camera from its own index record (``frames.json``),
    the building's volume (``check_features.json``), its local frame
    (``coordinates.json``) and the coverage table (``frame_coverage.csv``).
    None when no frame sees the building."""
    import pandas as pd

    from .cameras import parse_camera_from_eo_record
    from .crops import BoundingVolume3D

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

def _peak_mb_self() -> float:
    """This process's peak resident memory so far, in MB (the kernel reports
    kilobytes on Linux, bytes on macOS)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1 << 20 if sys.platform == "darwin" else 1 << 10), 1)


def run_script(cmd: Sequence[str], log_path: Path, timeout_s: float) -> Tuple[int, float, float, str]:
    """``cmd`` in its own process, its output appended to ``log_path``.
    Returns (exit code, seconds, the peak memory in MB of that process and
    the processes it waited on -- build_mesh waits on the colourise -- and
    its output)."""
    t = time.time()
    log_path = Path(log_path)
    part = log_path.with_suffix(".part")
    with open(part, "w", encoding="utf-8") as fh:
        p = subprocess.Popen([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, cwd=str(SCRIPTS.parent))
        status, usage = 0, None
        while True:
            pid, status, usage = os.wait4(p.pid, os.WNOHANG)
            if pid:
                break
            if time.time() - t > timeout_s:
                p.kill()
                _, status, usage = os.wait4(p.pid, 0)
                fh.write(f"\n[FAIL] timed out after {timeout_s:.0f} s\n")
                break
            time.sleep(0.2)
        p.returncode = os.waitstatus_to_exitcode(status)
    out = part.read_text(encoding="utf-8", errors="replace")
    part.unlink(missing_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(str(c) for c in cmd)}\n{out}")
    peak_mb = round(usage.ru_maxrss / (1 << 20 if sys.platform == "darwin" else 1 << 10), 1)
    return p.returncode, round(time.time() - t, 1), peak_mb, out


def _stage_times(out: str) -> Dict[str, float]:
    """The ``[time] <stage> <s> s`` lines build_mesh prints."""
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"^\[time\] (\w+) ([\d.]+) s", out, re.M)}


def build_one(gers: str, store, work: Path, neighbours_path: Path, scene_path: Optional[Path],
              timeout_s: float = 840.0, check: bool = True, audit: bool = True,
              log: Callable = print) -> Dict[str, Any]:
    """One building from its piece (already pulled into ``work/<gers>``):
    crops, the meshes, the check, the audit. Returns its summary (what
    ``building.json`` holds). Raises when there is no mesh."""
    from .prepare import cut_crops

    b = Path(work) / gers
    marker = json.loads((b / "piece.json").read_text(encoding="utf-8"))
    log_path = b / "log.txt"
    from .pieces import PROVENANCE

    s: Dict[str, Any] = {"gers": gers, "tile": marker.get("tile"), "name": marker.get("name"),
                         "parent_gers": marker.get("parent_gers"), "piece_hash": marker.get("inputs_hash"),
                         "code_version": BUILDING_CODE_VERSION, "seconds": {}, "peak_mb": {},
                         **{k: marker.get(k) for k in PROVENANCE if marker.get(k) is not None}}
    sec, peak = s["seconds"], s["peak_mb"]

    t = time.time()
    chosen = chosen_from_piece(b)
    res: Dict[str, Any] = {}
    if chosen is not None:
        cut_crops(chosen, b, res, log=lambda m: log(f"  {gers[:8]} {m.strip()}"))
    sec["crops"] = round(time.time() - t, 1)
    peak["crops"] = _peak_mb_self()
    s.update({"n_frames": 0 if chosen is None else len(chosen["cams"]), "crop_px": res.get("crop_px", 0)})

    py = sys.executable
    common = ["--neighbours", neighbours_path, "--neighbour-levels", "none"]
    if scene_path is not None:
        common += ["--scene-cloud", scene_path]
    rc, t_s, pk, out = run_script([py, SCRIPTS / "build_mesh.py", "--building", b, "--name", marker.get("name") or gers,
                                   "--refresh", "--complete", "--dump-textures", *common], log_path, timeout_s)
    times = _stage_times(out)
    sec["colourise"] = times.get("colourise", 0.0)
    sec["mesh"] = round(times.get("mesh", t_s) - times.get("colourise", 0.0), 1)
    sec["complete"] = times.get("complete", 0.0)
    sec["build_mesh_process"] = t_s
    peak["build_mesh"] = pk
    if rc != 0 or not (b / "3dtiles_mesh" / "tileset.json").exists():
        tail = "\n".join(out.splitlines()[-6:])
        raise RuntimeError(f"build_mesh exit {rc}: {tail}")
    ex = json.loads((b / "3dtiles_mesh" / "tileset.json").read_text(encoding="utf-8")).get("extras", {})
    s.update({k: ex.get(k) for k in ("path", "triangles", "archetype", "faces")})

    if check and s["n_frames"]:
        rc, sec["check"], peak["check"], out = run_script(
            [py, SCRIPTS / "check_building.py", "--building", b, "--footprints", neighbours_path,
             "--neighbour-levels", "none", *(["--scene-cloud", scene_path] if scene_path else [])],
            log_path, timeout_s)
        if rc == 0 and (b / "texture_check.json").exists():
            s["check"] = json.loads((b / "texture_check.json").read_text(encoding="utf-8")).get("building")
        else:
            s["check_error"] = "\n".join(out.splitlines()[-4:])
    if audit:
        rc, sec["audit"], peak["audit"], out = run_script(
            [py, SCRIPTS / "audit_building.py", "--building", b], log_path, timeout_s)
        if rc == 0 and (b / "audit.json").exists():
            a = json.loads((b / "audit.json").read_text(encoding="utf-8"))
            h, tx = a.get("holes") or {}, a.get("texture") or {}
            s["audit"] = {"hole_m2": h.get("total_m2"), "roof_wall_m2": (a.get("roof_walls") or {}).get("m2"),
                          "mesh_minus_lidar_m": (a.get("heights") or {}).get("mesh_minus_lidar_m"),
                          "blank_share": tx.get("blank_share")}
        else:
            s["audit_error"] = "\n".join(out.splitlines()[-4:])
    return s


def buildings_prefix(version: str = BUILDING_CODE_VERSION) -> str:
    """Where one version of the building step keeps its buildings."""
    return f"buildings/{version}"


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
    """The building's outputs to ``<prefix>/<gers>/`` (default: this
    version's folder) as a few objects (:func:`pack_building`),
    ``building.json`` last; the summary gains each file's size and each
    object's."""
    prefix = prefix or buildings_prefix()
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


def up_to_date(gers: str, marker: Dict[str, Any], store, prefix: Optional[str] = None) -> bool:
    """Whether ``<prefix>/<gers>/building.json`` (default: this version's
    folder) was built from this very piece by this code."""
    key = f"{prefix or buildings_prefix()}/{gers}/building.json"
    if not store.exists(key):
        return False
    try:
        done = store.read_json(key)
    except ValueError:
        return False
    return (done.get("piece_hash") == marker.get("inputs_hash")
            and done.get("code_version") == BUILDING_CODE_VERSION)


def building_step(gers: Sequence[str], store, work: Path, index, tiles_prefix: str = "tiles",
                  pieces_prefix: str = "pieces", out_prefix: Optional[str] = None, force: bool = False,
                  timeout_s: float = 840.0, check: bool = True, audit: bool = True, publish: bool = True,
                  keep: bool = False, out_store=None, log: Callable = print) -> Dict[str, Any]:
    """Step 3 for a batch of nearby buildings (one tile's, from the piece
    step's batches): pull each piece, read the LiDAR around them all once,
    build each, publish each (to ``out_store``, default ``store``). A
    building already built from the same piece by the same code is skipped. Returns {"built": [summaries], "skipped":
    [gers], "failed": {gers: reason}, "seconds": {...}}."""
    work = Path(work).resolve()                 # the scripts run from the repository root
    work.mkdir(parents=True, exist_ok=True)
    out_store = out_store or store
    out_prefix = out_prefix or buildings_prefix()
    t0 = time.time()
    built, skipped, failed = [], [], {}
    markers = {}
    for g in gers:
        try:
            m = pull_piece(g, store, work / g, pieces_prefix)
        except Exception as exc:                                    # noqa: BLE001
            failed[g] = f"no piece: {exc!r}"[:500]
            continue
        if not force and up_to_date(g, m, out_store, out_prefix):
            skipped.append(g)
            shutil.rmtree(work / g, ignore_errors=True)
            continue
        markers[g] = m
    t_pull = time.time() - t0

    t = time.time()
    by_tile: Dict[str, Path] = {}
    for g, m in markers.items():
        tile = m.get("tile")
        if tile not in by_tile:
            path = work / f"neighbours_{tile}.json"
            path.write_text(json.dumps({"buildings": neighbour_records(tile, store, tiles_prefix)}),
                            encoding="utf-8")
            by_tile[tile] = path
    t_neighbours = time.time() - t

    t = time.time()
    scene_path = None
    if markers:
        from .sources_choice import index_for

        origins = [LocalCoordinateFrame.load_json(work / g / "coordinates.json").origin_source_xyz for g in markers]
        # a batch is one tile's buildings: one phase (pieces from before the choice: Phase 2)
        phase = next(iter(markers.values())).get("lidar_phase")
        xyz = scene_cloud_for(origins, index_for(index, phase), log=log)
        if xyz is not None:
            scene_path = work / "scene.npz"
            np.savez(scene_path, xyz_ft=xyz)
            del xyz
    t_scene = time.time() - t
    peak_scene = _peak_mb_self()

    for g, m in markers.items():
        try:
            s = build_one(g, store, work, by_tile[m.get("tile")], scene_path, timeout_s, check, audit, log)
            s["seconds"]["batch_scene_share"] = round(t_scene / max(len(markers), 1), 1)
            if publish:
                t = time.time()
                publish_building(work / g, s, out_store, out_prefix)
                s["seconds"]["publish"] = round(time.time() - t, 1)
            built.append(s)
            log(f"  {g}: {s.get('path')}, {s.get('triangles')} triangles, "
                + ", ".join(f"{k} {v:.0f}s" for k, v in s["seconds"].items()))
        except Exception as exc:                                    # noqa: BLE001
            failed[g] = repr(exc)[:1000]
            log(f"  {g}: FAILED {exc}")
        finally:
            if not keep:
                shutil.rmtree(work / g, ignore_errors=True)
    if scene_path is not None and not keep:
        scene_path.unlink(missing_ok=True)
    return {"built": built, "skipped": skipped, "failed": failed,
            "seconds": {"pull": round(t_pull, 1), "neighbours": round(t_neighbours, 1), "scene": round(t_scene, 1),
                        "total": round(time.time() - t0, 1)},
            "peak_mb_scene": peak_scene}
