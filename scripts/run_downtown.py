#!/usr/bin/env python3
"""Run the whole downtown: every Overture footprint through prepare,
colourise and mesh, gathered into one tileset.

One subprocess per building (GDAL and curl are not fork-safe, and a tower's
crops are a few hundred megabytes decoded), a few at a time, with the
things every building would otherwise repeat done once up front:

* ``coordinates.json`` -- the downtown frame the merged tileset is placed in;
* ``frames_2024s1.gpkg`` -- the aerial frame index over the whole area;
* ``scene_phase2.npz`` -- the Phase 2 returns over the area plus 300 m, the
  occluder cloud every colourise windows instead of range-reading.

Each building's state lives in its own directory (``state.json``, ``log.txt``)
and the run's manifest is rebuilt from those, so a run can be stopped and
resumed and a lost manifest costs nothing. A building that fails is recorded
with the tail of its log and the run goes on; one that the colourise
declines (no camera sees it, too few returns) takes the box path and is not
a failure. After the mesh, the texture check (``check_building.py``) scores it
against the frames it was not textured from, while the crops are still on
disk. Crops are deleted once the mesh is built unless ``--keep-crops``.

Usage::

    python scripts/run_downtown.py run --footprints data/derived/louisville_downtown_overture_split.json
    python scripts/run_downtown.py run --only <gers> [<gers> ...] --keep-crops
    python scripts/run_downtown.py merge
    python scripts/run_downtown.py report
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from kyfromabove_3d.clouds import find_cloud                              # noqa: E402

NETWORK_SIGNS = ("CURL", "HTTP", "timed out", "ConnectionError", "IncompleteRead", "RemoteDisconnected",
                 "Connection reset", "503", "SlowDown")
STEPS = ("prepare", "colorize", "mesh")


# ---------------------------------------------------------------- the caches

def landmark_gers(landmarks_yaml: Path) -> dict[str, Path]:
    """GERS id -> building dir of the landmarks that already have meshes."""
    import yaml

    if not landmarks_yaml.exists():
        return {}
    out = {}
    for t in yaml.safe_load(landmarks_yaml.read_text(encoding="utf-8")).get("landmarks", []):
        if t.get("gers") and (ROOT / t["building_dir"] / "3dtiles_mesh" / "tileset.json").exists():
            out[t["gers"]] = ROOT / t["building_dir"]
    return out


def area_bounds_ft(records, pad_m: float = 300.0):
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform
    from shapely.wkt import loads as wkt_loads

    from kyfromabove_3d.coordinates import METERS_PER_US_SURVEY_FOOT

    to_ft = Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True).transform
    b = np.array([shp_transform(to_ft, wkt_loads(r["wkt"])).bounds for r in records])
    pad = pad_m / METERS_PER_US_SURVEY_FOOT
    return (float(b[:, 0].min() - pad), float(b[:, 1].min() - pad), float(b[:, 2].max() + pad), float(b[:, 3].max() + pad))


def build_caches(out: Path, records, log=print) -> dict:
    """The downtown frame, the frame index and the scene cloud; each made
    once and kept."""
    from kyfromabove_3d.coordinates import METERS_PER_US_SURVEY_FOOT, LocalCoordinateFrame
    from kyfromabove_3d.prepare import load_frames_index

    out.mkdir(parents=True, exist_ok=True)
    bounds = area_bounds_ft(records, pad_m=0.0)
    coords = out / "coordinates.json"
    if not coords.exists():
        grounds = [r["lidar_ground_z"] for r in records if r.get("lidar_ground_z") is not None]
        LocalCoordinateFrame(origin_source_xyz=(0.5 * (bounds[0] + bounds[2]), 0.5 * (bounds[1] + bounds[3]),
                                                float(np.median(grounds)) / METERS_PER_US_SURVEY_FOOT)).save_json(coords)
        log(f"downtown frame -> {coords}")
    frames = out / "frames_2024s1.gpkg"
    t = time.time()
    gdf = load_frames_index(frames, area_bounds_ft(records, pad_m=300.0))
    log(f"frame index: {len(gdf)} exposures ({time.time() - t:.0f}s) -> {frames}")
    scene = out / "scene_phase2.npz"
    if not scene.exists():
        import laspy

        from kyfromabove_3d.lidar import LidarTileIndex, open_copc

        t = time.time()
        wide = area_bounds_ft(records, pad_m=300.0)
        tiles = LidarTileIndex(ROOT / "data" / "raw", phases=("phase2",)).tiles_for_bounds(wide, phase="phase2")
        parts = []
        for tile in tiles:
            reader = open_copc(tile.url)
            h = reader.header
            pts = reader.query(laspy.copc.Bounds(mins=np.array([wide[0], wide[1], h.z_min]),
                                                 maxs=np.array([wide[2], wide[3], h.z_max])),
                               resolution=1.0 / METERS_PER_US_SURVEY_FOOT)
            c = np.asarray(pts.classification)
            keep = c != 7
            parts.append(np.column_stack([pts.x[keep], pts.y[keep], pts.z[keep]]))
        xyz = np.vstack(parts)
        np.savez(scene, xyz_ft=xyz)
        log(f"scene cloud: {len(xyz):,} returns at 1 m from {len(tiles)} tile(s) ({time.time() - t:.0f}s) -> {scene}")
    return {"coordinates": coords, "frames": frames, "scene": scene}


# ---------------------------------------------------------------- one building

def worker(args) -> int:
    """prepare -> colourise -> mesh -> cleanup for one building, writing
    state.json after every step. Runs in its own process."""
    from kyfromabove_3d.lidar import LidarTileIndex
    from kyfromabove_3d.prepare import load_frames_index, prepare_building

    data = json.loads(args.footprints.read_text(encoding="utf-8"))
    records = data["buildings"]
    rec = next((r for r in records if r["id"] == args.gers), None)
    if rec is None:
        print(f"[FAIL] {args.gers} not in {args.footprints}")
        return 1
    bdir = args.out / "buildings" / args.gers
    bdir.mkdir(parents=True, exist_ok=True)
    state_path = bdir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state.update({"gers": args.gers, "name": rec.get("name"), "attempts": state.get("attempts", 0) + 1})

    def save(**kw):
        state.update(kw)
        state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")

    force = set(args.force_step or [])
    t0 = time.time()
    try:
        # prepare
        # the crops are deleted after a mesh, and a mesh rebuild needs them
        # back: prepare again (10 s) when images/ is gone
        cams_file = bdir / "cameras_normalized.json"
        coords_file = bdir / "coordinates.json"
        old_origin = None
        if coords_file.exists():
            try:
                old_origin = json.loads(coords_file.read_text(encoding="utf-8")).get("origin_source_xyz")
            except Exception:
                old_origin = None
        need_crops = (not cams_file.exists()
                      or (bool(json.loads(cams_file.read_text(encoding="utf-8"))) and not (bdir / "images").exists()))
        if state.get("status") in (None, "pending", "failed") or "prepare" in force or need_crops:
            grounds = [r["lidar_ground_z"] for r in records if r.get("lidar_ground_z") is not None]
            frames = load_frames_index(args.out / "frames_2024s1.gpkg", area_bounds_ft(records, 300.0))
            index = LidarTileIndex(ROOT / "data" / "raw", phases=("phase2",))
            res = prepare_building(rec, bdir, frames, index, records, pad_ft=args.pad_ft,
                                   max_per_head=args.max_per_head, min_frames=args.min_frames,
                                   fallback_ground_m=float(np.median(grounds)) if grounds else None)
            save(status="prepared", prepare=res, error=None)
            new_origin = None
            if coords_file.exists():
                try:
                    new_origin = json.loads(coords_file.read_text(encoding="utf-8")).get("origin_source_xyz")
                except Exception:
                    new_origin = None
            if old_origin is not None and new_origin is not None and not np.allclose(old_origin, new_origin, atol=1e-3):
                # Origin moved: invalidate old colourise outputs and profiles to avoid mixing frames
                for fname in ("colorized_aerial.laz", "colorized_aerial.ply", "colorized_aerial_coverage.json",
                              "colorized.ply", "coverage.json",
                              "wall_planes.json", "wall_profiles.json", "levels.json"):
                    (bdir / fname).unlink(missing_ok=True)
                force.add("colorize")
        # colourise (as part of the mesh: build_mesh runs it and takes the box path if it declines)
        cmd = [sys.executable, str(SCRIPTS / "build_mesh.py"), "--building", str(bdir),
               "--name", rec.get("name") or args.gers, "--scene-cloud", str(args.out / "scene_phase2.npz")]
        if "colorize" in force or not find_cloud(bdir, "colorized_aerial").exists():
            cmd.append("--refresh")
        if args.no_prism_occluders:
            cmd.append("--no-prism-occluders")
        if args.complete:
            cmd.append("--complete")
        t = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           timeout=args.timeout_min * 60)
        print(r.stdout)
        if r.returncode != 0:
            raise RuntimeError(f"build_mesh exit {r.returncode}: " + "\n".join(r.stdout.splitlines()[-8:]))
        ts = json.loads((bdir / "3dtiles_mesh" / "tileset.json").read_text(encoding="utf-8"))
        ex = ts.get("extras", {})
        save(status="meshed", path=ex.get("path"), faces=ex.get("faces"), triangles=ex.get("triangles"),
             n_frames=ex.get("n_frames"), t_mesh=round(time.time() - t, 1), error=None)
        # the check: the mesh against the frames it was not textured from,
        # while the crops are still on disk; a check that fails does not fail the building
        if not args.no_check and ex.get("n_frames"):
            t = time.time()
            ccmd = [sys.executable, str(SCRIPTS / "check_building.py"), "--building", str(bdir)]
            if args.check_sheets:
                ccmd.append("--sheet")
            rc = subprocess.run(ccmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                timeout=args.timeout_min * 60)
            print(rc.stdout)
            if rc.returncode == 0:
                chk = json.loads((bdir / "texture_check.json").read_text(encoding="utf-8"))["building"]
                save(check=chk, check_error=None, t_check=round(time.time() - t, 1))
            else:
                save(check=None, check_error="\n".join(rc.stdout.splitlines()[-4:]))
        # the audit: the mesh against the LiDAR as a solid (holes, wall on a
        # flat roof, heights, blank wall, dots and lines); needs no crops,
        # and one that fails does not fail the building
        if not args.no_audit:
            t = time.time()
            acmd = [sys.executable, str(SCRIPTS / "audit_building.py"), "--building", str(bdir)]
            ra = subprocess.run(acmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                timeout=args.timeout_min * 60)
            print(ra.stdout)
            if ra.returncode == 0:
                save(audit=audit_brief(json.loads((bdir / "audit.json").read_text(encoding="utf-8"))),
                     audit_error=None, t_audit=round(time.time() - t, 1))
            else:
                save(audit=None, audit_error="\n".join(ra.stdout.splitlines()[-4:]))
        if not args.keep_crops:
            for sub in ("images", "validity_masks"):
                shutil.rmtree(bdir / sub, ignore_errors=True)
            (bdir / "occlusion.npz").unlink(missing_ok=True)       # made from the crops' cameras
            save(crops_kept=False)
        else:
            save(crops_kept=True)
        save(t_total=round(time.time() - t0, 1))
        return 0
    except Exception as exc:                                   # noqa: BLE001
        msg = str(exc)
        save(status="failed", error=msg[-2000:], t_total=round(time.time() - t0, 1),
             network=any(sig in msg for sig in NETWORK_SIGNS))
        print(f"[FAIL] {args.gers}: {msg[-500:]}")
        return 1


# -------------------------------------------------------------------- the run

def mark_split_parents(out: Path, records) -> int:
    """A footprint cut into parts (``parent_gers`` on the records) keeps its
    directory -- its cameras still serve the obliques panel for a click on
    the unsplit box -- but its state becomes ``split`` with the parts listed,
    and the merge leaves its mesh out."""
    parts: dict[str, list] = {}
    for r in records:
        if r.get("parent_gers"):
            parts.setdefault(r["parent_gers"], []).append(r["id"])
    n = 0
    for gers, ids in parts.items():
        pdir = out / "buildings" / gers
        if not pdir.exists():
            continue
        sp = pdir / "state.json"
        st = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {"gers": gers}
        if st.get("status") != "split" or st.get("parts") != sorted(ids):
            st.update({"status": "split", "parts": sorted(ids), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")})
            sp.write_text(json.dumps(st, indent=1), encoding="utf-8")
            n += 1
    return n


def read_states(out: Path) -> dict[str, dict]:
    states = {}
    for p in (out / "buildings").glob("*/state.json"):
        try:
            states[p.parent.name] = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return states


def write_manifest(out: Path, records, states: dict, landmarks: dict) -> dict:
    rows = []
    for r in records:
        s = states.get(r["id"], {})
        rows.append({"gers": r["id"], "name": r.get("name"),
                     "status": "landmark" if r["id"] in landmarks else s.get("status", "pending"),
                     "path": s.get("path"), "n_frames": s.get("n_frames"),
                     "n_on_footprint": (s.get("prepare") or {}).get("n_on_footprint"),
                     "t_total": s.get("t_total"), "error": s.get("error"), "attempts": s.get("attempts", 0)})
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    manifest = {"updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "counts": counts, "buildings": rows}
    tmp = out / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    os.replace(tmp, out / "manifest.json")
    return manifest


def audit_brief(rec: dict) -> dict:
    """The audit's headline numbers, for the manifest and the worst list."""
    h, t = rec.get("holes") or {}, rec.get("texture") or {}
    return {"hole_m2": h.get("total_m2"), "worst_face": next(iter(h.get("by_face") or {}), None),
            "worst_face_m2": next(iter((h.get("by_face") or {}).values()), None),
            "roof_wall_m2": (rec.get("roof_walls") or {}).get("m2"),
            "mesh_minus_lidar_m": (rec.get("heights") or {}).get("mesh_minus_lidar_m"),
            "product1_minus_lidar_m": (rec.get("heights") or {}).get("product1_minus_lidar_m"),
            "blank_share": t.get("blank_share"), "dots": t.get("dots"), "lines": t.get("lines"),
            "min_corner_deg": (rec.get("shape") or {}).get("min_corner_deg"),
            "min_width_m": (rec.get("shape") or {}).get("min_width_m")}


# the worst list: per measure, how "worse" reads off the brief
WORST = {"hole_m2": ("holes: LiDAR solid the mesh does not cover (m2, largest view)", lambda v: v),
         "roof_wall_m2": ("wall standing on a flat roof (m2)", lambda v: v),
         "mesh_minus_lidar_m": ("mesh top off the LiDAR's (m, either way)", abs),
         "product1_minus_lidar_m": ("product 1 height off the LiDAR's (m, either way)", abs),
         "blank_share": ("share of wall left blank", lambda v: v),
         "dots": ("round dots on the walls", lambda v: v),
         "lines": ("straight fill lines on the walls", lambda v: v),
         "min_width_m": ("thinnest outline (m, smaller is worse)", lambda v: -v)}


def audit_summary(out: Path, n: int = 20, log=print) -> None:
    """``audit_summary.json``: every building's brief, and the ``n`` worst per
    measure, by name -- where review starts, instead of browsing."""
    rows = []
    for d in sorted((out / "buildings").iterdir()):
        a = d / "audit.json"
        if not a.exists():
            continue
        rec = json.loads(a.read_text(encoding="utf-8"))
        rows.append({"building": d.name, "name": rec.get("name"), **audit_brief(rec)})
    worst = {}
    for key, (label, how) in WORST.items():
        have = [r for r in rows if r.get(key) is not None]
        have.sort(key=lambda r: -how(r[key]))
        worst[key] = {"measure": label, "buildings": [{"building": r["building"], "name": r["name"], key: r[key]}
                                                      for r in have[:n]]}
    (out / "audit_summary.json").write_text(json.dumps({"n": len(rows), "worst": worst, "buildings": rows}, indent=1),
                                            encoding="utf-8")
    if rows:
        log(f"audit: {len(rows)} buildings; worst holes: " + ", ".join(
            f"{r['name'] or r['building'][:8]} {r['hole_m2']:,.0f} m2" for r in worst["hole_m2"]["buildings"][:5]))


def do_merge(out: Path, landmarks: dict, log=print) -> None:
    sys.path.insert(0, str(SCRIPTS))
    from merge_downtown_tileset import merge

    boxes = ROOT / "outputs" / "overture_boxes" / "downtown_frame" / "tileset.json"
    ex = merge(out, out / "buildings", list(landmarks.values()), boxes if boxes.exists() else None)
    log(f"merged: {ex['buildings']} buildings, {ex['triangles']:,} triangles, paths {ex['paths']}")
    # the completed product (and its tinted copy), where any building has one;
    # the others stand in it as measured
    if any((out / "buildings").glob("*/3dtiles_complete/tileset.json")):
        for variant in ("complete", "complete_tint"):
            vout = out.parent / f"{out.name}_{variant}"
            merge(vout, out / "buildings", list(landmarks.values()), boxes if boxes.exists() else None, variant=variant)
        log(f"merged the completed product -> {out.parent / (out.name + '_complete')}")
    audit_summary(out, log=log)


def run(args) -> int:
    data = json.loads(args.footprints.read_text(encoding="utf-8"))
    records = data["buildings"]
    landmarks = landmark_gers(ROOT / "configs" / "landmarks.yaml")
    out = args.out
    caches = build_caches(out, records)
    mark_split_parents(out, records)
    states = read_states(out)
    todo = [r for r in records if r["id"] not in landmarks]
    if args.only:
        todo = [r for r in todo if r["id"] in set(args.only)]
    if not args.no_resume and not args.force_step:
        todo = [r for r in todo if states.get(r["id"], {}).get("status") != "meshed"]
    if args.limit:
        todo = todo[:args.limit]
    log_path = out / "run.log"
    print(f"{len(todo)} building(s) to run, {len(landmarks)} landmark(s) reused, {args.workers} worker(s); "
          f"manifest -> {out / 'manifest.json'}")
    write_manifest(out, records, states, landmarks)
    if not todo:
        do_merge(out, landmarks)
        return 0

    def one(rec):
        cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--footprints", str(args.footprints),
               "--gers", rec["id"], "--out", str(out), "--pad-ft", str(args.pad_ft),
               "--max-per-head", str(args.max_per_head), "--min-frames", str(args.min_frames),
               "--timeout-min", str(args.timeout_min)]
        if args.keep_crops:
            cmd.append("--keep-crops")
        if args.no_check:
            cmd.append("--no-check")
        if args.no_audit:
            cmd.append("--no-audit")
        if args.no_prism_occluders:
            cmd.append("--no-prism-occluders")
        if args.check_sheets:
            cmd.append("--check-sheets")
        if args.complete:
            cmd.append("--complete")
        if args.force_step:
            cmd += ["--force-step"] + list(args.force_step)
        bdir = out / "buildings" / rec["id"]
        bdir.mkdir(parents=True, exist_ok=True)
        for attempt in (1, 2):
            t = time.time()
            with open(bdir / "log.txt", "a", encoding="utf-8") as fh:
                fh.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} attempt {attempt}\n")
                fh.flush()                     # before the worker writes: its output belongs under this header
                try:
                    r = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT, text=True,
                                       timeout=(args.timeout_min + 25) * 60, env={**os.environ, "PYTHONUNBUFFERED": "1"})
                    rc = r.returncode
                except subprocess.TimeoutExpired:
                    fh.write(f"[FAIL] timed out after {args.timeout_min + 25} min\n")
                    rc = 124
            st = json.loads((bdir / "state.json").read_text(encoding="utf-8")) if (bdir / "state.json").exists() else {}
            if rc == 0 or not st.get("network"):
                break
        return rec["id"], rc, time.time() - t

    done = 0
    t_run = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, r): r for r in todo}
        for fut in as_completed(futures):
            rec = futures[fut]
            try:
                gers, rc, dt = fut.result()
            except Exception as exc:                           # noqa: BLE001
                gers, rc, dt = rec["id"], 1, 0.0
                print(f"[FAIL] {gers}: {exc}")
            done += 1
            st = read_states(out).get(gers, {})
            print(f"[{done}/{len(todo)}] {gers[:8]} {rec.get('name') or '':<32.32} {st.get('status', '?'):<8} "
                  f"{st.get('path') or '':<8} {dt:5.0f}s  {(st.get('error') or '')[:60]}")
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%H:%M:%S')} {gers} {st.get('status')} {st.get('path')} {dt:.0f}s {st.get('error') or ''}\n")
            if done % 10 == 0:
                write_manifest(out, records, read_states(out), landmarks)
            if args.merge_every and done % args.merge_every == 0:
                try:
                    do_merge(out, landmarks)
                except Exception as exc:                       # noqa: BLE001
                    print(f"[warn] merge failed: {exc}")
    manifest = write_manifest(out, records, read_states(out), landmarks)
    do_merge(out, landmarks)
    print(f"done in {(time.time() - t_run) / 60:.1f} min: {manifest['counts']}")
    return 0


def report(args) -> int:
    out = args.out
    data = json.loads(args.footprints.read_text(encoding="utf-8"))
    landmarks = landmark_gers(ROOT / "configs" / "landmarks.yaml")
    states = read_states(out)
    manifest = write_manifest(out, data["buildings"], states, landmarks)
    rows = manifest["buildings"]
    print(f"status: {manifest['counts']}")
    paths = {}
    for r in rows:
        if r["path"]:
            paths[r["path"]] = paths.get(r["path"], 0) + 1
    print(f"paths: {paths}")
    ts = np.array([r["t_total"] for r in rows if r.get("t_total")])
    if len(ts):
        print(f"time per building: median {np.median(ts):.0f}s, 90th {np.percentile(ts, 90):.0f}s, "
              f"max {ts.max():.0f}s, total {ts.sum() / 3600:.1f} h of worker time")
    failed = [r for r in rows if r["status"] == "failed"]
    for r in failed[:30]:
        print(f"  failed {r['gers'][:8]} {r.get('name') or '':<30.30} {(r.get('error') or '')[:100]}")
    (out / "report.json").write_text(json.dumps({"counts": manifest["counts"], "paths": paths,
                                                 "t_median": float(np.median(ts)) if len(ts) else None,
                                                 "failed": [r["gers"] for r in failed]}, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--footprints", type=Path, default=Path("data/derived/louisville_downtown_overture_split.json"),
                        help="the footprint list: overture_boxes.py split (default) or lidar-heights output")
    common.add_argument("--out", type=Path, default=Path("outputs/downtown"))
    common.add_argument("--pad-ft", type=float, default=90.0)
    common.add_argument("--max-per-head", type=int, default=4)
    common.add_argument("--min-frames", type=int, default=3)
    common.add_argument("--timeout-min", type=float, default=20.0)
    common.add_argument("--keep-crops", action="store_true")
    common.add_argument("--no-check", action="store_true", help="skip the texture check after the mesh")
    common.add_argument("--no-audit", action="store_true", help="skip the audit after the mesh (audit_building.py)")
    common.add_argument("--complete", action="store_true",
                        help="also build the completed product (unseen texels filled from each building's own facade)")
    common.add_argument("--no-prism-occluders", action="store_true",
                        help="texture with LiDAR-point occlusion alone (neighbours' facades mostly missing)")
    common.add_argument("--check-sheets", action="store_true", help="the check also draws texture_check.jpg")
    common.add_argument("--force-step", nargs="*", choices=STEPS, default=None)
    r = sub.add_parser("run", parents=[common]); r.add_argument("--workers", type=int, default=4)
    r.add_argument("--limit", type=int, default=0); r.add_argument("--only", nargs="*", default=None)
    r.add_argument("--no-resume", action="store_true"); r.add_argument("--merge-every", type=int, default=25)
    w = sub.add_parser("worker", parents=[common]); w.add_argument("--gers", required=True)
    sub.add_parser("merge", parents=[common])
    sub.add_parser("report", parents=[common])
    args = ap.parse_args()
    if args.cmd == "worker":
        return worker(args)
    if args.cmd == "merge":
        do_merge(args.out, landmark_gers(ROOT / "configs" / "landmarks.yaml"))
        return 0
    if args.cmd == "report":
        return report(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
