#!/usr/bin/env python3
"""The refactor's safety net: rebuild a fixed set of buildings and record
what each produced, so a refactoring step can be shown to change nothing.

The set is ``configs/safety_net.yaml``: the reference set, one or two of every
class, and buildings fixed by hand. Downtown Louisville's are built as the
laptop builds them (run_downtown.py); an entry with a ``tile`` and a ``seed``
(a store folder holding the tile's inputs, tile files and neighbors' pieces)
is measured and built as the cloud does (cloud_jobs measure and build) in a
store cut fresh from the seed, then unpacked to be recorded the same way. ``record`` rebuilds them the whole way --
the frames cut again, the colourise, the mesh and the completed product, the
texture check and the audit -- and writes, per building, a checksum of each
output that defines the products (the two meshes, the levels, the wall planes
and profiles, the reports with their timings left out), the geometry of each
mesh on its own (so a texture-only difference can be told from a shape one),
and the audit's measures (so a difference can be sized). ``check`` compares
two records and names every building and output that differs.

Usage::

    python scripts/safety_net.py record --label before          # rebuild, then record
    python scripts/safety_net.py record --label again --no-build # record what is on disk
    python scripts/safety_net.py check before again
    python scripts/safety_net.py record --label downtown --all  # every downtown building
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

CONFIG = ROOT / "configs" / "safety_net.yaml"
BUILDINGS = ROOT / "outputs" / "downtown" / "buildings"
RECORDS = ROOT / "outputs" / "refactor"
#: the store the entries with a tile are measured and built in, and where they are unpacked to be recorded
STORE = RECORDS / "store"
STORED = RECORDS / "stored"
MESHES = ("3dtiles_mesh/building.glb", "3dtiles_complete/building.glb", "3dtiles_complete/building_tint.glb")
#: outputs whose content is compared, with keys that hold a run's own timings or paths left out
REPORTS = ("levels.json", "wall_planes.json", "wall_profiles.json", "mirror_report.json", "classification.json",
           "colorized_aerial_coverage.json", "audit.json", "texture_check.json",
           "3dtiles_mesh/tileset.json", "3dtiles_complete/tileset.json")
VOLATILE = {"seconds", "path"}


def load_set(path: Path = CONFIG) -> list[dict]:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))["buildings"]


def building_dir(b: dict) -> Path:
    return STORED / b["id"] if b.get("tile") else BUILDINGS / b["id"]


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k not in VOLATILE}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def json_digest(p: Path) -> str:
    return hashlib.sha256(json.dumps(_strip(json.loads(p.read_text(encoding="utf-8"))), sort_keys=True).encode()).hexdigest()


def mesh_record(p: Path) -> dict:
    from kyfromabove_3d.texture_check import read_glb_geometry
    V, T = read_glb_geometry(p)
    geo = hashlib.sha256(np.round(V, 4).astype(np.float64).tobytes() + np.asarray(T, dtype=np.int64).tobytes()).hexdigest()
    return {"file": hashlib.sha256(p.read_bytes()).hexdigest(), "geometry": geo, "triangles": int(len(T)),
            "vertices": int(len(V))}


def measures(d: Path) -> dict:
    a = d / "audit.json"
    if not a.exists():
        return {}
    r = json.loads(a.read_text(encoding="utf-8"))
    sw, h = r.get("stray_walls") or {}, r.get("holes") or {}
    return {"holes_m2": h.get("total_m2"),
            "holes_views_m2": round(sum(v.get("hole_m2", 0) or 0 for v in h.get("views", [])), 1),
            "above_m2": sw.get("above_m2"), "outside_m2": sw.get("outside_m2"),
            "inside_m2": (r.get("inside_walls") or {}).get("m2"), "roof_gap_m2": (r.get("roof_gap") or {}).get("m2")}


def record_one(b: dict) -> dict:
    d = building_dir(b)
    rec = {"meshes": {}, "reports": {}, "measures": measures(d)}
    for m in MESHES:
        if (d / m).exists():
            rec["meshes"][m] = mesh_record(d / m)
    for r in REPORTS:
        if (d / r).exists():
            rec["reports"][r] = json_digest(d / r)
    return rec


def build(blist: list[dict], workers: int) -> int:
    """The whole chain for every building in the list, as the laptop builds it today."""
    rc = 0
    ids = [b["id"] for b in blist if not b.get("tile")]
    if ids:
        cmd = [sys.executable, str(SCRIPTS / "run_downtown.py"), "run", "--only", *ids,
               "--force-step", "prepare", "colorize", "mesh", "--complete", "--workers", str(workers)]
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    stored = [b for b in blist if b.get("tile")]
    if stored:
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "build-stored",
                            *[f"{b['seed']}:{b['tile']}:{b['id']}" for b in stored]], cwd=str(ROOT))
        rc = rc or r.returncode
    return rc


def build_stored(entries: list[str]) -> int:
    """Each ``seed:tile:gers`` measured and built as the cloud does, in a store
    cut fresh from its seed (no piece of its own, so measuring cuts it again),
    and unpacked to ``STORED/<gers>``."""
    import os
    import shutil
    import zipfile
    os.environ["STORE"] = str(STORE)
    os.environ.setdefault("RAW_DIR", str(ROOT / "data" / "raw"))
    from kyfromabove_3d.cloud_jobs import build_job, measure_job
    rc = 0
    for e in entries:
        seed, tile, g = e.split(":")
        seed = ROOT / seed
        shutil.rmtree(STORE, ignore_errors=True)
        for sub in ("inputs", "tiles", "pieces"):
            shutil.copytree(seed / sub, STORE / sub, ignore=shutil.ignore_patterns(g))
        print(g[:8], "measure", measure_job({"kind": "measure", "run": "safety_net", "tile": tile, "gers": [g]}),
              "build", r := build_job({"kind": "build", "run": "safety_net", "tile": tile, "gers": [g]}), flush=True)
        rc = rc or int(r.get("built", 0) != 1)
        out = STORED / g
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        src = STORE / "buildings" / g
        if (src / "levels.json").exists():
            shutil.copy(src / "levels.json", out / "levels.json")
        for z, sub in (("3dtiles_mesh.zip", "3dtiles_mesh"), ("3dtiles_complete.zip", "3dtiles_complete"),
                       ("review.zip", "")):
            if (src / z).exists():
                zipfile.ZipFile(src / z).extractall(out / sub)
    return rc


def all_downtown(blist: list[dict]) -> list[dict]:
    """Every downtown building in the last run's manifest."""
    man = json.loads((ROOT / "outputs" / "downtown" / "manifest.json").read_text(encoding="utf-8"))
    recs = man.get("buildings", man) if isinstance(man, dict) else man
    recs = recs.values() if isinstance(recs, dict) else recs
    ids = sorted(r["gers"] for r in recs)
    return [{"id": k} for k in ids]


def cmd_record(a) -> int:
    blist = load_set(a.config)
    if a.all:
        blist = all_downtown(blist)
    t0 = time.time()
    if not a.no_build:
        rc = build(blist, a.workers)
        if rc:
            print("[warn] a build failed; recording what is on disk")
    out = {"label": a.label, "when": time.strftime("%Y-%m-%dT%H:%M:%S"), "buildings": {}}
    for b in blist:
        out["buildings"][b["id"]] = record_one(b)
    RECORDS.mkdir(parents=True, exist_ok=True)
    p = RECORDS / f"safety_net_{a.label}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{len(blist)} buildings recorded ({(time.time() - t0) / 60:.1f} min) -> {p}")
    return 0


def compare(x: dict, y: dict) -> dict:
    """Per building, what differs between two records: {id: [lines]}."""
    out = {}
    for k in sorted(set(x["buildings"]) & set(y["buildings"])):        # (a set against all of downtown: the set)
        bx, by = x["buildings"][k], y["buildings"][k]
        lines = []
        for m in sorted(set(bx["meshes"]) | set(by["meshes"])):
            mx, my = bx["meshes"].get(m), by["meshes"].get(m)
            if mx is None or my is None:
                lines.append(f"{m}: only in one")
            elif mx["geometry"] != my["geometry"]:
                lines.append(f"{m}: shape differs ({mx['triangles']} -> {my['triangles']} triangles)")
            elif mx["file"] != my["file"]:
                lines.append(f"{m}: same shape, texture or file differs")
        for r in sorted(set(bx["reports"]) | set(by["reports"])):
            if bx["reports"].get(r) != by["reports"].get(r):
                lines.append(f"{r} differs")
        if lines:
            mx, my = bx["measures"], by["measures"]
            lines.append("audit: " + ", ".join(f"{f} {mx.get(f)} -> {my.get(f)}" for f in mx if mx.get(f) != my.get(f)))
            out[k] = lines
    return out


def cmd_check(a) -> int:
    x = json.loads((RECORDS / f"safety_net_{a.a}.json").read_text(encoding="utf-8"))
    y = json.loads((RECORDS / f"safety_net_{a.b}.json").read_text(encoding="utf-8"))
    diff = compare(x, y)
    n = len(set(x["buildings"]) & set(y["buildings"]))
    for k, lines in diff.items():
        print(f"{k[:40]}:")
        for ln in lines:
            print(f"    {ln}")
    print(f"{n - len(diff)} of {n} buildings identical ({a.a} vs {a.b})")
    return 1 if diff else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--label", required=True)
    r.add_argument("--no-build", action="store_true", help="record what is on disk")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--config", type=Path, default=CONFIG)
    r.add_argument("--all", action="store_true", help="every downtown building, not only the set")
    c = sub.add_parser("check")
    c.add_argument("a")
    c.add_argument("b")
    s = sub.add_parser("build-stored", help="(build's own step) seed:tile:gers entries, built as the cloud does")
    s.add_argument("entries", nargs="+")
    a = ap.parse_args()
    if a.cmd == "build-stored":
        return build_stored(a.entries)
    return cmd_record(a) if a.cmd == "record" else cmd_check(a)


if __name__ == "__main__":
    sys.exit(main())
