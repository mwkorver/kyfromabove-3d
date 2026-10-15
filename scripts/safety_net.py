#!/usr/bin/env python3
"""The refactor's safety net: rebuild a fixed set of buildings and record
what each produced, so a refactoring step can be shown to change nothing.

The set is ``configs/safety_net.yaml``: the reference set, one or two of every
class, and buildings fixed by hand. ``record`` rebuilds them the whole way --
the frames cut again, the colourise, the mesh and the completed product, the
texture check and the audit -- and writes, per building, a checksum of each
output that defines the products (the two meshes, the levels, the wall planes
and profiles, the reports with their timings left out), the geometry of each
mesh on its own (so a texture-only difference can be told from a shape one),
and the audit's measures (so a difference can be sized). ``check`` compares
two records and names every building and output that differs.

A landmark (``dir`` and ``case`` in the list) is built as it is today, on its
own with ``build_mesh.py --case``; it joins the ordinary path in step 2 of
the refactor.

Usage::

    python scripts/safety_net.py record --label before          # rebuild, then record
    python scripts/safety_net.py record --label again --no-build # record what is on disk
    python scripts/safety_net.py check before again
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
    return ROOT / b["dir"] if b.get("dir") else BUILDINGS / b["id"]


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
    from audit_building import audit
    rc = 0
    for b in [b for b in blist if b.get("dir")]:
        d = building_dir(b)
        print(f"landmark {b.get('name') or b['id']}: colourise and mesh", flush=True)
        r = subprocess.run([sys.executable, str(SCRIPTS / "build_mesh.py"), "--building", str(d), "--case", b["case"],
                            "--refresh", "--complete"], cwd=str(ROOT), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        (d / "log.txt").open("a", encoding="utf-8").write(r.stdout)
        if r.returncode != 0:
            print("\n".join(r.stdout.splitlines()[-6:]))
            rc = 1
            continue
        rec = audit(d, picture=d / "audit.png", case=b["case"])
        (d / "audit.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    ids = [b["id"] for b in blist if not b.get("dir")]
    cmd = [sys.executable, str(SCRIPTS / "run_downtown.py"), "run", "--only", *ids,
           "--force-step", "prepare", "colorize", "mesh", "--complete", "--workers", str(workers)]
    r = subprocess.run(cmd, cwd=str(ROOT))
    return rc or r.returncode


def cmd_record(a) -> int:
    blist = load_set(a.config)
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
    for k in sorted(set(x["buildings"]) | set(y["buildings"])):
        bx, by = x["buildings"].get(k), y["buildings"].get(k)
        if bx is None or by is None:
            out[k] = ["only in one record"]
            continue
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
    n = len(set(x["buildings"]) | set(y["buildings"]))
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
    c = sub.add_parser("check")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    return cmd_record(a) if a.cmd == "record" else cmd_check(a)


if __name__ == "__main__":
    sys.exit(main())
