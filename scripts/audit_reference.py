#!/usr/bin/env python3
"""Check a code change on the buildings that have broken before, without
looking at them.

The reference set (``configs/audit_reference.yaml``) is rebuilt with the
current code, each building is audited (``audit_building.py``: the mesh
against the LiDAR as a solid, and more), and the audit is compared with the
last accepted one (``tests/data/audit_baseline.json``). A measure that got
worse beyond its tolerance fails, naming the building, the face and the
measure; ``outputs/audit_compare.png`` shows the accepted and the current
picture of each building that changed.

Usage::

    python scripts/audit_reference.py rebuild      # the set, with the current code (about 10 minutes)
    python scripts/audit_reference.py compare      # exit 1 if anything got worse
    python scripts/audit_reference.py accept       # the current audit becomes the baseline

``pytest -m data`` runs the comparison as a test.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))

CONFIG = ROOT / "configs" / "audit_reference.yaml"
BASELINE = ROOT / "tests" / "data" / "audit_baseline.json"
BASELINE_PICTURES = ROOT / "outputs" / "audit_baseline"          # local only: the accepted audit.png per building
BUILDINGS = ROOT / "outputs" / "downtown" / "buildings"


def load_config(path: Path = CONFIG) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def building_dir(b: dict, buildings: Path = BUILDINGS) -> Path:
    """A downtown building's folder, or a landmark's own (``dir``)."""
    return ROOT / b["dir"] if b.get("dir") else buildings / b["id"]


def current(cfg: dict, buildings: Path = BUILDINGS, fresh: bool = False) -> dict:
    """The set's audits as they are on disk (auditing any that has none, or
    all with ``fresh``): {id: audit record}."""
    from audit_building import audit

    out = {}
    for b in cfg["buildings"]:
        d = building_dir(b, buildings)
        a = d / "audit.json"
        if not (d / "3dtiles_mesh" / "building.glb").exists():
            continue
        if fresh or not a.exists():
            rec = audit(d, picture=d / "audit.png", case=b.get("case"))
            a.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
        out[b["id"]] = json.loads(a.read_text(encoding="utf-8"))
    return out


def essentials(rec: dict) -> dict:
    """What the baseline keeps of an audit."""
    h, t = rec.get("holes") or {}, rec.get("texture") or {}
    return {"name": rec.get("name"), "hole_m2": h.get("total_m2"), "by_face": h.get("by_face") or {},
            "roof_wall_m2": (rec.get("roof_walls") or {}).get("m2"),
            "mesh_minus_lidar_m": (rec.get("heights") or {}).get("mesh_minus_lidar_m"),
            "blank_share": t.get("blank_share"), "dots": t.get("dots"), "lines": t.get("lines"),
            "min_corner_deg": (rec.get("shape") or {}).get("min_corner_deg"),
            "min_width_m": (rec.get("shape") or {}).get("min_width_m"),
            "views": {v["view"]: v["hole_m2"] for v in h.get("views") or []},
            "faces_in_view": {v["view"]: sorted({str(r["face"]) for r in v["holes"]}) for v in h.get("views") or []}}


def compare(cfg: dict, now: dict, base: dict):
    """(worse, better): lists of plain sentences, one per building, face and
    measure that moved beyond its tolerance."""
    tol = cfg.get("tolerances", {})
    ht = tol.get("hole_m2", {"abs": 10.0, "rel": 0.10})
    allow = {(e["id"], str(e["face"])): float(e["allow_m2"]) for e in cfg.get("exceptions") or []}
    worse, better = [], []

    def more(a, b):                                       # hole area a is worse than b beyond the tolerance
        return a > b + max(ht["abs"], ht["rel"] * b)

    for bid, rec in now.items():
        n, b = essentials(rec), base.get(bid)
        name = n["name"] or bid[:8]
        if b is None:
            worse.append(f"{name}: not in the baseline (accept to add it)")
            continue
        # A hole can be put down to a neighbouring face, or to a corner, from
        # one build to the next while staying the same hole: a face counts as
        # worse only if a view that shows its hole got worse as a whole.
        grown = {k for k, a in n["views"].items() if k in (b.get("views") or {}) and more(a, float(b["views"][k]))}
        for k in sorted(grown):
            worse.append(f"{name}, seen {k}: hole {n['views'][k]:,.0f} m2, was {b['views'][k]:,.0f}")
        for face, a in n["by_face"].items():
            was = float(b["by_face"].get(face, 0.0))
            if b.get("views") and not any(face in n["faces_in_view"].get(k, []) for k in grown):
                continue
            if more(a, max(was, allow.get((bid, face), 0.0))):
                worse.append(f"{name}, face {face}: hole {a:,.0f} m2, was {was:,.0f}")
        for face, was in b["by_face"].items():
            a = float(n["by_face"].get(face, 0.0))
            if more(was, a):
                better.append(f"{name}, face {face}: hole {a:,.0f} m2, was {was:,.0f}")
        if n["roof_wall_m2"] is not None and b["roof_wall_m2"] is not None:
            if n["roof_wall_m2"] > b["roof_wall_m2"] + tol.get("roof_wall_m2", 5.0):
                worse.append(f"{name}: wall on a flat roof {n['roof_wall_m2']:,.0f} m2, was {b['roof_wall_m2']:,.0f}")
            elif b["roof_wall_m2"] > n["roof_wall_m2"] + tol.get("roof_wall_m2", 5.0):
                better.append(f"{name}: wall on a flat roof {n['roof_wall_m2']:,.0f} m2, was {b['roof_wall_m2']:,.0f}")
        if n["mesh_minus_lidar_m"] is not None and b["mesh_minus_lidar_m"] is not None:
            if abs(n["mesh_minus_lidar_m"]) > abs(b["mesh_minus_lidar_m"]) + tol.get("height_m", 1.5):
                worse.append(f"{name}: mesh top {n['mesh_minus_lidar_m']:+.1f} m off the LiDAR's, was {b['mesh_minus_lidar_m']:+.1f}")
        if n["blank_share"] is not None and b["blank_share"] is not None:
            if n["blank_share"] > b["blank_share"] + tol.get("blank_share", 0.03):
                worse.append(f"{name}: blank wall {n['blank_share']:.0%}, was {b['blank_share']:.0%}")
            elif b["blank_share"] > n["blank_share"] + tol.get("blank_share", 0.03):
                better.append(f"{name}: blank wall {n['blank_share']:.0%}, was {b['blank_share']:.0%}")
        for k, label in (("dots", "round dots"), ("lines", "fill lines")):
            if n[k] is not None and b[k] is not None and n[k] > b[k] + tol.get(k, 0):
                worse.append(f"{name}: {n[k]} {label} on the walls, was {b[k]}")
        lim = tol.get("min_corner_deg", 30.0)
        if n["min_corner_deg"] is not None and n["min_corner_deg"] < lim <= (b["min_corner_deg"] or 180.0):
            worse.append(f"{name}: outline's sharpest corner {n['min_corner_deg']:.0f} deg, was {b['min_corner_deg']:.0f}")
        lim = tol.get("min_width_m", 4.0)
        if n["min_width_m"] is not None and n["min_width_m"] < lim <= (b["min_width_m"] or 1e9):
            worse.append(f"{name}: outline {n['min_width_m']:.1f} m at its thinnest, was {b['min_width_m']:.1f}")
    for bid in base:
        if bid not in now:
            worse.append(f"{base[bid].get('name') or bid[:8]}: no mesh on disk (rebuild)")
    return worse, better


def sheet(ids, now: dict, base: dict, path: Path, buildings: Path = BUILDINGS) -> None:
    """The accepted and the current audit picture of each building that
    changed, side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    dirs = {b["id"]: building_dir(b, buildings) for b in load_config()["buildings"]}
    dirs = {i: dirs.get(i, buildings / i) for i in ids}
    ids = [i for i in ids if (dirs[i] / "audit.png").exists()]
    if not ids:
        return
    fig, axs = plt.subplots(len(ids), 2, figsize=(16, 7.5 * len(ids)), squeeze=False)
    for row, bid in zip(axs, ids):
        for ax, (p, label) in zip(row, ((BASELINE_PICTURES / f"{bid}.png", "accepted"),
                                         (dirs[bid] / "audit.png", "now"))):
            ax.axis("off")
            if p.exists():
                ax.imshow(Image.open(p))
            name = (now.get(bid) or {}).get("name") or bid[:8]
            ax.set_title(f"{name}: {label}", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=70)
    plt.close(fig)


def diff_runs(cfg: dict, before_dir: Path, buildings: Path, ids, figure: Path) -> dict:
    """Every building's audit now against a saved earlier one (``before_dir``
    holds ``<id>.json``), by the reference set's rules: counts of buildings
    worse and better per measure, the sentences, and a chart."""
    now, base = {}, {}
    for bid in ids:
        a, b = buildings / bid / "audit.json", before_dir / f"{bid}.json"
        if a.exists() and b.exists():
            now[bid] = json.loads(a.read_text(encoding="utf-8"))
            base[bid] = essentials(json.loads(b.read_text(encoding="utf-8")))
    worse, better = compare(cfg, now, base)
    rows = [(essentials(now[i]), base[i]) for i in now]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    panels = [("hole_m2", "holes, largest view (m²)", lambda v: v),
              ("blank_share", "share of wall left blank", lambda v: v),
              ("lines", "straight fill lines", lambda v: v),
              ("dots", "round dots", lambda v: v),
              ("roof_wall_m2", "wall on a flat roof (m²)", lambda v: v),
              ("mesh_minus_lidar_m", "mesh top off the LiDAR's (m, either way)", abs)]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))
    for ax, (k, label, f) in zip(axs.ravel(), panels):
        pairs = np.array([(f(b[k]), f(n[k])) for n, b in rows if n[k] is not None and b[k] is not None], dtype=float)
        if not len(pairs):
            ax.axis("off"); continue
        lim = max(float(pairs.max()), 1e-6) * 1.05
        up = pairs[:, 1] > pairs[:, 0] + 1e-9
        down = pairs[:, 1] < pairs[:, 0] - 1e-9
        ax.scatter(pairs[~up & ~down, 0], pairs[~up & ~down, 1], s=10, color="#c3c2b7", label=f"same ({(~up & ~down).sum()})")
        ax.scatter(pairs[down, 0], pairs[down, 1], s=10, color="#2a78d6", label=f"lower now ({down.sum()})")
        ax.scatter(pairs[up, 0], pairs[up, 1], s=10, color="#e34948", label=f"higher now ({up.sum()})")
        ax.plot([0, lim], [0, lim], color="#898781", lw=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("before"); ax.set_ylabel("now")
        ax.set_title(f"{label}\nmedian {np.median(pairs[:, 0]):.3g} before, {np.median(pairs[:, 1]):.3g} now", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"{len(rows)} downtown buildings, audited before and after the rebuild: each dot a building, "
                 "below the line is better", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=90)
    plt.close(fig)
    return {"n": len(rows), "worse": worse, "better": better}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("rebuild", "compare", "accept", "diff", "snapshot"))
    ap.add_argument("--audit-outline", choices=("footprint", "lidar"), default="footprint",
                    help="snapshot: measure against the LiDAR solid within Overture's footprint (as the baseline) "
                         "or within each building's LiDAR outline (footprint_lidar.json)")
    ap.add_argument("--to", type=Path, default=None, help="snapshot: the folder for <id>.json")
    ap.add_argument("--before", type=Path, default=ROOT / "outputs" / "downtown" / "audit_before",
                    help="diff: a folder of earlier audit.json files, <id>.json")
    ap.add_argument("--figure", type=Path, default=ROOT / "docs" / "figures" / "audit_downtown_diff.png")
    ap.add_argument("--fresh", action="store_true", help="re-audit every building even if audit.json exists")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep-crops", action="store_true", help="keep the frame crops (for re-meshing by hand)")
    ap.add_argument("--match-walls", action="store_true", help="rebuild with the colourise's wall matching on")
    ap.add_argument("--lidar-outline", action="store_true", help="rebuild with each building's LiDAR outline (footprint_lidar.json)")
    args = ap.parse_args()
    cfg = load_config()
    ids = [b["id"] for b in cfg["buildings"] if not b.get("dir")]
    landmarks = [b for b in cfg["buildings"] if b.get("dir")]

    if args.action == "rebuild":
        # landmarks first, on their own (their crops stay on disk), then the
        # downtown ones, whose run also merges the tilesets with the landmarks
        from audit_building import audit
        rc = 0
        for b in landmarks:
            d = ROOT / b["dir"]
            print(f"landmark {b['name']}: colourise and mesh")
            r = subprocess.run([sys.executable, str(SCRIPTS / "build_mesh.py"), "--building", str(d), "--case", b["case"],
                                "--refresh", "--complete"] + (["--match-walls"] if args.match_walls else [])
                               + (["--lidar-outline"] if args.lidar_outline else []), cwd=str(ROOT), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
            (d / "log.txt").open("a", encoding="utf-8").write(r.stdout)
            if r.returncode != 0:
                print("\n".join(r.stdout.splitlines()[-6:])); rc = 1
                continue
            rec = audit(d, picture=d / "audit.png", case=b["case"])
            (d / "audit.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
        cmd = [sys.executable, str(SCRIPTS / "run_downtown.py"), "run", "--only", *ids,
               "--force-step", "colorize", "mesh", "--complete", "--workers", str(args.workers)]
        if args.keep_crops:
            cmd.append("--keep-crops")
        if args.match_walls:
            cmd.append("--match-walls")
        if args.lidar_outline:
            cmd.append("--lidar-outline")
        r = subprocess.run(cmd, cwd=str(ROOT))
        return rc or r.returncode

    if args.action == "snapshot":
        # every reference building audited now, against the outline asked for, into --to (the building's own
        # audit.json is left alone), so two builds can be compared against the same LiDAR solid
        from audit_building import audit
        args.to.mkdir(parents=True, exist_ok=True)
        for b in cfg["buildings"]:
            d = building_dir(b)
            if (d / "3dtiles_mesh" / "building.glb").exists():
                rec = audit(d, case=b.get("case"), outline=args.audit_outline)
                (args.to / f"{b['id']}.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
        print(f"snapshot ({args.audit_outline} outline) -> {args.to}")
        return 0

    if args.action == "diff":
        # every building the downtown run built, now against --before
        man = json.loads((ROOT / "outputs" / "downtown" / "manifest.json").read_text(encoding="utf-8"))
        recs = man if isinstance(man, list) else man.get("buildings", man)
        run_ids = [r["gers"] for r in (recs if isinstance(recs, list) else recs.values()) if r.get("status") == "meshed"]
        d = diff_runs(cfg, args.before, BUILDINGS, run_ids, args.figure)
        (ROOT / "outputs" / "downtown" / "audit_diff.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
        print(f"{d['n']} buildings: {len(d['worse'])} measures worse, {len(d['better'])} better; "
              f"chart -> {args.figure}; list -> outputs/downtown/audit_diff.json")
        return 0

    now = current(cfg, fresh=args.fresh)
    if args.action == "accept":
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps({bid: essentials(rec) for bid, rec in now.items()}, indent=1), encoding="utf-8")
        BASELINE_PICTURES.mkdir(parents=True, exist_ok=True)
        dirs = {b["id"]: building_dir(b) for b in cfg["buildings"]}
        for bid in now:
            src = dirs.get(bid, BUILDINGS / bid) / "audit.png"
            if src.exists():
                shutil.copy(src, BASELINE_PICTURES / f"{bid}.png")
        print(f"baseline: {len(now)} buildings -> {BASELINE.relative_to(ROOT)}")
        return 0

    if not BASELINE.exists():
        print(f"no baseline yet: run  python scripts/audit_reference.py accept")
        return 1
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    worse, better = compare(cfg, now, base)
    for s in better:
        print(f"  better  {s}")
    for s in worse:
        print(f"  WORSE   {s}")
    changed = sorted({bid for bid, rec in now.items()
                      if any(s.startswith((rec.get("name") or bid[:8]) + ":") or s.startswith((rec.get("name") or bid[:8]) + ",")
                             for s in worse + better)})
    sheet(changed, now, base, ROOT / "outputs" / "audit_compare.png")
    print(f"{len(now)} buildings compared: {len(worse)} worse, {len(better)} better"
          + (f"; pictures -> outputs/audit_compare.png" if changed else ""))
    return 1 if worse else 0


if __name__ == "__main__":
    sys.exit(main())
