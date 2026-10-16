#!/usr/bin/env python3
"""Gather every building's mesh into one 3D Tiles tileset for the downtown.

Each building's ``3dtiles_mesh/tileset.json`` places its glTF with its own
root transform (the building's local frame on the Earth) and a bounding
sphere in that frame. The downtown tileset keeps each building as a child
tile with a transform *relative* to the downtown root, so nothing about the
building files changes: the glb is copied under ``glb/<gers>.glb`` and the
child carries the building's extras (name, GERS id, path, triangles).

The root's content is the Overture boxes at LiDAR heights (grey prisms):
from far away Cesium draws those, and as the viewer closes in each child's
mesh replaces its box. Between root and buildings sits one level of empty
cells (about 150 m) so the viewer can cull whole blocks at once.

Usage::

    python scripts/merge_downtown_tileset.py --out outputs/downtown \\
        --buildings outputs/downtown/buildings
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def col_major(m: list) -> np.ndarray:
    return np.array(m, dtype=np.float64).reshape(4, 4).T


def to_col_major(M: np.ndarray) -> list:
    return [float(v) for v in M.T.ravel()]


def building_child(tileset_path: Path, root_inv: np.ndarray, glb_dir: Path) -> dict | None:
    ts = json.loads(tileset_path.read_text(encoding="utf-8"))
    root = ts["root"]
    glb_src = tileset_path.parent / root["content"]["uri"]
    if not glb_src.exists():
        return None
    state_path = tileset_path.parent.parent / "state.json"
    if state_path.exists():
        try:
            st = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            st = {}
        parts = st.get("parts") or []
        if st.get("status") == "split" and parts and all(
                (tileset_path.parent.parent.parent / part / "3dtiles_mesh" / "building.glb").exists() for part in parts):
            return None                                      # its parts stand in for it
    ex = ts.get("extras", {})
    gers = ex.get("gers") or tileset_path.parent.parent.name
    dst = glb_dir / f"{gers}.glb"
    if not dst.exists() or dst.stat().st_mtime < glb_src.stat().st_mtime:
        shutil.copy2(glb_src, dst)
    T_b = col_major(root["transform"])
    T_rel = root_inv @ T_b
    return {
        "transform": to_col_major(T_rel),
        "boundingVolume": root["boundingVolume"],
        "geometricError": 0.0,
        "content": {"uri": f"glb/{gers}.glb"},
        "extras": {k: ex.get(k) for k in ("name", "gers", "parent_gers", "triangles", "path", "archetype", "faces", "n_frames", "texels", "fill") if k in ex},
        "_world_centre": (T_b @ np.append(np.array(root["boundingVolume"]["sphere"][:3]), 1.0))[:3],
        "_radius": root["boundingVolume"]["sphere"][3],
    }


#: which of a building's tilesets each product merges: the measured mesh, or
#: the completed one (and its tinted copy) where it has been built
VARIANTS = {None: ("3dtiles_mesh/tileset.json",),
            "complete": ("3dtiles_complete/tileset.json", "3dtiles_mesh/tileset.json"),
            "complete_tint": ("3dtiles_complete/tileset_tint.json", "3dtiles_mesh/tileset.json")}


def building_tileset(d: Path, variant: str | None) -> Path | None:
    """The tileset of building dir ``d`` for ``variant``: the completed one
    where built, else the measured one (nothing invented there); none in the
    measured product for a building marked not in the LiDAR."""
    for rel in VARIANTS[variant]:
        if (d / rel).exists():
            if variant is None:
                # the measured product holds only what the LiDAR measured: a
                # building it shows nothing of (built after the flight) is in
                # the completed product alone
                extras = json.loads((d / rel).read_text(encoding="utf-8")).get("extras", {})
                if extras.get("not_in_lidar"):
                    return None
            return d / rel
    return None


def merge(out: Path, buildings_dir: Path, extra: list[Path], boxes: Path | None, cell_m: float = 150.0,
          variant: str | None = None) -> dict:
    from build_mesh import enu_transform

    out.mkdir(parents=True, exist_ok=True)
    if not (out / "coordinates.json").exists():               # a product merged beside the downtown's
        shutil.copy2(buildings_dir.parent / "coordinates.json", out / "coordinates.json")
    coords = json.loads((out / "coordinates.json").read_text(encoding="utf-8"))
    transform, origin_geo = enu_transform(coords)
    T_root = col_major(transform)
    root_inv = np.linalg.inv(T_root)
    glb_dir = out / "glb"
    glb_dir.mkdir(parents=True, exist_ok=True)

    dirs = sorted(d for d in buildings_dir.iterdir() if d.is_dir()) + [Path(e) for e in extra]
    paths = [t for t in (building_tileset(d, variant) for d in dirs) if t is not None]
    children = [c for c in (building_child(p, root_inv, glb_dir) for p in paths) if c is not None]
    # Every building was placed through the same geoid; one that was not
    # (a rebuild without the grid) sits 30-odd metres above the others.
    from pyproj import Transformer
    to_geo = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    _, _, h_root = to_geo.transform(*T_root[:3, 3])
    misplaced = []
    for c in children:
        T_b = T_root @ col_major(c["transform"])
        _, _, h_b = to_geo.transform(*T_b[:3, 3])
        if abs(h_b - h_root) > 20.0:
            misplaced.append((c["extras"].get("name") or c["extras"].get("gers"), round(h_b - h_root, 1)))
    if misplaced:
        raise SystemExit("[FAIL] building(s) placed at a different height from the downtown frame (rebuilt without "
                         f"the GEOID grid?): {misplaced[:5]}")

    # Cells: group children by their centre in the downtown frame.
    cells: dict[tuple[int, int], list] = {}
    for c in children:
        local = (root_inv @ np.append(c.pop("_world_centre"), 1.0))[:3]
        r = c.pop("_radius")
        c["_local"] = local; c["_r"] = r
        key = (int(np.floor(local[0] / cell_m)), int(np.floor(local[1] / cell_m)))
        cells.setdefault(key, []).append(c)
    cell_tiles = []
    for key, kids in cells.items():
        pts = np.array([k["_local"] for k in kids]); rs = np.array([k["_r"] for k in kids])
        centre = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        radius = float(np.max(np.linalg.norm(pts - centre, axis=1) + rs))
        for k in kids:
            k.pop("_local"); k.pop("_r")
        cell_tiles.append({"boundingVolume": {"sphere": centre.tolist() + [radius]},
                           "geometricError": 8.0, "refine": "REPLACE", "children": kids})
    allc = np.array([t["boundingVolume"]["sphere"][:3] for t in cell_tiles]) if cell_tiles else np.zeros((1, 3))
    allr = np.array([t["boundingVolume"]["sphere"][3] for t in cell_tiles]) if cell_tiles else np.array([100.0])
    centre = 0.5 * (allc.min(axis=0) + allc.max(axis=0))
    radius = float(np.max(np.linalg.norm(allc - centre, axis=1) + allr))
    root = {"transform": transform, "boundingVolume": {"sphere": centre.tolist() + [radius]},
            "geometricError": 60.0, "refine": "REPLACE", "children": cell_tiles}
    if boxes and boxes.exists():
        bt = json.loads(boxes.read_text(encoding="utf-8"))
        # the boxes were built in this same frame, so their glb drops in as-is
        if np.allclose(col_major(bt["root"]["transform"]), T_root, atol=1e-6):
            shutil.copy2(boxes.parent / bt["root"]["content"]["uri"], out / "boxes.glb")
            root["content"] = {"uri": "boxes.glb"}
        else:
            print("[warn] boxes tileset is in a different frame; root has no coarse content")
    by_path = {}
    for c in children:
        by_path[c["extras"].get("path", "?")] = by_path.get(c["extras"].get("path", "?"), 0) + 1
    tileset = {
        "asset": {"version": "1.1", "generator": "kyfromabove-3d merge_downtown_tileset.py"},
        "geometricError": 2000.0, "root": root,
        "extras": {"name": "Downtown Louisville", "kind": "mesh", "buildings": len(children),
                   "triangles": int(sum(c["extras"].get("triangles", 0) for c in children)),
                   "paths": by_path, "cells": len(cell_tiles), "origin_lon_lat_h": origin_geo},
    }
    (out / "tileset.json").write_text(json.dumps(tileset, indent=1), encoding="utf-8")
    return tileset["extras"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("outputs/downtown"))
    ap.add_argument("--buildings", type=Path, default=None, help="default <out>/buildings")
    ap.add_argument("--extra", type=Path, nargs="*", default=[], help="other building dirs to include (the landmarks)")
    ap.add_argument("--boxes", type=Path, default=None, help="a boxes tileset built in the downtown frame, for the root")
    ap.add_argument("--cell-m", type=float, default=150.0)
    ap.add_argument("--variant", choices=["complete", "complete_tint"], default=None,
                    help="merge the completed product (or its tinted copy) instead of the measured one")
    args = ap.parse_args()
    ex = merge(args.out, args.buildings or (args.out / "buildings"), args.extra, args.boxes, args.cell_m, args.variant)
    print(f"{ex['buildings']} buildings, {ex['triangles']:,} triangles, {ex['cells']} cells, paths {ex['paths']} -> {args.out / 'tileset.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
