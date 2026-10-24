#!/usr/bin/env python3
"""Audit one building's mesh against evidence it was not built from.

Measures what, so far, only a person looking at the viewer has caught
(``kyfromabove_3d.audit``): holes where the LiDAR says there is building
and the mesh shows none, wall standing on a flat roof, the height against
the LiDAR's and product 1's, blank wall and the dots and lines of the gap
fill, and the outline's sharpest corner and thinnest width. Needs no frame
crops: the glb, the LiDAR clip, the footprint and ``levels.json``.

Writes ``audit.json`` and ``audit.png`` (the eight side views and the top
view, holes in magenta) into the building directory.

Usage::

    python scripts/audit_building.py --building outputs/downtown/buildings/<gers>
    python scripts/audit_building.py --building <dir> --glb other.glb --out /tmp/x   # another mesh of it
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from kyfromabove_3d import audit as A                                            # noqa: E402
from kyfromabove_3d.clouds import find_cloud, read_cloud                         # noqa: E402
from kyfromabove_3d.project import repo_relative                                 # noqa: E402


def lidar_model(b: Path, outline: str = "footprint") -> dict:
    """The building as the mesh step sets it up (build_mesh.py): the outline
    in the local frame, the levels, and the LiDAR roof field over the
    outline, with the same cell and height cap. ``outline="lidar"`` takes the
    outline from the building's footprint_lidar.json (the LiDAR roof's edge,
    scripts/lidar_outline.py) instead of Overture's footprint, so meshes built
    on either outline can be measured against the same LiDAR solid."""
    import shapely

    from kyfromabove_3d.stages.build_mesh import roof_heightfield
    from kyfromabove_3d.footprints import resolve_footprint

    coords = json.loads((b / "coordinates.json").read_text(encoding="utf-8"))
    ox, oy, _ = coords["origin_source_xyz"]
    if outline == "lidar" and (b / "footprint_lidar.json").exists():
        poly, fp = resolve_footprint(b, ox, oy, footprint=b / "footprint_lidar.json")
    else:
        poly, fp = resolve_footprint(b, ox, oy)
    lidar = read_cloud(find_cloud(b, "lidar_local"))
    xyz = np.column_stack([lidar["x"], lidar["y"], lidar["z"]]).astype(np.float64)
    ground_z = 0.0
    height_m = fp.get("height_m") or fp.get("lidar_height_m")
    n_on = int(fp.get("lidar_n") or 0) or int(shapely.contains_xy(poly, xyz[:, 0], xyz[:, 1]).sum())
    ts_path = b / "3dtiles_mesh" / "tileset.json"
    extras = json.loads(ts_path.read_text(encoding="utf-8")).get("extras", {}) if ts_path.exists() else {}
    path = extras.get("path") or ("box" if n_on < 30 or not height_m else "profile")
    lv_path = b / "levels.json"
    levels = json.loads(lv_path.read_text(encoding="utf-8")) if lv_path.exists() and path != "box" else None
    if levels:
        roof_top = max(float(lv["z_hi"]) for lv in levels) - ground_z
    elif height_m:
        roof_top = float(height_m)
    else:
        roof_top = max(float(np.percentile(xyz[:, 2], 99)) - ground_z, 6.0) if len(xyz) else 6.0
    ring = np.array(poly.exterior.coords)
    if not levels:
        levels = [{"level": 0, "ring": ring.tolist(), "z_lo": ground_z, "z_hi": ground_z + roof_top}]
    cell = 0.75 if n_on / max(poly.area, 1.0) >= 4.0 else 1.5
    xs, ys, Z, valid = roof_heightfield(xyz, poly, ground_z, cell, max_z=ground_z + roof_top + 1.5)
    if path == "box" or valid.sum() < 4:
        # the mesh is a box at the building's height: so is the solid
        X, Y = np.meshgrid(xs, ys)
        valid = shapely.contains_xy(poly, X, Y)
        Z = np.where(valid, ground_z + roof_top, np.nan)
    # Under an overhang (the Hyatt's drum over its shaft) a roof field's column
    # would stand to the ground: with the LiDAR outline, the cells over the
    # overhang's plan start at its underside instead (overhangs.json, written by
    # the colourise from the returns on the shaft's wall: find_overhangs).
    floor = np.full(Z.shape, ground_z + 1.5)
    under = []
    oh_path = b / "overhangs.json"
    if outline == "lidar" and oh_path.exists() and path != "box":
        X, Y = np.meshgrid(xs, ys)
        for oh in json.loads(oh_path.read_text(encoding="utf-8")):
            inside = shapely.contains_xy(shapely.Polygon(oh["cut_m"]), X, Y) & valid
            floor[inside] = np.maximum(floor[inside], float(oh["under_z"]))
            under.append({"under_z": oh["under_z"], "plan_m2": round(float(inside.sum() * cell * cell), 1),
                          "n_line_returns": oh.get("n_line_returns")})
    return {"poly": poly, "fp": fp, "xyz": xyz, "ground_z": ground_z, "levels": levels, "roof_top": roof_top,
            "xs": xs, "ys": ys, "Z": Z, "valid": valid, "path": path, "extras": extras, "floor": floor, "overhangs": under}


def level_edges(levels):
    """The faces, numbered as build_mesh.py numbers them."""
    from kyfromabove_3d.features import LEVEL_FACE_STRIDE
    from kyfromabove_3d.features import wall_planes as footprint_edges

    edges = []
    for lv in levels:
        ring = np.asarray(lv["ring"], dtype=float)
        for e in footprint_edges(ring, float(lv["z_lo"]), float(lv["z_hi"]), min_length_m=0.3, arcs=(int(lv["level"]) == 0)):
            e.face += int(lv["level"]) * LEVEL_FACE_STRIDE
            edges.append(e)
    return edges


def audit(b: Path, glb: Path | None = None, picture: Path | None = None,
          outline: str = "footprint") -> dict:
    from kyfromabove_3d.stages.overture_boxes import min_corner_deg, roof_returns

    from kyfromabove_3d.texture_check import read_glb_geometry

    t0 = time.time()
    glb = glb or (b / "3dtiles_mesh" / "building.glb")
    V, T = read_glb_geometry(glb)
    m = lidar_model(b, outline=outline)
    rec = {"building": b.name, "name": m["fp"].get("name"), "path": m["path"], "glb": repo_relative(glb), "audit_outline": outline}

    # holes: the mesh against the LiDAR solid, from eight sides and above
    pts, top = A.lidar_solid(m["xs"], m["ys"], m["Z"], m["valid"], floor_z=m["floor"])
    edges = level_edges(m["levels"])
    face = np.full(len(pts), -1, dtype=np.int64)
    face[~top] = A.nearest_face(pts[~top], edges)
    mesh_pts, _ = A.sample_mesh(V, T)
    from kyfromabove_3d.features import ArcWall
    normals = {e.face: np.asarray(e.normal, dtype=float) for e in edges if not isinstance(e, ArcWall)}
    h = A.holes(mesh_pts, A.Solid(pts, face), face_normals=normals)
    rec["holes"] = {k: v for k, v in h.items() if k != "images"}
    if m["overhangs"]:
        # what the LiDAR solid leaves out under overhangs, so a change in the measure is seen
        rec["overhangs"] = m["overhangs"]

    # wall standing on a flat roof
    rec["roof_walls"] = A.roof_walls(V, T, m["xs"], m["ys"], m["Z"], m["valid"], np.asarray(m["levels"][0]["ring"]))
    # wall drawn where the LiDAR has no building
    rec["stray_walls"] = A.stray_walls(V, T, m["xs"], m["ys"], m["Z"], m["valid"])
    # the strip just inside the walls that the roof leaves open, from above
    rec["roof_gap"] = A.roof_gap(V, T, [lv["ring"] for lv in m["levels"]])
    # wall drawn inside the building, under an unbroken roof
    rec["inside_walls"] = A.inside_walls(V, T, m["xs"], m["ys"], m["Z"], m["valid"], np.asarray(m["levels"][0]["ring"]))

    # heights
    above = m["xyz"][m["xyz"][:, 2] > m["ground_z"] + 2.0]
    inside = roof_returns(m["poly"], above) if len(above) else above
    lidar_top = float(np.percentile(inside[:, 2], 99)) - m["ground_z"] if len(inside) else None
    mesh_top = float(V[:, 2].max()) - m["ground_z"] if len(V) else None
    p1 = m["fp"].get("height_m") or m["fp"].get("lidar_height_m")
    rec["heights"] = {"mesh_top_m": None if mesh_top is None else round(mesh_top, 1),
                      "lidar_p99_m": None if lidar_top is None else round(lidar_top, 1),
                      "product1_m": None if p1 is None else round(float(p1), 1),
                      "mesh_minus_lidar_m": None if None in (mesh_top, lidar_top) else round(mesh_top - lidar_top, 1),
                      "product1_minus_lidar_m": None if None in (p1, lidar_top) else round(float(p1) - lidar_top, 1)}

    # texture: product 2's per-texel sources on the walls
    ff = b / "3dtiles_complete" / "fill_faces.npz"
    if ff.exists():
        z = np.load(ff, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        srcs = [z[f"src_{i}"] for i, mm in enumerate(meta) if mm.get("kind") != "roof" and f"src_{i}" in z]
        rec["texture"] = A.texture_artefacts(srcs, float(z["texel_m"]))
    else:
        rec["texture"] = None
    tc = b / "texture_check.json"
    rec["check_agreement"] = json.loads(tc.read_text(encoding="utf-8"))["building"].get("ncc") if tc.exists() else None

    # the outline's shape (a split part squashed to a point)
    rec["shape"] = {"min_corner_deg": round(float(min_corner_deg(m["poly"])), 1), "min_width_m": A.min_width_m(m["poly"])}
    rec["seconds"] = round(time.time() - t0, 1)
    if picture is not None:
        A.draw(h["images"], picture, title=f"{rec['name'] or b.name}")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--building", type=Path, required=True)
    ap.add_argument("--glb", type=Path, default=None, help="audit this mesh of the building instead of its own")
    ap.add_argument("--out", type=Path, default=None, help="where to write audit.json / audit.png (default: the building)")
    ap.add_argument("--no-picture", action="store_true")
    args = ap.parse_args(argv)
    out = args.out or args.building
    out.mkdir(parents=True, exist_ok=True)
    rec = audit(args.building, args.glb, None if args.no_picture else out / "audit.png")
    (out / "audit.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    h = rec["holes"]
    print(f"{rec['name'] or rec['building']}: holes {h['total_m2']:,.0f} m2 (largest view)"
          + ("; by face " + ", ".join(f"{f}: {a:,.0f}" for f, a in list(h["by_face"].items())[:5]) if h["by_face"] else "")
          + f"; wall on flat roof {rec['roof_walls']['m2']:,.0f} m2; wall where the LiDAR has no building {rec['stray_walls']['m2']:,.0f} m2; top {rec['heights']['mesh_top_m']} m "
          f"(LiDAR {rec['heights']['lidar_p99_m']}, product 1 {rec['heights']['product1_m']})"
          + (f"; blank wall {rec['texture']['blank_share']:.0%}, {rec['texture']['dots']} dots, {rec['texture']['lines']} lines"
             if rec["texture"] else "")
          + f"; {rec['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
