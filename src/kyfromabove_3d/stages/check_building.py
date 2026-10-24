#!/usr/bin/env python3
"""Check a building's mesh against the aerial frames it was not textured from.

Reads ``3dtiles_mesh/building.glb``, lays each face on a grid, and looks at
it through every frame that sees it (the same facing, crop, mask and depth
tests the colourise uses). Each frame is scored on the points it did not
texture: how well the brightness pattern agrees (normalised correlation, 1
= the same, 0 = unrelated) and the shift, in metres on the face, that lines
the two up best. Writes ``texture_check.json``; with ``--sheet`` also
``texture_check.jpg``, one row per frame: the frame, the mesh drawn into
that frame's view, and each face coloured by how well it agreed there.

Needs the frame crops on disk (``run_downtown.py run ... --keep-crops``).

Usage::

    python scripts/check_building.py --building outputs/downtown/buildings/<gers> --sheet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]                                    # the repository (src/kyfromabove_3d/stages/..)

from kyfromabove_3d.clouds import find_cloud, read_cloud                         # noqa: E402
from kyfromabove_3d.features import LEVEL_FACE_STRIDE                            # noqa: E402
from kyfromabove_3d.features import wall_planes as footprint_edges              # noqa: E402
from kyfromabove_3d.texture_check import (face_grids, read_glb, score_face,     # noqa: E402
                                          summarise_face)


def building_planes(b: Path):
    """The footprint edges as build_mesh numbers them: level k's from k * STRIDE."""
    levels = json.loads((b / "levels.json").read_text(encoding="utf-8")) if (b / "levels.json").exists() else None
    if not levels:
        fp = json.loads((b / "footprint.json").read_text(encoding="utf-8"))
        from kyfromabove_3d.footprints import footprint_wgs84_to_local
        ox, oy, _ = json.loads((b / "coordinates.json").read_text(encoding="utf-8"))["origin_source_xyz"]
        ring = np.asarray(footprint_wgs84_to_local(fp["wkt"], ox, oy).exterior.coords)
        levels = [{"level": 0, "ring": ring.tolist(), "z_lo": 0.0, "z_hi": float(fp.get("height_m") or 10.0)}]
    planes = []
    for lv in levels:
        edges = footprint_edges(np.asarray(lv["ring"], dtype=float), float(lv["z_lo"]), float(lv["z_hi"]),
                                min_length_m=0.3, arcs=(int(lv["level"]) == 0))
        for e in edges:
            e.face += int(lv["level"]) * LEVEL_FACE_STRIDE
        planes += edges
    return planes


def check(b: Path, spacing_m: float = 0.25, scene_cloud: Path | None = None, radius_m: float = 250.0,
          footprints: Path | None = None, neighbour_levels: str | None = None):
    """(faces summary, grids, cams, per-face scores) for building dir ``b``."""
    from kyfromabove_3d.stages.colorize_building import load_cameras, neighbour_prisms, occluder_cloud, project_visible

    glb = b / "3dtiles_mesh" / "building.glb"
    prims = read_glb(glb)
    grids = face_grids(prims, building_planes(b), spacing_m=spacing_m)
    lid = read_cloud(find_cloud(b, "lidar_local"))
    clip = np.column_stack([lid["x"], lid["y"], lid["z"]]).astype(np.float64)
    from kyfromabove_3d.stages.colorize_building import levels_dir, occlusion_maps
    kw = {} if neighbour_levels is None else {"buildings_dir": levels_dir(neighbour_levels)}
    cams = occlusion_maps(b, load_cameras(b), occluder_cloud(b, scene_cloud, radius_m, clip),
                          neighbour_prisms(b, footprints, radius_m, **kw))
    if not cams:
        raise SystemExit(f"[FAIL] no frame crops on disk in {b}; re-cut them (run_downtown.py ... --keep-crops)")
    project = lambda p, n, c: project_visible(p, n, c, 0.15, 1.5)                 # noqa: E731
    scores = {g.key: score_face(g, cams, project) for g in grids}
    faces = {g.key: summarise_face(g, scores[g.key]) for g in grids}
    return faces, grids, cams, scores


def summary(faces: dict) -> dict:
    """The building's line: area-weighted over the faces that were checked."""
    rows = [f for f in faces.values() if f["ncc"] is not None]
    area = np.array([f["area_m2"] for f in rows])
    def wmedian(vals):
        vals = np.asarray(vals, dtype=float)
        o = np.argsort(vals); c = np.cumsum(area[o])
        return float(vals[o][np.searchsorted(c, c[-1] / 2)])
    total = sum(f["area_m2"] for f in faces.values())
    shifted = sum(f["area_m2"] for f in rows if f["shift_m"] is not None and f["shift_m"] >= 0.5)
    unexplained = sum(f["area_m2"] for f in rows if f.get("low_without_shift"))
    return {
        "faces": len(faces), "faces_checked": len(rows),
        "area_m2": round(total, 1), "area_checked_m2": round(float(area.sum()), 1) if len(rows) else 0.0,
        "ncc": round(wmedian([f["ncc"] for f in rows]), 3) if rows else None,
        "ncc_source": round(wmedian([f["ncc_source"] if f["ncc_source"] is not None else np.nan for f in rows]), 3)
        if rows and any(f["ncc_source"] is not None for f in rows) else None,
        "area_shifted_half_metre_m2": round(shifted, 1),
        "area_low_without_shift_m2": round(unexplained, 1),
    }


def sheet(path: Path, grids, cams, scores, n_rows: int = 4, width_px: int = 460):
    """One row per frame (the four with most points checked, distinct heads
    first): the frame, the mesh drawn into its view, each face coloured by
    its agreement there (red 0 -> green 1; grey: this frame textured it)."""
    from PIL import Image, ImageDraw
    from kyfromabove_3d.stages.colorize_building import project_visible

    per_frame = {}
    for key, sc in scores.items():
        for s in sc:
            if not s.source:
                per_frame.setdefault(s.frame, {})[key] = s
    ranked = sorted(per_frame, key=lambda f: -sum(s.n for s in per_frame[f].values()))
    chosen, heads = [], set()
    for f in ranked:                                                  # one per head, then the rest
        h = next(c["head"] for c in cams if c["id"] == f)
        if h not in heads:
            chosen.append(f); heads.add(h)
    chosen += [f for f in ranked if f not in chosen]
    chosen = chosen[:n_rows]
    if not chosen:
        return None
    rows = []
    for fid in chosen:
        cam = next(c for c in cams if c["id"] == fid)
        img = cam["img"].astype(np.float64)
        dim = img * 0.35
        mesh, agree = dim.copy(), dim.copy()
        for g in grids:
            pts, nrm = g.xyz[g.valid], g.normal[g.valid]
            idx, u, v, _ = project_visible(pts, nrm, cam, 0.15, 1.5)
            if not len(idx):
                continue
            col = g.model[g.valid][idx]
            s = per_frame[fid].get(g.key)
            if s is None or not np.isfinite(s.ncc):
                acol = np.array([150.0, 150.0, 150.0])
            else:
                t = float(np.clip(s.ncc, 0, 1))
                acol = np.array([220 * (1 - t) + 40 * t, 60 * (1 - t) + 190 * t, 60.0])
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    uu = np.clip(u + du, 0, img.shape[1] - 1); vv = np.clip(v + dv, 0, img.shape[0] - 1)
                    mesh[vv, uu] = col
                    agree[vv, uu] = acol
        scale = width_px / img.shape[1]
        size = (width_px, max(1, int(img.shape[0] * scale)))
        tiles = [Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize(size) for a in (img, mesh, agree)]
        row = Image.new("RGB", (size[0] * 3 + 8, size[1] + 18), (255, 255, 255))
        for i, t in enumerate(tiles):
            row.paste(t, (i * (size[0] + 4), 18))
        ImageDraw.Draw(row).text((4, 3), f"{cam['head']}  {fid.rsplit('/', 1)[-1]}", fill=(0, 0, 0))
        rows.append(row)
    W = max(r.width for r in rows)
    out = Image.new("RGB", (W, sum(r.height for r in rows)), (255, 255, 255))
    y = 0
    for r in rows:
        out.paste(r, (0, y)); y += r.height
    out.save(path, quality=88)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--building", type=Path, required=True)
    ap.add_argument("--spacing-m", type=float, default=0.25)
    ap.add_argument("--sheet", action="store_true", help="also write texture_check.jpg")
    ap.add_argument("--scene-cloud", type=Path, default=ROOT / "outputs" / "downtown" / "scene_phase2.npz",
                    help="the area's occluder cloud (run_downtown.py writes it)")
    ap.add_argument("--occluder-radius-m", type=float, default=250.0)
    ap.add_argument("--footprints", type=Path,
                    default=ROOT / "data" / "derived" / "louisville_downtown_overture_split.json",
                    help="the neighbours, as solid prisms at their LiDAR heights")
    ap.add_argument("--neighbour-levels", default=None,
                    help="where the neighbours' levels.json are; 'none' for their outlines at LiDAR height only")
    args = ap.parse_args(argv)
    t0 = time.time()
    faces, grids, cams, scores = check(args.building, args.spacing_m, args.scene_cloud, args.occluder_radius_m,
                                       args.footprints, args.neighbour_levels)
    s = summary(faces)
    s["seconds"] = round(time.time() - t0, 1)
    (args.building / "texture_check.json").write_text(json.dumps({"building": s, "faces": faces}, indent=1),
                                                      encoding="utf-8")
    print(f"texture check: {s['faces_checked']} of {s['faces']} faces checked "
          f"({s['area_checked_m2']:.0f} of {s['area_m2']:.0f} m2), agreement {s['ncc']} "
          f"(ceiling {s['ncc_source']}), {s['area_shifted_half_metre_m2']:.0f} m2 shifted 0.5 m or more, "
          f"{s['area_low_without_shift_m2']:.0f} m2 low with no shift to explain it "
          f"({s['seconds']:.0f}s)")
    for k, f in sorted(faces.items(), key=lambda kv: (kv[1]["ncc"] is None, kv[1]["ncc"] or 0)):
        if f["ncc"] is not None:
            print(f"  {k:10s} {f['area_m2']:7.0f} m2  agreement {f['ncc']:.2f} (ceiling {f['ncc_source']}) "
                  f"over {f['frames_checked']} frames; clear shift {f['shift_m']} m in {f['clear_shifts']}")
    if args.sheet:
        p = sheet(args.building / "texture_check.jpg", grids, cams, scores)
        print(f"  sheet -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
