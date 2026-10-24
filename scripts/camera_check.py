#!/usr/bin/env python3
"""How well the aerial frames line up with the LiDAR, frame by frame and head by head.

The LiDAR's height steps -- a parapet's top, a building's outline, a rooftop
unit, anywhere the surface drops by ``--step-m`` or more within a cell -- are
sharp lines seen from the air. Each step's top is projected into every frame
through the frame's camera model, and the frame's picture is slid until those
points sit on the strongest brightness edges: the slide that does it is the
frame's offset from the LiDAR, in pixels and, back on the ground at the
points' height, in metres east and north. A correct camera model needs no
slide. The same frame seen through several buildings' windows should need the
same slide, which checks the measure itself.

Steps:

  measure   every cached frame window of the reference buildings -> camera_check.json
  plot      the offsets by camera head

Usage::

    python scripts/camera_check.py measure
    python scripts/camera_check.py plot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kyfromabove_3d.stages import wall_visibility as wv  # noqa: E402

from kyfromabove_3d.wall_match import bilinear_gray  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "outputs" / "wall_visibility" / "reference"


def step_tops(las_path: Path, cell_m: float, step_m: float, max_points: int):
    """The tops of the LiDAR's height steps, in the building's local frame:
    cells of a highest-return grid with a neighbour ``step_m`` or more lower."""
    import laspy
    las = laspy.read(las_path)
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    x0, y0 = x.min(), y.min()
    ix, iy = ((x - x0) / cell_m).astype(int), ((y - y0) / cell_m).astype(int)
    g = np.full((iy.max() + 1, ix.max() + 1), -np.inf)
    np.maximum.at(g, (iy, ix), z)
    have = np.isfinite(g)
    drop = np.zeros_like(have)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nb = np.roll(np.roll(g, dy, 0), dx, 1)
        nb_ok = np.roll(np.roll(have, dy, 0), dx, 1)
        drop |= have & nb_ok & (g - nb >= step_m)
    drop[0, :] = drop[-1, :] = drop[:, 0] = drop[:, -1] = False
    jy, jx = np.nonzero(drop)
    pts = np.column_stack([x0 + (jx + 0.5) * cell_m, y0 + (jy + 0.5) * cell_m, g[jy, jx]])
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
    return pts


def best_slide(img: np.ndarray, u: np.ndarray, v: np.ndarray, max_px: int):
    """The slide (du, dv) putting the points on the strongest edges: coarse
    every 2 px, then every 0.5 px; and how clear it is (its score over the
    median score of all the slides tried)."""
    from scipy.ndimage import gaussian_filter
    gs = gaussian_filter(img, 1.2)
    gy, gx = np.gradient(gs)
    mag = np.hypot(gx, gy).astype(np.float32)

    def score(du, dv):
        val, ok = bilinear_gray(mag, u + du, v + dv)
        return float(val[ok].mean()) if ok.mean() > 0.5 else -1.0
    grid = np.arange(-max_px, max_px + 1e-9, 2.0)
    S = np.array([[score(du, dv) for du in grid] for dv in grid])
    j, i = np.unravel_index(np.argmax(S), S.shape)
    du, dv = grid[i], grid[j]
    fine = np.arange(-2.0, 2.001, 0.5)
    du, dv = max(((du + a, dv + b) for a in fine for b in fine), key=lambda t: score(*t))
    valid = S[S > 0]
    clear = score(du, dv) / float(np.median(valid)) if len(valid) else np.nan
    edge = abs(du) >= max_px - 1 or abs(dv) >= max_px - 1
    return float(du), float(dv), float(clear), bool(edge)


def ground_shift(cam, X: np.ndarray, du: float, dv: float, x0: float, y0: float) -> np.ndarray:
    """The slide (du, dv) as a move on the ground at the points' heights, in
    metres (east, north in the tileset frame): each point's pixel moved by the
    slide, cast back to its own height, averaged."""
    K, R, C = cam["K"], cam["R_cw"], cam["C"]
    pc = (X - C) @ R.T
    u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2] + du
    v = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2] + dv
    rays = np.column_stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)]) @ R   # camera -> world
    t = (X[:, 2] - C[2]) / rays[:, 2]
    moved = C[None, :] + t[:, None] * rays
    return np.median(moved[:, :2] - X[:, :2], axis=0)


def measure_building(job) -> List[dict]:
    import cv2
    (b, frames, cache, index, args) = job
    if not (b.dir / "lidar_local.laz").exists():
        print(f"  {b.name}: no lidar_local.laz, skipped", flush=True)
        return []
    cams = {c["id"]: c for c in wv.cameras_in_tileset_frame(frames, b)}
    local = step_tops(b.dir / "lidar_local.laz", args.cell_m, args.step_m, args.max_points)
    X = local @ b.M[:3, :3].T + b.M[:3, 3]
    out = []
    for k, w in index.items():
        if w["building"] != b.key or w["frame"] not in cams:
            continue
        c = cams[w["frame"]]
        pc = (X - c["C"]) @ c["R_cw"].T
        u = c["K"][0, 0] * pc[:, 0] / pc[:, 2] + c["K"][0, 2] - w["x0"]
        v = c["K"][1, 1] * pc[:, 1] / pc[:, 2] + c["K"][1, 2] - w["y0"]
        inside = (pc[:, 2] > 0) & (u > args.max_px) & (v > args.max_px) & (u < w["w"] - args.max_px) & (v < w["h"] - args.max_px)
        if inside.sum() < args.min_points:
            continue
        img = cv2.imread(str(cache / k), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        du, dv, clear, edge = best_slide(img, u[inside], v[inside], args.max_px)
        gsd = float(np.median(pc[inside, 2]) / c["K"][0, 0])
        east, north = ground_shift(c, X[inside], du, dv, w["x0"], w["y0"])
        out.append(dict(building=b.key, name=b.name, frame=w["frame"].split("/")[-1], head=w["frame"].split("/")[-1].split("_")[0],
                        points=int(inside.sum()), du_px=round(du, 2), dv_px=round(dv, 2), slide_m=round(float(np.hypot(du, dv) * gsd), 3),
                        east_m=round(float(east), 3), north_m=round(float(north), 3), gsd_m=round(gsd, 4),
                        clear=round(clear, 2), at_limit=edge))
    return out


def cmd_measure(a) -> int:
    import multiprocessing as mp
    blds = wv.reference_buildings(a.reference, a.landmarks, wv.tileset_transforms(a.tileset), a.buildings)
    frames = wv.read_frames(a.frames)
    index = json.loads((a.cache / "index.json").read_text(encoding="utf-8"))
    jobs = [(b, frames, a.cache, index, a) for b in blds]
    out, t0 = [], time.time()
    with mp.Pool(a.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(measure_building, jobs)):
            out += res
            print(f"  {k + 1}/{len(jobs)} buildings, {len(out)} frame windows ({time.time() - t0:.0f} s)", flush=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    good = [r for r in out if r["clear"] >= a.min_clear and not r["at_limit"]]
    print(f"{len(good)} of {len(out)} windows with a clear slide")
    for hd in sorted({r["head"] for r in good}):
        g = [r for r in good if r["head"] == hd]
        e, n, s = (np.array([r[k] for r in g]) for k in ("east_m", "north_m", "slide_m"))
        print(f"  {hd:5s} {len(g):3d} windows: median move east {np.median(e):+.2f} m, north {np.median(n):+.2f} m; "
              f"size of the slide median {np.median(s):.2f} m (90% under {np.percentile(s, 90):.2f} m)")
    print(f"-> {a.out}")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("measure", help="every cached frame window's slide onto the LiDAR's height steps")
    p.add_argument("--reference", type=Path, default=ROOT / "configs" / "audit_reference.yaml")
    p.add_argument("--landmarks", type=Path, default=ROOT / "configs" / "landmarks.yaml")
    p.add_argument("--tileset", type=Path, default=ROOT / "outputs" / "downtown")
    p.add_argument("--buildings", type=Path, default=ROOT / "outputs" / "downtown" / "buildings")
    p.add_argument("--frames", type=Path, default=ROOT / "outputs" / "downtown" / "frames_2024s1.gpkg")
    p.add_argument("--cache", type=Path, default=ROOT / "data" / "derived" / "frames_cache")
    p.add_argument("--cell-m", type=float, default=0.5, help="the LiDAR height grid's cell")
    p.add_argument("--step-m", type=float, default=0.5, help="a drop this big to a neighbouring cell is a height step")
    p.add_argument("--max-points", type=int, default=6000)
    p.add_argument("--min-points", type=int, default=300)
    p.add_argument("--max-px", type=int, default=30, help="the largest slide tried, pixels each way (about 2.3 m)")
    p.add_argument("--min-clear", type=float, default=1.5, help="a slide's score over the median slide's, to count")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", type=Path, default=REF / "camera_check.json")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = parse_args(argv)
    return dict(measure=cmd_measure)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
