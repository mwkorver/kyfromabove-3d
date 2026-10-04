#!/usr/bin/env python3
"""Why a wall produced the points it did, tile by tile.

:mod:`colorize_building` reports a count -- "18 tiles fitted, 10 too far from a
plane" -- which does not say whether a wall came out sparse because the building
is complicated or because the LiDAR never reached it. Those want opposite
responses: the first is a limit of the method, the second is a limit of the
capture, and only one of them is worth trying to fix.

Each wall is drawn flat, in metres along and up, with its returns and the
per-tile plane residual. A tile is green where the returns lie on a plane, red
where they do not, and outlined in grey with a bare count where there were too
few returns to judge.

On the Humana Building this settles the question. The gap in its resampled wall
is not the fit refusing: the left third of that face carries **zero returns
above 36 m**, and where returns do exist the tiles fit to 0.12-0.25 m, well
inside the 0.35 m cut. What the fit actually refused is the podium band at the
bottom, where the surface genuinely steps.

Usage::

    python scripts/diagnose_wall_fit.py --building outputs/humana/humana \
        --gers d2786314-4fb3-4da6-8f81-621873d40c08
"""

from __future__ import annotations

import argparse

import json, csv, sys, numpy as np, matplotlib
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer
from shapely.wkt import loads
from shapely.ops import transform as st
from shapely.geometry import Polygon
from shapely import contains_xy
from kyfromabove_3d.clouds import find_cloud, read_cloud
from kyfromabove_3d.features import split_surface_zones, facade_faces
from kyfromabove_3d.coordinates import METERS_PER_US_SURVEY_FOOT as M
csv.field_size_limit(1 << 30)

_ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--building", required=True)
_ap.add_argument("--gers", required=True)
_ap.add_argument("--footprint-csv", default="data/derived/ky_facade_tiled_wkt.csv")
_ap.add_argument("--tile-m", type=float, default=12.0)
_ap.add_argument("--max-plane-rms-m", type=float, default=0.35)
_ap.add_argument("--faces", type=int, default=2,
                 help="how many of the best-sampled walls to draw")
_ap.add_argument("--out", default=None)
_args = _ap.parse_args()
b, gers = _args.building, _args.gers
OUT = _args.out or f"{b}/wall_fit.png"
TILE, MAXRMS = _args.tile_m, _args.max_plane_rms_m
ox, oy, oz = json.load(open(f"{b}/coordinates.json"))["origin_source_xyz"]
ply = read_cloud(find_cloud(Path(b), "lidar_local"))
xyz = np.column_stack([ply["x"], ply["y"], ply["z"]]).astype(float)
wk = {r["id"]: r["wkt"] for r in csv.DictReader(
    open(_args.footprint_csv, newline="", encoding="utf-8"))}
to_ft = Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True).transform
ring = np.array(st(to_ft, loads(wk[gers])).exterior.coords)
ring_m = np.column_stack([(ring[:, 0]-ox)*M, (ring[:, 1]-oy)*M])
on = contains_xy(Polygon(ring_m).buffer(2.0), xyz[:, 0], xyz[:, 1])
bx = xyz[on]; z = split_surface_zones(bx)
h = float(np.percentile(bx[:, 2], 99) - bx[:, 2].min()); w = bx[z.wall]
faces = facade_faces(ring_m, height_m=h, wall_xy=w[:, :2], min_length_m=4.0, max_offset_m=2.0)
v = ring_m[:-1] if np.allclose(ring_m[0], ring_m[-1]) else ring_m
if np.sum((np.roll(v[:,0],-1)-v[:,0])*(np.roll(v[:,1],-1)+v[:,1])) > 0: v = v[::-1]

big = sorted(faces, key=lambda f: -f.n_returns)[:_args.faces]
fig, axes = plt.subplots(1, len(big), figsize=(7.2*len(big), 7.4), squeeze=False)
for ax, f in zip(axes[0], big):
    p0, p1 = v[f.index], v[(f.index+1) % len(v)]
    seg = p1-p0; L = np.hypot(*seg)
    t = np.clip(((w[:, :2]-p0) @ seg)/max(L**2, 1e-9), 0, 1); cl = p0 + t[:, None]*seg
    pts = w[np.linalg.norm(w[:, :2]-cl, axis=1) <= 2.0]
    e1 = np.array([seg[0]/L, seg[1]/L, 0.0]); e2 = np.array([0.,0.,1.])
    origin = np.array([p0[0], p0[1], pts[:,2].min()])
    rel = pts-origin; a, bb = rel@e1, rel@e2
    ax.scatter(a, bb, s=2.0, c="#8fa3b8", lw=0, label="LiDAR wall return")
    kept = skipped = 0
    for ai in np.arange(a.min(), a.max(), TILE):
        for bi in np.arange(bb.min(), bb.max(), TILE):
            m = (a>=ai)&(a<ai+TILE)&(bb>=bi)&(bb<bi+TILE)
            n = int(m.sum())
            if n < 40:
                ax.add_patch(plt.Rectangle((ai,bi),TILE,TILE,fill=False,
                             edgecolor="#444",lw=0.8,ls=":"))
                ax.text(ai+TILE/2, bi+TILE/2, f"{n}", ha="center", va="center",
                        fontsize=7, color="#666"); continue
            tp = pts[m]; c = tp.mean(0)
            nrm = np.linalg.svd(tp-c, full_matrices=False)[2][2]
            rms = float(np.sqrt(np.mean(((tp-c)@nrm)**2)))
            ok = rms <= MAXRMS
            kept += ok; skipped += not ok
            ax.add_patch(plt.Rectangle((ai,bi),TILE,TILE,
                         facecolor=("#2a9d8f" if ok else "#d1495b"), alpha=0.22,
                         edgecolor=("#2a9d8f" if ok else "#d1495b"), lw=1.3))
            ax.text(ai+TILE/2, bi+TILE/2, f"{rms:.2f}", ha="center", va="center",
                    fontsize=8, color=("#1d6f66" if ok else "#a03445"), fontweight="bold")
    ax.set_title(f"face {f.index} — {f.length_m:.0f} m wide, {len(pts):,} returns\n"
                 f"{kept} tiles fitted (green), {skipped} too far from a plane (red)",
                 fontsize=11)
    ax.set_xlabel("metres along the wall"); ax.set_ylabel("metres up the wall")
    ax.set_aspect("equal"); ax.grid(alpha=0.15)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
axes[0][0].legend(fontsize=9, markerscale=4, loc="upper left", frameon=False)
fig.suptitle("Humana Building: what the tiled plane fit accepted, and what it refused\n"
             f"numbers are each tile's plane-fit residual in metres; the cut is {MAXRMS} m",
             fontsize=12.5)
fig.tight_layout(rect=[0,0,1,0.93])
fig.savefig(OUT, dpi=125, bbox_inches="tight")
print("figure:", OUT)
