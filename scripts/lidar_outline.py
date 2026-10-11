#!/usr/bin/env python3
"""Each building's outline from the LiDAR, with Overture's footprint only as the seed.

The building's roof is found in the LiDAR itself: a 0.5 m grid of the highest
non-ground returns, grown from Overture's polygon (shrunk 1 m, so it starts
on roof) through cells at least 2.5 m up whose height changes by less than
1.5 m from cell to cell, so it stops where the roof drops to the street or to
a lower neighbour. It never enters another building's Overture polygon (a
party wall stays Overture's line) and never runs more than 3 m past this one.

That roof is then turned into an outline two ways:

  traced   the roof's own edge, simplified: the LiDAR's shape, whatever it is
  edges    Overture's edges, each moved along its own perpendicular to where
           the roof ends (the median along the edge), the corners
           re-intersected: the same faces, at the LiDAR's positions; an edge
           with a neighbour's polygon just outside it stays put (a party
           wall), and one the roof does not reach along most of its length
           stays put too

``measure`` writes ``lidar_outline.json`` (per building: the three outlines,
their areas, how much of the LiDAR roof each covers, and per edge how far it
moved and why) and ``lidar_outline.png`` (a plan of every building). Nothing
in the pipeline reads it yet.

Usage::

    python scripts/lidar_outline.py measure
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wall_visibility as wv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "outputs" / "wall_visibility" / "reference"
FOOTPRINTS = ROOT / "data" / "derived" / "louisville_downtown_overture_split.json"
NOT_ROOF_CLASSES = (7, 18)             # low and high noise; ground stays, so the street reads as street, not as a gap


def load_building(b, footprints_path: Path = FOOTPRINTS):
    """Everything this script needs from a building's folder, in either
    layout -- the downtown run's (``footprint.json``, ``lidar_local.laz``) or
    the older cases' (``coordinates.json``, ``lidar_local.ply``) -- with the
    footprint, when the folder has none, from the downtown Overture file by
    GERS id. Returns a dict: origin (x, y), wkt, height_m (LiDAR), xyz, cls,
    ground_z (the local height of the street: 0 in the downtown frame, the
    median ground return otherwise), or None with a reason."""
    if (b.dir / "footprint.json").exists():
        fp = json.loads((b.dir / "footprint.json").read_text(encoding="utf-8"))
        origin, wkt, height = fp["origin_source_xyz"][:2], fp["wkt"], fp.get("height_m")
    else:
        if not (b.dir / "coordinates.json").exists():
            return None, "no coordinates.json"
        origin = json.loads((b.dir / "coordinates.json").read_text(encoding="utf-8"))["origin_source_xyz"][:2]
        rec = next((r for r in json.loads(footprints_path.read_text(encoding="utf-8"))["buildings"] if r["id"] == b.gers), None)
        if rec is None:
            return None, "not in the Overture file"
        wkt, height = rec["wkt"], float(rec["lidar_height_m"]) if rec.get("lidar_height_m") not in (None, "None") else None
    if (b.dir / "lidar_local.laz").exists():
        import laspy
        las = laspy.read(b.dir / "lidar_local.laz")
        xyz, cls = np.column_stack([las.x, las.y, las.z]), np.asarray(las.classification)
    elif (b.dir / "lidar_local.ply").exists():
        from kyfromabove_3d.geometry import read_ply
        p = read_ply(b.dir / "lidar_local.ply")
        xyz, cls = np.column_stack([p["x"], p["y"], p["z"]]), np.asarray(p["classification"])
    else:
        return None, "no local LiDAR"
    ground = 0.0 if (b.dir / "footprint.json").exists() else float(np.median(xyz[cls == 2, 2])) if (cls == 2).any() else float(np.percentile(xyz[:, 2], 2))
    return dict(origin=origin, wkt=wkt, height_m=height, xyz=xyz, cls=cls, ground_z=ground), None


def roof_grid(xyz, cls, x0, y0, nx, ny, cell):
    """The highest return in each cell (noise left out); an empty cell takes its
    nearest filled neighbour's height within 1 m (the LiDAR is ~3 returns a
    square metre, so a 0.5 m grid has gaps), else NaN."""
    from scipy.ndimage import distance_transform_edt
    keep = ~np.isin(cls, NOT_ROOF_CLASSES)
    x, y, z = xyz[keep, 0], xyz[keep, 1], xyz[keep, 2]
    ix, iy = ((x - x0) / cell).astype(int), ((y - y0) / cell).astype(int)
    ok = (ix >= 0) & (iy >= 0) & (ix < nx) & (iy < ny)
    g = np.full((ny, nx), -np.inf)
    np.maximum.at(g, (iy[ok], ix[ok]), z[ok])
    empty = ~np.isfinite(g)
    d, (jy, jx) = distance_transform_edt(empty, return_indices=True)
    g = np.where(empty & (d * cell <= 1.0), g[jy, jx], g)
    return np.where(np.isfinite(g), g, np.nan)


def grow(g, seed, allowed, max_step_m):
    """Cells reachable from ``seed`` through ``allowed`` cells, each step
    changing height by less than ``max_step_m``."""
    out = seed & allowed
    q = deque(zip(*np.nonzero(out)))
    ny, nx = g.shape
    while q:
        j, i = q.popleft()
        h = g[j, i]
        for dj, di in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            jj, ii = j + dj, i + di
            if 0 <= jj < ny and 0 <= ii < nx and allowed[jj, ii] and not out[jj, ii] and abs(g[jj, ii] - h) < max_step_m:
                out[jj, ii] = True
                q.append((jj, ii))
    return out


def mask_polygon(mask, x0, y0, cell, simplify_m):
    """The mask's largest piece as a polygon in metres, simplified."""
    import rasterio.features
    from affine import Affine
    from shapely.geometry import shape
    tr = Affine(cell, 0, x0, 0, cell, y0)
    polys = [shape(geom) for geom, v in rasterio.features.shapes(mask.astype(np.uint8), mask=mask, transform=tr) if v == 1]
    if not polys:
        return None
    p = max(polys, key=lambda p: p.area)
    return p.simplify(simplify_m, preserve_topology=True)


def outward(P: np.ndarray) -> float:
    """+1 when the ring (without its closing point) runs counter-clockwise,
    so an edge's outward side is right of its direction of travel."""
    return 1.0 if np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]) > 0 else -1.0


def outline_from_moves(P: np.ndarray, moves, max_corner_move_m: float) -> np.ndarray:
    """The ring with each edge k of ``P`` (corners, no closing point) moved
    ``moves[k]`` metres outward along its perpendicular, corners where
    neighbouring moved edges meet (a corner that would move more than
    ``max_corner_move_m`` stays). Returns the closed ring."""
    n, sgn = len(P), outward(P)
    lines = []
    for k in range(n):
        p, q = P[k], P[(k + 1) % n]
        d = (q - p) / max(np.linalg.norm(q - p), 1e-9)
        lines.append((p + moves[k] * sgn * np.array([d[1], -d[0]]), d))
    new = []
    for k in range(n):
        (a0, d0), (a1, d1) = lines[k - 1], lines[k]
        A = np.column_stack([d0, -d1])
        if abs(np.linalg.det(A)) < np.sin(np.radians(10)):              # nearly straight on: no sharp corner to find
            c = 0.5 * (a1 + (a0 + d0 * ((P[k] - a0) @ d0)))
        else:
            s_, _ = np.linalg.solve(A, a1 - a0)
            c = a0 + s_ * d0
        if np.linalg.norm(c - P[k]) > max_corner_move_m:
            c = P[k]
        new.append(c)
    new.append(new[0])
    return np.array(new)


def move_edges(ring, inside, height_at, others, a):
    """Overture's edges moved to where the roof ends. ``inside(xy)`` says
    whether points are on the LiDAR roof, ``height_at(xy)`` the LiDAR's
    height there. Walking out from the roof along each edge's perpendicular,
    gaps up to ``--gap-m`` are stepped over (a patchy roof); the roof's end
    counts only where the height then drops ``--max-step-m`` or more, so a
    neighbour's footprint, which the roof is not grown into, is not taken for
    an edge. An edge with a neighbour just outside and no drop is a party
    wall and stays put. Returns the new ring and, per edge, its move and why."""
    from shapely.geometry import Point
    P = np.asarray(ring[:-1], float)
    n = len(P)
    area2 = np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    sgn = 1.0 if area2 > 0 else -1.0                                   # counter-clockwise: outward is right of travel
    step = 0.1
    offs = np.arange(-a.reach_m, a.reach_m + 1e-9, step)
    gap = int(round(a.gap_m / step))
    edges = []
    for k in range(n):
        p, q = P[k], P[(k + 1) % n]
        L = float(np.linalg.norm(q - p))
        d = (q - p) / max(L, 1e-9)
        nrm = sgn * np.array([d[1], -d[0]])
        rec = dict(edge=k, length_m=round(L, 2))
        if L < 2.0:
            edges.append(dict(rec, status="too short", move_m=0.0, support=0.0))
            continue
        t = np.arange(min(1.0, L / 4), L - min(1.0, L / 4) + 1e-9, 0.5)
        S = p[None, :] + t[:, None] * d[None, :]
        neighbour = others is not None and np.mean([others.contains(Point(*xy)) for xy in S + 1.0 * nrm[None, :]]) >= 0.5
        moves = []
        for s in S:
            line = s[None, :] + offs[:, None] * nrm[None, :]
            on = inside(line)
            if not on.any():
                continue
            k2 = int(np.argmax(on))                                      # the first roof cell from the inner end
            while True:                                                  # walk out, stepping over short gaps
                nxt = np.flatnonzero(on[k2 + 1:k2 + 2 + gap])
                if not len(nxt):
                    break
                k2 = k2 + 1 + int(nxt[-1]) if on[k2 + 1 + int(nxt[-1])] else k2 + 1
                if k2 + 1 >= len(on):
                    break
            if k2 + 1 >= len(on):                                        # roof runs past the reach: no edge here
                continue
            h_in = height_at(line[k2:k2 + 1])[0]
            beyond = height_at(line[k2 + 5:k2 + 16])                     # 0.5-1.5 m past the roof's last cell
            h_out = float(np.nanmedian(beyond)) if np.isfinite(beyond).any() else -99.0
            if np.isfinite(h_out) and h_in - h_out < a.max_step_m:       # no drop: the roof was stopped, not ended
                continue
            moves.append(offs[k2] + step / 2)
        support = len(moves) / len(S)
        if support < a.min_support:
            edges.append(dict(rec, status="party wall" if neighbour else "roof does not reach it", move_m=0.0, support=round(support, 2)))
            continue
        m = float(np.median(moves))
        edges.append(dict(rec, status="moved", move_m=round(m, 2), support=round(support, 2),
                          spread_m=round(float(np.percentile(np.abs(np.array(moves) - m), 75)), 2)))

    def corners():
        return outline_from_moves(P, [e["move_m"] for e in edges], a.max_corner_move_m)
    from shapely.geometry import Polygon
    new = corners()
    while not Polygon(new).is_valid:                                     # crossed itself: undo the biggest move, again
        moved = [e for e in edges if e["status"] == "moved"]
        if not moved:
            break
        worst = max(moved, key=lambda e: abs(e["move_m"]))
        worst.update(status="undone: the outline crossed itself", undone_move_m=worst["move_m"], move_m=0.0)
        new = corners()
    return new, edges


def measure_building(job) -> Optional[dict]:
    from shapely import contains_xy
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    from kyfromabove_3d.footprints import footprint_wgs84_to_local
    b, recs, a = job
    B, why = load_building(b, a.footprints)
    if B is None:
        return dict(building=b.key, name=b.name, status=why)
    ox, oy = B["origin"]
    ov = footprint_wgs84_to_local(B["wkt"], ox, oy)
    if ov.geom_type != "Polygon":
        ov = max(ov.geoms, key=lambda g: g.area)
    others = []
    for r in recs:
        if r["id"] == b.gers:
            continue
        g = footprint_wgs84_to_local(r["wkt"], ox, oy)
        if not g.is_valid:
            g = g.buffer(0)
        if g.distance(ov) < 10 and not g.equals(ov) and g.intersection(ov).area < 0.5 * min(g.area, ov.area):
            others.append(g.buffer(-0.5))
    others_u = unary_union(others) if others else None
    xyz, cls = B["xyz"], B["cls"]
    c = a.cell_m
    minx, miny, maxx, maxy = ov.bounds
    x0, y0 = minx - 8, miny - 8
    nx, ny = int((maxx + 8 - x0) / c) + 1, int((maxy + 8 - y0) / c) + 1
    g = roof_grid(xyz, cls, x0, y0, nx, ny, c)
    cx, cy = np.meshgrid(x0 + (np.arange(nx) + 0.5) * c, y0 + (np.arange(ny) + 0.5) * c)
    up = np.nan_to_num(g, nan=-999) >= B["ground_z"] + a.min_height_m
    seed = up & contains_xy(ov.buffer(-1.0), cx, cy)
    allowed = up & contains_xy(ov.buffer(a.reach_m), cx, cy)
    if others_u is not None:
        allowed &= ~contains_xy(others_u, cx, cy)
    if seed.sum() < 20:
        return dict(building=b.key, name=b.name, status="too little roof in the LiDAR")
    roof = grow(np.nan_to_num(g, nan=-99), seed, allowed, a.max_step_m)
    from scipy.ndimage import binary_fill_holes, binary_opening
    roof = binary_opening(roof, iterations=1) | seed                     # trim one-cell spurs (a lamp, a branch)
    holes = binary_fill_holes(roof) & ~roof
    from scipy.ndimage import label
    lab, nlab = label(holes)
    small = np.isin(lab, [i for i in range(1, nlab + 1) if (lab == i).sum() * c * c < 4.0])
    roof = roof | small
    traced = mask_polygon(roof, x0, y0, c, a.simplify_m)

    def inside(xy):
        i = ((xy[:, 0] - x0) / c).astype(int)
        j = ((xy[:, 1] - y0) / c).astype(int)
        ok = (i >= 0) & (j >= 0) & (i < nx) & (j < ny)
        out = np.zeros(len(xy), bool)
        out[ok] = roof[j[ok], i[ok]]
        return out
    ring = np.asarray(ov.exterior.coords)
    def height_at(xy):
        i = ((xy[:, 0] - x0) / c).astype(int)
        j = ((xy[:, 1] - y0) / c).astype(int)
        ok = (i >= 0) & (j >= 0) & (i < nx) & (j < ny)
        out = np.full(len(xy), np.nan)
        out[ok] = g[j[ok], i[ok]]
        return out
    new_ring, edges = move_edges(ring, inside, height_at, others_u, a)
    new = Polygon(new_ring)
    roof_poly = traced if traced is not None else Polygon()
    roof_area = float(roof.sum() * c * c)

    def cover(p):
        if not p.is_valid:
            p = p.buffer(0)
        return dict(area_m2=round(p.area, 1), iou=round(p.intersection(roof_poly).area / max(p.union(roof_poly).area, 1e-9), 3),
                    roof_outside_m2=round(roof_poly.difference(p).area, 1), not_roof_inside_m2=round(p.difference(roof_poly).area, 1))
    return dict(building=b.key, name=b.name, status="ok", origin=[ox, oy], ground_z=round(B["ground_z"], 2), roof_area_m2=round(roof_area, 1),
                overture=dict(ring=np.round(ring, 3).tolist(), **cover(ov)),
                edges_moved=dict(ring=np.round(new_ring, 3).tolist(), **cover(new)),
                traced=dict(ring=np.round(np.asarray(traced.exterior.coords), 3).tolist() if traced is not None else None,
                            area_m2=round(roof_poly.area, 1)),
                edges=edges, neighbours=len(others),
                grid=dict(x0=x0, y0=y0, cell=c, shape=[ny, nx]))


def cmd_measure(a) -> int:
    import multiprocessing as mp
    blds = wv.reference_buildings(a.reference, a.landmarks, wv.tileset_transforms(a.tileset), a.buildings)
    recs = json.loads(a.footprints.read_text(encoding="utf-8"))["buildings"]
    out, t0 = [], time.time()
    with mp.Pool(a.workers) as pool:
        for res in pool.imap_unordered(measure_building, [(b, recs, a) for b in blds]):
            out.append(res)
    out.sort(key=lambda r: r["name"])
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    ok = [r for r in out if r["status"] == "ok"]
    print(f"{len(ok)} of {len(out)} buildings outlined ({time.time() - t0:.0f} s); skipped: "
          f"{[(r['name'], r['status']) for r in out if r['status'] != 'ok']}")
    from collections import Counter
    st = Counter(e["status"] for r in ok for e in r["edges"])
    print("edges:", dict(st))
    mv = np.array([e["move_m"] for r in ok for e in r["edges"] if e["status"] == "moved"])
    print(f"moved edges: median move {np.median(mv):+.2f} m (+ out), size median {np.median(np.abs(mv)):.2f} m, over 0.5 m on "
          f"{np.mean(np.abs(mv) > 0.5):.0%}, over 1 m on {np.mean(np.abs(mv) > 1):.0%}")
    for k in ("overture", "edges_moved"):
        print(f"{k:12s}: share of the LiDAR roof matched (overlap over union) median {np.median([r[k]['iou'] for r in ok]):.3f}; "
              f"roof left outside {sum(r[k]['roof_outside_m2'] for r in ok):.0f} m2, non-roof inside {sum(r[k]['not_roof_inside_m2'] for r in ok):.0f} m2")
    print(f"-> {a.out}")
    return 0


def cmd_plot(a) -> int:
    """A plan of every building: the LiDAR's heights, Overture's footprint
    (teal, dashed), Overture's edges moved to the roof's end (orange) and the
    roof's own traced edge (white)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in json.loads(a.out.read_text(encoding="utf-8")) if r["status"] == "ok"]
    if a.only:
        rows = [r for r in rows if any(r["building"].startswith(k) for k in a.only)]
    blds = {b.key: b for b in wv.reference_buildings(a.reference, a.landmarks, wv.tileset_transforms(a.tileset), a.buildings)}
    n = len(rows)
    cols = min(n, a.cols)
    fig, axs = plt.subplots((n + cols - 1) // cols, cols, figsize=(5.2 * cols, 5.0 * ((n + cols - 1) // cols)), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, r in zip(axs.ravel(), rows):
        b = blds[r["building"]]
        B, _ = load_building(b)
        G = r["grid"]
        ny, nx = G["shape"]
        g = roof_grid(B["xyz"], B["cls"], G["x0"], G["y0"], nx, ny, G["cell"])
        ext = [G["x0"], G["x0"] + nx * G["cell"], G["y0"], G["y0"] + ny * G["cell"]]
        ax.imshow(np.ma.masked_invalid(g), origin="lower", extent=ext, cmap="Greys_r", vmin=B["ground_z"], vmax=float(np.nanpercentile(g, 99)))
        ov, mv = np.array(r["overture"]["ring"]), np.array(r["edges_moved"]["ring"])
        ax.plot(ov[:, 0], ov[:, 1], color="#3cc2bb", lw=2, ls=(0, (4, 2)))
        if r["traced"]["ring"]:
            tr = np.array(r["traced"]["ring"])
            ax.plot(tr[:, 0], tr[:, 1], color="white", lw=0.9)
        ax.plot(mv[:, 0], mv[:, 1], color="#f0a020", lw=1.8)
        for e in r["edges"]:
            if e["status"] == "party wall":
                k = e["edge"]
                ax.plot(ov[k:k + 2, 0], ov[k:k + 2, 1], color="#ff5ad2", lw=3)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
        moved = [abs(e["move_m"]) for e in r["edges"] if e["status"] == "moved"]
        ax.set_title(f'{r["name"][:30]}\nedges moved {len(moved)}, typically {np.median(moved) if moved else 0:.2f} m; '
                     f'match to roof {r["overture"]["iou"]:.2f} -> {r["edges_moved"]["iou"]:.2f}', fontsize=9.5)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color="#3cc2bb", lw=2, ls=(0, (4, 2))), Line2D([], [], color="#f0a020", lw=2),
                Line2D([], [], color="black", lw=1), Line2D([], [], color="#ff5ad2", lw=3)],
               ["Overture footprint", "Overture's edges moved to where the LiDAR roof ends", "the LiDAR roof's own edge, traced (white)",
                "party wall: left on Overture's line"], loc="upper center", ncol=4, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(a.plot_out, dpi=a.dpi)
    print(f"{n} buildings -> {a.plot_out}")
    return 0


def cmd_sheet(a) -> int:
    """Overture's footprint (teal) and the moved edges (orange) drawn on one
    frame from each camera head, each corner at the LiDAR roof's height
    just inside it, for the buildings named."""
    import cv2
    rows = {r["building"]: r for r in json.loads(a.out.read_text(encoding="utf-8")) if r["status"] == "ok"}
    blds = {b.key: b for b in wv.reference_buildings(a.reference, a.landmarks, wv.tileset_transforms(a.tileset), a.buildings)}
    frames = wv.read_frames(a.frames)
    index = json.loads((a.cache / "index.json").read_text(encoding="utf-8"))
    TEAL, ORANGE, H = (199, 194, 60), (32, 160, 240), 330
    out_rows = []
    for key in [k for k in rows if any(k.startswith(o) for o in a.only)]:
        r, b = rows[key], blds[key]
        B, _ = load_building(b)
        G = r["grid"]
        ny, nx = G["shape"]
        g = roof_grid(B["xyz"], B["cls"], G["x0"], G["y0"], nx, ny, G["cell"])

        def roof_line(ring):
            """The ring sampled every 0.5 m, each sample at the LiDAR roof's
            height just inside it (the highest cell 0.5-2 m in), as 3D points."""
            P = np.asarray(ring, float)
            area2 = np.sum(P[:-1, 0] * P[1:, 1] - P[1:, 0] * P[:-1, 1])
            sgn = 1.0 if area2 > 0 else -1.0
            pts = []
            for k in range(len(P) - 1):
                p0, p1 = P[k], P[k + 1]
                L = np.linalg.norm(p1 - p0)
                d = (p1 - p0) / max(L, 1e-9)
                inward = -sgn * np.array([d[1], -d[0]])
                for t in np.arange(0, L, 0.5):
                    q0 = p0 + t * d
                    vals = []
                    for f in (0.5, 1.0, 1.5, 2.0):
                        q = q0 + f * inward
                        i, j = int((q[0] - G["x0"]) / G["cell"]), int((q[1] - G["y0"]) / G["cell"])
                        if 0 <= i < nx and 0 <= j < ny and np.isfinite(g[j, i]):
                            vals.append(g[j, i])
                    pts.append([q0[0], q0[1], max(vals) if vals else np.nan])
            pts.append(pts[0])
            pts = np.array(pts)
            pts[:, 2] = np.where(np.isfinite(pts[:, 2]), pts[:, 2], np.nanmedian(pts[:, 2]))
            return pts
        rings3 = [(roof_line(r["overture"]["ring"]) @ b.M[:3, :3].T + b.M[:3, 3], TEAL),
                  (roof_line(r["edges_moved"]["ring"]) @ b.M[:3, :3].T + b.M[:3, 3], ORANGE)]
        cams = {c["id"]: c for c in wv.cameras_in_tileset_frame(frames, b)}
        tiles, seen = [], set()
        for kk, w in sorted(index.items()):
            hd = w["frame"].split("/")[-1].split("_")[0]
            if w["building"] != b.key or w["frame"] not in cams or hd in seen:
                continue
            c = cams[w["frame"]]
            uvs = []
            for X, col in rings3:
                pc = (X - c["C"]) @ c["R_cw"].T
                uvs.append((np.column_stack([c["K"][0, 0] * pc[:, 0] / pc[:, 2] + c["K"][0, 2] - w["x0"],
                                             c["K"][1, 1] * pc[:, 1] / pc[:, 2] + c["K"][1, 2] - w["y0"]]), col))
            allp = np.vstack([u for u, _ in uvs])
            inside = (allp[:, 0] >= 0) & (allp[:, 1] >= 0) & (allp[:, 0] < w["w"]) & (allp[:, 1] < w["h"])
            if inside.mean() < 0.85:
                continue
            allp = allp[inside]
            img = cv2.imread(str(a.cache / kk))
            for u, col in uvs:
                cv2.polylines(img, [u.round().astype(np.int32)], False, col, 2, cv2.LINE_AA)
            x0, y0 = np.maximum(allp.min(0) - 40, 0).astype(int)
            x1, y1 = np.minimum(allp.max(0) + 40, [img.shape[1], img.shape[0]]).astype(int)
            crop = img[y0:y1, x0:x1]
            sc = H / crop.shape[0]
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * sc)), H), interpolation=cv2.INTER_AREA)
            lab = np.full((22, crop.shape[1], 3), 255, np.uint8)
            cv2.putText(lab, f"{hd} {w['frame'].split('/')[-1]}", (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
            tiles.append(np.vstack([crop, lab]))
            seen.add(hd)
        if not tiles:
            continue
        row = np.hstack([np.hstack([t, np.full((t.shape[0], 10, 3), 255, np.uint8)]) for t in tiles])
        cap = np.full((28, row.shape[1], 3), 255, np.uint8)
        cv2.putText(cap, f"{r['name']}: teal = Overture footprint, orange = edges moved to where the LiDAR roof ends",
                    (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        out_rows.append(np.vstack([cap, row, np.full((14, row.shape[1], 3), 255, np.uint8)]))
    Wd = max(x.shape[1] for x in out_rows)
    sheet = np.vstack([cv2.copyMakeBorder(x, 0, 0, 0, Wd - x.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x in out_rows])
    cv2.imwrite(str(a.sheet_out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(out_rows)} buildings -> {a.sheet_out} ({sheet.shape[1]} x {sheet.shape[0]})")
    return 0


def cmd_fetch_nadir(a) -> int:
    """Windows of the straight-down (Color) frames over each building's roof:
    the ``--per-building`` frames in which it sits closest to directly below,
    each window the building plus ``--margin-m`` at the ground and at its
    roof's height, read from the frame files on S3 window by window."""
    import cv2

    from kyfromabove_3d.footprints import footprint_wgs84_to_local
    from kyfromabove_3d.imagery import FrameImageReader, cog_url
    blds = wv.reference_buildings(a.reference, a.landmarks, wv.tileset_transforms(a.tileset), a.buildings)
    frames = wv.read_frames(a.frames)
    url = {str(f["ID"]): f["S3URL"] for f in frames}
    a.nadir_cache.mkdir(parents=True, exist_ok=True)
    index_path = a.nadir_cache / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    wins = []
    for b in blds:
        B, why = load_building(b)
        if B is None:
            print(f"  {b.name}: {why}, skipped", flush=True)
            continue
        ov = footprint_wgs84_to_local(B["wkt"], *B["origin"]).buffer(a.margin_m)
        ring = np.asarray(ov.exterior.coords)
        top = B["ground_z"] + float(B["height_m"] or 30.0) + 2.0
        X = np.vstack([np.column_stack([ring, np.full(len(ring), B["ground_z"])]), np.column_stack([ring, np.full(len(ring), top)])])
        Xw = X @ b.M[:3, :3].T + b.M[:3, 3]
        centre = b.M[:3, 3]
        cand = []
        for c in wv.cameras_in_tileset_frame(frames, b):
            if not c["id"].split("/")[-1].startswith("Color"):
                continue
            pc = (Xw - c["C"]) @ c["R_cw"].T
            if (pc[:, 2] <= 0).any():
                continue
            u = c["K"][0, 0] * pc[:, 0] / pc[:, 2] + c["K"][0, 2]
            v = c["K"][1, 1] * pc[:, 1] / pc[:, 2] + c["K"][1, 2]
            x0, y0 = int(max(0, np.floor(u.min()))), int(max(0, np.floor(v.min())))
            x1, y1 = int(min(c["W"], np.ceil(u.max()))), int(min(c["H"], np.ceil(v.max())))
            if x1 - x0 < 50 or y1 - y0 < 50:
                continue
            full = (x1 - x0) * (y1 - y0) / max((u.max() - u.min()) * (v.max() - v.min()), 1)
            ray = centre - c["C"]
            off = float(np.degrees(np.arccos(-ray[2] / np.linalg.norm(ray))))
            if full >= 0.9:                                             # the building and its margin all in the frame
                cand.append((off, dict(building=b.key, frame=c["id"], x0=x0, y0=y0, w=x1 - x0, h=y1 - y0, off_nadir_deg=round(off, 2))))
        cand.sort(key=lambda t: t[0])
        wins += [w for _, w in cand[:a.per_building]]
    todo = [w for w in wins if not (a.nadir_cache / w["building"] / (w["frame"].split("/")[-1] + ".png")).exists()]
    print(f"{len(wins)} windows over {len(blds)} buildings, {len(wins) - len(todo)} already here; fetching {len(todo)} "
          f"({sum(w['w'] * w['h'] for w in todo) / 1e6:.0f} Mpx)", flush=True)
    t0, got = time.time(), 0
    for k, w in enumerate(todo):
        img = FrameImageReader(cog_url(url[w["frame"]]), frame_id=w["frame"]).read_window(w["x0"], w["y0"], w["w"], w["h"])
        p = a.nadir_cache / w["building"] / (w["frame"].split("/")[-1] + ".png")
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), cv2.cvtColor(img.rgb, cv2.COLOR_RGB2BGR))
        index[str(p.relative_to(a.nadir_cache))] = w
        got += p.stat().st_size
        index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
        print(f"  {k + 1}/{len(todo)} windows, {got / 1e6:.0f} MB on disk, {time.time() - t0:.0f} s", flush=True)
    return 0


def true_ortho(b, r, cam, win, img, cell_m=0.1):
    """The straight-down frame resampled onto the LiDAR's surface (its
    highest-return grid) every ``cell_m``: the roof as seen from directly
    above, with no lean. Returns (image, extent)."""
    import cv2
    from scipy.ndimage import zoom
    B, _ = load_building(b)
    G = r["grid"]
    ny, nx = G["shape"]
    g = roof_grid(B["xyz"], B["cls"], G["x0"], G["y0"], nx, ny, G["cell"])
    g = np.where(np.isfinite(g), g, np.nanmedian(g))
    f = G["cell"] / cell_m
    gz = zoom(g, f, order=0)
    yy, xx = np.mgrid[0:gz.shape[0], 0:gz.shape[1]]
    X = np.column_stack([G["x0"] + (xx.ravel() + 0.5) * cell_m, G["y0"] + (yy.ravel() + 0.5) * cell_m, gz.ravel()])
    Xw = X @ b.M[:3, :3].T + b.M[:3, 3]
    pc = (Xw - cam["C"]) @ cam["R_cw"].T
    u = (cam["K"][0, 0] * pc[:, 0] / pc[:, 2] + cam["K"][0, 2] - win["x0"]).astype(np.float32).reshape(gz.shape)
    v = (cam["K"][1, 1] * pc[:, 1] / pc[:, 2] + cam["K"][1, 2] - win["y0"]).astype(np.float32).reshape(gz.shape)
    ortho = cv2.remap(img, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return ortho, [G["x0"], G["x0"] + gz.shape[1] * cell_m, G["y0"], G["y0"] + gz.shape[0] * cell_m]


def cmd_ortho(a) -> int:
    """Every building's roof from directly above (the straight-down frame
    nearest overhead, resampled onto the LiDAR's surface), with Overture's
    footprint (teal, dashed) and the moved edges (orange) on it."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in json.loads(a.out.read_text(encoding="utf-8")) if r["status"] == "ok"]
    if a.only:
        rows = [r for r in rows if any(r["building"].startswith(k) for k in a.only)]
    blds = {b.key: b for b in wv.reference_buildings(a.reference, a.landmarks, wv.tileset_transforms(a.tileset), a.buildings)}
    frames = wv.read_frames(a.frames)
    index = json.loads((a.nadir_cache / "index.json").read_text(encoding="utf-8"))
    cols = min(len(rows), a.cols)
    nrow = (len(rows) + cols - 1) // cols
    fig, axs = plt.subplots(nrow, cols, figsize=(a.size * cols, a.size * nrow), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, r in zip(axs.ravel(), rows):
        b = blds[r["building"]]
        mine = sorted(((w["off_nadir_deg"], k, w) for k, w in index.items() if w["building"] == b.key), key=lambda t: t[0])
        if not mine:
            continue
        off, k, w = mine[0]
        cam = {c["id"]: c for c in wv.cameras_in_tileset_frame(frames, b)}[w["frame"]]
        img = cv2.imread(str(a.nadir_cache / k))[:, :, ::-1]
        ortho, ext = true_ortho(b, r, cam, w, img)
        ax.imshow(ortho, origin="lower", extent=ext)
        ov, mv = np.array(r["overture"]["ring"]), np.array(r["edges_moved"]["ring"])
        ax.plot(ov[:, 0], ov[:, 1], color="#3cc2bb", lw=1.6, ls=(0, (4, 2)))
        ax.plot(mv[:, 0], mv[:, 1], color="#f0a020", lw=1.4)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
        ax.set_title(f'{r["name"][:30]} ({w["frame"].split("/")[-1]}, {off:.1f} deg off overhead)', fontsize=9)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color="#3cc2bb", lw=2, ls=(0, (4, 2))), Line2D([], [], color="#f0a020", lw=2)],
               ["Overture footprint", "Overture's edges moved to where the LiDAR roof ends"], loc="upper center", ncol=2, fontsize=11, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(a.ortho_out, dpi=a.dpi)
    print(f"{len(rows)} buildings -> {a.ortho_out}")
    return 0


ACROSS = np.arange(-1.0, 1.0001, 0.05)       # metres across the edge, + out from the moved line


def roof_height_along(g, G, S, inward, inside_m=(0.5, 1.0, 1.5, 2.0)):
    """The LiDAR roof's height just inside each point of ``S``: the highest
    grid cell ``inside_m`` metres in."""
    ny, nx = G["shape"]
    out = np.full(len(S), np.nan)
    for i, s_ in enumerate(S):
        vals = []
        for f in inside_m:
            q = s_ + f * inward
            ii, jj = int((q[0] - G["x0"]) / G["cell"]), int((q[1] - G["y0"]) / G["cell"])
            if 0 <= ii < nx and 0 <= jj < ny and np.isfinite(g[jj, ii]):
                vals.append(g[jj, ii])
        out[i] = max(vals) if vals else np.nan
    return out


def edge_in_frame(b, line_pts, nrm, heights, cam, win, gray, window_m, outer_share):
    """Where one straight-down frame puts the edge: the brightness profile
    across it (strip points ``ACROSS`` metres out from the moved line, each at
    the roof's height there), averaged along the edge; the edge is the
    outermost of its strong steps (``outer_share`` of the strongest) within
    ``window_m``. Returns (offset, clarity) or (None, 0)."""
    from scipy.ndimage import gaussian_filter1d
    na = len(line_pts)
    A_, O_ = np.meshgrid(np.arange(na), ACROSS, indexing="ij")
    xy = line_pts[A_.ravel()] + O_.ravel()[:, None] * nrm[None, :]
    X = np.column_stack([xy, heights[A_.ravel()]]) @ b.M[:3, :3].T + b.M[:3, 3]
    pc = (X - cam["C"]) @ cam["R_cw"].T
    u = cam["K"][0, 0] * pc[:, 0] / pc[:, 2] + cam["K"][0, 2] - win["x0"]
    v = cam["K"][1, 1] * pc[:, 1] / pc[:, 2] + cam["K"][1, 2] - win["y0"]
    from kyfromabove_3d.wall_match import bilinear_gray
    val, ok = bilinear_gray(gray, u, v)
    val, ok = val.reshape(na, len(ACROSS)), ok.reshape(na, len(ACROSS))
    rows = ok.all(axis=1)
    if rows.sum() < max(4, 0.5 * na):
        return None, 0.0
    prof = gaussian_filter1d(val[rows].mean(axis=0), 1.0)
    grad = np.abs(np.gradient(prof))
    inwin = np.abs(ACROSS) <= window_m + 1e-9
    gw = np.where(inwin, grad, 0.0)
    top = gw.max()
    if top <= 0:
        return None, 0.0
    peaks = [i for i in range(1, len(gw) - 1) if inwin[i] and gw[i] >= gw[i - 1] and gw[i] >= gw[i + 1] and gw[i] >= outer_share * top]
    i = max(peaks) if peaks else int(np.argmax(gw))
    # sub-sample: the peak of a parabola through it and its neighbours
    if 0 < i < len(gw) - 1:
        den = gw[i - 1] - 2 * gw[i] + gw[i + 1]
        frac = 0.5 * (gw[i - 1] - gw[i + 1]) / den if den < 0 else 0.0
    else:
        frac = 0.0
    clarity = float(top / max(np.median(grad), 1e-6))
    return float(ACROSS[i] + frac * (ACROSS[1] - ACROSS[0])), clarity


def refine_building(job) -> Optional[dict]:
    import cv2
    r, b, frames, cache, index, a = job
    B, _ = load_building(b)
    G = r["grid"]
    ny, nx = G["shape"]
    g = roof_grid(B["xyz"], B["cls"], G["x0"], G["y0"], nx, ny, G["cell"])
    cams = {c["id"]: c for c in wv.cameras_in_tileset_frame(frames, b)}
    wins = sorted(((w["off_nadir_deg"], k, w) for k, w in index.items() if w["building"] == b.key), key=lambda t: t[0])[:a.max_frames]
    grays = {k: cv2.imread(str(cache / k), cv2.IMREAD_GRAYSCALE).astype(np.float32) for _, k, _ in wins}
    P = np.asarray(r["edges_moved"]["ring"])[:-1]
    P0 = np.asarray(r["overture"]["ring"])[:-1]
    sgn = outward(P0)
    edges = []
    for e in r["edges"]:
        k = e["edge"]
        rec = dict(edge=k, outline_status=e["status"], lidar_move_m=e["move_m"])
        if e["status"] != "moved" or e["length_m"] < 3.0:
            edges.append(dict(rec, status="not refined (" + e["status"] + ")", refine_m=0.0))
            continue
        p, q = P[k], P[(k + 1) % len(P)]
        L = float(np.linalg.norm(q - p))
        d = (q - p) / max(L, 1e-9)
        nrm = sgn * np.array([d[1], -d[0]])
        t = np.arange(min(1.0, L / 4), L - min(1.0, L / 4) + 1e-9, 0.25)
        S = p[None, :] + t[:, None] * d[None, :]
        h = roof_height_along(g, G, S, -nrm)
        if not np.isfinite(h).any():
            edges.append(dict(rec, status="no roof height", refine_m=0.0))
            continue
        h = np.where(np.isfinite(h), h, np.nanmedian(h))
        per = {}
        for off, kk, w in wins:
            o, clarity = edge_in_frame(b, S, nrm, h, cams[w["frame"]], w, grays[kk], a.window_m, a.outer_share)
            if o is not None and clarity >= a.min_clarity:
                per[w["frame"].split("/")[-1]] = round(o, 3)
        vals = np.array(list(per.values()))
        if len(vals) < 2:
            edges.append(dict(rec, status="fewer than 2 frames see a clear edge", refine_m=0.0, frames=per))
            continue
        m = float(np.median(vals))
        spread = float(np.max(np.abs(vals - m)))
        at_limit = abs(m) >= a.window_m - 0.03
        ok = spread <= a.agree_m and not at_limit
        edges.append(dict(rec, status="refined" if ok else ("at the window's limit" if at_limit else "frames disagree"),
                          refine_m=round(m, 3) if ok else 0.0, frames_say_m=round(m, 3), spread_m=round(spread, 3), frames=per))
    moves = [e["lidar_move_m"] + e["refine_m"] for e in edges]
    ring = outline_from_moves(P0, moves, a.max_corner_move_m)
    from shapely.geometry import Polygon
    return dict(building=r["building"], name=r["name"], edges=edges, ring=np.round(ring, 3).tolist(),
                valid=bool(Polygon(ring).is_valid), frames_used=[w["frame"].split("/")[-1] for _, _, w in wins])


def cmd_refine(a) -> int:
    """Each moved edge refined in the straight-down frames (see edge_in_frame);
    writes lidar_outline_refined.json."""
    import multiprocessing as mp
    rows = [r for r in json.loads(a.out.read_text(encoding="utf-8")) if r["status"] == "ok"]
    blds = {b.key: b for b in wv.reference_buildings(a.reference, a.landmarks, wv.tileset_transforms(a.tileset), a.buildings)}
    frames = wv.read_frames(a.frames)
    index = json.loads((a.nadir_cache / "index.json").read_text(encoding="utf-8"))
    with mp.Pool(a.workers) as pool:
        out = list(pool.imap_unordered(refine_building, [(r, blds[r["building"]], frames, a.nadir_cache, index, a) for r in rows]))
    out.sort(key=lambda r: r["name"])
    a.refined_out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    from collections import Counter
    E = [e for r in out for e in r["edges"]]
    print(Counter(e["status"] if not e["status"].startswith("not refined") else "not refined" for e in E))
    ref = np.array([e["refine_m"] for e in E if e["status"] == "refined"])
    if len(ref):
        print(f"{len(ref)} edges refined: the frames move them a median {np.median(ref):+.2f} m (+ out) from the LiDAR's line; "
              f"size median {np.median(np.abs(ref)):.2f} m, over 0.25 m on {np.mean(np.abs(ref) > .25):.0%}")
    sp = np.array([e["spread_m"] for e in E if "spread_m" in e])
    print(f"frames' agreement on an edge (furthest frame from their median): median {np.median(sp):.2f} m; 75% under {np.percentile(sp, 75):.2f} m")
    print(f"outlines that cross themselves: {[r['name'] for r in out if not r['valid']]}")
    print(f"-> {a.refined_out}")
    return 0


def cmd_edge_sheet(a) -> int:
    """Close-ups of chosen edges from directly above: the middle ``--span-m``
    of the edge, 3 m either side, at 5 cm, sampled from the straight-down
    frame nearest overhead at the roof's height, turned so the edge runs
    across and outside is up. Lines: Overture (teal, dashed), the LiDAR's
    move (orange), the frames' refinement (magenta)."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from kyfromabove_3d.wall_match import bilinear_gray
    rows = {r["building"]: r for r in json.loads(a.out.read_text(encoding="utf-8")) if r["status"] == "ok"}
    ref = {r["building"]: r for r in json.loads(a.refined_out.read_text(encoding="utf-8"))}
    blds = {b.key: b for b in wv.reference_buildings(a.reference, a.landmarks, wv.tileset_transforms(a.tileset), a.buildings)}
    frames = wv.read_frames(a.frames)
    index = json.loads((a.nadir_cache / "index.json").read_text(encoding="utf-8"))
    picks = []
    rng = np.random.default_rng(a.seed)
    for status, n in (("refined", a.n_refined), ("frames disagree", a.n_disagree), ("at the window's limit", a.n_limit)):
        pool = [(bk, e) for bk, r in ref.items() for e in r["edges"] if e["status"] == status]
        for i in rng.permutation(len(pool))[:n]:
            picks.append(pool[i])
    cols = 3
    fig, axs = plt.subplots((len(picks) + cols - 1) // cols, cols, figsize=(6.2 * cols, 3.0 * ((len(picks) + cols - 1) // cols)), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (bk, e) in zip(axs.ravel(), picks):
        r, b = rows[bk], blds[bk]
        B, _ = load_building(b)
        G = r["grid"]
        ny, nx = G["shape"]
        g = roof_grid(B["xyz"], B["cls"], G["x0"], G["y0"], nx, ny, G["cell"])
        P0 = np.asarray(r["overture"]["ring"])[:-1]
        sgn = outward(P0)
        k = e["edge"]
        p, q = P0[k], P0[(k + 1) % len(P0)]
        L = float(np.linalg.norm(q - p))
        d = (q - p) / L
        nrm = sgn * np.array([d[1], -d[0]])
        mid, half = L / 2, min(a.span_m, L - 1) / 2
        t = np.arange(mid - half, mid + half, 0.05)
        o = np.arange(-3.0, 3.0, 0.05)
        base = p + e["lidar_move_m"] * nrm
        h = roof_height_along(g, G, base[None, :] + t[:, None] * d[None, :], -nrm)
        h = np.where(np.isfinite(h), h, np.nanmedian(h))
        T, O = np.meshgrid(t, o)
        xy = base[None, :] + T.ravel()[:, None] * d[None, :] + O.ravel()[:, None] * nrm[None, :]
        X = np.column_stack([xy, np.tile(h, len(o))]) @ b.M[:3, :3].T + b.M[:3, 3]
        _, kk, w = min(((w["off_nadir_deg"], kk, w) for kk, w in index.items() if w["building"] == bk), key=lambda z: z[0])
        cam = {c["id"]: c for c in wv.cameras_in_tileset_frame(frames, b)}[w["frame"]]
        img = cv2.imread(str(a.nadir_cache / kk))[:, :, ::-1].astype(np.float32)
        pc = (X - cam["C"]) @ cam["R_cw"].T
        u = cam["K"][0, 0] * pc[:, 0] / pc[:, 2] + cam["K"][0, 2] - w["x0"]
        v = cam["K"][1, 1] * pc[:, 1] / pc[:, 2] + cam["K"][1, 2] - w["y0"]
        rgb = np.stack([bilinear_gray(img[..., c_], u, v)[0] for c_ in range(3)], axis=-1).reshape(len(o), len(t), 3)
        ax.imshow(np.clip(rgb, 0, 255).astype(np.uint8), origin="lower", extent=[t[0], t[-1], o[0], o[-1]], aspect="equal")
        ax.axhline(-e["lidar_move_m"], color="#3cc2bb", lw=1.6, ls=(0, (4, 2)))
        ax.axhline(0, color="#f0a020", lw=1.6)
        if "frames_say_m" in e:
            ax.axhline(e["frames_say_m"], color="#ff3cc8", lw=1.6)
        ax.axis("on"); ax.set_yticks([-2, 0, 2]); ax.tick_params(labelsize=7); ax.set_xticks([])
        ax.set_title(f'{r["name"][:24]} edge {k}: {e["status"]}'
                     + (f'; frames say {e["frames_say_m"]:+.2f} m, spread {e["spread_m"]:.2f}' if "frames_say_m" in e else ""), fontsize=8.5)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color="#3cc2bb", lw=2, ls=(0, (4, 2))), Line2D([], [], color="#f0a020", lw=2), Line2D([], [], color="#ff3cc8", lw=2)],
               ["Overture", "LiDAR's line (step 1)", "the straight-down frames' edge (step 2)"], loc="upper center", ncol=3, fontsize=10, frameon=False)
    fig.text(0.5, 0.005, "Each picture: the middle of one edge from directly above, outside at the top; numbers on the left are metres out from the LiDAR's line",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    fig.savefig(a.edges_out, dpi=a.dpi)
    print(f"{len(picks)} edges -> {a.edges_out}")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("measure", help="each reference building's outline from the LiDAR")
    p.add_argument("--reference", type=Path, default=ROOT / "configs" / "audit_reference.yaml")
    p.add_argument("--landmarks", type=Path, default=ROOT / "configs" / "landmarks.yaml")
    p.add_argument("--tileset", type=Path, default=ROOT / "outputs" / "downtown")
    p.add_argument("--buildings", type=Path, default=ROOT / "outputs" / "downtown" / "buildings")
    p.add_argument("--footprints", type=Path, default=FOOTPRINTS)
    p.add_argument("--cell-m", type=float, default=0.5)
    p.add_argument("--min-height-m", type=float, default=2.5, help="roof is at least this far above the ground")
    p.add_argument("--max-step-m", type=float, default=1.5, help="a height change this big between cells ends the roof")
    p.add_argument("--reach-m", type=float, default=3.0, help="the furthest the outline may run past Overture's")
    p.add_argument("--min-support", type=float, default=0.5, help="share of an edge the roof's end must be found along")
    p.add_argument("--gap-m", type=float, default=1.0, help="gaps in the roof this short are stepped over")
    p.add_argument("--max-corner-move-m", type=float, default=4.0)
    p.add_argument("--simplify-m", type=float, default=0.35)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", type=Path, default=REF / "lidar_outline.json")
    q = sub.add_parser("plot", help="a plan of every building's three outlines")
    for act in p._actions[1:]:
        if act.dest in ("reference", "landmarks", "tileset", "buildings", "out"):
            q._add_action(act)
    q.add_argument("--only", nargs="*", default=None, help="building keys (prefixes)")
    q.add_argument("--cols", type=int, default=4)
    q.add_argument("--dpi", type=int, default=80)
    q.add_argument("--plot-out", type=Path, default=REF / "lidar_outline.png")
    sh = sub.add_parser("sheet", help="Overture's and the moved outline drawn on the frames")
    for act in p._actions[1:]:
        if act.dest in ("reference", "landmarks", "tileset", "buildings", "out"):
            sh._add_action(act)
    sh.add_argument("--frames", type=Path, default=ROOT / "outputs" / "downtown" / "frames_2024s1.gpkg")
    sh.add_argument("--cache", type=Path, default=ROOT / "data" / "derived" / "frames_cache")
    sh.add_argument("--only", nargs="+", required=True, help="building keys (prefixes)")
    sh.add_argument("--sheet-out", type=Path, default=REF / "lidar_outline_sheet.jpg")
    fn = sub.add_parser("fetch-nadir", help="straight-down frame windows over each building's roof")
    for act in p._actions[1:]:
        if act.dest in ("reference", "landmarks", "tileset", "buildings"):
            fn._add_action(act)
    fn.add_argument("--frames", type=Path, default=ROOT / "outputs" / "downtown" / "frames_2024s1.gpkg")
    fn.add_argument("--nadir-cache", type=Path, default=ROOT / "data" / "derived" / "frames_cache_nadir")
    fn.add_argument("--per-building", type=int, default=4)
    fn.add_argument("--margin-m", type=float, default=15.0)
    o = sub.add_parser("ortho", help="each roof from directly above with both outlines on it")
    for act in p._actions[1:]:
        if act.dest in ("reference", "landmarks", "tileset", "buildings", "out"):
            o._add_action(act)
    o.add_argument("--frames", type=Path, default=ROOT / "outputs" / "downtown" / "frames_2024s1.gpkg")
    o.add_argument("--nadir-cache", type=Path, default=ROOT / "data" / "derived" / "frames_cache_nadir")
    o.add_argument("--only", nargs="*", default=None)
    o.add_argument("--cols", type=int, default=3)
    o.add_argument("--size", type=float, default=6.5)
    o.add_argument("--dpi", type=int, default=90)
    o.add_argument("--ortho-out", type=Path, default=REF / "lidar_outline_ortho.jpg")
    rf = sub.add_parser("refine", help="each moved edge refined in the straight-down frames")
    for act in p._actions[1:]:
        if act.dest in ("reference", "landmarks", "tileset", "buildings", "out", "workers", "max_corner_move_m"):
            rf._add_action(act)
    rf.add_argument("--frames", type=Path, default=ROOT / "outputs" / "downtown" / "frames_2024s1.gpkg")
    rf.add_argument("--nadir-cache", type=Path, default=ROOT / "data" / "derived" / "frames_cache_nadir")
    rf.add_argument("--max-frames", type=int, default=4)
    rf.add_argument("--window-m", type=float, default=0.5, help="the furthest the frames may move an edge from the LiDAR's line")
    rf.add_argument("--outer-share", type=float, default=0.6, help="a brightness step this share of the strongest counts")
    rf.add_argument("--min-clarity", type=float, default=3.0, help="the step over the profile's median gradient")
    rf.add_argument("--agree-m", type=float, default=0.15, help="every frame within this of their median")
    rf.add_argument("--refined-out", type=Path, default=REF / "lidar_outline_refined.json")
    es = sub.add_parser("edge-sheet", help="close-ups of edges from directly above, with the three lines")
    for act in rf._actions[1:]:
        if act.dest in ("reference", "landmarks", "tileset", "buildings", "out", "frames", "nadir_cache", "refined_out"):
            es._add_action(act)
    es.add_argument("--n-refined", type=int, default=12)
    es.add_argument("--n-disagree", type=int, default=6)
    es.add_argument("--n-limit", type=int, default=3)
    es.add_argument("--span-m", type=float, default=10.0)
    es.add_argument("--seed", type=int, default=1)
    es.add_argument("--dpi", type=int, default=90)
    es.add_argument("--edges-out", type=Path, default=REF / "lidar_outline_edges.jpg")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = parse_args(argv)
    return dict(measure=cmd_measure, plot=cmd_plot, sheet=cmd_sheet, ortho=cmd_ortho, refine=cmd_refine, **{'edge-sheet': cmd_edge_sheet}, **{'fetch-nadir': cmd_fetch_nadir})[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
