#!/usr/bin/env python3
"""Where a whole wall face stands, at its depth and its angle, by matching the frames that see it.

``wall_match.py`` matched each 5 m band of a face on its own, with a trial
wall parallel to the footprint edge. On the walls the LiDAR missed that
fitted badly -- not because the frames see them worse (as many frames, as
far apart, at the same detail) but because those bands are short (10 m
against 27, so a third is 3 m of wall) and about half of those whose thirds
disagree step one way along the wall: a wall turned against its Overture
edge. Here the trial wall covers the face's whole height and has a depth at
each end, so it can turn:

    sweep   per straight face: the frames that see it (every cached one),
            a parallel sweep for the rough depth, then the depth at each
            end within +-2 m of it, on a 0.25 m grid, then finer about the
            best. Kept: the best wall, how well the frames agree on it, the
            best rival wall at least ``--rival-m`` away at either end, the
            wall each half of the face (left/right, lower/upper) prefers,
            and each band's own best depth near the face's wall (a band
            that prefers a wall a metre off is a setback). Where the LiDAR
            saw the face, a line fitted through its wall returns goes
            beside it.
    atlas   for looking: each swept face laid flat from its most face-on
            frame, its depth against height (2 m strips, where the frames
            agree best, relative to the face's matched wall), the LiDAR's
            bands beside it, and the facade's pattern traces (floor lines,
            window bays, the storey height they imply)
    score   against the LiDAR's line on the faces it measured, held out by
            building, with the footprint edge and band matching beside it;
            then what it does on the faces the LiDAR missed.

Depths are metres off the footprint edge, + out, at the two ends of the
stretch matched (the face less 1.5 m at each corner).
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import wall_visibility as wv  # noqa: E402

from kyfromabove_3d.wall_match import MIN_BASELINE_DEG, bilinear_gray  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GRID_M = 0.25                 # the search's pixel
SIGMA_PX = 3.0                # the high-pass at GRID_M (about 0.8 m, as the band matcher's)
CORNER_M = 1.5


# ------------------------------------------------------------------ the face's wall, laid out

def face_views(b, face, plane, vis, cams, imgs, min_seen=0.3):
    """The face's grid (A along, Z up, ``GRID_M``) over its wall points, and
    for each frame that sees at least ``min_seen`` of them, its view: the
    camera, the image window and where on the grid it sees the wall."""
    sel = vis["face_of"] == face
    if sel.sum() < 30:
        return None
    sa, sz, us = vis["along"][sel], vis["z"][sel], vis["usable"][:, sel]
    L = float(plane.length_m)
    lo, hi = max(CORNER_M, sa.min() - 0.5), min(L - CORNER_M, sa.max() + 0.5)
    if hi - lo < 4.0:
        return None
    A, Z = np.meshgrid(np.arange(lo, hi, GRID_M), np.arange(sz.min() - 0.5, sz.max() + 0.5, GRID_M))
    # each grid cell takes the visibility of its nearest 1 m sample
    ka = np.clip(np.round(A - sa.min()).astype(int), 0, None)
    kz = np.clip(np.round(Z - sz.min()).astype(int), 0, None)
    ia, iz = np.round(sa - sa.min()).astype(int), np.round(sz - sz.min()).astype(int)
    lut = -np.ones((max(kz.max(), iz.max()) + 1, max(ka.max(), ia.max()) + 1), int)
    lut[iz, ia] = np.arange(len(sa))
    near = lut[kz, ka]
    views = {}
    for j in range(us.shape[0]):
        f = str(vis["frames"][j])
        if us[j].mean() < min_seen or f not in imgs:
            continue
        img, x0, y0 = imgs[f]
        views[f] = dict(K=cams[f]["K"], R_cw=cams[f]["R_cw"], C=cams[f]["C"], gray=img, x0=x0, y0=y0,
                        mask=(near >= 0) & us[j][np.maximum(near, 0)])
    return A, Z, views, (lo, hi)


# ------------------------------------------------------------------ where the face is really wall

FIELD_CELL_M = 0.5
INSIDE_M = (0.5, 1.0, 1.5, 2.0, 2.5)     # behind the wall: the building's roof must stand above the point
OUTSIDE_M = (1.0, 1.5, 2.0, 2.5, 3.0)    # in front of the wall's line: no roof or block above the point
BURY_MARGIN_M = 0.5
TRIM_SHARE = 0.5                          # rows and columns kept when at least this share of them is wall
MAX_FRAME_DEG = 60.0                      # a frame looks within this of straight at the face
MIN_FRAME_SEEN = 0.5                      # and sees at least this share of the trimmed face
MIN_FRAME_AGREE = 0.15                    # and its view of the face matches another frame's at least this well


def height_field(scene_path: Path, cell_m: float = FIELD_CELL_M):
    """The highest mesh vertex in each cell of a plan grid over every
    downtown mesh (the tileset frame, z up): the roofs, a building's own lower
    parts included. Returns (z grid, x0, y0, cell)."""
    from scipy.ndimage import maximum_filter
    V = np.load(scene_path)["V"].astype(np.float64)
    x0, y0 = V[:, 0].min() - 4.0, V[:, 1].min() - 4.0
    ix = ((V[:, 0] - x0) / cell_m).astype(int)
    iy = ((V[:, 1] - y0) / cell_m).astype(int)
    g = np.full((iy.max() + 9, ix.max() + 9), -np.inf, np.float32)
    np.maximum.at(g, (iy, ix), V[:, 2].astype(np.float32))
    return maximum_filter(g, size=3), x0, y0, cell_m        # the meshes' vertices are 0.75 m apart: close the gaps


def field_at(field, xy: np.ndarray) -> np.ndarray:
    g, x0, y0, c = field
    ix = np.clip(((xy[:, 0] - x0) / c).astype(int), 0, g.shape[1] - 1)
    iy = np.clip(((xy[:, 1] - y0) / c).astype(int), 0, g.shape[0] - 1)
    return g[iy, ix]


def wall_cells(b, plane, A, Z, lo, hi, d0, d1, field) -> np.ndarray:
    """Which cells of the face's grid, on its matched wall (d0 at ``lo``, d1
    at ``hi``), are open wall: the roof just behind stands above the cell, and
    nothing just in front -- a lower roof, the building's own podium, a
    neighbour -- does. In front is measured from the wall's line or the
    matched wall, whichever stands further out, so a wall matched behind its
    mesh's line is not buried by its own roof."""
    off = d0 + (d1 - d0) * (A.ravel() - lo) / (hi - lo)
    n = np.r_[np.asarray(plane.normal, float), 0.0] @ b.M[:3, :3].T
    P = make_to_world(b, plane, A, Z, lo, hi)(d0, d1)
    top = np.max([field_at(field, P[:, :2] - k * n[:2]) for k in INSIDE_M], axis=0)
    front = np.maximum(off, 0.0) - off
    above = np.min([field_at(field, P[:, :2] + (front + k)[:, None] * n[None, :2]) for k in OUTSIDE_M], axis=0)
    ok = (P[:, 2] < top - BURY_MARGIN_M) & ~(above > P[:, 2] + BURY_MARGIN_M)
    return ok.reshape(A.shape)


def trim_extent(wall: np.ndarray, share: float = TRIM_SHARE):
    """(row slice, column slice) of the grid spanning the rows and the columns
    that are at least ``share`` wall, or None if none is."""
    rows = np.flatnonzero(wall.mean(1) >= share)
    if not len(rows):
        return None
    cols = np.flatnonzero(wall[rows[0]:rows[-1] + 1].mean(0) >= share)
    if not len(cols):
        return None
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def frame_checks(b, plane, A, Z, lo, hi, d0, d1, views, wall) -> dict:
    """For each frame: the share of the trimmed face it sees, how far off
    straight on it looks (degrees), and how well its view of the face, on the
    matched wall, matches its best partner's (high-passed correlation);
    ``keep`` when all three pass."""
    from kyfromabove_3d.texture_check import highpass, masked_ncc
    to_world = make_to_world(b, plane, A, Z, lo, hi)
    P = to_world(d0, d1)
    n3 = np.r_[np.asarray(plane.normal, float), 0.0] @ b.M[:3, :3].T
    centre = P[wall.ravel()].mean(axis=0) if wall.any() else P.mean(axis=0)
    hp, out = {}, {}
    for f, w in views.items():
        pc = (P - w["C"]) @ w["R_cw"].T
        u = w["K"][0, 0] * pc[:, 0] / pc[:, 2] + w["K"][0, 2] - w["x0"]
        v = w["K"][1, 1] * pc[:, 1] / pc[:, 2] + w["K"][1, 2] - w["y0"]
        g, ok = bilinear_gray(w["gray"], u, v)
        m = w["mask"] & ok.reshape(A.shape) & wall
        hp[f] = (highpass(g.reshape(A.shape), m, SIGMA_PX), m)
        ray = w["C"] - centre
        out[f] = dict(seen=round(float(m.sum() / max(wall.sum(), 1)), 3),
                      angle=round(float(np.degrees(np.arccos(np.clip(ray @ n3 / np.linalg.norm(ray), -1, 1)))), 1), agree=None)
    for f in views:
        best = None
        for g_ in views:
            if g_ == f:
                continue
            m = hp[f][1] & hp[g_][1]
            if m.sum() >= 150:
                r = masked_ncc(hp[f][0], hp[g_][0], m)
                if np.isfinite(r) and (best is None or r > best):
                    best = float(r)
        out[f]["agree"] = None if best is None else round(best, 3)
        out[f]["keep"] = (out[f]["seen"] >= MIN_FRAME_SEEN and out[f]["angle"] <= MAX_FRAME_DEG
                          and best is not None and best >= MIN_FRAME_AGREE)
    return out


def make_to_world(b, plane, A, Z, lo, hi):
    """(d0, d1) -> the trial wall's points in the cameras' frame: depth d0 at
    ``lo`` along, d1 at ``hi``, straight between."""
    a = A.ravel()
    t = (a - lo) / (hi - lo)
    p0 = np.asarray(plane.p0, float)
    d = (np.asarray(plane.p1, float) - p0) / float(plane.length_m)
    n = np.asarray(plane.normal, float)
    z = Z.ravel()

    def to_world(d0, d1):
        off = d0 + (d1 - d0) * t
        xy = p0[None, :] + a[:, None] * d[None, :] + off[:, None] * n[None, :]
        return np.column_stack([xy, z]) @ b.M[:3, :3].T + b.M[:3, 3]
    return to_world


def score_walls(to_world, shape, views, walls, pairs, parts):
    """Agreement (weighted pairwise high-passed correlation) for each trial
    wall (d0, d1), for the whole face and each part (a boolean grid)."""
    from kyfromabove_3d.texture_check import highpass, masked_ncc
    nz, na = shape
    out = np.full((len(walls), 1 + len(parts)), np.nan)
    allp = [np.ones(shape, bool)] + list(parts)
    for i, (d0, d1) in enumerate(walls):
        P = to_world(d0, d1)
        hp = {}
        for f, w in views.items():
            pc = (P - w["C"]) @ w["R_cw"].T
            u = w["K"][0, 0] * pc[:, 0] / pc[:, 2] + w["K"][0, 2] - w["x0"]
            v = w["K"][1, 1] * pc[:, 1] / pc[:, 2] + w["K"][1, 2] - w["y0"]
            g, ok = bilinear_gray(w["gray"], u, v)
            m = w["mask"] & ok.reshape(nz, na)
            hp[f] = (highpass(g.reshape(nz, na), m, SIGMA_PX), m)
        for k, part in enumerate(allp):
            s, wt = 0.0, 0.0
            for f1, f2 in pairs:
                m = hp[f1][1] & hp[f2][1] & part
                n = int(m.sum())
                if n < 150:
                    continue
                r = masked_ncc(hp[f1][0], hp[f2][0], m)
                if np.isfinite(r):
                    s += r * np.sqrt(n)
                    wt += np.sqrt(n)
            out[i, k] = s / wt if wt > 0 else np.nan
    return out


def best_of(walls, scores):
    k = int(np.nanargmax(scores)) if np.isfinite(scores).any() else None
    return (None, np.nan) if k is None else (walls[k], float(scores[k]))


def match_face(b, face, plane, vis, cams, imgs, args):
    got = face_views(b, face, plane, vis, cams, imgs)
    if got is None:
        return None
    A, Z, views, (lo, hi) = got
    return match_grid(b, face, plane, vis, A, Z, views, lo, hi, args)


MIN_FACE_AGREE = 0.4     # a face whose best pair of frames matches worse than this is not one wall at its depth


def slice_face(A, Z, views, ext, wall=None, keep=None):
    """The face's grid cut to ``ext`` (row slice, column slice), its views'
    masks cut with it (and to ``wall``, already cut), only the frames in
    ``keep`` (all if None)."""
    rs, cs = ext
    At, Zt = A[rs, cs], Z[rs, cs]
    vt = {f: dict(w, mask=w["mask"][rs, cs] & (wall if wall is not None else True))
          for f, w in views.items() if keep is None or f in keep}
    return At, Zt, vt, (float(At.min()), float(At.max()) + GRID_M)


def depth_at(d0, d1, lo, hi, a):
    return d0 + (d1 - d0) * (a - lo) / (hi - lo)


def match_trimmed(b, face, plane, vis, cams, imgs, args, field):
    """``match_face`` on the face trimmed to its open wall, with only the
    frames that see it properly. Trimmed first on the wall's line, then again
    on the matched wall; frames that see under half the trimmed face or look
    more than 60 degrees off straight on are left out before matching, and
    those whose view matches no other frame's after it (then matched again).
    A face whose best pair of frames matches worse than ``MIN_FACE_AGREE`` is
    marked not ``reliable``: not one wall at that depth."""
    got = face_views(b, face, plane, vis, cams, imgs)
    if got is None:
        return None
    A, Z, views, (lo, hi) = got
    d0 = d1 = 0.0
    r, keep, checks = None, None, {}
    for _ in range(3):
        wall = wall_cells(b, plane, A, Z, lo, hi, d0, d1, field)
        ext = trim_extent(wall)
        if ext is None:
            return dict(face=int(face), reliable=False, why="no open wall")
        At, Zt, vt, (lo_t, hi_t) = slice_face(A, Z, views, ext, wall[ext], keep)
        checks = frame_checks(b, plane, At, Zt, lo_t, hi_t, depth_at(d0, d1, lo, hi, lo_t), depth_at(d0, d1, lo, hi, hi_t),
                              vt, wall[ext])
        ok = {f for f, c in checks.items() if c["seen"] >= MIN_FRAME_SEEN and c["angle"] <= MAX_FRAME_DEG
              and (r is None or (c["agree"] is not None and c["agree"] >= MIN_FRAME_AGREE))}
        vt = {f: w for f, w in vt.items() if f in ok}
        if len(vt) < 2:
            return dict(face=int(face), reliable=False, why="fewer than 2 frames see it properly",
                        frames_checked={f.split("/")[-1]: c for f, c in checks.items()})
        new = match_grid(b, face, plane, vis, At, Zt, vt, lo_t, hi_t, args)
        if new is None:
            return None
        nd0, nd1 = depth_at(new["d0"], new["d1"], lo_t, hi_t, lo), depth_at(new["d0"], new["d1"], lo_t, hi_t, hi)
        same = r is not None and set(vt) == set(keep or vt) and abs(nd0 - d0) < 0.3 and abs(nd1 - d1) < 0.3
        r, keep, d0, d1 = new, set(vt), nd0, nd1
        if same:
            break
    checks = frame_checks(b, plane, At, Zt, lo_t, hi_t, r["d0"], r["d1"], {f: w for f, w in vt.items()}, wall[ext])
    best = max([c["agree"] for c in checks.values() if c["agree"] is not None] or [-1.0])
    r.update(reliable=bool(best >= MIN_FACE_AGREE), best_agree=round(best, 3),
             why=None if best >= MIN_FACE_AGREE else f"its frames match each other only {best:.2f}",
             frames_kept=sorted(f.split("/")[-1] for f in vt), untrimmed=dict(z=[round(float(Z.min()), 2), round(float(Z.max()), 2)],
                                                                                along=[round(lo, 2), round(hi, 2)]))
    return r


def match_grid(b, face, plane, vis, A, Z, views, lo, hi, args):
    from kyfromabove_3d.wall_match import pairs_apart
    if len(views) < 2:
        return None
    to_world = make_to_world(b, plane, A, Z, lo, hi)
    pairs = pairs_apart(views, to_world(0.0, 0.0).mean(axis=0))
    if not pairs:
        return None
    na, nz = A.shape[1], A.shape[0]
    cols = np.arange(na)[None, :].repeat(nz, 0)
    rows = np.arange(nz)[:, None].repeat(na, 1)
    halves = [cols < na // 2, cols >= na // 2, rows < nz // 2, rows >= nz // 2]   # left, right, lower (rows run up), upper
    # 1. parallel, the whole search
    par = [(o, o) for o in np.arange(args.depth_min, args.depth_max + 1e-9, 0.25)]
    sp = score_walls(to_world, A.shape, views, par, pairs, [])
    (d_par, _), _ = best_of(par, sp[:, 0])
    if d_par is None:
        return None
    # 2. a depth at each end, +-2 m about it
    g = np.arange(-2.0, 2.0001, 0.25)
    two = [(d_par + x, d_par + y) for x in g for y in g
           if args.depth_min <= d_par + x <= args.depth_max and args.depth_min <= d_par + y <= args.depth_max]
    s2 = score_walls(to_world, A.shape, views, two, pairs, halves)
    (c0, c1), _ = best_of(two, s2[:, 0])
    # 3. finer about the best
    f = np.arange(-0.25, 0.2501, 0.05)
    fine = [(c0 + x, c1 + y) for x in f for y in f]
    sf = score_walls(to_world, A.shape, views, fine, pairs, [])
    (b0, b1), peak = best_of(fine, sf[:, 0])
    # the rival: the best wall at least rival_m from it at either end, among all tried
    cand = par + two
    sc = np.r_[sp[:, 0], s2[:, 0]]
    far = np.array([max(abs(x - b0), abs(y - b1)) >= args.rival_m for x, y in cand])
    rival = float(np.nanmax(sc[far])) if np.isfinite(sc[far]).any() else float("nan")
    halves_best = [list(best_of(two, s2[:, 1 + k])[0] or (None, None)) for k in range(4)]
    # the parallel wall's own best, for the angle's worth
    par_fine = [(d_par + x, d_par + x) for x in np.arange(-0.25, 0.2501, 0.05)]
    spf = score_walls(to_world, A.shape, views, par_fine, pairs, [])
    (pp, _), par_peak = best_of(par_fine, spf[:, 0])
    # each band's own preferred shift off the face's wall (setbacks)
    bands = []
    zb = sorted({(round(float(r["z_lo"]), 2), round(float(r["z_hi"]), 2)) for r in vis["bands_of"].get(face, [])})
    shifts = np.arange(-1.5, 1.5001, 0.25)
    for z_lo, z_hi in zb:
        part = (Z >= z_lo) & (Z < z_hi)
        if part.sum() < 200:
            continue
        sb = score_walls(to_world, A.shape, views, [(b0 + s, b1 + s) for s in shifts], pairs, [part])[:, 1]
        if np.isfinite(sb).any():
            k = int(np.nanargmax(sb))
            bands.append(dict(z_lo=z_lo, z_hi=z_hi, shift_m=round(float(shifts[k]), 2), agreement=round(float(sb[k]), 3)))
    return dict(face=int(face), along=[round(lo, 2), round(hi, 2)], z=[round(float(Z.min()), 1), round(float(Z.max()), 1)],
                frames=len(views), pairs=len(pairs), d0=round(b0, 2), d1=round(b1, 2), agreement=round(peak, 3),
                rival=round(rival, 3), margin=round(peak - rival, 3), parallel=round(pp, 2), parallel_agreement=round(par_peak, 3),
                halves=halves_best, bands=bands)


# ------------------------------------------------------------------ the LiDAR's answer

def lidar_line(b, face, plane, profile_rows, lo, hi):
    """The LiDAR's wall on a face it measured: a line (offset against along)
    fitted through the returns of its observed bands, trimmed twice of
    returns more than 0.3 m off it. (depth at lo, depth at hi, n) or None."""
    import laspy
    obs = [r for r in profile_rows if r["status"] == "observed" and r.get("offset_m") is not None]
    if len(obs) < 2:
        return None
    path = b.dir / "lidar_local.laz"
    if not path.exists():
        return None
    las = laspy.read(path)
    xyz = np.column_stack([las.x, las.y, las.z])
    along, off = plane.local(xyz[:, :2])
    keep = np.zeros(len(xyz), bool)
    for r in obs:
        keep |= ((xyz[:, 2] >= r["z_lo"]) & (xyz[:, 2] < r["z_hi"]) & (np.abs(off - r["offset_m"]) <= 0.6)
                 & (along >= max(lo, r["along_lo"])) & (along <= min(hi, r["along_hi"])))
    a, o = along[keep], off[keep]
    if len(a) < 60 or a.max() - a.min() < 0.4 * (hi - lo):
        return None
    for _ in range(3):
        k, c = np.polyfit(a, o, 1)
        r_ = np.abs(o - (k * a + c))
        if (r_ > 0.3).mean() < 0.02:
            break
        a, o = a[r_ <= max(0.3, np.percentile(r_, 80))], o[r_ <= max(0.3, np.percentile(r_, 80))]
    return round(float(k * lo + c), 2), round(float(k * hi + c), 2), int(len(a))


# ------------------------------------------------------------------ the atlas

def storey_period(profile: np.ndarray, step_m: float, lo_m: float = 2.6, hi_m: float = 6.5):
    """The repeat of a trace (floor lines up a facade): the lag, in metres,
    where its autocorrelation peaks between ``lo_m`` and ``hi_m``, and how
    strong that peak is (0-1)."""
    x = profile - profile.mean()
    if len(x) < 2 * hi_m / step_m or not np.any(x):
        return None, 0.0
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / ac[0]
    k0, k1 = int(lo_m / step_m), min(int(hi_m / step_m), len(ac) - 1)
    k = k0 + int(np.argmax(ac[k0:k1]))
    return round(k * step_m, 2), round(float(ac[k]), 3)


def image_lines(rows: np.ndarray, step_m: float, min_gap_m: float = 0.6):
    """The horizontal lines a laid-flat facade shows, from its row-by-row edge
    strength (bottom up, one value per ``step_m``): the heights of the clear
    peaks, at least ``min_gap_m`` apart, and how far apart they usually are
    (the median gap, kept only when at least 3 lines are found and most gaps
    are within 15% of it), with the share of gaps that agree."""
    x = np.asarray(rows, float)
    if len(x) < 5 or not np.any(x):
        return dict(lines=[], spacing_m=None, regular=0.0)
    sm = np.convolve(x, [0.25, 0.5, 0.25], "same")
    cut = np.median(sm) + 1.5 * np.median(np.abs(sm - np.median(sm)))
    w = max(1, int(round(0.3 / step_m)))
    peaks = [i for i in range(len(sm)) if sm[i] > cut and sm[i] == sm[max(0, i - w):i + w + 1].max()]
    keep = []
    for i in sorted(peaks, key=lambda i: -sm[i]):                       # strongest first, then spaced apart
        if all(abs(i - j) * step_m >= min_gap_m for j in keep):
            keep.append(i)
    z = np.sort(np.array(keep)) * step_m
    spacing, regular = None, 0.0
    if len(z) >= 3:
        gaps = np.diff(z)
        med = float(np.median(gaps))
        regular = float(np.mean(np.abs(gaps - med) <= 0.15 * med))
        if regular >= 0.5:
            spacing = round(med, 2)
    return dict(lines=[round(float(v), 2) for v in z], spacing_m=spacing, regular=round(regular, 2))


def find_steps(curve, min_jump_m: float = 0.5, min_margin: float = 0.02):
    """Heights where the wall steps: the median depth of the 3 confident 2 m
    slices below a height against the 3 above differ by ``min_jump_m``; of
    neighbouring heights within 2.5 m, the biggest. The atlas page applies
    the same rule."""
    c = [p for p in curve if p[3] is None or p[3] >= min_margin]
    out = []
    for k in range(3, len(c) - 2):
        below = float(np.median([p[1] for p in c[k - 3:k]]))
        above = float(np.median([p[1] for p in c[k:k + 3]]))
        if abs(above - below) >= min_jump_m:
            s_ = dict(z=0.5 * (c[k - 1][0] + c[k][0]), below=below, above=above)
            if out and s_["z"] - out[-1]["z"] < 2.5:
                if abs(above - below) > abs(out[-1]["above"] - out[-1]["below"]):
                    out[-1] = s_
            else:
                out.append(s_)
    return out


# ------------------------------------------------------------------ steps on the image's lines

SHIFTS = np.arange(-2.5, 2.5001, 0.25)   # trial depths off the face's matched wall
STEP_COST = 0.02     # a step must buy this share of the face's whole agreement
MIN_RUN_M = 2.0      # each run of one depth at least this tall
MIN_JUMP_M = 0.5     # a step changes the depth by at least this
MIN_CLEAR = 0.02     # each run's depth beats any depth 0.75 m away by this much agreement
LINE_EDGE_M = 0.5    # lines this close to the face's bottom or top are its own edges, not breaks


def pieces_of(z0: float, z1: float, lines) -> List[float]:
    """The heights a face is cut at: its bottom, each line the image shows
    (away from the face's own edges) and its top."""
    return [z0] + [z for z in sorted(lines) if z0 + LINE_EDGE_M < z < z1 - LINE_EDGE_M] + [z1]


def runs_from_scores(S: np.ndarray, W: np.ndarray, edges, shifts=SHIFTS, step_cost=STEP_COST, min_run_m=MIN_RUN_M,
                     min_jump_m=MIN_JUMP_M, min_clear=MIN_CLEAR) -> List[dict]:
    """Group neighbouring pieces into runs of one depth. ``S`` is the frames'
    agreement for each trial depth (rows) and piece (columns; NaN where the
    piece was not seen), ``W`` each piece's weight, ``edges`` the heights
    between pieces (one more than the pieces). The grouping kept is the one
    with the best weighted agreement after paying ``step_cost`` of the whole
    face's weight for each step, among runs at least ``min_run_m`` tall; then
    a step whose depths differ by less than ``min_jump_m``, or beside a run
    whose depth is not clear by ``min_clear``, is dropped (its runs joined)
    until none is left. Returns [{z_lo, z_hi, shift, clear}] bottom up."""
    ok = np.isfinite(S)
    Sw = np.where(ok, S, 0.0) * W[None, :]
    cs = np.concatenate([np.zeros((len(shifts), 1)), np.cumsum(Sw, 1)], 1)
    cw = np.concatenate([np.zeros((len(shifts), 1)), np.cumsum(ok * W[None, :], 1)], 1)
    total = float(np.max(cw[:, -1])) or 1.0
    far = np.abs(shifts[:, None] - shifts[None, :]) >= 0.75

    def run(i, j):                                   # pieces i..j-1
        tot, wt = cs[:, j] - cs[:, i], cw[:, j] - cw[:, i]
        if not np.any(wt > 0):
            return 0.0, None, 0.0
        mean = np.where(wt > 0, tot / np.maximum(wt, 1e-9), -np.inf)
        k = int(np.argmax(mean))
        rival = np.max(np.where(far[k], mean, -np.inf))
        clear = float(mean[k] - rival) if np.isfinite(rival) else 0.0
        return float(tot[k]), k, clear

    n = S.shape[1]
    tall = lambda i, j: edges[j] - edges[i] >= min_run_m - 1e-9
    best, back = [0.0] + [-np.inf] * n, [None] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j):
            if not (tall(i, j) or (i == 0 and j == n)) or not np.isfinite(best[i]):
                continue
            v = best[i] + run(i, j)[0] - (step_cost * total if i > 0 else 0.0)
            if v > best[j]:
                best[j], back[j] = v, i
    cuts, j = [], n
    while j and back[j] is not None:
        cuts.append(j)
        j = back[j]
    cuts = [0] + sorted(cuts) if j == 0 else [0, n]
    while True:                                      # drop the weakest failing step, then look again
        runs = [run(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1)]
        bad = []
        for k in range(1, len(runs)):
            (_, a, ca), (_, b_, cb) = runs[k - 1], runs[k]
            jump = abs(shifts[b_] - shifts[a]) if a is not None and b_ is not None else 0.0
            if jump < min_jump_m or min(ca, cb) < min_clear:
                bad.append((jump, min(ca, cb), k))
        if not bad:
            break
        cuts.pop(min(bad)[2])
    return [dict(z_lo=round(float(edges[cuts[k]]), 2), z_hi=round(float(edges[cuts[k + 1]]), 2),
                 shift=None if r[1] is None else round(float(shifts[r[1]]), 2), clear=round(r[2], 3))
            for k, r in enumerate(runs)]


def steps_of(runs) -> List[dict]:
    return [dict(z=runs[k]["z_lo"], below=runs[k - 1]["shift"], above=runs[k]["shift"]) for k in range(1, len(runs))]


def line_steps(to_world, shape, views, d0, d1, pairs, Z, edges):
    """The face's runs and steps from all its frame pairs, each step marked
    ``confirmed`` when both halves of the pairs, taken alone, also put a step
    within 1 m of it, the same way (in or out)."""
    parts = [(Z >= a) & (Z < b) for a, b in zip(edges[:-1], edges[1:])]
    W = np.sqrt(np.array([p.sum() for p in parts], float))
    walls = [(d0 + x, d1 + x) for x in SHIFTS]

    def runs_with(pp):
        return runs_from_scores(score_walls(to_world, shape, views, walls, pp, parts)[:, 1:], W, edges) if pp else []
    S = score_walls(to_world, shape, views, walls, pairs, parts)[:, 1:]
    pieces = []
    for k in range(len(parts)):                      # each piece's own best depth, and how clear it is
        col = S[:, k]
        if not np.isfinite(col).any():
            pieces.append([edges[k], edges[k + 1], None, None])
            continue
        j = int(np.nanargmax(col))
        rival = np.where(np.abs(SHIFTS - SHIFTS[j]) >= 0.75, col, np.nan)
        pieces.append([round(float(edges[k]), 2), round(float(edges[k + 1]), 2), round(float(SHIFTS[j]), 2),
                       round(float(col[j] - np.nanmax(rival)), 3) if np.isfinite(rival).any() else None])
    runs = runs_from_scores(S, W, edges)
    halves = [steps_of(runs_with(pairs[0::2])), steps_of(runs_with(pairs[1::2]))] if len(pairs) >= 2 else [[], []]
    steps = steps_of(runs)
    for s_ in steps:
        way = np.sign(s_["above"] - s_["below"])
        s_["confirmed"] = all(any(abs(h["z"] - s_["z"]) <= 1.0 and np.sign(h["above"] - h["below"]) == way for h in half)
                              for half in halves)
    return runs, steps, pieces


# ------------------------------------------------------------------ is the end of a face wall, or a level surface?

LEVEL_MARGIN = 0.1       # a level surface must beat the upright wall's agreement by this much
LEVEL_EDGE_M = 1.0       # and stand within this of the piece's edge (a deck at its top, a roof at its bottom)
LEVEL_MAX_CUT = 0.5      # never cut away more than this share of the face's height
LEVEL_REACH_M = 1.0      # level heights tried: from this below the piece to this above it
LEVEL_COVER = 0.8        # a level surface must explain at least this share of the piece's pixels
LEVEL_MIN_DEG = 2.0      # frames this far apart already tell a wall from a deck metres in front of it
LEVEL_ABSTAIN = 0.2      # a frame matching neither explanation better than this sits the vote out


def _local_rays(b, plane, A, Z, lo, hi, d0, d1, C):
    """The reference camera's rays to the cells (A, Z) on the matched wall, in
    the building's local frame: origin (along, out, z) and direction, each per cell."""
    Minv = np.linalg.inv(b.M)
    c = Minv[:3, :3] @ np.asarray(C, float) + Minv[:3, 3]
    p0 = np.asarray(plane.p0, float)
    dv = (np.asarray(plane.p1, float) - p0) / float(plane.length_m)
    nv = np.asarray(plane.normal, float)
    a, z = A.ravel(), Z.ravel()
    off = d0 + (d1 - d0) * (a - lo) / (hi - lo)
    o_c = np.r_[(c[:2] - p0) @ dv, (c[:2] - p0) @ nv, c[2]]
    return o_c, np.column_stack([a - o_c[0], off - o_c[1], z - o_c[2]])


def _to_world(b, plane, q: np.ndarray) -> np.ndarray:
    """(along, out, z) in the face's terms -> the cameras' frame."""
    p0 = np.asarray(plane.p0, float)
    dv = (np.asarray(plane.p1, float) - p0) / float(plane.length_m)
    nv = np.asarray(plane.normal, float)
    xy = p0[None, :] + q[:, :1] * dv[None, :] + q[:, 1:2] * nv[None, :]
    return np.column_stack([xy, q[:, 2]]) @ b.M[:3, :3].T + b.M[:3, 3]


def level_piece(b, plane, A, Z, lo, hi, d0, d1, views, ref, end: str) -> dict:
    """Is this band of the face (the cells A, Z) an upright wall or a level
    surface? The reference frame's pixels for the cells stay fixed; each is
    explained as the wall at a trial depth (``SHIFTS`` off the matched wall)
    or as a level surface at a trial height, and the other frames' pictures,
    sampled where each explanation puts the pixel, are matched against the
    reference's. A level surface counts only on the side it can be: in front
    of the wall at a face's ``bottom`` (a deck, a lower roof), behind it at its
    ``top`` (the roof beyond the wall's top edge)."""
    from kyfromabove_3d.texture_check import highpass, masked_ncc
    w0 = views[ref]
    o_c, D = _local_rays(b, plane, A, Z, lo, hi, d0, d1, w0["C"])
    shape = A.shape
    P0 = _to_world(b, plane, o_c[None, :] + D)                       # the cells on the matched wall
    pc = (P0 - w0["C"]) @ w0["R_cw"].T
    u = w0["K"][0, 0] * pc[:, 0] / pc[:, 2] + w0["K"][0, 2] - w0["x0"]
    v = w0["K"][1, 1] * pc[:, 1] / pc[:, 2] + w0["K"][1, 2] - w0["y0"]
    g0, ok0 = bilinear_gray(w0["gray"], u, v)
    m0 = w0["mask"] & ok0.reshape(shape)
    g0 = g0.reshape(shape)
    beta = (d1 - d0) / (hi - lo)
    centre = P0.mean(axis=0)
    look = lambda w: (centre - w["C"]) / np.linalg.norm(centre - w["C"])
    if not m0.any():
        return dict(z=[round(float(Z.min()), 2), round(float(Z.max()) + GRID_M, 2)], end=end, upright=None, upright_shift=None,
                    level=None, level_h=None, is_level=False)
    others = [f for f in views if f != ref and np.degrees(np.arccos(np.clip(look(views[f]) @ look(w0), -1, 1))) >= LEVEL_MIN_DEG]

    def agreement(t, keep):
        """Each other frame's (correlation, pixels) for this explanation."""
        q = o_c[None, :] + t[:, None] * D
        X = _to_world(b, plane, q)
        out = np.full((len(others), 2), np.nan)
        for i, f in enumerate(others):
            w = views[f]
            pc = (X - w["C"]) @ w["R_cw"].T
            uu = w["K"][0, 0] * pc[:, 0] / pc[:, 2] + w["K"][0, 2] - w["x0"]
            vv = w["K"][1, 1] * pc[:, 1] / pc[:, 2] + w["K"][1, 2] - w["y0"]
            g, ok = bilinear_gray(w["gray"], uu, vv)
            m = m0 & (ok & keep & (pc[:, 2] > 0)).reshape(shape)
            n = int(m.sum())
            if n < 150:
                continue
            r = masked_ncc(highpass(g0, m, SIGMA_PX), highpass(g.reshape(shape), m, SIGMA_PX), m)
            if np.isfinite(r):
                out[i] = r, n
        return out

    a_c = o_c[0]
    up = []
    for x in SHIFTS:                                                  # the upright wall, d0 + x .. d1 + x
        alpha = d0 + x - beta * lo
        den = D[:, 1] - beta * D[:, 0]
        t = np.where(np.abs(den) > 1e-9, (alpha + beta * a_c - o_c[1]) / np.where(np.abs(den) > 1e-9, den, 1), -1)
        up.append(agreement(t, t > 0))
    z_lo, z_hi = float(Z.min()), float(Z.max()) + GRID_M
    heights = np.arange(z_lo - LEVEL_REACH_M, z_hi + LEVEL_REACH_M + 1e-9, 0.25)
    lv = []
    for h in heights:                                                 # a level surface at height h
        t = np.where(np.abs(D[:, 2]) > 1e-9, (h - o_c[2]) / np.where(np.abs(D[:, 2]) > 1e-9, D[:, 2], 1), -1)
        q = o_c[None, :] + t[:, None] * D
        wall_at = d0 + beta * (q[:, 0] - lo)
        side = q[:, 1] >= wall_at - 0.25 if end == "bottom" else q[:, 1] <= wall_at + 0.25
        keep = (t > 0) & side
        lv.append(agreement(t, keep) if keep.reshape(shape)[m0].mean() >= LEVEL_COVER else np.full((len(others), 2), np.nan))
    U, Lv = np.array(up), np.array(lv)                                # (explanations, frames, [r, n])
    if not len(others):
        U, Lv = np.full((len(SHIFTS), 0, 2), np.nan), np.full((len(lv), 0, 2), np.nan)
    # frames that match neither explanation well (they do not see this band as the reference does) sit out
    best_f = np.fmax(np.nanmax(np.where(np.isfinite(U[..., 0]), U[..., 0], -1), axis=0, initial=-1),
                     np.nanmax(np.where(np.isfinite(Lv[..., 0]), Lv[..., 0], -1), axis=0, initial=-1)) if U.shape[1] else np.zeros(0)
    vote = best_f >= LEVEL_ABSTAIN

    def pooled(X):
        r, n = X[:, vote, 0], X[:, vote, 1]
        w = np.where(np.isfinite(r), np.sqrt(np.nan_to_num(n)), 0.0)
        return np.where(w.sum(1) > 0, (np.nan_to_num(r) * w).sum(1) / np.maximum(w.sum(1), 1e-9), np.nan)
    up, lv = (pooled(U), pooled(Lv)) if vote.any() else (np.full(len(SHIFTS), np.nan), np.full(len(Lv), np.nan))
    ku = int(np.nanargmax(up)) if np.isfinite(up).any() else None
    kl = int(np.nanargmax(lv)) if np.isfinite(lv).any() else None
    res = dict(z=[round(z_lo, 2), round(z_hi, 2)], end=end,
               upright=None if ku is None else round(float(up[ku]), 3), upright_shift=None if ku is None else round(float(SHIFTS[ku]), 2),
               level=None if kl is None else round(float(lv[kl]), 3), level_h=None if kl is None else round(float(heights[kl]), 2),
               frames_voting=int(vote.sum()), frames=len(others))
    edge = z_hi if end == "bottom" else z_lo
    res["is_level"] = bool(kl is not None and (ku is None or lv[kl] >= up[ku] + LEVEL_MARGIN)
                           and abs(heights[kl] - edge) <= LEVEL_EDGE_M)
    return res


def level_ends(b, plane, A, Z, lo, hi, d0, d1, views, edges) -> dict:
    """Test the face's pieces (between ``edges``) for level surfaces from the
    bottom up and from the top down, each way stopping at the first piece that
    is wall, and never cutting more than ``LEVEL_MAX_CUT`` of the face's
    height. Returns the face's new bottom and top and every piece tested."""
    p0 = np.asarray(plane.p0, float)
    n3 = np.r_[np.asarray(plane.normal, float), 0.0] @ b.M[:3, :3].T
    centre = make_to_world(b, plane, A, Z, lo, hi)(d0, d1).mean(axis=0)
    ref = max(views, key=lambda f: float((views[f]["C"] - centre) @ n3 / np.linalg.norm(views[f]["C"] - centre)) * views[f]["mask"].mean())
    z0, z1 = edges[0], edges[-1]
    bottom, top, tests = z0, z1, []
    rows = lambda a_, b_: (Z[:, 0] >= a_ - 1e-6) & (Z[:, 0] < b_ - 1e-6)
    for k in range(len(edges) - 1):                                   # from the bottom up
        if edges[k + 1] - z0 > LEVEL_MAX_CUT * (z1 - z0):
            break
        r_ = rows(edges[k], edges[k + 1])
        if r_.sum() < 2:
            continue
        t = level_piece(b, plane, A[r_], Z[r_], lo, hi, d0, d1, {f: dict(w, mask=w["mask"][r_]) for f, w in views.items()}, ref, "bottom")
        tests.append(t)
        if not t["is_level"]:
            break
        bottom = edges[k + 1]
    for k in range(len(edges) - 1, 0, -1):                            # from the top down
        if z1 - edges[k - 1] > LEVEL_MAX_CUT * (z1 - z0) or edges[k - 1] < bottom:
            break
        r_ = rows(edges[k - 1], edges[k])
        if r_.sum() < 2:
            continue
        t = level_piece(b, plane, A[r_], Z[r_], lo, hi, d0, d1, {f: dict(w, mask=w["mask"][r_]) for f, w in views.items()}, ref, "top")
        tests.append(t)
        if not t["is_level"]:
            break
        top = edges[k - 1]
    return dict(bottom=round(float(bottom), 2), top=round(float(top), 2), ref=ref.split("/")[-1], tests=tests)


def lidar_depths(b, plane, d0, d1, lo, hi, z0, z1, bin_m=1.0, min_n=8):
    """Where the LiDAR's own wall returns stand, height by height, off the
    face's matched wall: [[z, shift, n]] for each ``bin_m`` with at least
    ``min_n`` returns within 2.5 m of that wall, inside the face's span."""
    import laspy
    path = b.dir / "lidar_local.laz"
    if not path.exists():
        return []
    las = laspy.read(path)
    xyz = np.column_stack([las.x, las.y, las.z])
    along, off = plane.local(xyz[:, :2])
    wall = d0 + (d1 - d0) * (along - lo) / (hi - lo)
    sh = off - wall
    keep = (along >= lo) & (along <= hi) & (np.abs(sh) <= 2.5) & (xyz[:, 2] >= z0) & (xyz[:, 2] < z1)
    out = []
    for z in np.arange(z0, z1, bin_m):
        m = keep & (xyz[:, 2] >= z) & (xyz[:, 2] < z + bin_m)
        if m.sum() >= min_n:
            out.append([round(float(z + bin_m / 2), 1), round(float(np.median(sh[m])), 2), int(m.sum())])
    return out


def oblique_crops(b, plane, views, cams, rgbs, d0, d1, lo, hi, z0, z1, steps, out_dir: Path, stem: str,
                  n: int = 3, max_px: int = 900):
    """Up to ``n`` of the frames that see the face, as they took it: from
    different camera heads where there is a choice, those seeing most of the
    face first; each cropped round the face with a margin, the matched wall's
    outline drawn in teal and any step in red. Returns [{file, frame, head}]."""
    import cv2
    p0 = np.asarray(plane.p0, float)
    dv = (np.asarray(plane.p1, float) - p0) / float(plane.length_m)
    nv = np.asarray(plane.normal, float)

    def world(a, z, d):
        xy = p0 + a * dv + d * nv
        return np.r_[xy, z] @ b.M[:3, :3].T + b.M[:3, 3]

    def depth_at(a):
        return d0 + (d1 - d0) * (a - lo) / (hi - lo)

    ranked = sorted(views, key=lambda f: -views[f]["mask"].mean())
    picked, heads = [], set()
    for f in ranked:                                   # one per head first
        h = f.split("/")[-1].split("_")[0]
        if h not in heads:
            picked.append(f)
            heads.add(h)
    for f in ranked:
        if f not in picked:
            picked.append(f)
    out = []
    for f in picked[:n]:
        c, (im, x0, y0) = cams[f], rgbs[f]

        def uv(P):
            pc = (P - c["C"]) @ c["R_cw"].T
            return np.array([c["K"][0, 0] * pc[0] / pc[2] + c["K"][0, 2] - x0, c["K"][1, 1] * pc[1] / pc[2] + c["K"][1, 2] - y0])
        quad = np.array([uv(world(a, z, depth_at(a))) for a, z in ((lo, z0), (hi, z0), (hi, z1), (lo, z1))])
        span = quad.max(0) - quad.min(0)
        m = 0.3 * span.max() + 40
        X0, Y0 = np.maximum(quad.min(0) - m, 0).astype(int)
        X1, Y1 = np.minimum(quad.max(0) + m, [im.shape[1], im.shape[0]]).astype(int)
        if X1 - X0 < 40 or Y1 - Y0 < 40:
            continue
        crop = im[Y0:Y1, X0:X1].copy()
        k = max(1.0, max(crop.shape[:2]) / max_px)
        crop = cv2.resize(crop, None, fx=1 / k, fy=1 / k, interpolation=cv2.INTER_AREA) if k > 1 else crop
        t = lambda P: tuple(((P - [X0, Y0]) / k).round().astype(int))
        w_ = max(2, int(round(2 * max(crop.shape[:2]) / 600)))
        cv2.polylines(crop, [np.array([t(q) for q in quad], np.int32)], True, (120, 124, 15), w_, cv2.LINE_AA)
        for s_ in steps:
            cv2.line(crop, t(uv(world(lo, s_["z"], depth_at(lo)))), t(uv(world(hi, s_["z"], depth_at(hi)))), (30, 38, 179),
                     w_ if s_.get("confirmed", True) else max(1, w_ // 2), cv2.LINE_AA)
        name = f"{stem}_{f.split('/')[-1]}.jpg"
        cv2.imwrite(str(out_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out.append(dict(file=f"obliques/{name}", frame=f.split("/")[-1], head=f.split("/")[-1].split("_")[0]))
    return out


def atlas_building(job) -> List[dict]:
    import base64

    import cv2

    from kyfromabove_3d.features import load_wall_planes
    from kyfromabove_3d.wall_match import pairs_apart
    (b, frames, vis, cache, index, faces, args) = job
    cams = {c["id"]: c for c in wv.cameras_in_tileset_frame(frames, b)}
    planes = {p.face: p for p in load_wall_planes(b.dir / "wall_planes.json")}
    profiles = json.loads((b.dir / "wall_profiles.json").read_text(encoding="utf-8"))
    imgs, rgbs = {}, {}
    for k, w in index.items():
        if w["building"] == b.key:
            im = cv2.imread(str(cache / k))
            rgbs[w["frame"]] = (im, w["x0"], w["y0"])
            imgs[w["frame"]] = (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32), w["x0"], w["y0"])
    out = []
    for r in faces:
        plane = planes.get(r["face"])
        if "d0" not in r:                                   # the sweep measured nothing here
            continue
        got = face_views(b, r["face"], plane, vis, cams, imgs) if plane is not None else None
        if got is None:
            continue
        A, Z, views, (lo, hi) = got
        if "untrimmed" in r:
            # the sweep's trimmed face: its rows and columns, its open wall on the matched wall, its kept frames
            if "f" not in _FIELD:
                _FIELD["f"] = height_field(args.scene)
            rs = np.flatnonzero((Z[:, 0] >= r["z"][0] - 0.06) & (Z[:, 0] <= r["z"][1] + 0.06))
            cs = np.flatnonzero((A[0] >= r["along"][0] - 0.01) & (A[0] < r["along"][1] - 0.01))
            if not len(rs) or not len(cs):
                continue
            ext = slice(rs[0], rs[-1] + 1), slice(cs[0], cs[-1] + 1)
            lo_t, hi_t = r["along"]
            wall = wall_cells(b, plane, A, Z, lo, hi, depth_at(r["d0"], r["d1"], lo_t, hi_t, lo),
                              depth_at(r["d0"], r["d1"], lo_t, hi_t, hi), _FIELD["f"])
            A, Z, views, (lo, hi) = slice_face(A, Z, views, ext, wall[ext], {f for f in views if f.split("/")[-1] in r["frames_kept"]})
        if len(views) < 2:
            continue
        to_world = make_to_world(b, plane, A, Z, lo, hi)
        pairs = pairs_apart(views, to_world(0.0, 0.0).mean(axis=0))
        # depth against height: 2 m strips every metre, shifted off the face's wall
        shifts = np.arange(-2.5, 2.5001, 0.25)
        zc = np.arange(Z.min() + 1.0, Z.max() - 0.99, 1.0)
        strips = [(Z >= z - 1.0) & (Z < z + 1.0) for z in zc]
        sc = score_walls(to_world, A.shape, views, [(r["d0"] + x, r["d1"] + x) for x in shifts], pairs, strips) if pairs else None
        curve = []
        if sc is not None:
            for k, z in enumerate(zc):
                col = sc[:, 1 + k]
                if np.isfinite(col).any():
                    j = int(np.nanargmax(col))
                    second = np.nanmax(np.where(np.abs(shifts - shifts[j]) >= 0.75, col, np.nan)) if np.isfinite(col).sum() > 3 else np.nan
                    curve.append([round(float(z), 1), round(float(shifts[j]), 2), round(float(col[j]), 3),
                                  None if not np.isfinite(second) else round(float(col[j] - second), 3)])
        # the facade laid flat from the most face-on frame that sees most of it, at 0.1 m
        n3 = np.r_[np.asarray(plane.normal, float), 0.0] @ b.M[:3, :3].T
        centre = to_world(r["d0"], r["d1"]).mean(axis=0)

        def facing(f):
            ray = cams[f]["C"] - centre
            return float(ray @ n3 / np.linalg.norm(ray)) * views[f]["mask"].mean()
        best = max(views, key=facing)
        A1, Z1 = np.meshgrid(np.arange(lo, hi, 0.1), np.arange(Z.min(), Z.max(), 0.1))
        tw = make_to_world(b, plane, A1, Z1, lo, hi)
        P = tw(r["d0"], r["d1"])
        c, (im, x0, y0) = cams[best], rgbs[best]
        pc = (P - c["C"]) @ c["R_cw"].T
        u = (c["K"][0, 0] * pc[:, 0] / pc[:, 2] + c["K"][0, 2] - x0).astype(np.float32).reshape(A1.shape)
        v = (c["K"][1, 1] * pc[:, 1] / pc[:, 2] + c["K"][1, 2] - y0).astype(np.float32).reshape(A1.shape)
        flat = cv2.remap(im, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[::-1]   # up is up
        # as seen standing in front of it: the face runs p0 -> p1 left to right
        # only when its outward normal is on the right of that direction
        dv = (np.asarray(plane.p1, float) - np.asarray(plane.p0, float)) / float(plane.length_m)
        nv = np.asarray(plane.normal, float)
        mirrored = float(dv[0] * nv[1] - dv[1] * nv[0]) > 0
        if mirrored:
            flat = flat[:, ::-1]
        g = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY).astype(np.float32)
        valid = cv2.erode((g > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0   # not the image's own edges
        gy = np.abs(np.diff(g, axis=0, prepend=g[:1])) * valid
        gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1])) * valid
        rows = (gy.sum(1) / np.maximum(valid.sum(1), 1))[::-1]                 # bottom up, per 0.1 m
        cols = gx.sum(0) / np.maximum(valid.sum(0), 1)                       # left to right as seen
        lines = image_lines(rows, 0.1)
        bay, bay_strength = storey_period(cols, 0.1, 1.0, 9.0)
        s_ = min(1.0, 700 / max(flat.shape[:2]))
        small = cv2.resize(flat, None, fx=s_, fy=s_, interpolation=cv2.INTER_AREA) if s_ < 1 else flat
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        mean_d = 0.5 * (r["d0"] + r["d1"])
        band_rows = [dict(z_lo=x["z_lo"], z_hi=x["z_hi"], status=x["status"],
                          shift_m=None if x.get("offset_m") is None else round(x["offset_m"] - mean_d, 2))
                     for x in profiles.get(str(r["face"]), [])]
        z_lines = [round(float(Z.min()) + z, 2) for z in lines["lines"]]
        runs, steps, pieces = line_steps(to_world, A.shape, views, r["d0"], r["d1"], pairs, Z,
                                         pieces_of(float(Z.min()), float(Z.max()), z_lines)) if pairs else ([], [], [])
        reliable = r.get("reliable", True)
        steps = [s_ for s_ in steps if s_["confirmed"]] if reliable else []   # only steps the held-back frames agree on
        lidar = lidar_depths(b, plane, r["d0"], r["d1"], lo, hi, float(Z.min()), float(Z.max())) if r.get("lidar") else []
        obl = oblique_crops(b, plane, views, cams, rgbs, r["d0"], r["d1"], lo, hi, float(Z.min()), float(Z.max()), steps,
                            args.obliques_dir, f"{b.key[:8]}_{r['face']}") if args.obliques_dir else []
        out.append(dict(building=b.key, name=b.name, face=r["face"], measured=bool(r.get("lidar")), along=[lo, hi], obliques=obl,
                        reliable=reliable, why=r.get("why"), untrimmed=r.get("untrimmed"),
                        z=[round(float(Z.min()), 1), round(float(Z.max()), 1)], d0=r["d0"], d1=r["d1"], margin=r["margin"],
                        frame=best.split("/")[-1], frames=len(views), curve=curve, bands=band_rows,
                        floor_trace=[round(float(x), 2) for x in rows], bay_trace=[round(float(x), 2) for x in cols[::5]],
                        lines=z_lines, line_spacing_m=lines["spacing_m"], runs=runs, line_steps=steps, pieces=pieces,
                        slice_steps=find_steps(curve), lidar_depths=lidar,
                        line_regular=lines["regular"], bay_m=bay, bay_strength=bay_strength,
                        image="data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode(), image_px=list(small.shape[1::-1])))
    return out


def cmd_atlas(a) -> int:
    import multiprocessing as mp
    blds = wv.reference_buildings(a.reference, a.landmarks, wv.tileset_transforms(a.tileset), a.buildings)
    frames = wv.read_frames(a.frames)
    bands = json.loads((a.visibility / "bands.json").read_text(encoding="utf-8"))
    d = np.load(a.visibility / "points.npz")
    index = json.loads((a.cache / "index.json").read_text(encoding="utf-8"))
    faces = json.loads(a.faces.read_text(encoding="utf-8"))
    jobs = []
    for b in blds:
        mine = [r for r in faces if r["building"] == b.key]
        if not mine:
            continue
        ids = [i for i, r in enumerate(bands) if r["building"] == b.key and r["wall_fraction"] >= 0.5]
        sel = np.isin(d["band"], ids)
        vis = dict(face_of=np.array([bands[i]["face"] for i in d["band"][sel]]), along=d["along"][sel], z=d["z"][sel],
                   usable=d["usable"][:, sel], frames=d["frames"], bands_of={})
        jobs.append((b, frames, vis, a.cache, index, mine, a))
    if a.obliques_dir:
        a.obliques_dir.mkdir(parents=True, exist_ok=True)
    out, t0 = [], time.time()
    with mp.Pool(a.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(atlas_building, jobs)):
            out += res
            print(f"  {k + 1}/{len(jobs)} buildings, {len(out)} faces ({time.time() - t0:.0f} s)", flush=True)
    out.sort(key=lambda r: (r["name"], r["face"]))
    a.atlas_out.write_text(json.dumps(out, default=float), encoding="utf-8")
    print(f"{len(out)} faces -> {a.atlas_out} ({a.atlas_out.stat().st_size / 1e6:.1f} MB)")
    return 0


_FIELD: dict = {}


def flatten(b, plane, lo, hi, z_lo, z_hi, d0, d1, cam, rgb, step_m: float = 0.1) -> np.ndarray:
    """The face laid flat from one frame, on its matched wall, ``step_m`` a
    pixel, up the right way and as seen standing in front of it."""
    import cv2
    A1, Z1 = np.meshgrid(np.arange(lo, hi, step_m), np.arange(z_lo, z_hi, step_m))
    P = make_to_world(b, plane, A1, Z1, lo, hi)(d0, d1)
    im, x0, y0 = rgb
    pc = (P - cam["C"]) @ cam["R_cw"].T
    u = (cam["K"][0, 0] * pc[:, 0] / pc[:, 2] + cam["K"][0, 2] - x0).astype(np.float32).reshape(A1.shape)
    v = (cam["K"][1, 1] * pc[:, 1] / pc[:, 2] + cam["K"][1, 2] - y0).astype(np.float32).reshape(A1.shape)
    flat = cv2.remap(im, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[::-1]
    dv = (np.asarray(plane.p1, float) - np.asarray(plane.p0, float)) / float(plane.length_m)
    nv = np.asarray(plane.normal, float)
    return flat[:, ::-1] if float(dv[0] * nv[1] - dv[1] * nv[0]) > 0 else flat


def trim_building(job) -> List[dict]:
    """Each face's trim and frame checks; for the faces in ``show``, the
    pictures for the before/after sheet too."""
    import cv2

    from kyfromabove_3d.features import load_wall_planes
    (b, frames, vis, cache, index, faces, args, show) = job
    if "f" not in _FIELD:
        _FIELD["f"] = height_field(args.scene)
    field = _FIELD["f"]
    cams = {c["id"]: c for c in wv.cameras_in_tileset_frame(frames, b)}
    planes = {p.face: p for p in load_wall_planes(b.dir / "wall_planes.json")}
    imgs, rgbs = {}, {}
    for k, w in index.items():
        if w["building"] == b.key:
            im = cv2.imread(str(cache / k))
            rgbs[w["frame"]] = (im, w["x0"], w["y0"])
            imgs[w["frame"]] = (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32), w["x0"], w["y0"])
    out = []
    for r in faces:
        plane = planes.get(r["face"])
        got = face_views(b, r["face"], plane, vis, cams, imgs) if plane is not None else None
        if got is None:
            continue
        A, Z, views, (lo, hi) = got
        wall = wall_cells(b, plane, A, Z, lo, hi, r["d0"], r["d1"], field)
        ext = trim_extent(wall)
        rec = dict(building=b.key, name=b.name, face=r["face"], z=[round(float(Z.min()), 2), round(float(Z.max()), 2)],
                   along=[round(lo, 2), round(hi, 2)], wall_share=round(float(wall.mean()), 3), trimmed=None, frames={})
        if ext is not None:
            rs, cs = ext
            At, Zt, Wt = A[rs, cs], Z[rs, cs], wall[rs, cs]
            lo_t, hi_t = float(At.min()), float(At.max()) + GRID_M
            vt = {f: dict(w, mask=w["mask"][rs, cs]) for f, w in views.items()}
            # the trimmed face keeps the matched wall's line: its depths at the trimmed ends
            dl = lambda a: r["d0"] + (r["d1"] - r["d0"]) * (a - lo) / (hi - lo)
            rec["trimmed"] = dict(z=[round(float(Zt.min()), 2), round(float(Zt.max()) + GRID_M, 2)], along=[round(lo_t, 2), round(hi_t, 2)])
            rec["frames"] = frame_checks(b, plane, At, Zt, lo_t, hi_t, dl(lo_t), dl(hi_t), vt, Wt)
        if (b.key[:8], r["face"]) in show:
            n3 = np.r_[np.asarray(plane.normal, float), 0.0] @ b.M[:3, :3].T
            centre = make_to_world(b, plane, A, Z, lo, hi)(r["d0"], r["d1"]).mean(axis=0)
            facing = lambda f: float((cams[f]["C"] - centre) @ n3 / np.linalg.norm(cams[f]["C"] - centre)) * views[f]["mask"].mean()
            best = max(views, key=facing)
            rec["pic_before"] = flatten(b, plane, lo, hi, float(Z.min()), float(Z.max()), r["d0"], r["d1"], cams[best], rgbs[best])
            rec["pic_frames"] = {f: flatten(b, plane, lo, hi, float(Z.min()), float(Z.max()), r["d0"], r["d1"], cams[f], rgbs[f])
                                 for f in views}
            rec["pic_best"] = best
        out.append(rec)
    return out


def cmd_trim(a) -> int:
    """Trim every face to its open wall and check its frames; write
    ``trim.json`` and a before/after sheet of the faces in ``--show``."""
    import multiprocessing as mp
    blds = wv.reference_buildings(a.reference, a.landmarks, wv.tileset_transforms(a.tileset), a.buildings)
    frames = wv.read_frames(a.frames)
    bands = json.loads((a.visibility / "bands.json").read_text(encoding="utf-8"))
    d = np.load(a.visibility / "points.npz")
    index = json.loads((a.cache / "index.json").read_text(encoding="utf-8"))
    faces = json.loads(a.faces.read_text(encoding="utf-8"))
    show = {(k.split("_")[0], int(k.split("_")[1])) for k in a.show}
    jobs = []
    for b in blds:
        mine = [r for r in faces if r["building"] == b.key]
        if not mine:
            continue
        ids = [i for i, r in enumerate(bands) if r["building"] == b.key and r["wall_fraction"] >= 0.5]
        sel = np.isin(d["band"], ids)
        vis = dict(face_of=np.array([bands[i]["face"] for i in d["band"][sel]]), along=d["along"][sel], z=d["z"][sel],
                   usable=d["usable"][:, sel], frames=d["frames"], bands_of={})
        jobs.append((b, frames, vis, a.cache, index, mine, a, show))
    out = []
    with mp.Pool(a.workers) as pool:
        for res in pool.imap_unordered(trim_building, jobs):
            out += res
    out.sort(key=lambda r: (r["name"], r["face"]))
    pics = [r for r in out if "pic_before" in r]
    a.trim_out.write_text(json.dumps([{k: v for k, v in r.items() if not k.startswith("pic_")} for r in out], default=float), encoding="utf-8")
    print(f"{len(out)} faces -> {a.trim_out}")
    if pics:
        trim_sheet(pics, a.sheet)
    return 0


def level_building(job) -> List[dict]:
    import cv2

    from kyfromabove_3d.features import load_wall_planes
    (b, frames, vis, cache, index, faces, args, lines_of) = job
    cams = {c["id"]: c for c in wv.cameras_in_tileset_frame(frames, b)}
    planes = {p.face: p for p in load_wall_planes(b.dir / "wall_planes.json")}
    imgs = {}
    for k, w in index.items():
        if w["building"] == b.key:
            imgs[w["frame"]] = (cv2.imread(str(cache / k), cv2.IMREAD_GRAYSCALE).astype(np.float32), w["x0"], w["y0"])
    out = []
    for r in faces:
        plane = planes.get(r["face"])
        got = face_views(b, r["face"], plane, vis, cams, imgs) if plane is not None and "d0" in r else None
        if got is None:
            continue
        A, Z, views, (lo, hi) = got
        if len(views) < 2:
            continue
        z0, z1 = float(Z.min()), float(Z.max()) + GRID_M
        res = level_ends(b, plane, A, Z, lo, hi, r["d0"], r["d1"], views, pieces_of(z0, z1, lines_of.get((b.key, r["face"]), [])))
        out.append(dict(building=b.key, name=b.name, face=r["face"], z=[round(z0, 2), round(z1, 2)], **res))
    return out


def cmd_level(a) -> int:
    """Test the pieces at each end of every face for level surfaces (a deck, a
    lower roof, the roof beyond the top edge); write ``level.json``."""
    import multiprocessing as mp
    blds = wv.reference_buildings(a.reference, a.landmarks, wv.tileset_transforms(a.tileset), a.buildings)
    frames = wv.read_frames(a.frames)
    bands = json.loads((a.visibility / "bands.json").read_text(encoding="utf-8"))
    d = np.load(a.visibility / "points.npz")
    index = json.loads((a.cache / "index.json").read_text(encoding="utf-8"))
    faces = json.loads(a.faces.read_text(encoding="utf-8"))
    lines_of = {(r["building"], r["face"]): r["lines"] for r in json.loads(a.atlas.read_text(encoding="utf-8"))}
    jobs = []
    for b in blds:
        mine = [r for r in faces if r["building"] == b.key]
        if not mine:
            continue
        ids = [i for i, r in enumerate(bands) if r["building"] == b.key and r["wall_fraction"] >= 0.5]
        sel = np.isin(d["band"], ids)
        vis = dict(face_of=np.array([bands[i]["face"] for i in d["band"][sel]]), along=d["along"][sel], z=d["z"][sel],
                   usable=d["usable"][:, sel], frames=d["frames"], bands_of={})
        jobs.append((b, frames, vis, a.cache, index, mine, a, lines_of))
    out, t0 = [], time.time()
    with mp.Pool(a.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(level_building, jobs)):
            out += res
            print(f"  {k + 1}/{len(jobs)} buildings, {len(out)} faces ({time.time() - t0:.0f} s)", flush=True)
    out.sort(key=lambda r: (r["name"], r["face"]))
    a.level_out.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    cut = [r for r in out if r["bottom"] > r["z"][0] or r["top"] < r["z"][1]]
    print(f"{len(out)} faces; a level surface cut from {len(cut)} "
          f"(bottom {sum(r['bottom'] > r['z'][0] for r in out)}, top {sum(r['top'] < r['z'][1] for r in out)}) -> {a.level_out}")
    return 0


def trim_sheet(pics, path: Path) -> None:
    """One row per face: the face as it is today from its most face-on frame,
    what trimming cuts away shaded and the rest outlined, then each frame's
    view of the face, outlined teal if kept and red if dropped, with why."""
    import cv2
    font, H = cv2.FONT_HERSHEY_SIMPLEX, 240
    teal, red, ink = (120, 124, 15), (30, 38, 179), (40, 40, 40)

    def label(width, lines):
        lab = np.full((22 * len(lines) + 6, width, 3), 255, np.uint8)
        for k, (t, c) in enumerate(lines):
            cv2.putText(lab, t, (4, 18 + 22 * k), font, 0.5, c, 1, cv2.LINE_AA)
        return lab

    rows = []
    for r in pics:
        im = r["pic_before"].copy()
        z0, z1 = r["z"][0], r["z"][1] + GRID_M
        h, w = im.shape[:2]
        if r["trimmed"]:
            ya = max(0, int(round((z1 - r["trimmed"]["z"][1]) / 0.1))); yb = int(round((z1 - r["trimmed"]["z"][0]) / 0.1))
            xa = max(0, int(round((r["trimmed"]["along"][0] - r["along"][0]) / 0.1))); xb = int(round((r["trimmed"]["along"][1] - r["along"][0]) / 0.1))
            faded = cv2.addWeighted(im, 0.3, np.full_like(im, 255), 0.7, 0)
            faded[ya:yb, xa:xb] = im[ya:yb, xa:xb]
            im = faded
            cv2.rectangle(im, (xa, ya), (min(w, xb) - 1, min(h, yb) - 1), teal, 2)
        s_ = H / h
        W0 = max(1, int(w * s_))
        if W0 > 900:                                            # very long faces: squeeze, say so
            W0 = 900
        tiles = [np.vstack([cv2.resize(im, (W0, H)), label(W0, [("today, from the most face-on frame;", ink), ("shaded = trimmed away", ink)])])]
        for f, pic in r["pic_frames"].items():
            c = r["frames"].get(f)
            ww = min(600, max(60, int(pic.shape[1] * s_)))
            t = cv2.resize(pic, (ww, H))
            keep = bool(c and c["keep"])
            col = teal if keep else red
            cv2.rectangle(t, (1, 1), (ww - 2, H - 2), col, 4)
            why = []
            if c:
                if c["seen"] < MIN_FRAME_SEEN:
                    why.append(f"sees only {c['seen']:.0%}")
                if c["angle"] > MAX_FRAME_DEG:
                    why.append(f"{c['angle']:.0f} deg off straight")
                if c["agree"] is None or c["agree"] < MIN_FRAME_AGREE:
                    why.append("matches no other frame")
            lines = [("KEPT" if keep else "DROPPED", col)] + [(x, col) for x in why[:2]]
            if c and c["agree"] is not None:
                lines.append((f"match {c['agree']:.2f}", ink))
            lab = label(max(ww, 150), lines)
            t = cv2.copyMakeBorder(t, 0, 0, 0, lab.shape[1] - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) if lab.shape[1] > t.shape[1] else t
            tiles.append(np.vstack([t, lab]))
        hh = max(x.shape[0] for x in tiles)
        tiles = [cv2.copyMakeBorder(x, 0, hh - x.shape[0], 0, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x in tiles]
        row = np.hstack(tiles)
        cap = label(row.shape[1], [(f"{r['name']}, face {r['face']}: today {r['z'][0]:.1f} to {z1:.1f} m up"
                                    + (f"; trimmed {r['trimmed']['z'][0]:.1f} to {r['trimmed']['z'][1]:.1f} m" if r["trimmed"] else "; nothing left"), ink)])
        rows.append(np.vstack([cap, row, np.full((16, row.shape[1], 3), 255, np.uint8)]))
    Wd = max(x.shape[1] for x in rows)
    sheet = np.vstack([cv2.copyMakeBorder(x, 0, 0, 0, Wd - x.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x in rows])
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"sheet -> {path} ({sheet.shape[1]} x {sheet.shape[0]})")


PAGE_KEYS = ("building", "name", "face", "measured", "along", "z", "d0", "d1", "margin", "frame", "frames", "curve", "bands",
             "floor_trace", "bay_trace", "lines", "line_spacing_m", "line_regular", "runs", "line_steps", "pieces", "slice_steps", "lidar_depths", "image", "image_px", "obliques")


def cmd_page(a) -> int:
    """The atlas page: the template with the atlas inlined, and the oblique
    crops packed one JSON file per building (``obl/<key>.json``, crop path ->
    data URI) beside it, which the page fetches when a building is opened."""
    import base64
    import collections
    by = collections.defaultdict(dict)
    for f in sorted(a.obliques_dir.glob("*.jpg")):
        by[f.name.split("_")[0]]["obliques/" + f.name] = "data:image/jpeg;base64," + base64.b64encode(f.read_bytes()).decode()
    bundles = a.page_out.parent / "obl"
    bundles.mkdir(parents=True, exist_ok=True)
    for k, d in by.items():
        (bundles / f"{k}.json").write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")
    data = json.loads(a.atlas.read_text(encoding="utf-8"))
    s = json.dumps([{k: r[k] for k in PAGE_KEYS} for r in data], separators=(",", ":"), ensure_ascii=True)
    page = a.template.read_text(encoding="utf-8").replace("__ATLAS__", s.replace("</", "<\\/"))
    a.page_out.write_text(page, encoding="utf-8")
    print(f"{len(data)} faces -> {a.page_out} ({len(page) / 1e6:.1f} MB), {len(by)} oblique bundles in {bundles}")
    return 0


# ------------------------------------------------------------------ CLI

def sweep_building(job) -> List[dict]:
    import cv2

    from kyfromabove_3d.features import ArcWall, load_wall_planes
    (b, frames, vis, cache, index, args) = job
    cams = {c["id"]: c for c in wv.cameras_in_tileset_frame(frames, b)}
    planes = {p.face: p for p in load_wall_planes(b.dir / "wall_planes.json")}
    profiles = json.loads((b.dir / "wall_profiles.json").read_text(encoding="utf-8"))
    imgs = {}
    for k, w in index.items():
        if w["building"] == b.key:
            imgs[w["frame"]] = (cv2.imread(str(cache / k), cv2.IMREAD_GRAYSCALE).astype(np.float32), w["x0"], w["y0"])
    out = []
    for face in sorted(set(vis["face_of"].tolist())):
        plane = planes.get(int(face))
        if plane is None or isinstance(plane, ArcWall):
            continue
        if args.trim:
            if "f" not in _FIELD:
                _FIELD["f"] = height_field(args.scene)
            r = match_trimmed(b, face, plane, vis, cams, imgs, args, _FIELD["f"])
        else:
            r = match_face(b, face, plane, vis, cams, imgs, args)
        if r is None:
            continue
        rows = profiles.get(str(face), [])
        if "d0" not in r:                                    # trimmed away, or too few frames: nothing measured
            r.update(building=b.key, name=b.name, lidar=None)
            out.append(r)
            continue
        r.update(building=b.key, name=b.name, statuses="".join(x["status"][0] for x in rows),
                 verified=type(plane).__name__ == "VerifiedWallPlane",
                 lidar=lidar_line(b, face, plane, rows, *r["along"]))
        out.append(r)
    return out


def cmd_sweep(a) -> int:
    import multiprocessing as mp
    blds = wv.reference_buildings(a.reference, a.landmarks, wv.tileset_transforms(a.tileset), a.buildings)
    frames = wv.read_frames(a.frames)
    bands = json.loads((a.visibility / "bands.json").read_text(encoding="utf-8"))
    d = np.load(a.visibility / "points.npz")
    index = json.loads((a.cache / "index.json").read_text(encoding="utf-8"))
    jobs = []
    for b in blds:
        ids = [i for i, r in enumerate(bands) if r["building"] == b.key and r["wall_fraction"] >= 0.5]
        sel = np.isin(d["band"], ids)
        face_of = np.array([bands[i]["face"] for i in d["band"][sel]])
        bands_of = {}
        for i in ids:
            bands_of.setdefault(bands[i]["face"], []).append(bands[i])
        vis = dict(face_of=face_of, along=d["along"][sel], z=d["z"][sel], usable=d["usable"][:, sel], frames=d["frames"],
                   bands_of=bands_of)
        jobs.append((b, frames, vis, a.cache, index, a))
    t0, out = time.time(), []
    with mp.Pool(a.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(sweep_building, jobs)):
            out += res
            print(f"  {k + 1}/{len(jobs)} buildings, {len(out)} faces ({time.time() - t0:.0f} s)", flush=True)
    a.out.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    if a.trim:
        print(f"{sum(r.get('reliable', True) for r in out)} of {len(out)} faces reliable; "
              f"{sum('d0' not in r for r in out)} with nothing measured")
    meas = [r for r in out if r["lidar"]]
    if meas:
        e = np.array([max(abs(r["d0"] - r["lidar"][0]), abs(r["d1"] - r["lidar"][1])) for r in meas])
        print(f"{len(meas)} faces with a LiDAR line: both ends within 0.5 m on {np.mean(e <= 0.5) * 100:.0f}%, "
              f"within 1 m on {np.mean(e <= 1.0) * 100:.0f}% (before any cut)")
    print(f"-> {a.out}")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sweep", help="each straight face's wall from the frames that see it")
    p.add_argument("--reference", type=Path, default=ROOT / "configs" / "audit_reference.yaml")
    p.add_argument("--landmarks", type=Path, default=ROOT / "configs" / "landmarks.yaml")
    p.add_argument("--tileset", type=Path, default=ROOT / "outputs" / "downtown")
    p.add_argument("--buildings", type=Path, default=ROOT / "outputs" / "downtown" / "buildings")
    p.add_argument("--frames", type=Path, default=ROOT / "outputs" / "downtown" / "frames_2024s1.gpkg")
    p.add_argument("--visibility", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference")
    p.add_argument("--cache", type=Path, default=ROOT / "data" / "derived" / "frames_cache")
    p.add_argument("--depth-min", type=float, default=-8.0)
    p.add_argument("--depth-max", type=float, default=3.0)
    p.add_argument("--rival-m", type=float, default=0.75)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--scene", type=Path, default=ROOT / "data" / "derived" / "video" / "downtown_scene.npz",
                   help="every downtown mesh, in the tileset's frame (register_video.py scene)")
    p.add_argument("--trim", action=argparse.BooleanOptionalAction, default=False,
                   help="trim each face to its open wall and keep only frames that see it properly (off: on the "
                        "reference set it lost right steps as often as wrong ones, see faces_trimmed.json)")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference" / "faces.json")
    t = sub.add_parser("atlas", help="each face laid flat, with its depth against height and its pattern traces")
    for act in p._actions[1:]:
        if act.dest not in ("out",):
            t._add_action(act)
    t.add_argument("--faces", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference" / "faces.json")
    t.add_argument("--atlas-out", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference" / "atlas.json")
    t.add_argument("--obliques-dir", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference" / "obliques",
                   help="where the faces' oblique crops go (beside the atlas page)")
    m = sub.add_parser("trim", help="trim each face to its open wall and check its frames; a before/after sheet")
    for act in t._actions[1:]:
        if act.dest not in ("atlas_out", "obliques_dir"):
            m._add_action(act)
    m.add_argument("--show", nargs="*", default=[], help="faces for the sheet, as <building key's first 8>_<face>")
    m.add_argument("--trim-out", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference" / "trim.json")
    m.add_argument("--sheet", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference" / "trim_sheet.jpg")
    lv = sub.add_parser("level", help="test the pieces at each end of every face: wall, or a level surface?")
    for act in t._actions[1:]:
        if act.dest not in ("atlas_out", "obliques_dir"):
            lv._add_action(act)
    lv.add_argument("--atlas", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference" / "atlas.json",
                    help="the atlas whose image lines cut the faces into pieces")
    lv.add_argument("--level-out", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference" / "level.json")
    g = sub.add_parser("page", help="the atlas page, with the oblique crops packed per building beside it")
    g.add_argument("--atlas", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference" / "atlas.json")
    g.add_argument("--template", type=Path, default=ROOT / "tools" / "facade_atlas.html")
    g.add_argument("--obliques-dir", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference" / "obliques")
    g.add_argument("--page-out", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference" / "facade-atlas.html")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = parse_args(argv)
    return dict(sweep=cmd_sweep, atlas=cmd_atlas, trim=cmd_trim, level=cmd_level, page=cmd_page)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
