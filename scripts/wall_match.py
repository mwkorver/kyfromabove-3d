#!/usr/bin/env python3
"""Find how far back a wall stands by matching the aerial frames that see it.

``wall_visibility.py`` found which frames see each 5 m band of wall clearly.
A band's direction is the footprint's; its depth -- how far in or out of the
footprint edge the wall stands -- is what the LiDAR gives where it saw the
wall, and nothing gives where it did not (57 % of downtown's bands). Two
frames laid onto a trial wall at the right depth show the same texture in
the same place; at the wrong depth they slide apart. Sweeping the trial wall
through the possible depths and keeping the one where the frames agree best
gives the depth; scored first on bands the LiDAR measured.

    fetch   one full-resolution window per building per frame that sees
            its walls, read from the public frames (tiled JPEG, read by
            tile), saved under ``--cache`` and skipped when already there
    sweep   for every band two frames see from directions ``--min-baseline``
            degrees apart: each frame laid onto the trial wall at 10 cm a
            pixel (only where ``wall_visibility.py`` found the frame sees
            it), the pattern compared between every such pair (high-passed,
            normalised correlation) at depths from ``--depth-min`` to
            ``--depth-max`` m off the footprint edge (+ out), every 25 cm and
            then every 5 cm about the best. Per band: the depth, how well the
            frames agree there, and by how much it beats the best depth at
            least ``--rival-m`` away (a facade of repeated windows also
            agrees one bay off). Where the LiDAR measured the band, its depth
            goes beside it.
    show    one band: two frames laid onto the wall at chosen depths, as a
            red/cyan overlay (grey where they agree), and its agreement curve
    score   the sweep against the LiDAR, on the bands it measured: a
            confidence cut (agreement and margin) chosen on half the
            buildings and scored on the other half, both ways round; the
            error by building, by the way the wall faces and by height;
            figures to ``--figures``
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

ROOT = Path(__file__).resolve().parents[1]


def windows(vis_dir: Path, blds, frames, margin_px: int = 64, min_points: int = 20):
    """For each building and each frame that sees at least ``min_points`` of
    its wall points: the window enclosing them, in full-frame pixels."""
    d = np.load(vis_dir / "points.npz")
    bands = json.loads((vis_dir / "bands.json").read_text(encoding="utf-8"))
    fids, usable, X = list(d["frames"]), d["usable"], d["xyz"].astype(np.float64)
    bkey = np.array([bands[i]["building"] for i in d["band"]])
    out = []
    for b in blds:
        cams = {c["id"]: c for c in wv.cameras_in_tileset_frame(frames, b)}
        mine = bkey == b.key
        for j, fid in enumerate(fids):
            m = mine & usable[j]
            if m.sum() < min_points:
                continue
            c = cams[fid]
            pc = (X[m] - c["C"]) @ c["R_cw"].T
            u = c["K"][0, 0] * pc[:, 0] / pc[:, 2] + c["K"][0, 2]
            v = c["K"][1, 1] * pc[:, 1] / pc[:, 2] + c["K"][1, 2]
            x0, y0 = int(max(0, u.min() - margin_px)), int(max(0, v.min() - margin_px))
            x1, y1 = int(min(c["W"], u.max() + margin_px)), int(min(c["H"], v.max() + margin_px))
            out.append(dict(building=b.key, frame=fid, x0=x0, y0=y0, w=x1 - x0, h=y1 - y0, points=int(m.sum())))
    return out


def window_file(cache: Path, w: dict) -> Path:
    return cache / w["building"][:36] / (w["frame"].split("/")[-1] + ".png")


def cmd_fetch(a) -> int:
    import cv2

    from kyfromabove_3d.imagery import FrameImageReader, cog_url
    blds = wv.reference_buildings(a.reference, a.landmarks, wv.tileset_transforms(a.tileset), a.buildings)
    frames = wv.read_frames(a.frames)
    url = {str(f["ID"]): f["S3URL"] for f in frames}
    wins = windows(a.visibility, blds, frames)
    a.cache.mkdir(parents=True, exist_ok=True)
    index_path = a.cache / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    todo = [w for w in wins if not window_file(a.cache, w).exists()]
    print(f"{len(wins)} windows, {len(wins) - len(todo)} already here; fetching {len(todo)} "
          f"({sum(w['w'] * w['h'] for w in todo) / 1e6:.0f} Mpx)", flush=True)
    t0, got = time.time(), 0
    for k, w in enumerate(todo):
        img = FrameImageReader(cog_url(url[w["frame"]]), frame_id=w["frame"]).read_window(w["x0"], w["y0"], w["w"], w["h"])
        p = window_file(a.cache, w)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), cv2.cvtColor(img.rgb, cv2.COLOR_RGB2BGR))
        index[str(p.relative_to(a.cache))] = w
        got += p.stat().st_size
        if (k + 1) % 20 == 0 or k + 1 == len(todo):
            index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
            print(f"  {k + 1}/{len(todo)} windows, {got / 1e6:.0f} MB on disk, {time.time() - t0:.0f} s", flush=True)
    index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    return 0


# ------------------------------------------------------------------ the sweep

def band_geometry(b, rows_by_band):
    """For each band of the building: a function (along, offset) -> plan xy
    in the building's frame, and the LiDAR's offset where it measured one."""
    from kyfromabove_3d.features import ArcWall, load_wall_planes
    planes = {p.face: p for p in load_wall_planes(b.dir / "wall_planes.json")}
    profiles = json.loads((b.dir / "wall_profiles.json").read_text(encoding="utf-8"))
    out = {}
    for i, r in rows_by_band.items():
        p = planes[r["face"]]
        row = next(x for x in profiles[str(r["face"])] if abs(x["z_lo"] - r["z_lo"]) < 1e-6)
        if isinstance(p, ArcWall):
            to_xy = (lambda p: lambda a, o: p.world(a, np.full_like(a, o)))(p)
        else:
            L = float(np.hypot(*(np.asarray(p.p1) - np.asarray(p.p0))))
            d, n, p0 = (np.asarray(p.p1) - np.asarray(p.p0)) / L, np.asarray(p.normal, float), np.asarray(p.p0, float)
            to_xy = (lambda p0, d, n: lambda a, o: p0[None, :] + a[:, None] * d[None, :] + o * n[None, :])(p0, d, n)
        lidar = row["offset_m"] if row["status"] == "observed" else None
        out[i] = (to_xy, lidar)
    return out


def bilinear_gray(img: np.ndarray, u: np.ndarray, v: np.ndarray):
    """Luminance of ``img`` at (u, v), and where it is inside the image."""
    H, W = img.shape
    ok = (u >= 0) & (u <= W - 1.001) & (v >= 0) & (v <= H - 1.001)
    u, v = np.clip(u, 0, W - 1.001), np.clip(v, 0, H - 1.001)
    x0, y0 = u.astype(int), v.astype(int)
    fx, fy = u - x0, v - y0
    val = (img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy) + (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy
    return val, ok


def sweep_band(geom, A, Z, masks, frames_img, cams, M, offsets, sigma_px, pairs):
    """Agreement between the frames' views of the trial wall at each offset."""
    from kyfromabove_3d.texture_check import highpass, masked_ncc
    to_xy, _ = geom
    nz, na = A.shape
    scores = []
    for o in offsets:
        xy = to_xy(A.ravel(), o)
        P = np.column_stack([xy, Z.ravel()]) @ M[:3, :3].T + M[:3, 3]
        views = {}
        for f in masks:
            c, (img, x0, y0) = cams[f], frames_img[f]
            pc = (P - c["C"]) @ c["R_cw"].T
            u = c["K"][0, 0] * pc[:, 0] / pc[:, 2] + c["K"][0, 2] - x0
            v = c["K"][1, 1] * pc[:, 1] / pc[:, 2] + c["K"][1, 2] - y0
            g, ok = bilinear_gray(img, u, v)
            m = masks[f] & ok.reshape(nz, na)
            views[f] = (highpass(g.reshape(nz, na), m, sigma_px), m)
        s, w = 0.0, 0.0
        for f1, f2 in pairs:
            m = views[f1][1] & views[f2][1]
            n = int(m.sum())
            if n < 400:
                continue
            r = masked_ncc(views[f1][0], views[f2][0], m)
            if np.isfinite(r):
                s += r * np.sqrt(n)
                w += np.sqrt(n)
        scores.append(s / w if w > 0 else np.nan)
    return np.array(scores)


def sweep_building(job) -> List[dict]:
    import cv2
    (b, frames, rows, vis, cache, index, args) = job
    cams = {c["id"]: c for c in wv.cameras_in_tileset_frame(frames, b)}
    geom = band_geometry(b, rows)
    win = {(w["building"], w["frame"]): (k, w) for k, w in index.items()}
    imgs: dict = {}
    out = []
    step = 0.1
    for i, r in rows.items():
        if not r.get("stereo") or r["wall_fraction"] < 0.5:
            continue
        sel = vis["band"] == i
        sa, sz, us = vis["along"][sel], vis["z"][sel], vis["usable"][:, sel]
        seeing = [j for j in range(us.shape[0]) if us[j].mean() >= 0.5 and (b.key, vis["frames"][j]) in win]
        if len(seeing) < 2:
            continue
        a_lo, a_hi = sa.min() - 0.5, sa.max() + 0.5
        z_lo, z_hi = sz.min() - 0.5, sz.max() + 0.5
        A, Z = np.meshgrid(np.arange(a_lo, a_hi, step), np.arange(z_lo, z_hi, step))
        # each grid cell takes the visibility of its nearest 1 m sample
        ka = np.clip(np.round(A - sa.min()).astype(int), 0, None)
        kz = np.clip(np.round(Z - sz.min()).astype(int), 0, None)
        lut = -np.ones((kz.max() + 1, ka.max() + 1), int)
        lut[np.round(sz - sz.min()).astype(int), np.round(sa - sa.min()).astype(int)] = np.arange(len(sa))
        near = lut[kz, ka]
        masks, fids = {}, []
        for j in seeing:
            f = str(vis["frames"][j])
            masks[f] = (near >= 0) & us[j][np.maximum(near, 0)]
            fids.append(f)
            if f not in imgs:
                k, w = win[(b.key, f)]
                g = cv2.imread(str(cache / k), cv2.IMREAD_GRAYSCALE)
                imgs[f] = (g.astype(np.float32), w["x0"], w["y0"])
        centre = np.array([A.mean(), 0.0])
        cxy = geom[i][0](np.array([centre[0]]), 0.0)[0]
        cP = np.r_[cxy, Z.mean()] @ b.M[:3, :3].T + b.M[:3, 3]
        rays = {f: (cams[f]["C"] - cP) / np.linalg.norm(cams[f]["C"] - cP) for f in fids}
        pairs = [(f1, f2) for x, f1 in enumerate(fids) for f2 in fids[x + 1:]
                 if np.degrees(np.arccos(np.clip(rays[f1] @ rays[f2], -1, 1))) >= args.min_baseline]
        if not pairs:
            continue
        coarse = np.arange(args.depth_min, args.depth_max + 1e-9, 0.25)
        sc = sweep_band(geom[i], A, Z, masks, imgs, cams, b.M, coarse, args.sigma_px, pairs)
        if not np.isfinite(sc).any():
            continue
        k = int(np.nanargmax(sc))
        fine = np.arange(coarse[k] - 0.3, coarse[k] + 0.3001, 0.05)
        sf = sweep_band(geom[i], A, Z, masks, imgs, cams, b.M, fine, args.sigma_px, pairs)
        kf = int(np.nanargmax(sf))
        best, peak = float(fine[kf]), float(sf[kf])
        far = np.abs(coarse - best) >= args.rival_m
        rival = float(np.nanmax(sc[far])) if np.isfinite(sc[far]).any() else float("nan")
        lidar = geom[i][1]
        out.append(dict(r, band=i, depth_m=round(best, 2), agreement=round(peak, 3), rival=round(rival, 3),
                        margin=round(peak - rival, 3), rival_depth_m=float(coarse[far][np.nanargmax(sc[far])]) if np.isfinite(rival) else None,
                        pairs=len(pairs), frames_used=fids, lidar_m=lidar,
                        curve=[[round(float(o), 2), None if not np.isfinite(v) else round(float(v), 3)] for o, v in zip(coarse, sc)]))
    return out


def cmd_sweep(a) -> int:
    import multiprocessing as mp
    blds = wv.reference_buildings(a.reference, a.landmarks, wv.tileset_transforms(a.tileset), a.buildings)
    frames = wv.read_frames(a.frames)
    bands = json.loads((a.visibility / "bands.json").read_text(encoding="utf-8"))
    d = np.load(a.visibility / "points.npz")
    vis_all = {k: d[k] for k in ("band", "along", "z", "usable", "frames")}
    index = json.loads((a.cache / "index.json").read_text(encoding="utf-8"))
    jobs = []
    for b in blds:
        rows = {i: r for i, r in enumerate(bands) if r["building"] == b.key and (not a.status or r["status"] in a.status)}
        sel = np.isin(vis_all["band"], list(rows))
        vis = dict(band=vis_all["band"][sel], along=vis_all["along"][sel], z=vis_all["z"][sel],
                   usable=vis_all["usable"][:, sel], frames=vis_all["frames"])
        jobs.append((b, frames, rows, vis, a.cache, index, a))
    t0 = time.time()
    out = []
    with mp.Pool(a.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(sweep_building, jobs)):
            out += res
            print(f"  {k + 1}/{len(jobs)} buildings, {len(out)} bands swept ({time.time() - t0:.0f} s)", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out), encoding="utf-8")
    meas = [r for r in out if r["lidar_m"] is not None]
    if meas:
        e = np.abs(np.array([r["depth_m"] - r["lidar_m"] for r in meas]))
        print(f"{len(meas)} bands the LiDAR measured: depth within 0.3 m of it on {np.mean(e <= 0.3) * 100:.0f}%, "
              f"within 1 m on {np.mean(e <= 1.0) * 100:.0f}% (median error {np.median(e):.2f} m), before any confidence cut")
    print(f"-> {a.out}")
    return 0


# ------------------------------------------------------------------ one band, shown

def rectify(b, geom, cam, img, x0, y0, A, Z, offset):
    """One frame laid onto the trial wall at ``offset``: an (nz, na, 3) image."""
    import cv2
    to_xy, _ = geom
    P = np.column_stack([to_xy(A.ravel(), offset), Z.ravel()]) @ b.M[:3, :3].T + b.M[:3, 3]
    pc = (P - cam["C"]) @ cam["R_cw"].T
    u = (cam["K"][0, 0] * pc[:, 0] / pc[:, 2] + cam["K"][0, 2] - x0).astype(np.float32)
    v = (cam["K"][1, 1] * pc[:, 1] / pc[:, 2] + cam["K"][1, 2] - y0).astype(np.float32)
    out = cv2.remap(img, u.reshape(A.shape), v.reshape(A.shape), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return out[::-1]                                                         # up is up


def cmd_show(a) -> int:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = {r["band"]: r for r in json.loads(a.match.read_text(encoding="utf-8"))}
    r = rows[a.band]
    blds = {b.key: b for b in wv.reference_buildings(a.reference, a.landmarks, wv.tileset_transforms(a.tileset), a.buildings)}
    b = blds[r["building"]]
    cams = {c["id"]: c for c in wv.cameras_in_tileset_frame(wv.read_frames(a.frames), b)}
    geom = band_geometry(b, {a.band: r})[a.band]
    index = json.loads((a.cache / "index.json").read_text(encoding="utf-8"))
    win = {(w["building"], w["frame"]): (k, w) for k, w in index.items()}
    f1, f2 = r["frames_used"][:2]
    if len(r["frames_used"]) > 2:                                            # the pair furthest apart
        best = (-1.0, f1, f2)
        for x, g1 in enumerate(r["frames_used"]):
            for g2 in r["frames_used"][x + 1:]:
                best = max(best, (float(np.linalg.norm(cams[g1]["C"] - cams[g2]["C"])), g1, g2))
        f1, f2 = best[1], best[2]
    along = r["along"]
    A, Z = np.meshgrid(np.arange(along[0], along[1], 0.1), np.arange(r["z_lo"], r["z_hi"], 0.1))
    offsets = [("LiDAR", r["lidar_m"]), ("matching", r["depth_m"]), ("2 m further out", r["depth_m"] + 2.0)]
    offsets = [(n, o) for n, o in offsets if o is not None]
    views = []
    for name, o in offsets:
        g = []
        for f in (f1, f2):
            k, w = win[(b.key, f)]
            img = cv2.imread(str(a.cache / k), cv2.IMREAD_GRAYSCALE)
            g.append(rectify(b, geom, cams[f], img, w["x0"], w["y0"], A, Z, o))
        ana = np.dstack([g[1], g[1], g[0]])                                  # BGR: frame 1 red, frame 2 cyan
        views.append((name, o, ana))
    fig = plt.figure(figsize=(14, 2.2 * len(views) + 2.6))
    gs = fig.add_gridspec(len(views) + 1, 1, height_ratios=[1] * len(views) + [1.2])
    for k, (name, o, ana) in enumerate(views):
        ax = fig.add_subplot(gs[k])
        ax.imshow(ana[:, :, ::-1], aspect="equal")
        ax.set_axis_off()
        ax.set_title(f"wall laid {o:+.2f} m off the footprint edge ({name}): grey where the two frames agree, red/cyan fringes where they do not", fontsize=9)
    ax = fig.add_subplot(gs[-1])
    c = np.array([[x, np.nan if y is None else y] for x, y in r["curve"]], float)
    ax.plot(c[:, 0], c[:, 1], color="tab:orange", lw=2)
    if r["lidar_m"] is not None:
        ax.axvline(r["lidar_m"], color="tab:blue", ls="--", label="the LiDAR's depth")
    ax.axvline(r["depth_m"], color="tab:red", ls=":", label="matching's depth")
    ax.set_xlabel("trial depth off the footprint edge (m, + out)"); ax.set_ylabel("agreement")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"{r['name']}, face {r['face']}, {r['z_lo']:.0f}-{r['z_hi']:.0f} m up ({r['status']}); frames "
                 f"{f1.split('/')[-1]} (red) and {f2.split('/')[-1]} (cyan)", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out_png, dpi=100)
    print(f"-> {a.out_png}")
    return 0


# ------------------------------------------------------------------ the score

def choose_cut(rows, errors, max_gross: float = 0.03, gross_m: float = 1.0):
    """The loosest (agreement, margin) cut that keeps the share of accepted
    bands off by more than ``gross_m`` at or under ``max_gross``: a cut
    against matching that failed, not against the lean (see ``lean``)."""
    ag = np.array([r["agreement"] for r in rows])
    mg = np.nan_to_num(np.array([r["margin"] for r in rows], float), nan=1.0)
    best = (0.95, 1.0, 0)
    for ta in np.arange(0.0, 0.9, 0.025):
        for tm in np.arange(0.0, 0.2, 0.01):
            acc = (ag >= ta) & (mg >= tm)
            if acc.sum() >= 10 and np.mean(np.abs(errors[acc]) > gross_m) <= max_gross and acc.sum() > best[2]:
                best = (float(ta), float(tm), int(acc.sum()))
    return best[0], best[1]


def lean_removed(err: np.ndarray, building: np.ndarray, min_others: int = 3) -> np.ndarray:
    """Each band's error less the median error of the *other* measured bands
    of its building -- what an unknown band of that building could be
    corrected by. NaN where the building has fewer than ``min_others``."""
    out = np.full(len(err), np.nan)
    for i in range(len(err)):
        o = building == building[i]
        o[i] = False
        if o.sum() >= min_others:
            out[i] = err[i] - np.median(err[o])
    return out


def face_heading(b, face: int) -> float:
    """Which way a face looks, degrees clockwise from north (the building's frame)."""
    from kyfromabove_3d.features import load_wall_planes
    p = next(q for q in load_wall_planes(b.dir / "wall_planes.json") if q.face == face)
    n = np.asarray(p.normal, float)
    return float(np.degrees(np.arctan2(n[0], n[1])) % 360)


def cmd_score(a) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = json.loads(a.match.read_text(encoding="utf-8"))
    meas = [r for r in rows if r["lidar_m"] is not None]
    unk = [r for r in rows if r["status"] == "unknown"]
    err = np.array([r["depth_m"] - r["lidar_m"] for r in meas])
    blds = {b.key: b for b in wv.reference_buildings(a.reference, a.landmarks, wv.tileset_transforms(a.tileset), a.buildings)}
    heading = {}
    for r in rows:
        k = (r["building"], r["face"])
        if k not in heading:
            heading[k] = face_heading(blds[r["building"]], r["face"])
    keys = sorted({r["building"] for r in meas})
    half = [set(keys[0::2]), set(keys[1::2])]
    report = dict(bands_measured=len(meas), bands_unknown=len(unk), folds=[])
    accepted = np.zeros(len(meas), bool)
    for k in (0, 1):
        tr = np.array([r["building"] in half[k] for r in meas])
        ta, tm = choose_cut([r for r, t in zip(meas, tr) if t], err[tr], a.max_gross)
        te = ~tr
        acc = te & (np.array([r["agreement"] for r in meas]) >= ta) & (np.nan_to_num(np.array([r["margin"] for r in meas], float), nan=1.0) >= tm)
        accepted |= acc
        e = np.abs(err[acc])
        report["folds"].append(dict(gross_over_1m=round(float(np.mean(e > 1.0)), 3), cut_agreement=round(ta, 3), cut_margin=round(tm, 3), test_bands=int(te.sum()), accepted=int(acc.sum()),
                                    within_03=round(float(np.mean(e <= 0.3)), 3), within_05=round(float(np.mean(e <= 0.5)), 3),
                                    within_1=round(float(np.mean(e <= 1.0)), 3), median_m=round(float(np.median(e)), 3),
                                    p90_m=round(float(np.percentile(e, 90)), 3)))
    edge = np.abs([r["lidar_m"] for r in meas])
    e_acc = np.abs(err[accepted])
    bk = np.array([r["building"] for r in meas])
    corr = np.abs(lean_removed(err, bk))
    c_acc = corr[accepted & np.isfinite(corr)]
    report["held_out"] = dict(accepted=int(accepted.sum()), of=len(meas), within_03=round(float(np.mean(e_acc <= 0.3)), 3),
                              within_1=round(float(np.mean(e_acc <= 1.0)), 3), median_m=round(float(np.median(e_acc)), 3),
                              within_05=round(float(np.mean(e_acc <= 0.5)), 3),
                              lean_removed=dict(bands=int(len(c_acc)), within_03=round(float(np.mean(c_acc <= 0.3)), 3),
                                                within_05=round(float(np.mean(c_acc <= 0.5)), 3), median_m=round(float(np.median(c_acc)), 3)),
                              edge_guess_within_03=round(float(np.mean(edge[accepted] <= 0.3)), 3),
                              edge_guess_within_05=round(float(np.mean(edge[accepted] <= 0.5)), 3),
                              edge_guess_median_m=round(float(np.median(edge[accepted])), 3))
    # the lean: by the way the wall faces, and by building
    hd = np.array([heading[(r["building"], r["face"])] for r in meas])
    quad = ((hd + 45) % 360 // 90).astype(int)
    report["lean_by_facing"] = {n: dict(bands=int((quad == q).sum()), signed_median_m=round(float(np.median(err[quad == q])), 3))
                                for q, n in enumerate(("north", "east", "south", "west")) if (quad == q).any()}
    report["lean_by_building"] = {blds[k].name: round(float(np.median(err[[r["building"] == k for r in meas]])), 3) for k in keys}
    # the cut, chosen on all measured bands, applied to the unknown ones
    ta, tm = choose_cut(meas, err, a.max_gross)
    ua = [r for r in unk if r["agreement"] >= ta and (r["margin"] if r["margin"] == r["margin"] else 1.0) >= tm]
    report["unknown"] = dict(cut_agreement=round(ta, 3), cut_margin=round(tm, 3), swept=len(unk), accepted=len(ua),
                             moved_over_1m=int(sum(abs(r["depth_m"]) > 1.0 for r in ua)),
                             depth_median_m=round(float(np.median([r["depth_m"] for r in ua])), 3) if ua else None)
    a.report.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))

    a.figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    xs = np.linspace(0, 2.0, 201)
    for vals, lab, col, ls in ((np.abs(err), f"matching, every band ({len(err)})", "tab:orange", "-"),
                               (e_acc, f"matching, past the held-out cut ({len(e_acc)})", "tab:red", "-"),
                               (c_acc, f"the same, each building's lean taken out ({len(c_acc)})", "tab:purple", ":"),
                               (edge, "today's guess: wall on the footprint edge", "0.4", "--")):
        ax[0].plot(xs, [np.mean(vals <= x) * 100 for x in xs], color=col, ls=ls, lw=2, label=lab)
    ax[0].axvline(0.3, color="0.8", lw=1)
    ax[0].set_xlabel("how far from the LiDAR's depth (m)"); ax[0].set_ylabel("% of measured bands within that")
    ax[0].set_title("Depth from matching, against the LiDAR"); ax[0].legend(fontsize=8, loc="lower right"); ax[0].grid(alpha=0.3)
    lid = np.array([r["lidar_m"] for r in meas]); dep = np.array([r["depth_m"] for r in meas])
    ax[1].scatter(lid[~accepted], dep[~accepted], s=10, color="0.7", label="below the cut")
    ax[1].scatter(lid[accepted], dep[accepted], s=10, color="tab:red", label="past the cut")
    ax[1].plot([-8, 3], [-8, 3], color="k", lw=0.8)
    ax[1].set_xlabel("LiDAR: wall's depth off the footprint edge (m, + out)"); ax[1].set_ylabel("matching: the same (m)")
    ax[1].set_title("Band by band"); ax[1].legend(fontsize=8); ax[1].set_aspect("equal")
    names = sorted(report["lean_by_building"], key=lambda n: report["lean_by_building"][n])
    nb = [sum(blds[r["building"]].name == n for r in meas) for n in names]
    ax[2].barh(range(len(names)), [report["lean_by_building"][n] for n in names], color="tab:purple")
    ax[2].set_yticks(range(len(names))); ax[2].set_yticklabels([f"{n[:26]} ({c})" for n, c in zip(names, nb)], fontsize=7)
    ax[2].axvline(0, color="k", lw=0.8); ax[2].set_xlabel("matching minus LiDAR, median (m; - = further in)")
    ax[2].set_title("The lean, building by building")
    fig.suptitle("Wall depth by matching aerial frames, on the 19 reference buildings' bands the LiDAR measured", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.figures / "wall_match_score.png", dpi=100)
    print(f"-> {a.report}, {a.figures / 'wall_match_score.png'}")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch", help="the frame windows the matching reads")
    p.add_argument("--reference", type=Path, default=ROOT / "configs" / "audit_reference.yaml")
    p.add_argument("--landmarks", type=Path, default=ROOT / "configs" / "landmarks.yaml")
    p.add_argument("--tileset", type=Path, default=ROOT / "outputs" / "downtown")
    p.add_argument("--buildings", type=Path, default=ROOT / "outputs" / "downtown" / "buildings")
    p.add_argument("--frames", type=Path, default=ROOT / "outputs" / "downtown" / "frames_2024s1.gpkg")
    p.add_argument("--visibility", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference",
                   help="wall_visibility.py's output folder")
    p.add_argument("--cache", type=Path, default=ROOT / "data" / "derived" / "frames_cache")
    q = sub.add_parser("sweep", parents=[], help="each band's depth from the frames that see it")
    for act in p._actions[1:]:
        q._add_action(act)
    q.add_argument("--status", nargs="*", default=["observed", "unknown"], help="which bands (by profile status)")
    q.add_argument("--depth-min", type=float, default=-8.0)
    q.add_argument("--depth-max", type=float, default=3.0)
    q.add_argument("--min-baseline", type=float, default=10.0)
    q.add_argument("--rival-m", type=float, default=0.75)
    q.add_argument("--sigma-px", type=float, default=8.0, help="high-pass scale at 10 cm a pixel")
    q.add_argument("--workers", type=int, default=6)
    q.add_argument("--out", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference" / "match.json")
    r = sub.add_parser("score", help="the sweep against the LiDAR where it measured the wall")
    for act in p._actions[1:]:
        r._add_action(act)
    r.add_argument("--match", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference" / "match.json")
    r.add_argument("--max-gross", type=float, default=0.03, help="the cut keeps the share of accepted bands off by over 1 m under this")
    r.add_argument("--report", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference" / "match_score.json")
    r.add_argument("--figures", type=Path, default=ROOT / "docs" / "figures")
    w = sub.add_parser("show", help="one band laid out at the LiDAR's depth, matching's, and 2 m off")
    for act in p._actions[1:]:
        w._add_action(act)
    w.add_argument("--match", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference" / "match.json")
    w.add_argument("--band", type=int, required=True, help="the band's index (match.json's 'band')")
    w.add_argument("--out-png", type=Path, required=True)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = parse_args(argv)
    return dict(fetch=cmd_fetch, sweep=cmd_sweep, score=cmd_score, show=cmd_show)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
