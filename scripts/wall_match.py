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
    sheet   accepted unknown bands at random, each laid out twice -- where
            the mesh has the wall now and where matching puts it -- as
            red/cyan with each frame's brightness evened out
    compare-runs  two rebuilds of the reference set, matching off and on
            (folders of each building's audit.json, texture_check.json,
            wall_profiles.json, match_report.json): the audit's own
            comparison, the texture check on the faces that got matched
            walls against the rest, and the pipeline's matched depths
            against this study's
    outlines  one building's matched walls drawn on a frame twice: where the
            mesh stands them now (the footprint edge) and where matching
            puts them -- whichever hugs the wall's edges is right
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

from kyfromabove_3d.stages import wall_visibility as wv  # noqa: E402

from kyfromabove_3d.wall_match import (  # noqa: E402
    choose_cut,
    is_accepted,
    lean_removed,
    match_band,
)

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
        out[i] = (to_xy, lidar, int(row.get("n_returns") or 0))
    return out


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
        views = {f: dict(K=cams[f]["K"], R_cw=cams[f]["R_cw"], C=cams[f]["C"], gray=imgs[f][0], x0=imgs[f][1],
                         y0=imgs[f][2], mask=masks[f]) for f in fids}
        to_xy = geom[i][0]

        def to_world(a, o, to_xy=to_xy, Z=Z):
            return np.column_stack([to_xy(a, o), Z.ravel()]) @ b.M[:3, :3].T + b.M[:3, 3]

        res = match_band(to_world, A, Z, views)
        if res is None:
            continue
        out.append(dict(r, band=i, lidar_m=geom[i][1], lidar_returns=geom[i][2], **res))
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
    to_xy = geom[0]
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


def cmd_outlines(a) -> int:
    import cv2
    rows = json.loads(a.match.read_text(encoding="utf-8"))
    rep = json.loads(a.report.read_text(encoding="utf-8"))["unknown"]
    blds = {b.key: b for b in wv.reference_buildings(a.reference, a.landmarks, wv.tileset_transforms(a.tileset), a.buildings)}
    b = blds[a.building]
    cut = (rep["cut_margin"], rep["cut_pattern"])
    sel = [r for r in rows if r["building"] == b.key and r["status"] == "unknown" and is_accepted(r, cut) and abs(r["depth_m"]) >= a.min_move]
    if not sel:
        print(f"[FAIL] no matched band of {b.name} moved {a.min_move} m or more")
        return 1
    cams = {c["id"]: c for c in wv.cameras_in_tileset_frame(wv.read_frames(a.frames), b)}
    geom = band_geometry(b, {r["band"]: r for r in sel})
    index = json.loads((a.cache / "index.json").read_text(encoding="utf-8"))
    win = {w["frame"]: (k, w) for k, w in index.items() if w["building"] == b.key}

    def outline(r, f, o):
        A = np.array([r["along"][0], r["along"][1], r["along"][1], r["along"][0]], float)
        Z = np.array([r["z_lo"], r["z_lo"], r["z_hi"], r["z_hi"]], float)
        P = np.column_stack([geom[r["band"]][0](A, o), Z]) @ b.M[:3, :3].T + b.M[:3, 3]
        c = cams[f]
        pc = (P - c["C"]) @ c["R_cw"].T
        return np.column_stack([c["K"][0, 0] * pc[:, 0] / pc[:, 2] + c["K"][0, 2], c["K"][1, 1] * pc[:, 1] / pc[:, 2] + c["K"][1, 2]])

    # the frame that sees the most of these bands, weighted by how far the two outlines differ in it
    def worth(f):
        return sum(np.abs(outline(r, f, 0.0) - outline(r, f, r["depth_m"])).max() for r in sel if f in r["frames_used"])
    f = max((f for f in win if any(f in r["frames_used"] for r in sel)), key=worth)
    k, w = win[f]
    img = cv2.imread(str(a.cache / k))
    shown = 0
    for r in sel:
        if f not in r["frames_used"]:
            continue
        for o, col in ((0.0, (0, 140, 255)), (r["depth_m"], (60, 220, 60))):
            q = (outline(r, f, o) - [w["x0"], w["y0"]]).round().astype(np.int32)
            cv2.polylines(img, [q], True, col, 2, cv2.LINE_AA)
        shown += 1
    s_ = min(1.0, 1600 / max(img.shape[:2]))
    img = cv2.resize(img, None, fx=s_, fy=s_, interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, f"{b.name}, {f.split('/')[-1]}: {shown} matched bands. orange = where the mesh stands the wall now, green = where matching puts it",
                (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    a.out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out_png), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{b.name}: {shown} bands drawn on {f.split('/')[-1]} -> {a.out_png}")
    return 0


def cmd_sheet(a) -> int:
    import cv2
    rows = json.loads(a.match.read_text(encoding="utf-8"))
    rep = json.loads(a.report.read_text(encoding="utf-8"))["unknown"]
    cut = (rep["cut_margin"], rep["cut_pattern"])
    pool = [r for r in rows if r["status"] == "unknown" and is_accepted(r, cut) and abs(r["depth_m"]) >= a.min_move]
    rng = np.random.default_rng(a.seed)
    pick = [pool[i] for i in rng.choice(len(pool), min(a.n, len(pool)), replace=False)]
    blds = {b.key: b for b in wv.reference_buildings(a.reference, a.landmarks, wv.tileset_transforms(a.tileset), a.buildings)}
    frames = wv.read_frames(a.frames)
    index = json.loads((a.cache / "index.json").read_text(encoding="utf-8"))
    win = {(w["building"], w["frame"]): (k, w) for k, w in index.items()}

    def even(g):
        m = g > 0
        x = g.astype(np.float32)
        mu, sd = (x[m].mean(), x[m].std() + 1e-3) if m.any() else (0.0, 1.0)
        return np.clip(128 + (x - mu) / sd * 45, 0, 255).astype(np.uint8)

    tiles, W = [], 400
    for r in pick:
        b = blds[r["building"]]
        cams = {c["id"]: c for c in wv.cameras_in_tileset_frame(frames, b)}
        geom = band_geometry(b, {r["band"]: r})[r["band"]]
        fu = r["frames_used"]
        f1, f2 = max(((g1, g2) for x, g1 in enumerate(fu) for g2 in fu[x + 1:]),
                     key=lambda q: np.linalg.norm(cams[q[0]]["C"] - cams[q[1]]["C"]))
        lo, hi = r["along"]
        mid = (lo + hi) / 2
        lo, hi = max(lo, mid - 10), min(hi, mid + 10)
        A, Z = np.meshgrid(np.arange(lo, hi, 0.1), np.arange(r["z_lo"], r["z_hi"], 0.1))
        pair = []
        for o in (0.0, r["depth_m"]):
            g = []
            for f in (f1, f2):
                k, w = win[(b.key, f)]
                img = cv2.imread(str(a.cache / k), cv2.IMREAD_GRAYSCALE)
                g.append(even(rectify(b, geom, cams[f], img, w["x0"], w["y0"], A, Z, o)))
            an = np.dstack([g[1], g[1], g[0]])
            pair.append(cv2.resize(an, (W, max(20, int(W * an.shape[0] / an.shape[1]))), interpolation=cv2.INTER_AREA))
        h = pair[0].shape[0]
        lab = np.full((24, 2 * W + 12, 3), 255, np.uint8)
        cv2.putText(lab, f"{r['name'][:28]}, {r['z_lo']:.0f}-{r['z_hi']:.0f} m up: matching {r['depth_m']:+.1f} m",
                    (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([lab, np.hstack([pair[0], np.full((h, 12, 3), 255, np.uint8), pair[1]]),
                                np.full((10, 2 * W + 12, 3), 255, np.uint8)]))
    cols = [np.vstack(tiles[k::3]) for k in range(3) if tiles[k::3]]
    H = max(len(c) for c in cols)
    cols = [cv2.copyMakeBorder(c, 0, H - len(c), 0, 30, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for c in cols]
    body = np.hstack(cols)
    head = np.full((44, body.shape[1], 3), 255, np.uint8)
    cv2.putText(head, "Each pair: left = two frames laid on the wall where the mesh has it now; right = at the depth matching found. "
                "One frame red, the other cyan.", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(head, f"Grey = the frames agree there; red/cyan fringes = they do not. {len(pick)} accepted unknown bands moved "
                f"{a.min_move} m or more, at random (seed {a.seed}).", (4, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    a.out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out_png), np.vstack([head, body]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(pick)} of {len(pool)} bands -> {a.out_png}")
    return 0


def cmd_compare_runs(a) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import audit_reference as ar
    cfg = ar.load_config()
    off_audit, on_audit, rows = {}, {}, []
    tex = {"matched": [], "other": []}
    study = {(r["building"], r["face"], round(r["z_lo"], 1)): r for r in json.loads(a.match.read_text(encoding="utf-8"))}
    port = []
    for b in cfg["buildings"]:
        key = b["id"]
        d_off, d_on = a.off / key, a.on / key
        if not (d_off / "audit.json").exists() or not (d_on / "audit.json").exists():
            print(f"  {b['name']}: missing from one run")
            continue
        off_audit[key] = json.loads((d_off / "audit.json").read_text(encoding="utf-8"))
        on_audit[key] = json.loads((d_on / "audit.json").read_text(encoding="utf-8"))
        prof = json.loads((d_on / "wall_profiles.json").read_text(encoding="utf-8"))
        lidar = ("observed", "bridged", "extended", "mirrored")
        # the faces the mesh drew matched walls on: with a LiDAR wall band too (build_mesh)
        matched_faces = {f for f, bands in prof.items() if any(x["status"] == "matched" for x in bands)
                         and any(x["status"] in lidar for x in bands)}
        t_off = json.loads((d_off / "texture_check.json").read_text(encoding="utf-8"))["faces"] if (d_off / "texture_check.json").exists() else {}
        t_on = json.loads((d_on / "texture_check.json").read_text(encoding="utf-8"))["faces"] if (d_on / "texture_check.json").exists() else {}
        for f in set(t_off) & set(t_on):
            if t_off[f].get("kind") == "roof" or t_off[f].get("ncc") is None or t_on[f].get("ncc") is None:
                continue
            tex["matched" if f.replace("face ", "") in matched_faces else "other"].append((b["name"], f, t_off[f]["ncc"], t_on[f]["ncc"]))
        rep = d_on / "match_report.json"
        n_m = 0
        if rep.exists():
            r = json.loads(rep.read_text(encoding="utf-8"))
            n_m = r["matched"]
            for x in r["bands"]:
                s_ = study.get((key, x["face"], round(x["z_lo"], 1)))
                if s_ is not None and x["status"] == "unknown":
                    port.append((x["depth_m"], s_["depth_m"], x["accepted"], is_accepted(s_)))
        rows.append(dict(name=b["name"], matched_bands=n_m, matched_faces=len(matched_faces)))
    worse, better = ar.compare(cfg, on_audit, {k: ar.essentials(v) for k, v in off_audit.items()})
    print(f"audit, matching on against off: {len(worse)} measure(s) worse, {len(better)} better")
    for w in worse:
        print("   worse:", w)
    for w in better:
        print("   better:", w)
    for k in ("matched", "other"):
        v = tex[k]
        if v:
            dlt = np.array([on - off for _, _, off, on in v])
            print(f"texture check, {k} faces ({len(v)}): agreement {np.mean([x[2] for x in v]):.3f} -> {np.mean([x[3] for x in v]):.3f}; "
                  f"better by 0.02+ on {np.mean(dlt > 0.02) * 100:.0f}%, worse by 0.02+ on {np.mean(dlt < -0.02) * 100:.0f}%")
    if port:
        p = np.array([(x, y) for x, y, _, _ in port])
        agree = np.array([acc == sacc for _, _, acc, sacc in port])
        print(f"pipeline against the study on {len(port)} unknown bands: depth within 0.25 m on {np.mean(np.abs(p[:, 0] - p[:, 1]) <= 0.25) * 100:.0f}%; "
              f"same accept/reject on {agree.mean() * 100:.0f}%")
    out = dict(worse=worse, better=better, buildings=rows,
               texture={k: [dict(name=n, face=f, off=o, on=x) for n, f, o, x in v] for k, v in tex.items()},
               port=[dict(pipeline=x, study=y, pipeline_accepted=pa, study_accepted=sa) for x, y, pa, sa in port])
    a.report_out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {a.report_out}")
    return 0


# ------------------------------------------------------------------ the score

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
        cut = choose_cut([r for r, t in zip(meas, tr) if t], err[tr], a.max_gross)
        te = ~tr
        acc = te & np.array([is_accepted(r, cut) for r in meas])
        accepted |= acc
        e = np.abs(err[acc])
        report["folds"].append(dict(gross_over_1m=round(float(np.mean(e > 1.0)), 3), cut_margin=round(cut[0], 3), cut_pattern=round(cut[1], 1),
                                    test_bands=int(te.sum()), accepted=int(acc.sum()),
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
    # the hard measured bands, most like the unknown ones: the quarter with the
    # fewest LiDAR returns, or on a face that also has unknown bands
    faces_unknown = {(r["building"], r["face"]) for r in unk}
    q1 = np.percentile([r.get("lidar_returns", 0) for r in meas], 25)
    hard = np.array([r.get("lidar_returns", 0) <= q1 or (r["building"], r["face"]) in faces_unknown for r in meas])
    eh = np.abs(err[accepted & hard])
    report["held_out_hard"] = dict(bands=int(hard.sum()), accepted=int((accepted & hard).sum()),
                                   within_03=round(float(np.mean(eh <= 0.3)), 3), within_05=round(float(np.mean(eh <= 0.5)), 3),
                                   within_1=round(float(np.mean(eh <= 1.0)), 3), median_m=round(float(np.median(eh)), 3))
    # the lean: by the way the wall faces, and by building
    hd = np.array([heading[(r["building"], r["face"])] for r in meas])
    quad = ((hd + 45) % 360 // 90).astype(int)
    report["lean_by_facing"] = {n: dict(bands=int((quad == q).sum()), signed_median_m=round(float(np.median(err[quad == q])), 3))
                                for q, n in enumerate(("north", "east", "south", "west")) if (quad == q).any()}
    report["lean_by_building"] = {blds[k].name: round(float(np.median(err[[r["building"] == k for r in meas]])), 3) for k in keys}
    # the cut, chosen on all measured bands, applied to the unknown ones
    cut = choose_cut(meas, err, a.max_gross)
    ua = [r for r in unk if is_accepted(r, cut)]
    report["unknown"] = dict(cut_margin=round(cut[0], 3), cut_pattern=round(cut[1], 1), swept=len(unk), accepted=len(ua),
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
    o = sub.add_parser("outlines", help="a building's matched walls on a frame: where the mesh has them, where matching does")
    for act in p._actions[1:]:
        o._add_action(act)
    o.add_argument("--match", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference" / "match.json")
    o.add_argument("--report", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference" / "match_score.json")
    o.add_argument("--building", required=True, help="the reference set's id")
    o.add_argument("--min-move", type=float, default=1.0, help="only bands matching moves this far (m)")
    o.add_argument("--out-png", type=Path, required=True)
    h = sub.add_parser("sheet", help="accepted unknown bands at random, where the mesh has them and where matching does")
    for act in p._actions[1:]:
        h._add_action(act)
    h.add_argument("--match", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference" / "match.json")
    h.add_argument("--report", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference" / "match_score.json")
    h.add_argument("--n", type=int, default=24)
    h.add_argument("--min-move", type=float, default=0.5)
    h.add_argument("--seed", type=int, default=11)
    h.add_argument("--out-png", type=Path, required=True)
    c = sub.add_parser("compare-runs", help="two rebuilds of the reference set, matching off and on")
    c.add_argument("--off", type=Path, default=ROOT / "outputs" / "wall_match_ab" / "off")
    c.add_argument("--on", type=Path, default=ROOT / "outputs" / "wall_match_ab" / "on")
    c.add_argument("--match", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference" / "match.json")
    c.add_argument("--report-out", type=Path, default=ROOT / "outputs" / "wall_match_ab" / "compare.json")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = parse_args(argv)
    return dict(fetch=cmd_fetch, sweep=cmd_sweep, score=cmd_score, show=cmd_show, outlines=cmd_outlines, sheet=cmd_sheet,
                **{"compare-runs": cmd_compare_runs})[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
