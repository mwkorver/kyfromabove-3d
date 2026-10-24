#!/usr/bin/env python3
"""Where the frames put each roof edge, against where the mesh has it.

The mesh's roof edge along a straight face is the top of its wall line: the
wall at the top band's offset (the footprint edge where the LiDAR saw no
wall) and the mesh's roof height just inside. The frames show the edge at
about 7.5 cm a pixel.

For each face the edge is searched as a level line along the face, at a
trial offset (in or out of the mesh's wall line) and a trial height: each
frame scores how strong a brightness step it shows across the line where the
line projects, each camera head with an equal say, and the edge is the
outermost strong line (a parapet's shadow and the roof's seams lie inside
it). It counts as observed when at least two heads see it, it is not at the
search's limit, and leaving out any one head moves it by no more than
``--agree-m``.

Status (first run, reference set): not trustworthy yet -- on most faces the
search catches shadows, window bands or roof seams. What it did show is that
the mesh's edge is the problem, not the cameras (see camera_check.py): where
a wall sits on the footprint edge the LiDAR's own roof edge is a median 0.8 m
further in. A search started from the LiDAR's edge, within half a metre, is
the next attempt.

Steps:

  measure   every straight face of the reference buildings -> roof_edges.json
  plot      the gaps between the mesh's edge and the observed one
  sheet     the mesh's edge and the observed one drawn on the frames

Usage::

    python scripts/roof_edges.py measure
    python scripts/roof_edges.py plot
    python scripts/roof_edges.py sheet --faces 40ead71b_1006 ...
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
import face_match as fm  # noqa: E402
from kyfromabove_3d.stages import wall_visibility as wv  # noqa: E402

from kyfromabove_3d.wall_match import bilinear_gray  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "outputs" / "wall_visibility" / "reference"
STEP_ALONG_M = 0.25
WALL_STATUSES = ("observed", "bridged", "extended", "mirrored", "matched")


def head_of(frame: str) -> str:
    return frame.split("/")[-1].split("_")[0]


def mesh_edge(plane, rows, field, b, lo, hi):
    """The mesh's roof edge on this face: the top wall band's offset (0, the
    footprint line, where no band has one) and the mesh's height just inside it
    (the highest mesh vertex 0.5-2 m in, median along the face)."""
    off = 0.0
    for r in sorted(rows, key=lambda r: r["z_hi"]):
        if r.get("status") in WALL_STATUSES and r.get("offset_m") is not None:
            off = float(r["offset_m"])
    a = np.arange(lo, hi, 1.0)
    q = np.column_stack([a, np.full_like(a, off), np.zeros_like(a)])
    tops = []
    for k in (0.5, 1.0, 1.5, 2.0):
        q[:, 1] = off - k
        P = fm._to_world(b, plane, q)
        tops.append(fm.field_at(field, P[:, :2]))
    top = np.max(tops, axis=0)
    top = top[np.isfinite(top)]
    if not len(top):
        return off, None
    # the field is in the tileset frame; the building's local height is that less the frame's offset
    z_local = float(np.median(top)) - float(b.M[2, 3])
    return off, z_local


GRAD_O = np.arange(-3.0, 3.001, 0.1)       # in (-) or out (+) of the mesh's wall line
GRAD_H = np.arange(-2.0, 1.501, 0.1)       # down (-) or up (+) from the mesh's roof height
FINE = np.arange(-0.1, 0.101, 0.025)
OUTER_SHARE = 0.6


def frame_gradients(img: np.ndarray):
    """The frame's brightness gradient (x, y), lightly smoothed, and a typical
    strong gradient (90th percentile) to put frames on one scale."""
    from scipy.ndimage import gaussian_filter
    g = gaussian_filter(img, 1.0)
    gy, gx = np.gradient(g)
    return gx.astype(np.float32), gy.astype(np.float32), float(np.percentile(np.hypot(gx, gy), 90)) or 1.0


def edge_strength(b, plane, a, o, h, cam, grads, x0, y0):
    """How strong a brightness step a frame shows across the trial edge (offset
    ``o``, height ``h``) where it projects, along the line: the gradient's
    component across the projected line, over the frame's typical strong
    gradient. NaN where most of the line falls outside the frame."""
    gx, gy, scale = grads
    X = fm._to_world(b, plane, np.column_stack([a, np.full_like(a, o), np.full_like(a, h)]))
    pc = (X - cam["C"]) @ cam["R_cw"].T
    u = cam["K"][0, 0] * pc[:, 0] / pc[:, 2] + cam["K"][0, 2] - x0
    v = cam["K"][1, 1] * pc[:, 1] / pc[:, 2] + cam["K"][1, 2] - y0
    d = np.gradient(np.column_stack([u, v]), axis=0)
    n = np.column_stack([-d[:, 1], d[:, 0]]) / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    ex, ok = bilinear_gray(gx, u, v)
    ey, _ = bilinear_gray(gy, u, v)
    if ok.mean() < 0.6:
        return np.nan
    return float(np.mean(np.abs(ex * n[:, 0] + ey * n[:, 1])[ok]) / scale)


def gradient_search(b, plane, a, o_ref, h_ref, cams, grads, keep_heads=None):
    """Each trial edge's strength, averaged within each camera head and then
    over heads (each head an equal say): (grid of strengths over GRAD_O x GRAD_H
    per head)."""
    per_head = {}
    for f, (gr, x0, y0) in grads.items():
        hd = head_of(f)
        S = np.array([[edge_strength(b, plane, a, o_ref + do, h_ref + dh, cams[f], gr, x0, y0) for dh in GRAD_H] for do in GRAD_O])
        per_head.setdefault(hd, []).append(S)
    return {hd: np.nanmean(np.stack(v), axis=0) if np.isfinite(np.stack(v)).any() else np.stack(v)[0] for hd, v in per_head.items()}


def measure_building(job) -> List[dict]:
    import cv2

    from kyfromabove_3d.features import ArcWall, load_wall_planes
    (b, frames, cache, index, args) = job
    if "f" not in fm._FIELD:
        fm._FIELD["f"] = fm.height_field(args.scene)
    field = fm._FIELD["f"]
    cams = {c["id"]: c for c in wv.cameras_in_tileset_frame(frames, b)}
    imgs = {}
    for k, w in index.items():
        if w["building"] == b.key and w["frame"] in cams:
            imgs[w["frame"]] = (cv2.imread(str(cache / k), cv2.IMREAD_GRAYSCALE).astype(np.float32), w["x0"], w["y0"])
    profiles = json.loads((b.dir / "wall_profiles.json").read_text(encoding="utf-8"))
    out = []
    for plane in load_wall_planes(b.dir / "wall_planes.json"):
        if isinstance(plane, ArcWall):
            continue
        L = float(plane.length_m)
        lo, hi = 1.5, L - 1.5
        if hi - lo < 4.0:
            continue
        rec = dict(building=b.key, name=b.name, face=int(plane.face), length_m=round(L, 1))
        o_ref, h_ref = mesh_edge(plane, profiles.get(str(plane.face), []), field, b, lo, hi)
        rec.update(mesh_offset=round(o_ref, 2), mesh_height=None if h_ref is None else round(h_ref, 2))
        if h_ref is None:
            out.append(dict(rec, status="no roof found"))
            continue
        a = np.arange(lo + STEP_ALONG_M / 2, hi, STEP_ALONG_M)
        # frames whose window holds most of the mesh's edge, and in front of the camera
        X = fm._to_world(b, plane, np.column_stack([a, np.full_like(a, o_ref), np.full_like(a, h_ref)]))
        views = {}
        for f, (img, x0, y0) in imgs.items():
            c = cams[f]
            pc = (X - c["C"]) @ c["R_cw"].T
            u = c["K"][0, 0] * pc[:, 0] / pc[:, 2] + c["K"][0, 2] - x0
            v = c["K"][1, 1] * pc[:, 1] / pc[:, 2] + c["K"][1, 2] - y0
            inside = (pc[:, 2] > 0) & (u >= 0) & (v >= 0) & (u < img.shape[1] - 1) & (v < img.shape[0] - 1)
            if inside.mean() >= 0.6:
                views[f] = (img, x0, y0)
        heads = sorted({head_of(f) for f in views})
        rec.update(frames=len(views), heads=heads)
        if len(views) < 2 or len(heads) < 2:
            out.append(dict(rec, status="fewer than 2 camera heads see it"))
            continue
        grads = {f: (frame_gradients(img), x0, y0) for f, (img, x0, y0) in views.items()}
        per_head = gradient_search(b, plane, a, o_ref, h_ref, cams, grads)

        def peak(hds):
            """The outermost strong line: among the score's local peaks at least
            OUTER_SHARE of the strongest, the one furthest out (a parapet's
            shadow and the roof's seams lie inside the real edge)."""
            from scipy.ndimage import maximum_filter
            S = np.nanmean(np.stack([per_head[h] for h in hds]), axis=0)
            if not np.isfinite(S).any():
                return None, np.nan
            Sz = np.nan_to_num(S, nan=-9)
            top = Sz.max()
            cand = np.argwhere((Sz == maximum_filter(Sz, size=5)) & (Sz >= OUTER_SHARE * top))
            i, j = max(cand, key=lambda ij: (GRAD_O[ij[0]], Sz[ij[0], ij[1]]))
            return (float(GRAD_O[i]), float(GRAD_H[j])), float(S[i, j])
        (do, dh), score = peak(heads)
        coarse = (do, dh)
        # leave each head out in turn: the edge must not move by more than agree_m
        moves = {}
        for hd in heads:
            rest = [h for h in heads if h != hd]
            (o2, h2), _ = peak(rest)
            moves[hd] = round(float(np.hypot(o2 - do, h2 - dh)), 3)
        spread = max(moves.values()) if moves else None
        at_limit = abs(do) >= GRAD_O.max() - 0.05 or dh <= GRAD_H.min() + 0.05 or dh >= GRAD_H.max() - 0.05
        # refine about the coarse peak, every head an equal say
        def fine_score(fo, fh):
            vals = {}
            for f, (gr, x0, y0) in grads.items():
                vals.setdefault(head_of(f), []).append(edge_strength(b, plane, a, o_ref + fo, h_ref + fh, cams[f], gr, x0, y0))
            v = [np.nanmean(x) for x in vals.values() if np.isfinite(x).any()]
            return float(np.mean(v)) if v else -9.0
        do, dh = max(((coarse[0] + x, coarse[1] + y) for x in FINE for y in FINE), key=lambda t: fine_score(*t))
        observed = bool(np.isfinite(score) and score >= args.min_score and spread is not None and spread <= args.agree_m and not at_limit)
        rec.update(edge_offset=round(o_ref + do, 3), edge_height=round(h_ref + dh, 3), gap_out_m=round(float(do), 3),
                   gap_up_m=round(float(dh), 3), score=None if not np.isfinite(score) else round(score, 3),
                   leave_one_head_out_m=moves, status="observed" if observed else
                   ("unclear" if not np.isfinite(score) or score < args.min_score else
                    "at the search's limit" if at_limit else "heads disagree"))
        out.append(rec)
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
            print(f"  {k + 1}/{len(jobs)} buildings, {len(out)} faces ({time.time() - t0:.0f} s)", flush=True)
    out.sort(key=lambda r: (r["name"], r["face"]))
    a.out.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    from collections import Counter
    print(Counter(r["status"] for r in out))
    obs = [r for r in out if r["status"] == "observed"]
    if obs:
        g = np.array([abs(r["gap_out_m"]) for r in obs]); h = np.array([abs(r["gap_up_m"]) for r in obs])
        print(f"{len(obs)} observed: the mesh's edge is off in plan by a median {np.median(g):.2f} m "
              f"(over 0.3 m on {np.mean(g > 0.3):.0%}), in height by a median {np.median(h):.2f} m")
    print(f"-> {a.out}")
    return 0


def cmd_sheet(a) -> int:
    """For each face named: the mesh's edge (red) and the observed edge (teal)
    drawn on one frame from each camera head that sees it, over the middle
    10 m of the face, at the frame's own pixels."""
    import cv2

    from kyfromabove_3d.features import load_wall_planes
    rec = {f'{r["building"][:8]}_{r["face"]}': r for r in json.loads(a.edges.read_text(encoding="utf-8"))}
    blds = {b.key[:8]: b for b in wv.reference_buildings(a.reference, a.landmarks, wv.tileset_transforms(a.tileset), a.buildings)}
    frames = wv.read_frames(a.frames)
    index = json.loads((a.cache / "index.json").read_text(encoding="utf-8"))
    RED, TEAL, H = (30, 38, 179), (120, 124, 15), 280
    rows = []
    for k in [k for k in a.faces if k in rec and rec[k].get("edge_offset") is not None]:
        r = rec[k]; b = blds[k.split("_")[0]]
        plane = {p.face: p for p in load_wall_planes(b.dir / "wall_planes.json")}[r["face"]]
        cams = {c["id"]: c for c in wv.cameras_in_tileset_frame(frames, b)}
        L = float(plane.length_m); mid = L / 2
        aa = np.linspace(max(1.5, mid - 5), min(L - 1.5, mid + 5), 20)
        lines = [(r["mesh_offset"], r["mesh_height"], RED), (r["edge_offset"], r["edge_height"], TEAL)]
        tiles, seen = [], set()
        for kk, w in sorted(index.items()):
            hd = head_of(w["frame"])
            if w["building"] != b.key or w["frame"] not in cams or hd in seen:
                continue
            c = cams[w["frame"]]
            uvs = []
            for o, h, col in lines:
                X = fm._to_world(b, plane, np.column_stack([aa, np.full_like(aa, o), np.full_like(aa, h)]))
                pc = (X - c["C"]) @ c["R_cw"].T
                uvs.append((np.column_stack([c["K"][0, 0] * pc[:, 0] / pc[:, 2] + c["K"][0, 2] - w["x0"],
                                             c["K"][1, 1] * pc[:, 1] / pc[:, 2] + c["K"][1, 2] - w["y0"]]), col))
            allp = np.vstack([u for u, _ in uvs])
            if (allp < 0).any() or (allp[:, 0] > w["w"]).any() or (allp[:, 1] > w["h"]).any():
                continue
            img = cv2.imread(str(a.cache / kk))
            x0, y0 = np.maximum(allp.min(0) - 90, 0).astype(int)
            x1, y1 = np.minimum(allp.max(0) + 90, [img.shape[1], img.shape[0]]).astype(int)
            crop = img[y0:y1, x0:x1].copy()
            sc = H / crop.shape[0]
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * sc)), H), interpolation=cv2.INTER_CUBIC)
            for u, col in uvs:
                cv2.polylines(crop, [((u - [x0, y0]) * sc).round().astype(np.int32)], False, col, 2, cv2.LINE_AA)
            lab = np.full((22, crop.shape[1], 3), 255, np.uint8)
            cv2.putText(lab, f"{hd} {w['frame'].split('/')[-1]}", (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
            tiles.append(np.vstack([crop, lab]))
            seen.add(hd)
        if not tiles:
            continue
        row = np.hstack([np.hstack([t, np.full((t.shape[0], 10, 3), 255, np.uint8)]) for t in tiles[:5]])
        cap = np.full((28, row.shape[1], 3), 255, np.uint8)
        cv2.putText(cap, f'{r["name"][:30]}, face {r["face"]} ({r["status"]}): frames put the edge {r["gap_out_m"]:+.2f} m out, '
                    f'{r["gap_up_m"]:+.2f} m up from the mesh (red = mesh, teal = frames)', (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        rows.append(np.vstack([cap, row, np.full((12, row.shape[1], 3), 255, np.uint8)]))
    Wd = max(x.shape[1] for x in rows)
    sheet = np.vstack([cv2.copyMakeBorder(x, 0, 0, 0, Wd - x.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x in rows])
    cv2.imwrite(str(a.sheet_out), sheet)
    print(f"{len(rows)} faces -> {a.sheet_out} ({sheet.shape[1]} x {sheet.shape[0]})")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("measure", help="the observed roof edge of every straight face")
    p.add_argument("--reference", type=Path, default=ROOT / "configs" / "audit_reference.yaml")
    p.add_argument("--landmarks", type=Path, default=ROOT / "configs" / "landmarks.yaml")
    p.add_argument("--tileset", type=Path, default=ROOT / "outputs" / "downtown")
    p.add_argument("--buildings", type=Path, default=ROOT / "outputs" / "downtown" / "buildings")
    p.add_argument("--frames", type=Path, default=ROOT / "outputs" / "downtown" / "frames_2024s1.gpkg")
    p.add_argument("--cache", type=Path, default=ROOT / "data" / "derived" / "frames_cache")
    p.add_argument("--scene", type=Path, default=ROOT / "data" / "derived" / "video" / "downtown_scene.npz")
    p.add_argument("--min-score", type=float, default=0.0, help="the brightness step across the edge, over each frame's typical strong step, for an observed edge")
    p.add_argument("--agree-m", type=float, default=0.15, help="the furthest leaving out one camera head may move the edge")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", type=Path, default=REF / "roof_edges.json")
    s = sub.add_parser("sheet", help="the mesh's edge and the observed one on the frames, for chosen faces")
    for act in p._actions[1:]:
        if act.dest in ("reference", "landmarks", "tileset", "buildings", "frames", "cache"):
            s._add_action(act)
    s.add_argument("--edges", type=Path, default=REF / "roof_edges.json")
    s.add_argument("--faces", nargs="+", required=True, help="<building key's first 8>_<face>")
    s.add_argument("--sheet-out", type=Path, default=REF / "roof_edges_sheet.png")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = parse_args(argv)
    return dict(measure=cmd_measure, sheet=cmd_sheet)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
