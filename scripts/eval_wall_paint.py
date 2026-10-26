#!/usr/bin/env python3
"""Do walls painted from whole photographs beat walls coloured point by point?
The held-out test.

For every wall section a building's frames painted (``wall_paint.json``,
written by ``build_mesh.py``), and every frame that sees at least 30% of it:
leave that frame out, colour the section without it both ways, and compare
each with what the left-out frame shows.

* **new** -- ``mesh.paint.paint_grid``: the section warped whole from the
  best remaining frame, patched from the next;
* **old** -- what the products did before: a measured wall (``profile``)
  from its coloured LiDAR points, each recoloured from its own most face-on
  remaining frame and rasterised (``face_texture``'s way); a wall drawn up
  to the roof's edge (``prism``) texel by texel, each from its own most
  face-on remaining frame (``quad_texture_from_cameras``'s way).

Two measures, over the texels the left-out frame sees:

* pattern agreement -- correlation of the high-passed brightness (1 m
  shading removed), 1 = the same windows in the same places, 0 = unrelated;
* colour error -- the mean difference per channel, 0-255.

Both are of the whole texture as the product shows it, unseen texels filled
as the product fills them. Writes a CSV of every comparison and a figure,
reported by building size (height, and footprint area).

Usage::

    python scripts/eval_wall_paint.py --dirs outputs/downtown/buildings/<gers> ... \\
        --csv outputs/facade_check/wall_paint_holdout.csv --out docs/figures/wall_paint_holdout.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent / "src"))

from kyfromabove_3d.clouds import find_cloud, read_cloud                                     # noqa: E402
from kyfromabove_3d.colourise.colour import load_cameras, project_visible, sample_colour      # noqa: E402
from kyfromabove_3d.facade_fill import SEEN                                                  # noqa: E402
from kyfromabove_3d.mesh.paint import PAINT_FILE, paint_grid, section_grid                    # noqa: E402
from kyfromabove_3d.mesh.textures import _fill_holes, _solid_px                              # noqa: E402
from kyfromabove_3d.texture_check import highpass, luminance, masked_ncc                     # noqa: E402

#: the texel each way paints at: walls up to the roof's edge were sampled at 0.3 m before
TEXEL = {"profile": 0.15, "prism": 0.15}
TEXEL_OLD = {"profile": 0.15, "prism": 0.3}
#: (label, upper bound): building height in metres, footprint in square metres
HEIGHTS = (("under 10 m", 10), ("10-30 m", 30), ("30-80 m", 80), ("over 80 m", np.inf))
FOOTPRINTS = (("under 300 m2", 300), ("300-1,500 m2", 1500), ("1,500-5,000 m2", 5000), ("over 5,000 m2", np.inf))


def cams_with_depth(b: Path):
    """The building's frames and the depth maps its build used (``occlusion.npz``)."""
    cams = load_cameras(b)
    z = np.load(b / "occlusion.npz", allow_pickle=False)
    for i, c in enumerate(cams):
        c["depth"], c["downscale"] = z[f"d{i}"], 8
    return cams


def size_of(b: Path):
    """(height, footprint area): from the levels, or the footprint where the
    building was drawn as a box and has none."""
    from shapely.geometry import Polygon
    for name in ("levels.json", "levels_measured.json"):
        if (b / name).exists():
            levels = json.loads((b / name).read_text(encoding="utf-8"))
            return max(float(lv["z_hi"]) for lv in levels), float(Polygon(np.asarray(levels[0]["ring"])[:, :2]).area)
    from kyfromabove_3d.footprints import footprint_wgs84_to_local
    fp = json.loads((b / "footprint.json").read_text(encoding="utf-8"))
    ox, oy, _ = json.loads((b / "coordinates.json").read_text(encoding="utf-8"))["origin_source_xyz"]
    return float(fp.get("height_m") or fp.get("lidar_height_m") or 0.0), float(footprint_wgs84_to_local(fp["wkt"], ox, oy).area)


def old_profile(rec, sec, cxyz, cams, texel_m):
    """The section from its coloured points, recoloured from ``cams``."""
    d = cxyz[:, :2] - np.asarray(rec["p0"])
    a = d @ np.asarray(rec["e1"]); off = d @ np.asarray(rec["normal"])
    z = cxyz[:, 2]
    keep = ((np.abs(off - sec["off"]) < 0.6) & (a >= sec["a_lo"]) & (a <= sec["a_hi"])
            & (z >= sec["z_lo"]) & (z <= sec["z_hi"]))
    pts = cxyz[keep]
    xyz, n3 = section_grid(SimpleNamespace(p0=np.asarray(rec["p0"]), p1=np.asarray(rec["p0"]) + np.asarray(rec["e1"]),
                                           length_m=1.0, normal=np.asarray(rec["normal"])), sec, texel_m)
    H, W = xyz.shape[:2]
    img = np.zeros((H, W, 3)); cnt = np.zeros((H, W))
    if len(pts):
        rgb, seen, _, _ = sample_colour(pts, np.broadcast_to(n3, pts.shape), cams)
        ok = seen > 0
        u = np.clip(((a[keep][ok] - sec["a_lo"]) / (sec["a_hi"] - sec["a_lo"]) * W).astype(int), 0, W - 1)
        v = np.clip(((pts[ok, 2] - sec["z_lo"]) / (sec["z_hi"] - sec["z_lo"]) * H).astype(int), 0, H - 1)
        np.add.at(img, (v, u), rgb[ok].astype(np.float64))
        np.add.at(cnt, (v, u), 1.0)
    if not cnt.any():
        return None
    return _fill_holes(img, cnt, solid_px=_solid_px(texel_m))


def old_prism(xyz, n3, cams, texel_m):
    """The section texel by texel, each from its most face-on frame in ``cams``."""
    H, W = xyz.shape[:2]
    rgb, seen, _, _ = sample_colour(xyz.reshape(-1, 3), np.broadcast_to(n3, (H * W, 3)), cams)
    cnt = (seen > 0).astype(np.float64)
    if not cnt.any():
        return None
    img = rgb.astype(np.float64) * cnt[:, None]
    return _fill_holes(img.reshape(H, W, 3), cnt.reshape(H, W), max_fill_px=max(2, int(1.5 / texel_m)),
                       solid_px=_solid_px(texel_m))


def _resample(tex, H, W):
    """``tex`` (a Texture) on an ``H`` x ``W`` grid, nearest texel."""
    from kyfromabove_3d.mesh.textures import Texture
    h, w = tex.shape[:2]
    r = np.minimum((np.arange(H) * h / H).astype(int), h - 1); c = np.minimum((np.arange(W) * w / W).astype(int), w - 1)
    return Texture(np.asarray(tex)[r][:, c], tex.source[r][:, c])


def score(tex, ref, mask, sigma_px):
    """(pattern agreement, colour error, share seen) of ``tex`` (rows top-down) against ``ref`` over ``mask``."""
    if tex is None:
        return np.nan, np.nan, 0.0
    t = np.asarray(tex)[::-1].astype(np.float64)
    src = tex.source[::-1]
    a = highpass(luminance(t), mask, sigma_px)
    b = highpass(luminance(ref), mask, sigma_px)
    return (float(masked_ncc(a, b, mask)), float(np.abs(t[mask] - ref[mask]).mean()),
            float((src[mask] == SEEN).mean()))


def one_building(b: Path, max_frames: int = 3, min_area_m2: float = 6.0, kinds=(), prism_texel=None):
    try:
        return _one_building(b, max_frames, min_area_m2, kinds, prism_texel)
    except Exception as exc:                                    # noqa: BLE001
        print(f"[skip] {b.name}: {exc!r}", flush=True)
        return []


def _one_building(b: Path, max_frames: int, min_area_m2: float, kinds=(), prism_texel=None):
    if not (b / PAINT_FILE).exists() or not (b / "occlusion.npz").exists():
        print(f"[skip] {b.name}: no {PAINT_FILE} or occlusion.npz")
        return []
    records = json.loads((b / PAINT_FILE).read_text(encoding="utf-8"))
    cams = cams_with_depth(b)
    if not cams:
        print(f"[skip] {b.name}: no frame crops")
        return []
    # a building the colourise declined has no coloured points: its walls are
    # all up to the roof's edge, which the old way coloured from the frames
    cloud = find_cloud(b, "colorized_aerial")
    col = read_cloud(cloud) if cloud.exists() else {"x": [], "y": [], "z": []}
    cxyz = np.column_stack([col["x"], col["y"], col["z"]]).astype(np.float64).reshape(-1, 3)
    height, area = size_of(b)
    rows = []
    for k, rec in enumerate(records):
        sec = {"off": rec["off"], "a_lo": rec["along"][0], "a_hi": rec["along"][1], "z_lo": rec["z"][0], "z_hi": rec["z"][1]}
        sec_m2 = (sec["a_hi"] - sec["a_lo"]) * (sec["z_hi"] - sec["z_lo"])
        if sec_m2 < min_area_m2:
            continue
        if kinds and rec.get("kind", "profile") not in kinds:
            continue
        texel_m = prism_texel if (prism_texel and rec.get("kind") == "prism") else TEXEL[rec.get("kind", "profile")]
        plane = SimpleNamespace(p0=np.asarray(rec["p0"]), p1=np.asarray(rec["p0"]) + np.asarray(rec["e1"]),
                                length_m=1.0, normal=np.asarray(rec["normal"]))
        xyz, n3 = section_grid(plane, sec, texel_m)
        H, W = xyz.shape[:2]
        pts = xyz.reshape(-1, 3); nrm = np.broadcast_to(n3, pts.shape)
        seeing = []
        for c in cams:
            idx, u, v, f = project_visible(pts, nrm, c, 0.15, 1.5)
            if len(idx) >= 0.30 * len(pts):
                seeing.append((len(idx), c, idx, u, v))
        seeing.sort(key=lambda s: -s[0])
        for _, held, idx, u, v in seeing[:max_frames]:
            rest = [c for c in cams if c is not held]
            mask = np.zeros(len(pts), dtype=bool); mask[idx] = True
            ref = np.zeros((len(pts), 3)); ref[idx] = held["img"][v, u]
            mask = mask.reshape(H, W); ref = ref.reshape(H, W, 3)
            new, _ = paint_grid(xyz, n3, rest, texel_m)
            if rec.get("kind", "profile") == "profile":
                old = old_profile(rec, sec, cxyz, rest, texel_m)
            else:
                # at its own texel, then shown on this grid as a viewer sees it
                xyz_o, _ = section_grid(plane, sec, TEXEL_OLD["prism"])
                old = old_prism(xyz_o, n3, rest, TEXEL_OLD["prism"])
                old = None if old is None else _resample(old, H, W)
            sigma = 1.0 / texel_m
            n_ncc, n_err, n_seen = score(new, ref, mask, sigma)
            o_ncc, o_err, o_seen = score(old, ref, mask, sigma)
            rows.append({"building": b.name, "section": k, "face": rec["face"], "kind": rec.get("kind", "profile"),
                         "area_m2": round(sec_m2, 1), "height_m": round(height, 1), "footprint_m2": round(area, 1),
                         "held_out": held["id"], "texels": int(mask.sum()),
                         "ncc_old": round(o_ncc, 4), "ncc_new": round(n_ncc, 4),
                         "err_old": round(o_err, 2), "err_new": round(n_err, 2),
                         "seen_old": round(o_seen, 3), "seen_new": round(n_seen, 3)})
    print(f"{b.name[:8]}: {len(rows)} comparisons over {len(records)} section(s)", flush=True)
    return rows


def class_of(value, classes):
    return next(label for label, hi in classes if value < hi)


def figure(rows, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in rows if np.isfinite(r["ncc_old"]) and np.isfinite(r["ncc_new"])]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.8))
    ax = axs[0]
    for kind, colour in (("profile", "#2b6cb0"), ("prism", "#dd6b20")):
        k = [r for r in ok if r["kind"] == kind]
        ax.scatter([r["ncc_old"] for r in k], [r["ncc_new"] for r in k], s=8, alpha=0.5, c=colour,
                   label={"profile": "measured walls", "prism": "walls up to the roof's edge"}[kind])
    ax.plot([-0.3, 1], [-0.3, 1], "k--", lw=0.8)
    ax.set_xlim(-0.3, 1); ax.set_ylim(-0.3, 1)
    ax.set_xlabel("coloured point by point (before)"); ax.set_ylabel("painted from whole photos (now)")
    ax.set_title("Pattern agreement with a photo left out of the painting\n(1 = same windows in the same places; "
                 "above the line = now better)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    for ax, (name, key, classes) in zip(axs[1:], (("building height", "height_m", HEIGHTS),
                                                   ("footprint area", "footprint_m2", FOOTPRINTS))):
        labels = [lab for lab, _ in classes]
        med_o, med_n, better, n = [], [], [], []
        for lab in labels:
            g = [r for r in ok if class_of(r[key], classes) == lab]
            n.append(len({r["building"] for r in g}))
            med_o.append(np.median([r["ncc_old"] for r in g]) if g else np.nan)
            med_n.append(np.median([r["ncc_new"] for r in g]) if g else np.nan)
            better.append(np.mean([r["ncc_new"] > r["ncc_old"] for r in g]) if g else np.nan)
        x = np.arange(len(labels))
        ax.bar(x - 0.2, med_o, 0.4, color="#a0aec0", label="before (point by point)")
        ax.bar(x + 0.2, med_n, 0.4, color="#2b6cb0", label="now (whole photos)")
        for xi, bt, nb in zip(x, better, n):
            if np.isfinite(bt):
                ax.text(xi, 0.02, f"now better\nin {bt:.0%}\n({nb} bldgs)", ha="center", fontsize=8, color="white")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("median pattern agreement"); ax.set_title(f"By {name}", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=110)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=3, help="frames left out per section, the ones seeing most of it")
    ap.add_argument("--kinds", nargs="+", default=[], choices=["profile", "prism"], help="only these walls")
    ap.add_argument("--prism-texel", type=float, default=None, help="paint walls up to the roof's edge at this texel")
    ap.add_argument("--csv", type=Path, default=Path("outputs/facade_check/wall_paint_holdout.csv"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures/wall_paint_holdout.png"))
    args = ap.parse_args()

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(args.workers) as ex:
        n = len(args.dirs)
        rows = [r for rs in ex.map(one_building, args.dirs, [args.max_frames] * n, [6.0] * n, [tuple(args.kinds)] * n,
                                         [args.prism_texel] * n)
                for r in rs]
    if not rows:
        print("nothing compared")
        return 1
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    ok = [r for r in rows if np.isfinite(r["ncc_old"]) and np.isfinite(r["ncc_new"])]
    for kind in ("profile", "prism"):
        k = [r for r in ok if r["kind"] == kind]
        if k:
            print(f"{kind}: {len(k)} comparisons on {len({r['building'] for r in k})} buildings; pattern agreement "
                  f"{np.median([r['ncc_old'] for r in k]):.3f} -> {np.median([r['ncc_new'] for r in k]):.3f}, "
                  f"now better in {np.mean([r['ncc_new'] > r['ncc_old'] for r in k]):.0%}; colour error "
                  f"{np.median([r['err_old'] for r in k]):.1f} -> {np.median([r['err_new'] for r in k]):.1f}; seen "
                  f"{np.median([r['seen_old'] for r in k]):.0%} -> {np.median([r['seen_new'] for r in k]):.0%}")
    figure(rows, args.out)
    print(f"-> {args.csv}, {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
