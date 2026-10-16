#!/usr/bin/env python3
"""Render a coloured point cloud from viewpoints the capture never occupied.

A z-buffered point rasteriser. Each point covers a fixed number
of pixels at a metre of range and shrinks with distance, so the wall resampling
reads as a surface at the scale it was sampled and the LiDAR returns stay visibly
sparse beside it. That contrast is the thing worth looking at.

``--source`` renders only one kind of point, which is how to see what the
resampling actually added:

    measured   the LiDAR returns, coloured where they are
    resampled  the new points on the fitted wall planes
    both       everything (default)

Usage::

    python scripts/render_colorized.py --ply outputs/downtown/buildings/df2fccca-d6bd-4d4d-b4bc-673cfb5245aa/colorized.ply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from kyfromabove_3d.clouds import read_cloud


def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World-to-camera rotation in the OpenCV convention (x right, y down, z fwd)."""
    eye = np.asarray(eye, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, dtype=np.float64))
    if np.linalg.norm(x) < 1e-9:                    # looking straight down
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def render(xyz, rgb, eye, target, width=900, height=900, fov_deg=42.0,
           point_px_at_1m=1.6, background=(0.09, 0.10, 0.12), max_px=14):
    """Painter's-algorithm points with a z-buffer. Returns HxWx3 float RGB."""
    eye = np.asarray(eye, dtype=np.float64)
    R_cw = look_at(eye, target)
    cam = (xyz - eye) @ R_cw.T
    front = cam[:, 2] > 0.2
    if not front.any():
        return np.tile(np.asarray(background), (height, width, 1))
    cam, col = cam[front], rgb[front]

    f = 0.5 * height / np.tan(np.radians(fov_deg) / 2.0)
    u = f * cam[:, 0] / cam[:, 2] + width / 2.0
    v = f * cam[:, 1] / cam[:, 2] + height / 2.0
    rad = np.clip((point_px_at_1m * f / 1000.0) / cam[:, 2], 0.5, max_px)

    img = np.tile(np.asarray(background, dtype=np.float64), (height, width, 1))
    depth = np.full((height, width), np.inf)
    # Far to near, so nearer points overwrite -- with the z-buffer as the guard
    # for the points that straddle a pixel.
    order = np.argsort(-cam[:, 2])
    for i in order:
        r = int(rad[i])
        x0, x1 = int(u[i]) - r, int(u[i]) + r + 1
        y0, y1 = int(v[i]) - r, int(v[i]) + r + 1
        if x1 <= 0 or y1 <= 0 or x0 >= width or y0 >= height:
            continue
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, width), min(y1, height)
        z = cam[i, 2]
        patch = depth[y0:y1, x0:x1]
        m = z < patch
        if not m.any():
            continue
        patch[m] = z
        img[y0:y1, x0:x1][m] = col[i] / 255.0
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--elevation-deg", type=float, default=18.0,
                    help="how far above the horizon to orbit. Low, because the "
                         "point of the exercise is the walls, and the capture "
                         "itself never looked at them from below 30 degrees")
    ap.add_argument("--source", choices=["both", "measured", "resampled"],
                    default="both")
    ap.add_argument("--size", type=int, default=900)
    ap.add_argument("--range-mult", type=float, default=1.05,
                    help="how far back to stand; 1.0 just fits the cloud")
    ap.add_argument("--azimuth-deg", type=float, default=35.0,
                    help="first viewpoint; --views orbits from here")
    ap.add_argument("--compare", action="store_true",
                    help="one viewpoint, three ways: the LiDAR alone, the "
                         "resampled walls alone, and both. The comparison is "
                         "the point -- a coloured point cloud looks fine until "
                         "it is beside the surface it was meant to describe")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ply = read_cloud(args.ply)
    xyz = np.column_stack([ply["x"], ply["y"], ply["z"]]).astype(np.float64)
    rgb = np.column_stack([ply["red"], ply["green"], ply["blue"]]).astype(np.float64)
    src = ply.get("source")
    if src is not None and args.source != "both":
        keep = (src == 0) if args.source == "measured" else (src == 1)
        xyz, rgb = xyz[keep], rgb[keep]
    print(f"{len(xyz):,} points ({args.source})")
    if src is not None and args.source == "both":
        print(f"   {int((src == 0).sum()):,} measured, {int((src == 1).sum()):,} resampled")

    target = np.array([np.median(xyz[:, 0]), np.median(xyz[:, 1]),
                       (xyz[:, 2].min() + np.percentile(xyz[:, 2], 99)) / 2.0])
    reach = float(np.percentile(np.linalg.norm(xyz - target, axis=1), 98))
    dist = reach * args.range_mult / np.tan(np.radians(21.0))
    el = np.radians(args.elevation_deg)

    label = {"both": "LiDAR returns and resampled wall points",
             "measured": "LiDAR returns only, coloured from the obliques",
             "resampled": "only the points added on the fitted walls"}

    surf = ply.get("surface")
    if args.compare:
        if src is None and surf is None:
            print("[FAIL] this PLY has no 'source' or 'surface' property to compare on")
            return 1
        az = np.radians(args.azimuth_deg)
        eye = target + dist * np.array([np.cos(el) * np.cos(az),
                                        np.cos(el) * np.sin(az), np.sin(el)])

        if surf is not None and (surf == 3).any():
            panels = [
                ("LiDAR returns only", surf == 0),
                ("Fitted tiles only", surf == 1),
                ("Extruded walls only", surf == 3),
                ("Complete model", np.ones(len(xyz), dtype=bool)),
            ]
            fig, axes = plt.subplots(1, 4, figsize=(19.0, 5.6), squeeze=False)
            for ax, (name, m) in zip(axes[0], panels):
                ax.imshow(np.clip(render(xyz[m], rgb[m], eye, target,
                                         width=args.size, height=args.size), 0, 1))
                ax.set_title(f"{name}\n{int(m.sum()):,} points",
                             fontsize=10.5, color="#ddd")
                ax.axis("off")
            fig.suptitle(f"{args.ply.parent.name} — LiDAR, tiles, extruded walls, and complete model",
                         fontsize=13, y=0.99, color="#eee")
        else:
            panels = [("measured", src == 0), ("resampled", src != 0),
                      ("both", np.ones(len(src), dtype=bool))]
            fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.6), squeeze=False)
            for ax, (name, m) in zip(axes[0], panels):
                ax.imshow(np.clip(render(xyz[m], rgb[m], eye, target,
                                         width=args.size, height=args.size), 0, 1))
                ax.set_title(f"{label[name]}\n{int(m.sum()):,} points",
                             fontsize=10.5, color="#ddd")
                ax.axis("off")
            fig.suptitle(f"{args.ply.parent.name} — the same viewpoint, three ways",
                         fontsize=13, y=0.99, color="#eee")

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = args.out or args.ply.with_name("compare.png")
        fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="#16181c")
        print(f"figure: {out}")
        return 0

    fig, axes = plt.subplots(1, args.views,
                             figsize=(4.6 * args.views, 5.0), squeeze=False)
    for i, ax in enumerate(axes[0]):
        az = np.radians(args.azimuth_deg + 360.0 * i / args.views)
        eye = target + dist * np.array([np.cos(el) * np.cos(az),
                                        np.cos(el) * np.sin(az), np.sin(el)])
        ax.imshow(np.clip(render(xyz, rgb, eye, target,
                                 width=args.size, height=args.size), 0, 1))
        ax.set_title(f"{int(np.degrees(az)) % 360}°", fontsize=10, color="#ddd")
        ax.axis("off")
    fig.suptitle(f"{args.ply.parent.name} — {label[args.source]}",
                 fontsize=12.5, y=0.99, color="#eee")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = args.out or args.ply.with_name(f"orbit_{args.source}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="#16181c")
    print(f"figure: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
