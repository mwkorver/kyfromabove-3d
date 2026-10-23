#!/usr/bin/env python3
"""A VC Map oblique collection (``image.json``) from KyFromAbove frames.

VC Map's oblique view shows the original photos -- forward, backward, left,
right and straight down -- and moves between them and its 3D map. It needs,
for every photo, a 3x4 matrix taking a world point to a pixel (``p-to-image``),
its inverse for rays (``p-to-realworld``), the camera's position, the photo's
footprint on the ground and its view direction, all in one map projection.

Each matrix is fitted to points projected by our own camera model
(:mod:`kyfromabove_3d.cameras`, the one that colours the meshes): a grid of
pixels is cast onto horizontal planes from below the ground to above the
tallest roofs, the points are taken from Kentucky state plane feet and NAVD88
to UTM zone 16 north and heights above the WGS84 ellipsoid -- the same
geoid-gridded transform the 3D Tiles use -- and the 3x4 matrix that best maps
them back to their pixels is solved for. The fit's error is reported in pixels.

The photos stay where KyFromAbove keeps them (512-pixel JPEG-tiled COGs with
overviews); ``cogs.json`` beside ``image.json`` names each photo's COG, for the
page's tile adapter. Usage::

    python scripts/oblique_collection.py --frames outputs/downtown/frames_2024s1.gpkg \\
        --out viewers/vcmap/app/obl/KY_KYAPED_2024_Season1_3IN
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kyfromabove_3d.cameras import parse_camera_from_eo_record          # noqa: E402
from kyfromabove_3d.coordinates import METERS_PER_US_SURVEY_FOOT as FT   # noqa: E402
from kyfromabove_3d.coordinates import LocalCoordinateFrame              # noqa: E402

UTM16N = "+proj=utm +zone=16 +datum=WGS84 +units=m +no_defs"
TILE = 512
#: Heights (m, NAVD88) the pixel grid is cast onto, relative to the ground: the
#: fit holds from below street level to above the tallest downtown roof.
PLANES_M = (-20.0, 0.0, 60.0, 150.0, 260.0)


def to_utm_ellipsoid():
    """State plane feet + NAVD88 feet -> (UTM 16N metres, ellipsoidal height m),
    with the geoid grid or not at all (the 3D Tiles' rule, export_3dtiles.to_ecef)."""
    import pyproj
    from pyproj import Transformer
    from pyproj.transformer import TransformerGroup

    if os.environ.get("PROJ_NETWORK", "").upper() != "OFF":
        pyproj.network.set_network_enabled(True)
    group = TransformerGroup("EPSG:3089+6360", "EPSG:4979", always_xy=True)
    if not (group.best_available and group.transformers):
        raise SystemExit("[FAIL] the NAVD88 -> ellipsoid transform needs grids that are not available")
    to_ll = group.transformers[0]
    to_utm = Transformer.from_crs("EPSG:4979", UTM16N, always_xy=True)

    def f(x_ft, y_ft, z_ft):
        lon, lat, h = to_ll.transform(x_ft, y_ft, z_ft)
        e, n = to_utm.transform(lon, lat)
        out = np.column_stack([e, n, h])
        if not np.isfinite(out).all():
            raise SystemExit("[FAIL] the NAVD88 -> ellipsoid transform returned no height: the GEOID grid is missing")
        return out
    return f


def fit_projection(world: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """The 3x4 matrix P with uv ~ P [world, 1], by the normalised direct linear
    transform (Hartley): centre and scale both point sets, solve, undo."""
    mw, sw = world.mean(0), np.sqrt(3) / np.linalg.norm(world - world.mean(0), axis=1).mean()
    mu, su = uv.mean(0), np.sqrt(2) / np.linalg.norm(uv - uv.mean(0), axis=1).mean()
    Tw = np.diag([sw, sw, sw, 1.0]); Tw[:3, 3] = -sw * mw
    Tu = np.diag([su, su, 1.0]); Tu[:2, 2] = -su * mu
    Xn = (Tw @ np.c_[world, np.ones(len(world))].T).T
    un = (Tu @ np.c_[uv, np.ones(len(uv))].T).T
    rows = []
    for X, (u, v, _) in zip(Xn, un):
        rows.append(np.r_[X, np.zeros(4), -u * X])
        rows.append(np.r_[np.zeros(4), X, -v * X])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    P = np.linalg.inv(Tu) @ vt[-1].reshape(3, 4) @ Tw
    return P / np.linalg.norm(P[2, :3])


def project(P: np.ndarray, world: np.ndarray) -> np.ndarray:
    h = (P @ np.c_[world, np.ones(len(world))].T).T
    return h[:, :2] / h[:, 2:3]


def view_direction(axis_utm: np.ndarray) -> tuple[int, float]:
    """VC Map's direction (1 north, 2 east, 3 south, 4 west, 5 nadir) and its
    angle in radians from east toward north, from the optical axis."""
    e, n, u = axis_utm
    angle = math.atan2(n, e) % (2 * math.pi)
    if -u > 0.94:                                       # within ~20 degrees of straight down
        return 5, angle
    compass = (90.0 - math.degrees(angle)) % 360.0      # 0 north, clockwise
    return [1, 2, 3, 4][int(((compass + 45.0) % 360.0) // 90.0)], angle


def overview_resolutions(width: int, height: int) -> list[int]:
    """1, 2, 4 ... down to the level where the photo fits one tile, coarsest first
    (the COGs carry an overview at each)."""
    res = [1]
    while max(width, height) / res[-1] > TILE:
        res.append(res[-1] * 2)
    return res[::-1]


def main() -> int:
    import geopandas as gpd

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, required=True, help="frames with exterior orientation (frames_*.gpkg)")
    ap.add_argument("--out", type=Path, required=True, help="folder for image.json and cogs.json")
    ap.add_argument("--ground-navd88-m", type=float, default=140.0, help="the area's ground height, NAVD88 metres")
    args = ap.parse_args()

    frames = gpd.read_file(args.frames)
    convert = to_utm_ellipsoid()
    origin = (float(frames.X.mean()), float(frames.Y.mean()), args.ground_navd88_m / FT)
    lcf = LocalCoordinateFrame(origin_source_xyz=origin)

    def local_to_utm(pts_m):
        src = pts_m / FT + np.asarray(origin)
        return convert(src[:, 0], src[:, 1], src[:, 2])

    cams, images, cogs, worst = {}, [], {}, []
    for _, rec in frames.iterrows():
        cam = parse_camera_from_eo_record(rec, lcf)
        key = str(rec.CameraID)
        W, H = cam.width_px, cam.height_px
        if key not in cams:
            cams[key] = {"size": [W, H], "principal-point": [cam.cx_px, H - cam.cy_px],
                         "tile-resolution": overview_resolutions(W, H), "index": len(cams)}
        # the pixel grid cast onto planes from below the ground to above the roofs
        gu, gv = np.meshgrid(np.linspace(0, W, 9), np.linspace(0, H, 7))
        uv = np.c_[gu.ravel(), gv.ravel()]
        c, rays = cam.unproject(uv)
        pts, pix = [], []
        for z in PLANES_M:
            t = (z - c[2]) / rays[:, 2]
            ok = t > 0
            pts.append(c + rays[ok] * t[ok, None]); pix.append(uv[ok])
        world = local_to_utm(np.vstack(pts)); pix = np.vstack(pix)
        P = fit_projection(world, pix)
        err = np.linalg.norm(project(P, world) - pix, axis=1)
        worst.append(float(err.max()))
        M, p4 = P[:, :3], P[:, 3]
        centre = -np.linalg.solve(M, p4)
        # footprint: the photo's corners on the ground plane, and its centre
        corners_uv = np.array([[0, H], [W, H], [W, 0], [0, 0], [W / 2, H / 2]], float)
        _, cr = cam.unproject(corners_uv)
        tc = -c[2] / cr[:, 2]
        ground = local_to_utm(c + cr * tc[:, None])
        # the optical axis in UTM: from the centre through the ground point at the photo's middle
        axis = ground[4] - centre
        direction, angle = view_direction(axis / np.linalg.norm(axis))
        name = str(rec.ShotID)
        cogs[name] = str(rec.S3URL)
        images.append([name, W, H, cams[key]["tile-resolution"], direction, angle,
                       ground[:4].round(3).tolist(), ground[4].round(3).tolist(), cams[key]["index"],
                       centre.round(4).tolist(), np.linalg.inv(M).tolist(), P.tolist()])

    header = ["name", "width", "height", "tile-resolution", "view-direction", "view-direction-angle",
              "groundCoordinates", "centerPointOnGround", "camera-index", "projection-center",
              "p-to-realworld", "p-to-image"]
    ordered = sorted(cams.items(), key=lambda kv: kv[1]["index"])
    first = ordered[0][1]
    doc = {"version": "v3.5-0-kyfromabove",
           "generalImageInfo": {"width": first["size"][0], "height": first["size"][1],
                                "tile-resolution": first["tile-resolution"], "tile-width": TILE, "tile-height": TILE,
                                "crs": UTM16N,
                                "cameraParameter": [{"name": k, "size": v["size"], "principal-point": v["principal-point"],
                                                     "tileResolution": v["tile-resolution"]} for k, v in ordered]},
           "images": [header] + images}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "image.json").write_text(json.dumps(doc), encoding="utf-8")
    (args.out / "cogs.json").write_text(json.dumps(cogs, indent=0), encoding="utf-8")
    w = np.asarray(worst)
    sizes = ", ".join(f"{k} {v['size'][0]}x{v['size'][1]}" for k, v in ordered)
    print(f"{len(images)} photos, cameras {sizes}; "
          f"matrix fit error: largest {w.max():.3f} px, median of each photo's largest {np.median(w):.3f} px "
          f"-> {args.out / 'image.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
