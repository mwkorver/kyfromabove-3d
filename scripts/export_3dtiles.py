#!/usr/bin/env python3
"""Write a building's coloured point cloud as OGC 3D Tiles.

The colourised cloud (``colorized.ply``) lives in a local metre frame:
metres east/north/up from an origin given in Kentucky's state plane
(EPSG:3089, US survey feet) with NAVD88 heights. 3D Tiles want positions in
Earth-centred Earth-fixed metres (EPSG:4978) on the ellipsoid, so each point
goes local -> state plane feet -> longitude/latitude/ellipsoidal height
(EPSG:4979, through the GEOID grid pyproj fetches on demand; NAVD88 sits
about 34.6 m above the ellipsoid in Louisville) -> ECEF.

The tileset is two levels: a root tile holding every eighth point, for a
distant view, and eight children -- four plan quadrants times two height
halves -- holding everything, refined by ADD so the children's points join
the root's rather than replace them. Content is ``.pnts`` (POSITION +
RGB, RTC_CENTER for float precision), which every 3D Tiles viewer reads.

Usage::

    python scripts/export_3dtiles.py --building outputs/pnc_tower/pnc_tower
    # -> outputs/pnc_tower/pnc_tower/3dtiles/tileset.json
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

from kyfromabove_3d.clouds import read_cloud


def to_ecef(xyz_local: np.ndarray, coords: dict) -> tuple[np.ndarray, np.ndarray]:
    """Local metres -> ECEF metres, and lon/lat/height for the record."""
    import pyproj
    from pyproj import Transformer

    # the geoid grid: fetched over the network into PROJ's cache, unless
    # PROJ_NETWORK=OFF (the Lambda image, which carries the grids itself)
    if os.environ.get("PROJ_NETWORK", "").upper() != "OFF":
        pyproj.network.set_network_enabled(True)
    ox, oy, oz = coords["origin_source_xyz"]
    ft = coords.get("scale_source_to_meter", 0.3048006096012192)
    crs = coords.get("horizontal_crs", "EPSG:3089")
    # NAVD88 height in US survey feet is EPSG:6360.
    src = f"{crs}+6360"
    x_ft = xyz_local[:, 0] / ft + ox
    y_ft = xyz_local[:, 1] / ft + oy
    z_ft = xyz_local[:, 2] / ft + oz
    # Without the GEOID grid PROJ does not fail: it falls back to a
    # "ballpark" operation. With no vertical shift at all that is 34 m too
    # high in Louisville (the three landmarks were once rebuilt that way and
    # hung above their neighbours' roofs); with the network switched off it
    # found another ballpark and put the whole downtown 1.2 m high and
    # 0.7 m aside (September 2026). ``only_best`` caught neither. Take the
    # best operation only when PROJ says it is available, and stop
    # otherwise, saying which grids are missing.
    from pyproj.transformer import TransformerGroup
    try:
        group = TransformerGroup(src, "EPSG:4979", always_xy=True)
    except Exception as exc:                                   # noqa: BLE001
        raise SystemExit(f"[FAIL] no NAVD88 -> ellipsoid transform for {src}: {str(exc)[:200]}")
    if not (group.best_available and group.transformers):
        missing = sorted({g.short_name for op in group.unavailable_operations[:1] for g in op.grids if not g.available})
        raise SystemExit("[FAIL] the NAVD88 -> ellipsoid transform needs grids that are not available "
                         f"(offline, or an empty PROJ grid cache): {', '.join(missing) or 'unknown'}")
    lon, lat, h = group.transformers[0].transform(x_ft, y_ft, z_ft)
    if not (np.isfinite(lon).all() and np.isfinite(lat).all() and np.isfinite(h).all()):
        raise SystemExit("[FAIL] the NAVD88 -> ellipsoid transform returned no height: the GEOID grid is missing")
    to_ecef_t = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    X, Y, Z = to_ecef_t.transform(lon, lat, h)
    return np.column_stack([X, Y, Z]), np.column_stack([lon, lat, h])


def write_pnts(path: Path, ecef: np.ndarray, rgb: np.ndarray) -> dict:
    """One .pnts tile. Positions are float32 relative to RTC_CENTER."""
    center = ecef.mean(axis=0)
    rel = (ecef - center).astype(np.float32)
    n = len(rel)
    ft_json = {"POINTS_LENGTH": n, "RTC_CENTER": center.tolist(),
               "POSITION": {"byteOffset": 0}, "RGB": {"byteOffset": n * 12}}
    ft_bytes = json.dumps(ft_json).encode("utf-8")
    ft_bytes += b" " * ((8 - (28 + len(ft_bytes)) % 8) % 8)      # body starts 8-byte aligned
    body = rel.tobytes() + rgb.astype(np.uint8).tobytes()
    body += b"\0" * ((8 - len(body) % 8) % 8)
    total = 28 + len(ft_bytes) + len(body)
    header = struct.pack("<4sIIIIII", b"pnts", 1, total, len(ft_bytes), len(body), 0, 0)
    path.write_bytes(header + ft_bytes + body)
    radius = float(np.linalg.norm(rel, axis=1).max())
    return {"center": center.tolist(), "radius": radius, "n": n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--building", type=Path, required=True)
    ap.add_argument("--ply", type=Path, default=None, help="default <building>/colorized.ply")
    ap.add_argument("--out", type=Path, default=None, help="default <building>/3dtiles/")
    ap.add_argument("--root-every", type=int, default=8, help="the root keeps every Nth point")
    ap.add_argument("--root-error-m", type=float, default=6.0,
                    help="geometric error of the root: the screen-space size at which the "
                         "viewer swaps in the children")
    args = ap.parse_args()

    ply_path = args.ply or (args.building / "colorized.ply")
    out = args.out or (args.building / "3dtiles")
    out.mkdir(parents=True, exist_ok=True)
    coords = json.loads((args.building / "coordinates.json").read_text(encoding="utf-8"))

    ply = read_cloud(ply_path)
    xyz = np.column_stack([ply["x"], ply["y"], ply["z"]]).astype(np.float64)
    rgb = np.column_stack([ply["red"], ply["green"], ply["blue"]]).astype(np.uint8)
    print(f"{len(xyz):,} points from {ply_path}")

    ecef, geo = to_ecef(xyz, coords)
    print(f"placed at lon {geo[:, 0].mean():.6f}, lat {geo[:, 1].mean():.6f}, "
          f"ellipsoidal height {geo[:, 2].min():.1f}-{geo[:, 2].max():.1f} m "
          f"(NAVD88 {xyz[:, 2].min() + coords['origin_source_xyz'][2] * 0.3048:.1f}-"
          f"{xyz[:, 2].max() + coords['origin_source_xyz'][2] * 0.3048:.1f} m)")

    # Root: a thinning of everything.
    root_idx = np.arange(0, len(xyz), args.root_every)
    root = write_pnts(out / "root.pnts", ecef[root_idx], rgb[root_idx])

    # Children: plan quadrants x height halves, in the local frame.
    cx, cy = np.median(xyz[:, 0]), np.median(xyz[:, 1])
    cz = np.median(xyz[:, 2])
    children = []
    for qi, (sx, sy) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
        for hi, sz in enumerate([-1, 1]):
            m = ((xyz[:, 0] - cx) * sx >= 0) & ((xyz[:, 1] - cy) * sy >= 0) & ((xyz[:, 2] - cz) * sz >= 0)
            if m.sum() == 0:
                continue
            name = f"q{qi}h{hi}.pnts"
            info = write_pnts(out / name, ecef[m], rgb[m])
            children.append({
                "boundingVolume": {"sphere": info["center"] + [info["radius"]]},
                "geometricError": 0.0,
                "content": {"uri": name},
            })
            print(f"  {name}: {info['n']:,} points")

    tileset = {
        "asset": {"version": "1.0", "generator": "kyfromabove-3d export_3dtiles.py"},
        "geometricError": args.root_error_m * 4,
        "root": {
            "boundingVolume": {"sphere": root["center"] + [root["radius"]]},
            "geometricError": args.root_error_m,
            "refine": "ADD",
            "content": {"uri": "root.pnts"},
            "children": children,
        },
        "extras": {
            "source_ply": str(ply_path), "points": int(len(xyz)),
            "lon": float(geo[:, 0].mean()), "lat": float(geo[:, 1].mean()),
            "height_ellipsoidal_m": [float(geo[:, 2].min()), float(geo[:, 2].max())],
            "vertical_datum_of_source": coords.get("vertical_datum", "NAVD88"),
        },
    }
    (out / "tileset.json").write_text(json.dumps(tileset, indent=1), encoding="utf-8")
    size = sum(p.stat().st_size for p in out.glob("*.pnts")) / 1e6
    print(f"tileset -> {out / 'tileset.json'}  (root {root['n']:,} points + {len(children)} children, {size:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
