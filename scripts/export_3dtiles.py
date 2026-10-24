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

    python scripts/export_3dtiles.py --building outputs/downtown/buildings/1614a997-3d6f-475f-9bc5-bb5a9dbc878d      # PNC Tower
    # -> <building>/3dtiles/tileset.json
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

from kyfromabove_3d.clouds import read_cloud
from kyfromabove_3d.coordinates import to_ecef


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
