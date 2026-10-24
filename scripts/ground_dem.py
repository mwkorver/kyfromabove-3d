#!/usr/bin/env python3
"""The ground under the downtown, from the LiDAR, for the viewer's terrain.

The compare viewer drew its map on the bare ellipsoid: the OpenStreetMap
tiles at height zero, while every building sits at its true ellipsoidal
height -- about 106 m in downtown Louisville (the NAVD88 ground of ~140 m
less the 34 m the geoid lies above the ellipsoid there). From above the
gap hides behind the buildings; from the street they hang a hundred
metres over the map. The fix is not to lower the buildings (their height
is right) but to give the viewer the ground: this script grids the
LiDAR's ground-classified returns over the area at ``--cell`` metres,
fills the gaps from the nearest cell, and writes the heights as
ellipsoidal metres. ``intake_server.py`` serves it as terrain tiles
(``/terrain``) that Cesium's CustomHeightmapTerrainProvider reads.

Usage::

    python scripts/ground_dem.py --footprints data/derived/louisville_downtown_overture_split.json \\
        --frame outputs/downtown --out outputs/downtown/ground_dem.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kyfromabove_3d.coordinates import METERS_PER_US_SURVEY_FOOT       # noqa: E402


def build(footprints: Path, frame: Path, out: Path, cell_m: float = 2.0, resolution_m: float = 1.0) -> dict:
    from scipy.ndimage import distance_transform_edt, median_filter
    from scipy.stats import binned_statistic_2d

    from kyfromabove_3d.stages.build_mesh import enu_transform
    from kyfromabove_3d.stages.overture_boxes import area_cloud

    recs = json.loads(footprints.read_text(encoding="utf-8"))["buildings"]
    _, xyz, cls, ox, oy = area_cloud(recs, frame, resolution_m)
    ground = xyz[cls == 2]
    coords = json.loads((frame / "coordinates.json").read_text(encoding="utf-8"))
    _, (lon0, lat0, h0) = enu_transform(coords)
    # NAVD88 -> ellipsoidal: the frame origin's own shift, constant to a
    # centimetre over a kilometre or two
    shift = float(h0) - float(coords["origin_source_xyz"][2]) * METERS_PER_US_SURVEY_FOOT
    lo = ground[:, :2].min(axis=0) - cell_m
    hi = ground[:, :2].max(axis=0) + cell_m
    xe = np.arange(lo[0], hi[0] + cell_m, cell_m)
    ye = np.arange(lo[1], hi[1] + cell_m, cell_m)
    Z, _, _, _ = binned_statistic_2d(ground[:, 0], ground[:, 1], ground[:, 2], statistic="median", bins=[xe, ye])
    Z = Z.T                                                         # [row = y, col = x]
    valid = np.isfinite(Z)
    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    Z = Z[iy, ix]                                                   # gaps (under buildings) from the nearest ground cell
    Z = median_filter(Z, size=3)
    Z = Z + shift
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x0=float(xe[0]), y0=float(ye[0]), cell=float(cell_m), Z=Z.astype(np.float32),
                        origin_lon_lat_h=np.array([lon0, lat0, h0]), origin_source_xyz=np.array(coords["origin_source_xyz"]),
                        shift_m=shift, n_ground=int(len(ground)))
    info = {"cells": list(Z.shape), "cell_m": cell_m, "filled_from_neighbours": int((~valid).sum()),
            "z_ellipsoidal_m": [float(Z.min()), float(Z.max())], "navd88_to_ellipsoid_m": round(shift, 3),
            "n_ground_returns": int(len(ground))}
    print(f"ground DEM {Z.shape[1]} x {Z.shape[0]} cells of {cell_m} m from {len(ground):,} ground returns; "
          f"{(~valid).sum():,} cells filled from their neighbours; heights {Z.min():.1f}-{Z.max():.1f} m "
          f"(ellipsoidal; NAVD88 {shift:+.2f} m) -> {out}")
    return info


class GroundDEM:
    """Sample the DEM at lon/lat (bilinear; clamped to the edge outside it)."""

    def __init__(self, path: Path):
        d = np.load(path)
        self.x0, self.y0, self.cell, self.Z = float(d["x0"]), float(d["y0"]), float(d["cell"]), d["Z"]
        self.ox, self.oy, _ = d["origin_source_xyz"]
        from pyproj import Transformer
        self._to_ft = Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True).transform

    def heights(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        x_ft, y_ft = self._to_ft(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
        x = (np.asarray(x_ft) - self.ox) * METERS_PER_US_SURVEY_FOOT
        y = (np.asarray(y_ft) - self.oy) * METERS_PER_US_SURVEY_FOOT
        ny, nx = self.Z.shape
        fx = np.clip((x - self.x0) / self.cell - 0.5, 0, nx - 1)
        fy = np.clip((y - self.y0) / self.cell - 0.5, 0, ny - 1)
        ix = np.floor(fx).astype(int); iy = np.floor(fy).astype(int)
        ix1 = np.minimum(ix + 1, nx - 1); iy1 = np.minimum(iy + 1, ny - 1)
        tx = fx - ix; ty = fy - iy
        z = ((1 - tx) * (1 - ty) * self.Z[iy, ix] + tx * (1 - ty) * self.Z[iy, ix1]
             + (1 - tx) * ty * self.Z[iy1, ix] + tx * ty * self.Z[iy1, ix1])
        return z.astype(np.float32)

    def tile(self, x: int, y: int, level: int, w: int = 32, h: int = 32) -> bytes:
        """A Cesium heightmap tile in the geographic tiling scheme (level 0 is
        two tiles across): ``w x h`` heights from the north-west corner, row
        by row, little-endian float32."""
        n_x, n_y = 2 << level, 1 << level
        dlon, dlat = 360.0 / n_x, 180.0 / n_y
        west, north = -180.0 + x * dlon, 90.0 - y * dlat
        lon = west + (np.arange(w) / (w - 1)) * dlon
        lat = north - (np.arange(h) / (h - 1)) * dlat
        LON, LAT = np.meshgrid(lon, lat)
        return self.heights(LON.ravel(), LAT.ravel()).tobytes()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--footprints", type=Path, default=Path("data/derived/louisville_downtown_overture_split.json"))
    ap.add_argument("--frame", type=Path, default=Path("outputs/downtown"), help="a dir with coordinates.json: the frame")
    ap.add_argument("--out", type=Path, default=Path("outputs/downtown/ground_dem.npz"))
    ap.add_argument("--cell-m", type=float, default=2.0)
    ap.add_argument("--resolution-m", type=float, default=1.0)
    args = ap.parse_args()
    build(args.footprints, args.frame, args.out, args.cell_m, args.resolution_m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
