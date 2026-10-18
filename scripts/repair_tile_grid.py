#!/usr/bin/env python3
"""Repair a KyFromAbove LiDAR tile list whose tile shapes are broken.

KyFromAbove's Phase 3 list of 15 September 2026
(``s3://kyfromabove/elevation/PointCloud/TileGrids/kyfromabove_phase3_pointcloud_5k_grid.gpkg``)
names every tile, its file and size correctly, but stores every tile's shape
with infinite coordinates, so no area meets any tile and the pipeline reads
no LiDAR (trial1, N078E248). All phases share one 5,000 ft tile grid with one
naming: a shape is taken from the Phase 2 list's tile of the same name
(identical wherever both lists have the tile), or, for a name the Phase 2
list lacks, made from the name itself (row N and column E count 5,000 ft
steps in EPSG:3089), the rule fitted on the Phase 2 tiles and checked
against them. Rows whose shapes are already finite are kept as they are.

Usage::

    python scripts/repair_tile_grid.py data/raw/old_grids/kyfromabove_phase3_pointcloud_5k_grid_2026-09-15.gpkg \\
        --reference data/raw/kyfromabove_phase2_pointcloud_5k_grid.gpkg --out data/raw/kyfromabove_phase3_pointcloud_5k_grid.gpkg
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

GRID_CRS = "EPSG:3857"
TILE_CRS = "EPSG:3089"


def finite(geoms) -> np.ndarray:
    """Which shapes have finite, non-empty bounds."""
    out = []
    for g in geoms:
        out.append(g is not None and not g.is_empty and bool(np.isfinite(g.bounds).all()))
    return np.array(out, bool)


def name_rule(reference):
    """The tile name's position rule, fitted on ``reference`` (finite shapes,
    GRID_CRS): x0 = a + 5000 E, y1 = b - 5000 N in TILE_CRS feet. Returns
    (a, b, the largest corner error on the reference, in feet)."""
    ref = reference.to_crs(TILE_CRS)
    b = np.array([g.bounds for g in ref.geometry])
    full = (np.abs(b[:, 2] - b[:, 0] - 5000.0) < 1.0) & (np.abs(b[:, 3] - b[:, 1] - 5000.0) < 1.0)
    ref, b = ref[full], b[full]                     # the state border's tiles are clipped to it
    rc = np.array([tuple(map(int, re.match(r"N(\d+)E(\d+)", t).groups())) for t in ref.tilename])
    a0 = np.median(b[:, 0] - 5000.0 * rc[:, 1])
    b0 = np.median(b[:, 3] + 5000.0 * rc[:, 0])
    err = max(np.abs(b[:, 0] - (a0 + 5000.0 * rc[:, 1])).max(), np.abs(b[:, 3] - (b0 - 5000.0 * rc[:, 0])).max())
    return a0, b0, float(err)


def shape_from_name(tilename: str, a0: float, b0: float):
    from pyproj import Transformer
    from shapely.geometry import Polygon

    n, e = map(int, re.match(r"N(\d+)E(\d+)", tilename).groups())
    x0, y1 = a0 + 5000.0 * e, b0 - 5000.0 * n
    t = Transformer.from_crs(TILE_CRS, GRID_CRS, always_xy=True).transform
    return Polygon([t(x, y) for x, y in ((x0, y1 - 5000), (x0 + 5000, y1 - 5000), (x0 + 5000, y1), (x0, y1))])


def repair(grid, reference):
    """``grid`` with every non-finite shape replaced (see the module). Returns
    (repaired grid, counts)."""
    grid = grid.copy()
    ok = finite(grid.geometry)
    ref = reference[finite(reference.geometry)].drop_duplicates("tilename").set_index("tilename")
    a0, b0, err = name_rule(ref.reset_index())
    by_ref = by_name = 0
    geoms = list(grid.geometry)
    for i, t in enumerate(grid.tilename):
        if ok[i]:
            continue
        if t in ref.index:
            geoms[i] = ref.geometry[t]
            by_ref += 1
        else:
            geoms[i] = shape_from_name(t, a0, b0)
            by_name += 1
    grid = grid.set_geometry(geoms, crs=GRID_CRS)
    return grid, {"kept": int(ok.sum()), "from_reference": by_ref, "from_name": by_name, "name_rule_error_ft": err}


def main(argv=None) -> int:
    import geopandas as gpd

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grid", type=Path, help="the tile list to repair")
    ap.add_argument("--reference", type=Path, required=True, help="a list with good shapes (Phase 2)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    grid = gpd.read_file(args.grid)
    grid = grid.set_crs(GRID_CRS, allow_override=True) if grid.crs is None else grid.to_crs(GRID_CRS)
    ref = gpd.read_file(args.reference)
    ref = ref.set_crs(GRID_CRS, allow_override=True) if ref.crs is None else ref.to_crs(GRID_CRS)
    fixed, n = repair(grid, ref)
    if n["name_rule_error_ft"] > 5.0:
        print(f"the name rule misses the reference tiles by up to {n['name_rule_error_ft']:.1f} ft: not used", file=sys.stderr)
        return 1
    assert finite(fixed.geometry).all()
    fixed.to_file(args.out, driver="GPKG", layer=args.out.stem)
    print(f"{len(fixed)} tiles -> {args.out}: {n['kept']} kept, {n['from_reference']} shaped from the reference by name, "
          f"{n['from_name']} from the tile name (rule within {n['name_rule_error_ft']:.2f} ft of the reference)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
