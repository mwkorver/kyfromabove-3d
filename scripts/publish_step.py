#!/usr/bin/env python3
"""Step 4 and what the viewer needs, from a store to a store (a folder or
``s3://bucket``): for each tile, its batched 3D Tiles (``tiles/<t>/3dtiles/
<product>/``, :func:`kyfromabove_3d.publish.publish_tile`); the statewide
tree over them (``kentucky/<product>/``); the terrain from the tiles' ground
grids (``terrain/``); and the viewer page with its config (``viewer/``).

Usage::

    python scripts/publish_step.py --inputs s3://kyfromabove-3d --tiles N076E228 N076E229 --tree --terrain --viewer
    python scripts/publish_step.py --inputs outputs/pub_inputs --store outputs/pub_out1 --tiles N076E229 --tree
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: The viewer's views, in its button order: Overture's own heights, the
#: LiDAR heights (product 1), the measured meshes (2), the completed (3).
PRODUCTS = ("overture", "boxes", "measured", "complete")
TITLES = {"overture": "Overture", "boxes": "LiDAR heights", "measured": "Measured", "complete": "Completed"}
DEFAULT_VIEW = "measured"


def one_tile(tile: str, inputs: str, store: str, raw: str, products, available=None,
             buildings_prefix: str | None = None) -> list:
    """Step 4 for one tile, in a worker process."""
    from kyfromabove_3d.pieces import phase2_grid
    from kyfromabove_3d.publish import publish_tile
    from kyfromabove_3d.store import store_from

    grid = phase2_grid(Path(raw))
    src, out = store_from(inputs), store_from(store)
    res = []
    for p in products:
        res.append(publish_tile(tile, src, grid, p, buildings_prefix=buildings_prefix, out_store=out,
                                log=lambda m: None, available=available))
    if inputs != store:                         # the tree reads each tile's records from the output store
        out.put_json(f"tiles/{tile}/buildings.json", src.read_json(f"tiles/{tile}/buildings.json"))
    return res


def terrain(tiles, src, out, grid, max_level: int, log=print) -> dict:
    """Static terrain over the tiles, from their ground grids."""
    import io

    import numpy as np
    from pyproj import Transformer

    from kyfromabove_3d.coordinates import METERS_PER_US_SURVEY_FOOT as FT
    from kyfromabove_3d.publish import _scripts, _tile_nums
    from kyfromabove_3d.terrain import GroundMosaic, write_heightmap_tiles

    bm, _ = _scripts()
    to_ll = Transformer.from_crs("EPSG:3089", "EPSG:4326", always_xy=True).transform
    grounds, shifts, boxes = {}, {}, {}
    for t in tiles:
        key = f"tiles/{t}/ground.npz"
        if not src.exists(key):
            log(f"[warn] {t}: no ground.npz; the terrain there comes from its neighbours")
            continue
        d = dict(np.load(io.BytesIO(src.get_bytes(key))))
        g = {k: (v.item() if v.ndim == 0 else v) for k, v in d.items()}
        Z = g["Z_navd88_m"]
        cx = g["x0"] + g["cell_ft"] * Z.shape[1] / 2
        cy = g["y0"] + g["cell_ft"] * Z.shape[0] / 2
        z = float(np.median(Z))
        _, geo = bm.enu_transform({"origin_source_xyz": [cx, cy, z / FT], "horizontal_crs": "EPSG:3089",
                                   "scale_source_to_meter": FT})
        grounds[t], shifts[t] = g, geo[2] - z
        lon, lat = to_ll([g["x0"], g["x0"] + g["cell_ft"] * Z.shape[1]], [g["y0"], g["y0"] + g["cell_ft"] * Z.shape[0]])
        n, e = _tile_nums(t)
        b = boxes.setdefault((n // 16, e // 16), [180.0, 90.0, -180.0, -90.0])
        boxes[(n // 16, e // 16)] = [min(b[0], min(lon)), min(b[1], min(lat)), max(b[2], max(lon)), max(b[3], max(lat))]
    if not grounds:
        return {"tiles": 0}
    mosaic = GroundMosaic(grounds, shifts)
    return write_heightmap_tiles(mosaic, [tuple(b) for b in boxes.values()],
                                 lambda p, data: out.put_bytes(f"terrain/{p}", data), max_level=max_level, log=log)


def viewer(out, start: dict) -> None:
    out.put_bytes("viewer/index.html", (ROOT / "viewer" / "index.html").read_bytes(), "text/html; charset=utf-8")
    cfg = {"tilesets": {p: f"/kentucky/{p}/tileset.json" for p in PRODUCTS}, "titles": TITLES,
           "default": DEFAULT_VIEW,
           "terrain": {"type": "heightmap", "url": "/terrain/"}, "start": start}
    out.put_json("viewer/config.json", cfg)


def main() -> int:
    from kyfromabove_3d.pieces import phase2_grid
    from kyfromabove_3d.publish import publish_tree
    from kyfromabove_3d.store import store_from

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True,
                    help="the tiles' buildings.json and ground.npz, and buildings/<version>/<gers>/")
    ap.add_argument("--building-version", default=None,
                    help="which version of the building step's results to publish (default: the current code's)")
    ap.add_argument("--store", default=None, help="where to write (default: --inputs)")
    ap.add_argument("--tiles", nargs="*", default=[])
    ap.add_argument("--tiles-file", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--products", nargs="+", default=list(PRODUCTS))
    ap.add_argument("--no-tiles", action="store_true", help="skip step 4 (tree, terrain, viewer only)")
    ap.add_argument("--tree", action="store_true")
    ap.add_argument("--terrain", action="store_true")
    ap.add_argument("--terrain-max-level", type=int, default=15)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--start", type=float, nargs=3, default=(-85.76, 38.24, 12000.0), metavar=("LON", "LAT", "HEIGHT"))
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    args = ap.parse_args()
    tiles = list(args.tiles) + (args.tiles_file.read_text().split() if args.tiles_file else [])
    store = args.store or args.inputs
    src, out = store_from(args.inputs), store_from(store)
    grid = phase2_grid(args.raw)
    t0 = time.time()
    if tiles and not args.no_tiles:
        from kyfromabove_3d.building import BUILDING_CODE_VERSION, buildings_prefix
        from kyfromabove_3d.publish import built_buildings

        bprefix = buildings_prefix(args.building_version or BUILDING_CODE_VERSION)
        available = built_buildings(src, bprefix)                 # one listing, not a question per building
        print(f"{len(available):,} built building(s) in {args.inputs}/{bprefix}", flush=True)
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(one_tile, t, args.inputs, store, str(args.raw), args.products, available, bprefix): t
                    for t in tiles}
            for f in as_completed(futs):
                done += 1
                for s in f.result():
                    if s.get("meshed") or done % 25 == 0 or done == len(tiles):
                        print(f"[{done}/{len(tiles)}] {s['tile']} {s['product']}: {s.get('meshed', 0)} meshed, "
                              f"{s.get('boxed', 0)} boxes, {s.get('areas', 0)} area file(s)", flush=True)
    if args.tree:
        for p in args.products:
            publish_tree(out, grid, p, tiles=tiles or None)
    if args.terrain:
        terrain(tiles, src, out, grid, args.terrain_max_level)
    if args.viewer:
        lon, lat, h = args.start
        viewer(out, {"lon": lon, "lat": lat, "height": h, "heading": 15, "pitch": -40})
        print("viewer -> viewer/index.html, viewer/config.json")
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
