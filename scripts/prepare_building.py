#!/usr/bin/env python3
"""Prepare one building's working directory from an Overture record.

The lean preparation ``kyfromabove_3d.prepare`` does -- frame, LiDAR clip,
aerial cameras and crops, no QA -- for one GERS id out of a footprint JSON
(the file ``overture_boxes.py lidar-heights`` writes). The frame index is
cached beside the output so the second building does not query it again.

Usage::

    python scripts/prepare_building.py --footprints data/derived/louisville_downtown_overture_lidar.json \\
        --gers <id> --out outputs/downtown/buildings/<id> --frames-cache outputs/downtown/frames_2024s1.gpkg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def downtown_bounds_ft(records) -> tuple[float, float, float, float]:
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform
    from shapely.wkt import loads as wkt_loads

    to_ft = Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True).transform
    b = np.array([shp_transform(to_ft, wkt_loads(r["wkt"])).bounds for r in records])
    pad = 300.0 / 0.3048006096012192
    return (float(b[:, 0].min() - pad), float(b[:, 1].min() - pad), float(b[:, 2].max() + pad), float(b[:, 3].max() + pad))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--footprints", type=Path, required=True)
    ap.add_argument("--gers", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames-cache", type=Path, required=True)
    ap.add_argument("--pad-ft", type=float, default=90.0)
    ap.add_argument("--max-per-head", type=int, default=4)
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = ap.parse_args()

    from kyfromabove_3d.lidar import LidarTileIndex
    from kyfromabove_3d.prepare import load_frames_index, prepare_building

    data = json.loads(args.footprints.read_text(encoding="utf-8"))
    records = data["buildings"]
    rec = next((r for r in records if r["id"] == args.gers), None)
    if rec is None:
        print(f"[FAIL] {args.gers} not in {args.footprints}")
        return 1
    grounds = [r["lidar_ground_z"] for r in records if r.get("lidar_ground_z") is not None]
    frames = load_frames_index(args.frames_cache, downtown_bounds_ft(records))
    index = LidarTileIndex(args.raw_dir, phases=("phase2",))
    print(f"{rec.get('name') or rec['id']}: {len(frames)} frames in the index")
    result = prepare_building(rec, args.out, frames, index, records, pad_ft=args.pad_ft,
                              max_per_head=args.max_per_head, min_frames=args.min_frames,
                              fallback_ground_m=float(np.median(grounds)) if grounds else None)
    (args.out / "prepare.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"prepared -> {args.out} ({result['t_total']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
