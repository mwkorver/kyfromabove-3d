#!/usr/bin/env python3
"""Check whether an ortho collection is a true ortho or a DTM-based one.

Only a true ortho can be segmented directly for building footprints. On a
standard ortho a roof is displaced radially by ``height * tan(theta)``, by an
amount that varies per building and jumps across mosaic seamlines, and that
cannot be recovered from the ortho itself.

Run this once per collection before trusting it::

    python scripts/check_ortho.py
    python scripts/check_ortho.py --prefix imagery/orthos/Phase2/KY_KYAPED_2022_6IN \
                                  --suffix _2022_6IN_cog.tif
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from kyfromabove_3d import (
    OrthoCollection,
    Project,
    assess_ortho_rectification,
    build_planar_anchors,
    ground_elevation_under,
    ortho_tilename,
    readers_for_cameras,
    roof_check_features,
)
from kyfromabove_3d.qa import prefetch_windows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--prefix", default=OrthoCollection.prefix,
                    help="S3 key prefix of the ortho collection")
    ap.add_argument("--suffix", default=OrthoCollection.suffix,
                    help="filename suffix after the tile name")
    ap.add_argument("--camera", default="Color",
                    help="look direction to use as reference (default: the nadir head)")
    ap.add_argument("--json", type=Path, default=None, help="write the result as JSON")
    args = ap.parse_args()

    proj = Project(config_path=args.config)
    collection = OrthoCollection(prefix=args.prefix, suffix=args.suffix)

    lidar = proj.lidar
    features, roof_xyz = roof_check_features(lidar)
    corners = np.array([f.xyz_local_m for f in features
                        if f.feature_id.startswith("RoofCorner")])
    height = float(corners[:, 2].mean() - ground_elevation_under(lidar, corners))

    cameras = [c for c in proj.cameras if c.camera_name.startswith(args.camera)]
    if not cameras:
        print(f"no {args.camera} exposures over this AOI; "
              f"available: {sorted({c.camera_name.split('_')[0] for c in proj.cameras})}")
        return 1
    camera = cameras[len(cameras) // 2]

    anchors = build_planar_anchors(lidar)
    readers = readers_for_cameras([camera])
    window = prefetch_windows(anchors, [camera], readers, 14.0, progress=False)[camera.frame_id]

    x, y, _ = proj.origin_source_xyz
    url = collection.url_for_point(x, y)
    print(f"collection : {args.prefix}")
    print(f"tile       : {ortho_tilename(x, y)}")
    print(f"reference  : {camera.shot_id} ({args.camera} head, validated camera model)")
    print(f"anchors    : {len(anchors)}   building height {height:.2f} m")
    print()

    result = assess_ortho_rectification(
        url, camera, window, anchors, proj.coord_frame, building_height_m=height,
    )

    print(f"  ground  n={result.n_ground:>4}   {result.ground_displacement_m*100:7.1f} cm"
          f"   scatter {result.ground_scatter_m*100:6.1f} cm   <- control")
    print(f"  roof    n={result.n_roof:>4}   {result.roof_displacement_m*100:7.1f} cm"
          f"   scatter {result.roof_scatter_m*100:6.1f} cm")
    print(f"  difference        {result.roof_minus_ground_m*100:+7.1f} cm")
    if result.implied_off_nadir_deg is not None:
        print(f"  implies {result.implied_off_nadir_deg:.1f} deg off-nadir on a {height:.2f} m building")
    print(f"  rejections: {result.rejections}")
    print()
    print(f"  VERDICT: {result.verdict.replace('_', ' ').upper()}")
    print(f"  {result.reason}")
    print(f"  usable for footprint segmentation: "
          f"{'yes' if result.usable_for_footprints else 'NO'}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = result.to_dict()
        payload.update({"collection": args.prefix, "tile": ortho_tilename(x, y),
                        "reference_frame": camera.frame_id, "url": url})
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.json}")

    return 0 if result.verdict != "inconclusive" else 2


if __name__ == "__main__":
    sys.exit(main())
