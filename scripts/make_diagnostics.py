#!/usr/bin/env python3
"""Render the figures that explain the findings, not the per-frame QA figures.

``validate_building.py`` already emits per-frame overlays and residual fields.
These are different: each one exists because a specific conclusion in the
documentation is hard to believe from a table alone.

    building_profile.png   what the bounding volume actually contains, and how
                           far the two AOIs are apart in scale
    roof_section.png       why a domed crown cannot be four corners at one
                           median elevation
    anchor_visibility.png  that building_002's low acceptance is occlusion --
                           bimodal per-anchor rates, not a uniform degradation
    residual_vs_distance.png  where cross-view residuals actually spike

Usage::

    python scripts/make_diagnostics.py
    python scripts/make_diagnostics.py --out outputs/diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from kyfromabove_3d import Project, build_planar_anchors
from kyfromabove_3d.features import roof_check_features, roof_footprint
from kyfromabove_3d.plots import (
    plot_anchor_visibility,
    plot_building_profile,
    plot_residual_vs_distance,
    plot_roof_section,
)

AOIS = {
    "building_001 — Tractor Supply, Grayson": ("configs/phase1.yaml", 15.0, "largest"),
    "building_002 — 400 West Market, Louisville": ("configs/louisville.yaml", 180.0, "tallest"),
}


def legacy_volume(volume: dict, roof_min_z: float) -> dict:
    """The volume the old ``roof - 8 m`` rule would have produced."""
    v = dict(volume)
    v["min_z_m"] = roof_min_z - 8.0 - 2.0          # the -8 rule, then pad_z
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("outputs/diagnostics"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    args.out.mkdir(parents=True, exist_ok=True)

    clouds, roofs = {}, {}
    for label, (cfg, max_h, select) in AOIS.items():
        proj = Project(config_path=cfg)
        lidar = proj.lidar
        xyz = np.asarray(lidar.xyz_local_m)
        cls = np.asarray(lidar.classification)
        _, roof_xyz = roof_check_features(lidar, max_height_m=max_h, select=select)
        vol = json.loads((proj.building_dir / "check_features.json").read_text())
        clouds[label] = {
            "xyz": xyz, "cls": cls,
            "volume": vol["volume_local_m"],
            "legacy_volume": legacy_volume(vol["volume_local_m"],
                                           float(roof_xyz[:, 2].min())),
            "title": (f"{len(xyz):,} points · roof at "
                      f"{np.median(roof_xyz[:, 2]):.0f} m"),
        }
        roofs[label] = (proj, roof_xyz, vol)
        print(f"[load] {label}: {len(xyz):,} points, roof {len(roof_xyz):,}")

    plot_building_profile(clouds, save_to=args.out / "building_profile.png")
    print(f"[write] {args.out / 'building_profile.png'}")

    # The dome, from the tower.
    tower = [k for k in roofs if "002" in k][0]
    proj, roof_xyz, vol = roofs[tower]
    plot_roof_section(roof_xyz, roof_footprint(roof_xyz), vol["features"],
                      save_to=args.out / "roof_section.png")
    print(f"[write] {args.out / 'roof_section.png'}")

    # Anchor visibility, both AOIs, from the cross-direction reports.
    for label, (proj, roof_xyz, vol) in roofs.items():
        report = proj.qa_dir / "cross_direction_report.json"
        if not report.exists():
            print(f"[skip] no cross-direction report for {label}")
            continue
        anchors = {a.anchor_id: (a.xyz_local_m[0], a.xyz_local_m[1])
                   for a in build_planar_anchors(proj.lidar)}
        tally = defaultdict(lambda: [0, 0])
        for row in json.loads(report.read_text())["matches"]:
            tally[row["anchor_id"]][0] += 1
            tally[row["anchor_id"]][1] += bool(row["accepted"])
        rate = {k: acc / n for k, (n, acc) in tally.items() if n}
        tag = "building_002" if "002" in label else "building_001"
        plot_anchor_visibility(anchors, rate, footprint=roof_footprint(roof_xyz),
                               title=f"{label} — cross-direction match rate",
                               save_to=args.out / f"anchor_visibility_{tag}.png")
        print(f"[write] {args.out / f'anchor_visibility_{tag}.png'}")

        v = vol["volume_local_m"]
        cx = (v["min_x_m"] + v["max_x_m"]) / 2
        cy = (v["min_y_m"] + v["max_y_m"]) / 2
        acc = [r for r in json.loads(report.read_text())["matches"]
               if r["accepted"] and r["anchor_id"] in anchors]
        if acc:
            d = np.array([np.hypot(anchors[r["anchor_id"]][0] - cx,
                                   anchors[r["anchor_id"]][1] - cy) for r in acc])
            res = np.array([r["residual_px"] for r in acc])
            fp = roof_footprint(roof_xyz)
            half = float(np.hypot(fp[:, 0] - cx, fp[:, 1] - cy).mean())
            plot_residual_vs_distance(d, res, roof_half_extent_m=half,
                                      save_to=args.out / f"residual_vs_distance_{tag}.png")
            print(f"[write] {args.out / f'residual_vs_distance_{tag}.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
