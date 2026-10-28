#!/usr/bin/env python3
"""Draw where KyFromAbove's four oblique collections were flown, over the
state, and print the share of Kentucky they cover together.

Each collection's outline is its ``ImageFrameBoundary.gpkg`` (a few hundred
KB, read in place); the state is the union of the Phase 2 LiDAR grid.

Usage::

    python scripts/plot_oblique_coverage.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kyfromabove_3d.urban import S3_BASE, collection_full_name

COLLECTIONS = {"2024_Season1": "#2a78b5", "2023_Season1": "#3a9a5b",
               "2023_Season2": "#d08a2c", "2022_Season2": "#b0457c"}
STATE_PLANE = 3089            # areas in Kentucky single-zone state plane (feet)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=Path, default=Path("data/raw/kyfromabove_phase2_pointcloud_5k_grid.gpkg"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures/oblique_coverage.png"))
    args = ap.parse_args()

    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.ops import unary_union

    # The Phase 2 grid ships with no CRS; it is web mercator (docs/LIDAR_SOURCES.md).
    grid = gpd.read_file(args.grid).set_crs(3857, allow_override=True)
    state = gpd.GeoSeries([grid.union_all()], crs=3857)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    state.to_crs(4326).plot(ax=ax, color="#e9e9e9", edgecolor="#555", linewidth=0.8)
    outlines = []
    for c, colour in COLLECTIONS.items():
        g = gpd.read_file(f"/vsicurl/{S3_BASE}/FlightInformationData_GPKG/"
                          f"{collection_full_name(c)}_ImageFrameBoundary.gpkg").to_crs(4326)
        outlines.append(g.union_all())
        g.plot(ax=ax, color=colour, alpha=0.45, edgecolor=colour, linewidth=0.6)
        ax.plot([], [], color=colour, lw=8, alpha=0.6, label=c.replace("_", " "))
    ky = state.to_crs(STATE_PLANE).iloc[0]
    flown = gpd.GeoSeries([unary_union(outlines)], crs=4326).to_crs(STATE_PLANE).iloc[0]
    share = ky.intersection(flown).area / ky.area

    ax.set_xlim(-89.7, -81.8)
    ax.set_ylim(36.4, 39.25)
    ax.set_aspect(1.25)
    ax.legend(loc="lower left", frameon=False, title="oblique photo collection")
    ax.set_title("Where KyFromAbove's oblique photos were flown (gray: the state, from the LiDAR grid)\n"
                 f"Together they cover {share:.1%} of Kentucky, not only the cities", loc="left", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"{share:.2%} of Kentucky under an oblique collection -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
