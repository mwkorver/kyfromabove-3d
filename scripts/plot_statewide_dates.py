#!/usr/bin/env python3
"""The pictures for the statewide estimate (docs/CLOUD.md, "What all of Kentucky costs").

From ``survey_statewide_dates.py``'s files and ``estimate_statewide_cost.py``'s
numbers:

* ``statewide_flown.png`` -- when each LiDAR phase and each oblique
  collection was flown, as buildings under it per month;
* ``statewide_pairs_map.png`` -- tile by tile, the LiDAR flight the run would
  use, the oblique collection, and how far apart in time the two are;
* ``statewide_gap.png`` -- the buildings by that gap;
* ``statewide_cost.png`` -- the run's cost by part.

Usage::

    python scripts/plot_statewide_dates.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
from matplotlib.patches import Patch                                # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NONE_GRAY, OTHER_GRAY = "#d9d8d3", "#8a8984"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
# diverging: obliques later than the LiDAR (blue arm), earlier (red arm), gray within 6 months
DIV_LATER = ["#86b6ef", "#3987e5", "#1c5cab"]
DIV_EARLIER = ["#f2a7a6", "#e34948", "#a52a2a"]
MID = "#c9c8c3"
COLLECTION_NAME = {"2024_Season1": "2024 Season 1", "2023_Season1": "2023 Season 1",
                   "2023_Season2": "2023 Season 2", "2022_Season2": "2022 Season 2"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def flown(p, lid, obl, out):
    """Buildings under each acquisition, by month of flight."""
    b = p.set_index("tile")["buildings"]
    rows = []
    lid = lid[lid["first"].notna()].copy()
    lid["date"] = pd.to_datetime(lid["first"]) + (pd.to_datetime(lid["last"]) - pd.to_datetime(lid["first"])) / 2
    for ph, label in (("phase2", "LiDAR Phase 2"), ("phase3", "LiDAR Phase 3")):
        g = lid[lid.phase == ph]
        rows.append((label, g["date"], g["tile"].map(b).fillna(0).values, BLUE))
    for c in ("2022_Season2", "2023_Season1", "2023_Season2", "2024_Season1"):
        g = obl[(obl.collection == c) & (obl.n >= 5)]
        rows.append((f"Obliques {COLLECTION_NAME[c]}", pd.to_datetime(g["median"]), g["tile"].map(b).fillna(0).values, ORANGE))
    months = pd.date_range("2019-01-01", "2026-06-01", freq="MS")
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
    style(ax)
    ymax = 0.42
    for i, (label, dates, w, col) in enumerate(rows):
        h, _ = np.histogram(pd.to_datetime(dates).values.astype("datetime64[D]").astype(float),
                            bins=months.values.astype("datetime64[D]").astype(float), weights=w)
        scale = ymax / max(h.max(), 1)
        y0 = len(rows) - 1 - i
        for m0, m1, v in zip(months[:-1], months[1:], h):
            if v > 0:
                ax.bar(m0, v * scale * 2, width=(m1 - m0).days * 0.9, bottom=y0 - v * scale, color=col, align="edge")
        total = w.sum()
        first, last = pd.to_datetime(dates).min(), pd.to_datetime(dates).max()
        ax.text(pd.Timestamp("2026-06-15"), y0, f"{total / 1e6:.2f} M buildings under it\n{first:%b %Y} – {last:%b %Y}",
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=9.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(pd.Timestamp("2019-01-01"), pd.Timestamp("2028-06-01"))
    ax.set_title("When Kentucky was flown: each bar is the buildings under that capture, month by month\n"
                 "(LiDAR dated by the GPS time of its returns; obliques by each photo's FlightDate)",
                 fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def pairs_map(p, grid, out):
    g = grid.merge(p, left_on="tilename", right_on="tile", how="inner")
    g = g.to_crs("EPSG:3089")
    fig, axs = plt.subplots(3, 1, figsize=(11, 13.5), facecolor=SURFACE)
    for ax in axs:
        ax.set_facecolor(SURFACE)
        ax.set_axis_off()
    # 1: the LiDAR the run would use, by year flown
    yr = pd.to_datetime(g["lidar_date"]).dt.year
    groups = [("2019", (2019,), "#cde2fb"), ("2020-2021", (2020, 2021), "#86b6ef"), ("2022", (2022,), "#3987e5"),
              ("2023", (2023,), "#256abf"), ("2024", (2024,), "#184f95"), ("2025-2026", (2025, 2026), "#0d366b")]
    col = yr.map({y: c for _, ys, c in groups for y in ys})
    g.plot(ax=axs[0], color=col, linewidth=0)
    p3 = g[g.phase == "phase3"].dissolve()
    p3.boundary.plot(ax=axs[0], color=ORANGE, linewidth=1.2)
    axs[0].legend(handles=[Patch(color=c, label=t) for t, _, c in groups]
                  + [Patch(facecolor="none", edgecolor=ORANGE, label="built from Phase 3 (outlined)")],
                  ncol=7, loc="lower left", fontsize=8, frameon=False, bbox_to_anchor=(0, -0.08))
    axs[0].set_title("The LiDAR each tile would be built from, by the year it was flown", fontsize=10.5, color=INK, loc="left")
    # 2: the oblique collection
    cmap = {"2024_Season1": BLUE, "2023_Season1": ORANGE, "2023_Season2": AQUA, "2022_Season2": OTHER_GRAY}
    g.plot(ax=axs[1], color=g["collection"].map(cmap).fillna(NONE_GRAY), linewidth=0)
    axs[1].legend(handles=[Patch(color=cmap[c], label=f"{COLLECTION_NAME[c]}") for c in cmap]
                  + [Patch(color=NONE_GRAY, label="no obliques (boxes only)")],
                  ncol=5, loc="lower left", fontsize=8, frameon=False, bbox_to_anchor=(0, -0.08))
    axs[1].set_title("The oblique collection each tile would be colored from", fontsize=10.5, color=INK, loc="left")
    # 3: the gap
    m = g["gap_days"] / 30.44

    def gap_col(x):
        if pd.isna(x):
            return NONE_GRAY
        a = abs(x)
        if a < 6:
            return MID
        arm = DIV_LATER if x > 0 else DIV_EARLIER
        return arm[0] if a < 12 else arm[1] if a < 24 else arm[2]
    g.plot(ax=axs[2], color=m.map(gap_col), linewidth=0)
    labels = [("obliques 2+ years after the LiDAR", DIV_LATER[2]), ("1-2 years after", DIV_LATER[1]),
              ("6-12 months after", DIV_LATER[0]), ("within 6 months", MID), ("6-12 months before", DIV_EARLIER[0]),
              ("1-2 years before", DIV_EARLIER[1]), ("2+ years before", DIV_EARLIER[2])]
    axs[2].legend(handles=[Patch(color=c, label=t) for t, c in labels], ncol=4, loc="lower left", fontsize=8,
                  frameon=False, bbox_to_anchor=(0, -0.14))
    axs[2].set_title("How far apart in time the two were flown", fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def gap_bars(p, out):
    B = p.buildings.sum()
    m = p.gap_days / 30.44
    bins = [("2+ years\nbefore", -1e9, -24, DIV_EARLIER[2]), ("1-2 years\nbefore", -24, -12, DIV_EARLIER[1]),
            ("6-12 months\nbefore", -12, -6, DIV_EARLIER[0]), ("within\n6 months", -6, 6, MID),
            ("6-12 months\nafter", 6, 12, DIV_LATER[0]), ("1-2 years\nafter", 12, 24, DIV_LATER[1]),
            ("2+ years\nafter", 24, 1e9, DIV_LATER[2])]
    vals = [p.buildings[(m >= a) & (m < b)].sum() for _, a, b, _ in bins]
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    x = np.arange(len(bins))
    ax.bar(x, np.array(vals) / 1e3, width=0.72, color=[c for *_, c in bins], edgecolor=SURFACE, linewidth=2)
    for xi, v in zip(x, vals):
        ax.text(xi, v / 1e3 + 12, f"{v / 1e3:,.0f}k\n{v / B:.0%}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(x, [b[0] for b in bins], fontsize=9, color=INK)
    ax.set_ylabel("buildings, thousands", color=INK2, fontsize=9)
    ax.set_ylim(0, max(vals) / 1e3 * 1.25)
    nobl = p.buildings[p.collection.isna()].sum()
    ax.set_title(f"When the obliques were flown, against the LiDAR each building would be shaped from\n"
                 f"(all {B / 1e6:.2f} million buildings; {nobl:,.0f} on tiles no oblique covers are left out)",
                 fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def cost_bars(cost, out):
    mid, lo, hi = cost["cases"]["middle"], cost["cases"]["low"], cost["cases"]["high"]
    jh = mid["job_hours"]
    total_jh = sum(jh.values())
    mach = mid["usd"]["machines"]
    parts = [(f"build ({jh['build']:,} job-hours)", mach * jh["build"] / total_jh, BLUE),
             (f"measure ({jh['measure']:,})", mach * jh["measure"] / total_jh, BLUE),
             (f"publish ({jh['publish']:,})", mach * jh["publish"] / total_jh, BLUE),
             (f"tile ({jh['tile']:,})", mach * jh["tile"] / total_jh, BLUE),
             (f"tree ({jh['tree']:,})", mach * jh["tree"] / total_jh, BLUE),
             ("S3 uploads (26 million)", mid["usd"]["s3_uploads"], ORANGE),
             ("S3 reads (35 million)", mid["usd"]["s3_reads"], ORANGE)]
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    style(ax)
    y = np.arange(len(parts))[::-1]
    ax.barh(y, [v for _, v, _ in parts], height=0.62, color=[c for *_, c in parts])
    for yi, (_, v, _) in zip(y, parts):
        ax.text(v + 3, yi, f"${v:,.0f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [t for t, _, _ in parts], fontsize=9.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("dollars, one run of every step for all of Kentucky", color=INK2, fontsize=9)
    ax.legend(handles=[Patch(color=BLUE, label="Spot machines (c7g.4xlarge)"), Patch(color=ORANGE, label="S3 requests")],
              loc="lower right", frameon=False, fontsize=9)
    ax.set_title(f"About ${mid['usd']['total']:,} (between ${lo['usd']['total']:,} and ${hi['usd']['total']:,}); "
                 f"{mid['machine_hours']:,} machine-hours, {mid['days_on_machines']} days on 8 machines",
                 fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.subplots_adjust(left=0.2)
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("outputs/survey/statewide"))
    ap.add_argument("--cost", type=Path, default=Path("docs/data/statewide_cost.json"))
    ap.add_argument("--figures", type=Path, default=Path("docs/figures"))
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = ap.parse_args()
    import geopandas as gpd

    from kyfromabove_3d.lidar import LidarTileIndex
    p = pd.read_pickle(args.dir / "pairing.pkl")
    lid = pd.read_pickle(args.dir / "lidar_dates.pkl")
    obl = pd.read_pickle(args.dir / "oblique_dates.pkl")
    grid = gpd.read_file(args.raw_dir / "kyfromabove_phase2_pointcloud_5k_grid.gpkg")
    if grid.crs is None:
        grid = grid.set_crs(LidarTileIndex.GRID_CRS)
    flown(p, lid, obl, args.figures / "statewide_flown.png")
    pairs_map(p, grid, args.figures / "statewide_pairs_map.png")
    gap_bars(p, args.figures / "statewide_gap.png")
    cost_bars(json.loads(args.cost.read_text()), args.figures / "statewide_cost.png")
    print("figures ->", args.figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
