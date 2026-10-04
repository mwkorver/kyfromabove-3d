#!/usr/bin/env python3
"""Two pictures of a downtown run: the footprints coloured by the path each
building took (profile, prism-heavy, box, failed), and how long each took.

Usage::

    python scripts/plot_downtown_run.py --out outputs/downtown --figures docs/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("outputs/downtown"))
    ap.add_argument("--footprints", type=Path, default=Path("data/derived/louisville_downtown_overture_lidar.json"))
    ap.add_argument("--figures", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform
    from shapely.wkt import loads as wkt_loads

    from kyfromabove_3d.coordinates import METERS_PER_US_SURVEY_FOOT
    from kyfromabove_3d.footprints import largest_ring

    man = json.loads((args.out / "manifest.json").read_text(encoding="utf-8"))
    by_gers = {b["gers"]: b for b in man["buildings"]}
    states = {}
    for p in (args.out / "buildings").glob("*/state.json"):
        states[p.parent.name] = json.loads(p.read_text(encoding="utf-8"))
    recs = json.loads(args.footprints.read_text(encoding="utf-8"))["buildings"]
    to_ft = Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True).transform
    coords = json.loads((args.out / "coordinates.json").read_text(encoding="utf-8"))
    ox, oy, _ = coords["origin_source_xyz"]

    def kind(g):
        b = by_gers.get(g, {})
        if b.get("status") == "landmark":
            return "landmark"
        if b.get("status") == "failed":
            return "failed"
        if b.get("status") != "meshed":
            return "pending"
        faces = (states.get(g) or {}).get("faces") or {}
        if b.get("path") == "box":
            return "box"
        if faces.get("profile", 0) == 0:
            return "prism only"
        if faces.get("prism", 0) > faces.get("profile", 0):
            return "mostly prism"
        return "profile"

    colours = {"profile": "#2a78d6", "mostly prism": "#1baf7a", "prism only": "#eda100", "box": "#eb6834",
               "failed": "#e34948", "pending": "#c3c2b7", "landmark": "#4a3aa7"}
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.5), facecolor="#fcfcfb", gridspec_kw={"width_ratios": [1.4, 1]})
    for a in ax:
        a.set_facecolor("#fcfcfb")
    counts = {}
    for r in recs:
        g = shp_transform(to_ft, wkt_loads(r["wkt"]))
        ring = np.array(largest_ring(g).exterior.coords)
        xy = np.column_stack([(ring[:, 0] - ox) * METERS_PER_US_SURVEY_FOOT, (ring[:, 1] - oy) * METERS_PER_US_SURVEY_FOOT])
        k = kind(r["id"]); counts[k] = counts.get(k, 0) + 1
        ax[0].add_patch(MplPolygon(xy, closed=True, fc=colours[k], ec="#fcfcfb", lw=0.4))
    ax[0].set_aspect("equal"); ax[0].autoscale()
    ax[0].set_title("Downtown Louisville: how each footprint was built", loc="left", fontsize=11)
    ax[0].set_xlabel("metres east of the downtown origin"); ax[0].set_ylabel("metres north")
    for k, c in colours.items():
        if counts.get(k):
            ax[0].scatter([], [], color=c, label=f"{k} ({counts[k]})")
    ax[0].legend(frameon=False, fontsize=8, loc="lower left")
    for sp in ("top", "right"):
        ax[0].spines[sp].set_visible(False)

    ts = np.array([s.get("t_total") for s in states.values() if s.get("t_total")])
    ax[1].hist(ts, bins=np.arange(0, max(ts.max(), 60) + 5, 5), color="#2a78d6", edgecolor="#fcfcfb")
    ax[1].set_xlabel("seconds per building (prepare + colourise + mesh)"); ax[1].set_ylabel("buildings")
    ax[1].set_title(f"Time per building: median {np.median(ts):.0f} s, 90th percentile {np.percentile(ts, 90):.0f} s",
                    loc="left", fontsize=11)
    for sp in ("top", "right"):
        ax[1].spines[sp].set_visible(False)
    fig.tight_layout()
    args.figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figures / "downtown_run_paths.png", dpi=150)
    print(counts, "->", args.figures / "downtown_run_paths.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
