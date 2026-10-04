#!/usr/bin/env python3
"""The corner snap over a set of buildings: how many corners could be
observed, and the gap at each corner before and after.

Reads every ``corner_snap.json`` under the buildings directory (or the
ones named with ``--only``) and draws two things: the status of every
corner (snapped, measured, no observation, skipped as a party wall or a
curved wall, refused) and the gap distribution before and after for the
corners that had observations.

Usage::

    python scripts/plot_corner_snap.py --only <gers> ... --out docs/figures/corner_snap_gaps.png
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buildings", type=Path, default=Path("outputs/downtown/buildings"))
    ap.add_argument("--only", nargs="*", default=None, help="building ids (prefixes allowed)")
    ap.add_argument("--out", type=Path, default=Path("docs/figures/corner_snap_gaps.png"))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dirs = sorted(d for d in args.buildings.iterdir() if (d / "corner_snap.json").exists())
    if args.only:
        dirs = [d for d in dirs if any(d.name.startswith(o) for o in args.only)]
    status = collections.Counter()
    before, after, moves = [], [], []
    for d in dirs:
        c = json.loads((d / "corner_snap.json").read_text(encoding="utf-8"))
        for r in c["corners"]:
            status[r["status"]] += 1
            if r["status"] in ("snapped", "measured") and r["gap_before_m"] is not None and r["gap_after_m"] is not None:
                before.append(r["gap_before_m"]); after.append(r["gap_after_m"])
            if r["status"] == "snapped":
                moves.append(float(np.hypot(*(np.array(r["after"]) - np.array(r["before"])))))
    total = sum(status.values())
    print(f"{len(dirs)} buildings, {total} corners: " + ", ".join(f"{k} {v}" for k, v in status.most_common()))
    if before:
        print(f"corners with observations: {len(before)}; gap median {np.median(before):.2f} m -> {np.median(after):.2f} m, "
              f"90th {np.percentile(before, 90):.2f} -> {np.percentile(after, 90):.2f} m; "
              f"{len(moves)} moved, median {np.median(moves) if moves else 0:.2f} m")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    order = ["snapped", "measured", "refused", "no-observation", "skipped-party", "skipped-arc", "skipped-no-plane"]
    labels = {"snapped": "snapped to the frames' edge", "measured": "observed, already there", "refused": "refused (too far, or the two faces disagree)",
              "no-observation": "no frame shows the edge", "skipped-party": "party wall: nothing to see",
              "skipped-arc": "on a curved wall", "skipped-no-plane": "an edge of a curved wall, or under 0.5 m"}
    vals = [status.get(k, 0) for k in order]
    axs[0].barh([labels[k] for k in order][::-1], vals[::-1], color=["#2a9d8f", "#8ab17d", "#e76f51", "#999999", "#bbbbbb", "#bbbbbb", "#cccccc"][::-1])
    for i, v in enumerate(vals[::-1]):
        axs[0].text(v + 0.5, i, str(v), va="center", fontsize=9)
    axs[0].set_title(f"what happened to each of {total} corners ({len(dirs)} buildings)", fontsize=10)
    axs[0].set_xlabel("corners")
    if before:
        bins = np.arange(0, max(max(before), max(after), 0.5) + 0.25, 0.25)
        axs[1].hist(before, bins=bins, alpha=0.6, label=f"before (median {np.median(before):.2f} m)", color="#e76f51")
        axs[1].hist(after, bins=bins, alpha=0.6, label=f"after (median {np.median(after):.2f} m)", color="#2a9d8f")
        axs[1].set_xlabel("gap at the corner between the observed wall end and the modelled corner (m)")
        axs[1].set_ylabel("corners")
        axs[1].set_title("the corners the frames could place", fontsize=10)
        axs[1].legend(fontsize=9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
