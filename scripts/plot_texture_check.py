#!/usr/bin/env python3
"""The texture check over a set of buildings, as two pictures.

Left: each building's agreement with the frames it was not textured from,
beside its ceiling (the agreement with the frames it *was* textured from,
which the texture and its compression allow at best). Right: every face,
its agreement against its area, walls and roofs apart -- where the low
scores are.

Usage::

    python scripts/plot_texture_check.py --dirs outputs/downtown/buildings/<gers> ... --out docs/figures/texture_check.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def label(d: Path) -> str:
    ts = d / "3dtiles_mesh" / "tileset.json"
    ex = json.loads(ts.read_text(encoding="utf-8")).get("extras", {}) if ts.exists() else {}
    name = ex.get("name") or d.name
    if name == ex.get("gers"):
        name = d.name[:8] + (" " + d.name[-2:] if d.name[-2] == "-" else "")
    return name[:26]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/texture_check.png"))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, faces = [], []
    for d in args.dirs:
        p = d / "texture_check.json"
        if not p.exists():
            continue
        c = json.loads(p.read_text(encoding="utf-8"))
        b = c["building"]
        if b["ncc"] is None:
            continue
        rows.append((label(d), b["ncc"], b["ncc_source"], b["area_checked_m2"]))
        for f in c["faces"].values():
            if f["ncc"] is not None:
                faces.append((f["kind"], f["area_m2"], f["ncc"]))
    rows.sort(key=lambda r: r[1])
    print(f"{len(rows)} buildings, {len(faces)} faces; building agreement median {np.median([r[1] for r in rows]):.2f}")

    fig, axs = plt.subplots(1, 2, figsize=(13, 0.32 * len(rows) + 1.8), gridspec_kw={"width_ratios": [1.1, 1]})
    y = np.arange(len(rows))
    ax = axs[0]
    for i, (_, a, c, _) in enumerate(rows):
        if c is not None:
            ax.plot([a, c], [i, i], color="#c3c2b7", lw=2, zorder=1)
    ax.scatter([r[1] for r in rows], y, s=40, color="#2a78d6", zorder=2, label="agreement with the other frames")
    ax.scatter([r[2] if r[2] is not None else np.nan for r in rows], y, s=40, facecolor="white",
               edgecolor="#898781", zorder=2, label="ceiling: agreement with its own source frame")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlim(-0.1, 1.0); ax.set_xlabel("pattern agreement (1 = the same, 0 = unrelated), area-weighted median")
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False)
    ax = axs[1]
    for kind, colour, marker in (("wall", "#eb6834", "o"), ("roof", "#1baf7a", "s")):
        f = [x for x in faces if x[0] == kind]
        ax.scatter([x[1] for x in f], [x[2] for x in f], s=14, color=colour, marker=marker, alpha=0.7,
                   label=f"{kind} faces ({len(f)})")
    ax.set_xscale("log"); ax.set_ylim(-0.4, 1.0)
    ax.set_xlabel("face area (m2, log scale)"); ax.set_ylabel("pattern agreement")
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False)
    for a in axs:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
