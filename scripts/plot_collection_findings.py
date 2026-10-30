#!/usr/bin/env python3
"""Two pictures for docs/FUTURE_COLLECTION.md, from every published tile's buildings.json.

Two files: how far apart the LiDAR and the photos were flown, per pairing of
flights, with the textured buildings and the LiDAR-height boxes apart; and the LiDAR
density of the tile each building sits in. Both read the provenance that
every building in a published tile carries (lidar_date, oblique_date,
gap_months, lidar_points_per_m2), textured or not. A building is textured
when the store holds its buildings/<gers>/3dtiles_mesh.zip.

Usage::

    aws s3 cp s3://kyfromabove-3d/tiles/ tiles/ --recursive --exclude '*' --include '*/buildings.json'
    aws s3 ls s3://kyfromabove-3d/buildings/ --recursive > buildings_listing.txt
    python scripts/plot_collection_findings.py --tiles tiles --meshes buildings_listing.txt --out docs/figures
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiles", type=Path, required=True, help="folder of <tile>/buildings.json")
    ap.add_argument("--meshes", type=Path, required=True, help="`aws s3 ls --recursive` of the store's buildings/")
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meshed = {line.split()[3].split("/")[1] for line in args.meshes.read_text(encoding="utf-8").splitlines()
              if line.rstrip().endswith("/3dtiles_mesh.zip")}
    rows = []
    for p in sorted(args.tiles.glob("*/buildings.json")):
        for b in json.loads(p.read_text(encoding="utf-8"))["buildings"]:
            rows.append((b["lidar_date"], b["oblique_date"], b["gap_months"], b["lidar_points_per_m2"],
                         b["id"] in meshed))

    def pairing(r):
        return r[0][:7], r[1][:7], round(r[2])

    textured = collections.Counter(pairing(r) for r in rows if r[4])
    boxes = collections.Counter(pairing(r) for r in rows if not r[4])
    keys = sorted(set(textured) | set(boxes), key=lambda k: k[2])
    labels = []
    for ld, od, gap in keys:
        when = "LiDAR after photos" if gap < 0 else ("same month" if gap == 0 else "photos after LiDAR")
        labels.append(f"LiDAR {ld}\nphotos {od}\n{abs(gap)} months apart\n({when})")
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    w = 0.38
    for off, counts, name, color in ((-w / 2, textured, "textured buildings", "#e76f51"),
                                     (w / 2, boxes, "LiDAR-height boxes only", "#b8c4cf")):
        total = sum(counts.values())
        vals = [counts.get(k, 0) for k in keys]
        bars = ax.bar([i + off for i in range(len(keys))], vals, w, color=color,
                      label=f"{name}: {total:,}")
        for b, n in zip(bars, vals):
            if n:
                ax.text(b.get_x() + b.get_width() / 2, n + 1500, f"{n:,}\n{100 * n / total:.0f}%",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(keys)), labels, fontsize=9)
    ax.set_ylabel("buildings")
    ax.set_ylim(0, max(max(textured.values()), max(boxes.values())) * 1.25)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title(f"How far apart the LiDAR and the photos were flown, for the {len(rows):,} buildings "
                 "in the published tiles", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "collection_lidar_photo_gap.png")

    # the density is the tile's, from its LAS header, so buildings in one tile share a value
    dens = sorted(r[3] for r in rows)
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    ax.hist([min(x, 10) for x in dens], bins=40, range=(0, 10), color="#457b9d")
    med = dens[len(dens) // 2]
    ax.axvline(med, color="#1d3557", lw=1.2, ls="--")
    ax.text(med + 0.15, ax.get_ylim()[1] * 0.9,
            f"half the buildings sit in tiles with fewer than {med:.1f} points per m²\n"
            f"(about one point every {100 / med ** 0.5:.0f} cm, seen from above)", fontsize=9, color="#1d3557")
    ax.set_xlabel("LiDAR points per square meter in the tile each building sits in")
    ax.set_ylabel("buildings")
    ax.set_title("How densely the LiDAR samples the ground and roofs, per building", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "collection_lidar_density.png")
    print(f"{len(rows):,} buildings; textured {dict(textured)}; boxes {dict(boxes)}; median density {med:.2f} points/m²")


if __name__ == "__main__":
    main()
