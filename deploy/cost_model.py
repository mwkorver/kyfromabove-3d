#!/usr/bin/env python3
"""What stage 1 costs, statewide and for the downtown pilot.

Inputs are measured, not assumed, where they can be: buildings per LiDAR
tile (Overture over the Phase 2 grid), seconds per building and per tile
(the laptop and the Lambda image, downtown), peak memory, piece size and
files per piece. Prices are AWS list prices for us-west-2 (September 2026,
Lambda on ARM); check them before relying on the totals.

Rerun it with the pilot's measured seconds (``--s-per-building``,
``--s-per-tile``) once the pilot has run in AWS.

Usage::

    python deploy/cost_model.py --per-tile per_tile.npy --figure docs/figures/cloud_cost.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PRICE = {
    "lambda_gb_s_arm": 0.0000133334,    # $ per GB-second, arm64
    "lambda_request": 0.20 / 1e6,
    "s3_put": 0.005 / 1000,
    "s3_get": 0.0004 / 1000,
    "s3_gb_month": 0.023,
    "ddb_write_unit": 0.625 / 1e6,      # on-demand
    "sqs_request": 0.40 / 1e6,          # after the first million a month
    "logs_gb": 0.50,
    "eventbridge_aws_event": 0.0,       # events AWS services publish are free
}


def stage1_cost(per_tile: np.ndarray, s_per_building: float, s_per_tile: float, memory_gb: float,
                puts_per_piece: int, piece_mb: float, lambda_limit_s: float = 900.0,
                chunk_buildings: int = 800) -> dict:
    """One run of stage 1 over the given tiles (buildings per tile), and the
    month of storage after it. Tiles too big for one Lambda are split into
    chunks of ``chunk_buildings``, each paying the per-tile overhead."""
    n_b = float(per_tile.sum())
    chunks = np.where(s_per_tile + s_per_building * per_tile > lambda_limit_s,
                      np.ceil(per_tile / chunk_buildings), 1.0)
    invocations = float(chunks.sum())
    seconds = invocations * s_per_tile + n_b * s_per_building
    gb_s = seconds * memory_gb
    record_batches = n_b / 10.0
    c = {
        "Lambda, cutting pieces": gb_s * PRICE["lambda_gb_s_arm"] + invocations * PRICE["lambda_request"],
        "S3 writes (PUT)": (n_b * puts_per_piece + 3 * len(per_tile)) * PRICE["s3_put"],
        "Lambda, recording pieces": record_batches * 0.5 * 0.5 * PRICE["lambda_gb_s_arm"]
                                    + record_batches * PRICE["lambda_request"],
        "DynamoDB (row + by-tile index)": n_b * 2 * PRICE["ddb_write_unit"],
        "S3 reads (GET)": n_b * PRICE["s3_get"],
        "SQS": max(0.0, (3 * invocations + n_b + 2 * record_batches) - 1e6) * PRICE["sqs_request"],
        "CloudWatch logs": ((2e3 * invocations + 300 * (invocations + record_batches)) / 1e9) * PRICE["logs_gb"],
        "EventBridge": n_b * puts_per_piece * PRICE["eventbridge_aws_event"],
    }
    return {"buildings": n_b, "tiles": len(per_tile), "invocations": invocations, "hours_of_lambda": seconds / 3600,
            "gb_s": gb_s, "one_run": c, "one_run_total": sum(c.values()),
            "storage_gb": n_b * piece_mb / 1024, "storage_per_month": n_b * piece_mb / 1024 * PRICE["s3_gb_month"],
            "tiles_split": int((chunks > 1).sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-tile", type=Path, required=True, help="buildings per tile (.npy)")
    ap.add_argument("--s-per-building", type=float, default=0.8)
    ap.add_argument("--s-per-tile", type=float, default=30.0)
    ap.add_argument("--figure", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    per = np.load(args.per_tile)
    down = np.array([179, 72, 90, 37])                                     # the pilot: downtown's 4 tiles (owned)

    scenarios = {
        "as built now\n6 GB, 7 files a piece": dict(memory_gb=6.0, puts_per_piece=7),
        "right-sized\n2 GB, 7 files a piece": dict(memory_gb=2.0, puts_per_piece=7),
        "right-sized, bundled\n2 GB, 2 files a piece": dict(memory_gb=2.0, puts_per_piece=2),
        "bundled, twice as slow\n2 GB, 1.6 s a building": dict(memory_gb=2.0, puts_per_piece=2, slow=True),
    }
    out = {}
    for name, sc in scenarios.items():
        spb = args.s_per_building * (2 if sc.get("slow") else 1)
        out[name] = stage1_cost(per, spb, args.s_per_tile, sc["memory_gb"], sc["puts_per_piece"], piece_mb=0.15)
    pilot = stage1_cost(down, args.s_per_building, args.s_per_tile, 2.0, 2, piece_mb=0.25)
    for name, r in out.items():
        print(f"{name.replace(chr(10), ', '):48s} ${r['one_run_total']:8.2f} a run; "
              f"{r['hours_of_lambda']:,.0f} Lambda-hours; storage {r['storage_gb']:,.0f} GB, ${r['storage_per_month']:.2f}/month")
    print(f"pilot (downtown, 4 tiles, {int(down.sum())} buildings): ${pilot['one_run_total']:.3f}")
    if args.json:
        args.json.write_text(json.dumps({"scenarios": out, "pilot": pilot, "price": PRICE}, indent=1, default=float))

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        parts = list(next(iter(out.values()))["one_run"])
        colours = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6", "#e3a008", "#c3c2b7", "#898781", "#e34948"]
        fig, ax = plt.subplots(figsize=(11, 5.2))
        names = list(out)
        left = np.zeros(len(names))
        for part, col in zip(parts, colours):
            v = np.array([out[n]["one_run"][part] for n in names])
            if v.max() < 0.5:
                continue
            ax.barh(names, v, left=left, color=col, label=part, height=0.6, edgecolor="white", linewidth=2)
            left += v
        for y, n in enumerate(names):
            ax.text(left[y] + 4, y, f"${out[n]['one_run_total']:,.0f}", va="center", fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel("dollars for one run of stage 1 over Kentucky (2.79 million buildings, 41,566 LiDAR tiles)")
        ax.set_xlim(0, left.max() * 1.15)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title("Stage 1, statewide: what one run costs, by where the money goes "
                     f"(plus about ${out[names[2]]['storage_per_month']:.0f} a month to keep the pieces)", fontsize=10)
        fig.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=100)
        print(f"-> {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
