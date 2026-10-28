#!/usr/bin/env python3
"""What one run of every step costs for all of Kentucky, from the two runs measured on the queue.

The inputs, each measured:

* **the work, tile by tile** (``survey_statewide_dates.py``'s ``pairing.pkl``):
  every LiDAR tile, its buildings, and whether any oblique collection covers it
  (a tile with none gets boxes and no build jobs);
* **the time of each kind of job** (``queue_<run>.json``, the job records of
  trial1 and cities1): the tile, measure and publish jobs as a time per tile
  plus a time per building, fitted over the 58 tiles; the build job as a time
  per building by footprint size (<100, 100-300, 300-1,000 and >=1,000 m2),
  fitted the same way and applied to the state's own mix of sizes; the tree
  job, 2.1 s a tile;
* **what a job-hour costs**: cities1 ran 149 job-hours on 13.7 machine-hours
  of c7g.4xlarge Spot (10.9 at once, idle and boots included; 13.4 while
  building), at $0.284 an hour; disk and addresses added 4% to that;
* **S3**: cities1's 9.3 uploads and lists and 12.4 reads a building, its
  0.53 MB stored a building.

Each figure is a low, a middle and a high: the middle the measured rate, the
low and high the spread the runs themselves show. Writes the numbers to
``--out`` and prints them.

Usage::

    python scripts/estimate_statewide_cost.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = [0, 100, 300, 1000, 1e12]
BAND_NAMES = ["<100", "100-300", "300-1k", ">=1k"]
PUT_PER_1000 = 0.005
GET_PER_1000 = 0.0004
GB_MONTH = 0.023


def fits(q: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Seconds per tile and per building for each kind of job, from the runs."""
    from scipy.optimize import nnls
    out = {}
    # fitted on the tiles under 500 buildings: Kentucky's tile is one of them
    # (median 28 buildings, 89% under 100), and over all 58 a city tile's
    # minutes set the line's start at 95 s a tile for a measure job that takes
    # 15-65 s on a tile of 20-60 buildings
    small = q[q["n"] < 500]
    for k in ("tile_s", "measure_s", "publish_s"):
        coef, _ = nnls(np.column_stack([np.ones(len(small)), small["n"]]), small[k].values)
        out[k] = {"per_tile": float(coef[0]), "per_building": float(coef[1])}
    b = b.assign(band=pd.cut(b.area_m2, BANDS, labels=BAND_NAMES, right=False))
    cnt = b.groupby(["tile", "band"], observed=False).size().unstack(fill_value=0)
    d = q.merge(cnt, left_on="tile", right_index=True, how="left").fillna(0)
    coef, _ = nnls(d[BAND_NAMES].values.astype(float), d["build_s"].values)
    out["build_s"] = dict(zip(BAND_NAMES, map(float, coef)))
    per = d["build_s"] / d["built"].clip(lower=1)
    out["build_spread"] = [float(x) for x in np.percentile(per[d["built"] > 100], [10, 50, 90])]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("outputs/survey/statewide"))
    ap.add_argument("--out", type=Path, default=Path("docs/data/statewide_cost.json"))
    ap.add_argument("--machines", type=int, default=8, help="at once (128 vCPU Spot quota / 16)")
    ap.add_argument("--spot", type=float, nargs=3, default=(0.22, 0.284, 0.29),
                    help="c7g.4xlarge Spot $/h: low, middle (cities1 paid), high (us-west-2, October 2026)")
    args = ap.parse_args()

    p = pd.read_pickle(args.dir / "pairing.pkl")
    q = pd.read_csv(args.dir / "queue_per_tile.csv")
    b = pd.read_pickle(args.dir / "buildings.pkl")
    f = fits(q, b)

    work = p[p.buildings > 0]
    n_b = float(work.buildings.sum())
    n_t = len(work)
    built = work[work.collection.notna()]
    n_built = float(built.buildings.sum())
    mix = (b[b.tile.isin(built.tile)].assign(band=lambda x: pd.cut(x.area_m2, BANDS, labels=BAND_NAMES, right=False))
           .band.value_counts(normalize=True).reindex(BAND_NAMES).values)
    build_s = float(np.dot(mix, [f["build_s"][k] for k in BAND_NAMES]))
    lo_b, _, hi_b = f["build_spread"]

    def job_hours(build_per_bldg: float) -> dict:
        h = {
            "tile": n_t * f["tile_s"]["per_tile"] + n_b * f["tile_s"]["per_building"],
            "measure": n_t * f["measure_s"]["per_tile"] + n_b * f["measure_s"]["per_building"],
            "build": n_built * build_per_bldg * 1.02,          # walls painted whole since cities1: +5-30% on towers only
            "publish": n_t * f["publish_s"]["per_tile"] + n_b * f["publish_s"]["per_building"],
            "tree": len(p) * 2.1,
        }
        return {k: v / 3600.0 for k, v in h.items()}

    cases = {}
    for name, bps, per_machine, price in (("low", lo_b, 13.4, args.spot[0]),
                                          ("middle", build_s, 10.9, args.spot[1]),
                                          ("high", hi_b, 10.9, args.spot[2])):
        jh = job_hours(bps)
        mh = sum(jh.values()) / per_machine
        machines = mh * price * 1.04
        puts = n_b * (9.3 if name != "high" else 16.8) * PUT_PER_1000 / 1000
        gets = n_b * 12.4 * GET_PER_1000 / 1000
        cases[name] = {"build_s_per_building": round(bps, 2), "job_hours": {k: round(v) for k, v in jh.items()},
                       "machine_hours": round(mh), "days_on_machines": round(mh / args.machines / 24, 1),
                       "usd": {"machines": round(machines), "s3_uploads": round(puts), "s3_reads": round(gets),
                               "total": round(machines + puts + gets)}}
    res = {"tiles_with_buildings": n_t, "buildings": int(n_b), "buildings_built": int(n_built),
           "buildings_boxes_only": int(n_b - n_built), "storage_tb": round(n_b * 0.53 / 1e6, 2),
           "storage_usd_month": round(n_b * 0.53 / 1000 * GB_MONTH), "fits": f, "cases": cases}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
