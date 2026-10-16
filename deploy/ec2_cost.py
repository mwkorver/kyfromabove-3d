#!/usr/bin/env python3
"""What one run of step 3 costs on Graviton EC2 Spot, beside Lambda.

Lambda is priced from the calibration: the billed seconds of real buildings
in AWS, per footprint-size band (the results the old
``deploy/run_building_step.py`` wrote, removed in cleanup 2026-10-16).

EC2 Spot is priced from processor-seconds per building and the machine's
hourly Spot price. Until a Spot machine has been measured, processor-seconds
are bracketed:

* **slow end** -- the Lambda's seconds times its 1.7 processors: Graviton2,
  and as if both processors were busy the whole time;
* **fast end** -- the laptop's seconds for one building at a time, on one
  processor. The laptop's cores are faster than any Graviton, so this is
  the floor.

A measured machine (``--measured``: a bench run's ``--results`` lines, with
its ``.meta.json`` beside them -- how many ran at once and how long the whole
run took, from the old ``deploy/run_ec2_bench.py``, removed 2026-10-16) replaces the bracket: each
building's share of the machine is its seconds over the number at once,
scaled so the shares add up to the run's length.

Overheads on EC2: ``--interrupted`` of the work redone after Spot
interruptions, ``--boot-min`` a machine to start and pull the image, gp3
disk, and the same S3 writes as Lambda.

Usage::

    python deploy/ec2_cost.py --calibration outputs/step3_calib.jsonl outputs/step3_calib_warm.jsonl \\
        --sample outputs/step3_sample.jsonl --tiles-dir outputs/step1_local_v3/tiles --contention 1.68 1.08
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from building_cost import (BAND_NAMES, BANDS, STAGES, band_of, block_counts, load_sample,  # noqa: E402
                           per_band, state_counts)
from cost_model import PRICE                                                  # noqa: E402

LAMBDA_MB = 3008
LAMBDA_VCPU = LAMBDA_MB / 1769.0          # Lambda gives one full vCPU at 1,769 MB
#: Objects uploaded a building: three zips, levels.json and building.json
#: (kyfromabove_3d.building.pack_building; about 20 single files before).
FILES_PER_BUILDING = 5
EBS_GB, EBS_GB_MONTH = 30, 0.08           # gp3


def lambda_seconds_per_band(paths, areas):
    """Mean billed Lambda seconds a building, per band, over the calibration
    runs; a band a run missed (a timeout) takes the other runs' mean."""
    per = [[] for _ in BAND_NAMES]
    for p in paths:
        for ln in Path(p).read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            if r.get("error") or not r.get("billed_s") or r["gers"] not in areas:
                continue
            per[band_of(areas[r["gers"]])].append(float(r["billed_s"]))
    return np.array([np.mean(p) if p else np.nan for p in per])


def lambda_cost(counts, secs):
    gb_s = float((counts * secs).sum()) * LAMBDA_MB / 1024
    lam = gb_s * PRICE["lambda_gb_s_arm"] + counts.sum() * PRICE["lambda_request"]
    return lam + s3_cost(counts), float((counts * secs).sum())


def s3_cost(counts):
    n = float(counts.sum())
    return n * FILES_PER_BUILDING * PRICE["s3_put"] + n * 10 * PRICE["s3_get"]


def ec2_cost(counts, cpu_s, price_h, vcpus=16, interrupted=0.05, boot_min=5.0, hours_per_machine=4.0):
    """One run on Spot machines of ``vcpus`` processors at ``price_h`` an
    hour, the work being ``cpu_s`` processor-seconds a building per band."""
    work_h = float((counts * cpu_s).sum()) / 3600.0 / vcpus * (1.0 + interrupted)
    machines = max(1.0, np.ceil(work_h / hours_per_machine))
    machine_h = work_h + machines * boot_min / 60.0
    ebs = machine_h * EBS_GB * EBS_GB_MONTH / 730.0
    return {"machine_hours": machine_h, "compute": machine_h * price_h, "disk": ebs, "s3": s3_cost(counts),
            "total": machine_h * price_h + ebs + s3_cost(counts), "processor_hours": machine_h * vcpus}


def measured(path: Path, areas):
    """(meta, machine-seconds a building per band) from one bench run."""
    meta = json.loads(Path(str(path).replace(".jsonl", ".meta.json")).read_text(encoding="utf-8"))
    rows = [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r.get("built") and r["gers"] in areas]
    share = np.array([r["wall_s"] / meta["workers"] for r in rows])
    share *= meta["seconds"] / share.sum()                   # the tail, when fewer than all ran at once
    per = [[] for _ in BAND_NAMES]
    for r, sh in zip(rows, share):
        per[band_of(areas[r["gers"]])].append(sh)
    return meta, np.array([np.mean(p) if p else np.nan for p in per])


def spot_price(instance: str, region: str = "us-west-2"):
    """(cheapest, median) current Spot price over the region's zones."""
    import datetime

    import boto3
    r = boto3.client("ec2", region_name=region).describe_spot_price_history(
        InstanceTypes=[instance], ProductDescriptions=["Linux/UNIX"],
        StartTime=datetime.datetime.now(datetime.timezone.utc))
    p = sorted(float(x["SpotPrice"]) for x in r["SpotPriceHistory"])
    return p[0], float(np.median(p))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", nargs="+", required=True)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--tiles-dir", type=Path, required=True)
    ap.add_argument("--state-hist", type=Path, default=Path(__file__).with_name("ky_area_hist.json"))
    ap.add_argument("--contention", type=float, nargs=2, default=(1.0, 1.0))
    ap.add_argument("--instance", default="c8g.4xlarge")
    ap.add_argument("--vcpus", type=int, default=16)
    ap.add_argument("--spot-price", type=float, default=None, help="$/hour (default: the live median)")
    ap.add_argument("--measured", type=Path, nargs="*", default=[],
                    help="bench runs' --results files (outputs/ec2_bench/<label>.jsonl, beside <label>.meta.json)")
    ap.add_argument("--interrupted", type=float, default=0.05)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--figure", type=Path, default=None)
    args = ap.parse_args()

    rows = load_sample(args.sample)
    areas = {r["gers"]: r["area_m2"] for r in rows}
    for r in rows:
        r["seconds"] = {k: v / (args.contention[1] if k == "crops" else args.contention[0])
                        for k, v in r["seconds"].items()}
    lap = np.array([sum(b["seconds"].values()) if b else np.nan for b in per_band(rows)])
    lam_s = lambda_seconds_per_band(args.calibration, areas)
    block, ratio = block_counts(args.tiles_dir)
    state = state_counts(args.state_hist, ratio)
    if args.spot_price is None:
        lo, med = spot_price(args.instance)
        price = med
        print(f"{args.instance} Spot now: ${lo:.4f}/h in the cheapest zone, ${med:.4f}/h median")
    else:
        price = args.spot_price

    ends = {"slow end (Lambda's speed, both processors busy)": lam_s * LAMBDA_VCPU,
            "fast end (the laptop's speed, one processor)": lap}
    if args.measured:
        ends = {f"measured, {meta['workers']} at once": machine_s * args.vcpus
                for meta, machine_s in (measured(p, areas) for p in args.measured)}
    out = {"spot_price_h": price, "instance": args.instance, "lambda_seconds": lam_s.tolist(),
           "laptop_seconds": lap.tolist(), "results": {}}
    print("seconds a building, by band:  Lambda " + str(np.round(lam_s, 0).tolist())
          + "   laptop " + str(np.round(lap, 0).tolist()))
    for where, counts in (("2x2 block", block), ("Kentucky", state)):
        lam, lam_s_total = lambda_cost(counts, lam_s)
        print(f"{where}: Lambda ${lam:,.2f}  ({lam_s_total / 3600:,.0f} Lambda-hours)")
        out["results"][where] = {"lambda": lam}
        for name, cpu_s in ends.items():
            e = ec2_cost(counts, cpu_s, price, args.vcpus, args.interrupted)
            out["results"][where][name] = e
            at32 = e["machine_hours"] / (32 // args.vcpus)
            at1000 = e["machine_hours"] / (1000 // args.vcpus)
            print(f"   EC2 Spot, {name}: ${e['total']:,.2f} ({e['machine_hours']:,.0f} machine-hours; "
                  f"{at32:,.0f} h at today's 32-processor quota, {at1000:,.1f} h at 1,000)")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, default=float))
    if args.figure:
        figure(out, args.figure)
    return 0


def figure(out, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = out["results"]["Kentucky"]
    names = ["Lambda (measured)"] + [k for k in r if k != "lambda"]
    vals = [r["lambda"]] + [r[k]["total"] for k in names[1:]]
    labels = ["Lambda, 3,008 MB\n(measured, 10 buildings)"] + [
        "EC2 Spot, " + k.replace(" (", "\n(") for k in names[1:]]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    y = np.arange(len(vals))
    ax.barh(y, vals, color=["#eb6834"] + ["#2a78d6"] * (len(vals) - 1), height=0.55)
    for yy, v in zip(y, vals):
        ax.text(v + max(vals) * 0.01, yy, f"${v:,.0f}", va="center", fontsize=10)
    ax.set_yticks(y, labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_xlabel("dollars for one run of the building step over Kentucky's 2.79 million buildings")
    ax.set_title(f"Lambda against Graviton EC2 Spot ({out['instance']} at ${out['spot_price_h']:.3f} an hour)",
                 fontsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    print(f"-> {path}")


if __name__ == "__main__":
    raise SystemExit(main())
