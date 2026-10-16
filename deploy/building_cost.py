#!/usr/bin/env python3
"""What step 3, the building step, costs: for downtown's 2x2 block and for
Kentucky, stage by stage.

Measured, not assumed, where it can be:

* **seconds and peak memory of each stage, per building** -- a sample of
  buildings built on the laptop by the old ``scripts/building_step.py``
  (removed 2026-10-16; its results files remain), drawn evenly from five footprint-size bands so the
  towers are measured as well as the houses (``--results``);
* **how many buildings of each size** -- the 2x2 block's 2,361 from the tile
  step, and Kentucky's 2.79 million from Overture's outlines over the Phase 2
  grid (``deploy/ky_area_hist.json``: their bounding boxes, scaled to
  footprint area by the block's own ratio);
* **the size of what is kept** per building, from the sample.

Not yet measured: how much slower a 3 GB ARM Lambda (about two vCPUs) is than
the laptop. Steps 1 and 2 bracket it: the piece step ran at the laptop's
speed (network-bound), the tile step about 2.4 times slower (CPU-bound). The
building step is mostly CPU-bound, so both ends are priced; the calibration
run (a few buildings in AWS) replaces the bracket with one number.

Prices: AWS list prices, us-west-2, September 2026 (``cost_model.PRICE``).

Usage::

    python deploy/building_cost.py --results outputs/step3_sample.jsonl \\
        --tiles-dir outputs/step1_local_v3/tiles --figure docs/figures/building_step_cost.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cost_model import PRICE                                                  # noqa: E402

#: footprint area bands (m²), the ones the sample was drawn from
BANDS = (0, 150, 400, 1500, 5000, float("inf"))
BAND_NAMES = ("under 150 m²", "150-400 m²", "400-1,500 m²", "1,500-5,000 m²", "over 5,000 m²")
#: the stages, in the order they run, and what each is called on the chart
STAGES = {"crops": "cutting the frame crops", "scene": "reading the LiDAR around it",
          "colourise": "colouring the LiDAR from the frames", "mesh": "building the mesh",
          "complete": "the completed product", "check": "the texture check", "audit": "the audit"}
MEMORY_GB = 3008 / 1024
LAMBDA_LIMIT_S = 900.0
CONCURRENCY = 890                  # this account's unreserved Lambda concurrency


def load_sample(path: Path):
    """One row per building built: area, height, seconds per stage, peak MB, bytes kept, files kept."""
    rows = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(ln)
        if not r.get("built"):
            continue
        b = r["built"][0]
        sec = dict(b["seconds"])
        sec["scene"] = sec.pop("batch_scene_share", 0.0)
        rows.append({"gers": r["gers"], "area_m2": r.get("area_m2"), "height_m": r.get("height_m"),
                     "seconds": {k: float(sec.get(k, 0.0)) for k in STAGES},
                     "peak_mb": max((b.get("peak_mb") or {}).values() or [0.0]),
                     "bytes": b.get("bytes", 0),
                     # uploads: the objects stored (zips and all) where recorded, else each file
                     "files": len(b.get("objects") or b.get("files", {})) + 1})
    return rows


def band_of(area):
    return int(np.searchsorted(BANDS, area, side="right") - 1)


def per_band(rows):
    """Per size band: the mean seconds of each stage, mean bytes and files, the largest peak memory."""
    out = []
    for i in range(len(BAND_NAMES)):
        rs = [r for r in rows if band_of(r["area_m2"]) == i]
        if not rs:
            out.append(None)
            continue
        out.append({"n": len(rs),
                    "seconds": {k: float(np.mean([r["seconds"][k] for r in rs])) for k in STAGES},
                    "seconds_max": float(max(sum(r["seconds"].values()) for r in rs)),
                    "bytes": float(np.mean([r["bytes"] for r in rs])),
                    "files": float(np.mean([r["files"] for r in rs])),
                    "peak_mb": float(max(r["peak_mb"] for r in rs))})
    return out


def block_counts(tiles_dir: Path):
    """The 2x2 block's buildings per size band, and each one's bounding-box area over its footprint area."""
    from pyproj import Geod
    from shapely.wkt import loads

    geod = Geod(ellps="WGS84")
    counts, ratio = np.zeros(len(BAND_NAMES)), []
    for f in sorted(Path(tiles_dir).glob("*/buildings.json")):
        for r in json.loads(f.read_text(encoding="utf-8"))["buildings"]:
            g = loads(r["wkt"])
            a = abs(geod.geometry_area_perimeter(g)[0])
            counts[band_of(a)] += 1
            x0, y0, x1, y1 = g.bounds
            lat = np.radians((y0 + y1) / 2)
            ratio.append(((x1 - x0) * 111320 * np.cos(lat) * (y1 - y0) * 110574) / max(a, 1.0))
    return counts, float(np.median(ratio))


def state_counts(hist_path: Path, bbox_ratio: float):
    """Kentucky's buildings per size band, from the bounding-box histogram."""
    h = json.loads(Path(hist_path).read_text(encoding="utf-8"))
    edges, counts = np.asarray(h["edges_m2"]), np.asarray(h["counts"], dtype=float)
    mids = np.sqrt(edges[:-1] * edges[1:]) / bbox_ratio
    out = np.zeros(len(BAND_NAMES))
    for m, c in zip(mids, counts):
        out[band_of(m)] += c
    return out


def cost(counts, bands, factor: float, batch: int = 5):
    """One run over ``counts`` buildings per band, the laptop's seconds times
    ``factor`` (the crops, which wait on the network, are not scaled).
    Returns dollars per stage and the totals."""
    per_stage = {k: 0.0 for k in STAGES}
    n = float(counts.sum())
    puts = kept = over = 0.0
    for c, b in zip(counts, bands):
        if b is None or not c:
            continue
        for k in STAGES:
            s = b["seconds"][k] * (1.0 if k == "crops" else factor)
            per_stage[k] += c * s
        puts += c * b["files"]
        kept += c * b["bytes"]
        over += c if b["seconds_max"] * factor > LAMBDA_LIMIT_S else 0.0
    seconds = sum(per_stage.values())
    dollars = {STAGES[k]: v * MEMORY_GB * PRICE["lambda_gb_s_arm"] for k, v in per_stage.items()}
    dollars["writing the results to S3"] = puts * PRICE["s3_put"]
    dollars["reading the pieces"] = n * 10 * PRICE["s3_get"]
    dollars["Lambda requests"] = n / batch * PRICE["lambda_request"]
    dollars["CloudWatch logs"] = n * 4e3 / 1e9 * PRICE["logs_gb"]
    return {"buildings": n, "lambda_hours": seconds / 3600, "dollars": dollars, "total": sum(dollars.values()),
            "storage_gb": kept / 1e9, "storage_per_month": kept / 1e9 * PRICE["s3_gb_month"],
            "wall_clock_hours": seconds / CONCURRENCY / 3600, "at_risk_of_15_minutes": over}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="the old scripts/building_step.py --results lines")
    ap.add_argument("--tiles-dir", type=Path, required=True, help="the 2x2 block's tiles/<name>/buildings.json")
    ap.add_argument("--state-hist", type=Path, default=Path(__file__).with_name("ky_area_hist.json"))
    ap.add_argument("--contention", type=float, nargs=2, default=(1.0, 1.0), metavar=("COMPUTING", "CROPS"),
                    help="the sample's laptop seconds over the same buildings built one at a time (the sample "
                         "ran four at once): for the computing stages, and for the crops (which wait on the network)")
    ap.add_argument("--factors", type=float, nargs=2, default=(1.0, 2.4),
                    help="Lambda's seconds over the laptop's: the fast and the slow end")
    ap.add_argument("--figure", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    rows = load_sample(args.results)
    for r in rows:
        r["seconds"] = {k: v / (args.contention[1] if k == "crops" else args.contention[0])
                        for k, v in r["seconds"].items()}
    bands = per_band(rows)
    block, ratio = block_counts(args.tiles_dir)
    state = state_counts(args.state_hist, ratio)
    print(f"{len(rows)} buildings measured; bounding box / footprint area on the block: {ratio:.2f}")
    for name, b, nb, ns in zip(BAND_NAMES, bands, block, state):
        if b:
            print(f"  {name:16s} {b['n']:2d} measured, {sum(b['seconds'].values()):6.1f} s a building "
                  f"(slowest {b['seconds_max']:.0f} s), peak {b['peak_mb']:.0f} MB, {b['bytes'] / 1e6:.2f} MB kept; "
                  f"block {int(nb):5d}, state {int(ns):9,d}")
    res = {}
    for where, counts in (("block", block), ("state", state)):
        for f in args.factors:
            r = cost(counts, bands, f)
            res[f"{where} x{f:g}"] = r
            print(f"{where:5s} at {f:g}x the laptop: ${r['total']:,.2f} a run, {r['lambda_hours']:,.0f} Lambda-hours, "
                  f"{r['wall_clock_hours']:.1f} h at {CONCURRENCY} at once; keeps {r['storage_gb']:,.1f} GB "
                  f"(${r['storage_per_month']:,.2f}/month); {int(r['at_risk_of_15_minutes'])} at risk of 15 min")
    if args.json:
        args.json.write_text(json.dumps({"bands": dict(zip(BAND_NAMES, bands)), "results": res,
                                         "block_counts": block.tolist(), "state_counts": state.tolist(),
                                         "bbox_ratio": ratio, "price": PRICE}, indent=1, default=float))
    if args.figure:
        figure(rows, res, args.factors, args.figure)
    return 0


def figure(rows, res, factors, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#2a78d6", "#8b5cf6", "#eb6834", "#1baf7a", "#e3a008", "#898781", "#c3c2b7"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})

    # left: seconds per building against its size, each stage stacked
    rows = sorted(rows, key=lambda r: r["area_m2"])
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for (k, label), col in zip(STAGES.items(), colours):
        v = np.array([r["seconds"][k] for r in rows])
        a1.bar(x, v, bottom=bottom, color=col, width=0.8, label=label, edgecolor="white", linewidth=0.6)
        bottom += v
    step = max(1, len(rows) // 8)
    a1.set_xticks(x[::step], [f"{r['area_m2']:,.0f}" for r in rows[::step]], fontsize=8)
    a1.set_xlabel("each measured building, smallest footprint to largest (m²)")
    a1.set_ylabel("seconds on the laptop, one building at a time")
    a1.set_title("Where one building's time goes", fontsize=11)
    a1.legend(frameon=False, fontsize=8, loc="upper left")

    # right: dollars for one statewide run, per stage, at both ends of the speed bracket
    lo, hi = res[f"state x{factors[0]:g}"], res[f"state x{factors[1]:g}"]
    parts = [p for p in lo["dollars"] if max(lo["dollars"][p], hi["dollars"][p]) >= 1.0]
    parts = sorted(parts, key=lambda p: hi["dollars"][p])
    y = np.arange(len(parts))
    a2.barh(y - 0.2, [lo["dollars"][p] for p in parts], height=0.38, color="#86b6ef",
            label=f"if Lambda runs at the laptop's speed: ${lo['total']:,.0f}")
    a2.barh(y + 0.2, [hi["dollars"][p] for p in parts], height=0.38, color="#2a78d6",
            label=f"if {factors[1]:g} times slower (as the tile step was): ${hi['total']:,.0f}")
    for yy, p in zip(y, parts):
        a2.text(hi["dollars"][p] * 1.01 + 5, yy + 0.2, f"${hi['dollars'][p]:,.0f}", va="center", fontsize=8)
    a2.set_yticks(y, parts, fontsize=9)
    a2.set_xlabel("dollars for one run over Kentucky's 2.79 million buildings")
    a2.set_title("One statewide run, by stage", fontsize=11)
    a2.legend(frameon=False, fontsize=9, loc="lower right")
    for ax in (a1, a2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    print(f"-> {path}")


if __name__ == "__main__":
    raise SystemExit(main())
