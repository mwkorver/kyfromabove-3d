#!/usr/bin/env python3
"""Step 3, the building step, on this machine: what the Lambda
``kyfromabove-3d-building-step`` does, for measuring it before it runs in
AWS and checking it after.

Each building: its piece from ``pieces/<gers>/``, the frame crops, the
colourise, the mesh and the completed product, the texture check and the
audit; the results to ``buildings/<version>/<gers>/`` of ``--store`` (five objects) with
``building.json`` last (the seconds and peak memory of every stage, the size
of every file). One line per building also goes to ``--results``.

Buildings are named (``--gers``) or sampled from tiles the tile step has
measured (``--tiles ... --sample 40``): the same number from each footprint
size band, so the towers are measured as well as the houses.

Usage::

    python scripts/building_step.py --inputs s3://kyfromabove-3d --store outputs/step3_local \\
        --tiles N076E228 N076E229 N077E228 N077E229 --sample 40 --workers 4 --results step3.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: footprint area bands (m², the outline's own area) the sample is drawn from
AREA_BANDS = (0, 150, 400, 1500, 5000, float("inf"))


def footprint_area_m2(wkt: str) -> float:
    from pyproj import Geod
    from shapely.wkt import loads

    return abs(Geod(ellps="WGS84").geometry_area_perimeter(loads(wkt))[0])


def sample(records, n: int, seed: int):
    """``n`` buildings, the same number from each area band (fewer when a
    band runs short, the rest from the others)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    bands = [[] for _ in AREA_BANDS[:-1]]
    for r in records:
        a = footprint_area_m2(r["wkt"])
        for i in range(len(bands)):
            if AREA_BANDS[i] <= a < AREA_BANDS[i + 1]:
                bands[i].append(r["id"])
    per = int(np.ceil(n / len(bands)))
    out = []
    for b in bands:
        out += list(rng.choice(b, size=min(per, len(b)), replace=False))
    rest = [g for b in bands for g in b if g not in set(out)]
    if len(out) < n and rest:
        out += list(rng.choice(rest, size=min(n - len(out), len(rest)), replace=False))
    return sorted(out[:n])


def one(gers: str, inputs: str, store: str, raw: str, work: str, keep: bool, force: bool) -> dict:
    """One building in this worker process."""
    from kyfromabove_3d.building import building_step
    from kyfromabove_3d.sources_choice import indexes
    from kyfromabove_3d.store import store_from

    t = time.time()
    res = building_step([gers], store_from(inputs), Path(work), indexes(Path(raw)),
                        out_store=store_from(store), keep=keep, force=force, log=lambda m: None)
    res["gers"], res["wall_s"] = gers, round(time.time() - t, 1)
    return res


def main() -> int:
    from kyfromabove_3d.store import store_from

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True, help="where the pieces and the tile step's results are")
    ap.add_argument("--store", required=True, help="where to write buildings/<version>/<gers>/: a folder, or s3://bucket")
    ap.add_argument("--gers", nargs="*", default=[])
    ap.add_argument("--gers-file", type=Path, default=None, help="more buildings, whitespace-separated")
    ap.add_argument("--tiles", nargs="*", default=[])
    ap.add_argument("--sample", type=int, default=0, help="buildings to sample from --tiles (0: all of them)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2,
                    help="numeric threads per building (a 3 GB Lambda has about two vCPUs)")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--work", type=Path, default=Path("outputs/step3_work"))
    ap.add_argument("--results", type=Path, default=None, help="one JSON line per building")
    ap.add_argument("--keep", action="store_true", help="keep each building's working folder (crops and all)")
    ap.add_argument("--force", action="store_true", help="rebuild even when built from the same piece")
    args = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(args.threads)

    if not (args.raw / "kyfromabove_phase2_pointcloud_5k_grid.gpkg").exists():
        raise SystemExit(f"no Phase 2 LiDAR grid in {args.raw} (in the image it is /opt/data/raw)")
    inputs = store_from(args.inputs)
    gers = list(args.gers) + (args.gers_file.read_text().split() if args.gers_file else [])
    info = {}
    for t in args.tiles:
        recs = inputs.read_json(f"tiles/{t}/buildings.json")["buildings"]
        for r in recs:
            info[r["id"]] = {"tile": t, "area_m2": round(footprint_area_m2(r["wkt"]), 1),
                             "height_m": r.get("lidar_height_m")}
        if not args.sample and not gers and not args.gers_file:
            gers += [r["id"] for r in recs]
    if args.sample:
        recs = [r for t in args.tiles for r in inputs.read_json(f"tiles/{t}/buildings.json")["buildings"]]
        gers += sample(recs, args.sample, args.seed)
    print(f"{len(gers)} building(s), {args.workers} at a time")
    results = open(args.results, "a", encoding="utf-8") if args.results else None
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, g, args.inputs, args.store, str(args.raw), str(args.work / g), args.keep, args.force): g
                for g in gers}
        for k, f in enumerate(as_completed(futs), 1):
            g = futs[f]
            try:
                res = f.result()
            except Exception as exc:                                # noqa: BLE001
                res = {"gers": g, "failed": {g: repr(exc)[:500]}, "built": [], "skipped": []}
            res.update(info.get(g, {}))
            b = res["built"][0] if res.get("built") else {}
            sec = b.get("seconds", {})
            state = "built" if b else ("skipped" if res.get("skipped") else "FAILED")
            print(f"[{k}/{len(gers)}] {g[:13]} {state:7s} {res.get('area_m2', 0):8.0f} m² "
                  f"{(res.get('height_m') or 0):5.1f} m  {res.get('wall_s', 0):6.1f} s  "
                  + " ".join(f"{s} {v:.0f}" for s, v in sec.items())
                  + (f"  {list(res['failed'].values())[0][:120]}" if res.get("failed") else ""), flush=True)
            if results:
                results.write(json.dumps(res, default=str) + "\n")
                results.flush()
    print(f"{len(gers)} building(s) in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
