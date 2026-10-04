#!/usr/bin/env python3
"""Run step 2 in AWS: every tile's buildings in batches, one Lambda per
batch, all at once.

Reads each tile's ``tiles/<name>/buildings.json`` from the bucket, cuts its
buildings into batches of nearby ones, invokes ``kyfromabove-3d-piece-step``
once per batch in parallel, and prints per tile what was written and what
the Lambdas reported (their duration, peak memory, start-up), with the
seconds a building -- the numbers the batch size is set from.

Usage::

    python deploy/run_piece_step.py --tiles N076E228 N076E229 N077E228 N077E229 --batch 25
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def invoke(client, fn: str, tile: str, gers: list) -> dict:
    t = time.time()
    r = client.invoke(FunctionName=fn, InvocationType="RequestResponse", LogType="Tail",
                      Payload=json.dumps({"tile": tile, "gers": gers}).encode())
    body = json.loads(r["Payload"].read() or b"null")
    tail = base64.b64decode(r.get("LogResult", "")).decode("utf-8", "replace")
    rep = re.search(r"Duration: ([\d.]+) ms.*?Max Memory Used: (\d+) MB", tail)
    init = re.search(r"Init Duration: ([\d.]+) ms", tail)
    return {"tile": tile, "n": len(gers), "error": r.get("FunctionError"), "result": body,
            "seconds": float(rep.group(1)) / 1000 if rep else None, "peak_mb": int(rep.group(2)) if rep else None,
            "init_s": float(init.group(1)) / 1000 if init else 0.0, "wall_s": time.time() - t}


def main() -> int:
    import boto3
    from botocore.config import Config

    from kyfromabove_3d.pieces import batches
    from kyfromabove_3d.store import S3Store

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--per-tile", type=int, default=20,
                    help="at most this many batches of one tile at once: they all read the tile's one LiDAR file")
    ap.add_argument("--bucket", default="kyfromabove-3d")
    ap.add_argument("--function", default="kyfromabove-3d-piece-step")
    ap.add_argument("--region", default="us-west-2")
    args = ap.parse_args()
    store = S3Store(args.bucket, region=args.region)
    jobs = []
    for t in args.tiles:
        rec = store.read_json(f"tiles/{t}/buildings.json")
        jobs += [(t, b) for b in batches(rec["buildings"], args.batch)]
    client = boto3.client("lambda", region_name=args.region,
                          config=Config(read_timeout=960, connect_timeout=10, retries={"max_attempts": 0},
                                        max_pool_connections=min(len(jobs), 200)))
    print(f"{len(jobs)} batches of up to {args.batch} buildings over {len(args.tiles)} tile(s): "
          f"invoking at once, at most {args.per_tile} per tile")
    import threading
    gates = {t: threading.Semaphore(args.per_tile) for t in args.tiles}

    def run(job):
        with gates[job[0]]:
            return invoke(client, args.function, *job)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(len(jobs), 200)) as pool:
        res = list(pool.map(run, jobs))
    wall = time.time() - t0
    bad = [r for r in res if r["error"] or (r["result"] or {}).get("failed")]
    for t in args.tiles:
        rs = [r for r in res if r["tile"] == t and not r["error"]]
        pieces = sum(r["result"]["pieces"] for r in rs)
        print(f"{t}: {len(rs)} batches, {pieces} pieces, {sum(r['result']['written'] for r in rs)} written, "
              f"{sum(r['result']['unchanged'] for r in rs)} unchanged, "
              f"{sum(len(r['result']['failed']) for r in rs)} failed")
    ok = [r for r in res if not r["error"]]
    if ok:
        per_b = [r["result"]["seconds_per_building"] for r in ok]
        setup = [r["result"]["seconds_setup"] for r in ok]
        print(f"Lambda: median {statistics.median([r['seconds'] for r in ok]):.1f} s a batch, longest "
              f"{max(r['seconds'] for r in ok):.1f} s; {statistics.median(per_b):.2f} s a building (median); "
              f"set-up median {statistics.median(setup):.1f} s, longest {max(setup):.1f} s; start-up (init) "
              f"median {statistics.median([r['init_s'] for r in ok]):.1f} s; peak memory "
              f"{max(r['peak_mb'] or 0 for r in ok)} MB; GB-seconds {sum(r['seconds'] for r in ok) * 2.048:,.0f}")
    print(f"all {len(jobs)} batches done in {wall:.0f} s of wall-clock time")
    for r in bad[:10]:
        print(f"  problem in {r['tile']}: {r['error'] or ''} {json.dumps(r['result'])[:400]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
