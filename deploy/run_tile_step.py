#!/usr/bin/env python3
"""Run step 1 in AWS: one Lambda per LiDAR tile, all at once.

Invokes ``kyfromabove-3d-tile-step`` for each tile in parallel and waits
for them; prints each tile's result and the Lambda's own report of its
duration and peak memory (the last lines of its log, returned with the
invocation).

Usage::

    python deploy/run_tile_step.py --tiles N076E228 N076E229 N077E228 N077E229
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor


def invoke(client, fn: str, tile: str) -> dict:
    r = client.invoke(FunctionName=fn, InvocationType="RequestResponse", LogType="Tail",
                      Payload=json.dumps({"tile": tile}).encode())
    body = json.loads(r["Payload"].read() or b"null")
    tail = base64.b64decode(r.get("LogResult", "")).decode("utf-8", "replace")
    rep = re.search(r"Duration: ([\d.]+) ms.*?Max Memory Used: (\d+) MB", tail)
    init = re.search(r"Init Duration: ([\d.]+) ms", tail)
    return {"tile": tile, "error": r.get("FunctionError"), "result": body,
            "seconds": round(float(rep.group(1)) / 1000, 1) if rep else None,
            "peak_mb": int(rep.group(2)) if rep else None,
            "cold_start_s": round(float(init.group(1)) / 1000, 1) if init else None}


def main() -> int:
    import boto3
    from botocore.config import Config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", nargs="+", required=True)
    ap.add_argument("--function", default="kyfromabove-3d-tile-step")
    ap.add_argument("--region", default="us-west-2")
    args = ap.parse_args()
    client = boto3.client("lambda", region_name=args.region,
                          config=Config(read_timeout=960, connect_timeout=10, retries={"max_attempts": 0}))
    with ThreadPoolExecutor(max_workers=len(args.tiles)) as pool:
        results = list(pool.map(lambda t: invoke(client, args.function, t), args.tiles))
    bad = 0
    for r in results:
        if r["error"]:
            bad += 1
            print(f"{r['tile']}: FAILED ({r['error']}): {json.dumps(r['result'])[:500]}")
            continue
        s = r["result"]
        print(f"{r['tile']}: {s['buildings']} buildings ({s['split']} split), {s['measured']} measured, "
              f"{s['frames']} frames, {s['returns']:,} returns; Lambda {r['seconds']} s"
              + (f" (+{r['cold_start_s']} s start-up)" if r["cold_start_s"] else "")
              + f", peak {r['peak_mb']} MB")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
