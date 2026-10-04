#!/usr/bin/env python3
"""Run step 3 in AWS: named buildings, one Lambda each, all at once.

Invokes ``kyfromabove-3d-building-step`` with one building per call, so
each Lambda's own report (its billed seconds and the most memory the whole
container used) belongs to one building, and writes one JSON line per
building to ``--results``: the stage seconds the building step measured,
the Lambda's report, and the memory size it ran at. Those are what
``deploy/building_cost.py --lambda`` compares with the laptop.

Usage::

    python deploy/run_building_step.py --gers <gers> ... --results calibration.jsonl [--force]
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def invoke(client, fn: str, gers: str, force: bool) -> dict:
    t = time.time()
    r = client.invoke(FunctionName=fn, InvocationType="RequestResponse", LogType="Tail",
                      Payload=json.dumps({"gers": [gers], "force": force}).encode())
    body = json.loads(r["Payload"].read() or b"null")
    tail = base64.b64decode(r.get("LogResult", "")).decode("utf-8", "replace")
    rep = re.search(r"Duration: ([\d.]+) ms.*?Billed Duration: (\d+) ms.*?Memory Size: (\d+) MB.*?"
                    r"Max Memory Used: (\d+) MB", tail)
    init = re.search(r"Init Duration: ([\d.]+) ms", tail)
    return {"gers": gers, "error": r.get("FunctionError"), "result": body,
            "seconds": float(rep.group(1)) / 1000 if rep else None,
            "billed_s": int(rep.group(2)) / 1000 if rep else None,
            "memory_mb": int(rep.group(3)) if rep else None, "peak_mb": int(rep.group(4)) if rep else None,
            "init_s": float(init.group(1)) / 1000 if init else 0.0, "wall_s": round(time.time() - t, 1)}


def main() -> int:
    import boto3
    from botocore.config import Config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gers", nargs="+", required=True)
    ap.add_argument("--force", action="store_true", help="rebuild buildings already built from the same piece")
    ap.add_argument("--results", required=True)
    ap.add_argument("--function", default="kyfromabove-3d-building-step")
    ap.add_argument("--region", default="us-west-2")
    args = ap.parse_args()
    client = boto3.client("lambda", region_name=args.region,
                          config=Config(read_timeout=960, connect_timeout=10, retries={"max_attempts": 0},
                                        max_pool_connections=len(args.gers)))
    print(f"{len(args.gers)} building(s), one Lambda each, all at once")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(args.gers)) as pool:
        res = list(pool.map(lambda g: invoke(client, args.function, g, args.force), args.gers))
    bad = 0
    with open(args.results, "a", encoding="utf-8") as fh:
        for r in res:
            fh.write(json.dumps(r) + "\n")
            body = r["result"] or {}
            b = (body.get("buildings") or [{}])[0]
            sec = b.get("seconds") or {}
            ok = not r["error"] and body.get("built") == 1
            bad += not ok
            print(f"{r['gers'][:13]} {'built' if ok else 'PROBLEM'}  {r['seconds'] or 0:6.1f} s at "
                  f"{r['memory_mb']} MB, peak {r['peak_mb']} MB, init {r['init_s']:.1f} s  "
                  + " ".join(f"{k} {v:.0f}" for k, v in sec.items())
                  + ("" if ok else f"  {r['error'] or ''} {json.dumps(body)[:300]}"))
    print(f"done in {time.time() - t0:.0f} s of wall-clock time")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
