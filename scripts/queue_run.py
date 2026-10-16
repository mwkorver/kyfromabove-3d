#!/usr/bin/env python3
"""A run on the queue workers, from the laptop: send its jobs, follow it.

A run measures every building before it builds any (a building's occlusion
reads its neighbours' measured levels): ``measure`` sends one message per
tile; once ``status`` shows the measure jobs done, ``build`` sends the build
batches -- 8-16 nearby buildings each, one tile's. A building whose
fingerprint matches the one published is skipped, so a build after a code
change rebuilds what the change touched, and a run sent twice costs little.
``retry-failed`` sends again the batches holding buildings that failed.
``local`` runs the same jobs on this machine, one after another, with no
queue and no AWS account (``--store`` a folder; ``--limit`` builds only the
first few buildings).

Usage::

    python scripts/queue_run.py measure --queue-url <url> --run r1 --tiles N077E228 N077E229
    python scripts/queue_run.py status  --run r1
    python scripts/queue_run.py build   --queue-url <url> --run r1 --tiles N077E228 N077E229 --batch 12
    python scripts/queue_run.py retry-failed --queue-url <url> --run r1
    python scripts/queue_run.py local   --run r1 --store outputs/one_tile --tiles N076E229 --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _sqs(args):
    import boto3
    return boto3.client("sqs", region_name=args.region, endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None)


def _send(sqs, url: str, jobs) -> int:
    n = 0
    jobs = list(jobs)
    for i in range(0, len(jobs), 10):
        sqs.send_message_batch(QueueUrl=url, Entries=[{"Id": str(k), "MessageBody": json.dumps(j)}
                                                       for k, j in enumerate(jobs[i:i + 10])])
        n += len(jobs[i:i + 10])
    return n


def cmd_measure(args, store) -> int:
    n = _send(_sqs(args), args.queue_url, ({"id": f"measure-{t}", "kind": "measure", "run": args.run, "tile": t}
                                           for t in args.tiles))
    print(f"{n} measure job(s) sent")
    return 0


def cmd_build(args, store) -> int:
    from kyfromabove_3d.pieces import batches
    jobs = []
    for t in args.tiles:
        recs = store.read_json(f"tiles/{t}/buildings.json")["buildings"]
        for k, b in enumerate(batches(recs, args.batch)):
            jobs.append({"id": f"build-{t}-{k:03d}", "kind": "build", "run": args.run, "tile": t, "gers": b})
    print(f"{_send(_sqs(args), args.queue_url, jobs)} build job(s) sent")
    return 0


def cmd_status(args, store) -> int:
    counts = {k: sum(1 for _ in store.list(f"runs/{args.run}/{k}/")) for k in ("done", "attempts", "parked", "failed")}
    print(f"run {args.run}: {counts['done']} job(s) done, {counts['attempts']} with a failed try, "
          f"{counts['parked']} parked, {counts['failed']} building(s) failed")
    for key in list(store.list(f"runs/{args.run}/parked/"))[:10]:
        print("  parked:", key.rsplit("/", 1)[-1], store.read_json(key).get("last_error", "")[-200:])
    return 0


def cmd_retry_failed(args, store) -> int:
    by_tile = {}
    for key in store.list(f"runs/{args.run}/failed/"):
        g = key.rsplit("/", 1)[-1][:-5]
        tile = store.read_json(f"pieces/{g}/piece.json").get("tile") if store.exists(f"pieces/{g}/piece.json") else None
        if tile:
            by_tile.setdefault(tile, []).append(g)
    jobs = [{"id": f"retry-{t}-{k:03d}", "kind": "build", "run": args.run, "tile": t, "gers": gs[k:k + args.batch]}
            for t, gs in by_tile.items() for k in range(0, len(gs), args.batch)]
    print(f"{_send(_sqs(args), args.queue_url, jobs)} retry job(s) sent")
    return 0


def cmd_local(args, store) -> int:
    """The run's jobs in this process: every tile measured, then built."""
    os.environ["STORE"] = args.store
    os.environ.setdefault("RAW_DIR", str(Path(__file__).resolve().parents[1] / "data" / "raw"))
    from kyfromabove_3d.cloud_jobs import build_job, measure_job
    from kyfromabove_3d.pieces import batches
    for t in args.tiles:
        print("measure", t, measure_job({"kind": "measure", "run": args.run, "tile": t}), flush=True)
    left = args.limit or None
    for t in args.tiles:
        recs = store.read_json(f"tiles/{t}/buildings.json")["buildings"]
        for b in batches(recs, args.batch):
            if left is not None:
                b = b[:left]
                left -= len(b)
            if b:
                print("build", t, build_job({"kind": "build", "run": args.run, "tile": t, "gers": b}), flush=True)
            if left == 0:
                return 0
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("measure", "build", "status", "retry-failed", "local"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--queue-url", default=os.environ.get("QUEUE_URL"))
    ap.add_argument("--tiles", nargs="*", default=[])
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="local: build only this many buildings (0: all)")
    ap.add_argument("--store", default=os.environ.get("STORE", "s3://kyfromabove-3d"))
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    args = ap.parse_args(argv)
    from kyfromabove_3d.store import store_from
    store = store_from(args.store, endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None)
    return {"measure": cmd_measure, "build": cmd_build, "status": cmd_status,
            "retry-failed": cmd_retry_failed, "local": cmd_local}[args.cmd](args, store)


if __name__ == "__main__":
    sys.exit(main())
