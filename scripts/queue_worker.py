#!/usr/bin/env python3
"""A Spot machine's worker: takes jobs from the run's SQS queue and runs them
(kyfromabove_3d.queue_worker, kyfromabove_3d.cloud_jobs) until the queue
stays empty or the Spot notice comes.

Usage (on the machine, in the pipeline's image)::

    STORE=s3://kyfromabove-3d RAW_DIR=/opt/data/raw WORK_DIR=/var/tmp/work \\
        python scripts/queue_worker.py --queue-url <url> --run <run> --slots 12
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue-url", required=True)
    ap.add_argument("--run", required=True, help="the run's name: its progress is kept under runs/<run>/")
    ap.add_argument("--slots", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="jobs at once (a build job is one process; default: the processors less two)")
    ap.add_argument("--visibility-s", type=int, default=900)
    ap.add_argument("--heartbeat-s", type=float, default=300.0)
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    ap.add_argument("--max-failures", type=int, default=3)
    ap.add_argument("--idle-exit-s", type=float, default=600.0, help="stop when the queue has been empty this long")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="take no new job after this long; finish the running ones and stop (before the machine's end)")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    args = ap.parse_args(argv)
    import boto3

    import signal

    from kyfromabove_3d.cloud_jobs import _store
    from kyfromabove_3d.queue_worker import QueueWorker, RunLog, asg_protector
    sqs = boto3.client("sqs", region_name=args.region, endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None)
    w = QueueWorker(sqs, args.queue_url, "kyfromabove_3d.cloud_jobs:run_job", RunLog(_store(), args.run),
                    slots=args.slots, visibility_s=args.visibility_s, heartbeat_s=args.heartbeat_s,
                    timeout_s=args.timeout_s, max_failures=args.max_failures, log=lambda m: print(m, flush=True),
                    protect=asg_protector(log=lambda m: print(m, flush=True)),
                    max_seconds=args.max_hours * 3600.0 if args.max_hours else None)
    # a shutdown (docker stop, the machine's end): the running jobs go back to the queue at once
    signal.signal(signal.SIGTERM, lambda *a: w.stop.set())
    stats = w.run(idle_exit_s=args.idle_exit_s)
    print(f"worker stopped: {stats}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
