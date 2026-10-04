#!/usr/bin/env python3
"""Start stage 1: one message per LiDAR tile on the tiles queue.

The tiles come from ``inputs/tiles.json`` in the bucket (``stage0_index.py``
writes it), or are named. The queue URL comes from the stack's outputs
unless given.

Usage::

    python deploy/start_tiles.py --all
    python deploy/start_tiles.py --tiles N076E228 N077E228
    python deploy/start_tiles.py --all --endpoint http://localhost:4566     # LocalStack
"""

from __future__ import annotations

import argparse
import json
import sys


def stack_outputs(cf, stack: str) -> dict:
    out = cf.describe_stacks(StackName=stack)["Stacks"][0].get("Outputs", [])
    return {o["OutputKey"]: o["OutputValue"] for o in out}


def main() -> int:
    import boto3

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--tiles", nargs="+")
    which.add_argument("--all", action="store_true", help="every tile in the bucket's inputs/tiles.json")
    ap.add_argument("--stack", default="KyFromAbovePieces")
    ap.add_argument("--queue-url", default=None)
    ap.add_argument("--bucket", default=None)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--region", default="us-west-2")
    args = ap.parse_args()

    kw = {"endpoint_url": args.endpoint, "region_name": args.region}
    outs = {} if (args.queue_url and (args.bucket or args.tiles)) else stack_outputs(boto3.client("cloudformation", **kw),
                                                                                     args.stack)
    queue = args.queue_url or outs["TilesQueueUrl"]
    tiles = args.tiles
    if args.all:
        s3 = boto3.client("s3", **kw)
        body = s3.get_object(Bucket=args.bucket or outs["BucketName"], Key="inputs/tiles.json")["Body"].read()
        tiles = [t["tile"] for t in json.loads(body)["tiles"]]
    sqs = boto3.client("sqs", **kw)
    for i in range(0, len(tiles), 10):
        batch = tiles[i:i + 10]
        sqs.send_message_batch(QueueUrl=queue, Entries=[{"Id": str(k), "MessageBody": json.dumps({"tile": t})}
                                                        for k, t in enumerate(batch)])
    print(f"{len(tiles)} tile(s) queued -> {queue}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
