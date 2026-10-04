#!/usr/bin/env python3
"""Let the viewer's CloudFront distribution read the served prefixes of
``s3://kyfromabove-3d``, by adding one statement to the bucket's own policy.

The bucket and its policy (HTTPS only) were made by hand, so the viewer stack
does not own the policy; it prints the statement CloudFront needs as the
``BucketPolicyStatement`` output. This adds that statement (replacing an
earlier one with the same Sid) and keeps every other statement as it is.
Without ``--apply`` it only prints the policy it would write.

Usage::

    python deploy/viewer_policy.py            # show
    python deploy/viewer_policy.py --apply    # write
"""

from __future__ import annotations

import argparse
import json
import sys

BUCKET = "kyfromabove-3d"


def merged(current: dict, statement: dict) -> dict:
    """``current`` with ``statement`` added, or in place of the one with its Sid."""
    others = [s for s in current.get("Statement", []) if s.get("Sid") != statement["Sid"]]
    return {"Version": current.get("Version", "2012-10-17"), "Statement": others + [statement]}


def main() -> int:
    import boto3

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", default="kyfromabove-3d-viewer")
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    cf = boto3.client("cloudformation", region_name=args.region)
    outs = {o["OutputKey"]: o["OutputValue"] for o in cf.describe_stacks(StackName=args.stack)["Stacks"][0]["Outputs"]}
    statement = json.loads(outs["BucketPolicyStatement"])
    s3 = boto3.client("s3", region_name=args.region)
    current = json.loads(s3.get_bucket_policy(Bucket=BUCKET)["Policy"])
    new = merged(current, statement)
    kept = [s.get("Sid") for s in current["Statement"] if s.get("Sid") != statement["Sid"]]
    print(json.dumps(new, indent=1))
    print(f"kept: {kept}; added: {statement['Sid']}", file=sys.stderr)
    if args.apply:
        s3.put_bucket_policy(Bucket=BUCKET, Policy=json.dumps(new))
        print("written", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
