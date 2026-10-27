#!/usr/bin/env python3
"""The project bucket's undo: S3 versioning, old versions kept a week.

Every run overwrites the published files in place (the trees under
``kentucky/``, each tile's 3D Tiles, each building). With versioning on, an
overwritten or deleted file is kept as an old version for ``KEEP_DAYS`` days
and then dropped, so a bad run can be rolled back within the week and the
bucket does not grow for good. The bucket was made by hand, before the CDK
app, which only imports it; this script is where its settings are written
down.

    python deploy/bucket_settings.py show
    python deploy/bucket_settings.py apply
    # put a prefix back as it was at a time (UTC); a dry run unless --go
    python deploy/bucket_settings.py restore --prefix kentucky/measured/ --at 2026-10-24T20:00:00Z [--go]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

BUCKET = os.environ.get("KFA_BUCKET", "kyfromabove-3d")
REGION = "us-west-2"
#: how long an overwritten or deleted file is kept
KEEP_DAYS = 7

LIFECYCLE = {"Rules": [{
    "ID": f"old-versions-{KEEP_DAYS}-days",
    "Status": "Enabled",
    "Filter": {"Prefix": ""},
    # an overwritten or deleted file's old version goes KEEP_DAYS after it was replaced
    "NoncurrentVersionExpiration": {"NoncurrentDays": KEEP_DAYS},
    # and a delete marker with nothing left behind it goes too
    "Expiration": {"ExpiredObjectDeleteMarker": True},
    # uploads in parts that never finished
    "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": KEEP_DAYS},
}]}


def show(s3, bucket: str) -> Dict:
    v = s3.get_bucket_versioning(Bucket=bucket).get("Status", "never enabled")
    try:
        rules = s3.get_bucket_lifecycle_configuration(Bucket=bucket)["Rules"]
    except s3.exceptions.ClientError as exc:
        if "NoSuchLifecycleConfiguration" not in str(exc):
            raise
        rules = []
    return {"versioning": v, "lifecycle": rules}


def apply(s3, bucket: str) -> Dict:
    """Versioning on and the lifecycle above. The lifecycle replaces any the
    bucket had (it had none)."""
    s3.put_bucket_versioning(Bucket=bucket, VersioningConfiguration={"Status": "Enabled"})
    s3.put_bucket_lifecycle_configuration(Bucket=bucket, LifecycleConfiguration=LIFECYCLE)
    return show(s3, bucket)


def restore(s3, bucket: str, prefix: str, at: datetime, go: bool = False, log=print) -> Dict[str, int]:
    """Every file under ``prefix`` put back as it was at ``at``: a file
    changed since gets its version from then copied back on top; a file made
    since, or deleted since, is deleted or brought back. Files whose newest
    version is already the one from then are left alone. Only what versioning
    kept: changes older than ``KEEP_DAYS`` cannot be undone."""
    newest: Dict[str, dict] = {}                 # each key's newest version or delete marker
    then: Dict[str, Optional[dict]] = {}         # each key's newest version at ``at`` (None: absent then)
    pages = s3.get_paginator("list_object_versions").paginate(Bucket=bucket, Prefix=prefix)
    for page in pages:
        entries = [dict(v, deleted=False) for v in page.get("Versions", [])]
        entries += [dict(m, deleted=True) for m in page.get("DeleteMarkers", [])]
        for e in entries:
            k = e["Key"]
            if e["IsLatest"]:                    # S3 says which (times are only to the second)
                newest[k] = e
            if e["LastModified"] <= at:
                best = then.get(k)
                if best is None or e["LastModified"] > best["LastModified"]:
                    then[k] = e
    counts = {"unchanged": 0, "put_back": 0, "deleted": 0}
    plan: List[tuple] = []
    for k, cur in sorted(newest.items()):
        old = then.get(k)
        if old is not None and old["deleted"]:
            old = None                           # deleted at that time: absent then
        if old is None:
            if not cur["deleted"]:
                plan.append(("delete", k, None))
            else:
                counts["unchanged"] += 1
        elif not cur["deleted"] and (old["VersionId"] == cur["VersionId"] or old["ETag"] == cur["ETag"]):
            counts["unchanged"] += 1                                    # the same bytes (a restore copies them)
        else:
            plan.append(("put_back", k, old["VersionId"]))
    for action, k, vid in plan:
        if go:
            if action == "delete":
                s3.delete_object(Bucket=bucket, Key=k)                      # a delete marker: itself undoable
            else:
                s3.copy_object(Bucket=bucket, Key=k, CopySource={"Bucket": bucket, "Key": k, "VersionId": vid},
                               MetadataDirective="COPY")
        counts["deleted" if action == "delete" else "put_back"] += 1
    log(f"{'restored' if go else 'dry run'}: {prefix} as at {at.isoformat()}: {counts['put_back']} put back, "
        f"{counts['deleted']} deleted (made since), {counts['unchanged']} unchanged")
    return counts


def _when(text: str) -> datetime:
    t = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("show", "apply", "restore"))
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--prefix", help="restore: the files to put back (e.g. kentucky/measured/)")
    ap.add_argument("--at", type=_when, help="restore: the time to go back to, ISO 8601 (UTC unless given)")
    ap.add_argument("--go", action="store_true", help="restore: do it (otherwise a dry run that only counts)")
    args = ap.parse_args(argv)
    import boto3

    s3 = boto3.client("s3", region_name=REGION)
    if args.cmd == "restore":
        if not (args.prefix and args.at):
            ap.error("restore needs --prefix and --at")
        restore(s3, args.bucket, args.prefix, args.at, go=args.go)
    else:
        print((apply if args.cmd == "apply" else show)(s3, args.bucket))
    return 0


if __name__ == "__main__":
    sys.exit(main())
