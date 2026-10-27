#!/usr/bin/env python3
"""Published files the live site no longer points to.

Every file step 4 writes is named by a hash of its contents, so a republish
writes new names and leaves the old files behind. This reads the live tree
(``kentucky/<product>/tileset.json``, its group tilesets, each tile's
tileset) for each product, collects every file they reference, and lists
every other object under the same folders (``kentucky/<product>/`` and
``tiles/*/3dtiles/<product>/``). Nothing else in the bucket is looked at.

It only lists, unless ``--delete`` is given: the listing goes to ``--out``
(one key per line, with the total size) so it can be read first.

Usage::

    python deploy/s3_orphans.py --out outputs/pilot/s3_orphans.txt
    python deploy/s3_orphans.py --out outputs/pilot/s3_orphans.txt --delete    # removes exactly that list
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import sys
from pathlib import Path

BUCKET = os.environ.get("KFA_BUCKET", "kyfromabove-3d")
PRODUCTS = ("overture", "boxes", "measured", "complete")


def referenced(s3, product: str) -> set:
    """Every key the live tree of ``product`` reaches, the tilesets included."""
    seen, todo = set(), [f"kentucky/{product}/tileset.json"]
    while todo:
        key = todo.pop()
        if key in seen:
            continue
        seen.add(key)
        if not key.endswith(".json"):
            continue
        try:
            ts = json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
        except s3.exceptions.NoSuchKey:
            continue
        base = posixpath.dirname(key)

        def walk(node):
            uri = (node.get("content") or {}).get("uri")
            if uri:
                todo.append(posixpath.normpath(posixpath.join(base, uri.split("?")[0])))
            for c in node.get("children", []):
                walk(c)
        walk(ts["root"])
    return seen


def main() -> int:
    import boto3

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--delete", action="store_true", help="delete the keys listed in --out (list first)")
    args = ap.parse_args()
    s3 = boto3.client("s3", region_name=args.region)

    if args.delete:
        keys = [l.split("\t")[0] for l in args.out.read_text().splitlines() if l and not l.startswith("#")]
        for i in range(0, len(keys), 1000):
            s3.delete_objects(Bucket=BUCKET, Delete={"Objects": [{"Key": k} for k in keys[i:i + 1000]], "Quiet": True})
        print(f"deleted {len(keys):,} objects")
        return 0

    live = set()
    for p in PRODUCTS:
        live |= referenced(s3, p)
    orphans, size = [], 0
    pag = s3.get_paginator("list_objects_v2")
    prefixes = [f"kentucky/{p}/" for p in PRODUCTS] + ["tiles/"]
    for prefix in prefixes:
        for page in pag.paginate(Bucket=BUCKET, Prefix=prefix):
            for o in page.get("Contents", []):
                k = o["Key"]
                parts = k.split("/")
                in_scope = prefix != "tiles/" or (len(parts) > 3 and parts[2] == "3dtiles" and parts[3] in PRODUCTS)
                if in_scope and k not in live:
                    orphans.append((k, o["Size"]))
                    size += o["Size"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"# {len(orphans):,} objects, {size / 1e6:.1f} MB, not reached from the live trees of "
                        f"{', '.join(PRODUCTS)}\n" + "".join(f"{k}\t{n}\n" for k, n in orphans))
    print(f"{len(live):,} live files; {len(orphans):,} not referenced, {size / 1e6:.1f} MB -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
