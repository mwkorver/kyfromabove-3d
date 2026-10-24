"""The bucket's undo (deploy/bucket_settings.py) on a moto bucket: versioning
and the week's lifecycle set, and a prefix put back as it was before a bad
run -- overwritten files restored, new files removed, deleted files back."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))


def _contents(s3, prefix):
    keys = [o["Key"] for o in s3.list_objects_v2(Bucket="kfa3d", Prefix=prefix).get("Contents", [])]
    return {k: s3.get_object(Bucket="kfa3d", Key=k)["Body"].read() for k in keys}


def test_apply_then_restore_a_prefix_to_before_a_bad_run():
    moto = pytest.importorskip("moto")
    import boto3

    import bucket_settings as B

    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-west-2")
        s3.create_bucket(Bucket="kfa3d", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
        assert B.show(s3, "kfa3d") == {"versioning": "never enabled", "lifecycle": []}
        got = B.apply(s3, "kfa3d")
        assert got["versioning"] == "Enabled"
        (rule,) = got["lifecycle"]
        assert rule["NoncurrentVersionExpiration"]["NoncurrentDays"] == B.KEEP_DAYS

        for k, body in {"kentucky/measured/tileset.json": b"good root", "kentucky/measured/groups/g1.json": b"g1",
                        "kentucky/measured/groups/g2.json": b"g2", "kentucky/boxes/tileset.json": b"boxes"}.items():
            s3.put_object(Bucket="kfa3d", Key=k, Body=body)
        before = _contents(s3, "kentucky/measured/")
        time.sleep(1.1)                                          # S3 times are to the second
        good = datetime.now(timezone.utc)
        time.sleep(1.1)
        # the bad run: a file overwritten, one deleted, one added; another prefix changed too
        s3.put_object(Bucket="kfa3d", Key="kentucky/measured/tileset.json", Body=b"broken root")
        s3.delete_object(Bucket="kfa3d", Key="kentucky/measured/groups/g2.json")
        s3.put_object(Bucket="kfa3d", Key="kentucky/measured/groups/g3.json", Body=b"stray")
        s3.put_object(Bucket="kfa3d", Key="kentucky/boxes/tileset.json", Body=b"boxes v2")

        dry = B.restore(s3, "kfa3d", "kentucky/measured/", good, log=lambda *_: None)
        assert dry == {"put_back": 2, "deleted": 1, "unchanged": 1}
        assert _contents(s3, "kentucky/measured/") != before      # a dry run changes nothing

        B.restore(s3, "kfa3d", "kentucky/measured/", good, go=True, log=lambda *_: None)
        assert _contents(s3, "kentucky/measured/") == before
        assert _contents(s3, "kentucky/boxes/") == {"kentucky/boxes/tileset.json": b"boxes v2"}   # outside: as is
        again = B.restore(s3, "kfa3d", "kentucky/measured/", good, log=lambda *_: None)
        assert again["put_back"] == 0 and again["deleted"] == 0
