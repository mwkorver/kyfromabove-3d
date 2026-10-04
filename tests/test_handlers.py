"""The Lambda handlers against mocked AWS (moto): a marker's EventBridge
event becomes a row in the pieces table; a tile with no inputs is reported
as that message's failure, not the batch's."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

moto = pytest.importorskip("moto")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

import handlers                                                               # noqa: E402


def _event(bucket, key):
    """What EventBridge delivers for an S3 object, wrapped in an SQS message."""
    return {"messageId": "m1", "body": json.dumps({
        "source": "aws.s3", "detail-type": "Object Created", "time": "2026-09-24T12:00:00Z",
        "detail": {"bucket": {"name": bucket}, "object": {"key": key}}})}


def test_a_markers_event_becomes_a_row_in_the_pieces_table(monkeypatch):
    import boto3
    with moto.mock_aws():
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        s3 = boto3.client("s3", region_name="us-west-2")
        s3.create_bucket(Bucket="kfa", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
        ddb = boto3.client("dynamodb", region_name="us-west-2")
        ddb.create_table(TableName="pieces", KeySchema=[{"AttributeName": "gers", "KeyType": "HASH"}],
                         AttributeDefinitions=[{"AttributeName": "gers", "AttributeType": "S"}],
                         BillingMode="PAY_PER_REQUEST")
        marker = {"gers": "g1-a", "tile": "N076E228", "code_version": "stage1-1", "parent_gers": "g1",
                  "lidar_height_m": 17.7, "n_frames": 12, "heads": ["Bwd", "Fwd"], "inputs_hash": "abc"}
        s3.put_object(Bucket="kfa", Key="pieces/g1-a/piece.json", Body=json.dumps(marker))
        monkeypatch.setenv("TABLE", "pieces")
        out = handlers.record_piece({"Records": [_event("kfa", "pieces/g1-a/piece.json")]}, None)
        assert out == {"batchItemFailures": []}
        item = boto3.resource("dynamodb", region_name="us-west-2").Table("pieces").get_item(Key={"gers": "g1-a"})["Item"]
        assert item["status"] == "ready" and item["tile"] == "N076E228" and float(item["lidar_height_m"]) == 17.7
        assert item["key"] == "pieces/g1-a/piece.json" and item["heads"] == ["Bwd", "Fwd"]
        # a marker that is not there: that message fails, alone
        out = handlers.record_piece({"Records": [_event("kfa", "pieces/missing/piece.json")]}, None)
        assert out == {"batchItemFailures": [{"itemIdentifier": "m1"}]}


def test_a_tile_without_inputs_fails_only_its_own_message(monkeypatch, tmp_path):
    import boto3
    with moto.mock_aws():
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        boto3.client("s3", region_name="us-west-2").create_bucket(
            Bucket="kfa", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
        monkeypatch.setenv("BUCKET", "kfa")
        monkeypatch.setenv("WORK_DIR", str(tmp_path))
        monkeypatch.setattr(handlers, "_grid_and_index", lambda: (None, None))
        out = handlers.make_pieces({"Records": [{"messageId": "t1", "body": json.dumps({"tile": "N000E000"})}]}, None)
        assert out == {"batchItemFailures": [{"itemIdentifier": "t1"}]}
        assert list(tmp_path.iterdir()) == []                                   # its work folder cleaned up


def test_the_building_step_reports_each_building_and_cleans_up(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from kyfromabove_3d import building
    seen = {}

    def fake(gers, store, work, index, force=False, timeout_s=0.0, **k):
        seen.update(gers=gers, force=force, timeout_s=timeout_s, bucket=store.bucket)
        (Path(work) / "crop.png").write_bytes(b"x")
        return {"built": [{"gers": "g1", "seconds": {"crops": 1.0}, "peak_mb": {"crops": 300.0}, "bytes": 10,
                           "triangles": 76, "path": "profile", "files": {"a": 10}}],
                "skipped": ["g2"], "failed": {}, "seconds": {"total": 2.0}, "peak_mb_scene": 350.0}

    with moto.mock_aws():
        monkeypatch.setenv("BUCKET", "kfa")
        monkeypatch.setenv("WORK_DIR", str(tmp_path))
        monkeypatch.setattr(handlers, "_grid_and_index", lambda: (None, None))
        monkeypatch.setattr(building, "building_step", fake)
        out = handlers.building_step({"gers": ["g1", "g2"]},
                                     SimpleNamespace(get_remaining_time_in_millis=lambda: 900_000))
    assert seen == {"gers": ["g1", "g2"], "force": False, "timeout_s": 870.0, "bucket": "kfa"}
    assert out["built"] == 1 and out["skipped"] == 1 and out["buildings"][0]["triangles"] == 76
    assert "files" not in out["buildings"][0]                               # the summary stays small
    assert list(tmp_path.iterdir()) == []                                   # its work folder cleaned up
