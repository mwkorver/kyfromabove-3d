"""The whole event chain on mocked AWS (moto), with the stack's own rule
pattern: pieces published to S3 -> EventBridge -> the pieces queue ->
record_piece -> one row per piece. The marker, written last, is the only
object that raises an event; a re-run with nothing changed raises none; a
changed piece raises one again."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

moto = pytest.importorskip("moto")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))

import handlers                                                               # noqa: E402
from kyfromabove_3d import pieces as P                                        # noqa: E402
from kyfromabove_3d.store import S3Store                                      # noqa: E402

REGION = "us-west-2"


def _piece_pattern():
    """The rule's key pattern as the stack deploys it (without importing CDK)."""
    src = (ROOT / "deploy" / "stack.py").read_text(encoding="utf-8")
    line = next(ln for ln in src.splitlines() if ln.startswith("PIECE_PATTERN_KEY"))
    return json.loads(line.split("=", 1)[1].strip().replace("'", '"'))


def _piece(folder: Path, gers: str, height: float) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "footprint.json").write_text(json.dumps({"gers": gers, "height_m": height}))
    (folder / "clip.json").write_text("{}")
    (folder / "piece.json").write_text(json.dumps({"gers": gers, "tile": "N076E228", "lidar_height_m": height,
                                                   "inputs_hash": P.inputs_hash(folder)}))
    return folder


def _drain(sqs, url):
    got = []
    while True:
        m = sqs.receive_message(QueueUrl=url, MaxNumberOfMessages=10).get("Messages", [])
        if not m:
            return got
        got += m
        for x in m:
            sqs.delete_message(QueueUrl=url, ReceiptHandle=x["ReceiptHandle"])


def test_published_pieces_arrive_as_rows_once_each(tmp_path, monkeypatch):
    import boto3
    with moto.mock_aws():
        monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
        s3, ev, sqs = (boto3.client(n, region_name=REGION) for n in ("s3", "events", "sqs"))
        s3.create_bucket(Bucket="kfa", CreateBucketConfiguration={"LocationConstraint": REGION})
        s3.put_bucket_notification_configuration(Bucket="kfa",
                                                 NotificationConfiguration={"EventBridgeConfiguration": {}})
        q = sqs.create_queue(QueueName="pieces-ready")["QueueUrl"]
        arn = sqs.get_queue_attributes(QueueUrl=q, AttributeNames=["QueueArn"])["Attributes"]["QueueArn"]
        ev.put_rule(Name="PieceArrived", EventPattern=json.dumps({
            "source": ["aws.s3"], "detail-type": ["Object Created"],
            "detail": {"bucket": {"name": ["kfa"]}, "object": {"key": _piece_pattern()}}}))
        ev.put_targets(Rule="PieceArrived", Targets=[{"Id": "q", "Arn": arn}])
        boto3.client("dynamodb", region_name=REGION).create_table(
            TableName="pieces", KeySchema=[{"AttributeName": "gers", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "gers", "AttributeType": "S"}], BillingMode="PAY_PER_REQUEST")
        monkeypatch.setenv("TABLE", "pieces")
        store = S3Store("kfa", client=s3)
        table = boto3.resource("dynamodb", region_name=REGION).Table("pieces")

        folders = [_piece(tmp_path / g, g, h) for g, h in (("g1-a", 40.0), ("g1-b", 22.0), ("g2", 9.5))]
        assert all(P.publish_piece(f, store) for f in folders)
        P.publish_tile("N076E228", {"pieces": folders, "records": [], "splits": []}, [f.name for f in folders], store)
        msgs = _drain(sqs, q)
        assert sorted(json.loads(m["Body"])["detail"]["object"]["key"] for m in msgs) == \
            ["pieces/g1-a/piece.json", "pieces/g1-b/piece.json", "pieces/g2/piece.json"]    # markers only
        out = handlers.record_piece({"Records": [{"messageId": m["MessageId"], "body": m["Body"]} for m in msgs]}, None)
        assert out == {"batchItemFailures": []}
        assert table.scan()["Count"] == 3 and float(table.get_item(Key={"gers": "g1-b"})["Item"]["lidar_height_m"]) == 22.0

        # the same tile again, nothing changed: no writes, no events
        assert not any(P.publish_piece(f, store) for f in folders)
        assert _drain(sqs, q) == []
        # one building's height changed: its piece, and only it, again
        _piece(folders[2], "g2", 11.0)
        assert [P.publish_piece(f, store) for f in folders] == [False, False, True]
        assert [json.loads(m["Body"])["detail"]["object"]["key"] for m in _drain(sqs, q)] == ["pieces/g2/piece.json"]
