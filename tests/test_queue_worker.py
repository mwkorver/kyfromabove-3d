"""The queue worker (kyfromabove_3d.queue_worker) against mocked SQS and S3
(moto): a job done is deleted and recorded; one that fails, crashes or hangs
comes back and is parked after its failures; a Spot notice returns the job
at once without counting it; the heartbeat keeps a long job hidden."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
moto = pytest.importorskip("moto")

from kyfromabove_3d.queue_worker import QueueWorker, RunLog  # noqa: E402


@pytest.fixture
def aws(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    with moto.mock_aws():
        import boto3
        from kyfromabove_3d.store import S3Store
        sqs = boto3.client("sqs", region_name="us-west-2")
        boto3.client("s3", region_name="us-west-2").create_bucket(
            Bucket="runs-test", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
        url = sqs.create_queue(QueueName="jobs")["QueueUrl"]
        yield sqs, url, S3Store("runs-test")


class Spy:
    """The SQS client, counting the visibility changes (the heartbeat)."""
    def __init__(self, sqs):
        self.sqs, self.visibility = sqs, []

    def __getattr__(self, name):
        return getattr(self.sqs, name)

    def change_message_visibility(self, **kw):
        self.visibility.append(kw["VisibilityTimeout"])
        return self.sqs.change_message_visibility(**kw)


def worker(sqs, url, store, handler, **kw):
    kw = {"visibility_s": 30, "heartbeat_s": 600, "timeout_s": 60, "max_failures": 2, "poll_s": 0.1, "wait_s": 0,
          "notice": lambda: False, "log": lambda *a: None, **kw}
    return QueueWorker(sqs, url, f"queue_jobs:{handler}", RunLog(store, "r1"), **kw)


def send(sqs, url, **job):
    sqs.send_message(QueueUrl=url, MessageBody=json.dumps({"id": "j1", "kind": "build", "gers": ["a", "b"], **job}))


def left(sqs, url):
    return len(sqs.receive_message(QueueUrl=url, MaxNumberOfMessages=10, WaitTimeSeconds=0).get("Messages", []))


def test_a_job_done_is_deleted_and_recorded(aws):
    sqs, url, store = aws
    send(sqs, url)
    stats = worker(sqs, url, store, "ok").run(max_jobs=1)
    assert stats["done"] == 1 and left(sqs, url) == 0
    assert store.read_json("runs/r1/done/j1.json")["result"] == {"n": 2}


@pytest.mark.parametrize("handler", ["fail", "crash"])
def test_a_job_that_fails_comes_back_then_is_parked(aws, handler):
    sqs, url, store = aws
    send(sqs, url)
    w = worker(sqs, url, store, handler)
    w.run(max_jobs=1)
    assert store.read_json("runs/r1/attempts/j1.json")["failures"] == 1 and left(sqs, url) == 1
    # (moto keeps the message visible again: the next take is the second failure)
    time.sleep(0.2)
    w.run(max_jobs=1)
    assert store.exists("runs/r1/parked/j1.json") and left(sqs, url) == 0


def test_a_job_that_hangs_times_out(aws):
    sqs, url, store = aws
    send(sqs, url, sleep=30)
    worker(sqs, url, store, "slow", timeout_s=2).run(max_jobs=1)
    assert "timed out" in store.read_json("runs/r1/attempts/j1.json")["last_error"]


def test_a_spot_notice_returns_the_job_without_counting_it(aws):
    sqs, url, store = aws
    send(sqs, url, sleep=30)
    t0 = time.time()
    stats = worker(sqs, url, store, "slow", notice=lambda: time.time() - t0 > 2).run()
    assert stats["returned"] == 1 and time.time() - t0 < 20
    assert not store.exists("runs/r1/attempts/j1.json")
    assert left(sqs, url) == 1                                   # visible again, at once


def test_the_heartbeat_keeps_a_long_job_hidden(aws):
    sqs, url, store = aws
    send(sqs, url, sleep=3)
    spy = Spy(sqs)
    worker(spy, url, store, "slow", heartbeat_s=1, visibility_s=30).run(max_jobs=1)
    assert spy.visibility.count(30) >= 2


def test_a_busy_machine_is_protected_from_scale_in_and_released_when_idle(aws):
    sqs, url, store = aws
    send(sqs, url)
    seen = []
    w = worker(sqs, url, store, "ok", protect=seen.append)
    w.run(max_jobs=1)
    assert seen == [True, False]


def test_after_its_lifetime_the_worker_takes_no_new_job(aws):
    sqs, url, store = aws
    send(sqs, url)
    w = worker(sqs, url, store, "ok", max_seconds=0.0)
    stats = w.run(idle_exit_s=5)
    assert stats["done"] == 0 and left(sqs, url) == 1                    # the job waits for another machine


def test_a_stop_request_returns_the_running_job_at_once(aws):
    import threading
    sqs, url, store = aws
    send(sqs, url, sleep=30)
    w = worker(sqs, url, store, "slow")
    threading.Timer(2.0, w.stop.set).start()                            # SIGTERM sets this
    t0 = time.time()
    stats = w.run(max_jobs=1)
    assert stats["returned"] == 1 and time.time() - t0 < 20
    assert left(sqs, url) == 1                                            # back on the queue, not counted
    assert not list(store.list("runs/r1/attempts/"))


def test_off_ec2_the_machine_type_is_none_and_asked_once(monkeypatch):
    import kyfromabove_3d.queue_worker as qw

    monkeypatch.setattr(qw, "IMDS", "http://127.0.0.1:9")          # nothing answers there
    monkeypatch.setattr(qw, "_MACHINE", [])
    assert qw.machine_type(timeout_s=0.2) is None and qw._MACHINE == [None]
    monkeypatch.setattr(qw, "IMDS", "http://unused.invalid")
    assert qw.machine_type() is None                                 # the first answer is kept
