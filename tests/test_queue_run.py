"""A run's three job kinds, sent and run without AWS: the tile, measure and
build messages (moto's SQS), the worker's handler choosing a job by its kind,
``local`` running the kinds in order, and ``status`` counting each kind."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import queue_run                                                            # noqa: E402

from kyfromabove_3d import cloud_jobs                                      # noqa: E402
from kyfromabove_3d.store import LocalStore                                # noqa: E402


@pytest.fixture
def sqs(monkeypatch):
    moto = pytest.importorskip("moto")
    import boto3

    with moto.mock_aws():
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        client = boto3.client("sqs", region_name="us-west-2")
        url = client.create_queue(QueueName="jobs")["QueueUrl"]
        monkeypatch.setattr(queue_run, "_sqs", lambda args: client)
        yield client, url


def _bodies(client, url):
    out = []
    while True:
        r = client.receive_message(QueueUrl=url, MaxNumberOfMessages=10)
        msgs = r.get("Messages", [])
        if not msgs:
            return out
        for m in msgs:
            out.append(json.loads(m["Body"]))
            client.delete_message(QueueUrl=url, ReceiptHandle=m["ReceiptHandle"])


def test_the_tile_command_sends_one_tile_job_per_tile(sqs, tmp_path):
    client, url = sqs
    queue_run.main(["tile", "--run", "r9", "--queue-url", url, "--store", str(tmp_path),
                    "--tiles", "N076E228", "N076E229"])
    got = sorted(_bodies(client, url), key=lambda j: j["tile"])
    assert got == [{"id": "tile-N076E228", "kind": "tile", "run": "r9", "tile": "N076E228"},
                   {"id": "tile-N076E229", "kind": "tile", "run": "r9", "tile": "N076E229"}]


def test_the_worker_runs_a_job_by_its_kind(monkeypatch):
    seen = []
    for kind in ("tile", "measure", "build"):
        monkeypatch.setattr(cloud_jobs, f"{kind}_job", lambda job, k=kind: seen.append(k) or {"kind": k})
    for kind in ("tile", "measure", "build"):
        assert cloud_jobs.run_job({"kind": kind, "tile": "T"}) == {"kind": kind}
    assert seen == ["tile", "measure", "build"]
    with pytest.raises(KeyError):
        cloud_jobs.run_job({"kind": "nope"})


def test_local_makes_every_tile_before_measuring_any_and_measures_before_building(monkeypatch, tmp_path):
    monkeypatch.setenv("STORE", "unset")                 # local sets these for its jobs; put back after
    monkeypatch.setenv("RAW_DIR", str(tmp_path))
    store = LocalStore(tmp_path)
    for t in ("A", "B"):
        sq = "POLYGON ((-85.76 38.25, -85.7599 38.25, -85.7599 38.2501, -85.76 38.2501, -85.76 38.25))"
        store.put_json(f"tiles/{t}/buildings.json", {"buildings": [{"id": f"{t}1", "wkt": sq},
                                                                   {"id": f"{t}2", "wkt": sq}]})
    calls = []
    monkeypatch.setattr(cloud_jobs, "tile_job", lambda job: calls.append(("tile", job["tile"])) or {})
    monkeypatch.setattr(cloud_jobs, "measure_job", lambda job: calls.append(("measure", job["tile"])) or {})
    monkeypatch.setattr(cloud_jobs, "build_job", lambda job: calls.append(("build", tuple(job["gers"]))) or {})
    queue_run.main(["local", "--run", "r", "--store", str(tmp_path), "--tiles", "A", "B", "--limit", "3"])
    kinds = [c[0] for c in calls]
    assert kinds[:4] == ["tile", "tile", "measure", "measure"] and set(kinds[4:]) == {"build"}
    assert sum(len(c[1]) for c in calls if c[0] == "build") == 3          # --limit
    calls.clear()
    queue_run.main(["local", "--run", "r", "--store", str(tmp_path), "--tiles", "A", "--skip-tile"])
    assert [c[0] for c in calls][0] == "measure"


def test_status_counts_each_kind_of_job_done(tmp_path, capsys):
    store = LocalStore(tmp_path)
    for name in ("tile-A", "tile-B", "measure-A", "build-A-000", "build-A-001", "build-A-002"):
        store.put_json(f"runs/r/done/{name}.json", {})
    store.put_json("runs/r/failed/g1.json", {"error": "x"})
    queue_run.main(["status", "--run", "r", "--store", str(tmp_path)])
    out = capsys.readouterr().out
    assert "6 job(s) done (2 tile, 1 measure, 3 build)" in out and "1 building(s) failed" in out
