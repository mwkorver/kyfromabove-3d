"""The queue worker: an EC2 Spot machine's loop over an SQS queue (refactor
step 4, 2026-10-15).

Each message is one job -- one tile's buildings to measure, or a batch of
8-16 nearby buildings to build (``kyfromabove_3d.cloud_jobs``). The worker:

* takes one message at a time per slot and runs its job in a child process
  of its own (spawned: GDAL and curl are not fork-safe), so a job that
  crashes, hangs or runs out of memory takes only itself down;
* keeps the message hidden from the other machines while the job runs (the
  heartbeat: its visibility renewed every ``heartbeat_s``);
* on success deletes the message and writes ``runs/<run>/done/<job>.json``;
  a building that failed inside a job is the job's to record
  (``runs/<run>/failed/<gers>.json``) -- one bad building does not fail the
  fifteen beside it;
* when a job crashes, hangs past ``timeout_s`` or returns an error, counts it
  (``runs/<run>/attempts/<job>.json``) and lets the message come back; after
  ``max_failures`` real failures the job is parked
  (``runs/<run>/parked/<job>.json``) and its message deleted. A Spot
  interruption is not a failure: SQS's own receive count would count it, so
  the dead-letter queue's limit is set high and only these counts decide;
* watches the machine's metadata for the Spot two-minute notice: on it, stops
  the running jobs, makes their messages visible again at once (another
  machine takes them; buildings already published are skipped there by their
  fingerprint) and stops. A stop request (``stop``: SIGTERM, a shutdown) does
  the same;
* keeps its machine from being scaled in while it has jobs running
  (``protect(True)``, and ``protect(False)`` when idle): the group, sized to
  the queue, removes idle machines at the tail of a run, never one finishing
  a straggler;
* after ``max_seconds`` takes no new job, finishes the ones it has and stops
  -- before the machine's hard end, so no job is cut off there and its log is
  saved.

Intermediate files stay on the machine's own disk. What a job publishes --
each building's products, ``building.json`` last -- is the checkpoint.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

IMDS = "http://169.254.169.254"


def spot_notice(timeout_s: float = 1.0) -> bool:
    """Whether this machine has its Spot interruption notice (IMDSv2:
    ``/latest/meta-data/spot/instance-action`` answers 200 once it has).
    False off EC2 or when the metadata service does not answer."""
    import urllib.request

    try:
        req = urllib.request.Request(f"{IMDS}/latest/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        token = urllib.request.urlopen(req, timeout=timeout_s).read().decode()
        req = urllib.request.Request(f"{IMDS}/latest/meta-data/spot/instance-action",
                                     headers={"X-aws-ec2-metadata-token": token})
        return urllib.request.urlopen(req, timeout=timeout_s).status == 200
    except Exception:                                               # noqa: BLE001 (404, no IMDS, timeout)
        return False


def asg_protector(log: Callable = print, timeout_s: float = 1.0) -> Callable[[bool], None]:
    """``protect(on)`` for this machine in its Auto Scaling group (scale-in
    protection while it has jobs running); a no-op off EC2 or outside a group."""
    import urllib.request

    try:
        req = urllib.request.Request(f"{IMDS}/latest/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        token = urllib.request.urlopen(req, timeout=timeout_s).read().decode()
        req = urllib.request.Request(f"{IMDS}/latest/meta-data/instance-id", headers={"X-aws-ec2-metadata-token": token})
        instance = urllib.request.urlopen(req, timeout=timeout_s).read().decode()
        import boto3
        asg = boto3.client("autoscaling")
        found = asg.describe_auto_scaling_instances(InstanceIds=[instance])["AutoScalingInstances"]
        group = found[0]["AutoScalingGroupName"] if found else None
    except Exception:                                               # noqa: BLE001 (no IMDS, no group)
        return lambda on: None
    if group is None:
        return lambda on: None

    def protect(on: bool) -> None:
        try:
            asg.set_instance_protection(InstanceIds=[instance], AutoScalingGroupName=group,
                                        ProtectedFromScaleIn=bool(on))
        except Exception as exc:                                    # noqa: BLE001
            log(f"[warn] scale-in protection {on}: {exc}")
    return protect


def _run_job(handler_path: str, job: Dict[str, Any], conn) -> None:
    """In the child: import the handler (``module:function``), run the job,
    send back its result or the error."""
    import importlib

    try:
        mod, fn = handler_path.split(":")
        result = getattr(importlib.import_module(mod), fn)(job)
        conn.send({"ok": True, "result": result})
    except BaseException:                                           # noqa: BLE001
        conn.send({"ok": False, "error": traceback.format_exc()[-4000:]})
    finally:
        conn.close()


@dataclass
class RunLog:
    """Where a run's progress is kept: ``runs/<run>/`` in the store."""
    store: Any
    run: str

    def key(self, kind: str, name: str) -> str:
        return f"runs/{self.run}/{kind}/{name}.json"

    def put(self, kind: str, name: str, obj: Dict[str, Any]) -> None:
        self.store.put_json(self.key(kind, name), {**obj, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    def attempts(self, name: str) -> int:
        k = self.key("attempts", name)
        return int(self.store.read_json(k).get("failures", 0)) if self.store.exists(k) else 0


@dataclass
class QueueWorker:
    """One machine's loop over a queue (see the module's docstring)."""
    sqs: Any
    queue_url: str
    handler: str                                     # "module:function", run in a child per job
    runs: RunLog
    slots: int = 1                                   # jobs at once
    visibility_s: int = 900
    heartbeat_s: float = 300.0
    timeout_s: float = 3600.0
    max_failures: int = 3
    poll_s: float = 5.0
    wait_s: int = 20                                 # SQS long poll
    notice: Callable[[], bool] = spot_notice
    protect: Callable[[bool], None] = lambda on: None
    max_seconds: Optional[float] = None              # take no new job after this; finish and stop
    stop: threading.Event = field(default_factory=threading.Event)
    log: Callable = print
    stats: Dict[str, int] = field(default_factory=lambda: {"done": 0, "failed": 0, "parked": 0, "returned": 0})

    def _receive(self) -> Optional[Dict[str, Any]]:
        r = self.sqs.receive_message(QueueUrl=self.queue_url, MaxNumberOfMessages=1, WaitTimeSeconds=self.wait_s,
                                     VisibilityTimeout=self.visibility_s)
        msgs = r.get("Messages", [])
        return msgs[0] if msgs else None

    def _visible(self, msg, seconds: int) -> None:
        try:
            self.sqs.change_message_visibility(QueueUrl=self.queue_url, ReceiptHandle=msg["ReceiptHandle"],
                                               VisibilityTimeout=seconds)
        except Exception as exc:                                    # noqa: BLE001 (already gone)
            self.log(f"[warn] visibility: {exc}")

    def _start(self, msg) -> Dict[str, Any]:
        job = json.loads(msg["Body"])
        name = str(job.get("id") or msg["MessageId"])
        parent, child = mp.Pipe(duplex=False)
        p = mp.get_context("spawn").Process(target=_run_job, args=(self.handler, job, child), daemon=True)
        p.start()
        child.close()
        self.log(f"job {name}: started ({job.get('kind', '?')}, {len(job.get('gers', []))} building(s))")
        return {"msg": msg, "job": job, "name": name, "proc": p, "conn": parent, "t0": time.time(),
                "beat": time.time()}

    def _finish(self, run: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        name, msg = run["name"], run["msg"]
        if outcome.get("ok"):
            self.sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=msg["ReceiptHandle"])
            self.runs.put("done", name, {"job": run["job"], "result": outcome.get("result"),
                                         "seconds": round(time.time() - run["t0"], 1)})
            self.stats["done"] += 1
            self.log(f"job {name}: done ({time.time() - run['t0']:.0f} s)")
            return
        failures = self.runs.attempts(name) + 1
        self.runs.put("attempts", name, {"failures": failures, "last_error": outcome.get("error")})
        if failures >= self.max_failures:
            self.runs.put("parked", name, {"job": run["job"], "failures": failures, "last_error": outcome.get("error")})
            self.sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=msg["ReceiptHandle"])
            self.stats["parked"] += 1
            self.log(f"job {name}: parked after {failures} failures")
        else:
            self._visible(msg, 0)                       # back on the queue now, for another try
            self.stats["failed"] += 1
            self.log(f"job {name}: failed ({failures} of {self.max_failures}); returned to the queue")

    def _return(self, run: Dict[str, Any]) -> None:
        """A Spot notice: the job stopped, its message visible at once, no failure counted."""
        if run["proc"].is_alive():
            run["proc"].terminate()
            run["proc"].join(10)
        self._visible(run["msg"], 0)
        self.stats["returned"] += 1
        self.log(f"job {run['name']}: returned to the queue (Spot notice)")

    def run(self, max_jobs: Optional[int] = None, idle_exit_s: Optional[float] = None) -> Dict[str, int]:
        """Until the Spot notice, ``max_jobs`` jobs taken, or the queue idle
        for ``idle_exit_s``. Returns the counts."""
        running: List[Dict[str, Any]] = []
        taken, idle_since, t_start, busy = 0, time.time(), time.time(), False
        while True:
            if self.stop.is_set() or self.notice():
                self.log("stop requested" if self.stop.is_set() else "Spot notice: stopping")
                for r in running:
                    self._return(r)
                break
            if busy != bool(running):                       # scale-in protection while jobs run
                busy = bool(running)
                self.protect(busy)
            late = self.max_seconds is not None and time.time() - t_start >= self.max_seconds
            # results
            for r in list(running):
                if r["conn"].poll():
                    try:
                        outcome = r["conn"].recv()
                    except EOFError:
                        outcome = {"ok": False, "error": "the job's process ended without a result"}
                    r["proc"].join(10)
                    running.remove(r)
                    self._finish(r, outcome)
                elif not r["proc"].is_alive():
                    running.remove(r)
                    self._finish(r, {"ok": False, "error": f"the job's process died (exit {r['proc'].exitcode})"})
                elif time.time() - r["t0"] > self.timeout_s:
                    r["proc"].terminate()
                    r["proc"].join(10)
                    running.remove(r)
                    self._finish(r, {"ok": False, "error": f"timed out after {self.timeout_s:.0f} s"})
                elif time.time() - r["beat"] >= self.heartbeat_s:
                    self._visible(r["msg"], self.visibility_s)          # the heartbeat
                    r["beat"] = time.time()
            # new work
            if len(running) < self.slots and (max_jobs is None or taken < max_jobs) and not late:
                msg = self._receive()
                if msg is not None:
                    running.append(self._start(msg))
                    taken += 1
                    idle_since = time.time()
                    continue
            if not running:
                if late:
                    self.log("lifetime reached: stopping")
                    break
                if max_jobs is not None and taken >= max_jobs:
                    break
                if idle_exit_s is not None and time.time() - idle_since >= idle_exit_s:
                    break
            time.sleep(self.poll_s if running else 0.0)
        if busy:
            self.protect(False)
        return dict(self.stats)
