"""Stand-in jobs for tests/test_queue_worker.py, run in the worker's child process."""
import os
import time


def ok(job):
    return {"n": len(job.get("gers", []))}


def fail(job):
    raise RuntimeError("a job that fails")


def crash(job):
    os._exit(3)


def slow(job):
    time.sleep(float(job.get("sleep", 30)))
    return {"slept": True}
