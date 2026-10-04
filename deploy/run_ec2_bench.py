#!/usr/bin/env python3
"""Time the building step on the Spot bench machine (``Ec2BenchStack``),
under a spending cap.

Each run is ``scripts/building_step.py`` inside the Lambda's own image on
the machine, sent through SSM, writing to ``s3://kyfromabove-3d/bench/ec2/``
(never ``buildings/``). After each run the container's peak memory is read
from its cgroup and its ``--results`` lines are copied to
``bench/ec2/results/<label>.jsonl`` and to ``outputs/ec2_bench/``.

Before every run the spend so far is tallied -- the machine's hours times its
zone's Spot price, its disk, and ``--already`` (e.g. the Lambda calibration)
-- and the run is refused if the tally plus its expected cost passes
``--cap``.

Usage::

    python deploy/run_ec2_bench.py --label one_at_a_time --workers 1 --gers ... --expect-min 15
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

BUCKET = "kyfromabove-3d"
TILES = ["N076E228", "N076E229", "N077E228", "N077E229"]
EBS_GB, EBS_GB_MONTH = 40, 0.08


def instance(ec2, stack_outputs):
    iid = stack_outputs["InstanceId"]
    d = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]
    return d


def tally(ec2, inst, already: float):
    """Dollars spent so far on the bench: hours up x the zone's Spot price
    now (the price is capped by the Spot request anyway), disk, plus
    ``already``."""
    up_h = (datetime.datetime.now(datetime.timezone.utc) - inst["LaunchTime"]).total_seconds() / 3600
    az = inst["Placement"]["AvailabilityZone"]
    h = ec2.describe_spot_price_history(InstanceTypes=[inst["InstanceType"]], ProductDescriptions=["Linux/UNIX"],
                                        AvailabilityZone=az,
                                        StartTime=datetime.datetime.now(datetime.timezone.utc))["SpotPriceHistory"]
    price = float(h[0]["SpotPrice"]) if h else 0.35
    return up_h * price + up_h * EBS_GB * EBS_GB_MONTH / 730 + already, price, up_h


def ssm_run(ssm, iid: str, script: str, timeout_s: int = 7200) -> tuple:
    cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                           Parameters={"commands": [script], "executionTimeout": [str(timeout_s)]},
                           TimeoutSeconds=600)["Command"]["CommandId"]
    while True:
        time.sleep(10)
        try:
            r = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
        except ssm.exceptions.InvocationDoesNotExist:
            continue
        if r["Status"] not in ("Pending", "InProgress", "Delayed"):
            return r["Status"], r.get("StandardOutputContent", ""), r.get("StandardErrorContent", "")


def main() -> int:
    import boto3

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--gers", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--expect-min", type=float, required=True, help="the run's expected minutes, for the cap")
    ap.add_argument("--cap", type=float, default=5.0)
    ap.add_argument("--already", type=float, default=0.14, help="dollars already spent on this step elsewhere")
    ap.add_argument("--stack", default="kyfromabove-3d-ec2-bench")
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--wait-ready", action="store_true", help="only wait until the machine has pulled the image")
    ap.add_argument("--gers-file", type=Path, default=None, help="buildings, one per line or space-separated")
    ap.add_argument("--cache", action="store_true", help="read the frames through the machine's nginx frame cache")
    ap.add_argument("--cache-fresh", action="store_true", help="start the frame cache empty first")
    ap.add_argument("--slice", default="1m")
    ap.add_argument("--env", nargs="*", default=[], help="KEY=VALUE settings for the building container")
    ap.add_argument("--print-script", action="store_true", help="print the shell script a run would send, and stop")
    args = ap.parse_args()

    if args.print_script:
        inst = {"InstanceId": "i-print", "InstanceType": "c8g.4xlarge", "Placement": {"AvailabilityZone": "-"},
                "State": {"Name": "-"}}
        spent, price, up_h = 0.0, 0.25, 0.0
    else:
        cf = boto3.client("cloudformation", region_name=args.region)
        outs = {o["OutputKey"]: o["OutputValue"] for o in cf.describe_stacks(StackName=args.stack)["Stacks"][0]["Outputs"]}
        ec2 = boto3.client("ec2", region_name=args.region)
        ssm = boto3.client("ssm", region_name=args.region)
        inst = instance(ec2, outs)
        spent, price, up_h = tally(ec2, inst, args.already)
    iid = inst["InstanceId"]
    print(f"{iid} ({inst['InstanceType']}, {inst['Placement']['AvailabilityZone']}, {inst['State']['Name']}): "
          f"up {up_h * 60:.0f} min at ${price:.4f}/h; spent so far ${spent:.3f} of ${args.cap:.2f}")
    if args.wait_ready:
        for _ in range(60):
            try:
                st, out, _ = ssm_run(ssm, iid, "test -f /var/tmp/bench-ready && echo ready || echo waiting", 60)
                if "ready" in out and "waiting" not in out:
                    print("ready")
                    return 0
            except Exception as exc:                                  # noqa: BLE001 (agent not up yet)
                print(f"  not reachable yet ({type(exc).__name__})")
            time.sleep(20)
        return 1
    if args.gers_file:
        args.gers += args.gers_file.read_text().split()
    expect = args.expect_min / 60 * price
    if spent + expect > args.cap:
        print(f"[STOP] ${spent:.3f} spent + ${expect:.3f} expected would pass the ${args.cap:.2f} cap")
        return 2

    name = f"bench-{args.label}"
    cache_env = "-e COG_CACHE_URL=http://127.0.0.1:8001 " if args.cache else ""
    cache_env += "".join(f"-e {kv} " for kv in args.env)
    cmd = (f"docker run --name {name} --network host --entrypoint python -w /var/task {cache_env}"
           f"-e OMP_NUM_THREADS={args.threads} -v /var/tmp/bench:/out building-step:bench "
           f"scripts/building_step.py --inputs s3://{BUCKET} --store s3://{BUCKET}/bench/ec2/{args.label} "
           f"--raw /opt/data/raw --work /tmp/work --workers {args.workers} --threads {args.threads} --force "
           f"--results /out/{args.label}.jsonl --tiles {' '.join(TILES)} --gers {' '.join(args.gers)}")
    fresh = [
        "docker rm -f kfa-cog-cache >/dev/null 2>&1; docker volume rm kfa-cog-cache >/dev/null 2>&1",
        ": > /var/tmp/cog_log/cog.log",
        f"docker run -d --name kfa-cog-cache --network host -e SLICE={args.slice} -e CACHE_MAX=20g "
        "-v /opt/cog_cache/nginx.conf.template:/etc/nginx/templates/cog.conf.template:ro "
        "-v kfa-cog-cache:/var/cache/nginx/cog -v /var/tmp/cog_log:/var/log/nginx "
        "public.ecr.aws/nginx/nginx:1.27-alpine >/dev/null",
        "for i in $(seq 30); do curl -s -o /dev/null -I http://127.0.0.1:8001/imagery/obliques/Phase3/"
        "KY_KYAPED_2024_Season1_3IN/Bwd_7022_46837.tif && break; sleep 1; done",
    ] if args.cache and args.cache_fresh else []
    stats = [
        "n0=$(cat /var/tmp/bench/log_start 2>/dev/null || echo 0)",
        "tail -n +$((n0 + 1)) /var/tmp/cog_log/cog.log | awk '{s[$6]++; if ($8 ~ /^[0-9]+$/) up += $8} "
        "END {printf \"CACHE requests=%d hit=%d miss=%d upstream_bytes=%d\\n\", NR, s[\"HIT\"], s[\"MISS\"], up}'",
    ] if args.cache else []
    script = "\n".join([
        "set -o pipefail", "mkdir -p /var/tmp/bench", *fresh,
        "wc -l < /var/tmp/cog_log/cog.log 2>/dev/null > /var/tmp/bench/log_start || echo 0 > /var/tmp/bench/log_start",
        f"docker rm -f {name} >/dev/null 2>&1 || true",
        "t0=$(date +%s)", f"{cmd} > /var/tmp/bench/{args.label}.log 2>&1; rc=$?", "t1=$(date +%s)",
        f"cid=$(docker inspect -f '{{{{.Id}}}}' {name})",
        "peak=$(cat /sys/fs/cgroup/system.slice/docker-$cid.scope/memory.peak "
        "/sys/fs/cgroup/docker/$cid/memory.peak 2>/dev/null | head -1)",
        "peak=${peak:-0}",
        f"docker rm {name} >/dev/null",
        f"aws s3 cp /var/tmp/bench/{args.label}.jsonl s3://{BUCKET}/bench/ec2/results/{args.label}.jsonl --quiet",
        f"aws s3 cp /var/tmp/bench/{args.label}.log s3://{BUCKET}/bench/ec2/results/{args.label}.log --quiet",
        f"grep -E '^\\[|building\\(s\\)' /var/tmp/bench/{args.label}.log | tail -60",
        *stats,
        'echo "RESULT rc=$rc seconds=$((t1-t0)) peak_bytes=$peak"'])
    if args.print_script:
        print(script)
        return 0
    print(f"run {args.label}: {len(args.gers)} building(s), {args.workers} at once (expect ~{args.expect_min:.0f} min)")
    st, out, err = ssm_run(ssm, iid, script)
    print(out[-6000:])
    if err.strip():
        print("stderr:", err[-1500:])
    local = Path("outputs/ec2_bench")
    local.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name=args.region)
    try:
        s3.download_file(BUCKET, f"bench/ec2/results/{args.label}.jsonl", str(local / f"{args.label}.jsonl"))
    except Exception as exc:                                          # noqa: BLE001
        print(f"no results file: {exc}")
    for ln in out.splitlines():
        if ln.startswith("RESULT"):
            kv = dict(p.split("=") for p in ln.split()[1:])
            cache = next((dict(p.split("=") for p in c.split()[1:]) for c in out.splitlines()
                          if c.startswith("CACHE ")), None)
            meta = {"label": args.label, "workers": args.workers, "threads": args.threads, "n": len(args.gers),
                    "env": args.env,
                    "cache": bool(args.cache), "slice": args.slice if args.cache else None,
                    "cache_stats": {k: int(v) for k, v in cache.items()} if cache else None,
                    "seconds": int(kv["seconds"]), "peak_gb": int(kv["peak_bytes"]) / 1e9, "rc": int(kv["rc"]),
                    "instance": inst["InstanceType"], "price_h": price}
            (local / f"{args.label}.meta.json").write_text(json.dumps(meta, indent=1))
            print(json.dumps(meta))
    spent, _, _ = tally(ec2, instance(ec2, outs), args.already)
    print(f"spent so far ${spent:.3f} of ${args.cap:.2f}")
    return 0 if st == "Success" else 1


if __name__ == "__main__":
    sys.exit(main())
