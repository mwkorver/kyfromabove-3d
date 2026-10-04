#!/usr/bin/env python3
"""Build a list of buildings on the Spot machines of ``Ec2BenchStack``
(deployed with ``-c bench_count=N``), under a spending cap, then run step 4,
the statewide tree and the terrain on the first machine.

Each machine gets a share of the tiles (whole tiles, so neighbours share
frames and LiDAR) and runs ``scripts/building_step.py --workers 16`` inside
the pipeline's image, reading pieces from and writing ``buildings/<gers>/``
to ``s3://kyfromabove-3d``. Every few minutes the spend is tallied -- the
machines' hours at their zones' Spot prices, their disks, and an estimate of
the S3 writes so far -- with ``--already`` (the pilot's spend before this),
and both machines are terminated if the tally reaches ``--cap``.

Usage::

    python deploy/run_spot_build.py --groups "N077E235 N078E236" "N076E228 N076E229 ..." --already 1.82
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ec2_bench import EBS_GB, EBS_GB_MONTH, ssm_run                          # noqa: E402

BUCKET = "kyfromabove-3d"
PUT_USD = 0.005 / 1000
FILES_PER_BUILDING = 22
STALL_MIN = 8
ATTEMPTS = 6


def machines(region: str, stack: str):
    import boto3

    cf = boto3.client("cloudformation", region_name=region)
    outs = {o["OutputKey"]: o["OutputValue"] for o in cf.describe_stacks(StackName=stack)["Stacks"][0]["Outputs"]}
    ids = outs.get("InstanceIds", outs["InstanceId"]).split(",")
    ec2 = boto3.client("ec2", region_name=region)
    # only the machines still up: Spot can take one back mid-run
    live = [i["InstanceId"] for r in ec2.describe_instances(InstanceIds=ids)["Reservations"] for i in r["Instances"]
            if i["State"]["Name"] in ("pending", "running")]
    for i in ids:
        if i not in live:
            print(f"[warn] {i} is no longer running (its spend so far belongs in --already)", flush=True)
    return live, ec2


def spend(ec2, ids, built: int) -> tuple:
    """(dollars so far on the machines and writes, per-machine detail)."""
    now = datetime.datetime.now(datetime.timezone.utc)
    usd, detail = 0.0, []
    for inst in ec2.describe_instances(InstanceIds=ids)["Reservations"]:
        for i in inst["Instances"]:
            h = (now - i["LaunchTime"]).total_seconds() / 3600
            p = ec2.describe_spot_price_history(InstanceTypes=[i["InstanceType"]], ProductDescriptions=["Linux/UNIX"],
                                                AvailabilityZone=i["Placement"]["AvailabilityZone"],
                                                StartTime=now)["SpotPriceHistory"]
            price = float(p[0]["SpotPrice"]) if p else 0.35
            usd += h * price + h * EBS_GB * EBS_GB_MONTH / 730
            detail.append((i["InstanceId"], i["State"]["Name"], round(h, 2), price))
    usd += built * FILES_PER_BUILDING * PUT_USD
    return usd, detail


def built_so_far(s3) -> int:
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix="bench/pilot/progress/"):
        for o in page.get("Contents", []):
            n += int(o["Key"].rsplit("/", 1)[1].split("_")[1])
    return n


def main() -> int:
    import boto3

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="*", default=[], help="one quoted, space-separated tile list per machine")
    ap.add_argument("--all-tiles", type=Path, required=True, help="every tile to publish (boxes and meshes)")
    ap.add_argument("--cap", type=float, default=10.0)
    ap.add_argument("--already", type=float, required=True)
    ap.add_argument("--stack", default="kyfromabove-3d-ec2-bench")
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--check-min", type=float, default=5.0)
    ap.add_argument("--resume-after", default=None, metavar="ISO_UTC",
                    help="don't start the builds: watch ones started after this time (e.g. when the laptop's "
                         "login lapsed) until each machine's log lands in S3, then publish")
    ap.add_argument("--code-from-laptop", action="store_true",
                    help="run this checkout's src/kyfromabove_3d and scripts/ (mounted over the image's copies) "
                         "instead of rebuilding the image for a small fix")
    ap.add_argument("--publish-only", action="store_true",
                    help="no building step: step 4 and the tree on the first machine (e.g. a new box format)")
    ap.add_argument("--no-terrain", action="store_true", help="leave the published terrain as it is")
    args = ap.parse_args()
    if not args.groups and not args.publish_only:
        raise SystemExit("--groups, or --publish-only")
    ids, ec2 = machines(args.region, args.stack)
    if len(ids) < len(args.groups):
        raise SystemExit(f"{len(ids)} machine(s) for {len(args.groups)} group(s)")
    s3 = boto3.client("s3", region_name=args.region)
    ssm = boto3.client("ssm", region_name=args.region)
    s3.put_object(Bucket=BUCKET, Key="bench/pilot/all_tiles.txt", Body=args.all_tiles.read_bytes())
    fetch_code, mounts = "true", ""
    if args.code_from_laptop:
        import io
        import tarfile

        root = Path(__file__).resolve().parents[1]
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for f in sorted((root / "src" / "kyfromabove_3d").glob("*.py")):
                tar.add(f, arcname=f"pkg/{f.name}")
            for f in sorted((root / "scripts").glob("*.py")):
                tar.add(f, arcname=f"scripts/{f.name}")
        s3.put_object(Bucket=BUCKET, Key="bench/pilot/code.tgz", Body=buf.getvalue())
        fetch_code = (f"rm -rf /var/tmp/pilot/code && mkdir -p /var/tmp/pilot/code && aws s3 cp s3://{BUCKET}/"
                      "bench/pilot/code.tgz - --quiet | tar -xz -C /var/tmp/pilot/code")
        mounts = ("-v /var/tmp/pilot/code/pkg:/var/lang/lib/python3.12/site-packages/kyfromabove_3d:ro "
                  "-v /var/tmp/pilot/code/scripts:/var/task/scripts:ro ")
        print("code: this checkout's package and scripts, mounted over the image's", flush=True)
    groups = [g.split() for g in args.groups]
    if not args.resume_after and not args.publish_only:
        s3.delete_objects(Bucket=BUCKET, Delete={"Objects": [
            {"Key": o["Key"]} for o in s3.list_objects_v2(Bucket=BUCKET, Prefix="bench/pilot/progress/").get("Contents", [])
        ] or [{"Key": "bench/pilot/progress/none"}]})
    for i, tiles in enumerate([] if (args.resume_after or args.publish_only) else groups):
        gers = [r["id"] for t in tiles for r in json.loads(s3.get_object(Bucket=BUCKET, Key=f"tiles/{t}/buildings.json")
                                                             ["Body"].read())["buildings"]]
        s3.put_object(Bucket=BUCKET, Key=f"bench/pilot/gers_{i}.txt", Body=" ".join(gers).encode())
        print(f"machine {i} ({ids[i]}): {len(tiles)} tiles, {len(gers):,} buildings", flush=True)

    def build(i: int, tiles):
        # The results go up every 5 minutes (progress counts for the tally) and at the end.
        # The frames are read through the machine's nginx frame cache at a numeric address,
        # so GDAL's libcurl never looks up a host name: libcurl 8.16 cancels its lookup
        # threads, which can leave glibc's getaddrinfo lock held and hang every later
        # connection in that process (seen September 27: all 16 workers stuck in __check_pf).
        # A build that stalls for STALL_MIN anyway is killed and started again; the
        # buildings already published are skipped.
        run = ("docker run --name pilot --network host --entrypoint python -w /var/task -e OMP_NUM_THREADS=1 "
               "-e COG_CACHE_URL=http://127.0.0.1:8001 " + mounts +
               "-v /var/tmp/pilot:/out building-step:bench scripts/building_step.py "
               f"--inputs s3://{BUCKET} --store s3://{BUCKET} --gers-file /out/gers.txt --tiles {' '.join(tiles)} "
               "--raw /opt/data/raw --work /tmp/work --workers 16 --threads 1 --results /out/results.jsonl "
               ">> /var/tmp/pilot/log.txt 2>&1")
        script = "\n".join([
            "set -o pipefail", "mkdir -p /var/tmp/pilot /var/tmp/cog_log", fetch_code,
            f"aws s3 cp s3://{BUCKET}/bench/pilot/gers_{i}.txt /var/tmp/pilot/gers.txt --quiet",
            "docker rm -f pilot >/dev/null 2>&1 || true", "rm -f /var/tmp/pilot/results.jsonl /var/tmp/pilot/log.txt",
            "docker inspect kfa-cog-cache >/dev/null 2>&1 || docker run -d --name kfa-cog-cache --network host "
            "-e SLICE=1m -e CACHE_MAX=20g -v /opt/cog_cache/nginx.conf.template:/etc/nginx/templates/cog.conf.template:ro "
            "-v kfa-cog-cache:/var/cache/nginx/cog -v /var/tmp/cog_log:/var/log/nginx "
            "public.ecr.aws/nginx/nginx:1.27-alpine >/dev/null",
            "for k in $(seq 30); do curl -s -o /dev/null -f -r 0-99 http://127.0.0.1:8001/imagery/obliques/Phase3/"
            "KY_KYAPED_2024_Season1_3IN/Bwd_7022_46837.tif && break; sleep 1; done",
            "curl -s -o /dev/null -f -r 0-99 http://127.0.0.1:8001/imagery/obliques/Phase3/"
            "KY_KYAPED_2024_Season1_3IN/Bwd_7022_46837.tif || { echo 'RESULT the frame cache does not answer'; exit 1; }",
            "( last=-1; still=0; while sleep 60; do n=$(wc -l < /var/tmp/pilot/results.jsonl 2>/dev/null || echo 0); "
            f"if [ \"$n\" = \"$last\" ]; then still=$((still + 1)); else still=0; last=$n; fi; "
            f"if [ $still -ge {STALL_MIN} ]; then echo \"[stall] $n results, none for {STALL_MIN} min: restarting\" "
            ">> /var/tmp/pilot/log.txt; docker kill pilot >/dev/null 2>&1; still=0; fi; done ) & D=$!",
            "( while sleep 300; do n=$(wc -l < /var/tmp/pilot/results.jsonl 2>/dev/null || echo 0); "
            f"aws s3 cp /var/tmp/pilot/results.jsonl s3://{BUCKET}/bench/pilot/results_{i}.jsonl --quiet 2>/dev/null; "
            f"aws s3 rm s3://{BUCKET}/bench/pilot/progress/ --recursive --exclude '*' --include 'm{i}_*' --quiet; "
            f"echo | aws s3 cp - s3://{BUCKET}/bench/pilot/progress/m{i}_$n --quiet; done ) & W=$!",
            "t0=$(date +%s)",
            f"for attempt in $(seq {ATTEMPTS}); do docker rm -f pilot >/dev/null 2>&1; {run}; rc=$?; "
            "[ $rc -eq 0 ] && break; echo \"[attempt $attempt] exit $rc\" >> /var/tmp/pilot/log.txt; done",
            "kill $W $D 2>/dev/null",
            f"aws s3 cp /var/tmp/pilot/results.jsonl s3://{BUCKET}/bench/pilot/results_{i}.jsonl --quiet",
            f"aws s3 cp /var/tmp/pilot/log.txt s3://{BUCKET}/bench/pilot/log_{i}.txt --quiet",
            "n=$(wc -l < /var/tmp/pilot/results.jsonl)",
            f"aws s3 rm s3://{BUCKET}/bench/pilot/progress/ --recursive --exclude '*' --include 'm{i}_*' --quiet",
            f"echo | aws s3 cp - s3://{BUCKET}/bench/pilot/progress/m{i}_$n --quiet",
            "grep -c ' built ' /var/tmp/pilot/log.txt; grep -c 'FAILED' /var/tmp/pilot/log.txt",
            'echo "RESULT rc=$rc seconds=$(( $(date +%s) - t0 )) buildings=$n"'])
        st, out, err = ssm_run(ssm, ids[i], script, timeout_s=6 * 3600)
        results[i] = (st, out[-1500:], err[-800:])

    results: dict = {}
    t0 = time.time()
    if args.publish_only:
        def running() -> bool:
            return False
    elif args.resume_after:
        since = datetime.datetime.fromisoformat(args.resume_after).astimezone(datetime.timezone.utc)

        def running() -> bool:
            for i in range(len(groups)):
                try:
                    head = s3.head_object(Bucket=BUCKET, Key=f"bench/pilot/log_{i}.txt")
                except s3.exceptions.ClientError:
                    return True
                if head["LastModified"] <= since:
                    return True
            return False
    else:
        threads = [threading.Thread(target=build, args=(i, g), daemon=True) for i, g in enumerate(groups)]
        for th in threads:
            th.start()

        def running() -> bool:
            return any(th.is_alive() for th in threads)
    stopped = False
    while running():
        time.sleep(args.check_min * 60)
        try:
            n = built_so_far(s3)
            usd, detail = spend(ec2, ids, n)
        except Exception as exc:                                     # noqa: BLE001 (a lapsed login: keep watching)
            print(f"[warn] spend check failed ({type(exc).__name__}); the machines' own shutdown still stands",
                  flush=True)
            s3, ec2 = boto3.client("s3", region_name=args.region), boto3.client("ec2", region_name=args.region)
            continue
        total = args.already + usd
        print(f"[{(time.time() - t0) / 60:5.0f} min] {n:,} built; this run ${usd:.2f}, pilot ${total:.2f} of "
              f"${args.cap:.2f}; {detail}", flush=True)
        if total >= args.cap - 0.25 and not stopped:
            print("[STOP] the cap is within 25 cents: terminating the machines", flush=True)
            ec2.terminate_instances(InstanceIds=ids)
            stopped = True
            break
    if not (args.resume_after or args.publish_only):
        for th in threads:
            th.join(timeout=5)
    for i in range(0 if args.publish_only else len(groups)):
        tail = s3.get_object(Bucket=BUCKET, Key=f"bench/pilot/log_{i}.txt")["Body"].read().decode().splitlines()
        print(f"machine {i}: {sum(' built ' in l for l in tail)} built, {sum('FAILED' in l for l in tail)} failed; "
              f"{tail[-1] if tail else ''}", flush=True)
    for i, r in sorted(results.items()):
        print(f"machine {i}: {r[0]}\n{r[1]}\n{r[2]}", flush=True)
    if stopped:
        return 2

    # step 4, the tree and the terrain, on the first machine
    terrain = "" if args.no_terrain else " --terrain"
    script = "\n".join([
        "mkdir -p /var/tmp/pilot",
        f"aws s3 cp s3://{BUCKET}/bench/pilot/all_tiles.txt /var/tmp/pilot/all_tiles.txt --quiet", fetch_code,
        "docker run --rm --network host --entrypoint python -w /var/task " + mounts +
        "-v /var/tmp/pilot:/out building-step:bench "
        f"scripts/publish_step.py --inputs s3://{BUCKET} --tiles-file /out/all_tiles.txt --raw /opt/data/raw --workers 16 --tree" + terrain + " "
        "> /var/tmp/pilot/publish.txt 2>&1; rc=$?",
        f"aws s3 cp /var/tmp/pilot/publish.txt s3://{BUCKET}/bench/pilot/publish.txt --quiet",
        "tail -8 /var/tmp/pilot/publish.txt", 'echo "RESULT rc=$rc"'])
    st, out, err = ssm_run(ssm, ids[0], script, timeout_s=3 * 3600)
    print("publish:", st, out[-2500:], err[-600:], flush=True)
    # a publish-only run writes no buildings: stale progress markers would count as writes
    usd, detail = spend(ec2, ids, 0 if args.publish_only else built_so_far(s3))
    print(f"done: this run ${usd:.2f}, pilot ${args.already + usd:.2f} of ${args.cap:.2f}; {detail}", flush=True)
    return 0 if st == "Success" else 1


if __name__ == "__main__":
    sys.exit(main())
