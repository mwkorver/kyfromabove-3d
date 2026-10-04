#!/usr/bin/env python3
"""Cut a batch of neighbouring buildings' crops direct from S3 and through
the frame cache (``deploy/cog_cache``), and compare: seconds a building,
the cache's hits and misses, the bytes fetched from S3, and whether every
crop is byte-for-byte the same.

Each building is cut in a fresh process, as each building step's scripts run
on a machine, so nothing is shared but the cache. The cache must be running
(``sh deploy/cog_cache/run_local.sh <slice>``); ``--fresh`` restarts it
empty with ``--slice`` first.

Usage::

    python deploy/cog_cache/bench.py --pieces outputs/step2_local/pieces --tile-json outputs/step1_local_v3/tiles/N076E229/buildings.json \\
        --n 20 --label direct
    python deploy/cog_cache/bench.py ... --label s1m_cold --cache --fresh --slice 1m
    python deploy/cog_cache/bench.py ... --label s1m_warm --cache
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CUT = r'''
import json, sys, time, shutil
from pathlib import Path
sys.path.insert(0, "src")
from kyfromabove_3d.building import chosen_from_piece
from kyfromabove_3d.prepare import cut_crops
piece, out = Path(sys.argv[1]), Path(sys.argv[2])
shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
for f in piece.iterdir():
    if f.is_file(): shutil.copy(f, out / f.name)
t = time.time(); ch = chosen_from_piece(out); res = {}
if ch is not None:
    cut_crops(ch, out, res, log=lambda m: None)
print(json.dumps({"seconds": round(time.time() - t, 2), "frames": 0 if ch is None else len(ch["cams"])}))
'''


def log_lines(path: Path):
    return path.read_text().splitlines() if path.exists() else []


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from kyfromabove_3d.pieces import batches

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pieces", type=Path, required=True)
    ap.add_argument("--tile-json", type=Path, required=True, help="the tile step's buildings.json (for neighbours)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--batch-index", type=int, default=3, help="which batch of neighbours (a downtown one)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--cache", action="store_true", help="read through http://127.0.0.1:8001")
    ap.add_argument("--fresh", action="store_true", help="restart the cache empty first")
    ap.add_argument("--slice", default="1m")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "cog_cache" / "runs")
    args = ap.parse_args()

    recs = [r for r in json.loads(args.tile_json.read_text())["buildings"] if (args.pieces / r["id"] / "piece.json").exists()]
    gers = batches(recs, args.n)[args.batch_index]
    log = ROOT / "outputs" / "cog_cache" / "log" / "cog.log"
    if args.fresh:
        subprocess.run(["docker", "rm", "-f", "kfa-cog-cache"], capture_output=True)
        subprocess.run(["docker", "volume", "rm", "kfa-cog-cache"], capture_output=True)
        subprocess.run(["sh", str(ROOT / "deploy" / "cog_cache" / "run_local.sh"), args.slice], check=True)
        time.sleep(2)
    if args.cache:                                   # the cache must answer before anything is timed
        import urllib.request
        for _ in range(20):
            try:
                urllib.request.urlopen(urllib.request.Request(
                    "http://127.0.0.1:8001/imagery/obliques/Phase3/KY_KYAPED_2024_Season1_3IN/Bwd_7022_46837.tif",
                    method="HEAD"), timeout=10)
                break
            except OSError:
                time.sleep(1)
        else:
            raise SystemExit("[FAIL] the frame cache is not answering on 127.0.0.1:8001 (docker logs kfa-cog-cache)")
    start = len(log_lines(log))
    env = dict(os.environ)
    env.pop("COG_CACHE_URL", None)
    if args.cache:
        env["COG_CACHE_URL"] = "http://127.0.0.1:8001"
    out = args.out / args.label
    shutil.rmtree(out, ignore_errors=True)
    t0 = time.time()
    per = []
    for g in gers:
        r = subprocess.run([sys.executable, "-c", CUT, str(args.pieces / g), str(out / g)], cwd=ROOT, env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{g}: FAILED {r.stderr[-400:]}")
            continue
        per.append({"gers": g, **json.loads(r.stdout.strip().splitlines()[-1])})
    wall = time.time() - t0
    lines = log_lines(log)[start:]
    status = {}
    upstream = 0
    for ln in lines:
        f = ln.split()
        st = f[-4] if len(f) >= 4 else "?"
        status[st] = status.get(st, 0) + 1
        if f[-2].isdigit():
            upstream += int(f[-2])
    summary = {"label": args.label, "cache": args.cache, "slice": args.slice if args.cache else None,
               "buildings": len(per), "frames": sum(p["frames"] for p in per), "wall_s": round(wall, 1),
               "crop_s_mean": round(sum(p["seconds"] for p in per) / max(len(per), 1), 2),
               "requests_to_cache": len(lines), "cache_status": status, "bytes_from_s3": upstream, "per": per}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{args.label}: {len(per)} buildings, {summary['frames']} frames, {wall:.0f} s, "
          f"{summary['crop_s_mean']:.2f} s a building" + (f"; {len(lines)} requests {status}, "
          f"{upstream / 1e6:.1f} MB from S3" if args.cache else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
