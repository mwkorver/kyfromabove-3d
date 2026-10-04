#!/usr/bin/env python3
"""Re-run build_mesh.py over every downtown building and re-merge the tileset.

The mesh step only: the colourise's outputs (colorized_aerial.laz,
wall_profiles.json, levels.json) are reused. It needs each building's frame
crops on disk (a run with ``--keep-crops``); build_mesh stops on a building
whose crops were deleted, because a mesh built without them takes no colour
from the frames. ``run_downtown.py run --force-step mesh`` re-cuts the crops
and is the way to rebuild after an ordinary run. Every failure is listed, and
the downtown tileset is merged only when every building built.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--buildings-dir", type=Path, default=ROOT / "outputs" / "downtown" / "buildings")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "downtown")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=300, help="seconds per building")
    args = ap.parse_args()

    building_dirs = [
        d for d in sorted(args.buildings_dir.iterdir())
        if d.is_dir() and (d / "3dtiles_mesh" / "tileset.json").exists()
    ]
    if args.limit > 0:
        building_dirs = building_dirs[:args.limit]

    print(f"Rebuilding {len(building_dirs)} downtown building meshes with {args.workers} workers...")
    t0 = time.time()

    def build_one(bdir: Path) -> tuple[str, str, float, dict]:
        """(name, error or "", seconds, the new tileset's extras)."""
        t_start = time.time()
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{SCRIPTS}:{ROOT / 'src'}"
        ts_file = bdir / "3dtiles_mesh" / "tileset.json"
        before = ts_file.stat().st_mtime if ts_file.exists() else 0.0
        try:
            r = subprocess.run([sys.executable, str(SCRIPTS / "build_mesh.py"), "--building", str(bdir)],
                               cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=args.timeout, env=env)
        except subprocess.TimeoutExpired:
            return bdir.name, f"timed out after {args.timeout} s", time.time() - t_start, {}
        dt = time.time() - t_start
        if r.returncode != 0:
            tail = r.stdout.strip().splitlines()[-1:] or ["(no output)"]
            return bdir.name, f"exit {r.returncode}: {tail[0]}", dt, {}
        # the extras of the tileset this run wrote -- not an older one left on disk
        if not ts_file.exists() or ts_file.stat().st_mtime <= before:
            return bdir.name, "build_mesh exited 0 but wrote no new tileset", dt, {}
        return bdir.name, "", dt, json.loads(ts_file.read_text(encoding="utf-8")).get("extras", {})

    counts: dict[str, int] = {}
    failures: list[tuple[str, str]] = []
    total_triangles = 0
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_one, d) for d in building_dirs]
        for fut in as_completed(futures):
            bname, err, dt, info = fut.result()
            done += 1
            if err:
                failures.append((bname, err))
                print(f"[{done}/{len(building_dirs)}] {bname} FAILED: {err}")
                continue
            arch = info.get("archetype") or "unlabelled"
            counts[arch] = counts.get(arch, 0) + 1
            total_triangles += int(info.get("triangles", 0))
            if done % 10 == 0 or done == len(building_dirs):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 0.1)
                eta = (len(building_dirs) - done) / max(rate, 0.01)
                print(f"[{done}/{len(building_dirs)}] {bname[:8]} {arch:<15} {dt:.0f}s ({rate:.1f} bldg/s, ETA {eta:.0f}s)")

    print(f"\nRemeshed {done - len(failures)} of {done} buildings in {time.time() - t0:.1f}s.")
    print("Structure labels:", counts)
    print(f"Total mesh triangles: {total_triangles:,}")
    if failures:
        print(f"\n{len(failures)} building(s) failed; the downtown tileset was NOT merged:")
        for bname, err in failures:
            print(f"  {bname}: {err}")
        return 1

    # Merge downtown root tileset
    print("\nMerging downtown root tileset...")
    from run_downtown import do_merge, landmark_gers
    landmarks = landmark_gers(ROOT / "configs" / "landmarks.yaml")
    do_merge(args.out, landmarks)

    # Also update S3 demo if script exists
    s3_demo_script = SCRIPTS / "export_s3_demo.py"
    if s3_demo_script.exists():
        print("\nUpdating S3 compare demo metadata...")
        subprocess.run([sys.executable, str(s3_demo_script)], cwd=str(ROOT))

    return 0


if __name__ == "__main__":
    sys.exit(main())
