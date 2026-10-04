#!/usr/bin/env python3
"""Move buildings stored the old way to the new layout.

The building step used to write about twenty files a building to
``buildings/<gers>/``. It now writes five objects to
``buildings/<version>/<gers>/`` (``kyfromabove_3d.building.pack_building``):
a zip per mesh product, ``review.zip``, ``levels.json`` and ``building.json``
last. This repacks the old ones with that same code, under the version their
own ``building.json`` records, so step 4 publishes them as before.

Three passes, each safe to repeat:

* no flag: count what is there, change nothing;
* ``--copy``: write each old building in the new layout (a building already
  there is skipped);
* ``--delete-old``: remove an old building's files, only where its copy
  lists every one of them; its ``building.json`` goes last, so a building
  half deleted is still found and finished on the next run.

Usage::

    python deploy/migrate_buildings.py --store s3://kyfromabove-3d
    python deploy/migrate_buildings.py --store s3://kyfromabove-3d --copy --workers 16
    python deploy/migrate_buildings.py --store s3://kyfromabove-3d --delete-old --workers 16
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kyfromabove_3d.building import LAYOUT, BUILDING_CODE_VERSION, buildings_prefix, pack_building  # noqa: E402
from kyfromabove_3d.store import store_from                                                         # noqa: E402


def old_buildings(store) -> Dict[str, List[str]]:
    """{gers: its files' paths inside its folder} for every building in the
    old layout: ``buildings/<gers>/building.json`` one level down, not under
    a version folder."""
    keys: Dict[str, List[str]] = {}
    for k in store.list("buildings/"):
        parts = k.split("/")
        if len(parts) >= 3:
            keys.setdefault(parts[1], []).append("/".join(parts[2:]))
    return {g: rels for g, rels in keys.items() if "building.json" in rels}


def copy_one(store, gers: str, rels: List[str]) -> str:
    old = store.read_json(f"buildings/{gers}/building.json")
    prefix = buildings_prefix(old.get("code_version") or BUILDING_CODE_VERSION)
    if store.exists(f"{prefix}/{gers}/building.json"):
        return "already copied"
    with tempfile.TemporaryDirectory() as tmp:
        b = Path(tmp)
        for rel in rels:
            if rel != "building.json":
                store.get_file(f"buildings/{gers}/{rel}", b / rel)
        objects, files = pack_building(b)
    left_out = sorted(set(rels) - set(files) - {"building.json"})
    if left_out:
        return f"not copied: {len(left_out)} file(s) the new layout does not keep, e.g. {left_out[0]}"
    summary = dict(old, files=files, bytes=int(sum(files.values())),
                   objects={k: len(v) for k, v in sorted(objects.items())}, layout=LAYOUT)
    for name in sorted(objects):
        store.put_bytes(f"{prefix}/{gers}/{name}", objects[name],
                        "application/zip" if name.endswith(".zip") else "application/json")
    store.put_json(f"{prefix}/{gers}/building.json", summary)
    return "copied"


def delete_one(store, gers: str, rels: List[str]) -> str:
    old = store.read_json(f"buildings/{gers}/building.json")
    prefix = buildings_prefix(old.get("code_version") or BUILDING_CODE_VERSION)
    key = f"{prefix}/{gers}/building.json"
    if not store.exists(key):
        return "kept: no copy yet"
    new = store.read_json(key)
    missing = [r for r in rels if r != "building.json" and r not in new.get("files", {})]
    if new.get("layout") != LAYOUT or missing:
        return f"kept: the copy lacks {len(missing)} file(s)"
    for rel in rels:
        if rel != "building.json":
            store.delete(f"buildings/{gers}/{rel}")
    store.delete(f"buildings/{gers}/building.json")
    return "deleted"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="a folder, or s3://bucket")
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--delete-old", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.copy and args.delete_old:
        raise SystemExit("--copy, check the copies, then --delete-old: not both at once")
    store = store_from(args.store)
    old = old_buildings(store)
    print(f"{len(old):,} building(s) in the old layout, {sum(map(len, old.values())):,} file(s)", flush=True)
    if not (args.copy or args.delete_old):
        return 0
    step = copy_one if args.copy else delete_one
    tally: Dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(lambda g: step(store, g, old[g]), sorted(old)), 1):
            kind = res.split(":")[0]
            tally[kind] = tally.get(kind, 0) + 1
            if kind not in ("copied", "deleted", "already copied") or i % 500 == 0 or i == len(old):
                print(f"[{i}/{len(old)}] {res}", flush=True)
    print(", ".join(f"{v:,} {k}" for k, v in sorted(tally.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
