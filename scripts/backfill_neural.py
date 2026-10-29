#!/usr/bin/env python3
"""Product 4 (the borrowed fill, ``3dtiles_neural``) for buildings built
before it existed, in a store: stored as a build with ``--neural`` stores it.

For each building with a mesh on ``--tiles`` and no ``3dtiles_neural.zip``:
its product 2 (``3dtiles_mesh.zip``: the glb and its tileset) and its
product 3's texels (``3dtiles_complete/fill_faces.npz``, in ``review.zip``)
are read, the unseen texels are filled by ``scripts/build_neural.py`` (the
same fill the build job runs), and the result is packed as
``building.pack_building`` packs it: ``3dtiles_neural.zip`` (tileset, glb,
lossless texture pages), the tinted copy added to ``review.zip``, and
``building.json`` updated last. Products 2 and 3 are not touched.

Then publish the ``neural`` and ``neural_tint`` views of those tiles and
their trees (``publish_step.py --products neural neural_tint --tree``).

Usage::

    python scripts/backfill_neural.py --store s3://kyfromabove-3d --tiles N076E228 N076E229 --dry-run
    python scripts/backfill_neural.py --store s3://kyfromabove-3d --tiles N076E228 N076E229 --workers 8
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kyfromabove_3d.archive import read_zip, zip_bytes                       # noqa: E402
from kyfromabove_3d.building import BUILDINGS_PREFIX, REVIEW_ZIP            # noqa: E402
from kyfromabove_3d.publish import built_buildings                          # noqa: E402
from kyfromabove_3d.store import store_from                                 # noqa: E402

NEURAL = "3dtiles_neural"


def backfill(gers: str, store_spec: str, prefix: str = BUILDINGS_PREFIX) -> dict:
    """One building's product 4, made and stored (see the module)."""
    from build_neural import build_building_neural

    store = store_from(store_spec)
    base = f"{prefix}/{gers}"
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        b = Path(tmp) / gers
        mesh = read_zip(store, f"{base}/3dtiles_mesh.zip")
        ts = json.loads(mesh.read("tileset.json"))
        (b / "3dtiles_mesh").mkdir(parents=True)
        (b / "3dtiles_mesh" / "tileset.json").write_bytes(mesh.read("tileset.json"))
        (b / "3dtiles_mesh" / "building.glb").write_bytes(mesh.read(ts["root"]["content"]["uri"]))
        review_bytes = store.get_bytes(f"{base}/{REVIEW_ZIP}")
        review = zipfile.ZipFile(io.BytesIO(review_bytes))
        (b / "3dtiles_complete").mkdir()
        (b / "3dtiles_complete" / "fill_faces.npz").write_bytes(review.read("3dtiles_complete/fill_faces.npz"))

        res = build_building_neural(b, texture_format="ktx2", dump_textures=True)

        out = b / NEURAL
        product = {p.relative_to(out).as_posix(): p for p in sorted(out.rglob("*")) if p.is_file()}
        in_zip = {k: v for k, v in product.items() if k in ("tileset.json", "building.glb") or k.startswith("textures/")}
        rest = {f"{NEURAL}/{k}": v for k, v in product.items() if k not in in_zip}
        members = {n: review.read(n) for n in review.namelist() if not n.startswith(f"{NEURAL}/")}
        members.update({n: p.read_bytes() for n, p in rest.items()})
        zipped = zip_bytes(in_zip)
        new_review = zip_bytes(members)

        summary = store.read_json(f"{base}/building.json")
        files = {k: v for k, v in summary.get("files", {}).items() if not k.startswith(f"{NEURAL}/")}
        files.update({f"{NEURAL}/{k}": p.stat().st_size for k, p in product.items()})
        summary["files"] = dict(sorted(files.items()))
        summary["bytes"] = int(sum(files.values()))
        objects = dict(summary.get("objects", {}))
        objects[f"{NEURAL}.zip"] = len(zipped)
        objects[REVIEW_ZIP] = len(new_review)
        summary["objects"] = dict(sorted(objects.items()))
        summary["neural_backfill"] = {"seconds": res["seconds"], "fill": res["fill"]}

        store.put_bytes(f"{base}/{NEURAL}.zip", zipped, "application/zip")
        store.put_bytes(f"{base}/{REVIEW_ZIP}", new_review, "application/zip")
        store.put_json(f"{base}/building.json", summary)                       # last: the checkpoint
    return {"gers": gers, "name": res["name"], "seconds": round(time.time() - t0, 1),
            "fill": res["fill"], "bytes": len(zipped)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="s3://bucket or a folder")
    ap.add_argument("--tiles", nargs="*", default=[])
    ap.add_argument("--tiles-file", type=Path, default=None)
    ap.add_argument("--gers", nargs="*", default=[], help="only these buildings")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true", help="list what would be made, and the bytes to read")
    args = ap.parse_args()
    tiles = list(args.tiles) + (args.tiles_file.read_text().split() if args.tiles_file else [])
    store = store_from(args.store)
    available = built_buildings(store, BUILDINGS_PREFIX)
    todo = []
    for t in tiles:
        for r in store.read_json(f"tiles/{t}/buildings.json")["buildings"]:
            have = available.get(r["id"], set())
            if "3dtiles_mesh" in have and NEURAL not in have and (not args.gers or r["id"] in args.gers):
                todo.append(r["id"])
    print(f"{len(todo):,} building(s) with a mesh and no borrowed fill on {len(tiles)} tile(s)", flush=True)
    if args.dry_run:
        size = sum(store.size(f"{BUILDINGS_PREFIX}/{g}/{n}") for g in todo for n in ("3dtiles_mesh.zip", REVIEW_ZIP))
        print(f"would read {size / 1e9:.2f} GB")
        return 0
    t0, done, failed = time.time(), 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(backfill, g, args.store): g for g in todo}
        for f in as_completed(futs):
            try:
                r = f.result()
                done += 1
                if done % 50 == 0 or done == len(todo):
                    print(f"[{done}/{len(todo)}] {r['name']}: {r['seconds']} s, {time.time() - t0:.0f} s so far", flush=True)
            except Exception as e:                                             # noqa: BLE001
                failed.append((futs[f], repr(e)))
                print(f"[fail] {futs[f]}: {e!r}", flush=True)
    print(f"done: {done} made, {len(failed)} failed, in {time.time() - t0:.0f} s")
    if failed:
        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/backfill_neural_failed.json").write_text(json.dumps(failed, indent=1), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
