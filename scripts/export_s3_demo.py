#!/usr/bin/env python3
"""Export static metadata and terrain tiles for the standalone S3 compare demo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from ground_dem import GroundDEM
from intake_server import frame_records, frame_url

DEST = ROOT / "viewers" / "s3_compare"


def export_terrain_tiles(dem_path: Path, out_dir: Path, min_level: int = 0, max_level: int = 16):
    """Export static 32x32 float heightmap tiles covering downtown Louisville."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dem = GroundDEM(dem_path)

    # Downtown Louisville bounds
    # bbox: min_lon, min_lat, max_lon, max_lat
    west, south, east, north = -85.775, 38.240, -85.740, 38.265

    tile_count = 0
    for level in range(min_level, max_level + 1):
        n_x, n_y = 2 << level, 1 << level
        dlon, dlat = 360.0 / n_x, 180.0 / n_y

        min_tx = max(0, int((west + 180.0) / dlon))
        max_tx = min(n_x - 1, int((east + 180.0) / dlon))
        min_ty = max(0, int((90.0 - north) / dlat))
        max_ty = min(n_y - 1, int((90.0 - south) / dlat))

        for tx in range(min_tx, max_tx + 1):
            for ty in range(min_ty, max_ty + 1):
                data = dem.tile(tx, ty, level, w=32, h=32)
                tile_file = out_dir / f"{level}_{tx}_{ty}.bin"
                tile_file.write_bytes(data)
                tile_count += 1

    print(f"Exported {tile_count} terrain tiles (levels {min_level}..{max_level}) to {out_dir}")


def export_building_frames(out_dir: Path, copy_crops: bool = False, precut: bool = False):
    """Export static frames.json for each building with crop_manifest.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir.parent / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    bdir = ROOT / "outputs" / "downtown" / "buildings"

    count = 0
    crops_count = 0
    lookup = {}
    for p in bdir.iterdir():
        if not p.is_dir():
            continue
        gers = p.name
        mf = p / "crop_manifest.json"
        if not mf.exists():
            continue

        rec = frame_records(gers)
        if not rec:
            continue

        if precut:
            print(f"Pre-cutting crops for {gers} ({rec.get('name', gers[:8])})...")
            from intake_server import cut_crop
            for i in range(len(rec.get("frames", []))):
                cut_crop(gers, i, 320)
                cut_crop(gers, i, 1600)

        # Check if pre-cut crops exist in crops_cache
        cc = p / "crops_cache"
        has_crops = False
        if cc.exists() and any(cc.iterdir()):
            target_cc = crops_dir / gers
            if copy_crops:
                import shutil
                if not target_cc.exists():
                    shutil.copytree(cc, target_cc)
            else:
                if not target_cc.exists():
                    try:
                        target_cc.symlink_to(cc.resolve(), target_is_directory=True)
                    except Exception as e:
                        print(f"[warn] Failed to symlink crops for {gers}: {e}")
            has_crops = True
            crops_count += 1
        rec["has_crops"] = has_crops

        # Add S3 URLs and pixel windows to each frame
        raw_manifest = json.loads(mf.read_text(encoding="utf-8"))
        for i, f in enumerate(rec.get("frames", [])):
            if i < len(raw_manifest):
                m = raw_manifest[i]
                f["s3_url"] = frame_url(m["frame_id"])
                f["x0_px"] = int(m["x0_px"])
                f["y0_px"] = int(m["y0_px"])
                f["width_px"] = int(m["width_px"])
                f["height_px"] = int(m["height_px"])

        out_file = out_dir / f"{gers}.json"
        out_file.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        lookup[gers] = {
            "name": rec.get("name"),
            "height_m": rec.get("height_m"),
            "n_frames": len(rec.get("frames", [])),
            "ring_lonlat": rec.get("ring_lonlat"),
            "has_crops": has_crops,
        }
        count += 1

    (out_dir.parent / "buildings_index.json").write_text(json.dumps(lookup, indent=1), encoding="utf-8")
    print(f"Exported frames for {count} buildings ({crops_count} with static crops) to {out_dir}")


#: the viewer's tilesets/: each product's tileset directory, linked from outputs/
#: (``aws s3 sync --follow-symlinks`` uploads what they point at)
TILESET_LINKS = {"downtown": "outputs/downtown", "downtown_complete": "outputs/downtown_complete",
                 "downtown_complete_tint": "outputs/downtown_complete_tint", "overture_boxes": "outputs/overture_boxes"}


def link_tilesets(dest: Path) -> None:
    """Link every product's tileset directory into ``dest/tilesets`` (the ones
    built): products 1, 2 and 3 of docs/PRODUCTS.md."""
    import os
    d = dest / "tilesets"
    d.mkdir(parents=True, exist_ok=True)
    for name, rel in TILESET_LINKS.items():
        target = ROOT / rel
        link = d / name
        if not target.exists() or link.exists() or link.is_symlink():
            continue
        link.symlink_to(os.path.relpath(target, d), target_is_directory=True)
        print(f"linked tilesets/{name} -> {rel}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export static metadata, terrain, and crops for S3 demo")
    parser.add_argument("--skip-terrain", action="store_true", help="Skip re-generating terrain tiles")
    parser.add_argument("--copy-crops", action="store_true", help="Copy crops rather than symlinking (for S3 upload)")
    parser.add_argument("--precut", action="store_true", help="Pre-cut all missing crops using rasterio/intake_server")
    args = parser.parse_args()

    print(f"Preparing static assets in {DEST}...")
    link_tilesets(DEST)
    dem_path = ROOT / "outputs" / "downtown" / "ground_dem.npz"
    if not args.skip_terrain:
        if dem_path.exists():
            export_terrain_tiles(dem_path, DEST / "data" / "terrain")
        else:
            print(f"[warn] {dem_path} not found; skipping terrain tiles")
    else:
        print("Skipping terrain tiles (--skip-terrain).")

    export_building_frames(DEST / "data" / "frames", copy_crops=args.copy_crops, precut=args.precut)
    print("Static export complete.")


if __name__ == "__main__":
    main()
