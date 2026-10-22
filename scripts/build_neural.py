#!/usr/bin/env python3
"""Build Product 4 (``3dtiles_neural/``) for one or more already-built buildings.

Reads ``<building>/3dtiles_mesh/{building.glb,tileset.json}`` and
``<building>/3dtiles_complete/fill_faces.npz``, fills every unseen facade and
roof texel from the best-matching seen patches on the building's walls
(:mod:`kyfromabove_3d.neural_fill`), and writes:

* ``<building>/3dtiles_neural/building.glb``
* ``<building>/3dtiles_neural/building_tint.glb`` (neural-filled texels tinted violet)
* ``<building>/3dtiles_neural/tileset.json``
* ``<building>/3dtiles_neural/tileset_tint.json``

For buildings already built with ``build_mesh.py --complete``: refills
them without re-running the colourise or the mesh (``build_mesh.py --neural``
does the same during a build).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kyfromabove_3d.facade_fill import KNOWN, NAMES, shares
from kyfromabove_3d.mesh.gltf import TEXTURE_FORMATS, replace_glb_images
from kyfromabove_3d.mesh.textures import Texture, TextureAtlas
from kyfromabove_3d.neural_fill import complete_textures_neural, neural_fill, reference_bank


def build_building_neural(
    b: Path,
    texture_format: str = "ktx2",
    dump_textures: bool = False,
) -> dict:
    t0 = time.time()
    mesh_glb = b / "3dtiles_mesh" / "building.glb"
    mesh_ts = b / "3dtiles_mesh" / "tileset.json"
    npz_path = b / "3dtiles_complete" / "fill_faces.npz"
    if not mesh_glb.exists() or not mesh_ts.exists() or not npz_path.exists():
        raise FileNotFoundError(
            f"{b}: needs 3dtiles_mesh/{{building.glb,tileset.json}} and 3dtiles_complete/fill_faces.npz"
        )

    z = np.load(npz_path, allow_pickle=False)
    metas = json.loads(str(z["meta"]))
    texel_m = float(z["texel_m"])

    wall_imgs: list[Texture] = []
    wall_metas: list[dict] = []
    roof_tex: Texture | None = None

    for i, m in enumerate(metas):
        if f"img_{i}" not in z or f"src_{i}" not in z:
            continue
        ti = Texture(z[f"img_{i}"], z[f"src_{i}"])
        if m.get("kind") == "roof":
            roof_tex = ti
        else:
            wall_imgs.append(ti)
            wall_metas.append(m)

    out = b / "3dtiles_neural"
    out.mkdir(parents=True, exist_ok=True)

    ref_sources = reference_bank(wall_imgs, wall_metas, texel_m)
    done = complete_textures_neural(wall_imgs, wall_metas, texel_m, ref_sources=ref_sources)
    roof = None
    if roof_tex is not None and getattr(roof_tex, "source", None) is not None:
        roof = neural_fill(
            np.asarray(roof_tex),
            roof_tex.source,
            texel_m,
            kind="roof",
            ref_sources=ref_sources,
        )
    fill = shares([sv for _, sv in done] + ([roof[1]] if roof else []))

    def pages_of(images: list[np.ndarray]) -> list[np.ndarray]:
        a = TextureAtlas(4096, 2)
        for im in images:
            a.add(im)
        return a.pack() if images else []

    def tint(img: np.ndarray, src: np.ndarray | None) -> np.ndarray:
        if src is None:
            return img
        out_im = np.asarray(img).astype(np.float64).copy()
        inv = ~np.isin(src, KNOWN)
        out_im[inv] = 0.45 * out_im[inv] + 0.55 * np.array([155.0, 80.0, 240.0])  # violet: neural
        return np.clip(out_im, 0, 255).astype(np.uint8)

    for glb_name, images, roof_img in (
        ("building.glb", [im for im, _ in done], roof[0] if roof else None),
        ("building_tint.glb", [tint(im, sv) for im, sv in done], tint(*roof) if roof else None),
    ):
        swap = {f"walls_{k}": pg for k, pg in enumerate(pages_of(images))}
        if roof_img is not None:
            swap["roof"] = roof_img
        replace_glb_images(
            mesh_glb,
            out / glb_name,
            swap,
            extras_patch={"product": "neural", "fill": fill},
            texture_format=texture_format,
            dump_dir=(out / "textures") if (dump_textures and glb_name == "building.glb") else None,
        )

    tileset = json.loads(mesh_ts.read_text(encoding="utf-8"))
    for ts_name, uri in (("tileset.json", "building.glb"), ("tileset_tint.json", "building_tint.glb")):
        ts = json.loads(json.dumps(tileset))
        ts["root"]["content"]["uri"] = uri
        ts.setdefault("extras", {}).update({"product": "neural", "fill": fill})
        (out / ts_name).write_text(json.dumps(ts, indent=1), encoding="utf-8")

    dt = time.time() - t0
    b_name = tileset.get("extras", {}).get("name") or b.name
    summary = ", ".join(f"{NAMES[k]} {fill[NAMES[k]]:.0%}" for k in sorted(NAMES) if fill.get(NAMES[k], 0) > 0)
    print(f"[neural] {b_name} ({b.name}) in {dt:.1f}s -> {out / 'building.glb'}: {summary}")
    return {"building": b.name, "name": b_name, "seconds": round(dt, 2), "fill": fill}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("buildings", nargs="+", type=Path, help="Building output folder(s)")
    ap.add_argument("--texture", choices=TEXTURE_FORMATS, default="ktx2")
    ap.add_argument("--dump-textures", action="store_true", help="Save lossless PNG atlas pages in 3dtiles_neural/textures/")
    args = ap.parse_args()

    for b in args.buildings:
        build_building_neural(b, texture_format=args.texture, dump_textures=args.dump_textures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
