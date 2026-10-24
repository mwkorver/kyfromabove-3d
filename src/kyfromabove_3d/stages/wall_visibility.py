#!/usr/bin/env python3
"""Which aerial frames see each slice of wall, clearly and face-on enough to use.

The wall profiles (``wall_profiles.json``) cut every face into 5 m bands and
say what the LiDAR could tell about each: ``observed`` where returns put the
wall, and ``unknown``, ``mirrored``, ``extended`` ... where they did not.
Downtown, 57 % of the bands are unknown. The frames are flown at 45 degrees
and see walls the LiDAR (flown from above) barely touches; this measures,
from geometry alone, how well:

    for every band, points a metre apart across and up its wall, and for
    every frame of the flight (not just the ones the building kept):

      in the picture  inside the frame, clear of its edge
      not blocked     the downtown meshes, drawn through the frame's camera
                      at 1/``--scale`` size, show nothing nearer than the
                      point (within ``--depth-tol-m`` plus a pixel's width)
      face-on enough  the camera within ``--max-angle`` degrees of the wall's
                      outward normal; past that a wall is a sliver
      detail          centimetres of wall per pixel: distance / focal length,
                      stretched by 1 / cos(angle)

A frame *sees* a band when at least half its points pass. Per band: how many
frames see it; whether two of them look from directions at least
``--min-baseline`` degrees apart (what matching between frames needs to
find depth); the finest detail any of them gives; and how many of them the
building's own camera selection (``cameras_normalized.json``) kept.

The blockers are the downtown meshes, so trees, and buildings outside the
tileset, block nothing: the counts are an upper bound, most of all low down.
An unknown band's wall is put at the footprint edge (offset 0), where the
mesh's prism face stands.

Usage::

    python scripts/wall_visibility.py --reference configs/audit_reference.yaml \\
        --scene data/derived/video/downtown_scene.npz --out outputs/wall_visibility/reference
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from kyfromabove_3d.stages import register_video as rv

ROOT = Path(__file__).resolve().parents[3]                                    # the repository (src/kyfromabove_3d/stages/..)


# ------------------------------------------------------------------ inputs

def tileset_transforms(tileset_dir: Path) -> Dict[str, np.ndarray]:
    """Each glb's 4x4 transform from its building's local frame to the
    tileset's frame (the one ``register_video.py scene`` builds)."""
    ts = json.loads((tileset_dir / "tileset.json").read_text(encoding="utf-8"))
    out: Dict[str, np.ndarray] = {}

    def walk(node, M):
        M = M @ rv.col_major(node["transform"]) if node.get("transform") else M
        uri = node.get("content", {}).get("uri", "")
        if uri.startswith("glb/"):
            out[Path(uri).stem] = M
        for child in node.get("children", []):
            walk(child, M)

    walk(ts["root"], np.linalg.inv(rv.col_major(ts["root"]["transform"])))
    return out


@dataclass
class Building:
    key: str
    name: str
    dir: Path
    gers: str
    M: np.ndarray = field(repr=False)                                        # local -> tileset frame


def reference_buildings(reference: Path, landmarks: Path, transforms: Dict[str, np.ndarray],
                        buildings_dir: Path) -> List[Building]:
    import yaml
    ref = yaml.safe_load(reference.read_text(encoding="utf-8"))["buildings"]
    lm = {d.get("case") or d["key"]: d for d in yaml.safe_load(landmarks.read_text(encoding="utf-8"))["landmarks"]}
    out = []
    for b in ref:
        bdir = buildings_dir / b["id"]
        gers = lm[b["case"]]["gers"] if b.get("case") in lm else b["id"]
        if gers not in transforms:
            print(f"[WARN] {b['name']}: not in the tileset, skipped")
            continue
        out.append(Building(b["id"], b["name"], bdir, gers, transforms[gers]))
    return out


def read_frames(gpkg: Path) -> List[dict]:
    con = sqlite3.connect(gpkg)
    con.row_factory = sqlite3.Row
    table = con.execute("select table_name from gpkg_contents").fetchone()[0]
    return [dict(r) for r in con.execute(f'select * from "{table}"')]


def cameras_in_tileset_frame(frames: List[dict], b: Building):
    """Every frame's camera through the building's own local frame (as the
    pipeline makes it), then carried into the tileset frame by its transform."""
    from kyfromabove_3d.cameras import parse_camera_from_eo_record
    from kyfromabove_3d.coordinates import LocalCoordinateFrame
    coords = json.loads((b.dir / "coordinates.json").read_text(encoding="utf-8"))
    lf = LocalCoordinateFrame(tuple(coords["origin_source_xyz"]))
    A, t = b.M[:3, :3], b.M[:3, 3]
    out = []
    for f in frames:
        c = parse_camera_from_eo_record(f, lf)
        out.append(dict(id=str(f["ID"]), head=str(f["CamName"]).split("_")[0], W=c.width_px, H=c.height_px,
                        K=np.array([[c.fx_px, 0, c.cx_px], [0, c.fy_px, c.cy_px], [0, 0, 1.0]]),
                        R_cw=c.R_cw @ A.T, C=A @ c.camera_center_local_m + t))
    return out


@dataclass
class Samples:
    xyz: np.ndarray                                                          # (N, 3) tileset frame
    normal: np.ndarray                                                       # (N, 3) outward, horizontal
    band: np.ndarray                                                         # (N,) index into bands
    bands: List[dict]                                                        # building, face, z, status, ...
    along: np.ndarray = None                                                 # (N,) metres along the face
    z: np.ndarray = None                                                     # (N,) height in the building's frame


def wall_samples(b: Building, step_m: float = 1.0) -> Samples:
    """Points a metre apart on every band's wall, in the tileset frame."""
    from kyfromabove_3d.features import ArcWall, load_wall_planes
    planes = {p.face: p for p in load_wall_planes(b.dir / "wall_planes.json")}
    profiles = json.loads((b.dir / "wall_profiles.json").read_text(encoding="utf-8"))
    A, t = b.M[:3, :3], b.M[:3, 3]
    xyz, nrm, idx, bands, al, zs = [], [], [], [], [], []
    for face, rows in profiles.items():
        p = planes.get(int(face))
        if p is None:
            continue
        L = float(getattr(p, "length_m", np.hypot(*(np.asarray(p.p1) - np.asarray(p.p0)))))
        for row in rows:
            lo = row.get("along_lo") if row.get("status") == "observed" else None
            hi = row.get("along_hi") if row.get("status") == "observed" else None
            lo, hi = (0.0 if lo is None else lo), (L if hi is None else hi)
            if hi - lo < 1.0 or row["z_hi"] - row["z_lo"] < 0.5:
                continue
            off = row["offset_m"] if row.get("offset_m") is not None and row["status"] != "unknown" else 0.0
            a = np.arange(lo + step_m / 2, hi, step_m)
            z = np.arange(row["z_lo"] + step_m / 2, row["z_hi"], step_m)
            aa, zz = np.meshgrid(a, z)
            aa, zz = aa.ravel(), zz.ravel()
            if isinstance(p, ArcWall):
                xy = p.world(aa, np.full_like(aa, off))
                _, out_n = p.basis(aa)
            else:
                d = (np.asarray(p.p1) - np.asarray(p.p0)) / L
                n = np.asarray(p.normal, float)
                xy = np.asarray(p.p0)[None, :] + aa[:, None] * d[None, :] + off * n[None, :]
                out_n = np.tile(n, (len(aa), 1))
            pts = np.column_stack([xy, zz])
            n3 = np.column_stack([out_n, np.zeros(len(aa))])
            xyz.append(pts @ A.T + t)
            nrm.append(n3 @ A.T)
            idx.append(np.full(len(aa), len(bands)))
            al.append(aa)
            zs.append(zz)
            bands.append(dict(building=b.key, face=int(face), z_lo=row["z_lo"], z_hi=row["z_hi"], status=row["status"],
                              n_points=int(len(aa)), along=[round(lo, 2), round(hi, 2)]))
    if not xyz:
        return Samples(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, int), [], np.zeros(0), np.zeros(0))
    return Samples(np.vstack(xyz), np.vstack(nrm), np.concatenate(idx), bands, np.concatenate(al), np.concatenate(zs))


def roof_grid(V: np.ndarray, cell_m: float = 0.5):
    """The highest mesh vertex in each cell of a plan grid: the roof (and the
    tops of the walls) of one building's mesh. Returns (z grid, x0, y0)."""
    x0, y0 = V[:, 0].min() - 2.0, V[:, 1].min() - 2.0
    ix = ((V[:, 0] - x0) / cell_m).astype(int)
    iy = ((V[:, 1] - y0) / cell_m).astype(int)
    g = np.full((iy.max() + 3, ix.max() + 3), -np.inf)
    np.maximum.at(g, (iy, ix), V[:, 2])
    # the height field's vertices are 0.75 m apart: close the gaps between them
    from scipy.ndimage import maximum_filter
    return maximum_filter(g, size=3), x0, y0


def wall_exists(xyz: np.ndarray, normal: np.ndarray, grid, cell_m: float = 0.5, margin_m: float = 0.5) -> np.ndarray:
    """Whether the building stands behind each wall point: its mesh's roof,
    0.5-2.5 m inside the wall, is higher than the point. A band of a face
    traced along a podium's edge runs on up past the podium's roof; there
    the LiDAR rightly found nothing, and the point is air, not hidden wall."""
    g, x0, y0 = grid
    top = np.full(len(xyz), -np.inf)
    for d in (0.5, 1.0, 1.5, 2.0, 2.5):
        p = xyz[:, :2] - d * normal[:, :2]
        ix = np.clip(((p[:, 0] - x0) / cell_m).astype(int), 0, g.shape[1] - 1)
        iy = np.clip(((p[:, 1] - y0) / cell_m).astype(int), 0, g.shape[0] - 1)
        top = np.maximum(top, g[iy, ix])
    return xyz[:, 2] < top - margin_m


# ------------------------------------------------------------------ the test

_G: dict = {}


def _init(scene_path, samples, scale, depth_tol, max_angle):
    _G.update(scene=rv.load_scene(Path(scene_path)), s=samples, scale=scale, tol=depth_tol, cosmax=np.cos(np.radians(max_angle)))


def test_frame(cam) -> Optional[tuple]:
    """(usable mask, cm per pixel) for every sample in one frame, or None if it sees none of them."""
    s, scale = _G["s"], _G["scale"]
    X, n = s.xyz, s.normal
    pc = (X - cam["C"]) @ cam["R_cw"].T
    z = pc[:, 2]
    front = z > 1.0
    u = cam["K"][0, 0] * pc[:, 0] / np.where(front, z, 1) + cam["K"][0, 2]
    v = cam["K"][1, 1] * pc[:, 1] / np.where(front, z, 1) + cam["K"][1, 2]
    m = 0.01 * min(cam["W"], cam["H"])
    inside = front & (u > m) & (u < cam["W"] - m) & (v > m) & (v < cam["H"] - m)
    to_cam = cam["C"] - X
    dist = np.linalg.norm(to_cam, axis=1)
    cosang = np.einsum("ij,ij->i", to_cam, n) / dist
    cand = inside & (cosang >= _G["cosmax"])
    if not cand.any():
        return None
    w, h = int(round(cam["W"] / scale)), int(round(cam["H"] / scale))
    Ks = cam["K"].copy()
    Ks[:2] /= scale
    Ks[2, 2] = 1.0
    _, depth = rv.render(_G["scene"], cam["R_cw"], cam["C"], Ks, w, h)
    iu = np.clip((u[cand] / scale).astype(int), 0, w - 1)
    iv = np.clip((v[cand] / scale).astype(int), 0, h - 1)
    pix_m = z[cand] / Ks[0, 0]                                               # a small pixel's width at the point
    d = depth[iv, iu]
    clear = z[cand] <= d + _G["tol"] + 2.0 * pix_m
    usable = np.zeros(len(X), bool)
    usable[np.flatnonzero(cand)[clear]] = True
    cm = np.full(len(X), np.nan, np.float32)
    cm[usable] = (100.0 * z / cam["K"][0, 0] / np.maximum(cosang, 1e-3))[usable]
    return cam["id"], usable, cm


# ------------------------------------------------------------------ summary

def summarise(s: Samples, cams: List[dict], results: Dict[str, tuple], selected: Dict[str, set],
              min_baseline_deg: float) -> List[dict]:
    out = []
    by_band = [np.flatnonzero(s.band == i) for i in range(len(s.bands))]
    cam_of = {c["id"]: c for c in cams}
    for i, info in enumerate(s.bands):
        pts = by_band[i]
        if len(pts) == 0:                                                    # all air: nothing to see
            out.append(dict(info, frames=0, frames_kept=0, heads=[], widest_deg=0.0, stereo=False, best_cm_per_px=None))
            continue
        centre = s.xyz[pts].mean(0)
        seeing, cms = [], []
        for fid, (usable, cm) in results.items():
            frac = usable[pts].mean()
            if frac >= 0.5:
                seeing.append(fid)
                cms.append(float(np.nanmedian(cm[pts][usable[pts]])))
        rays = [cam_of[f]["C"] - centre for f in seeing]
        rays = [r / np.linalg.norm(r) for r in rays]
        widest = 0.0
        for a in range(len(rays)):
            for c in range(a + 1, len(rays)):
                widest = max(widest, float(np.degrees(np.arccos(np.clip(rays[a] @ rays[c], -1, 1)))))
        heads = sorted({cam_of[f]["head"] for f in seeing})
        out.append(dict(info, frames=len(seeing), frames_kept=sum(f in selected.get(info["building"], set()) for f in seeing),
                        heads=heads, widest_deg=round(widest, 1), stereo=widest >= min_baseline_deg,
                        best_cm_per_px=round(min(cms), 1) if cms else None))
    return out


def kept_frames(b: Building) -> set:
    p = b.dir / "cameras_normalized.json"
    if not p.exists():
        return set()
    d = json.loads(p.read_text(encoding="utf-8"))
    return {str(v["frame_id"]) for v in (d.values() if isinstance(d, dict) else d)}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, default=ROOT / "configs" / "audit_reference.yaml")
    ap.add_argument("--landmarks", type=Path, default=ROOT / "configs" / "landmarks.yaml")
    ap.add_argument("--tileset", type=Path, default=ROOT / "outputs" / "downtown")
    ap.add_argument("--buildings", type=Path, default=ROOT / "outputs" / "downtown" / "buildings")
    ap.add_argument("--frames", type=Path, default=ROOT / "outputs" / "downtown" / "frames_2024s1.gpkg")
    ap.add_argument("--scene", type=Path, required=True, help="register_video.py scene's .npz of the same tileset")
    ap.add_argument("--scale", type=float, default=16.0, help="draw the blockers at 1/scale of the frame's size")
    ap.add_argument("--depth-tol-m", type=float, default=2.0)
    ap.add_argument("--max-angle", type=float, default=70.0, help="degrees off the wall's normal a frame may look")
    ap.add_argument("--min-baseline", type=float, default=10.0, help="degrees two frames' views must differ for depth")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    transforms = tileset_transforms(a.tileset)
    blds = reference_buildings(a.reference, a.landmarks, transforms, a.buildings)
    frames = read_frames(a.frames)
    # The cameras through each building's frame must land in the same place:
    # a check on the chain of frames, before anything is measured.
    per_b = [cameras_in_tileset_frame(frames, b) for b in blds]
    spread = max(float(np.linalg.norm(per_b[k][j]["C"] - per_b[0][j]["C"])) for k in range(len(blds)) for j in range(len(frames)))
    print(f"{len(blds)} buildings, {len(frames)} frames; a camera's position through different buildings' frames "
          f"differs by at most {spread:.2f} m")
    cams = per_b[0]
    parts = [wall_samples(b) for b in blds]
    # Keep only points with the building behind them (see wall_exists), and
    # record per band how much of it was wall at all.
    sc = np.load(a.scene)
    names = list(sc["names"])
    for b, p in zip(blds, parts):
        ids = [i for i, n in enumerate(names) if n == b.gers]
        V = sc["V"][sc["T"][np.isin(sc["B"], ids)].ravel()].astype(np.float64)
        keep = wall_exists(p.xyz, p.normal, roof_grid(V))
        for i, band in enumerate(p.bands):
            on = p.band == i
            band["wall_fraction"] = round(float(keep[on].mean()), 3) if on.any() else 0.0
            band["n_points"] = int(keep[on].sum())
        p.xyz, p.normal, p.band, p.along, p.z = p.xyz[keep], p.normal[keep], p.band[keep], p.along[keep], p.z[keep]
    off = np.cumsum([0] + [len(p.bands) for p in parts])
    s = Samples(np.vstack([p.xyz for p in parts]), np.vstack([p.normal for p in parts]),
                np.concatenate([p.band + o for p, o in zip(parts, off)]), [x for p in parts for x in p.bands],
                np.concatenate([p.along for p in parts]), np.concatenate([p.z for p in parts]))
    print(f"{len(s.bands)} bands, {len(s.xyz)} wall points; testing them in every frame ...", flush=True)
    t0 = time.time()
    results = {}
    with mp.Pool(a.workers, initializer=_init, initargs=(str(a.scene), s, a.scale, a.depth_tol_m, a.max_angle)) as pool:
        for k, r in enumerate(pool.imap_unordered(test_frame, cams)):
            if r is not None:
                results[r[0]] = (r[1], r[2])
            if (k + 1) % 40 == 0:
                print(f"  {k + 1} frames ({time.time() - t0:.0f} s)", flush=True)
    selected = {b.key: kept_frames(b) for b in blds}
    rows = summarise(s, cams, results, selected, a.min_baseline)
    names = {b.key: b.name for b in blds}
    for r in rows:
        r["name"] = names[r["building"]]
    (a.out / "bands.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    np.savez_compressed(a.out / "points.npz", xyz=s.xyz.astype(np.float32), band=s.band, along=s.along.astype(np.float32),
                        z=s.z.astype(np.float32),
                        frames=np.array(list(results)), usable=np.stack([v[0] for v in results.values()]))
    air = [r for r in rows if r["wall_fraction"] < 0.5]
    print(f"{len(air)} of {len(rows)} bands are mostly air (no building behind them): "
          + ", ".join(f"{st} {sum(r['status'] == st for r in air)}" for st in sorted({r['status'] for r in air})))
    for st in ("observed", "unknown"):
        sel = [r for r in rows if r["status"] == st and r["wall_fraction"] >= 0.5]
        if sel:
            n = np.array([r["frames"] for r in sel])
            print(f"{st}: {len(sel)} bands; seen by none {np.mean(n == 0) * 100:.0f}%, one {np.mean(n == 1) * 100:.0f}%, "
                  f"two or more {np.mean(n >= 2) * 100:.0f}% (two from directions {a.min_baseline:.0f} deg apart: "
                  f"{np.mean([r['stereo'] for r in sel]) * 100:.0f}%); median {np.median(n):.0f} frames")
    print(f"-> {a.out}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
