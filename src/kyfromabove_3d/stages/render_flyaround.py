#!/usr/bin/env python3
"""Fly a virtual camera around a built building and save what it sees.

Test data for streaming reconstruction (LingBot-Map and models like it): a
video of a Kentucky building whose true camera path is known exactly. A model
that turns video into camera poses can be run on the frames and scored
against ``cameras.json``; the step that registers its poses to the LiDAR, and
the colouring that reads the frames, can be tested the same way before any
real drone or phone video exists.

The building is a mesh this project already built (``building.glb`` from
``build_mesh.py``), so its shape is the LiDAR's and its colour the aerial
frames'. What a real capture has and this does not: neighbouring buildings,
trees, people, cars, sky, glass that reflects, light that changes, motion
blur, rolling shutter. A pass here proves the plumbing, not the method on
real video.

Two paths:

    orbit   a drone circling the building: a helix of ``--turns`` turns at
            ``--radius`` m, climbing from ``--height-start`` to
            ``--height-end`` m above the street, always looking at the
            building's middle. The last turn passes over the first at another
            height, so drift shows.
    walk    a person with a phone walking all the way round, ``--offset`` m
            out from the building's outline, at eye height (``--eye``),
            facing the wall and tilted up ``--pitch`` degrees.

The ground under it is a procedural pattern, not real ground: it is there so
a tracker has something below the building to lock onto, and ``--ground none``
leaves it out.

Output, under ``--out``:

    frames/000000.png ...   the frames
    cameras.json            the intrinsics (OpenCV: pixel centres at integer
                            coordinates) and, per frame, R_cw and t_cw
                            (x_cam = R_cw @ X + t_cw, camera axes x right,
                            y down, z forward) and the camera centre C, in the
                            building's local frame: x east, y north, z up,
                            metres, z = 0 the street under the building. The
                            frame's origin in state plane is copied from the
                            building's coordinates.json when there is one.
    trajectory_tum.txt      the same path as TUM lines (t tx ty tz qx qy qz
                            qw, camera to world), which evo and most
                            trajectory benchmarks read
    flyaround.mp4           the frames as a video (``--video``)

A reconstruction from video has no scale of its own, so score it after a
similarity (Sim(3)) alignment to these poses, not a rigid one.

Usage::

    python scripts/render_flyaround.py --building outputs/downtown/buildings/d2786314-4fb3-4da6-8f81-621873d40c08 --out outputs/flyaround/humana
    python scripts/render_flyaround.py --glb tests/data/audit/omni_before_mesh.glb --path walk --out /tmp/omni_walk
    python scripts/render_flyaround.py --glb buildings/v3/<gers>/3dtiles_mesh.zip --out /tmp/x --video

KTX2 textures (build_mesh.py's default) are unpacked with the ``basisu``
command, as the texture check does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from kyfromabove_3d.texture_check import _decode_image, _open_glb, _z_up

BACKGROUND = (176, 190, 204)                  # a flat, pale sky: nothing to track in it
CHUNK_FRAGMENTS = 4_000_000                   # candidate pixels rasterised at once


# ------------------------------------------------------------------ the scene

@dataclass
class Scene:
    """Triangles ready to rasterise, in the building's local frame (z up).

    ``tex`` is each triangle's texture index, -1 for one coloured by
    ``colour`` (per vertex, 0-1 sRGB) instead."""
    tris: np.ndarray                          # (T, 3, 3)
    uv: np.ndarray                            # (T, 3, 2), glTF: v = 0 is the image's first row
    colour: np.ndarray                        # (T, 3, 3)
    tex: np.ndarray                           # (T,)
    textures: List[np.ndarray] = field(default_factory=list)

    def __add__(self, other: "Scene") -> "Scene":
        n = len(self.textures)
        return Scene(np.concatenate([self.tris, other.tris]), np.concatenate([self.uv, other.uv]),
                     np.concatenate([self.colour, other.colour]),
                     np.concatenate([self.tex, np.where(other.tex >= 0, other.tex + n, -1)]),
                     self.textures + other.textures)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """glTF vertex colours are linear; the frames are sRGB."""
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def glb_path(source: Path, tmp: Path) -> Path:
    """The glb itself, the mesh of a building folder, or the glb in a stored
    product zip (``buildings/<version>/<gers>/3dtiles_mesh.zip``)."""
    source = Path(source)
    if source.is_dir():
        p = source / "3dtiles_mesh" / "building.glb"
        if not p.exists():
            raise SystemExit(f"[FAIL] {source} has no 3dtiles_mesh/building.glb: build its mesh first")
        return p
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as z:
            glbs = sorted(n for n in z.namelist() if n.endswith(".glb") and "_tint" not in n)
            if not glbs:
                raise SystemExit(f"[FAIL] {source} holds no glb")
            name = "building.glb" if "building.glb" in glbs else glbs[0]
            out = tmp / "building.glb"
            out.write_bytes(z.read(name))
            return out
    return source


def load_glb(path: Path) -> Scene:
    """Every triangle of a glb written by build_mesh.py: textured, vertex
    coloured or flat. Node transforms are not applied; build_mesh writes none."""
    gltf, view, accessor = _open_glb(path)
    images: Dict[int, int] = {}
    textures: List[np.ndarray] = []
    parts = []
    for mesh in gltf["meshes"]:
        for p in mesh["primitives"]:
            attrs = p["attributes"]
            P = _z_up(accessor(attrs["POSITION"]).astype(np.float64))
            I = (accessor(p["indices"]).astype(np.int64).reshape(-1, 3) if "indices" in p
                 else np.arange(len(P)).reshape(-1, 3))
            mat = gltf["materials"][p["material"]] if "material" in p else {}
            pbr = mat.get("pbrMetallicRoughness", {})
            ref = pbr.get("baseColorTexture")
            tex_id = -1
            uv = np.zeros((len(P), 2))
            colour = np.tile(np.asarray(pbr.get("baseColorFactor", [0.7, 0.7, 0.7, 1.0])[:3], float),
                             (len(P), 1))
            colour = _linear_to_srgb(colour)
            if ref is not None and "TEXCOORD_0" in attrs:
                tex = gltf["textures"][ref["index"]]
                src = tex.get("source", tex.get("extensions", {}).get("KHR_texture_basisu", {}).get("source"))
                if src not in images:
                    img = gltf["images"][src]
                    try:
                        textures.append(_decode_image(view(img["bufferView"]), img.get("mimeType", "image/png")))
                    except FileNotFoundError:
                        raise SystemExit("[FAIL] the mesh's textures are KTX2 and the basisu command is not on "
                                         "PATH: brew install basis_universal, or rebuild the mesh with "
                                         "--texture png") from None
                    images[src] = len(textures) - 1
                tex_id = images[src]
                uv = accessor(attrs["TEXCOORD_0"]).astype(np.float64)
            elif "COLOR_0" in attrs:
                colour = _linear_to_srgb(accessor(attrs["COLOR_0"]).astype(np.float64)[:, :3])
            parts.append(Scene(P[I], uv[I], colour[I], np.full(len(I), tex_id), []))
    if not parts:
        raise SystemExit(f"[FAIL] {path} has no triangles")
    scene = Scene(np.concatenate([s.tris for s in parts]), np.concatenate([s.uv for s in parts]),
                  np.concatenate([s.colour for s in parts]), np.concatenate([s.tex for s in parts]), textures)
    return scene


def ground_scene(centre_xy: np.ndarray, half: float, z: float, seed: int = 7) -> Scene:
    """A square of ground ``2 * half`` m across at height ``z``, textured with
    smooth noise at several scales."""
    from scipy.ndimage import gaussian_filter
    n = 2048
    rng = np.random.default_rng(seed)
    g = sum(gaussian_filter(rng.normal(size=(n, n)), s, mode="wrap") * s for s in (2.0, 8.0, 32.0))
    g = (g - g.mean()) / g.std()
    base = np.array([118.0, 124.0, 104.0])                                   # dull grass and asphalt
    img = np.clip(base[None, None, :] + 34.0 * g[..., None], 0, 255).astype(np.uint8)
    x, y = centre_xy
    a, b, c, d = ([x - half, y - half, z], [x + half, y - half, z], [x + half, y + half, z], [x - half, y + half, z])
    tris = np.array([[a, b, c], [a, c, d]], dtype=np.float64)
    uv = np.array([[[0, 1], [1, 1], [1, 0]], [[0, 1], [1, 0], [0, 0]]], dtype=np.float64)   # north at the top
    return Scene(tris, uv, np.zeros_like(tris), np.zeros(2, dtype=np.int64), [img])


# ------------------------------------------------------------------ the camera

def intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """A pinhole K with square pixels, principal point in the middle (OpenCV:
    pixel centres at integer coordinates)."""
    f = 0.5 * width / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array([[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]])


def look_rotation(forward: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World-to-camera rotation looking along ``forward`` (x right, y down, z forward)."""
    z = np.asarray(forward, dtype=np.float64)
    z = z / np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, dtype=np.float64))
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x)
    return np.vstack([x, np.cross(z, x), z])


@dataclass
class Pose:
    R_cw: np.ndarray
    C: np.ndarray

    @property
    def t_cw(self) -> np.ndarray:
        return -self.R_cw @ self.C


def orbit_path(centre: np.ndarray, target_z: float, radius: float, z0: float, z1: float,
               turns: float, frames_per_turn: int) -> List[Pose]:
    """A helix round ``centre`` climbing from ``z0`` to ``z1``, counter-clockwise
    seen from above, starting due south, every frame looking at the building's
    middle at ``target_z``."""
    n = max(2, int(round(turns * frames_per_turn)))
    s = np.linspace(0.0, 1.0, n)
    a = -np.pi / 2 + 2 * np.pi * turns * s
    target = np.array([centre[0], centre[1], target_z])
    out = []
    for ai, zi in zip(a, z0 + (z1 - z0) * s):
        C = np.array([centre[0] + radius * np.cos(ai), centre[1] + radius * np.sin(ai), zi])
        out.append(Pose(look_rotation(target - C), C))
    return out


def walk_path(xy: np.ndarray, offset: float, step: float, eye: float, pitch_deg: float,
              ground_z: float) -> List[Pose]:
    """Counter-clockwise round the outline of ``xy`` (its convex hull, pushed
    out ``offset`` m with rounded corners), a frame every ``step`` m, facing
    the building and tilted up ``pitch_deg``."""
    from scipy.ndimage import gaussian_filter1d
    from shapely.geometry import MultiPoint
    from shapely.geometry.polygon import orient
    ring = orient(MultiPoint([tuple(p) for p in xy]).convex_hull.buffer(offset, quad_segs=16), 1.0).exterior
    n = max(8, int(round(ring.length / step)))
    pts = np.array([ring.interpolate(d).coords[0] for d in np.linspace(0.0, ring.length, n, endpoint=False)])
    d = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    # smoothed along the walk, so the heading turns through a corner instead of snapping
    sigma = max(1.0, 2.0 / step)
    d = np.column_stack([gaussian_filter1d(d[:, 0], sigma, mode="wrap"), gaussian_filter1d(d[:, 1], sigma, mode="wrap")])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    inward = np.column_stack([-d[:, 1], d[:, 0]])                           # left of a counter-clockwise walk
    p = np.radians(pitch_deg)
    return [Pose(look_rotation(np.array([np.cos(p) * w[0], np.cos(p) * w[1], np.sin(p)])),
                 np.array([q[0], q[1], ground_z + eye])) for q, w in zip(pts, inward)]


def shake(poses: Sequence[Pose], metres: float, degrees: float, seed: int = 1) -> List[Pose]:
    """Hand-held wobble: smooth random offsets of the position and the view
    direction. The returned poses are the true ones, wobble included."""
    if metres <= 0 and degrees <= 0:
        return list(poses)
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(seed)
    n = len(poses)
    smooth = lambda k: gaussian_filter1d(rng.normal(size=(n, k)), 3.0, axis=0, mode="wrap") * np.sqrt(2 * np.sqrt(np.pi) * 3.0)
    dC, dR = smooth(3) * metres, smooth(3) * np.radians(degrees)
    return [Pose(Rotation.from_rotvec(r).as_matrix() @ p.R_cw, p.C + c) for p, c, r in zip(poses, dC, dR)]


# ------------------------------------------------------------------ the raster

def clip_near(P: np.ndarray, attrs: np.ndarray, near: float):
    """Triangles in camera coordinates cut at the plane z = ``near``: what is
    in front of it, as triangles, and which input triangle each came from.

    ``attrs`` (T, 3, k) are carried along, interpolated where an edge is cut;
    they vary linearly over a triangle in 3D, so that is exact. A triangle
    with one corner behind leaves a quad (two triangles), with two behind a
    smaller triangle, with three nothing. The corners keep their winding."""
    behind = P[..., 2] <= near
    n = behind.sum(1)
    out_P, out_A, src = [P[n == 0]], [attrs[n == 0]], [np.flatnonzero(n == 0)]
    for k in (1, 2):
        sel = np.flatnonzero(n == k)
        if not len(sel):
            continue
        # turn each triangle so its odd corner out (the one behind, or the one in front) comes first
        odd = np.argmax(behind[sel] if k == 1 else ~behind[sel], axis=1)
        order = (odd[:, None] + np.arange(3)[None, :]) % 3
        Q = np.take_along_axis(P[sel], order[..., None], 1)
        B = np.take_along_axis(attrs[sel], order[..., None], 1)
        s_b = ((near - Q[:, 0, 2]) / (Q[:, 1, 2] - Q[:, 0, 2]))[:, None]
        s_c = ((near - Q[:, 0, 2]) / (Q[:, 2, 2] - Q[:, 0, 2]))[:, None]
        Pab, Pac = Q[:, 0] + s_b * (Q[:, 1] - Q[:, 0]), Q[:, 0] + s_c * (Q[:, 2] - Q[:, 0])
        Aab, Aac = B[:, 0] + s_b * (B[:, 1] - B[:, 0]), B[:, 0] + s_c * (B[:, 2] - B[:, 0])
        if k == 1:                                                           # a behind: (ab, b, c) and (ab, c, ac)
            out_P += [np.stack([Pab, Q[:, 1], Q[:, 2]], 1), np.stack([Pab, Q[:, 2], Pac], 1)]
            out_A += [np.stack([Aab, B[:, 1], B[:, 2]], 1), np.stack([Aab, B[:, 2], Aac], 1)]
            src += [sel, sel]
        else:                                                                # only a in front: (a, ab, ac)
            out_P.append(np.stack([Q[:, 0], Pab, Pac], 1))
            out_A.append(np.stack([B[:, 0], Aab, Aac], 1))
            src.append(sel)
    return np.concatenate(out_P), np.concatenate(out_A), np.concatenate(src)


def render(scene: Scene, K: np.ndarray, R_cw: np.ndarray, t_cw: np.ndarray, width: int, height: int,
           near: float = 0.3, background=BACKGROUND):
    """The scene through a pinhole camera: an (H, W, 3) uint8 image and the
    (H, W) depth along the optical axis (inf where nothing was hit).

    A z-buffered triangle rasteriser with perspective-correct interpolation,
    in numpy. Triangles are cut at ``near`` m in front of the camera, so a
    wall that runs past it is drawn up to the edge of the frame."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = scene.tris @ R_cw.T + t_cw                                           # (T, 3, 3) camera coordinates
    P, attrs, src = clip_near(P, np.concatenate([scene.uv, scene.colour], axis=2), near)
    z = P[..., 2]
    u = fx * P[..., 0] / z + cx
    v = fy * P[..., 1] / z + cy
    x0 = np.ceil(u.min(1)).clip(0, width); x1 = np.floor(u.max(1)).clip(-1, width - 1)
    y0 = np.ceil(v.min(1)).clip(0, height); y1 = np.floor(v.max(1)).clip(-1, height - 1)
    area = (u[:, 1] - u[:, 0]) * (v[:, 2] - v[:, 0]) - (u[:, 2] - u[:, 0]) * (v[:, 1] - v[:, 0])
    keep = (x1 >= x0) & (y1 >= y0) & (np.abs(area) > 1e-12)
    idx = np.flatnonzero(keep)
    u, v, iz, area = u[idx], v[idx], 1.0 / z[idx], area[idx]
    x0, y0 = x0[idx].astype(np.int64), y0[idx].astype(np.int64)
    bw, bh = x1[idx].astype(np.int64) - x0 + 1, y1[idx].astype(np.int64) - y0 + 1
    count = bw * bh
    # The barycentric weights l1, l2 are affine in the pixel's (x, y), and so
    # is 1/depth: each triangle's three planes, worked out once, make a
    # candidate pixel a few multiplications.
    du1, dv1, du2, dv2 = u[:, 1] - u[:, 0], v[:, 1] - v[:, 0], u[:, 2] - u[:, 0], v[:, 2] - v[:, 0]
    a1, b1 = dv2 / area, -du2 / area
    c1 = -(a1 * u[:, 0] + b1 * v[:, 0])
    a2, b2 = -dv1 / area, du1 / area
    c2 = -(a2 * u[:, 0] + b2 * v[:, 0])
    d1, d2 = iz[:, 1] - iz[:, 0], iz[:, 2] - iz[:, 0]

    depth = np.full(height * width, np.inf)
    tri_at = np.full(height * width, -1, dtype=np.int64)
    l12_at = np.zeros((height * width, 2))
    ends = np.cumsum(count)
    start = 0
    while start < len(idx):
        # as many triangles as fit in one chunk of candidate pixels (at least one)
        stop = max(start + 1, int(np.searchsorted(ends, (ends[start - 1] if start else 0) + CHUNK_FRAGMENTS, "right")))
        c = count[start:stop]
        t = np.repeat(np.arange(start, stop), c)
        off = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
        w_t = bw[t]
        px = x0[t] + off % w_t
        py = y0[t] + off // w_t
        l1 = a1[t] * px + b1[t] * py + c1[t]
        l2 = a2[t] * px + b2[t] * py + c2[t]
        inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l1 + l2 <= 1.0 + 1e-9)
        t, l1, l2 = t[inside], l1[inside], l2[inside]
        pix = py[inside] * width + px[inside]
        d = 1.0 / (iz[t, 0] + l1 * d1[t] + l2 * d2[t])
        nearer = d < depth[pix]
        pix, d, t, l1, l2 = pix[nearer], d[nearer], t[nearer], l1[nearer], l2[nearer]
        order = np.lexsort((d, pix))                                         # nearest first within each pixel
        first = np.ones(len(order), dtype=bool)
        first[1:] = pix[order][1:] != pix[order][:-1]
        o = order[first]
        depth[pix[o]], tri_at[pix[o]] = d[o], t[o]
        l12_at[pix[o]] = np.column_stack([l1[o], l2[o]])
        start = stop

    img = np.tile(np.asarray(background, dtype=np.float64), (height * width, 1))
    hit = np.flatnonzero(tri_at >= 0)
    k_at, l12 = tri_at[hit], l12_at[hit]
    # screen-space weights to the surface's own: each corner's weighted by its 1/depth
    w = np.column_stack([1.0 - l12[:, 0] - l12[:, 1], l12[:, 0], l12[:, 1]]) * iz[k_at]
    w /= w.sum(1, keepdims=True)
    t = idx[k_at]
    tex = scene.tex[src[t]]
    flat = tex < 0
    img[hit[flat]] = 255.0 * np.einsum("nk,nkc->nc", w[flat], attrs[t[flat], :, 2:])
    for k in np.unique(tex[~flat]):
        m = tex == k
        uv = np.einsum("nk,nkc->nc", w[m], attrs[t[m], :, :2])
        img[hit[m]] = _bilinear(scene.textures[k], uv)
    return (np.clip(img, 0, 255).round().astype(np.uint8).reshape(height, width, 3),
            depth.reshape(height, width))


def _bilinear(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """``img`` at glTF texture coordinates, clamped at the edges as the
    building's sampler is."""
    H, W = img.shape[:2]
    x = np.clip(uv[:, 0] * W - 0.5, 0, W - 1)
    y = np.clip(uv[:, 1] * H - 0.5, 0, H - 1)
    xa, ya = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    xb, yb = np.minimum(xa + 1, W - 1), np.minimum(ya + 1, H - 1)
    fx, fy = (x - xa)[:, None], (y - ya)[:, None]
    f = lambda r, c: img[r, c].astype(np.float64)                         # gather first: a texture can be large
    return (f(ya, xa) * (1 - fx) + f(ya, xb) * fx) * (1 - fy) + (f(yb, xa) * (1 - fx) + f(yb, xb) * fx) * fy


def render_frame(scene: Scene, K: np.ndarray, pose: Pose, width: int, height: int, supersample: int) -> np.ndarray:
    """One frame, rendered ``supersample`` times larger each way and averaged
    down, so edges and distant texture do not shimmer from frame to frame."""
    s = max(1, int(supersample))
    if s == 1:
        return render(scene, K, pose.R_cw, pose.t_cw, width, height)[0]
    Ks = K.copy()
    Ks[:2, :2] *= s
    Ks[:2, 2] = (K[:2, 2] + 0.5) * s - 0.5                                    # same pixel grid, s times finer
    big = render(scene, Ks, pose.R_cw, pose.t_cw, width * s, height * s)[0]
    return big.reshape(height, s, width, s, 3).mean(axis=(1, 3)).round().astype(np.uint8)


_JOB: Dict[str, Any] = {}


def _start(job: Dict[str, Any]) -> None:
    """A worker's copy of the scene and camera, set once when it starts."""
    _JOB.update(job)


def _frame(item) -> Optional[np.ndarray]:
    """Frame ``i``, rendered and saved; the image back only when a video wants it."""
    from PIL import Image
    i, pose = item
    j = _JOB
    img = render_frame(j["scene"], j["K"], pose, j["width"], j["height"], j["supersample"])
    Image.fromarray(img).save(j["frames_dir"] / f"{i:06d}.png")
    return img if j["video"] else None


# ------------------------------------------------------------------ the record

def tum_line(time_s: float, pose: Pose) -> str:
    """One TUM trajectory line: the camera's centre and its camera-to-world rotation."""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(pose.R_cw.T).as_quat()                          # x y z w
    return " ".join(f"{x:.6f}" for x in (time_s, *pose.C, *q))


def find_coordinates(source: Path) -> Optional[Dict[str, Any]]:
    """The building's coordinates.json (its local frame's origin in state plane), if beside it."""
    for d in (source, *source.parents[:3]):
        p = (d if d.is_dir() else d.parent) / "coordinates.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


class _nullpool:
    """One worker: the frames in this process, with a Pool's interface."""

    def __init__(self, job: Dict[str, Any]):
        _start(job)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap(self, fn, items):
        return map(fn, items)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--building", type=Path, help="a building folder (reads 3dtiles_mesh/building.glb)")
    src.add_argument("--glb", type=Path, help="a building.glb, or a stored product zip holding one")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--path", choices=("orbit", "walk"), default="orbit")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov", type=float, default=70.0, help="horizontal field of view, degrees")
    ap.add_argument("--supersample", type=int, default=1,
                    help="render this many times larger each way and average down (smoother edges, slower)")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="frames rendered at once")
    ap.add_argument("--fps", type=float, default=10.0, help="the frames' timestamps, and the video's rate")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after this many (a quick look)")
    o = ap.add_argument_group("orbit")
    o.add_argument("--radius", type=float, default=None, help="m from the building's middle (default: from its size)")
    o.add_argument("--height-start", type=float, default=None, help="m above the street (default: from its height)")
    o.add_argument("--height-end", type=float, default=None)
    o.add_argument("--turns", type=float, default=2.0)
    o.add_argument("--frames-per-turn", type=int, default=120)
    w = ap.add_argument_group("walk")
    w.add_argument("--offset", type=float, default=8.0, help="m out from the building's outline")
    w.add_argument("--step", type=float, default=1.0, help="m walked between frames")
    w.add_argument("--eye", type=float, default=1.6, help="camera height above the street, m")
    w.add_argument("--pitch", type=float, default=20.0, help="camera tilted up, degrees")
    ap.add_argument("--shake-m", type=float, default=0.0, help="hand-held wobble of the position, m")
    ap.add_argument("--shake-deg", type=float, default=0.0, help="hand-held wobble of the view, degrees")
    ap.add_argument("--ground", choices=("pattern", "none"), default="pattern")
    ap.add_argument("--ground-z", type=float, default=0.0, help="the street's height in the local frame")
    ap.add_argument("--video", action="store_true", help="also write flyaround.mp4")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    source = args.building or args.glb
    with tempfile.TemporaryDirectory() as tmp:
        glb = glb_path(source, Path(tmp))
        sha = hashlib.sha256(glb.read_bytes()).hexdigest()
        gltf, _, _ = _open_glb(glb)
        extras = (gltf.get("nodes") or [{}])[0].get("extras", {})
        scene = load_glb(glb)

    V = scene.tris.reshape(-1, 3)
    lo, hi = V.min(0), V.max(0)
    centre = (lo[:2] + hi[:2]) / 2.0
    half_diag = float(np.linalg.norm(hi[:2] - lo[:2]) / 2.0)
    top = float(hi[2] - args.ground_z)
    if args.path == "orbit":
        radius = args.radius or 1.6 * half_diag + 20.0
        z0 = args.ground_z + (args.height_start if args.height_start is not None else 0.5 * top + 10.0)
        z1 = args.ground_z + (args.height_end if args.height_end is not None else top + 25.0)
        poses = orbit_path(centre, args.ground_z + 0.5 * top, radius, z0, z1, args.turns, args.frames_per_turn)
        reach = radius
        path = {"kind": "orbit", "radius_m": radius, "height_start_m": z0 - args.ground_z,
                "height_end_m": z1 - args.ground_z, "turns": args.turns, "frames_per_turn": args.frames_per_turn}
    else:
        poses = walk_path(V[:, :2], args.offset, args.step, args.eye, args.pitch, args.ground_z)
        reach = half_diag + args.offset
        path = {"kind": "walk", "offset_m": args.offset, "step_m": args.step, "eye_m": args.eye,
                "pitch_deg": args.pitch}
    poses = shake(poses, args.shake_m, args.shake_deg)
    path.update(shake_m=args.shake_m, shake_deg=args.shake_deg)
    if args.max_frames:
        poses = poses[:args.max_frames]
    if args.ground == "pattern":
        scene = scene + ground_scene(centre, max(4.0 * reach, 3.0 * top, 150.0), args.ground_z)

    K = intrinsics(args.width, args.height, args.fov)
    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video = None
    if args.video:
        import cv2
        video = cv2.VideoWriter(str(args.out / "flyaround.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                args.fps, (args.width, args.height))
    job = {"scene": scene, "K": K, "width": args.width, "height": args.height,
           "supersample": args.supersample, "frames_dir": frames_dir, "video": video is not None}
    with multiprocessing.get_context().Pool(args.workers, _start, (job,)) if args.workers > 1 else _nullpool(job) as pool:
        for i, img in enumerate(pool.imap(_frame, enumerate(poses))):
            if video is not None:
                video.write(img[..., ::-1])
            if i % 20 == 0 or i == len(poses) - 1:
                print(f"frame {i + 1}/{len(poses)}", flush=True)
    if video is not None:
        video.release()
    records, tum = [], []
    for i, pose in enumerate(poses):
        t = i / args.fps
        records.append({"index": i, "file": f"frames/{i:06d}.png", "time_s": round(t, 6),
                        "R_cw": pose.R_cw.round(9).tolist(), "t_cw": pose.t_cw.round(6).tolist(),
                        "C": pose.C.round(6).tolist()})
        tum.append(tum_line(t, pose))

    record = {
        "what": "a synthetic fly-around of a built building: frames and the true camera of each",
        "source": {"glb": str(source), "sha256": sha, "name": extras.get("name"), "gers": extras.get("gers")},
        "frame": {"axes": "x east, y north, z up, metres; z = 0 the street under the building",
                  "coordinates": find_coordinates(Path(source))},
        "camera": {"model": "pinhole, no distortion", "width": args.width, "height": args.height,
                   "hfov_deg": args.fov, "K": K.round(6).tolist(),
                   "convention": "OpenCV: x_cam = R_cw @ X + t_cw, x right, y down, z forward; "
                                 "pixel centres at integer coordinates"},
        "path": path,
        "ground": {"kind": args.ground, "z": args.ground_z,
                   "note": "procedural pattern, not the real ground" if args.ground == "pattern" else None},
        "not_simulated": ["neighbouring buildings", "trees, people, cars", "sky", "reflections",
                          "changing light", "motion blur", "rolling shutter", "lens distortion"],
        "frames": records,
    }
    (args.out / "cameras.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    (args.out / "trajectory_tum.txt").write_text("# t tx ty tz qx qy qz qw (camera to world, local frame, m)\n"
                                                 + "\n".join(tum) + "\n", encoding="utf-8")
    print(f"{len(poses)} frames -> {frames_dir}; cameras -> {args.out / 'cameras.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
