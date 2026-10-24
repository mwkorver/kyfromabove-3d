"""Check a built mesh against the aerial frames it was not textured from.

Every texel of a mesh takes its colour from the single most face-on frame
that sees it; the building's other frames go unused. This module reads the
finished ``building.glb`` back, lays a regular grid over each face (a wall
in its own along/up frame, the roof in plan), and looks at each grid point
through every frame that sees it. A frame is scored only on the points it
did *not* texture -- comparing a texel with the frame it was copied from
measures nothing.

The score compares the *pattern*, not the colour: frames differ in sun,
exposure and, on glass, reflections. Both the model's face image and the
frame's view of it are reduced to brightness with the slow shading removed
(``highpass``), and compared by normalised cross-correlation -- 1 is the
same pattern, 0 unrelated. The shift that lines them up best, searched
within ``max_shift_m``, is reported in metres on the face: a face whose
texture sits 0.4 m off in the other frames' view shows it as a 0.4 m shift.

Pure functions over numpy; the file reading is ``read_glb`` alone.
"""

from __future__ import annotations

import json
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .features import ArcWall

_COMPONENTS = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


# ------------------------------------------------------------------ the glb

@dataclass
class Primitive:
    """One textured primitive of the mesh, in the building's local frame (z up)."""
    material: str
    positions: np.ndarray                  # (N, 3)
    uv: np.ndarray                         # (N, 2), glTF convention: v = 0 is the image's first row
    triangles: np.ndarray                  # (T, 3) vertex indices
    texture: np.ndarray                    # (H, W, 3) uint8


def _decode_image(data: bytes, mime: str) -> np.ndarray:
    from PIL import Image
    import io
    if mime != "image/ktx2":
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    return decode_ktx2(data)


def decode_ktx2(data: bytes) -> np.ndarray:
    """A KTX2 texture's full-size image, (H, W, 3) uint8, by the ``basisu``
    command. Ours are ETC1S (mesh.gltf.BASISU_ARGS), whose blocks are ETC1's:
    unpacked to ETC1 alone the image is the same, pixel for pixel, as the
    uncompressed one, in a fifteenth of the time (0.45 s, not 7, for a
    1876 x 1112 roof) -- basisu otherwise writes every GPU format and mip."""
    from PIL import Image
    etc1s = struct.unpack_from("<I", data, 44)[0] == 1           # the header's supercompression: BasisLZ
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "t.ktx2"
        src.write_bytes(data)
        r = subprocess.run(["basisu", "-unpack", "-no_ktx", *(["-format_only", "0"] if etc1s else []),
                            "-file", str(src)], cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        pattern = "t_unpacked_rgb_ETC1_RGB_level_0_*.png" if etc1s else "t_unpacked_rgb_RGBA32_level_0_*.png"
        out = sorted(Path(tmp).glob(pattern))
        if r.returncode != 0 or not out:
            raise RuntimeError(f"basisu could not unpack a KTX2 texture: {r.stdout[-300:]}")
        return np.asarray(Image.open(out[0]).convert("RGB"))


def _open_glb(path: Path):
    """(gltf json, view(i) -> bytes, accessor(i) -> array) of a glb."""
    b = Path(path).read_bytes()
    if b[:4] != b"glTF":
        raise ValueError(f"{path} is not a glb")
    jl = struct.unpack("<I", b[12:16])[0]
    gltf = json.loads(b[20:20 + jl])
    bin_at = 20 + jl
    bl = struct.unpack("<I", b[bin_at:bin_at + 4])[0]
    blob = b[bin_at + 8:bin_at + 8 + bl]

    def view(i):
        v = gltf["bufferViews"][i]
        o = v.get("byteOffset", 0)
        return blob[o:o + v["byteLength"]]

    def accessor(i):
        a = gltf["accessors"][i]
        dt = _COMPONENTS[a["componentType"]]
        arr = np.frombuffer(view(a["bufferView"]), dtype=dt, count=a["count"] * _WIDTH[a["type"]],
                            offset=a.get("byteOffset", 0))
        return arr.reshape(a["count"], _WIDTH[a["type"]]) if _WIDTH[a["type"]] > 1 else arr

    return gltf, view, accessor


def _z_up(P: np.ndarray) -> np.ndarray:
    return np.column_stack([P[:, 0], -P[:, 2], P[:, 1]])                  # glTF y-up -> local z-up


def read_glb_geometry(path: Path):
    """Every triangle of a glb, textured, plain or vertex-coloured alike:
    (vertices (N, 3) in the building's local frame, z up; triangles (T, 3))."""
    gltf, _, accessor = _open_glb(path)
    verts, tris, n = [], [], 0
    for mesh in gltf["meshes"]:
        for p in mesh["primitives"]:
            P = _z_up(accessor(p["attributes"]["POSITION"]).astype(np.float64))
            T = (accessor(p["indices"]).astype(np.int64).reshape(-1, 3) if "indices" in p
                 else np.arange(len(P)).reshape(-1, 3))
            verts.append(P); tris.append(T + n); n += len(P)
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.vstack(verts), np.vstack(tris)


def read_glb(path: Path) -> List[Primitive]:
    """The textured primitives of a glb written by build_mesh.py. Plain and
    vertex-coloured primitives are skipped: they carry no pattern to check."""
    gltf, view, accessor = _open_glb(path)
    images: Dict[int, np.ndarray] = {}

    def texture_of(material) -> Optional[np.ndarray]:
        ref = material.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if ref is None:
            return None
        tex = gltf["textures"][ref["index"]]
        src = tex.get("source", tex.get("extensions", {}).get("KHR_texture_basisu", {}).get("source"))
        if src not in images:
            img = gltf["images"][src]
            images[src] = _decode_image(view(img["bufferView"]), img.get("mimeType", "image/png"))
        return images[src]

    prims = []
    for mesh in gltf["meshes"]:
        for p in mesh["primitives"]:
            mat = gltf["materials"][p["material"]]
            if "TEXCOORD_0" not in p["attributes"]:
                continue
            tex = texture_of(mat)
            if tex is None:
                continue
            P = accessor(p["attributes"]["POSITION"]).astype(np.float64)
            prims.append(Primitive(
                material=mat.get("name", ""),
                positions=_z_up(P),
                uv=accessor(p["attributes"]["TEXCOORD_0"]).astype(np.float64),
                triangles=accessor(p["indices"]).astype(np.int64).reshape(-1, 3),
                texture=tex))
    return prims


# ------------------------------------------------------------ faces as grids

@dataclass
class FaceGrid:
    """A face laid out on a regular grid: ``xyz``/``normal``/``model`` are
    (H, W, 3) and ``valid`` (H, W) marks the grid points on the mesh."""
    key: str                               # "face 3", "face 101", "roof"
    kind: str                              # "wall" or "roof"
    spacing_m: float
    xyz: np.ndarray
    normal: np.ndarray
    model: np.ndarray
    valid: np.ndarray
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def area_m2(self) -> float:
        return float(self.valid.sum()) * self.spacing_m ** 2


def _bilinear(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    x = np.clip(uv[:, 0] * W - 0.5, 0, W - 1)
    y = np.clip(uv[:, 1] * H - 0.5, 0, H - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    f = img.astype(np.float64)
    return ((f[y0, x0] * (1 - fx) + f[y0, x1] * fx) * (1 - fy) + (f[y1, x0] * (1 - fx) + f[y1, x1] * fx) * fy)


def _rasterise(tris2d: np.ndarray, tri_ids: np.ndarray, grid_u: np.ndarray, grid_v: np.ndarray):
    """For each grid point (u, v), the first triangle (in ``tris2d``, (T, 3, 2))
    that contains it and its barycentric weights. Returns (tri (H, W) with -1
    where none, bary (H, W, 3))."""
    H, W = len(grid_v), len(grid_u)
    tri = np.full((H, W), -1, dtype=np.int64)
    bary = np.zeros((H, W, 3))
    step_u = grid_u[1] - grid_u[0] if W > 1 else 1.0
    step_v = grid_v[1] - grid_v[0] if H > 1 else 1.0
    for t, (a, b, c) in zip(tri_ids, tris2d):
        lo, hi = np.minimum(np.minimum(a, b), c), np.maximum(np.maximum(a, b), c)
        i0 = max(int(np.floor((lo[0] - grid_u[0]) / step_u)), 0)
        i1 = min(int(np.ceil((hi[0] - grid_u[0]) / step_u)) + 1, W)
        j0 = max(int(np.floor((lo[1] - grid_v[0]) / step_v)), 0)
        j1 = min(int(np.ceil((hi[1] - grid_v[0]) / step_v)) + 1, H)
        if i0 >= i1 or j0 >= j1:
            continue
        U, V = np.meshgrid(grid_u[i0:i1], grid_v[j0:j1])
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-12:
            continue
        l1 = ((b[1] - c[1]) * (U - c[0]) + (c[0] - b[0]) * (V - c[1])) / d
        l2 = ((c[1] - a[1]) * (U - c[0]) + (a[0] - c[0]) * (V - c[1])) / d
        l3 = 1.0 - l1 - l2
        inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9) & (tri[j0:j1, i0:i1] < 0)
        if not inside.any():
            continue
        sub_t, sub_b = tri[j0:j1, i0:i1], bary[j0:j1, i0:i1]
        sub_t[inside] = t
        sub_b[inside] = np.stack([l1[inside], l2[inside], l3[inside]], axis=1)
    return tri, bary


def face_grids(prims: Sequence[Primitive], planes: Sequence[Any], spacing_m: float = 0.25,
               max_off_m: float = 3.0, min_area_m2: float = 4.0, max_cells: int = 200_000) -> List[FaceGrid]:
    """Group the textured triangles into faces and lay each on a grid.

    A triangle whose normal is within 60 degrees of vertical is roof (one
    face, gridded in plan). Any other is wall, and belongs to the footprint
    edge ``planes`` (``WallPlane``/``ArcWall``, faces numbered as build_mesh
    numbers them) nearest to it -- within ``max_off_m`` of the edge's line,
    inside its length, facing its way. Triangles near no edge are left out.
    A face's grid spacing grows when it would exceed ``max_cells`` points.
    """
    groups: Dict[str, List] = {}
    for pi, pr in enumerate(prims):
        P = pr.positions[pr.triangles]                              # (T, 3, 3)
        n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        ln = np.linalg.norm(n, axis=1)
        keep = ln > 1e-9
        n[keep] /= ln[keep, None]
        cen = P.mean(axis=1)
        roof = keep & (np.abs(n[:, 2]) >= 0.5)
        for t in np.flatnonzero(roof):
            groups.setdefault("roof", []).append((pi, t))
        wall = np.flatnonzero(keep & ~roof)
        if not len(wall):
            continue
        best_d = np.full(len(wall), np.inf)
        best_f = np.full(len(wall), -1)
        for pl in planes:
            along, off = pl.local(cen[wall, :2])
            if isinstance(pl, ArcWall):                             # outward is radial, away from
                radial = cen[wall, :2] - pl.centre[None, :]         # the centre on a bulge
                radial /= np.maximum(np.linalg.norm(radial, axis=1), 1e-9)[:, None]
                faces_out = pl.out_sign * np.einsum("ij,ij->i", n[wall, :2], radial) > 0.7
            else:
                faces_out = (n[wall, :2] @ np.asarray(pl.normal, dtype=float)[:2]) > 0.7
            ok = faces_out & (along >= -0.5) & (along <= pl.length_m + 0.5) & (np.abs(off) <= max_off_m)
            better = ok & (np.abs(off) < best_d)
            best_d[better] = np.abs(off[better])
            best_f[better] = pl.face
        for t, f in zip(wall, best_f):
            if f >= 0:
                groups.setdefault(f"face {int(f)}", []).append((pi, t))

    by_face = {pl.face: pl for pl in planes}
    out = []
    for key, members in groups.items():
        pis = np.array([m[0] for m in members])
        tis = np.array([m[1] for m in members])
        P = np.stack([prims[p].positions[prims[p].triangles[t]] for p, t in members])       # (T, 3, 3)
        UV = np.stack([prims[p].uv[prims[p].triangles[t]] for p, t in members])             # (T, 3, 2)
        if key == "roof":
            flat = P[:, :, :2]
            plane = None
        else:
            plane = by_face[int(key.split()[1])]
            along, _ = plane.local(P[:, :, :2].reshape(-1, 2))
            flat = np.stack([along.reshape(-1, 3), P[:, :, 2]], axis=2)
        lo, hi = flat.reshape(-1, 2).min(axis=0), flat.reshape(-1, 2).max(axis=0)
        s = spacing_m
        while ((hi[0] - lo[0]) / s + 1) * ((hi[1] - lo[1]) / s + 1) > max_cells:
            s *= 1.5
        gu = np.arange(lo[0] + s / 2, hi[0], s)
        gv = np.arange(lo[1] + s / 2, hi[1], s)
        if len(gu) < 2 or len(gv) < 2:
            continue
        tri, bary = _rasterise(flat, np.arange(len(members)), gu, gv)
        valid = tri >= 0
        if valid.sum() * s * s < min_area_m2:
            continue
        H, W = valid.shape
        xyz = np.zeros((H, W, 3)); nrm = np.zeros((H, W, 3)); model = np.zeros((H, W, 3))
        tv, bv = tri[valid], bary[valid]
        xyz[valid] = np.einsum("nk,nkd->nd", bv, P[tv])
        tn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        tn /= np.maximum(np.linalg.norm(tn, axis=1), 1e-12)[:, None]
        nrm[valid] = tn[tv]
        uvs = np.einsum("nk,nkd->nd", bv, UV[tv])
        prim_of = pis[tv]
        vi, vj = np.nonzero(valid)
        for p in np.unique(prim_of):
            m = prim_of == p
            model[vi[m], vj[m]] = _bilinear(prims[p].texture, uvs[m])
        out.append(FaceGrid(key=key, kind="roof" if key == "roof" else "wall", spacing_m=s,
                            xyz=xyz, normal=nrm, model=model, valid=valid,
                            extras={"plane": plane}))
    return out


# ----------------------------------------------------------------- the score

def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def highpass(img: np.ndarray, mask: np.ndarray, sigma_px: float) -> np.ndarray:
    """``img`` less its local mean (a Gaussian of ``sigma_px``, normalised over
    ``mask`` so the face's edge is not read as dark): the pattern without the
    slow shading. Outside ``mask`` it is 0."""
    from scipy.ndimage import gaussian_filter
    m = mask.astype(np.float64)
    num = gaussian_filter(np.where(mask, img, 0.0), sigma_px, mode="constant")
    den = gaussian_filter(m, sigma_px, mode="constant")
    out = np.where(mask, img - num / np.maximum(den, 1e-6), 0.0)
    return out


def masked_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Normalised cross-correlation of ``a`` and ``b`` over ``mask``."""
    if mask.sum() < 3:
        return float("nan")
    x, y = a[mask], b[mask]
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 1e-9 else float("nan")


def _ncc_at(a, b, mask_a, mask_b, du, dv, min_overlap):
    """Correlation of ``a[v, u]`` with ``b[v + dv, u + du]`` over both masks."""
    H, W = a.shape
    ya0, ya1 = max(0, -dv), min(H, H - dv)
    xa0, xa1 = max(0, -du), min(W, W - du)
    if ya1 - ya0 < 2 or xa1 - xa0 < 2:
        return float("nan")
    m = mask_a[ya0:ya1, xa0:xa1] & mask_b[ya0 + dv:ya1 + dv, xa0 + du:xa1 + du]
    if m.sum() < min_overlap:
        return float("nan")
    return masked_ncc(a[ya0:ya1, xa0:xa1], b[ya0 + dv:ya1 + dv, xa0 + du:xa1 + du], m)


def _half(img: np.ndarray, mask: np.ndarray):
    """2x2 block mean over the mask; a block counts where all four are in it."""
    H, W = (img.shape[0] // 2) * 2, (img.shape[1] // 2) * 2
    m = mask[:H, :W].reshape(H // 2, 2, W // 2, 2)
    v = np.where(mask, img, 0.0)[:H, :W].reshape(H // 2, 2, W // 2, 2)
    return v.mean(axis=(1, 3)), m.all(axis=(1, 3))


def best_shift(a: np.ndarray, b: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray, max_px: int,
               min_overlap: int = 100):
    """The integer shift (du, dv) of ``b`` against ``a`` that maximises their
    correlation over the overlap of the masks: ``a[v, u]`` is compared with
    ``b[v + dv, u + du]``. Coarse to fine: every shift within ``max_px`` at
    half resolution, then the nine around the best at full resolution.
    Returns (ncc at that shift, du, dv, ncc at zero)."""
    zero = _ncc_at(a, b, mask_a, mask_b, 0, 0, min_overlap)
    ah, mah = _half(a, mask_a)
    bh, mbh = _half(b, mask_b)
    r = max(int(np.ceil(max_px / 2)), 1)
    coarse = (-2.0, 0, 0)
    for dv in range(-r, r + 1):
        for du in range(-r, r + 1):
            c = _ncc_at(ah, bh, mah, mbh, du, dv, max(min_overlap // 4, 10))
            if np.isfinite(c) and c > coarse[0]:
                coarse = (c, du, dv)
    best = (zero if np.isfinite(zero) else -2.0, 0, 0)
    for dv in range(2 * coarse[2] - 1, 2 * coarse[2] + 2):
        for du in range(2 * coarse[1] - 1, 2 * coarse[1] + 2):
            if abs(du) > max_px or abs(dv) > max_px:
                continue
            c = _ncc_at(a, b, mask_a, mask_b, du, dv, min_overlap)
            if np.isfinite(c) and c > best[0]:
                best = (c, du, dv)
    return (best[0] if best[0] > -2 else float("nan")), best[1], best[2], zero


@dataclass
class FrameScore:
    frame: str
    head: str
    n: int                                 # grid points scored
    ncc: float                             # at the model's own placement
    ncc_best: float                        # at the best shift
    shift_m: tuple                         # (along or x, up or y) in metres
    source: bool                           # True: scored on the points this frame textured (the ceiling)
    at_limit: bool = False                 # the best shift is at the edge of the search: not a measurement


def score_face(g: FaceGrid, cams: Sequence[dict], project: Callable, min_points: int = 150,
               min_share: float = 0.10, sigma_m: float = 1.0, max_shift_m: float = 1.5) -> List[FrameScore]:
    """Every frame's view of face ``g``, scored against the model.

    ``project(xyz, normals, cam)`` is colorize_building.project_visible (the
    facing, crop, mask and depth-buffer tests). The source of each grid
    point is the frame that faces it most squarely among those that see it
    -- the rule the texturing used. A frame is scored on the points it is
    not the source of; its score on its own points is returned too, marked
    ``source``, as the ceiling the texture and its compression allow.
    """
    pts = g.xyz[g.valid]; nrm = g.normal[g.valid]
    n = len(pts)
    views, facing = [], np.full((len(cams), n), -np.inf)
    for k, cam in enumerate(cams):
        idx, u, v, f = project(pts, nrm, cam)
        col = np.full((n, 3), np.nan)
        col[idx] = cam["img"][v, u]
        facing[k, idx] = f
        views.append(col)
    src = np.argmax(facing, axis=0)
    seen_any = np.isfinite(facing.max(axis=0))
    H, W = g.valid.shape
    sigma_px = sigma_m / g.spacing_m
    max_px = int(round(max_shift_m / g.spacing_m))
    model_hp = highpass(luminance(g.model), g.valid, sigma_px)
    out = []
    for k, cam in enumerate(cams):
        seen = np.isfinite(views[k][:, 0])
        for is_src in (False, True):
            use = seen & seen_any & ((src == k) if is_src else (src != k))
            if use.sum() < min_points or use.sum() < min_share * n:
                continue
            mask = np.zeros((H, W), dtype=bool); mask[g.valid] = use
            img = np.zeros((H, W, 3)); img[g.valid] = np.nan_to_num(views[k])
            frame_hp = highpass(luminance(img), mask, sigma_px)
            if is_src:
                c0 = masked_ncc(model_hp, frame_hp, mask)
                cb, du, dv = c0, 0, 0
            else:
                cb, du, dv, c0 = best_shift(model_hp, frame_hp, g.valid, mask, max_px)
            out.append(FrameScore(frame=cam.get("id", str(k)), head=cam.get("head", ""), n=int(use.sum()),
                                  ncc=c0, ncc_best=cb, shift_m=(du * g.spacing_m, dv * g.spacing_m), source=is_src,
                                  at_limit=max(abs(du), abs(dv)) >= max_px))
    return out


def summarise_face(g: FaceGrid, scores: Sequence[FrameScore], clear_gain: float = 0.10,
                   low: float = 0.3) -> Dict[str, Any]:
    """One face's line in texture_check.json. A shift counts as *clear* when
    lining up improves the correlation by at least ``clear_gain`` and the
    best shift is inside the search, not at its edge -- on a facade that
    repeats every metre and a half (a curtain wall's mullions) the edge of a
    1.5 m search is where the next repeat lines up, and that is not a
    measurement. The face's shift is the median of the clear ones."""
    held = [s for s in scores if not s.source and np.isfinite(s.ncc)]
    own = [s for s in scores if s.source and np.isfinite(s.ncc)]
    clear = [s for s in held if np.isfinite(s.ncc_best) and s.ncc_best - s.ncc >= clear_gain and not s.at_limit]
    shifts = np.array([np.hypot(*s.shift_m) for s in clear]) if clear else np.zeros(0)
    ncc = float(np.median([s.ncc for s in held])) if held else None
    # frames from one flight line share sun and exposure: agreement per line
    from .corners import flight_line_of
    lines: Dict[str, List[float]] = {}
    for s in held:
        lines.setdefault(flight_line_of(s.frame), []).append(s.ncc)
    return {
        "kind": g.kind, "area_m2": round(g.area_m2, 1), "spacing_m": round(g.spacing_m, 3),
        "frames_checked": len(held),
        "ncc": round(ncc, 3) if ncc is not None else None,
        "ncc_by_flight_line": {k: round(float(np.median(v)), 3) for k, v in sorted(lines.items())},
        # low agreement that no shift explains: glass reflecting each view
        # differently, a structure seen through (columns, open decks), or an
        # occluder the depth maps miss -- not a placement error; see the sheet
        "low_without_shift": bool(ncc is not None and ncc < low and not clear),
        "ncc_source": round(float(np.median([s.ncc for s in own])), 3) if own else None,
        "clear_shifts": len(clear),
        "shift_m": round(float(np.median(shifts)), 2) if len(shifts) else None,
        "frames": [{"frame": s.frame, "head": s.head, "n": s.n, "ncc": round(s.ncc, 3),
                    "ncc_best": round(s.ncc_best, 3), "shift_m": [round(v, 2) for v in s.shift_m],
                    "source": s.source, "at_limit": s.at_limit} for s in scores],
    }
