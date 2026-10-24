"""Step 4, a LiDAR tile's buildings as batched 3D Tiles, the way Cesium's
own building tilesets are laid out: many buildings in each tile file, each
still a feature with its own id and properties.

For one tile and one product (``measured`` or ``complete``) this writes, to
``tiles/<tile>/3dtiles/<product>/``:

* ``tileset.json`` -- placed in an east-north-up frame at the tile's centre;
  its root carries the tile's product 1 boxes and, as children, the areas;
* ``boxes.glb`` -- every building of the tile as its footprint extruded to
  its LiDAR height (product 1), each a feature: what the viewer draws from a
  kilometre or more away;
* ``cells/<hash>.glb`` -- one per area. An area starts as the whole tile and
  is split into four while it holds more than ``max_buildings`` buildings or
  ``max_bytes`` of meshes, so a rural tile is a file or two and downtown many
  small ones. Each area's glb holds all its buildings: their triangles moved
  into the tile's frame, their texture pages repacked into a few shared
  atlases (from the lossless pages the building step keeps, encoded to KTX2
  once), a feature id per building (``EXT_mesh_features``) and a property
  table (``EXT_structural_metadata``: GERS id, name, height, archetype,
  path). A building with no mesh stands in its area as its box, so nothing
  disappears when the viewer swaps boxes for meshes.

Files are named by a hash of their contents: a changed building gives its
area a new name, so a CDN never serves a stale one.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .coordinates import METERS_PER_US_SURVEY_FOOT

#: A tile's boxes are swapped for its areas within about 54 x this many metres
#: (CesiumJS's 16 px screen error, a 1,000 px view at 60 degrees): ~1 km.
TILE_ERROR_M = 20.0
#: An area is split into four while it holds more than this many buildings...
MAX_BUILDINGS = 100
#: ...or more than this many bytes of building meshes.
MAX_BYTES = 1_500_000
MAX_DEPTH = 8
#: Each product and the mesh folders it draws from, in order of preference.
#: "overture" (Overture's own heights) and "boxes" (LiDAR heights, product
#: 1) have none: every building is a box.
#: The "_tint" ones are the review copies of products 3 and 4, every
#: invented texel tinted (magenta the completed fill, violet the borrowed),
#: read from the tinted copy each build keeps in its review.zip (a source
#: ending in TINT); a building without a fill shows its measured mesh.
PRODUCTS = {"overture": (), "boxes": (),
            "measured": ("3dtiles_mesh",), "complete": ("3dtiles_complete", "3dtiles_mesh"),
            "neural": ("3dtiles_neural", "3dtiles_complete", "3dtiles_mesh"),
            "complete_tint": ("3dtiles_complete:tint", "3dtiles_mesh"),
            "neural_tint": ("3dtiles_neural:tint", "3dtiles_complete:tint", "3dtiles_mesh")}
TINT = ":tint"
BOX_RGB = (205, 203, 196)
#: The sources_choice fields shown on each building (see pieces.PROVENANCE).
PROVENANCE_FIELDS = ("lidar_phase", "lidar_date", "oblique_collection", "oblique_date", "gap_months")
GLTF_TO_ZUP = np.array([[1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])   # glTF y-up -> z-up


def _scripts():
    """build_mesh.py's writer and packer, and overture_boxes.py's extrusion."""
    from .stages import build_mesh, overture_boxes

    return build_mesh, overture_boxes


def col_major(m) -> np.ndarray:
    return np.array(m, dtype=np.float64).reshape(4, 4).T


def to_col_major(M: np.ndarray) -> List[float]:
    return [float(v) for v in np.asarray(M).T.ravel()]


# ------------------------------------------------------------------ reading

def read_glb(data: bytes) -> Tuple[dict, bytes]:
    """(the glTF JSON, the binary chunk)."""
    n = struct.unpack_from("<I", data, 12)[0]
    js = json.loads(data[20:20 + n])
    rest = data[20 + n:]
    binary = rest[8:8 + struct.unpack_from("<I", rest, 0)[0]] if len(rest) >= 8 else b""
    return js, binary


_CTYPES = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5122: np.int16, 5121: np.uint8, 5120: np.int8}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def accessor(js: dict, binary: bytes, i: int) -> np.ndarray:
    """An accessor's values as stored (a quantized one stays in its
    integers: the node matrix, or ``normalized``, turns them back)."""
    a = js["accessors"][i]
    v = js["bufferViews"][a["bufferView"]]
    n = _NCOMP[a["type"]]
    dt = np.dtype(_CTYPES[a["componentType"]])
    start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = v.get("byteStride", 0)
    if stride and stride != n * dt.itemsize:                        # rows padded out
        rows = np.frombuffer(binary, dtype=np.uint8, count=stride * (a["count"] - 1) + n * dt.itemsize, offset=start)
        rows = np.lib.stride_tricks.as_strided(rows, shape=(a["count"], n * dt.itemsize), strides=(stride, 1))
        arr = np.ascontiguousarray(rows).view(dt).reshape(a["count"], n)
        return arr if n > 1 else arr[:, 0]
    arr = np.frombuffer(binary, dtype=dt, count=a["count"] * n, offset=start)
    return arr.reshape(a["count"], n) if n > 1 else arr


def building_parts(glb: bytes, pages: Dict[str, np.ndarray]) -> List[dict]:
    """A building's primitives as arrays in its own glTF frame (y-up):
    ``pos``, ``nrm``, ``idx`` and one of ``uv`` + ``page`` (a textured
    primitive, the page from ``pages`` by material name), ``rgb`` (flat
    colour) or ``col`` (per-vertex colour, linear)."""
    js, binary = read_glb(glb)
    out = []
    for prim in js["meshes"][0]["primitives"]:
        at = prim["attributes"]
        mat = js["materials"][prim["material"]]
        p = {"pos": accessor(js, binary, at["POSITION"]).astype(np.float64),
             "nrm": accessor(js, binary, at["NORMAL"]).astype(np.float64),
             "idx": accessor(js, binary, prim["indices"]).astype(np.int64)}
        pbr = mat.get("pbrMetallicRoughness", {})
        if "baseColorTexture" in pbr and "TEXCOORD_0" in at:
            name = mat["name"]
            if name not in pages:
                raise KeyError(f"no lossless page {name!r} for a textured primitive (built without --dump-textures?)")
            p["uv"] = accessor(js, binary, at["TEXCOORD_0"]).astype(np.float64)   # glTF: t from the top
            p["page"] = name
        elif "COLOR_0" in at:
            p["col"] = accessor(js, binary, at["COLOR_0"]).astype(np.float64)
        else:
            f = pbr.get("baseColorFactor", [0.8, 0.8, 0.8, 1.0])
            p["rgb"] = tuple(int(round(c * 255)) for c in f[:3])
        out.append(p)
    return out


# --------------------------------------------------------------- the areas

def split_areas(centres: np.ndarray, sizes: np.ndarray, max_buildings: int = MAX_BUILDINGS,
                max_bytes: int = MAX_BYTES, max_depth: int = MAX_DEPTH) -> List[np.ndarray]:
    """Indices of the buildings in each area: a quadtree over their centres
    (x, y in the tile's frame), a square split into four while it holds more
    than ``max_buildings`` or ``max_bytes``. Every building is in exactly one
    area; empty squares are dropped."""
    if len(centres) == 0:
        return []
    lo, hi = centres.min(axis=0), centres.max(axis=0)
    half = max(float((hi - lo).max()) / 2.0, 1.0) + 1e-6
    mid = (lo + hi) / 2.0
    out: List[np.ndarray] = []

    def go(idx: np.ndarray, cx: float, cy: float, h: float, depth: int):
        if len(idx) == 0:
            return
        if (len(idx) <= max_buildings and sizes[idx].sum() <= max_bytes) or depth >= max_depth or len(idx) == 1:
            out.append(idx)
            return
        x, y = centres[idx, 0], centres[idx, 1]
        for east in (False, True):
            for north in (False, True):
                m = ((x >= cx) == east) & ((y >= cy) == north)
                go(idx[m], cx + (h / 2 if east else -h / 2), cy + (h / 2 if north else -h / 2), h / 2, depth + 1)

    go(np.arange(len(centres)), float(mid[0]), float(mid[1]), half, 0)
    return out


def box_volume(lo: np.ndarray, hi: np.ndarray) -> Dict[str, list]:
    """A 3D Tiles box: its centre and three half-axes."""
    c = (lo + hi) / 2.0
    h = np.maximum((hi - lo) / 2.0, 0.01)
    return {"box": [*c.tolist(), float(h[0]), 0.0, 0.0, 0.0, float(h[1]), 0.0, 0.0, 0.0, float(h[2])]}


# ------------------------------------------------------------- the writer

class FeatureGlb:
    """A glb of many buildings: primitives grouped by material, a feature id
    per vertex, and a property table per building. Wraps build_mesh's
    GlbWriter for the textures (KTX2) and the binary layout."""

    SCHEMA = {"id": "kyfromabove_3d", "classes": {"building": {"name": "Building", "properties": {
        "gers": {"type": "STRING", "description": "Overture GERS id (a split part: <gers>-a, -b)"},
        "name": {"type": "STRING"},
        "height_m": {"type": "SCALAR", "componentType": "FLOAT32",
                     "description": "the height drawn: LiDAR (product 1), or Overture's in the Overture view"},
        "height_from": {"type": "STRING", "description": "lidar, height (Overture's), floors (x 3.5 m), default (6 m)"},
        "overture_height_m": {"type": "SCALAR", "componentType": "FLOAT32", "description": "Overture's own height"},
        "lidar_height_m": {"type": "SCALAR", "componentType": "FLOAT32",
                           "description": "the LiDAR height, beside Overture's (Overture view)"},
        "archetype": {"type": "STRING"},
        "path": {"type": "STRING", "description": "how the mesh was built; 'box' where there is no mesh"},
        "product": {"type": "STRING"},
        "lidar_phase": {"type": "STRING", "description": "KyFromAbove LiDAR phase the shape came from"},
        "lidar_date": {"type": "STRING", "description": "when that LiDAR was flown"},
        "oblique_collection": {"type": "STRING", "description": "KyFromAbove oblique collection the colour came from"},
        "oblique_date": {"type": "STRING", "description": "when those frames were taken"},
        "gap_months": {"type": "SCALAR", "componentType": "FLOAT32",
                       "description": "months from the LiDAR to the frames (negative: frames first)"}}}}}

    def __init__(self, texture_format: str = "ktx2"):
        build_mesh, _ = _scripts()
        self.w = build_mesh.GlbWriter(texture_format)
        self.features: List[dict] = []
        self.quant: Optional[Tuple[np.ndarray, np.ndarray]] = None      # (translation, scale), glTF y-up

    def _view8(self, data: bytes) -> int:
        """A buffer view on an 8-byte boundary (the metadata extension asks for it)."""
        while len(self.w.bin) % 8:
            self.w.bin += b"\0"
        self.w.views.append({"buffer": 0, "byteOffset": len(self.w.bin), "byteLength": len(data)})
        self.w.bin += data
        return len(self.w.views) - 1

    def primitive(self, pos_zup, nrm_zup, idx, fid, material, uv_vup=None, col=None):
        w = self.w
        p = np.column_stack([pos_zup[:, 0], pos_zup[:, 2], -pos_zup[:, 1]]).astype(np.float32)
        n = np.column_stack([nrm_zup[:, 0], nrm_zup[:, 2], -nrm_zup[:, 1]]).astype(np.float32)
        attrs = {"POSITION": w._accessor(p, 5126, "VEC3", 34962, minmax=True),
                 "NORMAL": w._accessor(n, 5126, "VEC3", 34962)}
        if uv_vup is not None:
            t = np.column_stack([uv_vup[:, 0], 1.0 - uv_vup[:, 1]]).astype(np.float32)
            attrs["TEXCOORD_0"] = w._accessor(t, 5126, "VEC2", 34962)
        if col is not None:
            attrs["COLOR_0"] = w._accessor(np.asarray(col, dtype=np.float32), 5126, "VEC3", 34962)
        fid = np.asarray(fid)
        # float32: vertex attributes must sit on 4-byte boundaries (a
        # uint16 id failed the glTF validator), and float32 holds whole
        # numbers exactly far past any file's building count
        attrs["_FEATURE_ID_0"] = w._accessor(fid.astype(np.float32), 5126, "SCALAR", 34962)
        w.prims.append({"attributes": attrs, "material": material,
                        "indices": w._accessor(np.asarray(idx).astype(np.uint32), 5125, "SCALAR", 34963),
                        "extensions": {"EXT_mesh_features": {"featureIds": [
                            {"featureCount": int(len(np.unique(fid))), "attribute": 0, "propertyTable": 0}]}}})

    def _strided(self, arr: np.ndarray, ctype: int, stride: int, target: int, normalized: bool = False,
                 minmax: bool = False) -> int:
        """A VEC3 accessor over rows padded to ``stride`` bytes (a vertex
        attribute must start every row on a 4-byte boundary)."""
        a4 = np.zeros((len(arr), 4), dtype=arr.dtype)
        a4[:, :3] = arr
        v = self.w._view(a4.tobytes(), target)
        self.w.views[v]["byteStride"] = stride
        acc = {"bufferView": v, "componentType": ctype, "count": int(len(arr)), "type": "VEC3"}
        if normalized:
            acc["normalized"] = True
        if minmax:
            acc["min"], acc["max"] = arr.min(axis=0).tolist(), arr.max(axis=0).tolist()
        self.w.accessors.append(acc)
        return len(self.w.accessors) - 1

    def box_primitives(self, pos_zup, nrm_zup, idx, fid, material, max_vertices: int = 65535):
        """One group of boxes in one material: :meth:`box_groups`."""
        self.box_groups([(pos_zup, nrm_zup, idx, fid, material)], max_vertices)

    def box_groups(self, groups, max_vertices: int = 65535):
        """Untextured geometry (the boxes), small: ``groups`` of (positions,
        normals, indices, feature ids, material), all in one frame for the
        whole glb -- positions as int16 (``KHR_mesh_quantization``; the node
        matrix scales them back), normals as int8, and 2-byte indices, each
        group's geometry cut, between buildings, into primitives of at most
        ``max_vertices``. About 16 bytes a vertex instead of 28, and 2 an
        index instead of 4. Vertices and triangles must come building by
        building, as :func:`_merge` gives them. Once per glb."""
        assert self.quant is None, "one quantized frame per glb"
        yup = lambda a: np.column_stack([a[:, 0], a[:, 2], -a[:, 1]]).astype(np.float64)
        allp = np.vstack([yup(np.asarray(g[0])) for g in groups])
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        scale = np.maximum(hi - lo, 1e-6) / 65534.0
        self.quant = (lo + 32767.0 * scale, scale)                       # p = translation + q * scale
        for pos_zup, nrm_zup, idx, fid, material in groups:
            q = np.round((yup(np.asarray(pos_zup)) - lo) / scale).astype(np.int64) - 32767
            q16 = q.astype(np.int16)
            n8 = np.round(yup(np.asarray(nrm_zup)) * 127.0).astype(np.int8)
            fid, tri = np.asarray(fid), np.asarray(idx).reshape(-1, 3)
            starts = np.flatnonzero(np.r_[True, fid[1:] != fid[:-1]]).tolist() + [len(fid)]
            chunks, a, last = [], 0, 0
            for s in starts[1:]:
                if s - a > max_vertices:
                    assert last > a, "one building has more vertices than a primitive may"
                    chunks.append((a, last))
                    a = last
                last = s
            chunks.append((a, len(fid)))
            first = tri.min(axis=1)
            for a, b in chunks:
                t = tri[(first >= a) & (first < b)] - a
                assert t.size == 0 or (t.min() >= 0 and t.max() < b - a)
                f = fid[a:b]
                attrs = {"POSITION": self._strided(q16[a:b], 5122, 8, 34962, minmax=True),
                         "NORMAL": self._strided(n8[a:b], 5120, 4, 34962, normalized=True),
                         "_FEATURE_ID_0": self.w._accessor(f.astype(np.float32), 5126, "SCALAR", 34962)}
                self.w.prims.append({"attributes": attrs, "material": material,
                                     "indices": self.w._accessor(t.ravel().astype(np.uint16), 5123, "SCALAR", 34963),
                                     "extensions": {"EXT_mesh_features": {"featureIds": [
                                         {"featureCount": int(len(np.unique(f))), "attribute": 0, "propertyTable": 0}]}}})

    def _property_table(self) -> dict:
        props = {}
        for name, spec in self.SCHEMA["classes"]["building"]["properties"].items():
            vals = [f.get(name) for f in self.features]
            if all(v is None for v in vals) and spec["type"] == "STRING":
                # a text property no feature has is left out (its values would be a zero-length view,
                # which is invalid); a number is kept, all NaN: the validator fails on a number left out
                continue
            if spec["type"] == "STRING":
                enc = [("" if v is None else str(v)).encode("utf-8") for v in vals]
                offsets = np.cumsum([0] + [len(e) for e in enc]).astype(np.uint32)
                props[name] = {"values": self._view8(b"".join(enc)), "stringOffsets": self._view8(offsets.tobytes()),
                               "stringOffsetType": "UINT32"}
            else:
                arr = np.array([np.nan if v is None else float(v) for v in vals], dtype=np.float32)
                props[name] = {"values": self._view8(arr.tobytes())}
        return {"class": "building", "count": len(self.features), "properties": props}

    def to_bytes(self, extras: dict, node_matrix: Optional[np.ndarray] = None) -> bytes:
        """The glb. ``node_matrix`` (4x4, z-up like a tileset transform)
        places the content itself -- for content whose tiles may carry no
        transform, because tilesets beneath them carry their own."""
        w = self.w
        table = self._property_table()
        while len(w.bin) % 4:
            w.bin += b"\0"
        used = ["KHR_materials_unlit", "EXT_mesh_features", "EXT_structural_metadata"]
        required = []
        if w.textures and w.texture_extension:
            used.append(w.texture_extension)
            required.append(w.texture_extension)
        # the node's matrix lives in glTF's y-up space: the placement, then
        # (quantized content) the int16 positions back to metres
        N = None if node_matrix is None else np.linalg.inv(GLTF_TO_ZUP) @ node_matrix @ GLTF_TO_ZUP
        if self.quant is not None:
            used.append("KHR_mesh_quantization")
            required.append("KHR_mesh_quantization")
            Q = np.diag([*self.quant[1], 1.0])
            Q[:3, 3] = self.quant[0]
            N = Q if N is None else N @ Q
        gltf = {"asset": {"version": "2.0", "generator": "kyfromabove-3d publish.py"},
                "extensionsUsed": used, "scene": 0, "scenes": [{"nodes": [0]}],
                "nodes": [{"mesh": 0, "name": extras.get("name", "area"), "extras": extras,
                           **({"matrix": to_col_major(N)} if N is not None else {})}],
                "meshes": [{"primitives": w.prims}], "materials": w.materials,
                "textures": w.textures, "images": w.images,
                "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
                "accessors": w.accessors, "bufferViews": w.views, "buffers": [{"byteLength": len(w.bin)}],
                "extensions": {"EXT_structural_metadata": {"schema": self.SCHEMA, "propertyTables": [table]}}}
        if not w.textures:
            del gltf["textures"], gltf["images"], gltf["samplers"]
        if required:
            gltf["extensionsRequired"] = required
        js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        js += b" " * ((4 - len(js) % 4) % 4)
        total = 12 + 8 + len(js) + 8 + len(w.bin)
        buf = io.BytesIO()
        buf.write(struct.pack("<4sII", b"glTF", 2, total))
        buf.write(struct.pack("<II", len(js), 0x4E4F534A)); buf.write(js)
        buf.write(struct.pack("<II", len(w.bin), 0x004E4942)); buf.write(bytes(w.bin))
        return buf.getvalue()


def _merge(groups: Dict[Any, dict], key, pos, nrm, idx, fid, uv=None, col=None):
    g = groups.setdefault(key, {"pos": [], "nrm": [], "idx": [], "fid": [], "uv": [], "col": [], "n": 0})
    g["pos"].append(pos); g["nrm"].append(nrm); g["idx"].append(idx + g["n"]); g["fid"].append(fid)
    if uv is not None:
        g["uv"].append(uv)
    if col is not None:
        g["col"].append(col)
    g["n"] += len(pos)


def area_glb(members: Sequence[dict], M_of: Callable[[dict], np.ndarray], product: str,
             texture_format: str = "ktx2", page_px: int = 4096) -> Tuple[bytes, np.ndarray, np.ndarray]:
    """One area's glb. ``members``: each a building (``record``, and either
    ``parts`` + ``pages`` from :func:`building_parts` or ``box`` = (P, N, I)
    in the tile's frame, z-up). ``M_of(member)`` is the 4x4 taking the
    building's glTF content (after the y-up -> z-up turn) into the tile's
    frame. Returns (the glb, the content's lower and upper corners, z-up).
    An area of boxes only is written small (:func:`boxes_glb`)."""
    if not any("parts" in m for m in members):
        return boxes_glb([m["box"] for m in members], [box_feature(m["record"], m.get("box_product", "boxes"))
                                                       for m in members],
                         {"name": "area", "buildings": len(members), "product": product}, texture_format,
                         colours=[m.get("colour", BOX_RGB) for m in members])
    build_mesh, _ = _scripts()
    out = FeatureGlb(texture_format)
    atlas = build_mesh.TextureAtlas(page_px=page_px)
    handle: Dict[Tuple[int, str], int] = {}
    for fi, m in enumerate(members):
        for name, img in (m.get("pages") or {}).items():
            handle[(fi, name)] = atlas.add(np.asarray(img)[..., :3])
    if atlas.images:
        atlas.pack()
    groups: Dict[Any, dict] = {}
    for fi, m in enumerate(members):
        r = m["record"]
        out.features.append({"gers": r["id"], "name": r.get("name"), "height_m": r.get("lidar_height_m"),
                             "archetype": m.get("archetype"), "path": m.get("path") or ("box" if "box" in m else None),
                             "product": product, **{k: r.get(k) for k in PROVENANCE_FIELDS}})
        if "box" in m:
            P, N, I = m["box"]
            _merge(groups, ("rgb", BOX_RGB), P, N, I, np.full(len(P), fi))
            continue
        M = M_of(m) @ GLTF_TO_ZUP
        R = M[:3, :3]
        for p in m["parts"]:
            pos = (M @ np.column_stack([p["pos"], np.ones(len(p["pos"]))]).T).T[:, :3]
            nrm = (R @ p["nrm"].T).T
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            fid = np.full(len(pos), fi)
            if "uv" in p:
                pi, x, y, w, h = atlas.places[handle[(fi, p["page"])]]
                H, W = atlas.pages[pi].shape[:2]
                s = (x + p["uv"][:, 0] * w) / W
                t = (y + p["uv"][:, 1] * h) / H                     # both measured from the top
                _merge(groups, ("page", pi), pos, nrm, p["idx"], fid, uv=np.column_stack([s, 1.0 - t]))
            elif "col" in p:
                _merge(groups, ("col",), pos, nrm, p["idx"], fid, col=p["col"])
            else:
                _merge(groups, ("rgb", p["rgb"]), pos, nrm, p["idx"], fid)
    mats: Dict[Any, int] = {}
    for key in sorted(groups, key=str):
        if key[0] == "page" and key not in mats:
            mats[key] = out.w.material_texture(atlas.pages[key[1]], f"page_{key[1]}")
        elif key[0] == "rgb" and key not in mats:
            mats[key] = out.w.material_flat(key[1], f"rgb_{key[1][0]}_{key[1][1]}_{key[1][2]}", unlit=key[1] != BOX_RGB)
        elif key[0] == "col" and key not in mats:
            mats[key] = out.w.material_flat((255, 255, 255), "vertex_colour")
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for key in sorted(groups, key=str):
        g = groups[key]
        pos = np.vstack(g["pos"])
        lo, hi = np.minimum(lo, pos.min(axis=0)), np.maximum(hi, pos.max(axis=0))
        out.primitive(pos, np.vstack(g["nrm"]), np.concatenate(g["idx"]), np.concatenate(g["fid"]), mats[key],
                      uv_vup=np.vstack(g["uv"]) if g["uv"] else None, col=np.vstack(g["col"]) if g["col"] else None)
    data = out.to_bytes({"name": "area", "buildings": len(members), "product": product})
    return data, lo, hi


# -------------------------------------------------------------- the tile

def tile_frame(tile: str, records: Sequence[dict], grid) -> dict:
    """coordinates.json for the tile's frame: its centre, at the median of
    its buildings' LiDAR ground."""
    from .pieces import tile_origin_ft

    x, y = tile_origin_ft(grid, tile)
    g = [r["lidar_ground_z"] for r in records if r.get("lidar_ground_z") is not None]
    z_ft = float(np.median(g)) / METERS_PER_US_SURVEY_FOOT if g else 0.0
    return {"origin_source_xyz": [x, y, z_ft], "horizontal_crs": "EPSG:3089", "horizontal_unit": "US survey foot",
            "vertical_datum": "NAVD88", "vertical_unit": "US survey foot",
            "scale_source_to_meter": METERS_PER_US_SURVEY_FOOT}


#: A box's outline is simplified to within this (metres) where the box is the
#: closest view of a building; a 1 m error is under a pixel from 60 m away.
BOX_SIMPLIFY_M = 1.0


def prism(poly, z0: float, h: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """A footprint pushed up to a height, z-up: each wall a quad of its own
    (flat shading needs its own normals), the roof one fan of the outline's
    own corners (constrained Delaunay), no floor. (P, N, I); None for a
    degenerate outline."""
    import shapely
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient

    poly = Polygon(poly.exterior) if poly.geom_type == "Polygon" else None
    if poly is None or poly.is_empty:
        return None
    if not poly.is_valid:
        fixed = poly.buffer(0)
        if fixed.is_empty:
            return None
        poly = fixed if fixed.geom_type == "Polygon" else max(fixed.geoms, key=lambda g: g.area)
        poly = Polygon(poly.exterior)
    if poly.area < 1e-6:
        return None
    poly = orient(poly, 1.0)                                       # counter-clockwise
    v = np.array(poly.exterior.coords)[:-1, :2]
    P, N, I = [], [], []
    for a, b in zip(v, np.roll(v, -1, axis=0)):
        n = np.array([b[1] - a[1], -(b[0] - a[0]), 0.0])
        n /= np.linalg.norm(n) + 1e-12
        k = len(P)
        P += [[a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z0 + h], [a[0], a[1], z0 + h]]
        N += [n.tolist()] * 4
        I += [k, k + 1, k + 2, k, k + 2, k + 3]
    k0 = len(P)
    at = {(round(x, 6), round(y, 6)): k0 + i for i, (x, y) in enumerate(v)}
    P += [[x, y, z0 + h] for x, y in v]
    N += [[0.0, 0.0, 1.0]] * len(v)
    for tri in shapely.constrained_delaunay_triangles(poly).geoms:
        c = np.array(tri.exterior.coords)[:3, :2]
        if (c[1, 0] - c[0, 0]) * (c[2, 1] - c[0, 1]) - (c[1, 1] - c[0, 1]) * (c[2, 0] - c[0, 0]) < 0:
            c = c[::-1]
        ids = []
        for x, y in c:
            key = (round(x, 6), round(y, 6))
            if key not in at:                                      # a corner the triangulation added
                at[key] = len(P)
                P.append([x, y, z0 + h])
                N.append([0.0, 0.0, 1.0])
            ids.append(at[key])
        I += ids
    return np.array(P, float), np.array(N, float), np.array(I, np.int64)


def box_height(r: dict, product: str = "boxes") -> Tuple[Optional[float], Optional[str]]:
    """The height a building's box is drawn to, and where it came from:
    Overture's own (its ``height``, else floors x 3.5 m, else a 6 m default:
    ``overture_boxes.height_of``) in the "overture" product; the LiDAR
    height in every other. (None, None) without one."""
    if product == "overture":
        _, ob = _scripts()
        h, kind = ob.height_of(r, prefer_lidar=False)
        return float(h), kind
    if not r.get("lidar_height_m"):
        return None, None
    return float(r["lidar_height_m"]), "lidar"


def box_colour(kind: Optional[str], product: str = "boxes") -> Tuple[int, int, int]:
    """Overture's boxes say where their height came from (``overture_boxes.
    COLOURS``: grey its own height, blue-grey floors, orange the default);
    LiDAR boxes are one grey."""
    if product == "overture":
        _, ob = _scripts()
        return tuple(ob.COLOURS.get(kind, BOX_RGB))
    return BOX_RGB


def record_box(r: dict, frame: dict, shape: str = "outline", simplify_m: float = BOX_SIMPLIFY_M,
               product: str = "boxes") -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """A building's box (footprint to its height, :func:`box_height`) in the
    tile's frame, z-up, standing on the LiDAR ground -- for Overture's
    heights too, so it sits on the published terrain; None without a ground
    or a height. ``shape``: "outline" (the footprint simplified to within
    ``simplify_m``) or "rectangle" (the smallest rectangle around it, at any
    angle: for views from afar)."""
    import shapely

    from .footprints import footprint_wgs84_to_local

    h, _ = box_height(r, product)
    if r.get("lidar_ground_z") is None or not h:
        return None
    ox, oy, oz = frame["origin_source_xyz"]
    try:
        poly = footprint_wgs84_to_local(r["wkt"], ox, oy)
        if shape == "rectangle":
            poly = shapely.oriented_envelope(poly)
        elif simplify_m > 0:
            poly = poly.simplify(simplify_m, preserve_topology=True)
        z0 = float(r["lidar_ground_z"]) - oz * METERS_PER_US_SURVEY_FOOT
        return prism(poly, z0, h)
    except Exception:                                               # noqa: BLE001 (a degenerate outline)
        return None


def box_feature(r: dict, product: str = "boxes") -> dict:
    """A box's properties: the height drawn and where it came from, and in
    the Overture view the LiDAR height beside it."""
    h, kind = box_height(r, product)
    f = {"gers": r["id"], "name": r.get("name"), "height_m": h, "height_from": kind,
         "overture_height_m": r.get("height"), "path": "box", "product": product,
         **{k: r.get(k) for k in PROVENANCE_FIELDS}}
    if product == "overture":
        f["lidar_height_m"] = r.get("lidar_height_m")
    return f


def boxes_glb(boxes: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]], features: Sequence[dict], extras: dict,
              texture_format: str = "ktx2", node_matrix: Optional[np.ndarray] = None,
              colours: Optional[Sequence[Tuple[int, int, int]]] = None):
    """One glb of boxes (each (P, N, I) with its feature, and its colour --
    default the LiDAR grey), quantized in one frame
    (:meth:`FeatureGlb.box_groups`), one primitive group per colour.
    Returns (the glb, lower and upper corners, z-up) or (None, None, None)
    when there is no box."""
    out = FeatureGlb(texture_format)
    groups: Dict[Any, dict] = {}
    for i, (box, f) in enumerate(zip(boxes, features)):
        c = tuple(colours[i]) if colours is not None else BOX_RGB
        _merge(groups, c, *box, np.full(len(box[0]), len(out.features)))
        out.features.append(f)
    if not groups:
        return None, None, None
    parts, lo, hi = [], np.full(3, np.inf), np.full(3, -np.inf)
    for c in sorted(groups):
        g = groups[c]
        pos = np.vstack(g["pos"])
        lo, hi = np.minimum(lo, pos.min(axis=0)), np.maximum(hi, pos.max(axis=0))
        name = "box" if c == BOX_RGB else f"box_{c[0]}_{c[1]}_{c[2]}"
        parts.append((pos, np.vstack(g["nrm"]), np.concatenate(g["idx"]), np.concatenate(g["fid"]),
                      out.w.material_flat(c, name, unlit=False)))
    out.box_groups(parts)
    return out.to_bytes(extras, node_matrix=node_matrix), lo, hi


def built_buildings(store, prefix: Optional[str] = None) -> Dict[str, set]:
    """{gers: the mesh products it has} from one listing of ``prefix``
    (default: ``buildings/``) -- instead of asking the store about
    each of a tile's buildings (a million requests for the pilot's 316,000,
    most of which have no mesh). A product is there when its zip is."""
    from .building import BUILDINGS_PREFIX

    prefix = (prefix or BUILDINGS_PREFIX).rstrip("/")
    subs = {s.removesuffix(TINT) for v in PRODUCTS.values() for s in v}
    out: Dict[str, set] = {}
    for k in store.list(f"{prefix}/"):
        parts = k[len(prefix) + 1:].split("/")
        if len(parts) == 2 and parts[1].endswith(".zip") and parts[1][:-4] in subs:
            out.setdefault(parts[0], set()).add(parts[1][:-4])
    return out


def load_building(store, gers: str, product: str, prefix: Optional[str] = None,
                  available: Optional[Dict[str, set]] = None) -> Optional[dict]:
    """A built building's glb, its tileset (placement) and its lossless
    pages, for ``product`` (the completed one falls back to the measured
    one where it was not made), from that product's zip; None if it has no
    mesh. ``available`` (:func:`built_buildings`) spares asking the store;
    without it each zip is read straight away and a missing one passed over --
    no existence check first (a boxes-only building cost one per view's zip).
    A tinted copy (a source ending in TINT) comes from the review zip, its
    pages decoded from the glb's own textures (no lossless ones are kept)."""
    from PIL import Image

    from .archive import read_zip
    from .building import BUILDINGS_PREFIX, REVIEW_ZIP

    prefix = prefix or BUILDINGS_PREFIX
    review = None
    for sub in PRODUCTS[product]:
        folder = sub.removesuffix(TINT)
        if available is not None and folder not in available.get(gers, ()):
            continue
        if sub.endswith(TINT):
            try:
                review = review or read_zip(store, f"{prefix}/{gers}/{REVIEW_ZIP}")
                ts = json.loads(review.read(f"{folder}/tileset_tint.json"))
            except (FileNotFoundError, KeyError):
                continue
            glb = review.read(f"{folder}/{ts['root']['content']['uri']}")
            return {"tileset": ts, "glb": glb, "pages": glb_pages(glb), "source": sub}
        try:
            z = read_zip(store, f"{prefix}/{gers}/{sub}.zip")
        except FileNotFoundError:
            continue
        names = set(z.namelist())
        ts = json.loads(z.read("tileset.json"))
        glb = z.read(ts["root"]["content"]["uri"])
        js, _ = read_glb(glb)
        pages = {}
        for mat in js.get("materials", []):
            if "baseColorTexture" in mat.get("pbrMetallicRoughness", {}):
                k = f"textures/{mat['name']}.png"
                if k in names:
                    pages[mat["name"]] = np.asarray(Image.open(io.BytesIO(z.read(k))).convert("RGB"))
        return {"tileset": ts, "glb": glb, "pages": pages, "source": sub}
    return None


def glb_pages(glb: bytes) -> Dict[str, np.ndarray]:
    """{material name: its base colour texture, (H, W, 3) uint8}, decoded
    from the images in the glb itself."""
    from .texture_check import _decode_image

    js, binary = read_glb(glb)
    pages = {}
    for mat in js.get("materials", []):
        ref = mat.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if ref is None:
            continue
        tex = js["textures"][ref["index"]]
        img = js["images"][tex.get("source", tex.get("extensions", {}).get("KHR_texture_basisu", {}).get("source"))]
        view = js["bufferViews"][img["bufferView"]]
        at = view.get("byteOffset", 0)
        pages[mat["name"]] = _decode_image(binary[at:at + view["byteLength"]], img.get("mimeType", "image/png"))
    return pages


def publish_tile(tile: str, store, grid, product: str = "measured", tiles_prefix: str = "tiles",
                 buildings_prefix: Optional[str] = None, out_store=None, max_buildings: int = MAX_BUILDINGS,
                 max_bytes: int = MAX_BYTES, texture_format: str = "ktx2", log: Callable = print,
                 available: Optional[Dict[str, set]] = None) -> dict:
    """Step 4 for one tile and one product. Reads the tile step's
    ``buildings.json`` and each building's outputs from ``store``; writes
    the tile's tileset, boxes and areas to ``out_store`` (default ``store``).
    ``available``: the built buildings (:func:`built_buildings`), known
    beforehand. Returns a summary."""
    import os

    import pyproj

    build_mesh, _ = _scripts()
    # PROJ picks its operations when a transformer is made, and to_ecef turns
    # the network (the grids) on the first time it runs: made before that,
    # the tile's centre came out 0.19 m from the same centre made after, so
    # the same tile gave different bytes. Settle it first, as to_ecef will.
    if os.environ.get("PROJ_NETWORK", "").upper() != "OFF":
        pyproj.network.set_network_enabled(True)
    out_store = out_store or store
    recs = store.read_json(f"{tiles_prefix}/{tile}/buildings.json")["buildings"]
    frame = tile_frame(tile, recs, grid)
    transform, origin_geo = build_mesh.enu_transform(frame)
    T_tile = col_major(transform)
    T_inv = np.linalg.inv(T_tile)
    base = f"{tiles_prefix}/{tile}/3dtiles/{product}"

    # the products with meshes show product 1's boxes where there is none
    box_product = product if not PRODUCTS[product] else "boxes"
    rects, rect_features, rect_colours = [], [], []
    members, meshed, boxed, missing = [], 0, 0, 0
    for r in recs:
        rect = record_box(r, frame, "rectangle", product=box_product)
        colour = box_colour(box_height(r, box_product)[1], box_product)
        if rect is not None:
            rects.append(rect)
            rect_features.append(box_feature(r, box_product))
            rect_colours.append(colour)
        b = (load_building(store, r["id"], product, buildings_prefix, available)
             if PRODUCTS[product] else None)
        if b is not None:
            T_b = col_major(b["tileset"]["root"]["transform"])
            M = T_inv @ T_b
            ex = b["tileset"].get("extras", {})
            members.append({"record": r, "parts": building_parts(b["glb"], b["pages"]), "pages": b["pages"],
                            "M": M, "archetype": ex.get("archetype"), "path": ex.get("path"),
                            "origin_lon_lat_h": ex.get("origin_lon_lat_h"),
                            "bytes": len(b["glb"]), "centre": M[:3, 3]})
            meshed += 1
            continue
        box = record_box(r, frame, product=box_product) if rect is not None else None
        if box is not None:
            members.append({"record": r, "box": box, "bytes": 600, "colour": colour, "box_product": box_product,
                            "centre": np.append(box[0][:, :2].mean(axis=0), 0.0)})
            boxed += 1
        else:
            missing += 1
    # Every building placed through the same geoid: its ellipsoidal height
    # less its NAVD88 ground is the geoid there, which varies by well under a
    # metre across a tile. One placed without the grid is ~34 m off. (Not a
    # height test: a Hazard tile's hills rise 150 m and more.)
    geoid_tile = origin_geo[2] - frame["origin_source_xyz"][2] * METERS_PER_US_SURVEY_FOOT
    off = []
    for m in members:
        h = (m.get("origin_lon_lat_h") or [None, None, None])[2]
        g = m["record"].get("lidar_ground_z")
        if h is not None and g is not None and abs((h - g) - geoid_tile) > 5.0:
            off.append((m["record"]["id"], round(float((h - g) - geoid_tile), 1)))
    if off:
        raise SystemExit(f"[FAIL] {len(off)} building(s) placed through a different geoid from the tile's "
                         f"(built without the grid?): {off[:5]}")

    # The tile's own content, seen from about 1 to 3 km: every building as a
    # rectangle to its height (a house is a few pixels there; the level's
    # geometric error allows 20 m). Up close its areas replace it: the
    # meshes, and outline boxes for the buildings without one.
    lo_all, hi_all = np.full(3, np.inf), np.full(3, -np.inf)
    box_key = None
    data, lo, hi = boxes_glb(rects, rect_features, {"name": f"{tile} rectangles", "buildings": len(rects)},
                             texture_format, colours=rect_colours)
    if data is not None:
        lo_all, hi_all = lo, hi
        box_key = f"{base}/boxes_{hashlib.sha1(data).hexdigest()[:16]}.glb"
        out_store.put_bytes(box_key, data, "model/gltf-binary", content_encoding="gzip")

    children, area_bytes = [], []
    if members:
        centres = np.array([m["centre"][:2] for m in members])
        sizes = np.array([m["bytes"] for m in members], dtype=float)
        for idx in split_areas(centres, sizes, max_buildings, max_bytes):
            area = [members[i] for i in idx]
            data, lo, hi = area_glb(area, lambda m: m["M"], product, texture_format)
            if data is None:
                continue
            key = f"{base}/cells/{hashlib.sha1(data).hexdigest()[:16]}.glb"
            out_store.put_bytes(key, data, "model/gltf-binary", content_encoding="gzip")
            area_bytes.append(len(data))
            lo_all, hi_all = np.minimum(lo_all, lo), np.maximum(hi_all, hi)
            children.append({"boundingVolume": box_volume(lo, hi), "geometricError": 0.0,
                             "content": {"uri": key[len(base) + 1:]},
                             "extras": {"buildings": len(area)}})
    if not np.isfinite(lo_all).all():
        return {"tile": tile, "product": product, "buildings": len(recs), "meshed": 0, "areas": 0}
    root = {"transform": transform, "boundingVolume": box_volume(lo_all, hi_all),
            "geometricError": TILE_ERROR_M if children else 0.0, "refine": "REPLACE"}
    if box_key:
        root["content"] = {"uri": box_key[len(base) + 1:]}
    if children:
        root["children"] = children
    summary = {"tile": tile, "product": product, "buildings": len(recs), "meshed": meshed, "boxed": boxed,
               "missing": missing, "areas": len(children), "area_bytes": int(sum(area_bytes)),
               "area_buildings_max": int(max((c["extras"]["buildings"] for c in children), default=0)),
               "origin_lon_lat_h": origin_geo}
    tileset = {"asset": {"version": "1.1", "generator": "kyfromabove-3d publish.py"},
               "geometricError": 500.0, "root": root, "extras": summary}
    out_store.put_json(f"{base}/tileset.json", tileset)
    log(f"{tile} {product}: {meshed} meshed, {boxed} as boxes, {missing} without a height; "
        f"{len(children)} area file(s), {sum(area_bytes) / 1e6:.1f} MB")
    return summary


# ------------------------------------------------------------ the tree

#: The levels above the tiles: (tiles a side, geometric error, the buildings
#: its coarse content keeps). Replaced by the next level within about 54 x
#: the error: 16 km for the 16 x 16 groups, 3.2 km for the 4 x 4. A building
#: is kept when its footprint's width (the square root of its area) or its
#: height reaches ``min_size_m`` -- a third of the 4 x 4's error, a sixth of
#: the 16 x 16's (so the overview still shows the big ones). Around
#: Louisville that keeps 10% and 0.6% of the buildings.
GROUP_LEVELS = ((16, 300.0, {"min_size_m": 50.0}),
                (4, 60.0, {"min_size_m": 20.0}))
ROOT_ERROR_M = 5000.0


def _tile_nums(tile: str) -> Tuple[int, int]:
    import re

    m = re.fullmatch(r"N(\d+)E(\d+)", tile)
    return int(m.group(1)), int(m.group(2))


def tileset_region(ts: dict) -> List[float]:
    """A tileset's root box (under its transform) as a 3D Tiles region:
    [west, south, east, north] in radians and heights in metres."""
    from pyproj import Transformer

    T = col_major(ts["root"]["transform"])
    b = ts["root"]["boundingVolume"]["box"]
    c, ax = np.array(b[:3]), np.array(b[3:]).reshape(3, 3)
    corners = np.array([c + sx * ax[0] + sy * ax[1] + sz * ax[2]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    world = (T @ np.column_stack([corners, np.ones(8)]).T).T[:, :3]
    lon, lat, h = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True).transform(*world.T)
    return [float(np.radians(np.min(lon))), float(np.radians(np.min(lat))), float(np.radians(np.max(lon))),
            float(np.radians(np.max(lat))), float(np.min(h)), float(np.max(h))]


def union_region(regions: Sequence[Sequence[float]]) -> List[float]:
    r = np.array(regions)
    return [float(r[:, 0].min()), float(r[:, 1].min()), float(r[:, 2].max()), float(r[:, 3].max()),
            float(r[:, 4].min()), float(r[:, 5].max())]


def footprint_area_m2(wkt: str) -> float:
    from pyproj import Geod
    from shapely.wkt import loads

    return abs(Geod(ellps="WGS84").geometry_area_perimeter(loads(wkt))[0])


def coarse_glb(records: Sequence[dict], grid, tiles: Sequence[str], keep: dict,
               texture_format: str = "ktx2", product: str = "boxes") -> Optional[bytes]:
    """The buildings of ``records`` big enough for ``keep`` as rectangles to
    their heights (the product's: :func:`box_height`), in a frame at the
    group's centre carried by the glb's own node matrix."""
    build_mesh, _ = _scripts()
    from .pieces import tile_origin_ft

    box_product = product if not PRODUCTS.get(product, ()) else "boxes"
    big = []
    for r in records:
        h, _ = box_height(r, box_product)
        if h and r.get("lidar_ground_z") is not None and max(footprint_area_m2(r["wkt"]) ** 0.5, h) >= keep["min_size_m"]:
            big.append(r)
    if not big:
        return None
    o = np.array([tile_origin_ft(grid, t) for t in tiles]).mean(axis=0)
    g = [r["lidar_ground_z"] for r in big]
    frame = {"origin_source_xyz": [float(o[0]), float(o[1]), float(np.median(g)) / METERS_PER_US_SURVEY_FOOT],
             "horizontal_crs": "EPSG:3089", "scale_source_to_meter": METERS_PER_US_SURVEY_FOOT}
    transform, _ = build_mesh.enu_transform(frame)
    boxes, feats, colours = [], [], []
    for r in big:
        box = record_box(r, frame, "rectangle", product=box_product)
        if box is not None:
            boxes.append(box)
            feats.append(box_feature(r, box_product))
            colours.append(box_colour(box_height(r, box_product)[1], box_product))
    data, _, _ = boxes_glb(boxes, feats, {"name": "coarse boxes", "buildings": len(boxes)}, texture_format,
                           node_matrix=col_major(transform), colours=colours)
    return data


def publish_trees(store, grid, products: Sequence[str], tiles: Optional[Sequence[str]] = None,
                  log: Callable = print, **kw) -> Dict[str, dict]:
    """:func:`publish_tree` for several views over one listing of the tiles,
    each tile's ``buildings.json`` read once for all of them. Returns
    {product: its summary, with ``tile_names``}."""
    tiles = sorted(store.list_dirs(f"{kw.get('tiles_prefix', 'tiles')}/")) if tiles is None else list(tiles)
    cache: Dict[str, list] = {}
    return {p: publish_tree(store, grid, p, tiles=tiles, records_cache=cache, log=log, **kw) for p in products}


def publish_tree(store, grid, product: str = "measured", tiles: Optional[Sequence[str]] = None,
                 tiles_prefix: str = "tiles", root_prefix: str = "kentucky", out_store=None,
                 log: Callable = print, records_cache: Optional[Dict[str, list]] = None) -> dict:
    """The statewide tree for ``product`` over the tiles that have a published
    tileset: ``<root_prefix>/<product>/tileset.json`` (the root, no content)
    and one external tileset per 16 x 16 group of tiles
    (``groups/<id>.json``), holding its coarse boxes, its 4 x 4 subgroups'
    and references to its tiles' tilesets. No tile in the tree carries a
    transform (a tileset beneath carries its own), so the coarse content
    places itself through its glb's node matrix."""
    import os

    import pyproj

    if os.environ.get("PROJ_NETWORK", "").upper() != "OFF":
        pyproj.network.set_network_enabled(True)
    out_store = out_store or store
    base = f"{root_prefix}/{product}"
    from concurrent.futures import ThreadPoolExecutor

    if tiles is None:
        tiles = store.list_dirs(f"{tiles_prefix}/")                     # one level: not every key under them
    cache = records_cache if records_cache is not None else {}

    def read(t):
        """The tile's tileset region and records; None where it has no tileset for the view."""
        try:
            reg = tileset_region(store.read_json(f"{tiles_prefix}/{t}/3dtiles/{product}/tileset.json"))
        except FileNotFoundError:
            return t, None, None
        recs = cache.get(t)
        if recs is None:
            recs = store.read_json(f"{tiles_prefix}/{t}/buildings.json")["buildings"]
        return t, reg, recs
    region = {}
    records = {}
    with ThreadPoolExecutor(32) as ex:                                  # the reads wait on the network
        for t, reg, recs in ex.map(read, sorted(tiles)):
            if reg is not None:
                region[t], records[t] = reg, recs
                cache[t] = recs
    tiles = sorted(region)
    (n16, e16_err, keep16), (n4, e4_err, keep4) = GROUP_LEVELS
    by16: Dict[Tuple[int, int], List[str]] = {}
    for t in tiles:
        n, e = _tile_nums(t)
        by16.setdefault((n // n16, e // n16), []).append(t)
    root_children = []
    for (gn, ge), members in sorted(by16.items()):
        gid = f"g{gn:03d}_{ge:03d}"
        by4: Dict[Tuple[int, int], List[str]] = {}
        for t in members:
            n, e = _tile_nums(t)
            by4.setdefault((n // n4, e // n4), []).append(t)
        kids4 = []
        for (fn, fe), sub in sorted(by4.items()):
            refs = [{"boundingVolume": {"region": region[t]}, "geometricError": 20.0,
                     "content": {"uri": f"../../../{tiles_prefix}/{t}/3dtiles/{product}/tileset.json"}}
                    for t in sorted(sub)]
            node = {"boundingVolume": {"region": union_region([region[t] for t in sub])},
                    "geometricError": e4_err, "refine": "REPLACE", "children": refs}
            data = coarse_glb([r for t in sub for r in records[t]], grid, sub, keep4, product=product)
            if data is not None:
                k = f"{base}/groups/{gid}_{fn:03d}_{fe:03d}_{hashlib.sha1(data).hexdigest()[:12]}.glb"
                out_store.put_bytes(k, data, "model/gltf-binary", content_encoding="gzip")
                node["content"] = {"uri": k[len(base) + len("/groups/"):]}
            kids4.append(node)
        g_region = union_region([region[t] for t in members])
        g_root = {"boundingVolume": {"region": g_region}, "geometricError": e16_err, "refine": "REPLACE",
                  "children": kids4}
        data = coarse_glb([r for t in members for r in records[t]], grid, members, keep16, product=product)
        if data is not None:
            k = f"{base}/groups/{gid}_{hashlib.sha1(data).hexdigest()[:12]}.glb"
            out_store.put_bytes(k, data, "model/gltf-binary", content_encoding="gzip")
            g_root["content"] = {"uri": k[len(base) + len("/groups/"):]}
        out_store.put_json(f"{base}/groups/{gid}.json",
                           {"asset": {"version": "1.1", "generator": "kyfromabove-3d publish.py"},
                            "geometricError": e16_err, "root": g_root,
                            "extras": {"group": gid, "tiles": len(members)}})
        root_children.append({"boundingVolume": {"region": g_region}, "geometricError": e16_err,
                              "content": {"uri": f"groups/{gid}.json"}})
    buildings = int(sum(len(v) for v in records.values()))
    summary = {"product": product, "tiles": len(tiles), "groups": len(root_children), "buildings": buildings}
    out_store.put_json(f"{base}/tileset.json", {
        "asset": {"version": "1.1", "generator": "kyfromabove-3d publish.py"}, "geometricError": ROOT_ERROR_M,
        "root": {"boundingVolume": {"region": union_region([c["boundingVolume"]["region"] for c in root_children])},
                 "geometricError": ROOT_ERROR_M, "refine": "REPLACE", "children": root_children},
        "extras": summary})
    log(f"tree {product}: {len(tiles)} tile(s) in {len(root_children)} group(s), {buildings:,} buildings")
    return {**summary, "tile_names": tiles}
