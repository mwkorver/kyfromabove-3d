"""The mesh written out: glTF (GLB) with its textures (KTX2, WebP, JPEG or PNG), and the
transform that places the local frame on the Earth for 3D Tiles."""

from __future__ import annotations

import io
import json
import shutil
import struct
from pathlib import Path

import numpy as np

from .textures import _mult4

TEXTURE_FORMATS = ("ktx2", "webp", "jpeg", "png")

# KTX2/Basis: the texture stays compressed on the GPU (about an eighth of a
# decoded PNG/JPEG/WebP) and needs no decode on the main thread -- what a
# city-scale view runs out of first. Encoded by the basisu command (Homebrew
# basis_universal): ETC1S, file size about a JPEG's, 4 bits per texel on the
# GPU. Cesium (KHR_texture_basisu) and three.js read it.
BASISU_ARGS = ["-ktx2", "-mipmap", "-q", "200", "-comp_level", "2"]


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """glTF vertex colours are linear; image pixels are sRGB."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


class GlbWriter:
    """Minimal glTF 2.0 binary writer: textured or flat-colour primitives.

    ``texture_format``: ``ktx2`` (the default; ``KHR_texture_basisu``,
    needs the ``basisu`` command), ``webp`` (``EXT_texture_webp``),
    ``jpeg`` or ``png`` (core glTF). The extension formats carry no
    fallback image -- a fallback would embed a second copy and undo the
    saving -- so they are listed as required. ``dump_dir`` keeps a PNG of
    every texture beside the glb, for looking at.
    """

    def __init__(self, texture_format: str = "ktx2", dump_dir: Path | None = None):
        if texture_format not in TEXTURE_FORMATS:
            raise ValueError(f"texture format {texture_format!r}; one of {TEXTURE_FORMATS}")
        if texture_format == "ktx2" and not shutil.which("basisu"):
            raise SystemExit("[FAIL] the ktx2 textures need the basisu encoder and it is not on PATH: "
                             "brew install basis_universal, or pass --texture webp|jpeg|png")
        self.texture_format = texture_format
        self.dump_dir = dump_dir
        self.bin = bytearray()
        self.views, self.accessors, self.images, self.textures, self.materials = [], [], [], [], []
        self.prims = []

    def _encode(self, image: np.ndarray, name: str) -> tuple[bytes, str]:
        """The texture's bytes and MIME type in the writer's format."""
        from PIL import Image
        if self.dump_dir is not None:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(self.dump_dir / f"{name}.png")
        fmt = self.texture_format
        if fmt == "ktx2":
            import subprocess
            import tempfile
            h, w = image.shape[:2]
            if h % 4 or w % 4:
                image = np.pad(image, ((0, _mult4(h) - h), (0, _mult4(w) - w), (0, 0)), mode="edge")
            with tempfile.TemporaryDirectory() as td:
                src, dst = Path(td) / f"{name}.png", Path(td) / f"{name}.ktx2"
                Image.fromarray(image).save(src)
                r = subprocess.run(["basisu", *BASISU_ARGS, "-output_file", str(dst), str(src)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                if r.returncode != 0 or not dst.exists():
                    raise RuntimeError(f"basisu failed on {name} ({image.shape[1]}x{image.shape[0]}): "
                                       + "\n".join(r.stdout.splitlines()[-5:]))
                return dst.read_bytes(), "image/ktx2"
        buf = io.BytesIO()
        if fmt == "webp":
            Image.fromarray(image).save(buf, format="WEBP", quality=85, method=4)
            return buf.getvalue(), "image/webp"
        if fmt == "jpeg":
            Image.fromarray(image).save(buf, format="JPEG", quality=90, subsampling=0)
            return buf.getvalue(), "image/jpeg"
        Image.fromarray(image).save(buf, format="PNG")
        return buf.getvalue(), "image/png"

    @property
    def texture_extension(self) -> str | None:
        return {"ktx2": "KHR_texture_basisu", "webp": "EXT_texture_webp"}.get(self.texture_format)

    def _view(self, data: bytes, target=None) -> int:
        while len(self.bin) % 4:
            self.bin += b"\0"
        v = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            v["target"] = target
        self.bin += data
        self.views.append(v)
        return len(self.views) - 1

    def _accessor(self, arr: np.ndarray, ctype: int, atype: str, target: int, minmax=False) -> int:
        arr = np.ascontiguousarray(arr)
        a = {"bufferView": self._view(arr.tobytes(), target), "componentType": ctype,
             "count": int(arr.shape[0]), "type": atype}
        if minmax:
            a["min"] = arr.min(axis=0).tolist(); a["max"] = arr.max(axis=0).tolist()
        self.accessors.append(a)
        return len(self.accessors) - 1

    def material_texture(self, image: np.ndarray, name: str) -> int:
        data, mime = self._encode(np.ascontiguousarray(image, dtype=np.uint8), name)
        self.images.append({"bufferView": self._view(data), "mimeType": mime, "name": name})
        source = len(self.images) - 1
        ext = self.texture_extension
        self.textures.append({"sampler": 0, "extensions": {ext: {"source": source}}} if ext
                             else {"source": source, "sampler": 0})
        # unlit: the imagery already carries the sun; a second lighting pass
        # in the viewer would darken every face the viewer's sun is behind
        self.materials.append({"name": name, "doubleSided": True, "extensions": {"KHR_materials_unlit": {}},
                               "pbrMetallicRoughness": {"baseColorTexture": {"index": len(self.textures) - 1},
                                                        "metallicFactor": 0.0, "roughnessFactor": 1.0}})
        return len(self.materials) - 1

    def material_flat(self, rgb, name: str, unlit: bool = True) -> int:
        m = {"name": name, "doubleSided": True,
             "pbrMetallicRoughness": {"baseColorFactor": [float(c) / 255 for c in rgb] + [1.0],
                                      "metallicFactor": 0.0, "roughnessFactor": 1.0}}
        if unlit:
            m["extensions"] = {"KHR_materials_unlit": {}}
        self.materials.append(m)
        return len(self.materials) - 1

    def primitive_coloured(self, pos: np.ndarray, nrm: np.ndarray, rgb: np.ndarray, idx: np.ndarray, material: int):
        """Per-vertex colour (COLOR_0) instead of a texture."""
        p = np.column_stack([pos[:, 0], pos[:, 2], -pos[:, 1]]).astype(np.float32)
        n = np.column_stack([nrm[:, 0], nrm[:, 2], -nrm[:, 1]]).astype(np.float32)
        attrs = {"POSITION": self._accessor(p, 5126, "VEC3", 34962, minmax=True),
                 "NORMAL": self._accessor(n, 5126, "VEC3", 34962),
                 "COLOR_0": self._accessor(_srgb_to_linear(rgb / 255.0).astype(np.float32), 5126, "VEC3", 34962)}
        self.prims.append({"attributes": attrs, "material": material,
                           "indices": self._accessor(idx.astype(np.uint32), 5125, "SCALAR", 34963)})

    def primitive(self, pos: np.ndarray, nrm: np.ndarray, uv, idx: np.ndarray, material: int):
        # local z-up -> glTF y-up: (x, y, z) -> (x, z, -y)
        p = np.column_stack([pos[:, 0], pos[:, 2], -pos[:, 1]]).astype(np.float32)
        n = np.column_stack([nrm[:, 0], nrm[:, 2], -nrm[:, 1]]).astype(np.float32)
        attrs = {"POSITION": self._accessor(p, 5126, "VEC3", 34962, minmax=True),
                 "NORMAL": self._accessor(n, 5126, "VEC3", 34962)}
        if uv is not None:
            t = np.column_stack([uv[:, 0], 1.0 - uv[:, 1]]).astype(np.float32)
            attrs["TEXCOORD_0"] = self._accessor(t, 5126, "VEC2", 34962)
        self.prims.append({"attributes": attrs, "material": material,
                           "indices": self._accessor(idx.astype(np.uint32), 5125, "SCALAR", 34963)})

    def write(self, path: Path, extras: dict):
        while len(self.bin) % 4:
            self.bin += b"\0"
        used = ["KHR_materials_unlit"]
        if self.textures and self.texture_extension:
            used.append(self.texture_extension)
        gltf = {"asset": {"version": "2.0", "generator": "kyfromabove-3d build_mesh.py"},
                "extensionsUsed": used, "scene": 0, "scenes": [{"nodes": [0]}],
                "nodes": [{"mesh": 0, "name": extras.get("name", "building"), "extras": extras}],
                "meshes": [{"primitives": self.prims}],
                "materials": self.materials, "textures": self.textures, "images": self.images,
                "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
                "accessors": self.accessors, "bufferViews": self.views,
                "buffers": [{"byteLength": len(self.bin)}]}
        if not self.textures:
            del gltf["textures"], gltf["images"], gltf["samplers"]
        elif self.texture_extension:
            gltf["extensionsRequired"] = [self.texture_extension]
        js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        js += b" " * ((4 - len(js) % 4) % 4)
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        with open(path, "wb") as fh:
            fh.write(struct.pack("<4sII", b"glTF", 2, total))
            fh.write(struct.pack("<II", len(js), 0x4E4F534A)); fh.write(js)
            fh.write(struct.pack("<II", len(self.bin), 0x004E4942)); fh.write(bytes(self.bin))
        return total


class GlbRecorder:
    """A GlbWriter that remembers every primitive and material call, so the
    same mesh can be written again with other textures (:meth:`replay`) --
    the completed product is the measured one, triangle for triangle."""

    RECORDED = ("primitive", "primitive_coloured", "material_texture", "material_flat")

    def __init__(self, writer: "GlbWriter"):
        self.writer = writer
        self.calls = []

    def __getattr__(self, name):
        fn = getattr(self.writer, name)
        if name not in self.RECORDED:
            return fn

        def recorded(*a, **k):
            self.calls.append((name, a, k))
            return fn(*a, **k)
        return recorded

    def replay(self, writer: "GlbWriter", textures: dict):
        """Every recorded call on ``writer``; a texture material named in
        ``textures`` takes that image instead."""
        for name, a, k in self.calls:
            if name == "material_texture" and a[1] in textures:
                a = (textures[a[1]], a[1])
            getattr(writer, name)(*a, **k)


def quads_to_arrays(quads):
    P, N, T, I = [], [], [], []
    for corners, n, q_uv, _ in quads:
        b = len(P)
        P.extend(corners.tolist()); N.extend([list(n)] * 4)
        T.extend(q_uv.tolist() if q_uv is not None else [[0, 0]] * 4)
        I.extend([b, b + 1, b + 2, b, b + 2, b + 3])
    return np.array(P), np.array(N), np.array(T), np.array(I)


# -------------------------------------------------------------------------- tileset

def enu_transform(coords: dict):
    """Column-major 4x4 placing the local metre frame (east, north, up at the
    origin) in ECEF, plus the origin's lon/lat/ellipsoidal height."""
    from export_3dtiles import to_ecef
    ecef, geo = to_ecef(np.zeros((1, 3)), coords)
    lon, lat = np.radians(geo[0, 0]), np.radians(geo[0, 1])
    sl, cl, sp, cp = np.sin(lon), np.cos(lon), np.sin(lat), np.cos(lat)
    east = [-sl, cl, 0.0]
    north = [-sp * cl, -sp * sl, cp]
    up = [cp * cl, cp * sl, sp]
    m = east + [0.0] + north + [0.0] + up + [0.0] + ecef[0].tolist() + [1.0]
    return [float(v) for v in m], geo[0].tolist()
