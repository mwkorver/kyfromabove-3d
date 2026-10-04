#!/usr/bin/env python3
"""The local viewer server: this project's buildings on a globe, on localhost.

    GET  /tiles?target=KEY[,KEY]|all=1[&mesh=1]  one or more buildings' 3D Tiles (tools/tiles_viewer.html)
    GET  /compare              two globes, one camera: the meshes against Overture's boxes
                               (tools/compare_viewer.html; left=, right=, heights=, *_product=)
    GET  /viewer?target=KEY    a building's coloured point cloud (tools/pointcloud_viewer.html)
    GET  /files/<path>         any file under outputs/ (tilesets, glb, KTX2, PNG, the published viewer)
    GET  /building_at?lon&lat  the building under a point
    GET  /frames?gers=         the aerial frames behind a building, with its footprint on each
    GET  /crop?gers=&frame=    one of those crops, re-cut from the frame on S3 when it is gone
    GET  /terrain?x&y&level    a Cesium heightmap tile of the LiDAR ground (scripts/ground_dem.py)
    POST /snapshot?name=       save a view of the map (a JPEG data URL) under outputs/snapshots/

The landmark buildings (PNC Tower, 400 West Market, the Humana Building) are
listed in ``configs/landmarks.yaml``.

Usage::

    python scripts/intake_server.py               # http://localhost:8765
    python scripts/intake_server.py --port 8770
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from html import escape as html_escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import yaml

ROOT = Path(__file__).resolve().parent.parent
LANDMARKS_FILE = ROOT / "configs" / "landmarks.yaml"


# --- landmarks -----------------------------------------------------------------

#: Tilesets that are not one landmark building: name -> path under the repo.
PSEUDO_TILESETS = {"downtown": "outputs/downtown/tileset.json",
                   "downtown_complete": "outputs/downtown_complete/tileset.json",
                   "downtown_complete_tint": "outputs/downtown_complete_tint/tileset.json"}

#: the three products (docs/PRODUCTS.md): what each pane can show
PRODUCTS = {
    "1": ("outputs/overture_boxes/downtown_lidar/tileset.json",
          "Product 1: Overture footprints at their LiDAR heights (measured)"),
    "1o": ("outputs/overture_boxes/downtown/tileset.json",
           "Overture buildings at Overture's own heights (for comparison)"),
    "2": ("outputs/downtown/tileset.json",
          "Product 2: LiDAR geometry, colour only where a frame saw it (measured)"),
    "3": ("outputs/downtown_complete/tileset.json",
          "Product 3: product 2 with unseen texels invented from each building's own frames"),
    "3t": ("outputs/downtown_complete_tint/tileset.json",
           "Product 3, invented texels tinted magenta"),
}


def products() -> Dict[str, Dict[str, Optional[str]]]:
    """Each product's tileset URL (cache-busting mtime; None when not built) and title."""
    out = {}
    for k, (rel, title) in PRODUCTS.items():
        f = ROOT / rel
        out[k] = {"url": f"/files/{rel}?v={int(f.stat().st_mtime)}" if f.exists() else None, "title": title}
    return out
GROUND_DEM = ROOT / "outputs" / "downtown" / "ground_dem.npz"
_ground_dem = None


def ground_dem():
    """The LiDAR ground DEM the viewer's terrain comes from, loaded once
    (``scripts/ground_dem.py`` writes it); None when it has not been built."""
    global _ground_dem
    if _ground_dem is None and GROUND_DEM.exists():
        sys.path.insert(0, str(ROOT / "scripts"))
        from ground_dem import GroundDEM
        _ground_dem = GroundDEM(GROUND_DEM)
    return _ground_dem


def pseudo_tileset_url(name: str) -> Optional[str]:
    """/files/ URL (with a cache-busting mtime) for a pseudo tileset, or None."""
    rel = PSEUDO_TILESETS.get(name)
    if not rel or not (ROOT / rel).exists():
        return None
    return f"/files/{rel}?v={int((ROOT / rel).stat().st_mtime)}"


def load_landmarks(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    blob = yaml.safe_load((path or LANDMARKS_FILE).read_text(encoding="utf-8")) or {}
    return list(blob.get("landmarks", []))


def landmark_by_key(key: str, targets: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """A landmark by its GERS id -- the id everything in the project is
    keyed on -- or by its short key."""
    for t in targets or load_landmarks():
        if key in (t["gers"], t["key"]):
            return t
    raise KeyError(key)


FOOTPRINT_CSV = ROOT / "data" / "derived" / "ky_facade_tiled_wkt.csv"
_FOOTPRINTS: Optional[List[Dict[str, Any]]] = None


def footprints() -> List[Dict[str, Any]]:
    """The large-building footprints (WGS84 WKT, estimated height), loaded once."""
    global _FOOTPRINTS
    if _FOOTPRINTS is None:
        import csv
        from shapely.wkt import loads as wkt_loads
        rows: List[Dict[str, Any]] = []
        if FOOTPRINT_CSV.exists():
            csv.field_size_limit(1 << 30)
            with open(FOOTPRINT_CSV, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    try:
                        g = wkt_loads(r["wkt"])
                        rows.append({"id": r["id"], "geom": g, "lat": g.centroid.y, "lon": g.centroid.x,
                                     "height_m": float(r.get("est_height_m") or r.get("height") or 0.0)})
                    except Exception:                       # noqa: BLE001
                        continue
        _FOOTPRINTS = rows
    return _FOOTPRINTS


# ------------------------------------------------- the obliques behind a building
#
# Every building directory keeps crop_manifest.json -- per aerial frame,
# the window that was cut from it and the crop's camera -- long after the
# crops themselves are deleted. That is enough to re-cut a crop from the
# S3 frame on demand, and to draw the footprint on it.

DOWNTOWN_DIR = Path("outputs/downtown")
FRAME_URL_PATTERN = "https://kyfromabove.s3.amazonaws.com/imagery/obliques/Phase3/{frame_id}.tif"
_BUILDINGS: Optional[List[Dict[str, Any]]] = None
_FRAME_URLS: Optional[Dict[str, str]] = None
_FRAME_RECORDS: Dict[str, Dict[str, Any]] = {}
_READERS: Dict[str, Any] = {}
_CUT_LOCKS: Dict[str, threading.Lock] = {}
_CUT_LOCKS_GUARD = threading.Lock()


def building_index() -> List[Dict[str, Any]]:
    """Every building with a directory: the downtown run's plus the
    landmarks, each with its footprint polygon (WGS84) for point lookups."""
    global _BUILDINGS
    if _BUILDINGS is None:
        from shapely.wkt import loads as wkt_loads
        rows: List[Dict[str, Any]] = []
        for fp in sorted((ROOT / DOWNTOWN_DIR / "buildings").glob("*/footprint.json")):
            try:
                e = json.loads(fp.read_text(encoding="utf-8"))
                rows.append({"gers": e["gers"], "name": e.get("name") or e["gers"], "dir": fp.parent,
                             "geom": wkt_loads(e["wkt"])})
            except Exception:                               # noqa: BLE001
                continue
        for t in load_landmarks():
            if any(r["gers"] == t.get("gers") for r in rows):
                continue
            g = landmark_geometry(t)
            if g is not None and (ROOT / t["building_dir"]).exists():
                rows.append({"gers": t["gers"], "name": t["name"], "dir": ROOT / t["building_dir"], "geom": g["geom"]})
        _BUILDINGS = rows
    return _BUILDINGS


def building_dir_of(gers: str) -> Optional[Dict[str, Any]]:
    for r in building_index():
        if r["gers"] == gers:
            return r
    return None


def building_at(lon: float, lat: float, tolerance_deg: float = 6e-6) -> Optional[Dict[str, Any]]:
    """The building whose footprint holds the point -- or is nearest within
    about half a metre: a click on a wall lands exactly on the footprint's
    edge, which ``contains`` refuses by a rounding hair."""
    from shapely.geometry import Point
    pt = Point(lon, lat)
    best, best_d = None, tolerance_deg
    for r in building_index():
        if r["geom"].contains(pt):
            return r
        d = r["geom"].distance(pt)
        if d < best_d:
            best, best_d = r, d
    return best


def frame_url(frame_id: str) -> str:
    """The frame's COG on S3: from the run's cached index, else the pattern."""
    global _FRAME_URLS
    if _FRAME_URLS is None:
        _FRAME_URLS = {}
        gp = ROOT / DOWNTOWN_DIR / "frames_2024s1.gpkg"
        if gp.exists():
            try:
                import geopandas as gpd
                df = gpd.read_file(gp)
                _FRAME_URLS = dict(zip(df["ID"].astype(str), df["S3URL"].astype(str)))
            except Exception:                               # noqa: BLE001
                _FRAME_URLS = {}
    return _FRAME_URLS.get(frame_id) or FRAME_URL_PATTERN.format(frame_id=frame_id)


def frame_records(gers: str) -> Optional[Dict[str, Any]]:
    """The building's frames: for each crop, where the camera looks and where
    the footprint lands on the crop (base ring at the ground, roof ring at the
    building's height), so a viewer can pick the one facing the same way and
    draw the outline."""
    if gers in _FRAME_RECORDS:
        return _FRAME_RECORDS[gers]
    b = building_dir_of(gers)
    if b is None or not (b["dir"] / "crop_manifest.json").exists():
        return None
    import numpy as np
    from kyfromabove_3d.cameras import VirtualObliqueCamera
    from kyfromabove_3d.footprints import footprint_wgs84_to_local

    d = b["dir"]
    manifest = json.loads((d / "crop_manifest.json").read_text(encoding="utf-8"))
    coords = json.loads((d / "coordinates.json").read_text(encoding="utf-8"))
    ox, oy, oz = coords["origin_source_xyz"]
    height_m = None
    wkt = None
    fp = d / "footprint.json"
    if fp.exists():
        e = json.loads(fp.read_text(encoding="utf-8"))
        height_m, wkt = e.get("height_m"), e.get("wkt")
    lv_p = d / "levels.json"                     # the LiDAR's height, when the colourise measured it
    lvs = json.loads(lv_p.read_text(encoding="utf-8")) if lv_p.exists() else None
    if lvs:
        height_m = max(float(lv["z_hi"]) for lv in lvs) - min(float(lv["z_lo"]) for lv in lvs)
    if wkt is None:
        wkt = b["geom"].wkt
    poly = footprint_wgs84_to_local(wkt, ox, oy)
    ring = np.array(poly.exterior.coords)
    if height_m is None:
        # a landmark: the ground and the top from its wall planes
        wp = d / "wall_planes.json"
        if wp.exists():
            planes = json.loads(wp.read_text(encoding="utf-8")).get("planes", [])
            height_m = max((p["z_hi"] for p in planes), default=30.0)
        else:
            height_m = 30.0
    ground_z = 0.0
    wp = d / "wall_planes.json"
    if not fp.exists() and wp.exists():
        planes = json.loads(wp.read_text(encoding="utf-8")).get("planes", [])
        ground_z = min((p["z_lo"] for p in planes), default=0.0)
    base = np.column_stack([ring, np.full(len(ring), ground_z)])
    roof = np.column_stack([ring, np.full(len(ring), ground_z + float(height_m))])
    frames = []
    for i, m in enumerate(manifest):
        cam = VirtualObliqueCamera(frame_id=m["frame_id"], camera_name="crop",
                                   camera_center_local_m=np.array(m["camera_center_local_m"], dtype=float),
                                   R_cw=np.array(m["R_cw"], dtype=float),
                                   width_px=int(m["width_px"]), height_px=int(m["height_px"]),
                                   fx_px=float(m["fx_px"]), fy_px=float(m["fy_px"]),
                                   cx_px=float(m["cx_px"]), cy_px=float(m["cy_px"]))
        look = np.array(m["R_cw"], dtype=float)[2]
        az = float(np.degrees(np.arctan2(look[0], look[1])) % 360.0)
        el = float(np.degrees(np.arcsin(-look[2])))
        uv_b, _, _ = cam.project(base)
        uv_r, _, _ = cam.project(roof)
        head = m["frame_id"].rsplit("/", 1)[-1].split("_")[0]
        img = d / "images" / f"frame_{i + 1:03d}.png"
        frames.append({"index": i, "frame_id": m["frame_id"], "head": head, "azimuth_deg": round(az, 1),
                       "elevation_deg": round(el, 1), "nadir": bool(abs(look[2]) > 0.9), "look_enu": look.tolist(),
                       "width_px": int(m["width_px"]), "height_px": int(m["height_px"]),
                       "base_px": np.round(uv_b, 1).tolist(), "roof_px": np.round(uv_r, 1).tolist(),
                       "on_disk": img.exists()})
    ring_lonlat = [list(c) for c in b["geom"].exterior.coords]
    rec = {"gers": gers, "name": b["name"], "height_m": float(height_m), "frames": frames,
           "ring_lonlat": ring_lonlat, "dir": str(d.relative_to(ROOT)) if str(d).startswith(str(ROOT)) else str(d)}
    st = d / "3dtiles_mesh" / "tileset.json"
    if st.exists():
        ex = json.loads(st.read_text(encoding="utf-8")).get("extras", {})
        rec.update({
            "path": ex.get("path"),
            "archetype": ex.get("archetype"),
            "faces": ex.get("faces"),
            "triangles": ex.get("triangles"),
        })
    _FRAME_RECORDS[gers] = rec
    return rec


def cut_crop(gers: str, index: int, max_px: int = 1600) -> Optional[Path]:
    """The crop as a JPEG no wider than ``max_px``: the PNG on disk when it
    is still there, else re-cut from the frame on S3 and cached beside the
    building. One lock per crop, so two clicks do not both cut it."""
    from PIL import Image

    b = building_dir_of(gers)
    if b is None:
        return None
    d = b["dir"]
    manifest = json.loads((d / "crop_manifest.json").read_text(encoding="utf-8"))
    if not 0 <= index < len(manifest):
        return None
    cache = d / "crops_cache" / f"{index:03d}_{max_px}.jpg"
    if cache.exists():
        return cache
    key = f"{gers}/{index}/{max_px}"
    with _CUT_LOCKS_GUARD:
        lock = _CUT_LOCKS.setdefault(key, threading.Lock())
    with lock:
        if cache.exists():
            return cache
        m = manifest[index]
        png = d / "images" / f"frame_{index + 1:03d}.png"
        if png.exists():
            img = Image.open(png).convert("RGB")
        else:
            from kyfromabove_3d.imagery import FrameImageReader
            reader = _READERS.get(m["frame_id"])
            if reader is None:
                reader = _READERS[m["frame_id"]] = FrameImageReader(frame_url(m["frame_id"]), m["frame_id"])
            win = reader.read_window(int(m["x0_px"]), int(m["y0_px"]), int(m["width_px"]), int(m["height_px"]))
            img = Image.fromarray(win.rgb)
        if max(img.size) > max_px:
            s = max_px / max(img.size)
            img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp.jpg")
        img.save(tmp, "JPEG", quality=86)
        os.replace(tmp, cache)
    return cache


def landmark_geometry(t: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The landmark's footprint and height, from its footprint row."""
    for f in footprints():
        if f["id"] == t["gers"]:
            return f
    return None


# --- the human manifest --------------------------------------------------------


# --- registration jobs -----------------------------------------------------------


# --- HTTP ------------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "kyfromabove-3d-viewers/0.2"

    def _send(self, code: int, body: Any) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, ctype: str) -> None:
        self._send_bytes(path.read_bytes(), ctype)

    def _cors(self) -> None:
        # any localhost page
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Meta")

    def do_OPTIONS(self) -> None:                           # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:    # quieter than the default
        sys.stderr.write("  %s %s\n" % (self.command, self.path))

    def do_GET(self) -> None:                               # noqa: N802
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/viewer":
                t = landmark_by_key(q["target"])
                self._send_bytes(viewer_html(t).encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/tiles":
                # One target, several (comma-separated), or all=1 for every
                # building that has a tileset.
                ts = load_landmarks()
                keys = [k.strip() for k in q.get("target", "").split(",") if k.strip()]
                pseudo = [k for k in keys if k in PSEUDO_TILESETS]
                if q.get("all") == "1":
                    chosen = ts
                else:
                    chosen = [landmark_by_key(k, ts) for k in keys if k not in PSEUDO_TILESETS]
                # mesh=1 shows the textured surfaces (scripts/build_mesh.py)
                # instead of the point tiles.
                sub = "3dtiles_mesh" if q.get("mesh") == "1" else "3dtiles"
                ready = [t for t in chosen if (ROOT / t["building_dir"] / sub / "tileset.json").exists()]
                purls = [u for u in (pseudo_tileset_url(k) for k in pseudo) if u]
                if not ready and not purls:
                    self._send(404, {"error": f"no 3D Tiles yet: run scripts/{'build_mesh' if sub.endswith('mesh') else 'export_3dtiles'}.py --building <dir>"})
                else:
                    tpl = ROOT / "tools" / "tiles_viewer.html"
                    # the tileset's mtime as a query parameter, which Cesium
                    # passes on to every tile it fetches, so a rebuilt
                    # building is not served from the browser's cache
                    urls = purls + [f"/files/{t['building_dir']}/{sub}/tileset.json?v="
                                    f"{int((ROOT / t['building_dir'] / sub / 'tileset.json').stat().st_mtime)}" for t in ready]
                    name = (ready[0]["name"] if len(ready) == 1 and not purls
                            else ", ".join(pseudo) if purls and not ready else f"{len(urls)} tilesets")
                    html = (tpl.read_text(encoding="utf-8").replace("__NAME__", name)
                            .replace("__TERRAIN__", "true" if ground_dem() is not None else "false")
                            .replace("__TILESETS__", json.dumps(urls)))
                    self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/compare":
                # Two globes side by side, one camera: this project's meshes on
                # the left, Overture's buildings as boxes on the right (or any
                # two tileset lists: left=, right= as comma-separated /files/
                # tileset paths).
                ts = load_landmarks()
                ready = [t for t in ts if (ROOT / t["building_dir"] / "3dtiles_mesh" / "tileset.json").exists()]
                landmarks = [f"/files/{t['building_dir']}/3dtiles_mesh/tileset.json?v={int((ROOT / t['building_dir'] / '3dtiles_mesh' / 'tileset.json').stat().st_mtime)}"
                             for t in ready]
                # left=downtown (the merged tileset), left=targets (the
                # landmarks alone), or a list of /files/ paths; the default
                # is the downtown when it has been built.
                left_q = q.get("left", "")
                if left_q in PSEUDO_TILESETS:
                    left = [u for u in [pseudo_tileset_url(left_q)] if u]
                elif left_q == "targets" or (not left_q and not pseudo_tileset_url("downtown")):
                    left = landmarks
                elif not left_q:
                    left = [pseudo_tileset_url("downtown")]
                else:
                    left = left_q.split(",")
                left_name = ("This project: downtown from LiDAR + aerial frames" if left and "downtown" in left[0]
                             else "This project: LiDAR + aerial frames")
                # Overture's footprints at the LiDAR's heights when measured
                # (heights=lidar, the default), or at Overture's own (heights=overture).
                sub_b = "downtown_lidar" if q.get("heights", "lidar") == "lidar" else "downtown"
                boxes = ROOT / "outputs" / "overture_boxes" / sub_b / "tileset.json"
                right = q.get("right", "").split(",") if q.get("right") else (
                    [f"/files/outputs/overture_boxes/{sub_b}/tileset.json?v={int(boxes.stat().st_mtime)}"] if boxes.exists() else [])
                right_name = ("Overture footprints at the LiDAR's heights" if sub_b == "downtown_lidar"
                              else "Overture buildings, extruded to their height")
                if not left or not right:
                    self._send(404, {"error": "need meshes (scripts/build_mesh.py) and boxes (scripts/overture_boxes.py) to compare"})
                else:
                    tpl = ROOT / "tools" / "compare_viewer.html"
                    html = (tpl.read_text(encoding="utf-8").replace("__NAME__", "LiDAR mesh vs Overture")
                            .replace("__LEFT_NAME__", html_escape(q.get("left_name", left_name)))
                            .replace("__RIGHT_NAME__", html_escape(q.get("right_name", right_name)))
                            .replace("__LEFT__", json.dumps(left)).replace("__RIGHT__", json.dumps(right))
                            .replace("__PRODUCTS__", json.dumps(products()))
                            .replace("__LEFT_PRODUCT__", json.dumps(q.get("left_product") or ("2" if not q.get("left") else None)))
                            .replace("__RIGHT_PRODUCT__", json.dumps(q.get("right_product") or ("1" if not q.get("right") else None)))
                            .replace("__TERRAIN__", "true" if ground_dem() is not None else "false"))
                    self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path.startswith("/files/"):
                rel = Path(u.path[len("/files/"):])
                full = (ROOT / rel).resolve()
                if ".." in rel.parts or not str(full).startswith(str((ROOT / "outputs").resolve())) or not full.is_file():
                    self._send(404, {"error": "only files under outputs/ are served"})
                else:
                    ctype = {"ply": "application/octet-stream", "png": "image/png", "json": "application/json",
                             "glb": "model/gltf-binary", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                             "pnts": "application/octet-stream", "html": "text/html; charset=utf-8",
                             "ktx2": "image/ktx2"}.get(full.suffix[1:], "application/octet-stream")
                    self._send_file(full, ctype)
            elif u.path == "/building_at":
                r = building_at(float(q["lon"]), float(q["lat"]))
                self._send(200, {"gers": r["gers"], "name": r["name"]} if r else {"gers": None})
            elif u.path == "/frames":
                rec = frame_records(q.get("gers", ""))
                if rec is None:
                    self._send(404, {"error": "no crop manifest for that building"})
                else:
                    self._send(200, rec)
            elif u.path == "/crop":
                path = cut_crop(q.get("gers", ""), int(q.get("frame", "0")), int(q.get("max", "1600")))
                if path is None:
                    self._send(404, {"error": "no such crop"})
                else:
                    data = path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "max-age=86400")
                    self._cors()
                    self.end_headers()
                    self.wfile.write(data)
            elif u.path == "/terrain":
                # a Cesium heightmap tile of the LiDAR ground (scripts/ground_dem.py)
                dem = ground_dem()
                if dem is None:
                    self._send(404, {"error": "no ground DEM: python scripts/ground_dem.py"})
                else:
                    data = dem.tile(int(q.get("x", "0")), int(q.get("y", "0")), int(q.get("level", "0")),
                                    int(q.get("w", "32")), int(q.get("h", "32")))
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "max-age=3600")
                    self._cors()
                    self.end_headers()
                    self.wfile.write(data)
            elif u.path == "/health":
                self._send(200, {"ok": True, "root": str(ROOT)})
            else:
                self._send(404, {"error": f"no route {u.path}"})
        except KeyError as exc:
            self._send(404, {"error": f"unknown building {exc}"})
        except Exception as exc:                            # noqa: BLE001
            self._send(500, {"error": str(exc)[:300]})

    def do_POST(self) -> None:                              # noqa: N802
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        try:
            if u.path == "/snapshot":
                # A view of the map, saved from the page: the body is a JPEG
                # data URL from the Cesium canvas.
                import base64
                import re as _re
                name = _re.sub(r"[^A-Za-z0-9_.-]", "_", q.get("name", "view"))[:80] or "view"
                if not body.startswith(b"data:image/jpeg;base64,"):
                    self._send(400, {"error": "expected a JPEG data URL"})
                    return
                out = ROOT / "outputs" / "snapshots"
                out.mkdir(parents=True, exist_ok=True)
                (out / f"{name}.jpg").write_bytes(base64.b64decode(body.split(b",", 1)[1]))
                self._send(200, {"saved": f"outputs/snapshots/{name}.jpg"})
            else:
                self._send(404, {"error": f"no route {u.path}"})
        except KeyError as exc:
            self._send(404, {"error": f"unknown building {exc}"})
        except Exception as exc:                            # noqa: BLE001
            self._send(500, {"error": str(exc)[:300]})


def viewer_html(t: Dict[str, Any]) -> str:
    """The project's three.js point-cloud viewer (tools/pointcloud_viewer.html),
    pointed at this landmark's colorized.ply through /files/."""
    src = ROOT / "tools" / "pointcloud_viewer.html"
    if not src.exists():
        return "<p>tools/pointcloud_viewer.html is missing.</p>"
    html = src.read_text(encoding="utf-8")
    ply = f"/files/{t['building_dir']}/colorized.ply"
    html = re.sub(r'const plyPath = "[^"]*";', f'const plyPath = "{ply}";', html, count=1)
    html = html.replace("PNC Tower", t["name"])
    return html


def serve(port: int) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"viewer server on http://127.0.0.1:{port}  (landmarks: {LANDMARKS_FILE.relative_to(ROOT)})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    serve(args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
