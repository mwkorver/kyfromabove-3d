"""The local viewer server: its routes, the buildings under a point, the
frames and crops behind a building, the terrain."""

from __future__ import annotations

import io
import json
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import intake_server as srv  # noqa: E402


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A throwaway project: one landmark."""
    root = tmp_path
    bdir = root / "outputs" / "t" / "t"
    bdir.mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "configs" / "landmarks.yaml").write_text(
        "landmarks:\n  - key: t\n    name: Test Tower\n    gers: g-1\n    building_dir: outputs/t/t\n")
    monkeypatch.setattr(srv, "ROOT", root)
    monkeypatch.setattr(srv, "LANDMARKS_FILE", root / "configs" / "landmarks.yaml")
    return root


@pytest.fixture
def server(project):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, json.loads(r.read())


def test_an_unknown_building_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(server + "/viewer?target=nope")
    assert exc.value.code == 404


def test_the_photo_routes_are_gone(server):
    for path in ("/targets", "/queries?target=t", "/status?target=t", "/register?target=t"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(server + path)
        assert exc.value.code == 404


def test_a_landmark_is_found_by_key_or_gers(project):
    assert srv.landmark_by_key("t")["gers"] == "g-1" == srv.landmark_by_key("g-1")["gers"]


@pytest.fixture
def downtown(project, monkeypatch):
    """One downtown building with a crop manifest but no crops: a 10 m
    square at the origin, 20 m tall, seen by a camera 500 m straight above
    (so the footprint projects centred) and by an oblique looking north."""
    root = project
    gers = "aaaa1111-0000-0000-0000-000000000001"
    bdir = root / "outputs" / "downtown" / "buildings" / gers
    bdir.mkdir(parents=True)
    ox, oy = 4919000.0, 3980000.0
    ft = 0.3048006096012192
    from pyproj import Transformer
    to_ll = Transformer.from_crs("EPSG:3089", "EPSG:4326", always_xy=True).transform
    ring = [(-5, -5), (5, -5), (5, 5), (-5, 5), (-5, -5)]
    lonlat = [to_ll(ox + x / ft, oy + y / ft) for x, y in ring]
    wkt = "POLYGON ((" + ", ".join(f"{a} {b}" for a, b in lonlat) + "))"
    (bdir / "footprint.json").write_text(json.dumps({"gers": gers, "name": "Test Shop", "wkt": wkt, "height_m": 20.0}))
    (bdir / "coordinates.json").write_text(json.dumps({"origin_source_xyz": [ox, oy, 450.0]}))
    down = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]                       # camera z along -up: looking straight down
    north = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]                        # camera z along +north
    (bdir / "crop_manifest.json").write_text(json.dumps([
        {"frame_id": "C/Color_1_1", "x0_px": 100, "y0_px": 200, "width_px": 400, "height_px": 400,
         "fx_px": 4000.0, "fy_px": 4000.0, "cx_px": 200.0, "cy_px": 200.0, "R_cw": down,
         "camera_center_local_m": [0.0, 0.0, 500.0]},
        {"frame_id": "C/Fwd_1_2", "x0_px": 0, "y0_px": 0, "width_px": 400, "height_px": 400,
         "fx_px": 4000.0, "fy_px": 4000.0, "cx_px": 200.0, "cy_px": 200.0, "R_cw": north,
         "camera_center_local_m": [0.0, -500.0, 10.0]},
    ]))
    monkeypatch.setattr(srv, "_BUILDINGS", None)
    monkeypatch.setattr(srv, "_FRAME_RECORDS", {})
    monkeypatch.setattr(srv, "_FRAME_URLS", {})
    return gers, bdir, lonlat


def test_building_at_finds_the_footprint_under_a_point(server, downtown):
    gers, _, lonlat = downtown
    lon = sum(p[0] for p in lonlat[:4]) / 4; lat = sum(p[1] for p in lonlat[:4]) / 4
    status, body = get(f"{server}/building_at?lon={lon}&lat={lat}")
    assert status == 200 and body["gers"] == gers and body["name"] == "Test Shop"
    status, body = get(f"{server}/building_at?lon={lon + 0.01}&lat={lat}")
    assert body["gers"] is None


def test_frames_report_each_camera_and_project_the_footprint_onto_the_crop(server, downtown):
    gers, _, _ = downtown
    status, rec = get(f"{server}/frames?gers={gers}")
    assert status == 200 and rec["name"] == "Test Shop" and len(rec["frames"]) == 2
    nadir, fwd = rec["frames"]
    assert nadir["head"] == "Color" and nadir["nadir"] and abs(nadir["elevation_deg"] - 90.0) < 1e-6
    assert fwd["head"] == "Fwd" and not fwd["nadir"] and abs(fwd["azimuth_deg"] - 0.0) < 1e-6
    # straight down from 500 m with f = 4000 px: a 10 m square is 80 px wide, centred on the principal point
    base = np.array(nadir["base_px"]); roof = np.array(nadir["roof_px"])
    assert np.allclose(base[:4].mean(axis=0), [200.0, 200.0], atol=0.5)
    assert abs((base[:, 0].max() - base[:, 0].min()) - 80.0) < 0.5
    assert (roof[:, 0].max() - roof[:, 0].min()) > (base[:, 0].max() - base[:, 0].min())   # the roof is nearer
    assert len(rec["ring_lonlat"]) == 5


def test_crop_recuts_from_the_frame_when_the_png_is_gone(server, downtown, monkeypatch):
    gers, bdir, _ = downtown
    calls = []

    class FakeReader:
        def __init__(self, url, frame_id=None):
            calls.append(url)

        def read_window(self, x0, y0, w, h):
            rgb = np.zeros((h, w, 3), dtype=np.uint8); rgb[..., 0] = 200
            return type("W", (), {"rgb": rgb})()

    import kyfromabove_3d.imagery as im
    monkeypatch.setattr(im, "FrameImageReader", FakeReader)
    with urllib.request.urlopen(f"{server}/crop?gers={gers}&frame=1&max=100", timeout=10) as r:
        assert r.status == 200 and r.headers["Content-Type"] == "image/jpeg"
        img = Image.open(io.BytesIO(r.read()))
    assert img.size == (100, 100) and img.getpixel((50, 50))[0] > 180
    assert calls == ["https://kyfromabove.s3.amazonaws.com/imagery/obliques/Phase3/C/Fwd_1_2.tif"]
    assert (bdir / "crops_cache" / "001_100.jpg").exists()
    # a second request is served from the cache, not re-cut
    with urllib.request.urlopen(f"{server}/crop?gers={gers}&frame=1&max=100", timeout=10) as r:
        assert r.status == 200
    assert len(calls) == 1
    # a PNG still on disk is used as-is
    (bdir / "images").mkdir()
    Image.fromarray(np.full((400, 400, 3), 30, dtype=np.uint8)).save(bdir / "images" / "frame_001.png")
    with urllib.request.urlopen(f"{server}/crop?gers={gers}&frame=0&max=100", timeout=10) as r:
        img = Image.open(io.BytesIO(r.read()))
    assert img.getpixel((50, 50))[0] < 60 and len(calls) == 1


def test_terrain_tiles_come_from_the_lidar_ground_dem(server, tmp_path, monkeypatch):
    """A DEM of a 20 x 20 m plateau at 100 m rising to 110 m eastward; the
    tile at the deepest level over it reads those heights, north-west first,
    and a tile far away reads the clamped edge rather than nothing."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from ground_dem import GroundDEM
    from pyproj import Transformer
    ft = 0.3048006096012192
    lon0, lat0 = -85.7565, 38.2549
    to_ft = Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True).transform
    ox, oy = to_ft(lon0, lat0)
    Z = np.tile(np.linspace(100.0, 110.0, 11, dtype=np.float32), (11, 1))       # rises with x (east)
    path = tmp_path / "ground_dem.npz"
    np.savez_compressed(path, x0=-10.0, y0=-10.0, cell=2.0, Z=Z, origin_lon_lat_h=np.array([lon0, lat0, 105.0]),
                        origin_source_xyz=np.array([ox, oy, 460.0]), shift_m=-34.5, n_ground=1)
    dem = GroundDEM(path)
    z = dem.heights(np.array([lon0, lon0 + 0.0001]), np.array([lat0, lat0]))
    assert 104.0 < z[0] < 106.0 and z[1] > z[0]                                   # centre ~105, east higher
    far = dem.heights(np.array([-80.0]), np.array([40.0]))
    assert 100.0 <= far[0] <= 110.0                                               # clamped to the edge
    monkeypatch.setattr(srv, "_ground_dem", dem)
    monkeypatch.setattr(srv, "GROUND_DEM", path)
    level = 18
    n_x = 2 << level
    x = int((lon0 + 180.0) / (360.0 / n_x)); y = int((90.0 - lat0) / (180.0 / (1 << level)))
    try:
        with urllib.request.urlopen(f"{server}/terrain?x={x}&y={y}&level={level}&w=4&h=4") as r:
            buf = np.frombuffer(r.read(), dtype=np.float32)
    except urllib.error.HTTPError as exc:
        raise AssertionError(exc.read().decode()) from exc
    assert buf.shape == (16,) and 100.0 <= buf.min() and buf.max() <= 110.0
    assert buf[3] >= buf[0]                                                       # the east column is higher
    for tpl in ("compare_viewer.html", "tiles_viewer.html"):
        html = (Path(__file__).resolve().parent.parent / "tools" / tpl).read_text()
        assert "__TERRAIN__" in html and "CustomHeightmapTerrainProvider" in html and "/terrain?" in html
