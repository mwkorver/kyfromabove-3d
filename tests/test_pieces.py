"""Stage 1's bookkeeping, with no network: which tile owns a footprint, which
tiles see it as halo, and publishing a piece -- its marker last, an
unchanged piece not rewritten (so no event), a changed one rewritten."""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import box

from kyfromabove_3d import pieces as P
from kyfromabove_3d.store import LocalStore

to_wgs = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
X0, Y0 = -9546000.0, 4612000.0                       # downtown Louisville, EPSG:3857


def _grid():
    """Two 1.5 km tiles side by side, W and E."""
    return gpd.GeoDataFrame({"tilename": ["W", "E"], "key": ["w.copc.laz", "e.copc.laz"], "aws_url": ["", ""]},
                            geometry=[box(X0, Y0, X0 + 1500, Y0 + 1500), box(X0 + 1500, Y0, X0 + 3000, Y0 + 1500)],
                            crs="EPSG:3857")


def _rec(gid, x0, y0, x1, y1):
    xs, ys = to_wgs([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])
    return {"id": gid, "wkt": "POLYGON ((" + ", ".join(f"{x} {y}" for x, y in zip(xs, ys)) + "))"}


def test_a_footprint_belongs_to_the_tile_holding_its_centroid_and_is_halo_to_its_neighbour():
    g = _grid()
    recs = [_rec("inside_w", X0 + 100, Y0 + 100, X0 + 140, Y0 + 140),
            _rec("straddles_mostly_e", X0 + 1490, Y0 + 500, X0 + 1560, Y0 + 540),      # centroid in E
            _rec("near_edge_w", X0 + 1450, Y0 + 900, X0 + 1480, Y0 + 930)]             # 20 m from E
    assert P.owner_tiles(recs, g) == ["W", "E", "W"]
    by = P.records_by_tile(recs, g)
    assert [r["id"] for r in by["W"]["owned"]] == ["inside_w", "near_edge_w"]
    assert [r["id"] for r in by["E"]["owned"]] == ["straddles_mostly_e"]
    assert sorted(r["id"] for r in by["E"]["halo"]) == ["near_edge_w"]
    assert [r["id"] for r in by["W"]["halo"]] == ["straddles_mostly_e"]


def test_a_footprint_outside_the_grid_has_no_tile():
    assert P.owner_tiles([_rec("far", X0 - 9000, Y0, X0 - 8960, Y0 + 40)], _grid()) == [None]


class _Recording(LocalStore):
    def __init__(self, root):
        super().__init__(root)
        self.keys = []

    def put_bytes(self, key, data, content_type=None, content_encoding=None):
        self.keys.append(key); super().put_bytes(key, data, content_type, content_encoding)

    def put_file(self, key, path):
        self.keys.append(key); super().put_file(key, path)


def _piece(folder, gers="g1", height=12.0):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "footprint.json").write_text(json.dumps({"gers": gers, "height_m": height}))
    (folder / "clip.json").write_text(json.dumps({"aoi_ft": [0, 1, 0, 1, 0, 1]}))
    (folder / "coordinates.json").write_text("{}")
    (folder / "piece.json").write_text(json.dumps({"gers": gers, "inputs_hash": P.inputs_hash(folder)}))
    return folder


def test_a_piece_is_published_marker_last_and_an_unchanged_one_is_not_rewritten(tmp_path):
    store = _Recording(tmp_path / "store")
    f = _piece(tmp_path / "work" / "g1")
    assert P.publish_piece(f, store) is True
    assert store.keys[-1] == "pieces/g1/piece.json"                       # the event fires on the whole piece
    assert "pieces/g1/footprint.json" in store.keys and len(store.keys) == 4
    store.keys.clear()
    assert P.publish_piece(f, store) is False and store.keys == []         # nothing written: no event
    _piece(f, height=14.0)                                                 # the LiDAR height changed
    assert P.publish_piece(f, store) is True and store.keys[-1] == "pieces/g1/piece.json"


def test_the_inputs_hash_follows_the_footprint_and_the_code_version(tmp_path, monkeypatch):
    f = _piece(tmp_path / "g1")
    h = P.inputs_hash(f)
    assert P.inputs_hash(f) == h
    monkeypatch.setattr(P, "CODE_VERSION", P.CODE_VERSION + "-next")
    assert P.inputs_hash(f) != h


def test_tile_origin_and_bounds_are_in_state_plane_feet():
    g = _grid()
    ox, oy = P.tile_origin_ft(g, "W")
    minx, miny, maxx, maxy = P.tile_bounds_ft(g, "W")
    assert minx < ox < maxx and miny < oy < maxy
    assert 3500 < maxx - minx < 5500                                       # 1.5 km of web-mercator at 38 N, in feet
    with pytest.raises(KeyError):
        P.tile_origin_ft(g, "nowhere")


def test_the_tile_step_measures_its_own_buildings_and_keeps_the_halo_apart(monkeypatch):
    """Two outlines owned by tile W and one halo neighbour, over a made-up
    LiDAR: ground at 0 m, roofs at 10 m and 25 m. Every owned building comes
    back once, measured; the halo is measured too but returned separately."""
    from kyfromabove_3d import product1

    g = _grid()
    recs = [_rec("low", X0 + 100, Y0 + 100, X0 + 140, Y0 + 140),
            _rec("tall", X0 + 300, Y0 + 300, X0 + 330, Y0 + 330)]
    halo = [_rec("neighbour", X0 + 1510, Y0 + 900, X0 + 1540, Y0 + 930)]

    def fake_cloud(polys, ox, oy, resolution_m=None, index=None, log=print):
        rng = np.random.default_rng(0)
        pts, cls = [], []
        for p, h in zip(polys, (10.0, 25.0, 8.0)):
            minx, miny, maxx, maxy = p.bounds
            n = 3000
            xy = np.column_stack([rng.uniform(minx - 15, maxx + 15, n), rng.uniform(miny - 15, maxy + 15, n)])
            from shapely import contains_xy
            inside = contains_xy(p, xy[:, 0], xy[:, 1])
            z = np.where(inside, h, 0.0) + rng.normal(0, 0.05, n)
            pts.append(np.column_stack([xy, z])); cls.append(np.where(inside, 1, 2))
        return np.vstack(pts), np.concatenate(cls)

    monkeypatch.setattr(product1, "area_cloud_local", fake_cloud)
    out = P.tile_step("W", recs, halo, g, index=None, log=lambda *a: None)
    got = {r["id"]: r for r in out["buildings"]}
    assert sorted(got) == ["low", "tall"]                                   # each owned building once
    assert abs(got["low"]["lidar_height_m"] - 10.0) < 0.3 and abs(got["tall"]["lidar_height_m"] - 25.0) < 0.3
    assert [r["id"] for r in out["halo"]] == ["neighbour"] and abs(out["halo"][0]["lidar_height_m"] - 8.0) < 0.3
    assert out["stats"]["buildings"] == 2 and out["stats"]["measured"] == 3 and out["stats"]["split"] == 0


def test_the_tile_steps_results_are_published_summary_last(tmp_path):
    store = _Recording(tmp_path / "store")
    frames = tmp_path / "frames.gpkg"
    frames.write_bytes(b"GPKG")
    res = {"buildings": [{"id": "a", "lidar_height_m": 9.0}], "halo": [], "splits": [],
           "stats": {"tile": "W", "buildings": 1, "frames": 12}}
    P.publish_tile_step("W", res, frames, store)
    assert store.keys == ["tiles/W/buildings.json", "tiles/W/frames.gpkg", "tiles/W/summary.json"]
    assert store.read_json("tiles/W/buildings.json")["buildings"][0]["id"] == "a"
    assert store.read_json("tiles/W/summary.json")["frames"] == 12


def test_the_point_grid_finds_exactly_what_a_full_scan_finds():
    from kyfromabove_3d.product1 import PointGrid
    rng = np.random.default_rng(3)
    xy = np.column_stack([rng.uniform(-800, 800, 200_000), rng.uniform(-600, 700, 200_000)])
    g = PointGrid(xy)
    for box in [(-10, -10, 25, 40), (-900, -900, -790, -500), (100.0, 100.0, 100.0, 100.0), (-2000, -2000, 2000, 2000),
                (790, 690, 900, 900), (3.3, -17.1, 3.4, 200.7)]:
        full = np.flatnonzero((xy[:, 0] > box[0]) & (xy[:, 0] < box[2]) & (xy[:, 1] > box[1]) & (xy[:, 1] < box[3]))
        assert np.array_equal(g.box(*box), full), box                     # same points, same order
    assert len(PointGrid(np.zeros((0, 2))).box(0, 0, 1, 1)) == 0


def test_a_split_parts_outline_is_written_to_seven_decimals():
    from shapely.geometry import Polygon
    from kyfromabove_3d.product1 import rounded_wkt
    # a split part from N076E229, as the projection round trip left it
    part = Polygon([(-85.73762225631944, 38.254229713841774), (-85.7376328, 38.25423090000001),
                    (-85.737613, 38.25434), (-85.7377413, 38.25435440000002), (-85.7377254, 38.254441700000015),
                    (-85.73758713643477, 38.25442622125094)])
    assert part.is_valid
    w = rounded_wkt(part)
    decimals = [len(v.split(".")[1]) for v in w.replace("POLYGON ((", "").replace("))", "").replace(",", " ").split()]
    assert max(decimals) <= 7 and "38.2542297" in w and "38.2544417" in w
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])                       # invalid to begin with
    assert rounded_wkt(bowtie) == bowtie.wkt                                # left as it is


def test_a_split_outline_drops_a_corner_that_lies_on_a_straight_edge():
    """A cut can leave a point 1 mm off an edge on one machine and not on
    another; written out, it must be gone on both."""
    from shapely.geometry import Polygon
    square = Polygon([(0, 0), (10, 0), (10, 5), (10.001, 7.5), (10, 10), (0, 10)])     # (10.001, 7.5): 1 mm off
    simple = square.simplify(0.01, preserve_topology=True)
    assert len(simple.exterior.coords) == 5 and not any(abs(x - 10.001) < 1e-9 for x, _ in simple.exterior.coords)


def test_batches_hold_every_building_once_and_keep_neighbours_together():
    recs = [_rec(f"b{i:02d}", X0 + 30 * (i % 6), Y0 + 300 * (i // 6), X0 + 30 * (i % 6) + 20, Y0 + 300 * (i // 6) + 20)
            for i in range(18)]
    got = P.batches(recs, 5)
    flat = [g for b in got for g in b]
    assert sorted(flat) == sorted(r["id"] for r in recs) and len(flat) == 18
    assert [len(b) for b in got] == [5, 5, 5, 3]
    assert set(got[0]) <= {f"b{i:02d}" for i in range(6)}                   # the first row's buildings together


def test_the_frames_a_building_gets_are_self_contained_pointers():
    import pandas as pd
    from types import SimpleNamespace
    fid = "KY_KYAPED_2024_Season1_3IN/Bwd_7019_47217"
    frames = gpd.GeoDataFrame({"ID": [fid, "other"], "S3URL": ["https://kyfromabove.s3.amazonaws.com/x/Bwd_7019_47217.tif", "u"],
                               "CameraID": ["Bwd", "Fwd"], "X": [4914722.7, 1.0], "Omega": [-44.9, 0.0],
                               "CamWidthPx": [np.int64(14144), np.int64(1)], "FlightDate": [pd.Timestamp("2024-03-20"), pd.NaT]},
                              geometry=[box(0, 0, 1, 1), box(0, 0, 1, 1)], crs="EPSG:3089")
    table = pd.DataFrame({"frame_id": [fid], "u_min": [4412.2], "u_max": [4915.4], "v_min": [7025.9], "v_max": [7614.0],
                          "span_u_px": [503.2], "span_v_px": [588.1], "corners_on_sensor": [8], "margin_px": [2945.9],
                          "selected": [True]})
    rec = P.frames_record({"cams": [SimpleNamespace(frame_id=fid)], "table": table}, frames)
    assert len(rec) == 1 and rec[0]["url"].endswith("Bwd_7019_47217.tif") and rec[0]["direction"] == "Bwd"
    assert rec[0]["window_px"]["u_min"] == 4412.2 and rec[0]["corners_on_sensor"] == 8
    assert rec[0]["eo"]["CamWidthPx"] == 14144 and rec[0]["eo"]["FlightDate"].startswith("2024-03-20")
    json.dumps(rec)                                                         # plain JSON all the way down
    assert P.frames_record(None, frames) == []


def test_the_piece_step_writes_each_buildings_piece_with_its_frames(tmp_path, monkeypatch):
    from kyfromabove_3d import prepare
    recs = [_rec("a", X0 + 100, Y0 + 100, X0 + 140, Y0 + 140), _rec("b", X0 + 150, Y0 + 100, X0 + 180, Y0 + 130)]
    for r in recs:
        r.update({"lidar_ground_z": 140.0, "lidar_height_m": 12.0})
    seen = {}

    def fake_prepare_piece(rec, folder, frames_gdf, index, pool, *a, **k):
        folder.mkdir(parents=True, exist_ok=True)
        for n in ("footprint.json", "clip.json", "coordinates.json", "check_features.json"):
            (folder / n).write_text(json.dumps({"gers": rec["id"]}))
        seen[rec["id"]] = sorted(r["id"] for r in pool)
        return {"n_frames": 0, "heads": []}, None

    monkeypatch.setattr(prepare, "prepare_piece", fake_prepare_piece)
    out = P.piece_step("W", ["a", "b", "zzz"], recs, [], frames_gdf=None, index=None, work=tmp_path)
    assert sorted(f.name for f in out["pieces"]) == ["a", "b"] and list(out["failed"]) == ["zzz"]
    for f in out["pieces"]:
        assert json.loads((f / "frames.json").read_text()) == []
        m = json.loads((f / "piece.json").read_text())
        assert m["tile"] == "W" and m["inputs_hash"] == P.inputs_hash(f)
    assert seen["a"] == ["a", "b"]                                          # a neighbour within 50 m is in the pool


def test_the_hash_ignores_last_bit_differences_between_machines(tmp_path):
    a, b = _piece(tmp_path / "a"), _piece(tmp_path / "b")
    (a / "footprint.json").write_text(json.dumps({"gers": "g1", "origin_source_xyz": [4920000.5, 3980683.110304273, 461.9]}))
    (b / "footprint.json").write_text(json.dumps({"gers": "g1", "origin_source_xyz": [4920000.5, 3980683.1103042746, 461.9]}))
    (a / "frame_coverage.csv").write_text("frame_id,u_min\nF1,381.2640773860139\n")
    (b / "frame_coverage.csv").write_text("frame_id,u_min\nF1,381.26407738601395\n")
    assert P.inputs_hash(a) == P.inputs_hash(b)
    (b / "footprint.json").write_text(json.dumps({"gers": "g1", "origin_source_xyz": [4920000.5, 3980683.2, 461.9]}))
    assert P.inputs_hash(a) != P.inputs_hash(b)                             # a real move still counts


def test_slow_down_is_retried_and_a_missing_file_is_not(monkeypatch):
    import time as _time
    from kyfromabove_3d import lidar
    monkeypatch.setattr(_time, "sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("503 Server Error: Service Unavailable for url: https://kyfromabove.s3...")
        return "points"

    assert lidar.with_retries(flaky) == "points" and len(calls) == 3
    calls.clear()

    def missing():
        calls.append(1)
        raise RuntimeError("404 Client Error: Not Found")

    with pytest.raises(RuntimeError):
        lidar.with_retries(missing)
    assert len(calls) == 1


def test_the_tile_step_reads_disjoint_boxes_covering_every_footprint_and_its_margin():
    from kyfromabove_3d.product1 import read_boxes
    rng = np.random.default_rng(1)
    # a rural tile: a few houses far apart, and a cluster
    xy = np.vstack([rng.uniform(-700, 700, (12, 2)), rng.normal([300, -200], 30, (20, 2))])
    allb = np.column_stack([xy - 8, xy + 8])
    pad = 20.0
    lo, hi = allb[:, :2].min(axis=0) - pad, allb[:, 2:].max(axis=0) + pad
    boxes = read_boxes(allb, pad, lo, hi)
    # disjoint (half-open) and covering each footprint's padded bounds: sample points in them
    pts = np.vstack([rng.uniform(b[:2] - pad, b[2:] + pad, (200, 2)) for b in allb])
    inside = np.array([[(bx0 <= x < bx1) and (by0 <= y < by1) for bx0, by0, bx1, by1 in boxes] for x, y in pts])
    assert (inside.sum(axis=1) == 1).all()
    area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
    assert area < 0.5 * (hi[0] - lo[0]) * (hi[1] - lo[1])             # well under one box round them all
