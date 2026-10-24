"""The downtown batch: preparing a building from its Overture record, the
fallback geometry in the mesh, and gathering buildings into one tileset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from conftest import make_camera                                     # noqa: E402
from kyfromabove_3d.coordinates import METERS_PER_US_SURVEY_FOOT     # noqa: E402
from kyfromabove_3d.footprints import largest_ring                   # noqa: E402
from kyfromabove_3d.prepare import (                                 # noqa: E402
    aoi_bounds_ft, building_frame, building_volume, select_cameras,
)

FT = METERS_PER_US_SURVEY_FOOT


def test_frame_puts_the_origin_at_the_centroid_and_ground_at_the_lidar_ground():
    ring_ft = np.array([[100.0, 200.0], [200.0, 200.0], [200.0, 300.0], [100.0, 300.0], [100.0, 200.0]])
    f = building_frame({"id": "x", "lidar_ground_z": 140.0}, ring_ft)
    assert f.origin_source_xyz[:2] == (140.0, 240.0)
    assert abs(f.origin_source_xyz[2] * FT - 140.0) < 1e-9
    f2 = building_frame({"id": "y", "lidar_ground_z": None}, ring_ft, fallback_ground_m=138.0)
    assert abs(f2.origin_source_xyz[2] * FT - 138.0) < 1e-9
    with pytest.raises(ValueError):
        building_frame({"id": "z"}, ring_ft)


def test_aoi_reaches_thirty_metres_over_the_roof_and_ten_under_ground():
    ring_ft = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    minx, maxx, miny, maxy, zlo, zhi = aoi_bounds_ft(ring_ft, ground_ft=1500.0, height_m=50.0, pad_ft=90.0)
    assert (minx, maxx, miny, maxy) == (-90.0, 190.0, -90.0, 190.0)
    assert abs((zhi - 1500.0) * FT - 80.0) < 1e-6 and abs((1500.0 - zlo) * FT - 10.0) < 1e-6
    # a building with no measured height is treated as 6 m, never as 0
    _, _, _, _, _, zhi2 = aoi_bounds_ft(ring_ft, 1500.0, None)
    assert abs((zhi2 - 1500.0) * FT - 36.0) < 1e-6


def test_volume_is_the_footprint_not_whatever_the_lidar_holds():
    ring_m = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]])
    v = building_volume(ring_m, height_m=30.0, pad_xy_m=3.0, pad_z_m=2.0)
    assert (v.min_x_m, v.max_x_m, v.min_y_m, v.max_y_m) == (-3.0, 23.0, -3.0, 13.0)
    assert (v.min_z_m, v.max_z_m) == (-2.0, 32.0)


def test_largest_ring_takes_the_biggest_polygon_and_drops_holes():
    from shapely.geometry import MultiPolygon

    big = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], holes=[[(4, 4), (6, 4), (6, 6), (4, 6)]])
    small = Polygon([(20, 20), (21, 20), (21, 21), (20, 21)])
    ring = largest_ring(MultiPolygon([big, small]))
    assert abs(ring.area - 100.0) < 1e-9 and len(ring.interiors) == 0


def _frames_gdf(cams, ring_ft):
    """A stand-in for the cached frame index: one row per camera whose
    footprint contains the building, with the EO columns the parser reads."""
    import geopandas as gpd
    from shapely.geometry import box

    rows = []
    for c in cams:
        rows.append({"ID": c.frame_id, "CameraID": c.frame_id.rsplit("/", 1)[-1].split("_")[0],
                     "X": 0.0, "Y": 0.0, "Z": 0.0, "Omega": 0.0, "Phi": 0.0, "Kappa": 0.0,
                     "CamWidthPx": c.width_px, "CamHeighPx": c.height_px, "CamFocalMm": 120.0,
                     "CamCCDResU": 3.76, "CamPpxMm": 0.0, "CamPpyMm": 0.0,
                     "geometry": box(ring_ft[:, 0].min() - 5000, ring_ft[:, 1].min() - 5000,
                                     ring_ft[:, 0].max() + 5000, ring_ft[:, 1].max() + 5000)})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:3089")


def test_select_cameras_caps_per_head_and_relaxes_the_corner_test(monkeypatch):
    """Six Fwd cameras all see the building: only the best four by margin
    are kept. When none holds all eight corners, the six-corner rule lets
    the nearly-full ones through rather than leaving the building blind."""
    import kyfromabove_3d.prepare as prep

    ring_m = np.array([[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]])
    volume = building_volume(ring_m, 20.0)
    frame = building_frame({"id": "x", "lidar_ground_z": 0.0}, np.zeros((4, 2)))
    cams = [make_camera(f"C/Fwd_{i}", np.array([0.0, -1500.0 - 100 * i, 1600.0]), omega=45.0) for i in range(6)]
    # the parser is replaced: the table already holds camera models
    monkeypatch.setattr(prep, "parse_camera_from_eo_record", lambda r, f: next(c for c in cams if c.frame_id == r["ID"]),
                        raising=False)
    import kyfromabove_3d.cameras as camsmod
    monkeypatch.setattr(camsmod, "parse_camera_from_eo_record",
                        lambda r, f: next(c for c in cams if c.frame_id == r["ID"]))
    ring_ft = ring_m / FT
    chosen, table = select_cameras(_frames_gdf(cams, ring_ft), ring_ft, frame, volume, max_per_head=4, min_frames=3)
    assert len(chosen) == 4 and table["selected"].sum() == 4
    # a volume too tall for any frame: relax to six corners
    tall = building_volume(ring_m, 4000.0)
    chosen2, table2 = select_cameras(_frames_gdf(cams, ring_ft), ring_ft, frame, tall, max_per_head=4, min_frames=3)
    assert (table2["corners_on_sensor"] < 8).all()
    assert len(chosen2) >= 3 or len(chosen2) == 0        # relaxed, or honestly nothing


def test_edge_top_z_reads_the_roof_beside_the_edge_not_the_tower():
    from kyfromabove_3d.stages.build_mesh import edge_top_z

    xs = np.arange(0.5, 40, 1.0); ys = np.arange(0.5, 20, 1.0)
    X, Y = np.meshgrid(xs, ys)
    Z = np.where(X > 20, 60.0, 12.0)                       # podium 12 m, tower 60 m on the east half
    valid = np.ones_like(Z, dtype=bool)
    assert abs(edge_top_z(xs, ys, Z, valid, np.array([0.0, 0.0]), np.array([0.0, 20.0])) - 12.0) < 1e-9
    assert abs(edge_top_z(xs, ys, Z, valid, np.array([40.0, 0.0]), np.array([40.0, 20.0])) - 60.0) < 1e-9


def test_party_edges_are_the_ones_along_a_neighbour():
    from kyfromabove_3d.stages.build_mesh import party_edges
    from kyfromabove_3d.features import wall_planes

    ring = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    faces = wall_planes(ring, 0.0, 10.0, min_length_m=0.3)
    neighbour = Polygon([(10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0)])   # shares the east edge
    party = party_edges(ring, [neighbour], faces)
    east = [f.face for f in faces if abs(f.normal[0] - 1.0) < 1e-6]
    assert set(party) == set(east)
    (s0, s1, top), = party[east[0]]
    assert abs(s0) < 0.6 and abs(s1 - 10.0) < 0.6 and top is None      # the whole edge; no height known
    # a tall neighbour on the south half of the east edge, a low one on the north half
    from kyfromabove_3d.stages.build_mesh import party_steps
    tall = Polygon([(10.0, 0.0), (20.0, 0.0), (20.0, 5.0), (10.0, 5.0)])
    low = Polygon([(10.0, 5.0), (20.0, 5.0), (20.0, 10.0), (10.0, 10.0)])
    party = party_edges(ring, [tall, low], faces, tops=[12.0, 4.0])
    st = party[east[0]]
    assert len(st) == 2 and st[0][2] == 12.0 and st[1][2] == 4.0
    steps = party_steps(st, 0.0, 10.0, 9.0)
    # south half: the neighbour is taller than this wall -> plain all the way; north half: plain to 4 m, seen above
    assert len(steps) == 2 and steps[0][2] == 9.0 and steps[1][2] == 4.0
    assert abs(steps[0][1] - 5.0) <= 0.6 and steps[0][0] == 0.0 and steps[1][1] == 10.0   # the cut within the 0.5 m tolerance
    assert party_steps([], 0.0, 10.0, 9.0) == [(0.0, 10.0, None)]


def test_split_parents_are_marked_and_left_out_of_the_merge(tmp_path):
    from run_downtown import mark_split_parents
    out = tmp_path
    parent = out / "buildings" / "abc"
    (parent / "3dtiles_mesh").mkdir(parents=True)
    (parent / "state.json").write_text(json.dumps({"gers": "abc", "status": "meshed"}))
    records = [{"id": "abc-a", "parent_gers": "abc"}, {"id": "abc-b", "parent_gers": "abc"}, {"id": "xyz"}]
    assert mark_split_parents(out, records) == 1
    st = json.loads((parent / "state.json").read_text())
    assert st["status"] == "split" and st["parts"] == ["abc-a", "abc-b"]
    assert mark_split_parents(out, records) == 0                        # already marked
    from merge_downtown_tileset import building_child
    (parent / "3dtiles_mesh" / "building.glb").write_bytes(b"glb")
    (parent / "3dtiles_mesh" / "tileset.json").write_text(json.dumps(
        {"root": {"content": {"uri": "building.glb"}, "transform": list(np.eye(4).ravel()),
                  "boundingVolume": {"sphere": [0, 0, 0, 1]}}, "extras": {"gers": "abc"}}))
    (tmp_path / "glb").mkdir()
    assert building_child(parent / "3dtiles_mesh" / "tileset.json", np.eye(4), tmp_path / "glb") is not None   # parts not built yet: the parent stays
    for part in ("abc-a", "abc-b"):
        (out / "buildings" / part / "3dtiles_mesh").mkdir(parents=True)
        (out / "buildings" / part / "3dtiles_mesh" / "building.glb").write_bytes(b"glb")
    assert building_child(parent / "3dtiles_mesh" / "tileset.json", np.eye(4), tmp_path / "glb") is None


def test_atlas_packs_images_and_maps_uv_back_onto_them():
    from kyfromabove_3d.stages.build_mesh import TextureAtlas

    atlas = TextureAtlas(page_px=64, gap_px=0)
    a = np.full((10, 20, 3), 10, dtype=np.uint8); b = np.full((30, 30, 3), 200, dtype=np.uint8)
    ha, hb = atlas.add(a), atlas.add(b)
    pages = atlas.pack()
    assert len(pages) == 1
    # points just inside image a's corners land on pixels of a
    pi, uv = atlas.uv(ha, np.array([[0.05, 0.05], [0.95, 0.95]]))
    H, W = pages[pi].shape[:2]
    for u, v in uv:
        row = int((1.0 - v) * H); col = int(u * W)
        assert pages[pi][row, col, 0] == 10
    pi, uv = atlas.uv(hb, np.array([[0.5, 0.5]]))
    row = int((1.0 - uv[0, 1]) * H); col = int(uv[0, 0] * W)
    assert pages[pi][row, col, 0] == 200


def test_merge_places_each_building_relative_to_the_downtown_root(tmp_path):
    from merge_downtown_tileset import col_major, merge

    out = tmp_path / "downtown"; out.mkdir()
    (out / "coordinates.json").write_text(json.dumps({"origin_source_xyz": [4919402.0, 3980509.0, 460.0]}))
    b = out / "buildings" / "g1" / "3dtiles_mesh"; b.mkdir(parents=True)
    (b / "building.glb").write_bytes(b"glTF" + bytes(20))
    from kyfromabove_3d.stages.build_mesh import enu_transform
    t_b, _ = enu_transform({"origin_source_xyz": [4919500.0, 3980600.0, 458.0]})
    (b / "tileset.json").write_text(json.dumps({
        "root": {"transform": t_b, "boundingVolume": {"sphere": [1.0, 2.0, 10.0, 25.0]}, "content": {"uri": "building.glb"}},
        "extras": {"gers": "g1", "name": "One", "triangles": 12, "path": "prism"}}))
    ex = merge(out, out / "buildings", [], None)
    ts = json.loads((out / "tileset.json").read_text())
    assert ex["buildings"] == 1 and ex["paths"] == {"prism": 1} and (out / "glb" / "g1.glb").exists()
    child = ts["root"]["children"][0]["children"][0]
    T_root = col_major(ts["root"]["transform"]); T_rel = col_major(child["transform"])
    assert np.allclose(T_root @ T_rel, col_major(t_b), atol=1e-6)
    assert child["content"]["uri"] == "glb/g1.glb" and child["extras"]["name"] == "One"


def test_to_ecef_uses_the_geoid_and_stops_without_it(monkeypatch):
    """Louisville's geoid lies 34.55 m below NAVD88 zero: the export must
    land there to the centimetre. Without the grid PROJ does not fail -- it
    falls back to a ballpark 1.2 m (network off) or 34 m (no shift) out --
    so the export must stop instead."""
    import pyproj
    from kyfromabove_3d.coordinates import to_ecef

    coords = {"origin_source_xyz": [4919401.519615987, 3980508.6132660736, 460.66]}   # downtown's origin
    _, geo = to_ecef(np.zeros((1, 3)), coords)
    navd88_m = 460.66 * 0.3048006096012192
    assert abs((navd88_m - geo[0, 2]) - 34.55) < 0.05

    real = pyproj.network.set_network_enabled
    monkeypatch.setattr(pyproj.network, "set_network_enabled", lambda *a, **kw: real(False))
    try:
        with pytest.raises(SystemExit, match="grids that are not available"):
            to_ecef(np.zeros((1, 3)), coords)
    finally:
        real(True)


def test_merge_refuses_a_building_placed_at_the_wrong_height(tmp_path):
    from merge_downtown_tileset import merge

    out = tmp_path / "downtown"; out.mkdir()
    (out / "coordinates.json").write_text(json.dumps({"origin_source_xyz": [4919402.0, 3980509.0, 460.0]}))
    b = out / "buildings" / "g1" / "3dtiles_mesh"; b.mkdir(parents=True)
    (b / "building.glb").write_bytes(b"glTF" + bytes(20))
    from kyfromabove_3d.stages.build_mesh import enu_transform
    t_b, _ = enu_transform({"origin_source_xyz": [4919500.0, 3980600.0, 458.0]})
    # lift the building 34 m: the signature of a rebuild without the geoid
    M = np.array(t_b).reshape(4, 4).T
    up = M[:3, 3] / np.linalg.norm(M[:3, 3])
    M[:3, 3] += 34.0 * up
    (b / "tileset.json").write_text(json.dumps({
        "root": {"transform": [float(v) for v in M.T.ravel()], "boundingVolume": {"sphere": [0, 0, 10, 25]},
                 "content": {"uri": "building.glb"}}, "extras": {"gers": "g1", "name": "Lifted"}}))
    with pytest.raises(SystemExit):
        merge(out, out / "buildings", [], None)


def _two_height_cloud(rng, low_h, high_h, x_cut=20.0, w=40.0, d=20.0, density=6):
    """Returns over a w x d rectangle: a roof at ``low_h`` west of ``x_cut``,
    ``high_h`` east of it (both above ground 0), with 30 cm of noise."""
    n = int(w * d * density)
    xy = rng.uniform([0, 0], [w, d], (n, 2))
    z = np.where(xy[:, 0] < x_cut, low_h, high_h) + rng.normal(0, 0.3, n)
    return np.column_stack([xy, z])


def test_roof_step_split_finds_two_buildings_side_by_side_but_not_a_tower_on_a_podium():
    from kyfromabove_3d.stages.overture_boxes import roof_step_split

    rng = np.random.default_rng(3)
    poly = Polygon([(0, 0), (40, 0), (40, 20), (0, 20)])
    found = roof_step_split(poly, _two_height_cloud(rng, 6.0, 15.0))
    assert found is not None and abs(found["step_m"] - 9.0) < 1.0 and abs(found["share_low"] - 0.5) < 0.1
    (x0, y0), (x1, y1) = found["line"]
    assert abs(x0 - 20.0) < 1.5 and abs(x1 - 20.0) < 1.5                # the cut runs north-south at x = 20
    # a step under 3 m is a roof feature, not a second building
    assert roof_step_split(poly, _two_height_cloud(rng, 6.0, 8.0)) is None
    # a tower in the middle of a podium is enclosed by the low roof: the levels rule's case
    n = 6000
    xy = rng.uniform([0, 0], [40, 20], (n, 2))
    tower = (xy[:, 0] > 14) & (xy[:, 0] < 26) & (xy[:, 1] > 6) & (xy[:, 1] < 14)
    z = np.where(tower, 30.0, 8.0) + rng.normal(0, 0.3, n)
    assert roof_step_split(poly, np.column_stack([xy, z])) is None


def _tiered_cloud(rng, heights, cuts, w=60.0, d=20.0, density=6):
    """Returns over a w x d rectangle stepping along x: ``heights[i]`` between
    ``cuts[i-1]`` and ``cuts[i]``, with 30 cm of noise."""
    n = int(w * d * density)
    xy = rng.uniform([0, 0], [w, d], (n, 2))
    z = np.asarray(heights)[np.searchsorted(cuts, xy[:, 0])] + rng.normal(0, 0.3, n)
    return np.column_stack([xy, z])


def test_three_roof_heights_in_a_row_are_three_buildings():
    """39a33bd6's case: one outline, three roofs stepping down along it."""
    from kyfromabove_3d.stages.overture_boxes import cut_into_tiers, roof_tiers

    rng = np.random.default_rng(5)
    poly = Polygon([(0, 0), (60, 0), (60, 20), (0, 20)])
    found = roof_tiers(poly, _tiered_cloud(rng, [4.5, 8.0, 12.5], [20.0, 40.0]))
    assert found is not None and len(found["tiers"]) == 3
    assert [round(t["z"]) for t in found["tiers"]] == [5, 8, 13]
    for seam in found["seams"]:                                  # both seams run across the building
        (x0, _), (x1, _) = seam["line"]
        assert abs(x0 - x1) < 3.0 and min(abs(x0 - 20.0), abs(x0 - 40.0)) < 1.5
    parts, why = cut_into_tiers(poly, found)
    assert why is None and len(parts) == 3
    assert [round(g.area / 100) for _, g in parts] == [4, 4, 4]  # 20 x 20 m each, the tallest first
    assert parts[0][1].centroid.x > 40 and parts[2][1].centroid.x < 20


def test_two_roof_heights_are_still_two_buildings_and_a_tower_on_a_podium_none():
    from kyfromabove_3d.stages.overture_boxes import cut_into_tiers, roof_tiers

    rng = np.random.default_rng(6)
    poly = Polygon([(0, 0), (40, 0), (40, 20), (0, 20)])
    cloud = _two_height_cloud(rng, 6.0, 15.0)
    assert roof_tiers(poly, cloud, n_tiers=3) is None               # two roofs are not three
    found = roof_tiers(poly, cloud, n_tiers=2)
    assert found is not None and len(found["tiers"]) == 2
    parts, why = cut_into_tiers(poly, found)
    assert why is None and len(parts) == 2
    n = 6000
    xy = rng.uniform([0, 0], [40, 20], (n, 2))
    tower = (xy[:, 0] > 14) & (xy[:, 0] < 26) & (xy[:, 1] > 6) & (xy[:, 1] < 14)
    assert roof_tiers(poly, np.column_stack([xy, np.where(tower, 30.0, 8.0) + rng.normal(0, 0.3, n)]), n_tiers=2) is None


def test_a_cut_that_leaves_a_sliver_is_refused():
    """A seam running along the building from one corner, as 39a33bd6's did."""
    from kyfromabove_3d.stages.overture_boxes import cut_into_tiers, min_corner_deg

    poly = Polygon([(0, 0), (60, 0), (60, 20), (0, 20)])
    assert min_corner_deg(poly) == pytest.approx(90.0)
    ys = np.arange(0.5, 20, 1.0)
    xs = np.arange(0.5, 60, 1.0)
    X, Y = np.meshgrid(xs, ys)
    below = Y < X / 3.0                                         # the triangle under the diagonal
    found = {"tiers": [{"z": 5.0, "share": 0.5, "cells": np.column_stack([X[below], Y[below]])},
                       {"z": 12.0, "share": 0.5, "cells": np.column_stack([X[~below], Y[~below]])}],
             "seams": [{"between": (0, 1), "line": ((-60.0, -20.0), (120.0, 40.0)), "rms_m": 0.1}]}
    parts, why = cut_into_tiers(poly, found)
    assert parts == [] and "degree corner" in why


def test_roof_height_is_measured_inside_the_edge_where_a_neighbours_wall_spills():
    """9947f2b0's case: a 15 m roof, and a 110 m neighbour's facade returns in a
    30 cm strip along one edge of the outline."""
    from kyfromabove_3d.stages.overture_boxes import roof_returns

    rng = np.random.default_rng(8)
    poly = Polygon([(0, 0), (40, 0), (40, 20), (0, 20)])
    roof = np.column_stack([rng.uniform(0, 40, 6000), rng.uniform(0, 20, 6000), rng.normal(15.0, 0.3, 6000)])
    spill = np.column_stack([rng.uniform(39.7, 40.0, 800), rng.uniform(0, 20, 800), rng.uniform(20, 110, 800)])
    above = np.vstack([roof, spill])
    assert np.percentile(above[:, 2], 99) > 90                      # the whole footprint: the neighbour
    assert abs(np.percentile(roof_returns(poly, above)[:, 2], 99) - 15.7) < 0.5
    tiny = Polygon([(0, 0), (2.5, 0), (2.5, 2.5), (0, 2.5)])          # no inside to speak of: all of it
    few = np.column_stack([rng.uniform(0, 2.5, 30), rng.uniform(0, 2.5, 30), np.full(30, 6.0)])
    assert len(roof_returns(tiny, few)) == 30
