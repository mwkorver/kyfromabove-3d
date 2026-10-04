"""Step 4, the batched 3D Tiles, with no network and no basisu: synthetic
buildings (a textured wall, a flat roof) written by build_mesh's own writer
with PNG textures, batched by publish into area glbs."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from kyfromabove_3d import publish as P                                     # noqa: E402

build_mesh = pytest.importorskip("build_mesh")


def _building_glb(colour, name="b"):
    """A 10 x 6 x 8 m box-ish building in its own frame (z-up): one textured
    wall (a 16 x 8 image of ``colour`` with a white left half) and a flat
    grey roof. Returns (glb bytes, {page name: image})."""
    w = build_mesh.GlbWriter("png")
    img = np.zeros((8, 16, 3), np.uint8)
    img[:] = colour
    img[:, :8] = 255
    mat = w.material_texture(img, "walls_0")
    pos = np.array([[0, 0, 0], [10, 0, 0], [10, 0, 8], [0, 0, 8]], float)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)          # v up, as build_mesh gives it
    w.primitive(pos, np.tile([0.0, -1.0, 0.0], (4, 1)), uv, np.array([0, 1, 2, 0, 2, 3]), mat)
    roof = w.material_flat((120, 120, 120), "roof_flat")
    rp = np.array([[0, 0, 8], [10, 0, 8], [10, 6, 8], [0, 6, 8]], float)
    w.primitive(rp, np.tile([0.0, 0.0, 1.0], (4, 1)), None, np.array([0, 1, 2, 0, 2, 3]), roof)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / f"{name}.glb"
        w.write(path, {"name": name})
        data = path.read_bytes()
    return data, {"walls_0": img}


def _enu(lon, lat, h):
    """An east-north-up placement at lon/lat/h (degrees, metres), as a
    tileset transform (4x4)."""
    from pyproj import Transformer
    X, Y, Z = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True).transform(lon, lat, h)
    lo, la = np.radians(lon), np.radians(lat)
    e = [-np.sin(lo), np.cos(lo), 0.0]
    n = [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)]
    u = [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = e, n, u, [X, Y, Z]
    return M


def _decode(glb):
    js, binary = P.read_glb(glb)
    return js, binary


def test_every_building_is_in_exactly_one_area_and_areas_respect_the_limits():
    rng = np.random.default_rng(0)
    centres = np.vstack([rng.normal([0, 0], 40, (400, 2)), rng.uniform(-700, 700, (60, 2))])
    sizes = rng.uniform(5_000, 80_000, len(centres))
    areas = P.split_areas(centres, sizes, max_buildings=50, max_bytes=1_000_000)
    flat = np.concatenate(areas)
    assert sorted(flat.tolist()) == list(range(len(centres)))
    assert all((len(a) <= 50 and sizes[a].sum() <= 1_000_000) or len(a) == 1 for a in areas)
    assert len(areas) < len(centres) / 3                                     # batched, not one per building
    assert P.split_areas(np.zeros((0, 2)), np.zeros(0)) == []


def test_a_batched_building_lands_where_its_own_tileset_put_it():
    T_tile = _enu(-85.745, 38.260, 105.0)
    T_b = [_enu(-85.7440, 38.2610, 106.3), _enu(-85.7462, 38.2588, 104.1)]
    members = []
    for i, T in enumerate(T_b):
        glb, pages = _building_glb((200, 30 * i, 40), f"b{i}")
        members.append({"record": {"id": f"g{i}", "name": f"Building {i}", "lidar_height_m": 8.0},
                        "parts": P.building_parts(glb, pages), "pages": pages,
                        "M": np.linalg.inv(T_tile) @ T, "archetype": "box", "path": "profile"})
    data, lo, hi = P.area_glb(members, lambda m: m["M"], "measured", texture_format="png")
    js, binary = _decode(data)
    # every vertex each building's own tileset places, in the world
    own = {}
    for f, m in enumerate(members):
        pts = np.vstack([(P.GLTF_TO_ZUP @ np.column_stack([p["pos"], np.ones(len(p["pos"]))]).T).T for p in m["parts"]])
        own[f] = (T_b[f] @ pts.T).T[:, :3]
    checked = 0
    for prim in js["meshes"][0]["primitives"]:
        pos = P.accessor(js, binary, prim["attributes"]["POSITION"]).astype(float)
        fid = P.accessor(js, binary, prim["attributes"]["_FEATURE_ID_0"]).astype(int)
        world = (T_tile @ P.GLTF_TO_ZUP @ np.column_stack([pos, np.ones(len(pos))]).T).T[:, :3]
        for v, f in zip(world, fid):
            assert np.min(np.linalg.norm(own[f] - v, axis=1)) < 0.001          # within a millimetre
            checked += 1
    assert checked == sum(len(o) for o in own.values())
    assert (lo < hi).all()


def test_textures_survive_the_repacking_and_the_properties_come_back():
    from PIL import Image

    T = np.eye(4)
    members = []
    colours = [(200, 20, 40), (20, 180, 60), (30, 40, 210)]
    for i, c in enumerate(colours):
        glb, pages = _building_glb(c, f"t{i}")
        M = np.eye(4); M[0, 3] = 30.0 * i
        members.append({"record": {"id": f"gers-{i}", "name": ["A", None, "C"][i], "lidar_height_m": 8.0 + i},
                        "parts": P.building_parts(glb, pages), "pages": pages, "M": M, "path": "profile"})
    data, _, _ = P.area_glb(members, lambda m: m["M"], "complete", texture_format="png")
    js, binary = _decode(data)
    # one textured primitive for all three (one atlas page), one flat roof primitive
    kinds = sorted("TEXCOORD_0" in p["attributes"] for p in js["meshes"][0]["primitives"])
    assert kinds == [False, True]
    tex = next(p for p in js["meshes"][0]["primitives"] if "TEXCOORD_0" in p["attributes"])
    uv = P.accessor(js, binary, tex["attributes"]["TEXCOORD_0"]).astype(float)
    fid = P.accessor(js, binary, tex["attributes"]["_FEATURE_ID_0"]).astype(int)
    img_i = js["textures"][js["materials"][tex["material"]]["pbrMetallicRoughness"]["baseColorTexture"]["index"]]["source"]
    view = js["bufferViews"][js["images"][img_i]["bufferView"]]
    atlas = np.asarray(Image.open(io.BytesIO(binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]])).convert("RGB"))
    H, W = atlas.shape[:2]
    for f, c in enumerate(colours):
        # a point three-quarters along the wall, halfway up: the coloured (right) half
        s = uv[fid == f]
        s_mid = np.array([0.25 * s[:, 0].min() + 0.75 * s[:, 0].max(), s[:, 1].mean()])
        px = atlas[int(s_mid[1] * H), int(s_mid[0] * W)]
        assert np.abs(px.astype(int) - np.array(c)).max() <= 2
    # the property table: gers and names back, a missing name empty
    ext = js["extensions"]["EXT_structural_metadata"]
    table = ext["propertyTables"][0]
    assert table["count"] == 3 and ext["schema"]["classes"]["building"]["properties"]["gers"]["type"] == "STRING"

    def strings(prop):
        v = js["bufferViews"][table["properties"][prop]["values"]]
        o = js["bufferViews"][table["properties"][prop]["stringOffsets"]]
        offs = np.frombuffer(binary, np.uint32, o["byteLength"] // 4, o["byteOffset"])
        raw = binary[v["byteOffset"]:v["byteOffset"] + v["byteLength"]]
        return [raw[a:b].decode() for a, b in zip(offs[:-1], offs[1:])]
    assert strings("gers") == ["gers-0", "gers-1", "gers-2"]
    assert strings("name") == ["A", "", "C"]
    for p in js["meshes"][0]["primitives"]:
        fids = p["extensions"]["EXT_mesh_features"]["featureIds"][0]
        assert fids["propertyTable"] == 0 and fids["attribute"] == 0
        a = js["accessors"][p["attributes"]["_FEATURE_ID_0"]]
        assert a["componentType"] == 5126 and js["bufferViews"][a["bufferView"]]["byteOffset"] % 4 == 0
    for v in (table["properties"][k]["values"] for k in table["properties"]):
        assert js["bufferViews"][v]["byteOffset"] % 8 == 0                   # the extension asks for 8


def test_a_node_matrix_places_content_like_a_tile_transform():
    T = _enu(-85.745, 38.26, 105.0)
    g = P.FeatureGlb("png")
    g.features.append({"gers": "x"})
    g.primitive(np.array([[1.0, 2.0, 3.0], [2, 2, 3], [1, 3, 3]]), np.tile([0, 0, 1.0], (3, 1)), np.array([0, 1, 2]),
                np.zeros(3), g.w.material_flat((1, 2, 3), "m"))
    js, binary = P.read_glb(g.to_bytes({"name": "n"}, node_matrix=T))
    N = P.col_major(js["nodes"][0]["matrix"])
    pos = P.accessor(js, binary, js["meshes"][0]["primitives"][0]["attributes"]["POSITION"]).astype(float)
    # CesiumJS: node matrix, then glTF y-up -> z-up
    world = (P.GLTF_TO_ZUP @ N @ np.column_stack([pos, np.ones(3)]).T).T[:, :3]
    want = (T @ np.array([1.0, 2.0, 3.0, 1.0]))[:3]
    assert np.linalg.norm(world[0] - want) < 0.001


def test_a_region_holds_its_tileset_and_unions_hold_their_parts():
    T = _enu(-85.745, 38.26, 105.0)
    ts = {"root": {"transform": P.to_col_major(T), "boundingVolume": P.box_volume(np.array([-700.0, -700, -5]),
                                                                                  np.array([700.0, 700, 60]))}}
    r = P.tileset_region(ts)
    assert r[0] < np.radians(-85.745) < r[2] and r[1] < np.radians(38.26) < r[3] and r[4] < 105 < r[5]
    u = P.union_region([r, [r[0] - 0.001, r[1], r[2], r[3], r[4] - 10, r[5]]])
    assert u[0] == r[0] - 0.001 and u[4] == r[4] - 10


# ------------------------------------------------------------- the boxes

def test_a_prism_shares_its_roof_corners_and_its_roof_covers_the_outline():
    from shapely.geometry import Polygon

    L = Polygon([(0, 0), (20, 0), (20, 8), (8, 8), (8, 15), (0, 15)])            # concave: an L
    Pp, Nn, I = P.prism(L, z0=100.0, h=9.0)
    walls, roof = 4 * 6, 6
    assert len(Pp) == walls + roof                                               # one point a roof corner
    tri = I.reshape(-1, 3)
    roof_tris = tri[(Pp[tri][:, :, 2] == 109.0).all(axis=1) & (Nn[tri][:, 0, 2] == 1.0)]
    a = Pp[roof_tris]
    area = 0.5 * np.abs((a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1])
                        - (a[:, 1, 1] - a[:, 0, 1]) * (a[:, 2, 0] - a[:, 0, 0])).sum()
    assert abs(area - L.area) < 1e-6                                             # no gap, no overlap
    cross = np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0])
    assert (cross[:, 2] > 0).all()                                               # every roof face up
    assert P.prism(Polygon([(0, 0), (1, 0), (2, 0)]), 0.0, 5.0) is None          # no area, no box


def _boxes(n, seed=0):
    from shapely.geometry import box as sbox
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x, y = rng.uniform(-700, 700, 2)
        out.append(P.prism(sbox(x, y, x + rng.uniform(6, 30), y + rng.uniform(6, 30)),
                           rng.uniform(100, 140), rng.uniform(3, 40)))
    return out


def _world(js, binary, T=None):
    """Every primitive's vertices placed as CesiumJS places them: the node
    matrix (dequantizing), glTF y-up -> z-up, then a tile transform."""
    N = P.col_major(js["nodes"][0]["matrix"]) if "matrix" in js["nodes"][0] else np.eye(4)
    out = []
    for prim in js["meshes"][0]["primitives"]:
        pos = P.accessor(js, binary, prim["attributes"]["POSITION"]).astype(float)
        fid = P.accessor(js, binary, prim["attributes"]["_FEATURE_ID_0"]).astype(int)
        idx = P.accessor(js, binary, prim["indices"]).astype(int)
        w = ((T if T is not None else np.eye(4)) @ P.GLTF_TO_ZUP @ N @ np.column_stack([pos, np.ones(len(pos))]).T).T
        out.append((w[:, :3], fid, idx, prim))
    return out


def _strided_accessor(js, binary, i):
    a = js["accessors"][i]
    v = js["bufferViews"][a["bufferView"]]
    dt = {5122: np.int16, 5120: np.int8}[a["componentType"]]
    rows = np.frombuffer(binary, dt, a["count"] * v["byteStride"] // np.dtype(dt).itemsize, v.get("byteOffset", 0))
    return rows.reshape(a["count"], -1)[:, :3]


def test_quantized_boxes_land_within_two_centimetres_and_stay_small():
    boxes = _boxes(40)
    feats = [{"gers": f"g{i}", "path": "box"} for i in range(len(boxes))]
    data, lo, hi = P.boxes_glb(boxes, feats, {"name": "t"}, texture_format="png")
    js, binary = P.read_glb(data)
    assert "KHR_mesh_quantization" in js["extensionsRequired"]
    want = {i: b[0] for i, b in enumerate(boxes)}
    for world, fid, idx, prim in _world(js, binary):
        pos = js["accessors"][prim["attributes"]["POSITION"]]
        assert pos["componentType"] == 5122 and js["bufferViews"][pos["bufferView"]]["byteStride"] == 8
        nrm = js["accessors"][prim["attributes"]["NORMAL"]]
        assert nrm["componentType"] == 5120 and nrm["normalized"]
        assert js["accessors"][prim["indices"]]["componentType"] == 5123           # 2-byte indices
        for f in np.unique(fid):
            got = world[fid == f]
            assert np.abs(got - want[f]).max() < 0.02                           # within 2 cm
    np.testing.assert_allclose(lo, np.vstack([b[0] for b in boxes]).min(axis=0))
    # the int16 positions sit inside the accessor's min and max, as the validator checks
    q = _strided_accessor(js, binary, js["meshes"][0]["primitives"][0]["attributes"]["POSITION"])
    a = js["accessors"][js["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert (q.min(axis=0) == np.array(a["min"])).all() and (q.max(axis=0) == np.array(a["max"])).all()


def test_box_geometry_is_cut_between_buildings_into_primitives_small_enough_for_2_byte_indices():
    boxes = _boxes(30)
    g = P.FeatureGlb("png")
    groups = {}
    for i, b in enumerate(boxes):
        P._merge(groups, "b", *b, np.full(len(b[0]), i))
        g.features.append({"gers": f"g{i}"})
    gb = groups["b"]
    g.box_primitives(np.vstack(gb["pos"]), np.vstack(gb["nrm"]), np.concatenate(gb["idx"]),
                     np.concatenate(gb["fid"]), g.w.material_flat(P.BOX_RGB, "box"), max_vertices=100)
    js, binary = P.read_glb(g.to_bytes({"name": "t"}))
    prims = _world(js, binary)
    assert len(prims) > 3
    seen, tris = set(), 0
    for world, fid, idx, prim in prims:
        assert len(world) <= 100 and idx.max() < len(world)
        assert not (set(fid.tolist()) & seen)                                    # a building in one primitive
        seen |= set(fid.tolist())
        tris += len(idx) // 3
    assert seen == set(range(30)) and tris == sum(len(b[2]) // 3 for b in boxes)


def test_quantized_content_with_a_node_matrix_lands_like_a_tile_transform():
    T = _enu(-85.745, 38.26, 105.0)
    boxes = _boxes(5)
    data, _, _ = P.boxes_glb(boxes, [{"gers": str(i)} for i in range(5)], {"name": "c"}, "png", node_matrix=T)
    js, binary = P.read_glb(data)
    for world, fid, _, _ in _world(js, binary):
        for f in np.unique(fid):
            want = (T @ np.column_stack([boxes[f][0], np.ones(len(boxes[f][0]))]).T).T[:, :3]
            assert np.abs(world[fid == f] - want).max() < 0.02


def test_an_area_of_boxes_only_is_written_small():
    boxes = _boxes(3)
    members = [{"record": {"id": f"g{i}", "lidar_height_m": 9.0}, "box": b} for i, b in enumerate(boxes)]
    data, lo, hi = P.area_glb(members, lambda m: None, "measured", texture_format="png")
    js, _ = P.read_glb(data)
    assert "KHR_mesh_quantization" in js["extensionsRequired"] and (lo < hi).all()


def test_the_group_levels_keep_buildings_by_their_largest_dimension():
    from shapely.geometry import box as sbox
    ((_, _, keep16), (_, _, keep4)) = P.GROUP_LEVELS
    assert keep4["min_size_m"] < keep16["min_size_m"]
    wide = sbox(0, 0, 30, 30).area ** 0.5                                        # a 30 m square: 30 m
    assert wide >= keep4["min_size_m"] and wide < keep16["min_size_m"]


def test_the_built_buildings_come_from_one_listing(tmp_path):
    from kyfromabove_3d.store import LocalStore
    st = LocalStore(tmp_path)
    st.put_json("buildings/g1/3dtiles_mesh/tileset.json", {})
    st.put_json("buildings/g1/3dtiles_complete/tileset.json", {})
    st.put_json("buildings/g2/3dtiles_mesh/tileset.json", {})
    st.put_json("buildings/g3/audit.json", {})                                  # no mesh
    st.put_json("buildings/g4/3dtiles_mesh/textures/tileset.json", {})         # deeper: not a mesh folder
    got = P.built_buildings(st)
    assert got == {"g1": {"3dtiles_mesh", "3dtiles_complete"}, "g2": {"3dtiles_mesh"}}
    assert P.load_building(st, "g3", "measured", available=got) is None       # asks nothing of the store


# ------------------------------------------------------------- the views

def test_each_view_draws_its_own_height():
    r = {"id": "g", "lidar_height_m": 11.0, "height": 14.0, "num_floors": 3}
    assert P.box_height(r, "boxes") == (11.0, "lidar")
    assert P.box_height(r, "measured") == (11.0, "lidar")                      # boxes in a meshed view
    assert P.box_height(r, "overture") == (14.0, "height")                     # Overture's own
    assert P.box_height({"id": "g", "num_floors": 3}, "overture") == (3 * 3.5, "floors")
    assert P.box_height({"id": "g"}, "overture") == (6.0, "default")           # the guess shows
    assert P.box_height({"id": "g"}, "boxes") == (None, None)
    assert P.box_colour("default", "overture") != P.box_colour("height", "overture")
    assert P.box_colour("lidar", "boxes") == P.BOX_RGB
    f = P.box_feature(r, "overture")
    assert f["height_m"] == 14.0 and f["height_from"] == "height" and f["lidar_height_m"] == 11.0
    assert "lidar_height_m" not in P.box_feature(r, "boxes")


def test_boxes_in_several_colours_share_one_frame_and_land_within_two_centimetres():
    boxes = _boxes(12)
    colours = [P.box_colour(k, "overture") for k in ["height", "floors", "default"] * 4]
    feats = [{"gers": f"g{i}", "height_from": ["height", "floors", "default"][i % 3]} for i in range(12)]
    data, _, _ = P.boxes_glb(boxes, feats, {"name": "t"}, texture_format="png", colours=colours)
    js, binary = P.read_glb(data)
    mats = {p["material"] for p in js["meshes"][0]["primitives"]}
    assert len(mats) == 3 and len(js["materials"]) == 3                       # one per colour
    got = {}
    for world, fid, idx, prim in _world(js, binary):
        rgb = js["materials"][prim["material"]]["pbrMetallicRoughness"]["baseColorFactor"][:3]
        for f in np.unique(fid):
            assert np.abs(world[fid == f] - boxes[f][0]).max() < 0.02
            got[int(f)] = tuple(round(c * 255) for c in rgb)
    assert got == {i: tuple(colours[i]) for i in range(12)}                    # each box in its colour


def test_the_new_properties_come_back_from_the_property_table():
    r = {"id": "g1", "lidar_height_m": 11.0, "height": 14.0}
    data, _, _ = P.boxes_glb(_boxes(1), [P.box_feature(r, "overture")], {"name": "t"}, texture_format="png")
    js, binary = P.read_glb(data)
    table = js["extensions"]["EXT_structural_metadata"]["propertyTables"][0]
    for k in ("height_from", "overture_height_m", "lidar_height_m"):
        assert k in table["properties"]
    v = js["bufferViews"][table["properties"]["overture_height_m"]["values"]]
    assert np.frombuffer(binary, np.float32, 1, v["byteOffset"])[0] == 14.0


def test_the_box_views_draw_every_building_as_a_box_even_where_there_is_a_mesh(tmp_path):
    from kyfromabove_3d.store import LocalStore

    raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    if not (raw / "kyfromabove_phase2_pointcloud_5k_grid.gpkg").exists():
        pytest.skip("the LiDAR tile grid is not on this machine")
    from kyfromabove_3d.pieces import phase2_grid

    grid = phase2_grid(raw)
    recs = json.loads((Path(__file__).parent / "data" / "publish_small" / "N076E229_buildings.json").read_text())
    st = LocalStore(tmp_path)
    st.put_json("tiles/N076E229/buildings.json", recs)
    built = {recs["buildings"][0]["id"]: {"3dtiles_mesh"}}                        # said to have a mesh: never read
    got = {}
    for product in ("overture", "boxes"):
        s = P.publish_tile("N076E229", st, grid, product, texture_format="png", log=lambda m: None, available=built)
        assert s["meshed"] == 0 and s["boxed"] == len(recs["buildings"])
        ts = st.read_json(f"tiles/N076E229/3dtiles/{product}/tileset.json")
        assert ts["root"]["content"]["uri"].startswith("boxes_") and ts["root"]["children"]
        js, binary = P.read_glb(st.get_bytes(f"tiles/N076E229/3dtiles/{product}/{ts['root']['content']['uri']}"))
        got[product] = {tuple(round(c * 255) for c in m["pbrMetallicRoughness"]["baseColorFactor"][:3])
                        for m in js["materials"]}
    assert got["boxes"] == {P.BOX_RGB}
    assert P.box_colour("default", "overture") in got["overture"]              # the 6 m guesses, in orange
    assert P.box_colour("height", "overture") in got["overture"]


def test_a_number_no_building_has_is_kept_and_a_text_is_left_out():
    g = P.FeatureGlb("png")
    g.features = [{"gers": "a"}, {"gers": "b"}]                                  # no heights, no names
    g.primitive(np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.tile([0, 0, 1.0], (3, 1)), np.array([0, 1, 2]),
                np.zeros(3), g.w.material_flat((1, 2, 3), "m"))
    js, binary = P.read_glb(g.to_bytes({"name": "t"}))
    props = js["extensions"]["EXT_structural_metadata"]["propertyTables"][0]["properties"]
    assert "overture_height_m" in props and "name" not in props
    v = js["bufferViews"][props["overture_height_m"]["values"]]
    assert np.isnan(np.frombuffer(binary, np.float32, 2, v["byteOffset"])).all()
