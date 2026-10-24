"""The building mesh: wall rectangles from profile bands, the roof height
field, and a glTF that reads back."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from kyfromabove_3d.stages.build_mesh import (                                  # noqa: E402
    GlbWriter, _fill_holes, face_texture, outside_modelled_wall, plane_bands, roll_quads,
    quads_to_arrays, roof_heightfield, roof_mesh, wall_geometry,
)
from kyfromabove_3d.features import FaceBand, VerifiedWallPlane, WallPlane   # noqa: E402


def _band(z_lo, z_hi, offset, status="observed", **kw):
    return FaceBand(z_lo=z_lo, z_hi=z_hi, offset_m=offset, n_returns=100, spread_m=0.1,
                    along_lo=0.0, along_hi=40.0, status=status, **kw)


def _plane():
    # a 40 m face along +x, outward normal -y
    return WallPlane(face=0, p0=np.array([0.0, 0.0]), p1=np.array([40.0, 0.0]),
                     normal=np.array([0.0, -1.0]), z_lo=0.0, z_hi=30.0)


def test_wall_geometry_makes_one_rectangle_per_band_and_a_ledge_at_a_setback():
    bands = [_band(0, 10, 0.0), _band(10, 20, -5.0)]          # the upper band stands 5 m back
    quads, extent = wall_geometry(_plane(), bands)
    kinds = [q[3] for q in quads]
    assert kinds.count("wall") == 2
    assert kinds.count("ledge") == 1
    ledge = next(q for q in quads if q[3] == "ledge")
    assert np.allclose(ledge[0][:, 2], 10.0)                  # at the band boundary
    assert ledge[1][2] > 0                                    # faces up: the set-back exposes the podium roof
    assert extent == (0.0, 20.0, 40.0)


def test_recess_segment_splits_the_band_and_adds_return_walls():
    bands = [_band(0, 10, 0.0, segments=[(10.0, 20.0, -2.0)])]
    quads, _ = wall_geometry(_plane(), bands)
    walls = [q for q in quads if q[3] == "wall"]
    returns = [q for q in quads if q[3] == "return"]
    assert len(walls) == 3 and len(returns) == 2
    recessed = walls[1][0]
    assert np.allclose(recessed[:, 1], 2.0)                   # normal is -y, so -2 m offset is +2 in y


def test_isolated_open_band_without_solid_walls_returns_no_geometry():
    # An open band (e.g. from roof returns misclassified as loggia) with no solid
    # wall bands on the face must not produce floating wall geometry
    open_band = _band(50, 55, None, status="open", back_offset_m=-6.0)
    quads, extent = wall_geometry(_plane(), [open_band])
    assert quads == []
    assert extent is None


def test_wall_geometry_does_not_extend_beyond_profile_bands_to_plane_z_hi():
    # If a plane has z_hi=100m (e.g. tower maximum) but the face profile stops at 20m
    # (e.g. lower wing / setback), wall_geometry must not extrude wall panels up to 100m
    plane = WallPlane(face=0, p0=np.array([0.0, 0.0]), p1=np.array([40.0, 0.0]),
                      normal=np.array([0.0, -1.0]), z_lo=0.0, z_hi=100.0)
    bands = [_band(0, 10, 0.0), _band(10, 20, 0.0)]
    quads, extent = wall_geometry(plane, bands)
    assert extent == (0.0, 20.0, 40.0)
    max_z = max(q[0][:, 2].max() for q in quads)
    assert max_z <= 20.0


def test_verified_plane_bands_cover_the_face_and_skip_open_bands():
    plane = VerifiedWallPlane(face=0, p0=np.array([0.0, 0.0]), p1=np.array([40.0, 0.0]),
                              normal=np.array([0.0, -1.0]), z_lo=0.0, z_hi=30.0, z_top=30.0)
    prof = [_band(0, 10, 0.3), _band(10, 20, None, status="open"), _band(20, 30, 0.2)]
    out = plane_bands(plane, prof)
    assert [(b.z_lo, b.z_hi) for b in out] == [(0, 10), (20, 30)]
    assert all(b.offset_m == 0.0 and b.along_hi == 40.0 for b in out)


def test_face_texture_rasterises_points_on_the_band_only():
    plane = _plane()
    bands = [_band(0, 10, 0.0)]
    patch = np.array([[x + 0.5, 0.0, z + 0.5] for x in range(3, 8) for z in range(2, 7)])   # 5 m x 5 m of wall
    on = np.vstack([patch, [[35.0, 0.1, 2.0]]])                # on the wall
    off = np.array([[20.0, -3.0, 5.0]])                      # 3 m in front of it
    xyz = np.vstack([on, off])
    rgb = np.array([[200, 0, 0]] * len(patch) + [[0, 200, 0], [0, 0, 200]], dtype=np.uint8)
    tex, n = face_texture(plane, bands, (0.0, 10.0, 40.0), xyz, rgb, texel_m=1.0)
    assert n == len(patch) + 1
    assert tex.shape == (10, 40, 3)
    assert tex[-6, 5].tolist() == [200, 0, 0]                 # row 4 from the bottom, flipped for the image
    assert not (tex == [0, 0, 200]).all(axis=2).any()        # the off-wall point never lands


def test_fill_holes_bridges_a_small_gap_and_leaves_the_rest_plain():
    img = np.zeros((1, 40, 3)); cnt = np.zeros((1, 40))
    img[0, 0] = [255, 0, 0]; cnt[0, 0] = 1
    img[0, 6] = [255, 0, 0]; cnt[0, 6] = 1
    img[0, 39] = [0, 0, 255]; cnt[0, 39] = 1
    out = _fill_holes(img, cnt, max_fill_px=5)
    assert out[0, 3].tolist() == [255, 0, 0]                  # between two seen within 5 px: nearest
    assert out[0, 20].tolist() == [255, 0, 0]                 # beyond: the median of what was filled
    assert out.source[0, 9] == 3 and out.source[0, 3] == 2    # past the last seen point: plain, not grown


def test_roof_heightfield_and_mesh_keep_flat_cells_and_split_steep_ones():
    rng = np.random.default_rng(0)
    poly = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    # a 10 m block on the west half, a 30 m block on the east half
    xs = rng.uniform(0, 20, 20000); ys = rng.uniform(0, 20, 20000)
    z = np.where(xs < 10, 10.0, 30.0) + rng.normal(0, 0.05, 20000)
    xyz = np.column_stack([xs, ys, z])
    gx, gy, Z, valid = roof_heightfield(xyz, poly, ground_z=0.0, cell=1.0)
    assert valid.sum() > 300
    assert abs(np.nanmedian(Z[:, :8]) - 10.0) < 0.3 and abs(np.nanmedian(Z[:, 12:]) - 30.0) < 0.3
    verts, flat, steep = roof_mesh(gx, gy, Z, valid, wall_quads=[])
    assert len(flat) > 500 and len(steep) > 10
    # the step between the blocks is the steep set
    cen = verts[steep].mean(axis=1)
    assert np.all(np.abs(cen[:, 0] - 10.0) < 1.5)


def test_outside_modelled_wall_flags_cells_between_the_wall_and_the_edge():
    plane = _plane()
    profiles = {0: [_band(40, 80, -20.0)]}                    # the tower stands 20 m back at 40-80 m
    cen = np.array([[20.0, 5.0, 60.0],                        # 5 m in from the edge: under the overhang
                    [20.0, 25.0, 60.0],                       # behind the wall
                    [20.0, 5.0, 35.0],                        # no band of its own; the 40-80 band is within 30 m and speaks
                    [20.0, 5.0, 5.0]])                        # podium height: nothing known within 30 m, left alone
    out = outside_modelled_wall(cen, [plane], profiles)
    assert out.tolist() == [True, False, True, False]


def test_glb_writes_a_valid_binary_with_textures_and_extras(tmp_path):
    glb = GlbWriter("png")
    tex = np.full((4, 8, 3), 90, dtype=np.uint8)
    mat = glb.material_texture(tex, "face_0")
    quads, _ = wall_geometry(_plane(), [_band(0, 10, 0.0)])
    P, N, T, I = quads_to_arrays(quads)
    glb.primitive(P, N, T, I, mat)
    glb.primitive_coloured(P, N, np.full((len(P), 3), 200.0), I, glb.material_flat((255, 255, 255), "steep"))
    path = tmp_path / "b.glb"
    total = glb.write(path, {"name": "test", "gers": "abc"})
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF" and version == 2 and length == len(data) == total
    jlen, jtype = struct.unpack_from("<II", data, 12)
    js = json.loads(data[20:20 + jlen])
    assert jtype == 0x4E4F534A
    assert js["nodes"][0]["extras"]["gers"] == "abc"
    assert "KHR_materials_unlit" in js["extensionsUsed"]
    assert len(js["meshes"][0]["primitives"]) == 2
    assert js["images"][0]["mimeType"] == "image/png"
    assert "extensionsRequired" not in js and js["textures"][0]["source"] == 0
    # y-up: the wall's height went into y, and its -y normal into +z
    pos = js["accessors"][js["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert pos["max"][1] == 10.0 and pos["min"][1] == 0.0
    blen, btype = struct.unpack_from("<II", data, 20 + jlen)
    assert btype == 0x004E4942 and 20 + jlen + 8 + blen == len(data)


def test_overture_box_is_a_closed_prism_with_the_height_the_record_gives():
    from kyfromabove_3d.stages.overture_boxes import extrude, height_of, height_source

    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], dtype=float)
    P, N, I = extrude(square, z0=2.0, h=30.0)
    assert len(I) // 3 == 4 * 2 + 2                          # four wall quads and a two-triangle top
    assert P[:, 2].min() == 2.0 and P[:, 2].max() == 32.0
    assert height_of({"height": 45.5, "num_floors": 3}) == (45.5, "height")
    assert height_of({"height": None, "num_floors": 4}) == (14.0, "floors")
    assert height_of({"height": None, "num_floors": None})[1] == "default"
    rec = {"height": 21.9, "datasets": ["OpenStreetMap", "Microsoft ML Buildings"], "props": ["", "/properties/height"]}
    assert height_source(rec) == "Microsoft ML Buildings"
    assert height_source({"height": 156.0, "datasets": ["OpenStreetMap"], "props": [""]}) == "OpenStreetMap"


def test_overture_boxes_stand_on_the_ground_in_the_frame(tmp_path):
    """lidar_ground_z is an absolute elevation; the box must be lowered by
    the frame's origin height, or it floats (every downtown box did, by
    140 m, until this test)."""
    from kyfromabove_3d.stages.overture_boxes import build

    frame = tmp_path / "frame"; frame.mkdir()
    (frame / "coordinates.json").write_text(json.dumps({"origin_source_xyz": [4918763.6, 3981088.7, 455.0]}))
    (frame / "wall_planes.json").write_text(json.dumps({"planes": [
        {"face": 0, "p0": [0, 0], "p1": [10, 0], "normal": [0, -1], "z_lo": 1.5, "z_hi": 30.0}]}))
    ground_abs = 455.0 * 0.3048006096012192 + 2.0                 # 2 m above the origin
    wkt = "POLYGON ((-85.7579 38.2562, -85.7575 38.2562, -85.7575 38.2565, -85.7579 38.2565, -85.7579 38.2562))"
    (tmp_path / "b.json").write_text(json.dumps({"release": "t", "buildings": [
        {"id": "x", "name": "x", "height": None, "num_floors": None, "wkt": wkt, "datasets": [], "props": [],
         "lidar_ground_z": ground_abs, "lidar_height_m": 20.0}]}))
    build(tmp_path / "b.json", frame, tmp_path / "out", [], None, heights="lidar")
    data = (tmp_path / "out" / "boxes.glb").read_bytes()
    jlen, = struct.unpack_from("<I", data, 12)
    js = json.loads(data[20:20 + jlen])
    pos = js["accessors"][js["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert abs(pos["min"][1] - 2.0) < 1e-3 and abs(pos["max"][1] - 22.0) < 1e-3


def _field(fn, x0=0, x1=40, y0=0, y1=30, cell=1.0):
    xs = np.arange(x0 + cell / 2, x1, cell); ys = np.arange(y0 + cell / 2, y1, cell)
    X, Y = np.meshgrid(xs, ys)
    Z = fn(X, Y).astype(float)
    return xs, ys, Z, np.isfinite(Z)


def test_a_wall_band_only_stands_where_there_is_roof_behind_it():
    """A 40 m edge along y = 0 (wall facing -y); the tower behind it (y > 0)
    is 60 m tall only between x = 14 and x = 26, the rest is a 12 m podium.
    A band at 40 m survives only over the tower; a band at 10 m spans the
    edge."""
    from kyfromabove_3d.stages.build_mesh import RoofField, _runs

    xs, ys, Z, valid = _field(lambda X, Y: np.where((X > 14) & (X < 26), 60.0, 12.0))
    roof = RoofField(xs, ys, Z, valid)
    plane = _plane()                                             # p0 (0,0) -> p1 (40,0), normal -y
    hi = _band(40, 45, 0.0); lo = _band(5, 10, 0.0)
    runs_hi = _runs(plane, hi, roof); runs_lo = _runs(plane, lo, roof)
    assert len(runs_hi) == 1 and abs(runs_hi[0][0] - 14) < 1.6 and abs(runs_hi[0][1] - 26) < 1.6
    assert len(runs_lo) == 1 and runs_lo[0][0] < 1 and runs_lo[0][1] > 39
    # no roof at all behind: nothing
    xs2, ys2, Z2, valid2 = _field(lambda X, Y: np.full(X.shape, np.nan))
    assert _runs(plane, hi, RoofField(xs2, ys2, Z2, valid2)) == []


def test_edge_top_profile_steps_from_podium_to_tower():
    from kyfromabove_3d.stages.build_mesh import RoofField

    xs, ys, Z, valid = _field(lambda X, Y: np.where(X > 20, 60.0, 12.0))
    roof = RoofField(xs, ys, Z, valid)
    edge = WallPlane(face=0, p0=np.array([0.0, 0.0]), p1=np.array([40.0, 0.0]), normal=np.array([0.0, -1.0]), z_lo=0.0, z_hi=60.0)
    st = roof.top_along(edge, within_m=2.0, step_m=4.0)
    heights = [z for _, _, z in st]
    assert abs(heights[0] - 12.0) < 1e-9 and abs(heights[-1] - 60.0) < 1e-9
    assert all(abs(z - 12.0) < 1e-9 for _, b, z in st if b <= 20) and all(abs(z - 60.0) < 1e-9 for a, _, z in st if a >= 20)


def test_wall_geometry_sloped_ground_draping():
    from kyfromabove_3d.features import GroundField

    # Ring: 4 vertices (square from 0 to 40)
    # Vertex 0: z=1.0, Vertex 1: z=3.0 (slope of 2m along 40m edge)
    ring = np.array([[0, 0], [40, 0], [40, 40], [0, 40], [0, 0]], dtype=float)
    vertex_z = np.array([1.0, 3.0, 2.5, 0.5, 1.0], dtype=float)
    gf = GroundField(vertex_z=vertex_z, ring_m=ring)

    bands = [_band(0, 10, 0.0), _band(10, 20, 0.0)]
    quads, _ = wall_geometry(_plane(), bands, ground_field=gf, edge_face=0)

    walls = [q for q in quads if q[3] == "wall"]
    assert len(walls) == 2

    # Bottom band (z_lo == 0) should be draped to ground:
    # Corner 0 at s=0 -> za = 1.0; Corner 1 at s=40 -> zc = 3.0
    bot_corners = walls[0][0]
    assert np.isclose(bot_corners[0, 2], 1.0)
    assert np.isclose(bot_corners[1, 2], 3.0)
    # Top of bottom band remains flat at z_hi = 10
    assert np.isclose(bot_corners[2, 2], 10.0)
    assert np.isclose(bot_corners[3, 2], 10.0)

    # Upper band (z_lo == 10, z_hi == 20) remains completely flat
    top_corners = walls[1][0]
    assert np.allclose(top_corners[:2, 2], 10.0)
    assert np.allclose(top_corners[2:, 2], 20.0)


def test_wall_geometry_upper_level_remains_flat():
    from kyfromabove_3d.features import GroundField, LEVEL_FACE_STRIDE

    ring = np.array([[0, 0], [40, 0], [40, 40], [0, 40], [0, 0]], dtype=float)
    vertex_z = np.array([1.0, 3.0, 2.5, 0.5, 1.0], dtype=float)
    gf = GroundField(vertex_z=vertex_z, ring_m=ring)

    bands = [_band(25, 50, 0.0)]
    # Upper level face: edge_face = LEVEL_FACE_STRIDE (e.g. 100)
    quads, _ = wall_geometry(_plane(), bands, ground_field=gf, edge_face=LEVEL_FACE_STRIDE)

    walls = [q for q in quads if q[3] == "wall"]
    assert len(walls) == 1
    corners = walls[0][0]
    # Baseline must remain strictly horizontal at z_lo = 25.0
    assert np.allclose(corners[:2, 2], 25.0)
    assert np.allclose(corners[2:, 2], 50.0)



def test_an_open_band_gets_its_back_wall_and_a_verified_plane_keeps_bands_over_blank_field():
    from kyfromabove_3d.stages.build_mesh import RoofField, _runs, wall_geometry
    from kyfromabove_3d.features import VerifiedWallPlane

    # a colonnade band between two observed bands: the wall is drawn 6 m back
    bands = [_band(0, 5, 0.0), FaceBand(5, 10, 0.2, 100, 0.0, 4.0, 36.0, "open", back_offset_m=-6.0), _band(10, 15, 0.0)]
    quads, _ = wall_geometry(_plane(), bands)
    walls = [q for q in quads if q[3] == "wall"]
    assert len(walls) == 3
    recess = walls[1][0]
    assert np.allclose(recess[:, 1], 6.0) and 5.0 <= recess[:, 2].min() and recess[:, 2].max() <= 10.0
    assert sum(1 for q in quads if q[3] == "ledge") == 2
    # an open band with no back surface extent stays a hole
    bands2 = [_band(0, 5, 0.0), FaceBand(5, 10, 0.2, 100, 0.0, 0.0, 0.0, "open", back_offset_m=-6.0)]
    quads2, _ = wall_geometry(_plane(), bands2)
    assert len([q for q in quads2 if q[3] == "wall"]) == 1
    # a verified plane keeps its band where the field has no cells at all
    xs, ys, Z, valid = _field(lambda X, Y: np.full(X.shape, np.nan))
    roof = RoofField(xs, ys, Z, valid)
    plain = _plane()
    verified = VerifiedWallPlane(face=0, p0=plain.p0, p1=plain.p1, normal=plain.normal, z_lo=0.0, z_hi=30.0, z_top=30.0)
    assert _runs(plain, _band(20, 25, 0.0), roof) == []
    assert len(_runs(verified, _band(20, 25, 0.0), roof)) == 1


def test_roof_smoothing_drops_a_one_cell_spike_but_keeps_a_step():
    from kyfromabove_3d.stages.build_mesh import roof_heightfield, roof_mesh
    from shapely.geometry import Polygon

    rng = np.random.default_rng(1)
    poly = Polygon([(0, 0), (30, 0), (30, 20), (0, 20)])
    xs = rng.uniform(0, 30, 30000); ys = rng.uniform(0, 20, 30000)
    z = np.where(xs > 20, 30.0, 12.0) + rng.normal(0, 0.03, 30000)
    spike = (np.abs(xs - 8) < 0.3) & (np.abs(ys - 10) < 0.3)
    z[spike] = 40.0                                          # a mast, one cell wide
    xyz = np.column_stack([xs, ys, z])
    gx, gy, Z, valid = roof_heightfield(xyz, poly, ground_z=0.0, cell=1.0)
    assert abs(Z[valid & (np.meshgrid(gx, gy)[0] < 18)].max() - 12.0) < 0.5   # the spike is smoothed away
    verts, flat, steep = roof_mesh(gx, gy, Z, valid, wall_quads=[])
    cen = verts[steep].mean(axis=1)
    assert len(steep) > 0 and np.all(np.abs(cen[:, 0] - 20.0) < 1.6)       # only the 18 m step is steep


def _photo_like(h=512, w=512):
    """A texture with photograph-like content: gradients, brick lines, noise."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([(xx * 255 // w), (yy * 255 // h), ((xx + yy) * 255 // (w + h))], axis=-1).astype(np.float64)
    img[(yy % 24) < 2] *= 0.5
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _glb_json(path):
    data = path.read_bytes()
    jlen, _ = struct.unpack_from("<II", data, 12)
    return json.loads(data[20:20 + jlen]), data


def _one_textured_glb(tmp_path, fmt, tex):
    glb = GlbWriter(fmt)
    mat = glb.material_texture(tex, "face_0")
    quads, _ = wall_geometry(_plane(), [_band(0, 10, 0.0)])
    P, N, T, I = quads_to_arrays(quads)
    glb.primitive(P, N, T, I, mat)
    path = tmp_path / f"b_{fmt}.glb"
    glb.write(path, {"name": "t"})
    return path


def test_jpeg_and_webp_textures_are_much_smaller_than_png(tmp_path):
    tex = _photo_like()
    sizes = {}
    for fmt in ("png", "jpeg", "webp"):
        js, data = _glb_json(_one_textured_glb(tmp_path, fmt, tex))
        sizes[fmt] = len(data)
        mime = js["images"][0]["mimeType"]
        if fmt == "png":
            assert mime == "image/png" and "extensionsRequired" not in js
        elif fmt == "jpeg":
            assert mime == "image/jpeg" and "extensionsRequired" not in js and "source" in js["textures"][0]
        else:
            assert mime == "image/webp"
            assert js["extensionsRequired"] == ["EXT_texture_webp"] and "EXT_texture_webp" in js["extensionsUsed"]
            assert js["textures"][0]["extensions"]["EXT_texture_webp"]["source"] == 0 and "source" not in js["textures"][0]
    assert sizes["jpeg"] * 3 < sizes["png"]
    assert sizes["webp"] * 3 < sizes["png"]


def test_ktx2_textures_are_basis_with_the_extension_required(tmp_path):
    import shutil
    import pytest
    if not shutil.which("basisu"):
        pytest.skip("basisu (brew install basis_universal) not on PATH")
    tex = _photo_like(300, 202)                           # not multiples of 4: padded for the encoder
    js, data = _glb_json(_one_textured_glb(tmp_path, "ktx2", tex))
    assert js["images"][0]["mimeType"] == "image/ktx2"
    assert js["extensionsRequired"] == ["KHR_texture_basisu"] and "KHR_texture_basisu" in js["extensionsUsed"]
    t = js["textures"][0]
    assert t["extensions"]["KHR_texture_basisu"]["source"] == 0 and "source" not in t
    view = js["bufferViews"][js["images"][0]["bufferView"]]
    bin_start = 20 + len(data[20:].split(b"BIN\0", 1)[0]) + 4
    img = data[bin_start + view["byteOffset"]: bin_start + view["byteOffset"] + view["byteLength"]]
    assert img[:12] == b"\xabKTX 20\xbb\r\n\x1a\n"       # the KTX2 magic
    # KTX2 header: pixelWidth at byte 20, pixelHeight at 24 -- padded to multiples of 4
    w, h = struct.unpack_from("<II", img, 20)
    assert (w, h) == (204, 300)
    _, png = _glb_json(_one_textured_glb(tmp_path, "png", tex))
    assert len(data) * 2 < len(png)


def test_glb_writer_refuses_ktx2_without_the_encoder(monkeypatch):
    import shutil
    import pytest
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="basisu"):
        GlbWriter("ktx2")


def test_texture_pages_and_roof_have_sides_in_multiples_of_four():
    from kyfromabove_3d.stages.build_mesh import TextureAtlas, _mult4
    assert [_mult4(n) for n in (1, 4, 5, 8, 4095)] == [4, 4, 8, 8, 4096]
    atlas = TextureAtlas(page_px=64)
    a = atlas.add(np.full((10, 13, 3), 1, dtype=np.uint8))
    atlas.add(np.full((7, 30, 3), 2, dtype=np.uint8))
    pages = atlas.pack()
    assert all(pg.shape[0] % 4 == 0 and pg.shape[1] % 4 == 0 for pg in pages)
    # the uv still lands on the image: the corner of the first image
    pi, uv = atlas.uv(a, np.array([[0.0, 1.0], [1.0, 0.0]]))
    H, W = pages[pi].shape[:2]
    assert pages[pi][int((1 - uv[0, 1]) * H), int(uv[0, 0] * W)][0] == 1
    assert pages[pi][min(H - 1, int((1 - uv[1, 1]) * H)) - 1, int(uv[1, 0] * W) - 1][0] == 1


def _arc_ring():
    """A 20 x 16 m block whose east end is a half circle of radius 8 m drawn as 12 edges."""
    th = np.linspace(-np.pi / 2, np.pi / 2, 13)
    arc = np.column_stack([20 + 8 * np.cos(th), 8 * np.sin(th)])
    return np.vstack([[0, -8], arc, [0, 8], [0, -8]])


def test_curved_wall_is_one_face_and_its_quads_lie_on_the_circle():
    from kyfromabove_3d.features import ArcWall, wall_planes
    planes = wall_planes(_arc_ring(), 0.0, 20.0, min_length_m=0.3, arcs=True)
    arcs = [p for p in planes if isinstance(p, ArcWall)]
    assert len(arcs) == 1 and len(planes) == 4                       # three straight edges and the arc
    arc = arcs[0]
    assert abs(arc.radius - 8.0) < 1e-6 and abs(abs(arc.turn) - np.pi) < 1e-6 and arc.convex
    assert abs(arc.length_m - 8 * np.pi) < 1e-6
    # a band the whole way along, in the unrolled frame; rolled back it hugs the circle
    band = FaceBand(z_lo=0.0, z_hi=12.0, offset_m=0.0, n_returns=500, spread_m=0.1,
                    along_lo=0.0, along_hi=arc.length_m, status="observed")
    quads, extent = wall_geometry(arc, [band])
    rolled = roll_quads(quads, arc, step_m=1.5)
    assert len(rolled) >= int(np.ceil(arc.length_m / 1.5))
    for corners, n, uv, kind in rolled:
        r = np.hypot(corners[:, 0] - 20.0, corners[:, 1])
        assert np.abs(r - 8.0).max() < 0.02                          # on the circle
        mid = corners[:, :2].mean(axis=0)
        assert n[:2] @ (mid - np.array([20.0, 0.0])) > 0.9 * 8       # normal points away from the centre
        assert uv is not None and 0.0 <= uv.min() and uv.max() <= 1.0 + 1e-9
    # unroll / roll agree, and a return 2 m outside the wall reads as offset +2
    s_, d_ = arc.local(np.array([[30.0, 0.0]]))
    assert abs(s_[0] - 4 * np.pi) < 1e-6 and abs(d_[0] - 2.0) < 1e-6
    assert np.abs(arc.world(s_, d_) - np.array([[30.0, 0.0]])).max() < 1e-9


def test_a_rounded_corner_is_not_a_curved_wall():
    from kyfromabove_3d.features import ArcWall, wall_planes
    th = np.linspace(0, np.pi / 2, 6)                                # a 1.5 m radius corner, 5 edges, 2.4 m of arc
    corner = np.column_stack([18.5 + 1.5 * np.cos(th), 8.5 + 1.5 * np.sin(th)])
    ring = np.vstack([[0, 0], [20, 0], corner, [0, 10], [0, 0]])
    planes = wall_planes(ring, 0.0, 10.0, min_length_m=0.3, arcs=True)
    assert not any(isinstance(p, ArcWall) for p in planes)


def test_roof_field_queries_follow_a_curved_wall():
    from kyfromabove_3d.features import wall_planes, ArcWall
    from kyfromabove_3d.stages.build_mesh import RoofField
    # roof only inside the circle (r < 8) at 12 m
    xs, ys, Z, valid = _field(lambda X, Y: np.where(np.hypot(X - 20, Y) < 7.5, 12.0, np.nan), y0=-15, y1=15)
    roof = RoofField(xs, ys, Z, valid)
    arc = [p for p in wall_planes(_arc_ring(), 0.0, 20.0, min_length_m=0.3, arcs=True) if isinstance(p, ArcWall)][0]
    st, ok = roof.behind(arc, 0.0, 10.0, 0.0, arc.length_m)
    assert ok.all()                                                   # roof behind every station of the arc
    st, ok = roof.behind(arc, 0.0, 14.0, 0.0, arc.length_m)
    assert not ok.any()                                               # nothing at 14 m
    tops = [z for _, _, z in roof.top_along(arc, within_m=2.0, step_m=3.0)]
    assert all(abs(z - 12.0) < 1e-9 for z in tops)


def test_roof_texture_from_cloud():
    from kyfromabove_3d.stages.build_mesh import roof_texture_from_cloud
    # 20x20 m building with roof returns at z=10.0
    poly = Polygon([(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)])
    xs = np.linspace(-5, 25, 31)
    ys = np.linspace(-5, 25, 31)
    X, Y = np.meshgrid(xs, ys)
    Z = np.full(X.shape, 10.0)
    valid = (X >= 0) & (X <= 20) & (Y >= 0) & (Y <= 20)

    # Synthetic point cloud on roof
    gx, gy = np.meshgrid(np.linspace(1, 19, 19), np.linspace(1, 19, 19))
    cxyz = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 10.0)])
    crgb = np.full((len(cxyz), 3), [180, 180, 180], dtype=np.uint8)

    tex, bbox, n_pts, n_in = roof_texture_from_cloud(xs, ys, Z, valid, poly, cxyz, crgb, texel_m=0.5)
    assert tex is not None
    assert tex.shape[0] > 0 and tex.shape[1] > 0
    assert n_pts == len(cxyz)
    assert bbox == (0.0, 0.0, 20.0, 20.0)
    # Check that pixels are textured with the grey color
    assert np.allclose(tex.mean(axis=(0, 1)), 180.0, atol=5.0)


def test_roof_sloped_fraction_tells_a_gable_from_a_flat_roof_with_a_step():
    from kyfromabove_3d.stages.build_mesh import roof_sloped_fraction
    cell = 0.75
    x = np.arange(0, 30, cell)
    X, Y = np.meshgrid(x, x)
    valid = np.ones(X.shape, dtype=bool)
    flat = np.full(X.shape, 12.0)
    assert roof_sloped_fraction(flat, valid, cell) == 0.0
    # a 30-degree gable: every interior cell slopes
    gable = 12.0 + np.tan(np.radians(30)) * (15.0 - np.abs(X - 15.0))
    assert roof_sloped_fraction(gable, valid, cell) > 0.9
    # two flat roofs 20 m apart: the step is a wall, not a slope
    stepped = np.where(X < 15.0, 12.0, 32.0)
    assert roof_sloped_fraction(stepped, valid, cell) < 0.05
    # the step as the height field draws it -- a ramp two cells wide -- and
    # a 2 m plant room: still flat roofs
    ramped = np.interp(X, [0.0, 14.25, 15.75, 30.0], [12.0, 12.0, 15.5, 15.5])
    ramped = np.where((np.abs(X - 22) < 1.5) & (np.abs(Y - 10) < 1.5), ramped + 2.0, ramped)
    assert roof_sloped_fraction(ramped, valid, cell) < 0.05


def _face_points(n, y, rgb, seed):
    rng = np.random.default_rng(seed)
    xyz = np.column_stack([rng.uniform(1, 39, n), np.full(n, y), rng.uniform(1, 9, n)])
    return xyz, np.tile(np.array(rgb, dtype=np.uint8), (n, 1))


def test_face_texture_never_borrows_a_facade_in_front_of_the_band():
    from kyfromabove_3d.stages.build_mesh import face_texture
    band = [_band(0, 10, 0.0)]                        # outward is -y
    own, own_rgb = _face_points(10, 0.0, (120, 120, 120), 1)
    front, front_rgb = _face_points(600, -3.0, (200, 40, 40), 2)   # a neighbour 3 m out
    behind, behind_rgb = _face_points(600, 1.0, (40, 40, 200), 3)  # the wall 1 m in
    extent = (0.0, 10.0, 40.0)

    _, n = face_texture(_plane(), band, extent, np.vstack([own, front]), np.vstack([own_rgb, front_rgb]), 0.5)
    assert n == 10                                    # the neighbour is not painted on

    xyz, rgb = np.vstack([own, behind]), np.vstack([own_rgb, behind_rgb])
    _, n = face_texture(_plane(), band, extent, xyz, rgb, 0.5)
    assert n == 600                                   # the wall just behind is taken instead
    _, n = face_texture(_plane(), band, extent, xyz, rgb, 0.5, party=True)
    assert n == 10                                    # but never on a party wall


def test_a_lone_point_stays_a_point_and_a_sparse_raster_still_fills():
    """The small-gap fill bridges gaps between seen texels; round a lone point
    it must not grow a disc (the round dots on the Omni's walls)."""
    from kyfromabove_3d.facade_fill import NEAR, PLAIN, SEEN
    from kyfromabove_3d.mesh.textures import _fill_holes
    img = np.zeros((60, 60, 3)); cnt = np.zeros((60, 60))
    img[30, 30] = (40, 40, 40); cnt[30, 30] = 1                     # one dark point on an unseen wall
    img[:, :5] = 200; cnt[:, :5] = 1                                 # a seen strip, for the median
    t = _fill_holes(img, cnt)
    src = t.source[::-1]                                             # back to the input's row order
    assert src[30, 30] == SEEN
    assert (src[25:36, 25:36] != NEAR).all()                         # no disc round it
    assert (src[25:36, 25:36][src[25:36, 25:36] != SEEN] == PLAIN).all()
    # a regular raster of points every 3 texels: every gap is between -> filled
    img2 = np.zeros((30, 30, 3)); cnt2 = np.zeros((30, 30))
    img2[::3, ::3] = 120; cnt2[::3, ::3] = 1
    t2 = _fill_holes(img2, cnt2)
    assert (t2.source[::-1][:28, :28] != PLAIN).all()                # within the raster's extent


def test_roof_field_keeps_a_rooftop_structure_and_drops_a_neighbours_spill():
    """Above the building's top, returns along the edge are a neighbour's wall;
    returns inside are something standing on the roof."""
    from kyfromabove_3d.stages.build_mesh import roof_heightfield
    rng = np.random.default_rng(4)
    poly = Polygon([(0, 0), (30, 0), (30, 30), (0, 30)])
    roof = np.column_stack([rng.uniform(0, 30, 9000), rng.uniform(0, 30, 9000), rng.normal(70.0, 0.2, 9000)])
    penthouse = np.column_stack([rng.uniform(13, 18, 800), rng.uniform(13, 18, 800), rng.normal(85.0, 0.2, 800)])
    spill = np.column_stack([rng.uniform(29.5, 30, 600), rng.uniform(0, 30, 600), rng.uniform(75, 160, 600)])
    xs, ys, Z, valid = roof_heightfield(np.vstack([roof, penthouse, spill]), poly, 0.0, 1.0, max_z=71.5)
    X, Y = np.meshgrid(xs, ys)
    mid = valid & (abs(X - 15.5) < 2) & (abs(Y - 15.5) < 2)
    edge = valid & (X > 28)
    assert np.nanmedian(Z[mid]) > 84                       # the penthouse is kept
    assert np.nanmax(Z[edge]) < 72                         # the neighbour's wall is not


def test_an_open_band_is_a_recess_only_under_a_solid_band():
    """A colonnade carries floors above it; an open band at the top of a face
    is the roof seen over the parapet and must not stand a wall on the roof."""
    colonnade = [_band(0, 5, None, status="open", back_offset_m=-3.0), _band(5, 30, 0.0)]
    quads, _ = wall_geometry(_plane(), colonnade)
    assert any(abs(q[0][:, 1].mean() - 3.0) < 1e-6 for q in quads if q[3] == "wall")   # drawn 3 m back
    roof_seen = [_band(0, 25, 0.0), _band(25, 30, None, status="open", back_offset_m=-5.0)]
    quads, extent = wall_geometry(_plane(), roof_seen)
    assert all(q[0][:, 2].max() <= 25.0 + 1e-6 for q in quads)                       # nothing above the wall
    assert extent[1] == 25.0


def test_gap_fill_on_a_texture_narrower_than_its_reach():
    """A prism face two texels wide: the 10-texel reach runs off its edge."""
    from kyfromabove_3d.stages.build_mesh import _fill_holes
    img = np.zeros((28, 2, 3)); cnt = np.zeros((28, 2))
    img[0] = 100; cnt[0] = 1; img[5] = 100; cnt[5] = 1
    t = _fill_holes(img, cnt, max_fill_px=10)
    assert t.shape == (28, 2, 3)


def test_on_a_wall_specks_are_dropped_solid_patches_grow_and_blank_takes_its_floors_colour():
    """Walls: a scattered speck is not trusted (bridging specks drew lines,
    growing them drew dots); a solidly seen patch grows its colour up to the
    reach; unseen wall takes the median of its own floor, not the face's."""
    from kyfromabove_3d.facade_fill import NEAR, PLAIN, SEEN
    from kyfromabove_3d.mesh.textures import _fill_holes
    H, W = 60, 80
    img = np.zeros((H, W, 3)); cnt = np.zeros((H, W))
    img[:30, :30] = 60; cnt[:30, :30] = 1                            # lower floors, seen, in shadow
    img[30:, :30] = 210; cnt[30:, :30] = 1                           # upper floors, seen, in sun
    for y, x in ((10, 60), (45, 55), (50, 70)):                      # lone specks on the unseen part
        img[y, x] = 0; cnt[y, x] = 1
    t = _fill_holes(img, cnt, max_fill_px=10, solid_px=7)
    src, out = t.source[::-1], np.asarray(t)[::-1]                   # back to the input's row order
    assert (src[:, :30] == SEEN).all()
    assert src[10, 60] == PLAIN and src[45, 55] == PLAIN             # specks dropped, and no disc round them
    assert (src[:, 30:40] == NEAR).all() and (src[:, 45:] == PLAIN).all()   # grown 10 texels, no further
    assert out[5, 70].tolist() == [60, 60, 60]                       # blank low down: the lower floors' colour
    assert out[55, 70].tolist() == [210, 210, 210]                   # blank high up: the upper floors'


def test_a_small_seen_blob_grows_no_ring_and_a_speckled_floor_keeps_the_specks_colour():
    from kyfromabove_3d.facade_fill import NEAR, PLAIN, SEEN
    from kyfromabove_3d.mesh.textures import _fill_holes
    H, W = 80, 80
    img = np.zeros((H, W, 3)); cnt = np.zeros((H, W))
    img[:30, :] = 60; cnt[:30, :] = 1                                # lower floors seen solidly, in shadow
    img[40:45, 40:45] = 90; cnt[40:45, 40:45] = 1                    # a blob under 2 m across
    img[60::4, ::4] = 230; cnt[60::4, ::4] = 1                       # the top floors: specks only, pale
    t = _fill_holes(img, cnt, max_fill_px=10, solid_px=7)
    src, out = t.source[::-1], np.asarray(t)[::-1]
    assert (src[40:45, 40:45] == SEEN).all()
    assert (src[42, 45:50] == PLAIN).all() and (src[45:50, 42] == PLAIN).all()   # no ring round the blob
    assert (src[64:, :] == PLAIN).all() and (src[:40] != NEAR).any() # the specks are dropped...
    assert out[75, 2].tolist() == [230, 230, 230]                    # ...but their floor takes their colour


def test_an_edge_reads_the_roof_on_its_own_side_only():
    """A tower wing's edge stands on its podium: the podium's roof outside
    the edge must not pull the wall's height down (the Omni's south wing)."""
    from kyfromabove_3d.stages.build_mesh import RoofField
    xs, ys, Z, valid = _field(lambda X, Y: np.where(Y > 0, 57.0, 17.0), y0=-10)   # wing north of y=0, podium south
    roof = RoofField(xs, ys, Z, valid)
    edge = WallPlane(face=1000, p0=np.array([0.0, 0.0]), p1=np.array([40.0, 0.0]), normal=np.array([0.0, -1.0]),
                     z_lo=17.0, z_hi=57.0)
    assert all(abs(z - 57.0) < 1e-9 for _, _, z in roof.top_along(edge, within_m=2.0, step_m=3.0))


def test_a_coarse_roof_field_still_gives_every_stretch_of_an_edge_a_height():
    """1.5 m cells (sparse LiDAR): the inside strip must reach far enough to
    hold a cell in every 3 m stretch, or that stretch gets no wall."""
    from kyfromabove_3d.stages.build_mesh import RoofField
    xs, ys, Z, valid = _field(lambda X, Y: np.full(X.shape, 24.0), y0=-0.6, cell=1.5)   # one row 0.5-2.5 m in
    roof = RoofField(xs, ys, Z, valid)
    edge = WallPlane(face=16, p0=np.array([0.0, 0.0]), p1=np.array([19.0, 0.0]), normal=np.array([0.0, -1.0]),
                     z_lo=0.0, z_hi=24.0)
    tops = [z for _, _, z in roof.top_along(edge, within_m=2.0, step_m=3.0)]
    assert all(np.isfinite(z) and abs(z - 24.0) < 1e-9 for z in tops)
