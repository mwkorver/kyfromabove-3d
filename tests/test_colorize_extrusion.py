"""Unit tests for wall plane extrusion and tile deduplication."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add scripts directory to sys.path so colorize_building can be imported
scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from kyfromabove_3d.stages.colorize_building import extrude_verified_faces, write_ply
from kyfromabove_3d.features import WallPlane
from kyfromabove_3d.geometry import read_ply


def test_extrude_verified_faces_and_deduplication():
    # Nominal plane: face 0 along x from 0 to 30, y=0, normal [0, -1]
    plane = WallPlane(
        face=0,
        p0=np.array([0.0, 0.0]),
        p1=np.array([30.0, 0.0]),
        normal=np.array([0.0, -1.0]),
        z_lo=0.0,
        z_hi=40.0,
    )

    # 2 fitted tiles: at (5, 0, 10) and (20, 0, 20), tile_m=10
    tile_m = 10.0
    tiles_by_face = {
        0: [
            {"centroid": np.array([5.0, 0.0, 10.0]), "normal": np.array([0.0, -1.0, 0.0]), "rms": 0.02},
            {"centroid": np.array([20.0, 0.0, 20.0]), "normal": np.array([0.0, -1.0, 0.0]), "rms": 0.02},
        ]
    }

    ext_xyz, ext_n, ext_face, verified_planes = extrude_verified_faces(
        nominal_planes=[plane],
        tiles_by_face=tiles_by_face,
        spacing_m=1.0,
        tile_m=tile_m,
        max_offset_rms_m=0.30,
        max_normal_dev_deg=3.0,
        roof_z=40.0,
    )

    assert 0 in verified_planes
    assert len(ext_xyz) > 0
    assert len(ext_n) == len(ext_xyz)
    assert len(ext_face) == len(ext_xyz)
    assert (ext_face == 0).all()

    # Verify z_top is highest_z + tile_m/2 = 20 + 5 = 25
    assert abs(verified_planes[0].z_top - 25.0) < 1e-4

    # Verify that NO extruded points fall inside tile footprints:
    # Tile 0: x in (0, 10), z in (5, 15)
    in_tile0 = (np.abs(ext_xyz[:, 0] - 5.0) < tile_m / 2.0) & (np.abs(ext_xyz[:, 2] - 10.0) < tile_m / 2.0)
    assert not in_tile0.any(), "Extruded points must not duplicate Tile 0"

    # Tile 1: x in (15, 25), z in (15, 25)
    in_tile1 = (np.abs(ext_xyz[:, 0] - 20.0) < tile_m / 2.0) & (np.abs(ext_xyz[:, 2] - 20.0) < tile_m / 2.0)
    assert not in_tile1.any(), "Extruded points must not duplicate Tile 1"

    # But points outside tiles (e.g. z < 5 or gap between x 10 and 15) DO exist!
    gap_points = ext_xyz[(ext_xyz[:, 0] >= 10.0) & (ext_xyz[:, 0] <= 15.0)]
    assert len(gap_points) > 0


def test_extrude_non_planar_face_declines():
    plane = WallPlane(
        face=0,
        p0=np.array([0.0, 0.0]),
        p1=np.array([30.0, 0.0]),
        normal=np.array([0.0, -1.0]),
        z_lo=0.0,
        z_hi=40.0,
    )
    # Setback step of 1.0 m -> non-planar
    tiles_by_face = {
        0: [
            {"centroid": np.array([5.0, 0.0, 10.0]), "normal": np.array([0.0, -1.0, 0.0]), "rms": 0.02},
            {"centroid": np.array([20.0, 1.0, 10.0]), "normal": np.array([0.0, -1.0, 0.0]), "rms": 0.02},
        ]
    }
    ext_xyz, ext_n, ext_face, verified_planes = extrude_verified_faces(
        nominal_planes=[plane],
        tiles_by_face=tiles_by_face,
        spacing_m=1.0,
        tile_m=10.0,
        max_offset_rms_m=0.30,
        max_normal_dev_deg=3.0,
        roof_z=40.0,
    )
    assert len(verified_planes) == 0
    assert len(ext_xyz) == 0


def test_write_and_read_ply_with_surface_property(tmp_path):
    path = tmp_path / "test_surface.ply"
    xyz = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ], dtype=np.float32)
    rgb = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
    ], dtype=np.uint8)
    source = np.array([0, 1, 2], dtype=np.uint8)
    surface = np.array([0, 1, 3], dtype=np.uint8)

    write_ply(path, xyz, rgb, extra=source, extras={"surface": surface})

    cols = read_ply(path)
    assert "source" in cols
    assert "surface" in cols
    assert np.array_equal(cols["source"], source)
    assert np.array_equal(cols["surface"], surface)


def test_extrude_prism_completion():
    # Box with 4 faces:
    # Face 0 (length 10, y=0), Face 1 (length 30, x=10), Face 2 (length 10, y=30), Face 3 (length 30, x=0)
    planes = [
        WallPlane(0, np.array([0.0, 0.0]), np.array([10.0, 0.0]), np.array([0.0, -1.0]), 0.0, 40.0),
        WallPlane(1, np.array([10.0, 0.0]), np.array([10.0, 30.0]), np.array([1.0, 0.0]), 0.0, 40.0),
        WallPlane(2, np.array([10.0, 30.0]), np.array([0.0, 30.0]), np.array([0.0, 1.0]), 0.0, 40.0),
        WallPlane(3, np.array([0.0, 30.0]), np.array([0.0, 0.0]), np.array([-1.0, 0.0]), 0.0, 40.0),
    ]

    # Fitted tiles only on Face 1 and Face 3 (60 m / 80 m = 75% >= 60%)
    tiles_by_face = {
        1: [
            {"centroid": np.array([10.0, 5.0, 10.0]), "normal": np.array([1.0, 0.0, 0.0]), "rms": 0.02},
            {"centroid": np.array([10.0, 20.0, 25.0]), "normal": np.array([1.0, 0.0, 0.0]), "rms": 0.02},
        ],
        3: [
            {"centroid": np.array([0.0, 25.0, 10.0]), "normal": np.array([-1.0, 0.0, 0.0]), "rms": 0.02},
            {"centroid": np.array([0.0, 10.0, 25.0]), "normal": np.array([-1.0, 0.0, 0.0]), "rms": 0.02},
        ],
    }

    # When prism_completion=False: only faces 1 and 3 are extruded
    ext_xyz_v, _, ext_face_v, vp_v = extrude_verified_faces(
        nominal_planes=planes,
        tiles_by_face=tiles_by_face,
        spacing_m=1.0,
        tile_m=10.0,
        prism_completion=False,
    )
    assert set(vp_v.keys()) == {1, 3}
    assert set(np.unique(ext_face_v)) == {1, 3}

    # When prism_completion=True: all 4 faces (0, 1, 2, 3) are extruded
    ext_xyz_p, _, ext_face_p, vp_p = extrude_verified_faces(
        nominal_planes=planes,
        tiles_by_face=tiles_by_face,
        spacing_m=1.0,
        tile_m=10.0,
        prism_completion=True,
    )
    assert set(vp_p.keys()) == {0, 1, 2, 3}
    assert set(np.unique(ext_face_p)) == {0, 1, 2, 3}
    assert vp_p[0].inferred is True
    assert vp_p[2].inferred is True
    assert vp_p[1].inferred is False
    assert vp_p[3].inferred is False
    assert len(ext_xyz_p) > len(ext_xyz_v)


def test_write_and_read_laz_round_trips_colour_and_extras(tmp_path):
    from kyfromabove_3d.clouds import find_cloud, read_cloud
    path = tmp_path / "test_surface.laz"
    rng = np.random.default_rng(1)
    xyz = rng.uniform(-80, 80, (500, 3))
    rgb = rng.integers(0, 256, (500, 3)).astype(np.uint8)
    source = (np.arange(500) % 4).astype(np.uint8)
    surface = (np.arange(500) % 5).astype(np.uint8)

    write_ply(path, xyz, rgb, extra=source, extras={"surface": surface})

    cols = read_cloud(path)
    assert np.abs(np.column_stack([cols["x"], cols["y"], cols["z"]]) - xyz).max() < 0.0006   # 1 mm resolution
    for i, c in enumerate(("red", "green", "blue")):
        assert cols[c].dtype == np.uint8 and np.array_equal(cols[c], rgb[:, i])
    assert np.array_equal(cols["source"], source) and np.array_equal(cols["surface"], surface)
    # the finder prefers the laz, falls back to a ply, and names the laz when neither exists
    assert find_cloud(tmp_path, "test_surface") == path
    ply = tmp_path / "old.ply"
    write_ply(ply, xyz, rgb, extra=source, extras={"surface": surface})
    assert find_cloud(tmp_path, "old") == ply
    assert find_cloud(tmp_path, "none").suffix == ".laz"
    # and the two formats read the same columns
    old = read_cloud(ply)
    assert np.array_equal(old["red"], cols["red"]) and np.abs(old["x"] - cols["x"]).max() < 0.0006


def test_lidar_clip_as_laz_keeps_intensity_and_classification(tmp_path):
    from kyfromabove_3d.clouds import read_cloud, write_cloud
    xyz = np.array([[0.0, 0.0, 0.0], [1.5, -2.25, 30.125]])
    p = write_cloud(tmp_path / "lidar_local.laz", xyz, intensity=np.array([0, 40000]), classification=np.array([2, 6]))
    cols = read_cloud(p)
    assert cols["intensity"].tolist() == [0, 40000]           # 16-bit kept (the PLY clipped to 8)
    assert cols["classification"].tolist() == [2, 6] and "red" not in cols
    assert np.abs(cols["z"] - xyz[:, 2]).max() < 0.0006
