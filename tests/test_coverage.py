"""Unit tests for facade coverage grids and fillable deficit tracking."""

from __future__ import annotations

import numpy as np

from kyfromabove_3d.coverage import FaceCoverage, facade_coverage


def test_face_coverage_dataclass():
    along = np.array([0.0, 3.0, 6.0])
    up = np.array([0.0, 3.0, 6.0])
    n_lidar = np.array([[5, 0], [10, 2]])
    n_col = np.array([[2, 0], [5, 1]])
    n_uncol = np.array([[3, 0], [5, 1]])
    n_photo = np.array([[1, 0], [0, 1]])

    fc = FaceCoverage(
        face=0,
        bearing_deg=90.0,
        along_edges_m=along,
        up_edges_m=up,
        n_lidar=n_lidar,
        n_coloured=n_col,
        n_uncoloured=n_uncol,
        n_from_photo=n_photo,
    )

    # Fillable where uncoloured > 0
    assert fc.fillable[0, 0] is True or fc.fillable[0, 0] == 1
    assert fc.fillable[0, 1] is False or fc.fillable[0, 1] == 0
    assert fc.n_fillable_cells == 3

    # Unfillable where n_lidar == 0
    assert fc.unfillable[0, 1] is True or fc.unfillable[0, 1] == 1
    assert fc.unfillable[0, 0] is False or fc.unfillable[0, 0] == 0
    assert fc.n_unfillable_cells == 1

    d = fc.to_dict()
    assert d["face"] == 0
    assert d["bearing_deg"] == 90.0
    assert d["n_fillable_cells"] == 3


def test_facade_coverage_synthetic_box():
    # Square building footprint 20m x 20m
    ring = np.array([
        [0.0, 0.0],
        [20.0, 0.0],
        [20.0, 20.0],
        [0.0, 20.0],
        [0.0, 0.0],
    ])
    height_m = 15.0

    # Create synthetic resampled points on Face 0 (along y=0 from x=0 to 20)
    xs = np.linspace(1.0, 19.0, 20)
    zs = np.linspace(1.0, 14.0, 15)
    X, Z = np.meshgrid(xs, zs)
    resampled_xyz = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])

    # Half coloured, half uncoloured
    coloured_mask = resampled_xyz[:, 0] < 10.0
    from_photo_mask = resampled_xyz[:, 2] > 10.0

    # Lidar wall returns on all faces
    lidar_wall_xyz = resampled_xyz.copy()

    coverages = facade_coverage(
        resampled_xyz=resampled_xyz,
        coloured_mask=coloured_mask,
        from_photo_mask=from_photo_mask,
        lidar_wall_xyz=lidar_wall_xyz,
        ring_m=ring,
        height_m=height_m,
        cell_m=3.0,
        min_face_length_m=5.0,
    )

    assert len(coverages) == 4
    # Face 0 should have both coloured and uncoloured cells
    f0 = coverages[0]
    assert f0.n_fillable_cells > 0
    assert f0.n_coloured.sum() > 0
    assert f0.n_uncoloured.sum() > 0
    assert f0.n_photo_cells > 0


def test_facade_coverage_extrusion_and_verification():
    from kyfromabove_3d.features import VerifiedWallPlane

    ring = np.array([
        [0.0, 0.0],
        [20.0, 0.0],
        [20.0, 20.0],
        [0.0, 20.0],
        [0.0, 0.0],
    ])
    height_m = 15.0

    # Face 0 points: lower half (z < 6) are tiles (surface=1), upper half (z >= 6) are extruded (surface=3)
    xs = np.linspace(1.0, 19.0, 20)
    zs = np.linspace(1.0, 14.0, 15)
    X, Z = np.meshgrid(xs, zs)
    resampled_xyz = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])

    surface_mask = np.where(resampled_xyz[:, 2] < 6.0, 1, 3).astype(np.uint8)
    coloured_mask = np.ones(len(resampled_xyz), dtype=bool)
    from_photo_mask = resampled_xyz[:, 2] >= 10.0

    # LiDAR only exists on lower half (z < 6.0)
    lidar_wall_xyz = resampled_xyz[resampled_xyz[:, 2] < 6.0]

    vp = VerifiedWallPlane(
        face=0,
        p0=np.array([0.0, 0.0]),
        p1=np.array([20.0, 0.0]),
        normal=np.array([0.0, -1.0]),
        z_lo=0.0,
        z_hi=15.0,
        n_tiles=4,
        rms_m=0.05,
        max_normal_dev_deg=0.8,
        z_top=15.0,
        offset_m=0.0,
        yaw_dev_deg=0.1,
    )

    coverages = facade_coverage(
        resampled_xyz=resampled_xyz,
        coloured_mask=coloured_mask,
        from_photo_mask=from_photo_mask,
        lidar_wall_xyz=lidar_wall_xyz,
        ring_m=ring,
        height_m=height_m,
        cell_m=3.0,
        min_face_length_m=5.0,
        surface_mask=surface_mask,
        verified_planes={0: vp},
    )

    f0 = coverages[0]
    assert f0.plane_verified is True
    assert f0.verified_rms_m == 0.05
    assert f0.n_extruded_cells > 0
    assert f0.n_tile_cells > 0
    # Even though upper half has NO LiDAR returns, every cell with extruded points is fillable!
    assert f0.n_unfillable_cells == 0
    n_cells = (len(f0.along_edges_m) - 1) * (len(f0.up_edges_m) - 1)
    assert f0.n_fillable_cells == n_cells

    d = f0.to_dict()
    assert d["plane_verified"] is True
    assert d["n_extruded_cells"] == f0.n_extruded_cells
    assert d["n_tile_cells"] == f0.n_tile_cells
    assert d["verified_rms_m"] == 0.05

