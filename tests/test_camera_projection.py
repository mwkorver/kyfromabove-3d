"""Unit tests for VirtualObliqueCamera projection, unprojection, and rotation geometry."""

import numpy as np
import pytest
from kyfromabove_3d.cameras import (
    VirtualObliqueCamera,
    photogrammetric_omega_phi_kappa_to_rcw,
    rot_x,
    rot_y,
    rot_z,
)


@pytest.fixture
def sample_camera():
    """Create a synthetic calibrated oblique camera."""
    # Camera at (0, -100, 100) looking down and forward at origin
    C_local = np.array([0.0, -100.0, 100.0], dtype=np.float64)
    omega_deg = 45.0
    phi_deg = 0.0
    kappa_deg = 0.0
    R_cw = photogrammetric_omega_phi_kappa_to_rcw(omega_deg, phi_deg, kappa_deg)

    return VirtualObliqueCamera(
        frame_id="test_cam_001",
        camera_name="TestCam_v1",
        camera_center_local_m=C_local,
        R_cw=R_cw,
        width_px=14144,
        height_px=10560,
        fx_px=32813.83,
        fy_px=32813.83,
        cx_px=7072.0,
        cy_px=5280.0,
    )


def test_rotation_matrix_properties():
    """Verify orthonormality and positive determinant of photogrammetric rotation."""
    for omega in [-45.0, 0.0, 45.0]:
        for phi in [-30.0, 0.0, 30.0]:
            for kappa in [-180.0, -90.0, 0.0, 90.0, 180.0]:
                R = photogrammetric_omega_phi_kappa_to_rcw(omega, phi, kappa)
                # Check determinant is +1
                det = np.linalg.det(R)
                assert np.isclose(det, 1.0, atol=1e-12)
                # Check R @ R.T == I
                ident = R @ R.T
                assert np.allclose(ident, np.eye(3), atol=1e-12)


def test_optical_axis_projection(sample_camera):
    """A point directly along the optical axis must project to principal point (cx, cy)."""
    # Optical axis direction in world coordinates: R_wc @ [0, 0, 1]
    ray_dir_w = sample_camera.R_wc @ np.array([0.0, 0.0, 1.0])
    distance_m = 150.0
    point_on_axis = sample_camera.camera_center_local_m + distance_m * ray_dir_w

    uv, zc, valid = sample_camera.project(point_on_axis)
    assert valid
    assert np.isclose(zc, distance_m, atol=1e-9)
    assert np.isclose(uv[0], sample_camera.cx_px, atol=1e-6)
    assert np.isclose(uv[1], sample_camera.cy_px, atol=1e-6)


def test_behind_camera_rejection(sample_camera):
    """Points behind the camera (zc <= 0) must be flagged as invalid."""
    ray_dir_w = sample_camera.R_wc @ np.array([0.0, 0.0, 1.0])
    # Point behind camera
    point_behind = sample_camera.camera_center_local_m - 50.0 * ray_dir_w

    uv, zc, valid = sample_camera.project(point_behind)
    assert not valid
    assert zc < 0.0


def test_unproject_project_roundtrip(sample_camera):
    """Unprojecting image pixels to 3D rays and reprojecting must yield identical pixels."""
    test_pixels = np.array(
        [
            [7072.0, 5280.0],  # Center
            [2000.0, 3000.0],  # Top-left
            [12000.0, 8000.0],  # Bottom-right
            [5000.0, 5000.0],
            [10000.0, 4000.0],
        ],
        dtype=np.float64,
    )

    cam_center, ray_dirs_w = sample_camera.unproject(test_pixels)
    assert np.allclose(cam_center, sample_camera.camera_center_local_m)

    # Place 3D points at arbitrary distance 200m along rays
    pts_3d = cam_center + 200.0 * ray_dirs_w

    uv_reproj, zc, valid = sample_camera.project(pts_3d)
    assert np.all(valid)
    assert np.allclose(uv_reproj, test_pixels, atol=1e-6)


def test_rotation_convention_is_pinned():
    """Pin the omega/phi/kappa convention against silent regression.

    The orthonormality and optical-axis tests above pass for *any* rotation
    matrix, so they cannot tell a correct convention from a wrong one. This one
    states the convention explicitly:

        R_cw = diag(1, -1, -1) @ (Rx(omega) Ry(phi) Rz(kappa)).T

    Composing the same three rotations in the reverse order was the original
    bug. It measured 26 px of disagreement between consecutive exposures on the
    real imagery while every self-consistency test still passed.
    """
    flip = np.diag([1.0, -1.0, -1.0])
    for omega, phi, kappa in [(45.0, 0.6, 0.5), (-30.0, 10.0, -120.0), (0.0, 0.0, 0.0),
                              (44.96, 0.63, 0.61), (12.0, -7.5, 200.0)]:
        w, p, k = np.radians([omega, phi, kappa])
        expected = flip @ (rot_x(w) @ rot_y(p) @ rot_z(k)).T
        assert np.allclose(photogrammetric_omega_phi_kappa_to_rcw(omega, phi, kappa), expected)


def test_reversed_rotation_order_differs_materially():
    """The wrong order is not a harmless algebraic rearrangement.

    At omega ~ 45 degrees the two orderings point the optical axis degrees
    apart, which is why it moved the ground footprint by metres.
    """
    w, p, k = np.radians([44.96, 0.63, 0.61])
    flip = np.diag([1.0, -1.0, -1.0])
    right = flip @ (rot_x(w) @ rot_y(p) @ rot_z(k)).T
    wrong = flip @ (rot_x(-w) @ rot_y(-p) @ rot_z(-k))

    axis_right = right.T @ np.array([0.0, 0.0, 1.0])
    axis_wrong = wrong.T @ np.array([0.0, 0.0, 1.0])
    separation_deg = np.degrees(np.arccos(np.clip(axis_right @ axis_wrong, -1, 1)))
    assert separation_deg > 0.2

    # At this sensor's focal length that pointing error is ~136 px, about 9.4 m
    # on the ground -- far too large to be dismissed as a convention detail.
    focal_px = 32813.83
    assert focal_px * np.radians(separation_deg) > 50.0


def test_principal_point_y_offset_is_subtracted():
    """CamPpyMm is a photo-frame (+y up) offset; image rows increase downward.

    Determined by measurement: with the opposite sign a Left exposure yields
    zero usable correspondences against a validated Fwd exposure. Getting it
    backwards displaces the principal point by 3552 px on the side cameras.
    """
    from kyfromabove_3d.cameras import parse_camera_from_eo_record
    from kyfromabove_3d.coordinates import LocalCoordinateFrame

    record = {
        "ID": "test/Left_0001", "X": 5725184.6, "Y": 4025570.9, "Z": 5800.0,
        "Omega": 0.04, "Phi": -9.0, "Kappa": -18.29,
        "CamName": "Left_434S82525X818326-f120_v2",
        "CamWidthPx": 14144, "CamHeighPx": 10560,
        "CamFocalMm": 123.38, "CamCCDResU": 3.76,
        "CamPpxMm": 0.0, "CamPpyMm": 6.68,
    }
    cam = parse_camera_from_eo_record(
        record, LocalCoordinateFrame(origin_source_xyz=(5725184.6, 4025570.9, 600.0))
    )
    expected_offset = 6.68 / (3.76e-3)
    assert cam.cy_px == pytest.approx(10560 / 2.0 - expected_offset)
    assert cam.cx_px == pytest.approx(14144 / 2.0)
    assert expected_offset > 1700, "the offset is large enough that the sign matters"
