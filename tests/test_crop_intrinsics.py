"""Crop geometry, windowing and validity-mask behaviour."""

import numpy as np
import pytest

from kyfromabove_3d.cameras import VirtualObliqueCamera, photogrammetric_omega_phi_kappa_to_rcw
from kyfromabove_3d.coordinates import LocalCoordinateFrame
from kyfromabove_3d.crops import BoundingVolume3D, BuildingCropExtractor
from kyfromabove_3d.imagery import ImageWindow


class FakeReader:
    """Stands in for a COG: everything inside the sensor is 200, outside is padded."""

    def __init__(self, width=14144, height=10560):
        self.width, self.height = width, height

    def read_window(self, x0, y0, w, h):
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        valid = np.zeros((h, w), dtype=bool)
        ix0, iy0 = max(x0, 0), max(y0, 0)
        ix1, iy1 = min(x0 + w, self.width), min(y0 + h, self.height)
        if ix1 > ix0 and iy1 > iy0:
            rgb[iy0 - y0: iy1 - y0, ix0 - x0: ix1 - x0] = 200
            valid[iy0 - y0: iy1 - y0, ix0 - x0: ix1 - x0] = True
        return ImageWindow(rgb=rgb, valid=valid, x0=x0, y0=y0)


@pytest.fixture
def test_setup():
    coord_frame = LocalCoordinateFrame(origin_source_xyz=(5725184.6, 4025570.9, 600.0))
    camera = VirtualObliqueCamera(
        frame_id="crop_test_cam", camera_name="TestCam",
        camera_center_local_m=np.array([0.0, -500.0, 500.0]),
        R_cw=photogrammetric_omega_phi_kappa_to_rcw(45.0, 0.0, 0.0),
        width_px=14144, height_px=10560,
        fx_px=32813.83, fy_px=32813.83, cx_px=7072.0, cy_px=5280.0,
    )
    volume = BoundingVolume3D(-25.0, 25.0, -25.0, 25.0, -2.0, 10.0)
    return coord_frame, camera, volume


def test_bounding_volume_corners(test_setup):
    _, _, vol = test_setup
    corners = vol.get_8_corners()
    assert corners.shape == (8, 3)
    assert np.isclose(np.min(corners[:, 0]), vol.min_x_m)
    assert np.isclose(np.max(corners[:, 2]), vol.max_z_m)


def test_crop_intrinsics_projection_consistency(test_setup):
    """A point must land at the same place through the crop camera as the full frame."""
    coord_frame, camera, vol = test_setup
    extractor = BuildingCropExtractor(coord_frame)
    x0, y0, w, h = extractor.compute_crop_extent(camera, vol, padding_px=50)

    pt = np.array([5.0, -10.0, 4.0])
    uv_full, _, _ = camera.project(pt)
    crop_cam = VirtualObliqueCamera(
        frame_id="crop_view", camera_name=camera.camera_name,
        camera_center_local_m=camera.camera_center_local_m.copy(), R_cw=camera.R_cw.copy(),
        width_px=w, height_px=h, fx_px=camera.fx_px, fy_px=camera.fy_px,
        cx_px=camera.cx_px - x0, cy_px=camera.cy_px - y0,
    )
    uv_crop, _, _ = crop_cam.project(pt)
    assert np.isclose(uv_crop[0], uv_full[0] - x0, atol=1e-6)
    assert np.isclose(uv_crop[1], uv_full[1] - y0, atol=1e-6)


def test_crop_extent_is_not_clamped(test_setup):
    """The ideal window is reported honestly even when it leaves the sensor.

    Clamping preserved the declared geometry while changing the pixels, which is
    how two Phase 1 crops truncated the building and still reported a PASS.
    """
    coord_frame, camera, vol = test_setup
    edge_cam = VirtualObliqueCamera(
        frame_id="edge", camera_name="TestCam",
        camera_center_local_m=camera.camera_center_local_m.copy(), R_cw=camera.R_cw.copy(),
        width_px=camera.width_px, height_px=camera.height_px,
        fx_px=camera.fx_px, fy_px=camera.fy_px, cx_px=50.0, cy_px=40.0,
    )
    x0, y0, w, h = BuildingCropExtractor(coord_frame).compute_crop_extent(
        edge_cam, vol, padding_px=100
    )
    assert x0 < 0 or y0 < 0, "a window running off the sensor must report negative origin"
    assert w > 0 and h > 0


def test_too_small_fixed_size_raises(test_setup):
    """Requesting a crop that cannot hold the building must fail, not truncate."""
    coord_frame, camera, vol = test_setup
    extractor = BuildingCropExtractor(coord_frame)
    _, _, w, h = extractor.compute_crop_extent(camera, vol, padding_px=120)
    with pytest.raises(ValueError, match="cut off the building"):
        extractor.extract_crop(camera, FakeReader(), vol, padding_px=120,
                               fixed_square_size=max(w, h) // 2)


def test_validity_mask_marks_off_sensor_pixels(test_setup):
    """The mask must record real sensor coverage, not just non-black pixels.

    Every Phase 1 mask was uniformly 255 and carried no information.
    """
    coord_frame, camera, vol = test_setup
    edge_cam = VirtualObliqueCamera(
        frame_id="edge", camera_name="TestCam",
        camera_center_local_m=camera.camera_center_local_m.copy(), R_cw=camera.R_cw.copy(),
        width_px=camera.width_px, height_px=camera.height_px,
        fx_px=camera.fx_px, fy_px=camera.fy_px, cx_px=60.0, cy_px=60.0,
    )
    _, mask, spec = BuildingCropExtractor(coord_frame).extract_crop(
        edge_cam, FakeReader(), vol, padding_px=120, allow_truncation=True
    )
    assert (mask == 0).any(), "off-sensor region must be marked invalid"
    assert (mask == 255).any()
    assert spec.sensor_coverage_fraction < 1.0
    assert not spec.volume_contained
