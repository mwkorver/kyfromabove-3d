"""Unit tests for coordinate conversions and local metric frame."""

import json
import numpy as np
import pytest
from kyfromabove_3d.coordinates import (
    METERS_PER_US_SURVEY_FOOT,
    US_SURVEY_FEET_PER_METER,
    LocalCoordinateFrame,
    meters_to_us_survey_feet,
    us_survey_feet_to_meters,
)


def test_us_survey_foot_constant():
    """Verify exact scale conversion ratio."""
    assert np.isclose(METERS_PER_US_SURVEY_FOOT, 1200.0 / 3937.0)
    assert np.isclose(US_SURVEY_FEET_PER_METER, 3937.0 / 1200.0)
    assert np.isclose(METERS_PER_US_SURVEY_FOOT * US_SURVEY_FEET_PER_METER, 1.0)


def test_scalar_roundtrip():
    """Test scalar forward and inverse conversions."""
    val_ft = 5725184.6
    val_m = us_survey_feet_to_meters(val_ft)
    val_ft_back = meters_to_us_survey_feet(val_m)
    assert np.abs(val_ft - val_ft_back) < 1e-9


def test_array_roundtrip():
    """Test array forward and inverse conversions."""
    pts_ft = np.array(
        [
            [5725000.0, 4025000.0, 600.0],
            [5725500.0, 4025500.0, 650.0],
            [5726000.0, 4026000.0, 700.0],
        ],
        dtype=np.float64,
    )
    pts_m = us_survey_feet_to_meters(pts_ft)
    pts_ft_back = meters_to_us_survey_feet(pts_m)
    assert np.all(np.abs(pts_ft - pts_ft_back) < 1e-9)


def test_local_coordinate_frame():
    """Test LocalCoordinateFrame transformation and origin centering."""
    origin_xyz = (5725184.6, 4025570.9, 600.0)
    frame = LocalCoordinateFrame(origin_source_xyz=origin_xyz)

    # Origin in local frame should be exactly (0, 0, 0)
    local_origin = frame.to_local(origin_xyz)
    assert np.allclose(local_origin, [0.0, 0.0, 0.0], atol=1e-9)

    # Round-trip origin
    src_origin_back = frame.to_source(local_origin)
    assert np.allclose(src_origin_back, origin_xyz, atol=1e-9)

    # Arbitrary test points
    test_pts_source = np.array(
        [
            [5725284.6, 4025570.9, 600.0],  # +100 ft East
            [5725184.6, 4025670.9, 600.0],  # +100 ft North
            [5725184.6, 4025570.9, 700.0],  # +100 ft Up
        ],
        dtype=np.float64,
    )
    test_pts_local = frame.to_local(test_pts_source)

    # Expected distances in meters: 100 * (1200 / 3937) ≈ 30.48006 m
    expected_m = 100.0 * METERS_PER_US_SURVEY_FOOT
    assert np.isclose(test_pts_local[0, 0], expected_m)
    assert np.isclose(test_pts_local[1, 1], expected_m)
    assert np.isclose(test_pts_local[2, 2], expected_m)

    # Inverse transformation
    roundtrip = frame.to_source(test_pts_local)
    assert np.allclose(roundtrip, test_pts_source, atol=1e-9)


def test_coordinate_frame_serialization(tmp_path):
    """Test JSON saving and loading of LocalCoordinateFrame."""
    origin_xyz = (5725184.6, 4025570.9, 600.0)
    frame = LocalCoordinateFrame(origin_source_xyz=origin_xyz)
    json_path = tmp_path / "coords.json"

    frame.save_json(json_path)
    assert json_path.exists()

    loaded = LocalCoordinateFrame.load_json(json_path)
    assert loaded.origin_source_xyz == origin_xyz
    assert loaded.horizontal_crs == "EPSG:3089"
    assert loaded.vertical_datum == "NAVD88"
    assert np.isclose(loaded.scale_source_to_meter, METERS_PER_US_SURVEY_FOOT)
