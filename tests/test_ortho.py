"""Ortho tile grid maths, window alignment, and the rectification check."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from kyfromabove_3d.coordinates import LocalCoordinateFrame
from kyfromabove_3d.ortho import (
    KY_ORTHO_ORIGIN_X_FT,
    KY_ORTHO_ORIGIN_Y_FT,
    OrthoCollection,
    OrthoReader,
    assess_ortho_rectification,
    ortho_tilename,
    ortho_tiles_for_bounds,
)

from conftest import sample_texture

ORIGIN = (5725184.6, 4025570.9, 600.0)


def test_tile_grid_matches_published_world_files():
    """Pinned against the 39-byte .tfw sidecars the grid was derived from."""
    # N037E390's own upper-left corner must land in N037E390.
    assert ortho_tilename(KY_ORTHO_ORIGIN_X_FT + 1, KY_ORTHO_ORIGIN_Y_FT - 1) == "N037E390"
    # +5000 ft east is E391; 5000 ft south is N038.
    assert ortho_tilename(KY_ORTHO_ORIGIN_X_FT + 5001, KY_ORTHO_ORIGIN_Y_FT - 1) == "N037E391"
    assert ortho_tilename(KY_ORTHO_ORIGIN_X_FT + 1, KY_ORTHO_ORIGIN_Y_FT - 5001) == "N038E390"
    # The Grayson pilot building.
    assert ortho_tilename(5725184.6, 4025570.9) == "N067E390"


def test_bounds_spanning_a_tile_edge_returns_both_tiles():
    """The pilot AOI genuinely straddles two ortho tiles; one is not enough."""
    tiles = ortho_tiles_for_bounds(5725000.0, 4025400.0, 5725350.0, 4025800.0)
    assert tiles == ["N067E389", "N067E390"]


def test_collection_builds_urls():
    url = OrthoCollection().url_for_point(5725184.6, 4025570.9)
    assert url.endswith("N067E390_2022_Season2_3IN_cog.tif")


def _write_ortho(path, texture, coord_frame, half_m=70.0, res_ft=0.25, shift_ft=0.0):
    """Synthesise a georeferenced ortho of the z=0 textured plane.

    ``shift_ft`` deliberately mis-georeferences it, to check the control.
    """
    s = coord_frame.scale_source_to_meter
    half_ft = half_m / s
    ox, oy, _ = coord_frame.origin_source_xyz
    minx, maxy = ox - half_ft, oy + half_ft
    n = int(2 * half_ft / res_ft)
    cols, rows = np.meshgrid(np.arange(n), np.arange(n))
    xs = minx + (cols + 0.5) * res_ft
    ys = maxy - (rows + 0.5) * res_ft
    local = coord_frame.to_local(
        np.column_stack([xs.ravel() + shift_ft, ys.ravel(), np.zeros(xs.size)])
    )
    vals = np.clip(sample_texture(texture, local[:, :2]), 0, 255).astype(np.uint8).reshape(n, n)
    with rasterio.open(
        path, "w", driver="GTiff", height=n, width=n, count=3, dtype="uint8",
        crs="EPSG:3089", transform=from_origin(minx, maxy, res_ft, res_ft),
    ) as dst:
        for b in range(1, 4):
            dst.write(vals, b)
    return path


def test_read_bounds_is_boundless_so_array_and_transform_agree(tmp_path, texture_grid):
    """A window running off the tile must not silently desynchronise.

    ``src.read`` clips such a window while ``window_transform`` describes the
    unclipped origin; at a tile edge that produced an 862 px (65 m) shift with
    no error raised.
    """
    frame = LocalCoordinateFrame(origin_source_xyz=ORIGIN)
    path = _write_ortho(tmp_path / "ortho.tif", texture_grid, frame, half_m=40.0)
    reader = OrthoReader(path)

    minx, miny, maxx, maxy = reader.bounds
    # Deliberately hang 200 ft off the western edge.
    rgb, transform, valid = reader.read_bounds(minx - 200.0, miny, minx + 200.0, maxy)

    expected_cols = int(round(400.0 / reader.res[0]))
    assert rgb.shape[1] == expected_cols, "array width must match the requested window"
    # Pixel 0 of the array must correspond to the requested origin, not the tile's.
    assert transform.c == pytest.approx(minx - 200.0, abs=reader.res[0])
    # The off-tile half is flagged invalid rather than passed off as imagery.
    assert not valid[:, :10].any()
    assert valid[:, -10:].any()


def test_control_withholds_a_verdict_when_georeferencing_is_wrong(
    tmp_path, texture_grid, ground_anchors, synthetic_pair
):
    """A mis-georeferenced ortho must be reported inconclusive, not diagnosed.

    Ground anchors are the control: if they do not rectify, the measurement is
    untrustworthy and no true/standard verdict can be justified.
    """
    cam, win, _, _ = synthetic_pair
    frame = LocalCoordinateFrame(origin_source_xyz=ORIGIN)
    bad = _write_ortho(tmp_path / "shifted.tif", texture_grid, frame, shift_ft=25.0)

    result = assess_ortho_rectification(
        bad, cam, win, ground_anchors, frame,
        patch_radius_m=4.0, search_radius_m=3.0, gsd_m=0.10, min_anchors=4,
    )
    assert result.verdict == "inconclusive"
    assert not result.usable_for_footprints
    assert "ground control failed" in result.reason or "need >=" in result.reason


def test_incoherent_matches_are_reported_as_a_scene_mismatch(
    tmp_path, texture_grid, ground_anchors, synthetic_pair
):
    """Noise and a georeferencing shift are different failures; say which.

    An ortho of a *different* scene yields correlations that scatter as widely
    as they displace. Reporting that as "check your CRS" sends the reader after
    the wrong problem -- on the real 2019 6-inch collection the actual cause was
    a three-year vintage gap against a 2022 reference exposure.
    """
    from scipy.signal import convolve2d

    cam, win, _, _ = synthetic_pair
    frame = LocalCoordinateFrame(origin_source_xyz=ORIGIN)

    rng = np.random.default_rng(1234)      # deliberately not the camera's texture
    n = texture_grid.shape[0]
    other = convolve2d(rng.integers(0, 256, size=(n, n)).astype(float),
                       np.ones((3, 3)) / 9.0, mode="same", boundary="symm")
    unrelated = _write_ortho(tmp_path / "other_scene.tif", other, frame)

    result = assess_ortho_rectification(
        unrelated, cam, win, ground_anchors, frame,
        patch_radius_m=4.0, search_radius_m=3.0, gsd_m=0.10,
        min_anchors=4, min_ncc=0.0,
    )
    assert result.verdict == "inconclusive"
    assert not result.control_passed
    assert "different scenes" in result.reason
    assert result.ground_scatter_m >= 0.5 * result.ground_displacement_m


def test_missing_roof_anchors_is_inconclusive_not_true_ortho(
    tmp_path, texture_grid, ground_anchors, synthetic_pair
):
    """With no roof anchors there is nothing to compare; do not default to pass."""
    cam, win, _, _ = synthetic_pair
    frame = LocalCoordinateFrame(origin_source_xyz=ORIGIN)
    good = _write_ortho(tmp_path / "ortho.tif", texture_grid, frame)

    result = assess_ortho_rectification(
        good, cam, win, ground_anchors, frame,
        patch_radius_m=4.0, search_radius_m=3.0, gsd_m=0.10, min_anchors=4,
    )
    assert result.n_roof == 0
    assert result.verdict == "inconclusive"
    assert not result.usable_for_footprints
