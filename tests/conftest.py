"""Synthetic scene fixtures for testing the QA measurement path offline."""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from kyfromabove_3d.cameras import VirtualObliqueCamera, photogrammetric_omega_phi_kappa_to_rcw
from kyfromabove_3d.features import PlanarAnchor, tangent_basis
from kyfromabove_3d.imagery import ImageWindow

TEXTURE_CELL_M = 0.4
TEXTURE_HALF_EXTENT_M = 120.0


@pytest.fixture(scope="session")
def texture_grid() -> np.ndarray:
    """A deterministic, well-textured ground pattern for correlation to lock onto."""
    rng = np.random.default_rng(20260906)
    n = int(2 * TEXTURE_HALF_EXTENT_M / TEXTURE_CELL_M)
    grid = rng.integers(0, 256, size=(n, n)).astype(np.float64)
    # Mild smoothing so bilinear resampling is well behaved.
    k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0
    from scipy.signal import convolve2d

    return convolve2d(grid, k, mode="same", boundary="symm")


def sample_texture(texture: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear lookup of the ground texture at local metric XY."""
    gx = (xy[:, 0] + TEXTURE_HALF_EXTENT_M) / TEXTURE_CELL_M
    gy = (xy[:, 1] + TEXTURE_HALF_EXTENT_M) / TEXTURE_CELL_M
    return map_coordinates(texture, [gy, gx], order=1, mode="nearest")


def make_camera(frame_id: str, centre: np.ndarray, omega=45.0, phi=0.0, kappa=0.0):
    return VirtualObliqueCamera(
        frame_id=frame_id,
        camera_name="SyntheticOblique",
        camera_center_local_m=np.asarray(centre, dtype=np.float64),
        R_cw=photogrammetric_omega_phi_kappa_to_rcw(omega, phi, kappa),
        width_px=14144,
        height_px=10560,
        fx_px=32813.83,
        fy_px=32813.83,
        cx_px=7072.0,
        cy_px=5280.0,
    )


def render_window(
    camera: VirtualObliqueCamera, texture: np.ndarray, anchors, margin_px: int = 220
) -> ImageWindow:
    """Inverse-render the z=0 textured plane as this camera would see it.

    Each output pixel is unprojected to a ray and intersected with the ground
    plane, so the rendering is exact for the plane and independent of the
    forward-projection code path that ``sample_plane_patch`` exercises.
    """
    pts = np.array([a.xyz_local_m for a in anchors], dtype=np.float64)
    uv, _, _ = camera.project(pts)
    x0 = int(np.floor(uv[:, 0].min())) - margin_px
    y0 = int(np.floor(uv[:, 1].min())) - margin_px
    w = int(np.ceil(uv[:, 0].max())) + margin_px - x0
    h = int(np.ceil(uv[:, 1].max())) + margin_px - y0

    us, vs = np.meshgrid(np.arange(x0, x0 + w), np.arange(y0, y0 + h))
    pix = np.column_stack([us.ravel().astype(np.float64), vs.ravel().astype(np.float64)])
    centre, dirs = camera.unproject(pix)

    # Intersect each ray with the ground plane z = 0.
    t = -centre[2] / dirs[:, 2]
    ground = centre[None, :] + t[:, None] * dirs
    vals = sample_texture(texture, ground[:, :2]).reshape(h, w)
    vals = np.clip(vals, 0, 255).astype(np.uint8)

    return ImageWindow(
        rgb=np.dstack([vals, vals, vals]),
        valid=np.ones((h, w), dtype=bool),
        x0=x0,
        y0=y0,
    )


@pytest.fixture(scope="session")
def ground_anchors():
    """A small grid of anchors on the z=0 plane."""
    anchors = []
    for i, x in enumerate(np.linspace(-20, 20, 4)):
        for j, y in enumerate(np.linspace(-20, 20, 4)):
            normal = np.array([0.0, 0.0, 1.0])
            e1, e2 = tangent_basis(normal)
            anchors.append(
                PlanarAnchor(
                    xyz_local_m=np.array([x, y, 0.0]),
                    normal=normal,
                    e1=e1,
                    e2=e2,
                    planarity_rms_m=0.0,
                    height_above_ground_m=0.0,
                    label="ground",
                    anchor_id=f"S{i}{j}",
                )
            )
    return anchors


@pytest.fixture(scope="session")
def synthetic_pair(texture_grid, ground_anchors):
    """Two synthetic exposures ~250 m apart along a flight line, and their renders."""
    cam_a = make_camera("synth/frame_a", np.array([0.0, -1600.0, 1600.0]))
    cam_b = make_camera("synth/frame_b", np.array([0.0, -1350.0, 1610.0]))
    win_a = render_window(cam_a, texture_grid, ground_anchors)
    win_b = render_window(cam_b, texture_grid, ground_anchors)
    return cam_a, win_a, cam_b, win_b
