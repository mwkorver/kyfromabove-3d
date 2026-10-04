"""Building crop extraction with honest windows and honest masks.

Two behaviours are deliberately different from the first implementation:

* The crop window is **padded**, not clamped. Clamping a window back inside the
  sensor keeps the declared geometry but silently changes what the pixels show;
  that is how two of the five Phase 1 crops ended up truncating the building
  while still reporting a clean PASS.
* The validity mask records **where the window actually sampled the sensor**.
  The previous mask flagged any non-black pixel as valid, which on a clamped
  window is every pixel -- all five delivered masks were uniformly 255 and
  carried no information at all.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .cameras import VirtualObliqueCamera
from .coordinates import LocalCoordinateFrame
from .imagery import FrameImageReader, ImageWindow


@dataclass
class BoundingVolume3D:
    """Axis-aligned bounding box in local metric coordinates."""

    min_x_m: float
    max_x_m: float
    min_y_m: float
    max_y_m: float
    min_z_m: float
    max_z_m: float

    @classmethod
    def from_footprint_and_elevation(
        cls,
        footprint_local_xy: np.ndarray,  # (N, 2)
        min_z_m: float,
        max_z_m: float,
        pad_xy_m: float = 5.0,
        pad_z_m: float = 3.0,
    ) -> "BoundingVolume3D":
        min_x = np.min(footprint_local_xy[:, 0]) - pad_xy_m
        max_x = np.max(footprint_local_xy[:, 0]) + pad_xy_m
        min_y = np.min(footprint_local_xy[:, 1]) - pad_xy_m
        max_y = np.max(footprint_local_xy[:, 1]) + pad_xy_m
        return cls(min_x, max_x, min_y, max_y, min_z_m - pad_z_m, max_z_m + pad_z_m)

    def get_8_corners(self) -> np.ndarray:
        """Return (8, 3) vertices of the 3D volume."""
        return np.array(
            [
                [self.min_x_m, self.min_y_m, self.min_z_m],
                [self.max_x_m, self.min_y_m, self.min_z_m],
                [self.max_x_m, self.max_y_m, self.min_z_m],
                [self.min_x_m, self.max_y_m, self.min_z_m],
                [self.min_x_m, self.min_y_m, self.max_z_m],
                [self.max_x_m, self.min_y_m, self.max_z_m],
                [self.max_x_m, self.max_y_m, self.max_z_m],
                [self.min_x_m, self.max_y_m, self.max_z_m],
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "min_x_m": float(self.min_x_m), "max_x_m": float(self.max_x_m),
            "min_y_m": float(self.min_y_m), "max_y_m": float(self.max_y_m),
            "min_z_m": float(self.min_z_m), "max_z_m": float(self.max_z_m),
        }


@dataclass
class CropSpecification:
    """Geometry of an extracted crop, including what it failed to capture."""

    frame_id: str
    x0_px: int  # column offset in the source image; may be negative (padded)
    y0_px: int  # row offset in the source image; may be negative (padded)
    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    R_cw: np.ndarray
    camera_center_local_m: np.ndarray
    sensor_coverage_fraction: float = 1.0
    volume_contained: bool = True
    volume_corners_on_sensor: int = 8
    required_width_px: int = 0
    required_height_px: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "x0_px": int(self.x0_px),
            "y0_px": int(self.y0_px),
            "width_px": int(self.width_px),
            "height_px": int(self.height_px),
            "fx_px": float(self.fx_px),
            "fy_px": float(self.fy_px),
            "cx_px": float(self.cx_px),
            "cy_px": float(self.cy_px),
            "R_cw": self.R_cw.tolist(),
            "camera_center_local_m": self.camera_center_local_m.tolist(),
            "sensor_coverage_fraction": float(self.sensor_coverage_fraction),
            "volume_contained": bool(self.volume_contained),
            "volume_corners_on_sensor": int(self.volume_corners_on_sensor),
            "required_width_px": int(self.required_width_px),
            "required_height_px": int(self.required_height_px),
        }


class BuildingCropExtractor:
    """Extracts pinhole crops for a 3D building volume.

    The source imagery carries no distortion model -- the EO table supplies only
    focal length, pixel pitch and principal point -- so crops are plain pinhole
    windows onto imagery the vendor has already rectified. They are not
    undistorted here, and calling them "undistorted crops" overstates what the
    pipeline does.
    """

    def __init__(self, coord_frame: LocalCoordinateFrame):
        self.coord_frame = coord_frame

    def compute_crop_extent(
        self,
        camera: VirtualObliqueCamera,
        volume_3d: BoundingVolume3D,
        padding_px: int = 100,
    ) -> Tuple[int, int, int, int]:
        """Ideal window (x0, y0, w, h) enclosing the volume, *without* clamping.

        x0/y0 may be negative and the window may extend past the sensor: that is
        the honest answer, and the caller decides what to do about it.
        """
        corners = volume_3d.get_8_corners()
        uv, zc, _ = camera.project(corners)

        if not np.any(zc > 0.1):
            raise ValueError(f"Building volume is entirely behind camera {camera.frame_id}")

        valid_uv = uv[zc > 0.1]
        min_u = int(np.floor(np.min(valid_uv[:, 0]))) - padding_px
        max_u = int(np.ceil(np.max(valid_uv[:, 0]))) + padding_px
        min_v = int(np.floor(np.min(valid_uv[:, 1]))) - padding_px
        max_v = int(np.ceil(np.max(valid_uv[:, 1]))) + padding_px

        return min_u, min_v, max_u - min_u, max_v - min_v

    def extract_crop(
        self,
        camera: VirtualObliqueCamera,
        reader: Union[FrameImageReader, str, Path],
        volume_3d: BoundingVolume3D,
        padding_px: int = 120,
        fixed_square_size: Optional[int] = None,
        allow_truncation: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, CropSpecification]:
        """Extract a crop, its validity mask, and the crop camera intrinsics.

        Args:
            camera: source exposure model.
            reader: a ``FrameImageReader`` or an image path/URL.
            volume_3d: building volume the crop must contain.
            padding_px: margin around the projected volume.
            fixed_square_size: force a square crop of this size. If the volume
                needs more room than this, the crop truncates the building.
            allow_truncation: permit that truncation. Default False raises, so a
                too-small fixed size fails loudly instead of shipping a crop that
                cuts the subject in half.

        Returns:
            ``(rgb, validity_mask, crop_spec)``. The mask is 255 where the pixel
            came from the sensor and 0 where the window ran off it.
        """
        if not hasattr(reader, "read_window"):
            reader = FrameImageReader(reader, frame_id=camera.frame_id)

        x0, y0, w, h = self.compute_crop_extent(camera, volume_3d, padding_px)
        required_w, required_h = w, h

        if fixed_square_size is not None:
            centre_x = x0 + w / 2.0
            centre_y = y0 + h / 2.0
            if (w > fixed_square_size or h > fixed_square_size) and not allow_truncation:
                raise ValueError(
                    f"{camera.frame_id}: volume needs a {w}x{h}px window but "
                    f"fixed_square_size={fixed_square_size} was requested, which would "
                    f"cut off the building. Raise the size, lower padding_px, or pass "
                    f"allow_truncation=True to accept a partial crop."
                )
            if w > fixed_square_size or h > fixed_square_size:
                warnings.warn(
                    f"{camera.frame_id}: crop truncates the building "
                    f"({w}x{h}px needed, {fixed_square_size}px given)",
                    stacklevel=2,
                )
            x0 = int(round(centre_x - fixed_square_size / 2.0))
            y0 = int(round(centre_y - fixed_square_size / 2.0))
            w = h = int(fixed_square_size)

        window = reader.read_window(x0, y0, w, h)
        rgb = window.rgb
        validity = (window.valid * 255).astype(np.uint8)

        # How much of the volume this crop actually holds.
        corners = volume_3d.get_8_corners()
        uv, zc, _ = camera.project(corners)
        on_sensor = (
            (zc > 0.1)
            & (uv[:, 0] >= 0) & (uv[:, 0] <= camera.width_px)
            & (uv[:, 1] >= 0) & (uv[:, 1] <= camera.height_px)
        )
        in_crop = (
            (zc > 0.1)
            & (uv[:, 0] >= x0) & (uv[:, 0] <= x0 + w)
            & (uv[:, 1] >= y0) & (uv[:, 1] <= y0 + h)
        )

        spec = CropSpecification(
            frame_id=camera.frame_id,
            x0_px=x0,
            y0_px=y0,
            width_px=w,
            height_px=h,
            fx_px=camera.fx_px,
            fy_px=camera.fy_px,
            cx_px=camera.cx_px - float(x0),
            cy_px=camera.cy_px - float(y0),
            R_cw=camera.R_cw.copy(),
            camera_center_local_m=camera.camera_center_local_m.copy(),
            sensor_coverage_fraction=window.coverage_fraction,
            volume_contained=bool(in_crop.all() and on_sensor.all()),
            volume_corners_on_sensor=int(on_sensor.sum()),
            required_width_px=int(required_w),
            required_height_px=int(required_h),
        )
        return rgb, validity, spec
