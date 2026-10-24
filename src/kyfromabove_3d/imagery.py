"""Windowed access to the source oblique COGs.

Two things this module exists to get right:

1. **Padded, not clamped, reads.** A window that runs off the sensor is padded
   with zeros and reported through a validity mask, rather than being silently
   slid back inside the image. Clamping is what let Phase 1 emit crops whose
   nominal geometry did not match their content.
2. **One read per frame.** QA measurement touches many small patches spread
   across the building. Reading each one over HTTP is slow, so a single
   enclosing window is fetched and every patch is then sampled from RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import rasterio
from rasterio.windows import Window

GDAL_ENV = dict(
    AWS_NO_SIGN_REQUEST="YES",
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MAX_RETRY="3",
    GDAL_HTTP_RETRY_DELAY="1",
)


@dataclass
class ImageWindow:
    """A rectangular region of a source image, with out-of-sensor padding marked.

    ``rgb`` is (H, W, 3) uint8. ``valid`` is (H, W) bool and is False exactly
    where the requested window fell outside the sensor. ``x0``/``y0`` are the
    requested top-left in full-image pixel coordinates and may be negative.
    """

    rgb: np.ndarray
    valid: np.ndarray
    x0: int
    y0: int

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def coverage_fraction(self) -> float:
        """Fraction of the window that was actually sampled from the sensor."""
        return float(self.valid.mean())

    def contains(self, x0: int, y0: int, w: int, h: int) -> bool:
        return (
            x0 >= self.x0
            and y0 >= self.y0
            and x0 + w <= self.x0 + self.width
            and y0 + h <= self.y0 + self.height
        )

    def to_local_xy(self, uv_full: np.ndarray) -> np.ndarray:
        """Convert full-image pixel coordinates into this window's frame."""
        uv = np.asarray(uv_full, dtype=np.float64)
        return uv - np.array([self.x0, self.y0], dtype=np.float64)


class FrameImageReader:
    """Reads windows from one source image (local path or HTTPS COG URL)."""

    def __init__(self, path_or_url: Union[str, Path], frame_id: Optional[str] = None):
        self.path_or_url = str(path_or_url)
        self.frame_id = frame_id or self.path_or_url
        self._cached: Optional[ImageWindow] = None
        with rasterio.Env(**GDAL_ENV):
            with rasterio.open(self.path_or_url) as src:
                self.width = int(src.width)
                self.height = int(src.height)
                self.count = int(src.count)

    def read_window(self, x0: int, y0: int, w: int, h: int) -> ImageWindow:
        """Read (x0, y0, w, h); regions outside the sensor come back zero-padded."""
        x0, y0, w, h = int(x0), int(y0), int(w), int(h)
        if w <= 0 or h <= 0:
            raise ValueError(f"Window must have positive size, got {w}x{h}")

        if self._cached is not None and self._cached.contains(x0, y0, w, h):
            c = self._cached
            r0, c0 = y0 - c.y0, x0 - c.x0
            return ImageWindow(
                rgb=c.rgb[r0 : r0 + h, c0 : c0 + w].copy(),
                valid=c.valid[r0 : r0 + h, c0 : c0 + w].copy(),
                x0=x0,
                y0=y0,
            )

        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        valid = np.zeros((h, w), dtype=bool)

        # Intersection of the requested window with the sensor.
        ix0, iy0 = max(x0, 0), max(y0, 0)
        ix1, iy1 = min(x0 + w, self.width), min(y0 + h, self.height)
        if ix1 > ix0 and iy1 > iy0:
            with rasterio.Env(**GDAL_ENV):
                with rasterio.open(self.path_or_url) as src:
                    arr = src.read(
                        indexes=[1, 2, 3],
                        window=Window(ix0, iy0, ix1 - ix0, iy1 - iy0),
                    )
            patch = np.transpose(arr, (1, 2, 0))
            rgb[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = patch
            valid[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = True

        return ImageWindow(rgb=rgb, valid=valid, x0=x0, y0=y0)

    def prefetch(self, x0: int, y0: int, w: int, h: int) -> ImageWindow:
        """Read a window and keep it, so later sub-windows are served from RAM."""
        self._cached = self.read_window(x0, y0, w, h)
        return self._cached

    def prefetch_covering(
        self, uv: np.ndarray, margin_px: int = 96, max_side: int = 6000
    ) -> ImageWindow:
        """Cache a single window enclosing all of ``uv`` plus a margin."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        finite = uv[np.all(np.isfinite(uv), axis=1)]
        if len(finite) == 0:
            raise ValueError("No finite pixel coordinates to cover")
        x0 = int(np.floor(finite[:, 0].min())) - margin_px
        y0 = int(np.floor(finite[:, 1].min())) - margin_px
        x1 = int(np.ceil(finite[:, 0].max())) + margin_px
        y1 = int(np.ceil(finite[:, 1].max())) + margin_px

        # Keep the fetch bounded; clip to the sensor plus one margin of padding.
        x0 = max(x0, -margin_px)
        y0 = max(y0, -margin_px)
        x1 = min(x1, self.width + margin_px)
        y1 = min(y1, self.height + margin_px)
        w = min(x1 - x0, max_side)
        h = min(y1 - y0, max_side)
        return self.prefetch(x0, y0, w, h)

    def clear_cache(self) -> None:
        self._cached = None

    def __repr__(self) -> str:
        return f"FrameImageReader({self.frame_id!r}, {self.width}x{self.height})"


#: KyFromAbove's bucket, as the frame index gives each frame's address.
KFA_HTTPS = "https://kyfromabove.s3.amazonaws.com/"


def cog_url(url: str) -> str:
    """A frame's address, through the machine's shared frame cache when
    ``COG_CACHE_URL`` is set (an nginx in front of S3,
    ``deploy/cog_cache``): ``http://127.0.0.1:8001/<key>`` for
    ``https://kyfromabove.s3.amazonaws.com/<key>``. The same bytes either way."""
    import os

    base = os.environ.get("COG_CACHE_URL", "").rstrip("/")
    if base and url.startswith(KFA_HTTPS):
        return f"{base}/{url[len(KFA_HTTPS):]}"
    return url


def readers_for_cameras(cameras: Sequence, local_dir: Optional[Path] = None) -> Dict[str, FrameImageReader]:
    """Build one reader per camera, preferring a local .tif when present."""
    readers: Dict[str, FrameImageReader] = {}
    for cam in cameras:
        src = cog_url(cam.s3_url)
        if local_dir is not None:
            local = Path(local_dir) / (Path(cam.frame_id).name + ".tif")
            if local.exists():
                src = str(local)
        readers[cam.frame_id] = FrameImageReader(src, frame_id=cam.frame_id)
    return readers
