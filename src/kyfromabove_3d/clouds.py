"""The per-building point clouds on disk: LAZ, with PLY still readable.

A building's LiDAR clip (``lidar_local``) and its coloured cloud
(``colorized_aerial``) used to be PLY -- the clip as ASCII text, five times
the size it need be. Both are now written as LAZ through laspy (lazrs does
the compression): the clip about a fifth of the text file, the coloured
cloud about a quarter of the binary PLY, and the clip keeps the LiDAR's
16-bit intensity that the PLY clipped to 8 bits.

Every reader goes through :func:`read_cloud`, which returns the same dict
of columns for either format, so the 378 downtown directories written
before the change need no conversion: :func:`find_cloud` looks for the
``.laz`` first and falls back to the ``.ply``.

Columns: ``x y z`` (float64, local metres), ``intensity`` (uint16),
``classification`` (uint8), ``red green blue`` (uint8), and any extra
uint8 columns (``source``, ``surface``) the colourise records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

CLOUD_SUFFIXES = (".laz", ".las", ".ply")


def find_cloud(directory: Path, stem: str) -> Path:
    """``<directory>/<stem>.laz`` when it exists, else ``.las``, else ``.ply``
    (the path even if none exists, so a missing file fails where it is read)."""
    directory = Path(directory)
    for suffix in CLOUD_SUFFIXES:
        p = directory / f"{stem}{suffix}"
        if p.exists():
            return p
    return directory / f"{stem}.laz"


def read_cloud(path: Path) -> Dict[str, np.ndarray]:
    """A cloud's columns, from LAZ/LAS or PLY."""
    path = Path(path)
    if path.suffix.lower() == ".ply":
        from .geometry import read_ply
        return read_ply(path)
    import laspy

    las = laspy.read(str(path))
    cols: Dict[str, np.ndarray] = {
        "x": np.asarray(las.x, dtype=np.float64),
        "y": np.asarray(las.y, dtype=np.float64),
        "z": np.asarray(las.z, dtype=np.float64),
        "intensity": np.asarray(las.intensity, dtype=np.uint16),
        "classification": np.asarray(las.classification, dtype=np.uint8),
    }
    if "red" in las.point_format.dimension_names:
        for c in ("red", "green", "blue"):
            # LAS colour is 16-bit; the pipeline's colour is 8-bit
            cols[c] = (np.asarray(las[c], dtype=np.uint16) >> 8).astype(np.uint8)
    for dim in las.point_format.extra_dimension_names:
        cols[dim] = np.asarray(las[dim])
    return cols


def write_cloud(path: Path, xyz: np.ndarray, rgb: Optional[np.ndarray] = None,
                intensity: Optional[np.ndarray] = None, classification: Optional[np.ndarray] = None,
                extras: Optional[Dict[str, np.ndarray]] = None) -> Path:
    """Write a cloud as LAZ (or LAS when the path says so): point format 2
    when there is colour, else 0; 1 mm coordinate resolution; every
    ``extras`` column as a uint8 extra-bytes dimension."""
    import laspy

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    header = laspy.LasHeader(point_format=2 if rgb is not None else 0, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.floor(xyz.min(axis=0)) if len(xyz) else np.zeros(3)
    for name in (extras or {}):
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.uint8))
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if intensity is not None:
        las.intensity = np.asarray(intensity).astype(np.uint16)
    if classification is not None:
        las.classification = np.asarray(classification).astype(np.uint8)
    if rgb is not None:
        rgb = np.asarray(rgb).reshape(-1, 3)
        # 8-bit -> 16-bit exactly (0 -> 0, 255 -> 65535), so >> 8 gives it back
        las.red = rgb[:, 0].astype(np.uint16) * 257
        las.green = rgb[:, 1].astype(np.uint16) * 257
        las.blue = rgb[:, 2].astype(np.uint16) * 257
    for name, values in (extras or {}).items():
        las[name] = np.asarray(values).astype(np.uint8)
    las.write(str(path))
    return path
