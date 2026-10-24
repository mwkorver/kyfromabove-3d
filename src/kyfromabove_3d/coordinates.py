"""Coordinate frame reconciliation and transformations for Kentucky From Above data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np

# Exact US Survey Foot definition: 1 meter = 39.37 inches, 1 foot = 12 inches
# s = 1200 / 3937 meters per US survey foot ≈ 0.3048006096012192 m/ft
METERS_PER_US_SURVEY_FOOT: float = 1200.0 / 3937.0
US_SURVEY_FEET_PER_METER: float = 3937.0 / 1200.0


def us_survey_feet_to_meters(val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert US survey feet to meters in double precision."""
    return np.asarray(val, dtype=np.float64) * METERS_PER_US_SURVEY_FOOT


def meters_to_us_survey_feet(val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert meters to US survey feet in double precision."""
    return np.asarray(val, dtype=np.float64) * US_SURVEY_FEET_PER_METER


@dataclass
class LocalCoordinateFrame:
    """Building-centered local metric Cartesian frame.
    
    Transforms source EPSG:3089 coordinates (US Survey Feet, NAVD88)
    into a local right-handed metric frame centered at `origin_source_xyz`.
    
    In the local frame:
      +X: East (meters)
      +Y: North (meters)
      +Z: Up (meters)
    """

    origin_source_xyz: Tuple[float, float, float]
    horizontal_crs: str = "EPSG:3089"
    horizontal_unit: str = "US survey foot"
    vertical_datum: str = "NAVD88"
    vertical_unit: str = "US survey foot"
    scale_source_to_meter: float = METERS_PER_US_SURVEY_FOOT

    def __post_init__(self):
        self.origin_source_xyz = (
            float(self.origin_source_xyz[0]),
            float(self.origin_source_xyz[1]),
            float(self.origin_source_xyz[2]),
        )

    @property
    def origin_source_array(self) -> np.ndarray:
        return np.array(self.origin_source_xyz, dtype=np.float64)

    def to_local(self, xyz_source: np.ndarray | Tuple[float, float, float]) -> np.ndarray:
        """Convert source coordinates (ft) to local metric coordinates (m).
        
        X_local = s * (X_source - O_source)
        """
        pts = np.asarray(xyz_source, dtype=np.float64)
        is_1d = pts.ndim == 1
        if is_1d:
            pts = pts.reshape(1, 3)

        diff = pts - self.origin_source_array
        local_pts = diff * self.scale_source_to_meter

        return local_pts[0] if is_1d else local_pts

    def to_source(self, xyz_local: np.ndarray | Tuple[float, float, float]) -> np.ndarray:
        """Convert local metric coordinates (m) back to source coordinates (ft).
        
        X_source = O_source + (1 / s) * X_local
        """
        pts = np.asarray(xyz_local, dtype=np.float64)
        is_1d = pts.ndim == 1
        if is_1d:
            pts = pts.reshape(1, 3)

        diff = pts / self.scale_source_to_meter
        source_pts = diff + self.origin_source_array

        return source_pts[0] if is_1d else source_pts

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_source_xyz": list(self.origin_source_xyz),
            "horizontal_crs": self.horizontal_crs,
            "horizontal_unit": self.horizontal_unit,
            "vertical_datum": self.vertical_datum,
            "vertical_unit": self.vertical_unit,
            "scale_source_to_meter": self.scale_source_to_meter,
            "local_axes": {
                "+X": "East (meters)",
                "+Y": "North (meters)",
                "+Z": "Up (meters)",
            },
        }

    def save_json(self, filepath: Union[str, Path]) -> None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LocalCoordinateFrame:
        origin = tuple(d["origin_source_xyz"])
        return cls(
            origin_source_xyz=(origin[0], origin[1], origin[2]),
            horizontal_crs=d.get("horizontal_crs", "EPSG:3089"),
            horizontal_unit=d.get("horizontal_unit", "US survey foot"),
            vertical_datum=d.get("vertical_datum", "NAVD88"),
            vertical_unit=d.get("vertical_unit", "US survey foot"),
            scale_source_to_meter=d.get("scale_source_to_meter", METERS_PER_US_SURVEY_FOOT),
        )

    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> LocalCoordinateFrame:
        with open(filepath, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)
