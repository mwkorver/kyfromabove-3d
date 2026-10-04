"""Project root discovery and a session object for the camera-validation work.

``Project`` resolves everything against the repository root, so nothing
depends on the working directory, and caches the expensive loads
(GeoPackages, LAZ) so repeated calls are cheap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from .cameras import VirtualObliqueCamera, parse_camera_from_eo_record
from .coordinates import LocalCoordinateFrame

_ROOT_MARKERS = ("pyproject.toml", "configs/phase1.yaml")


def find_project_root(start: Optional[Union[str, Path]] = None) -> Path:
    """Walk upward from ``start`` (default: cwd) looking for the repo root.

    Falls back to the parent of the installed package, which is correct for the
    editable install used in development.
    """
    candidates: List[Path] = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(Path(__file__).resolve())

    for candidate in candidates:
        for parent in [candidate, *candidate.parents]:
            if all((parent / marker).exists() for marker in _ROOT_MARKERS):
                return parent
    # src/kyfromabove_3d/project.py -> repo root
    return Path(__file__).resolve().parents[2]


def repo_relative(path: Union[str, Path], root: Optional[Path] = None) -> str:
    """``path`` as written into a report: relative to the repository root when
    it is inside it, so a committed report carries ``outputs/...`` and not the
    absolute path of whoever's machine ran it; otherwise as given."""
    root = root or Path(__file__).resolve().parents[2]
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


@dataclass
class Project:
    """A loaded Phase 1 configuration plus lazily-materialised data handles.

    Typical use::

        from kyfromabove_3d import Project
        proj = Project()
        cams = proj.cameras
        lidar = proj.lidar
    """

    config_path: Optional[Union[str, Path]] = None
    root: Path = field(init=False)
    config: Dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = find_project_root()
        cfg_path = (
            Path(self.config_path)
            if self.config_path is not None
            else self.root / "configs" / "phase1.yaml"
        )
        if not cfg_path.is_absolute():
            cfg_path = self.root / cfg_path
        self.config_path = cfg_path
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    # ------------------------------------------------------------------ paths
    def _path(self, key: str) -> Path:
        return self.root / self.config["paths"][key]

    @property
    def raw_dir(self) -> Path:
        return self._path("raw")

    @property
    def derived_dir(self) -> Path:
        return self._path("derived")

    @property
    def output_dir(self) -> Path:
        return self._path("output")

    @property
    def qa_dir(self) -> Path:
        """Under the building, not the output tree.

        It used to be ``output_dir / "qa"``, which two buildings sharing a
        ``paths.output`` silently write over each other in: running PNC Tower
        into ``outputs/urban`` destroyed ``building_002``'s reprojection report,
        the file the facade evidence came from, and nothing said so. Phase 2
        already wrote to ``outputs/phase2/<building>/qa``; this is the rest of
        the project catching up to that.
        """
        return self.building_dir / "qa"

    @property
    def building_id(self) -> str:
        return self.config["aoi"]["building_ids"][0]

    @property
    def building_dir(self) -> Path:
        return self.output_dir / self.building_id

    @property
    def collection(self) -> str:
        """The oblique collection this AOI is flown by.

        Defaults to the pilot's collection so building_001's config, which
        predates there being a choice, keeps working unchanged.
        """
        return (self.config.get("selection", {}).get("collection")
                or "KY_KYAPED_2022_Season2_3IN")

    def _index(self, kind: str):
        """A local index if one is downloaded, else a ``/vsicurl`` path.

        The 2024 indexes are 553 MB and 1.07 GB. Neither needs downloading: a
        GeoPackage is SQLite with an RTree, so a spatial query is served by
        range request. Returning a str rather than a Path is the signal that it
        is remote.
        """
        from .urban import collection_full_name, eo_index_url, frames_index_url

        name = collection_full_name(self.collection)
        local = self.raw_dir / f"{name}_ImageFrame{kind}.gpkg"
        if local.exists():
            return local
        return eo_index_url(self.collection) if kind == "EO" else frames_index_url(self.collection)

    @property
    def eo_gpkg(self):
        return self._index("EO")

    @property
    def frames_gpkg(self):
        return self._index("s")

    @property
    def laz_file(self) -> Path:
        """The pilot's single pre-downloaded tile.

        Retained only for the Grayson AOI. Any other AOI resolves its tiles
        through :class:`LidarTileIndex` instead -- see :attr:`lidar`.
        """
        return self.raw_dir / "N067E390.laz"

    # ------------------------------------------------------------- geo/config
    @property
    def origin_source_xyz(self) -> Tuple[float, float, float]:
        x, y, z = self.config["coordinates"]["local_origin_source_xyz"]
        return (float(x), float(y), float(z))

    @cached_property
    def coord_frame(self) -> LocalCoordinateFrame:
        return LocalCoordinateFrame(
            origin_source_xyz=self.origin_source_xyz,
            horizontal_crs=self.config["aoi"]["horizontal_crs"],
            vertical_datum=self.config["coordinates"]["vertical_datum"],
        )

    @property
    def aoi_bounds_source_ft(self) -> Tuple[float, float, float, float, float, float]:
        """(minx, maxx, miny, maxy, minz, maxz) in EPSG:3089 US survey feet."""
        minx, miny, maxx, maxy = self.config["aoi"]["bounds"]
        # A narrow default is a trap: it silently returns *zero* points for any
        # AOI outside it rather than erroring. [550, 750] is the Grayson range,
        # and Louisville's ground alone is 463 ft with a roof at 1004 ft. The
        # default is now wide enough to be inert; blunders are excluded by
        # classification, not by elevation.
        zmin, zmax = self.config["aoi"].get("z_bounds_ft", [-1000.0, 20000.0])
        return (float(minx), float(maxx), float(miny), float(maxy), float(zmin), float(zmax))

    # ------------------------------------------------------------------ data
    @cached_property
    def candidate_frames(self):
        """DataFrame of exposures whose footprint covers the AOI."""
        from .sources import discover_candidate_frames

        return discover_candidate_frames(
            eo_gpkg_path=self.eo_gpkg,
            frames_gpkg_path=self.frames_gpkg,
            aoi_point_or_bounds=self.aoi_bounds_ft_xy,
            flight_line=self.flight_lines,
            camera_direction=self.camera_directions,
            max_frames=self.config["selection"].get("target_frame_count"),
        )

    @property
    def camera_directions(self) -> List[str]:
        sel = self.config["selection"]
        dirs = sel.get("camera_directions") or sel.get("source_direction_label", "Fwd")
        return [dirs] if isinstance(dirs, str) else list(dirs)

    @property
    def flight_lines(self):
        """Configured flight lines, or None to accept every line."""
        return self.config["selection"].get("flight_ids") or self.config["selection"].get("flight_id")

    @property
    def aoi_bounds_ft_xy(self) -> Tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy) as shapely/geopandas expect."""
        minx, miny, maxx, maxy = self.config["aoi"]["bounds"]
        return (float(minx), float(miny), float(maxx), float(maxy))

    @cached_property
    def cameras(self) -> List[VirtualObliqueCamera]:
        return [
            parse_camera_from_eo_record(dict(row), self.coord_frame)
            for _, row in self.candidate_frames.iterrows()
        ]

    @cached_property
    def lidar(self):
        """Points for this AOI, merged across every covering tile.

        Resolves tiles through the index rather than reading a fixed file, so an
        AOI anywhere in Kentucky works. The pilot's pre-downloaded tile is used
        when it is the one the index returns, which keeps building_001 offline.
        """
        from .lidar import LidarTileIndex, extract_lidar_for_aois

        index = LidarTileIndex(self.raw_dir)
        clouds = extract_lidar_for_aois(
            {self.building_id: self.aoi_bounds_source_ft},
            self.coord_frame, index,
            local_dir=self.raw_dir, progress=False,
        )
        return clouds[self.building_id]

    def building_context(self, pad_xy_m: float = 6.0, pad_z_m: float = 2.0):
        """LiDAR, check features, building volume and coverage-selected cameras.

        The same setup ``scripts/validate_building.py`` performs, in one place so
        the later scripts cannot drift from the Phase 1 selection. Returns a
        dict with keys ``lidar``, ``check_features``, ``roof_xyz``, ``footprint``,
        ``volume``, ``all_cameras``, ``cameras`` and ``coverage``.
        """
        import numpy as np

        from .cameras import parse_camera_from_eo_record
        from .crops import BoundingVolume3D
        from .features import roof_check_features
        from .sources import discover_candidate_frames, select_frames_by_coverage

        lidar = self.lidar
        check_features, roof_xyz = roof_check_features(lidar)
        if not check_features:
            raise ValueError("could not isolate a roof surface from the LiDAR")

        footprint = np.array([f.xyz_local_m[:2] for f in check_features[:4]])
        volume = BoundingVolume3D.from_footprint_and_elevation(
            footprint,
            min_z_m=float(roof_xyz[:, 2].min()) - 8.0,
            max_z_m=float(roof_xyz[:, 2].max()),
            pad_xy_m=pad_xy_m,
            pad_z_m=pad_z_m,
        )

        frames = discover_candidate_frames(
            self.eo_gpkg, self.frames_gpkg, self.aoi_bounds_ft_xy,
            flight_line=self.flight_lines, camera_direction=self.camera_directions,
            max_frames=None, require_containment=True,
        )
        all_cameras = [parse_camera_from_eo_record(dict(r), self.coord_frame)
                       for _, r in frames.iterrows()]
        cameras, coverage = select_frames_by_coverage(all_cameras, volume)

        return {
            "lidar": lidar, "check_features": check_features, "roof_xyz": roof_xyz,
            "footprint": footprint, "volume": volume,
            "all_cameras": all_cameras, "cameras": cameras, "coverage": coverage,
        }

    def camera_by_frame_id(self, frame_id: str) -> VirtualObliqueCamera:
        for cam in self.cameras:
            if cam.frame_id == frame_id or cam.frame_id.endswith(frame_id):
                return cam
        raise KeyError(f"No camera matching {frame_id!r}")

    def __repr__(self) -> str:
        # Abbreviate the home directory: an absolute path in printed output
        # is noise and pins one machine's layout.
        root = self.root
        try:
            root = "~/" + str(root.relative_to(Path.home()))
        except ValueError:
            pass
        return (
            f"Project(root={root}, building={self.building_id!r}, "
            f"origin_ft={self.origin_source_xyz})"
        )
