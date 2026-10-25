"""LiDAR extraction, coordinate transformation, and point filtering for building AOIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import laspy
import numpy as np

from .coordinates import LocalCoordinateFrame


@dataclass
class LidarCloud:
    """In-memory LiDAR point cloud container."""

    xyz_source_ft: np.ndarray  # (N, 3) in EPSG:3089 US survey feet
    xyz_local_m: np.ndarray  # (N, 3) in local metric frame
    intensity: np.ndarray  # (N,)
    classification: np.ndarray  # (N,)
    return_number: np.ndarray  # (N,)
    number_of_returns: np.ndarray  # (N,)

    def __len__(self) -> int:
        return len(self.xyz_source_ft)

    def filter_by_bounds_source(
        self, bounds_source: Tuple[float, float, float, float, float, float]
    ) -> LidarCloud:
        """Filter points by (minx, maxx, miny, maxy, minz, maxz) in source feet."""
        minx, maxx, miny, maxy, minz, maxz = bounds_source
        mask = (
            (self.xyz_source_ft[:, 0] >= minx)
            & (self.xyz_source_ft[:, 0] <= maxx)
            & (self.xyz_source_ft[:, 1] >= miny)
            & (self.xyz_source_ft[:, 1] <= maxy)
            & (self.xyz_source_ft[:, 2] >= minz)
            & (self.xyz_source_ft[:, 2] <= maxz)
        )
        return self._apply_mask(mask)

    def filter_by_bounds_local(
        self, bounds_local: Tuple[float, float, float, float, float, float]
    ) -> LidarCloud:
        """Filter points by (minx, maxx, miny, maxy, minz, maxz) in local meters."""
        minx, maxx, miny, maxy, minz, maxz = bounds_local
        mask = (
            (self.xyz_local_m[:, 0] >= minx)
            & (self.xyz_local_m[:, 0] <= maxx)
            & (self.xyz_local_m[:, 1] >= miny)
            & (self.xyz_local_m[:, 1] <= maxy)
            & (self.xyz_local_m[:, 2] >= minz)
            & (self.xyz_local_m[:, 2] <= maxz)
        )
        return self._apply_mask(mask)

    def filter_by_classification(self, classes: List[int]) -> LidarCloud:
        """Filter points by ASPRS classification codes."""
        mask = np.isin(self.classification, classes)
        return self._apply_mask(mask)

    def decimate(self, factor: int) -> LidarCloud:
        """Uniformly decimate point cloud by keeping every factor-th point."""
        if factor <= 1:
            return self
        idx = np.arange(0, len(self), factor)
        return self._apply_mask(idx)

    def _apply_mask(self, mask: np.ndarray) -> LidarCloud:
        return LidarCloud(
            xyz_source_ft=self.xyz_source_ft[mask],
            xyz_local_m=self.xyz_local_m[mask],
            intensity=self.intensity[mask],
            classification=self.classification[mask],
            return_number=self.return_number[mask],
            number_of_returns=self.number_of_returns[mask],
        )

    def save_laz(self, filepath: Union[str, Path], use_local_coords: bool = True) -> Path:
        """Save the cloud as LAZ: a fifth of the ASCII PLY, and the full
        16-bit intensity kept. Read back with ``clouds.read_cloud``."""
        from .clouds import write_cloud
        pts = self.xyz_local_m if use_local_coords else self.xyz_source_ft
        return write_cloud(Path(filepath), pts, intensity=self.intensity, classification=self.classification)

    def save_ply(self, filepath: Union[str, Path], use_local_coords: bool = True) -> None:
        """Save point cloud as an ASCII PLY file (the old format; ``save_laz`` is smaller)."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        pts = self.xyz_local_m if use_local_coords else self.xyz_source_ft
        n_pts = len(pts)

        header = (
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {n_pts}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar intensity\n"
            "property uchar classification\n"
            "end_header\n"
        )

        with open(p, "w", encoding="utf-8") as f:
            f.write(header)
            for i in range(n_pts):
                inte = int(np.clip(self.intensity[i] / 256.0, 0, 255))
                cls_ = int(self.classification[i])
                f.write(f"{pts[i, 0]:.4f} {pts[i, 1]:.4f} {pts[i, 2]:.4f} {inte} {cls_}\n")


class LidarSubsampler:
    """Extract and transform LiDAR point clouds for an AOI."""

    def __init__(self, coord_frame: LocalCoordinateFrame):
        self.coord_frame = coord_frame

    def extract_aoi_from_laz(
        self,
        laz_filepath: Union[str, Path],
        aoi_bounds_source_ft: Tuple[float, float, float, float, float, float],
    ) -> LidarCloud:
        """Extract points from a LAZ file within the source bounds (minx, maxx, miny, maxy, minz, maxz)."""
        minx, maxx, miny, maxy, minz, maxz = aoi_bounds_source_ft
        las = laspy.read(laz_filepath)

        x, y, z = np.array(las.x), np.array(las.y), np.array(las.z)
        mask = (
            (x >= minx)
            & (x <= maxx)
            & (y >= miny)
            & (y <= maxy)
            & (z >= minz)
            & (z <= maxz)
        )

        pts_source = np.column_stack([x[mask], y[mask], z[mask]])
        pts_local = self.coord_frame.to_local(pts_source)

        intensity = np.array(las.intensity[mask]) if hasattr(las, "intensity") else np.zeros(len(pts_source))
        classification = (
            np.array(las.classification[mask])
            if hasattr(las, "classification")
            else np.zeros(len(pts_source), dtype=np.uint8)
        )
        return_num = (
            np.array(las.return_number[mask])
            if hasattr(las, "return_number")
            else np.ones(len(pts_source), dtype=np.uint8)
        )
        num_returns = (
            np.array(las.number_of_returns[mask])
            if hasattr(las, "number_of_returns")
            else np.ones(len(pts_source), dtype=np.uint8)
        )

        return LidarCloud(
            xyz_source_ft=pts_source,
            xyz_local_m=pts_local,
            intensity=intensity,
            classification=classification,
            return_number=return_num,
            number_of_returns=num_returns,
        )


def extract_lidar_aoi(
    laz_filepath: Union[str, Path],
    coord_frame: LocalCoordinateFrame,
    aoi_bounds_source_ft: Tuple[float, float, float, float, float, float],
) -> LidarCloud:
    """Convenience helper to extract and transform LiDAR for an AOI."""
    subsampler = LidarSubsampler(coord_frame)
    return subsampler.extract_aoi_from_laz(laz_filepath, aoi_bounds_source_ft)


# ------------------------------------------------------------- tiled sourcing
#: Bytes of a LAS public header block worth fetching. LAS 1.4's block is 375
#: bytes; 512 leaves room without pulling any point data.
LAS_HEADER_BYTES = 512


@dataclass
class LasHeaderSummary:
    """What a LAS/LAZ public header block says, without reading any points.

    Every KyFromAbove tile is tens of megabytes, so comparing phases by opening
    them is not practical. The public header carries the point count, the bounds
    and the flight date, and a single HTTP range request retrieves it. That is
    what makes a statewide comparison cheap enough to actually run.
    """

    version: str
    created: str
    point_format: int
    compressed: bool
    n_points: int
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]

    @property
    def width(self) -> float:
        return self.x_range[1] - self.x_range[0]

    @property
    def height(self) -> float:
        return self.y_range[1] - self.y_range[0]

    @property
    def z_span(self) -> float:
        """Vertical extent, in the file's own units.

        A useful blunder detector. Real terrain inside a 5000 ft Kentucky tile
        spans a few hundred feet; the 2017 Phase 1 tiles span thousands, because
        they retain returns from birds and cloud that later phases do not.
        """
        return self.z_range[1] - self.z_range[0]

    def density_per_m2(self, unit_metres: float = 0.3048006096) -> float:
        """Points per square metre, from the header's own bounding box.

        ``unit_metres`` converts the file's horizontal unit; KyFromAbove is
        EPSG:3089, US survey feet.
        """
        area = (self.width * unit_metres) * (self.height * unit_metres)
        return self.n_points / area if area > 0 else float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version, "created": self.created,
            "point_format": self.point_format, "compressed": self.compressed,
            "n_points": self.n_points, "x_range": list(self.x_range),
            "y_range": list(self.y_range), "z_range": list(self.z_range),
            "density_per_m2": self.density_per_m2(), "z_span": self.z_span,
        }


def read_las_header(source: Union[str, Path], timeout: float = 60.0) -> LasHeaderSummary:
    """Parse a LAS/LAZ public header from a URL (range request) or local path.

    Deliberately hand-rolled against the LAS specification rather than routed
    through laspy: laspy opens the file, and the entire point of this is not to.
    """
    import struct

    want = LAS_HEADER_BYTES
    text = str(source)
    if text.startswith(("http://", "https://")):
        import urllib.request

        req = urllib.request.Request(text, headers={"Range": f"bytes=0-{want - 1}"})
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            raw = fh.read()
    else:
        raw = Path(source).read_bytes()[:want]

    if raw[:4] != b"LASF":
        raise ValueError(f"{source}: not a LAS/LAZ file (no LASF signature)")
    if len(raw) < 375:
        raise ValueError(f"{source}: header truncated at {len(raw)} bytes")

    major, minor = raw[24], raw[25]
    day, year = struct.unpack_from("<HH", raw, 90)
    raw_format = raw[104]
    # Bit 7 of the point data record format is laszip's compression flag.
    compressed = bool(raw_format & 0x80)
    point_format = raw_format & 0x3F
    legacy_n, = struct.unpack_from("<I", raw, 107)
    max_x, min_x, max_y, min_y, max_z, min_z = struct.unpack_from("<6d", raw, 179)

    n_points = legacy_n
    if minor >= 4:
        # LAS 1.4 moved the count to a 64-bit field; the legacy one is zero for
        # point formats above 5, so falling back to it silently reports 0.
        n64, = struct.unpack_from("<Q", raw, 247)
        n_points = int(n64) if n64 else legacy_n

    created = ""
    if year:
        import datetime

        try:
            created = (datetime.date(year, 1, 1)
                       + datetime.timedelta(days=max(day, 1) - 1)).isoformat()
        except (ValueError, OverflowError):
            created = str(year)

    return LasHeaderSummary(
        version=f"{major}.{minor}", created=created, point_format=point_format,
        compressed=compressed, n_points=int(n_points),
        x_range=(min_x, max_x), y_range=(min_y, max_y), z_range=(min_z, max_z),
    )


def open_copc(tile_or_url, timeout: float = 60.0):
    """Open a COPC reader, tolerating the grid's mis-spelled URLs.

    Measured against the published Phase 2 grid: **10% of its 898 ``.copz.laz``
    entries return 404, and every one of them exists under ``.copc.laz``** --
    roughly 90 tiles statewide. None of the 45,269 ``.copc.laz`` entries fail.
    So the spelling is not merely inconsistent, as
    :class:`LidarTileIndex` already handles; for some tiles it is simply wrong.

    Trying the other spelling on failure costs one HEAD-shaped round trip on the
    ~2% of tiles that need it, and turns a hard 404 into a transparent success.
    """
    import laspy

    url = getattr(tile_or_url, "url", tile_or_url)
    candidates = [url]
    if ".copz." in url:
        candidates.append(url.replace(".copz.", ".copc."))
    elif ".copc." in url:
        candidates.append(url.replace(".copc.", ".copz."))

    last = None
    for candidate in candidates:
        try:
            return with_retries(lambda: laspy.CopcReader.open(candidate))
        except Exception as exc:                       # noqa: BLE001
            last = exc
            if is_transient(exc):
                raise                                  # still busy after the retries: not a wrong spelling
    raise FileNotFoundError(
        f"no COPC readable for {url!r}; tried {len(candidates)} spelling(s)"
    ) from last


def is_transient(exc: BaseException) -> bool:
    """An error worth retrying: S3 asking to slow down (503), a gateway or
    server hiccup, a dropped connection or a timeout -- not a 404."""
    text = f"{type(exc).__name__} {exc}"
    return any(s in text for s in ("503", "SlowDown", "Service Unavailable", "500 Server Error",
                                   "502", "504", "ConnectionError", "ConnectTimeout", "ReadTimeout",
                                   "Connection reset", "RemoteDisconnected", "IncompleteRead"))


def with_retries(call, tries: int = 7, first_wait_s: float = 0.5, cap_s: float = 20.0):
    """``call()``, retried with exponential back-off and jitter on transient
    errors: what S3 asks of a caller that gets 503 Slow Down. Forty-five
    Lambdas reading one tile's COPC file at once got 503s on 7 of 1,121
    buildings (September 2026)."""
    import random
    import time

    for attempt in range(tries):
        try:
            return call()
        except Exception as exc:                       # noqa: BLE001
            if attempt == tries - 1 or not is_transient(exc):
                raise
            time.sleep(min(cap_s, first_wait_s * 2 ** attempt) * (0.5 + random.random()))


#: Readers kept open for the life of the process (a Lambda container): each
#: building's clip would otherwise re-read the file's header and hierarchy.
_READERS: Dict[str, Any] = {}


def _close_reader(r) -> None:
    """Close a COPC reader: laspy's ``CopcReader`` closes through ``__exit__``
    (it has no ``close``); a failure to close is not worth failing a read."""
    try:
        r.__exit__(None, None, None)
    except Exception:                                           # noqa: BLE001
        pass


def copc_reader(url: str):
    """A COPC reader for ``url``, opened once per process and reused (the
    16 most recent)."""
    if url not in _READERS:
        if len(_READERS) >= 16:
            _close_reader(_READERS.pop(next(iter(_READERS))))
        _READERS[url] = open_copc(url)
    return _READERS[url]


def copc_query(url: str, bounds, resolution=None):
    """Points of ``url`` inside ``bounds`` (optionally capped at a
    ``resolution`` in the file's units), with retries; a reader that failed
    mid-read is dropped and opened afresh."""
    def once():
        try:
            r = copc_reader(url)
            return r.query(bounds) if resolution is None else r.query(bounds, resolution=resolution)
        except Exception:
            r = _READERS.pop(url, None)
            if r is not None:
                _close_reader(r)
            raise
    return with_retries(once)


@dataclass
class LidarTile:
    """One KyFromAbove 5k point-cloud tile."""

    tilename: str
    key: str
    url: str
    phase: str
    is_copc: bool = False
    size_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tilename": self.tilename, "key": self.key, "url": self.url,
            "phase": self.phase, "is_copc": self.is_copc,
            "size_bytes": self.size_bytes,
        }


class LidarTileIndex:
    """Resolves AOIs to LiDAR tiles using the published 5k grid GeoPackages.

    .. note::
       The grids are in **EPSG:3857**, not the EPSG:3089 the rest of this
       project works in, and the Phase 2 grid ships with **no CRS set at all**.
       Comparing an EPSG:3089 AOI against them directly silently returns no
       tiles rather than raising, so both are handled explicitly here.
    """

    GRID_CRS = "EPSG:3857"

    #: Preference order, measured rather than assumed. Phase 2 is COPC statewide
    #: (46,167 tiles) and is the densest everywhere the phases overlap: 4.4 and
    #: 4.8 pts/m^2 over built-up Ashland and Pikeville against Phase 3's 2.7 and
    #: 3.8, despite Phase 3 being ten months newer. Phase 3 is a re-flight of
    #: eastern Kentucky, not new ground -- 6,846 of its 6,847 tiles are already
    #: in Phase 2. Phase 1 is last because it is plain LAZ: no range queries, so
    #: a whole tile must be fetched to read a building out of it, and it is 2017
    #: rather than 2024. One tile (N052E324) is Phase 3 + Phase 1 only, which is
    #: the sole case where this ordering changes the answer.
    PHASE_PREFERENCE = ("phase2", "phase3", "phase1")

    def __init__(self, raw_dir: Union[str, Path], phases: Sequence[str] = PHASE_PREFERENCE):
        import geopandas as gpd

        self.raw_dir = Path(raw_dir)
        self.grids: Dict[str, Any] = {}
        for phase in phases:
            path = self.raw_dir / f"kyfromabove_{phase}_pointcloud_5k_grid.gpkg"
            if not path.exists():
                continue
            grid = gpd.read_file(path)
            if grid.crs is None:
                grid = grid.set_crs(self.GRID_CRS)
            grid = grid.to_crs(self.GRID_CRS)
            b = grid.geometry.bounds.to_numpy()
            bad = ~np.isfinite(b).all(axis=1)
            if bad.any():
                # KyFromAbove's Phase 3 list of 2026-09-15 has every shape at infinity: no area
                # would meet a tile, and every read would quietly find no LiDAR
                raise ValueError(f"{path.name}: {int(bad.sum())} of {len(grid)} tile shapes are not finite "
                                 f"(e.g. {grid.tilename[bad].iloc[0]}); repair it with scripts/repair_tile_grid.py")
            self.grids[phase] = grid

    @property
    def phases(self) -> List[str]:
        return list(self.grids)

    def tiles_for_bounds(
        self,
        bounds_source_ft: Tuple[float, float, float, float],
        phase: Optional[str] = None,
        source_crs: str = "EPSG:3089",
    ) -> List[LidarTile]:
        """Tiles covering (minx, miny, maxx, maxy) given in ``source_crs``."""
        from shapely.geometry import box
        from shapely.ops import transform
        from pyproj import Transformer

        to_grid = Transformer.from_crs(source_crs, self.GRID_CRS, always_xy=True).transform
        aoi = transform(to_grid, box(*bounds_source_ft))

        for name in ([phase] if phase else self.phases):
            grid = self.grids.get(name)
            if grid is None:
                continue
            hit = grid[grid.intersects(aoi)]
            if len(hit) == 0:
                continue
            tiles = []
            for _, row in hit.iterrows():
                size = row.get("size")
                tiles.append(
                    LidarTile(
                        tilename=str(row["tilename"]),
                        key=str(row["key"]),
                        url=str(row["aws_url"]),
                        phase=name,
                        # KyFromAbove writes Phase 2/3 as COPC, spelled .copz.laz
                        # in some index rows and .copc.laz in others.
                        is_copc=(".copc." in row["key"] or ".copz." in row["key"]),
                        size_bytes=int(size) if size not in (None, "") else None,
                    )
                )
            return tiles
        return []


def _cloud_from_las(las_points, coord_frame) -> LidarCloud:
    xyz = np.column_stack([np.array(las_points.x), np.array(las_points.y), np.array(las_points.z)])
    return LidarCloud(
        xyz_source_ft=xyz,
        xyz_local_m=coord_frame.to_local(xyz),
        intensity=np.array(las_points.intensity),
        classification=np.array(las_points.classification),
        return_number=np.array(las_points.return_number),
        number_of_returns=np.array(las_points.number_of_returns),
    )


def concat_clouds(clouds: Sequence[LidarCloud]) -> LidarCloud:
    """Merge point clouds, e.g. the parts of one AOI that fall on different tiles."""
    clouds = [c for c in clouds if len(c) > 0]
    if not clouds:
        empty = np.zeros((0, 3))
        z = np.zeros(0)
        return LidarCloud(empty, empty, z, z, z, z)
    if len(clouds) == 1:
        return clouds[0]
    return LidarCloud(
        xyz_source_ft=np.vstack([c.xyz_source_ft for c in clouds]),
        xyz_local_m=np.vstack([c.xyz_local_m for c in clouds]),
        intensity=np.concatenate([c.intensity for c in clouds]),
        classification=np.concatenate([c.classification for c in clouds]),
        return_number=np.concatenate([c.return_number for c in clouds]),
        number_of_returns=np.concatenate([c.number_of_returns for c in clouds]),
    )


#: the survey's vegetation classes (ASPRS low, medium, high)
VEGETATION = (3, 4, 5)
#: the class :func:`label_trees` gives a tree's returns (ASPRS high vegetation)
TREE_CLASS = 5
#: the tree rule's version: a building's clip cut before it is cut again
#: (prepare_piece's clip.json)
TREE_RULE = 3


def label_trees(xyz_m: np.ndarray, cls: np.ndarray, number_of_returns: np.ndarray, smooth_m: float = 0.08,
                k: int = 12, max_gap_m: float = 2.0, vote: float = 0.3, split_share: float = 0.6,
                max_tree_m: float = 40.0, rejoin_m: float = 0.15, k_rejoin: int = 5, patch_link_m: float = 1.5,
                min_patch: int = 40, patch_split_share: float = 0.5, edge_m: float = 2.0, edge_up_m: float = 1.0,
                chunk: int = 500_000) -> np.ndarray:
    """``cls`` with the unclassified returns (1) that are bare branches
    relabeled trees (:data:`TREE_CLASS`); ``xyz_m`` in meters,
    ``number_of_returns`` each return's pulse's echo count.

    The state's Phase 2 LiDAR labels ground and noise only: no tree and no
    building class on any of the 302 published tiles. Every step that leaves
    vegetation out left nothing out, and a tree over a house was its roof: in
    St. Matthews 41 of 218 houses measured over 15 m, 81288b3f 24.9 m for a
    6.9 m roof. Flown leaves-off, a crown is a scatter whose pulses split:

    * rough: fewer than ``vote`` of a return's ``k`` nearest unclassified
      neighbors are smooth -- they and their own ``k - 1`` nearest (within
      ``max_gap_m``) lie within ``smooth_m`` of one plane, at any angle, so
      walls stay; a ridge, where two slopes meet in no one plane, is carried
      by the smooth slopes either side;
    * split: at least ``split_share`` of those neighbors' pulses gave more
      than one echo. Rough alone took the Cathedral of the Assumption's spire
      (its top 84 m to 50 m) and up to half a glass tower's own returns: stone,
      glass and rooftop plant are rough, but their pulses mostly end in one
      echo (26-46% split, against 83% in St. Matthews' crowns);
    * low: no more than ``max_tree_m`` over the nearest ground return. A
      lattice mast splits pulses as a crown does: the Kentucky Home Life
      Building's, 70 to 94 m up, and PNC Tower's."""
    from scipy.spatial import cKDTree

    cls = np.asarray(cls).copy()
    u = np.flatnonzero(cls == 1)
    if len(u) < k:
        return cls
    xyz_m = np.asarray(xyz_m, dtype=np.float64)
    o = xyz_m[u].mean(axis=0)
    P = xyz_m[u] - o
    split = np.asarray(number_of_returns)[u] > 1
    g = np.flatnonzero(cls == 2)
    if len(g):
        G = xyz_m[g] - o
        _, gi = cKDTree(G[:, :2]).query(P[:, :2], workers=-1)
        low = P[:, 2] - G[gi, 2] <= max_tree_m
    else:
        low = np.ones(len(P), dtype=bool)
    tree = cKDTree(P)
    smooth = np.empty(len(P), dtype=bool)
    for s in range(0, len(P), chunk):
        d, ii = tree.query(P[s:s + chunk], k=k, workers=-1)
        Q = P[ii]
        Q = Q - Q.mean(axis=1, keepdims=True)
        spread = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", Q, Q) / k)[:, 0]
        smooth[s:s + chunk] = (np.sqrt(np.maximum(spread, 0.0)) < smooth_m) & (d[:, -1] <= max_gap_m)
    # the vote, a second query rather than every neighbor list kept (a
    # tile's area holds tens of millions of returns)
    is_tree = np.empty(len(P), dtype=bool)
    branchy = np.empty(len(P), dtype=bool)
    for s in range(0, len(P), chunk):
        _, ii = tree.query(P[s:s + chunk], k=k, workers=-1)
        branchy[s:s + chunk] = (split[ii].mean(axis=1) >= split_share) & low[s:s + chunk]
        is_tree[s:s + chunk] = (smooth[ii].mean(axis=1) < vote) & branchy[s:s + chunk]
    # A roof return with crown returns among its neighbors fits no plane and
    # was called a tree (18% of a test roof under a crown): it rejoins the
    # surface when it lies within ``rejoin_m`` of the plane through its k
    # nearest kept neighbors, those smooth themselves. A crown return far
    # off any surface stays a tree.
    cand, kept = np.flatnonzero(is_tree), np.flatnonzero(~is_tree)
    if len(cand) and len(kept) >= k_rejoin:
        ktree = cKDTree(P[kept])
        for s in range(0, len(cand), chunk):
            c = cand[s:s + chunk]
            d, ii = ktree.query(P[c], k=k_rejoin, workers=-1)
            Q = P[kept[ii]]
            mu = Q.mean(axis=1, keepdims=True)
            w, v = np.linalg.eigh(np.einsum("nki,nkj->nij", Q - mu, Q - mu) / k_rejoin)
            off = np.abs(np.einsum("ni,ni->n", P[c] - mu[:, 0], v[:, :, 0]))
            ok = (np.sqrt(np.maximum(w[:, 0], 0.0)) < smooth_m) & (d[:, -1] <= max_gap_m) & (off < rejoin_m)
            is_tree[c[ok]] = False
    # A few branches can lie smooth by chance: a clump of a dozen returns,
    # standing alone, set 6453e105's top at 18.7 m for an 8.7 m roof. A
    # roof is one surface of hundreds: a kept patch (returns linked within
    # ``patch_link_m`` of one of their ``k_rejoin`` nearest) of fewer than
    # ``min_patch`` returns -- about 10 m2 at the survey's 4 a square metre
    # -- is a tree, when its own pulses split (``patch_split_share``) or its
    # neighborhood's do, and it stands low enough: a piece of spire does not.
    kept = np.flatnonzero(~is_tree)
    if len(kept) > k_rejoin:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        d, ii = cKDTree(P[kept]).query(P[kept], k=k_rejoin + 1, workers=-1)
        link = d[:, 1:] < patch_link_m
        rows = np.repeat(np.arange(len(kept)), k_rejoin)[link.ravel()]
        _, lab = connected_components(coo_matrix((np.ones(len(rows)), (rows, ii[:, 1:][link])),
                                                 shape=(len(kept), len(kept))), directed=False)
        n = np.bincount(lab)
        clump_split = (np.bincount(lab, weights=split[kept]) / n)[lab] >= patch_split_share
        small = (n[lab] < min_patch) & (clump_split | branchy[kept]) & low[kept]
        is_tree[kept[small]] = True
    # A building's edges -- parapet, cornice, eave -- fit no one plane and a
    # pulse that clips one splits: 81% of what the rule called a tree on the
    # Courtyard Louisville Downtown lay within 1.5 m of its outline, and its
    # walls came out short of their parapets (holes 97 -> 244 m2). An edge
    # stands at or below the surface beside it, a crown that matters over
    # it: a return with kept smooth returns within ``edge_m`` across, and no
    # more than ``edge_up_m`` above the highest of them, rejoins (once, not
    # chained: a crown's lowest tips just over a roof come back, adding at
    # most that to a house's top, never the crown).
    cand, kept = np.flatnonzero(is_tree), np.flatnonzero(~is_tree & smooth)     # surfaces, not a crown's leftovers
    if len(cand) and len(kept) >= 8:
        d, ii = cKDTree(P[kept, :2]).query(P[cand, :2], k=8, distance_upper_bound=edge_m, workers=-1)
        near = np.isfinite(d)
        top = np.where(near, P[kept[np.minimum(ii, len(kept) - 1)], 2], -np.inf).max(axis=1)
        is_tree[cand[near.any(axis=1) & (P[cand, 2] <= top + edge_up_m)]] = False
    cls[u[is_tree]] = TREE_CLASS
    return cls


def extract_lidar_for_aois(
    aois: Dict[str, Tuple[float, float, float, float, float, float]],
    coord_frame,
    index: LidarTileIndex,
    phase: Optional[str] = None,
    local_dir: Optional[Union[str, Path]] = None,
    allow_missing_tiles: bool = False,
    progress: bool = True,
) -> Dict[str, LidarCloud]:
    """Extract per-AOI point clouds, reading each source tile at most once.

    ``aois`` maps an id to ``(minx, maxx, miny, maxy, minz, maxz)`` in source
    feet. Two strategies, chosen per tile:

    * **COPC tiles** (Phase 2 and 3) support HTTP range queries, so each AOI is
      fetched directly and no whole tile is ever read.
    * **Plain LAZ tiles** (Phase 1) must be read whole. Reading one per *AOI*
      costs 28 MB every time; grouping AOIs by tile and reading once amortises
      that across every building on the tile, which is the difference between
      linear-in-buildings and linear-in-tiles.

    An AOI that straddles a tile boundary is read from **every** covering tile
    and the parts merged. Taking only the first covering tile silently returns a
    truncated cloud, which downstream looks like a sparse building rather than
    an error.

    Raises if a required plain-LAZ tile is not present locally, unless
    ``allow_missing_tiles``; a partially-covered AOI is worse than a loud
    failure, so the default is to complain.
    """
    import laspy

    # An AOI may appear under several tiles; invert to tile -> AOIs.
    by_tile: Dict[Tuple[str, str], List[str]] = {}
    tile_meta: Dict[Tuple[str, str], LidarTile] = {}
    tiles_per_aoi: Dict[str, int] = {}
    unresolved: List[str] = []

    for aoi_id, b in aois.items():
        minx, maxx, miny, maxy = b[0], b[1], b[2], b[3]
        tiles = index.tiles_for_bounds((minx, miny, maxx, maxy), phase=phase)
        if not tiles:
            unresolved.append(aoi_id)
            continue
        tiles_per_aoi[aoi_id] = len(tiles)
        for tile in tiles:
            key = (tile.phase, tile.tilename)
            by_tile.setdefault(key, []).append(aoi_id)
            tile_meta[key] = tile

    if unresolved and progress:
        print(f"  no tile coverage for {len(unresolved)} AOI(s): {unresolved[:5]}")

    parts: Dict[str, List[LidarCloud]] = {aoi_id: [] for aoi_id in tiles_per_aoi}
    missing: Dict[str, List[str]] = {}

    for key, aoi_ids in by_tile.items():
        tile = tile_meta[key]
        local = Path(local_dir) / Path(tile.key).name if local_dir else None
        source = str(local) if local and local.exists() else tile.url

        if tile.is_copc:
            for aoi_id in aoi_ids:
                minx, maxx, miny, maxy, minz, maxz = aois[aoi_id]
                pts = copc_query(source, laspy.copc.Bounds(mins=np.array([minx, miny, minz]),
                                                           maxs=np.array([maxx, maxy, maxz])))
                parts[aoi_id].append(_cloud_from_las(pts, coord_frame))
            if progress:
                print(f"  {tile.tilename} ({tile.phase}, COPC): {len(aoi_ids)} AOI(s) "
                      f"by range query, no full-tile read")
        else:
            if local is None or not local.exists():
                for aoi_id in aoi_ids:
                    missing.setdefault(aoi_id, []).append(tile.tilename)
                if progress:
                    print(f"  {tile.tilename} ({tile.phase}, plain LAZ): NOT LOCAL, skipped")
                continue
            las = laspy.read(str(local))
            whole = _cloud_from_las(las, coord_frame)
            for aoi_id in aoi_ids:
                parts[aoi_id].append(whole.filter_by_bounds_source(aois[aoi_id]))
            if progress:
                mb = local.stat().st_size / 1048576
                print(f"  {tile.tilename} ({tile.phase}, plain LAZ): read once "
                      f"({mb:.0f} MB) for {len(aoi_ids)} AOI(s)")

    if missing and not allow_missing_tiles:
        detail = "; ".join(f"{k} needs {v}" for k, v in list(missing.items())[:5])
        raise FileNotFoundError(
            f"{len(missing)} AOI(s) would be truncated by missing local plain-LAZ "
            f"tiles ({detail}). Fetch them, use a COPC phase, or pass "
            f"allow_missing_tiles=True to accept partial clouds."
        )

    out = {aoi_id: concat_clouds(p) for aoi_id, p in parts.items()}
    for c in out.values():                      # the survey labels no trees (label_trees)
        c.classification = label_trees(c.xyz_local_m, c.classification, c.number_of_returns)
    return out
