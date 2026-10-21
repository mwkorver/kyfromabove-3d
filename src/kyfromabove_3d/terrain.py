"""The ground under the buildings, for the viewer's terrain.

Buildings are placed at their true ellipsoidal heights (downtown Louisville's
ground is about 104 m above the ellipsoid), so the viewer's ground must be
the same ground or they float or sink. Two steps:

* :func:`tile_ground` -- per LiDAR tile, its ground returns (class 2) read
  at a coarse resolution over the whole tile, gridded (the median of each
  cell; a cell with none, under a building or on water, from the nearest
  that has some), in NAVD88 metres: ``tiles/<tile>/ground.npz``. Part of the
  tile step.
* :class:`GroundMosaic` and :func:`write_heightmap_tiles` -- the tiles'
  grids, shifted to ellipsoidal heights through the geoid (each tile's own
  shift, from placing its centre), written as static Cesium terrain:
  ``heightmap-1.0`` tiles in the geographic scheme plus ``layer.json``.
  Outside the covered tiles a sample takes the nearest tile's height, so the
  coarse levels have no cliffs.
"""

from __future__ import annotations

import io
import json
import struct
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .coordinates import METERS_PER_US_SURVEY_FOOT

#: Heightmap-1.0: 65 x 65 samples, stored as (h + 1000) * 5 in 16 bits.
SAMPLES = 65


def tile_ground(tile: str, grid, index, cell_m: float = 5.0, resolution_ft: float = 6.0,
                log: Callable = print) -> dict:
    """The tile's ground: NAVD88 metres on a ``cell_m`` grid in state-plane
    feet, as an npz-ready dict (x0/y0 of the grid's corner in feet, cell in
    feet, Z [row = y, col = x])."""
    import laspy
    from scipy.ndimage import distance_transform_edt
    from scipy.stats import binned_statistic_2d

    from .lidar import copc_query
    from .pieces import tile_bounds_ft

    b = tile_bounds_ft(grid, tile)
    xs, ys, zs = [], [], []
    for t in index.tiles_for_bounds(b):                            # a single-phase index
        pts = copc_query(t.url, laspy.copc.Bounds(mins=np.array([b[0], b[1], -1e5]), maxs=np.array([b[2], b[3], 1e5])),
                         resolution=resolution_ft)
        g = np.asarray(pts.classification) == 2
        xs.append(np.asarray(pts.x)[g]); ys.append(np.asarray(pts.y)[g]); zs.append(np.asarray(pts.z)[g])
    x, y, z = (np.concatenate(a) if a else np.zeros(0) for a in (xs, ys, zs))
    cell_ft = cell_m / METERS_PER_US_SURVEY_FOOT
    xe = np.arange(b[0], b[2] + cell_ft, cell_ft)
    ye = np.arange(b[1], b[3] + cell_ft, cell_ft)
    if len(z) == 0:
        raise RuntimeError(f"{tile}: no ground returns")
    Z, _, _, _ = binned_statistic_2d(x, y, z * METERS_PER_US_SURVEY_FOOT, statistic="median", bins=[xe, ye])
    Z = Z.T
    valid = np.isfinite(Z)
    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    Z = Z[iy, ix]
    log(f"{tile}: ground {Z.shape[1]} x {Z.shape[0]} cells of {cell_m:g} m from {len(z):,} returns, "
        f"{(~valid).sum():,} filled; {Z.min():.1f}-{Z.max():.1f} m NAVD88")
    return {"x0": float(xe[0]), "y0": float(ye[0]), "cell_ft": float(cell_ft), "Z_navd88_m": Z.astype(np.float32),
            "n_ground": int(len(z)), "filled": int((~valid).sum())}


def save_ground(d: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **d)
    return buf.getvalue()


class GroundMosaic:
    """Ellipsoidal ground heights at any lon/lat, from the tiles' grids.
    ``grounds``: {tile: the dict of :func:`tile_ground`}; ``shifts``: {tile:
    NAVD88 -> ellipsoid metres at its centre}."""

    def __init__(self, grounds: Dict[str, dict], shifts: Dict[str, float]):
        from pyproj import Transformer
        from scipy.spatial import cKDTree

        self.tiles = sorted(grounds)
        self.g = [grounds[t] for t in self.tiles]
        self.shift = np.array([shifts[t] for t in self.tiles])
        import threading

        # a pyproj Transformer is not safe to share between threads (the terrain
        # tiles are made 32 at a time): one per thread
        self._local = threading.local()
        self.lo = np.array([[g["x0"], g["y0"]] for g in self.g])
        self.hi = np.array([[g["x0"] + g["cell_ft"] * g["Z_navd88_m"].shape[1],
                             g["y0"] + g["cell_ft"] * g["Z_navd88_m"].shape[0]] for g in self.g])
        self.tree = cKDTree((self.lo + self.hi) / 2.0)
        # the covered area in lon/lat, a little wide: samples beyond it (a
        # level-0 tile spans half the globe, where state-plane feet are not
        # finite) are clamped to it, and take the nearest tile's edge anyway
        to_ll = Transformer.from_crs("EPSG:3089", "EPSG:4326", always_xy=True).transform
        lon, lat = to_ll(np.r_[self.lo[:, 0], self.hi[:, 0]], np.r_[self.lo[:, 1], self.hi[:, 1]])
        self.bbox = (float(np.min(lon)) - 0.05, float(np.min(lat)) - 0.05, float(np.max(lon)) + 0.05,
                     float(np.max(lat)) + 0.05)

    def heights(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        w, s, e, n = self.bbox
        lon = np.clip(np.asarray(lon, float).ravel(), w, e)
        lat = np.clip(np.asarray(lat, float).ravel(), s, n)
        tf = getattr(self._local, "to_ft", None)
        if tf is None:
            from pyproj import Transformer
            tf = self._local.to_ft = Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True).transform
        x, y = tf(lon, lat)
        x, y = np.asarray(x), np.asarray(y)
        out = np.empty(len(x))
        # inside a tile: that tile (the highest-numbered where grids overlap, as
        # a loop over every tile gave); outside all: the nearest tile, its edge.
        # Only the 9 nearest tiles are tested -- a point inside a grid has that
        # grid's centre among them -- not every tile for every point (statewide,
        # ~41,000 tiles for each of ~450,000 terrain tiles)
        k = min(9, len(self.tiles))
        _, cand = self.tree.query(np.column_stack([x, y]), k=k)
        cand = np.asarray(cand).reshape(len(x), k)
        near = cand[:, 0]
        lo, hi = self.lo[cand], self.hi[cand]
        inside = ((x[:, None] >= lo[..., 0]) & (x[:, None] < hi[..., 0]) &
                  (y[:, None] >= lo[..., 1]) & (y[:, None] < hi[..., 1]))
        best = np.where(inside, cand, -1).max(axis=1)
        which = np.where(best >= 0, best, near)
        for i in np.unique(which):
            m = which == i
            g = self.g[i]
            Z = g["Z_navd88_m"]
            ny, nx = Z.shape
            fx = np.clip((x[m] - g["x0"]) / g["cell_ft"] - 0.5, 0, nx - 1)
            fy = np.clip((y[m] - g["y0"]) / g["cell_ft"] - 0.5, 0, ny - 1)
            ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
            ix1, iy1 = np.minimum(ix + 1, nx - 1), np.minimum(iy + 1, ny - 1)
            tx, ty = fx - ix, fy - iy
            out[m] = ((1 - tx) * (1 - ty) * Z[iy, ix] + tx * (1 - ty) * Z[iy, ix1]
                      + (1 - tx) * ty * Z[iy1, ix] + tx * ty * Z[iy1, ix1]) + self.shift[i]
        return out


def tile_rect(level: int, x: int, y: int) -> Tuple[float, float, float, float]:
    """(west, south, east, north) degrees of a geographic-scheme TMS tile
    (level 0 is two tiles across, y counts from the south)."""
    dlon, dlat = 360.0 / (2 << level), 180.0 / (1 << level)
    w, s = -180.0 + x * dlon, -90.0 + y * dlat
    return w, s, w + dlon, s + dlat


def tiles_over(level: int, bbox: Tuple[float, float, float, float]) -> List[Tuple[int, int]]:
    w, s, e, n = bbox
    dlon, dlat = 360.0 / (2 << level), 180.0 / (1 << level)
    x0, x1 = int((w + 180.0) // dlon), int((e + 180.0) // dlon)
    y0, y1 = int((s + 90.0) // dlat), int((n + 90.0) // dlat)
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def heightmap_tile(sample: Callable, level: int, x: int, y: int, child_mask: int) -> bytes:
    """One heightmap-1.0 tile: 65 x 65 heights from the north-west corner,
    row by row southwards, as (h + 1000) * 5 in uint16; the child mask; a
    water mask of 0 (land)."""
    w, s, e, n = tile_rect(level, x, y)
    lon = np.linspace(w, e, SAMPLES)
    lat = np.linspace(n, s, SAMPLES)
    LON, LAT = np.meshgrid(lon, lat)
    h = sample(LON.ravel(), LAT.ravel())
    v = np.clip(np.round((h + 1000.0) * 5.0), 0, 65535).astype("<u2")
    return v.tobytes() + struct.pack("<BB", child_mask & 0x0F, 0)


def write_heightmap_tiles(mosaic: GroundMosaic, bboxes: Sequence[Tuple[float, float, float, float]],
                          put: Callable[[str, bytes], None], max_level: int = 15,
                          log: Callable = print) -> dict:
    """Static Cesium terrain over ``bboxes`` (lon/lat degrees), levels 0 to
    ``max_level``: ``put(path, bytes)`` for each ``{z}/{x}/{y}.terrain`` and
    ``layer.json``. Returns counts."""
    have: Dict[int, set] = {0: {(0, 0), (1, 0)}}          # CesiumJS starts from both level-0 tiles
    for z in range(1, max_level + 1):
        have[z] = set()
        for bb in bboxes:
            have[z].update(tiles_over(z, bb))
    from concurrent.futures import ThreadPoolExecutor

    def one(zxy):
        z, x, y = zxy
        mask = 0
        if z < max_level:
            for bit, (cx, cy) in ((1, (2 * x, 2 * y)), (2, (2 * x + 1, 2 * y)),
                                  (4, (2 * x, 2 * y + 1)), (8, (2 * x + 1, 2 * y + 1))):
                if (cx, cy) in have[z + 1]:
                    mask |= bit
        put(f"{z}/{x}/{y}.terrain", heightmap_tile(mosaic.heights, z, x, y, mask))
    todo = [(z, x, y) for z in range(max_level + 1) for (x, y) in sorted(have[z])]
    with ThreadPoolExecutor(32) as ex:                               # the uploads wait on the network
        list(ex.map(one, todo))
    n = len(todo)
    available = [[{"startX": 0, "endX": 1, "startY": 0, "endY": 0}]]
    for z in range(1, max_level + 1):
        # one rectangle per bbox at this level
        rects = []
        for bb in bboxes:
            t = tiles_over(z, bb)
            rects.append({"startX": min(a for a, _ in t), "endX": max(a for a, _ in t),
                          "startY": min(b for _, b in t), "endY": max(b for _, b in t)})
        available.append(rects)
    layer = {"tilejson": "2.1.0", "name": "kyfromabove-3d ground", "format": "heightmap-1.0", "version": "1.0.0",
             "scheme": "tms", "tiles": ["{z}/{x}/{y}.terrain"], "projection": "EPSG:4326",
             "bounds": [-180.0, -90.0, 180.0, 90.0], "minzoom": 0, "maxzoom": max_level, "available": available,
             "attribution": "ground from KyFromAbove Phase 2 LiDAR"}
    put("layer.json", json.dumps(layer).encode("utf-8"))
    log(f"terrain: {n} tiles, levels 0-{max_level}")
    return {"tiles": n, "levels": max_level + 1}
