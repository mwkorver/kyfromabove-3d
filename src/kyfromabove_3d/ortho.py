"""Orthoimagery access, and a check for what kind of ortho you actually have.

A **true ortho** is rectified through a surface model that includes buildings,
so every surface lands at its own X, Y and a roof sits directly over its
footprint. A **standard ortho** is rectified through a bare-earth DTM, so only
the ground is correct: a roof is displaced radially by ``height * tan(theta)``,
where theta is the off-nadir angle of whichever frame contributed that patch of
mosaic.

The difference decides whether an ortho can be segmented for building
footprints. It usually is not documented, it cannot be recovered from the ortho
itself -- there is no camera and no record of the contributing frame -- and it
is invisible unless you measure it. On the KyFromAbove 2022 Season 2 3-inch
ortho this measures 174 cm of roof displacement on a 5.6 m building.

:func:`assess_ortho_rectification` measures it, using the ground as its own
control: if ground patches do not rectify to within a pixel or so, the
measurement itself is untrustworthy and the verdict is withheld rather than
guessed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from .imagery import GDAL_ENV

# KyFromAbove ortho tiles: 5000 ft square on a regular grid in EPSG:3089, with
# tile N037E390 anchored at this upper-left corner. Derived from the 39-byte
# .tfw sidecars rather than the 45 MB grid index -- and note that
# imagery/orthos/tile-grids/imagery_grid.fgb is mislabelled: it indexes Phase 3
# LiDAR, not orthoimagery.
KY_ORTHO_ORIGIN_X_FT = 5725000.125
KY_ORTHO_ORIGIN_Y_FT = 4179999.875
KY_ORTHO_ORIGIN_E = 390
KY_ORTHO_ORIGIN_N = 37
KY_ORTHO_TILE_FT = 5000.0


def ortho_tilename(x_ft: float, y_ft: float) -> str:
    """Tile name covering an EPSG:3089 point, e.g. ``N067E390``."""
    e = KY_ORTHO_ORIGIN_E + math.floor((x_ft - KY_ORTHO_ORIGIN_X_FT) / KY_ORTHO_TILE_FT)
    n = KY_ORTHO_ORIGIN_N + math.floor((KY_ORTHO_ORIGIN_Y_FT - y_ft) / KY_ORTHO_TILE_FT)
    return f"N{n:03d}E{e:03d}"


def ortho_tiles_for_bounds(
    minx_ft: float, miny_ft: float, maxx_ft: float, maxy_ft: float
) -> List[str]:
    """Every tile touching an EPSG:3089 extent. AOIs do straddle tile edges."""
    names = {
        ortho_tilename(x, y)
        for x in (minx_ft, maxx_ft)
        for y in (miny_ft, maxy_ft)
    }
    return sorted(names)


@dataclass
class OrthoCollection:
    """One published ortho collection, e.g. the 2022 Season 2 3-inch mosaic."""

    prefix: str = "imagery/orthos/Phase3/KY_KYAPED_2022_Season2_3IN"
    suffix: str = "_2022_Season2_3IN_cog.tif"
    bucket_url: str = "https://kyfromabove.s3.amazonaws.com"

    def url_for_tile(self, tilename: str) -> str:
        return f"{self.bucket_url}/{self.prefix}/{tilename}{self.suffix}"

    def url_for_point(self, x_ft: float, y_ft: float) -> str:
        return self.url_for_tile(ortho_tilename(x_ft, y_ft))


class OrthoReader:
    """Windowed reads from a georeferenced ortho tile."""

    def __init__(self, path_or_url: Union[str, Path]):
        self.path_or_url = str(path_or_url)
        with rasterio.Env(**GDAL_ENV):
            with rasterio.open(self.path_or_url) as src:
                self.crs = src.crs
                self.res = src.res
                self.bounds = tuple(src.bounds)
                self.width, self.height = src.width, src.height

    def read_bounds(
        self, minx: float, miny: float, maxx: float, maxy: float
    ) -> Tuple[np.ndarray, Any, np.ndarray]:
        """Read a window by map bounds; returns ``(rgb, transform, valid)``.

        .. warning::
           The read is **boundless**. Without that, ``src.read`` silently clips a
           window at the tile edge while ``window_transform`` still describes the
           unclipped origin, so the array and its georeferencing disagree by
           exactly the clipped amount. At a tile boundary this produced an 862 px
           (65 m) shift with no error and no warning. ``valid`` marks the
           zero-filled region that fell outside the tile.
        """
        with rasterio.Env(**GDAL_ENV):
            with rasterio.open(self.path_or_url) as src:
                window = from_bounds(minx, miny, maxx, maxy, src.transform)
                arr = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
                transform = src.window_transform(window)
        rgb = np.transpose(arr, (1, 2, 0))
        return rgb, transform, rgb.sum(axis=2) > 0

    def __repr__(self) -> str:
        return f"OrthoReader({Path(self.path_or_url).name}, {self.width}x{self.height}, {self.crs})"


@dataclass
class OrthoAssessment:
    """Result of measuring how an ortho rectifies elevated surfaces."""

    n_ground: int
    n_roof: int
    ground_displacement_m: float
    roof_displacement_m: float
    control_passed: bool
    ground_scatter_m: float
    roof_scatter_m: float
    verdict: str                      # true_ortho | standard_ortho | inconclusive
    reason: str
    building_height_m: Optional[float] = None
    implied_off_nadir_deg: Optional[float] = None
    rejections: Dict[str, int] = field(default_factory=dict)

    @property
    def roof_minus_ground_m(self) -> float:
        return self.roof_displacement_m - self.ground_displacement_m

    @property
    def usable_for_footprints(self) -> bool:
        """Only a true ortho can be segmented directly for building footprints."""
        return self.verdict == "true_ortho"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict, "reason": self.reason,
            "control_passed": bool(self.control_passed),
            "usable_for_footprints": self.usable_for_footprints,
            "n_ground": int(self.n_ground), "n_roof": int(self.n_roof),
            "ground_displacement_m": float(self.ground_displacement_m),
            "roof_displacement_m": float(self.roof_displacement_m),
            "ground_scatter_m": float(self.ground_scatter_m),
            "roof_scatter_m": float(self.roof_scatter_m),
            "roof_minus_ground_m": float(self.roof_minus_ground_m),
            "building_height_m": self.building_height_m,
            "implied_off_nadir_deg": self.implied_off_nadir_deg,
            "rejections": self.rejections,
        }

    def __repr__(self) -> str:
        return (
            f"OrthoAssessment({self.verdict}, ground={self.ground_displacement_m*100:.1f} cm, "
            f"roof={self.roof_displacement_m*100:.1f} cm, n={self.n_ground}+{self.n_roof})"
        )

def assess_ortho_rectification(
    ortho: Union[str, Path, OrthoReader],
    reference_camera,
    reference_window,
    anchors: Sequence,
    coord_frame,
    building_height_m: Optional[float] = None,
    patch_radius_m: float = 5.0,
    search_radius_m: float = 5.0,
    gsd_m: float = 0.08,
    min_ncc: float = 0.5,
    min_texture_std: float = 8.0,
    max_ground_displacement_m: float = 0.30,
    min_anchors: int = 10,
    margin_ft: float = 500.0,
) -> OrthoAssessment:
    """Measure whether an ortho rectifies roofs like it rectifies the ground.

    Each anchor's surroundings are sampled twice: through ``reference_camera``
    (a real exposure with a validated model) and through the ortho's map
    projection, which places content by X, Y alone. Correlating the two gives a
    displacement per anchor.

    Ground anchors are the control. A DTM-based ortho rectifies the ground
    correctly, so ground displacement near zero confirms the measurement is
    sound; if it is not near zero, something else is wrong (CRS, tile, camera)
    and no verdict is issued.

    Args:
        ortho: tile path/URL, or an :class:`OrthoReader`.
        reference_camera: a validated exposure, ideally the nadir head.
        reference_window: an :class:`~kyfromabove_3d.imagery.ImageWindow`
            covering the anchors in that exposure.
        anchors: :class:`~kyfromabove_3d.features.PlanarAnchor` list, labelled
            ``"roof"`` and ``"ground"``.
        building_height_m: used only to report an implied off-nadir angle.
        max_ground_displacement_m: control threshold.
    """
    import cv2

    from .qa import sample_plane_patch

    reader = ortho if isinstance(ortho, OrthoReader) else OrthoReader(ortho)

    pts_local = np.array([a.xyz_local_m for a in anchors], dtype=np.float64)
    src_ft = coord_frame.to_source(pts_local)
    rgb, transform, valid = reader.read_bounds(
        src_ft[:, 0].min() - margin_ft, src_ft[:, 1].min() - margin_ft,
        src_ft[:, 0].max() + margin_ft, src_ft[:, 1].max() + margin_ft,
    )
    inv = ~transform

    def ortho_patch(anchor, radius_m: float) -> Optional[np.ndarray]:
        n = int(round(2 * radius_m / gsd_m))
        offs = (np.arange(n) - (n - 1) / 2.0) * gsd_m
        a, b = np.meshgrid(offs, offs)
        grid = (
            anchor.xyz_local_m[None, None, :]
            + a[..., None] * anchor.e1[None, None, :]
            + b[..., None] * anchor.e2[None, None, :]
        ).reshape(-1, 3)
        ft = coord_frame.to_source(grid)          # z deliberately unused
        cols, rows = inv @ (ft[:, 0], ft[:, 1])
        mx = np.asarray(cols, np.float32).reshape(n, n)
        my = np.asarray(rows, np.float32).reshape(n, n)
        if mx.min() < 0 or my.min() < 0 or mx.max() > rgb.shape[1] - 1 or my.max() > rgb.shape[0] - 1:
            return None
        ok = cv2.remap(valid.astype(np.uint8) * 255, mx, my, cv2.INTER_NEAREST) > 127
        if ok.mean() < 0.99:
            return None
        return cv2.remap(rgb, mx, my, cv2.INTER_LINEAR)

    measured: Dict[str, List[float]] = {"ground": [], "roof": []}
    rejections: Dict[str, int] = {}

    def reject(name: str) -> None:
        rejections[name] = rejections.get(name, 0) + 1

    for anchor in anchors:
        if anchor.label not in measured:
            continue
        ref = sample_plane_patch(reference_window, reference_camera, anchor, patch_radius_m, gsd_m)
        if ref is None or ref.valid_fraction < 0.99:
            reject("reference_unusable"); continue
        if ref.texture_std < min_texture_std:
            reject("low_texture"); continue
        tgt = ortho_patch(anchor, patch_radius_m + search_radius_m)
        if tgt is None:
            reject("outside_ortho_tile"); continue

        surface = cv2.matchTemplate(
            cv2.cvtColor(tgt, cv2.COLOR_RGB2GRAY).astype(np.float32),
            ref.gray, cv2.TM_CCOEFF_NORMED,
        )
        _, peak, _, loc = cv2.minMaxLoc(surface)
        if peak < min_ncc:
            reject("low_ncc"); continue
        c, r = int(loc[0]), int(loc[1])
        if r in (0, surface.shape[0] - 1) or c in (0, surface.shape[1] - 1):
            reject("peak_at_search_edge"); continue
        dx = (c - (surface.shape[1] - 1) / 2.0) * gsd_m
        dy = (r - (surface.shape[0] - 1) / 2.0) * gsd_m
        measured[anchor.label].append((dx, dy))

    def summarise(vectors: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Median displacement, and how coherent it is.

        Scatter is the median distance from the median vector. When scatter is
        comparable to the displacement itself, the correlations are not finding
        one consistent shift -- they are noise, which means something other than
        georeferencing is wrong.
        """
        if not vectors:
            return float("nan"), float("nan")
        v = np.asarray(vectors, dtype=np.float64)
        centre = np.median(v, axis=0)
        return (
            float(np.median(np.hypot(v[:, 0], v[:, 1]))),
            float(np.median(np.hypot(v[:, 0] - centre[0], v[:, 1] - centre[1]))),
        )

    n_ground, n_roof = len(measured["ground"]), len(measured["roof"])
    ground, ground_scatter = summarise(measured["ground"])
    roof, roof_scatter = summarise(measured["roof"])

    # Order matters: a failed control means the measurement itself is unsound,
    # which is a more fundamental problem than having too few roof anchors, and
    # reporting the anchor count first sends the reader after the wrong thing.
    if n_ground < min_anchors:
        return OrthoAssessment(
            n_ground, n_roof, ground, roof, False, ground_scatter, roof_scatter,
            "inconclusive",
            f"need >= {min_anchors} ground anchors to validate the measurement, "
            f"got {n_ground}",
            building_height_m, None, rejections,
        )

    control_passed = ground <= max_ground_displacement_m
    if not control_passed:
        # Distinguish a real georeferencing shift from correlations that simply
        # failed. A coherent shift has scatter well below its magnitude; noise
        # has scatter comparable to it, which points at the two images showing
        # different scenes -- most often different acquisition vintages.
        if ground_scatter >= 0.5 * ground:
            why = (f"ground correlations are incoherent (scatter {ground_scatter*100:.0f} cm "
                   f"against a {ground*100:.0f} cm median), so they are noise rather than "
                   f"a shift -- the reference exposure and this ortho most likely show "
                   f"different scenes; check that their acquisition vintages match")
        else:
            why = (f"ground is coherently displaced by {ground*100:.1f} cm, which is a "
                   f"georeferencing disagreement -- check CRS, tile and reference camera")
        return OrthoAssessment(
            n_ground, n_roof, ground, roof, False, ground_scatter, roof_scatter,
            "inconclusive", f"ground control failed: {why}",
            building_height_m, None, rejections,
        )

    if n_roof < min_anchors:
        return OrthoAssessment(
            n_ground, n_roof, ground, roof, control_passed, ground_scatter, roof_scatter,
            "inconclusive",
            f"the measurement is sound (ground {ground*100:.1f} cm) but there are only "
            f"{n_roof} roof anchors, fewer than the {min_anchors} needed to judge how "
            f"elevated surfaces are rectified",
            building_height_m, None, rejections,
        )

    diff = roof - ground
    implied = (
        float(np.degrees(np.arctan(diff / building_height_m)))
        if building_height_m and diff > 0 else None
    )
    if abs(diff) <= max_ground_displacement_m:
        verdict = "true_ortho"
        reason = (f"roofs rectify like the ground ({diff*100:+.1f} cm apart); "
                  f"safe to segment for footprints")
    else:
        verdict = "standard_ortho"
        reason = (f"roofs displaced {diff*100:+.1f} cm relative to the ground, "
                  f"so a segmented roof is not a footprint")
    return OrthoAssessment(
        n_ground, n_roof, ground, roof, control_passed, ground_scatter, roof_scatter,
        verdict, reason, building_height_m, implied, rejections,
    )
