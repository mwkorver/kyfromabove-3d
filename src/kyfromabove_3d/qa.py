"""Measured multi-view QA for the oblique camera model.

The Phase 1 QA this replaces computed its "observed" image position by
projecting the check feature through the very camera it was testing, then
differenced that against a second projection of the same point. The result was
0.000 px by construction and would have stayed 0.000 px with the rotation
convention transposed or the focal length halved.

What is measured here instead:

``measure_cross_view_residuals``
    For each LiDAR anchor, the surrounding image content is rectified onto the
    anchor's locally-fitted surface plane in two different exposures. If the
    camera model is right, the two rectified patches show the same thing at the
    same place. Normalised cross-correlation locates the actual offset. The
    image data decides the answer, so a wrong pose, focal length or datum
    produces a non-zero number.

``triangulate_anchors``
    The matched image positions are triangulated and compared against the LiDAR
    position of the same point. This is a cross-sensor check: it exercises
    absolute scale and the foot-to-metre conversion, which a purely relative
    metric cannot see.

Both are relative to LiDAR and to the delivered exterior orientation. Neither
establishes absolute georeferencing accuracy -- that needs surveyed ground
control, which this dataset does not include.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .cameras import VirtualObliqueCamera
from .features import CheckFeature, PlanarAnchor
from .imagery import FrameImageReader, ImageWindow


# ------------------------------------------------------------ patch sampling
@dataclass
class PlanePatch:
    """Image content resampled onto an anchor's tangent plane.

    Column index maps to the ``e1`` axis, row index to ``e2``; ``gsd_m`` is the
    plane-space size of one canonical pixel. Rectifying both views onto the same
    plane removes the perspective difference between exposures, so correlation
    stays meaningful across the full 18 degrees of view-angle spread.
    """

    patch: np.ndarray  # (N, N, 3) uint8
    valid: np.ndarray  # (N, N) bool
    gsd_m: float
    radius_m: float
    centre_uv: np.ndarray  # projection of the anchor in full-image pixels

    @property
    def gray(self) -> np.ndarray:
        return cv2.cvtColor(self.patch, cv2.COLOR_RGB2GRAY).astype(np.float32)

    @property
    def texture_std(self) -> float:
        if not self.valid.any():
            return 0.0
        return float(self.gray[self.valid].std())

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.mean())


def sample_plane_patch(
    window: ImageWindow,
    camera: VirtualObliqueCamera,
    anchor: PlanarAnchor,
    radius_m: float,
    gsd_m: float,
) -> Optional[PlanePatch]:
    """Rectify image content around ``anchor`` onto its tangent plane."""
    n_px = int(round(2.0 * radius_m / gsd_m))
    if n_px < 8:
        return None

    offs = (np.arange(n_px, dtype=np.float64) - (n_px - 1) / 2.0) * gsd_m
    a, b = np.meshgrid(offs, offs)  # a -> columns (e1), b -> rows (e2)
    pts = (
        anchor.xyz_local_m[None, None, :]
        + a[..., None] * anchor.e1[None, None, :]
        + b[..., None] * anchor.e2[None, None, :]
    ).reshape(-1, 3)

    uv, zc, _ = camera.project(pts)
    if not np.all(zc > 0.1):
        return None

    local = window.to_local_xy(uv)
    map_x = local[:, 0].reshape(n_px, n_px).astype(np.float32)
    map_y = local[:, 1].reshape(n_px, n_px).astype(np.float32)

    patch = cv2.remap(
        window.rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    valid_src = window.valid.astype(np.uint8) * 255
    valid = cv2.remap(
        valid_src, map_x, map_y, interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ) > 127

    in_window = (
        (map_x >= 0) & (map_x <= window.width - 1) & (map_y >= 0) & (map_y <= window.height - 1)
    )
    valid &= in_window

    centre_uv, _, _ = camera.project(anchor.xyz_local_m)
    return PlanePatch(
        patch=patch, valid=valid, gsd_m=gsd_m, radius_m=radius_m, centre_uv=np.asarray(centre_uv)
    )


# -------------------------------------------------------------------- matching
@dataclass
class MatchResult:
    """One measured ref->target correspondence for a single anchor."""

    anchor_id: str
    label: str
    ref_frame: str
    tgt_frame: str
    accepted: bool
    reject_reason: str = ""
    ncc: float = float("nan")
    offset_e1_m: float = float("nan")
    offset_e2_m: float = float("nan")
    residual_m: float = float("nan")
    residual_px: float = float("nan")
    observed_uv: Optional[Tuple[float, float]] = None
    predicted_uv: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "anchor_id": self.anchor_id,
            "label": self.label,
            "ref_frame": self.ref_frame,
            "tgt_frame": self.tgt_frame,
            "accepted": bool(self.accepted),
            "reject_reason": self.reject_reason,
            "ncc": float(self.ncc),
            "offset_e1_m": float(self.offset_e1_m),
            "offset_e2_m": float(self.offset_e2_m),
            "residual_m": float(self.residual_m),
            "residual_px": float(self.residual_px),
        }
        if self.observed_uv is not None:
            d["observed_uv"] = [float(v) for v in self.observed_uv]
        if self.predicted_uv is not None:
            d["predicted_uv"] = [float(v) for v in self.predicted_uv]
        return d


def _subpixel_peak(surface: np.ndarray, r: int, c: int) -> Tuple[float, float]:
    """Parabolic refinement of a correlation peak; falls back to integer."""
    dr = dc = 0.0
    if 0 < r < surface.shape[0] - 1:
        a, b, cc = surface[r - 1, c], surface[r, c], surface[r + 1, c]
        denom = a - 2 * b + cc
        if abs(denom) > 1e-12:
            dr = 0.5 * (a - cc) / denom
    if 0 < c < surface.shape[1] - 1:
        a, b, cc = surface[r, c - 1], surface[r, c], surface[r, c + 1]
        denom = a - 2 * b + cc
        if abs(denom) > 1e-12:
            dc = 0.5 * (a - cc) / denom
    return float(np.clip(dr, -1, 1)), float(np.clip(dc, -1, 1))


def match_anchor_between_views(
    anchor: PlanarAnchor,
    ref_camera: VirtualObliqueCamera,
    ref_window: ImageWindow,
    tgt_camera: VirtualObliqueCamera,
    tgt_window: ImageWindow,
    patch_radius_m: float = 6.0,
    search_radius_m: float = 3.0,
    gsd_m: float = 0.10,
    min_ncc: float = 0.5,
    min_texture_std: float = 8.0,
    min_valid_fraction: float = 0.98,
    min_facing_cos: float = 0.0,
) -> MatchResult:
    """Locate the reference patch inside the target view and measure the offset.

    ``min_facing_cos`` culls back faces. A roof faces up and every camera in an
    aerial survey is above it, so for horizontal anchors this never fires and
    the default of 0.0 preserves existing behaviour exactly. A **facade faces
    one way**: measured on 400 West Market, only 30% of anchor/camera pairs have
    the camera on the visible side, and the other 70% are geometrically
    impossible. Attempting them produces guaranteed ``low_ncc`` rejections that
    depress the acceptance rate and read as the matcher failing rather than as
    the wall being in the way.

    The test needs the anchor's normal to point *out* of the building; see
    ``features._outward_normal``.
    """
    if min_facing_cos > 0.0:
        n = np.asarray(anchor.normal, dtype=np.float64)
        for cam in (ref_camera, tgt_camera):
            d = np.asarray(cam.camera_center_local_m, dtype=np.float64) - np.asarray(
                anchor.xyz_local_m, dtype=np.float64)
            nd = np.linalg.norm(d)
            if nd < 1e-9 or float(n @ d) / nd < min_facing_cos:
                return MatchResult(
                    anchor_id=anchor.anchor_id, label=anchor.label,
                    ref_frame=ref_camera.frame_id, tgt_frame=tgt_camera.frame_id,
                    accepted=False, reject_reason="back_face",
                )
    res = MatchResult(
        anchor_id=anchor.anchor_id,
        label=anchor.label,
        ref_frame=ref_camera.frame_id,
        tgt_frame=tgt_camera.frame_id,
        accepted=False,
    )

    ref = sample_plane_patch(ref_window, ref_camera, anchor, patch_radius_m, gsd_m)
    if ref is None:
        res.reject_reason = "ref_unprojectable"
        return res
    if ref.valid_fraction < min_valid_fraction:
        res.reject_reason = "ref_off_sensor"
        return res
    if ref.texture_std < min_texture_std:
        res.reject_reason = "ref_low_texture"
        return res

    tgt = sample_plane_patch(
        tgt_window, tgt_camera, anchor, patch_radius_m + search_radius_m, gsd_m
    )
    if tgt is None:
        res.reject_reason = "tgt_unprojectable"
        return res
    if tgt.valid_fraction < min_valid_fraction:
        res.reject_reason = "tgt_off_sensor"
        return res

    surface = cv2.matchTemplate(tgt.gray, ref.gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(surface)
    c, r = int(max_loc[0]), int(max_loc[1])
    res.ncc = float(max_val)

    if max_val < min_ncc:
        res.reject_reason = "low_ncc"
        return res
    if r in (0, surface.shape[0] - 1) or c in (0, surface.shape[1] - 1):
        res.reject_reason = "peak_at_search_edge"
        return res

    dr, dc = _subpixel_peak(surface, r, c)
    expected_c = (surface.shape[1] - 1) / 2.0
    expected_r = (surface.shape[0] - 1) / 2.0
    off_a = ((c + dc) - expected_c) * gsd_m
    off_b = ((r + dr) - expected_r) * gsd_m

    res.offset_e1_m = float(off_a)
    res.offset_e2_m = float(off_b)
    res.residual_m = float(np.hypot(off_a, off_b))

    # Express the plane-space offset as a displacement in target image pixels.
    shifted = anchor.xyz_local_m + off_a * anchor.e1 + off_b * anchor.e2
    uv_pred, _, _ = tgt_camera.project(anchor.xyz_local_m)
    uv_obs, _, _ = tgt_camera.project(shifted)
    res.predicted_uv = (float(uv_pred[0]), float(uv_pred[1]))
    res.observed_uv = (float(uv_obs[0]), float(uv_obs[1]))
    res.residual_px = float(np.hypot(uv_obs[0] - uv_pred[0], uv_obs[1] - uv_pred[1]))
    res.accepted = True
    return res


# --------------------------------------------------------------------- report
@dataclass
class ReprojectionReport:
    """Aggregated cross-view residuals."""

    matches: List[MatchResult]
    pair_mode: str = "consecutive"
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> List[MatchResult]:
        return [m for m in self.matches if m.accepted]

    @property
    def residuals_px(self) -> np.ndarray:
        return np.array([m.residual_px for m in self.accepted], dtype=np.float64)

    @property
    def residuals_m(self) -> np.ndarray:
        return np.array([m.residual_m for m in self.accepted], dtype=np.float64)

    def filter_anchors(self, anchor_ids) -> "ReprojectionReport":
        """A copy carrying only matches on the given anchors.

        Used to re-read an existing measurement restricted to the building,
        rather than re-running the matching: the matches are identical, only
        the population being summarised changes.
        """
        keep = set(anchor_ids)
        return replace(self, matches=[m for m in self.matches
                                      if m.anchor_id in keep])

    def stats(self) -> Dict[str, Any]:
        errs = self.residuals_px
        rejects: Dict[str, int] = {}
        for m in self.matches:
            if not m.accepted:
                rejects[m.reject_reason] = rejects.get(m.reject_reason, 0) + 1

        # A back-face rejection is not a failed match. The camera was behind the
        # surface, so the pair was never a candidate, and counting it as a
        # failure is like scoring a photograph that does not contain the
        # building. Measured on 400 West Market: 8,182 of 15,763 attempts, which
        # reads as 33% acceptance against a true 70%. Roof-only AOIs record zero
        # of them, so this leaves the pilot untouched.
        back_face = rejects.get("back_face", 0)
        eligible = max(len(self.matches) - back_face, 0)

        if len(errs) == 0:
            return {
                "attempted": len(self.matches),
                "back_face": back_face,
                "eligible": eligible,
                "accepted": 0,
                "acceptance_rate": 0.0,
                "rejections": rejects,
            }

        per_pair: Dict[str, Dict[str, float]] = {}
        for m in self.accepted:
            key = f"{Path(m.ref_frame).name} -> {Path(m.tgt_frame).name}"
            per_pair.setdefault(key, {"count": 0, "errs": []})
            per_pair[key]["count"] += 1
            per_pair[key]["errs"].append(m.residual_px)
        # Split each pair into a systematic component (a constant offset, which a
        # bundle adjustment can absorb into the pose) and the scatter about it
        # (which it cannot). For cross-flight-line pairs the split is the whole
        # question of whether the extra exposures are usable.
        per_pair_out = {}
        for k, v in per_pair.items():
            members = [
                m for m in self.accepted
                if f"{Path(m.ref_frame).name} -> {Path(m.tgt_frame).name}" == k
            ]
            de1 = np.array([m.offset_e1_m for m in members])
            de2 = np.array([m.offset_e2_m for m in members])
            bias = np.array([np.median(de1), np.median(de2)])
            scatter_m = np.hypot(de1 - bias[0], de2 - bias[1])
            # metres -> pixels via this pair's own measured ratio
            mag_m = np.hypot(de1, de2)
            scale = np.median(np.array(v["errs"])[mag_m > 1e-9] / mag_m[mag_m > 1e-9]) \
                if np.any(mag_m > 1e-9) else 0.0
            per_pair_out[k] = {
                "count": int(v["count"]),
                "median_px": float(np.median(v["errs"])),
                "p90_px": float(np.percentile(v["errs"], 90)),
                "max_px": float(np.max(v["errs"])),
                "systematic_px": float(np.hypot(*bias) * scale),
                "scatter_px": float(np.median(scatter_m) * scale),
            }

        per_label: Dict[str, Dict[str, float]] = {}
        for label in sorted({m.label for m in self.accepted}):
            sel = [m.residual_px for m in self.accepted if m.label == label]
            per_label[label] = {
                "count": len(sel),
                "median_px": float(np.median(sel)),
                "p90_px": float(np.percentile(sel, 90)),
            }

        # A systematic component (non-zero mean vector) points at a model error;
        # a zero-mean scatter points at matching noise.
        de1 = np.array([m.offset_e1_m for m in self.accepted])
        de2 = np.array([m.offset_e2_m for m in self.accepted])

        return {
            "attempted": len(self.matches),
            "accepted": len(errs),
            "back_face": back_face,
            "eligible": eligible,
            # Over pairs that were geometrically possible, not over every pair
            # attempted -- see the note above.
            "acceptance_rate": float(len(errs) / max(eligible, 1)),
            "rejections": rejects,
            "pair_mode": self.pair_mode,
            "median_error_px": float(np.median(errs)),
            "mean_error_px": float(np.mean(errs)),
            "p90_error_px": float(np.percentile(errs, 90)),
            "max_error_px": float(np.max(errs)),
            "median_error_m": float(np.median(self.residuals_m)),
            "systematic_offset_m": [float(np.median(de1)), float(np.median(de2))],
            "systematic_offset_magnitude_m": float(np.hypot(np.median(de1), np.median(de2))),
            "per_pair": per_pair_out,
            "per_surface_label": per_label,
            "params": self.params,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.stats(), "matches": [m.to_dict() for m in self.matches]}

    def save_json(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return p

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame([m.to_dict() for m in self.matches])

    def __repr__(self) -> str:
        s = self.stats()
        if not s.get("accepted"):
            return f"ReprojectionReport(attempted={s['attempted']}, accepted=0, rejections={s['rejections']})"
        return (
            f"ReprojectionReport(n={s['accepted']}/{s['attempted']}, "
            f"median={s['median_error_px']:.2f}px, p90={s['p90_error_px']:.2f}px, "
            f"max={s['max_error_px']:.2f}px)"
        )

def resolve_reference_index(
    cameras: Sequence[VirtualObliqueCamera],
    reference: Optional[Union[int, str]] = None,
) -> int:
    """Resolve a reference camera given an index, a frame id, or a suffix of one."""
    if reference is None:
        return len(cameras) // 2
    if isinstance(reference, int):
        return reference
    for i, cam in enumerate(cameras):
        if cam.frame_id == reference or cam.frame_id.endswith(reference) or cam.shot_id == reference:
            return i
    raise KeyError(f"No camera matching reference {reference!r}")


def _frame_pairs(
    cameras: Sequence[VirtualObliqueCamera],
    mode: str,
    reference: Optional[Union[int, str]] = None,
) -> List[Tuple[int, int]]:
    n = len(cameras)
    if mode == "consecutive":
        return [(i, i + 1) for i in range(n - 1)]
    if mode == "reference":
        ref = resolve_reference_index(cameras, reference)
        return [(ref, j) for j in range(n) if j != ref]
    if mode == "all":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    raise ValueError(f"Unknown pair mode {mode!r}; use consecutive, reference or all")


def anchors_in_volume(anchors, volume, pad_m: float = 0.0):
    """Anchors whose centre falls inside a building volume.

    Cross-direction registration is reported so that reconstruction quality can
    be anticipated, and what matters for that is the **building**. The anchor
    set spans the whole AOI, which for the pilot is ~140 m of parking lot, road
    and treeline -- and the visibility maps show its roof matches well while
    that surround does not. An unrestricted median therefore describes how well
    treeline correlates across look directions, which is not the question.

    The same correction as scoring the building rather than the photograph in
    a held-out render score (Phase 2, since removed); report both numbers, not one.
    """
    lo = np.array([volume.min_x_m - pad_m, volume.min_y_m - pad_m,
                   volume.min_z_m - pad_m])
    hi = np.array([volume.max_x_m + pad_m, volume.max_y_m + pad_m,
                   volume.max_z_m + pad_m])
    kept = []
    for a in anchors:
        xyz = np.asarray(a.xyz_local_m, dtype=np.float64)
        if np.all((xyz >= lo) & (xyz <= hi)):
            kept.append(a)
    return kept


def measure_cross_view_residuals(
    anchors: Sequence[PlanarAnchor],
    cameras: Sequence[VirtualObliqueCamera],
    readers: Dict[str, FrameImageReader],
    pair_mode: str = "consecutive",
    patch_radius_m: float = 6.0,
    search_radius_m: float = 3.0,
    gsd_m: float = 0.10,
    min_ncc: float = 0.5,
    min_texture_std: float = 8.0,
    min_facing_cos: float = 0.0,
    windows: Optional[Dict[str, ImageWindow]] = None,
    reference: Optional[Union[int, str]] = None,
    progress: bool = True,
) -> ReprojectionReport:
    """Measure ref->target residuals for every anchor over the chosen frame pairs.

    ``reference`` selects the reference exposure for ``pair_mode="reference"``,
    as an index, a frame id, or a shot id. It defaults to the middle camera,
    which is rarely the one you want once several look directions are mixed in.
    """
    if windows is None:
        windows = prefetch_windows(anchors, cameras, readers, patch_radius_m + search_radius_m)

    pairs = _frame_pairs(cameras, pair_mode, reference)
    matches: List[MatchResult] = []
    for pi, (i, j) in enumerate(pairs):
        ref_cam, tgt_cam = cameras[i], cameras[j]
        ref_win, tgt_win = windows[ref_cam.frame_id], windows[tgt_cam.frame_id]
        for anchor in anchors:
            matches.append(
                match_anchor_between_views(
                    anchor, ref_cam, ref_win, tgt_cam, tgt_win,
                    patch_radius_m=patch_radius_m,
                    min_facing_cos=min_facing_cos,
                    search_radius_m=search_radius_m,
                    gsd_m=gsd_m,
                    min_ncc=min_ncc,
                    min_texture_std=min_texture_std,
                )
            )
        if progress:
            print(
                f"  pair {pi + 1}/{len(pairs)}: "
                f"{Path(ref_cam.frame_id).name} -> {Path(tgt_cam.frame_id).name}"
            )

    return ReprojectionReport(
        matches=matches,
        pair_mode=pair_mode,
        params={
            "patch_radius_m": patch_radius_m,
            "search_radius_m": search_radius_m,
            "gsd_m": gsd_m,
            "min_ncc": min_ncc,
            "min_texture_std": min_texture_std,
            "n_anchors": len(anchors),
            "n_frames": len(cameras),
            "reference": (
                cameras[resolve_reference_index(cameras, reference)].frame_id
                if pair_mode == "reference" else None
            ),
        },
    )


def prefetch_windows(
    anchors: Sequence[PlanarAnchor],
    cameras: Sequence[VirtualObliqueCamera],
    readers: Dict[str, FrameImageReader],
    max_radius_m: float,
    progress: bool = True,
) -> Dict[str, ImageWindow]:
    """Fetch one enclosing window per frame so matching runs from memory."""
    pts = np.array([a.xyz_local_m for a in anchors], dtype=np.float64)
    windows: Dict[str, ImageWindow] = {}
    for cam in cameras:
        uv, zc, _ = cam.project(pts)
        uv = uv[zc > 0.1]
        # Generous pixel margin: one plane-space radius plus resampling slack.
        margin = int(np.ceil(max_radius_m / 0.03)) + 64
        windows[cam.frame_id] = readers[cam.frame_id].prefetch_covering(uv, margin_px=margin)
        if progress:
            w = windows[cam.frame_id]
            print(
                f"  fetched {Path(cam.frame_id).name}: {w.width}x{w.height} "
                f"at ({w.x0},{w.y0}), sensor coverage {100 * w.coverage_fraction:.0f}%"
            )
    return windows


# --------------------------------------------------------------- triangulation
@dataclass
class TriangulationResult:
    anchor_id: str
    label: str
    n_views: int
    xyz_triangulated_m: np.ndarray
    xyz_lidar_m: np.ndarray
    discrepancy_m: float
    horizontal_m: float
    vertical_m: float
    max_ray_angle_deg: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "label": self.label,
            "n_views": int(self.n_views),
            "xyz_triangulated_m": [float(v) for v in self.xyz_triangulated_m],
            "xyz_lidar_m": [float(v) for v in self.xyz_lidar_m],
            "discrepancy_m": float(self.discrepancy_m),
            "horizontal_m": float(self.horizontal_m),
            "vertical_m": float(self.vertical_m),
            "max_ray_angle_deg": float(self.max_ray_angle_deg),
        }


def _triangulate_rays(centres: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Least-squares closest point to a bundle of 3D rays."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(centres, dirs):
        d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.solve(A, b)


def triangulate_anchors(
    anchors: Sequence[PlanarAnchor],
    cameras: Sequence[VirtualObliqueCamera],
    readers: Dict[str, FrameImageReader],
    reference_index: Optional[Union[int, str]] = None,
    min_views: int = 3,
    windows: Optional[Dict[str, ImageWindow]] = None,
    **match_kwargs: Any,
) -> List[TriangulationResult]:
    """Triangulate NCC-matched image positions and compare against LiDAR.

    This is the cross-sensor half of the QA: the image observations come from
    correlation, the comparison point comes from LiDAR, and a scale or datum
    error shows up as a systematic vertical or horizontal discrepancy.
    """
    patch_radius_m = match_kwargs.get("patch_radius_m", 6.0)
    search_radius_m = match_kwargs.get("search_radius_m", 3.0)
    if windows is None:
        windows = prefetch_windows(
            anchors, cameras, readers, patch_radius_m + search_radius_m
        )

    ref_i = resolve_reference_index(cameras, reference_index)
    ref_cam = cameras[ref_i]
    results: List[TriangulationResult] = []

    for anchor in anchors:
        centres, dirs = [], []
        # The reference view contributes its predicted position; every other
        # view contributes an NCC-measured one.
        for j, cam in enumerate(cameras):
            if j == ref_i:
                uv, zc, _ = cam.project(anchor.xyz_local_m)
                if zc <= 0.1:
                    continue
                obs = np.asarray(uv)
            else:
                m = match_anchor_between_views(
                    anchor, ref_cam, windows[ref_cam.frame_id], cam, windows[cam.frame_id],
                    **match_kwargs,
                )
                if not m.accepted:
                    continue
                obs = np.array(m.observed_uv)
            c, d = cam.unproject(obs)
            centres.append(c)
            dirs.append(d)

        if len(centres) < min_views:
            continue

        centres_a = np.array(centres)
        dirs_a = np.array(dirs)
        xyz = _triangulate_rays(centres_a, dirs_a)
        diff = xyz - anchor.xyz_local_m

        angles = [
            np.degrees(np.arccos(np.clip(dirs_a[p] @ dirs_a[q], -1, 1)))
            for p in range(len(dirs_a))
            for q in range(p + 1, len(dirs_a))
        ]
        results.append(
            TriangulationResult(
                anchor_id=anchor.anchor_id,
                label=anchor.label,
                n_views=len(centres),
                xyz_triangulated_m=xyz,
                xyz_lidar_m=anchor.xyz_local_m.copy(),
                discrepancy_m=float(np.linalg.norm(diff)),
                horizontal_m=float(np.hypot(diff[0], diff[1])),
                vertical_m=float(diff[2]),
                max_ray_angle_deg=float(max(angles) if angles else 0.0),
            )
        )
    return results


def triangulation_summary(results: Sequence[TriangulationResult]) -> Dict[str, Any]:
    if not results:
        return {"count": 0}
    d = np.array([r.discrepancy_m for r in results])
    h = np.array([r.horizontal_m for r in results])
    v = np.array([r.vertical_m for r in results])
    return {
        "count": len(results),
        "median_3d_discrepancy_m": float(np.median(d)),
        "p90_3d_discrepancy_m": float(np.percentile(d, 90)),
        "median_horizontal_m": float(np.median(h)),
        "median_vertical_bias_m": float(np.median(v)),
        "vertical_std_m": float(np.std(v)),
        "median_views": float(np.median([r.n_views for r in results])),
        "median_max_ray_angle_deg": float(np.median([r.max_ray_angle_deg for r in results])),
    }


# ------------------------------------------------------- multi-direction help
def group_cameras(
    cameras: Sequence[VirtualObliqueCamera],
) -> Dict[Tuple[str, str], List[VirtualObliqueCamera]]:
    """Group exposures by (look direction, flight line).

    Exposures within a group share a camera head and a single EO solution;
    exposures in different groups do not. The distinction matters because
    within-group agreement tests the camera model, while cross-group agreement
    also carries line-to-line georeferencing bias.
    """
    groups: Dict[Tuple[str, str], List[VirtualObliqueCamera]] = {}
    for cam in cameras:
        key = (cam.camera_name.split("_")[0], str(cam.flight_line))
        groups.setdefault(key, []).append(cam)
    for members in groups.values():
        members.sort(key=lambda c: str(c.shot_id))
    return groups


def measure_within_group_residuals(
    anchors: Sequence[PlanarAnchor],
    cameras: Sequence[VirtualObliqueCamera],
    readers: Dict[str, FrameImageReader],
    windows: Optional[Dict[str, ImageWindow]] = None,
    progress: bool = True,
    **match_kwargs: Any,
) -> ReprojectionReport:
    """Consecutive-pair residuals measured inside each (direction, line) group.

    This is the camera-model check: it compares like with like, so a failure
    points at the model rather than at the difficulty of correlating two views
    100 degrees apart.
    """
    groups = group_cameras(cameras)
    matches: List[MatchResult] = []
    for (direction, line), members in sorted(groups.items()):
        if len(members) < 2:
            if progress:
                print(f"  {direction} line {line}: only {len(members)} frame, skipped")
            continue
        report = measure_cross_view_residuals(
            anchors, members, readers, pair_mode="consecutive",
            windows=windows, progress=False, **match_kwargs,
        )
        matches.extend(report.matches)
        if progress:
            s = report.stats()
            print(f"  {direction} line {line}: {len(members)} frames, "
                  f"median {s.get('median_error_px', float('nan')):.2f} px "
                  f"({s.get('accepted', 0)}/{s.get('attempted', 0)})")

    params = dict(match_kwargs)
    params.update({"n_anchors": len(anchors), "n_frames": len(cameras),
                   "n_groups": len(groups)})
    return ReprojectionReport(matches=matches, pair_mode="within-group consecutive",
                              params=params)


def view_geometry_summary(cameras: Sequence[VirtualObliqueCamera],
                          target_local_m: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Angular diversity of a camera set as seen from the target.

    ``azimuth_concentration`` is the mean-resultant length R of the viewing
    azimuths: 1.0 means every exposure looks from the same bearing, and values
    near 0 mean they surround the target. It is the number that says whether a
    reconstruction is being asked to invent the sides it never saw.
    """
    target = np.zeros(3) if target_local_m is None else np.asarray(target_local_m, float)
    C = np.array([c.camera_center_local_m for c in cameras], dtype=np.float64)
    d = target[None, :] - C
    d = d / np.linalg.norm(d, axis=1, keepdims=True)

    az = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360.0
    el = np.degrees(np.arcsin(-d[:, 2]))

    # A near-nadir exposure looks straight down, so its azimuth is arbitrary and
    # must not be allowed to swing the concentration figure. Weight each bearing
    # by the horizontal component of its view ray, which is cos(elevation): a
    # true nadir view then contributes nothing, as it should.
    weights = np.cos(np.radians(el))
    rad = np.radians(az)
    wsum = weights.sum()
    R = (
        float(np.hypot((weights * np.cos(rad)).sum(), (weights * np.sin(rad)).sum()) / wsum)
        if wsum > 1e-9 else 0.0
    )
    near_nadir = int((el > 75.0).sum())
    angles = [
        float(np.degrees(np.arccos(np.clip(d[i] @ d[j], -1, 1))))
        for i in range(len(d)) for j in range(i + 1, len(d))
    ]
    return {
        "n_exposures": len(cameras),
        "look_directions": sorted({c.camera_name.split("_")[0] for c in cameras}),
        "flight_lines": sorted({str(c.flight_line) for c in cameras}),
        "azimuths_deg": [float(a) for a in np.sort(az)],
        "elevation_min_deg": float(el.min()),
        "elevation_max_deg": float(el.max()),
        "n_near_nadir": near_nadir,
        "max_ray_separation_deg": float(max(angles)) if angles else 0.0,
        "azimuth_concentration": R,
    }
