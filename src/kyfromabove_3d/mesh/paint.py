"""Walls painted from whole photographs: each flat section of a wall is
warped from the one frame that sees it best, and patched from the next where
something stands in that frame's way.

The products had coloured a measured wall from its LiDAR points (each point
from its own most face-on frame, the points rasterised) and a wall drawn up
to the roof's edge texel by texel, each from its own frame. A wall holds few
returns -- the LiDAR is flown from above -- so the first was blocky, and the
second a patchwork of frames lit and seen differently. Warped whole, a
section keeps the frame's windows and mullions in line.

What stands in the way is the same test as everywhere else
(:func:`kyfromabove_3d.colourise.colour.project_visible`): the frame faces
the wall, the texel is inside the crop and its validity mask, and nothing in
the frame's depth map -- every LiDAR return, trees included, and the
neighbours as prisms (:func:`kyfromabove_3d.colourise.occlusion.occlusion_maps`)
-- is more than 1.5 m in front of it. A tree in front of a house is not
painted onto the house; the next frame paints that part, or it stays
unseen, and the completed product fills it as it fills any unseen wall.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from kyfromabove_3d.features import ArcWall, WallPlane
from kyfromabove_3d.texture_check import highpass, luminance, masked_ncc

from .textures import MAX_TEX_PX, Texture, _fill_holes, _solid_px

#: Wall pieces within this distance of each other (in and out of the
#: outline) count as one flat section and share one photograph.
SECTION_TOL_M = 0.5

#: How much a frame's facing the wall counts in choosing the one that paints
#: it (2.0 scored the same on the held-out test: 55% against 54%).
FACING_POWER = 0.9
#: A frame paints what the best could not see unless its pattern clearly
#: disagrees with what is painted, over at least ``MIN_OVERLAP_PX`` texels both
#: see (correlation of the high-passed brightness). The nadir frame, seeing
#: the Hyatt's 4 m pier at a grazing angle (-0.57 against the frame beside the
#: best), had painted it half a floor out of step. A higher bar (0.2) turned
#: away frames on plain walls, where the correlation is noise round zero, and
#: left them unseen.
AGREE_MIN = -0.1
MIN_OVERLAP_PX = 100
#: A frame takes part when it sees at least this share of the grid, and
#: patches when it adds at least ``MIN_NEW_SHARE`` of it.
MIN_SEE_SHARE = 0.10
MIN_NEW_SHARE = 0.015
#: Beside what blocked the best frame, its texels go where the next frame's
#: colour differs from it by more than this (0-255, per channel); None: never.
EDGE_DIFF = 42.0


def wall_sections(plane: WallPlane, wall_quads: Sequence[Any]) -> List[Dict[str, Any]]:
    """A face's wall quads (``wall_geometry``'s, kind ``"wall"``) grouped into
    flat sections: pieces standing at the same depth, within
    ``SECTION_TOL_M``, form one; a step in or out starts another. A
    photograph warped onto a wall that is not where the building is shows
    whatever stands behind it in the wrong place (400 West Market's crown,
    2-8 m back, painted onto a flat 164 m face). Each section is
    ``{"off", "a_lo", "a_hi", "z_lo", "z_hi", "quads"}``: its area-weighted
    depth out of the outline, its extent along the face and up, and its
    quads."""
    if isinstance(plane, ArcWall) or not wall_quads:
        return []
    p0 = np.asarray(plane.p0, dtype=np.float64)
    e1 = (np.asarray(plane.p1, dtype=np.float64) - p0) / max(plane.length_m, 1e-6)
    nv = np.asarray(plane.normal, dtype=np.float64)
    pieces = []
    for q in wall_quads:
        c = np.asarray(q[0], dtype=np.float64)
        a = (c[:, :2] - p0) @ e1
        off = float(np.mean((c[:, :2] - p0) @ nv))
        pieces.append((off, a, c[:, 2], float(np.ptp(a) * np.ptp(c[:, 2])), q))
    pieces.sort(key=lambda p: -p[0])
    groups: List[List[Any]] = []
    for p in pieces:
        if groups and groups[-1][0][0] - p[0] <= SECTION_TOL_M:
            groups[-1].append(p)
        else:
            groups.append([p])
    sections = []
    for g in groups:
        w = np.array([max(p[3], 1e-3) for p in g])
        sections.append({
            "off": float(np.average([p[0] for p in g], weights=w)),
            "a_lo": float(min(p[1].min() for p in g)),
            "a_hi": float(max(p[1].max() for p in g)),
            "z_lo": float(min(p[2].min() for p in g)),
            "z_hi": float(max(p[2].max() for p in g)),
            "quads": [p[4] for p in g],
        })
    return sections


def section_uv(plane: WallPlane, sec: Dict[str, Any], corners: np.ndarray) -> np.ndarray:
    """Each corner's place in the section's picture: u along the face from
    ``a_lo``, v up from ``z_lo``."""
    p0 = np.asarray(plane.p0, dtype=np.float64)
    e1 = (np.asarray(plane.p1, dtype=np.float64) - p0) / max(plane.length_m, 1e-6)
    a = (np.asarray(corners)[:, :2] - p0) @ e1
    u = (a - sec["a_lo"]) / max(sec["a_hi"] - sec["a_lo"], 1e-6)
    v = (np.asarray(corners)[:, 2] - sec["z_lo"]) / max(sec["z_hi"] - sec["z_lo"], 1e-6)
    return np.clip(np.column_stack([u, v]), 0.0, 1.0)


def _size_px(length_m: float, texel_m: float) -> int:
    return max(2, min(MAX_TEX_PX, int(np.ceil(length_m / texel_m))))


def section_grid(plane: WallPlane, sec: Dict[str, Any], texel_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """The section's texel centres in the building's frame, ``(H, W, 3)`` with
    row 0 at the bottom (``z_lo``), and its outward normal ``(3,)``."""
    W = _size_px(sec["a_hi"] - sec["a_lo"], texel_m)
    H = _size_px(sec["z_hi"] - sec["z_lo"], texel_m)
    a = sec["a_lo"] + (np.arange(W) + 0.5) / W * (sec["a_hi"] - sec["a_lo"])
    z = sec["z_lo"] + (np.arange(H) + 0.5) / H * (sec["z_hi"] - sec["z_lo"])
    p0 = np.asarray(plane.p0, dtype=np.float64)
    e1 = (np.asarray(plane.p1, dtype=np.float64) - p0) / max(plane.length_m, 1e-6)
    nv = np.asarray(plane.normal, dtype=np.float64)
    xy = p0 + a[:, None] * e1 + sec["off"] * nv                       # (W, 2)
    xyz = np.concatenate([np.broadcast_to(xy[None], (H, W, 2)),
                          np.broadcast_to(z[:, None, None], (H, W, 1))], axis=-1)
    return xyz, np.array([nv[0], nv[1], 0.0])


def match_color_transfer(src: np.ndarray, ref: np.ndarray, overlap_mask: np.ndarray) -> np.ndarray:
    """``src`` with each channel's gain and bias matched to ``ref`` over
    ``overlap_mask``: two frames see a wall in different sun and exposure."""
    out = src.astype(np.float32).copy()
    if overlap_mask.sum() < 50:
        return src
    for c in range(3):
        s_vals = src[..., c][overlap_mask].astype(np.float32)
        r_vals = ref[..., c][overlap_mask].astype(np.float32)
        gain = np.clip(float(np.std(r_vals)) / max(float(np.std(s_vals)), 4.0), 0.65, 1.45)
        bias = float(np.mean(r_vals)) - gain * float(np.mean(s_vals))
        out[..., c] = np.clip(out[..., c] * gain + bias, 0, 255)
    return out.astype(np.uint8)


def _project(cam: dict, xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates (float) of ``(N, 3)`` points in ``cam``'s crop."""
    pc = (cam["R_cw"] @ (xyz - cam["C"]).T).T
    z = np.where(pc[:, 2] > 1e-6, pc[:, 2], 1e-6)
    proj = (cam["K"] @ (pc / z[:, None]).T).T
    return proj[:, 0], proj[:, 1]


def paint_grid(xyz: np.ndarray, normal: np.ndarray, cams: Sequence[dict], texel_m: float,
               min_facing_cos: float = 0.15, occlusion_slack_m: float = 1.5,
               ) -> Tuple[Optional[Texture], Dict[str, Any]]:
    """Paint the flat grid of texel centres ``xyz`` ``(H, W, 3)`` (row 0 the
    bottom), facing ``normal``, from the frames ``cams``.

    Every frame that sees at least a tenth of it is warped onto it whole
    (``cv2.remap``, cubic). The best -- the share it sees unblocked, how
    squarely it faces the wall, and how well its pattern agrees with the
    other frames' -- paints it; the next ones paint what the best could not
    see, their colour matched to it where both see and feathered in, each
    only if its pattern agrees with what is painted (``AGREE_MIN``). Where
    the next frame agrees overall but differs sharply from the best just
    beside what blocked the best, the best's texels there are dropped: the
    occluder's edge, wider than the depth map drew it.

    Returns ``(texture, info)``: a :class:`Texture` whose seen texels came
    from a frame -- the rest near or plain, as ``_fill_holes`` decides -- and
    ``{"frames": [ids, best first], "seen": share}``; ``(None, info)`` when no
    frame sees the grid."""
    from kyfromabove_3d.colourise.colour import project_visible

    H, W = xyz.shape[:2]
    pts = xyz.reshape(-1, 3)
    nrm = np.broadcast_to(np.asarray(normal, dtype=np.float64), pts.shape)
    step_m = texel_m
    cands = []
    for cam in cams:
        idx, _, _, _ = project_visible(pts, nrm, cam, min_facing_cos, occlusion_slack_m)
        if len(idx) < MIN_SEE_SHARE * len(pts):
            continue
        sees = np.zeros(len(pts), dtype=bool)
        sees[idx] = True
        sees = sees.reshape(H, W)
        u, v = _project(cam, pts)
        rgb = cv2.remap(cam["img"], u.reshape(H, W).astype(np.float32), v.reshape(H, W).astype(np.float32),
                        cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        mid = pts[len(pts) // 2]
        ray = cam["C"] - mid
        facing = float(ray @ normal / max(np.linalg.norm(ray), 1e-6))
        share = float(sees.mean())
        cands.append({"id": cam.get("id", ""), "sees": sees, "rgb": rgb,
                      "score": share ** 1.2 * max(facing, 1e-3) ** FACING_POWER})
    if not cands:
        return None, {"frames": [], "seen": 0.0}
    if len(cands) >= 2:
        hps = [highpass(luminance(c["rgb"].astype(np.float64)), c["sees"], sigma_px=max(2.0, 0.4 / texel_m))
               for c in cands]
        for i, c in enumerate(cands):
            corrs = []
            for j, c2 in enumerate(cands):
                m = c["sees"] & c2["sees"]
                if i != j and m.mean() >= 0.25:
                    r = masked_ncc(hps[i], hps[j], m)
                    if np.isfinite(r):
                        corrs.append(r)
            c["score"] *= 1.0 + 0.40 * max(0.0, float(np.median(corrs)) if corrs else 0.0)
    cands.sort(key=lambda c: -c["score"])
    best = cands[0]
    comp = best["rgb"].copy()
    seen = best["sees"].copy()
    used = [best["id"]]
    sigma_px = max(2.0, 0.4 / step_m)

    def agrees(c) -> bool:
        both = seen & c["sees"]
        if both.sum() < MIN_OVERLAP_PX:
            return True                                 # too little in common to judge
        r = masked_ncc(highpass(luminance(comp.astype(np.float64)), both, sigma_px),
                       highpass(luminance(c["rgb"].astype(np.float64)), both, sigma_px), both)
        return bool(np.isfinite(r) and r >= AGREE_MIN)

    if EDGE_DIFF is not None and len(cands) >= 2 and (~seen).any() and agrees(cands[1]):
        nxt = cands[1]
        edge = binary_dilation(~seen, iterations=max(4, int(round(4.0 / step_m)))) & seen & nxt["sees"]
        if edge.sum() > 50:
            nxt_m = match_color_transfer(nxt["rgb"], comp, seen & nxt["sees"] & ~edge)
            diff = np.abs(comp.astype(np.float32) - nxt_m.astype(np.float32)).mean(axis=-1)
            outlier = edge & (diff > EDGE_DIFF)
            if outlier.any():
                seen &= ~binary_dilation(outlier, iterations=max(2, int(round(1.0 / step_m))))
    for c in cands[1:]:
        missing = ~seen & c["sees"]
        if missing.mean() < MIN_NEW_SHARE or not agrees(c):
            continue
        matched = match_color_transfer(c["rgb"], comp, seen & c["sees"])
        dist_in = distance_transform_edt(~seen)
        alpha = np.clip(dist_in / max(3.0, 1.0 / step_m), 0.0, 1.0)[..., None]
        zone = c["sees"] & (dist_in > 0)
        blended = (alpha * matched.astype(np.float32) + (1.0 - alpha) * comp.astype(np.float32)).astype(np.uint8)
        comp = np.where(zone[..., None], blended, comp)
        seen |= c["sees"]
        used.append(c["id"])
    cnt = seen.astype(np.float64)
    tex = _fill_holes(comp.astype(np.float64) * cnt[..., None], cnt,
                      max_fill_px=max(2, int(1.5 / texel_m)), solid_px=_solid_px(texel_m))
    return tex, {"frames": used, "seen": float(seen.mean())}


def paint_section(plane: WallPlane, sec: Dict[str, Any], cams: Sequence[dict], texel_m: float):
    """:func:`paint_grid` over a flat section of ``plane`` (:func:`wall_sections`)."""
    xyz, n3 = section_grid(plane, sec, texel_m)
    return paint_grid(xyz, n3, cams, texel_m)


#: which frames painted which wall section, per building (``paint_record``)
PAINT_FILE = "wall_paint.json"


def paint_record(plane: WallPlane, face: int, sec: Dict[str, Any], frames: Sequence[str],
                 kind: str = "profile") -> Dict[str, Any]:
    """What a section is and which frames painted it, for ``wall_paint.json``:
    its face, its ``kind`` (``"profile"``, a measured wall; ``"prism"``, one
    drawn up to the roof's edge), the face's line (``p0``, ``e1``,
    ``normal``), its depth out of the line, its extent along it and up."""
    p0 = np.asarray(plane.p0, dtype=np.float64)
    e1 = (np.asarray(plane.p1, dtype=np.float64) - p0) / max(plane.length_m, 1e-6)
    return {"face": int(face), "kind": kind, "p0": [round(float(v), 3) for v in p0], "e1": [round(float(v), 6) for v in e1],
            "normal": [round(float(v), 6) for v in np.asarray(plane.normal, dtype=np.float64)[:2]],
            "off": round(float(sec["off"]), 3), "along": [round(float(sec["a_lo"]), 3), round(float(sec["a_hi"]), 3)],
            "z": [round(float(sec["z_lo"]), 3), round(float(sec["z_hi"]), 3)], "frames": list(frames)}


def painted_by(records: Sequence[Dict[str, Any]], xyz: np.ndarray, tol_m: float = 0.6) -> List[frozenset]:
    """For each point of ``xyz`` ``(N, 3)``, the frames that painted the wall
    section it lies on (``paint_record``), empty where none did."""
    if not len(records) or not len(xyz):
        return [frozenset()] * len(xyz)
    on = np.zeros((len(records), len(xyz)), dtype=bool)
    for k, r in enumerate(records):
        d = xyz[:, :2] - np.asarray(r["p0"])
        a = d @ np.asarray(r["e1"]); off = d @ np.asarray(r["normal"])
        on[k] = ((a >= r["along"][0] - 0.05) & (a <= r["along"][1] + 0.05) & (xyz[:, 2] >= r["z"][0] - 0.05)
                 & (xyz[:, 2] <= r["z"][1] + 0.05) & (np.abs(off - r["off"]) <= tol_m))
    # points on the same sections share one set
    pattern, inv = np.unique(on.T, axis=0, return_inverse=True)
    sets = [frozenset(f for k in np.flatnonzero(p) for f in records[k]["frames"]) for p in pattern]
    return [sets[i] for i in np.ravel(inv)]
