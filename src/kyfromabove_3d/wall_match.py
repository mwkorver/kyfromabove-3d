"""A wall's depth from the aerial frames that see it, by matching them.

A wall band's direction is the footprint's; how far in or out of the
footprint edge it stands is what the LiDAR gives where it saw the wall, and
nothing gives where it did not. Two frames laid onto a trial wall at the
right depth show the same pattern in the same place; at the wrong depth
they slide apart. :func:`match_band` sweeps the trial wall through the
possible depths and keeps the one where the frames agree best;
:func:`is_accepted` says whether to believe it; :func:`match_profiles` does
both for a building's wall profile in the colourise, and turns the unknown
bands it believes into ``matched`` ones.

Measured on the 19 reference buildings (``scripts/wall_match.py``,
``docs/WALL_VISIBILITY.md``): where the LiDAR measured the wall, the bands
that pass are within 0.5 m of it on 85 %, against 34 % for the footprint
edge. A per-building lean is left -- the windows matched sit behind the face
the laser hits -- which :func:`building_lean` takes out with the building's
own measured bands.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

STEP_M = 0.1                      # the trial wall's pixel
DEPTH_MIN, DEPTH_MAX = -8.0, 3.0  # metres off the footprint edge, + out
COARSE_M, FINE_M = 0.25, 0.05
RIVAL_M = 0.75                    # a rival depth is at least this far from the best
MIN_BASELINE_DEG = 10.0           # two frames this far apart can find depth
MAX_ANGLE_DEG = 70.0              # a frame further than this off face-on sees a sliver of wall
SIGMA_PX = 8.0                    # the high-pass scale, at STEP_M a pixel
EDGE_M = 0.5                      # a depth this close to the search's ends is the search running out
THIRDS_M = 0.5                    # a band's thirds must agree this well
# The cut past the fixed rules: (margin, pattern). Chosen, held out by
# building, on the measured bands of the reference set: the two halves gave
# margins of 0.0 and 0.05 and patterns of 0; the stricter is kept.
CUT = (0.05, 0.0)
MIN_LEAN_BANDS = 3                # a building's lean needs this many measured bands


def bilinear_gray(img: np.ndarray, u: np.ndarray, v: np.ndarray):
    """``img`` at (u, v), and where that is inside it."""
    H, W = img.shape
    ok = (u >= 0) & (u <= W - 1.001) & (v >= 0) & (v <= H - 1.001)
    u, v = np.clip(u, 0, W - 1.001), np.clip(v, 0, H - 1.001)
    x0, y0 = u.astype(int), v.astype(int)
    fx, fy = u - x0, v - y0
    val = (img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy) + (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy
    return val, ok


def sweep(to_world: Callable, A: np.ndarray, Z: np.ndarray, views: Dict[str, dict], offsets: Sequence[float],
          pairs: Sequence[Tuple[str, str]], segments: Sequence[np.ndarray] = (), sigma_px: float = SIGMA_PX):
    """Agreement between the views of the trial wall at each offset.

    ``to_world(along, offset)`` gives the wall's points (N, 3) in the
    cameras' frame for the grid ``A`` (along) and ``Z`` (up); each view is
    ``dict(K, R_cw, C, gray, x0, y0, mask)`` -- ``gray`` the frame (or a
    window of it at ``x0, y0``) and ``mask`` (nz, na) where it sees the wall.
    Returns an array (1 + len(segments), len(offsets)) -- the whole band,
    then each segment (a boolean over the band's columns) -- and, per offset,
    the pattern's strength (the weaker view's high-passed spread)."""
    from kyfromabove_3d.texture_check import highpass, masked_ncc
    nz, na = A.shape
    parts = [np.ones(na, bool)] + list(segments)
    scores, strength = [], []
    for o in offsets:
        P = to_world(A.ravel(), o)
        hp = {}
        for f, w in views.items():
            pc = (P - w["C"]) @ w["R_cw"].T
            u = w["K"][0, 0] * pc[:, 0] / pc[:, 2] + w["K"][0, 2] - w.get("x0", 0)
            v = w["K"][1, 1] * pc[:, 1] / pc[:, 2] + w["K"][1, 2] - w.get("y0", 0)
            g, ok = bilinear_gray(w["gray"], u, v)
            m = w["mask"] & ok.reshape(nz, na)
            hp[f] = (highpass(g.reshape(nz, na), m, sigma_px), m)
        row = []
        for k, cols in enumerate(parts):
            min_n = 400 if k == 0 else 150                                   # pixels two views must share
            s, wt = 0.0, 0.0
            for f1, f2 in pairs:
                m = hp[f1][1] & hp[f2][1] & cols[None, :]
                n = int(m.sum())
                if n < min_n:
                    continue
                r = masked_ncc(hp[f1][0], hp[f2][0], m)
                if np.isfinite(r):
                    s += r * np.sqrt(n)
                    wt += np.sqrt(n)
            row.append(s / wt if wt > 0 else np.nan)
        scores.append(row)
        strength.append(min((float(v[0][v[1]].std()) for v in hp.values() if v[1].sum() > 50), default=0.0))
    return np.array(scores).T, np.array(strength)


def pairs_apart(views: Dict[str, dict], centre: np.ndarray, min_deg: float = MIN_BASELINE_DEG) -> List[Tuple[str, str]]:
    """The pairs of views that look at ``centre`` from directions at least ``min_deg`` apart."""
    ray = {f: (w["C"] - centre) / np.linalg.norm(w["C"] - centre) for f, w in views.items()}
    ids = list(views)
    return [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]
            if np.degrees(np.arccos(np.clip(ray[a] @ ray[b], -1, 1))) >= min_deg]


def match_band(to_world: Callable, A: np.ndarray, Z: np.ndarray, views: Dict[str, dict]) -> Optional[dict]:
    """The depth where the views agree best: every ``COARSE_M`` over the
    search, then every ``FINE_M`` about the best. With it: the agreement
    there, the best agreement at least ``RIVAL_M`` away and the margin
    between, each third of the band's own depth, and the pattern's strength.
    None when no two views are far enough apart or share enough wall."""
    pairs = pairs_apart(views, to_world(A.ravel(), 0.0).mean(axis=0))
    if not pairs:
        return None
    coarse = np.arange(DEPTH_MIN, DEPTH_MAX + 1e-9, COARSE_M)
    na = A.shape[1]
    thirds = [(np.arange(na) * 3 // na) == t for t in range(3)]
    sc_all, _ = sweep(to_world, A, Z, views, coarse, pairs, thirds)
    sc = sc_all[0]
    if not np.isfinite(sc).any():
        return None
    k = int(np.nanargmax(sc))
    fine = np.arange(coarse[k] - 0.3, coarse[k] + 0.3001, FINE_M)
    sf_all, strength = sweep(to_world, A, Z, views, fine, pairs)
    sf = sf_all[0]
    kf = int(np.nanargmax(sf))
    best, peak = float(fine[kf]), float(sf[kf])
    far = np.abs(coarse - best) >= RIVAL_M
    rival = float(np.nanmax(sc[far])) if np.isfinite(sc[far]).any() else float("nan")
    return dict(depth_m=round(best, 2), agreement=round(peak, 3), rival=round(rival, 3), margin=round(peak - rival, 3),
                rival_depth_m=float(coarse[far][np.nanargmax(sc[far])]) if np.isfinite(rival) else None,
                pairs=len(pairs), frames_used=list(views),
                thirds_m=[float(coarse[int(np.nanargmax(t))]) if np.isfinite(t).any() else None for t in sc_all[1:]],
                pattern=round(float(strength[kf]), 2),
                curve=[[round(float(o), 2), None if not np.isfinite(v) else round(float(v), 3)] for o, v in zip(coarse, sc)])


# ------------------------------------------------------------------ believing it

def thirds_spread(r: dict) -> float:
    """How far apart the depths of the band's thirds (and the whole band's)
    are; inf with fewer than two thirds matched."""
    t = [x for x in r.get("thirds_m") or [] if x is not None]
    return float(max(t + [r["depth_m"]]) - min(t + [r["depth_m"]])) if len(t) >= 2 else float("inf")


def passes_rules(r: dict) -> bool:
    """The fixed rules: not at the end of the search (a depth there is the
    search running out, not a wall), and the band's thirds agree -- a flat
    wall agrees with itself; a reflection or a rounded wall seldom does."""
    return DEPTH_MIN + EDGE_M < r["depth_m"] < DEPTH_MAX - EDGE_M and thirds_spread(r) <= THIRDS_M


def is_accepted(r: dict, cut: Tuple[float, float] = CUT) -> bool:
    tm, tp = cut
    m = r["margin"] if r["margin"] == r["margin"] else 1.0
    return passes_rules(r) and m >= tm and r.get("pattern", 99.0) >= tp


def choose_cut(rows: Sequence[dict], errors: np.ndarray, max_gross: float = 0.03, gross_m: float = 1.0):
    """Past the fixed rules, the loosest (margin, pattern) cut that keeps the
    share of accepted bands off by more than ``gross_m`` at or under
    ``max_gross``: a cut against matching that failed, not against the lean."""
    ok = np.array([passes_rules(r) for r in rows])
    mg = np.nan_to_num(np.array([r["margin"] for r in rows], float), nan=1.0)
    pt = np.array([r.get("pattern", 99.0) for r in rows])
    best = (1.0, 99.0, 0)
    for tm in np.arange(0.0, 0.2, 0.01):
        for tp in np.arange(0.0, 40.0, 1.0):
            acc = ok & (mg >= tm) & (pt >= tp)
            if acc.sum() >= 10 and np.mean(np.abs(errors[acc]) > gross_m) <= max_gross and acc.sum() > best[2]:
                best = (float(tm), float(tp), int(acc.sum()))
    return best[0], best[1]


def lean_removed(err: np.ndarray, building: np.ndarray, min_others: int = MIN_LEAN_BANDS) -> np.ndarray:
    """Each band's error less the median error of the *other* measured bands
    of its building -- what an unknown band of that building could be
    corrected by. NaN where the building has fewer than ``min_others``."""
    out = np.full(len(err), np.nan)
    for i in range(len(err)):
        o = building == building[i]
        o[i] = False
        if o.sum() >= min_others:
            out[i] = err[i] - np.median(err[o])
    return out


def building_lean(measured: Sequence[Tuple[dict, float]]) -> Tuple[float, int]:
    """The building's lean -- the median of matched minus LiDAR depth over
    its accepted measured bands -- and how many bands it rests on; 0 with
    fewer than ``MIN_LEAN_BANDS``."""
    d = [r["depth_m"] - lidar for r, lidar in measured if is_accepted(r)]
    return (float(np.median(d)), len(d)) if len(d) >= MIN_LEAN_BANDS else (0.0, len(d))


# ------------------------------------------------------------------ in the colourise

def band_grid(plane, lo: float, hi: float, z_lo: float, z_hi: float):
    """The band's wall as a grid of points ``STEP_M`` apart: (A along, Z up)."""
    return np.meshgrid(np.arange(lo + STEP_M / 2, hi, STEP_M), np.arange(z_lo + STEP_M / 2, z_hi, STEP_M))


def match_profiles(planes: Sequence, profiles: Dict[int, list], cams: Sequence[dict], visible: Callable,
                   skip_faces: Sequence[int] = (), corner_m: float = 1.5, min_seen: float = 0.5, log=print) -> dict:
    """Match a building's measured and unknown bands in its own crops; turn
    the unknown bands that pass into ``matched`` ones, at the matched depth
    less the building's lean (:func:`building_lean`). ``profiles`` is changed
    in place.

    ``cams`` are the colourise's cameras (``K``, ``R_cw``, ``C``, ``img``,
    with their depth buffers); ``visible(xyz, normals, cam)`` is its
    visibility test, returning the indices of the points the camera sees.
    A camera is a view of the band where it sees ``min_seen`` of the band's
    wall at the depth the profile has (the footprint edge for an unknown
    band). Returns the report: every band tried, the lean, what changed."""
    gray = {}
    rows, measured = [], []
    by_face = {p.face: p for p in planes}
    for face, bands in profiles.items():
        p = by_face.get(face)
        if p is None or face in skip_faces:
            continue
        L = float(p.length_m)
        normal = np.asarray(p.normal, float)
        for k, b in enumerate(bands):
            if b.status not in ("observed", "unknown"):
                continue
            lo, hi = (b.along_lo, b.along_hi) if b.status == "observed" else (corner_m, L - corner_m)
            if hi - lo < 3.0 or b.z_hi - b.z_lo < 1.0:
                continue
            A, Z = band_grid(p, lo, hi, b.z_lo, b.z_hi)
            prior = b.offset_m if b.status == "observed" and b.offset_m is not None else 0.0

            def to_world(a, o, p=p, Z=Z):
                return np.column_stack([p.world(a, np.full(len(a), o)), Z.ravel()])

            P0 = to_world(A.ravel(), prior)
            if hasattr(p, "basis"):                                          # a curved wall: the outward radial
                _, out = p.basis(A.ravel())
                nrm = np.column_stack([out, np.zeros(len(out))])
            else:
                nrm = np.tile([normal[0], normal[1], 0.0], (len(P0), 1))
            views = {}
            for c in cams:
                idx = visible(P0, nrm, c)
                if len(idx) < min_seen * len(P0):
                    continue
                if c["id"] not in gray:
                    gray[c["id"]] = c["img"].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
                m = np.zeros(len(P0), bool)
                m[idx] = True
                views[c["id"]] = dict(K=c["K"], R_cw=c["R_cw"], C=c["C"], gray=gray[c["id"]], mask=m.reshape(A.shape))
            if len(views) < 2:
                continue
            r = match_band(to_world, A, Z, views)
            if r is None:
                continue
            r.update(face=int(face), band=k, z_lo=b.z_lo, z_hi=b.z_hi, status=b.status, along=[lo, hi],
                     lidar_m=b.offset_m if b.status == "observed" else None, accepted=is_accepted(r))
            r.pop("curve", None)
            rows.append(r)
            if b.status == "observed" and b.offset_m is not None:
                measured.append((r, b.offset_m))
    lean, n_lean = building_lean(measured)
    n_matched = 0
    for r in rows:
        if r["status"] != "unknown" or not r["accepted"]:
            continue
        b = profiles[r["face"]][r["band"]]
        # Matched away from the corners (a neighbour's wall stands there), but
        # drawn the face's whole length, as a mirrored band is: the corner
        # clip (clip_profiles_by_neighbours) then stops it where the next
        # wall stands. Drawn 1.5 m short, it left a slit at every corner.
        L = float(by_face[r["face"]].length_m)
        profiles[r["face"]][r["band"]] = replace(b, offset_m=round(r["depth_m"] - lean, 2), along_lo=0.0,
                                                 along_hi=L, status="matched")
        r["offset_m"] = round(r["depth_m"] - lean, 2)
        n_matched += 1
    n_unknown = sum(r["status"] == "unknown" for r in rows)
    meas_ok = [abs(r["depth_m"] - r["lidar_m"]) for r in rows if r["status"] == "observed" and r["accepted"]]
    log(f"    walls matched in the frames: {n_matched} of {n_unknown} unknown band(s) tried; lean {lean:+.2f} m "
        f"from {n_lean} measured band(s)" + (f"; measured bands matched within 0.5 m: {np.mean(np.array(meas_ok) <= 0.5) * 100:.0f}% "
                                             f"of {len(meas_ok)}" if meas_ok else ""))
    return dict(lean_m=round(lean, 3), lean_bands=n_lean, cut=list(CUT), matched=n_matched, tried_unknown=n_unknown,
                tried_measured=sum(r["status"] == "observed" for r in rows), bands=rows)
