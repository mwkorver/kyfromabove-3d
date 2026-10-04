"""Snap a footprint's corners to the edges the aerial frames show.

The LiDAR fixes each wall's plane -- its offset, to about 0.10 m -- but says
little about where a face *ends*: the corners come from Overture's outline,
which is a metre off as a matter of course, and from thin corner returns.
A building's edge, though, is a line the frames show crisply, on glass as
well as brick: past the wall's true end, a strip sampled along the
extended plane holds sky, ground or the neighbour, and the silhouette is a
strong vertical edge. Each frame that sees a corner votes for where the
wall ends; the votes agree at the true end and disagree on anything that
is not on the plane (a neighbour's corner behind it projects to a
different place from each viewpoint). The corner is moved to the observed
end and the unverified line beside it is slid to pass through it; moving
a line moves its other corner, so the pass repeats round the ring until
nothing moves. A LiDAR-verified plane is never moved.

Two hazards the literature on LiDAR-plus-image-edge fusion reports first,
and their guards here: a shadow reads as an edge (so a vote is kept only
where the texture's *structure* drops past the edge, as it does at sky or
ground, or when two flight lines agree on it), and a neighbour's edge
lands on the extended plane (so votes from one flight line alone take the
candidate nearest the current corner, not the strongest).

Everything here is a pure function over numpy arrays; the caller supplies
the frame sampler (``colorize_building.sample_colour`` with one camera).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import WallPlane, _ccw_ring

Sampler = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]   # (xyz, normals) -> (rgb, seen)


@dataclass
class EdgeVote:
    frame: str
    line: str                  # flight line the frame belongs to
    along_m: float             # where the wall ends, metres along the face from p0
    strength: float            # column gradient energy at the edge, relative to the strip's median
    far_structure: float       # texture structure beyond the edge / before it (< 1: it drops, a silhouette)
    n_seen: int


@dataclass
class CornerRecord:
    corner: int
    status: str                # snapped | measured | no-observation | skipped-party | skipped-arc | skipped-no-plane | refused
    before: List[float]
    after: List[float]
    gap_before_m: Optional[float] = None
    gap_after_m: Optional[float] = None
    votes: Dict[str, int] = field(default_factory=dict)      # "end of face k" -> n votes
    note: str = ""


def flight_line_of(frame_id: str) -> str:
    """``KY_.../Bwd_7021_47003`` -> ``7021``."""
    parts = frame_id.rsplit("/", 1)[-1].split("_")
    return parts[1] if len(parts) >= 3 else frame_id


def edge_strip(plane: WallPlane, sampler: Sampler, s_centre: float, z_lo: float, z_hi: float,
               half_m: float = 3.0, texel_m: float = 0.10):
    """A grey strip of the plane about ``s_centre`` (``along`` in
    ``[s_centre - half, s_centre + half]``, ``z`` in ``[z_lo, z_hi]``) sampled
    through one frame. Returns ``(gray (H, W), seen (H, W), along (W,))``;
    ``gray`` is nan where the frame does not see the texel."""
    along = np.arange(s_centre - half_m, s_centre + half_m + 1e-9, texel_m)
    zs = np.arange(z_lo, z_hi + 1e-9, texel_m)
    if len(along) < 8 or len(zs) < 8:
        return None
    A, Z = np.meshgrid(along, zs)
    xy = plane.world(A.ravel(), np.zeros(A.size))
    pts = np.column_stack([xy, Z.ravel()])
    n3 = np.repeat(np.array([[plane.normal[0], plane.normal[1], 0.0]]), len(pts), axis=0)
    rgb, seen = sampler(pts, n3)
    seen = (np.asarray(seen) > 0).reshape(A.shape)
    gray = (np.asarray(rgb, dtype=np.float64) @ np.array([0.299, 0.587, 0.114])).reshape(A.shape)
    gray[~seen] = np.nan
    return gray, seen, along


def edge_candidates(gray: np.ndarray, seen: np.ndarray, along: np.ndarray, min_seen: float = 0.5,
                    min_strength: float = 2.0, window_m: float = 1.5, max_candidates: int = 3,
                    outside: str = "high") -> List[Tuple[float, float, float]]:
    """Where the wall ends in one strip: the columns where the horizontal
    gradient (summed down the column) peaks. Returns up to
    ``max_candidates`` of ``(along_m, strength, far_structure)`` sorted by
    strength, ``far_structure`` being the texture structure on the
    ``outside`` side of the edge (``"high"``: greater ``along``; ``"low"``:
    lesser) over the structure on the wall side -- under 0.6 the texture
    stops there, a silhouette; near 1 it continues, a window jamb or a
    shadow. Empty when the frame sees too little of the strip."""
    if seen.mean() < min_seen:
        return []
    texel = float(along[1] - along[0])
    # gradient across columns, and the vertical texture in each column
    gx = np.abs(gray[:, 2:] - gray[:, :-2]) / 2.0                     # (H, W-2)
    gz = np.abs(gray[2:, :] - gray[:-2, :]) / 2.0                     # (H-2, W)
    e = np.nanmean(gx, axis=0)                                        # (W-2,)
    e = np.nan_to_num(e, nan=0.0)
    k = max(1, int(round(0.2 / texel)))
    e = np.convolve(e, np.ones(k) / k, mode="same")
    ref = float(np.median(e[e > 0])) if (e > 0).any() else 0.0
    if ref <= 0:
        return []
    # structure as contrast (gradient over brightness), so a shadowed wall
    # keeps its structure and only sky or ground loses it
    level = np.maximum(np.nan_to_num(np.nanmean(gray, axis=0), nan=0.0), 1.0)
    struct = np.nan_to_num(np.nanmean(gz, axis=0), nan=0.0) / level    # (W,)
    w = max(2, int(round(window_m / texel)))
    cands = []
    order = np.argsort(e)[::-1]
    taken = np.zeros(len(e), dtype=bool)
    for i in order:
        if len(cands) >= max_candidates or e[i] < min_strength * ref:
            break
        if taken[max(0, i - k):i + k + 1].any():
            continue
        c = i + 1                                                     # column in the strip's frame
        before = struct[max(0, c - w):c]
        after = struct[c + 1:c + 1 + w]
        if len(before) < 3 or len(after) < 3:
            continue
        wall, out = (before, after) if outside == "high" else (after, before)
        far = float(np.mean(out) / max(np.mean(wall), 1e-6))
        cands.append((float(along[c]), float(e[i] / ref), far))
        taken[max(0, i - k):i + k + 1] = True
    return cands


def observe_end(plane: WallPlane, end: str, cams: Sequence[dict], make_sampler: Callable[[dict], Sampler],
                z_lo: float, z_hi: float, half_m: float = 3.0, texel_m: float = 0.10,
                agree_m: float = 0.3, min_votes: int = 2) -> Tuple[Optional[float], List[EdgeVote], str]:
    """Where face ``plane`` ends at its ``end`` (``"start"``: along 0,
    ``"end"``: along L), from every frame that sees the strip. Returns
    ``(along_m or None, votes, why)``."""
    s0 = 0.0 if end == "start" else plane.length_m
    # the wall side of the corner is inside [0, L]; the strip reaches half_m past it
    votes: List[EdgeVote] = []
    for cam in cams:
        strip = edge_strip(plane, make_sampler(cam), s0, z_lo, z_hi, half_m, texel_m)
        if strip is None:
            continue
        gray, seen, along = strip
        cands = edge_candidates(gray, seen, along, outside="high" if end == "end" else "low")
        if not cands:
            continue
        line = flight_line_of(cam["id"])
        # only a silhouette counts: the texture stops past the edge. A window
        # jamb, a pier's edge or a shadow keeps the wall's structure going and
        # is not where the wall ends. Of the silhouettes in a strip, the one
        # nearest the current corner is the frame's vote.
        sil = [c for c in cands if c[2] < 0.6]
        if not sil:
            continue
        a, strength, far = min(sil, key=lambda c: abs(c[0] - s0))
        votes.append(EdgeVote(cam["id"], line, a, strength, far, int(seen.sum())))
    if not votes:
        return None, votes, "no frame sees the corner"
    # cluster the votes along the face; the observation is the best cluster
    positions = np.array([v.along_m for v in votes])
    best = None
    for i, v in enumerate(votes):
        members = [w for w in votes if abs(w.along_m - v.along_m) <= agree_m]
        lines = {w.line for w in members}
        frames = {w.frame for w in members}
        if len(frames) < min_votes:
            continue
        # score: frames agreeing, lines agreeing, then nearness to the current corner
        key = (len(lines), len(frames), -abs(v.along_m - s0))
        if best is None or key > best[0]:
            best = (key, members)
    if best is None:
        # one flight line only: take the candidate nearest the current corner
        # that at least two frames agree on
        near = sorted(votes, key=lambda w: abs(w.along_m - s0))
        for v in near:
            members = [w for w in votes if abs(w.along_m - v.along_m) <= agree_m]
            if len({w.frame for w in members}) >= min_votes:
                best = ((1, len(members), -abs(v.along_m - s0)), members)
                break
    if best is None:
        return None, votes, f"{len(votes)} vote(s), none agreeing"
    members = best[1]
    return float(np.median([w.along_m for w in members])), votes, f"{len({w.frame for w in members})} frames, {len({w.line for w in members})} line(s)"


def _line_intersection(a: WallPlane, b: WallPlane) -> Optional[np.ndarray]:
    da = a.p1 - a.p0; db = b.p1 - b.p0
    den = da[0] * db[1] - da[1] * db[0]
    if abs(den) < 1e-9 * max(np.hypot(*da) * np.hypot(*db), 1e-9):
        return None
    t = ((b.p0[0] - a.p0[0]) * db[1] - (b.p0[1] - a.p0[1]) * db[0]) / den
    return a.p0 + t * da


def _shift_through(plane: WallPlane, point: np.ndarray) -> None:
    """Translate the plane's line along its normal so ``point`` lies on it."""
    d = float((point - plane.p0) @ plane.normal)
    plane.p0 = plane.p0 + d * plane.normal
    plane.p1 = plane.p1 + d * plane.normal


def snap_corners(ring_m: np.ndarray, planes: Sequence[WallPlane], verified: set,
                 observations: Dict[Tuple[int, str], Optional[float]], skip: Dict[int, str] | None = None,
                 max_shift_m: float = 2.0, max_disagree_m: float = 1.0, tol_m: float = 0.05,
                 max_passes: int = 5) -> Tuple[np.ndarray, List[WallPlane], List[CornerRecord]]:
    """Move the ring's corners to the observed ends of their faces.

    ``planes`` are the ring's straight faces (``face`` = edge index in the
    CCW ring); ``verified`` the faces whose plane the LiDAR fixed;
    ``observations`` maps ``(face, "start"|"end")`` to the observed end in
    metres along that face, or None; ``skip`` maps a corner index to a
    reason (party wall, arc). Returns the new ring, the planes (copies,
    moved), and a record per corner.
    """
    ring = _ccw_ring(ring_m)
    n = len(ring)
    by_face = {}
    for p in planes:
        q = WallPlane(face=p.face, p0=p.p0.copy(), p1=p.p1.copy(), normal=p.normal.copy(), z_lo=p.z_lo, z_hi=p.z_hi)
        by_face[p.face] = q
    original = {f: (q.p0.copy(), q.p1.copy()) for f, q in by_face.items()}
    # an observation is a point in the world: the face's end as seen in the
    # frames, measured along the face as it was when the strips were sampled
    ends: Dict[Tuple[int, str], Optional[np.ndarray]] = {}
    for (f, which), s_obs in observations.items():
        if s_obs is None or f not in by_face:
            continue
        o0, o1 = original[f]
        e = (o1 - o0) / max(float(np.hypot(*(o1 - o0))), 1e-9)
        ends[(f, which)] = o0 + float(s_obs) * e
    skip = skip or {}
    new_ring = ring.copy()
    records = {k: CornerRecord(corner=k, status="no-observation", before=ring[k].tolist(), after=ring[k].tolist()) for k in range(n)}
    for k, why in skip.items():
        records[k].status = why

    def corner_faces(k):
        return (k - 1) % n, k                                          # edge into the corner, edge out of it

    def gap_at(k, fa, fb) -> Optional[float]:
        A, B = by_face.get(fa), by_face.get(fb)
        EA, EB = ends.get((fa, "end")), ends.get((fb, "start"))
        if A is None or B is None:
            return None
        P = _line_intersection(A, B)
        if P is None:
            return None
        g = 0.0; m = 0
        if EA is not None:
            g += float(np.linalg.norm(EA - P)); m += 1
        if EB is not None:
            g += float(np.linalg.norm(EB - P)); m += 1
        return g if m else None

    for k in range(n):
        fa, fb = corner_faces(k)
        records[k].gap_before_m = gap_at(k, fa, fb)

    for _ in range(max_passes):
        moved = 0.0
        for k in range(n):
            if k in skip:
                continue
            fa, fb = corner_faces(k)
            A, B = by_face.get(fa), by_face.get(fb)
            if A is None or B is None:
                records[k].status = "skipped-no-plane"
                continue
            EA, EB = ends.get((fa, "end")), ends.get((fb, "start"))
            P = _line_intersection(A, B)
            if EA is None and EB is None:
                continue
            if EA is not None and EB is not None and np.linalg.norm(EA - EB) > max_disagree_m and not (fa in verified and fb in verified):
                records[k].status = "refused"; records[k].note = f"the two faces' edges disagree by {np.linalg.norm(EA - EB):.1f} m"
                continue
            if fa in verified and fb in verified:
                C = P if P is not None else ring[k]
                records[k].status = "measured"
            elif fa in verified:
                C = EA if EA is not None else P
                if C is None:
                    continue
                _shift_through(B, C)
            elif fb in verified:
                C = EB if EB is not None else P
                if C is None:
                    continue
                _shift_through(A, C)
            else:
                C = 0.5 * (EA + EB) if (EA is not None and EB is not None) else (EA if EA is not None else EB)
                _shift_through(A, C); _shift_through(B, C)
            if np.linalg.norm(C - ring[k]) > max_shift_m:
                # put the lines back and leave the corner alone
                for f in (fa, fb):
                    by_face[f].p0, by_face[f].p1 = original[f][0].copy(), original[f][1].copy()
                records[k].status = "refused"; records[k].note = f"a {np.linalg.norm(C - ring[k]):.1f} m move"
                continue
            moved = max(moved, float(np.linalg.norm(C - new_ring[k])))
            new_ring[k] = C
            if records[k].status != "measured":
                records[k].status = "snapped"
        # re-derive every corner from its two lines, where both exist
        for k in range(n):
            if k in skip or records[k].status in ("refused", "skipped-no-plane", "no-observation"):
                continue
            fa, fb = corner_faces(k)
            P = _line_intersection(by_face[fa], by_face[fb])
            if P is not None and np.linalg.norm(P - ring[k]) <= max_shift_m:
                new_ring[k] = P
        for f, q in by_face.items():
            c0, c1 = new_ring[f].copy(), new_ring[(f + 1) % n].copy()
            if f in verified:
                # a verified line keeps its direction and offset; its ends slide along it
                o0, o1 = original[f]
                e = (o1 - o0) / max(float(np.hypot(*(o1 - o0))), 1e-9)
                c0 = o0 + float((c0 - o0) @ e) * e
                c1 = o0 + float((c1 - o0) @ e) * e
            q.p0, q.p1 = c0, c1
        if moved <= tol_m:
            break

    for k in range(n):
        fa, fb = corner_faces(k)
        if records[k].status == "snapped" and np.linalg.norm(new_ring[k] - ring[k]) < 0.02:
            records[k].status = "measured"                              # observed, and already there
        records[k].after = new_ring[k].tolist()
        records[k].gap_after_m = gap_at(k, fa, fb)
        records[k].votes = {f"face {fa} end": 1 if (fa, "end") in ends else 0,
                            f"face {fb} start": 1 if (fb, "start") in ends else 0}
    out_planes = []
    for p in planes:
        q = by_face[p.face]
        seg = q.p1 - q.p0
        L = float(np.hypot(*seg))
        if L > 1e-6:
            q.normal = np.array([seg[1], -seg[0]]) / L
        out_planes.append(q)
    return np.vstack([new_ring, new_ring[:1]]), out_planes, [records[k] for k in range(n)]
