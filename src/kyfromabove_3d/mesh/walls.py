"""The walls of the mesh: each face's profile bands as quads, capped at the roof where
the LiDAR did not see them, with ledges, returns and corners."""

from __future__ import annotations

import numpy as np

from kyfromabove_3d.features import (
    LEVEL_FACE_STRIDE,
    WALL_STATUSES,
    ArcWall,
    FaceBand,
    GroundField,
    WallPlane,
    band_extent,
)

# wall bands the LiDAR did not see at their height: they stop at the roof just
# inside the wall line instead of their band's top
CAPPED_STATUSES = ("edge", "extended", "mirrored")

# how far a wall the LiDAR saw may stand over the roof just inside it
PARAPET_M = 1.5


# --------------------------------------------------------------------------- walls

def _orient(corners: np.ndarray, want_normal: np.ndarray) -> np.ndarray:
    """Wind a quad so its face normal points along ``want_normal``."""
    n = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    return corners if np.dot(n, want_normal) >= 0 else corners[::-1].copy()


class RoofField:
    """The roof height field, queryable along an edge: is there roof at a
    given height just inside a wall line, and how high is the roof beside
    an edge. Built once per building from ``roof_heightfield``."""

    def __init__(self, xs, ys, Z, valid):
        self.xs, self.ys, self.Z, self.valid = xs, ys, Z, valid
        self.cell = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
        X, Y = np.meshgrid(xs, ys)
        self.pts = np.column_stack([X[valid], Y[valid]])
        self.zs = Z[valid]

    def _near(self, xy: np.ndarray, radius: float):
        """Max roof height within ``radius`` of each xy (nan where none)."""
        if not len(self.pts):
            return np.full(len(xy), np.nan)
        from scipy.spatial import cKDTree
        if not hasattr(self, "_tree"):
            self._tree = cKDTree(self.pts)
        out = np.full(len(xy), np.nan)
        for i, nb in enumerate(self._tree.query_ball_point(xy, radius)):
            if nb:
                out[i] = self.zs[nb].max()
        return out

    def behind(self, plane: WallPlane, offset: float, z: float, a0: float, a1: float,
               inward_m=(1.0, 4.0), step_m: float = 1.0, slack_m: float = 1.5,
               keep_when_no_data: bool = False):
        """Stations along the edge from ``a0`` to ``a1`` (metres from p0)
        and whether the roof just inside the wall line reaches ``z``: no
        roof behind a wall at that height means no wall there."""
        L = plane.length_m
        if L < 1e-6 or a1 - a0 < step_m:
            return np.array([]), np.array([], dtype=bool)
        st = np.arange(a0 + step_m / 2, a1, step_m)
        depths = np.arange(inward_m[0], inward_m[1] + 1e-6, self.cell)
        ok = np.zeros(len(st), dtype=bool)
        seen = np.zeros(len(st), dtype=bool)                 # any cell at all inside the wall line
        for d in depths:
            # (along, offset) -> plan through the face's own frame: straight
            # for a plane, rolled back onto the circle for a curved wall
            h = self._near(plane.world(st, np.full(len(st), offset - d)), max(self.cell * 0.75, 0.6))
            seen |= np.isfinite(h)
            ok |= np.nan_to_num(h, nan=-np.inf) >= z - slack_m
        if keep_when_no_data:
            # "no cell here" is not "no roof here": a plane the fitted
            # tiles verified keeps its band where the field is blank
            # (PNC's crown is thin in the returns along its short faces)
            ok |= ~seen
        return st, ok

    def tops_along(self, plane: WallPlane, offset: float, a0: float, a1: float, first_m: float = 0.25,
                   reach_m: float = 4.0, depth_m: float = 1.0, step_m: float = 1.0):
        """Stations from ``a0`` to ``a1`` and the roof's height just inside
        the wall line at each: the highest cell within ``depth_m`` of the
        first one met going in (from ``first_m``, up to ``reach_m``) -- where
        the field is blank along the edge the roof is found further in, and a
        tower standing back behind a podium's edge is not mistaken for the
        podium's roof. nan where there is none."""
        if a1 - a0 < 1e-6:
            return np.array([]), np.array([])
        n = max(1, int(round((a1 - a0) / step_m)))
        st = a0 + (np.arange(n) + 0.5) * (a1 - a0) / n
        depths = np.arange(first_m, reach_m + 1e-6, self.cell)
        H = np.array([self._near(plane.world(st, np.full(n, offset - d)), max(self.cell * 0.75, 0.6)) for d in depths])
        top = np.full(n, np.nan)
        for j in range(n):
            ok = np.flatnonzero(np.isfinite(H[:, j]))
            if len(ok):
                d0 = depths[ok[0]]
                top[j] = np.nanmax(H[(depths >= d0) & (depths <= d0 + depth_m), j])
        return st, top

    def outside_along(self, plane, offset: float, a0: float, a1: float, near_m=(0.5, 2.5), step_m: float = 1.0):
        """The roof's height just outside a wall line, from ``a0`` to ``a1``:
        the median, per station, of the cells ``near_m`` outside it -- the
        podium's roof beside a tower's wall; nan where there is none."""
        if a1 - a0 < 1e-6:
            return np.array([]), np.array([])
        n = max(1, int(round((a1 - a0) / step_m)))
        st = a0 + (np.arange(n) + 0.5) * (a1 - a0) / n
        hs = np.array([self._near(plane.world(st, np.full(n, offset + d)), max(self.cell * 0.75, 0.6))
                       for d in np.arange(near_m[0], near_m[1] + 1e-6, self.cell)])
        with np.errstate(all="ignore"):
            return st, np.nanmedian(np.where(np.isfinite(hs), hs, np.nan), axis=0)

    def capped_runs(self, plane, offset, z_lo, z_hi, a0, a1, min_run_m: float = 1.0, tol_m: float = 0.5,
                    above_m: float = 0.0, keep_unknown: bool = False):
        """[a0, a1] cut where the roof's height just inside changes, each
        stretch's wall top the roof there plus ``above_m`` (at most
        ``z_hi``); stretches whose roof is not above ``z_lo`` are dropped. A
        wall drawn under a roof edge (an ``edge`` band) stops at the roof,
        not at its band's top. ``keep_unknown``: a stretch with no roof found
        beside it, or none above ``z_lo``, keeps ``z_hi`` instead of going --
        for a wall the LiDAR saw, which the field cannot overrule."""
        st, top = self.tops_along(plane, offset, a0, a1)
        if not len(st):
            return [(a0, a1, z_hi)] if keep_unknown else []
        half = 0.5 * (a1 - a0) / len(st)
        out = []
        for s, t in zip(st, np.minimum(top + above_m, z_hi)):
            if not np.isfinite(t) or t <= z_lo + 0.2:
                if not keep_unknown:
                    continue
                t = z_hi
            if out and abs(out[-1][1] - (s - half)) < 1e-6 and abs(out[-1][2] - t) <= tol_m:
                out[-1][1] = s + half
                out[-1][3].append(t)
            else:
                out.append([s - half, s + half, t, [t]])
        return [(max(a0, a), min(a1, c), float(np.median(ts))) for a, c, _, ts in out if c - a >= min_run_m]

    def runs_with_roof(self, plane, offset, z, a0, a1, min_run_m: float = 1.5, keep_when_no_data: bool = False):
        """The stretches of [a0, a1] that have roof behind them."""
        st, ok = self.behind(plane, offset, z, a0, a1, keep_when_no_data=keep_when_no_data)
        if not len(st):
            return []
        runs, start = [], None
        half = 0.5 * (st[1] - st[0]) if len(st) > 1 else 0.5
        for s, o in zip(st, ok):
            if o and start is None:
                start = max(a0, s - half)
            if not o and start is not None:
                runs.append((start, min(a1, s - half))); start = None
        if start is not None:
            runs.append((start, a1))
        return [(a, b) for a, b in runs if b - a >= min_run_m]

    def top_along(self, edge: WallPlane, within_m: float = 2.0, step_m: float = 3.0):
        """The roof height beside an edge, per ``step_m`` stretch: the
        median valid cell within ``within_m`` *inside* the edge; nan where
        there is none. Inside only: a tower's edge stands on its podium's
        roof, and the cells on both sides mixed the two -- the Omni's 57 m
        south wing read 38-49 m along its south wall, and 17 m (no wall at
        all) wherever the podium's cells were the majority."""
        L = edge.length_m
        n_st = max(1, int(np.ceil(L / step_m)))
        edges = np.linspace(0.0, L, n_st + 1)
        out = []
        if not len(self.pts):
            return [(a, b, np.nan) for a, b in zip(edges[:-1], edges[1:])]
        t, off = edge.local(self.pts)
        t = np.clip(t, 0.0, L)
        dist = np.linalg.norm(self.pts - edge.world(t, np.zeros(len(t))), axis=1)
        # the first half metre inside mixes the wall's returns into the roof's;
        # a coarse field (1.5 m cells, sparse LiDAR) cannot spare its first row,
        # and reaches two cells in (the Kentucky Science Center lost a 3 m
        # stretch of wall to a stretch with one cell in reach)
        skip = 0.5 if self.cell <= 1.0 else 0.0
        reach = max(within_m, 2.0 * self.cell) + skip
        for a, b in zip(edges[:-1], edges[1:]):
            m = (dist <= reach) & (off <= -skip) & (t >= a) & (t < b)
            out.append((float(a), float(b), float(np.median(self.zs[m])) if m.any() else np.nan))
        return out


def _runs(plane: WallPlane, b: FaceBand, roof: "RoofField | None" = None):
    """(along_lo, along_hi, offset) runs of one band across its extent --
    and, given the roof field, only where there is roof behind the wall."""
    s0, s1 = band_extent(plane, b)
    if s1 - s0 < 0.2:
        return []
    cuts = {s0, s1}
    for a0, a1, _ in b.segments:
        for c in (a0, a1):
            if s0 < c < s1:
                cuts.add(c)
    cuts = sorted(cuts)
    runs = []
    for a, c in zip(cuts[:-1], cuts[1:]):
        mid = 0.5 * (a + c)
        off = b.offset_m
        for a0, a1, oseg in b.segments:
            if a0 <= mid < a1:
                off = oseg
        runs.append((a, c, off))
    # merge neighbours at the same offset
    merged = [list(runs[0])]
    for a, c, off in runs[1:]:
        if abs(off - merged[-1][2]) < 1e-6:
            merged[-1][1] = c
        else:
            merged.append([a, c, off])
    if roof is None:
        return [tuple(r) for r in merged]
    from kyfromabove_3d.features import VerifiedWallPlane
    verified = isinstance(plane, VerifiedWallPlane)
    clipped = []
    for a, c, off in merged:
        for ra, rc in roof.runs_with_roof(plane, off, b.z_lo, a, c, keep_when_no_data=verified):
            clipped.append((ra, rc, off))
    return clipped


def plane_bands(plane: WallPlane, bands: list[FaceBand], dense_per_m2: float = 0.5) -> list[FaceBand]:
    """The bands of a *verified* plane: the whole face, at the plane, in the
    profile's height steps -- the same rule as ``profile_keeps`` in the
    colourise. The profile removes only what it has evidence against: an
    open band (a loggia) is left out, a band whose returns are dense and
    stop short is cut to their extent, and the corner clips hold.
    """
    from kyfromabove_3d.features import VerifiedWallPlane
    if not isinstance(plane, VerifiedWallPlane):
        return bands
    L = float(np.hypot(*(plane.p1 - plane.p0)))
    out = []
    steps = [(b.z_lo, b.z_hi, b) for b in bands] or [(plane.z_lo, plane.z_hi, None)]
    for z_lo, z_hi, b in steps:
        if z_lo >= plane.z_hi - 0.05:
            # above the verified plane the profile speaks for itself
            if b is not None:
                out.append(b)
            continue
        z_hi = min(z_hi, plane.z_hi)
        if b is not None and b.status == "open":
            continue
        clip_lo, clip_hi = (b.clip_lo, b.clip_hi) if b is not None else (None, None)
        if b is not None and b.status == "observed" and (b.along_hi - b.along_lo) < 0.8 * L:
            area = max(b.along_hi - b.along_lo, 1.0) * (b.z_hi - b.z_lo)
            if b.n_returns / area >= dense_per_m2:
                s0, s1 = band_extent(plane, b, full_width_fraction=0.8)
                clip_lo = s0 if clip_lo is None else max(clip_lo, s0)
                clip_hi = s1 if clip_hi is None else min(clip_hi, s1)
        out.append(FaceBand(z_lo=z_lo, z_hi=z_hi, offset_m=0.0, n_returns=b.n_returns if b else 0,
                            spread_m=0.0, along_lo=0.0, along_hi=L, status="observed",
                            segments=[], clip_lo=clip_lo, clip_hi=clip_hi))
    return out


def wall_geometry(plane: WallPlane, bands: list[FaceBand], roof: "RoofField | None" = None,
                  ground_field: "GroundField | None" = None, edge_face: int | None = None,
                  foot_to_roof: bool = False, wall_cap: bool = True):
    """Quads for one face: (corners(4,3), normal(3), uv(4,2) or None, kind).

    ``kind`` is "wall" (textured with the face image), "ledge" or "return"
    (flat colour). UVs are in face texture space: u along the edge from p0,
    v up from the lowest band.
    """
    seg = plane.p1 - plane.p0
    L = float(np.hypot(*seg))
    e1 = np.array([seg[0] / L, seg[1] / L, 0.0])
    n3 = np.array([plane.normal[0], plane.normal[1], 0.0])
    p0 = np.array([plane.p0[0], plane.p0[1], 0.0])

    def P(s, off, z):
        return p0 + s * e1 + off * n3 + np.array([0.0, 0.0, z])

    # An open band -- a colonnade, a loggia -- has its wall behind the
    # piers; draw it there, so the void reads as a recess and not as a
    # missing wall (Humana's Main Street front was a hole at the colonnade).
    # Only under a solid band: a colonnade carries floors above it. An open
    # band with nothing solid above is the roof seen over the parapet, and
    # drawn as a recess it stood a wall panel 3-7 m inboard on the roof of
    # every face of the Kentucky Home Life Building. A face with no solid
    # band at all takes the prism instead.
    # (the wall above must have been seen: a band drawn under a roof edge
    # because nothing said it was not there does not carry a colonnade --
    # above RIVUE's open band it stood the wall behind the piers 7 m in, on
    # the roof)
    solid_tops = [b.z_lo for b in bands if b.status in WALL_STATUSES and b.status != "edge" and b.offset_m is not None]
    used = []
    has_solid = any(b.status in WALL_STATUSES and b.offset_m is not None for b in bands)
    for b in bands:
        if b.status in WALL_STATUSES and b.offset_m is not None:
            used.append(b)
        elif (b.status == "open" and b.back_offset_m is not None and (b.along_hi - b.along_lo) >= 3.0
              and any(z >= b.z_hi - 1e-6 for z in solid_tops)):
            used.append(FaceBand(b.z_lo, b.z_hi, b.back_offset_m, b.n_returns, b.spread_m, b.along_lo, b.along_hi,
                                 "observed", segments=[], clip_lo=b.clip_lo, clip_hi=b.clip_hi))
    if not has_solid:
        return [], None
    z0 = min(b.z_lo for b in used)
    z1 = max(b.z_hi for b in used)
    quads = []
    # An upper level's lowest band stands on the roof below it: where the
    # podium's roof beside the wall is lower than the level's foot (RIVUE's
    # tower starts at 9.4 m beside a 4.5 m podium roof) it reaches down to it,
    # stretch by stretch; the face's picture is extended down to match.
    z_first, foot_z = z0, None
    if foot_to_roof and roof is not None and edge_face is not None and edge_face >= LEVEL_FACE_STRIDE:
        b_low = min(used, key=lambda x: x.z_lo)
        _, below = roof.outside_along(plane, b_low.offset_m, 0.0, L)
        if len(below) and np.isfinite(below).any() and float(np.nanmin(below)) < z_first - 0.3:
            foot_z = float(np.nanmin(below))
            z0 = foot_z
    # with the wall cap off (--no-wall-cap, the build before it), only a wall drawn under a
    # roof edge stops at the roof
    capped, parapet = (CAPPED_STATUSES, PARAPET_M) if wall_cap else (("edge",), None)
    prev = None                                     # (runs, z_hi) of the band below
    for b in used:
        runs = _runs(plane, b, roof)
        is_bottom = (ground_field is not None and edge_face is not None and
                     edge_face < LEVEL_FACE_STRIDE and abs(b.z_lo - z0) < 1e-3)
        # a wall the LiDAR did not see at this height -- drawn under a roof
        # edge because nothing says it is not there, carried up from the wall
        # below, or copied from the face opposite -- stops at the roof just
        # inside, stretch by stretch. Carried up to the band's top, 2b18f52d's
        # east and west walls stood 3-7 m above the eave of its mansard, a
        # panel in front of the slope.
        # A wall it did see stops no higher than a parapet over that roof --
        # the top band only: the returns of 2870eeb0's 1 m parapet put a band
        # at 20-25 m, drawn to 24.6 m, 4 m over its 20.3 m roof. A seen band
        # with wall above it is the wall: where an upper level's outline takes
        # in a strip of the roof below (the Presbyterian Church headquarters'
        # tower) the roof just inside reads low.
        top_band = b.z_hi >= z1 - 1e-3
        if roof is None or (b.status not in capped and (parapet is None or not top_band)):
            tops = [(a, c, off, b.z_hi) for a, c, off in runs]
        # (a wall standing out past its outline holds the roof inside the
        # outline: from the wall in, Chase Bank Plaza's tower face read its
        # podium's roof 2 m out and lost its wall above 14 m)
        elif b.status in capped:
            tops = [(ra, rc, off, zt) for a, c, off in runs
                    for ra, rc, zt in roof.capped_runs(plane, min(off, 0.0), b.z_lo, b.z_hi, a, c)]
        else:
            tops = [(ra, rc, off, zt) for a, c, off in runs
                    for ra, rc, zt in roof.capped_runs(plane, min(off, 0.0), b.z_lo, b.z_hi, a, c, min_run_m=0.0,
                                                       above_m=parapet, keep_unknown=True)]
        runs = [(a, c, off) for a, c, off, _ in tops]
        foot = foot_z is not None and abs(b.z_lo - z_first) < 1e-3
        if foot:
            pieces = []
            for a, c, off, z_top in tops:
                st, below = roof.outside_along(plane, off, a, c)
                lo = float(np.nanmin(below)) if len(below) and np.isfinite(below).any() else np.nan
                pieces.append((a, c, off, z_top, max(lo, foot_z) if np.isfinite(lo) and lo < b.z_lo - 0.3 else b.z_lo))
        else:
            pieces = [(a, c, off, z_top, b.z_lo) for a, c, off, z_top in tops]
        for a, c, off, z_top, z_bot in pieces:
            if foot and z_bot < b.z_lo:
                corners = np.array([P(a, off, z_bot), P(c, off, z_bot), P(c, off, z_top), P(a, off, z_top)])
                uv = np.array([[a / L, (z_bot - z0) / (z1 - z0)], [c / L, (z_bot - z0) / (z1 - z0)],
                               [c / L, (z_top - z0) / (z1 - z0)], [a / L, (z_top - z0) / (z1 - z0)]])
                quads.append((corners, n3, uv, "wall"))
                continue
            if is_bottom:
                za = ground_field.elevation_along_edge(edge_face, a, L)
                zc = ground_field.elevation_along_edge(edge_face, c, L)
                corners = np.array([P(a, off, za), P(c, off, zc), P(c, off, z_top), P(a, off, z_top)])
            else:
                corners = np.array([P(a, off, b.z_lo), P(c, off, b.z_lo), P(c, off, z_top), P(a, off, z_top)])
            uv = np.array([[a / L, (b.z_lo - z0) / (z1 - z0)], [c / L, (b.z_lo - z0) / (z1 - z0)],
                           [c / L, (z_top - z0) / (z1 - z0)], [a / L, (z_top - z0) / (z1 - z0)]])
            quads.append((corners, n3, uv, "wall"))
        # return walls where the offset changes along the band
        for (a, c, off), (a2, c2, off2) in zip(runs[:-1], runs[1:]):
            if abs(off - off2) < 0.05:
                continue
            if is_bottom:
                zc = ground_field.elevation_along_edge(edge_face, c, L)
                corners = np.array([P(c, off, zc), P(c, off2, zc), P(c, off2, b.z_hi), P(c, off, b.z_hi)])
            else:
                corners = np.array([P(c, off, b.z_lo), P(c, off2, b.z_lo), P(c, off2, b.z_hi), P(c, off, b.z_hi)])
            # faces towards the deeper side's open air: along -e1 if the
            # next run stands further out, else +e1
            want = -e1 if off2 > off else e1
            quads.append((_orient(corners, want), want, None, "return"))
        # ledge between this band and the one below
        if prev is not None and abs(prev[1] - b.z_lo) < 0.05:
            for a, c, off in runs:
                for pa, pc, poff in prev[0]:
                    lo, hi = max(a, pa), min(c, pc)
                    if hi - lo < 0.2 or abs(off - poff) < 0.05:
                        continue
                    z = b.z_lo
                    corners = np.array([P(lo, poff, z), P(hi, poff, z), P(hi, off, z), P(lo, off, z)])
                    want = np.array([0.0, 0.0, 1.0 if off < poff else -1.0])
                    quads.append((_orient(corners, want), want, None, "ledge"))
        prev = (runs, b.z_hi)
    return quads, (z0, z1, L)


def corner_returns(plane: WallPlane, bands: list[FaceBand], neighbour_offset, roof: "RoofField | None" = None,
                   tol_m: float = 0.05):
    """Short walls closing the corners of a face's *matched* bands.

    A matched band moved off the footprint edge (by the frames, see
    kyfromabove_3d.wall_match) ends where the next face's wall does not:
    a prism beside it stands on the footprint line, a band beside it at its
    own depth. Seen past the corner, the gap opened onto the building's
    empty inside -- cfb7194a's two faces matched 1 m in made 1 m slits from
    ground to roof. Each end of each matched run within half a metre of the
    face's end gets a vertical quad from the run's end to where
    ``neighbour_offset(end, z)`` (0 = p0's corner, 1 = p1's) puts the next
    wall at that corner: a point in plan, or None to leave it open."""
    seg = plane.p1 - plane.p0
    L = float(np.hypot(*seg))
    e1 = np.array([seg[0] / L, seg[1] / L, 0.0])
    n3 = np.array([plane.normal[0], plane.normal[1], 0.0])
    p0 = np.array([plane.p0[0], plane.p0[1], 0.0])
    out = []
    for b in bands:
        if b.status != "matched" or b.offset_m is None:
            continue
        runs = _runs(plane, b, roof)
        if not runs:
            continue
        zm = 0.5 * (b.z_lo + b.z_hi)
        for end, (s_, off) in ((0, (runs[0][0], runs[0][2])), (1, (runs[-1][1], runs[-1][2]))):
            if (end == 0 and s_ > 0.5) or (end == 1 and s_ < L - 0.5):
                continue
            q = neighbour_offset(end, zm)
            if q is None:
                continue
            a = p0 + s_ * e1 + off * n3
            bq = np.array([q[0], q[1], 0.0])
            d = bq - a
            if np.hypot(d[0], d[1]) < tol_m:
                continue
            corners = np.array([a + [0, 0, b.z_lo], bq + [0, 0, b.z_lo], bq + [0, 0, b.z_hi], a + [0, 0, b.z_hi]])
            side = np.array([-d[1], d[0], 0.0])                              # horizontal, square to the gap
            side /= np.linalg.norm(side)
            want = side if side @ (n3 + (e1 if end == 1 else -e1)) >= 0 else -side    # out of the building
            out.append((_orient(corners, want), want, None, "return"))
    return out


def roll_quads(quads, arc: ArcWall, step_m: float = 1.5):
    """Quads built in an arc's unrolled frame -> quads on the arc. Each is
    cut every ``step_m`` along the arc (and across, for a ledge), its
    corners rolled back onto the circle and its normal turned with the
    frame; uv is interpolated the same way, so the face texture still
    lands where the profile put the wall."""
    out = []
    for corners, n, uv, kind in quads:
        c = np.asarray(corners, dtype=np.float64)
        span_u = abs(c[1, 0] - c[0, 0]); span_v = abs(c[3, 0] - c[0, 0])
        nu = max(1, int(np.ceil(span_u / step_m))); nv = max(1, int(np.ceil(span_v / step_m)))
        uvq = np.asarray(uv, dtype=np.float64) if uv is not None else None
        for i in range(nu):
            for j in range(nv):
                u0, u1 = i / nu, (i + 1) / nu
                v0, v1 = j / nv, (j + 1) / nv
                def at(u, v, arr):
                    return (1 - v) * ((1 - u) * arr[0] + u * arr[1]) + v * ((1 - u) * arr[3] + u * arr[2])
                sub = np.array([at(u0, v0, c), at(u1, v0, c), at(u1, v1, c), at(u0, v1, c)])
                mid = sub[:, 0].mean()
                world, n_world = arc.to_world_3d(sub, np.repeat(np.asarray(n, dtype=np.float64)[None, :], 4, axis=0))
                n_mid = arc.to_world_3d(np.array([[mid, 0.0, 0.0]]), np.asarray(n, dtype=np.float64)[None, :])[1][0]
                sub_uv = np.array([at(u0, v0, uvq), at(u1, v0, uvq), at(u1, v1, uvq), at(u0, v1, uvq)]) if uvq is not None else None
                out.append((_orient(world, n_mid), n_mid, sub_uv, kind))
    return out


class ArcGround:
    """The ground along a curved wall: the ground field's elevation at the
    arc's first vertex to its last, interpolated by distance along the arc
    (``elevation_along_edge`` keys by ring vertex, which an arc spans many of)."""

    def __init__(self, ground_field: GroundField, arc: ArcWall):
        self.z0 = ground_field.elevation_at_vertex(arc.first_edge)
        self.z1 = ground_field.elevation_at_vertex(arc.last_edge + 1)

    def elevation_along_edge(self, edge_idx: int, s: float, L: float) -> float:
        t = float(np.clip(s / max(L, 1e-6), 0.0, 1.0))
        return self.z0 + t * (self.z1 - self.z0)


def _wall_behind_band(off: np.ndarray, expect: np.ndarray, band_off: float, plane_off: float,
                      matched: np.ndarray, max_behind_m: float = 2.0) -> np.ndarray:
    """The coloured points of a band whose own offset caught few of them.

    The band's points may sit a little behind it -- the verified plane's
    offset, or the most common offset among the points -- where the profile
    put the band on a ledge or a mullion line. Only a wall *behind* the band
    (inside the building) and within ``max_behind_m`` is accepted, and only
    when it holds at least twice the points: in front of the band are the
    street, a light well's far side and a neighbour's facade, whose colours
    must not be painted on this one."""
    candidates = []
    if -max_behind_m <= plane_off - band_off < -0.2:
        candidates.append(plane_off)
    window = (off >= band_off - max_behind_m) & (off < band_off)
    if window.sum() >= 30:
        hist, edges = np.histogram(off[window], bins=max(int(max_behind_m / 0.1), 1),
                                   range=(band_off - max_behind_m, band_off))
        k = int(np.argmax(hist))
        candidates.append(0.5 * (edges[k] + edges[k + 1]))
    for cand in candidates:
        m = np.abs(off - (expect + cand - band_off)) < 0.6
        if m.sum() > max(2 * matched.sum(), 50):
            return m
    return matched
