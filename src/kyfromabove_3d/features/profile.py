"""A face as a profile of bands, for buildings that are not prisms: the bands, overhangs,
crowns, clipping by neighbours, extrusion, and hidden faces mirrored."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .walls import ArcWall, WallPlane, wall_planes

# --- a face as a profile of bands, for buildings that are not prisms --------

@dataclass
class FaceBand:
    """One horizontal band of a wall face and where the LiDAR says the wall is.

    ``offset_m`` is the wall's distance outward from the footprint edge (a
    negative offset is inside the drawn footprint); ``status`` says how the
    band is known: ``observed`` from its own returns, ``bridged`` between two
    observed bands that agree, or ``unknown``.
    """

    z_lo: float
    z_hi: float
    offset_m: Optional[float]
    n_returns: int
    spread_m: float                 # median absolute deviation of the returns about the offset
    along_lo: float                 # extent along the edge the returns cover, metres from p0
    along_hi: float
    status: str
    # Where the wall departs from the band's offset along the edge: runs of
    # (along_lo, along_hi, offset) at their own offset -- a recessed bay, a
    # projecting stair tower. Empty when the band is one flat wall.
    segments: List[Tuple[float, float, float]] = field(default_factory=list)
    # Where the neighbouring walls stand at this height, measured along this
    # edge from p0: the wall cannot run past them. None = unclipped.
    clip_lo: Optional[float] = None
    clip_hi: Optional[float] = None
    # An open band (a loggia, a colonnade) has two surfaces; this is the
    # deeper one, where the wall behind the piers stands. None otherwise.
    back_offset_m: Optional[float] = None
    # How much of the band's height the deeper surface's returns fill, in
    # 0.5 m steps (0-1). A wall behind piers runs up through the band; a roof
    # seen over a parapet, or two, sits at one or two heights. None if not open.
    back_fill: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"z_lo": self.z_lo, "z_hi": self.z_hi,
                "clip_lo": None if self.clip_lo is None else round(float(self.clip_lo), 1),
                "clip_hi": None if self.clip_hi is None else round(float(self.clip_hi), 1),
                "back_offset_m": None if self.back_offset_m is None else round(float(self.back_offset_m), 2),
                "back_fill": None if self.back_fill is None else round(float(self.back_fill), 2),
                "offset_m": None if self.offset_m is None else round(float(self.offset_m), 2),
                "n_returns": int(self.n_returns), "spread_m": round(float(self.spread_m), 2),
                "along_lo": round(float(self.along_lo), 1), "along_hi": round(float(self.along_hi), 1),
                "segments": [[round(a, 1), round(b, 1), round(c, 2)] for a, b, c in self.segments],
                "status": self.status}


def _band_segments(a: np.ndarray, o: np.ndarray, offset: float, L: float,
                   cell_m: float = 3.0, step_m: float = 0.8, min_cell: int = 6) -> List[Tuple[float, float, float]]:
    """Runs along the edge where the wall stands at a different offset from
    the band's. The profile is otherwise one offset per band, and that
    filled the recessed bay down the middle of Humana's south face -- 13 m
    wide, 65 m tall, 3-5 m deep -- flat. Each ``cell_m`` cell along the edge
    gets the modal offset of its own returns; cells that differ from the
    band's offset by more than ``step_m``, and agree with their neighbour,
    become a segment. Cells with too few returns keep the band's offset."""
    edges = np.arange(0.0, L + cell_m, cell_m)
    cells = []
    for c0, c1 in zip(edges[:-1], edges[1:]):
        m = (a >= c0) & (a < c1)
        if m.sum() < min_cell:
            cells.append(None)
            continue
        oc = o[m]
        hist, be = np.histogram(oc, bins=np.arange(np.floor(oc.min()), oc.max() + 0.5, 0.5))
        mode = be[int(np.argmax(hist))] + 0.25
        near = np.abs(oc - mode) <= 0.6
        cells.append(float(np.median(oc[near])) if near.sum() >= min_cell else None)
    segs: List[Tuple[float, float, float]] = []
    run_start = None
    run_vals: List[float] = []
    for i, v in enumerate(cells + [None]):
        differs = v is not None and abs(v - offset) > step_m
        if differs and (run_start is None or abs(v - run_vals[-1]) <= step_m / 2):
            if run_start is None:
                run_start = i
            run_vals.append(v)
            continue
        if run_start is not None and len(run_vals) >= 2:            # two cells: 6 m, not a stray
            segs.append((float(edges[run_start]), float(edges[i]), float(np.median(run_vals))))
        run_start, run_vals = (i, [v]) if differs else (None, [])
    return segs


def face_profile(plane: WallPlane, xyz: np.ndarray, band_m: float = 5.0,
                 offset_window_m: Optional[Tuple[float, float]] = None,
                 min_returns: int = 30, max_spread_m: float = 0.45,
                 min_along_fraction: float = 0.25,
                 max_bridge_m: float = 80.0, max_step_m: float = 0.8,
                 local_rise: float = 0.25, local_strip_m: float = 2.0) -> List[FaceBand]:
    """Where a face's wall stands, band by band, from the returns near it.

    A single plane per face fits a slab tower and fails a building like the
    Humana Building, whose east wall stands 2 m outside its footprint line
    for a hundred metres and then steps 6 m back at the crown, and whose
    south wall the flight saw from the ground to 40 m and again from 90 m
    up but never between. Measured band by band, each of those is simple:
    the returns in a band pile up at one offset from the footprint edge, and
    the wall is there.

    A band is *observed* when at least ``min_returns`` returns sit within
    0.6 m of the band's modal offset, spread no more than ``max_spread_m``
    about it, and reach along at least ``min_along_fraction`` of the edge.
    Unobserved bands between two observed bands whose offsets agree within
    ``max_step_m`` are *bridged* at that offset, up to ``max_bridge_m`` of
    gap -- the LiDAR saw the wall above and below and the same wall is
    between. A step larger than that is kept as a step, and nothing is
    bridged across it. Nothing is extruded above the highest observed band
    or below the lowest: a crown or a portico the LiDAR never reached stays
    unknown.
    """
    seg = plane.p1 - plane.p0
    L = float(np.hypot(*seg))
    if L < 1e-6 or len(xyz) == 0:
        return []
    if offset_window_m is None:
        # 8 m inside the footprint edge, 3 m outside. A tower standing
        # further back than that from the edge its podium was traced along
        # -- Humana's shaft is 25 m inside its north edge -- is out of this
        # window on purpose: widening it to the far side let returns from
        # deep inside the block turn solid walls "open" on the other faces,
        # and the set-back shaft was seen too thinly (about 20 returns per
        # band) to pass the evidence rule anyway. Such a face is reported
        # as unmodelled, and is a job for a photograph.
        offset_window_m = (-8.0, 3.0)
    rel = xyz[:, :2] - plane.p0
    along = (rel @ seg) / (L * L)
    off = rel @ plane.normal
    # The corners are left out: the neighbouring wall's returns sit there at
    # every depth and belong to it.
    corner = 1.5 / L
    in_face = (along >= corner) & (along <= 1.0 - corner) & (off >= offset_window_m[0]) & (off <= offset_window_m[1])
    z = xyz[:, 2]
    bands: List[FaceBand] = []
    returns_of: List[Tuple[np.ndarray, np.ndarray]] = []   # (along m, offset m) per band, for segments later
    z0 = plane.z_lo
    while z0 < plane.z_hi - 1e-6:
        z1 = min(z0 + band_m, plane.z_hi)
        m = in_face & (z >= z0) & (z < z1)
        returns_of.append((along[m] * L, off[m]))
        band = FaceBand(z0, z1, None, int(m.sum()), 0.0, 0.0, 0.0, "unknown")
        if m.sum() >= min_returns:
            o = off[m]
            hist, edges = np.histogram(o, bins=np.arange(np.floor(offset_window_m[0]), offset_window_m[1] + 0.5, 0.5))
            mode = edges[int(np.argmax(hist))] + 0.25
            near = np.abs(o - mode) <= 0.6
            # Two walls in one band -- piers at the face and a recess metres
            # behind them, each with a good share of the returns -- is a
            # colonnade or a loggia, not a solid wall. Humana's crown is one:
            # a plane through the piers gets painted with the sky between
            # them. Such a band is "open" and nothing is extruded on it.
            # Measured, not assumed: the second-highest bin at least 2 m
            # from the mode must hold at least half what the mode's bin
            # holds. Through-glass returns behind a window wall are spread
            # thin and do not reach that; a second wall does.
            # Through a glass wall the laser also returns from the floor slab
            # just inside, 2-3 m behind: that is one wall, not two. A recess
            # 4 m or more behind the piers is a loggia.
            h2 = hist.copy()
            k = int(np.argmax(hist))
            h2[max(0, k - 8):k + 9] = 0
            if h2.max() >= 0.5 * hist[k] and h2.max() >= min_returns / 2:
                # the deeper of the two surfaces is the wall behind the
                # piers; its extent along the edge is where its returns are
                second = edges[int(np.argmax(h2))] + 0.25
                back = float(min(mode, second))
                on_back = np.abs(o - back) <= 0.6
                a_back = along[m][on_back] * L
                # does the deeper surface run up through the band (a wall) or
                # sit at one or two heights (a roof seen over the parapet)?
                z_back = z[m][on_back]
                steps, _ = np.histogram(z_back, bins=np.arange(z0, z1 + 0.25, 0.5))
                fill = float((steps >= max(3, 0.02 * len(z_back))).mean()) if len(steps) else 0.0
                bands.append(FaceBand(z0, z1, float(mode), int(m.sum()), 0.0,
                                      float(a_back.min()) if len(a_back) else 0.0,
                                      float(a_back.max()) if len(a_back) else 0.0, "open",
                                      back_offset_m=back, back_fill=fill))
                z0 = z1
                continue
            if near.sum() >= min_returns:
                offset = float(np.median(o[near]))
                spread = float(np.median(np.abs(o[near] - offset)))
                a = along[m][near] * L
                lo, hi = float(np.percentile(a, 2)), float(np.percentile(a, 98))
                # A wall's returns run up through the band; a roof slab or a
                # ledge caught at one offset sits at one height. Humana's
                # porch roof, 25 m inside the north edge, read as a wall band
                # until this: its returns all lie within a metre of 101 m.
                z_near = z[m][near]
                tall = (np.percentile(z_near, 90) - np.percentile(z_near, 10)) >= 0.5 * (z1 - z0)
                if tall and local_rise > 0:
                    # ... and they do so where they are: in ``local_strip_m``
                    # strips along the face, a wall's returns still climb the band
                    # (by ``local_rise`` of it, in the median strip); a roof's sit at
                    # one height. Over the whole face a roof that rises along it
                    # passed for a wall: f60dbc1c's, 10 m at one end and 12.5 m at
                    # the other, stood a wall 3 m inside its flat-roofed block.
                    strip = np.floor(along[m][near] * L / local_strip_m).astype(int)
                    rises = [float(np.percentile(z_near[strip == k], 90) - np.percentile(z_near[strip == k], 10))
                             for k in np.unique(strip) if (strip == k).sum() >= 3]
                    if len(rises) >= 3:
                        tall = float(np.median(rises)) >= local_rise * (z1 - z0)
                if spread <= max_spread_m and (hi - lo) >= min_along_fraction * L and tall:
                    band = FaceBand(z0, z1, offset, int(near.sum()), spread, lo, hi, "observed",
                                    segments=_band_segments(along[m] * L, o, offset, L, step_m=max_step_m))
        bands.append(band)
        z0 = z1

    # A wall is vertical: its offset holds from one band to the next. A band
    # whose offset agrees with neither neighbour (within ``max_step_m`` / 2)
    # is a surface that leans -- a dome, a sloping crown, a cornice seen
    # edge-on -- and a vertical strip through it would be a pane on a dome.
    # Such bands are ``sloped``: recorded, never extruded. The returns that
    # found them carry the shape themselves.
    seen = list(bands)                                      # judged against what was observed, not what was just relabelled
    for i, b in enumerate(seen):
        if b.status != "observed":
            continue
        neighbours = [seen[j] for j in (i - 1, i + 1)
                      if 0 <= j < len(seen) and seen[j].status == "observed" and seen[j].offset_m is not None]
        if not neighbours:
            # If immediate neighbours are unobserved (a gap across a terrace or setback),
            # check the nearest observed band in each direction within 15 m (3 bands).
            for j in range(i - 1, max(-1, i - 4), -1):
                if seen[j].status == "observed" and seen[j].offset_m is not None:
                    if (b.z_lo - seen[j].z_hi) <= 15.0:
                        neighbours.append(seen[j])
                    break
            for j in range(i + 1, min(len(seen), i + 4)):
                if seen[j].status == "observed" and seen[j].offset_m is not None:
                    if (seen[j].z_lo - b.z_hi) <= 15.0:
                        neighbours.append(seen[j])
                    break
        if not neighbours:
            # An isolated band far above the ground (> 15 m) with no observed wall within 15 m
            # cannot be a vertical wall holding its offset across bands; it is a crown/dome feature.
            if b.z_lo - plane.z_lo > 15.0:
                bands[i] = FaceBand(b.z_lo, b.z_hi, b.offset_m, b.n_returns, b.spread_m, b.along_lo, b.along_hi, "sloped")
            continue
        if not any(abs(n.offset_m - b.offset_m) <= max_step_m / 2 for n in neighbours):
            bands[i] = FaceBand(b.z_lo, b.z_hi, b.offset_m, b.n_returns, b.spread_m, b.along_lo, b.along_hi, "sloped")

    # Extend a wall through thin bands at its ends. Below the lowest observed
    # band the returns are few -- the street is cluttered, the podium is
    # under trees and canopies -- but if what there is sits at the same
    # offset, the wall continues to the ground; without this, Humana's east
    # wall stopped at 12 m and hung over its own ground floor like a panel.
    # The same at the top.
    for step in (-1, 1):
        obs_idx = [i for i, b in enumerate(bands) if b.status == "observed"]
        if not obs_idx:
            break
        i = min(obs_idx) if step < 0 else max(obs_idx)
        anchor = bands[i]
        j = i + step
        while 0 <= j < len(bands) and bands[j].status == "unknown":
            b = bands[j]
            m = in_face & (z >= b.z_lo) & (z < b.z_hi)
            o = off[m]
            # Looser than an observed band, because the wall beside it is
            # already established: within a metre of it, a quarter of the
            # returns an observed band needs.
            near = np.abs(o - anchor.offset_m) <= 1.0
            if near.sum() < max(min_returns // 4, 6):
                break
            a = along[m][near] * L
            lo, hi = float(np.percentile(a, 2)), float(np.percentile(a, 98))
            bands[j] = FaceBand(b.z_lo, b.z_hi, float(np.median(o[near])), int(near.sum()),
                                float(np.median(np.abs(o[near] - np.median(o[near])))),
                                min(lo, anchor.along_lo), max(hi, anchor.along_hi), "extended")
            j += step

    # Bridge between observed bands that agree.
    obs = [i for i, b in enumerate(bands) if b.status == "observed"]
    for i, j in zip(obs, obs[1:]):
        if j - i <= 1:
            continue
        a, c = bands[i], bands[j]
        gap = c.z_lo - a.z_hi
        if abs(a.offset_m - c.offset_m) > max_step_m or gap > max_bridge_m:
            continue
        # An open band between them stays open (a terrace, a loggia); the
        # unknown bands are bridged when the wall above and below agree.
        lo, hi = min(a.along_lo, c.along_lo), max(a.along_hi, c.along_hi)
        # A bridged band was not observed as a wall, but its returns are
        # still there -- too few or too spread for the wall itself, yet
        # often plenty on a recessed bay. Segments come from its own returns
        # where it has them, else from a recess both ends share.
        shared = [sa for sa in a.segments if any(abs(sa[0] - sc[0]) < 3.0 and abs(sa[1] - sc[1]) < 3.0
                                                  and abs(sa[2] - sc[2]) <= max_step_m for sc in c.segments)]
        for k in range(i + 1, j):
            b = bands[k]
            if b.status == "open":
                continue
            frac = (b.z_lo - a.z_hi) / max(gap, 1e-6)
            offset_k = a.offset_m + frac * (c.offset_m - a.offset_m)
            a_k, o_k = returns_of[k]
            own = _band_segments(a_k, o_k, offset_k, L, step_m=max_step_m) if len(o_k) >= 12 else []
            bands[k] = FaceBand(b.z_lo, b.z_hi, offset_k, b.n_returns, 0.0, lo, hi, "bridged",
                                segments=own or list(shared))
    return bands


def band_extent(plane: WallPlane, b: FaceBand, full_width_fraction: float = 0.6,
                margin_m: float = 3.0) -> Tuple[float, float]:
    """How far along the edge a band's wall is taken to run: the whole edge
    when the returns reached most of it, else their own extent plus a margin;
    and never past where the neighbouring walls stand at this height."""
    L = float(np.hypot(*(plane.p1 - plane.p0)))
    if (b.along_hi - b.along_lo) >= full_width_fraction * L:
        s0, s1 = 0.0, L
    else:
        s0, s1 = max(0.0, b.along_lo - margin_m), min(L, b.along_hi + margin_m)
    if b.clip_lo is not None:
        s0 = max(s0, b.clip_lo)
    if b.clip_hi is not None:
        s1 = min(s1, b.clip_hi)
    return s0, s1


# The kinds of band that are wall. ``matched`` has its depth from the frames
# (kyfromabove_3d.wall_match), not the LiDAR: a wall to draw, but never an
# anchor -- bridging, extending and mirroring read ``observed`` bands only.
# ``edge`` is a band under a roof edge the LiDAR measured, drawn because
# nothing says there is no wall there (``fill_under_roof_edge``).
WALL_STATUSES = ("observed", "bridged", "extended", "mirrored", "matched", "edge")


def fill_under_roof_edge(bands: List[FaceBand], length_m: float, max_offset_m: float = 1.5,
                         max_back_fill: float = 0.5, max_wobble_m: float = 1.0) -> int:
    """A wall stands under a measured roof edge unless there is evidence of a
    gap. For a face whose top edge the LiDAR placed (the roof's end, with a
    drop beyond it), the bands the returns left unknown, and the ``sloped``
    ones within ``max_offset_m`` of the edge, become wall (``edge``): the
    Truist tower's face 1001 was seen to 47 m and then as glass whose depth
    wobbled by a metre, and 30 m of it under a roof edge the LiDAR measured
    was left a hole. A sloped band further in than that is filled too when
    it lies within ``max_wobble_m`` of the nearest drawn band's depth -- the
    wall holding its line, its depth wobbling on thin returns: the Truist's
    faces 1001 and 1004 at 52-57 m, 1.6 m in beside walls at 1.1-1.2 m. A
    band stepping in further than that (400 West Market's crown, 2-8 m in) is
    a slope and stays one. Each takes the depth of the face's nearest drawn band
    within ``max_offset_m`` of the edge, or the edge itself where there is
    none (the Omni's one band 2.4 m out, at its tower's foot, is the
    podium's, not the tower's). An ``open`` band whose deeper surface fills
    less than ``max_back_fill`` of its height is a roof seen behind the edge,
    not a second wall, and is filled like an unknown one: Hampton's long
    tower walls were open at 20-25 m over a flat roof at 23.1 m reaching
    to the edge. The evidence that keeps a gap is kept: an open band with a
    wall behind (RIVUE's penthouse 7 m in) is left as it is, and the mesh
    stops any wall band where the LiDAR roof behind it does not reach its
    height (``build_mesh.RoofField``). In place; returns how many bands were
    filled."""
    drawn = [b for b in bands if b.status in WALL_STATUSES and b.offset_m is not None
             and abs(b.offset_m) <= max_offset_m]
    n = 0
    for i, b in enumerate(bands):
        mid = 0.5 * (b.z_lo + b.z_hi)
        near = min(drawn, key=lambda d: (abs(0.5 * (d.z_lo + d.z_hi) - mid), d.z_lo), default=None)
        sloped_wall = b.status == "sloped" and b.offset_m is not None and (
            abs(b.offset_m) <= max_offset_m or (near is not None and abs(b.offset_m - near.offset_m) <= max_wobble_m))
        fill = (b.status == "unknown" or sloped_wall
                or (b.status == "open" and b.back_fill is not None and b.back_fill < max_back_fill))
        if not fill:
            continue
        bands[i] = FaceBand(b.z_lo, b.z_hi, near.offset_m if near is not None else 0.0, b.n_returns, 0.0,
                            0.0, length_m, "edge", clip_lo=b.clip_lo, clip_hi=b.clip_hi)
        n += 1
    return n


def find_overhangs(ring_m: np.ndarray, z_lo: float, z_hi: float, xyz: np.ndarray, band_m: float = 5.0,
                   min_setback_m: float = 2.0, min_bands: int = 2, max_face_offset_m: float = 1.0,
                   min_roof_above_m: float = 5.0, line_tol_m: float = 0.6, max_angle_deg: float = 5.0,
                   min_line_returns: int = 30, max_cut_share: float = 0.4,
                   min_face_m: float = 4.0) -> List[Dict[str, Any]]:
    """Where an upper part of a level stands out over the part below it --
    the Hyatt's drum, 60 m up, over its tower's shaft -- from the LiDAR alone.

    The level's outline is its roof's edge, so it is the drum's; a roof
    height field is one height per cell and cannot show that the shaft
    below stands 5 m further in. The returns can: a face whose wall is seen
    ``min_setback_m`` or more inside the edge over at least ``min_bands``
    bands from the level's foot up, never within ``max_face_offset_m`` of the
    edge below that, while the roof just inside the edge stands at least
    ``min_roof_above_m`` above the top of the set-back wall (``under_z``, the
    overhang's underside). The shaft's wall is then fitted as a line through
    the returns within ``line_tol_m`` of it below ``under_z`` (within
    ``max_angle_deg`` of the face), and the outline is cut back along that
    line between the two places the line crosses it either side of the face
    (the overhang's plan, ``cut_m``). The cut must take no more than
    ``max_cut_share`` of the outline, and every other face along the cut
    must have no wall seen within ``max_face_offset_m`` of its edge below
    ``under_z``. Returns one record per overhang: ``under_z``, the shaft
    line, ``ring_below_m`` (the outline below the overhang), ``cut_m``, the
    faces cut and the returns on the shaft line."""
    from shapely.geometry import LineString, Polygon
    from shapely.ops import split

    ring_m = np.asarray(ring_m, dtype=float)[:, :2]
    poly = Polygon(ring_m)
    if not poly.is_valid or poly.area < 1.0:
        return []
    near = xyz[(xyz[:, 2] >= z_lo - 2.0) & (xyz[:, 2] <= z_hi + 2.0)]
    planes = [pl for pl in wall_planes(ring_m, z_lo, z_hi, min_length_m=min_face_m) if not isinstance(pl, ArcWall)]
    measured = ("observed", "extended", "bridged")
    profiles = {pl.face: face_profile(pl, near, band_m=band_m) for pl in planes}

    def roof_inside(pl) -> float:
        """The roof just inside the edge: the 90th percentile of the returns
        0.25-1.5 m in, over the face's middle."""
        a, o = pl.local(xyz[:, :2])
        m = (a >= 0.2 * pl.length_m) & (a <= 0.8 * pl.length_m) & (o <= -0.25) & (o >= -1.5)
        return float(np.percentile(xyz[m, 2], 90)) if m.sum() >= 10 else -np.inf

    def wall_at_edge_below(pl, under_z: float) -> bool:
        return any(b.status in measured and b.offset_m is not None and abs(b.offset_m) <= max_face_offset_m
                   and b.z_lo < under_z for b in profiles.get(pl.face, []))

    out = []
    for pl in planes:
        bands = sorted(profiles[pl.face], key=lambda b: b.z_lo)
        run = []
        for b in bands[:2]:                                   # the set-back wall starts at the level's foot
            if b.status in measured and b.offset_m is not None and b.offset_m <= -min_setback_m:
                run = [b]
                break
        if not run:
            continue
        for b in bands[bands.index(run[0]) + 1:]:
            if b.status in measured and b.offset_m is not None and abs(b.offset_m - run[0].offset_m) <= 1.0:
                run.append(b)
            else:
                break
        under_z = run[-1].z_hi
        if len(run) < min_bands or wall_at_edge_below(pl, under_z) or roof_inside(pl) < under_z + min_roof_above_m:
            continue
        setback = float(np.median([b.offset_m for b in run]))
        # the shaft's wall: the returns on that line below the overhang, wherever they run along it
        a, o = pl.local(near[:, :2])
        on = (np.abs(o - setback) <= line_tol_m) & (near[:, 2] >= z_lo + 1.0) & (near[:, 2] <= under_z - 0.5)
        if on.sum() < min_line_returns:
            continue
        P = near[on, :2]
        c = P.mean(axis=0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        d = vt[0] / np.linalg.norm(vt[0])
        e1 = (pl.p1 - pl.p0) / pl.length_m
        if abs(float(d @ e1)) < np.cos(np.radians(max_angle_deg)):
            continue
        d = d if d @ e1 > 0 else -d
        # where the line crosses the outline either side of the face; the part outside it is the overhang
        mid_t = float((0.5 * (pl.p0 + pl.p1) - c) @ d)
        R = 4.0 * float(np.sqrt(poly.area)) + 100.0
        hits = LineString([c - R * d, c + R * d]).intersection(poly.exterior)
        ts = sorted(float((np.asarray(g.coords[0]) - c) @ d) for g in getattr(hits, "geoms", [hits]) if not g.is_empty)
        lo = [t for t in ts if t < mid_t - 0.5]
        hi = [t for t in ts if t > mid_t + 0.5]
        if not lo or not hi:
            continue
        cutter = LineString([c + (lo[-1] - 0.01) * d, c + (hi[0] + 0.01) * d])
        pieces = [g for g in split(poly, cutter).geoms if g.area > 0.5]
        side = [g for g in pieces if float((np.asarray(g.representative_point().coords[0]) - c) @ pl.normal) > 0]
        if len(pieces) < 2 or len(side) != 1:
            continue
        cut = side[0]
        below = poly.difference(cut)
        if below.geom_type != "Polygon" or cut.area > max_cut_share * poly.area:
            continue
        cut_faces = [q.face for q in planes if cut.exterior.buffer(0.05).contains(LineString([q.p0, q.p1]))]
        if any(wall_at_edge_below(q, under_z) for q in planes if q.face in cut_faces and q.face != pl.face):
            continue
        a_line = P @ d
        out.append({"face": int(pl.face), "under_z": round(float(under_z), 2), "setback_m": round(setback, 2),
                    "line_point": [round(float(v), 3) for v in c], "line_dir": [round(float(v), 4) for v in d],
                    "line_along_m": [round(float(a_line.min() - c @ d), 1), round(float(a_line.max() - c @ d), 1)],
                    "n_line_returns": int(on.sum()), "faces_cut": [int(f) for f in cut_faces],
                    "cut_m": np.asarray(cut.exterior.coords).round(3).tolist(), "cut_area_m2": round(float(cut.area), 1),
                    "ring_below_m": np.asarray(below.exterior.coords).round(3).tolist()})
        break                                                # one per level: the outline below changes the rest
    return out


def crown_bands(bands: List[FaceBand], reference_offset_m: Optional[float] = None,
                max_step_m: float = 2.0, top_z: Optional[float] = None,
                top_of: Optional[Callable[[FaceBand], Optional[float]]] = None) -> int:
    """An "open" band at the top of a face -- returns at two depths, with no
    wall above it -- is the wall's crown or parapet in front and the roof
    behind it. Where its front surface continues the wall directly below
    (within ``max_step_m``; or, on a verified plane with nothing measured
    below, the plane itself: ``reference_offset_m``), it is wall, at its front
    offset. Left open, PNC Tower's walls stopped at 152 m under a roof at
    155-160 m, a floating lid; drawn at its *back* offset it had stood panels
    3-7 m inboard on Kentucky Home Life's roof. Bottom-up, so a crown two
    bands tall continues itself. In place; returns how many were made wall."""
    # What stands below must be drawn as wall: on a verified plane every band
    # but an open one is (a sloped band too); elsewhere only the wall
    # statuses. A crown over nothing drawn would turn a face the prism path
    # walls to its roof edge into a lone band at the top -- Truist Tower's
    # face 1007 lost 12-71 m of wall that way.
    order = sorted(range(len(bands)), key=lambda i: bands[i].z_lo)
    wallish = WALL_STATUSES + (("sloped",) if reference_offset_m is not None else ())
    # the crown is above the face's highest wall band: an open band lower
    # down, with wall above it, is a colonnade or a loggia. Each open band up
    # there counts only over a band that is itself drawn -- on a face open all
    # the way (4a668624's face 1) its open ground band made wall left the open
    # bands above it bare, where the prism path had walled the whole face;
    # Chase Bank Plaza's face 1005 has wall to 19 m, then open, unmeasured and
    # open bands, and only the first continues the wall
    walls = [k for k, i in enumerate(order) if bands[i].status in WALL_STATUSES and bands[i].offset_m is not None]
    n = 0
    for k in range(walls[-1] + 1 if walls else 0, len(order)):
        b = bands[order[k]]
        if b.status != "open" or b.offset_m is None or k == 0:
            continue
        below = bands[order[k - 1]]
        if abs(below.z_hi - b.z_lo) > 0.05:
            continue
        if below.status in wallish and below.offset_m is not None:
            ref = below.offset_m
        elif reference_offset_m is not None and below.status != "open":
            ref = reference_offset_m                        # a verified plane: the plane is drawn over it
        else:
            continue                                        # nothing drawn below it to continue
        # a crown stands at the building's edge: within max_step_m of the wall
        # below it *and* of the footprint's edge -- Waterfront Plaza's open
        # band 6 m inboard, over a wall already set back, stood a panel on its
        # roof like Kentucky Home Life's
        if abs(b.offset_m - ref) > max_step_m or abs(b.offset_m) > max_step_m:
            continue
        # and no higher than the face, nor than its own returns reach (``top_of``):
        # a 5 m band over a roof 2 m above its foot put Fifth Third's top 3 m
        # above the LiDAR's
        z_hi = min(b.z_hi, top_z) if top_z is not None else b.z_hi
        own = top_of(b) if top_of is not None else None
        if own is not None:
            z_hi = min(z_hi, own)
        if z_hi - b.z_lo < 0.3:
            continue
        bands[order[k]] = FaceBand(b.z_lo, z_hi, b.offset_m, b.n_returns, b.spread_m, b.along_lo, b.along_hi,
                                   "observed", segments=list(b.segments), clip_lo=b.clip_lo, clip_hi=b.clip_hi)
        n += 1
    return n


def roof_behind_bands(bands: List[FaceBand], max_back_fill: float = 0.5, max_step_m: float = 2.0,
                      top_z: Optional[float] = None,
                      top_of: Optional[Callable[[FaceBand], Optional[float]]] = None) -> int:
    """An "open" band whose deeper surface lies flat -- at one or two
    heights, not up through the band (``back_fill`` under ``max_back_fill``)
    -- is the wall in front of a flat roof, not a colonnade: the band the
    roof falls in holds the wall at the edge and the roof running inward
    from it. It is drawn as wall at its front. 9947f2b0's roof at 14 m sits
    in its 10-15 m band on every face; read as a loggia, the wall was drawn
    5-8 m inboard for those 5 m, under a wall band above, and where the
    roof's returns outnumbered the wall's (the mode at the back) the crown
    rule found no front and the wall stopped at 10 m.

    The front is the band's own mode where that is the front surface, else
    the offset of the wall band touching it (below first, then above); either
    way within ``max_step_m`` of that wall and of the footprint's edge. With
    no wall drawn above, the band stops at ``top_z`` and at its own front
    returns' top (``top_of``), as a crown does. An open band with a wall
    behind its piers (back_fill high) is left open. In place; returns how
    many bands were made wall."""
    order = sorted(range(len(bands)), key=lambda i: bands[i].z_lo)
    n = 0
    for k, i in enumerate(order):
        b = bands[i]
        if b.status != "open" or b.back_fill is None or b.back_fill >= max_back_fill:
            continue
        below = bands[order[k - 1]] if k > 0 else None
        above = bands[order[k + 1]] if k + 1 < len(order) else None
        # the wall beside it must be one the LiDAR saw: a band copied from the
        # face opposite is no anchor -- on 2870eeb0's never-seen 86 m face the
        # crown made wall off a mirrored band stood a 10 m strip at its top and
        # took the face off the prism path (490 m2 open below it)
        touching = [x for x in (below, above)
                    if x is not None and x.status in ("observed", "extended", "bridged")
                    and x.offset_m is not None and abs(x.offset_m) <= max_step_m
                    and min(abs(x.z_hi - b.z_lo), abs(x.z_lo - b.z_hi)) <= 0.05]
        if not touching:
            continue
        anchor = touching[0]
        front_is_mode = (b.offset_m is not None and b.back_offset_m is not None
                         and b.offset_m - b.back_offset_m > 0.5
                         and abs(b.offset_m - anchor.offset_m) <= max_step_m and abs(b.offset_m) <= max_step_m)
        front = float(b.offset_m) if front_is_mode else float(anchor.offset_m)
        wall_above = any(x.status in WALL_STATUSES and x.status not in ("edge", "matched") and x.offset_m is not None
                         and x.z_lo >= b.z_hi - 1e-6 for x in bands)
        z_hi = b.z_hi
        if not wall_above:
            if top_z is not None:
                z_hi = min(z_hi, top_z)
            own = top_of(FaceBand(b.z_lo, b.z_hi, front, b.n_returns, 0.0, 0.0, 0.0, "observed")) if top_of else None
            if own is not None:
                z_hi = min(z_hi, own)
            if z_hi - b.z_lo < 0.3:
                continue
        bands[i] = FaceBand(b.z_lo, z_hi, front, b.n_returns, 0.0, anchor.along_lo, anchor.along_hi, "bridged",
                            segments=[], clip_lo=b.clip_lo, clip_hi=b.clip_hi)
        n += 1
    return n


def carry_extents_down(bands: List[FaceBand], max_offset_step_m: float = 1.0) -> int:
    """A wall stands on the wall below it. Where a band's returns stop short
    of the band directly above it at nearly the same offset -- a neighbour
    hid part of the wall from the scanner, in a staircase down the face (the
    Omni's west tower wall, 47-67 m up) -- the band runs at least as far as
    the one above. Only downward: a tower set back above a wider podium is
    untouched, and an open band (a colonnade) breaks the chain. In place;
    returns how many bands were widened."""
    order = sorted(range(len(bands)), key=lambda i: bands[i].z_lo)
    n = 0
    for lo_i, hi_i in zip(order[-2::-1], order[:0:-1]):
        b, above = bands[lo_i], bands[hi_i]
        if (b.status not in WALL_STATUSES or above.status not in WALL_STATUSES
                or b.offset_m is None or above.offset_m is None
                or abs(b.offset_m - above.offset_m) > max_offset_step_m or abs(b.z_hi - above.z_lo) > 0.05):
            continue
        lo, hi = min(b.along_lo, above.along_lo), max(b.along_hi, above.along_hi)
        if lo < b.along_lo - 0.05 or hi > b.along_hi + 0.05:
            b.along_lo, b.along_hi = lo, hi
            n += 1
    return n


def clip_profiles_by_neighbours(planes: List[WallPlane], profiles: Dict[int, List[FaceBand]],
                                min_setback_m: float = 1.0) -> int:
    """Where a neighbouring face's wall stands back from the footprint at
    some height, this face's wall cannot reach the corner at that height.

    Seen from Main Street, Humana's east wall ran the full 56 m of its edge
    at every height, while the north wall it meets stands 24.5 m back from
    the north corner above the podium: the east wall was drawn 24.5 m too
    wide up there, matching the base. Faces share corners, so each band is
    clipped at each end by the neighbour's offset at the same height --
    when the neighbour has a wall there; an unknown neighbour clips nothing.
    Returns how many bands were clipped.
    """
    by_face = {pl.face: pl for pl in planes}
    def neighbour_at(pl: WallPlane, corner: np.ndarray, own_end: str):
        for other in planes:
            if other.face == pl.face:
                continue
            if np.linalg.norm(other.p1 - corner) < 0.5 and own_end == "p0":
                return other
            if np.linalg.norm(other.p0 - corner) < 0.5 and own_end == "p1":
                return other
        return None
    n_clipped = 0
    for face, bands in profiles.items():
        pl = by_face.get(face)
        if pl is None:
            continue
        L = float(np.hypot(*(pl.p1 - pl.p0)))
        prev_pl = neighbour_at(pl, pl.p0, "p0")
        next_pl = neighbour_at(pl, pl.p1, "p1")
        for i, b in enumerate(bands):
            if b.status not in WALL_STATUSES:
                continue
            clip_lo = clip_hi = None
            for nb_pl, end in ((prev_pl, "lo"), (next_pl, "hi")):
                if nb_pl is None or nb_pl.face not in profiles:
                    continue
                nb = next((x for x in profiles[nb_pl.face] if x.z_lo <= b.z_lo < x.z_hi), None)
                if nb is None or nb.status not in WALL_STATUSES or nb.offset_m is None:
                    # The neighbour has no wall at this height (Humana's north
                    # wall is unseen under its porch): take its nearest known
                    # band within 25 m of height, so the clip does not jump
                    # back to the corner across a gap in the LiDAR.
                    known = [x for x in profiles[nb_pl.face] if x.status in WALL_STATUSES and x.offset_m is not None
                             and abs((x.z_lo + x.z_hi) / 2 - (b.z_lo + b.z_hi) / 2) <= 25.0]
                    if not known:
                        continue
                    nb = min(known, key=lambda x: abs((x.z_lo + x.z_hi) / 2 - (b.z_lo + b.z_hi) / 2))
                setback = -nb.offset_m
                if setback < min_setback_m:
                    continue
                if end == "lo":
                    clip_lo = setback
                else:
                    clip_hi = L - setback
            if clip_lo is not None or clip_hi is not None:
                bands[i] = FaceBand(b.z_lo, b.z_hi, b.offset_m, b.n_returns, b.spread_m, b.along_lo, b.along_hi,
                                    b.status, segments=b.segments, clip_lo=clip_lo, clip_hi=clip_hi)
                n_clipped += 1
    return n_clipped


def profile_keeps(plane: WallPlane, bands: List[FaceBand], xyz: np.ndarray,
                  dense_per_m2: float = 0.5) -> np.ndarray:
    """Which of ``xyz`` (points on a plane extruded over the whole face) the
    profile lets stand. A verified plane has the tiles' evidence behind it,
    so the profile removes only what it has evidence *against*: bands that
    are open (a loggia: two surfaces, one 4 m or more behind the other), and
    the part of a band beyond the returns' extent when the returns are dense
    enough (``dense_per_m2`` over the extent they cover) for their edge to
    mean something. Humana's north face above 40 m is only its east 30 m --
    the shaft is narrower than the podium -- and the returns there, two per
    square metre, stop sharply at the step; the plane painted sky on the
    rest. On a glass tower the returns are thin everywhere, a tenth of that,
    and thin is not absent: such bands are left alone."""
    seg = plane.p1 - plane.p0
    L = float(np.hypot(*seg))
    along = ((xyz[:, :2] - plane.p0) @ seg) / max(L, 1e-9)
    keep = np.ones(len(xyz), dtype=bool)
    for b in bands:
        in_band = (xyz[:, 2] >= b.z_lo) & (xyz[:, 2] < b.z_hi)
        if b.clip_lo is not None:
            keep &= ~(in_band & (along < b.clip_lo))
        if b.clip_hi is not None:
            keep &= ~(in_band & (along > b.clip_hi))
        if b.status == "open":
            keep &= ~in_band
        elif b.status == "observed" and (b.along_hi - b.along_lo) < 0.8 * L:
            area = max(b.along_hi - b.along_lo, 1.0) * (b.z_hi - b.z_lo)
            if b.n_returns / area >= dense_per_m2:
                s0, s1 = band_extent(plane, b, full_width_fraction=0.8)
                keep &= ~(in_band & ((along < s0) | (along > s1)))
    return keep


def extrude_profile(plane: WallPlane, bands: List[FaceBand], spacing_m: float,
                    full_width_fraction: float = 0.6, margin_m: float = 3.0) -> np.ndarray:
    """Grid points on the wall as the profile describes it: every observed
    or bridged band, at its own offset, across the full edge when the returns
    reached most of it and across their own extent plus a margin otherwise."""
    seg = plane.p1 - plane.p0
    L = float(np.hypot(*seg))
    if L < 1e-6:
        return np.zeros((0, 3))
    e1 = seg / L
    out = []
    for b in bands:
        if b.status == "open" and b.back_offset_m is not None and (b.along_hi - b.along_lo) >= 3.0:
            # the wall behind the piers of a colonnade
            b = FaceBand(b.z_lo, b.z_hi, b.back_offset_m, b.n_returns, 0.0, b.along_lo, b.along_hi, "observed",
                         clip_lo=b.clip_lo, clip_hi=b.clip_hi)
        if b.status not in WALL_STATUSES or b.offset_m is None:
            continue
        s0, s1 = band_extent(plane, b, full_width_fraction, margin_m)
        s = np.arange(s0, s1, spacing_m)
        zz = np.arange(b.z_lo, b.z_hi, spacing_m)
        if len(s) == 0 or len(zz) == 0:
            continue
        # Each point along the edge takes the segment's offset where one
        # covers it, and the band's elsewhere.
        off = np.full(len(s), b.offset_m)
        for a0, a1, oseg in b.segments:
            off[(s >= a0) & (s < a1)] = oseg
        S, Z = np.meshgrid(s, zz)
        O = np.broadcast_to(off[None, :], S.shape)
        xy = plane.p0[None, :] + S.ravel()[:, None] * e1[None, :] + O.ravel()[:, None] * plane.normal[None, :]
        out.append(np.column_stack([xy, Z.ravel()]))
    return np.vstack(out) if out else np.zeros((0, 3))


def opposite_face(plane: WallPlane, planes: List[WallPlane]) -> Optional[WallPlane]:
    """The face across the building from ``plane``: normals within 25 deg of
    opposite, and the largest overlap when projected onto this edge."""
    seg = plane.p1 - plane.p0
    L = float(np.hypot(*seg))
    if L < 1e-6:
        return None
    e1 = seg / L
    best, best_overlap = None, 0.0
    for other in planes:
        if other.face == plane.face or float(other.normal @ plane.normal) > -0.9:
            continue
        a0, a1 = sorted([float((other.p0 - plane.p0) @ e1), float((other.p1 - plane.p0) @ e1)])
        overlap = max(0.0, min(a1, L) - max(a0, 0.0))
        if overlap > best_overlap:
            best, best_overlap = other, overlap
    return best if best_overlap >= 0.5 * L else None


def mirror_hidden_bands(planes: List[WallPlane], profiles: Dict[int, List[FaceBand]],
                        max_shape_diff_m: float = 0.6, min_joint_bands: int = 3,
                        measured_faces: Optional[set] = None) -> Dict[int, Dict[str, Any]]:
    """Fill a face's unseen bands from the face opposite, when the two
    faces have been seen to have the same shape.

    Buildings are mostly symmetric across their middle: a setback at the
    twentieth floor on the west wall is there on the east wall too. Where a
    face is hidden -- Humana's east wall in PNC Tower's shadow, a tower's
    short end the flight never looked at -- the opposite face is usually
    the best evidence there is. It is used only as a *shape*: the offsets
    are compared band by band where both faces were observed, relative to
    each face's own lowest observed wall (so a footprint drawn a metre off
    on one side does not break the match), and the pair is taken as
    symmetric when they agree within ``max_shape_diff_m`` on at least
    ``min_joint_bands`` bands. Then each unknown band on the hidden face
    takes the opposite band's offset (shifted by the difference of the two
    anchors) and its extent mirrored along the edge, with status
    ``mirrored``. A bridged band is replaced too where the twin was
    *observed* at that height: a bridge interpolates between its anchors and
    cannot know about a step between them; the twin can. A face with no
    observed band at all takes the opposite face's offsets as they are.
    Observed, extended, open and sloped bands are never replaced: a face's
    own measurement outranks the prior.

    A face in ``measured_faces`` -- under a roof edge the LiDAR measured --
    with no observed band takes nothing from its twin: the roof edge is
    better evidence than the opposite face, and the bands are left unknown
    for ``fill_under_roof_edge`` to draw at the edge. The Hyatt's face 1023,
    never seen, was given its twin's 5.3 m setback -- the shaft under the
    drum, only on the other side -- and cut back the walls beside it.

    Returns, per face that received bands, a small record of what was done.
    """
    by_face = {pl.face: pl for pl in planes}
    report: Dict[int, Dict[str, Any]] = {}
    for face, bands in profiles.items():
        pl = by_face.get(face)
        if pl is None:
            continue
        opp = opposite_face(pl, planes)
        if opp is None or opp.face not in profiles:
            continue
        other = profiles[opp.face]
        L_f = float(np.hypot(*(pl.p1 - pl.p0)))
        L_g = float(np.hypot(*(opp.p1 - opp.p0)))
        def band_at(bs: List[FaceBand], z: float) -> Optional[FaceBand]:
            return next((b for b in bs if b.z_lo <= z < b.z_hi), None)
        own_obs = [b for b in bands if b.status == "observed"]
        opp_obs = [b for b in other if b.status == "observed"]
        if not opp_obs:
            continue
        # Anchors: each face's lowest observed wall, the reference its shape is measured from.
        anchor_g = float(np.median([b.offset_m for b in sorted(opp_obs, key=lambda b: b.z_lo)[:3]]))
        if own_obs:
            anchor_f = float(np.median([b.offset_m for b in sorted(own_obs, key=lambda b: b.z_lo)[:3]]))
            joint = [(b, band_at(other, (b.z_lo + b.z_hi) / 2)) for b in own_obs]
            joint = [(b, g) for b, g in joint if g is not None and g.status == "observed"]
            if len(joint) < min_joint_bands:
                continue
            diffs = [abs((b.offset_m - anchor_f) - (g.offset_m - anchor_g)) for b, g in joint]
            shape_diff = float(np.percentile(diffs, 75))     # one odd band is tolerated, a second is not
            if shape_diff > max_shape_diff_m:
                report[face] = {"from_face": opp.face, "symmetric": False, "shape_diff_m": round(shape_diff, 2), "n_joint": len(joint), "mirrored": 0}
                continue
            shift = anchor_f - anchor_g
        else:
            if measured_faces is not None and face in measured_faces:
                report[face] = {"from_face": opp.face, "symmetric": None, "shape_diff_m": None, "n_joint": 0, "mirrored": 0,
                                "anchored": False, "skipped": "never seen, under a measured roof edge"}
                continue
            joint, shape_diff, shift = [], None, 0.0
        n_mirrored = 0
        for i, b in enumerate(bands):
            # Unknown bands take the twin's word; so do bridged ones where
            # the twin was actually observed at that height -- a bridge
            # interpolates between anchors and cannot know about a step
            # between them, the twin can.
            if b.status not in ("unknown", "bridged"):
                continue
            g = band_at(other, (b.z_lo + b.z_hi) / 2)
            if g is None or g.offset_m is None:
                continue
            if b.status == "unknown" and g.status not in ("observed", "extended", "bridged"):
                continue
            if b.status == "bridged" and g.status != "observed":
                continue
            # The twin's band must itself be a wall that holds from one band
            # to the next -- the same test that keeps a vertical strip off a
            # dome. A twin band on a receding crown is not copied across.
            gi = other.index(g)
            steady = any(0 <= j < len(other) and other[j].status in WALL_STATUSES and other[j].offset_m is not None
                         and abs(other[j].offset_m - g.offset_m) <= 0.4 for j in (gi - 1, gi + 1))
            if not steady:
                continue
            lo = L_f * (1.0 - g.along_hi / L_g)
            hi = L_f * (1.0 - g.along_lo / L_g)
            bands[i] = FaceBand(b.z_lo, b.z_hi, g.offset_m + shift, b.n_returns, g.spread_m,
                                max(0.0, lo), min(L_f, hi), "mirrored")
            n_mirrored += 1
        report[face] = {"from_face": opp.face, "symmetric": True, "shape_diff_m": None if shape_diff is None else round(shape_diff, 2),
                        "n_joint": len(joint), "mirrored": n_mirrored, "anchored": bool(own_obs)}
    return report


def mirror_colour_from_twin(planes: List[WallPlane], profiles: Dict[int, List[FaceBand]],
                            mirror_report: Dict[int, Dict[str, Any]],
                            xyz: np.ndarray, face_of: np.ndarray, rgb: np.ndarray, seen: np.ndarray,
                            tolerance_m: float = 0.4) -> np.ndarray:
    """Colour a hidden face's inferred wall from the face opposite.

    Where a face's shape was taken from its twin, its colour can be too: a
    point on a mirrored or bridged band, or any point no camera coloured,
    takes the colour of the twin's point at the same height and the
    mirrored position along the edge, when the twin has a coloured point
    within ``tolerance_m`` there. Humana's east wall stands in PNC Tower's
    shadow and the obliques that reach it do so at a grazing angle, dark;
    its west wall was photographed. Observed bands keep their own colour.

    Modifies ``rgb`` and ``seen`` in place; returns a mask of the points
    whose colour came from the twin.
    """
    from scipy.spatial import cKDTree

    by_face = {pl.face: pl for pl in planes}
    took = np.zeros(len(xyz), dtype=bool)
    for face, rec in mirror_report.items():
        if not rec.get("symmetric"):
            continue
        pl, twin = by_face.get(face), by_face.get(rec["from_face"])
        if pl is None or twin is None:
            continue
        L_f = float(np.hypot(*(pl.p1 - pl.p0)))
        L_g = float(np.hypot(*(twin.p1 - twin.p0)))
        e_f = (pl.p1 - pl.p0) / L_f
        e_g = (twin.p1 - twin.p0) / L_g
        g_idx = np.flatnonzero((face_of == twin.face) & (seen > 0))
        if len(g_idx) < 100:
            continue
        # The twin's coloured points, laid out as they would sit on this face.
        a_g = (xyz[g_idx, :2] - twin.p0) @ e_g
        key_g = np.column_stack([L_f * (1.0 - a_g / L_g), xyz[g_idx, 2]])
        tree = cKDTree(key_g)
        f_idx = np.flatnonzero(face_of == face)
        if len(f_idx) == 0:
            continue
        z_f = xyz[f_idx, 2]
        inferred = seen[f_idx] == 0
        for b in profiles.get(face, []):
            if b.status in ("mirrored", "bridged"):
                inferred |= (z_f >= b.z_lo) & (z_f < b.z_hi)
        cand = f_idx[inferred]
        if len(cand) == 0:
            continue
        a_f = (xyz[cand, :2] - pl.p0) @ e_f
        d, j = tree.query(np.column_stack([a_f, xyz[cand, 2]]), distance_upper_bound=tolerance_m)
        hit = np.isfinite(d)
        src = g_idx[j[hit]]
        tgt = cand[hit]
        # Colour flows towards the darker side. A face in a neighbour's
        # shadow is coloured by grazing, shadowed oblique pixels; its twin
        # was lit or photographed. A point that already has colour keeps it
        # unless the twin's is clearly brighter; an uncoloured point takes
        # whatever the twin has.
        own_bright = rgb[tgt].astype(np.float64).mean(axis=1)
        twin_bright = rgb[src].astype(np.float64).mean(axis=1)
        take = (seen[tgt] == 0) | (twin_bright >= 1.2 * own_bright + 10.0)
        tgt, src = tgt[take], src[take]
        rgb[tgt] = rgb[src]
        seen[tgt] = np.maximum(seen[tgt], 1)
        took[tgt] = True
    return took
