#!/usr/bin/env python3
"""Build a textured glTF mesh of one building from its LiDAR, footprint,
wall profiles and imagery, and wrap it as OGC 3D Tiles.

The point export (``export_3dtiles.py``) draws the building as it was
sampled. This draws it as surfaces, from the same evidence:

* **walls** are the profile bands ``colorize_building.py`` wrote to
  ``wall_profiles.json`` -- each band a vertical rectangle at its own offset
  from the footprint edge, split where a recess or projection was found, with
  a horizontal ledge wherever the band above stands further in or out than
  the band below (a set-back tower on its podium) and a short return wall at
  each recess edge;
* the **roof** is a height field over the footprint, the top of the returns
  in each cell (their 80th percentile, so an overhang keeps its roof), with the near-vertical cells that coincide with a
  modelled wall dropped so the two do not fight;
* **wall textures** are rasterised from an aerial-only coloured cloud
  (``colorized_aerial.laz``: the colourise run with no photograph cameras,
  which this script makes for itself) -- the obliques and, where a face was
  mirrored, the face opposite -- one image per face, 0.15 m per pixel. No
  internet photograph reaches the mesh;
* the **roof texture** is sampled straight from the aerial frames through
  each camera model, the nadir frame winning wherever it sees the roof
  because it is the most face-on, with the LiDAR depth test against the
  building's own returns so a set-back tower does not print onto its podium.

Output is one ``building.glb`` (positions, normals, UVs, one material per
face plus the roof, the GERS id in the node's extras) and a
``tileset.json`` whose root ``transform`` places the local metre frame on the
Earth; the mesh itself stays in the building's local frame, converted to
glTF's Y-up.

Usage::

    python scripts/build_mesh.py --building outputs/humana/humana --gers <id>
    # -> outputs/humana/humana/3dtiles_mesh/tileset.json
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kyfromabove_3d.coordinates import METERS_PER_US_SURVEY_FOOT       # noqa: E402
from kyfromabove_3d.features import (                                  # noqa: E402
    WALL_STATUSES, ArcWall, FaceBand, WallPlane, VerifiedWallPlane, band_extent, load_wall_planes,
    LEVEL_FACE_STRIDE, rectify_footprint_corners, GroundField, ground_field_for_building,
)
from kyfromabove_3d.clouds import find_cloud, read_cloud               # noqa: E402

TEXEL_M = 0.15            # texture resolution, metres per pixel
ROOF_CELL_M = 0.75        # roof height-field cell
MAX_TEX_PX = 4096
TEXTURE_FORMATS = ("ktx2", "webp", "jpeg", "png")
# KTX2/Basis: the texture stays compressed on the GPU (about an eighth of a
# decoded PNG/JPEG/WebP) and needs no decode on the main thread -- what a
# city-scale view runs out of first. Encoded by the basisu command (Homebrew
# basis_universal): ETC1S, file size about a JPEG's, 4 bits per texel on the
# GPU. Cesium (KHR_texture_basisu) and three.js read it.
BASISU_ARGS = ["-ktx2", "-mipmap", "-q", "200", "-comp_level", "2"]


def _mult4(n: int) -> int:
    """Block-compressed textures need sides that are multiples of 4."""
    return max(4, int(np.ceil(n / 4.0)) * 4)


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

    def capped_runs(self, plane, offset, z_lo, z_hi, a0, a1, min_run_m: float = 1.0, tol_m: float = 0.5):
        """[a0, a1] cut where the roof's height just inside changes, each
        stretch's wall top the roof there (at most ``z_hi``); stretches
        whose roof is not above ``z_lo`` are dropped. A wall drawn under a
        roof edge (an ``edge`` band) stops at the roof, not at its band's top."""
        st, top = self.tops_along(plane, offset, a0, a1)
        if not len(st):
            return []
        half = 0.5 * (a1 - a0) / len(st)
        out = []
        for s, t in zip(st, np.minimum(top, z_hi)):
            if not np.isfinite(t) or t <= z_lo + 0.2:
                continue
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
                  ground_field: "GroundField | None" = None, edge_face: int | None = None):
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
    solid_tops = [b.z_lo for b in bands if b.status in WALL_STATUSES and b.offset_m is not None]
    used = []
    has_solid = bool(solid_tops)
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
    prev = None                                     # (runs, z_hi) of the band below
    for b in used:
        runs = _runs(plane, b, roof)
        is_bottom = (ground_field is not None and edge_face is not None and
                     edge_face < LEVEL_FACE_STRIDE and abs(b.z_lo - z0) < 1e-3)
        # a wall drawn under a roof edge because nothing says it is not there
        # stops at the roof just inside, stretch by stretch
        tops = ([(ra, rc, off, zt) for a, c, off in runs for ra, rc, zt in roof.capped_runs(plane, off, b.z_lo, b.z_hi, a, c)]
                if b.status == "edge" and roof is not None else [(a, c, off, b.z_hi) for a, c, off in runs])
        runs = [(a, c, off) for a, c, off, _ in tops]
        for a, c, off, z_top in tops:
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


def face_texture(plane: WallPlane, bands: list[FaceBand], extent, xyz: np.ndarray,
                 rgb: np.ndarray, texel_m: float = TEXEL_M, party: bool = False):
    """Rasterise the coloured cloud onto the face: a point counts where it
    lies on one of the face's band rectangles (within 0.6 m of the offset
    the profile gives at that height and position along the edge), or on a
    wall just behind a band that caught few points (``_wall_behind_band``;
    never on a ``party`` face)."""
    z0, z1, L = extent
    along, off = plane.local(xyz[:, :2])
    z = xyz[:, 2]
    near = (along >= -0.5) & (along <= L + 0.5) & (z >= z0 - 0.5) & (z <= z1 + 0.5)
    keep = np.zeros(len(xyz), dtype=bool)
    idx = np.flatnonzero(near)
    for b in bands:
        if b.status == "open" and b.back_offset_m is not None:
            b = FaceBand(b.z_lo, b.z_hi, b.back_offset_m, b.n_returns, 0.0, b.along_lo, b.along_hi, "observed")
        if b.status not in WALL_STATUSES or b.offset_m is None:
            continue
        inb = idx[(z[idx] >= b.z_lo - 0.2) & (z[idx] < b.z_hi + 0.2)]
        if not len(inb):
            continue
        expect = np.full(len(inb), b.offset_m)
        for a0, a1, oseg in b.segments:
            expect[(along[inb] >= a0) & (along[inb] < a1)] = oseg
        matched = np.abs(off[inb] - expect) < 0.6
        if not party:
            matched = _wall_behind_band(off[inb], expect, b.offset_m, float(getattr(plane, "offset_m", 0.0)), matched)
        keep[inb[matched]] = True
    W = max(2, min(MAX_TEX_PX, int(np.ceil(L / texel_m))))
    H = max(2, min(MAX_TEX_PX, int(np.ceil((z1 - z0) / texel_m))))
    sx, sy = W / L, H / (z1 - z0)
    img = np.zeros((H, W, 3), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.float64)
    k = np.flatnonzero(keep)
    coloured = k[rgb[k].sum(axis=1) > 15]
    if len(coloured):
        u = np.clip((along[coloured] * sx).astype(int), 0, W - 1)
        v = np.clip(((z[coloured] - z0) * sy).astype(int), 0, H - 1)
        np.add.at(img, (v, u), rgb[coloured].astype(np.float64))
        np.add.at(cnt, (v, u), 1.0)
    return _fill_holes(img, cnt, solid_px=_solid_px(texel_m)), int(len(coloured))


#: where a texel's colour came from -- the measured product keeps SEEN and
#: NEAR, and says PLAIN where nothing saw the wall; the completed product
#: fills PLAIN texels and records how (kyfromabove_3d.facade_fill)
from kyfromabove_3d.facade_fill import NEAR, PLAIN, SEEN                # noqa: E402


class Texture(np.ndarray):
    """A texture image that knows, per pixel, where its colour came from
    (``source``: SEEN -- a frame, or the cloud a frame coloured; NEAR -- the
    nearest seen pixel across a small gap; PLAIN -- the face's median, where
    nothing saw it). Rows run top-down, as the image does."""

    def __new__(cls, img: np.ndarray, source: np.ndarray):
        obj = np.asarray(img).view(cls)
        obj.source = source
        return obj

    def __array_finalize__(self, obj):
        self.source = getattr(obj, "source", None)


def _gap_between(filled: np.ndarray, max_px: int) -> np.ndarray:
    """Empty texels that lie *between* filled ones: a filled texel within
    ``max_px`` on both sides along one of four lines through it -- across,
    up, or either diagonal. A gap inside a sparse but regular raster of
    points is between; the ring round a lone point is not."""
    H, W = filled.shape

    def seen_towards(dy, dx):
        hit = np.zeros_like(filled)
        for k in range(1, max_px + 1):
            sy, sx = dy * k, dx * k
            if abs(sy) >= H or abs(sx) >= W:
                break                                   # past the texture's edge
            src = filled[max(sy, 0):H + min(sy, 0), max(sx, 0):W + min(sx, 0)]
            hit[max(-sy, 0):H + min(-sy, 0), max(-sx, 0):W + min(-sx, 0)] |= src
        return hit

    between = np.zeros_like(filled)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        between |= seen_towards(dy, dx) & seen_towards(-dy, -dx)
    return ~filled & between


def _solid_px(texel_m: float) -> int:
    """The window, about 1 m across and an odd number of texels, in which a
    seen texel must have seen neighbours to count on a wall."""
    return max(3, int(round(1.0 / texel_m)) | 1)


def _row_medians(out: np.ndarray, seen: np.ndarray, band_px: int, rows: np.ndarray,
                 min_seen: int = 30) -> np.ndarray:
    """Per row, the median seen colour within ``band_px`` rows of it -- the
    same floor, give or take. A floor too little seen takes the nearest floor
    that was; a face with none such, its own median."""
    H = len(seen)
    cum = np.concatenate([[0], np.cumsum(seen.sum(axis=1))])
    lo = np.maximum(np.arange(H) - band_px, 0); hi = np.minimum(np.arange(H) + band_px + 1, H)
    enough = np.flatnonzero(cum[hi] - cum[lo] >= min_seen)
    med = np.tile(np.median(out[seen], axis=0), (H, 1))
    if not len(enough):
        return med
    for r in np.flatnonzero(rows):
        k = enough[np.argmin(np.abs(enough - r))]
        med[r] = np.median(out[lo[k]:hi[k]][seen[lo[k]:hi[k]]], axis=0)
    return med


def _fill_holes(img: np.ndarray, cnt: np.ndarray, max_fill_px: int = 10, solid_px: int | None = None) -> Texture:
    """Mean per pixel; an empty pixel *between* filled ones -- a filled one
    within ``max_fill_px`` on both sides, across or up -- takes the nearest
    filled colour (a gap between returns); any other takes the face's median
    colour (a stretch of wall nothing saw), so unseen wall reads as plain
    surface. A lone point stays a point: nearest-fill in every direction had
    grown each one into a disc up to 3 m across, round dots on the walls. The
    result records which of the three each pixel is (:class:`Texture`).

    With ``solid_px`` (walls): a seen texel counts only where at least half
    the ``solid_px`` square round it was seen -- a scattered speck is dropped
    as unseen, since bridging specks drew lines and growing them drew dots --
    and colour then grows up to ``max_fill_px`` out from solid patches about
    2 m across or more (a smaller blob grew a ring: a dot again). Unseen wall
    takes the median colour of its own floor (the rows within ``max_fill_px``,
    specks included), or of the nearest floor seen enough -- not the whole
    face's, which shadow can darken."""
    from scipy.ndimage import distance_transform_edt, label, maximum_filter, uniform_filter

    filled = cnt > 0
    if not filled.any():
        return Texture(np.full(img.shape, 128, dtype=np.uint8), np.full(cnt.shape, PLAIN, dtype=np.uint8))
    out = np.zeros_like(img)
    out[filled] = img[filled] / cnt[filled][:, None]
    if solid_px is None:
        plain = np.tile(np.median(out[filled], axis=0), (len(filled), 1))
        near = _gap_between(filled, max_fill_px)
    else:
        core = uniform_filter(filled.astype(np.float64), solid_px, mode="nearest") >= 0.5
        solid = filled & maximum_filter(core, solid_px)        # a core and the seen rim round it
        if not solid.any():                    # specks only: a plain face in their colour
            return Texture(np.broadcast_to(np.clip(np.median(out[filled], axis=0), 0, 255).astype(np.uint8),
                                           img.shape).copy(), np.full(cnt.shape, PLAIN, dtype=np.uint8))
        seen_any, filled = filled, solid
        lab, n = label(solid, structure=np.ones((3, 3)))
        big = np.flatnonzero(np.bincount(lab.ravel(), minlength=n + 1)[1:] >= (2 * solid_px) ** 2) + 1
        grows = np.isin(lab, big)              # patches about 2 m across or more: a blob grows no ring
        near = ~filled & (distance_transform_edt(~grows) <= max_fill_px) if grows.any() else np.zeros_like(filled)
        # a floor's colour from everything seen on it, specks too: untrusted one by one, fair together
        plain = _row_medians(out, seen_any, max_fill_px, (~filled & ~near).any(axis=1))
    _, (iy, ix) = distance_transform_edt(~filled, return_indices=True)
    out = out[iy, ix]
    blank = ~filled & ~near
    out[blank] = np.broadcast_to(plain[:, None, :], out.shape)[blank]
    source = np.where(filled, SEEN, np.where(near, NEAR, PLAIN)).astype(np.uint8)
    # image rows run top-down; our v runs bottom-up
    return Texture(np.clip(out[::-1], 0, 255).astype(np.uint8), source[::-1].copy())


def tile_geometry(building: Path, xyz: np.ndarray, poly, cxyz: np.ndarray, crgb: np.ndarray,
                  texel_m: float = TEXEL_M, max_offset_m: float = 2.0):
    """The locally fitted wall tiles the colourise resamples (12 m squares
    fitted to the wall returns, kept where they lie on a plane), each as a
    textured rectangle in its own plane. These are the wall where the LiDAR
    saw it directly; the profile bands stand in where it did not."""
    from shapely import contains_xy

    from kyfromabove_3d.features import split_surface_zones
    from colorize_building import resample_walls

    ring = np.array(poly.exterior.coords)
    on_building = contains_xy(poly.buffer(max_offset_m), xyz[:, 0], xyz[:, 1])
    b_xyz = xyz[on_building]
    zones = split_surface_zones(b_xyz)
    if zones is None:
        return []
    height_m = float(np.percentile(b_xyz[:, 2], 99) - b_xyz[:, 2].min())
    _, _, _, _, _, tiles_by_face = resample_walls(b_xyz, zones.wall, ring, height_m, 0.15, 4.0, max_offset_m)
    out = []
    for face, tiles in tiles_by_face.items():
        for t in tiles:
            if "t1" not in t:
                continue
            c, n, t1, t2 = t["centroid"], t["normal"], t["t1"], t["t2"]
            (a0, a1), (b0, b1) = t["a_range"], t["b_range"]
            if a1 - a0 < 0.5 or b1 - b0 < 0.5:
                continue
            corners = np.array([c + a0 * t1 + b0 * t2, c + a1 * t1 + b0 * t2,
                                c + a1 * t1 + b1 * t2, c + a0 * t1 + b1 * t2])
            # texture: the coloured points within 0.3 m of the tile's plane
            rel = cxyz - c
            d = rel @ n
            ta, tb = rel @ t1, rel @ t2
            m = (np.abs(d) < 0.3) & (ta >= a0 - 0.2) & (ta <= a1 + 0.2) & (tb >= b0 - 0.2) & (tb <= b1 + 0.2)
            m &= crgb.sum(axis=1) > 15
            W = max(2, min(MAX_TEX_PX, int(np.ceil((a1 - a0) / texel_m))))
            H = max(2, min(MAX_TEX_PX, int(np.ceil((b1 - b0) / texel_m))))
            img = np.zeros((H, W, 3)); cnt = np.zeros((H, W))
            if m.any():
                u = np.clip(((ta[m] - a0) / (a1 - a0) * W).astype(int), 0, W - 1)
                v = np.clip(((tb[m] - b0) / (b1 - b0) * H).astype(int), 0, H - 1)
                np.add.at(img, (v, u), crgb[m].astype(np.float64)); np.add.at(cnt, (v, u), 1.0)
            tex = _fill_holes(img, cnt, solid_px=_solid_px(texel_m))
            uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
            out.append((_orient(corners, n), n, uv, tex, face, int(m.sum())))
    return out


# ---------------------------------------------------------------------------- roof

def roof_heightfield(xyz: np.ndarray, poly, ground_z: float, cell: float = ROOF_CELL_M,
                     max_z: Optional[float] = None, edge_m: float = 2.0):
    """Top height of the returns in each cell inside the footprint.

    Returns (xs, ys, Z, valid): cell-centre coordinates, heights, and which
    cells hold a value (their own returns, or the nearest cell's within 3 m).
    """
    from scipy.ndimage import distance_transform_edt
    from scipy.stats import binned_statistic_2d
    from shapely import contains_xy

    inside = contains_xy(poly.buffer(0.3), xyz[:, 0], xyz[:, 1]) & (xyz[:, 2] > ground_z + 2.0)
    if max_z is not None:
        # a taller neighbour's wall spills a metre or two over the outline;
        # above max_z only that strip is dropped. Anything standing on the
        # roof inside it -- a penthouse, a lantern, a water tank -- stays: the
        # Kentucky Home Life Building's 20 m2 rooftop structure to 94 m was
        # cut at 77 m and left as a ring of vertical panels
        import shapely
        high = inside & (xyz[:, 2] > max_z)
        if high.any():
            near_edge = shapely.distance(shapely.points(xyz[high, 0], xyz[high, 1]), poly.exterior) <= edge_m
            drop = np.flatnonzero(high)[near_edge]
            inside[drop] = False
    p = xyz[inside]
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx + cell / 2, maxx, cell)
    ys = np.arange(miny + cell / 2, maxy, cell)
    if len(xs) < 2 or len(ys) < 2 or len(p) < 3:
        # nothing above the ground here (a low structure, a footprint the
        # scanner missed): no roof from the returns
        xs = xs if len(xs) >= 2 else np.array([minx + cell / 2, minx + 1.5 * cell])
        ys = ys if len(ys) >= 2 else np.array([miny + cell / 2, miny + 1.5 * cell])
        Z = np.full((len(ys), len(xs)), np.nan)
        return xs, ys, Z, np.zeros(Z.shape, dtype=bool)
    xe = np.concatenate([xs - cell / 2, [xs[-1] + cell / 2]])
    ye = np.concatenate([ys - cell / 2, [ys[-1] + cell / 2]])
    # The top surface of each cell, not its median: a cell under an overhang
    # holds returns from the roof above and the roof below (Humana's top
    # floors stand out over the Main Street podium), and the median hangs a
    # curtain between the two. What the nadir sees is the upper one.
    med, _, _, _ = binned_statistic_2d(p[:, 0], p[:, 1], p[:, 2],
                                       statistic=lambda v: np.percentile(v, 80), bins=[xe, ye])
    cnt, _, _, _ = binned_statistic_2d(p[:, 0], p[:, 1], None, statistic="count", bins=[xe, ye])
    Z = med.T                                       # [iy, ix]
    have = (cnt.T >= 2) & np.isfinite(Z)
    X, Y = np.meshgrid(xs, ys)
    infp = contains_xy(poly, X.ravel(), Y.ravel()).reshape(X.shape)
    dist, (iy, ix) = distance_transform_edt(~have, return_indices=True)
    Zf = Z[iy, ix]
    valid = infp & (have | (dist * cell <= 3.0))
    Zf[~valid] = np.nan
    # A 3x3 median where at least five neighbours are valid: a one-cell
    # spike (a lattice dome's rib, a mast) is not a wall, and left alone it
    # became a fringe of small vertical triangles round every crown.
    from scipy.ndimage import generic_filter
    def _nanmedian(v):
        v = v[np.isfinite(v)]
        return np.median(v) if len(v) >= 5 else np.nan
    Zs = generic_filter(Zf, _nanmedian, size=3, mode="constant", cval=np.nan)
    smooth = valid & np.isfinite(Zs)
    Zf[smooth] = Zs[smooth]
    return xs, ys, Zf, valid


def roof_sloped_fraction(Z: np.ndarray, valid: np.ndarray, cell: float,
                         lo_deg: float = 10.0, hi_deg: float = 60.0, patch_m: float = 3.0) -> float:
    """Share of the roof cells that slope like a pitched roof: between
    ``lo_deg`` (flatter is a flat roof's drainage fall) and ``hi_deg``
    (steeper is a wall), in a sloping patch at least ``patch_m`` across.

    The patch rule matters downtown: the height field is a per-cell
    percentile, smoothed, so a sheer step between two flat roofs becomes a
    ramp one or two cells wide -- on a sample of 45 flat-plated roofs a
    fifth of the cells read as sloped that way, from steps, plant rooms and
    parapets. A pitched roof slopes over metres. Only cells whose
    neighbours are all valid count, so the footprint's edge is not a slope.
    """
    from scipy.ndimage import binary_opening
    if valid.sum() < 4:
        return 0.0
    gy, gx = np.gradient(np.where(valid, Z, np.nan), cell)
    ok = valid & np.isfinite(gx) & np.isfinite(gy)
    if not ok.any():
        return 0.0
    slope = np.degrees(np.arctan(np.hypot(np.nan_to_num(gx), np.nan_to_num(gy))))
    sloped = ok & (slope >= lo_deg) & (slope <= hi_deg)
    k = max(int(round(patch_m / cell)), 1)
    sloped = binary_opening(sloped, structure=np.ones((k, k), dtype=bool))
    return float(sloped.sum() / ok.sum())


def outside_modelled_wall(cen: np.ndarray, planes, profiles, slack_m: float = 1.0) -> np.ndarray:
    """Which of ``cen`` stand outside the wall the profile puts at their
    height: between a face's modelled wall and its footprint edge. A steep
    roof cell there is the underside of an overhang that the 2.5-D height
    field has filled in as wall -- Humana's porch cantilevers 24 m over its
    podium -- and not a wall the LiDAR saw."""
    out = np.zeros(len(cen), dtype=bool)
    for plane in planes:
        L = plane.length_m
        along, off = plane.local(cen[:, :2])
        on_face = (along >= -1.0) & (along <= L + 1.0) & (off > -0.6 * L) & (off < 1.0)
        known = [(0.5 * (b.z_lo + b.z_hi), b.offset_m) for b in profiles.get(plane.face, [])
                 if b.status in WALL_STATUSES and b.offset_m is not None]
        if not known or not on_face.any():
            continue
        # at a height with no band of its own, the nearest known band
        # within 30 m speaks for the wall
        kz = np.array([z for z, _ in known]); ko = np.array([o for _, o in known])
        i = np.abs(cen[:, 2][:, None] - kz[None, :]).argmin(axis=1)
        has = np.abs(cen[:, 2] - kz[i]) <= 30.0
        out[on_face & has & (off > ko[i] + slack_m)] = True
    return out


def roof_mesh(xs, ys, Z, valid, wall_quads, planes=(), profiles=None, max_slope_deg: float = 75.0,
              min_step_m: float = 2.0, levels=None):
    """Two triangles per cell of four valid corners, split into the flat
    ones (the roof, textured from the plan view) and the steep ones (a step
    or a wall the LiDAR saw that no band models -- Humana's bulging north
    side); a steep triangle standing where a wall quad already is gets
    dropped."""
    ny, nx = Z.shape
    vid = -np.ones((ny, nx), dtype=int)
    vid[valid] = np.arange(valid.sum())
    X, Y = np.meshgrid(xs, ys)
    verts = np.column_stack([X[valid], Y[valid], Z[valid]])
    tris = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            a, b, c, d = vid[iy, ix], vid[iy, ix + 1], vid[iy + 1, ix], vid[iy + 1, ix + 1]
            if min(a, b, c, d) < 0:
                continue
            tris.append((a, b, d))
            tris.append((a, d, c))
    tris = np.array(tris, dtype=np.int64).reshape(-1, 3)
    if not len(tris):
        return verts, tris, tris
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    nz = n[:, 2] / (np.linalg.norm(n, axis=1) + 1e-9)
    drop = np.max(np.stack([v0[:, 2], v1[:, 2], v2[:, 2]]), axis=0) - np.min(np.stack([v0[:, 2], v1[:, 2], v2[:, 2]]), axis=0)
    steep = (nz < np.cos(np.radians(max_slope_deg))) & (drop >= min_step_m)
    if steep.any() and wall_quads:
        cen = (v0 + v1 + v2) / 3
        near = np.zeros(len(tris), dtype=bool)
        # a wall quad's plan segment, and its height range
        segs = [(q[0][0][:2], q[0][1][:2], q[0][:, 2].min(), q[0][:, 2].max()) for q in wall_quads if q[3] == "wall"]
        for s_idx in np.flatnonzero(steep):
            cx, cy, cz = cen[s_idx]
            for pa, pb, zlo, zhi in segs:
                if cz < zlo - 1 or cz > zhi + 1:
                    continue
                d = pb - pa
                t = np.clip(np.dot([cx - pa[0], cy - pa[1]], d) / (np.dot(d, d) + 1e-9), 0, 1)
                if np.hypot(*(pa + t * d - [cx, cy])) < 1.5:
                    near[s_idx] = True
                    break
        if profiles:
            near |= outside_modelled_wall(cen, planes, profiles)
        if levels and len(levels) > 1:
            for lv in levels[1:]:
                ring_lv = np.asarray(lv["ring"], dtype=float)
                for pa, pb in zip(ring_lv[:-1], ring_lv[1:]):
                    d = pb - pa
                    L2 = float(np.dot(d, d))
                    if L2 < 1e-6:
                        continue
                    t = np.clip(((cen[:, 0] - pa[0]) * d[0] + (cen[:, 1] - pa[1]) * d[1]) / L2, 0, 1)
                    dist = np.hypot(cen[:, 0] - (pa[0] + t * d[0]), cen[:, 1] - (pa[1] + t * d[1]))
                    near |= steep & (dist < 2.5)
        tris, steep = tris[~(steep & near)], steep[~(steep & near)]
    return verts, tris[~steep], tris[steep]


def roof_texture(xs, ys, Z, valid, poly, cams, texel_m: float = TEXEL_M):
    """Plan-view texture of the roof sampled through the camera models."""
    from scipy.ndimage import distance_transform_edt
    from shapely import contains_xy

    from colorize_building import sample_colour

    minx, miny, maxx, maxy = poly.bounds
    W = _mult4(min(MAX_TEX_PX, int(np.ceil((maxx - minx) / texel_m))))
    H = _mult4(min(MAX_TEX_PX, int(np.ceil((maxy - miny) / texel_m))))
    tx = minx + (np.arange(W) + 0.5) * (maxx - minx) / W
    ty = miny + (np.arange(H) + 0.5) * (maxy - miny) / H
    TX, TY = np.meshgrid(tx, ty)
    # height from the nearest valid cell
    cell = xs[1] - xs[0]
    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    Zf = Z[iy, ix]
    gx = np.clip(((TX - xs[0]) / cell + 0.5).astype(int), 0, len(xs) - 1)
    gy = np.clip(((TY - ys[0]) / cell + 0.5).astype(int), 0, len(ys) - 1)
    tz = Zf[gy, gx]
    inside = contains_xy(poly, TX.ravel(), TY.ravel()) & valid[gy, gx].ravel()
    pts = np.column_stack([TX.ravel(), TY.ravel(), tz.ravel()])[inside]
    up = np.repeat(np.array([[0.0, 0.0, 1.0]]), len(pts), axis=0)
    rgb, seen, _, _ = sample_colour(pts, up, cams, min_facing_cos=0.3, occlusion_slack_m=1.5)
    img = np.zeros((H * W, 3)); cnt = np.zeros(H * W)
    ii = np.flatnonzero(inside)[seen > 0]
    img[ii] = rgb[seen > 0]; cnt[ii] = 1
    tex = _fill_holes(img.reshape(H, W, 3), cnt.reshape(H, W))
    return tex, (minx, miny, maxx, maxy), int((seen > 0).sum()), int(inside.sum())


def roof_texture_from_cloud(xs, ys, Z, valid, poly, cxyz: np.ndarray, crgb: np.ndarray,
                            texel_m: float = TEXEL_M, flat_z: float | None = None):
    """Plan-view texture of the roof rasterized from the colorized aerial point cloud.

    Used when camera crop PNGs are not on disk (which occurs after run_downtown
    cleans intermediate crops to save disk space).
    """
    from scipy.ndimage import distance_transform_edt
    from shapely import contains_xy

    minx, miny, maxx, maxy = poly.bounds
    W = _mult4(min(MAX_TEX_PX, int(np.ceil((maxx - minx) / texel_m))))
    H = _mult4(min(MAX_TEX_PX, int(np.ceil((maxy - miny) / texel_m))))

    inside = contains_xy(poly, cxyz[:, 0], cxyz[:, 1])
    has_color = crgb.sum(axis=1) > 15
    m = inside & has_color
    if not m.any():
        return None, (minx, miny, maxx, maxy), 0, 0

    if flat_z is not None:
        on_roof = m & (cxyz[:, 2] >= flat_z - 3.0)
        if not on_roof.any():
            on_roof = m
    elif valid.any():
        cell = xs[1] - xs[0]
        _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
        Zf = Z[iy, ix]
        gx = np.clip(((cxyz[:, 0] - xs[0]) / cell + 0.5).astype(int), 0, len(xs) - 1)
        gy = np.clip(((cxyz[:, 1] - ys[0]) / cell + 0.5).astype(int), 0, len(ys) - 1)
        on_roof = m & (np.abs(cxyz[:, 2] - Zf[gy, gx]) < 2.0)
        if not on_roof.any():
            on_roof = m
    else:
        top_z = cxyz[m, 2].max()
        on_roof = m & (cxyz[:, 2] >= top_z - 3.0)
        if not on_roof.any():
            on_roof = m

    pts_roof = cxyz[on_roof]
    rgb_roof = crgb[on_roof]

    u = np.clip(((pts_roof[:, 0] - minx) / max(maxx - minx, 1e-6) * W).astype(int), 0, W - 1)
    v = np.clip(((pts_roof[:, 1] - miny) / max(maxy - miny, 1e-6) * H).astype(int), 0, H - 1)

    img = np.zeros((H, W, 3), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.float64)
    np.add.at(img, (v, u), rgb_roof.astype(np.float64))
    np.add.at(cnt, (v, u), 1.0)

    tex = _fill_holes(img, cnt, max_fill_px=max(4, int(2.0 / texel_m)))
    return tex, (minx, miny, maxx, maxy), int(len(pts_roof)), int(inside.sum())


# ---------------------------------------------------------------------------- glTF

# ------------------------------------------------------------- the fallback

def edge_top_z(xs, ys, Z, valid, p0: np.ndarray, p1: np.ndarray, within_m: float = 2.0,
               default: float = 6.0) -> float:
    """The roof height along one footprint edge: the median of the valid
    height-field cells within ``within_m`` of the edge. A podium's edge is
    not extruded to the tower's top this way."""
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X[valid], Y[valid]])
    zs = Z[valid]
    if not len(pts):
        return default
    d = p1 - p0
    L2 = float(d @ d) + 1e-9
    t = np.clip(((pts - p0) @ d) / L2, 0.0, 1.0)
    dist = np.linalg.norm(pts - (p0 + t[:, None] * d), axis=1)
    near = zs[dist <= within_m]
    return float(np.median(near)) if len(near) >= 3 else default


def party_edges(ring_m: np.ndarray, neighbours: list, faces, within_m: float = 0.5,
                min_run_m: float = 1.0, tops: list | None = None) -> dict:
    """Where the faces are party walls: the stretches of each edge that lie
    within ``within_m`` of a neighbouring footprint. Nothing sees a shared
    wall -- no return, no camera -- so it is drawn plain rather than painted
    with whatever stands behind it. Only up to the neighbour's roof, though:
    above it the wall stands in the open and the cameras see it. Returns
    ``{face: [(along_lo, along_hi, the neighbour's roof as local z, or None
    when unknown), ...]}`` -- one stretch per neighbour the edge touches, so
    an edge shared with a tall building on one half and a low one on the
    other is plain to a different height on each."""
    from shapely.geometry import LineString

    if not neighbours:
        return {}
    out: dict = {}
    tops = list(tops) if tops is not None else [None] * len(neighbours)
    zones = [g.buffer(within_m) for g in neighbours]
    for f in faces:
        edge = LineString(f.world_line())
        if edge.length <= 0:
            continue
        stretches = []
        for zone, t in zip(zones, tops):
            inter = edge.intersection(zone)
            pieces = list(inter.geoms) if hasattr(inter, "geoms") else [inter]
            for piece in pieces:
                if piece.is_empty or piece.geom_type != "LineString" or piece.length < min_run_m:
                    continue
                along, _ = f.local(np.array(piece.coords))
                s0, s1 = float(along.min()), float(along.max())
                if s1 - s0 >= min_run_m:
                    stretches.append((s0, s1, None if t is None else float(t)))
        if stretches:
            out[f.face] = sorted(stretches)
    return out


def party_steps(stretches, a: float, b: float, z_top: float, min_open_m: float = 0.5):
    """Cut one prism run ``[a, b]`` up to ``z_top`` by the party stretches:
    ``(c0, c1, plain_top)`` pieces, ``plain_top`` the height the wall is
    plain up to (``None`` where the wall is in the open the whole way)."""
    cuts = {a, b}
    for s0, s1, _ in stretches:
        for c in (s0, s1):
            if a < c < b:
                cuts.add(c)
    cuts = sorted(cuts)
    out = []
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        mid = 0.5 * (c0 + c1)
        covering = [t for s0, s1, t in stretches if s0 - 1e-6 <= mid <= s1 + 1e-6]
        if not covering:
            plain = None
        elif any(t is None for t in covering):
            plain = z_top
        else:
            plain = min(max(covering), z_top)
            if plain >= z_top - min_open_m:
                plain = z_top
        if out and out[-1][2] == plain:
            out[-1] = (out[-1][0], c1, plain)
        else:
            out.append((c0, c1, plain))
    return out


def quad_texture_from_cameras(corners: np.ndarray, normal: np.ndarray, cams, texel_m: float = 0.3,
                              min_facing_cos: float = 0.15, arc: "ArcWall | None" = None):
    """A texture for one rectangle sampled straight from the aerial frames
    through their camera models (the same test as the roof), for a wall the
    profile did not model and so has no coloured points to rasterise. With
    ``arc`` the rectangle is in the arc's unrolled frame and every texel is
    sampled where it lands on the circle, facing outward there: one image
    for the whole curve rather than one per piece."""
    from colorize_building import sample_colour

    e1 = corners[1] - corners[0]; e2 = corners[3] - corners[0]
    L1, L2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    if L1 < 1e-6 or L2 < 1e-6:
        return None, 0
    W = max(2, min(MAX_TEX_PX, int(np.ceil(L1 / texel_m))))
    H = max(2, min(MAX_TEX_PX, int(np.ceil(L2 / texel_m))))
    u = (np.arange(W) + 0.5) / W; v = (np.arange(H) + 0.5) / H
    U, V = np.meshgrid(u, v)
    pts = corners[0][None, :] + U.ravel()[:, None] * e1[None, :] + V.ravel()[:, None] * e2[None, :]
    nrm = np.repeat(normal[None, :], len(pts), axis=0)
    if arc is not None:
        pts, nrm = arc.to_world_3d(pts, nrm)
    if not cams:
        return None, 0
    rgb, seen, _, _ = sample_colour(pts, nrm, cams, min_facing_cos=min_facing_cos, occlusion_slack_m=1.5)
    img = np.zeros((H * W, 3)); cnt = np.zeros(H * W)
    img[seen > 0] = rgb[seen > 0]; cnt[seen > 0] = 1
    if not cnt.any():
        return None, 0
    return _fill_holes(img.reshape(H, W, 3), cnt.reshape(H, W), max_fill_px=max(2, int(1.5 / texel_m)),
                       solid_px=_solid_px(texel_m)), int((seen > 0).sum())


def texel_sources(images) -> dict:
    """The share of texels by where their colour came from, over the images
    that record it (:class:`Texture`): ``{"seen", "near", "plain"}``, and the
    texel count. The measured product states its own coverage."""
    counts = np.zeros(4, dtype=np.int64)
    for img in images:
        src = getattr(img, "source", None)
        if src is not None:
            counts += np.bincount(src.ravel(), minlength=4)[:4]
    n = int(counts[1:].sum())
    return {"n": n, **{k: round(float(counts[v]) / max(n, 1), 4) for k, v in (("seen", SEEN), ("near", NEAR), ("plain", PLAIN))}}


class TextureAtlas:
    """Shelf-packs many small textures into a few pages, so a building is a
    handful of draw calls rather than one per face and tile. ``add`` takes an
    image and returns a handle; ``pack`` lays the pages out; ``uv`` maps a
    handle's local (u, v) in [0, 1] to page coordinates."""

    def __init__(self, page_px: int = 4096, gap_px: int = 2):
        self.page_px, self.gap = page_px, gap_px
        self.images = []
        self.meta = []
        self.places = {}
        self.pages = []

    def add(self, image: np.ndarray, meta: dict | None = None) -> int:
        """``meta``: what the image is of (the face, its extent), for the
        completed product; the image keeps its per-pixel source if it is a
        :class:`Texture`."""
        self.images.append(image)
        self.meta.append(meta or {})
        return len(self.images) - 1

    def pack(self):
        order = sorted(range(len(self.images)), key=lambda i: -self.images[i].shape[0])
        pages = []                                      # [ (x_cursor, y_cursor, shelf_h, canvas) ]
        for i in order:
            h, w = self.images[i].shape[:2]
            h, w = min(h, self.page_px), min(w, self.page_px)
            placed = False
            for pi, pg in enumerate(pages):
                x, y, sh, canvas = pg
                if x + w + self.gap <= self.page_px and y + h + self.gap <= self.page_px:
                    pass
                elif y + sh + self.gap + h + self.gap <= self.page_px:
                    x, y, sh = 0, y + sh + self.gap, 0
                else:
                    continue
                canvas[y:y + h, x:x + w] = self.images[i][:h, :w]
                self.places[i] = (pi, x, y, w, h)
                pages[pi] = (x + w + self.gap, y, max(sh, h), canvas)
                placed = True
                break
            if not placed:
                canvas = np.full((self.page_px, self.page_px, 3), 128, dtype=np.uint8)
                canvas[:h, :w] = self.images[i][:h, :w]
                self.places[i] = (len(pages), 0, 0, w, h)
                pages.append((w + self.gap, 0, h, canvas))
        # trim each page to what it used (to a multiple of 4: block compression)
        self.pages = []
        for pi, (x, y, sh, canvas) in enumerate(pages):
            used_w = _mult4(max(p[1] + p[3] for p in self.places.values() if p[0] == pi))
            used_h = _mult4(max(p[2] + p[4] for p in self.places.values() if p[0] == pi))
            self.pages.append(canvas[:used_h, :used_w].copy())
        return self.pages

    def uv(self, handle: int, uv_local: np.ndarray) -> tuple[int, np.ndarray]:
        """(page, uv on the page) for uv_local with v up from the image's bottom."""
        pi, x, y, w, h = self.places[handle]
        H, W = self.pages[pi].shape[:2]
        u = (x + uv_local[:, 0] * w) / W
        # the image row 0 is its top; v_local = 1 is the top
        v_img_top = (y + (1.0 - uv_local[:, 1]) * h) / H
        return pi, np.column_stack([u, 1.0 - v_img_top])


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """glTF vertex colours are linear; image pixels are sRGB."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


class GlbWriter:
    """Minimal glTF 2.0 binary writer: textured or flat-colour primitives.

    ``texture_format``: ``ktx2`` (the default; ``KHR_texture_basisu``,
    needs the ``basisu`` command), ``webp`` (``EXT_texture_webp``),
    ``jpeg`` or ``png`` (core glTF). The extension formats carry no
    fallback image -- a fallback would embed a second copy and undo the
    saving -- so they are listed as required. ``dump_dir`` keeps a PNG of
    every texture beside the glb, for looking at.
    """

    def __init__(self, texture_format: str = "ktx2", dump_dir: Path | None = None):
        if texture_format not in TEXTURE_FORMATS:
            raise ValueError(f"texture format {texture_format!r}; one of {TEXTURE_FORMATS}")
        if texture_format == "ktx2" and not shutil.which("basisu"):
            raise SystemExit("[FAIL] the ktx2 textures need the basisu encoder and it is not on PATH: "
                             "brew install basis_universal, or pass --texture webp|jpeg|png")
        self.texture_format = texture_format
        self.dump_dir = dump_dir
        self.bin = bytearray()
        self.views, self.accessors, self.images, self.textures, self.materials = [], [], [], [], []
        self.prims = []

    def _encode(self, image: np.ndarray, name: str) -> tuple[bytes, str]:
        """The texture's bytes and MIME type in the writer's format."""
        from PIL import Image
        if self.dump_dir is not None:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(self.dump_dir / f"{name}.png")
        fmt = self.texture_format
        if fmt == "ktx2":
            import subprocess
            import tempfile
            h, w = image.shape[:2]
            if h % 4 or w % 4:
                image = np.pad(image, ((0, _mult4(h) - h), (0, _mult4(w) - w), (0, 0)), mode="edge")
            with tempfile.TemporaryDirectory() as td:
                src, dst = Path(td) / f"{name}.png", Path(td) / f"{name}.ktx2"
                Image.fromarray(image).save(src)
                r = subprocess.run(["basisu", *BASISU_ARGS, "-output_file", str(dst), str(src)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                if r.returncode != 0 or not dst.exists():
                    raise RuntimeError(f"basisu failed on {name} ({image.shape[1]}x{image.shape[0]}): "
                                       + "\n".join(r.stdout.splitlines()[-5:]))
                return dst.read_bytes(), "image/ktx2"
        buf = io.BytesIO()
        if fmt == "webp":
            Image.fromarray(image).save(buf, format="WEBP", quality=85, method=4)
            return buf.getvalue(), "image/webp"
        if fmt == "jpeg":
            Image.fromarray(image).save(buf, format="JPEG", quality=90, subsampling=0)
            return buf.getvalue(), "image/jpeg"
        Image.fromarray(image).save(buf, format="PNG")
        return buf.getvalue(), "image/png"

    @property
    def texture_extension(self) -> str | None:
        return {"ktx2": "KHR_texture_basisu", "webp": "EXT_texture_webp"}.get(self.texture_format)

    def _view(self, data: bytes, target=None) -> int:
        while len(self.bin) % 4:
            self.bin += b"\0"
        v = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            v["target"] = target
        self.bin += data
        self.views.append(v)
        return len(self.views) - 1

    def _accessor(self, arr: np.ndarray, ctype: int, atype: str, target: int, minmax=False) -> int:
        arr = np.ascontiguousarray(arr)
        a = {"bufferView": self._view(arr.tobytes(), target), "componentType": ctype,
             "count": int(arr.shape[0]), "type": atype}
        if minmax:
            a["min"] = arr.min(axis=0).tolist(); a["max"] = arr.max(axis=0).tolist()
        self.accessors.append(a)
        return len(self.accessors) - 1

    def material_texture(self, image: np.ndarray, name: str) -> int:
        data, mime = self._encode(np.ascontiguousarray(image, dtype=np.uint8), name)
        self.images.append({"bufferView": self._view(data), "mimeType": mime, "name": name})
        source = len(self.images) - 1
        ext = self.texture_extension
        self.textures.append({"sampler": 0, "extensions": {ext: {"source": source}}} if ext
                             else {"source": source, "sampler": 0})
        # unlit: the imagery already carries the sun; a second lighting pass
        # in the viewer would darken every face the viewer's sun is behind
        self.materials.append({"name": name, "doubleSided": True, "extensions": {"KHR_materials_unlit": {}},
                               "pbrMetallicRoughness": {"baseColorTexture": {"index": len(self.textures) - 1},
                                                        "metallicFactor": 0.0, "roughnessFactor": 1.0}})
        return len(self.materials) - 1

    def material_flat(self, rgb, name: str, unlit: bool = True) -> int:
        m = {"name": name, "doubleSided": True,
             "pbrMetallicRoughness": {"baseColorFactor": [float(c) / 255 for c in rgb] + [1.0],
                                      "metallicFactor": 0.0, "roughnessFactor": 1.0}}
        if unlit:
            m["extensions"] = {"KHR_materials_unlit": {}}
        self.materials.append(m)
        return len(self.materials) - 1

    def primitive_coloured(self, pos: np.ndarray, nrm: np.ndarray, rgb: np.ndarray, idx: np.ndarray, material: int):
        """Per-vertex colour (COLOR_0) instead of a texture."""
        p = np.column_stack([pos[:, 0], pos[:, 2], -pos[:, 1]]).astype(np.float32)
        n = np.column_stack([nrm[:, 0], nrm[:, 2], -nrm[:, 1]]).astype(np.float32)
        attrs = {"POSITION": self._accessor(p, 5126, "VEC3", 34962, minmax=True),
                 "NORMAL": self._accessor(n, 5126, "VEC3", 34962),
                 "COLOR_0": self._accessor(_srgb_to_linear(rgb / 255.0).astype(np.float32), 5126, "VEC3", 34962)}
        self.prims.append({"attributes": attrs, "material": material,
                           "indices": self._accessor(idx.astype(np.uint32), 5125, "SCALAR", 34963)})

    def primitive(self, pos: np.ndarray, nrm: np.ndarray, uv, idx: np.ndarray, material: int):
        # local z-up -> glTF y-up: (x, y, z) -> (x, z, -y)
        p = np.column_stack([pos[:, 0], pos[:, 2], -pos[:, 1]]).astype(np.float32)
        n = np.column_stack([nrm[:, 0], nrm[:, 2], -nrm[:, 1]]).astype(np.float32)
        attrs = {"POSITION": self._accessor(p, 5126, "VEC3", 34962, minmax=True),
                 "NORMAL": self._accessor(n, 5126, "VEC3", 34962)}
        if uv is not None:
            t = np.column_stack([uv[:, 0], 1.0 - uv[:, 1]]).astype(np.float32)
            attrs["TEXCOORD_0"] = self._accessor(t, 5126, "VEC2", 34962)
        self.prims.append({"attributes": attrs, "material": material,
                           "indices": self._accessor(idx.astype(np.uint32), 5125, "SCALAR", 34963)})

    def write(self, path: Path, extras: dict):
        while len(self.bin) % 4:
            self.bin += b"\0"
        used = ["KHR_materials_unlit"]
        if self.textures and self.texture_extension:
            used.append(self.texture_extension)
        gltf = {"asset": {"version": "2.0", "generator": "kyfromabove-3d build_mesh.py"},
                "extensionsUsed": used, "scene": 0, "scenes": [{"nodes": [0]}],
                "nodes": [{"mesh": 0, "name": extras.get("name", "building"), "extras": extras}],
                "meshes": [{"primitives": self.prims}],
                "materials": self.materials, "textures": self.textures, "images": self.images,
                "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
                "accessors": self.accessors, "bufferViews": self.views,
                "buffers": [{"byteLength": len(self.bin)}]}
        if not self.textures:
            del gltf["textures"], gltf["images"], gltf["samplers"]
        elif self.texture_extension:
            gltf["extensionsRequired"] = [self.texture_extension]
        js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        js += b" " * ((4 - len(js) % 4) % 4)
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        with open(path, "wb") as fh:
            fh.write(struct.pack("<4sII", b"glTF", 2, total))
            fh.write(struct.pack("<II", len(js), 0x4E4F534A)); fh.write(js)
            fh.write(struct.pack("<II", len(self.bin), 0x004E4942)); fh.write(bytes(self.bin))
        return total


class GlbRecorder:
    """A GlbWriter that remembers every primitive and material call, so the
    same mesh can be written again with other textures (:meth:`replay`) --
    the completed product is the measured one, triangle for triangle."""

    RECORDED = ("primitive", "primitive_coloured", "material_texture", "material_flat")

    def __init__(self, writer: "GlbWriter"):
        self.writer = writer
        self.calls = []

    def __getattr__(self, name):
        fn = getattr(self.writer, name)
        if name not in self.RECORDED:
            return fn

        def recorded(*a, **k):
            self.calls.append((name, a, k))
            return fn(*a, **k)
        return recorded

    def replay(self, writer: "GlbWriter", textures: dict):
        """Every recorded call on ``writer``; a texture material named in
        ``textures`` takes that image instead."""
        for name, a, k in self.calls:
            if name == "material_texture" and a[1] in textures:
                a = (textures[a[1]], a[1])
            getattr(writer, name)(*a, **k)


def quads_to_arrays(quads):
    P, N, T, I = [], [], [], []
    for corners, n, q_uv, _ in quads:
        b = len(P)
        P.extend(corners.tolist()); N.extend([list(n)] * 4)
        T.extend(q_uv.tolist() if q_uv is not None else [[0, 0]] * 4)
        I.extend([b, b + 1, b + 2, b, b + 2, b + 3])
    return np.array(P), np.array(N), np.array(T), np.array(I)


# -------------------------------------------------------------------------- tileset

def enu_transform(coords: dict):
    """Column-major 4x4 placing the local metre frame (east, north, up at the
    origin) in ECEF, plus the origin's lon/lat/ellipsoidal height."""
    from export_3dtiles import to_ecef
    ecef, geo = to_ecef(np.zeros((1, 3)), coords)
    lon, lat = np.radians(geo[0, 0]), np.radians(geo[0, 1])
    sl, cl, sp, cp = np.sin(lon), np.cos(lon), np.sin(lat), np.cos(lat)
    east = [-sl, cl, 0.0]
    north = [-sp * cl, -sp * sl, cp]
    up = [cp * cl, cp * sl, sp]
    m = east + [0.0] + north + [0.0] + up + [0.0] + ecef[0].tolist() + [1.0]
    return [float(v) for v in m], geo[0].tolist()


# ----------------------------------------------------------------------------- main

def load_footprint(args, ox, oy):
    """(polygon in the local frame, name, gers, record) from --case, --gers
    or --footprint; the building's own footprint.json when none is given."""
    from kyfromabove_3d.footprints import resolve_footprint

    poly, entry = resolve_footprint(args.building, ox, oy, case=args.case, gers=args.gers,
                                    footprint=getattr(args, "footprint", None),
                                    footprints_json=args.footprints, footprint_csv=args.footprint_csv)
    return poly, entry.get("name") or entry.get("overture_name") or entry.get("gers"), entry.get("gers"), entry


def building_classification(b: Path, xyz, poly, levels, profiles, edges, party, Z, valid, cell):
    """The building's structural label, written to classification.json and
    the tileset's extras. A label only: it does not choose the mesh."""
    from shapely import contains_xy
    from kyfromabove_3d.features import classify_building_structure, split_surface_zones

    on_b = contains_xy(poly, xyz[:, 0], xyz[:, 1])
    zones = split_surface_zones(xyz[on_b]) if on_b.sum() >= 20 else None
    c = classify_building_structure(
        levels, zones, profiles, arc_edges={e.face for e in edges if isinstance(e, ArcWall)},
        sloped_fraction=roof_sloped_fraction(Z, valid, cell), party_faces=set(party))
    (b / "classification.json").write_text(json.dumps(c.to_dict(), indent=1), encoding="utf-8")
    print(f"  structure: {c.archetype} (roof {c.roof_type}, facade {c.facade_type})")
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--building", type=Path, required=True)
    ap.add_argument("--case", default=None)
    ap.add_argument("--gers", default=None)
    ap.add_argument("--footprint", type=Path, default=None,
                    help="a footprint.json written by prepare (default when the building dir has one)")
    ap.add_argument("--footprints", type=Path, default=Path("outputs/survey/case_footprints.json"))
    ap.add_argument("--footprint-csv", type=Path, default=Path("data/derived/ky_facade_tiled_wkt.csv"))
    ap.add_argument("--name", default=None, help="display name (default: from the footprint)")
    ap.add_argument("--texel-m", type=float, default=TEXEL_M)
    ap.add_argument("--roof-cell-m", type=float, default=0.0,
                    help="roof height-field cell; 0 = 0.75 m where the returns are dense (>= 4/m2), else 1.5 m")
    ap.add_argument("--out", type=Path, default=None, help="default <building>/3dtiles_mesh/")
    ap.add_argument("--ply", type=Path, default=None,
                    help="coloured cloud to texture the walls from (default <building>/colorized_aerial.laz, "
                         "or .ply from an older run; made here from the aerial frames alone when missing)")
    ap.add_argument("--texture", choices=TEXTURE_FORMATS, default="ktx2",
                    help="texture format in the glb: ktx2 (Basis, GPU-compressed; needs basisu), webp, jpeg, png")
    ap.add_argument("--dump-textures", action="store_true",
                    help="also keep every texture as a lossless PNG in <out>/textures/ (the measured and the "
                         "completed product's): the batched tiles (kyfromabove_3d.publish) repack them")
    ap.add_argument("--refresh", action="store_true", help="remake the aerial-only cloud even if it exists")
    ap.add_argument("--complete", action="store_true",
                    help="also write the completed product (3dtiles_complete/): unseen texels filled from the "
                         "building's own facade, and a tinted copy showing what was invented")
    ap.add_argument("--no-prism-occluders", action="store_true",
                    help="occlude with the LiDAR returns alone, in the colourise and the frame sampling here")
    ap.add_argument("--lidar-outline", action="store_true",
                    help="with --refresh: the colourise uses the building's footprint_lidar.json (scripts/lidar_outline.py)")
    ap.add_argument("--match-walls", action="store_true",
                    help="with --refresh: the colourise gives unknown wall bands the depth the frames agree on "
                         "(kyfromabove_3d.wall_match); matched walls are drawn, and tinted cyan in building_matched.glb")
    ap.add_argument("--allow-no-frames", action="store_true",
                    help="build even though the listed frames' crops are missing (no colour from the frames)")
    ap.add_argument("--neighbours", type=Path, default=None,
                    help="the area's footprints with LiDAR heights, for the neighbour prisms (default: downtown's)")
    ap.add_argument("--neighbour-levels", default=None,
                    help="where the neighbours' levels.json are; 'none' for their outlines at LiDAR height only")
    ap.add_argument("--scene-cloud", type=Path, default=None,
                    help="a cached occluder cloud (scripts/run_downtown.py writes one) passed on to the colourise")
    args = ap.parse_args()

    from colorize_building import load_cameras

    b = args.building
    t_start = time.time()
    out = args.out or (b / "3dtiles_mesh")
    out.mkdir(parents=True, exist_ok=True)
    coords = json.loads((b / "coordinates.json").read_text(encoding="utf-8"))
    ox, oy, _ = coords["origin_source_xyz"]
    poly, name, gers, fp_entry = load_footprint(args, ox, oy)
    name = args.name or name
    lidar = read_cloud(find_cloud(b, "lidar_local"))
    xyz = np.column_stack([lidar["x"], lidar["y"], lidar["z"]]).astype(np.float64)
    ply_path = args.ply or find_cloud(b, "colorized_aerial")
    if args.ply is None and (args.refresh or not ply_path.exists()):
        # The mesh's colour comes from the state's aerial frames alone: a
        # colourise with no --extra-camera, kept apart from colorized.ply
        # (which may carry admitted photographs for the point view). It also
        # writes the wall planes and profiles the mesh is built from.
        import subprocess
        ply_path = b / "colorized_aerial.laz"
        (b / "colorized_aerial.ply").unlink(missing_ok=True)        # an older run's copy
        cmd = [sys.executable, str(Path(__file__).resolve().parent / "colorize_building.py"),
               "--building", str(b), "--out", str(ply_path)]
        if args.case:
            cmd += ["--case", args.case]
        elif args.gers:
            cmd += ["--gers", args.gers]
        elif args.footprint:
            cmd += ["--footprint", str(args.footprint)]
        if args.scene_cloud:
            cmd += ["--scene-cloud", str(args.scene_cloud)]
        if args.neighbours:
            cmd += ["--neighbours", str(args.neighbours)]
        if args.neighbour_levels:
            cmd += ["--neighbour-levels", str(args.neighbour_levels)]
        if args.no_prism_occluders:
            cmd.append("--no-prism-occluders")
        if args.match_walls:
            cmd.append("--match-walls")
        if args.lidar_outline:
            cmd.append("--lidar-outline")
        print(f"colouring from the aerial frames only -> {ply_path.name}")
        t_col = time.time()
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # a line the building step reads to price each stage
        print(f"[time] colourise {time.time() - t_col:.1f} s")
        # the profile's decisions go into the building's log (which faces got
        # a crown, which bands were widened): a rebuild can then be traced
        for ln in r.stdout.splitlines():
            if ln.startswith("    face ") or ln.startswith("    ") and "band(s)" in ln:
                print("  colourise:" + ln)
        if r.returncode != 0:
            # No cameras, or too few returns to split into ground, wall and
            # roof: the building still gets a mesh, by the box path below.
            tail = [ln for ln in r.stdout.splitlines() if ln.startswith("[FAIL]")]
            print(f"  colourise declined ({tail[-1] if tail else f'exit {r.returncode}'}); box path")
            colourise_failed = True
        else:
            colourise_failed = False
    else:
        colourise_failed = not ply_path.exists()
    if (b / "wall_planes.json").exists() and (b / "wall_profiles.json").exists() and not colourise_failed:
        planes = load_wall_planes(b / "wall_planes.json")
        prof = json.loads((b / "wall_profiles.json").read_text(encoding="utf-8"))
        profiles = {int(k): [FaceBand(**d) for d in v] for k, v in prof.items()}
    else:
        planes, profiles = [], {}
    if ply_path.exists() and not colourise_failed:
        col = read_cloud(ply_path)
        if "source" in col and int((np.asarray(col["source"]) == 2).sum()):
            raise SystemExit(f"[FAIL] {ply_path} carries colour from photographs "
                             f"({int((np.asarray(col['source']) == 2).sum()):,} points); the mesh takes "
                             f"aerial imagery only. Pass an aerial-only cloud or --refresh.")
        cxyz = np.column_stack([col["x"], col["y"], col["z"]]).astype(np.float64)
        crgb = np.column_stack([col["red"], col["green"], col["blue"]]).astype(np.uint8)
    else:
        cxyz, crgb = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    print(f"{name}: {len(planes)} faces, {len(xyz):,} returns, {len(cxyz):,} coloured points from {ply_path.name}")

    from kyfromabove_3d.features import wall_planes as footprint_edges
    from kyfromabove_3d.footprints import footprint_wgs84_to_local

    # What a frame can see, as the colourise decides it: the area's returns
    # (the building's own clip reaches 27 m past it and misses the tower
    # across the street) and every neighbour as a solid prism.
    cams = load_cameras(b) if (b / "cameras_normalized.json").exists() else []
    if cams:
        from colorize_building import FOOTPRINTS, levels_dir, neighbour_prisms, occluder_cloud, occlusion_maps
        occ = occluder_cloud(b, args.scene_cloud, 250.0, xyz) if args.scene_cloud else xyz
        kw = {} if args.neighbour_levels is None else {"buildings_dir": levels_dir(args.neighbour_levels)}
        prisms = [] if args.no_prism_occluders else neighbour_prisms(b, args.neighbours or FOOTPRINTS, **kw)
        occlusion_maps(b, cams, occ, prisms)
    # The frames are listed but their crops are gone (run_downtown deletes them
    # after a mesh): a mesh built now would take no colour from the frames --
    # every prism face and roof falls back to the point cloud or to grey. A
    # mesh-only rebuild did that to 386 downtown buildings (September 2026).
    # run_downtown.py re-cuts the crops before a mesh; stop here otherwise.
    listed = json.loads((b / "cameras_normalized.json").read_text(encoding="utf-8")) \
        if (b / "cameras_normalized.json").exists() else {}
    if listed and not cams and not args.allow_no_frames:
        raise SystemExit(f"[FAIL] {len(listed)} frames are listed but none of their crops is on disk; "
                         "re-cut them (scripts/run_downtown.py run --only <gers> --force-step mesh) "
                         "or pass --allow-no-frames")
    heads = sorted({c["head"] for c in cams})
    ring_m = np.array(poly.exterior.coords)
    height_m = fp_entry.get("height_m") or fp_entry.get("lidar_height_m")
    n_on = int(fp_entry.get("lidar_n") or 0) or int(__import__("shapely").contains_xy(poly, xyz[:, 0], xyz[:, 1]).sum())
    neighbours = [footprint_wgs84_to_local(nb["wkt"], ox, oy) for nb in fp_entry.get("neighbours", [])]
    # a neighbour's roof in this building's frame (z = 0 at this ground)
    nb_tops = [((nb["lidar_ground_z"] + nb["lidar_height_m"]) - fp_entry["ground_z_m"])
               if nb.get("lidar_ground_z") is not None and nb.get("lidar_height_m") and fp_entry.get("ground_z_m") is not None
               else None for nb in fp_entry.get("neighbours", [])]
    ground_z = 0.0 if (b / "footprint.json").exists() else min([p.z_lo for p in planes] or [0.0])

    # The path this building takes. A footprint the LiDAR barely touched
    # (a shed under a tree, a kiosk) gets a box: its Overture/LiDAR height,
    # a flat roof, prism walls. Everything else takes the profile where the
    # returns gave one and a prism face at the roof's edge height where not.
    path = "box" if (n_on < 30 or not height_m or colourise_failed) else "profile"
    # The bodies: the footprint alone, or the levels the colourise found (a
    # tower on a podium: the footprint up to the podium top and the tower's
    # own outline above it, its faces numbered from 100).
    from kyfromabove_3d.features import LEVEL_FACE_STRIDE
    lv_path = b / "levels.json"
    levels = json.loads(lv_path.read_text(encoding="utf-8")) if lv_path.exists() and path != "box" else None
    if levels:
        roof_top = max(float(lv["z_hi"]) for lv in levels) - ground_z
    elif height_m:
        roof_top = float(height_m)
    else:
        roof_top = max(float(np.percentile(xyz[:, 2], 99)) - ground_z, 6.0) if len(xyz) else 6.0
    if not levels:
        levels = [{"level": 0, "ring": ring_m.tolist(), "z_lo": ground_z, "z_hi": ground_z + roof_top}]
    cell = args.roof_cell_m if args.roof_cell_m > 0 else (0.75 if n_on / max(poly.area, 1.0) >= 4.0 else 1.5)
    xs, ys, Z, valid = roof_heightfield(xyz, poly, ground_z, cell, max_z=ground_z + roof_top + 1.5)
    if path == "box" or valid.sum() < 4:
        # a flat roof at the building's height; the height field is not trusted
        valid[:] = False

    # Multi-Plane Corner Snapping: rectify footprint corners from adjacent verified wall plane lines
    level0_planes = [p for p in planes if p.face < LEVEL_FACE_STRIDE]
    if any(isinstance(p, VerifiedWallPlane) for p in level0_planes):
        rect_ring, rect_planes = rectify_footprint_corners(levels[0]["ring"], level0_planes)
        levels[0]["ring"] = rect_ring.round(3).tolist()
        rect_by_face = {p.face: p for p in rect_planes}
        planes = [rect_by_face.get(p.face, p) for p in planes]

    cls_arr = np.asarray(lidar["classification"]) if "classification" in lidar else None
    ground_field = ground_field_for_building(
        xyz,
        np.asarray(levels[0]["ring"]),
        classification=cls_arr,
        fallback_ground_z=ground_z,
    )

    edges, edge_level = [], {}
    for lv in levels:
        ring_lv = np.asarray(lv["ring"], dtype=float)
        fp_edges = footprint_edges(ring_lv, float(lv["z_lo"]), float(lv["z_hi"]), min_length_m=0.3, arcs=(int(lv["level"]) == 0))
        assert len(fp_edges) < LEVEL_FACE_STRIDE, f"level {lv['level']} has {len(fp_edges)} edges, exceeds stride {LEVEL_FACE_STRIDE}"
        for e in fp_edges:
            e.face += int(lv["level"]) * LEVEL_FACE_STRIDE
            edges.append(e); edge_level[e.face] = lv
    by_face = {p.face: p for p in planes}
    party = party_edges(ring_m, neighbours, [e for e in edges if edge_level[e.face]["level"] == 0], tops=nb_tops)
    classification = building_classification(b, xyz, poly, levels, profiles, edges, party, Z, valid, cell)
    if len(levels) > 1:
        print(f"  {len(levels)} levels: " + "; ".join(f"level {lv['level']} {float(lv['z_lo']):.0f}-{float(lv['z_hi']):.0f} m" for lv in levels))
    # The roof field the walls are checked against: a wall band may only
    # stand where there is roof behind it at that height. RIVUE Tower's
    # mirrored bands ran a 104 m sheet along the block's edge at tower
    # height; the tower is 30 m wide.
    roof = RoofField(xs, ys, Z, valid) if valid.any() else None

    glb = GlbRecorder(GlbWriter(args.texture, dump_dir=(out / "textures") if args.dump_textures else None))
    atlas = TextureAtlas()
    textured = []                                    # (handle, quads with uv in the image's [0,1] frame)
    flat_quads = []                                  # (rgb, quads)
    all_wall_quads = []
    n_tris = 0
    n_profile = n_prism = n_party = 0
    n_matched_faces, matched_m2 = 0, 0.0              # walls whose depth came from the frames (wall_match)
    used_bands = {}
    n_arc = 0
    def neighbour_wall_at(pl, end: int, z: float):
        """Where the face meeting ``pl`` at its p0 (``end`` 0) or p1 (1)
        has its wall at height ``z``, in plan: on its band's line if it has a
        wall band there, else on the footprint (where its prism stands);
        None when no straight face meets it there."""
        corner = pl.p0 if end == 0 else pl.p1
        # the footprint's edges, not the wall planes: an edge under 3 m has
        # no plane (cfb7194a's face 1006) and stands as a prism on the line
        for nb in edges:
            if nb.face == pl.face or isinstance(nb, ArcWall):
                continue
            if np.hypot(*((nb.p1 if end == 0 else nb.p0) - corner)) > 0.05:
                continue
            nb_plane = by_face.get(nb.face)
            if nb_plane is None or isinstance(nb_plane, ArcWall):
                return np.asarray(corner, float)
            nb_bands = plane_bands(nb_plane, profiles.get(nb.face, []))
            lidar = any(x.status in WALL_STATUSES and x.status != "matched" and x.offset_m is not None for x in nb_bands)
            off = 0.0
            for x in nb_bands:
                if (x.z_lo <= z < x.z_hi and x.status in WALL_STATUSES and x.offset_m is not None
                        and (lidar or x.status != "matched")):
                    off = float(x.offset_m)
            return np.asarray(corner, float) + off * np.asarray(nb_plane.normal, float)
        return None

    for edge in edges:
        plane = by_face.get(edge.face)
        arc = plane if isinstance(plane, ArcWall) else (edge if isinstance(edge, ArcWall) else None)
        if plane is not None and isinstance(edge, ArcWall) != isinstance(plane, ArcWall):
            plane = None                                 # the ring changed shape since the colourise
        bands = plane_bands(plane, profiles.get(plane.face, [])) if plane is not None else []
        # A matched band (its depth from the frames) fills a wall the LiDAR
        # left open on a face it measured; it is never the only evidence for
        # a face. A face with no LiDAR wall band takes the prism, as it would
        # without matching -- one matched band on the Omni's all-unknown face
        # 1000 turned a full prism into one 5 m strip and a 1,000 m2 hole.
        if not any(b.status in WALL_STATUSES and b.status != "matched" and b.offset_m is not None for b in bands):
            bands = [b for b in bands if b.status != "matched"]
        quads, extent = (
            wall_geometry(plane, bands, roof, ground_field=(ArcGround(ground_field, arc) if arc is not None and ground_field is not None else ground_field),
                          edge_face=edge.face)
            if (plane is not None and path != "box")
            else ([], None)
        )
        if quads and arc is None and any(b.status == "matched" for b in bands):
            quads = quads + corner_returns(plane, bands, lambda end, z, pl=plane: neighbour_wall_at(pl, end, z), roof)
        if quads:
            used_bands[plane.face] = bands
            if arc is not None:
                # texture from the coloured points in the unrolled frame; the
                # rectangles back onto the circle, cut every 1.5 m along it
                near = np.abs(np.hypot(*(cxyz[:, :2] - arc.centre).T) - arc.radius) <= 12.0
                tex, n_pts = face_texture(plane, bands, extent, cxyz[near], crgb[near], args.texel_m,
                                          party=edge.face in party)
                quads = roll_quads(quads, arc)
                n_arc += 1
            else:
                tex, n_pts = face_texture(plane, bands, extent, cxyz, crgb, args.texel_m, party=edge.face in party)
            walls = [q for q in quads if q[3] == "wall"]
            matched = [(float(b.z_lo), float(b.z_hi)) for b in bands if b.status == "matched"]
            matched_m2 += sum((b.z_hi - b.z_lo) * (b.along_hi - b.along_lo) for b in bands if b.status == "matched")
            n_matched_faces += bool(matched)
            textured.append((atlas.add(tex, {"face": edge.face, "kind": "profile", "z": (extent[0], extent[1]),
                                             "along": (0.0, extent[2]), **({"matched": matched} if matched else {})}), walls))
            flat = [q for q in quads if q[3] != "wall"]
            if flat:
                flat_quads.append((tex.reshape(-1, 3).mean(axis=0), flat))
            all_wall_quads.extend(quads)
            n_profile += 1
            z0, z1, L = extent
            print(f"  face {edge.face}: {len(walls)} wall rectangle(s), {len(flat)} ledge/return(s), "
                  f"{L:.0f} m x {z1 - z0:.0f} m, texture {tex.shape[1]}x{tex.shape[0]} px from {n_pts:,} points")
            continue
        # prism faces at the roof's height beside this edge, stretch by
        # stretch: a one-storey base gets its 13 m wall along the whole
        # edge and a step where the base ends; a stretch with no roof
        # beside it gets no wall
        seg = edge.p1 - edge.p0
        L = float(np.hypot(*seg)); e1 = seg / L
        n3 = np.array([edge.normal[0], edge.normal[1], 0.0])
        lv = edge_level[edge.face]
        lv_floor, lv_ceiling = max(ground_z, float(lv["z_lo"])), float(lv["z_hi"]) + 1.0
        edge_ground = ArcGround(ground_field, arc) if (arc is not None and ground_field is not None) else ground_field
        if roof is not None:
            stations = [(a, b, min(z, lv_ceiling)) for a, b, z in roof.top_along(edge)
                        if np.isfinite(z) and z - lv_floor >= 0.5]
        else:
            stations = [(0.0, L, ground_z + roof_top)] if roof_top >= 0.5 else []
        # merge neighbouring stretches at the same height
        runs = []
        for a, b, z in stations:
            if runs and abs(runs[-1][2] - z) < 1.0 and abs(runs[-1][1] - a) < 1e-6:
                runs[-1] = (runs[-1][0], b, 0.5 * (runs[-1][2] + z))
            else:
                runs.append((a, b, z))
        made = False
        plain_any = False
        for a, b, z_top in runs:
            # a party wall is plain up to the neighbour's roof and seen above
            # it, stretch by stretch along the edge
            for c0, c1, plain_top in party_steps(party.get(edge.face, []), a, b, z_top):
                if edge_ground is not None and edge.face < LEVEL_FACE_STRIDE and abs(lv_floor - ground_z) < 1e-3:
                    za = edge_ground.elevation_along_edge(edge.face, c0, L)
                    zb = edge_ground.elevation_along_edge(edge.face, c1, L)
                else:
                    za = lv_floor
                    zb = lv_floor
                pa = edge.p0 + c0 * e1; pb = edge.p0 + c1 * e1
                steps = [(za, zb, z_top, True)] if plain_top is None else (
                    [(za, zb, z_top, False)] if plain_top >= z_top else
                    [(za, zb, plain_top, False), (plain_top, plain_top, z_top, True)])
                for z_a, z_b, z_t, seen_wall in steps:
                    corners = _orient(np.array([[*pa, z_a], [*pb, z_b], [*pb, z_t], [*pa, z_t]]), n3)
                    quad = (corners, n3, np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64), "wall")
                    # a curved edge's prism face follows the circle, in pieces
                    # that share one texture sampled along the curve
                    pieces = roll_quads([quad], arc) if arc is not None else [quad]
                    all_wall_quads.extend(pieces)
                    made = True
                    if not seen_wall:
                        plain_any = True
                        flat_quads.append((None, pieces))
                        continue
                    tex, n_seen = quad_texture_from_cameras(corners, n3, cams, arc=arc)
                    if tex is None:
                        if len(cxyz):
                            dummy_band = FaceBand(float(z_a), float(z_t), 0.0, 10, 0.0, 0.0, L, "observed")
                            tex_cloud, n_pts = face_texture(edge, [dummy_band], (float(z_a), float(z_t), L), cxyz, crgb, args.texel_m)
                            if n_pts > 0:
                                textured.append((atlas.add(tex_cloud, {"face": edge.face, "kind": "prism", "z": (float(z_a), float(z_t)), "along": (c0, c1)}), pieces))
                                continue
                        flat_quads.append((None, pieces))
                    else:
                        textured.append((atlas.add(tex, {"face": edge.face, "kind": "prism", "z": (float(z_a), float(z_t)), "along": (c0, c1)}), pieces))
        if made:
            if plain_any:
                n_party += 1
            else:
                n_prism += 1
    if n_prism or n_party or n_arc:
        print(f"  {n_profile} face(s) from the profile ({n_arc} curved), {n_prism} prism face(s) at the roof's edge height, "
              f"{n_party} party wall(s) drawn plain")

    # Fitted wall tiles: the wall where the LiDAR saw it directly.
    tiles = tile_geometry(b, xyz, poly, cxyz, crgb, args.texel_m) if path != "box" else []
    for corners, n, uv, tex, face, n_pts in tiles:
        textured.append((atlas.add(tex, {"face": face, "kind": "tile"}), [(corners, n, uv, "wall")]))
    if tiles:
        print(f"  {len(tiles)} fitted wall tile(s) as textured rectangles, "
              f"{sum(t[5] for t in tiles):,} coloured points behind them")

    # Roof
    if valid.any():
        verts, tris, steep = roof_mesh(xs, ys, Z, valid, all_wall_quads, [by_face[f] for f in used_bands], used_bands, levels=levels)
    else:
        from overture_boxes import extrude
        P_, N_, I_ = extrude(ring_m, ground_z, roof_top)
        top = N_[:, 2] > 0.5
        verts = P_[top]; tris = I_.reshape(-1, 3)[np.all(top[I_.reshape(-1, 3)], axis=1)]
        remap = -np.ones(len(P_), dtype=int); remap[np.flatnonzero(top)] = np.arange(top.sum())
        tris = remap[tris]; steep = np.zeros((0, 3), dtype=int)
    if len(steep):
        # The steep cells are surfaces the planar walls do not describe.
        # They keep the LiDAR's shape and take their colour from whichever
        # aerial frame faces them, one colour per triangle.
        from colorize_building import sample_colour
        v0, v1, v2 = verts[steep[:, 0]], verts[steep[:, 1]], verts[steep[:, 2]]
        n = np.cross(v1 - v0, v2 - v0); n /= np.linalg.norm(n, axis=1)[:, None] + 1e-9
        cen = (v0 + v1 + v2) / 3
        if cams:
            rgb, seen, _, _ = sample_colour(cen, n, cams, min_facing_cos=0.15, occlusion_slack_m=1.5)
        else:
            rgb, seen = np.zeros((len(cen), 3), dtype=np.uint8), np.zeros(len(cen), dtype=int)
        fill = np.median(rgb[seen > 0], axis=0) if (seen > 0).any() else np.array([110, 110, 110])
        rgb[seen == 0] = fill
        P = np.vstack([v0, v1, v2]).reshape(3, -1, 3).transpose(1, 0, 2).reshape(-1, 3)
        N = np.repeat(n, 3, axis=0)
        C = np.repeat(rgb, 3, axis=0)
        glb.primitive_coloured(P, N, C, np.arange(len(P)), glb.material_flat((255, 255, 255), "steep"))
        n_tris += len(steep)
        print(f"  {len(steep):,} steep roof triangle(s) kept as LiDAR-shaped wall, coloured from the frames "
              f"({(seen > 0).sum():,} seen)")
    tex = None
    bbox = None
    source_desc = ""
    if cams and len(verts):
        if valid.any():
            tex, bbox, n_seen, n_tex = roof_texture(xs, ys, Z, valid, poly, cams, args.texel_m)
        else:
            flat_valid = np.ones_like(valid); Zf = np.full(Z.shape, ground_z + roof_top)
            tex, bbox, n_seen, n_tex = roof_texture(xs, ys, Zf, flat_valid, poly, cams, args.texel_m)
        source_desc = f"{n_seen:,} of {n_tex:,} texels seen by a camera ({', '.join(heads)})"
    elif len(verts) and len(cxyz) and (crgb.sum(axis=1) > 15).any():
        flat_z = (ground_z + roof_top) if path == "box" else None
        tex, bbox, n_pts, n_in = roof_texture_from_cloud(xs, ys, Z, valid, poly, cxyz, crgb, args.texel_m, flat_z=flat_z)
        source_desc = f"{n_pts:,} coloured point returns from {ply_path.name}"

    roof_tex = tex if (tex is not None and len(verts)) else None
    if tex is not None and len(verts):
        minx, miny, maxx, maxy = bbox
        uv = np.column_stack([(verts[:, 0] - minx) / max(maxx - minx, 1e-6), (verts[:, 1] - miny) / max(maxy - miny, 1e-6)])
        up = np.repeat(np.array([[0.0, 0.0, 1.0]]), len(verts), axis=0)
        glb.primitive(verts, up, uv, tris.ravel(), glb.material_texture(tex, "roof"))
        print(f"  roof: {valid.sum():,} cells of {cell} m, {len(tris):,} triangles, "
              f"{(Z[valid].min() if valid.any() else roof_top):.0f}-{(Z[valid].max() if valid.any() else roof_top):.0f} m; "
              f"texture {tex.shape[1]}x{tex.shape[0]} px ({source_desc})")
    elif len(verts):
        up = np.repeat(np.array([[0.0, 0.0, 1.0]]), len(verts), axis=0)
        glb.primitive(verts, up, None, tris.ravel(), glb.material_flat((150, 150, 148), "roof_plain"))
        print(f"  roof: {len(tris):,} triangles, plain (no camera or cloud colour sees this building)")
    n_tris += len(tris)

    # Walls: the atlas pages, one primitive each; the plain quads in one more.
    pages = atlas.pack() if textured else []
    page_mats = [glb.material_texture(pg, f"walls_{k}") for k, pg in enumerate(pages)]
    per_page = {k: [] for k in range(len(pages))}
    for handle, quads in textured:
        for corners, n, uv, kind in quads:
            pi, uv_page = atlas.uv(handle, uv)
            per_page[pi].append((corners, n, uv_page, kind))
    for k, quads in per_page.items():
        if quads:
            P, N, T, I = quads_to_arrays(quads)
            glb.primitive(P, N, T, I, page_mats[k])
            n_tris += len(I) // 3
    if flat_quads:
        seen_cols = [c for c, _ in flat_quads if c is not None]
        plain = np.median(np.array(seen_cols), axis=0) if seen_cols else np.array([150, 150, 148])
        Pq, Nq, Cq, Iq = [], [], [], []
        for c, quads in flat_quads:
            P, N, T, I = quads_to_arrays(quads)
            Iq.extend((np.asarray(I) + len(Pq)).tolist()); Pq.extend(P.tolist()); Nq.extend(N.tolist())
            Cq.extend([list(c if c is not None else plain)] * len(P))
        glb.primitive_coloured(np.array(Pq), np.array(Nq), np.array(Cq, dtype=float), np.array(Iq), glb.material_flat((255, 255, 255), "plain"))
        n_tris += len(Iq) // 3
    if pages:
        print(f"  textures: {len(textured)} images packed into {len(pages)} page(s) "
              f"({', '.join(f'{pg.shape[1]}x{pg.shape[0]}' for pg in pages)})")

    texels = texel_sources([img for img in atlas.images] + ([roof_tex] if roof_tex is not None else []))
    extras = {"name": name, "gers": gers, "path": path, "archetype": classification.archetype, "texels": texels,
              "levels": len(levels), "n_frames": len(cams), "heads": heads,
              **({"parent_gers": fp_entry["parent_gers"], "split": fp_entry.get("split")} if fp_entry.get("parent_gers") else {}),
              "faces": {"profile": n_profile, "prism": n_prism, "party": n_party}, "height_m": roof_top,
              "source": "KyFromAbove LiDAR and aerial frames (no photographs); Overture footprint",
              **({"matched": {"faces": n_matched_faces, "area_m2": round(matched_m2, 1)}} if n_matched_faces else {})}
    size = glb.write(out / "building.glb", extras)

    # Tileset: the mesh in its local frame, placed by the root transform.
    transform, origin_geo = enu_transform(coords)
    allp = np.vstack([q[0] for q in all_wall_quads] + [t[0] for t in tiles] + ([verts] if len(verts) else []) + [np.column_stack([ring_m, np.zeros(len(ring_m))])])
    centre = 0.5 * (allp.min(axis=0) + allp.max(axis=0))
    radius = float(np.linalg.norm(allp - centre, axis=1).max())
    tileset = {
        "asset": {"version": "1.1", "generator": "kyfromabove-3d build_mesh.py"},
        "geometricError": 500.0,
        "root": {"transform": transform,
                 "boundingVolume": {"sphere": centre.tolist() + [radius]},
                 "geometricError": 0.0, "refine": "REPLACE",
                 "content": {"uri": "building.glb"}},
        "extras": {"name": name, "gers": gers, "triangles": int(n_tris), "kind": "mesh", "path": path,
                   "archetype": classification.archetype, "texels": texels,
                   "faces": {"profile": n_profile, "prism": n_prism, "party": n_party}, "n_frames": len(cams),
                   "origin_lon_lat_h": origin_geo,
                   **({"parent_gers": fp_entry["parent_gers"]} if fp_entry.get("parent_gers") else {}),
                   **({"matched": extras["matched"]} if "matched" in extras else {})},
    }
    (out / "tileset.json").write_text(json.dumps(tileset, indent=1), encoding="utf-8")
    if n_matched_faces:
        write_matched_tint(out, glb, atlas, args, extras, tileset)
    print(f"mesh -> {out / 'building.glb'} ({size / 1e6:.1f} MB, {n_tris:,} triangles); tileset -> {out / 'tileset.json'}")
    print(f"[time] mesh {time.time() - t_start:.1f} s (with the colourise)")
    if args.complete:
        t_done = time.time()
        write_completed(args.building, glb, atlas, roof_tex, args, extras, tileset)
        print(f"[time] complete {time.time() - t_done:.1f} s")
    return 0


def write_matched_tint(out: Path, glb: "GlbRecorder", atlas: TextureAtlas, args, extras: dict, tileset: dict):
    """The measured mesh again with the walls whose depth came from the
    frames (``matched`` bands, kyfromabove_3d.wall_match) tinted cyan for the
    viewer: ``building_matched.glb`` and ``tileset_matched.json``. Measured,
    not invented -- product 2 -- but from the frames, not the LiDAR, and
    shown as such. A face texture's rows run top-down over its ``z`` range."""
    images = []
    for img, meta in zip(atlas.images, atlas.meta):
        img = np.asarray(img)
        if not meta.get("matched"):
            images.append(img)
            continue
        z0, z1 = meta["z"]
        H = img.shape[0]
        out_img = img.astype(np.float64).copy()
        for zb0, zb1 in meta["matched"]:
            r0 = int(np.clip(np.floor((z1 - zb1) / max(z1 - z0, 1e-6) * H), 0, H))
            r1 = int(np.clip(np.ceil((z1 - zb0) / max(z1 - z0, 1e-6) * H), 0, H))
            out_img[r0:r1] = 0.45 * out_img[r0:r1] + 0.55 * np.array([40.0, 210.0, 230.0])      # cyan: matched
        images.append(np.clip(out_img, 0, 255).astype(np.uint8))
    a = TextureAtlas(atlas.page_px, atlas.gap)
    for im in images:
        a.add(im)
    w = GlbWriter(args.texture)
    glb.replay(w, {f"walls_{k}": pg for k, pg in enumerate(a.pack())})
    w.write(out / "building_matched.glb", extras)
    ts = json.loads(json.dumps(tileset))
    ts["root"]["content"]["uri"] = "building_matched.glb"
    (out / "tileset_matched.json").write_text(json.dumps(ts, indent=1), encoding="utf-8")
    print(f"matched walls shown -> {out / 'building_matched.glb'} ({extras['matched']['faces']} face(s), "
          f"{extras['matched']['area_m2']:.0f} m2)")


def write_completed(b: Path, glb: "GlbRecorder", atlas: TextureAtlas, roof_tex, args, extras: dict, tileset: dict):
    """The completed product (product 3): the measured mesh again, call for
    call, with every texel no frame saw filled from the building's own
    facade (kyfromabove_3d.facade_fill: its twin, its rhythm, patches of
    itself) -- ``3dtiles_complete/building.glb``, and ``building_tint.glb``
    with the invented texels tinted for the viewer. The wall textures and
    their sources go to ``fill_faces.npz`` for the held-out test."""
    from kyfromabove_3d.facade_fill import KNOWN, NAMES, POLICY, complete_textures, patch_fill, shares

    out = b / "3dtiles_complete"
    out.mkdir(parents=True, exist_ok=True)
    rep = b / "mirror_report.json"
    twins = {int(f): int(r["from_face"]) for f, r in json.loads(rep.read_text(encoding="utf-8")).items()
             if r.get("symmetric")} if rep.exists() else {}
    done = complete_textures(atlas.images, atlas.meta, args.texel_m, twins)
    roof = None
    if roof_tex is not None and getattr(roof_tex, "source", None) is not None and "patch" in POLICY["roof"]:
        roof = patch_fill(np.asarray(roof_tex), roof_tex.source, args.texel_m)
    fill = shares([sv for _, sv in done] + ([roof[1]] if roof else []))

    def pages_of(images):
        a = TextureAtlas(atlas.page_px, atlas.gap)
        for im in images:
            a.add(im)
        return a.pack() if images else []

    def tint(img, src):
        if src is None:
            return img
        img = np.asarray(img).astype(np.float64).copy()
        inv = ~np.isin(src, KNOWN)
        img[inv] = 0.45 * img[inv] + 0.55 * np.array([230.0, 40.0, 200.0])      # magenta: invented
        return np.clip(img, 0, 255).astype(np.uint8)

    for name, images, roof_img in (
            ("building.glb", [im for im, _ in done], roof[0] if roof else None),
            ("building_tint.glb", [tint(im, sv) for im, sv in done], tint(*roof) if roof else None)):
        swap = {f"walls_{k}": pg for k, pg in enumerate(pages_of(images))}
        if roof_img is not None:
            swap["roof"] = roof_img
        # the completed pages kept lossless too (--dump-textures), for the
        # batched tiles to repack; not the tinted copy, which is for review
        w = GlbWriter(args.texture, dump_dir=(out / "textures") if args.dump_textures and name == "building.glb"
                      else None)
        glb.replay(w, swap)
        w.write(out / name, {**extras, "product": "completed", "fill": fill})
    for name, uri in (("tileset.json", "building.glb"), ("tileset_tint.json", "building_tint.glb")):
        ts = json.loads(json.dumps(tileset))
        ts["root"]["content"]["uri"] = uri
        ts["extras"].update({"product": "completed", "fill": fill})
        (out / name).write_text(json.dumps(ts, indent=1), encoding="utf-8")
    images, metas = list(atlas.images), list(atlas.meta)
    if roof_tex is not None and getattr(roof_tex, "source", None) is not None:
        images.append(roof_tex); metas.append({"kind": "roof"})
    walls = {f"img_{i}": np.asarray(im) for i, im in enumerate(images) if getattr(im, "source", None) is not None}
    walls.update({f"src_{i}": im.source for i, im in enumerate(images) if getattr(im, "source", None) is not None})
    np.savez_compressed(out / "fill_faces.npz", meta=json.dumps(metas), texel_m=args.texel_m,
                        twins=json.dumps(twins), **walls)
    print(f"completed -> {out / 'building.glb'}: " + ", ".join(f"{NAMES[k]} {fill[NAMES[k]]:.0%}" for k in sorted(NAMES)))


if __name__ == "__main__":
    sys.exit(main())
