"""Wall planes from the outline: faces, verified and curved walls, prism faces, their
files, and the outline's corners squared up."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .surface import FACADE_CAPTURE_PER_M2


@dataclass
class FacadeFace:
    """One wall of a building, and whether the LiDAR reached it.

    A facade is not one surface. **A wall faces one way**, and an aerial scanner
    flying one direction over it reaches some faces and not others -- measured on
    400 West Market, only 30% of camera-anchor pairs have the camera on the
    visible side at all. A single whole-building return density averages the
    face the aircraft flew along with the face it never saw, and reports a
    building as captured when half of it is missing.

    ``bearing_deg`` is the compass direction the wall looks towards, so a run of
    uncaptured faces sharing a bearing is a flight-direction gap rather than an
    occlusion.
    """

    index: int
    length_m: float
    area_m2: float
    bearing_deg: float
    n_returns: int
    returns_per_m2: float
    captured: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "face": self.index, "length_m": self.length_m,
            "area_m2": self.area_m2, "bearing_deg": self.bearing_deg,
            "n_returns": self.n_returns,
            "returns_per_m2": self.returns_per_m2, "captured": self.captured,
        }


def facade_faces(
    vertices_m: np.ndarray,
    height_m: float,
    wall_xy: Optional[np.ndarray] = None,
    min_length_m: float = 3.0,
    max_offset_m: float = 2.0,
    capture_per_m2: float = FACADE_CAPTURE_PER_M2,
) -> List[FacadeFace]:
    """Split a footprint into walls and count the returns that reached each.

    ``vertices_m`` is the footprint's exterior ring, ``(N, 2)`` in metres, closed
    or not; winding is fixed internally so the bearings come out pointing away
    from the building whichever way the ring was written. ``wall_xy`` is the
    ``(M, 2)`` plan position of the wall-zone returns, from
    :func:`split_surface_zones`.

    Each return is assigned to its nearest wall segment and dropped if it is
    further than ``max_offset_m`` from every one -- an Overture footprint is
    ML-derived and sits a metre or two off, so a return from the real wall lands
    just inside or outside the drawn one.

    Segments under ``min_length_m`` are skipped. They are mostly the small steps
    a traced footprint puts on a corner, and a face two metres wide gets a
    return density dominated by which side of the corner a point was assigned
    to.
    """
    v = np.asarray(vertices_m, dtype=np.float64)[:, :2]
    if len(v) > 1 and np.allclose(v[0], v[-1]):
        v = v[:-1]
    if len(v) < 3 or height_m <= 0:
        return []

    # Force counter-clockwise, so the outward side of a segment is consistently
    # to its right and the bearings mean something.
    if np.sum((np.roll(v[:, 0], -1) - v[:, 0]) * (np.roll(v[:, 1], -1) + v[:, 1])) > 0:
        v = v[::-1]

    p0, p1 = v, np.roll(v, -1, axis=0)
    seg = p1 - p0
    length = np.hypot(seg[:, 0], seg[:, 1])
    keep = length >= min_length_m
    if not keep.any():
        return []

    nearest = None
    if wall_xy is not None and len(wall_xy):
        w = np.asarray(wall_xy, dtype=np.float64)[:, :2]
        # Perpendicular distance to each segment, clamped to its extent so a
        # point beyond an end is measured to the end rather than to the
        # infinite line.
        d = w[:, None, :] - p0[None, :, :]
        denom = np.maximum((seg ** 2).sum(axis=1), 1e-9)
        t = np.clip((d * seg[None, :, :]).sum(axis=2) / denom, 0.0, 1.0)
        closest = p0[None, :, :] + t[:, :, None] * seg[None, :, :]
        dist = np.linalg.norm(w[:, None, :] - closest, axis=2)
        dist[:, ~keep] = np.inf
        nearest = np.where(dist.min(axis=1) <= max_offset_m, dist.argmin(axis=1), -1)

    faces: List[FacadeFace] = []
    for i in np.flatnonzero(keep):
        # Outward normal of a CCW ring is the segment turned clockwise.
        nx, ny = seg[i, 1] / length[i], -seg[i, 0] / length[i]
        area = float(length[i] * height_m)
        n = int((nearest == i).sum()) if nearest is not None else 0
        per_m2 = n / area if area > 0 else 0.0
        faces.append(FacadeFace(
            index=int(i),
            length_m=float(length[i]),
            area_m2=area,
            bearing_deg=float(np.degrees(np.arctan2(nx, ny)) % 360.0),
            n_returns=n,
            returns_per_m2=per_m2,
            captured=bool(per_m2 >= capture_per_m2),
        ))
    return faces


@dataclass
class WallPlane:
    """One wall of a building as a vertical plane: the footprint edge from
    ``p0`` to ``p1`` extruded from ``z_lo`` to ``z_hi``, facing ``normal``.

    On a curtain-wall tower the LiDAR reaches the wall thinly -- PNC Tower's
    walls hold 0.57 returns per square metre, one every 1.3 m -- so a camera
    ray looking for the nearest return within a metre usually finds none, and
    the plane-fit tiles that the aerial pass colours cannot be fitted at all
    (2,710 of its 2,727 wall cells). But the wall *is* a plane: the footprint
    says where, the roof plate says how high. This is that plane, so a ray can
    be lifted onto it exactly, every ray, with no return needed.
    """

    face: int
    p0: np.ndarray            # (2,) metres, footprint edge start
    p1: np.ndarray            # (2,) metres, footprint edge end
    normal: np.ndarray        # (2,) unit, pointing away from the building
    z_lo: float
    z_hi: float

    @property
    def length_m(self) -> float:
        return float(np.hypot(*(self.p1 - self.p0)))

    def local(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(along from p0, offset outward) of plan positions ``xy``."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        seg = self.p1 - self.p0
        rel = xy - self.p0[None, :]
        return rel @ (seg / max(self.length_m, 1e-9)), rel @ self.normal

    def world(self, along, off) -> np.ndarray:
        """Plan positions from (along, offset outward)."""
        along = np.asarray(along, dtype=np.float64).reshape(-1)
        off = np.asarray(off, dtype=np.float64).reshape(-1)
        e1 = (self.p1 - self.p0) / max(self.length_m, 1e-9)
        return self.p0[None, :] + along[:, None] * e1[None, :] + off[:, None] * self.normal[None, :]

    def world_line(self, step_m: float = 1.0) -> np.ndarray:
        """The wall's line in plan, as points from p0 to p1."""
        return np.vstack([self.p0, self.p1])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "face": int(self.face),
            "p0": self.p0.tolist(),
            "p1": self.p1.tolist(),
            "normal": self.normal.tolist(),
            "z_lo": float(self.z_lo),
            "z_hi": float(self.z_hi),
            "verified": False,
        }


@dataclass
class VerifiedWallPlane(WallPlane):
    """A WallPlane whose location and planarity are verified by fitted LiDAR tiles,
    or inferred via prism completion from adjacent verified planar walls.

    Extends WallPlane with tile statistics, evidence-based z_top, and inference flag.
    """

    n_tiles: int = 0
    rms_m: float = 0.0
    max_normal_dev_deg: float = 0.0
    z_top: float = 0.0
    offset_m: float = 0.0
    yaw_dev_deg: float = 0.0
    inferred: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "verified": True,
            "inferred": bool(self.inferred),
            "n_tiles": int(self.n_tiles),
            "rms_m": float(self.rms_m),
            "max_normal_dev_deg": float(self.max_normal_dev_deg),
            "z_top": float(self.z_top),
            "offset_m": float(self.offset_m),
            "yaw_dev_deg": float(self.yaw_dev_deg),
        })
        return d


@dataclass
class ArcWall(WallPlane):
    """A curved wall as one face: a circular arc through a run of the
    footprint's short edges, handled in an **unrolled** frame.

    Overture draws a curve as a chain of edges under 3 m, each too short
    to hold a profile of its own, so a curved wall came out as a fan of
    flat slats with a seam at each. Here the run is one face. Its plane
    frame is the arc unrolled flat: ``p0 = (0, 0)``, ``p1 = (L, 0)`` with
    ``L`` the arc length, ``normal = (0, -1)`` (outward is to the right of
    travel, as on every other face), so the profile, the bands, the
    texture and the wall rectangles are computed by the same code as a
    straight face. :meth:`roll` puts finished points back on the arc and
    :meth:`unroll` brings returns into the frame; ``x`` is distance along
    the arc from its first vertex, ``y`` is ``-`` the outward offset.

    ``turn`` is the signed angle the arc sweeps (radians; positive when
    the angle about the centre grows along the wall), ``convex`` whether
    outward points away from the centre (a bulge) or towards it (a dent).
    """

    centre: np.ndarray = field(default_factory=lambda: np.zeros(2))
    radius: float = 1.0
    theta0: float = 0.0
    turn: float = 0.0
    convex: bool = True
    first_edge: int = 0
    last_edge: int = 0
    rms_m: float = 0.0

    @classmethod
    def make(cls, face: int, centre, radius: float, theta0: float, turn: float, convex: bool,
             first_edge: int, last_edge: int, z_lo: float, z_hi: float, rms_m: float = 0.0) -> "ArcWall":
        L = float(abs(turn) * radius)
        return cls(face=int(face), p0=np.zeros(2), p1=np.array([L, 0.0]), normal=np.array([0.0, -1.0]),
                   z_lo=float(z_lo), z_hi=float(z_hi), centre=np.asarray(centre, dtype=np.float64),
                   radius=float(radius), theta0=float(theta0), turn=float(turn), convex=bool(convex),
                   first_edge=int(first_edge), last_edge=int(last_edge), rms_m=float(rms_m))

    @property
    def direction(self) -> float:
        return 1.0 if self.turn >= 0 else -1.0

    @property
    def out_sign(self) -> float:
        return 1.0 if self.convex else -1.0

    def theta(self, along) -> np.ndarray:
        return self.theta0 + self.direction * np.asarray(along, dtype=np.float64) / self.radius

    def basis(self, along):
        """Unit tangent (the unrolled +x) and outward radial at ``along``."""
        th = self.theta(along).reshape(-1)
        radial = np.column_stack([np.cos(th), np.sin(th)])
        tangent = self.direction * np.column_stack([-np.sin(th), np.cos(th)])
        return tangent, self.out_sign * radial

    def unroll(self, xy: np.ndarray) -> np.ndarray:
        """Plan positions -> the unrolled frame ``(x = along, y = -outward)``."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        rel = xy - self.centre[None, :]
        r = np.hypot(rel[:, 0], rel[:, 1])
        th = np.arctan2(rel[:, 1], rel[:, 0])
        # the angle from the arc's start, wound so the arc itself is [0, |turn|]
        # and a little before the start reads negative rather than a lap later
        margin = min(np.pi / 4, 0.5 * (2 * np.pi - abs(self.turn)))
        a = (self.direction * (th - self.theta0) + margin) % (2 * np.pi) - margin
        d = self.out_sign * (r - self.radius)
        return np.column_stack([a * self.radius, -d])

    def roll(self, sy: np.ndarray) -> np.ndarray:
        """The unrolled frame ``(x, y)`` -> plan positions on the arc."""
        sy = np.asarray(sy, dtype=np.float64).reshape(-1, 2)
        _, radial_out = self.basis(sy[:, 0])
        return self.centre[None, :] + self.radius * self.out_sign * radial_out + (-sy[:, 1])[:, None] * radial_out

    def local(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u = self.unroll(xy)
        return u[:, 0], -u[:, 1]

    def world(self, along, off) -> np.ndarray:
        along = np.asarray(along, dtype=np.float64).reshape(-1)
        off = np.asarray(off, dtype=np.float64).reshape(-1)
        return self.roll(np.column_stack([along, -off]))

    def world_line(self, step_m: float = 1.0) -> np.ndarray:
        n = max(2, int(np.ceil(self.length_m / step_m)) + 1)
        return self.world(np.linspace(0.0, self.length_m, n), np.zeros(n))

    def to_world_3d(self, pts: np.ndarray, normals: Optional[np.ndarray] = None):
        """Unrolled ``(x, y, z)`` points (and normals) -> world."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        xy = self.roll(pts[:, :2])
        out = np.column_stack([xy, pts[:, 2]])
        if normals is None:
            return out
        t, r_out = self.basis(pts[:, 0])
        nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        nxy = nrm[:, :1] * t + nrm[:, 1:2] * (-r_out)
        return out, np.column_stack([nxy, nrm[:, 2]])

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({"kind": "arc", "centre": self.centre.tolist(), "radius": float(self.radius),
                  "theta0": float(self.theta0), "turn": float(self.turn), "convex": bool(self.convex),
                  "first_edge": int(self.first_edge), "last_edge": int(self.last_edge), "rms_m": float(self.rms_m)})
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArcWall":
        return cls.make(d["face"], d["centre"], d["radius"], d["theta0"], d["turn"], d["convex"],
                        d["first_edge"], d["last_edge"], d["z_lo"], d["z_hi"], d.get("rms_m", 0.0))


def _ccw_ring(vertices_m: np.ndarray) -> np.ndarray:
    """The ring open and counter-clockwise, as ``facade_faces`` and
    ``wall_planes`` see it, so edge indices agree everywhere."""
    v = np.asarray(vertices_m, dtype=np.float64)[:, :2]
    if len(v) > 1 and np.allclose(v[0], v[-1]):
        v = v[:-1]
    if len(v) >= 3 and np.sum((np.roll(v[:, 0], -1) - v[:, 0]) * (np.roll(v[:, 1], -1) + v[:, 1])) > 0:
        v = v[::-1]
    return v


def _fit_circle(pts: np.ndarray):
    """Algebraic least-squares circle: (centre, radius, rms of the radial residual)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    bvec = -(x * x + y * y)
    (a, b, c), *_ = np.linalg.lstsq(A, bvec, rcond=None)
    cx, cy = -a / 2, -b / 2
    r2 = cx * cx + cy * cy - c
    if r2 <= 0:
        return None
    R = float(np.sqrt(r2))
    res = np.hypot(x - cx, y - cy) - R
    return np.array([cx, cy]), R, float(np.sqrt((res ** 2).mean()))


def arc_faces(vertices_m: np.ndarray, z_lo: float, z_hi: float, min_edges: int = 4, max_edge_m: float = 4.0,
              turn_deg: Tuple[float, float] = (3.0, 35.0), min_total_deg: float = 35.0,
              max_rms_m: float = 0.3, max_radius_m: float = 200.0, min_length_m: float = 6.0) -> List[ArcWall]:
    """The curved walls of a footprint: runs of at least ``min_edges``
    consecutive edges no longer than ``max_edge_m`` that turn the same way
    by ``turn_deg`` at every vertex and by ``min_total_deg`` in all, whose
    vertices fit a circle within ``max_rms_m`` and whose arc is at least
    ``min_length_m`` long (a rounded corner of a metre or two stays a
    corner). A run of every edge is one full circle. Each arc's ``face`` is
    its first edge's index."""
    v = _ccw_ring(vertices_m)
    n = len(v)
    if n < min_edges + 1 or z_hi <= z_lo:
        return []
    seg = np.roll(v, -1, axis=0) - v
    length = np.hypot(seg[:, 0], seg[:, 1])
    heading = np.arctan2(seg[:, 1], seg[:, 0])
    # turn[i]: at the vertex between edge i and edge i+1
    turn = (np.roll(heading, -1) - heading + np.pi) % (2 * np.pi) - np.pi
    lo, hi = np.radians(turn_deg[0]), np.radians(turn_deg[1])
    short = length <= max_edge_m
    # joint i: edges i and i+1 may belong to one arc
    joint = short & np.roll(short, -1) & (np.abs(turn) >= lo) & (np.abs(turn) <= hi)
    same = np.sign(turn) == np.sign(np.roll(turn, 1))            # turn[i] agrees with turn[i-1]
    runs: List[List[int]] = []
    if joint.all() and same.all():
        runs = [list(range(n))]                                    # a full circle
    else:
        start = int(np.flatnonzero(~joint)[0]) + 1 if (~joint).any() else 0
        cur = [start % n]
        for k in range(1, n):
            i = (start + k) % n
            prev = (i - 1) % n
            if joint[prev] and (len(cur) == 1 or same[prev]):
                cur.append(i)
            else:
                runs.append(cur); cur = [i]
        runs.append(cur)
    arcs: List[ArcWall] = []
    for run in runs:
        if len(run) < min_edges:
            continue
        total = float(sum(turn[i] for i in run[:-1]))
        if abs(total) < np.radians(min_total_deg):
            continue
        verts = np.array([v[i] for i in run] + [v[(run[-1] + 1) % n]])
        full = len(run) == n
        fit = _fit_circle(verts[:-1] if full else verts)
        if fit is None:
            continue
        centre, R, rms = fit
        if rms > max_rms_m or R > max_radius_m or R < 0.5 or R * abs(total) < min_length_m:
            continue
        rel = verts - centre[None, :]
        th = np.arctan2(rel[:, 1], rel[:, 0])
        dth = (np.diff(th) + np.pi) % (2 * np.pi) - np.pi
        sweep = float(dth.sum())
        if full:
            sweep = float(np.sign(sweep) * 2 * np.pi)
        mid = run[len(run) // 2]
        out_n = np.array([seg[mid, 1], -seg[mid, 0]])              # right of travel: outward on a CCW ring
        convex = bool(out_n @ (v[mid] - centre) > 0)
        arcs.append(ArcWall.make(face=run[0], centre=centre, radius=R, theta0=float(th[0]), turn=sweep,
                                 convex=convex, first_edge=run[0], last_edge=run[-1], z_lo=z_lo, z_hi=z_hi, rms_m=rms))
    return arcs


def verify_face_plane(
    plane: WallPlane,
    tile_centroids: np.ndarray,
    tile_normals: np.ndarray,
    max_offset_rms_m: float = 0.30,
    max_normal_dev_deg: float = 3.0,
    min_tiles: int = 2,
    tile_m: float = 12.0,
    roof_z: Optional[float] = None,
) -> Optional[VerifiedWallPlane]:
    """Verify whether a face's fitted LiDAR tiles define a single vertical plane.

    Fits a vertical plane (offset and yaw relative to the nominal footprint edge)
    to the tile centroids via least squares. If at least ``min_tiles`` exist,
    every tile centroid is within ``max_offset_rms_m`` (both RMS and max residual),
    and every tile normal is within ``max_normal_dev_deg`` of the fitted plane's
    outward normal, the face is declared planar-verified.

    Returns:
        VerifiedWallPlane with refined p0, p1, and normal, extruded from
        z_lo to z_top (highest tile top, capped at roof_z), or None if unverified.
    """
    c = np.asarray(tile_centroids, dtype=np.float64)
    n = np.asarray(tile_normals, dtype=np.float64)
    if len(c) < min_tiles or len(c) != len(n):
        return None

    p0, p1 = plane.p0, plane.p1
    seg = p1 - p0
    L = float(np.hypot(seg[0], seg[1]))
    if L < 1e-6:
        return None
    e1 = seg / L
    n_out = plane.normal

    # Project centroids into nominal edge frame: s along edge, d along normal
    rel = c[:, :2] - p0
    s = rel @ e1
    d = rel @ n_out

    # Fit line d(s) = d_0 + k * s
    if np.ptp(s) < 1.0:
        # Tiles stacked vertically at roughly the same along-wall position; no yaw
        k = 0.0
        d_0 = float(np.mean(d))
    else:
        k, d_0 = np.polyfit(s, d, deg=1)
        k = float(k)
        d_0 = float(d_0)

    yaw_rad = np.arctan(k)
    yaw_deg = float(np.degrees(np.abs(yaw_rad)))
    if yaw_deg > max_normal_dev_deg:
        return None

    # Centroid residuals to the fitted line
    d_fit = d_0 + k * s
    residuals = d - d_fit
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    max_res = float(np.max(np.abs(residuals)))
    if rms > max_offset_rms_m or max_res > max_offset_rms_m:
        return None

    # Fitted outward normal in 2D
    dir_line = e1 + k * n_out
    dir_line /= np.linalg.norm(dir_line)
    n_fitted_2d = np.array([dir_line[1], -dir_line[0]], dtype=np.float64)
    if n_fitted_2d @ n_out < 0:
        n_fitted_2d = -n_fitted_2d
    n_fitted_2d /= np.linalg.norm(n_fitted_2d)

    # Check tile normal deviations (in 3D). One tile among many leaning a
    # degree past the limit is a mullion or a canopy in that tile, not a
    # different wall: up to a quarter of the tiles may miss the limit as
    # long as the rest hold it. PNC Tower's west face lost its verification
    # -- and its two short faces their prism completion -- when dropping
    # 185 noise returns nudged one tile of twelve from 3.4 to 4.4 degrees.
    n_plane_3d = np.array([n_fitted_2d[0], n_fitted_2d[1], 0.0], dtype=np.float64)
    dots = np.clip(n @ n_plane_3d, -1.0, 1.0)
    normal_devs_deg = np.degrees(np.arccos(dots))
    within = normal_devs_deg <= max_normal_dev_deg
    if within.sum() < max(min_tiles, int(np.ceil(0.75 * len(n)))):
        return None
    max_normal_dev = float(np.max(normal_devs_deg[within]))

    # Refine p0 and p1 along the fitted line
    p0_ref = p0 + d_0 * n_out
    p1_ref = p0 + L * e1 + (d_0 + k * L) * n_out

    highest_z = float(np.max(c[:, 2]))
    top_cand = highest_z + tile_m / 2.0
    z_top = float(min(roof_z if roof_z is not None else plane.z_hi, top_cand))

    return VerifiedWallPlane(
        face=plane.face,
        p0=p0_ref,
        p1=p1_ref,
        normal=n_fitted_2d,
        z_lo=plane.z_lo,
        z_hi=z_top,
        n_tiles=len(c),
        rms_m=rms,
        max_normal_dev_deg=max_normal_dev,
        z_top=z_top,
        offset_m=d_0,
        yaw_dev_deg=yaw_deg,
    )


def infer_prism_faces(
    nominal_planes: List[WallPlane],
    verified_planes: Dict[int, VerifiedWallPlane],
    min_perimeter_ratio: float = 0.60,
    roof_z: Optional[float] = None,
) -> Dict[int, VerifiedWallPlane]:
    """Infer unverified connecting facade faces for buildings with verified planar curtain walls.

    When major facade faces of a building are verified planar curtain walls
    (accounting for at least ``min_perimeter_ratio`` of the total nominal footprint perimeter),
    the building is inferred to be a vertical prism. The remaining unverified faces
    (typically narrow end walls where LiDAR returns are depleted by specular grazing reflections)
    are completed as planar prism faces bridging between adjacent verified planes,
    extruded up to the minimum z_top of the adjacent verified faces (capped at roof_z).

    Non-planar buildings (such as post-modern architecture with setbacks, arches, or loggias,
    e.g. Humana Building with 0% verified perimeter) will fail the perimeter evidence gate
    and return an empty dictionary, ensuring zero geometry is invented.

    Args:
        nominal_planes: list of nominal footprint WallPlanes in perimeter order.
        verified_planes: dict of directly verified WallPlanes (e.g. from verify_face_plane).
        min_perimeter_ratio: fraction of total perimeter that must be verified (default 0.60).
        roof_z: optional roof elevation cap.

    Returns:
        Dict mapping face index to inferred VerifiedWallPlane instances (with inferred=True).
    """
    if not verified_planes or not nominal_planes:
        return {}

    lengths = [float(np.hypot(p.p1[0] - p.p0[0], p.p1[1] - p.p0[1])) for p in nominal_planes]
    total_perimeter = sum(lengths)
    if total_perimeter <= 0:
        return {}

    verified_perimeter = sum(
        lengths[idx] for idx, p in enumerate(nominal_planes) if p.face in verified_planes
    )
    ratio = verified_perimeter / total_perimeter
    if ratio < min_perimeter_ratio:
        return {}

    n_faces = len(nominal_planes)
    face_to_idx = {p.face: idx for idx, p in enumerate(nominal_planes)}
    inferred: Dict[int, VerifiedWallPlane] = {}

    for p in nominal_planes:
        if p.face in verified_planes:
            continue

        curr_idx = face_to_idx[p.face]
        prev_p = nominal_planes[(curr_idx - 1) % n_faces]
        next_p = nominal_planes[(curr_idx + 1) % n_faces]

        # Snap endpoints to adjacent verified walls if available, closing the polygon
        if prev_p.face in verified_planes:
            p0 = verified_planes[prev_p.face].p1.copy()
        else:
            p0 = p.p0.copy()

        if next_p.face in verified_planes:
            p1 = verified_planes[next_p.face].p0.copy()
        else:
            p1 = p.p1.copy()

        # Compute outward normal perpendicular to the refined edge segment
        seg = p1 - p0
        L = float(np.hypot(seg[0], seg[1]))
        if L > 1e-6:
            norm_cand = np.array([seg[1], -seg[0]]) / L
            if norm_cand @ p.normal < 0:
                norm_cand = -norm_cand
            normal = norm_cand
        else:
            normal = p.normal.copy()

        # Extrusion height: min z_top of adjacent verified faces (or all verified if neither)
        has_prev = prev_p.face in verified_planes
        has_next = next_p.face in verified_planes
        if has_prev and has_next:
            z_cand = min(verified_planes[prev_p.face].z_top, verified_planes[next_p.face].z_top)
        elif has_prev:
            z_cand = verified_planes[prev_p.face].z_top
        elif has_next:
            z_cand = verified_planes[next_p.face].z_top
        else:
            z_cand = min(v.z_top for v in verified_planes.values())

        if roof_z is not None:
            z_top = float(min(roof_z, z_cand))
        else:
            z_top = float(min(p.z_hi, z_cand))

        inferred[p.face] = VerifiedWallPlane(
            face=p.face,
            p0=p0,
            p1=p1,
            normal=normal,
            z_lo=p.z_lo,
            z_hi=z_top,
            n_tiles=0,
            rms_m=0.0,
            max_normal_dev_deg=0.0,
            z_top=z_top,
            offset_m=0.0,
            yaw_dev_deg=0.0,
            inferred=True,
        )

    return inferred


def wall_planes(vertices_m: np.ndarray, z_lo: float, z_hi: float,
                min_length_m: float = 3.0, arcs: bool = False) -> List[WallPlane]:
    """The footprint's edges as vertical planes between ``z_lo`` and ``z_hi``.

    Uses the same winding fix and outward-normal convention as
    :func:`facade_faces`, so ``WallPlane.face`` matches ``FacadeFace.index``
    for the same ring, and the same ``min_length_m`` drops the small corner
    steps a traced footprint carries. With ``arcs`` the runs of short edges
    that :func:`arc_faces` reads as curved walls become one :class:`ArcWall`
    each (its ``face`` the first edge's index) in place of their edges.
    """
    v = _ccw_ring(vertices_m)
    if len(v) < 3 or z_hi <= z_lo:
        return []
    p0, p1 = v, np.roll(v, -1, axis=0)
    seg = p1 - p0
    length = np.hypot(seg[:, 0], seg[:, 1])
    planes: List[WallPlane] = []
    in_arc = set()
    arcs = arc_faces(v, z_lo, z_hi) if arcs else []
    for a in arcs:
        i = a.first_edge
        while True:
            in_arc.add(i)
            if i == a.last_edge:
                break
            i = (i + 1) % len(v)
    for i in np.flatnonzero(length >= min_length_m):
        if int(i) in in_arc:
            continue
        normal = np.array([seg[i, 1], -seg[i, 0]]) / length[i]
        planes.append(WallPlane(face=int(i), p0=p0[i].copy(), p1=p1[i].copy(),
                                normal=normal, z_lo=float(z_lo), z_hi=float(z_hi)))
    planes.extend(arcs)
    return sorted(planes, key=lambda pl: pl.face)


def save_wall_planes(path: Union[Path, str], planes: List[WallPlane]) -> None:
    """Save a list of WallPlane or VerifiedWallPlane instances to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {"planes": [pl.to_dict() for pl in planes]}
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_wall_planes(path: Union[Path, str]) -> List[WallPlane]:
    """Load a list of WallPlane / VerifiedWallPlane instances from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("planes", data if isinstance(data, list) else [])
    planes: List[WallPlane] = []
    for item in items:
        if item.get("kind") == "arc":
            planes.append(ArcWall.from_dict(item))
        elif item.get("verified", False):
            planes.append(VerifiedWallPlane(
                face=int(item["face"]),
                p0=np.array(item["p0"], dtype=np.float64),
                p1=np.array(item["p1"], dtype=np.float64),
                normal=np.array(item["normal"], dtype=np.float64),
                z_lo=float(item["z_lo"]),
                z_hi=float(item["z_hi"]),
                n_tiles=int(item.get("n_tiles", 0)),
                rms_m=float(item.get("rms_m", 0.0)),
                max_normal_dev_deg=float(item.get("max_normal_dev_deg", 0.0)),
                z_top=float(item.get("z_top", item["z_hi"])),
                offset_m=float(item.get("offset_m", 0.0)),
                yaw_dev_deg=float(item.get("yaw_dev_deg", 0.0)),
                inferred=bool(item.get("inferred", False)),
            ))
        else:
            planes.append(WallPlane(
                face=int(item["face"]),
                p0=np.array(item["p0"], dtype=np.float64),
                p1=np.array(item["p1"], dtype=np.float64),
                normal=np.array(item["normal"], dtype=np.float64),
                z_lo=float(item["z_lo"]),
                z_hi=float(item["z_hi"]),
            ))
    return planes


def rectify_footprint_corners(
    ring_m: np.ndarray,
    planes: List[WallPlane],
    max_corner_shift_m: float = 2.0,
    min_angle_deg: float = 15.0,
) -> Tuple[np.ndarray, List[WallPlane]]:
    """Rectify footprint corners by mathematically intersecting adjacent verified wall lines.

    For each corner between adjacent footprint edges i and (i+1) % N:
    If adjacent faces have verified or inferred planes, their 2D line equations
    n_i @ x = n_i @ p_0,i and n_{i+1} @ x = n_{i+1} @ p_0,{i+1} are solved for their
    exact intersection point P. If the angle between their outward normals is at
    least min_angle_deg and the solved intersection point is within max_corner_shift_m
    of the nominal corner vertex, the corner is updated to P.

    This aligns the footprint polygon with the physical LiDAR curtain-wall planes to
    millimeter accuracy, removing gaps and overhang offsets.

    Returns:
        (rectified_ring_m, updated_planes)
    """
    ring = np.asarray(ring_m, dtype=np.float64)[:, :2]
    if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    N = len(ring)
    if N < 3:
        return ring_m, planes

    face_to_plane = {p.face: p for p in planes}
    min_sin = float(np.sin(np.radians(min_angle_deg)))

    new_corners = ring.copy()
    for i in range(N):
        next_i = (i + 1) % N
        pA = face_to_plane.get(i)
        pB = face_to_plane.get(next_i)

        is_verified_A = isinstance(pA, VerifiedWallPlane) if pA else False
        is_verified_B = isinstance(pB, VerifiedWallPlane) if pB else False
        if not (is_verified_A or is_verified_B):
            continue

        # Line A
        if pA is not None:
            nA = np.asarray(pA.normal, dtype=np.float64)[:2]
            nA_len = np.linalg.norm(nA)
            if nA_len < 1e-6:
                continue
            nA = nA / nA_len
            cA = float(nA @ pA.p0[:2])
        else:
            segA = ring[(i + 1) % N] - ring[i]
            lenA = np.hypot(segA[0], segA[1])
            if lenA < 1e-6:
                continue
            nA = np.array([segA[1], -segA[0]]) / lenA
            cA = float(nA @ ring[i])

        # Line B
        if pB is not None:
            nB = np.asarray(pB.normal, dtype=np.float64)[:2]
            nB_len = np.linalg.norm(nB)
            if nB_len < 1e-6:
                continue
            nB = nB / nB_len
            cB = float(nB @ pB.p0[:2])
        else:
            segB = ring[(next_i + 1) % N] - ring[next_i]
            lenB = np.hypot(segB[0], segB[1])
            if lenB < 1e-6:
                continue
            nB = np.array([segB[1], -segB[0]]) / lenB
            cB = float(nB @ ring[next_i])

        M = np.array([nA, nB], dtype=np.float64)
        det = float(np.linalg.det(M))
        if abs(det) < min_sin:
            nominal_corner = ring[next_i]
            if is_verified_B:
                pt = nominal_corner + (cB - float(nB @ nominal_corner)) * nB
            elif is_verified_A:
                pt = nominal_corner + (cA - float(nA @ nominal_corner)) * nA
            else:
                continue
            shift = float(np.hypot(pt[0] - nominal_corner[0], pt[1] - nominal_corner[1]))
            if shift <= max_corner_shift_m:
                new_corners[next_i] = pt
            continue

        try:
            pt = np.linalg.solve(M, np.array([cA, cB], dtype=np.float64))
        except np.linalg.LinAlgError:
            continue

        nominal_corner = ring[next_i]
        shift = float(np.hypot(pt[0] - nominal_corner[0], pt[1] - nominal_corner[1]))
        if shift <= max_corner_shift_m:
            new_corners[next_i] = pt

    rectified_ring = np.vstack([new_corners, new_corners[:1]])

    updated_planes: List[WallPlane] = []
    for p in planes:
        face = p.face
        if face < N:
            c0 = new_corners[face]
            c1 = new_corners[(face + 1) % N]
            if isinstance(p, VerifiedWallPlane):
                updated_p = VerifiedWallPlane(
                    face=p.face,
                    p0=c0.copy(),
                    p1=c1.copy(),
                    normal=p.normal.copy(),
                    z_lo=p.z_lo,
                    z_hi=p.z_hi,
                    n_tiles=p.n_tiles,
                    rms_m=p.rms_m,
                    max_normal_dev_deg=p.max_normal_dev_deg,
                    z_top=p.z_top,
                    offset_m=p.offset_m,
                    yaw_dev_deg=p.yaw_dev_deg,
                    inferred=p.inferred,
                )
            else:
                updated_p = WallPlane(
                    face=p.face,
                    p0=c0.copy(),
                    p1=c1.copy(),
                    normal=p.normal.copy(),
                    z_lo=p.z_lo,
                    z_hi=p.z_hi,
                )
            updated_planes.append(updated_p)
        else:
            updated_planes.append(p)

    return rectified_ring, updated_planes
