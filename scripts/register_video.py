#!/usr/bin/env python3
"""The true camera path of a real video, from the LiDAR meshes this project built.

``score_trajectory.py`` scores a video model's camera path against a true
one; ``render_flyaround.py`` makes the truth for a synthetic video. A real
video (a drone clip over downtown, say) has no recorded path. This gets one
from the buildings in it:

    scene    the downtown tileset's meshes in one frame (metres east, north
             and up from the tileset's origin), saved as one .npz
    pnp      one frame's camera from points measured by hand: each a pixel
             and the same point on a mesh (a spire tip, a roof corner). With
             the lens free, its width is found too. Use points at different
             distances: from points all far off, a camera raised 20 m and
             tilted down 3 degrees fits nearly as well as the right one.
    refine   nudge a camera so the meshes' edges fall on the frame's edges
             (with its points, lightly, as an anchor). It moves a camera that
             is close; it cannot rescue one that is 20 m out.
    track    every frame's camera, from a solved one: corners on the
             buildings in the first frame, their 3D read off the meshes
             through the solved camera, followed frame to frame (Lucas-
             Kanade, checked backwards) and the camera solved from them in
             each frame (PnP, outliers dropped). As corners leave, new ones
             are seeded the same way through the frame's new camera. The
             hand-measured points are followed too and weighted heavily:
             their 3D is exact and never handed on, so they hold the path.
    truth    one path from a track forward from the first solved frame and
             one backward from the last: each frame leans on the nearer end,
             and where the two disagree is how far the truth can be trusted.
    overlay  the meshes' outlines drawn over the frames, for the true path
             or for a video model's path placed on two solved frames.

The cameras are OpenCV pinholes (x right, y down, z forward), ``R_cw`` the
world-to-camera rotation and ``C`` the centre. The truth is written as TUM
(camera to world), which ``score_trajectory.py --truth`` reads.

What it assumes: the lens is fixed (no zoom) and has no distortion worth
modelling; the meshes are where the buildings are (the LiDAR's, to a few
decimetres); the buildings fill enough of each frame to track. Water, sky
and streets are not in the meshes, so nothing is tracked on them.

Usage (the DX0001_003049 test, frames every second frame of the clip)::

    python scripts/register_video.py scene --tileset outputs/downtown --out data/derived/video/downtown_scene.npz
    python scripts/register_video.py pnp --scene S --frame F/000000.jpg --points P/points_A.json --out P/cam_A.json
    python scripts/register_video.py track --scene S --frames F --camera P/cam_A.json --start 0 --out P/track_fwd.json
    python scripts/register_video.py refine --scene S --frame F/000403.jpg --camera P/track_fwd.json@403 \\
        --points P/points_D.json --out P/cam_D.json
    python scripts/register_video.py track --scene S --frames F --camera P/cam_D.json --start 403 --backward --out P/track_bwd.json
    python scripts/register_video.py truth --forward P/track_fwd.json --backward P/track_bwd.json --fps 11.988 --out P/truth_tum.txt
    python scripts/register_video.py overlay --scene S --frames F --truth P/truth_tum.txt --camera P/cam_A.json --out P/overlay_truth.mp4
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LIGHT = np.array([0.35, -0.55, 0.75]) / np.linalg.norm([0.35, -0.55, 0.75])


# ------------------------------------------------------------------ the scene

def col_major(m: Sequence[float]) -> np.ndarray:
    return np.array(m, dtype=np.float64).reshape(4, 4).T


def gather_scene(tileset_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Every ``glb/<gers>.glb`` the tileset places, in the tileset's own
    frame: the child transforms applied, the root's (frame to Earth) undone.
    Returns vertices (N, 3), triangles (T, 3), each triangle's building
    index, and the building names."""
    from kyfromabove_3d.texture_check import read_glb_geometry
    ts = json.loads((tileset_dir / "tileset.json").read_text(encoding="utf-8"))
    V: List[np.ndarray] = []
    T: List[np.ndarray] = []
    B: List[np.ndarray] = []
    names: List[str] = []
    n = 0

    def walk(node, M):
        nonlocal n
        M = M @ col_major(node["transform"]) if node.get("transform") else M
        uri = node.get("content", {}).get("uri", "")
        if uri.startswith("glb/"):
            v, t = read_glb_geometry(tileset_dir / uri)
            if len(t):
                V.append((np.c_[v, np.ones(len(v))] @ M.T)[:, :3])
                T.append(t + n)
                B.append(np.full(len(t), len(names)))
                names.append(Path(uri).stem)
                n += len(v)
        for child in node.get("children", []):
            walk(child, M)

    walk(ts["root"], np.linalg.inv(col_major(ts["root"]["transform"])))
    return np.vstack(V), np.vstack(T), np.concatenate(B), names


def load_scene(path: Path):
    """The scene .npz as ``render_flyaround.Scene``, each face shaded by
    which way it faces (so walls meeting at a corner differ), no texture."""
    import render_flyaround as rf
    d = np.load(path)
    tris = d["V"].astype(np.float64)[d["T"]]
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    shade = 0.35 + 0.6 * np.abs(nrm @ LIGHT)
    colour = np.repeat((shade[:, None] * np.array([0.92, 0.90, 0.86]))[:, None, :], 3, axis=1)
    return rf.Scene(tris, np.zeros((len(tris), 3, 2)), colour, np.full(len(tris), -1), [])


def render(scene, R_cw: np.ndarray, C: np.ndarray, K: np.ndarray, W: int, H: int):
    """Shaded image and depth (inf where no mesh) through the camera."""
    import render_flyaround as rf
    return rf.render(scene, K, R_cw, -R_cw @ C, W, H, near=1.0, background=(150, 170, 200))


def silhouette(depth: np.ndarray, jump: float = 0.03) -> np.ndarray:
    """Where the depth jumps (relative) or the mesh ends: the outlines."""
    hit = np.isfinite(depth)
    d = np.where(hit, np.log(np.where(hit, depth, 1.0)), 9.0)
    return ((np.abs(np.diff(d, axis=0, prepend=d[:1])) > jump) | (np.abs(np.diff(d, axis=1, prepend=d[:, :1])) > jump)
            | (hit ^ np.roll(hit, 1, 0)))


# ------------------------------------------------------------------ cameras

def intrinsics(W: int, H: int, hfov_deg: float) -> np.ndarray:
    f = W / 2 / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0.0, (W - 1) / 2], [0.0, f, (H - 1) / 2], [0.0, 0.0, 1.0]])


def hfov(K: np.ndarray, W: int) -> float:
    return float(np.degrees(2 * np.arctan(W / 2 / K[0, 0])))


def project(K: np.ndarray, R_cw: np.ndarray, C: np.ndarray, X: np.ndarray) -> np.ndarray:
    pc = (np.asarray(X, float) - C) @ R_cw.T
    pr = pc @ K.T
    return pr[:, :2] / pr[:, 2:]


def camera_json(W: int, H: int, K: np.ndarray, R_cw: np.ndarray, C: np.ndarray, **extra) -> dict:
    return dict(width=W, height=H, K=K.tolist(), R_cw=np.asarray(R_cw).tolist(), C=np.asarray(C).tolist(),
                hfov_deg=hfov(K, W), **extra)


def read_camera(spec: str) -> dict:
    """A camera json, or ``track.json@i``: frame i of a track."""
    path, _, idx = spec.partition("@")
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if idx:
        p = next(q for q in d["poses"] if q["i"] == int(idx))
        return dict(width=d["width"], height=d["height"], K=d["K"], R_cw=p["R_cw"], C=p["C"], hfov_deg=hfov(np.array(d["K"]), d["width"]))
    return d


def read_points(path: Path) -> Dict[str, Tuple[List[float], List[float]]]:
    """``{name: {"xyz": [E, N, U], "px": [x, y]}}`` -> ``{name: (xyz, px)}``."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: (v["xyz"], v["px"]) for k, v in d.items()}


def solve_pnp(X: np.ndarray, x: np.ndarray, W: int, H: int, hfov_deg: Optional[float] = None,
              search=(25.0, 90.0, 0.25)):
    """Camera from 3D-2D pairs. With ``hfov_deg`` the lens is fixed; without,
    every width in ``search`` is tried and the one that fits best is kept.
    Returns K, R_cw, C and each point's error in pixels."""
    import cv2
    X = np.asarray(X, np.float64)
    x = np.asarray(x, np.float64)
    widths = [hfov_deg] if hfov_deg else np.arange(*search)
    best = None
    for w in widths:
        K = intrinsics(W, H, w)
        for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP):
            try:
                ok, rv, tv = cv2.solvePnP(X, x, K, None, flags=flag)
            except cv2.error:
                continue
            if not ok:
                continue
            rv, tv = cv2.solvePnPRefineLM(X, x, K, None, rv, tv)
            R = cv2.Rodrigues(rv)[0]
            C = (-R.T @ tv).ravel()
            if np.any((X - C) @ R[2] <= 0):                 # a point behind the camera
                continue
            e = np.linalg.norm(project(K, R, C, X) - x, axis=1)
            if best is None or e.mean() < best[3].mean():
                best = (K, R, C, e)
    if best is None:
        raise SystemExit("[FAIL] no camera fits these points")
    return best


# ------------------------------------------------------------------ edges

def frame_edges(gray: np.ndarray, ignore: Optional[Tuple[int, int, int, int]], cap: float = 15.0):
    """The frame's edges and the distance to the nearest one (capped)."""
    import cv2
    e = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), 1.2), 40, 100) > 0
    if ignore:
        x0, y0, x1, y1 = ignore
        e[y0:y1, x0:x1] = False
    return e, np.minimum(cv2.distanceTransform((~e).astype(np.uint8), cv2.DIST_L2, 3), cap)


def model_edges(img: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Outlines plus the creases where differently facing walls meet."""
    import cv2
    e = (cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 8, 20) > 0) | silhouette(depth, 0.02)
    e[:2] = e[-2:] = False
    e[:, :2] = e[:, -2:] = False
    return e


def refine(scene, cam: dict, gray: np.ndarray, points=None, W: int = 960, free_lens: bool = False,
           weight: float = 0.3, ignore=None, maxfev: int = 900):
    """Move the camera so the meshes' edges fall on the frame's: the mean
    distance from each mesh edge pixel to the nearest frame edge, over the
    nearest 80 % (what the meshes lack, or the frame shows that they do not,
    is left out), plus ``weight`` times the points' mean error."""
    import cv2
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation
    s = W / cam["width"]
    H = round(cam["height"] * s)
    K0 = np.array(cam["K"])
    f0, cx, cy = K0[0, 0] * s, (K0[0, 2] + .5) * s - .5, (K0[1, 2] + .5) * s - .5
    R0, C0 = np.array(cam["R_cw"]), np.array(cam["C"])
    X = np.array([v[0] for v in (points or {}).values()]).reshape(-1, 3)
    x = np.array([v[1] for v in (points or {}).values()]).reshape(-1, 2) * s
    small = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
    ign = tuple(int(round(v * s)) for v in ignore) if ignore else None
    _, dt = frame_edges(small, ign)

    def unpack(q):
        K = np.array([[q[6] if free_lens else f0, 0, cx], [0, q[6] if free_lens else f0, cy], [0, 0, 1]])
        return K, Rotation.from_rotvec(q[3:6]).as_matrix() @ R0, q[:3]

    def parts(q):
        K, R, C = unpack(q)
        img, depth = render(scene, R, C, K, W, H)
        me = model_edges(img, depth)
        if ign:
            me[ign[1]:ign[3], ign[0]:ign[2]] = False
        v = np.sort(dt[me])
        edge = float(v[: max(1, int(0.8 * len(v)))].mean()) if len(v) else 15.0
        lm = float(np.linalg.norm(project(K, R, C, X) - x, axis=1).mean()) if len(x) else 0.0
        return edge, lm

    def cost(q):
        edge, lm = parts(q)
        return edge + weight * lm

    q0 = np.r_[C0, 0.0, 0.0, 0.0, f0][: 7 if free_lens else 6]
    steps = np.diag(np.array([4, 4, 2, 0.004, 0.004, 0.003, 0.01 * f0])[: len(q0)])
    r = minimize(cost, q0, method="Powell",
                 options=dict(direc=steps, xtol=1e-3, ftol=1e-4, maxfev=maxfev))
    (e0, l0), (e1, l1) = parts(q0), parts(r.x)
    K, R, C = unpack(r.x)
    Kf = K.copy()
    Kf[:2, :2] /= s
    Kf[:2, 2] = (K[:2, 2] + .5) / s - .5
    return camera_json(cam["width"], cam["height"], Kf, R, C, refine=dict(
        width=W, edge_px_before=e0 / s, edge_px_after=e1 / s, points_px_before=l0 / s, points_px_after=l1 / s,
        moved_m=float(np.linalg.norm(C - C0)), renders=int(r.nfev)))


# ------------------------------------------------------------------ tracking

def seed(scene, gray, K, R, C, near_existing, want, ignore=None):
    """Corners on the meshes, away from outlines (where a pixel's depth is
    uncertain) and from corners already tracked, with their 3D read off the
    mesh through the camera."""
    import cv2
    H, W = gray.shape
    _, depth = render(scene, R, C, K, W, H)
    ok = np.isfinite(depth) & ~cv2.dilate(silhouette(depth, 0.01).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    if ignore:
        x0, y0, x1, y1 = ignore
        ok[y0:y1, x0:x1] = False
    ok[:8] = ok[-8:] = False
    ok[:, :8] = ok[:, -8:] = False
    mask = ok.astype(np.uint8) * 255
    for p in near_existing:
        cv2.circle(mask, (int(p[0]), int(p[1])), 14, 0, -1)
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=max(1, want), qualityLevel=0.01, minDistance=14, mask=mask, blockSize=7)
    if pts is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 3))
    pts = pts.reshape(-1, 2)
    return pts.astype(np.float32), backproject(K, R, C, pts, depth)


def backproject(K, R, C, pts, depth):
    """Pixels to 3D through the camera, at the depth (along the optical axis) the mesh has there."""
    z = depth[np.round(pts[:, 1]).astype(int), np.round(pts[:, 0]).astype(int)]
    ray = np.linalg.inv(K) @ np.c_[pts, np.ones(len(pts))].T
    return (R.T @ (ray * z)).T + C


def pose_from(K, rv, tv, X, p, LX, Lp, weight):
    """Least squares on the corners and the landmarks (each landmark
    ``weight`` times a corner), with a soft loss so a bad match cannot pull."""
    import cv2
    from scipy.optimize import least_squares

    def res(q):
        r = []
        for XX, pp, w in ((X, p, 1.0), (LX, Lp, weight)):
            if len(XX):
                r.append(w * (cv2.projectPoints(XX, q[:3], q[3:], K, None)[0].reshape(-1, 2) - pp).ravel())
        return np.concatenate(r)

    q = least_squares(res, np.r_[rv.ravel(), tv.ravel()], loss="soft_l1", f_scale=1.5).x
    return q[:3].reshape(3, 1), q[3:].reshape(3, 1)


def track(scene, frames: Sequence[str], cam: dict, start: int, step: int, points=None, ignore=None,
          keep: int = 350, total: int = 600, reseed_every: int = 25, landmark_weight: float = 7.0, log=print) -> dict:
    """Every frame's camera from ``start`` to the end (``step`` 1) or the
    beginning (-1); see the module's ``track``."""
    import cv2
    K = np.array(cam["K"])
    R, C = np.array(cam["R_cw"]), np.array(cam["C"])
    order = list(range(start, len(frames))) if step > 0 else list(range(start, -1, -1))
    prev = cv2.imread(frames[order[0]], cv2.IMREAD_GRAYSCALE)
    H, W = prev.shape
    if (W, H) != (cam["width"], cam["height"]):
        raise SystemExit(f"[FAIL] the frames are {W}x{H}, the camera {cam['width']}x{cam['height']}")
    names = list(points or {})
    LX = np.array([points[k][0] for k in names], float).reshape(-1, 3)
    Lp = np.array([points[k][1] for k in names], np.float32).reshape(-1, 2)
    p, X = seed(scene, prev, K, R, C, [], total, ignore)
    rv, tv = cv2.Rodrigues(R)[0], (-R @ C).reshape(3, 1)
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    lkL = dict(winSize=(31, 31), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))

    def follow(img0, img1, pts, params):
        q, st, _ = cv2.calcOpticalFlowPyrLK(img0, img1, pts, None, **params)
        b, st2, _ = cv2.calcOpticalFlowPyrLK(img1, img0, q, None, **params)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(b - pts, axis=1) < 0.7)
        good &= (q[:, 0] > 4) & (q[:, 0] < W - 4) & (q[:, 1] > 4) & (q[:, 1] < H - 4)
        if ignore:
            x0, y0, x1, y1 = ignore
            good &= ~((q[:, 0] > x0) & (q[:, 0] < x1) & (q[:, 1] > y0) & (q[:, 1] < y1))
        return q, good

    poses = [dict(i=order[0], R_cw=R.tolist(), C=C.tolist(), corners=len(p), rms_px=0.0,
                  landmarks={k: Lp[j].tolist() for j, k in enumerate(names)})]
    t0 = time.time()
    for n_done, i in enumerate(order[1:], 1):
        gray = cv2.imread(frames[i], cv2.IMREAD_GRAYSCALE)
        q, good = follow(prev, gray, p, lk)
        p, X = q[good], X[good]
        if len(Lp):
            lq, lg = follow(prev, gray, Lp, lkL)
            Lp, LX, names = lq[lg], LX[lg], [k for k, g in zip(names, lg) if g]
        if len(p) < 6:
            raise SystemExit(f"[FAIL] frame {i}: only {len(p)} corners left to solve the camera from")
        _, rv, tv, inl = cv2.solvePnPRansac(X, p, K, None, rv.copy(), tv.copy(), useExtrinsicGuess=True,
                                             reprojectionError=2.5, iterationsCount=200, flags=cv2.SOLVEPNP_ITERATIVE)
        inl = inl.ravel() if inl is not None else np.arange(len(p))
        rv, tv = pose_from(K, rv, tv, X[inl], p[inl], LX, Lp, landmark_weight)
        R = cv2.Rodrigues(rv)[0]
        C = (-R.T @ tv).ravel()
        e = np.linalg.norm(project(K, R, C, X[inl]) - p[inl], axis=1)
        p, X = p[inl], X[inl]
        poses.append(dict(i=i, R_cw=R.tolist(), C=C.tolist(), corners=int(len(inl)), rms_px=float(np.sqrt((e ** 2).mean())),
                          landmarks={k: Lp[j].tolist() for j, k in enumerate(names)}))
        if len(p) < keep or n_done % reseed_every == 0:
            p2, X2 = seed(scene, gray, K, R, C, p, total - len(p), ignore)
            p, X = np.vstack([p, p2]).astype(np.float32), np.vstack([X, X2])
        prev = gray
        if n_done % 50 == 0:
            log(f"  frame {i}: {len(inl)} corners, {poses[-1]['rms_px']:.2f} px rms, {len(Lp)} landmarks, "
                f"camera at E {C[0]:.1f} N {C[1]:.1f} up {C[2]:.1f} ({time.time() - t0:.0f} s)")
    poses.sort(key=lambda d: d["i"])
    return dict(width=W, height=H, K=K.tolist(), start=start, step=step, frames=[str(f) for f in frames], poses=poses)


# ------------------------------------------------------------------ the truth

def blend(forward: dict, backward: dict):
    """Each frame's camera from the forward track (from the first solved
    frame) and the backward one (from the last), weighted by nearness to each
    end: position linearly, orientation by slerp. Returns the camera-to-world
    rotations, the centres, and where the two tracks disagree (metres,
    degrees) frame by frame."""
    from scipy.spatial.transform import Rotation, Slerp
    f = {p["i"]: p for p in forward["poses"]}
    b = {p["i"]: p for p in backward["poses"]}
    idx = sorted(set(f) & set(b))
    if not idx:
        raise SystemExit("[FAIL] the two tracks share no frame")
    a, z = idx[0], idx[-1]
    Rwc, C, gap_m, gap_deg = [], [], [], []
    for i in idx:
        w = (i - a) / max(1, z - a)
        Rf = Rotation.from_matrix(np.array(f[i]["R_cw"]).T)
        Rb = Rotation.from_matrix(np.array(b[i]["R_cw"]).T)
        Cf, Cb = np.array(f[i]["C"]), np.array(b[i]["C"])
        C.append((1 - w) * Cf + w * Cb)
        Rwc.append(Slerp([0.0, 1.0], Rotation.concatenate([Rf, Rb]))(w).as_matrix())
        gap_m.append(float(np.linalg.norm(Cf - Cb)))
        gap_deg.append(float(np.degrees((Rf.inv() * Rb).magnitude())))
    return idx, np.array(Rwc), np.array(C), np.array(gap_m), np.array(gap_deg)


def write_tum(path: Path, times, Rwc, C, header: str) -> None:
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(Rwc).as_quat()
    lines = [f"{t:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f} {a[0]:.8f} {a[1]:.8f} {a[2]:.8f} {a[3]:.8f}" for t, c, a in zip(times, C, q)]
    path.write_text(f"# {header}\n# t tx ty tz qx qy qz qw (camera to world)\n" + "\n".join(lines) + "\n", encoding="utf-8")


def read_tum(path: Path):
    from scipy.spatial.transform import Rotation
    d = np.loadtxt(path, ndmin=2)
    return d[:, 0], Rotation.from_quat(d[:, 4:8]).as_matrix(), d[:, 1:4]


# ------------------------------------------------------------------ CLI

def frame_list(folder: str) -> List[str]:
    fs = sorted(glob.glob(str(Path(folder) / "*.jpg")) + glob.glob(str(Path(folder) / "*.png")))
    if not fs:
        raise SystemExit(f"[FAIL] no frames in {folder}")
    return fs


def parse_box(s: Optional[str]):
    return tuple(int(v) for v in s.split(",")) if s else None


def cmd_scene(a) -> int:
    V, T, B, names = gather_scene(Path(a.tileset))
    np.savez_compressed(a.out, V=V.astype(np.float32), T=T.astype(np.int32), B=B.astype(np.int32), names=np.array(names))
    print(f"{len(names)} buildings, {len(T)} triangles; E {V[:, 0].min():.0f} to {V[:, 0].max():.0f} m, "
          f"N {V[:, 1].min():.0f} to {V[:, 1].max():.0f} m, up to {V[:, 2].max():.0f} m -> {a.out}")
    return 0


def cmd_pnp(a) -> int:
    import cv2
    H, W = cv2.imread(a.frame, cv2.IMREAD_GRAYSCALE).shape
    pts = read_points(Path(a.points))
    K, R, C, e = solve_pnp([v[0] for v in pts.values()], [v[1] for v in pts.values()], W, H, a.hfov)
    out = camera_json(W, H, K, R, C, pnp_px=float(e.mean()),
                      points={k: [list(map(float, v[0])), list(map(float, v[1]))] for k, v in pts.items()})
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    f = R[2]
    print(f"lens {hfov(K, W):.2f} deg across; points off by {', '.join(f'{k} {v:.1f}' for k, v in zip(pts, e))} px "
          f"(mean {e.mean():.1f}); camera at E {C[0]:.1f} N {C[1]:.1f} up {C[2]:.1f} m, heading "
          f"{np.degrees(np.arctan2(f[0], f[1])) % 360:.1f} deg, tilted down {np.degrees(np.arcsin(-f[2])):.1f} deg -> {a.out}")
    return 0


def cmd_refine(a) -> int:
    import cv2
    cam = read_camera(a.camera)
    pts = read_points(Path(a.points)) if a.points else None
    out = refine(load_scene(Path(a.scene)), cam, cv2.imread(a.frame, cv2.IMREAD_GRAYSCALE), pts,
                 free_lens=a.free_lens, ignore=parse_box(a.ignore))
    if pts:
        out["points"] = {k: [list(map(float, v[0])), list(map(float, v[1]))] for k, v in pts.items()}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    r = out["refine"]
    print(f"moved {r['moved_m']:.2f} m; edges {r['edge_px_before']:.2f} -> {r['edge_px_after']:.2f} px, points "
          f"{r['points_px_before']:.1f} -> {r['points_px_after']:.1f} px ({r['renders']} renders); camera at "
          f"{np.round(out['C'], 1).tolist()} -> {a.out}")
    return 0


def cmd_track(a) -> int:
    cam = read_camera(a.camera)
    pts = read_points(Path(a.points)) if a.points else ({k: tuple(v) for k, v in cam["points"].items()} if cam.get("points") else None)
    out = track(load_scene(Path(a.scene)), frame_list(a.frames), cam, a.start, -1 if a.backward else 1, pts, parse_box(a.ignore))
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    last = out["poses"][0 if a.backward else -1]
    print(f"{len(out['poses'])} frames; median {np.median([p['corners'] for p in out['poses'][1:]]):.0f} corners a frame, "
          f"median {np.median([p['rms_px'] for p in out['poses'][1:]]):.2f} px rms; ends at frame {last['i']} with the camera at "
          f"{np.round(last['C'], 1).tolist()} -> {a.out}")
    if a.check:
        ref = read_camera(a.check)
        d = np.array(last["C"]) - np.array(ref["C"])
        Rr = np.array(last["R_cw"]) @ np.array(ref["R_cw"]).T
        print(f"against {a.check}: {np.linalg.norm(d):.2f} m apart (E {d[0]:+.2f} N {d[1]:+.2f} up {d[2]:+.2f}), "
              f"pointing {np.degrees(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1))):.2f} deg apart")
    return 0


def cmd_truth(a) -> int:
    fw = json.loads(Path(a.forward).read_text(encoding="utf-8"))
    bw = json.loads(Path(a.backward).read_text(encoding="utf-8"))
    idx, Rwc, C, gap_m, gap_deg = blend(fw, bw)
    times = np.array(idx) / a.fps
    write_tum(Path(a.out), times, Rwc, C, "true camera path from register_video.py truth: downtown frame (m east, north, up), OpenCV camera axes")
    Path(a.out).with_suffix(".gaps.json").write_text(json.dumps(dict(frames=idx, gap_m=gap_m.tolist(), gap_deg=gap_deg.tolist())), encoding="utf-8")
    step = np.linalg.norm(np.diff(C, axis=0), axis=1).sum()
    print(f"{len(idx)} frames; the two tracks are a median {np.median(gap_m):.1f} m apart (largest {gap_m.max():.1f} m at "
          f"{times[gap_m.argmax()]:.1f} s) and point a median {np.median(gap_deg):.2f} deg apart (largest {gap_deg.max():.2f}); "
          f"the camera flies {step:.0f} m and climbs {C[-1, 2] - C[0, 2]:.0f} m -> {a.out}")
    return 0


def cmd_overlay(a) -> int:
    import cv2
    scene = load_scene(Path(a.scene))
    frames = frame_list(a.frames)
    cam = read_camera(a.camera)
    _, Rwc, C = read_tum(Path(a.truth))
    if len(C) != len(frames):
        raise SystemExit(f"[FAIL] {len(C)} poses for {len(frames)} frames")
    s = a.width / cam["width"]
    W, H = a.width, round(cam["height"] * s)
    K = np.array(cam["K"]) * s
    K[2, 2] = 1.0
    K[:2, 2] = (np.array(cam["K"])[:2, 2] + .5) * s - .5
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(a.fps),
                           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", a.out], stdin=subprocess.PIPE)
    for i, f in enumerate(frames):
        _, depth = render(scene, Rwc[i].T, C[i], K, W, H)
        img = cv2.resize(cv2.imread(f), (W, H), interpolation=cv2.INTER_AREA)
        img[silhouette(depth, 0.04)] = (0, 255, 255)
        cv2.putText(img, f"{i / a.fps:5.1f} s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{len(frames)} frames -> {a.out}")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="See the module docstring for the method.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scene", help="the tileset's meshes in one .npz")
    p.add_argument("--tileset", required=True, help="folder with tileset.json and glb/ (outputs/downtown)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("pnp", help="one frame's camera from hand-measured points")
    p.add_argument("--frame", required=True)
    p.add_argument("--points", required=True, help='json: {name: {"xyz": [E, N, U], "px": [x, y]}}')
    p.add_argument("--hfov", type=float, default=None, help="lens width in degrees; leave out to find it")
    p.add_argument("--out", required=True)
    p = sub.add_parser("refine", help="nudge a camera so the meshes' edges fall on the frame's")
    p.add_argument("--scene", required=True)
    p.add_argument("--frame", required=True)
    p.add_argument("--camera", required=True, help="camera json, or track.json@i")
    p.add_argument("--points", default=None, help="hand-measured points to anchor it (as for pnp)")
    p.add_argument("--free-lens", action="store_true", help="let the lens width change too")
    p.add_argument("--ignore", default=None, help="x0,y0,x1,y1 in the frame's pixels: a box to leave out (a watermark)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("track", help="every frame's camera from a solved one")
    p.add_argument("--scene", required=True)
    p.add_argument("--frames", required=True, help="folder of frames, in order by name")
    p.add_argument("--camera", required=True, help="the solved camera (json, or track.json@i)")
    p.add_argument("--start", type=int, required=True, help="its frame's index in --frames")
    p.add_argument("--backward", action="store_true", help="track towards the first frame")
    p.add_argument("--points", default=None, help="hand-measured points in the start frame (default: the camera's own)")
    p.add_argument("--ignore", default=None, help="x0,y0,x1,y1: a box where nothing is tracked (a watermark)")
    p.add_argument("--check", default=None, help="a camera solved separately for the last frame, to compare with")
    p.add_argument("--out", required=True)
    p = sub.add_parser("truth", help="one path from a forward and a backward track")
    p.add_argument("--forward", required=True)
    p.add_argument("--backward", required=True)
    p.add_argument("--fps", type=float, required=True, help="frames per second of the frame folder (for TUM times)")
    p.add_argument("--out", required=True, help="TUM file; the disagreement goes beside it as .gaps.json")
    p = sub.add_parser("overlay", help="the meshes' outlines over every frame, as a video")
    p.add_argument("--scene", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--truth", required=True, help="TUM path, one pose per frame")
    p.add_argument("--camera", required=True, help="a camera json for the lens")
    p.add_argument("--fps", type=float, default=12.0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--out", required=True)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = parse_args(argv)
    return dict(scene=cmd_scene, pnp=cmd_pnp, refine=cmd_refine, track=cmd_track, truth=cmd_truth, overlay=cmd_overlay)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
