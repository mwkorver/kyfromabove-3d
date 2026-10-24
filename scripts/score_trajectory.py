#!/usr/bin/env python3
"""Score camera poses estimated from video against the true ones.

The truth is ``cameras.json`` from ``render_flyaround.py`` (or any TUM
trajectory); the estimate is what a video reconstruction model made of the
same frames -- LingBot-Map's saved predictions, an ``.npz``/``.npy`` of
extrinsics, or a TUM file. A model working from video alone has no scale,
no north and no origin, so the estimate is first fitted to the truth with a
similarity transform (rotation, translation and one scale: Sim(3),
Umeyama's closed form on the camera centres). What is left is the model's
own error, in the truth's metres:

    ATE      absolute trajectory error: each camera centre's distance from
             where it truly was, after the fit (RMSE, median, max)
    rotation each camera's orientation error after the fit, degrees
    RPE      relative pose error over ``--rpe-deltas`` frames: how wrong the
             motion between two frames is, metres and degrees -- drift, with
             the fit taken out
    scale    metres per unit of the model, and how much it changes between
             the first half of the sequence and the second

``--align-frames N`` fits on the first N frames only and scores all of
them: the honest test of drift, since a real capture would be registered to
the LiDAR where it starts and left to run. Fitting on every frame is the
oracle: it hides drift by spreading it.

The fit here uses the truth. Registering a real capture, with no truth, to
the building's LiDAR is a different step; this one measures whether the
poses would be worth registering.

Estimate formats:

    a folder of frame_*.npz   LingBot-Map (``demo_render/batch_demo.py
                              --save_predictions``): ``extrinsic`` per frame,
                              and ``intrinsic`` when there
    file.npz / file.npy       ``extrinsic`` (S, 3, 4) or (S, 4, 4)
    file.txt                  TUM lines: t tx ty tz qx qy qz qw, camera to world

``--extrinsic`` says whether a matrix maps camera to world (c2w) or world
to camera (w2c); ``auto`` (the default) tries both and keeps the one that
fits, and says which. ``--axes`` is the camera's axes: ``opencv`` (x right,
y down, z forward, as the truth) or ``opengl`` (y up, z back). A wrong
choice shows as a large constant ``axes offset`` in the report.

Frames are paired by timestamp when both sides have one, else by order:
estimate frame i is truth frame ``--offset + i * --stride``.

Usage::

    python scripts/score_trajectory.py --truth outputs/flyaround/humana/cameras.json \\
        --estimate outputs/lingbot/humana/ --out outputs/flyaround/humana/score
    python scripts/score_trajectory.py --truth cameras.json --estimate est_tum.txt --align-frames 60 --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


# ------------------------------------------------------------------ trajectories

@dataclass
class Trajectory:
    """Camera-to-world rotations and camera centres, with times if known."""
    R_wc: np.ndarray                          # (N, 3, 3)
    C: np.ndarray                             # (N, 3)
    time: Optional[np.ndarray] = None         # (N,) seconds

    def __len__(self) -> int:
        return len(self.C)

    def take(self, i: np.ndarray) -> "Trajectory":
        return Trajectory(self.R_wc[i], self.C[i], None if self.time is None else self.time[i])


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(q).as_matrix()                                   # x y z w


def read_tum(path: Path) -> Trajectory:
    rows = [l.split() for l in Path(path).read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    a = np.array(rows, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 8:
        raise SystemExit(f"[FAIL] {path}: TUM lines have 8 numbers (t tx ty tz qx qy qz qw)")
    return Trajectory(_quat_to_R(a[:, 4:8]), a[:, 1:4], a[:, 0])


def read_truth(path: Path) -> Trajectory:
    """``cameras.json`` from render_flyaround.py, or a TUM file."""
    path = Path(path)
    if path.suffix != ".json":
        return read_tum(path)
    rec = json.loads(path.read_text(encoding="utf-8"))
    f = rec["frames"]
    R_cw = np.array([x["R_cw"] for x in f], dtype=np.float64)
    return Trajectory(np.transpose(R_cw, (0, 2, 1)), np.array([x["C"] for x in f], dtype=np.float64),
                      np.array([x["time_s"] for x in f], dtype=np.float64) if all("time_s" in x for x in f) else None)


def _as_4x4(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    while M.ndim > 3 and M.shape[0] == 1:                                      # a batch of one
        M = M[0]
    if M.ndim == 2:
        M = M[None]
    if M.shape[-2:] not in ((3, 4), (4, 4)):
        raise SystemExit(f"[FAIL] extrinsics are (S, 3, 4) or (S, 4, 4), not {M.shape}")
    out = np.tile(np.eye(4), (len(M), 1, 1))
    out[:, :3, :] = M[:, :3, :]
    return out


def read_estimate(path: Path) -> Dict[str, Any]:
    """The estimate's raw matrices (S, 4, 4), or a trajectory if it is TUM,
    and its intrinsics (S, 3, 3) when it has them."""
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("frame_*.npz"))
        if not files:
            raise SystemExit(f"[FAIL] {path} has no frame_*.npz")
        mats, Ks = [], []
        for f in files:
            with np.load(f) as z:
                if "extrinsic" not in z:
                    raise SystemExit(f"[FAIL] {f} has no 'extrinsic' (it has {sorted(z.files)})")
                mats.append(_as_4x4(z["extrinsic"])[0])
                if "intrinsic" in z:
                    Ks.append(np.asarray(z["intrinsic"], dtype=np.float64).reshape(-1, 3, 3)[0])
        return {"M": np.stack(mats), "K": np.stack(Ks) if len(Ks) == len(mats) else None,
                "time": None, "source": f"{len(files)} frame_*.npz"}
    if path.suffix == ".npz":
        with np.load(path) as z:
            key = "extrinsic" if "extrinsic" in z else next(
                (k for k in z.files if z[k].shape[-2:] in ((3, 4), (4, 4))), None)
            if key is None:
                raise SystemExit(f"[FAIL] {path} has no extrinsics (it has {sorted(z.files)})")
            K = np.asarray(z["intrinsic"], dtype=np.float64).reshape(-1, 3, 3) if "intrinsic" in z else None
            return {"M": _as_4x4(z[key]), "K": K, "time": None, "source": f"{path.name}:{key}"}
    if path.suffix == ".npy":
        return {"M": _as_4x4(np.load(path)), "K": None, "time": None, "source": path.name}
    t = read_tum(path)
    return {"trajectory": t, "K": None, "time": t.time, "source": f"{path.name} (TUM)"}


def interpret(est: Dict[str, Any], extrinsic: str, axes: str) -> Trajectory:
    """The estimate as camera-to-world in OpenCV axes."""
    if "trajectory" in est:
        R_wc, C = est["trajectory"].R_wc, est["trajectory"].C
    else:
        M = est["M"]
        if extrinsic == "c2w":
            R_wc, C = M[:, :3, :3], M[:, :3, 3]
        else:
            R_wc = np.transpose(M[:, :3, :3], (0, 2, 1))
            C = -np.einsum("nij,nj->ni", R_wc, M[:, :3, 3])
    if axes == "opengl":
        R_wc = R_wc @ OPENGL_TO_OPENCV
    return Trajectory(R_wc, C, est.get("time"))


def pair(truth: Trajectory, est: Trajectory, stride: int, offset: int, max_dt: Optional[float]):
    """Indices (truth, estimate) of the frames both have."""
    if truth.time is not None and est.time is not None:
        dt = np.median(np.diff(truth.time)) if len(truth) > 1 else 1.0
        tol = max_dt if max_dt is not None else 0.5 * dt
        j = np.clip(np.searchsorted(truth.time, est.time), 0, len(truth) - 1)
        k = np.clip(j - 1, 0, len(truth) - 1)
        near = np.where(np.abs(truth.time[k] - est.time) < np.abs(truth.time[j] - est.time), k, j)
        ok = np.abs(truth.time[near] - est.time) <= tol
        return near[ok], np.flatnonzero(ok)
    i_t = offset + stride * np.arange(len(est))
    ok = (i_t >= 0) & (i_t < len(truth))
    return i_t[ok], np.flatnonzero(ok)


# ------------------------------------------------------------------ the fit and the errors

def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """(s, R, t) minimising |dst - (s R src + t)|^2 over the rows: Umeyama (1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(b.T @ a / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var = (a ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var) if with_scale and var > 0 else 1.0
    return s, R, mu_d - s * R @ mu_s


def rotation_angle_deg(R: np.ndarray) -> np.ndarray:
    c = (np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def nearest_rotation(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def _stats(x: np.ndarray) -> Dict[str, float]:
    return {"rmse": float(np.sqrt(np.mean(x ** 2))), "mean": float(x.mean()), "median": float(np.median(x)),
            "max": float(x.max())}


def score(truth: Trajectory, est: Trajectory, align_frames: Optional[int] = None,
          rpe_deltas: Sequence[int] = (1, 10)) -> Dict[str, Any]:
    """The errors of ``est`` against ``truth``, frame for frame (paired already)."""
    n = len(truth)
    if n < 3:
        raise SystemExit(f"[FAIL] {n} paired frames; at least 3 are needed to fit")
    m = n if not align_frames else min(max(3, align_frames), n)
    s, R, t = umeyama(est.C[:m], truth.C[:m])
    C = s * est.C @ R.T + t
    R_wc = R @ est.R_wc
    pos = np.linalg.norm(C - truth.C, axis=1)
    R_err = np.transpose(truth.R_wc, (0, 2, 1)) @ R_wc                         # camera frame: est relative to truth
    offset = nearest_rotation(R_err.sum(0))
    rot = rotation_angle_deg(R_err)

    rpe = {}
    for d in rpe_deltas:
        if d >= n:
            continue
        a, b = np.arange(n - d), np.arange(d, n)
        # each motion in the first camera's own frame, so the fit's rotation drops out
        dT_t = np.einsum("nji,nj->ni", truth.R_wc[a], truth.C[b] - truth.C[a])
        dT_e = np.einsum("nji,nj->ni", R_wc[a], C[b] - C[a])
        dR_t = np.transpose(truth.R_wc[a], (0, 2, 1)) @ truth.R_wc[b]
        dR_e = np.transpose(R_wc[a], (0, 2, 1)) @ R_wc[b]
        rpe[str(d)] = {"translation_m": _stats(np.linalg.norm(dT_e - dT_t, axis=1)),
                       "rotation_deg": _stats(rotation_angle_deg(np.transpose(dR_t, (0, 2, 1)) @ dR_e))}

    half = n // 2
    scales = [umeyama(est.C[i], truth.C[i])[0] if len(i) >= 3 else None
              for i in (np.arange(half), np.arange(half, n))]
    path_m = float(np.linalg.norm(np.diff(truth.C, axis=0), axis=1).sum())
    extent_m = float(np.linalg.norm(truth.C.max(0) - truth.C.min(0)))
    q = max(1, n // 4)
    return {
        "frames": n, "aligned_on_frames": m,
        "fit": {"scale_m_per_unit": s, "R": R.tolist(), "t": t.tolist()},
        "ate_m": _stats(pos),
        "ate_first_quarter_m": _stats(pos[:q]), "ate_last_quarter_m": _stats(pos[-q:]),
        "ate_rmse_over_extent": float(np.sqrt(np.mean(pos ** 2)) / extent_m) if extent_m > 0 else None,
        "rotation_deg": _stats(rot),
        "axes_offset_deg": float(rotation_angle_deg(offset)),
        "rpe": rpe,
        "scale_first_half": scales[0], "scale_second_half": scales[1],
        "scale_drift": (scales[1] / scales[0] - 1.0) if scales[0] and scales[1] else None,
        "truth_path_m": path_m, "truth_extent_m": extent_m,
        "per_frame": {"position_error_m": pos.tolist(), "rotation_error_deg": rot.tolist()},
        "_aligned": Trajectory(R_wc, C, est.time),
    }


# ------------------------------------------------------------------ the report

def summary(r: Dict[str, Any]) -> str:
    a = r["ate_m"]
    lines = [
        f"{r['frames']} frames paired, fitted on {r['aligned_on_frames']}; the truth moves {r['truth_path_m']:.0f} m "
        f"and spans {r['truth_extent_m']:.0f} m",
        f"scale   {r['fit']['scale_m_per_unit']:.4g} m per unit"
        + (f"; first half {r['scale_first_half']:.4g}, second half {r['scale_second_half']:.4g} "
           f"({100 * r['scale_drift']:+.1f}%)" if r["scale_drift"] is not None else ""),
        f"ATE     rmse {a['rmse']:.3f} m, median {a['median']:.3f} m, max {a['max']:.3f} m "
        f"({100 * r['ate_rmse_over_extent']:.2f}% of the extent)" if r["ate_rmse_over_extent"] is not None else
        f"ATE     rmse {a['rmse']:.3f} m",
        f"        first quarter rmse {r['ate_first_quarter_m']['rmse']:.3f} m, "
        f"last quarter {r['ate_last_quarter_m']['rmse']:.3f} m",
        f"rot     median {r['rotation_deg']['median']:.2f} deg, max {r['rotation_deg']['max']:.2f} deg; "
        f"axes offset {r['axes_offset_deg']:.1f} deg",
    ]
    for d, e in r["rpe"].items():
        lines.append(f"RPE {d:>3} translation rmse {e['translation_m']['rmse']:.3f} m, "
                     f"rotation rmse {e['rotation_deg']['rmse']:.2f} deg")
    if r["axes_offset_deg"] > 20:
        lines.append("[WARN] a large constant rotation between the estimate's cameras and the truth's: "
                     "try the other --axes")
    return "\n".join(lines)


def plot(truth: Trajectory, r: Dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    est = r["_aligned"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.4]})
    ax0.plot(truth.C[:, 0], truth.C[:, 1], color="0.25", lw=2, label="truth")
    ax0.plot(est.C[:, 0], est.C[:, 1], color="tab:orange", lw=1.2, label="estimate, fitted")
    ax0.scatter(*truth.C[0, :2], color="0.25", zorder=3, s=20)
    m = r["aligned_on_frames"]
    if m < r["frames"]:
        ax0.plot(est.C[:m, 0], est.C[:m, 1], color="tab:blue", lw=2, label=f"fitted on these {m}")
    ax0.set_aspect("equal")
    ax0.set_xlabel("east (m)"); ax0.set_ylabel("north (m)")
    ax0.set_title("camera path from above")
    ax0.legend(frameon=False, fontsize=8)
    pos = np.asarray(r["per_frame"]["position_error_m"])
    rot = np.asarray(r["per_frame"]["rotation_error_deg"])
    ax1.plot(pos, color="tab:orange", lw=1.2)
    ax1.set_xlabel("frame"); ax1.set_ylabel("position error (m)", color="tab:orange")
    if m < r["frames"]:
        ax1.axvline(m - 0.5, color="tab:blue", lw=1, ls="--")
    ax2 = ax1.twinx()
    ax2.plot(rot, color="0.5", lw=0.8)
    ax2.set_ylabel("rotation error (deg)", color="0.5")
    ax2.set_ylim(0, max(1.0, 1.1 * rot.max()))                               # hundredths of a degree stay flat
    ax1.set_title(f"ATE rmse {r['ate_m']['rmse']:.2f} m, scale {r['fit']['scale_m_per_unit']:.3g} m/unit")
    for a in (ax0, ax1, ax2):
        a.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_tum(path: Path, traj: Trajectory) -> None:
    from scipy.spatial.transform import Rotation
    t = traj.time if traj.time is not None else np.arange(len(traj), dtype=np.float64)
    q = Rotation.from_matrix(traj.R_wc).as_quat()
    lines = [" ".join(f"{x:.6f}" for x in (ti, *c, *qi)) for ti, c, qi in zip(t, traj.C, q)]
    path.write_text("# t tx ty tz qx qy qz qw (camera to world, the truth's frame, m)\n" + "\n".join(lines) + "\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", type=Path, required=True, help="cameras.json from render_flyaround.py, or TUM")
    ap.add_argument("--estimate", type=Path, required=True, help="frame_*.npz folder, .npz, .npy or TUM .txt")
    ap.add_argument("--out", type=Path, default=None, help="folder for score.json, aligned_tum.txt, score.png")
    ap.add_argument("--extrinsic", choices=("auto", "c2w", "w2c"), default="auto")
    ap.add_argument("--axes", choices=("opencv", "opengl"), default="opencv")
    ap.add_argument("--stride", type=int, default=1, help="truth frames per estimate frame, pairing by order")
    ap.add_argument("--offset", type=int, default=0, help="the truth frame of estimate frame 0, pairing by order")
    ap.add_argument("--max-dt", type=float, default=None, help="seconds, pairing by time (default half a frame)")
    ap.add_argument("--align-frames", type=int, default=None, help="fit on the first N paired frames only")
    ap.add_argument("--rpe-deltas", type=int, nargs="+", default=[1, 10])
    ap.add_argument("--plot", action="store_true", help="write score.png (needs --out)")
    return ap.parse_args(argv)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    truth = read_truth(args.truth)
    raw = read_estimate(args.estimate)
    tried = ["c2w", "w2c"] if args.extrinsic == "auto" and "M" in raw else [args.extrinsic]
    results = {}
    for e in tried:
        est = interpret(raw, e, args.axes)
        i_t, i_e = pair(truth, est, args.stride, args.offset, args.max_dt)
        results[e] = score(truth.take(i_t), est.take(i_e), args.align_frames, args.rpe_deltas)
        results[e]["_truth"] = truth.take(i_t)
        results[e]["unpaired_estimate_frames"] = int(len(est) - len(i_e))
    chosen = min(results, key=lambda k: results[k]["ate_m"]["rmse"])
    r = results[chosen]
    r["extrinsic"] = chosen if "M" in raw else "c2w (TUM)"
    if len(results) > 1:
        r["extrinsic_ate_rmse_m"] = {k: v["ate_m"]["rmse"] for k, v in results.items()}
    r["axes"] = args.axes
    r["estimate"] = {"path": str(args.estimate), "source": raw["source"]}
    r["truth"] = str(args.truth)
    if raw.get("K") is not None:
        K = raw["K"]
        r["estimate_hfov_deg"] = float(np.median(np.degrees(2 * np.arctan2(K[:, 0, 2] + 0.5, K[:, 0, 0]))))
    if Path(args.truth).suffix == ".json":
        cam = json.loads(Path(args.truth).read_text(encoding="utf-8")).get("camera", {})
        if "hfov_deg" in cam:
            r["truth_hfov_deg"] = cam["hfov_deg"]
    return r


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    r = run(args)
    print(f"estimate: {r['estimate']['source']}, read as {r['extrinsic']}, {r['axes']} axes"
          + (f" (ATE rmse c2w {r['extrinsic_ate_rmse_m']['c2w']:.3f} m, w2c {r['extrinsic_ate_rmse_m']['w2c']:.3f} m)"
             if "extrinsic_ate_rmse_m" in r else ""))
    if r["unpaired_estimate_frames"]:
        print(f"[WARN] {r['unpaired_estimate_frames']} estimate frames had no truth frame to pair with")
    print(summary(r))
    if "estimate_hfov_deg" in r:
        print(f"field of view: estimated {r['estimate_hfov_deg']:.1f} deg"
              + (f", true {r['truth_hfov_deg']:.1f} deg" if "truth_hfov_deg" in r else ""))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        aligned, truth = r.pop("_aligned"), r.pop("_truth")
        (args.out / "score.json").write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
        write_tum(args.out / "aligned_tum.txt", aligned)
        if args.plot:
            plot(truth, {**r, "_aligned": aligned}, args.out / "score.png")
        print(f"-> {args.out / 'score.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
