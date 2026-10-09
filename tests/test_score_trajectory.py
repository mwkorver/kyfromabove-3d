"""The trajectory score against estimates made from the truth on purpose:
the truth itself in a model's own scale, frame and conventions, then with
noise, drift and a change of scale added. Each must come back as exactly
what was put in."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import render_flyaround as rf                                                # noqa: E402
import score_trajectory as st                                                # noqa: E402

FPS = 10.0
S_MODEL = 0.025                                   # model units per metre: a 100 m orbit is 2.5 units
R_MODEL = Rotation.from_euler("zyx", [40.0, -25.0, 110.0], degrees=True).as_matrix()
T_MODEL = np.array([0.3, -1.2, 4.0])


@pytest.fixture
def truth():
    """Two turns of a climbing orbit, as render_flyaround makes them."""
    return rf.orbit_path(np.array([12.0, -7.0]), 30.0, 90.0, 40.0, 85.0, 2.0, 60)


def write_truth(tmp_path: Path, poses) -> Path:
    frames = [{"index": i, "time_s": i / FPS, "R_cw": p.R_cw.tolist(), "t_cw": p.t_cw.tolist(), "C": p.C.tolist()}
              for i, p in enumerate(poses)]
    p = tmp_path / "cameras.json"
    p.write_text(json.dumps({"camera": {"hfov_deg": 70.0}, "frames": frames}))
    return p


def in_model_frame(poses, noise_m=0.0, drift=None, scale_of=None, seed=3):
    """Each pose as a model would report it: camera to world in its own frame
    (x_model = S R x + T). ``drift(i, n)`` adds metres to the centre, ``scale_of(i, n)``
    replaces S. Returns (R_wc (N, 3, 3), C (N, 3)) in the model's frame."""
    rng = np.random.default_rng(seed)
    n = len(poses)
    R_wc, C = [], []
    for i, p in enumerate(poses):
        c = p.C + (drift(i, n) if drift else 0.0) + rng.normal(scale=noise_m, size=3)
        s = scale_of(i, n) if scale_of else S_MODEL
        C.append(s * R_MODEL @ c + T_MODEL)
        R_wc.append(R_MODEL @ p.R_cw.T)
    return np.array(R_wc), np.array(C)


def save_frames(folder: Path, R_wc, C, kind="c2w", K=None):
    folder.mkdir(parents=True, exist_ok=True)
    for i, (R, c) in enumerate(zip(R_wc, C)):
        M = np.column_stack([R, c]) if kind == "c2w" else np.column_stack([R.T, -R.T @ c])
        extra = {"intrinsic": K} if K is not None else {}
        np.savez(folder / f"frame_{i:06d}.npz", extrinsic=M.astype(np.float32 if kind == "f32" else np.float64),
                 depth=np.zeros((4, 4, 1)), **extra)
    return folder


def args(truth_path, est_path, *extra):
    return st.parse_args(["--truth", str(truth_path), "--estimate", str(est_path), *extra])


def test_the_truth_in_another_frame_scores_zero(tmp_path, truth):
    R_wc, C = in_model_frame(truth)
    K = np.array([[300.0, 0, 258.5], [0, 300.0, 188.5], [0, 0, 1]])
    r = st.run(args(write_truth(tmp_path, truth), save_frames(tmp_path / "est", R_wc, C, K=K)))
    assert r["extrinsic"] == "c2w" and r["frames"] == 120
    assert r["ate_m"]["max"] < 1e-6 and r["rotation_deg"]["max"] < 1e-4
    assert r["axes_offset_deg"] < 1e-4
    assert np.isclose(r["fit"]["scale_m_per_unit"], 1 / S_MODEL, rtol=1e-9)
    assert abs(r["scale_drift"]) < 1e-9
    assert all(e["translation_m"]["max"] < 1e-6 for e in r["rpe"].values())
    assert np.isclose(r["estimate_hfov_deg"], np.degrees(2 * np.arctan2(259.0, 300.0)))


def test_world_to_camera_extrinsics_are_recognised(tmp_path, truth):
    R_wc, C = in_model_frame(truth)
    r = st.run(args(write_truth(tmp_path, truth), save_frames(tmp_path / "est", R_wc, C, kind="w2c")))
    assert r["extrinsic"] == "w2c" and r["ate_m"]["max"] < 1e-6
    assert r["extrinsic_ate_rmse_m"]["c2w"] > 1.0                             # the other reading does not fit
    forced = st.run(args(write_truth(tmp_path, truth), tmp_path / "est", "--extrinsic", "c2w"))
    assert forced["extrinsic"] == "c2w" and forced["ate_m"]["rmse"] > 1.0


def test_opengl_axes_show_as_a_half_turn_until_named(tmp_path, truth):
    R_wc, C = in_model_frame(truth)
    est = save_frames(tmp_path / "est", R_wc @ st.OPENGL_TO_OPENCV, C)
    wrong = st.run(args(write_truth(tmp_path, truth), est))
    assert wrong["ate_m"]["max"] < 1e-6                                       # positions do not care
    assert np.isclose(wrong["axes_offset_deg"], 180.0, atol=1e-3)
    assert "try the other --axes" in st.summary(wrong)
    right = st.run(args(write_truth(tmp_path, truth), est, "--axes", "opengl"))
    assert right["axes_offset_deg"] < 1e-4 and right["rotation_deg"]["max"] < 1e-4


def test_noise_comes_back_in_metres(tmp_path, truth):
    R_wc, C = in_model_frame(truth, noise_m=0.5)
    r = st.run(args(write_truth(tmp_path, truth), save_frames(tmp_path / "est", R_wc, C)))
    # 0.5 m on each axis is 0.87 m in 3D; the fit absorbs a little of it
    assert 0.7 < r["ate_m"]["rmse"] < 0.95
    assert np.isclose(r["fit"]["scale_m_per_unit"], 1 / S_MODEL, rtol=0.01)


def test_fitting_on_the_start_shows_drift_the_oracle_hides(tmp_path, truth):
    drift = lambda i, n: np.array([0.0, 0.0, 6.0]) * (i / n) ** 2              # 6 m of climb by the end
    R_wc, C = in_model_frame(truth, drift=drift)
    t, e = write_truth(tmp_path, truth), save_frames(tmp_path / "est", R_wc, C)
    oracle = st.run(args(t, e))
    start = st.run(args(t, e, "--align-frames", "30"))
    assert start["aligned_on_frames"] == 30
    assert start["ate_first_quarter_m"]["rmse"] < 0.3
    assert start["ate_last_quarter_m"]["rmse"] > 3.0
    assert start["ate_m"]["max"] > oracle["ate_m"]["max"] * 1.5               # the oracle spreads it out


def test_a_change_of_scale_halfway_is_reported(tmp_path, truth):
    R_wc, C = in_model_frame(truth, scale_of=lambda i, n: S_MODEL * (1.0 if i < n // 2 else 1 / 1.1))
    r = st.run(args(write_truth(tmp_path, truth), save_frames(tmp_path / "est", R_wc, C)))
    assert np.isclose(r["scale_drift"], 0.10, atol=1e-9)                      # the second half's metres per unit, 10% more


def test_a_strided_estimate_pairs_with_every_other_frame(tmp_path, truth):
    R_wc, C = in_model_frame(truth)
    r = st.run(args(write_truth(tmp_path, truth), save_frames(tmp_path / "est", R_wc[::2], C[::2]), "--stride", "2"))
    assert r["frames"] == 60 and r["ate_m"]["max"] < 1e-6
    off = st.run(args(write_truth(tmp_path, truth), tmp_path / "est"))         # paired 1:1, it cannot fit
    assert off["ate_m"]["rmse"] > 10.0


def test_tum_files_pair_by_time(tmp_path, truth):
    t_path = tmp_path / "truth_tum.txt"
    t_path.write_text("\n".join(rf.tum_line(i / FPS, p) for i, p in enumerate(truth)) + "\n")
    R_wc, C = in_model_frame(truth)
    q = Rotation.from_matrix(R_wc).as_quat()
    keep = np.arange(5, 120, 3)                                               # a later start, every third frame
    e_path = tmp_path / "est_tum.txt"
    e_path.write_text("\n".join(" ".join(map(str, (i / FPS, *C[i], *q[i]))) for i in keep) + "\n")
    r = st.run(args(t_path, e_path))
    assert r["frames"] == len(keep) and r["ate_m"]["max"] < 1e-5 and r["rotation_deg"]["max"] < 1e-3


def test_one_npz_of_float32_extrinsics(tmp_path, truth):
    R_wc, C = in_model_frame(truth)
    M = np.concatenate([R_wc, C[:, :, None]], axis=2).astype(np.float32)[None]   # (1, S, 3, 4)
    np.savez(tmp_path / "pred.npz", extrinsic=M)
    r = st.run(args(write_truth(tmp_path, truth), tmp_path / "pred.npz"))
    assert r["frames"] == 120 and r["ate_m"]["rmse"] < 0.01                    # float32 in a 2.5-unit frame


def test_the_script_writes_its_record(tmp_path, truth):
    R_wc, C = in_model_frame(truth, noise_m=0.2)
    out = tmp_path / "score"
    assert st.main(["--truth", str(write_truth(tmp_path, truth)), "--estimate",
                    str(save_frames(tmp_path / "est", R_wc, C)), "--out", str(out), "--plot"]) == 0
    rec = json.loads((out / "score.json").read_text())
    assert rec["frames"] == 120 and len(rec["per_frame"]["position_error_m"]) == 120
    assert (out / "score.png").stat().st_size > 10_000
    aligned = st.read_tum(out / "aligned_tum.txt")
    assert np.sqrt(np.mean(np.sum((aligned.C - np.array([p.C for p in truth])) ** 2, axis=1))) == pytest.approx(
        rec["ate_m"]["rmse"], rel=1e-4)


def test_umeyama_recovers_a_known_similarity():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(50, 3)) * 10
    R = Rotation.random(random_state=2).as_matrix()
    s, R_fit, t = st.umeyama(X, 3.7 * X @ R.T + [1.0, 2.0, 3.0])
    assert np.isclose(s, 3.7) and np.allclose(R_fit, R) and np.allclose(t, [1, 2, 3])
