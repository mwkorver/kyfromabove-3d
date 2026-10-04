"""Tests for the cross-view measurement path.

These are deliberately structured so they can fail. The failure mode being
guarded against is the original Phase 1 QA, which computed its "observed" image
position by projecting through the camera under test and therefore reported
0.000 px no matter what the camera model said.

A test that renders both views through the same model can only verify the
measurement machinery, not the correctness of the model's angle convention.
That is what ``test_rotation_convention_is_pinned`` and the network-marked
integration test are for.
"""

from __future__ import annotations

import numpy as np
import pytest

from kyfromabove_3d.cameras import VirtualObliqueCamera, photogrammetric_omega_phi_kappa_to_rcw
from kyfromabove_3d.qa import match_anchor_between_views

MATCH = dict(patch_radius_m=5.0, search_radius_m=3.0, gsd_m=0.08, min_texture_std=4.0)


def test_identical_views_measure_zero(synthetic_pair, ground_anchors):
    """A frame matched against itself must measure zero, not merely be assumed to."""
    cam_a, win_a, _, _ = synthetic_pair
    residuals = []
    for anchor in ground_anchors:
        m = match_anchor_between_views(anchor, cam_a, win_a, cam_a, win_a, **MATCH)
        assert m.accepted, m.reject_reason
        residuals.append(m.residual_px)
    assert np.median(residuals) < 0.05
    assert max(residuals) < 0.25


def test_consistent_views_agree(synthetic_pair, ground_anchors):
    """Two correctly-modelled exposures of the same scene must agree sub-pixel."""
    cam_a, win_a, cam_b, win_b = synthetic_pair
    residuals = []
    for anchor in ground_anchors:
        m = match_anchor_between_views(anchor, cam_a, win_a, cam_b, win_b, **MATCH)
        assert m.accepted, m.reject_reason
        residuals.append(m.residual_px)
    assert np.median(residuals) < 0.5


@pytest.mark.parametrize("shift_m", [0.3, 1.0, 2.0])
def test_injected_translation_is_recovered(synthetic_pair, ground_anchors, shift_m):
    """A known perspective-centre error must come back out of the measurement.

    This is the property the original QA lacked: its metric was blind to pose
    error by construction.
    """
    cam_a, win_a, cam_b, win_b = synthetic_pair
    bad = VirtualObliqueCamera(
        frame_id=cam_b.frame_id + "_perturbed",
        camera_name=cam_b.camera_name,
        camera_center_local_m=cam_b.camera_center_local_m + np.array([shift_m, 0.0, 0.0]),
        R_cw=cam_b.R_cw.copy(),
        width_px=cam_b.width_px, height_px=cam_b.height_px,
        fx_px=cam_b.fx_px, fy_px=cam_b.fy_px, cx_px=cam_b.cx_px, cy_px=cam_b.cy_px,
    )
    measured = []
    for anchor in ground_anchors:
        m = match_anchor_between_views(
            anchor, cam_a, win_a, bad, win_b,
            **{**MATCH, "search_radius_m": shift_m + 1.5},
        )
        if m.accepted:
            measured.append(m.residual_m)
    assert len(measured) >= len(ground_anchors) // 2
    assert np.median(measured) == pytest.approx(shift_m, rel=0.15)


def test_injected_rotation_is_detected(synthetic_pair, ground_anchors):
    """A 0.05 degree pointing error must register well above the noise floor."""
    cam_a, win_a, cam_b, win_b = synthetic_pair
    omega, phi, kappa = 45.0, 0.0, 0.0
    bad = VirtualObliqueCamera(
        frame_id=cam_b.frame_id + "_rot",
        camera_name=cam_b.camera_name,
        camera_center_local_m=cam_b.camera_center_local_m.copy(),
        R_cw=photogrammetric_omega_phi_kappa_to_rcw(omega + 0.05, phi, kappa),
        width_px=cam_b.width_px, height_px=cam_b.height_px,
        fx_px=cam_b.fx_px, fy_px=cam_b.fy_px, cx_px=cam_b.cx_px, cy_px=cam_b.cy_px,
    )
    measured = [
        m.residual_px
        for anchor in ground_anchors
        if (m := match_anchor_between_views(
            anchor, cam_a, win_a, bad, win_b, **{**MATCH, "search_radius_m": 5.0}
        )).accepted
    ]
    assert len(measured) >= len(ground_anchors) // 2
    assert np.median(measured) > 5.0


def test_wrong_rotation_order_is_caught(synthetic_pair, ground_anchors):
    """The specific bug this QA path was built to find must be detectable.

    ``Rx(-w) Ry(-p) Rz(-k)`` is the same three rotations in the wrong order. On
    the real obliques it put consecutive exposures 26 px apart while the old
    self-consistency QA still reported 0.000 px.
    """
    from kyfromabove_3d.cameras import rot_x, rot_y, rot_z

    cam_a, win_a, cam_b, win_b = synthetic_pair
    w, p, k = np.radians([45.0, 3.0, 4.0])
    wrong = np.diag([1.0, -1.0, -1.0]) @ (rot_x(-w) @ rot_y(-p) @ rot_z(-k))
    right = photogrammetric_omega_phi_kappa_to_rcw(45.0, 3.0, 4.0)
    assert not np.allclose(wrong, right), "conventions must actually differ"

    bad = VirtualObliqueCamera(
        frame_id=cam_b.frame_id + "_badconv", camera_name=cam_b.camera_name,
        camera_center_local_m=cam_b.camera_center_local_m.copy(), R_cw=wrong,
        width_px=cam_b.width_px, height_px=cam_b.height_px,
        fx_px=cam_b.fx_px, fy_px=cam_b.fy_px, cx_px=cam_b.cx_px, cy_px=cam_b.cy_px,
    )
    accepted = [
        m for anchor in ground_anchors
        if (m := match_anchor_between_views(
            anchor, cam_a, win_a, bad, win_b, **{**MATCH, "search_radius_m": 6.0}
        )).accepted
    ]
    # Either the offset is measured as large, or correlation cannot find the
    # patch at all within the search window. Both are detections; silently
    # returning zero is the failure this guards against.
    assert len(accepted) < len(ground_anchors) or np.median(
        [m.residual_px for m in accepted]
    ) > 10.0


# ------------------------------------------------- restricting to the building

def test_anchors_in_volume_selects_by_position():
    """The anchor set spans the whole AOI; the building is a subset of it."""
    from types import SimpleNamespace

    from kyfromabove_3d.crops import BoundingVolume3D
    from kyfromabove_3d.qa import anchors_in_volume

    vol = BoundingVolume3D(min_x_m=-10.0, max_x_m=10.0, min_y_m=-10.0,
                           max_y_m=10.0, min_z_m=0.0, max_z_m=50.0)
    inside = SimpleNamespace(anchor_id="in", xyz_local_m=np.array([0.0, 0.0, 25.0]))
    on_edge = SimpleNamespace(anchor_id="edge", xyz_local_m=np.array([10.0, 10.0, 50.0]))
    far_xy = SimpleNamespace(anchor_id="far", xyz_local_m=np.array([40.0, 0.0, 25.0]))
    too_low = SimpleNamespace(anchor_id="low", xyz_local_m=np.array([0.0, 0.0, -5.0]))

    got = {a.anchor_id for a in
           anchors_in_volume([inside, on_edge, far_xy, too_low], vol)}
    assert got == {"in", "edge"}, "the boundary is inclusive; outside is excluded"

    padded = {a.anchor_id for a in
              anchors_in_volume([inside, too_low], vol, pad_m=6.0)}
    assert padded == {"in", "low"}


def test_filter_anchors_resummarises_without_rematching():
    """The restricted number must come from the same matches, not a new run."""
    from kyfromabove_3d.qa import MatchResult, ReprojectionReport

    def m(anchor_id, residual, accepted=True):
        return MatchResult(
            anchor_id=anchor_id, label="roof", ref_frame="a", tgt_frame="b",
            accepted=accepted, reject_reason=None if accepted else "low_ncc",
            ncc=0.9, offset_e1_m=0.0, offset_e2_m=0.0,
            residual_m=residual * 0.05, residual_px=residual,
            observed_uv=(0.0, 0.0),
        )

    report = ReprojectionReport(matches=[m("A", 1.0), m("B", 9.0), m("C", 1.0)])
    assert report.stats()["accepted"] == 3

    restricted = report.filter_anchors({"A", "C"})
    assert restricted.stats()["accepted"] == 2
    assert restricted.stats()["median_error_px"] == pytest.approx(1.0)
    # The original is untouched -- both numbers are reported side by side.
    assert report.stats()["accepted"] == 3
    assert len(restricted.filter_anchors(set()).matches) == 0
