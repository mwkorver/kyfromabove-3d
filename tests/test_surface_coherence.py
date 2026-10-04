"""Telling real structure from range noise.

A high local plane residual says a point is not on a plane. It does not say
why, and the two causes want opposite treatment: a roof vent is detail worth
resolving with denser sampling, a noisy grazing-incidence facade return is
measurement error and densifying it resolves nothing.

The discriminator is spatial coherence — whether a point's neighbours deviate
from the surrounding surface *together*. These tests build both causes
explicitly, because on real data they are indistinguishable by eye.
"""

import numpy as np
import pytest

from kyfromabove_3d.features import surface_coherence


def grid(side=12.0, spacing=0.25):
    g = np.arange(0.0, side, spacing)
    gx, gy = np.meshgrid(g, g)
    return np.column_stack([gx.ravel(), gy.ravel()])


def test_noise_is_incoherent_and_structure_is_coherent():
    """The same residual magnitude, opposite verdicts."""
    rng = np.random.default_rng(0)
    xy = grid()

    noisy = np.column_stack([xy, rng.normal(0, 0.05, len(xy))])
    noise_coh, _ = surface_coherence(noisy)

    # A step of the same order as the noise band, but systematic.
    z = np.where(xy[:, 0] > 6.0, 0.20, 0.0) + rng.normal(0, 0.01, len(xy))
    stepped = np.column_stack([xy, z])
    step_coh, _ = surface_coherence(stepped)

    on_step = np.abs(xy[:, 0] - 6.0) < 0.8
    assert np.median(noise_coh) < 0.5, "independent noise must not read as structure"
    assert np.median(step_coh[on_step]) > 0.8, "a real step must read as structure"


def test_offset_records_which_way_and_how_far():
    """The magnitude is what says whether the LiDAR is under-sampling."""
    rng = np.random.default_rng(1)
    xy = grid()
    # Smaller than the outer neighbourhood's reach, or it becomes the surface.
    bump = ((xy[:, 0] - 6.0) ** 2 + (xy[:, 1] - 6.0) ** 2) < 0.6 ** 2  # < outer reach
    z = np.where(bump, 0.6, 0.0) + rng.normal(0, 0.01, len(xy))
    coh, off = surface_coherence(np.column_stack([xy, z]))

    on_top = ((xy[:, 0] - 6.0) ** 2 + (xy[:, 1] - 6.0) ** 2) < 0.35 ** 2
    assert on_top.sum() >= 4, "fixture must put a few points on the feature"
    assert np.median(coh[on_top]) > 0.8
    assert np.median(off[on_top]) > 0.2, "a raised feature offsets one way"
    # Well away from the bump the surface is a plane; nothing to report.
    far = ((xy[:, 0] - 6.0) ** 2 + (xy[:, 1] - 6.0) ** 2) > 4.0 ** 2
    assert abs(np.median(off[far])) < 0.05


def test_a_smooth_slope_is_not_flagged_as_structure():
    """Coherence must measure departure from the local surface, not tilt.

    A pitched roof is planar; it should look like a plane however steep it is.
    """
    rng = np.random.default_rng(2)
    xy = grid()
    for slope in (0.0, 0.3, 1.0):
        z = slope * xy[:, 0] + rng.normal(0, 0.01, len(xy))
        _, off = surface_coherence(np.column_stack([xy, z]))
        assert abs(np.median(off)) < 0.05, f"slope {slope} reported an offset"


def test_outer_k_bounds_the_structure_size_that_can_be_seen():
    """A feature wider than the outer neighbourhood *becomes* the local surface.

    Not a defect — it is what separates "detail the LiDAR under-samples" from
    "a whole roof level", which is a different question and a different tool.
    But it means outer_k has to be chosen against the point density, and a
    silent zero here would otherwise read as "no structure present".
    """
    rng = np.random.default_rng(3)
    xy = grid()
    r2 = (xy[:, 0] - 6.0) ** 2 + (xy[:, 1] - 6.0) ** 2
    wide = np.column_stack([xy, np.where(r2 < 3.0 ** 2, 0.6, 0.0)
                            + rng.normal(0, 0.01, len(xy))])
    _, off = surface_coherence(wide, outer_k=48)
    centre = r2 < 1.0 ** 2
    assert abs(np.median(off[centre])) < 0.05, "a 3 m feature is invisible at outer_k=48"

    # Reaching further finds it.
    _, off_wide = surface_coherence(wide, outer_k=900)
    assert abs(np.median(off_wide[centre])) > 0.15


def test_degenerate_input_returns_zeros_rather_than_raising():
    coh, off = surface_coherence(np.zeros((5, 3)))
    assert coh.shape == (5,) and off.shape == (5,)
    assert not coh.any() and not off.any()
