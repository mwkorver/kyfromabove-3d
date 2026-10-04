"""The one number that decides whether a building is worth densifying.

:func:`assess_surface_complexity` folds a local plane fit and a coherence
measure into a structure fraction. Both inputs are tested in
``test_surface_coherence.py``; what matters here is that the fold does not
confuse the two causes of roughness, that it refuses to answer rather than
answering zero when it cannot see enough surface, and that its false-positive
floor is known — a rough point has some chance of reading coherent by luck, and
the survey's headline numbers are only meaningful above that floor.

Fixtures are built at 0.6 m spacing with sub-decimetre noise, which is the
regime KyFromAbove is actually in. That matters: when the noise amplitude
approaches the point spacing the surface stops being a surface, the wide plane
fit becomes meaningless, and the floor rises to a third of rough points instead
of a tenth.
"""

import numpy as np
import pytest

from kyfromabove_3d.features import (
    NOISE_COHERENCE,
    ROUGH_RESIDUAL_M,
    STRUCTURE_COHERENCE,
    assess_surface_complexity,
)

SPACING_M = 0.6            # KyFromAbove Phase 2 over a building, near enough


def grid(side=24.0, spacing=SPACING_M):
    g = np.arange(0.0, side, spacing)
    gx, gy = np.meshgrid(g, g)
    return np.column_stack([gx.ravel(), gy.ravel()])


def flat_roof(sigma=0.02, seed=0):
    """A plate at the sensor's noise floor: nothing to resolve."""
    xy = grid()
    return np.column_stack([xy, np.random.default_rng(seed).normal(0, sigma, len(xy))])


def vented_roof(seed=1):
    """A plate carrying raised blocks: real structure, coherently offset."""
    xy = grid()
    z = np.random.default_rng(seed).normal(0, 0.02, len(xy))
    for cx, cy in ((6.0, 6.0), (14.0, 8.0), (9.0, 16.0)):
        z[((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2) < 0.9 ** 2] += 0.5
    return np.column_stack([xy, z])


def noisy_roof(sigma=0.15, seed=2):
    """A plate measured badly: rough nearly everywhere, coherent nowhere."""
    xy = grid()
    return np.column_stack([xy, np.random.default_rng(seed).normal(0, sigma, len(xy))])


def coherent_rate(got):
    """Share of *rough* points that read as structure.

    The conditional rate, not ``structure_frac``, is what separates the two
    causes. ``structure_frac`` also carries how much of the roof is rough at
    all, which is a property of the building rather than of the measure.
    """
    return got.structure_frac / max(got.rough_frac, 1e-9)


def test_flat_roof_has_nothing_to_resolve():
    """The commonest Kentucky roof must not read as worth densifying."""
    got = assess_surface_complexity(flat_roof())
    assert got is not None
    assert got.rough_frac < 0.01, f"a plate read {got.rough_frac:.1%} rough"
    assert got.structure_frac < 0.01


def test_structure_and_noise_are_told_apart():
    """Both surfaces are rough. Only one is worth points."""
    vents = assess_surface_complexity(vented_roof())
    noise = assess_surface_complexity(noisy_roof())
    assert vents is not None and noise is not None

    assert noise.rough_frac > 0.5, "the noisy fixture must actually be rough"
    assert vents.rough_frac > 0.02, "the vented fixture must have rough points"
    assert coherent_rate(vents) > 3 * coherent_rate(noise), (
        f"vents {coherent_rate(vents):.2f} vs noise {coherent_rate(noise):.2f}: "
        "the measure is tracking residual magnitude, not its cause")
    assert noise.noise_frac > noise.structure_frac, (
        "independent scatter must land in the noise bin, not the structure bin")


def test_pure_noise_false_positive_floor():
    """A rough point can read coherent by luck. Fix how often.

    Measured, not asserted from theory: the inner neighbourhood also contributes
    to the wide plane it is compared against, so the deviations are not
    independent and the rate is geometry-dependent. Under 20% of rough points on
    a regular grid at this density. Every reported structure fraction carries
    roughly ``0.1 x rough_frac`` of this, so the pilot's 11.5% rough contributes
    about 1 point of its 7.7% structure, and 400 West Market's 70% rough
    contributes about 7 of its 34%.
    """
    noise = assess_surface_complexity(noisy_roof())
    assert 0.0 < coherent_rate(noise) < 0.20, (
        f"noise floor moved to {coherent_rate(noise):.2f}; the survey's "
        "structure fractions are quoted against it")


def test_structure_offset_reports_the_feature_height():
    """The offset is what says whether denser sampling would resolve anything."""
    got = assess_surface_complexity(vented_roof())
    assert got is not None
    # Blocks stand 0.5 m proud; the median is over structure points that
    # straddle the block edges, so it lands below the full step.
    assert 0.03 < got.structure_offset_p50_m < 0.5


def test_too_few_points_returns_none_not_zero():
    """Absence of evidence must not be recorded as evidence of flatness."""
    assert assess_surface_complexity(np.zeros((10, 3))) is None
    xy = grid(side=3.0, spacing=1.0)                       # 9 points
    assert assess_surface_complexity(np.column_stack([xy, np.zeros(len(xy))])) is None


def test_sparse_cloud_returns_none_even_when_populous():
    """Enough points, but none with a neighbourhood tight enough to fit.

    ``fit_local_planes`` needs neighbours within 4 m; scattered over a
    kilometre, 400 points have none, and the fractions would otherwise be
    computed over a handful of accidental pairs.
    """
    scattered = np.random.default_rng(3).uniform(0, 1000, size=(400, 3))
    assert assess_surface_complexity(scattered) is None


def test_thresholds_are_the_documented_ones():
    """The survey and the case studies compare only if the cut points are fixed."""
    assert ROUGH_RESIDUAL_M == pytest.approx(0.08)
    assert STRUCTURE_COHERENCE == pytest.approx(0.7)
    assert NOISE_COHERENCE == pytest.approx(0.4)
    # The gap between the two coherence cuts is deliberate: rough points landing
    # between them are ambiguous and counted in neither bin.
    got = assess_surface_complexity(noisy_roof())
    assert got.structure_frac + got.noise_frac < got.rough_frac


# ------------------------------------------------------- walls versus roof

def box(width=24.0, height=40.0, spacing=0.6, wall_spacing=1.2, seed=4):
    """A flat-roofed box: dense roof, sparse walls, a little ground around it.

    The sparse walls are the point. An aerial scanner sees a vertical face
    nearly edge-on and gets far fewer returns off it than off the roof, which is
    what makes a whole-building average a mix of two different measurements.
    """
    rng = np.random.default_rng(seed)
    g = np.arange(0.0, width, spacing)
    gx, gy = np.meshgrid(g, g)
    roof = np.column_stack([gx.ravel(), gy.ravel(),
                            np.full(gx.size, height) + rng.normal(0, 0.02, gx.size)])
    v = np.arange(0.0, height, wall_spacing)
    wx, wz = np.meshgrid(np.arange(0.0, width, wall_spacing), v)
    wall = np.column_stack([wx.ravel(), np.zeros(wx.size), wz.ravel()])
    gr = np.column_stack([rng.uniform(-6, width + 6, 400),
                          rng.uniform(-6, width + 6, 400), np.zeros(400)])
    return np.vstack([roof, wall, gr])


def test_zones_find_the_roof_and_separate_the_wall():
    from kyfromabove_3d.features import split_surface_zones

    z = split_surface_zones(box())
    assert z is not None
    assert z.plate_z_m == pytest.approx(40.0, abs=0.6), "the roof is the modal band"
    assert z.plate_rms_m < 0.05, "a flat roof fits a plane tightly"
    assert z.roof.sum() > z.wall.sum(), "the fixture's roof is the denser surface"
    assert z.wall.sum() > 0 and z.ground.sum() > 0
    # Every point lands in exactly one zone.
    assert (z.ground.astype(int) + z.wall + z.roof == 1).all()


def test_zone_rates_withhold_a_rate_it_cannot_support():
    """A zone with a handful of points gets ``None``, not a number.

    The pilot has 22 wall returns. A percentage over 22 points invites a
    comparison against a tower's several thousand, which is exactly the mistake
    the split exists to prevent.
    """
    from kyfromabove_3d.features import split_surface_zones

    xyz = box(height=4.0, wall_spacing=2.0)      # low box: almost no wall band
    z = split_surface_zones(xyz)
    rates = z.zone_rates(np.zeros(len(xyz), dtype=bool))
    assert rates["wall_rate"] is None, "a near-empty wall must not be rated"
    assert rates["wall_frac"] is not None, "but its share is still reported"
    assert rates["roof_rate"] is not None
    # On a building this low the roof band reaches below the ground cutoff, and
    # the zones would overlap if each were tested independently. Most of
    # Kentucky's large buildings are under 8 m, so this is the common case.
    assert (z.ground.astype(int) + z.wall + z.roof == 1).all(), (
        "a point landed in two zones, and would be counted twice")


def test_plate_is_the_mode_not_a_high_percentile():
    """A mast above the roof must not drag the roof line up to meet it.

    PNC Tower's roof plate is at 153 m under a 163 m mast carrying 22 points;
    measured at the 98th percentile the roof line lands at 157 m, on neither.
    """
    from kyfromabove_3d.features import split_surface_zones

    xyz = box(height=40.0)
    mast = np.column_stack([np.full(30, 12.0), np.full(30, 12.0),
                            np.linspace(41.0, 50.0, 30)])
    z = split_surface_zones(np.vstack([xyz, mast]))
    assert z.plate_z_m == pytest.approx(40.0, abs=0.6)
    assert z.roof[-30:].all(), "the mast still counts as roof, just does not define it"


# ------------------------------------------------------------ facade extent

def test_facade_ratio_separates_a_warehouse_from_a_tower():
    """The number building selection was missing.

    A roof is scored on roughness. A facade is already flat -- 0.104 m median
    local-plane RMS on the tower shaft -- so roughness says nothing about it;
    what it lacks is colour. The facade term therefore measures how much wall
    there is, and the two shapes have to come out an order of magnitude apart or
    it is not adding anything the roof score did not already say.
    """
    from kyfromabove_3d.features import facade_extent

    # Tractor Supply: 171 m around, 9 m up, on a 1,813 m2 footprint.
    warehouse = facade_extent(perimeter_m=171, height_m=9,
                              footprint_area_m2=1813, n_wall_returns=22)
    # 400 West Market: a near-identical footprint, 164 m up.
    tower = facade_extent(perimeter_m=172, height_m=164,
                          footprint_area_m2=1845, n_wall_returns=6511)

    assert warehouse.to_roof_ratio < 1.0
    assert tower.to_roof_ratio > 10.0
    assert tower.area_m2 > 15 * warehouse.area_m2


def test_capture_separates_a_wall_that_exists_from_one_that_was_flown():
    """Computed area says what exists; returns say what the LiDAR reached.

    The pilot's 1,539 m2 of computed facade carries 22 returns. Treating that as
    1,539 m2 of imaging opportunity would be wrong, and only the second number
    catches it.
    """
    from kyfromabove_3d.features import facade_extent

    on_paper = facade_extent(perimeter_m=171, height_m=9,
                             footprint_area_m2=1813, n_wall_returns=22)
    flown = facade_extent(perimeter_m=180, height_m=156,
                          footprint_area_m2=1724, n_wall_returns=1920)

    assert not on_paper.captured, "22 returns over 1,539 m2 is not a captured wall"
    assert flown.captured
    assert on_paper.returns_per_m2 < flown.returns_per_m2 / 4


def test_facade_extent_survives_a_degenerate_building():
    """Zero height or zero footprint must not raise or divide by zero."""
    from kyfromabove_3d.features import facade_extent

    for kwargs in ({"perimeter_m": 0, "height_m": 10},
                   {"perimeter_m": 100, "height_m": 0}):
        got = facade_extent(footprint_area_m2=500, n_wall_returns=0, **kwargs)
        assert got.area_m2 == 0.0
        assert got.returns_per_m2 == 0.0
        assert got.captured is False
    flat = facade_extent(perimeter_m=100, height_m=5,
                         footprint_area_m2=0, n_wall_returns=10)
    assert flat.to_roof_ratio == 0.0


# ------------------------------------------------------------- facade faces

def rect(w=20.0, d=10.0):
    """A w x d rectangle, counter-clockwise from the origin."""
    return np.array([[0.0, 0.0], [w, 0.0], [w, d], [0.0, d]])


def test_each_wall_is_rated_on_its_own():
    """A whole-building average calls a building captured when half is missing.

    An aircraft flying one direction reaches some faces and not others -- on 400
    West Market only 30% of camera-anchor pairs have the camera on the visible
    side at all. Averaging the face it flew along with the face it never saw is
    the thing this exists to prevent.
    """
    from kyfromabove_3d.features import facade_faces

    north = np.column_stack([np.linspace(1, 19, 300), np.full(300, 10.0)])
    faces = facade_faces(rect(), height_m=30.0, wall_xy=north)

    assert len(faces) == 4
    got = [f for f in faces if f.n_returns > 0]
    assert len(got) == 1, "only the wall the returns are on may be credited"
    assert got[0].bearing_deg == pytest.approx(0.0, abs=1.0), "that wall faces north"
    assert got[0].captured
    assert not any(f.captured for f in faces if f is not got[0])


def test_winding_does_not_change_which_way_a_wall_looks():
    """An outward normal that depends on how the ring was written is useless.

    Overture polygons arrive in both windings, and the bearing is what a later
    imagery pass joins on.
    """
    from kyfromabove_3d.features import facade_faces

    north = np.column_stack([np.linspace(1, 19, 300), np.full(300, 10.0)])
    ccw = facade_faces(rect(), height_m=30.0, wall_xy=north)
    cw = facade_faces(rect()[::-1], height_m=30.0, wall_xy=north)

    assert sorted(round(f.bearing_deg) for f in ccw) == \
           sorted(round(f.bearing_deg) for f in cw)
    assert sum(f.n_returns for f in ccw) == sum(f.n_returns for f in cw)


def test_returns_beyond_the_offset_belong_to_no_wall():
    """An Overture footprint is ML-derived and sits a metre or two off.

    Returns near the drawn wall must be credited to it; a neighbouring
    building's wall well beyond it must not.
    """
    from kyfromabove_3d.features import facade_faces

    near = np.column_stack([np.linspace(1, 19, 200), np.full(200, 11.0)])   # 1 m out
    far = np.column_stack([np.linspace(1, 19, 200), np.full(200, 25.0)])    # 15 m out
    assert sum(f.n_returns for f in
               facade_faces(rect(), 30.0, wall_xy=near, max_offset_m=2.0)) == 200
    assert sum(f.n_returns for f in
               facade_faces(rect(), 30.0, wall_xy=far, max_offset_m=2.0)) == 0


def test_short_segments_are_not_rated():
    """A traced footprint puts two-metre steps on its corners.

    A face that narrow gets a density decided by which side of the corner a
    point was assigned to, not by whether the wall was reached.
    """
    from kyfromabove_3d.features import facade_faces

    notched = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 9.0], [18.0, 9.0],
                        [18.0, 10.0], [0.0, 10.0]])
    faces = facade_faces(notched, height_m=30.0, min_length_m=3.0)
    assert all(f.length_m >= 3.0 for f in faces)
    assert len(faces) < 6, "the notch's short segments must be dropped"


def test_degenerate_footprints_return_nothing():
    from kyfromabove_3d.features import facade_faces

    assert facade_faces(np.zeros((2, 2)), height_m=30.0) == []
    assert facade_faces(rect(), height_m=0.0) == []


# ---------------------------------------------------------- overhang shadow

def box_with_cornice(depth_m=3.0, height_m=30.0, spacing=0.5):
    """A 20 x 10 m box with a cornice of ``depth_m`` along its north face.

    The cornice is roof: horizontal returns at the top, standing proud of the
    wall plane. The wall beneath it is what an aerial scanner cannot reach.
    """
    ax = np.arange(0.5, 20, spacing); ay = np.arange(0.5, 10, spacing)
    gx, gy = np.meshgrid(ax, ay)
    roof = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, height_m)])
    cx, cy = np.meshgrid(ax, np.linspace(10.0, 10.0 + depth_m, 6))
    cornice = np.column_stack([cx.ravel(), cy.ravel(), np.full(cx.size, height_m)])
    wx, wz = np.meshgrid(ax, np.arange(1, height_m, 1.0))
    wall = np.column_stack([wx.ravel(), np.full(wx.size, 10.0), wz.ravel()])
    xyz = np.vstack([roof, cornice, wall])
    wall_mask = np.zeros(len(xyz), dtype=bool)
    wall_mask[-len(wall):] = True
    return xyz, wall_mask, np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]])


def test_a_cornice_shadows_the_wall_beneath_it_and_only_that():
    """The band below a projection is dark from the air; the other walls are not.

    A scanner reaching 20 degrees off vertical clears a 3 m cornice and lands
    3 / tan(20) = 8.2 m below it. Everything between is unreachable.
    """
    from kyfromabove_3d.features import predict_overhang_shadow

    xyz, wm, ring = box_with_cornice(depth_m=3.0, height_m=30.0)
    faces = predict_overhang_shadow(xyz, wm, ring, 30.0, max_off_nadir_deg=20.0)
    by = {f.face: f for f in faces}

    north = by[2]                                   # the face with the cornice
    assert north.shadowed.any()
    rows = np.flatnonzero(north.shadowed.any(axis=1))
    top = north.up_edges_m[rows.max() + 1]
    bottom = north.up_edges_m[rows.min()]
    assert top == pytest.approx(30.0, abs=3.0)
    assert bottom == pytest.approx(30.0 - 3.0 / np.tan(np.radians(20.0)), abs=3.0)
    assert north.shadow_depth_m.max() == pytest.approx(3.0, abs=0.3)

    for i, f in by.items():
        if i != 2:
            assert not f.shadowed.any(), f"face {i} has no cornice and must not be shadowed"


def test_a_setback_shadows_nothing():
    """A crown narrower than its shaft hides no wall. 400 West Market steps in."""
    from kyfromabove_3d.features import predict_overhang_shadow

    xyz, wm, ring = box_with_cornice(depth_m=3.0, height_m=30.0)
    # Move the "cornice" inside the footprint: now it is a setback ledge.
    xyz[:, 1] = np.where((xyz[:, 2] == 30.0) & (xyz[:, 1] > 10.0),
                         10.0 - (xyz[:, 1] - 10.0), xyz[:, 1])
    faces = predict_overhang_shadow(xyz, wm, ring, 30.0)
    assert not any(f.shadowed.any() for f in faces)


def test_a_steeper_scanner_shadows_less():
    """The shadow band scales with 1 / tan(angle): a wider scan reaches further under."""
    from kyfromabove_3d.features import predict_overhang_shadow

    xyz, wm, ring = box_with_cornice(depth_m=3.0, height_m=30.0)
    narrow = predict_overhang_shadow(xyz, wm, ring, 30.0, max_off_nadir_deg=10.0)
    wide = predict_overhang_shadow(xyz, wm, ring, 30.0, max_off_nadir_deg=30.0)
    n = next(f for f in narrow if f.face == 2).shadowed.sum()
    w = next(f for f in wide if f.face == 2).shadowed.sum()
    assert n > w, "a scanner that only looks nearly straight down is shadowed by more"
