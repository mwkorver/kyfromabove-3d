"""Anchors on vertical surfaces.

``build_planar_anchors`` was written for a horizontal building: it admits only
surfaces within ~37 degrees of level and grids cells on (x, y). On a tower that
yields almost nothing -- 24 anchors inside 400 West Market's volume, all on its
crown, none on 165 m of facade -- so the residuals it reports describe the
building's neighbours.

The facade branch is tested on a hollow box, because three of its properties
only have a defined answer when there is a real inside and outside: normals
must point out of the structure, cells must stratify vertically, and the
tangent basis must run along and up the wall rather than in two arbitrary
directions.
"""

import numpy as np
import pytest

from kyfromabove_3d.features import build_planar_anchors, facade_tangent_basis


def hollow_box(side=30.0, height=120.0, spacing=0.7, seed=0):
    """Four vertical walls and a roof, with nothing inside.

    Hollow on purpose: a real LiDAR cloud never contains returns from inside a
    building, and that emptiness is the signal used to orient normals outward.
    """
    rng = np.random.default_rng(seed)
    h = side / 2.0
    zs = np.arange(2.0, height, spacing)
    ts = np.arange(-h, h, spacing)
    walls = []
    for sign in (-1.0, 1.0):
        t, z = np.meshgrid(ts, zs)
        walls.append(np.column_stack([np.full(t.size, sign * h), t.ravel(), z.ravel()]))
        walls.append(np.column_stack([t.ravel(), np.full(t.size, sign * h), z.ravel()]))
    g = np.arange(-h, h, spacing)
    gx, gy = np.meshgrid(g, g)
    roof = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, height)])
    ground = np.column_stack([
        rng.uniform(-60, 60, 4000), rng.uniform(-60, 60, 4000), np.zeros(4000)])
    xyz = np.vstack(walls + [roof, ground])
    xyz += rng.normal(0, 0.01, xyz.shape)
    cls = np.concatenate([np.ones(sum(len(w) for w in walls) + len(roof)),
                          np.full(len(ground), 2)])
    from types import SimpleNamespace
    return SimpleNamespace(xyz_local_m=xyz, classification=cls)


def test_default_behaviour_is_unchanged():
    """The pilot must not move: facades are opt-in."""
    lidar = hollow_box()
    without = build_planar_anchors(lidar)
    assert all(a.label != "facade" for a in without)

    with_facades = build_planar_anchors(lidar, include_facades=True)
    assert len(with_facades) > len(without)
    # Every anchor the horizontal pass found is still there, unchanged.
    assert [a.label for a in with_facades[:len(without)]] == [a.label for a in without]


def test_facade_anchors_span_the_wall_vertically():
    """A planimetric grid returns per_cell anchors for a whole column."""
    lidar = hollow_box(height=120.0)
    anchors = [a for a in build_planar_anchors(lidar, include_facades=True)
               if a.label == "facade"]
    assert len(anchors) > 50, f"only {len(anchors)} facade anchors"
    z = np.array([a.xyz_local_m[2] for a in anchors])
    assert z.max() - z.min() > 80.0, "anchors must climb the wall, not cluster"
    # Every 20 m band of the wall should be represented.
    for lo in range(10, 100, 20):
        assert ((z >= lo) & (z < lo + 20)).sum() > 0, f"no anchors in {lo}-{lo + 20} m"


def test_facade_normals_point_out_of_the_building():
    """The sign says which side a camera must be on; an arbitrary one is useless."""
    lidar = hollow_box()
    anchors = [a for a in build_planar_anchors(lidar, include_facades=True)
               if a.label == "facade"]
    outward = 0
    for a in anchors:
        p = np.asarray(a.xyz_local_m)
        n = np.asarray(a.normal)
        # On a box centred at the origin, outward is away from the axis.
        radial = np.array([p[0], p[1], 0.0])
        if np.linalg.norm(radial) < 1e-6:
            continue
        if np.dot(n[:2], radial[:2] / np.linalg.norm(radial[:2])) > 0:
            outward += 1
    assert outward / len(anchors) > 0.9, \
        f"only {100 * outward / len(anchors):.0f}% of normals point outward"


def test_facade_basis_runs_along_and_up_the_wall():
    """So offset_e1_m and offset_e2_m mean something on a facade."""
    for normal in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.05]):
        e1, e2 = facade_tangent_basis(np.array(normal))
        assert abs(e1[2]) < 1e-9, "e1 must be horizontal -- along the wall"
        assert e2[2] > 0.9, "e2 must run up the wall"
        assert abs(np.dot(e1, e2)) < 1e-9
        assert abs(np.dot(e1, normal / np.linalg.norm(normal))) < 1e-9

    # A horizontal plane has no "along the wall"; it must fall back, not divide.
    e1, e2 = facade_tangent_basis(np.array([0.0, 0.0, 1.0]))
    assert np.isfinite(e1).all() and np.isfinite(e2).all()
    assert abs(np.dot(e1, e2)) < 1e-9


def test_facade_thresholds_are_separate_from_roof_thresholds():
    """A facade is noisier than a roof; one threshold cannot serve both."""
    lidar = hollow_box()
    # The fixture's walls sit near 0.003 m RMS, so 0.0001 must exclude them all.
    strict = build_planar_anchors(lidar, include_facades=True,
                                  facade_max_planarity_rms_m=0.0001)
    assert not [a for a in strict if a.label == "facade"]

    loose = build_planar_anchors(lidar, include_facades=True,
                                 facade_max_planarity_rms_m=0.20)
    assert [a for a in loose if a.label == "facade"]

    # facade_max_normal_z is an upper bound on |n_z|, so every facade anchor
    # returned must honour it. It cannot exclude a perfectly vertical wall,
    # whose |n_z| is ~0 and passes any threshold -- it excludes sloped surfaces.
    for limit in (0.35, 0.05):
        got = [a for a in build_planar_anchors(lidar, include_facades=True,
                                               facade_max_normal_z=limit)
               if a.label == "facade"]
        assert got
        assert all(abs(a.normal[2]) < limit for a in got)


# ------------------------------------------------------------ back-face culling

def test_back_face_culling_is_inert_for_roofs_and_active_for_facades():
    """A roof faces every aerial camera; a wall faces one way.

    Without culling, the ~70% of facade anchor/camera pairs whose camera is
    behind the wall are attempted anyway and rejected as ``low_ncc``. They are
    not matching failures -- they are pairs that should never have been tried,
    and counting them depresses the acceptance rate and hides the real one.
    """
    from types import SimpleNamespace

    from kyfromabove_3d.qa import match_anchor_between_views

    def anchor(normal):
        return SimpleNamespace(anchor_id="A", label="facade",
                               xyz_local_m=np.array([0.0, 0.0, 50.0]),
                               normal=np.asarray(normal, dtype=float),
                               e1=np.array([0.0, 1.0, 0.0]),
                               e2=np.array([0.0, 0.0, 1.0]))

    def camera(centre, name):
        return SimpleNamespace(frame_id=name,
                               camera_center_local_m=np.asarray(centre, dtype=float))

    east = camera([200.0, 0.0, 300.0], "east")
    west = camera([-200.0, 0.0, 300.0], "west")
    wall_facing_east = anchor([1.0, 0.0, 0.0])

    # The west camera is behind this wall: culled before any image is touched,
    # which is also why None windows are safe to pass here.
    res = match_anchor_between_views(wall_facing_east, west, None, east, None,
                                     min_facing_cos=0.20)
    assert res.accepted is False and res.reject_reason == "back_face"

    # The east camera can see it, so the pair proceeds. These doubles have no
    # .project, so reaching patch sampling raises -- which is the assertion:
    # the cull did not fire, rather than a claim about what the match returns.
    with pytest.raises(AttributeError):
        match_anchor_between_views(wall_facing_east, east, None, east, None,
                                   min_facing_cos=0.20)

    # Culling off (the default) must not change existing behaviour at all: even
    # the back-facing pair is attempted.
    with pytest.raises(AttributeError):
        match_anchor_between_views(wall_facing_east, west, None, east, None,
                                   min_facing_cos=0.0)

    # A roof normal faces up and every aerial camera is above it, so culling is
    # inert for the horizontal anchors the pilot is built from.
    roof = anchor([0.0, 0.0, 1.0])
    for cam in (east, west):
        with pytest.raises(AttributeError):
            match_anchor_between_views(roof, cam, None, cam, None,
                                       min_facing_cos=0.20)


def test_acceptance_is_measured_over_geometrically_possible_pairs():
    """A back-face rejection is not a failed match, so it is not a denominator.

    Counting it as one is like scoring a photograph that does not contain the
    building. On 400 West Market it turns a true 70% acceptance into 33%.
    """
    from kyfromabove_3d.qa import MatchResult, ReprojectionReport

    def m(accepted, reason=None, residual=1.0):
        return MatchResult(anchor_id="A", label="facade", ref_frame="a",
                           tgt_frame="b", accepted=accepted, reject_reason=reason,
                           ncc=0.9 if accepted else None,
                           offset_e1_m=0.0, offset_e2_m=0.0,
                           residual_m=residual * 0.05, residual_px=residual,
                           observed_uv=(0.0, 0.0) if accepted else None)

    matches = ([m(True)] * 7 + [m(False, "low_ncc")] * 3
               + [m(False, "back_face")] * 90)
    s = ReprojectionReport(matches=matches).stats()
    assert s["attempted"] == 100
    assert s["back_face"] == 90
    assert s["eligible"] == 10
    assert s["acceptance_rate"] == pytest.approx(0.7), \
        "7 of 10 possible pairs matched, not 7 of 100 attempts"

    # With no back faces -- every roof-only AOI -- nothing changes.
    plain = ReprojectionReport(matches=[m(True)] * 9 + [m(False, "low_ncc")]).stats()
    assert plain["back_face"] == 0
    assert plain["eligible"] == plain["attempted"] == 10
    assert plain["acceptance_rate"] == pytest.approx(0.9)
