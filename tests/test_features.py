"""Walls as vertical planes, and lifting camera rays onto them."""

from __future__ import annotations

import numpy as np


# --- walls as planes ---------------------------------------------------------

def _square():
    """A 20 m square footprint, corners at (0,0) (20,0) (20,20) (0,20)."""
    return np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]])


def test_wall_planes_face_outward_whichever_way_the_ring_is_wound():
    from kyfromabove_3d.features import wall_planes

    for ring in (_square(), _square()[::-1]):
        planes = wall_planes(ring, z_lo=0.0, z_hi=50.0)
        assert len(planes) == 4
        for pl in planes:
            mid = (pl.p0 + pl.p1) / 2.0
            # Stepping along the normal from the wall's middle leaves the square.
            out = mid + pl.normal * 1.0
            assert not (0 < out[0] < 20 and 0 < out[1] < 20), (pl.face, out)
            assert abs(np.linalg.norm(pl.normal) - 1.0) < 1e-9
        assert {round(pl.length_m) for pl in planes} == {20}


def test_ray_to_wall_planes_hits_the_near_wall_only():
    from kyfromabove_3d.features import ray_to_wall_planes, wall_planes

    planes = wall_planes(_square(), z_lo=0.0, z_hi=50.0)
    C = np.array([-30.0, 10.0, 25.0])                     # west of the square, mid-height
    rays = np.array([
        [1.0, 0.0, 0.0],                                  # straight at the west wall
        [1.0, 0.0, 1.0],                                  # climbs: hits the wall at z=55? no -- above the roof
        [1.0, 0.0, -0.5],                                 # descends: hits the west wall at z=10
        [-1.0, 0.0, 0.0],                                 # away from the building
        [1.0, 2.0, 0.0],                                  # passes north of the square: y=70 at x=0
    ])
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    pts, hit, face = ray_to_wall_planes(C, rays, planes)
    assert hit.tolist() == [True, False, True, False, False]
    assert np.allclose(pts[0], [0.0, 10.0, 25.0])         # the near (west) wall, not the far one at x=20
    assert np.allclose(pts[2], [0.0, 10.0, 10.0])
    assert np.isnan(pts[1]).all()
    west = next(pl.face for pl in planes if pl.normal[0] < -0.9)
    assert face[0] == west and face[2] == west


def test_ray_to_wall_planes_across_a_corner_takes_the_first_wall_met():
    from kyfromabove_3d.features import ray_to_wall_planes, wall_planes

    planes = wall_planes(_square(), z_lo=0.0, z_hi=50.0)
    C = np.array([-10.0, -10.0, 20.0])                    # south-west of the square
    # Aimed at (5, 20-ish): crosses the south wall's plane (y=0) at x=... no --
    # aimed to cross y=0 at x=-3 (outside the south wall's extent) and then
    # x=0 at y=... a ray towards (0+eps, 6): meets the west wall inside its extent.
    d = np.array([[10.0, 16.0, 0.0], [16.0, 10.0, 0.0]])  # first: west wall; second: south wall
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    pts, hit, face = ray_to_wall_planes(C, d, planes)
    assert hit.all()
    west = next(pl.face for pl in planes if pl.normal[0] < -0.9)
    south = next(pl.face for pl in planes if pl.normal[1] < -0.9)
    assert face.tolist() == [west, south]
    assert abs(pts[0, 0]) < 1e-9 and abs(pts[1, 1]) < 1e-9


# --- verify_face_plane -------------------------------------------------------

def test_verify_face_plane_offset_shift():
    from kyfromabove_3d.features import WallPlane, verify_face_plane

    plane = WallPlane(
        face=0,
        p0=np.array([0.0, 0.0]),
        p1=np.array([30.0, 0.0]),
        normal=np.array([0.0, -1.0]),
        z_lo=0.0,
        z_hi=60.0,
    )
    # 3 tiles shifted by 0.2m along outward normal [0, -1]
    centroids = np.array([
        [5.0, -0.2, 10.0],
        [15.0, -0.2, 25.0],
        [25.0, -0.2, 40.0],
    ])
    normals = np.array([
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
    ])
    vp = verify_face_plane(plane, centroids, normals, tile_m=12.0, roof_z=60.0)
    assert vp is not None
    assert abs(vp.offset_m - 0.2) < 1e-3
    assert vp.rms_m < 1e-3
    assert vp.yaw_dev_deg < 0.1
    assert abs(vp.z_top - 46.0) < 1e-3  # 40.0 + 12/2
    assert np.allclose(vp.normal, [0.0, -1.0])
    assert np.allclose(vp.p0, [0.0, -0.2])
    assert np.allclose(vp.p1, [30.0, -0.2])


def test_verify_face_plane_setback_fails():
    from kyfromabove_3d.features import WallPlane, verify_face_plane

    plane = WallPlane(
        face=0,
        p0=np.array([0.0, 0.0]),
        p1=np.array([30.0, 0.0]),
        normal=np.array([0.0, -1.0]),
        z_lo=0.0,
        z_hi=60.0,
    )
    # Graves setback: step of 1.0 m between lower and upper tiles
    centroids = np.array([
        [10.0, 0.0, 10.0],
        [20.0, 0.0, 10.0],
        [15.0, 1.0, 30.0],
    ])
    normals = np.tile([0.0, -1.0, 0.0], (3, 1))
    vp = verify_face_plane(plane, centroids, normals, max_offset_rms_m=0.30)
    assert vp is None


def test_verify_face_plane_single_tile_fails():
    from kyfromabove_3d.features import WallPlane, verify_face_plane

    plane = WallPlane(
        face=0,
        p0=np.array([0.0, 0.0]),
        p1=np.array([30.0, 0.0]),
        normal=np.array([0.0, -1.0]),
        z_lo=0.0,
        z_hi=60.0,
    )
    centroids = np.array([[15.0, 0.0, 20.0]])
    normals = np.array([[0.0, -1.0, 0.0]])
    assert verify_face_plane(plane, centroids, normals) is None


def test_verify_face_plane_yaw_refinement():
    from kyfromabove_3d.features import WallPlane, verify_face_plane

    plane = WallPlane(
        face=0,
        p0=np.array([0.0, 0.0]),
        p1=np.array([30.0, 0.0]),
        normal=np.array([0.0, -1.0]),
        z_lo=0.0,
        z_hi=60.0,
    )
    # 1.0 degree yaw tilt: tan(1 deg) ~ 0.017455
    k = np.tan(np.radians(1.0))
    s = np.array([5.0, 15.0, 25.0])
    # Along normal: d = k * s -> points are at y = -d
    d = k * s
    centroids = np.column_stack([s, -d, [15.0, 25.0, 35.0]])
    # Outward normal rotated by 1.0 deg: line has slope -k, outward normal is [-sin(rad), -cos(rad)]
    rad = np.radians(1.0)
    n_expected = np.array([-np.sin(rad), -np.cos(rad), 0.0])
    normals = np.tile(n_expected, (3, 1))

    vp = verify_face_plane(plane, centroids, normals, max_normal_dev_deg=3.0)
    assert vp is not None
    assert abs(vp.yaw_dev_deg - 1.0) < 0.05
    assert vp.rms_m < 1e-4
    assert np.allclose(vp.normal, n_expected[:2], atol=1e-3)


def test_wall_planes_save_and_load(tmp_path):
    from kyfromabove_3d.features import (
        WallPlane,
        VerifiedWallPlane,
        save_wall_planes,
        load_wall_planes,
    )

    planes = [
        WallPlane(
            face=0,
            p0=np.array([0.0, 0.0]),
            p1=np.array([20.0, 0.0]),
            normal=np.array([0.0, -1.0]),
            z_lo=10.0,
            z_hi=50.0,
        ),
        VerifiedWallPlane(
            face=1,
            p0=np.array([20.2, 0.0]),
            p1=np.array([20.2, 30.0]),
            normal=np.array([1.0, 0.0]),
            z_lo=10.0,
            z_hi=45.0,
            n_tiles=12,
            rms_m=0.08,
            max_normal_dev_deg=1.2,
            z_top=45.0,
            offset_m=0.2,
            yaw_dev_deg=0.5,
        ),
    ]

    path = tmp_path / "wall_planes.json"
    save_wall_planes(path, planes)
    loaded = load_wall_planes(path)

    assert len(loaded) == 2
    assert isinstance(loaded[0], WallPlane) and not isinstance(loaded[0], VerifiedWallPlane)
    assert isinstance(loaded[1], VerifiedWallPlane)

    assert loaded[0].face == 0
    assert np.allclose(loaded[0].p0, [0.0, 0.0])
    assert loaded[0].z_hi == 50.0

    assert loaded[1].face == 1
    assert loaded[1].n_tiles == 12
    assert abs(loaded[1].rms_m - 0.08) < 1e-5
    assert abs(loaded[1].offset_m - 0.2) < 1e-5
    assert abs(loaded[1].z_top - 45.0) < 1e-5
    assert loaded[1].inferred is False


def test_infer_prism_faces():
    from kyfromabove_3d.features import (
        WallPlane,
        VerifiedWallPlane,
        infer_prism_faces,
    )

    # 4-sided rectangular footprint:
    # Face 0: (0, 0) -> (10, 0), length 10
    # Face 1: (10, 0) -> (10, 30), length 30
    # Face 2: (10, 30) -> (0, 30), length 10
    # Face 3: (0, 30) -> (0, 0), length 30
    # Total perimeter = 80 m
    nominal = [
        WallPlane(0, np.array([0.0, 0.0]), np.array([10.0, 0.0]), np.array([0.0, -1.0]), 0.0, 50.0),
        WallPlane(1, np.array([10.0, 0.0]), np.array([10.0, 30.0]), np.array([1.0, 0.0]), 0.0, 50.0),
        WallPlane(2, np.array([10.0, 30.0]), np.array([0.0, 30.0]), np.array([0.0, 1.0]), 0.0, 50.0),
        WallPlane(3, np.array([0.0, 30.0]), np.array([0.0, 0.0]), np.array([-1.0, 0.0]), 0.0, 50.0),
    ]

    # Scenario A: Only Face 1 verified (30 m / 80 m = 37.5% < 60%).
    # Prism completion should decline to trigger (Humana Building case).
    verified_a = {
        1: VerifiedWallPlane(1, np.array([10.1, 0.0]), np.array([10.1, 30.0]),
                             np.array([1.0, 0.0]), 0.0, 48.0, n_tiles=5, z_top=48.0),
    }
    inferred_a = infer_prism_faces(nominal, verified_a, min_perimeter_ratio=0.60)
    assert len(inferred_a) == 0, "Prism prior must not trigger if verified perimeter < 60%"

    # Scenario B: Face 1 and Face 3 verified (30 + 30 = 60 m / 80 m = 75% >= 60%).
    # Prism completion triggers and completes Face 0 and Face 2 (PNC Tower case).
    verified_b = {
        1: VerifiedWallPlane(1, np.array([10.1, 0.0]), np.array([10.1, 30.0]),
                             np.array([1.0, 0.0]), 0.0, 48.0, n_tiles=5, z_top=48.0),
        3: VerifiedWallPlane(3, np.array([-0.1, 30.0]), np.array([-0.1, 0.0]),
                             np.array([-1.0, 0.0]), 0.0, 46.0, n_tiles=5, z_top=46.0),
    }
    inferred_b = infer_prism_faces(nominal, verified_b, min_perimeter_ratio=0.60)
    assert len(inferred_b) == 2
    assert 0 in inferred_b and 2 in inferred_b

    # Face 0 bridges from Face 3's p1 to Face 1's p0
    inf0 = inferred_b[0]
    assert inf0.inferred is True
    assert np.allclose(inf0.p0, [-0.1, 0.0])
    assert np.allclose(inf0.p1, [10.1, 0.0])
    assert np.allclose(inf0.normal, [0.0, -1.0])
    # z_top is min of adjacent (46 and 48) -> 46.0
    assert abs(inf0.z_top - 46.0) < 1e-4

    # Face 2 bridges from Face 1's p1 to Face 3's p0
    inf2 = inferred_b[2]
    assert inf2.inferred is True
    assert np.allclose(inf2.p0, [10.1, 30.0])
    assert np.allclose(inf2.p1, [-0.1, 30.0])
    assert np.allclose(inf2.normal, [0.0, 1.0])
    assert abs(inf2.z_top - 46.0) < 1e-4



# --- a face as a profile of bands ------------------------------------------------

def _wall_returns(plane, z_ranges, offset, along=(0.0, 1.0), per_band=80, seed=0):
    """Returns near a footprint edge: at ``offset`` m outward, within the
    given z ranges, across a fraction of the edge, with a little noise."""
    rng = np.random.default_rng(seed)
    seg = plane.p1 - plane.p0
    L = np.hypot(*seg)
    e1 = seg / L
    out = []
    for z0, z1 in z_ranges:
        n = int(per_band * (z1 - z0) / 5.0)
        s = rng.uniform(along[0] * L, along[1] * L, n)
        z = rng.uniform(z0, z1, n)
        o = offset + rng.normal(0, 0.1, n)
        xy = plane.p0 + s[:, None] * e1 + o[:, None] * plane.normal
        out.append(np.column_stack([xy, z]))
    return np.vstack(out)


def test_face_profile_observes_bands_and_bridges_the_gap_the_flight_missed():
    from kyfromabove_3d.features import face_profile, wall_planes

    pl = wall_planes(_square(), z_lo=0.0, z_hi=100.0)[0]
    # Seen from the ground to 40 m and from 80 m up, 1.5 m outside the edge; nothing between.
    xyz = _wall_returns(pl, [(0, 40), (80, 100)], offset=1.5)
    bands = face_profile(pl, xyz, band_m=5.0)
    status = {(b.z_lo, b.z_hi): b.status for b in bands}
    assert status[(0.0, 5.0)] == "observed" and status[(35.0, 40.0)] == "observed"
    assert status[(80.0, 85.0)] == "observed" and status[(95.0, 100.0)] == "observed"
    assert all(status[(z, z + 5.0)] == "bridged" for z in range(40, 80, 5))
    assert all(abs(b.offset_m - 1.5) < 0.15 for b in bands if b.status != "unknown")


def test_face_profile_does_not_bridge_a_step_and_stops_at_the_top_it_saw():
    from kyfromabove_3d.features import face_profile, wall_planes

    pl = wall_planes(_square(), z_lo=0.0, z_hi=100.0)[0]
    # A podium wall at +1.5 m to 30 m, a set-back shaft at -4 m from 50 m to 80 m, nothing above.
    xyz = np.vstack([_wall_returns(pl, [(0, 30)], offset=1.5), _wall_returns(pl, [(50, 80)], offset=-4.0, seed=1)])
    bands = face_profile(pl, xyz, band_m=5.0)
    status = {b.z_lo: b.status for b in bands}
    assert status[30.0] == "unknown" and status[45.0] == "unknown", "a 5.5 m step is not bridged"
    assert status[85.0] == "unknown" and status[95.0] == "unknown", "nothing above the highest observed band"


def test_face_profile_marks_a_loggia_open():
    from kyfromabove_3d.features import extrude_profile, face_profile, wall_planes

    pl = wall_planes(_square(), z_lo=0.0, z_hi=60.0)[0]
    # Solid wall to 40 m; from 40 to 60 m piers at the face AND a recess 6 m behind, equally strong.
    xyz = np.vstack([_wall_returns(pl, [(0, 40)], offset=0.0),
                     _wall_returns(pl, [(40, 60)], offset=0.0, seed=2),
                     _wall_returns(pl, [(40, 60)], offset=-6.0, seed=3)])
    bands = face_profile(pl, xyz, band_m=5.0)
    assert [b.status for b in bands if b.z_lo >= 40] == ["open"] * 4
    pts = extrude_profile(pl, bands, spacing_m=0.5)
    # the open bands are extruded at the wall BEHIND the piers, 6 m back,
    # never at the piers' line
    above = pts[pts[:, 2] >= 40.0]
    off = (above[:, :2] - pl.p0) @ pl.normal
    assert len(above) and np.allclose(off, -6.0, atol=0.6)
    assert all(abs(b.back_offset_m + 6.0) < 0.6 for b in bands if b.z_lo >= 40)


def test_profile_keeps_trims_only_on_evidence():
    from kyfromabove_3d.features import face_profile, profile_keeps, wall_planes

    big = _square() * 5.0                                   # a 100 m edge, like a real face
    pl = wall_planes(big, z_lo=0.0, z_hi=60.0)[0]
    seg = pl.p1 - pl.p0
    L = np.hypot(*seg)
    # Dense returns on the east half only above 30 m (the shaft is narrower than the podium);
    # full width below; per_band=400 over 50 m x 5 m is 1.6 returns/m2 -- dense.
    xyz = np.vstack([_wall_returns(pl, [(0, 30)], offset=0.0, per_band=400),
                     _wall_returns(pl, [(30, 60)], offset=0.0, along=(0.5, 1.0), per_band=400, seed=4)])
    bands = face_profile(pl, xyz, band_m=5.0)
    s = np.linspace(0.5, L - 0.5, 40)
    z = np.linspace(0.5, 59.5, 60)
    S, Z = np.meshgrid(s, z)
    grid = np.column_stack([pl.p0 + S.ravel()[:, None] * (seg / L), Z.ravel()])
    keep = profile_keeps(pl, bands, grid)
    along = S.ravel() / L
    assert keep[(Z.ravel() < 30)].all(), "full-width bands keep everything"
    assert not keep[(Z.ravel() > 35) & (along < 0.35)].any(), "the west half above the step is not wall"
    assert keep[(Z.ravel() > 35) & (along > 0.55)].all()
    # Sparse returns (a glass tower, 0.14 per m2): the same half-extent is not trusted, nothing is trimmed.
    thin = np.vstack([_wall_returns(pl, [(0, 60)], offset=0.0, along=(0.5, 1.0), per_band=35, seed=5)])
    bands_thin = face_profile(pl, thin, band_m=5.0)
    assert profile_keeps(pl, bands_thin, grid).all()


def test_face_profile_does_not_put_panes_on_a_dome():
    """A crown that recedes a metre or two every band is not a wall; its
    bands are sloped, and nothing vertical is extruded through them."""
    from kyfromabove_3d.features import extrude_profile, face_profile, wall_planes

    pl = wall_planes(_square() * 5.0, z_lo=0.0, z_hi=60.0)[0]
    parts = [_wall_returns(pl, [(0, 30)], offset=0.0, per_band=200)]              # a wall
    for k, z0 in enumerate(range(30, 60, 5)):                                       # then a dome: 1.5 m in per band
        parts.append(_wall_returns(pl, [(z0, z0 + 5)], offset=-1.5 * (k + 1), per_band=200, seed=10 + k))
    bands = face_profile(pl, np.vstack(parts), band_m=5.0)
    status = {b.z_lo: b.status for b in bands}
    assert all(status[z] == "observed" for z in (0.0, 10.0, 25.0))
    assert all(status[z] == "sloped" for z in (30.0, 40.0, 50.0)), status
    pts = extrude_profile(pl, bands, spacing_m=0.5)
    assert pts[:, 2].max() < 30.5


def test_face_profile_extends_a_wall_down_to_the_ground_through_thin_bands():
    """Few returns at street level, but at the wall's offset: the wall goes
    on down. Without this the extrusion started at the first well-seen band
    and hung over the ground floor like a panel."""
    from kyfromabove_3d.features import extrude_profile, face_profile, wall_planes

    pl = wall_planes(_square() * 5.0, z_lo=0.0, z_hi=60.0)[0]
    xyz = np.vstack([_wall_returns(pl, [(15, 60)], offset=1.8, per_band=80),
                     _wall_returns(pl, [(0, 15)], offset=1.8, per_band=12, seed=7)])   # 12 per band: thin
    bands = face_profile(pl, xyz, band_m=5.0)
    status = {b.z_lo: b.status for b in bands}
    assert status[15.0] == "observed"
    assert status[0.0] == "extended" and status[5.0] == "extended" and status[10.0] == "extended", status
    pts = extrude_profile(pl, bands, spacing_m=0.5)
    assert pts[:, 2].min() < 1.0
    # With nothing at all below, nothing is extended.
    bands2 = face_profile(pl, _wall_returns(pl, [(15, 60)], offset=1.8, per_band=80), band_m=5.0)
    assert {b.z_lo: b.status for b in bands2}[5.0] == "unknown"


def test_face_profile_keeps_a_recessed_bay_recessed():
    """A wall at the footprint line with a 12 m bay set 4 m back down its
    middle: the band's offset is the wall's, the bay is a segment at its own
    offset, and the extrusion steps back where the bay is."""
    from kyfromabove_3d.features import extrude_profile, face_profile, wall_planes

    pl = wall_planes(_square() * 5.0, z_lo=0.0, z_hi=40.0)[0]      # a 100 m edge
    wall = np.vstack([_wall_returns(pl, [(0, 40)], offset=0.0, along=(0.0, 0.44), per_band=150),
                      _wall_returns(pl, [(0, 40)], offset=0.0, along=(0.56, 1.0), per_band=150, seed=2)])
    bay = _wall_returns(pl, [(0, 40)], offset=-4.0, along=(0.44, 0.56), per_band=60, seed=3)
    bands = face_profile(pl, np.vstack([wall, bay]), band_m=5.0)
    assert all(b.status == "observed" for b in bands)
    for b in bands:
        assert abs(b.offset_m) < 0.2
        assert len(b.segments) == 1, b.segments
        a0, a1, off = b.segments[0]
        assert 40 <= a0 <= 46 and 54 <= a1 <= 60 and abs(off + 4.0) < 0.3
    pts = extrude_profile(pl, bands, spacing_m=0.5)
    seg = pl.p1 - pl.p0; L = np.hypot(*seg)
    along = ((pts[:, :2] - pl.p0) @ seg) / L
    offs = (pts[:, :2] - pl.p0) @ pl.normal
    assert np.all(np.abs(offs[(along > 47) & (along < 53)] + 4.0) < 0.3), "the bay is extruded 4 m back"
    assert np.all(np.abs(offs[(along < 40) | (along > 60)]) < 0.3), "the wall either side stays put"


def test_face_profile_rejects_a_roof_slab_as_a_wall_band():
    """A flat slab caught at one offset sits at one height; a wall's
    returns run up through the band. Humana's porch roof, 25 m inside the
    north edge, read as a wall band until this."""
    from kyfromabove_3d.features import face_profile, wall_planes

    pl = wall_planes(_square() * 5.0, z_lo=0.0, z_hi=40.0)[0]
    rng = np.random.default_rng(11)
    seg = pl.p1 - pl.p0; L = np.hypot(*seg); e1 = seg / L
    n = 400
    s = rng.uniform(5, L - 5, n)
    slab = pl.p0 + s[:, None] * e1 + rng.uniform(-6.0, -2.0, n)[:, None] * pl.normal    # a roof, 2-6 m inside
    slab = np.column_stack([slab, np.full(n, 21.0) + rng.normal(0, 0.15, n)])              # all at 21 m
    wall = _wall_returns(pl, [(0, 15)], offset=0.0, per_band=100)
    bands = face_profile(pl, np.vstack([wall, slab]), band_m=5.0)
    status = {b.z_lo: b.status for b in bands}
    assert status[0.0] == "observed" and status[10.0] == "observed"
    assert status[20.0] != "observed", status


def test_neighbour_clip_narrows_a_wall_where_the_next_face_stands_back():
    """The east wall of a square building runs the full edge, but above the
    podium the north wall stands 20 m back from the north corner: at those
    heights the east wall stops 20 m short of that corner."""
    from kyfromabove_3d.features import (FaceBand, band_extent, clip_profiles_by_neighbours,
                                          extrude_profile, wall_planes)

    planes = wall_planes(_square() * 5.0, z_lo=0.0, z_hi=40.0)     # 100 m square, faces 0..3 CCW
    east = next(p for p in planes if p.normal[0] > 0.9)
    north = next(p for p in planes if p.normal[1] > 0.9)
    L = 100.0
    def band(z0, off, status="observed"):
        return FaceBand(z0, z0 + 10.0, off, 100, 0.1, 2.0, 98.0, status)
    profiles = {
        east.face: [band(0, 0.0), band(10, 0.0), band(20, 0.0), band(30, 0.0)],
        north.face: [band(0, 0.0), band(10, 0.0), band(20, -20.0), band(30, -20.0)],   # set back above 20 m
    }
    n = clip_profiles_by_neighbours(planes, profiles)
    assert n == 2
    # The east face's p1 corner is the north-east corner (CCW ring), shared with the north face's p0.
    assert np.linalg.norm(east.p1 - north.p0) < 1e-9
    e = profiles[east.face]
    assert e[0].clip_hi is None and e[1].clip_hi is None
    assert abs(e[2].clip_hi - (L - 20.0)) < 1e-9 and abs(e[3].clip_hi - (L - 20.0)) < 1e-9
    assert band_extent(east, e[2]) == (0.0, 80.0)
    pts = extrude_profile(east, e, spacing_m=1.0)
    along = ((pts[:, :2] - east.p0) @ (east.p1 - east.p0)) / L
    assert along[pts[:, 2] < 20].max() > 98.0
    assert along[pts[:, 2] >= 20].max() <= 80.0


def _profiles_for_square(observed_west, observed_east, band_m=10.0, z_hi=60.0):
    """Profiles for the west and east faces of a 100 m square from
    {z_lo: offset} dicts; every other band unknown."""
    from kyfromabove_3d.features import FaceBand, wall_planes

    planes = wall_planes(_square() * 5.0, z_lo=0.0, z_hi=z_hi)
    west = next(p for p in planes if p.normal[0] < -0.9)
    east = next(p for p in planes if p.normal[0] > 0.9)
    def bands(obs):
        out = []
        z = 0.0
        while z < z_hi:
            if z in obs:
                out.append(FaceBand(z, z + band_m, obs[z], 100, 0.1, 5.0, 95.0, "observed"))
            else:
                out.append(FaceBand(z, z + band_m, None, 3, 0.0, 0.0, 0.0, "unknown"))
            z += band_m
        return out
    return planes, west, east, {west.face: bands(observed_west), east.face: bands(observed_east)}


def test_mirror_fills_a_hidden_face_from_its_twin_with_the_same_shape():
    """The west wall: flat to 30 m, then set back 4 m. The east wall was seen
    at the bottom and the top only, with the same setback; its middle is
    filled from the west, shifted to the east wall's own anchor."""
    from kyfromabove_3d.features import mirror_hidden_bands

    west_obs = {0.0: 0.0, 10.0: 0.0, 20.0: 0.0, 30.0: -4.0, 40.0: -4.0, 50.0: -4.0}
    east_obs = {0.0: 1.0, 10.0: 1.0, 20.0: 1.0, 50.0: -3.0}          # a footprint drawn 1 m off on this side
    planes, west, east, profiles = _profiles_for_square(west_obs, east_obs)
    rep = mirror_hidden_bands(planes, profiles)
    assert rep[east.face]["symmetric"] and rep[east.face]["mirrored"] == 2, rep
    e = {b.z_lo: b for b in profiles[east.face]}
    assert e[30.0].status == "mirrored" and abs(e[30.0].offset_m - (-3.0)) < 1e-6
    assert e[40.0].status == "mirrored" and abs(e[40.0].offset_m - (-3.0)) < 1e-6
    assert e[0.0].status == "observed", "a face's own evidence is never replaced"


def test_mirror_declines_when_the_faces_differ_in_shape():
    from kyfromabove_3d.features import mirror_hidden_bands

    west_obs = {0.0: 0.0, 10.0: 0.0, 20.0: 0.0, 30.0: -4.0, 40.0: -4.0}
    east_obs = {0.0: 0.0, 10.0: 0.0, 20.0: 0.0, 30.0: 0.0, 50.0: 0.0}   # the east wall does not step back
    planes, west, east, profiles = _profiles_for_square(west_obs, east_obs)
    rep = mirror_hidden_bands(planes, profiles)
    assert rep[east.face]["symmetric"] is False and rep[east.face]["mirrored"] == 0
    assert profiles[east.face][4].status == "unknown"


def test_mirror_takes_the_opposite_offsets_when_a_face_was_never_seen():
    from kyfromabove_3d.features import mirror_hidden_bands

    west_obs = {0.0: 0.0, 10.0: 0.0, 20.0: -2.0, 30.0: -2.0, 40.0: -2.0, 50.0: -2.0}
    planes, west, east, profiles = _profiles_for_square(west_obs, {})
    rep = mirror_hidden_bands(planes, profiles)
    assert rep[east.face]["mirrored"] == 6 and rep[east.face]["anchored"] is False
    e = profiles[east.face]
    assert all(b.status == "mirrored" for b in e)
    assert abs(e[0].offset_m) < 1e-6 and abs(e[3].offset_m + 2.0) < 1e-6
    # The extent is mirrored along the edge: the west's 5-95 becomes the east's 5-95 of a 100 m edge.
    assert abs(e[0].along_lo - 5.0) < 1e-6 and abs(e[0].along_hi - 95.0) < 1e-6


def test_mirror_replaces_a_bridge_where_the_twin_saw_a_step():
    """The east wall was seen at 0-20 m and at 50 m at the same offset, so
    its middle was bridged flat. The west wall, observed all the way up, has
    a 4 m setback at 30-40 m and returns at 50 m -- a notch. The twin's
    measurement outranks the interpolation."""
    from kyfromabove_3d.features import FaceBand, mirror_hidden_bands

    west_obs = {0.0: 0.0, 10.0: 0.0, 20.0: 0.0, 30.0: -4.0, 40.0: -4.0, 50.0: 0.0}
    east_obs = {0.0: 0.5, 10.0: 0.5, 20.0: 0.5, 50.0: 0.5}
    planes, west, east, profiles = _profiles_for_square(west_obs, east_obs)
    for i, b in enumerate(profiles[east.face]):
        if b.status == "unknown":
            profiles[east.face][i] = FaceBand(b.z_lo, b.z_hi, 0.5, 0, 0.0, 5.0, 95.0, "bridged")
    rep = mirror_hidden_bands(planes, profiles)
    e = {b.z_lo: b for b in profiles[east.face]}
    assert rep[east.face]["symmetric"] and rep[east.face]["mirrored"] == 2
    assert e[30.0].status == "mirrored" and abs(e[30.0].offset_m - (-3.5)) < 1e-6
    assert e[40.0].status == "mirrored" and abs(e[40.0].offset_m - (-3.5)) < 1e-6
    assert e[50.0].status == "observed"


def _returns(fn, x0=-30, x1=30, y0=-20, y1=20, density=3.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int((x1 - x0) * (y1 - y0) * density)
    xs = rng.uniform(x0, x1, n); ys = rng.uniform(y0, y1, n)
    return np.column_stack([xs, ys, fn(xs, ys) + rng.normal(0, 0.05, n)])


def test_building_levels_splits_a_tower_from_its_podium():
    from kyfromabove_3d.features import LEVEL_FACE_STRIDE, building_levels

    ring = np.array([[-30, -20], [30, -20], [30, 20], [-30, 20], [-30, -20]], dtype=float)
    # a 12 m podium over the whole footprint, a 60 m tower on its east half
    xyz = _returns(lambda x, y: np.where((x > 0) & (np.abs(y) < 12), 60.0, 12.0))
    levels = building_levels(xyz, ring, ground_z=0.0)
    assert [lv["level"] for lv in levels] == [0, 1]
    podium, tower = levels
    assert abs(podium["z_hi"] - 12.0) < 0.5 and podium["z_lo"] == 0.0
    assert abs(tower["z_lo"] - 12.0) < 0.5 and abs(tower["z_hi"] - 60.0) < 0.5
    from shapely.geometry import Polygon
    t = Polygon(tower["ring"])
    assert 600 < t.area < 850                                   # 30 x 24 = 720 m2, within a cell of the edges
    assert t.centroid.x > 10 and abs(t.centroid.y) < 2
    assert LEVEL_FACE_STRIDE == 1000


def test_building_levels_leaves_a_flat_roof_and_a_slab_alone():
    from kyfromabove_3d.features import building_levels

    ring = np.array([[-30, -20], [30, -20], [30, 20], [-30, 20], [-30, -20]], dtype=float)
    flat = building_levels(_returns(lambda x, y: np.full(len(x), 25.0)), ring, 0.0)
    assert len(flat) == 1 and abs(flat[0]["z_hi"] - 25.0) < 0.5
    # a mechanical penthouse: 4 m higher over 3 % of the roof is not a tower
    pent = building_levels(_returns(lambda x, y: np.where((np.abs(x) < 4) & (np.abs(y) < 4), 29.0, 25.0)), ring, 0.0)
    assert len(pent) == 1
    # a tall building whose whole roof is high: one level
    tall = building_levels(_returns(lambda x, y: np.full(len(x), 120.0)), ring, 0.0)
    assert len(tall) == 1


def test_building_levels_finds_two_towers_on_one_footprint():
    from kyfromabove_3d.features import building_levels

    ring = np.array([[-40, -20], [40, -20], [40, 20], [-40, 20], [-40, -20]], dtype=float)
    xyz = _returns(lambda x, y: np.where(((x < -10) | (x > 10)) & (np.abs(y) < 12) & (np.abs(x) < 35), 50.0, 8.0),
                   x0=-40, x1=40)
    levels = building_levels(xyz, ring, 0.0)
    assert [lv["level"] for lv in levels] == [0, 1, 2]
    from shapely.geometry import Polygon
    cx = sorted(Polygon(lv["ring"]).centroid.x for lv in levels[1:])
    assert cx[0] < -15 and cx[1] > 15


def test_building_levels_preserves_l_shaped_tower():
    from kyfromabove_3d.features import building_levels
    from shapely.geometry import Polygon

    ring = np.array([[-30, -30], [30, -30], [30, 30], [-30, 30], [-30, -30]], dtype=float)
    # 10 m podium over 60x60 m footprint; 50 m tower shaped like an L
    is_tower = lambda x, y: ((x >= -20) & (x <= 20) & (y >= -20) & (y <= -5)) | \
                            ((x >= -20) & (x <= -5) & (y >= -5) & (y <= 20))
    xyz = _returns(lambda x, y: np.where(is_tower(x, y), 50.0, 10.0), x0=-30, x1=30, y0=-30, y1=30)
    levels = building_levels(xyz, ring, 0.0)
    assert len(levels) == 2
    podium, tower = levels
    tower_poly = Polygon(tower["ring"])
    # An L-shape preserves its multi-corner form (>= 6 corners) rather than collapsing to a 4-point box
    assert len(tower_poly.exterior.coords) - 1 >= 6


def test_rectify_footprint_corners():
    from kyfromabove_3d.features import (
        WallPlane,
        VerifiedWallPlane,
        rectify_footprint_corners,
    )

    # 4-corner nominal box: (0,0) -> (10,0) -> (10,10) -> (0,10) -> (0,0)
    nominal_ring = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0],
        [0.0, 0.0],
    ])

    # Face 0 (normal [0, -1]) verified with 0.3 m outward shift: y = -0.3
    # Face 1 (normal [1, 0]) verified with 0.2 m outward shift: x = 10.2
    # Face 2 and Face 3 are nominal
    planes = [
        VerifiedWallPlane(0, np.array([0.0, -0.3]), np.array([10.0, -0.3]),
                          np.array([0.0, -1.0]), 0.0, 20.0, n_tiles=5, z_top=20.0),
        VerifiedWallPlane(1, np.array([10.2, 0.0]), np.array([10.2, 10.0]),
                          np.array([1.0, 0.0]), 0.0, 20.0, n_tiles=5, z_top=20.0),
        WallPlane(2, np.array([10.0, 10.0]), np.array([0.0, 10.0]),
                  np.array([0.0, 1.0]), 0.0, 20.0),
        WallPlane(3, np.array([0.0, 10.0]), np.array([0.0, 0.0]),
                  np.array([-1.0, 0.0]), 0.0, 20.0),
    ]

    rectified_ring, updated_planes = rectify_footprint_corners(
        nominal_ring, planes, max_corner_shift_m=2.0, min_angle_deg=15.0
    )

    # Corner 0 (between 3 and 0): Line 3 (x=0) and Line 0 (y=-0.3) -> (0.0, -0.3)
    assert np.allclose(rectified_ring[0], [0.0, -0.3])
    # Corner 1 (between 0 and 1): Line 0 (y=-0.3) and Line 1 (x=10.2) -> (10.2, -0.3)
    assert np.allclose(rectified_ring[1], [10.2, -0.3])
    # Corner 2 (between 1 and 2): Line 1 (x=10.2) and Line 2 (y=10.0) -> (10.2, 10.0)
    assert np.allclose(rectified_ring[2], [10.2, 10.0])
    # Corner 3 (between 2 and 3): neither verified -> stays nominal (0.0, 10.0)
    assert np.allclose(rectified_ring[3], [0.0, 10.0])
    # Closed ring
    assert np.allclose(rectified_ring[4], rectified_ring[0])

    # Planes updated to match corners
    assert np.allclose(updated_planes[0].p0, [0.0, -0.3])
    assert np.allclose(updated_planes[0].p1, [10.2, -0.3])
    assert np.allclose(updated_planes[1].p0, [10.2, -0.3])
    assert np.allclose(updated_planes[1].p1, [10.2, 10.0])


def test_rectify_footprint_corners_handles_collinear_edges():
    from kyfromabove_3d.features import (
        WallPlane,
        VerifiedWallPlane,
        rectify_footprint_corners,
    )

    # Ring with an extra collinear vertex along face 0: (0,0) -> (5,0) -> (10,0) -> (10,10) -> (0,10) -> (0,0)
    nominal_ring = np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0],
        [0.0, 0.0],
    ])

    # Edge 1 is a verified plane with normal [0, -1] shifted by 0.5 m outward (y = -0.5)
    # Edge 0 is collinear with Edge 1
    planes = [
        WallPlane(0, np.array([0.0, 0.0]), np.array([5.0, 0.0]), np.array([0.0, -1.0]), 0.0, 20.0),
        VerifiedWallPlane(1, np.array([5.0, -0.5]), np.array([10.0, -0.5]),
                          np.array([0.0, -1.0]), 0.0, 20.0, n_tiles=5, z_top=20.0, offset_m=0.5),
        WallPlane(2, np.array([10.0, 0.0]), np.array([10.0, 10.0]), np.array([1.0, 0.0]), 0.0, 20.0),
        WallPlane(3, np.array([10.0, 10.0]), np.array([0.0, 10.0]), np.array([0.0, 1.0]), 0.0, 20.0),
        WallPlane(4, np.array([0.0, 10.0]), np.array([0.0, 0.0]), np.array([-1.0, 0.0]), 0.0, 20.0),
    ]

    rectified_ring, _ = rectify_footprint_corners(nominal_ring, planes)
    # Corner 1 (between collinear Edge 0 and Edge 1) projects onto verified Line 1: y = -0.5
    assert np.allclose(rectified_ring[1], [5.0, -0.5])


def test_rectify_footprint_corners_rejects_excessive_shift():
    from kyfromabove_3d.features import (
        VerifiedWallPlane,
        rectify_footprint_corners,
    )

    nominal_ring = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0],
        [0.0, 0.0],
    ])

    # Outlier plane shifted by 10.0 m (exceeding max_corner_shift_m = 2.0 m)
    planes = [
        VerifiedWallPlane(0, np.array([0.0, -10.0]), np.array([10.0, -10.0]),
                          np.array([0.0, -1.0]), 0.0, 20.0, n_tiles=5, z_top=20.0),
        VerifiedWallPlane(1, np.array([10.0, 0.0]), np.array([10.0, 10.0]),
                          np.array([1.0, 0.0]), 0.0, 20.0, n_tiles=5, z_top=20.0),
    ]

    rectified_ring, _ = rectify_footprint_corners(
        nominal_ring, planes, max_corner_shift_m=2.0
    )
    # The excessive shift is rejected; corner stays at nominal
    assert np.allclose(rectified_ring[1], [10.0, 0.0])


def test_ground_field_and_edge_interpolation():
    from kyfromabove_3d.features import ground_field_for_building

    ring = np.array([
        [0.0, 0.0],
        [20.0, 0.0],
        [20.0, 10.0],
        [0.0, 10.0],
        [0.0, 0.0],
    ])

    # Generate synthetic ground points with a slope: z = 2.0 - 0.05 * x (drops 1 m from x=0 to x=20)
    xs = np.linspace(-5.0, 25.0, 40)
    ys = np.linspace(-5.0, 15.0, 30)
    X, Y = np.meshgrid(xs, ys)
    pts_x = X.ravel(); pts_y = Y.ravel()
    pts_z = 2.0 - 0.05 * pts_x
    xyz = np.column_stack([pts_x, pts_y, pts_z])
    cls = np.full(len(xyz), 2, dtype=np.uint8)

    gf = ground_field_for_building(xyz, ring, classification=cls, search_radius_m=6.0)

    # Vertex 0 (x=0, y=0) should have z ~ 2.0
    assert abs(gf.elevation_at_vertex(0) - 2.0) < 0.05
    # Vertex 1 (x=20, y=0) should have z ~ 1.0
    assert abs(gf.elevation_at_vertex(1) - 1.0) < 0.05

    # Along edge 0 (L=20 m):
    # at s = 10 m (midpoint), elevation should be linearly interpolated to ~1.5 m
    z_mid = gf.elevation_along_edge(edge_idx=0, s=10.0, L=20.0)
    assert abs(z_mid - 1.5) < 0.05


def test_face_profile_flags_stepped_crown_with_unobserved_gaps_as_sloped():
    """A stepped crown or dome where unobserved gaps (ledges/terrace returns)
    separate the receding bands: non-contiguous bands that step inward must be
    detected as sloped, preventing floating wall panels at roof level."""
    from kyfromabove_3d.features import extrude_profile, face_profile, wall_planes

    pl = wall_planes(_square() * 5.0, z_lo=0.0, z_hi=80.0)[0]
    # Continuous vertical wall from 0 to 40 m (offset 0.0)
    parts = [_wall_returns(pl, [(0, 40)], offset=0.0, per_band=200)]
    # Stepped crown with gaps:
    # 40..45 is missing/unobserved
    # 45..50 has returns at offset -2.0 m
    # 50..55 is missing/unobserved
    # 55..60 has returns at offset -4.0 m
    parts.append(_wall_returns(pl, [(45, 50)], offset=-2.0, per_band=200, seed=42))
    parts.append(_wall_returns(pl, [(55, 60)], offset=-4.0, per_band=200, seed=43))

    bands = face_profile(pl, np.vstack(parts), band_m=5.0)
    status = {b.z_lo: b.status for b in bands}

    # Vertical wall bands below 40 m are observed
    assert all(status[z] == "observed" for z in (0.0, 10.0, 20.0, 35.0))
    # Stepped bands at 45 m and 55 m are detected as sloped despite intervening gaps
    assert status[45.0] == "sloped"
    assert status[55.0] == "sloped"
    # Extrusion stops at the top of the vertical shaft (< 41 m)
    pts = extrude_profile(pl, bands, spacing_m=0.5)
    assert pts[:, 2].max() < 41.0


def test_level_face_stride_no_collision_on_large_rings():
    """When a footprint ring has > 100 edges (e.g. 120 vertices),
    LEVEL_FACE_STRIDE = 1000 ensures level-0 faces do not collide with level-1 faces."""
    from kyfromabove_3d.features import LEVEL_FACE_STRIDE, wall_planes

    # Create a 120-vertex circular polygon
    angles = np.linspace(0, 2 * np.pi, 121)[:-1]
    ring = np.column_stack([30.0 * np.cos(angles), 30.0 * np.sin(angles)])
    ring = np.vstack([ring, ring[:1]])

    planes_lv0 = wall_planes(ring, z_lo=0.0, z_hi=20.0, min_length_m=0.1)
    assert len(planes_lv0) == 120
    for p in planes_lv0:
        p.face += 0 * LEVEL_FACE_STRIDE

    # Level 1 has 4 edges
    ring_lv1 = np.array([[-10, -10], [10, -10], [10, 10], [-10, 10], [-10, -10]], dtype=float)
    planes_lv1 = wall_planes(ring_lv1, z_lo=20.0, z_hi=80.0, min_length_m=0.1)
    for p in planes_lv1:
        p.face += 1 * LEVEL_FACE_STRIDE

    all_face_ids = [p.face for p in planes_lv0] + [p.face for p in planes_lv1]
    # No duplicate face ids
    assert len(all_face_ids) == len(set(all_face_ids))
    # Max face id in level 0 is 119 (< 1000)
    assert max(p.face for p in planes_lv0) < LEVEL_FACE_STRIDE
    # Min face id in level 1 is 1000
    assert min(p.face for p in planes_lv1) >= LEVEL_FACE_STRIDE


def test_building_levels_rejects_adjacent_skyscraper_spillover():
    """A 15 m low-rise building abutting a 100 m skyscraper gets LiDAR returns
    on the shared edge up to 100 m. Because those upper returns cover < 5% of
    the footprint area, building_levels must reject them and set z_hi to the
    true ~15 m roof instead of 100 m."""
    from kyfromabove_3d.features import building_levels

    # 40 m x 40 m building footprint at ground 0
    ring = np.array([[0, 0], [40, 0], [40, 40], [0, 40], [0, 0]], dtype=float)
    rng = np.random.default_rng(42)

    # 10,000 points on the low-rise roof and ground (height 0-15 m, median roof ~14 m)
    rx = rng.uniform(0.5, 39.5, 8000)
    ry = rng.uniform(0.5, 39.5, 8000)
    rz = rng.uniform(13.5, 15.0, 8000)
    base_pts = np.column_stack([rx, ry, rz])

    # 1,000 returns along the eastern party wall (x=39.5 to 40.0) extending from 20 m to 100 m
    # (covers only 0.5m x 40m = 20 m^2 out of 1600 m^2 = 1.25% of footprint)
    wx = rng.uniform(39.5, 40.0, 1000)
    wy = rng.uniform(0.0, 40.0, 1000)
    wz = rng.uniform(20.0, 100.0, 1000)
    wall_spillover = np.column_stack([wx, wy, wz])

    xyz = np.vstack([base_pts, wall_spillover])

    levels = building_levels(xyz, ring, ground_z=0.0)
    assert len(levels) == 1
    # Ceiling should be ~15 m, NOT 100 m!
    assert levels[0]["z_hi"] < 17.0
    assert levels[0]["z_hi"] > 14.0



def test_building_levels_keeps_a_small_tower_inside_the_roof():
    """A 6 m square tower in the middle of a 15 m roof covers about 2% of
    the footprint -- under 5%, like a neighbour's spill -- but it is the
    building's own: it stands away from the edge, and the building keeps
    its full height."""
    from kyfromabove_3d.features import building_levels

    ring = np.array([[0, 0], [40, 0], [40, 40], [0, 40], [0, 0]], dtype=float)
    rng = np.random.default_rng(7)
    roof = np.column_stack([rng.uniform(0.5, 39.5, 8000), rng.uniform(0.5, 39.5, 8000),
                            rng.uniform(13.5, 15.0, 8000)])
    tower = np.column_stack([rng.uniform(17.0, 23.0, 3000), rng.uniform(17.0, 23.0, 3000),
                             rng.uniform(40.0, 45.0, 3000)])
    levels = building_levels(np.vstack([roof, tower]), ring, ground_z=0.0)
    assert len(levels) == 1
    assert levels[0]["z_hi"] > 30.0

# --- structural building classification --------------------------------------

def test_classify_simple_box():
    from kyfromabove_3d.features import (
        FaceBand,
        SurfaceZones,
        classify_building_structure,
    )

    levels = [{"level": 0, "ring": [[0, 0], [20, 0], [20, 20], [0, 20]], "z_lo": 0.0, "z_hi": 10.0}]
    zones = SurfaceZones(
        plate_z_m=10.0,
        plateau_frac=0.85,
        roof_plateau_frac=0.85,
        plate_rms_m=0.12,
        ground=np.zeros(100, dtype=bool),
        wall=np.zeros(100, dtype=bool),
        roof=np.ones(100, dtype=bool),
    )
    profiles = {
        0: [FaceBand(0.0, 10.0, 0.05, 100, 0.1, 0.0, 20.0, "observed")],
        1: [FaceBand(0.0, 10.0, -0.02, 100, 0.1, 0.0, 20.0, "observed")],
    }
    cls = classify_building_structure(levels, zones, profiles, arc_edges=set(), sloped_fraction=0.0)
    assert cls.archetype == "simple_box"
    assert cls.roof_type == "flat"
    assert cls.facade_type == "planar"
    assert cls.metrics["max_wall_offset_m"] == 0.05


def test_classify_tiered_box():
    from kyfromabove_3d.features import (
        FaceBand,
        SurfaceZones,
        classify_building_structure,
    )

    levels = [
        {"level": 0, "ring": [[0, 0], [40, 0], [40, 40], [0, 40]], "z_lo": 0.0, "z_hi": 20.0},
        {"level": 1, "ring": [[10, 10], [30, 10], [30, 30], [10, 30]], "z_lo": 20.0, "z_hi": 60.0},
    ]
    zones = SurfaceZones(
        plate_z_m=60.0,
        plateau_frac=0.75,
        roof_plateau_frac=0.75,
        plate_rms_m=0.18,
        ground=np.zeros(100, dtype=bool),
        wall=np.zeros(100, dtype=bool),
        roof=np.ones(100, dtype=bool),
    )
    profiles = {
        0: [FaceBand(0.0, 20.0, 0.0, 100, 0.1, 0.0, 40.0, "observed")],
    }
    cls = classify_building_structure(levels, zones, profiles, arc_edges=set(), sloped_fraction=0.0)
    assert cls.archetype == "tiered_box"
    assert cls.roof_type == "flat"
    assert cls.facade_type == "planar"
    assert cls.metrics["num_levels"] == 2


def test_classify_complex_roof():
    from kyfromabove_3d.features import (
        FaceBand,
        SurfaceZones,
        classify_building_structure,
    )

    levels = [{"level": 0, "ring": [[0, 0], [20, 0], [20, 20], [0, 20]], "z_lo": 0.0, "z_hi": 30.0}]
    # Pitch or pyramidal roof: plate_rms is high (0.85 m > 0.30 m) or plateau fraction is low
    zones = SurfaceZones(
        plate_z_m=30.0,
        plateau_frac=0.45,
        roof_plateau_frac=0.45,
        plate_rms_m=0.85,
        ground=np.zeros(100, dtype=bool),
        wall=np.zeros(100, dtype=bool),
        roof=np.ones(100, dtype=bool),
    )
    profiles = {
        0: [FaceBand(0.0, 30.0, 0.05, 100, 0.1, 0.0, 20.0, "observed")],
    }
    cls = classify_building_structure(levels, zones, profiles, arc_edges=set(), sloped_fraction=0.0)
    assert cls.archetype == "complex_roof"
    assert cls.roof_type == "complex"
    assert cls.facade_type == "planar"


def test_classify_sculpted_facade_cantilever():
    from kyfromabove_3d.features import (
        FaceBand,
        SurfaceZones,
        classify_building_structure,
    )

    levels = [{"level": 0, "ring": [[0, 0], [20, 0], [20, 20], [0, 20]], "z_lo": 0.0, "z_hi": 50.0}]
    zones = SurfaceZones(
        plate_z_m=50.0,
        plateau_frac=0.85,
        roof_plateau_frac=0.85,
        plate_rms_m=0.15,
        ground=np.zeros(100, dtype=bool),
        wall=np.zeros(100, dtype=bool),
        roof=np.ones(100, dtype=bool),
    )
    # Significant wall offset (e.g. 2.5 m cantilever like Humana Building)
    profiles = {
        0: [FaceBand(0.0, 50.0, 2.5, 100, 0.1, 0.0, 20.0, "observed")],
    }
    cls = classify_building_structure(levels, zones, profiles, arc_edges=set(), sloped_fraction=0.0)
    assert cls.archetype == "sculpted_facade"
    assert cls.facade_type == "sculpted"
    assert cls.metrics["max_wall_offset_m"] == 2.5


def test_classify_sculpted_facade_curved_or_open():
    from kyfromabove_3d.features import (
        FaceBand,
        SurfaceZones,
        classify_building_structure,
    )

    levels = [{"level": 0, "ring": [[0, 0], [20, 0], [20, 20], [0, 20]], "z_lo": 0.0, "z_hi": 20.0}]
    zones = SurfaceZones(
        plate_z_m=20.0,
        plateau_frac=0.85,
        roof_plateau_frac=0.85,
        plate_rms_m=0.15,
        ground=np.zeros(100, dtype=bool),
        wall=np.zeros(100, dtype=bool),
        roof=np.ones(100, dtype=bool),
    )
    # Case 1: has open loggia
    profiles_open = {
        0: [FaceBand(0.0, 10.0, 0.0, 100, 0.1, 0.0, 20.0, "open")],
    }
    cls_open = classify_building_structure(levels, zones, profiles_open, arc_edges=set(), sloped_fraction=0.0)
    assert cls_open.archetype == "sculpted_facade"

    # Case 2: has curved wall (arc_edges)
    cls_arc = classify_building_structure(levels, zones, {}, arc_edges={0, 1}, sloped_fraction=0.0)
    assert cls_arc.archetype == "sculpted_facade"


def test_building_classification_serialization():
    from kyfromabove_3d.features import BuildingClassification

    bc = BuildingClassification(
        archetype="simple_box",
        roof_type="flat",
        facade_type="planar",
        metrics={"plate_rms_m": 0.12},
    )
    d = bc.to_dict()
    assert d["archetype"] == "simple_box"
    assert "confidence" not in d
    assert d["roof_type"] == "flat"

    loaded = BuildingClassification.from_dict(d)
    assert loaded.archetype == bc.archetype
    assert loaded.roof_type == bc.roof_type
    assert loaded.facade_type == bc.facade_type
    assert loaded.metrics == bc.metrics


def test_classify_party_wall_awareness():
    from kyfromabove_3d.features import (
        FaceBand,
        SurfaceZones,
        classify_building_structure,
    )

    levels = [{"level": 0, "ring": [[0, 0], [20, 0], [20, 20], [0, 20]], "z_lo": 0.0, "z_hi": 10.0}]
    zones = SurfaceZones(
        plate_z_m=10.0,
        plateau_frac=0.85,
        plate_rms_m=0.12,
        ground=np.zeros(100, dtype=bool),
        wall=np.zeros(100, dtype=bool),
        roof=np.ones(100, dtype=bool),
        roof_plateau_frac=0.85,
    )
    # Face 0: street-facing, planar (offset 0.05m)
    # Face 1: party wall touching neighbor, returns detected at -5.0m offset or open
    profiles = {
        0: [FaceBand(0.0, 10.0, 0.05, 100, 0.1, 0.0, 20.0, "observed")],
        1: [FaceBand(0.0, 10.0, -5.0, 100, 0.1, 0.0, 20.0, "open")],
    }
    # Without party_faces awareness -> sculpted_facade
    cls_naive = classify_building_structure(levels, zones, profiles, arc_edges=set(), sloped_fraction=0.0)
    assert cls_naive.archetype == "sculpted_facade"

    # With party_faces={1} -> party wall is ignored, classified as simple_box!
    cls_aware = classify_building_structure(levels, zones, profiles, arc_edges=set(), sloped_fraction=0.0, party_faces={1})
    assert cls_aware.archetype == "simple_box"
    assert cls_aware.metrics["party_faces"] == 1
    assert cls_aware.metrics["max_wall_offset_m"] == 0.05







def test_classify_a_pitched_roof_is_not_flat_even_when_smooth():
    """A roof whose returns sit close to one plate can still be pitched: the
    share of sloping cells in the roof field decides."""
    from kyfromabove_3d.features import FaceBand, SurfaceZones, classify_building_structure

    levels = [{"level": 0, "ring": [[0, 0], [20, 0], [20, 20], [0, 20]], "z_lo": 0.0, "z_hi": 10.0}]
    zones = SurfaceZones(plate_z_m=10.0, plateau_frac=0.85, plate_rms_m=0.12,
                         ground=np.zeros(100, dtype=bool), wall=np.zeros(100, dtype=bool),
                         roof=np.ones(100, dtype=bool), roof_plateau_frac=0.85)
    profiles = {0: [FaceBand(0.0, 10.0, 0.05, 100, 0.1, 0.0, 20.0, "observed")]}
    cls = classify_building_structure(levels, zones, profiles, arc_edges=set(), sloped_fraction=0.30)
    assert cls.roof_type == "complex"
    assert cls.archetype == "complex_roof"


def test_a_wall_stands_on_the_wall_below_it():
    """Bands whose returns stop short under a band that spans the face (a
    neighbour hid them from the scanner) are widened to it; a set-back tower
    above a wider podium does not narrow the podium; an open band breaks it."""
    from kyfromabove_3d.features import FaceBand, carry_extents_down

    def band(z, lo, hi, off=0.0, status="observed"):
        return FaceBand(z, z + 5, off, 50, 0.1, lo, hi, status)

    bands = [band(0, 0, 90), band(5, 0, 50), band(10, 0, 40), band(15, 0, 90, status="extended")]
    assert carry_extents_down(bands) == 2
    assert [(b.along_lo, b.along_hi) for b in bands] == [(0, 90)] * 4
    podium = [band(0, 0, 90), band(5, 20, 60, off=-8.0)]              # a tower set 8 m back
    assert carry_extents_down(podium) == 0 and podium[0].along_hi == 90
    loggia = [band(0, 0, 30), band(5, 0, 90, status="open"), band(10, 0, 90)]
    assert carry_extents_down(loggia) == 0 and loggia[0].along_hi == 30


def test_an_open_band_at_the_top_is_the_walls_crown_when_its_front_continues_the_wall():
    """PNC: the wall's top band has returns at the facade line (a crown) and
    the roof behind; it is wall, at its front. Kentucky Home Life: an open band
    whose front stands 3 m inboard is not, nor what stands on it. A colonnade
    (floors above) is left to the recess rule."""
    from kyfromabove_3d.features import FaceBand, crown_bands

    def band(z, off, status="observed", back=None):
        return FaceBand(z, z + 5, off, 500, 0.1, 0.0, 50.0, status, back_offset_m=back)

    pnc = [band(142, None, "unknown"), band(147, 0.05, "sloped"), band(152, -0.25, "open", -5.25), band(157, None, "unknown")]
    assert crown_bands([FaceBand(**{**vars(x), "segments": []}) for x in pnc]) == 0   # a sloped band is not drawn off a verified plane
    assert crown_bands(pnc, reference_offset_m=0.0) == 1 and pnc[2].status == "observed" and pnc[2].offset_m == -0.25
    truist = [band(12, None, "unknown"), band(17, None, "unknown"), band(22, -0.5, "open", -6.0)]
    assert crown_bands(truist) == 0                                           # the prism path walls this face to its roof
    face3 = [band(142, None, "unknown"), band(147, None, "unknown"), band(152, -0.75, "open", -7.75)]
    assert crown_bands(face3, reference_offset_m=0.0) == 1                    # a verified plane: the plane is the wall
    assert crown_bands([band(147, None, "unknown"), band(152, -0.75, "open", -7.75)]) == 0   # nothing to continue
    omni = [band(92, -0.02, "mirrored"), band(97, -1.75, "open", -7.25), band(102, -1.25, "open", -6.25)]
    assert crown_bands(omni) == 2                                             # a crown two bands tall
    khl = [band(55, 0.0), band(60, -2.75, "open", -2.75), band(65, -0.25, "open", -7.75)]
    assert crown_bands(khl) == 0 and khl[1].status == "open" and khl[2].status == "open"
    colonnade = [band(0, 0.0, "open", -3.0), band(5, 0.0), band(10, 0.0)]
    assert crown_bands(colonnade) == 0
    setback = [band(45, -5.0), band(50, -5.5), band(55, -6.25, "open", -9.0)]  # Waterfront Plaza: 6 m inboard
    assert crown_bands(setback) == 0
    low = [band(10, 0.0), band(15, -0.5, "open", -6.0)]                       # the roof 2 m above the band's foot
    assert crown_bands(low, top_z=17.0) == 1 and low[1].z_hi == 17.0
    allopen = [band(0, 0.25, "open", -3.0), band(5, -7.75, "open", -9.0), band(10, 0.25, "open", -5.0)]
    assert crown_bands(allopen, reference_offset_m=0.0) == 0                  # 4a668624's face 1: open all the way
    chase = [band(4, 0.93, "extended"), band(9, 0.95), band(14, 1.0), band(19, 0.75, "open", -6.25),
             band(24, None, "unknown"), band(29, -1.25, "open", -7.75)]
    assert crown_bands(chase) == 1 and chase[3].status == "observed" and chase[5].status == "open"
    own = [band(10, 0.0), band(15, -0.5, "open", -6.0)]                       # its returns reach 16.2 m
    assert crown_bands(own, top_z=20.0, top_of=lambda b: 16.2) == 1 and own[1].z_hi == 16.2


def test_fill_under_roof_edge_draws_unseen_and_roof_backed_bands_but_keeps_a_wall_behind():
    from kyfromabove_3d.features import FaceBand, fill_under_roof_edge

    def band(z, status, off=None, back=None, fill=None):
        return FaceBand(z, z + 5.0, off, 50, 0.1, 0.0, 30.0, status, back_offset_m=back, back_fill=fill)

    bands = [band(0, "observed", -0.6), band(5, "unknown"), band(10, "sloped", -1.2), band(15, "sloped", -4.0),
             band(20, "open", -1.0, -7.0, 0.1), band(25, "open", -7.0, -7.0, 0.9)]
    assert fill_under_roof_edge(bands, 30.0) == 3
    assert [b.status for b in bands] == ["observed", "edge", "edge", "sloped", "edge", "open"]
    # each filled band at the depth of the nearest drawn band within 1.5 m of the edge
    assert all(b.offset_m == -0.6 for b in bands[1:3]) and bands[4].offset_m == -0.6
    # with nothing drawn near the edge, on the edge itself
    lone = [band(0, "observed", 2.4), band(5, "unknown")]
    fill_under_roof_edge(lone, 30.0)
    assert lone[1].status == "edge" and lone[1].offset_m == 0.0
    # a sloped band further in than 1.5 m is the wall holding its line when within 1 m of the wall beside it;
    # a crown stepping in further is a slope, and a band at street level 2.6 m outside the edge is not the wall
    truist = [band(37, "observed", -1.2), band(42, "extended", -1.12), band(47, "unknown"), band(52, "sloped", -1.58)]
    assert fill_under_roof_edge(truist, 30.0) == 2 and truist[3].status == "edge" and truist[3].offset_m == -1.12
    crown = [band(135, "observed", -1.3), band(140, "sloped", -3.4), band(145, "sloped", -4.9)]
    assert fill_under_roof_edge(crown, 30.0) == 0
    street = [band(0, "sloped", 2.6), band(5, "unknown")]
    fill_under_roof_edge(street, 30.0)
    assert street[0].status == "sloped" and street[1].offset_m == 0.0
