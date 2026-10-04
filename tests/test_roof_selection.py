"""Choosing the right roof when an AOI holds more than one.

The pilot AOI contains a single low building, so two shortcuts were invisible:
a 15 m ceiling on roof height, and connectivity computed in plan view. Neither
errors on a downtown AOI -- they return a confident answer about the wrong
building, and every measurement downstream inherits it.

The fixture here is the minimal shape that exposes both: a large podium roof
with a small tower roof stacked directly above it.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from kyfromabove_3d.features import extract_roof_points, roof_check_features


def slab(x0, y0, side, z, spacing=1.0, jitter=0.01, seed=0):
    """A flat horizontal surface, dense enough for local plane fitting."""
    rng = np.random.default_rng(seed)
    n = int(side / spacing) + 1
    g = np.linspace(0.0, side, n)
    gx, gy = np.meshgrid(g, g)
    x, y = gx.ravel() + x0, gy.ravel() + y0
    return np.column_stack([x, y, np.full(x.shape, float(z))
                            + rng.normal(0, jitter, x.shape)])


def stacked_scene():
    """Ground, a 60 m podium roof at 20 m, and a 16 m tower roof at 160 m.

    The tower footprint sits inside the podium's, so in plan view the two
    surfaces overlap completely -- which is exactly what a 2D clustering step
    cannot separate.
    """
    ground = slab(0.0, 0.0, 80.0, 0.0, spacing=2.0, seed=1)
    podium = slab(5.0, 5.0, 60.0, 20.0, spacing=1.0, seed=2)
    tower = slab(30.0, 30.0, 16.0, 160.0, spacing=0.7, seed=3)
    xyz = np.vstack([ground, podium, tower])
    cls = np.concatenate([np.full(len(ground), 2),
                          np.full(len(podium) + len(tower), 1)])
    counts = {"ground": len(ground), "podium": len(podium), "tower": len(tower)}
    return SimpleNamespace(xyz_local_m=xyz, classification=cls), counts


def test_the_default_ceiling_silently_excludes_a_tall_roof():
    lidar, _ = stacked_scene()
    low, _ = extract_roof_points(lidar)                       # default 15 m
    assert len(low) == 0 or low[:, 2].max() < 100, \
        "a 15 m ceiling must not admit a 160 m roof"

    tall, _ = extract_roof_points(lidar, max_height_m=None)
    assert tall[:, 2].max() > 150, "no ceiling must admit the tower roof"


def test_largest_picks_the_podium_and_tallest_picks_the_tower():
    lidar, counts = stacked_scene()
    assert counts["podium"] > counts["tower"], "fixture must make podium larger"

    _, big = roof_check_features(lidar, max_height_m=None, select="largest")
    assert np.median(big[:, 2]) == pytest.approx(20.0, abs=2.0)

    _, high = roof_check_features(lidar, max_height_m=None, select="tallest")
    assert np.median(high[:, 2]) == pytest.approx(160.0, abs=2.0)


def test_stacked_surfaces_are_separated_by_three_dimensional_connectivity():
    """The regression that mattered: in plan view these two roofs are one blob.

    With 2D connectivity they merge, and the merged cluster's median elevation
    is the *podium's*, because the podium has more points. "tallest" would then
    return a 20 m roof for a 160 m building, and say nothing was wrong.
    """
    lidar, _ = stacked_scene()
    _, roof = roof_check_features(lidar, max_height_m=None, select="tallest")

    assert np.median(roof[:, 2]) > 150.0
    # And it is only the tower: the podium's 60 m span must not be included.
    span = max(roof[:, 0].max() - roof[:, 0].min(), roof[:, 1].max() - roof[:, 1].min())
    assert span < 25.0, f"cluster spans {span:.0f} m; podium leaked in"


def test_an_isolated_single_storey_building_is_unaffected():
    """The pilot case: nothing stacked, so both modes agree."""
    ground = slab(0.0, 0.0, 80.0, 0.0, spacing=2.0, seed=4)
    roof = slab(20.0, 20.0, 40.0, 6.0, spacing=1.0, seed=5)
    lidar = SimpleNamespace(
        xyz_local_m=np.vstack([ground, roof]),
        classification=np.concatenate([np.full(len(ground), 2), np.full(len(roof), 1)]))

    _, a = roof_check_features(lidar, select="largest")
    _, b = roof_check_features(lidar, select="tallest")
    assert np.median(a[:, 2]) == pytest.approx(np.median(b[:, 2]), abs=0.5)
    assert len(a) == len(b)


def test_unknown_selection_mode_is_rejected():
    lidar, _ = stacked_scene()
    with pytest.raises(ValueError, match="largest.*tallest"):
        roof_check_features(lidar, max_height_m=None, select="biggest")


# --------------------------------------------------------------- curved roofs

def dome(cx, cy, radius, base_z, rise, n=4000, seed=7):
    """A dome: elevation falls off with radius, as 400 West Market's crown does."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2 * np.pi, n)
    x, y = cx + r * np.cos(th), cy + r * np.sin(th)
    z = base_z + rise * np.cos(np.pi / 2 * np.clip(r / radius, 0, 1))
    return np.column_stack([x, y, z])


def dome_scene():
    ground = slab(0.0, 0.0, 80.0, 0.0, spacing=2.0, seed=8)
    # Rise chosen so the surface survives extract_roof_points' near-horizontal
    # filter, as the real crown's apex cap does, while still being curved
    # enough to fail the plane test (0.575 m RMS against a 0.5 m threshold).
    crown = dome(40.0, 40.0, 9.0, 160.0, 2.0)
    return SimpleNamespace(
        xyz_local_m=np.vstack([ground, crown]),
        classification=np.concatenate([np.full(len(ground), 2), np.full(len(crown), 1)]),
    ), crown


def surface_z_near(xy, cloud, radius=2.0):
    d = np.hypot(cloud[:, 0] - xy[0], cloud[:, 1] - xy[1])
    return np.median(cloud[d <= radius, 2]) if (d <= radius).sum() >= 3 else None


def test_a_dome_does_not_get_rectangle_corners():
    """Four corners at one median elevation describe a plane, not a dome."""
    lidar, _ = dome_scene()
    feats, _ = roof_check_features(lidar, max_height_m=None, select="tallest")
    kinds = {f.provenance for f in feats}
    assert "lidar:minAreaRect" not in kinds
    assert kinds <= {"lidar:apex", "lidar:surface_sample"}
    assert len({f.xyz_local_m[2] for f in feats}) > 1, \
        "a curved roof must not report one elevation for every feature"


def test_every_feature_sits_on_measured_surface():
    """The invariant. A check feature is a measurement or it is nothing."""
    lidar, crown = dome_scene()
    feats, _ = roof_check_features(lidar, max_height_m=None, select="tallest")
    assert feats
    for f in feats:
        x, y, z = f.xyz_local_m
        local = surface_z_near((x, y), crown)
        assert local is not None, f"{f.feature_id} has no surface under it"
        assert abs(z - local) < 0.5, \
            f"{f.feature_id} floats {abs(z - local):.2f} m off the surface"


def test_the_rectangle_corners_land_off_the_dome_entirely():
    """Shows the failure the detection prevents, rather than asserting it.

    A min-area rectangle around a circular footprint puts its corners at
    radius*sqrt(2) -- outside the dome. So the old construction did not merely
    place corners at the wrong elevation; it placed them where the roof does
    not exist, and reported them as measured positions.
    """
    import cv2

    _, crown = dome_scene()
    corners = cv2.boxPoints(cv2.minAreaRect(crown[:, :2].astype(np.float32)))
    cx, cy = crown[:, 0].mean(), crown[:, 1].mean()
    radius = np.hypot(crown[:, 0] - cx, crown[:, 1] - cy).max()

    unsupported = 0
    for c in corners:
        assert np.hypot(c[0] - cx, c[1] - cy) > radius * 1.2, \
            "a rect corner should sit well outside a circular footprint"
        if surface_z_near((float(c[0]), float(c[1])), crown, radius=1.5) is None:
            unsupported += 1
    assert unsupported == 4, "every rect corner should be off the measured surface"


def test_a_flat_roof_still_gets_rectangle_corners():
    lidar, _ = stacked_scene()
    feats, _ = roof_check_features(lidar, max_height_m=None, select="tallest")
    assert {f.provenance for f in feats} == {"lidar:minAreaRect", "lidar:centroid"}
    assert len({round(f.xyz_local_m[2], 3) for f in feats}) == 1


# ------------------------------------------------- volume base is measured

def test_volume_base_is_measured_ground_not_roof_minus_a_constant():
    """The base of the bounding volume decides whether it holds the building.

    ``roof - 8 m`` encodes a single-storey building. It reaches ground on a
    1.3 m warehouse roof, so the pilot never exposed it; on a 160 m roof it
    produces a thin slab floating near the top with the whole tower outside.

    ``ground_elevation_under`` reads the apron just outside the footprint,
    because returns *inside* it are the roof.
    """
    from kyfromabove_3d import BoundingVolume3D
    from kyfromabove_3d.features import ground_elevation_under, roof_footprint

    ground = slab(0.0, 0.0, 120.0, 0.0, spacing=2.0, seed=11)
    tower = slab(50.0, 50.0, 20.0, 160.0, spacing=0.6, seed=12)
    lidar = SimpleNamespace(
        xyz_local_m=np.vstack([ground, tower]),
        classification=np.concatenate([np.full(len(ground), 2), np.full(len(tower), 1)]))

    _, roof = roof_check_features(lidar, max_height_m=None, select="tallest")
    footprint = roof_footprint(roof)
    corners3 = np.column_stack(
        [footprint, np.full(len(footprint), float(np.median(roof[:, 2])))])

    measured = ground_elevation_under(lidar, corners3)
    assert abs(measured) < 5.0, "apron ground sits near z=0 in this fixture"

    good = BoundingVolume3D.from_footprint_and_elevation(
        footprint, min_z_m=float(measured), max_z_m=float(roof[:, 2].max()),
        pad_xy_m=6.0, pad_z_m=2.0)
    assert good.max_z_m - good.min_z_m > 150.0, "volume must span the tower"
    assert good.min_z_m < 10.0, "volume must reach the ground"

    # The old rule produces a slab that excludes almost the entire building.
    legacy = BoundingVolume3D.from_footprint_and_elevation(
        footprint, min_z_m=float(roof[:, 2].min()) - 8.0,
        max_z_m=float(roof[:, 2].max()), pad_xy_m=6.0, pad_z_m=2.0)
    assert legacy.max_z_m - legacy.min_z_m < 30.0
    assert legacy.min_z_m > 100.0, "the slab floats near the roof"
