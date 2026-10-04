"""Finding tall structures, and not being fooled while doing it.

Two things here are easy to get wrong in ways that still produce a ranked list.
Measuring height against a single datum reports terrain slope as building
height, which matters because downtowns sit on river valleys. And ranking cells
by their maximum return lets one bird outrank an entire tower. Both are tested
by construction rather than by inspecting real output, where a wrong answer
looks like a plausible one.
"""

import numpy as np
import pytest

from kyfromabove_3d.urban import (
    OBLIQUE_COLLECTIONS,
    collections_covering,
    eo_table_name,
    find_tall_clusters,
    frames_index_url,
    heights_above_ground,
)

IDENTITY = lambda x, y: (x, y)          # noqa: E731 -- tests work in feet


def ground_plane(n_side=40, spacing=25.0, slope=0.0):
    """A grid of class-2 returns, optionally tilted."""
    xs = np.arange(n_side) * spacing
    gx, gy = np.meshgrid(xs, xs)
    x, y = gx.ravel(), gy.ravel()
    return np.stack([x, y, 100.0 + slope * x], axis=1)


def test_height_is_measured_against_local_ground_not_a_global_datum():
    """A sloping site must not read as a building at the high end."""
    slope = 0.05                                   # 5 ft rise per 100 ft
    ground = ground_plane(slope=slope)
    xyz = ground
    cls = np.full(len(xyz), 2)
    h = heights_above_ground(xyz, cls, grid_ft=100.0)
    assert np.nanmax(np.abs(h)) < 3.0, "sloping ground should read as ~zero height"

    # Against a single datum the far end would read as the full rise.
    naive = xyz[:, 2] - np.median(xyz[:, 2])
    assert naive.max() > 20.0


def test_cells_without_ground_return_nan_rather_than_a_guess():
    ground = ground_plane(n_side=4, spacing=25.0)   # occupies x,y in [0, 75]
    roof = np.array([[5000.0, 5000.0, 250.0]])      # far away, no ground nearby
    xyz = np.vstack([ground, roof])
    cls = np.concatenate([np.full(len(ground), 2), [1]])
    h = heights_above_ground(xyz, cls, grid_ft=100.0)
    assert np.isnan(h[-1])
    assert np.isfinite(h[:-1]).all()


def _tower(x0, y0, height, n=400, footprint=50.0, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, footprint, size=(n, 2)) + np.array([x0, y0])
    z = 100.0 + rng.uniform(height - 5.0, height, size=n)
    return np.column_stack([p, z])


def test_a_single_spike_does_not_outrank_a_tower():
    """Ranking by max would put one noise return above a real building."""
    ground = ground_plane()
    tower = _tower(300.0, 300.0, 400.0)
    spike = np.array([[700.0, 700.0, 100.0 + 900.0]])   # one point, very high
    xyz = np.vstack([ground, tower, spike])
    cls = np.concatenate([np.full(len(ground), 2), np.full(len(tower), 1), [1]])

    got = find_tall_clusters(xyz, cls, IDENTITY, min_points=40, max_results=5)
    assert got, "expected at least the tower"
    assert got[0].height_ft == pytest.approx(400.0, abs=10.0)
    assert all(c.height_ft < 500 for c in got), "the 900 ft spike must not rank"


def test_noise_classes_are_excluded():
    ground = ground_plane()
    blunder = _tower(700.0, 700.0, 1800.0, n=200)      # a cloud of class-7 points
    xyz = np.vstack([ground, blunder])
    cls = np.concatenate([np.full(len(ground), 2), np.full(len(blunder), 7)])
    assert find_tall_clusters(xyz, cls, IDENTITY, min_points=40) == []


def test_one_building_yields_one_cluster_not_many_cells():
    """A 200 ft footprint spans several 60 ft cells; that is still one building."""
    ground = ground_plane()
    wide = _tower(300.0, 300.0, 300.0, n=3000, footprint=200.0)
    xyz = np.vstack([ground, wide])
    cls = np.concatenate([np.full(len(ground), 2), np.full(len(wide), 1)])

    got = find_tall_clusters(xyz, cls, IDENTITY, min_points=40,
                             min_separation_ft=400.0, max_results=10)
    assert len(got) == 1

    # With separation suppressed, the same building fragments into many cells.
    fragments = find_tall_clusters(xyz, cls, IDENTITY, min_points=40,
                                   min_separation_ft=0.0, max_results=10)
    assert len(fragments) > 1


def test_two_separated_buildings_are_both_reported():
    ground = ground_plane(n_side=60)
    a = _tower(200.0, 200.0, 500.0, seed=1)
    b = _tower(1100.0, 1100.0, 300.0, seed=2)
    xyz = np.vstack([ground, a, b])
    cls = np.concatenate([np.full(len(ground), 2), np.full(len(a) + len(b), 1)])
    got = find_tall_clusters(xyz, cls, IDENTITY, min_points=40, max_results=10)
    assert len(got) == 2
    assert got[0].height_ft > got[1].height_ft
    assert got[0].height_m == pytest.approx(got[0].height_ft * 0.3048006096)


def test_collection_lookup_and_urls():
    assert "2024_Season1" in collections_covering(-85.759, 38.255)   # Louisville
    assert collections_covering(-100.0, 40.0) == []                  # not Kentucky
    # The pilot corridor must not claim the urban collections.
    assert "2024_Season1" not in collections_covering(-82.949, 38.332)

    url = frames_index_url("2024_Season1")
    assert url.startswith("/vsicurl/"), "must range-read, never download 1 GB"
    assert url.endswith("_ImageFrames.gpkg")
    assert eo_table_name("2023_Season2") == "KY_KYAPED_2023_Season2_3IN_ImageFrameEO"
    with pytest.raises(KeyError):
        frames_index_url("2019_Season9")


def test_every_collection_declares_an_extent_and_a_note():
    for name, meta in OBLIQUE_COLLECTIONS.items():
        assert meta["lon"][0] < meta["lon"][1]
        assert meta["lat"][0] < meta["lat"][1]
        assert meta["note"]
