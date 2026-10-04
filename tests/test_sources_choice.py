"""Which LiDAR and which obliques a tile's buildings are made from: the pair
closest together in time (the user's rule, September 2026)."""

from __future__ import annotations

import datetime as dt

import pytest

from kyfromabove_3d import sources_choice as S


def _l(phase, y, m, d=15):
    return {"phase": phase, "date": dt.datetime(y, m, d), "points_per_m2": 4.0}


def _o(c, y, m, d=15):
    return {"collection": c, "date": dt.datetime(y, m, d), "n_frames": 100}


def test_the_pair_closest_in_time_wins():
    l, o = S.choose([_l("phase2", 2019, 3), _l("phase3", 2024, 2)], [_o("2023_Season1", 2023, 3), _o("2024_Season1", 2024, 3)])
    assert (l["phase"], o["collection"]) == ("phase3", "2024_Season1")
    l, o = S.choose([_l("phase2", 2022, 12)], [_o("2023_Season1", 2023, 3), _o("2024_Season1", 2024, 3)])
    assert o["collection"] == "2023_Season1"                          # not simply the newest


def test_a_tie_goes_to_the_newer_lidar_and_an_undated_candidate_loses():
    l, _ = S.choose([_l("phase2", 2023, 1, 1), _l("phase3", 2023, 5, 1)], [_o("x", 2023, 3, 1)])
    assert l["phase"] == "phase2"                                     # 59 days against 61: the closer wins
    l, _ = S.choose([_l("phase2", 2023, 1, 1), _l("phase3", 2023, 5, 1)], [_o("x", 2023, 3, 2)])
    assert l["phase"] == "phase3"                                     # 60 days each: the newer LiDAR
    undated = {"phase": "phase3", "date": None, "points_per_m2": 3.0}
    l, _ = S.choose([undated, _l("phase2", 2019, 3)], [_o("x", 2024, 3)])
    assert l["phase"] == "phase2"
    with pytest.raises(RuntimeError):
        S.choose([], [_o("x", 2024, 3)])


def test_gps_times_read_as_the_dates_found_by_hand():
    # the COPC headers' midpoints of three tiles dated on 2026-09-27
    assert S.gps_to_datetime(354922918).strftime("%Y-%m-%d") == "2022-12-12"     # N076E229, Phase 2
    assert S.gps_to_datetime(360828329).strftime("%Y-%m") == "2023-02"           # N148E378, Phase 2
    assert S.gps_to_datetime(391094074).strftime("%Y-%m-%d") == "2024-02-04"     # N016E301, Phase 3
    assert S.gps_to_datetime(302400.0) is None                                  # week time: no date


def test_the_provenance_fields():
    p = S.provenance(_l("phase2", 2022, 12, 12), _o("2024_Season1", 2024, 3, 1))
    assert p["lidar_phase"] == "phase2" and p["lidar_date"] == "2022-12-12"
    assert p["oblique_collection"] == "2024_Season1" and p["oblique_date"] == "2024-03-01"
    assert p["gap_months"] == pytest.approx(14.6, abs=0.1)


def test_the_tile_step_stamps_its_own_buildings_but_not_its_halo(monkeypatch):
    import numpy as np

    from kyfromabove_3d import pieces as P
    from kyfromabove_3d import product1

    monkeypatch.setattr(P, "tile_origin_ft", lambda grid, t: (0.0, 0.0))
    monkeypatch.setattr(product1, "local_polygons_about", lambda recs, ox, oy: [None] * len(recs))
    monkeypatch.setattr(product1, "area_cloud_local", lambda *a, **k: (np.zeros((0, 3)), np.zeros(0)))
    monkeypatch.setattr(product1, "measure_heights", lambda recs, polys, xyz, cls: len(recs))
    monkeypatch.setattr(product1, "split_records", lambda owned, *a, **k: ([dict(r) for r in owned], []))
    prov = {"lidar_phase": "phase3", "gap_months": 1.0}
    res = P.tile_step("T", [{"id": "a"}], [{"id": "h"}], None, None, log=lambda m: None, provenance=prov)
    assert res["buildings"][0]["lidar_phase"] == "phase3" and "lidar_phase" not in res["halo"][0]
    assert res["stats"]["gap_months"] == 1.0
