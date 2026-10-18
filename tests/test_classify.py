"""The classifier on a real building too small for its roof measures:
3a7c18d7, a 6 m shed on N077E228 whose roof has no cell away from its own
edge. It used to stop the measure job (the lowest height of no cells); it is
measured now as a roof with too few cells, and classed as one the LiDAR
cannot see."""

from __future__ import annotations

import shutil
from pathlib import Path

from kyfromabove_3d.classify import classify_building

SHED = Path(__file__).parent / "data" / "piece_tiny_shed"      # footprint, local frame, LiDAR clip


def test_a_roof_with_no_cell_away_from_its_edge_is_measured_without_its_roof_measures(tmp_path):
    d = tmp_path / "3a7c18d7-3778-458a-bc26-66f1d886ea7c"
    shutil.copytree(SHED, d)
    rec = classify_building(d)
    assert rec["status"] == "ok" and rec["class"] == "none"
    assert rec["area_m2"] < 50 and rec["width_m"] < 7
    assert all(rec[k] is None for k in ("flat_share", "pitched_share", "main_roof_m", "lower_share"))
    assert rec["tiers"] == 0
    assert (d / "building_class.json").exists()
