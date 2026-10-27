"""The viewer's settings: its views, its start, and the places its "fly to"
buttons land at (from far away the tree shows only the big buildings)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from kyfromabove_3d.stages import publish_step                              # noqa: E402

from kyfromabove_3d.store import LocalStore                                # noqa: E402


def test_the_viewer_config_has_its_views_its_start_and_places_in_kentucky(tmp_path):
    store = LocalStore(tmp_path)
    start = {"lon": -85.76, "lat": 38.24, "height": 12000.0, "heading": 15, "pitch": -40}
    publish_step.viewer(store, start)
    cfg = store.read_json("viewer/config.json")
    assert set(cfg["tilesets"]) == set(publish_step.PRODUCTS) and cfg["start"] == start
    names = [p["name"] for p in cfg["places"]]
    assert "Downtown Louisville" in names and "Lexington" in names and len(set(names)) == len(names)
    for p in cfg["places"]:
        assert -89.6 < p["lon"] < -81.9 and 36.4 < p["lat"] < 39.2                 # Kentucky
        assert math.isfinite(p["ground_m"]) and 0 < p["ground_m"] < 1300
    page = store.get_bytes("viewer/index.html").decode("utf-8")
    assert 'id="places"' in page and "flyToBoundingSphere" in page
