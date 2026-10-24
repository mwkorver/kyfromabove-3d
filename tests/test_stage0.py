"""Stage 0 over a list of tiles: the box fetched covers the tiles and the
margin around them, and the tile list keeps the tiles written before."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage0_index                                                          # noqa: E402

from kyfromabove_3d.pieces import phase2_grid                               # noqa: E402
from kyfromabove_3d.store import LocalStore                                 # noqa: E402


def _grid():
    """The real Phase 2 tile grid: in data/raw on the laptop, not in git (CI skips)."""
    raw = ROOT / "data" / "raw"
    if not (raw / "kyfromabove_phase2_pointcloud_5k_grid.gpkg").exists():
        pytest.skip("the LiDAR tile grid is not on this machine")
    return phase2_grid(raw)


def test_the_box_holds_the_tiles_and_the_margin_around_them():
    from pyproj import Geod, Transformer

    grid = _grid()
    tiles = ["N090E300", "N090E301", "N091E300"]
    lon0, lat0, lon1, lat1 = stage0_index.tiles_bbox(grid, tiles, 400.0)
    to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
    x0, y0, x1, y1 = grid[grid["tilename"].isin(tiles)].total_bounds
    (tlon0, tlon1), (tlat0, tlat1) = to_ll([x0, x1], [y0, y1])
    geod = Geod(ellps="WGS84")
    for a, b in (((lon0, tlat0), (tlon0, tlat0)), ((tlon1, tlat1), (lon1, tlat1)),
                 ((tlon0, lat0), (tlon0, tlat0)), ((tlon1, tlat1), (tlon1, lat1))):
        d = geod.inv(a[0], a[1], b[0], b[1])[2]
        assert 395 < d < 405, d                                     # 400 m on the ground on every side


def test_the_tile_list_keeps_the_tiles_written_before(tmp_path, monkeypatch):
    store = LocalStore(tmp_path / "store")
    store.put_json("inputs/tiles.json", {"release": "old", "tiles": [{"tile": "N077E230", "owned": 5, "halo": 1},
                                                                     {"tile": "N090E300", "owned": 1, "halo": 0}]})
    grid = _grid()
    c = grid[grid["tilename"] == "N090E300"].geometry.iloc[0].centroid
    from pyproj import Transformer
    lon, lat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform(c.x, c.y)
    sq = f"POLYGON(({lon} {lat}, {lon + 1e-4} {lat}, {lon + 1e-4} {lat + 1e-4}, {lon} {lat + 1e-4}, {lon} {lat}))"
    src = tmp_path / "overture.json"
    src.write_text(json.dumps({"release": "new", "buildings": [{"id": "a", "wkt": sq}]}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["stage0_index.py", "--overture", str(src), "--store", str(tmp_path / "store"),
                                      "--raw", str(ROOT / "data" / "raw")])
    assert stage0_index.main() == 0
    listed = {t["tile"]: t for t in store.read_json("inputs/tiles.json")["tiles"]}
    assert set(listed) == {"N077E230", "N090E300"}
    assert listed["N077E230"]["owned"] == 5 and listed["N090E300"]["owned"] == 1 and listed["N090E300"]["release"] == "new"
