"""Step 3's bookkeeping, with no network: the cameras rebuilt from a piece's
own frame records, the neighbours gathered from the measured tiles, the
per-stage times and memory read back from the scripts, and publishing a
building -- its summary last, the crops left behind, an unchanged one
skipped."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from kyfromabove_3d import building as B
from kyfromabove_3d.store import LocalStore

PIECE = Path(__file__).parent / "data" / "piece_small"      # a house on N077E229, from step 2


def test_the_cameras_rebuilt_from_a_piece_see_the_building_where_the_piece_step_did(tmp_path):
    import pandas as pd

    from kyfromabove_3d.sources import assess_frame_coverage

    for f in PIECE.iterdir():
        shutil.copy(f, tmp_path / f.name)
    chosen = B.chosen_from_piece(tmp_path)
    frames = json.loads((PIECE / "frames.json").read_text())
    assert [c.frame_id for c in chosen["cams"]] == [f["frame_id"] for f in frames]
    again = assess_frame_coverage(chosen["cams"], chosen["volume"]).set_index("frame_id")
    for f in frames:
        w = f["window_px"]
        for k in ("u_min", "u_max", "v_min", "v_max"):
            assert again.loc[f["frame_id"], k] == pytest.approx(w[k], abs=0.1)
    table = pd.read_csv(PIECE / "frame_coverage.csv")
    assert len(chosen["table"]) == len(table) and chosen["table"]["selected"].sum() == len(frames)


def test_a_piece_no_frame_sees_gives_no_crops(tmp_path):
    (tmp_path / "frames.json").write_text("[]")
    assert B.chosen_from_piece(tmp_path) is None


def test_the_tiles_around_a_tile():
    got = B.adjacent_tiles("N076E228")
    assert len(got) == 8 and "N075E227" in got and "N077E229" in got and "N076E228" not in got
    assert B.adjacent_tiles("not-a-tile") == []


def _r(gid, h=10.0, parent=None):
    return {"id": gid, "wkt": "POLYGON ((0 0, 1 0, 1 1, 0 0))", "lidar_ground_z": 140.0, "lidar_height_m": h,
            "parent_gers": parent, "extra": "dropped"}


def test_the_neighbours_come_from_the_tile_and_the_measured_tiles_around_it(tmp_path):
    s = LocalStore(tmp_path)
    s.put_json("tiles/N076E228/buildings.json",
               {"buildings": [_r("a"), _r("b")],
                "halo": [_r("c"), _r("x"), _r("p"), _r("far")]})        # c, x, p, far: other tiles' footprints
    s.put_json("tiles/N076E229/buildings.json",                          # measured: its own records win
               {"buildings": [_r("c", h=30.0), _r("p-a", parent="p"), _r("p-b", parent="p")], "halo": []})
    got = {r["id"]: r for r in B.neighbour_records("N076E228", s)}
    assert set(got) == {"a", "b", "c", "p-a", "p-b", "x", "far"}         # the split parent p stands down
    assert got["c"]["lidar_height_m"] == 30.0                             # from its own tile, not the halo
    assert "extra" not in got["a"]                                        # only what the prisms read


def test_the_stage_times_are_read_from_build_meshs_output():
    out = "colouring...\n[time] colourise 6.0 s\n  face 1\n[time] mesh 11.6 s (with the colourise)\n[time] complete 0.4 s\n"
    assert B._stage_times(out) == {"colourise": 6.0, "mesh": 11.6, "complete": 0.4}


def test_a_script_is_timed_and_its_own_peak_memory_read(tmp_path):
    code = "import numpy as np; a = np.ones(60_000_000, dtype=np.uint8); a[::4096] = 2; print('ok', a.sum())"
    rc, s, mb, out = B.run_script([sys.executable, "-c", code], tmp_path / "log.txt", 60)
    assert rc == 0 and "ok" in out and s >= 0 and mb > 55                 # the 60 MB array is in its peak
    assert "ok" in (tmp_path / "log.txt").read_text()
    rc, *_ = B.run_script([sys.executable, "-c", "import sys; sys.exit(3)"], tmp_path / "log.txt", 60)
    assert rc == 3
    rc, s, _, out = B.run_script([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "log.txt", 1)
    assert rc != 0 and s < 10 and "timed out" in out


def _built(folder: Path, gers="g1"):
    b = folder / gers
    for d in ("3dtiles_mesh", "3dtiles_complete", "images", "validity_masks"):
        (b / d).mkdir(parents=True, exist_ok=True)
    (b / "3dtiles_mesh" / "building.glb").write_bytes(b"x" * 100)
    (b / "3dtiles_mesh" / "tileset.json").write_text("{}")
    (b / "3dtiles_complete" / "building.glb").write_bytes(b"y" * 50)
    (b / "images" / "frame_000.png").write_bytes(b"crop")
    (b / "audit.json").write_text("{}")
    (b / "lidar_local.laz").write_bytes(b"clip")
    (b / "log.txt").write_text("log")
    return b


def test_a_building_is_published_summary_last_without_its_crops(tmp_path):
    b = _built(tmp_path / "work")
    store = LocalStore(tmp_path / "bucket")
    order = []
    real = store.put_file
    store.put_file = lambda k, p: (order.append(k), real(k, p))[1]
    real_json = store.put_json
    store.put_json = lambda k, o: (order.append(k), real_json(k, o))[1]
    s = B.publish_building(b, {"gers": "g1", "piece_hash": "h", "code_version": B.BUILDING_CODE_VERSION}, store)
    assert order[-1] == "buildings/g1/building.json"
    assert "buildings/g1/3dtiles_mesh/building.glb" in order and "buildings/g1/3dtiles_complete/building.glb" in order
    assert not any("images" in k or "lidar_local" in k for k in order)    # crops and the clip stay behind
    assert s["files"]["3dtiles_mesh/building.glb"] == 100 and s["bytes"] == 100 + 2 + 50 + 2 + 3


def test_a_building_built_from_the_same_piece_by_the_same_code_is_up_to_date(tmp_path):
    store = LocalStore(tmp_path)
    assert not B.up_to_date("g1", {"inputs_hash": "h"}, store)
    store.put_json("buildings/g1/building.json", {"piece_hash": "h", "code_version": B.BUILDING_CODE_VERSION})
    assert B.up_to_date("g1", {"inputs_hash": "h"}, store)
    assert not B.up_to_date("g1", {"inputs_hash": "changed"}, store)
    store.put_json("buildings/g1/building.json", {"piece_hash": "h", "code_version": "older"})
    assert not B.up_to_date("g1", {"inputs_hash": "h"}, store)


def test_the_building_step_skips_an_up_to_date_building_and_reports_a_missing_piece(tmp_path):
    store = LocalStore(tmp_path / "bucket")
    for f in PIECE.iterdir():
        store.put_file(f"pieces/g1/{f.name}", f)
    marker = json.loads((PIECE / "piece.json").read_text())
    store.put_json("buildings/g1/building.json", {"piece_hash": marker["inputs_hash"],
                                                  "code_version": B.BUILDING_CODE_VERSION})
    res = B.building_step(["g1", "nope"], store, tmp_path / "work", index=None, log=lambda m: None)
    assert res["skipped"] == ["g1"] and res["built"] == [] and list(res["failed"]) == ["nope"]
    assert np.isclose(res["seconds"]["scene"], 0.0, atol=0.5)             # nothing to build: no LiDAR read
