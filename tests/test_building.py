"""Step 3's bookkeeping, with no network: the cameras rebuilt from a piece's
own frame records, the neighbours gathered from the measured tiles, the
per-stage times and memory read back from the scripts, and publishing a
building -- its summary last, the crops left behind, an unchanged one
skipped."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

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


def _built(folder: Path, gers="g1"):
    """A built building's folder as the scripts leave it, crops and all."""
    b = folder / gers
    for d in ("3dtiles_mesh/textures", "3dtiles_complete/textures", "images", "validity_masks"):
        (b / d).mkdir(parents=True, exist_ok=True)
    ts = json.dumps({"root": {"content": {"uri": "building.glb"}}})
    (b / "3dtiles_mesh" / "building.glb").write_bytes(b"x" * 100)
    (b / "3dtiles_mesh" / "tileset.json").write_text(ts)
    (b / "3dtiles_mesh" / "textures" / "walls_0.png").write_bytes(b"p" * 30)
    (b / "3dtiles_complete" / "building.glb").write_bytes(b"y" * 50)
    (b / "3dtiles_complete" / "tileset.json").write_text(ts)
    (b / "3dtiles_complete" / "textures" / "walls_0.png").write_bytes(b"q" * 20)
    (b / "3dtiles_complete" / "building_tint.glb").write_bytes(b"t" * 40)          # for review
    (b / "3dtiles_complete" / "fill_faces.npz").write_bytes(b"f" * 10)            # for review
    (b / "images" / "frame_000.png").write_bytes(b"crop")
    (b / "audit.json").write_text("{}")
    (b / "levels.json").write_text('{"levels": []}')
    (b / "lidar_local.laz").write_bytes(b"clip")
    (b / "log.txt").write_text("log")
    return b


def test_a_building_is_five_objects_summary_last_without_its_crops(tmp_path):
    import zipfile

    b = _built(tmp_path / "work")
    store = LocalStore(tmp_path / "bucket")
    order = []
    real = store.put_bytes
    store.put_bytes = lambda k, d, *a, **kw: (order.append(k), real(k, d, *a, **kw))[1]
    s = B.publish_building(b, {"gers": "g1", "fingerprint": "f"}, store)
    base = "buildings/g1"
    assert order[-1] == f"{base}/building.json"                           # the marker last
    assert sorted(order) == sorted(f"{base}/{n}" for n in (
        "3dtiles_mesh.zip", "3dtiles_complete.zip", "review.zip", "levels.json", "building.json"))
    mesh = zipfile.ZipFile(tmp_path / "bucket" / base / "3dtiles_mesh.zip")
    assert sorted(mesh.namelist()) == ["building.glb", "textures/walls_0.png", "tileset.json"]
    complete = zipfile.ZipFile(tmp_path / "bucket" / base / "3dtiles_complete.zip")
    assert sorted(complete.namelist()) == ["building.glb", "textures/walls_0.png", "tileset.json"]
    review = sorted(zipfile.ZipFile(tmp_path / "bucket" / base / "review.zip").namelist())
    assert review == ["3dtiles_complete/building_tint.glb", "3dtiles_complete/fill_faces.npz", "audit.json", "log.txt"]
    every = "".join(order) + "".join(review)
    assert "images" not in every and "lidar_local" not in every             # crops and the clip stay behind
    assert s["files"]["3dtiles_mesh/building.glb"] == 100 and s["bytes"] == sum(s["files"].values())
    assert set(s["objects"]) == {"3dtiles_mesh.zip", "3dtiles_complete.zip", "review.zip", "levels.json"}


def test_the_same_files_pack_into_the_same_bytes(tmp_path):
    """Deterministic zips: the same files, written at another moment, give the
    same objects (so an unchanged rebuild uploads identical bytes)."""
    import os
    import time

    b1 = _built(tmp_path / "one")
    one, _ = B.pack_building(b1)
    time.sleep(1.1)
    b2 = _built(tmp_path / "two")
    for p in b2.rglob("*"):
        os.utime(p, (1e9, 1e9))
    two, _ = B.pack_building(b2)
    assert one == two
