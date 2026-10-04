"""Moving buildings stored the old way (about twenty files under
``buildings/<gers>/``) to the new layout (five objects under
``buildings/<version>/<gers>/``), on a folder: copied by the building step's
own packing, the old files removed only once the copy holds every one."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

from kyfromabove_3d import building as B
from kyfromabove_3d.store import LocalStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

import migrate_buildings as M                                                  # noqa: E402

OLD = {"3dtiles_mesh/tileset.json": json.dumps({"root": {"content": {"uri": "building.glb"}}}).encode(),
       "3dtiles_mesh/building.glb": b"x" * 100,
       "3dtiles_mesh/textures/walls_0.png": b"p" * 30,
       "3dtiles_complete/tileset.json": json.dumps({"root": {"content": {"uri": "building.glb"}}}).encode(),
       "3dtiles_complete/building.glb": b"y" * 50,
       "3dtiles_complete/building_tint.glb": b"t" * 40,
       "audit.json": b"{}", "levels.json": b'{"levels": []}', "log.txt": b"log"}


def _old_building(store, gers="g1", version="v-old"):
    for rel, data in OLD.items():
        store.put_bytes(f"buildings/{gers}/{rel}", data)
    store.put_json(f"buildings/{gers}/building.json", {"gers": gers, "code_version": version, "piece_hash": "h"})


def test_an_old_building_is_copied_under_its_own_version_then_removed(tmp_path):
    st = LocalStore(tmp_path)
    _old_building(st)
    st.put_json("buildings/v-new/g9/building.json", {})                      # new layout: left alone
    old = M.old_buildings(st)
    assert list(old) == ["g1"] and len(old["g1"]) == len(OLD) + 1

    assert M.delete_one(st, "g1", old["g1"]) == "kept: no copy yet"         # nothing goes before a copy
    assert M.copy_one(st, "g1", old["g1"]) == "copied"
    assert M.copy_one(st, "g1", old["g1"]) == "already copied"
    new = sorted(k for k in st.list("buildings/v-old/g1/"))
    assert [k.rsplit("/", 1)[1] for k in new] == sorted(
        ["3dtiles_mesh.zip", "3dtiles_complete.zip", "review.zip", "levels.json", "building.json"])
    summary = st.read_json("buildings/v-old/g1/building.json")
    assert summary["piece_hash"] == "h" and summary["layout"] == B.LAYOUT
    assert summary["files"] == {rel: len(data) for rel, data in OLD.items()}
    mesh = zipfile.ZipFile(tmp_path / "buildings/v-old/g1/3dtiles_mesh.zip")
    assert mesh.read("building.glb") == OLD["3dtiles_mesh/building.glb"]

    assert M.delete_one(st, "g1", old["g1"]) == "deleted"
    assert not any(k.startswith("buildings/g1/") for k in st.list("buildings/"))
    assert not (tmp_path / "buildings" / "g1").exists()                     # no empty folders left
    assert M.old_buildings(st) == {}


def test_an_old_building_whose_copy_lacks_a_file_is_kept(tmp_path):
    st = LocalStore(tmp_path)
    _old_building(st)
    old = M.old_buildings(st)
    M.copy_one(st, "g1", old["g1"])
    st.put_bytes("buildings/g1/added_later.json", b"{}")                      # not in the copy
    assert M.delete_one(st, "g1", M.old_buildings(st)["g1"]).startswith("kept")
    assert st.exists("buildings/g1/building.json")
