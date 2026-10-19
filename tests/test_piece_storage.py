"""How a measured piece is stored: one zip and its marker (two uploads, not
a dozen), read back byte for byte, the old layout (a file each) still read;
and each tile's buildings' frames and levels in one object, which a build
batch reads for its neighbours instead of two small objects each."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from kyfromabove_3d import building as B
from kyfromabove_3d.store import LocalStore

PIECE = Path(__file__).parent / "data" / "piece_small"


class _Counting(LocalStore):
    def __init__(self, root):
        super().__init__(root)
        self.puts, self.reads = [], 0

    def put_bytes(self, key, data, *a, **kw):
        self.puts.append(key); return super().put_bytes(key, data, *a, **kw)

    def put_file(self, key, path, *a, **kw):
        self.puts.append(key); return super().put_file(key, path, *a, **kw)

    def get_bytes(self, key):
        self.reads += 1; return super().get_bytes(key)

    def get_file(self, key, path):
        self.reads += 1; return super().get_file(key, path)


def _measured_piece(tmp_path: Path, gers: str = "g1") -> Path:
    d = tmp_path / "work" / gers
    shutil.copytree(PIECE, d)
    m = json.loads((d / "piece.json").read_text(encoding="utf-8"))
    (d / "piece.json").write_text(json.dumps({**m, "gers": gers}, indent=1), encoding="utf-8")
    (d / "building_class.json").write_text('{"class": "house"}', encoding="utf-8")
    (d / "levels_measured.json").write_text('{"levels": [{"z": 7.5}]}', encoding="utf-8")
    return d


def _files(d: Path) -> dict:
    return {p.name: p.read_bytes() for p in d.iterdir() if p.is_file()}


def test_a_piece_is_one_zip_and_its_marker_and_comes_back_byte_for_byte(tmp_path):
    src = _measured_piece(tmp_path)
    store = _Counting(tmp_path / "bucket")
    B.publish_piece(src, store)
    assert store.puts == ["pieces/g1/piece.zip", "pieces/g1/piece.json"]          # the marker last
    assert store.read_json("pieces/g1/piece.json")["layout"] == B.PIECE_LAYOUT
    out = tmp_path / "pulled" / "g1"
    store.reads = 0
    B.pull_piece("g1", store, out)
    assert store.reads <= 2                                                       # the marker and the zip
    assert _files(out) == _files(src)                                             # piece.json too: no layout


def test_a_piece_stored_the_old_way_is_still_read(tmp_path):
    src = _measured_piece(tmp_path)
    store = LocalStore(tmp_path / "bucket")
    for p in src.iterdir():
        store.put_file(f"pieces/g1/{p.name}", p)                                  # a file each, as before
    out = tmp_path / "pulled" / "g1"
    B.pull_piece("g1", store, out)
    assert _files(out) == _files(src)


def test_a_tiles_measured_file_merges_and_gives_its_neighbours_back_byte_for_byte(tmp_path):
    store = LocalStore(tmp_path / "bucket")
    a, b, c = (_measured_piece(tmp_path / n, n) for n in ("ga", "gb", "gc"))
    assert B.publish_tile_measured("T1", [a, b], store) == 2
    assert B.publish_tile_measured("T1", [b], store) == 2                         # a re-measure keeps ga
    # gc: an old-layout neighbour, its two files in its piece
    for name in B.NEIGHBOUR_FILES:
        store.put_file(f"pieces/gc/{name}", c / name)
    out = tmp_path / "nb"
    n = B.pull_neighbours(["ga", "gb", "gc", "nowhere"], ["T1", "T2"], store, out)
    assert n == {"tile_files": 1, "from_tiles": 2, "files_from_pieces": 2}
    for d in (a, b, c):
        for name in B.NEIGHBOUR_FILES:
            assert (out / d.name / name).read_bytes() == (d / name).read_bytes()
    assert not (out / "nowhere").exists()


def _old_layout(store, src):
    for p in src.iterdir():
        store.put_file(f"pieces/{src.name}/{p.name}", p)


def test_repacking_an_old_piece_checks_it_then_removes_its_loose_files_and_can_run_again(tmp_path):
    src = _measured_piece(tmp_path)
    store = LocalStore(tmp_path / "bucket")
    _old_layout(store, src)
    r = B.repack_piece("g1", store, tmp_path / "w1")
    assert r["repacked"] and r["removed"] == len(list(src.iterdir())) - 1             # all but the marker
    assert sorted(k.split("/")[-1] for k in store.list("pieces/g1/")) == ["piece.json", "piece.zip"]
    out = tmp_path / "pulled" / "g1"
    B.pull_piece("g1", store, out)
    assert _files(out) == _files(src)
    store.put_file("pieces/g1/frames.json", src / "frames.json")                      # a leftover from a cut run
    r = B.repack_piece("g1", store, tmp_path / "w2")
    assert not r["repacked"] and r["removed"] == 1


def test_a_piece_whose_zip_does_not_check_keeps_its_loose_files(tmp_path, monkeypatch):
    src = _measured_piece(tmp_path)
    store = LocalStore(tmp_path / "bucket")
    _old_layout(store, src)
    real = B.publish_piece

    def broken(folder, st, prefix="pieces"):
        (Path(folder) / "frames.json").write_text("not the frames")                    # what gets zipped differs
        m = real(folder, st, prefix)
        (Path(folder) / "frames.json").write_bytes((src / "frames.json").read_bytes())
        return m
    monkeypatch.setattr(B, "publish_piece", broken)
    import pytest
    with pytest.raises(ValueError, match="does not give back"):
        B.repack_piece("g1", store, tmp_path / "w")
    assert store.exists("pieces/g1/frames.json") and store.exists("pieces/g1/coordinates.json")   # nothing removed
