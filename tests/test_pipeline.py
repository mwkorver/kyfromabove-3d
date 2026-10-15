"""The one building pipeline (kyfromabove_3d.pipeline): the fingerprint that
says whether a building must be built again."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kyfromabove_3d import pipeline as P  # noqa: E402

PIECE = Path(__file__).parent / "data" / "piece_small"


def _building(tmp: Path, name: str) -> Path:
    b = tmp / "buildings" / name
    b.mkdir(parents=True)
    for f in PIECE.iterdir():
        shutil.copy(f, b / f.name)
    return b


def test_the_fingerprint_holds_while_nothing_changes_and_moves_when_anything_does(tmp_path):
    b = _building(tmp_path, "a")
    fp = P.fingerprint(b, tmp_path / "buildings")
    assert fp == P.fingerprint(b, tmp_path / "buildings")
    # its own measured levels
    (b / P.MEASURED_LEVELS).write_text(json.dumps([{"level": 0, "ring": [[0, 0], [1, 0], [1, 1], [0, 0]], "z_lo": 0, "z_hi": 9}]))
    fp2 = P.fingerprint(b, tmp_path / "buildings")
    assert fp2 != fp
    # a neighbour's measured levels (they hide its walls from the frames)
    n = _building(tmp_path, "b")
    (n / P.MEASURED_LEVELS).write_text(json.dumps([{"level": 0, "ring": [[0, 0], [1, 0], [1, 1], [0, 0]], "z_lo": 0, "z_hi": 20}]))
    fp3 = P.fingerprint(b, tmp_path / "buildings")
    assert fp3 != fp2
    # its class's settings
    (b / "building_class.json").write_text(json.dumps({"class": "box/tall"}))
    assert P.fingerprint(b, tmp_path / "buildings") != fp3


def test_the_code_version_is_a_digest_of_the_code():
    v = P.code_version()
    assert len(v) == 16 and v == P.code_version()
