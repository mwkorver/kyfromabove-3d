"""The audit on real buildings (``pytest -m data``; needs the downtown
outputs on disk). Known answers first: the Omni as it was, with a hole in
its west tower wall and its south wing's wall missing, audits with both; the
Omni as it is audits with neither; the Seelbach's textures under the old
gap rule are full of lines. Then the reference set against its baseline:
nothing got worse (``scripts/audit_reference.py``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = pytest.mark.data

OMNI = ROOT / "outputs" / "downtown" / "buildings" / "b7daf154-da79-4a42-b938-a973aa1111b5"
KNOWN = ROOT / "tests" / "data" / "audit"


def _need(p: Path):
    if not p.exists():
        pytest.skip(f"{p.relative_to(ROOT)} is not on disk")


def _hole_m2(h: dict, face: str, z_lo: float, z_hi: float) -> float:
    """The largest hole on ``face`` reaching into [z_lo, z_hi], over the views."""
    return max([r["m2"] for v in h["views"] for r in v["holes"]
                if str(r["face"]) == face and r.get("z_hi", 0) > z_lo and r.get("z_lo", 1e9) < z_hi] or [0.0])


def test_the_omni_as_it_was_has_both_holes_and_as_it_is_has_neither():
    from kyfromabove_3d.stages.audit_building import audit
    _need(OMNI / "3dtiles_mesh" / "building.glb")
    before = audit(OMNI, glb=KNOWN / "omni_before_mesh.glb")["holes"]
    assert _hole_m2(before, "1012", 47, 72) >= 1000                       # the west tower wall's staircase
    assert max(_hole_m2(before, f, 20, 55) for f in ("1001", "1002")) >= 1000   # the south wing's wall
    now = audit(OMNI)["holes"]
    assert _hole_m2(now, "1012", 47, 72) < 50
    assert max(_hole_m2(now, f, 20, 55) for f in ("1001", "1002")) < 200


def test_the_old_gap_rule_left_lines_on_the_seelbach():
    from kyfromabove_3d.audit import texture_artefacts
    z = np.load(KNOWN / "seelbach_between_rule_fill_faces.npz", allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    srcs = [z[f"src_{i}"] for i, m in enumerate(meta) if m.get("kind") != "roof" and f"src_{i}" in z]
    old = texture_artefacts(srcs, float(z["texel_m"]))
    assert old["lines"] >= 20
    seel = OMNI.parent / "c5fe3ad2-ffa0-4800-9240-410277f7b332" / "3dtiles_complete" / "fill_faces.npz"
    _need(seel)
    z = np.load(seel, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    now = texture_artefacts([z[f"src_{i}"] for i, m in enumerate(meta) if m.get("kind") != "roof" and f"src_{i}" in z],
                            float(z["texel_m"]))
    assert now["lines"] <= old["lines"] // 10


def test_the_reference_set_got_no_worse():
    import audit_reference as R
    if not R.BASELINE.exists():
        pytest.skip("no baseline yet (scripts/audit_reference.py accept)")
    cfg = R.load_config()
    now = R.current(cfg)
    if not now:
        pytest.skip("no reference building on disk (scripts/audit_reference.py rebuild)")
    worse, _ = R.compare(cfg, now, json.loads(R.BASELINE.read_text(encoding="utf-8")))
    assert not worse, "\n".join(worse)
