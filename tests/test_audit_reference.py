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


def _facing(b: Path, azimuth: float, within: float = 30.0) -> list[str]:
    """The faces above the ground level whose outward side points within
    ``within`` degrees of ``azimuth`` (0 north, 90 east). The audit numbers
    faces from the building's outline as it is now, so a wall is found by the
    way it faces, not by a number that a change to the outline moves."""
    from kyfromabove_3d.features import LEVEL_FACE_STRIDE
    from kyfromabove_3d.stages.audit_building import level_edges, lidar_model
    out = []
    for e in level_edges(lidar_model(b)["levels"]):
        n = getattr(e, "normal", None)
        if n is None or e.face < LEVEL_FACE_STRIDE:
            continue
        az = np.degrees(np.arctan2(n[0], n[1])) % 360
        if abs((az - azimuth + 180) % 360 - 180) <= within:
            out.append(str(e.face))
    return out


def test_the_omni_as_it_was_has_both_holes_and_as_it_is_has_neither():
    from kyfromabove_3d.stages.audit_building import audit
    _need(OMNI / "3dtiles_mesh" / "building.glb")
    before = audit(OMNI, glb=KNOWN / "omni_before_mesh.glb")["holes"]
    now = audit(OMNI)["holes"]
    # the west tower wall's staircase, and the south wing's wall: the face
    # holed worst before must have no such hole now
    for azimuth, z_lo, z_hi, gone in ((270, 47, 72, 50), (180, 20, 55, 200)):
        face = max(_facing(OMNI, azimuth), key=lambda f: _hole_m2(before, f, z_lo, z_hi))
        assert _hole_m2(before, face, z_lo, z_hi) >= 1000
        assert _hole_m2(now, face, z_lo, z_hi) < gone


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
