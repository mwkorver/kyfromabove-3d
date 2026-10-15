"""One building pipeline: every building, on the laptop and in the cloud,
goes through the same two phases (refactor step 2, 2026-10-14).

**Measure** -- everything that comes from the footprints, the LiDAR and the
frame index, without a pixel:

1. the piece (``prepare.prepare_piece``): ``coordinates.json``,
   ``footprint.json``, the LiDAR clip, the frames chosen -- kept in
   ``frames.json`` so the build phase can cut their crops;
2. the class (``classify.classify_building``): ``building_class.json``;
3. the LiDAR outline, where the class builds on it
   (``lidar_outline.outline_for``), and each tower's own outline from the
   LiDAR (``lidar_outline.towers_for``): ``footprint_lidar.json``;
4. the levels as the LiDAR shows them (the colourise's own
   ``footprint_and_levels``): ``levels_measured.json``.

**Build** -- the frame crops cut from ``frames.json``, the colourise, the mesh
and the completed product, the texture check, the audit.

Every building of a run is measured before any is built. A building's
colourise hides the walls its neighbours stand in front of, as prisms at
their levels; it reads the neighbours' ``levels_measured.json``, which is the
same whichever of them was built first. (It read their ``levels.json``,
written by their own colourise after the frames had moved a corner: what it
found depended on the order the run happened to take.)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

#: the levels as the LiDAR shows them, before the frames move a corner
MEASURED_LEVELS = "levels_measured.json"


@dataclass
class Area:
    """What every building of a run shares: the footprints (the neighbours,
    with their LiDAR heights), the frame index, the LiDAR, and how frames
    are chosen."""
    records: Sequence[Dict[str, Any]]
    frames_gdf: Any
    index: Any
    pad_ft: float = 90.0
    max_per_head: int = 4
    min_frames: int = 3
    fallback_ground_m: Optional[float] = None


def write_measured_levels(bdir: Path, levels) -> Path:
    import numpy as np
    path = Path(bdir) / MEASURED_LEVELS
    path.write_text(json.dumps(
        [{"level": lv["level"], "ring": np.asarray(lv["ring"]).round(3).tolist(),
          "z_lo": round(float(lv["z_lo"]), 3), "z_hi": round(float(lv["z_hi"]), 3)} for lv in levels],
        indent=1), encoding="utf-8")
    return path


def measure(rec: Dict[str, Any], bdir: Path, area: Area, log: Callable = print) -> Dict[str, Any]:
    """Phase 1 for one building (see the module's docstring). Returns what
    it found, for the run's state."""
    import sys

    from .classify import classify_building
    from .pieces import frames_record
    from .prepare import prepare_piece
    from .settings import resolve

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import lidar_outline
    from colorize_building import ColouriseFailure, footprint_and_levels

    bdir = Path(bdir)
    t0 = time.time()
    out: Dict[str, Any] = {}
    res, chosen = prepare_piece(rec, bdir, area.frames_gdf, area.index, area.records, area.pad_ft,
                                area.max_per_head, area.min_frames, area.fallback_ground_m, log)
    (bdir / "frames.json").write_text(json.dumps(frames_record(chosen, area.frames_gdf), indent=1, default=float),
                                      encoding="utf-8")
    out["prepare"] = res
    c = classify_building(bdir)
    out["class"] = c.get("class") if c.get("status") == "ok" else None
    settings = resolve(bdir)
    r = None
    if settings.colourise.outline == "lidar":
        r = lidar_outline.outline_for(bdir, area.records)
        path = lidar_outline.write_lidar_outline(bdir, r)
        out["lidar_outline"] = "written" if path else r.get("status")
        if not path:
            r = None
    try:
        F = footprint_and_levels(bdir, settings.colourise)
        if r is not None and len(F.levels) > 1:
            # each tower's own outline from the LiDAR, seeded by the levels just found; then the levels
            # again, the towers on their outlines (and a tower over its shaft split at the overhang)
            towers = lidar_outline.towers_for(bdir, F.levels, r)
            if towers:
                lidar_outline.write_lidar_outline(bdir, r, towers)
                out["towers"] = len(towers)
                F = footprint_and_levels(bdir, settings.colourise)
        write_measured_levels(bdir, F.levels)
        out["levels"] = len(F.levels)
    except ColouriseFailure as exc:
        (bdir / MEASURED_LEVELS).unlink(missing_ok=True)
        out["levels"] = None
        out["levels_error"] = str(exc)
    out["t_measure"] = round(time.time() - t0, 1)
    return out


def cut_crops_from_piece(bdir: Path, log: Callable = print) -> Dict[str, Any]:
    """Phase 2's first part: the crops of the frames the measure phase chose
    (``frames.json``); {} when no frame sees the building."""
    from .building import chosen_from_piece
    from .prepare import cut_crops

    chosen = chosen_from_piece(bdir)
    res: Dict[str, Any] = {}
    if chosen is not None:
        cut_crops(chosen, bdir, res, log)
    return res


# ------------------------------------------------------------------ the fingerprint

#: what the build is made of besides the package: the scripts the build phase
#: runs, and the classes' settings
_CODE = ("scripts/colorize_building.py", "scripts/build_mesh.py", "scripts/check_building.py",
         "scripts/audit_building.py", "scripts/lidar_outline.py", "scripts/wall_visibility.py",
         "configs/class_strategies.yaml")
#: the measure phase's files a building is built from
_MEASURED = ("coordinates.json", "footprint.json", "clip.json", "check_features.json", "frame_coverage.csv",
             "frames.json", "building_class.json", "footprint_lidar.json", MEASURED_LEVELS)
_code_version: Optional[str] = None


def code_version() -> str:
    """A digest of the code a building is built with: the package and the
    build's scripts and class settings, by content (no git needed, so the
    same in a container)."""
    global _code_version
    if _code_version is None:
        import hashlib
        root = Path(__file__).resolve().parents[2]
        h = hashlib.sha256()
        for f in sorted(Path(__file__).resolve().parent.glob("*.py")) + [root / c for c in _CODE]:
            if f.exists():
                h.update(f.name.encode())
                h.update(f.read_bytes())
        _code_version = h.hexdigest()[:16]
    return _code_version


def fingerprint(bdir: Path, buildings_dir: Path, radius_m: float = 250.0) -> str:
    """What a building's products are made from: its measured inputs, its
    resolved settings, its neighbours' measured levels within ``radius_m``
    (they hide its walls from the frames) and the code. A building whose
    fingerprint has not changed since it was built need not be built again;
    one whose has must be -- whatever changed."""
    import hashlib

    import numpy as np

    from .pieces import _canonical
    from .settings import resolve

    bdir = Path(bdir)
    h = hashlib.sha256(code_version().encode())
    for name in _MEASURED:
        p = bdir / name
        h.update(name.encode())
        h.update(_canonical(name, p.read_bytes()) if p.exists() else b"-")
    laz = bdir / "lidar_local.laz"
    if laz.exists():
        import laspy
        with laspy.open(laz) as f:
            h.update(str(f.header.point_count).encode())
    h.update(json.dumps(resolve(bdir).to_dict(), sort_keys=True, default=str).encode())
    me = np.asarray(json.loads((bdir / "coordinates.json").read_text(encoding="utf-8"))["origin_source_xyz"][:2])
    for nb in sorted(Path(buildings_dir).iterdir()):
        lv, co = nb / MEASURED_LEVELS, nb / "coordinates.json"
        if nb == bdir or not lv.exists() or not co.exists():
            continue
        o = np.asarray(json.loads(co.read_text(encoding="utf-8"))["origin_source_xyz"][:2])
        if np.hypot(*(o - me)) * 0.3048006096012192 <= radius_m + 100.0:     # (origins are building centres)
            h.update(nb.name.encode())
            h.update(_canonical(MEASURED_LEVELS, lv.read_bytes()))
    return h.hexdigest()
