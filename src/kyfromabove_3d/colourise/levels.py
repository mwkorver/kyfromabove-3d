"""A building's outline and levels as the colourise measures them (the measure phase
of the pipeline reads these too)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kyfromabove_3d.clouds import find_cloud, read_cloud
from kyfromabove_3d.lidar import VEGETATION


class ColouriseFailure(Exception):
    """The colourise cannot go on; ``code`` is its exit code."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def footprint_and_levels(building: Path, CS, cli_lidar_outline: bool = False):
    """What the colourise finds before it looks at a frame: the building's
    returns (``CS.drop_classes`` left out, its neighbours' footprints cut out),
    its outline (Overture's, or the LiDAR's where the class or the command
    line asks), the ground and its levels -- the towers with the LiDAR's
    outlines, a level split under an overhang. The building step's measure
    phase records these levels for the neighbours' occlusion
    (``levels_measured.json``), the same whatever order the buildings are
    built in. Raises ColouriseFailure."""
    from types import SimpleNamespace
    coords = json.loads((building / "coordinates.json").read_text(encoding="utf-8"))
    ox, oy, oz = coords["origin_source_xyz"]

    lidar_path = find_cloud(building, "lidar_local")
    ply = read_cloud(lidar_path)
    xyz = np.column_stack([ply["x"], ply["y"], ply["z"]]).astype(np.float64)
    print(f"{len(xyz):,} LiDAR points from {lidar_path}")
    # the classification rides along with the points through every filter
    cls = np.asarray(ply["classification"]) if "classification" in ply else None
    if CS.drop_classes and cls is not None:
        # noise and water; the trees (lidar.label_trees) are left out of the
        # building's own returns alone (on_building, b_xyz below): a wall's plane, its edges and its profile
        # keep them, since a parapet or a sparse wall under a cornice can pass
        # for twigs (the Courtyard Louisville Downtown's walls stopped short,
        # holes 97 -> 228 m2), and a wall stands no higher than its level,
        # which the roof without the trees sets
        drop = np.isin(cls, tuple(c for c in CS.drop_classes if c not in VEGETATION))
        if drop.any():
            print(f"  {drop.sum():,} returns of class {sorted(set(cls[drop].tolist()))} left out")
            xyz, cls = xyz[~drop], cls[~drop]

    # No fallback anywhere below. An earlier version quietly took the first
    # entry when the lookup missed, and coloured 400 West Market against Tractor
    # Supply's footprint without saying so.
    from kyfromabove_3d.footprints import resolve_footprint
    # a class may be built on the LiDAR outline (configs/class_strategies.yaml: outline: lidar)
    lidar_outline = CS.outline == "lidar"
    if lidar_outline and not cli_lidar_outline:
        print("footprint: the class builds on the LiDAR outline")
    fp_file = None                                # the building's own footprint.json
    if lidar_outline and (building / "footprint_lidar.json").exists():
        # the outline from our own data; Overture's polygon stays in the record as outline.overture_wkt
        fp_file = building / "footprint_lidar.json"
        print("footprint: the LiDAR outline (footprint_lidar.json), not Overture's")
    elif lidar_outline:
        print("[warn] --lidar-outline: no footprint_lidar.json here; Overture's footprint used")
    try:
        poly_local, entry = resolve_footprint(building, ox, oy, footprint=fp_file)
    except (KeyError, ValueError) as exc:
        print(f"[FAIL] {exc}")
        raise ColouriseFailure(1, f"[FAIL] {exc}")
    ring_m = np.array(poly_local.exterior.coords, dtype=np.float64)
    # the outline's edges whose roof end the LiDAR found (a drop beyond it), in
    # the ring's own order: walls under them are drawn unless there is evidence
    # of a gap (fill_under_roof_edge, below)
    measured_edges = [(ring_m[int(i)].copy(), ring_m[int(i) + 1].copy())
                      for i, e in ((entry.get("outline") or {}).get("edges") or {}).items()
                      if e.get("step1") == "moved" and int(i) + 1 < len(ring_m)] if fp_file is not None and fp_file.name == "footprint_lidar.json" else []
    entry.setdefault("overture_name", entry.get("name"))
    if entry.get("neighbours"):
        # In a row of buildings the clip holds the neighbours too, and a
        # neighbour's roof at another height would be read as this
        # building's wall band. Their footprints (shrunk half a metre, so the
        # shared edge itself stays) are cut out of the returns.
        from shapely import contains_xy as _cxy

        from kyfromabove_3d.footprints import footprint_wgs84_to_local
        keep = np.ones(len(xyz), dtype=bool)
        for nb in entry["neighbours"]:
            g = footprint_wgs84_to_local(nb["wkt"], ox, oy).buffer(-0.5)
            if not g.is_empty:
                keep &= ~_cxy(g, xyz[:, 0], xyz[:, 1])
        if (~keep).any():
            print(f"  {(~keep).sum():,} returns inside {len(entry['neighbours'])} neighbouring footprint(s) left out")
            xyz = xyz[keep]
            cls = cls[keep] if cls is not None else None
    print(f"footprint: {entry.get('overture_name')} ({len(ring_m) - 1} vertices)")
    if CS.neighbour_spill_m > 0:
        # A taller neighbour's wall stands on the shared edge, where the cut above
        # leaves it, or a metre or two outside the outline: 9947f2b0, an 18 m
        # building beside 400 West Market, read 43 m tall and drew the tower's
        # wall as its own, as did 84 more downtown (cc3f65d1 50 m for 20 m, the
        # Marmaduke Building 44 m for 21 m). A return within 4 m of the outline
        # standing more than neighbour_spill_m over the top of the building's
        # inside (2 m in) is left out where it is outside the outline, or inside
        # it within 1.5 m of a neighbour's footprint. A wall of the building's own
        # standing proud of its outline (Humana's south wall) is not taller than
        # the building, and stays.
        import shapely as _sh
        from shapely.geometry import Polygon as _SP
        _poly = _SP(ring_m) if _SP(ring_m).is_valid else _SP(ring_m).buffer(0)
        _core = _poly.buffer(-2.0)
        in_core = _sh.contains_xy(_core, xyz[:, 0], xyz[:, 1]) if not _core.is_empty else np.zeros(len(xyz), bool)
        if in_core.sum() >= 50:
            top = float(np.percentile(xyz[in_core, 2], 99))
            pts = _sh.points(xyz[:, 0], xyz[:, 1])
            near_nb = np.zeros(len(xyz), dtype=bool)
            for nb in entry.get("neighbours") or []:
                from kyfromabove_3d.footprints import footprint_wgs84_to_local
                near_nb |= _sh.distance(footprint_wgs84_to_local(nb["wkt"], ox, oy), pts) < 1.5
            spill = (_sh.contains_xy(_poly.buffer(4.0), xyz[:, 0], xyz[:, 1]) & (xyz[:, 2] > top + CS.neighbour_spill_m)
                     & (~_sh.contains_xy(_poly, xyz[:, 0], xyz[:, 1]) | near_nb))
            if spill.any():
                print(f"  {spill.sum():,} returns of a taller neighbour's wall left out (over {top + CS.neighbour_spill_m:.1f} m, "
                      f"the inside's top {top:.1f} m)")
                xyz = xyz[~spill]
                cls = cls[~spill] if cls is not None else None

    # The zone split must run on THIS building, not the AOI. The AOI is 110 m
    # of downtown around the tower, so its modal elevation is the street, every
    # point lands above the roof line, and the whole cloud reads as roof --
    # which is the same way the tower's QA came to be measuring its neighbours.
    from shapely import contains_xy as _contains
    from shapely.geometry import Polygon
    poly_local = Polygon(ring_m)
    on_building = _contains(poly_local.buffer(CS.max_offset_m),
                            xyz[:, 0], xyz[:, 1])
    print(f"{on_building.sum():,} of {len(xyz):,} points fall on the footprint")
    if on_building.sum() < 200:
        print("[FAIL] the footprint and the LiDAR do not overlap")
        raise ColouriseFailure(2, "[FAIL] the footprint and the LiDAR do not overlap")
    if cls is not None and CS.drop_classes:
        tree = on_building & np.isin(cls, VEGETATION)
        if tree.any():
            print(f"  {tree.sum():,} tree returns over the footprint left out of the building")
            on_building = on_building & ~tree
    b_xyz = xyz[on_building]
    # The profile looks 3 m outside the footprint edge and 8 m inside; the
    # 2 m clip above would cut off a wall standing 2-3 m proud of a
    # footprint traced along its podium (Humana's south wall does).
    profile_xyz = xyz[_contains(poly_local.buffer(4.0), xyz[:, 0], xyz[:, 1])]
    # Ground elevation in the local frame. In the downtown/prepared frame, 0 is
    # the frame's ground level. Standalone cases use a robust lower percentile
    # so noise returns in cellars, river banks or pits (-10 m for RIVUE) do not
    # sit below street level and cause parking lots or plazas to count as roof.
    if (building / "footprint.json").exists():
        ground_z = 0.0
    elif cls is not None and (cls[on_building] == 2).sum() >= 20:
        ground_z = float(np.median(b_xyz[cls[on_building] == 2, 2]))
    else:
        ground_z = float(np.percentile(b_xyz[:, 2], 1.0))

    from kyfromabove_3d.features import building_levels
    roof_z = float(np.percentile(b_xyz[:, 2], 99))
    # the building's class may set how its levels are found (configs/class_strategies.yaml)
    from kyfromabove_3d.settings import LevelSettings
    if CS.levels != LevelSettings():
        print(f"  levels as the class sets them: {CS.levels}")
    lv_opts = CS.levels.kwargs()
    # fallback_only: the class's way is tried only where the tower tracing finds one level
    fallback_only = CS.levels.fallback_only
    if not CS.find_levels:
        levels = [{"level": 0, "ring": ring_m, "z_lo": ground_z, "z_hi": roof_z}]
    elif fallback_only:
        levels = building_levels(b_xyz, ring_m, ground_z)
        if len(levels) == 1:
            levels = building_levels(b_xyz, ring_m, ground_z, **lv_opts)
            if len(levels) > 1:
                print(f"  levels: one by the tower tracing; {len(levels)} by the class's way")
    else:
        levels = building_levels(b_xyz, ring_m, ground_z, **lv_opts)
    lidar_tower_levels: set[int] = set()                    # levels whose outline is the LiDAR's tower edge
    if lidar_outline and entry.get("towers_local") and len(levels) > 1:
        # each tower the colourise found takes the LiDAR's outline of it (footprint_lidar.json, lidar_outline.py
        # towers): traced on a grid fixed to the building, not re-traced from the footprint's corner each run
        from shapely.geometry import Polygon as _Poly
        ours = [_Poly(np.asarray(t["ring"], dtype=float)[:, :2]) for t in entry["towers_local"]]
        for lv in levels[1:]:
            mine = _Poly(np.asarray(lv["ring"], dtype=float)[:, :2])
            best = max(ours, key=lambda q: q.intersection(mine).area / max(q.union(mine).area, 1e-9))
            if best.intersection(mine).area / max(best.union(mine).area, 1e-9) >= 0.5:
                lv["ring"] = np.asarray(best.exterior.coords, dtype=float)
                lidar_tower_levels.add(lv["level"])
                print(f"  level {lv['level']}: the LiDAR's tower outline ({len(lv['ring']) - 1} corners)")
    # An upper part standing out over the part below it (the Hyatt's drum over
    # its shaft): the roof's outline is the drum's, the returns show the shaft
    # set back under it. Such a level is split at the overhang's underside, the
    # part below on the outline cut back to the shaft's wall (find_overhangs).
    # Only with the LiDAR's tower outlines; written to overhangs.json for the
    # mesh (the drum's underside) and the audit (the LiDAR solid under it).
    overhangs = []
    if lidar_outline and lidar_tower_levels:
        from kyfromabove_3d.features import find_overhangs
        split_levels = []
        for lv in levels:
            found = find_overhangs(lv["ring"], lv["z_lo"], lv["z_hi"], profile_xyz) if lv["level"] in lidar_tower_levels else []
            if not found:
                split_levels.append(lv)
                continue
            oh = found[0]
            split_levels.append({**lv, "ring": np.asarray(oh["ring_below_m"], dtype=float), "z_hi": oh["under_z"]})
            split_levels.append({**lv, "z_lo": oh["under_z"]})
            overhangs.append(oh)
            print(f"  level {lv['level']}: overhangs its shaft from {oh['under_z']:.1f} m up -- the wall below stands "
                  f"{-oh['setback_m']:.1f} m in ({oh['n_line_returns']} returns on it); split in two, {oh['cut_area_m2']:.0f} m2 cut "
                  f"from the outline below")
        if overhangs:
            old_to_new = {}
            for k, lv in enumerate(split_levels):
                old_to_new.setdefault(lv["level"], []).append(k)
                lv["level"] = k
            lidar_tower_levels = {k for old in lidar_tower_levels for k in old_to_new.get(old, [])}
            levels = split_levels
    roof_z = max(float(lv["z_hi"]) for lv in levels)
    return SimpleNamespace(ox=ox, oy=oy, oz=oz, xyz=xyz, ring_m=ring_m, entry=entry, measured_edges=measured_edges,
                           on_building=on_building, b_xyz=b_xyz, profile_xyz=profile_xyz, ground_z=ground_z,
                           levels=levels, lidar_tower_levels=lidar_tower_levels, overhangs=overhangs, roof_z=roof_z,
                           lidar_outline=lidar_outline)
