#!/usr/bin/env python3
"""Colour a building's LiDAR from the obliques, and resample its walls.

The LiDAR has no colour at all -- point format 6, intensity only, and no
KyFromAbove phase publishes RGB. The obliques are 3-inch, which on a wall the
scanner reached at 1.90 m spacing is **25x finer linearly**. That gap is the
whole opportunity, and it is an appearance gap, not a geometry one.

So this does two things, and deliberately not a third:

**Colours the existing returns.** Every point is projected into every camera that
can see it, and the best view wins. No geometry is changed or inferred.

**Resamples the walls.** A plane is fitted to each wall face's returns -- facades
measure 0.104 m RMS about a plane, flatter than the roof filter assumes is
available -- then sampled at imagery resolution and coloured the same way. The
new points take their depth from the LiDAR's plane and only their colour from
the imagery.

**It does not triangulate.** The median angle between two views of the same point
is 10 degrees, so a 0.15 m matching error becomes 0.87 m along the ray, and
measured on 400 West Market triangulated points land 1.22 m from the LiDAR --
filling a 1.90 m gap with a 1.22 m error. Resampling a fitted plane costs
0.104 m of plane fit plus 0.153 m of reprojection instead.

Usage::

    python scripts/colorize_building.py --building outputs/downtown/buildings/df2fccca-d6bd-4d4d-b4bc-673cfb5245aa      # 400 West Market
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from kyfromabove_3d.coordinates import METERS_PER_US_SURVEY_FOOT
from kyfromabove_3d.features import (
    split_surface_zones,
    wall_planes,
    save_wall_planes,
    VerifiedWallPlane,
    rectify_footprint_corners,
)
from kyfromabove_3d.project import repo_relative
from kyfromabove_3d.colourise.occlusion import (  # noqa: E402,F401
    add_prism_occluders, depth_buffers, FOOTPRINTS, levels_dir, may_land_on_crop, neighbour_prisms,
    occluder_cloud, OCCLUSION_FILE, occlusion_maps, ROOT, scene_cloud, scene_cloud_from_cache,
)
from kyfromabove_3d.colourise.colour import (  # noqa: E402,F401
    load_cameras, project_visible, sample_colour, write_ply,
)
from kyfromabove_3d.colourise.walls import (  # noqa: E402,F401
    extrude_verified_faces, resample_walls, under_measured_edge,
)
from kyfromabove_3d.colourise.levels import ColouriseFailure, footprint_and_levels  # noqa: E402,F401


def parser() -> argparse.ArgumentParser:
    from kyfromabove_3d.settings import ColouriseSettings
    D = ColouriseSettings()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--building", type=Path, required=True,
                    help="a pipeline output dir holding lidar_local.laz (or .ply), "
                         "cameras_normalized.json and images/")
    ap.add_argument("--spacing-m", type=float, default=D.spacing_m,
                    help="wall resample pitch. 3-inch imagery is 0.076 m; the "
                         "default is deliberately coarser, because the "
                         "reprojection is only good to 0.153 m and sampling "
                         "finer than that invents detail the geometry cannot "
                         "place")
    ap.add_argument("--min-face-m", type=float, default=D.min_face_m)
    ap.add_argument("--look-in-share", type=float, default=D.look_in_share,
                    help="a face looks inside its level at most this share of the level's narrowest width (and 8 m); 0 for the fixed 8 m")
    ap.add_argument("--max-offset-m", type=float, default=D.max_offset_m)
    ap.add_argument("--tile-m", type=float, default=D.tile_m,
                    help="side of the square a plane is fitted to. One plane per "
                         "whole face rejects any building with setbacks")
    ap.add_argument("--max-plane-rms-m", type=float, default=D.max_plane_rms_m,
                    help="how far a tile's returns may sit off their own plane "
                         "before the tile is skipped rather than invented")
    ap.add_argument("--min-facing-cos", type=float, default=D.min_facing_cos)
    ap.add_argument("--occlusion-slack-m", type=float, default=D.occlusion_slack_m,
                    help="how much closer something must be to count as "
                         "blocking. Generous, because the depth map is built "
                         "from sparse LiDAR and a tight value rejects a wall "
                         "against its own returns")
    ap.add_argument("--scene-cloud", type=Path, default=None,
                    help="an .npz with xyz_ft (source feet) covering the area, written once by run_downtown.py; "
                         "windowed to --occluder-radius-m instead of range-reading the tiles")
    ap.add_argument("--drop-classes", type=lambda v: tuple(int(x) for x in v.split(",")) if v else (),
                    default=D.drop_classes,
                    help="LiDAR classes left out (default vegetation 3-5, noise 7, water 9); '' keeps all")
    ap.add_argument("--no-corner-snap", dest="corner_snap", action="store_false",
                    help="leave the footprint's corners where Overture and the LiDAR put them; "
                         "by default each face's end is read from the frames and the corner snapped to it")
    ap.add_argument("--no-levels", dest="levels", action="store_false",
                    help="treat the footprint as one body even when the roof returns show a tower on a podium")
    ap.add_argument("--neighbours", type=Path, default=FOOTPRINTS,
                    help="the area's footprints with LiDAR heights: the neighbours, as solid prisms")
    ap.add_argument("--neighbour-levels", default=str(ROOT / "outputs" / "downtown" / "buildings"),
                    help="where the neighbours' own levels.json are (a tower on its podium); 'none' to use "
                         "their outlines at their LiDAR heights only (the cloud building step)")
    ap.add_argument("--no-prism-occluders", action="store_true",
                    help="occlude with the LiDAR returns alone (the neighbours' facades mostly missing)")
    ap.add_argument("--occluder-radius-m", type=float, default=D.occluder_radius_m,
                    help="half-width of the LiDAR extent read purely to occlude "
                         "with. Must reach past the AOI: the neighbours that "
                         "block a wall are by definition outside the box drawn "
                         "around the target")
    ap.add_argument("--prism-min-ratio", type=float, default=D.prism_min_ratio,
                    help="minimum fraction of nominal perimeter verified before inferring prism completion")
    ap.add_argument("--lidar-outline", action="store_true",
                    help="use the building's footprint_lidar.json (its outline from the LiDAR roof, refined in the "
                         "straight-down frames: scripts/lidar_outline.py) instead of Overture's footprint, when it has one")
    ap.add_argument("--match-walls", action="store_true",
                    help="give an unknown wall band the depth where the frames that see it agree (kyfromabove_3d.wall_match), "
                         "when they agree well enough; such a band is 'matched', at the matched depth less the building's "
                         "lean (from its measured bands, 0 with fewer than 3). Off by default; match_report.json says what it did")
    ap.add_argument("--profile-band-m", type=float, default=D.profile_band_m,
                    help="height of the bands a profiled face is measured in")
    ap.add_argument("--extrude-max-rms-m", type=float, default=D.extrude_max_rms_m,
                    help="maximum RMS distance of tile centroids to fitted face plane")
    ap.add_argument("--extrude-max-normal-deg", type=float, default=D.extrude_max_normal_deg,
                    help="maximum angular deviation of tile normals from face normal")
    ap.add_argument("--neighbour-spill-m", type=float, default=D.neighbour_spill_m,
                    help="a return near the edge this far over the building's inside top is a taller neighbour's wall "
                         "and is left out; 0 keeps them (before)")
    ap.add_argument("--band-local-rise", type=float, default=D.band_local_rise,
                    help="an observed band's returns climb this share of it in the median 2 m strip along the face; "
                         "0 tests the whole face's span only (before)")
    ap.add_argument("--out", type=Path, default=None)
    return ap


def settings_from_args(args):
    """The colourise's settings as the command line gives them (kyfromabove_3d.settings)."""
    from kyfromabove_3d.settings import ColouriseSettings
    return ColouriseSettings(
        spacing_m=args.spacing_m, min_face_m=args.min_face_m, look_in_share=args.look_in_share,
        max_offset_m=args.max_offset_m, tile_m=args.tile_m, max_plane_rms_m=args.max_plane_rms_m,
        min_facing_cos=args.min_facing_cos, occlusion_slack_m=args.occlusion_slack_m,
        drop_classes=tuple(args.drop_classes), corner_snap=args.corner_snap, find_levels=args.levels,
        prism_occluders=not args.no_prism_occluders,
        occluder_radius_m=args.occluder_radius_m,
        prism_min_ratio=args.prism_min_ratio, outline="lidar" if args.lidar_outline else "overture",
        match_walls=args.match_walls, profile_band_m=args.profile_band_m,
        extrude_max_rms_m=args.extrude_max_rms_m, extrude_max_normal_deg=args.extrude_max_normal_deg,
        neighbour_spill_m=args.neighbour_spill_m, band_local_rise=args.band_local_rise)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    import dataclasses
    from kyfromabove_3d.settings import ColouriseSettings, changed_from_defaults, resolve
    cli = settings_from_args(args)
    # the defaults, then the building's class's, then what the command line changed
    CS = resolve(args.building, colourise=changed_from_defaults(
        {f.name: getattr(cli, f.name) for f in dataclasses.fields(cli)}, ColouriseSettings)).colourise


    started = time.time()
    cams = load_cameras(args.building)
    print(f"{len(cams)} cameras with crops: "
          f"{', '.join(sorted({c['head'] for c in cams}))}")
    if not cams:
        print("[FAIL] no cameras with images on disk")
        return 2

    try:
        F = footprint_and_levels(args.building, CS, args.lidar_outline)
    except ColouriseFailure as exc:
        return exc.code
    ox, oy, oz, xyz = F.ox, F.oy, F.oz, F.xyz
    ring_m, entry, measured_edges, lidar_outline = F.ring_m, F.entry, F.measured_edges, F.lidar_outline
    on_building, b_xyz, profile_xyz, ground_z = F.on_building, F.b_xyz, F.profile_xyz, F.ground_z
    levels, lidar_tower_levels, overhangs, roof_z = F.levels, F.lidar_tower_levels, F.overhangs, F.roof_z
    from kyfromabove_3d.features import LEVEL_FACE_STRIDE
    from kyfromabove_3d.footprints import footprint_wgs84_to_local
    zones = split_surface_zones(b_xyz)
    if zones is None:
        print("[FAIL] cannot split zones")
        return 2
    oh_path = args.building / "overhangs.json"
    if overhangs:
        oh_path.write_text(json.dumps(overhangs, indent=1), encoding="utf-8")
    elif oh_path.exists():
        oh_path.unlink()
    height_m = roof_z - ground_z
    print(f"zones: {zones.ground.sum():,} ground / {zones.wall.sum():,} wall / "
          f"{zones.roof.sum():,} roof;  height {height_m:.0f} m")

    # Existing points: colour them where they are. Normals only decide which
    # cameras may see a point, so a coarse split is enough -- roof and ground
    # face up, wall points face outward from the building's axis.
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(xyz), 1))
    centre = np.array([ring_m[:, 0].mean(), ring_m[:, 1].mean()])
    wall_all = np.zeros(len(xyz), dtype=bool)
    wall_all[np.flatnonzero(on_building)[zones.wall]] = True
    out = xyz[wall_all][:, :2] - centre
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    normals[wall_all] = np.column_stack([
        np.divide(out, np.maximum(nrm, 1e-9)), np.zeros(len(out))])

    if args.scene_cloud:
        wider = scene_cloud_from_cache(args.scene_cloud, (ox, oy, oz), CS.occluder_radius_m)
    else:
        wider = scene_cloud((ox, oy, oz), CS.occluder_radius_m)
    if wider is None:
        occ = xyz
        print("[warn] no wider tile; occluding with the AOI alone, which "
              "cannot see the neighbours that block a wall")
    else:
        occ = np.column_stack([(wider[:, 0] - ox) * METERS_PER_US_SURVEY_FOOT,
                               (wider[:, 1] - oy) * METERS_PER_US_SURVEY_FOOT,
                               (wider[:, 2] - oz) * METERS_PER_US_SURVEY_FOOT])
    prisms = [] if (not CS.prism_occluders) else neighbour_prisms(
        args.building, args.neighbours, CS.occluder_radius_m, levels_dir(args.neighbour_levels))
    occlusion_maps(args.building, cams, occ, prisms)
    print(f"depth maps from {len(occ):,} returns within "
          f"{CS.occluder_radius_m:.0f} m ({len(xyz):,} in the AOI), so a "
          f"wall is not coloured through whatever stands in front of it")
    if prisms:
        print(f"  and {len(prisms)} neighbour prism(s): a tower's facade, which the LiDAR "
              f"barely touches, occludes as a solid")
    rgb, seen, _, from_photo_orig = sample_colour(xyz, normals, cams, CS.min_facing_cos,
                                                 CS.occlusion_slack_m)
    got = seen > 0
    print(f"\ncoloured {got.sum():,} of {len(xyz):,} existing points "
          f"({got.mean():.0%}); median {int(np.median(seen[got])) if got.any() else 0} "
          f"views each  ({time.time() - started:.0f}s)")
    on_idx = np.flatnonzero(on_building)
    for name, m in (("ground", zones.ground), ("wall", zones.wall),
                    ("roof", zones.roof)):
        if m.sum():
            print(f"    {name:<7} {got[on_idx[m]].mean():5.0%} coloured "
                  f"({m.sum():,} points)")

    # The curved walls: runs of short edges that fit a circle. Each is one
    # face, profiled in its unrolled frame; its edges are not fitted as planes.
    from kyfromabove_3d.features import ArcWall, arc_faces
    arc_edges = set()
    for a in arc_faces(ring_m, ground_z, ground_z + max(float(height_m or 0.0), 1.0)):
        i = a.first_edge
        while True:
            arc_edges.add(i)
            if i == a.last_edge:
                break
            i = (i + 1) % (len(ring_m) - 1)
    new_xyz, new_n, new_face, faces, stats, tiles_by_face = resample_walls(
        b_xyz, zones.wall, ring_m, height_m, CS.spacing_m,
        CS.min_face_m, CS.max_offset_m, tile_m=CS.tile_m,
        max_rms_m=CS.max_plane_rms_m, skip_faces=arc_edges)
    print(f"\n{len(faces)} wall faces, {sum(f.captured for f in faces)} reached")
    for i, length, n, kept, skipped in stats:
        print(f"    face {i}: {length:5.1f} m, {n:>6,} returns -> "
            f"{kept} tile(s) fitted, {skipped} too far from a plane")
    print(f"  resampled {len(new_xyz):,} new wall tile points at {CS.spacing_m} m")

    # The bodies the footprint holds. Overture draws the ground floor's
    # outline; a tower on a podium has a second outline the LiDAR knows.
    # Each level gets its own faces (level k's numbered from k * LEVEL_FACE_STRIDE) over
    # its own height range, and everything below works per face as before.
    nominal_planes = []
    for lv in levels:
        # arcs only on the footprint's own ring: a traced tower outline has
        # rounded corners that are not curved walls
        edges = wall_planes(lv["ring"], lv["z_lo"], lv["z_hi"], min_length_m=CS.min_face_m, arcs=(lv["level"] == 0))
        assert len(edges) < LEVEL_FACE_STRIDE, f"level {lv['level']} has {len(edges)} edges, exceeds stride {LEVEL_FACE_STRIDE}"
        for pl in edges:
            pl.face += lv["level"] * LEVEL_FACE_STRIDE
            nominal_planes.append(pl)
    is_arc = {pl.face: isinstance(pl, ArcWall) for pl in nominal_planes}
    for pl in nominal_planes:
        if is_arc[pl.face]:
            print(f"  face {pl.face}: a curved wall, edges {pl.first_edge}-{pl.last_edge}, radius {pl.radius:.1f} m, "
                  f"{np.degrees(abs(pl.turn)):.0f} deg, {pl.length_m:.0f} m along the arc, "
                  f"{'convex' if pl.convex else 'concave'}, vertices within {pl.rms_m:.2f} m of the circle")
    if len(levels) > 1:
        from shapely.geometry import Polygon as _Poly
        print("  levels: " + "; ".join(
            f"level {lv['level']}: {_Poly(lv['ring']).area:,.0f} m2, {lv['z_lo']:.0f}-{lv['z_hi']:.0f} m, "
            f"{len(lv['ring']) - 1} corners" for lv in levels))
    (args.building / "levels.json").write_text(json.dumps(
        [{"level": lv["level"], "ring": np.asarray(lv["ring"]).round(3).tolist(),
          "z_lo": round(float(lv["z_lo"]), 3), "z_hi": round(float(lv["z_hi"]), 3)} for lv in levels],
        indent=1), encoding="utf-8")
    level_of = {pl.face: pl.face // LEVEL_FACE_STRIDE for pl in nominal_planes}
    ring_of_level = {lv["level"]: np.asarray(lv["ring"]) for lv in levels}
    verified_planes: dict[int, VerifiedWallPlane] = {}
    new_surface = np.ones(len(new_xyz), dtype=np.uint8) if len(new_xyz) else np.empty(0, dtype=np.uint8)

    all_ext_xyz, all_ext_n, all_ext_face = [], [], []
    for lv in levels:
        lv_planes = [pl for pl in nominal_planes if level_of[pl.face] == lv["level"] and not is_arc[pl.face]]
        if not lv_planes:
            continue
        ext_xyz, ext_n, ext_face, lv_verified = extrude_verified_faces(
            nominal_planes=lv_planes,
            tiles_by_face=tiles_by_face,
            spacing_m=CS.spacing_m,
            tile_m=CS.tile_m,
            max_offset_rms_m=CS.extrude_max_rms_m,
            max_normal_dev_deg=CS.extrude_max_normal_deg,
            roof_z=float(lv["z_hi"]),
            prism_completion=True,
            min_prism_ratio=CS.prism_min_ratio,
        )
        verified_planes.update(lv_verified)
        if len(ext_xyz):
            all_ext_xyz.append(ext_xyz)
            all_ext_n.append(ext_n)
            all_ext_face.append(ext_face)
    if all_ext_xyz:
        ext_xyz = np.vstack(all_ext_xyz)
        ext_n = np.vstack(all_ext_n)
        ext_face = np.concatenate(all_ext_face)
        ext_surface = np.full(len(ext_xyz), 3, dtype=np.uint8)
        n_dir = sum(1 for v in verified_planes.values() if not v.inferred)
        n_inf = sum(1 for v in verified_planes.values() if v.inferred)
        inf_msg = f" ({n_dir} verified + {n_inf} inferred prism)" if n_inf else ""
        print(f"  extruded {len(ext_xyz):,} points across {len(verified_planes)} face(s){inf_msg}")
        new_xyz = np.vstack([new_xyz, ext_xyz]) if len(new_xyz) else ext_xyz
        new_n = np.vstack([new_n, ext_n]) if len(new_n) else ext_n
        new_face = np.concatenate([new_face, ext_face]) if len(new_face) else ext_face
        new_surface = np.concatenate([new_surface, ext_surface]) if len(new_surface) else ext_surface

    # The footprint's corners, before any face is profiled or textured: first
    # where two LiDAR-verified planes meet (their intersection), then where
    # the frames show a face ending (kyfromabove_3d.corners). Overture's
    # corners are a metre off as a matter of course; every band's extent and
    # every texture's placement inherit whatever the corners are.
    level0_straight = [p for p in nominal_planes if level_of[p.face] == 0 and not is_arc[p.face]]
    if verified_planes and len(levels) > 0 and level0_straight:
        candidate_planes = [verified_planes.get(p.face, p) for p in level0_straight]
        rect_ring, rect_planes = rectify_footprint_corners(levels[0]["ring"], candidate_planes)
        levels[0]["ring"] = rect_ring.round(3).tolist()
        for rp in rect_planes:
            if rp.face in verified_planes:
                verified_planes[rp.face] = rp
    if CS.corner_snap and len(levels) > 0 and level0_straight and cams:
        from kyfromabove_3d.corners import observe_end, snap_corners
        from kyfromabove_3d.features import _ccw_ring
        from kyfromabove_3d.footprints import footprint_wgs84_to_local
        # the ring as the faces index it (counter-clockwise, open)
        ring0 = _ccw_ring(np.asarray(levels[0]["ring"], dtype=np.float64))
        n_ring = len(ring0)
        # every straight edge of half a metre or more takes part (a corner
        # needs a line on both sides to be re-derived), the verified planes
        # standing in for their faces
        snap_planes = [verified_planes.get(p.face, p) for p in wall_planes(ring0, ground_z, roof_z, min_length_m=0.5, arcs=True)
                       if not isinstance(p, ArcWall)]
        verified_faces = {f for f in verified_planes if f < LEVEL_FACE_STRIDE}
        # corners with nothing to see: where either edge runs along a
        # neighbour (a party wall), or on a curved wall
        skip = {}
        from shapely.geometry import LineString as _LS
        nb_zone = None
        for nb in entry.get("neighbours", []):
            g = footprint_wgs84_to_local(nb["wkt"], ox, oy).buffer(0.5)
            nb_zone = g if nb_zone is None else nb_zone.union(g)
        if nb_zone is not None:
            for k in range(n_ring):
                e = _LS([ring0[k], ring0[(k + 1) % n_ring]])
                if e.length > 0 and e.intersection(nb_zone).length / e.length >= 0.7:
                    skip[k] = "skipped-party"; skip[(k + 1) % n_ring] = "skipped-party"
        for pl in nominal_planes:
            if is_arc[pl.face]:
                skip[pl.first_edge] = "skipped-arc"; skip[(pl.last_edge + 1) % n_ring] = "skipped-arc"
        aerial = [c for c in cams if not c.get("is_photo")]

        def make_sampler(cam):
            return lambda pts, nrm: sample_colour(pts, nrm, [cam], CS.min_facing_cos, CS.occlusion_slack_m)[:2]

        z_lo_s, z_hi_s = ground_z + 1.0, min(ground_z + 13.0, roof_z - 1.0)
        observations, notes = {}, {}
        if z_hi_s - z_lo_s >= 2.0:
            for pl in snap_planes:
                for which in ("start", "end"):
                    corner_k = pl.face if which == "start" else (pl.face + 1) % n_ring
                    if corner_k in skip:
                        continue
                    s_obs, votes, why = observe_end(pl, which, aerial, make_sampler, z_lo_s, z_hi_s)
                    observations[(pl.face, which)] = s_obs
                    notes[f"{pl.face}:{which}"] = {"along_m": None if s_obs is None else round(s_obs, 2),
                                                   "n_votes": len(votes), "why": why}
        new_ring, snapped, records = snap_corners(ring0, snap_planes, verified_faces, observations, skip=skip)
        n_snapped = sum(1 for r in records if r.status == "snapped")
        moves = [float(np.linalg.norm(np.array(r.after) - np.array(r.before))) for r in records if r.status == "snapped"]
        gaps_b = [r.gap_before_m for r in records if r.gap_before_m is not None]
        gaps_a = [r.gap_after_m for r in records if r.gap_after_m is not None]
        print(f"  corners: {n_snapped} of {n_ring} snapped to the frames' edges"
              + (f", moved a median of {np.median(moves):.2f} m (max {max(moves):.2f})" if moves else "")
              + (f"; corner gap {np.median(gaps_b):.2f} m -> {np.median(gaps_a):.2f} m (median)" if gaps_b and gaps_a else "")
              + f"; {sum(1 for r in records if r.status == 'no-observation')} unseen, "
              f"{sum(1 for r in records if r.status.startswith('skipped'))} skipped, {sum(1 for r in records if r.status == 'refused')} refused")
        levels[0]["ring"] = new_ring.round(3).tolist()
        by_face_snapped = {p.face: p for p in snapped}
        for pl in level0_straight:
            q = by_face_snapped.get(pl.face)
            if q is None:
                continue
            pl.p0, pl.p1, pl.normal = q.p0.copy(), q.p1.copy(), q.normal.copy()
            if pl.face in verified_planes:
                vp = verified_planes[pl.face]
                vp.p0, vp.p1, vp.normal = q.p0.copy(), q.p1.copy(), q.normal.copy()
        ring_m = new_ring.copy()
        (args.building / "corner_snap.json").write_text(json.dumps({
            "corners": [{"corner": r.corner, "status": r.status, "before": [round(v, 3) for v in r.before],
                         "after": [round(v, 3) for v in r.after],
                         "gap_before_m": None if r.gap_before_m is None else round(r.gap_before_m, 3),
                         "gap_after_m": None if r.gap_after_m is None else round(r.gap_after_m, 3),
                         "note": r.note} for r in records],
            "observations": notes, "strip_z_m": [round(z_lo_s, 2), round(z_hi_s, 2)], "n_frames": len(aerial)}, indent=1), encoding="utf-8")
    if len(levels) > 0:
        (args.building / "levels.json").write_text(json.dumps(
            [{"level": lv["level"], "ring": np.asarray(lv["ring"]).round(3).tolist(),
              "z_lo": round(float(lv["z_lo"]), 3), "z_hi": round(float(lv["z_hi"]), 3)} for lv in levels],
            indent=1), encoding="utf-8")

    # Faces no single plane describes: a profile of bands from the returns.
    profiles: dict[int, list] = {}
    mirrored: dict[int, dict] = {}
    from kyfromabove_3d.features import clip_profiles_by_neighbours, extrude_profile, face_profile, profile_keeps
    # Pass 1: every face's profile. Each face looks inside its level at most
    # --look-in-share of the level's narrowest width (and at most 8 m): with
    # 8 m regardless, on a tier or a row building 7-8 m across a face took
    # the far wall's returns for its own wall "7 m in" (e5214ab7); 200 of
    # the downtown's 350 buildings are under 16 m across.
    from shapely import maximum_inscribed_circle
    from shapely.geometry import Polygon as _LvPoly
    level_width = {lv["level"]: 2.0 * float(maximum_inscribed_circle(_LvPoly(np.asarray(lv["ring"], dtype=float)[:, :2]).buffer(0)).length)
                   for lv in levels}

    def look_in_window(plane):
        if CS.look_in_share <= 0:
            return None                                   # the fixed 8 m (before)
        w = level_width.get(level_of[plane.face])
        inner = 8.0 if w is None else min(8.0, max(2.0, CS.look_in_share * w))
        return (-inner, 3.0)
    for plane in nominal_planes:
        # a level's profile reads only the returns of its own height
        # range (a little over), so the podium's bands do not see the
        # tower's wall and the tower's do not see the podium's roof
        lv_xyz = profile_xyz[(profile_xyz[:, 2] >= plane.z_lo - 2.0) & (profile_xyz[:, 2] <= plane.z_hi + 2.0)]
        if is_arc[plane.face]:
            # a curved wall is profiled unrolled: the returns within 12 m
            # of the arc's radius brought into its flat frame
            near = np.abs(np.hypot(*(lv_xyz[:, :2] - plane.centre).T) - plane.radius) <= 12.0
            lv_xyz = np.column_stack([plane.unroll(lv_xyz[near, :2]), lv_xyz[near, 2]])
        bands = face_profile(plane, lv_xyz, band_m=CS.profile_band_m, offset_window_m=look_in_window(plane),
                             local_rise=CS.band_local_rise)
        # A tower can stand well back from the edge its podium was traced
        # along: Humana's north wall is 25 m inside the Main Street edge,
        # seen by a thin column of returns the 8 m window never reaches.
        # When the standard window finds no wall above the podium, look
        # deeper -- up to 60% of the way to the far side -- with a lower
        # floor, and keep the result only if it finds a set-back wall.
        # (Only when the building is one level: with the tower split
        # out, its walls are its own faces.)
        top_seen = max((b.z_hi for b in bands if b.status in ("observed", "bridged", "extended")), default=0.0)
        # ... and only where the roof just inside the wall is well below the
        # top, as at Humana's podium edge: where the roof reaches the edge
        # the wall is there, unseen, and looking 60% across found the far wall
        # (the Galt House's face 7, drawn 37 m in)
        def roof_inside_low(pl=plane):
            a_, o_ = pl.local(lv_xyz[:, :2])
            m_ = (a_ >= 0.2 * pl.length_m) & (a_ <= 0.8 * pl.length_m) & (o_ <= -0.25) & (o_ >= -2.0)
            return m_.sum() < 10 or float(np.percentile(lv_xyz[m_, 2], 90)) < ground_z + 0.6 * (roof_z - ground_z)
        if len(levels) == 1 and top_seen < 0.4 * (roof_z - ground_z) and not is_arc[plane.face] and roof_inside_low():
            depth = float(max(-((ring_m - plane.p0) @ plane.normal).min(), 8.0))
            deep = face_profile(plane, lv_xyz, band_m=CS.profile_band_m,
                                offset_window_m=(-0.6 * depth, 3.0), min_returns=20,
                                local_rise=CS.band_local_rise)
            found = [b for b in deep if b.status in ("observed", "bridged", "extended") and b.offset_m is not None and b.offset_m < -8.0]
            if found:
                print(f"    face {plane.face}: a wall set {-np.median([b.offset_m for b in found]):.0f} m back from the footprint edge, "
                      f"{found[0].z_lo:.0f}-{found[-1].z_hi:.0f} m up, found by looking {0.6 * depth:.0f} m in")
                # The standard window keeps its say on the podium; the
                # deep one speaks only for the bands above what it saw.
                bands = [b if b.z_lo < top_seen else d for b, d in zip(bands, deep)]
        profiles[plane.face] = bands

    # Pass 1b: an unknown band takes its depth from the frames, where two
    # that see it from directions apart agree on one (wall_match). Before
    # the mirror and the bridges, which read measured bands only.
    if CS.match_walls:
        from kyfromabove_3d.wall_match import MAX_ANGLE_DEG, match_profiles
        facing = float(np.cos(np.radians(MAX_ANGLE_DEG)))
        report = match_profiles(nominal_planes, profiles, cams,
                                lambda xyz, nrm, c: project_visible(xyz, nrm, c, facing)[0],
                                skip_faces=set(verified_planes))
        (args.building / "match_report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")

    # Pass 2a: a hidden face takes its shape from the face opposite, when
    # the two have been seen to share one -- buildings are mostly
    # symmetric across their middle, and the face in a neighbour's
    # shadow is best read from its twin.
    from kyfromabove_3d.features import mirror_hidden_bands
    # the faces under a roof edge the LiDAR measured -- with the LiDAR outline,
    # or, where the class sets upper_levels: roof_edge, every face of an upper
    # level, whose outline is traced from the LiDAR's roof (building_levels):
    # an unseen one takes the roof edge's word, not its twin's (pass 2a'')
    upper_measured = CS.upper_levels == "roof_edge"
    measured_levels = set(lidar_tower_levels) | ({lv["level"] for lv in levels if lv["level"] > 0} if upper_measured else set())
    under = (under_measured_edge(nominal_planes, is_arc, level_of, measured_levels, measured_edges if lidar_outline else [])
             if (lidar_outline or upper_measured) else set())
    mirrored = {}
    for lv in levels:
        lv_planes = [pl for pl in nominal_planes if level_of[pl.face] == lv["level"] and not is_arc[pl.face]]
        lv_prof = {pl.face: profiles[pl.face] for pl in lv_planes}
        # (a verified plane is extruded whole and is not filled from the roof edge: its twin still speaks)
        mirrored.update(mirror_hidden_bands(lv_planes, lv_prof, measured_faces=(under - set(verified_planes)) or None))
    # which faces are symmetric twins: the completed product's twin fill reads it
    (args.building / "mirror_report.json").write_text(
        json.dumps({str(f): rec for f, rec in mirrored.items()}, indent=1, default=float), encoding="utf-8")
    for f, rec in sorted(mirrored.items()):
        if rec["symmetric"] and rec["mirrored"]:
            print(f"    face {f}: {rec['mirrored']} unseen band(s) taken from face {rec['from_face']} opposite "
                  + (f"(shapes agree within {rec['shape_diff_m']} m on {rec['n_joint']} bands)" if rec.get("anchored") else "(this face was never seen; the opposite face's offsets as they are)"))
        elif rec.get("skipped"):
            print(f"    face {f}: nothing taken from face {rec['from_face']} opposite ({rec['skipped']})")
        elif not rec["symmetric"]:
            print(f"    face {f}: face {rec['from_face']} opposite has a different shape "
                  f"({rec['shape_diff_m']} m apart on {rec['n_joint']} bands); nothing mirrored")

    # Pass 2a'': a wall stands under a roof edge the LiDAR measured -- a
    # tower edge from the LiDAR's own outline of the tower, or an edge of
    # the outline whose roof end it found -- unless there is evidence of a
    # gap. Not on Overture's edge, which is not measured: only with the LiDAR
    # outline, or on upper levels where the class says so.
    if under:
        from kyfromabove_3d.features import fill_under_roof_edge
        n_faces = n_bands = 0
        for f in sorted(under):
            if f in verified_planes or f not in profiles:
                continue
            pl_f = next(pl for pl in nominal_planes if pl.face == f)
            if (k := fill_under_roof_edge(profiles[f], pl_f.length_m)):
                n_faces += 1
                n_bands += k
                print(f"    face {f}: {k} band(s) under the measured roof edge drawn as wall (none seen as a gap)")
        print(f"  {len(under)} face(s) under a measured roof edge; {n_bands} band(s) on {n_faces} of them drawn as wall")

    # Pass 2a': a wall stands on the wall below it -- where a neighbour
    # hid part of a band from the scanner, it runs as far as the band above.
    from kyfromabove_3d.features import carry_extents_down, crown_bands
    for f, bands in sorted(profiles.items()):
        pl_f = next((pl for pl in nominal_planes if pl.face == f), None)
        top_z = float(pl_f.z_hi) if pl_f is not None else None

        def crown_top(b, pl=pl_f):
            """How high the returns on the crown's front surface reach."""
            if pl is None or not len(profile_xyz):
                return None
            along, off = pl.local(profile_xyz[:, :2])
            m = ((along >= 0) & (along <= pl.length_m) & (np.abs(off - b.offset_m) < 0.75)
                 & (profile_xyz[:, 2] >= b.z_lo) & (profile_xyz[:, 2] <= b.z_hi + 1.0))
            return float(np.percentile(profile_xyz[m, 2], 98)) if m.sum() >= 10 else None

        if not is_arc[f] and (n_crown := crown_bands(bands, 0.0 if f in verified_planes else None, top_z=top_z,
                                                     top_of=crown_top)):
            print(f"    face {f}: {n_crown} open band(s) at the top drawn as the wall's crown, at their front")
        if not is_arc[f] and (n_carried := carry_extents_down(bands)):
            print(f"    face {f}: {n_carried} band(s) widened to the wall above them (returns stopped short)")

    # Pass 2b: faces meet at corners. Where a neighbour's wall stands back
    # from the footprint at some height, this face's wall stops there.
    n_clipped = 0
    for lv in levels:
        lv_planes = [pl for pl in nominal_planes if level_of[pl.face] == lv["level"] and not is_arc[pl.face]]
        n_clipped += clip_profiles_by_neighbours(lv_planes, {pl.face: profiles[pl.face] for pl in lv_planes})
    if n_clipped:
        print(f"    {n_clipped} band(s) clipped where a neighbouring wall stands back from the corner")

    # Pass 3: trim the verified planes, extrude the rest.
    prof_xyz, prof_n, prof_face = [], [], []
    for plane in nominal_planes:
        bands = profiles[plane.face]
        # A verified plane is extruded over the whole face; the profile
        # trims it to where the returns say the wall is, band by band --
        # not into an open loggia, not past the corner where the wall
        # steps in. Humana's north face is a plane, but its photo was
        # painting sky and the tower behind onto the part that is not wall.
        if plane.face in verified_planes:
            on_face = (new_face == plane.face) & (new_surface == 3)
            if on_face.any():
                keep_f = profile_keeps(plane, bands, new_xyz[on_face])
                drop = np.zeros(len(new_xyz), dtype=bool)
                drop[np.flatnonzero(on_face)[~keep_f]] = True
                n_obs = sum(b.status in ("observed", "bridged") for b in bands)
                n_open = sum(b.status == "open" for b in bands)
                print(f"    face {plane.face}: verified plane trimmed to the profile ({n_obs} band(s) with a wall, "
                      f"{n_open} open): {drop.sum():,} of {on_face.sum():,} extruded points removed")
                if drop.any():
                    new_xyz, new_n, new_face, new_surface = new_xyz[~drop], new_n[~drop], new_face[~drop], new_surface[~drop]
            continue
        n_obs = sum(b.status == "observed" for b in bands)
        n_br = sum(b.status == "bridged" for b in bands)
        n_sl = sum(b.status == "sloped" for b in bands)
        n_ex = sum(b.status == "extended" for b in bands)
        n_mi = sum(b.status == "mirrored" for b in bands)
        n_ma = sum(b.status == "matched" for b in bands)
        n_ed = sum(b.status == "edge" for b in bands)
        if n_obs == 0 and n_mi == 0 and n_ed == 0:                       # matched bands never stand alone (see build_mesh)
            print(f"    face {plane.face}: no band observed; nothing extruded")
            continue
        pts = extrude_profile(plane, bands, CS.spacing_m)
        if is_arc[plane.face] and len(pts):
            # back onto the arc, each point's normal the outward radial there
            pts, pts_n = plane.to_world_3d(pts, np.repeat(np.array([[0.0, -1.0, 0.0]]), len(pts), axis=0))
        else:
            pts_n = np.repeat(np.array([[plane.normal[0], plane.normal[1], 0.0]]), len(pts), axis=0)
        offs = sorted({round(b.offset_m, 1) for b in bands if b.offset_m is not None})
        zs = [b for b in bands if b.status in ("observed", "bridged", "extended", "mirrored", "matched", "edge")]
        print(f"    face {plane.face}: {n_obs} band(s) observed, {n_br} bridged, {n_ex} extended through thin bands, {n_mi} mirrored from the face opposite, "
              + (f"{n_ma} matched in the frames, " if n_ma else "") + (f"{n_ed} under the measured roof edge, " if n_ed else "") + f"{n_sl} sloped (not extruded); wall {zs[0].z_lo:.0f}-{zs[-1].z_hi:.0f} m up, "
              f"offsets {offs[0]:+.1f}..{offs[-1]:+.1f} m from the footprint; {len(pts):,} points")
        if len(pts):
            prof_xyz.append(pts)
            prof_n.append(pts_n)
            prof_face.append(np.full(len(pts), plane.face))
    if prof_xyz:
        px, pn, pf = np.vstack(prof_xyz), np.vstack(prof_n), np.concatenate(prof_face)
        print(f"  extruded {len(px):,} points on {len(prof_xyz)} profiled face(s)")
        new_xyz = np.vstack([new_xyz, px]) if len(new_xyz) else px
        new_n = np.vstack([new_n, pn]) if len(new_n) else pn
        new_face = np.concatenate([new_face, pf]) if len(new_face) else pf
        new_surface = np.concatenate([new_surface, np.full(len(px), 4, dtype=np.uint8)]) if len(new_surface) else np.full(len(px), 4, dtype=np.uint8)
    (args.building / "wall_profiles.json").write_text(json.dumps(
        {str(k): [b.to_dict() for b in v] for k, v in profiles.items()}, indent=1), encoding="utf-8")

    planes_to_save = [verified_planes.get(p.face, p) for p in nominal_planes]
    planes_path = args.building / "wall_planes.json"
    save_wall_planes(planes_path, planes_to_save)
    n_dir = sum(1 for v in verified_planes.values() if not v.inferred)
    n_inf = sum(1 for v in verified_planes.values() if v.inferred)
    inf_msg = f" ({n_dir} verified, {n_inf} inferred prism)" if n_inf else f" ({n_dir} verified)"
    print(f"  saved {len(planes_to_save)} wall planes{inf_msg} -> {planes_path}")

    from_photo_wall = np.zeros(len(new_xyz), dtype=bool)
    if len(new_xyz):
        new_rgb, new_seen, _, from_photo_wall = sample_colour(new_xyz, new_n, cams,
                                                             CS.min_facing_cos,
                                                             CS.occlusion_slack_m)
        # (a face read from its twin for shape takes no colour from it: the
        # measured product colours a texel only where a frame saw it, and the
        # completed product makes the twin copy on its textures, flagged)
        colour_from_twin = np.zeros(len(new_xyz), dtype=bool)
        keep = new_seen > 0
        print(f"  coloured {keep.sum():,} of them ({keep.mean():.0%}) "
              f"({time.time() - started:.0f}s)")

        # Measure facade coverage before discarding uncoloured points
        from kyfromabove_3d.coverage import facade_coverage
        coverages = facade_coverage(
            resampled_xyz=new_xyz,
            coloured_mask=keep,
            from_photo_mask=from_photo_wall,
            lidar_wall_xyz=b_xyz[zones.wall],
            ring_m=ring_m,
            height_m=height_m,
            cell_m=3.0,
            min_face_length_m=CS.min_face_m,
            max_offset_m=CS.max_offset_m,
            surface_mask=new_surface,
            verified_planes=verified_planes,
        )
        cov_data = {
            "building": repo_relative(args.building),
            "height_m": float(height_m),
            "faces": [c.to_dict() for c in coverages],
            "total_fillable_cells": sum(c.n_fillable_cells for c in coverages),
            "total_unfillable_cells": sum(c.n_unfillable_cells for c in coverages),
            "total_photo_cells": sum(c.n_photo_cells for c in coverages),
            "total_extruded_cells": sum(c.n_extruded_cells for c in coverages),
            "total_tile_cells": sum(c.n_tile_cells for c in coverages),
        }
        # A run with its own --out (the aerial-only cloud the mesh uses)
        # keeps its coverage beside it; coverage.json stays the picker's.
        cov_path = (args.building / "coverage.json" if args.out is None
                    else args.out.with_name(args.out.stem + "_coverage.json"))
        cov_path.write_text(json.dumps(cov_data, indent=2), encoding="utf-8")
        print(f"  wrote coverage data ({cov_data['total_fillable_cells']} fillable cells across {len(coverages)} faces) -> {cov_path}")

        new_xyz, new_rgb = new_xyz[keep], new_rgb[keep]
        from_photo_wall = from_photo_wall[keep]
        new_surface = new_surface[keep]
        colour_from_twin = colour_from_twin[keep]
    else:
        new_rgb = np.zeros((0, 3), dtype=np.uint8)
        colour_from_twin = np.zeros(0, dtype=bool)

    out_path = args.out or (args.building / "colorized.ply")
    all_xyz = np.vstack([xyz[got], new_xyz])
    all_rgb = np.vstack([rgb[got], new_rgb])
    source_orig = np.where(from_photo_orig[got], 2, 0).astype(np.uint8)
    source_new = np.where(from_photo_wall, 2, 1).astype(np.uint8) if len(new_xyz) else np.empty(0, dtype=np.uint8)
    if len(new_xyz):
        source_new[colour_from_twin] = 3            # colour taken from the face opposite
    source = np.concatenate([source_orig, source_new])
    surface_orig = np.zeros(len(xyz[got]), dtype=np.uint8)
    surface = np.concatenate([surface_orig, new_surface])
    write_ply(out_path, all_xyz, all_rgb, extra=source, extras={"surface": surface})
    n_from_photo = int((source == 2).sum())
    n_extruded = int(((surface == 3) | (surface == 4)).sum())
    n_tile = int((surface == 1).sum())
    print(f"\n{len(all_xyz):,} coloured points "
          f"({got.sum():,} measured + {n_tile:,} tile + {n_extruded:,} extruded; {n_from_photo:,} from photos) -> {out_path}")
    print("  'source' property: 0 = a LiDAR return, 1 = resampled on a fitted wall, 2 = from a photograph, 3 = colour taken from the face opposite")
    print("  'surface' property: 0 = a LiDAR return, 1 = fitted tile, 3 = extruded plane, 4 = extruded profile band")
    return 0


if __name__ == "__main__":
    sys.exit(main())
