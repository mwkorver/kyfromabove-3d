#!/usr/bin/env python3
"""Build a textured glTF mesh of one building from its LiDAR, footprint,
wall profiles and imagery, and wrap it as OGC 3D Tiles.

The point export (``export_3dtiles.py``) draws the building as it was
sampled. This draws it as surfaces, from the same evidence:

* **walls** are the profile bands ``colorize_building.py`` wrote to
  ``wall_profiles.json`` -- each band a vertical rectangle at its own offset
  from the footprint edge, split where a recess or projection was found, with
  a horizontal ledge wherever the band above stands further in or out than
  the band below (a set-back tower on its podium) and a short return wall at
  each recess edge;
* the **roof** is a height field over the footprint, the top of the returns
  in each cell (their 80th percentile, so an overhang keeps its roof), with the near-vertical cells that coincide with a
  modelled wall dropped so the two do not fight;
* **wall textures** are rasterised from an aerial-only coloured cloud
  (``colorized_aerial.laz``: the colourise run with no photograph cameras,
  which this script makes for itself) -- the obliques and, where a face was
  mirrored, the face opposite -- one image per face, 0.15 m per pixel. No
  internet photograph reaches the mesh;
* the **roof texture** is sampled straight from the aerial frames through
  each camera model, the nadir frame winning wherever it sees the roof
  because it is the most face-on, with the LiDAR depth test against the
  building's own returns so a set-back tower does not print onto its podium.

Output is one ``building.glb`` (positions, normals, UVs, one material per
face plus the roof, the GERS id in the node's extras) and a
``tileset.json`` whose root ``transform`` places the local metre frame on the
Earth; the mesh itself stays in the building's local frame, converted to
glTF's Y-up.

Usage::

    python scripts/build_mesh.py --building outputs/downtown/buildings/d2786314-4fb3-4da6-8f81-621873d40c08      # Humana
    # -> <building>/3dtiles_mesh/tileset.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from kyfromabove_3d.features import (                                  # noqa: E402
    WALL_STATUSES, ArcWall, FaceBand, VerifiedWallPlane, load_wall_planes,
    rectify_footprint_corners, ground_field_for_building,
)
from kyfromabove_3d.clouds import find_cloud, read_cloud               # noqa: E402
from kyfromabove_3d.lidar import VEGETATION                           # noqa: E402
from kyfromabove_3d.mesh.walls import (  # noqa: E402,F401
    _orient, _runs, ArcGround, corner_returns, plane_bands, roll_quads, RoofField, wall_geometry,
)
from kyfromabove_3d.mesh.textures import (  # noqa: E402,F401
    _fill_holes, _mult4, face_texture, quad_texture_from_cameras, texel_sources, TextureAtlas,
    tile_geometry,
)
from kyfromabove_3d.mesh.roof import (  # noqa: E402,F401
    outside_modelled_wall, roof_heightfield, roof_mesh, roof_sloped_fraction, roof_texture,
    roof_texture_from_cloud, tidy_roof_field,
)
from kyfromabove_3d.mesh.party import edge_top_z, party_edges, party_steps  # noqa: E402,F401
from kyfromabove_3d.mesh.paint import (  # noqa: E402
    PAINT_FILE, paint_quad, paint_record, paint_section, section_uv, wall_sections,
)
from kyfromabove_3d.mesh.gltf import (  # noqa: E402,F401
    enu_transform, GlbRecorder, GlbWriter, quads_to_arrays, TEXTURE_FORMATS,
)


# ----------------------------------------------------------------------------- main

def load_footprint(args, ox, oy):
    """(polygon in the local frame, name, gers, record) from --case, --gers
    or --footprint; the building's own footprint.json when none is given."""
    from kyfromabove_3d.footprints import resolve_footprint

    poly, entry = resolve_footprint(args.building, ox, oy)
    return poly, entry.get("name") or entry.get("overture_name") or entry.get("gers"), entry.get("gers"), entry


def building_classification(b: Path, xyz, poly, levels, profiles, edges, party, Z, valid, cell):
    """The building's structural label, written to classification.json and
    the tileset's extras. A label only: it does not choose the mesh."""
    from shapely import contains_xy
    from kyfromabove_3d.features import classify_building_structure, split_surface_zones

    on_b = contains_xy(poly, xyz[:, 0], xyz[:, 1])
    zones = split_surface_zones(xyz[on_b]) if on_b.sum() >= 20 else None
    c = classify_building_structure(
        levels, zones, profiles, arc_edges={e.face for e in edges if isinstance(e, ArcWall)},
        sloped_fraction=roof_sloped_fraction(Z, valid, cell), party_faces=set(party))
    (b / "classification.json").write_text(json.dumps(c.to_dict(), indent=1), encoding="utf-8")
    print(f"  structure: {c.archetype} (roof {c.roof_type}, facade {c.facade_type})")
    return c


def parser() -> argparse.ArgumentParser:
    from kyfromabove_3d.settings import MeshSettings
    D = MeshSettings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--building", type=Path, required=True)
    ap.add_argument("--name", default=None, help="display name (default: from the footprint)")
    ap.add_argument("--texel-m", type=float, default=D.texel_m)
    ap.add_argument("--no-roof-tidy", dest="roof_tidy", action="store_false",
                    help="the roof field as measured: no ramps over empty cells, no snapping to flat levels (before)")
    ap.add_argument("--no-wall-cap", dest="wall_cap", action="store_false",
                    help="walls carried up or copied keep their band's top, and seen ones too; only a wall drawn "
                         "under a roof edge stops at the roof (before)")
    ap.add_argument("--roof-cell-m", type=float, default=D.roof_cell_m,
                    help="roof height-field cell; 0 = 0.75 m where the returns are dense (>= 4/m2), else 1.5 m")
    ap.add_argument("--out", type=Path, default=None, help="default <building>/3dtiles_mesh/")
    ap.add_argument("--texture", choices=TEXTURE_FORMATS, default=D.texture,
                    help="texture format in the glb: ktx2 (Basis, GPU-compressed; needs basisu), webp, jpeg, png")
    ap.add_argument("--dump-textures", action="store_true",
                    help="also keep every texture as a lossless PNG in <out>/textures/ (the measured and the "
                         "completed product's): the batched tiles (kyfromabove_3d.publish) repack them")
    ap.add_argument("--refresh", action="store_true", help="remake the aerial-only cloud even if it exists")
    ap.add_argument("--complete", action="store_true",
                    help="also write the completed product (3dtiles_complete/): unseen texels filled from the "
                         "building's own facade, and a tinted copy showing what was invented")
    ap.add_argument("--neural", action="store_true",
                    help="also write the fourth product (3dtiles_neural/): unseen texels filled from the "
                         "best-matching seen patches on any of the building's walls")
    ap.add_argument("--no-prism-occluders", action="store_true",
                    help="occlude with the LiDAR returns alone, in the colourise and the frame sampling here")
    ap.add_argument("--lidar-outline", action="store_true",
                    help="with --refresh: the colourise uses the building's footprint_lidar.json (scripts/lidar_outline.py)")
    ap.add_argument("--match-walls", action="store_true",
                    help="with --refresh: the colourise gives unknown wall bands the depth the frames agree on "
                         "(kyfromabove_3d.wall_match); matched walls are drawn, and tinted cyan in building_matched.glb")
    ap.add_argument("--allow-no-frames", action="store_true",
                    help="build even though the listed frames' crops are missing (no colour from the frames)")
    ap.add_argument("--neighbours", type=Path, default=None,
                    help="the area's footprints with LiDAR heights, for the neighbour prisms (default: downtown's)")
    ap.add_argument("--neighbour-levels", default=None,
                    help="where the neighbours' levels.json are; 'none' for their outlines at LiDAR height only")
    ap.add_argument("--scene-cloud", type=Path, default=None,
                    help="a cached occluder cloud (scripts/run_downtown.py writes one) passed on to the colourise")
    return ap


def settings_from_args(args):
    """The mesh's settings as the command line gives them (kyfromabove_3d.settings)."""
    from kyfromabove_3d.settings import MeshSettings
    return MeshSettings(texel_m=args.texel_m, roof_tidy=args.roof_tidy, wall_cap=args.wall_cap,
                        roof_cell_m=args.roof_cell_m, texture=args.texture)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    import dataclasses
    from kyfromabove_3d.settings import ColouriseSettings, MeshSettings, changed_from_defaults, resolve
    cli = settings_from_args(args)
    # the defaults, then the building's class's, then what the command line changed (the
    # colourise's own options that pass through here among them)
    colourise_cli = {"prism_occluders": not args.no_prism_occluders,
                     "outline": "lidar" if args.lidar_outline else "overture", "match_walls": args.match_walls}
    BS = resolve(args.building, colourise=changed_from_defaults(colourise_cli, ColouriseSettings),
                 mesh=changed_from_defaults({f.name: getattr(cli, f.name) for f in dataclasses.fields(cli)},
                                            MeshSettings))
    M = BS.mesh

    from kyfromabove_3d.stages.colorize_building import load_cameras

    b = args.building
    t_start = time.time()
    out = args.out or (b / "3dtiles_mesh")
    out.mkdir(parents=True, exist_ok=True)
    coords = json.loads((b / "coordinates.json").read_text(encoding="utf-8"))
    ox, oy, _ = coords["origin_source_xyz"]
    poly, name, gers, fp_entry = load_footprint(args, ox, oy)
    name = args.name or name
    lidar = read_cloud(find_cloud(b, "lidar_local"))
    xyz = np.column_stack([lidar["x"], lidar["y"], lidar["z"]]).astype(np.float64)
    # the trees (lidar.label_trees) hide the frames' view but are no part of
    # the building: kept for the occlusion, left out of its shape
    cls_arr = np.asarray(lidar["classification"]) if "classification" in lidar else None
    xyz_all = xyz
    if cls_arr is not None:
        tree = np.isin(cls_arr, VEGETATION)
        xyz, cls_arr = xyz[~tree], cls_arr[~tree]
    ply_path = find_cloud(b, "colorized_aerial")
    if args.refresh or not ply_path.exists():
        # The mesh's colour comes from the state's aerial frames alone: a
        # colourise with no --extra-camera, kept apart from colorized.ply
        # (which may carry admitted photographs for the point view). It also
        # writes the wall planes and profiles the mesh is built from.
        ply_path = b / "colorized_aerial.laz"
        (b / "colorized_aerial.ply").unlink(missing_ok=True)        # an older run's copy
        cmd = ["--building", str(b), "--out", str(ply_path)]
        if args.scene_cloud:
            cmd += ["--scene-cloud", str(args.scene_cloud)]
        if args.neighbours:
            cmd += ["--neighbours", str(args.neighbours)]
        if args.neighbour_levels:
            cmd += ["--neighbour-levels", str(args.neighbour_levels)]
        if args.no_prism_occluders:
            cmd.append("--no-prism-occluders")
        if args.match_walls:
            cmd.append("--match-walls")
        if args.lidar_outline:
            cmd.append("--lidar-outline")
        print(f"colouring from the aerial frames only -> {ply_path.name}")
        t_col = time.time()
        from kyfromabove_3d.pipeline import run_main
        r = run_main("colorize_building", cmd)               # in this process (refactor step 3)
        # a line the building step reads to price each stage
        print(f"[time] colourise {time.time() - t_col:.1f} s")
        # the profile's decisions go into the building's log (which faces got
        # a crown, which bands were widened): a rebuild can then be traced
        for ln in r.stdout.splitlines():
            if ln.startswith("    face ") or ln.startswith("    ") and "band(s)" in ln:
                print("  colourise:" + ln)
        if r.returncode != 0:
            # No cameras, or too few returns to split into ground, wall and
            # roof: the building still gets a mesh, by the box path below.
            tail = [ln for ln in r.stdout.splitlines() if ln.startswith("[FAIL]")]
            print(f"  colourise declined ({tail[-1] if tail else f'exit {r.returncode}'}); box path")
            colourise_failed = True
        else:
            colourise_failed = False
    else:
        colourise_failed = not ply_path.exists()
    if (b / "wall_planes.json").exists() and (b / "wall_profiles.json").exists() and not colourise_failed:
        planes = load_wall_planes(b / "wall_planes.json")
        prof = json.loads((b / "wall_profiles.json").read_text(encoding="utf-8"))
        profiles = {int(k): [FaceBand(**d) for d in v] for k, v in prof.items()}
    else:
        planes, profiles = [], {}
    if ply_path.exists() and not colourise_failed:
        col = read_cloud(ply_path)
        if "source" in col and int((np.asarray(col["source"]) == 2).sum()):
            raise SystemExit(f"[FAIL] {ply_path} carries colour from photographs "
                             f"({int((np.asarray(col['source']) == 2).sum()):,} points); the mesh takes "
                             f"aerial imagery only. Pass an aerial-only cloud or --refresh.")
        cxyz = np.column_stack([col["x"], col["y"], col["z"]]).astype(np.float64)
        crgb = np.column_stack([col["red"], col["green"], col["blue"]]).astype(np.uint8)
    else:
        cxyz, crgb = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    print(f"{name}: {len(planes)} faces, {len(xyz):,} returns, {len(cxyz):,} coloured points from {ply_path.name}")

    from kyfromabove_3d.features import wall_planes as footprint_edges
    from kyfromabove_3d.footprints import footprint_wgs84_to_local

    # What a frame can see, as the colourise decides it: the area's returns
    # (the building's own clip reaches 27 m past it and misses the tower
    # across the street) and every neighbour as a solid prism.
    cams = load_cameras(b) if (b / "cameras_normalized.json").exists() else []
    if cams:
        from kyfromabove_3d.stages.colorize_building import FOOTPRINTS, levels_dir, neighbour_prisms, occluder_cloud, occlusion_maps
        occ = occluder_cloud(b, args.scene_cloud, 250.0, xyz_all) if args.scene_cloud else xyz_all
        kw = {} if args.neighbour_levels is None else {"buildings_dir": levels_dir(args.neighbour_levels)}
        prisms = [] if not BS.colourise.prism_occluders else neighbour_prisms(b, args.neighbours or FOOTPRINTS, **kw)
        occlusion_maps(b, cams, occ, prisms)
    # The frames are listed but their crops are gone (run_downtown deletes them
    # after a mesh): a mesh built now would take no colour from the frames --
    # every prism face and roof falls back to the point cloud or to grey. A
    # mesh-only rebuild did that to 386 downtown buildings (September 2026).
    # run_downtown.py re-cuts the crops before a mesh; stop here otherwise.
    listed = json.loads((b / "cameras_normalized.json").read_text(encoding="utf-8")) \
        if (b / "cameras_normalized.json").exists() else {}
    if listed and not cams and not args.allow_no_frames:
        raise SystemExit(f"[FAIL] {len(listed)} frames are listed but none of their crops is on disk; "
                         "re-cut them (scripts/run_downtown.py run --only <gers> --force-step mesh) "
                         "or pass --allow-no-frames")
    heads = sorted({c["head"] for c in cams})
    ring_m = np.array(poly.exterior.coords)
    height_m = fp_entry.get("height_m") or fp_entry.get("lidar_height_m")
    n_on = int(fp_entry.get("lidar_n") or 0) or int(__import__("shapely").contains_xy(poly, xyz[:, 0], xyz[:, 1]).sum())
    neighbours = [footprint_wgs84_to_local(nb["wkt"], ox, oy) for nb in fp_entry.get("neighbours", [])]
    # a neighbour's roof in this building's frame (z = 0 at this ground)
    nb_tops = [((nb["lidar_ground_z"] + nb["lidar_height_m"]) - fp_entry["ground_z_m"])
               if nb.get("lidar_ground_z") is not None and nb.get("lidar_height_m") and fp_entry.get("ground_z_m") is not None
               else None for nb in fp_entry.get("neighbours", [])]
    ground_z = 0.0 if (b / "footprint.json").exists() else min([p.z_lo for p in planes] or [0.0])

    # The path this building takes. A footprint the LiDAR barely touched
    # (a shed under a tree, a kiosk) gets a box: its Overture/LiDAR height,
    # a flat roof, prism walls. Everything else takes the profile where the
    # returns gave one and a prism face at the roof's edge height where not.
    # A footprint the LiDAR shows no building on (its class says so: Hotel
    # Bourre Bonne was built after the flight, its returns 78% ground) is a box
    # at its recorded height, coloured from the frames and marked not from the
    # LiDAR: the completed product holds it, the measured one leaves it out.
    not_in_lidar = (M.lidar == "absent" and bool(height_m) and poly.area >= M.absent_min_area_m2)
    if not_in_lidar:
        print(f"  not in the LiDAR ({BS.building_class}): a box at the recorded {float(height_m):.1f} m, "
              f"for the completed product only")
    path = "box" if (n_on < 30 or not height_m or colourise_failed or not_in_lidar) else "profile"
    # The bodies: the footprint alone, or the levels the colourise found (a
    # tower on a podium: the footprint up to the podium top and the tower's
    # own outline above it, its faces numbered from 100).
    from kyfromabove_3d.features import LEVEL_FACE_STRIDE
    lv_path = b / "levels.json"
    levels = json.loads(lv_path.read_text(encoding="utf-8")) if lv_path.exists() and path != "box" else None
    if levels:
        roof_top = max(float(lv["z_hi"]) for lv in levels) - ground_z
    elif height_m:
        roof_top = float(height_m)
    else:
        roof_top = max(float(np.percentile(xyz[:, 2], 99)) - ground_z, 6.0) if len(xyz) else 6.0
    if not levels:
        levels = [{"level": 0, "ring": ring_m.tolist(), "z_lo": ground_z, "z_hi": ground_z + roof_top}]
    cell = M.roof_cell_m if M.roof_cell_m > 0 else (0.75 if n_on / max(poly.area, 1.0) >= 4.0 else 1.5)
    xs, ys, Z, valid = roof_heightfield(xyz, poly, ground_z, cell, max_z=ground_z + roof_top + 1.5)
    if M.roof_tidy and valid.sum() >= 8:
        Z = tidy_roof_field(xyz, poly, ground_z, xs, ys, Z, valid)
    if path == "box" or valid.sum() < 4:
        # a flat roof at the building's height; the height field is not trusted
        valid[:] = False

    # Multi-Plane Corner Snapping: rectify footprint corners from adjacent verified wall plane lines
    level0_planes = [p for p in planes if p.face < LEVEL_FACE_STRIDE]
    if any(isinstance(p, VerifiedWallPlane) for p in level0_planes):
        rect_ring, rect_planes = rectify_footprint_corners(levels[0]["ring"], level0_planes)
        levels[0]["ring"] = rect_ring.round(3).tolist()
        rect_by_face = {p.face: p for p in rect_planes}
        planes = [rect_by_face.get(p.face, p) for p in planes]

    ground_field = ground_field_for_building(
        xyz,
        np.asarray(levels[0]["ring"]),
        classification=cls_arr,
        fallback_ground_z=ground_z,
    )

    edges, edge_level = [], {}
    for lv in levels:
        ring_lv = np.asarray(lv["ring"], dtype=float)
        fp_edges = footprint_edges(ring_lv, float(lv["z_lo"]), float(lv["z_hi"]), min_length_m=0.3, arcs=(int(lv["level"]) == 0))
        assert len(fp_edges) < LEVEL_FACE_STRIDE, f"level {lv['level']} has {len(fp_edges)} edges, exceeds stride {LEVEL_FACE_STRIDE}"
        for e in fp_edges:
            e.face += int(lv["level"]) * LEVEL_FACE_STRIDE
            edges.append(e); edge_level[e.face] = lv
    by_face = {p.face: p for p in planes}
    party = party_edges(ring_m, neighbours, [e for e in edges if edge_level[e.face]["level"] == 0], tops=nb_tops)
    # the building's class and its strategy (classify_buildings.py, configs/class_strategies.yaml)
    whole_party = set()
    if M.party_walls == "whole":
        for e in edges:
            if e.face in party and not isinstance(e, ArcWall):
                shared = sum(s1 - s0 for s0, s1, _ in party[e.face])
                if shared >= 0.5 * e.length_m:
                    whole_party.add(e.face)
        print(f"  class {BS.building_class}: {len(whole_party)} party wall(s) drawn whole, plain up to the neighbour's roof")
    classification = building_classification(b, xyz, poly, levels, profiles, edges, party, Z, valid, cell)
    if len(levels) > 1:
        print(f"  {len(levels)} levels: " + "; ".join(f"level {lv['level']} {float(lv['z_lo']):.0f}-{float(lv['z_hi']):.0f} m" for lv in levels))
    # The roof field the walls are checked against: a wall band may only
    # stand where there is roof behind it at that height. RIVUE Tower's
    # mirrored bands ran a 104 m sheet along the block's edge at tower
    # height; the tower is 30 m wide.
    roof = RoofField(xs, ys, Z, valid) if valid.any() else None

    glb = GlbRecorder(GlbWriter(M.texture, dump_dir=(out / "textures") if args.dump_textures else None))
    atlas = TextureAtlas()
    textured = []                                    # (handle, quads with uv in the image's [0,1] frame)
    flat_quads = []                                  # (rgb, quads)
    all_wall_quads = []
    n_tris = 0
    n_profile = n_prism = n_party = 0
    n_matched_faces, matched_m2 = 0, 0.0              # walls whose depth came from the frames (wall_match)
    used_bands = {}
    painted = []                                     # what painted each wall section (mesh.paint)
    n_arc = 0
    def neighbour_wall_at(pl, end: int, z: float):
        """Where the face meeting ``pl`` at its p0 (``end`` 0) or p1 (1)
        has its wall at height ``z``, in plan: on its band's line if it has a
        wall band there, else on the footprint (where its prism stands);
        None when no straight face meets it there."""
        corner = pl.p0 if end == 0 else pl.p1
        # the footprint's edges, not the wall planes: an edge under 3 m has
        # no plane (cfb7194a's face 1006) and stands as a prism on the line
        for nb in edges:
            if nb.face == pl.face or isinstance(nb, ArcWall):
                continue
            if np.hypot(*((nb.p1 if end == 0 else nb.p0) - corner)) > 0.05:
                continue
            nb_plane = by_face.get(nb.face)
            if nb_plane is None or isinstance(nb_plane, ArcWall):
                return np.asarray(corner, float)
            nb_bands = plane_bands(nb_plane, profiles.get(nb.face, []))
            lidar = any(x.status in WALL_STATUSES and x.status != "matched" and x.offset_m is not None for x in nb_bands)
            off = 0.0
            for x in nb_bands:
                if (x.z_lo <= z < x.z_hi and x.status in WALL_STATUSES and x.offset_m is not None
                        and (lidar or x.status != "matched")):
                    off = float(x.offset_m)
            return np.asarray(corner, float) + off * np.asarray(nb_plane.normal, float)
        return None

    for edge in edges:
        plane = by_face.get(edge.face)
        arc = plane if isinstance(plane, ArcWall) else (edge if isinstance(edge, ArcWall) else None)
        if plane is not None and isinstance(edge, ArcWall) != isinstance(plane, ArcWall):
            plane = None                                 # the ring changed shape since the colourise
        bands = plane_bands(plane, profiles.get(plane.face, [])) if plane is not None else []
        if edge.face in whole_party:
            bands = []                                   # the party-wall path below, not what was seen on it
        # A matched band (its depth from the frames) fills a wall the LiDAR
        # left open on a face it measured; it is never the only evidence for
        # a face. A face with no LiDAR wall band takes the prism, as it would
        # without matching -- one matched band on the Omni's all-unknown face
        # 1000 turned a full prism into one 5 m strip and a 1,000 m2 hole.
        if not any(b.status in WALL_STATUSES and b.status != "matched" and b.offset_m is not None for b in bands):
            bands = [b for b in bands if b.status != "matched"]
        quads, extent = (
            wall_geometry(plane, bands, roof, ground_field=(ArcGround(ground_field, arc) if arc is not None and ground_field is not None else ground_field),
                          edge_face=edge.face, foot_to_roof=M.tower_foot == "down_to_roof", wall_cap=M.wall_cap)
            if (plane is not None and path != "box")
            else ([], None)
        )
        if quads and arc is None and roof is not None and M.short_face_share > 0 and edge.face not in whole_party:
            # Measured walls that stop well short of the roof beside them, with
            # nothing measured above: the face takes the full wall along the
            # outline, as one with nothing measured at all does. SpringHill
            # Suites' face 10 was seen 5-10 m up only; drawn to 10 m, it left a
            # 20 m strip open under its 30 m roof (1,067 m2 of holes).
            top_wall = max((float(q[0][:, 2].max()) for q in quads if q[3] == "wall"), default=None)
            beside = [z for _, _, z in roof.top_along(edge) if np.isfinite(z)]
            if top_wall is not None and beside and \
                    top_wall < ground_z + M.short_face_share * (float(np.median(beside)) - ground_z):
                print(f"  face {edge.face}: walls measured to {top_wall:.0f} m under a roof at {np.median(beside):.0f} m; "
                      f"the full wall along the outline")
                quads, extent = [], None
        if quads and arc is None and any(b.status == "matched" for b in bands):
            quads = quads + corner_returns(plane, bands, lambda end, z, pl=plane: neighbour_wall_at(pl, end, z), roof)
        if quads:
            used_bands[plane.face] = bands
            matched = [(float(b.z_lo), float(b.z_hi)) for b in bands if b.status == "matched"]
            matched_m2 += sum((b.z_hi - b.z_lo) * (b.along_hi - b.along_lo) for b in bands if b.status == "matched")
            n_matched_faces += bool(matched)
            # a flat face is painted from whole photographs, one per flat
            # section of it (mesh.paint); a section no frame sees, and a
            # curved face, take the coloured points as before
            sections = wall_sections(plane, [q for q in quads if q[3] == "wall"]) if (arc is None and cams) else []
            from_points, colors, how = [], [], []
            for sec in sections:
                tex, info = paint_section(plane, sec, cams, M.texel_m)
                if tex is None:
                    from_points.extend(sec["quads"])
                    continue
                in_sec = [(max(z0, sec["z_lo"]), min(z1, sec["z_hi"])) for z0, z1 in matched
                          if min(z1, sec["z_hi"]) > max(z0, sec["z_lo"])]
                pieces = [(q[0], q[1], section_uv(plane, sec, np.asarray(q[0], dtype=np.float64)), q[3])
                          for q in sec["quads"]]
                textured.append((atlas.add(tex, {"face": edge.face, "kind": "profile", "z": (sec["z_lo"], sec["z_hi"]),
                                                 "along": (sec["a_lo"], sec["a_hi"]), "frames": info["frames"],
                                                 **({"matched": in_sec} if in_sec else {})}), pieces))
                painted.append(paint_record(plane, edge.face, sec, info["frames"]))
                colors.append(np.asarray(tex).reshape(-1, 3).mean(axis=0))
                how.append(f"{tex.shape[1]}x{tex.shape[0]} px from {info['frames'][0]}"
                           + (f" (+{len(info['frames']) - 1})" if len(info["frames"]) > 1 else ""))
            if not sections:
                from_points = [q for q in quads if q[3] == "wall"]
            if from_points:
                if arc is not None:
                    # texture from the coloured points in the unrolled frame; the
                    # rectangles back onto the circle, cut every 1.5 m along it
                    near = np.abs(np.hypot(*(cxyz[:, :2] - arc.centre).T) - arc.radius) <= 12.0
                    tex, n_pts = face_texture(plane, bands, extent, cxyz[near], crgb[near], M.texel_m,
                                              party=edge.face in party)
                    quads = roll_quads(quads, arc)
                    from_points = [q for q in quads if q[3] == "wall"]
                    n_arc += 1
                else:
                    tex, n_pts = face_texture(plane, bands, extent, cxyz, crgb, M.texel_m, party=edge.face in party)
                textured.append((atlas.add(tex, {"face": edge.face, "kind": "profile", "z": (extent[0], extent[1]),
                                                 "along": (0.0, extent[2]), **({"matched": matched} if matched else {})}),
                                 from_points))
                colors.append(np.asarray(tex).reshape(-1, 3).mean(axis=0))
                how.append(f"{tex.shape[1]}x{tex.shape[0]} px from {n_pts:,} points")
            walls = [q for q in quads if q[3] == "wall"]
            flat = [q for q in quads if q[3] != "wall"]
            if flat:
                flat_quads.append((np.mean(colors, axis=0), flat))
            all_wall_quads.extend(quads)
            n_profile += 1
            z0, z1, L = extent
            print(f"  face {edge.face}: {len(walls)} wall rectangle(s), {len(flat)} ledge/return(s), "
                  f"{L:.0f} m x {z1 - z0:.0f} m, texture " + "; ".join(how))
            continue
        # prism faces at the roof's height beside this edge, stretch by
        # stretch: a one-storey base gets its 13 m wall along the whole
        # edge and a step where the base ends; a stretch with no roof
        # beside it gets no wall
        seg = edge.p1 - edge.p0
        L = float(np.hypot(*seg)); e1 = seg / L
        n3 = np.array([edge.normal[0], edge.normal[1], 0.0])
        lv = edge_level[edge.face]
        lv_floor, lv_ceiling = max(ground_z, float(lv["z_lo"])), float(lv["z_hi"]) + 1.0
        edge_ground = ArcGround(ground_field, arc) if (arc is not None and ground_field is not None) else ground_field
        if roof is not None:
            stations = [(a, b, min(z, lv_ceiling)) for a, b, z in roof.top_along(edge)
                        if np.isfinite(z) and z - lv_floor >= 0.5]
        else:
            stations = [(0.0, L, ground_z + roof_top)] if roof_top >= 0.5 else []
        if edge.face in whole_party and stations:
            # a party wall drawn whole stands to the top of the wall the LiDAR
            # saw on it, where that is above the roof just inside -- a parapet
            # (644a6ecb's party wall stopped 1.5 m short of its own top) -- but
            # at most ``max_parapet_m`` above it: beside a taller neighbour the
            # returns along the line are the neighbour's wall, and raised to them
            # the held-out buildings stood 1,365 m2 of wall above their roofs
            seen = [x for x in profiles.get(edge.face, []) if x.status in ("observed", "extended", "bridged")
                    and x.offset_m is not None and abs(x.offset_m) <= 1.0]
            max_parapet_m = 2.0
            if seen:
                off = float(np.median([x.offset_m for x in seen]))
                # the wall's own returns, the trees' too: a parapet can pass
                # for twigs (colourise.levels, wall_xyz)
                along, across = edge.local(xyz_all[:, :2])
                near = (np.abs(across - off) <= 0.75) & (xyz_all[:, 2] > lv_floor + 0.5)
                raised = []
                for a, c, z in stations:
                    m = near & (along >= a) & (along <= c)
                    top = float(np.percentile(xyz_all[m, 2], 98)) if m.sum() >= 10 else -np.inf
                    raised.append((a, c, min(max(z, min(top, z + max_parapet_m)), lv_ceiling)))
                stations = raised
        # merge neighbouring stretches at the same height
        runs = []
        for a, b, z in stations:
            if runs and abs(runs[-1][2] - z) < 1.0 and abs(runs[-1][1] - a) < 1e-6:
                runs[-1] = (runs[-1][0], b, 0.5 * (runs[-1][2] + z))
            else:
                runs.append((a, b, z))
        made = False
        plain_any = False
        for a, b, z_top in runs:
            # a party wall is plain up to the neighbour's roof and seen above
            # it, stretch by stretch along the edge
            for c0, c1, plain_top in party_steps(party.get(edge.face, []), a, b, z_top):
                if edge_ground is not None and edge.face < LEVEL_FACE_STRIDE and abs(lv_floor - ground_z) < 1e-3:
                    za = edge_ground.elevation_along_edge(edge.face, c0, L)
                    zb = edge_ground.elevation_along_edge(edge.face, c1, L)
                else:
                    za = lv_floor
                    zb = lv_floor
                pa = edge.p0 + c0 * e1; pb = edge.p0 + c1 * e1
                steps = [(za, zb, z_top, True)] if plain_top is None else (
                    [(za, zb, z_top, False)] if plain_top >= z_top else
                    [(za, zb, plain_top, False), (plain_top, plain_top, z_top, True)])
                for z_a, z_b, z_t, seen_wall in steps:
                    corners = _orient(np.array([[*pa, z_a], [*pb, z_b], [*pb, z_t], [*pa, z_t]]), n3)
                    quad = (corners, n3, np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64), "wall")
                    # a curved edge's prism face follows the circle, in pieces
                    # that share one texture sampled along the curve
                    pieces = roll_quads([quad], arc) if arc is not None else [quad]
                    all_wall_quads.extend(pieces)
                    made = True
                    if not seen_wall:
                        plain_any = True
                        flat_quads.append((None, pieces))
                        continue
                    if arc is None:
                        # painted whole, as a measured wall is (mesh.paint), and
                        # as finely: at the 0.3 m the texel-by-texel sampling
                        # used, a house's porch columns were one blur
                        tex, info = paint_quad(corners, n3, cams, M.texel_m)
                        if tex is not None:
                            painted.append(paint_record(edge, edge.face, {"off": 0.0, "a_lo": c0, "a_hi": c1,
                                                                          "z_lo": float(min(z_a, z_b)),
                                                                          "z_hi": float(z_t)}, info["frames"], kind="prism"))
                    else:
                        tex, n_seen = quad_texture_from_cameras(corners, n3, cams, arc=arc)
                    if tex is None:
                        if len(cxyz):
                            dummy_band = FaceBand(float(z_a), float(z_t), 0.0, 10, 0.0, 0.0, L, "observed")
                            tex_cloud, n_pts = face_texture(edge, [dummy_band], (float(z_a), float(z_t), L), cxyz, crgb, M.texel_m)
                            if n_pts > 0:
                                textured.append((atlas.add(tex_cloud, {"face": edge.face, "kind": "prism", "z": (float(z_a), float(z_t)), "along": (c0, c1)}), pieces))
                                continue
                        flat_quads.append((None, pieces))
                    else:
                        textured.append((atlas.add(tex, {"face": edge.face, "kind": "prism", "z": (float(z_a), float(z_t)), "along": (c0, c1)}), pieces))
        if made:
            if plain_any:
                n_party += 1
            else:
                n_prism += 1
    # which frames painted which wall: the texture check scores a frame only
    # where it did not paint (texture_check.score_face)
    (args.building / PAINT_FILE).write_text(json.dumps(painted), encoding="utf-8")
    if n_prism or n_party or n_arc:
        print(f"  {n_profile} face(s) from the profile ({n_arc} curved), {n_prism} prism face(s) at the roof's edge height, "
              f"{n_party} party wall(s) drawn plain")

    # The underside of an upper part standing out over the part below it (the
    # Hyatt's drum over its shaft, colorize_building.py / find_overhangs): a
    # flat ceiling, facing down, over the plan the outline below was cut by.
    oh_path = args.building / "overhangs.json"
    if oh_path.exists() and path != "box":
        import shapely
        from shapely.geometry import Polygon as _OhPoly
        down = np.array([0.0, 0.0, -1.0])
        for oh in json.loads(oh_path.read_text(encoding="utf-8")):
            z = float(oh["under_z"])
            tris = shapely.constrained_delaunay_triangles(_OhPoly(oh["cut_m"]))
            quads = []
            for t in tris.geoms:
                c = np.asarray(t.exterior.coords)[:3]
                corners = _orient(np.array([[*c[0], z], [*c[1], z], [*c[2], z], [*c[2], z]]), down)
                quads.append((corners, down, None, "ledge"))
            if quads:
                flat_quads.append((None, quads))
                all_wall_quads.extend(quads)
                print(f"  the underside of the overhang at {z:.1f} m: {len(quads)} triangle(s) over {oh['cut_area_m2']:.0f} m2")

    # Fitted wall tiles: the wall where the LiDAR saw it directly.
    tiles = tile_geometry(b, xyz, poly, cxyz, crgb, M.texel_m) if path != "box" else []
    for corners, n, uv, tex, face, n_pts in tiles:
        textured.append((atlas.add(tex, {"face": face, "kind": "tile"}), [(corners, n, uv, "wall")]))
    if tiles:
        print(f"  {len(tiles)} fitted wall tile(s) as textured rectangles, "
              f"{sum(t[5] for t in tiles):,} coloured points behind them")

    # Roof
    if valid.any():
        verts, tris, steep = roof_mesh(xs, ys, Z, valid, all_wall_quads, [by_face[f] for f in used_bands], used_bands, levels=levels)
    else:
        from kyfromabove_3d.stages.overture_boxes import extrude
        P_, N_, I_ = extrude(ring_m, ground_z, roof_top)
        top = N_[:, 2] > 0.5
        verts = P_[top]; tris = I_.reshape(-1, 3)[np.all(top[I_.reshape(-1, 3)], axis=1)]
        remap = -np.ones(len(P_), dtype=int); remap[np.flatnonzero(top)] = np.arange(top.sum())
        tris = remap[tris]; steep = np.zeros((0, 3), dtype=int)
    if len(steep):
        # The steep cells are surfaces the planar walls do not describe.
        # They keep the LiDAR's shape and take their colour from whichever
        # aerial frame faces them, one colour per triangle.
        from kyfromabove_3d.stages.colorize_building import sample_colour
        v0, v1, v2 = verts[steep[:, 0]], verts[steep[:, 1]], verts[steep[:, 2]]
        n = np.cross(v1 - v0, v2 - v0); n /= np.linalg.norm(n, axis=1)[:, None] + 1e-9
        cen = (v0 + v1 + v2) / 3
        if cams:
            rgb, seen, _, _ = sample_colour(cen, n, cams, min_facing_cos=0.15, occlusion_slack_m=1.5)
        else:
            rgb, seen = np.zeros((len(cen), 3), dtype=np.uint8), np.zeros(len(cen), dtype=int)
        fill = np.median(rgb[seen > 0], axis=0) if (seen > 0).any() else np.array([110, 110, 110])
        rgb[seen == 0] = fill
        P = np.vstack([v0, v1, v2]).reshape(3, -1, 3).transpose(1, 0, 2).reshape(-1, 3)
        N = np.repeat(n, 3, axis=0)
        C = np.repeat(rgb, 3, axis=0)
        glb.primitive_coloured(P, N, C, np.arange(len(P)), glb.material_flat((255, 255, 255), "steep"))
        n_tris += len(steep)
        print(f"  {len(steep):,} steep roof triangle(s) kept as LiDAR-shaped wall, coloured from the frames "
              f"({(seen > 0).sum():,} seen)")
    tex = None
    bbox = None
    source_desc = ""
    if cams and len(verts):
        if valid.any():
            from shapely.geometry import Polygon as _RoofPoly
            roof_outline = _RoofPoly(np.asarray(levels[0]["ring"], dtype=float)[:, :2]).buffer(0)
            tex, bbox, n_seen, n_tex = roof_texture(xs, ys, Z, valid, poly, cams, M.texel_m, outline=roof_outline)
        else:
            flat_valid = np.ones_like(valid); Zf = np.full(Z.shape, ground_z + roof_top)
            tex, bbox, n_seen, n_tex = roof_texture(xs, ys, Zf, flat_valid, poly, cams, M.texel_m)
        source_desc = f"{n_seen:,} of {n_tex:,} texels seen by a camera ({', '.join(heads)})"
    elif len(verts) and len(cxyz) and (crgb.sum(axis=1) > 15).any():
        flat_z = (ground_z + roof_top) if path == "box" else None
        tex, bbox, n_pts, n_in = roof_texture_from_cloud(xs, ys, Z, valid, poly, cxyz, crgb, M.texel_m, flat_z=flat_z)
        source_desc = f"{n_pts:,} coloured point returns from {ply_path.name}"

    roof_tex = tex if (tex is not None and len(verts)) else None
    if tex is not None and len(verts):
        minx, miny, maxx, maxy = bbox
        uv = np.column_stack([(verts[:, 0] - minx) / max(maxx - minx, 1e-6), (verts[:, 1] - miny) / max(maxy - miny, 1e-6)])
        up = np.repeat(np.array([[0.0, 0.0, 1.0]]), len(verts), axis=0)
        glb.primitive(verts, up, uv, tris.ravel(), glb.material_texture(tex, "roof"))
        print(f"  roof: {valid.sum():,} cells of {cell} m, {len(tris):,} triangles, "
              f"{(Z[valid].min() if valid.any() else roof_top):.0f}-{(Z[valid].max() if valid.any() else roof_top):.0f} m; "
              f"texture {tex.shape[1]}x{tex.shape[0]} px ({source_desc})")
    elif len(verts):
        up = np.repeat(np.array([[0.0, 0.0, 1.0]]), len(verts), axis=0)
        glb.primitive(verts, up, None, tris.ravel(), glb.material_flat((150, 150, 148), "roof_plain"))
        print(f"  roof: {len(tris):,} triangles, plain (no camera or cloud colour sees this building)")
    n_tris += len(tris)

    # Walls: the atlas pages, one primitive each; the plain quads in one more.
    pages = atlas.pack() if textured else []
    page_mats = [glb.material_texture(pg, f"walls_{k}") for k, pg in enumerate(pages)]
    per_page = {k: [] for k in range(len(pages))}
    for handle, quads in textured:
        for corners, n, uv, kind in quads:
            pi, uv_page = atlas.uv(handle, uv)
            per_page[pi].append((corners, n, uv_page, kind))
    for k, quads in per_page.items():
        if quads:
            P, N, T, I = quads_to_arrays(quads)
            glb.primitive(P, N, T, I, page_mats[k])
            n_tris += len(I) // 3
    if flat_quads:
        seen_cols = [c for c, _ in flat_quads if c is not None]
        plain = np.median(np.array(seen_cols), axis=0) if seen_cols else np.array([150, 150, 148])
        Pq, Nq, Cq, Iq = [], [], [], []
        for c, quads in flat_quads:
            P, N, T, I = quads_to_arrays(quads)
            Iq.extend((np.asarray(I) + len(Pq)).tolist()); Pq.extend(P.tolist()); Nq.extend(N.tolist())
            Cq.extend([list(c if c is not None else plain)] * len(P))
        glb.primitive_coloured(np.array(Pq), np.array(Nq), np.array(Cq, dtype=float), np.array(Iq), glb.material_flat((255, 255, 255), "plain"))
        n_tris += len(Iq) // 3
    if pages:
        print(f"  textures: {len(textured)} images packed into {len(pages)} page(s) "
              f"({', '.join(f'{pg.shape[1]}x{pg.shape[0]}' for pg in pages)})")

    texels = texel_sources([img for img in atlas.images] + ([roof_tex] if roof_tex is not None else []))
    extras = {"name": name, "gers": gers, "path": path, "archetype": classification.archetype, "texels": texels,
              "levels": len(levels), "n_frames": len(cams), "heads": heads,
              **({"parent_gers": fp_entry["parent_gers"], "split": fp_entry.get("split")} if fp_entry.get("parent_gers") else {}),
              "faces": {"profile": n_profile, "prism": n_prism, "party": n_party}, "height_m": roof_top,
              "source": ("not in the LiDAR: a box at the recorded height, coloured from the KyFromAbove aerial frames; "
                         "Overture footprint" if not_in_lidar else
                         "KyFromAbove LiDAR and aerial frames (no photographs); Overture footprint"),
              **({"not_in_lidar": True} if not_in_lidar else {}),
              **({"matched": {"faces": n_matched_faces, "area_m2": round(matched_m2, 1)}} if n_matched_faces else {})}
    size = glb.write(out / "building.glb", extras)

    # Tileset: the mesh in its local frame, placed by the root transform.
    transform, origin_geo = enu_transform(coords)
    allp = np.vstack([q[0] for q in all_wall_quads] + [t[0] for t in tiles] + ([verts] if len(verts) else []) + [np.column_stack([ring_m, np.zeros(len(ring_m))])])
    centre = 0.5 * (allp.min(axis=0) + allp.max(axis=0))
    radius = float(np.linalg.norm(allp - centre, axis=1).max())
    tileset = {
        "asset": {"version": "1.1", "generator": "kyfromabove-3d build_mesh.py"},
        "geometricError": 500.0,
        "root": {"transform": transform,
                 "boundingVolume": {"sphere": centre.tolist() + [radius]},
                 "geometricError": 0.0, "refine": "REPLACE",
                 "content": {"uri": "building.glb"}},
        "extras": {"name": name, "gers": gers, "triangles": int(n_tris), "kind": "mesh", "path": path,
                   "archetype": classification.archetype, "texels": texels,
                   "faces": {"profile": n_profile, "prism": n_prism, "party": n_party}, "n_frames": len(cams),
                   "origin_lon_lat_h": origin_geo,
                   **({"parent_gers": fp_entry["parent_gers"]} if fp_entry.get("parent_gers") else {}),
                   **({"matched": extras["matched"]} if "matched" in extras else {}),
                   **({"not_in_lidar": True, "source": extras["source"]} if extras.get("not_in_lidar") else {})},
    }
    (out / "tileset.json").write_text(json.dumps(tileset, indent=1), encoding="utf-8")
    if n_matched_faces:
        write_matched_tint(out, glb, atlas, M, extras, tileset)
    print(f"mesh -> {out / 'building.glb'} ({size / 1e6:.1f} MB, {n_tris:,} triangles); tileset -> {out / 'tileset.json'}")
    print(f"[time] mesh {time.time() - t_start:.1f} s (with the colourise)")
    if args.complete:
        t_done = time.time()
        write_completed(args.building, glb, atlas, roof_tex, M, args.dump_textures, extras, tileset)
        print(f"[time] complete {time.time() - t_done:.1f} s")
    if args.neural:
        t_neur = time.time()
        write_neural(args.building, glb, atlas, roof_tex, M, args.dump_textures, extras, tileset)
        print(f"[time] neural {time.time() - t_neur:.1f} s")
    return 0


def write_matched_tint(out: Path, glb: "GlbRecorder", atlas: TextureAtlas, M, extras: dict, tileset: dict):
    """The measured mesh again with the walls whose depth came from the
    frames (``matched`` bands, kyfromabove_3d.wall_match) tinted cyan for the
    viewer: ``building_matched.glb`` and ``tileset_matched.json``. Measured,
    not invented -- product 2 -- but from the frames, not the LiDAR, and
    shown as such. A face texture's rows run top-down over its ``z`` range."""
    images = []
    for img, meta in zip(atlas.images, atlas.meta):
        img = np.asarray(img)
        if not meta.get("matched"):
            images.append(img)
            continue
        z0, z1 = meta["z"]
        H = img.shape[0]
        out_img = img.astype(np.float64).copy()
        for zb0, zb1 in meta["matched"]:
            r0 = int(np.clip(np.floor((z1 - zb1) / max(z1 - z0, 1e-6) * H), 0, H))
            r1 = int(np.clip(np.ceil((z1 - zb0) / max(z1 - z0, 1e-6) * H), 0, H))
            out_img[r0:r1] = 0.45 * out_img[r0:r1] + 0.55 * np.array([40.0, 210.0, 230.0])      # cyan: matched
        images.append(np.clip(out_img, 0, 255).astype(np.uint8))
    a = TextureAtlas(atlas.page_px, atlas.gap)
    for im in images:
        a.add(im)
    w = GlbWriter(M.texture)
    glb.replay(w, {f"walls_{k}": pg for k, pg in enumerate(a.pack())})
    w.write(out / "building_matched.glb", extras)
    ts = json.loads(json.dumps(tileset))
    ts["root"]["content"]["uri"] = "building_matched.glb"
    (out / "tileset_matched.json").write_text(json.dumps(ts, indent=1), encoding="utf-8")
    print(f"matched walls shown -> {out / 'building_matched.glb'} ({extras['matched']['faces']} face(s), "
          f"{extras['matched']['area_m2']:.0f} m2)")


def write_completed(b: Path, glb: "GlbRecorder", atlas: TextureAtlas, roof_tex, M, dump_textures: bool, extras: dict,
                    tileset: dict):
    """The completed product (product 3): the measured mesh again, call for
    call, with every texel no frame saw filled from the building's own
    facade (kyfromabove_3d.facade_fill: its twin, its rhythm, patches of
    itself) -- ``3dtiles_complete/building.glb``, and ``building_tint.glb``
    with the invented texels tinted for the viewer. The wall textures and
    their sources go to ``fill_faces.npz`` for the held-out test."""
    from kyfromabove_3d.facade_fill import KNOWN, NAMES, POLICY, complete_textures, patch_fill, shares

    out = b / "3dtiles_complete"
    out.mkdir(parents=True, exist_ok=True)
    rep = b / "mirror_report.json"
    twins = {int(f): int(r["from_face"]) for f, r in json.loads(rep.read_text(encoding="utf-8")).items()
             if r.get("symmetric")} if rep.exists() else {}
    done = complete_textures(atlas.images, atlas.meta, M.texel_m, twins)
    roof = None
    if roof_tex is not None and getattr(roof_tex, "source", None) is not None and "patch" in POLICY["roof"]:
        roof = patch_fill(np.asarray(roof_tex), roof_tex.source, M.texel_m)
    fill = shares([sv for _, sv in done] + ([roof[1]] if roof else []))

    def pages_of(images):
        a = TextureAtlas(atlas.page_px, atlas.gap)
        for im in images:
            a.add(im)
        return a.pack() if images else []

    def tint(img, src):
        if src is None:
            return img
        img = np.asarray(img).astype(np.float64).copy()
        inv = ~np.isin(src, KNOWN)
        img[inv] = 0.45 * img[inv] + 0.55 * np.array([230.0, 40.0, 200.0])      # magenta: invented
        return np.clip(img, 0, 255).astype(np.uint8)

    for name, images, roof_img in (
            ("building.glb", [im for im, _ in done], roof[0] if roof else None),
            ("building_tint.glb", [tint(im, sv) for im, sv in done], tint(*roof) if roof else None)):
        swap = {f"walls_{k}": pg for k, pg in enumerate(pages_of(images))}
        if roof_img is not None:
            swap["roof"] = roof_img
        # the completed pages kept lossless too (--dump-textures), for the
        # batched tiles to repack; not the tinted copy, which is for review
        w = GlbWriter(M.texture, dump_dir=(out / "textures") if dump_textures and name == "building.glb"
                      else None)
        glb.replay(w, swap)
        w.write(out / name, {**extras, "product": "completed", "fill": fill})
    for name, uri in (("tileset.json", "building.glb"), ("tileset_tint.json", "building_tint.glb")):
        ts = json.loads(json.dumps(tileset))
        ts["root"]["content"]["uri"] = uri
        ts["extras"].update({"product": "completed", "fill": fill})
        (out / name).write_text(json.dumps(ts, indent=1), encoding="utf-8")
    images, metas = list(atlas.images), list(atlas.meta)
    if roof_tex is not None and getattr(roof_tex, "source", None) is not None:
        images.append(roof_tex); metas.append({"kind": "roof"})
    walls = {f"img_{i}": np.asarray(im) for i, im in enumerate(images) if getattr(im, "source", None) is not None}
    walls.update({f"src_{i}": im.source for i, im in enumerate(images) if getattr(im, "source", None) is not None})
    np.savez_compressed(out / "fill_faces.npz", meta=json.dumps(metas), texel_m=M.texel_m,
                        twins=json.dumps(twins), **walls)
    print(f"completed -> {out / 'building.glb'}: " + ", ".join(f"{NAMES[k]} {fill[NAMES[k]]:.0%}" for k in sorted(NAMES)))


def write_neural(b: Path, glb: "GlbRecorder", atlas: TextureAtlas, roof_tex, M, dump_textures: bool, extras: dict,
                 tileset: dict):
    """The fourth product: unseen texels filled from the best-matching seen
    patches on any of the building's walls (:mod:`kyfromabove_3d.neural_fill`)
    -- ``3dtiles_neural/building.glb`` and ``building_tint.glb``."""
    from kyfromabove_3d.facade_fill import KNOWN, NAMES, shares
    from kyfromabove_3d.neural_fill import complete_textures_neural, neural_fill, reference_bank

    out = b / "3dtiles_neural"
    out.mkdir(parents=True, exist_ok=True)
    ref_sources = reference_bank(atlas.images, atlas.meta, M.texel_m)
    done = complete_textures_neural(atlas.images, atlas.meta, M.texel_m, ref_sources=ref_sources)
    roof = None
    if roof_tex is not None and getattr(roof_tex, "source", None) is not None:
        roof = neural_fill(np.asarray(roof_tex), roof_tex.source, M.texel_m, kind="roof",
                           ref_sources=ref_sources)
    fill = shares([sv for _, sv in done] + ([roof[1]] if roof else []))

    def pages_of(images):
        a = TextureAtlas(atlas.page_px, atlas.gap)
        for im in images:
            a.add(im)
        return a.pack() if images else []

    def tint(img, src):
        if src is None:
            return img
        img = np.asarray(img).astype(np.float64).copy()
        inv = ~np.isin(src, KNOWN)
        img[inv] = 0.45 * img[inv] + 0.55 * np.array([155.0, 80.0, 240.0])      # violet: neural
        return np.clip(img, 0, 255).astype(np.uint8)

    for name, images, roof_img in (
            ("building.glb", [im for im, _ in done], roof[0] if roof else None),
            ("building_tint.glb", [tint(im, sv) for im, sv in done], tint(*roof) if roof else None)):
        swap = {f"walls_{k}": pg for k, pg in enumerate(pages_of(images))}
        if roof_img is not None:
            swap["roof"] = roof_img
        w = GlbWriter(M.texture, dump_dir=(out / "textures") if dump_textures and name == "building.glb"
                      else None)
        glb.replay(w, swap)
        w.write(out / name, {**extras, "product": "neural", "fill": fill})
    for name, uri in (("tileset.json", "building.glb"), ("tileset_tint.json", "building_tint.glb")):
        ts = json.loads(json.dumps(tileset))
        ts["root"]["content"]["uri"] = uri
        ts["extras"].update({"product": "neural", "fill": fill})
        (out / name).write_text(json.dumps(ts, indent=1), encoding="utf-8")
    print(f"neural -> {out / 'building.glb'}: " + ", ".join(f"{NAMES[k]} {fill[NAMES[k]]:.0%}" for k in sorted(NAMES)))


if __name__ == "__main__":
    sys.exit(main())
