#!/usr/bin/env python3
"""End-to-end Phase 1 validation: ingest, model, crop, and *measure*.

Acceptance is decided by measurements that the imagery can fail:

* Frame coverage is tested by projecting the building volume onto each sensor,
  not by asking whether a footprint polygon contains the AOI centre point.
* Reprojection accuracy is measured by locating the same physical surface in two
  exposures via normalised cross-correlation. The previous implementation
  derived its "observed" image position by projecting through the camera under
  test, which made the reported error identically zero regardless of whether the
  camera model was right.
* Crops fail loudly rather than silently truncating the building, and validity
  masks record real sensor coverage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from kyfromabove_3d import (
    BoundingVolume3D,
    BuildingCropExtractor,
    Project,
    build_planar_anchors,
    building_surface,
    ground_elevation_under,
    readers_for_cameras,
    roof_check_features,
    roof_footprint,
)
from kyfromabove_3d.cameras import parse_camera_from_eo_record
from kyfromabove_3d.plots import (
    plot_building_in_crops,
    plot_flight_geometry,
    plot_frame_coverage,
    plot_lidar_overlay,
    plot_residual_field,
    plot_residual_histogram,
)
from kyfromabove_3d.qa import (
    anchors_in_volume,
    measure_cross_view_residuals,
    measure_within_group_residuals,
    prefetch_windows,
    triangulate_anchors,
    triangulation_summary,
    view_geometry_summary,
)
from kyfromabove_3d.sources import (
    build_source_manifest,
    discover_candidate_frames,
    select_frames_by_coverage,
)

# min_facing_cos culls pairs where the camera is behind the surface. It is inert
# for roofs -- every aerial camera is above one -- and removes ~70% of facade
# attempts, which are geometrically impossible and would otherwise be counted as
# low_ncc rejections. cos 0.20 is about 78 deg off the wall.
MATCH_PARAMS = dict(patch_radius_m=6.0, search_radius_m=3.0, gsd_m=0.10,
                    min_facing_cos=0.20)
PREFETCH_RADIUS_M = 12.0  # generous enough for the wider cross-direction search


def run_pipeline(config_path: str | None = None, skip_manifest: bool = False) -> int:
    proj = Project(config_path=config_path)
    cfg = proj.config
    qa_cfg = cfg["qa"]

    print("=" * 70)
    print(f"Phase 1 validation: {cfg['project']['name']} / {proj.building_id}")
    print("=" * 70)

    for d in (proj.building_dir / "images", proj.building_dir / "validity_masks",
              proj.qa_dir / "overlays", proj.derived_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Coordinate frame -------------------------------------------------
    proj.coord_frame.save_json(proj.output_dir / "coordinates.json")
    proj.coord_frame.save_json(proj.building_dir / "coordinates.json")
    print(f"[coords] local origin {proj.origin_source_xyz} (EPSG:3089 ft)")

    # 2. Provenance -------------------------------------------------------
    if skip_manifest:
        print("[manifest] skipped (--skip-manifest)")
    else:
        # Only assets that exist locally can be hashed. The 2024 indexes are
        # 553 MB and 1.07 GB and are read in place over /vsicurl, so they are
        # recorded by URL in the notes rather than downloaded to be SHA-256'd.
        from kyfromabove_3d.urban import collection_full_name

        coll = proj.collection
        base = ("https://kyfromabove.s3.amazonaws.com/imagery/obliques/Phase3/"
                "FlightInformationData_GPKG")
        candidates = [
            (proj.eo_gpkg, f"{base}/{collection_full_name(coll)}_ImageFrameEO.gpkg",
             f"{coll}_EO", f"Exterior orientation table, {coll}"),
            (proj.frames_gpkg, f"{base}/{collection_full_name(coll)}_ImageFrames.gpkg",
             f"{coll}_Frames", f"Image frame footprints, {coll}"),
        ]
        assets = [a for a in candidates if isinstance(a[0], Path) and Path(a[0]).exists()]
        remote = [a[1] for a in candidates if a not in assets]
        from kyfromabove_3d.lidar import LidarTileIndex

        tiles = LidarTileIndex(proj.raw_dir).tiles_for_bounds(proj.aoi_bounds_ft_xy)
        remote += [t.url for t in tiles]
        manifest = build_source_manifest(
            project_name=cfg["project"]["name"],
            assets=assets,
            crs=cfg["aoi"]["horizontal_crs"],
            vertical_datum=cfg["coordinates"]["vertical_datum"],
            root=proj.root,
            metadata_notes={
                "collection": coll,
                "sensor_family": cfg["selection"].get("sensor_family"),
                "camera_heads": proj.camera_directions,
                "pixel_pitch_um": 3.76,
                "read_in_place_not_downloaded": remote,
                "lidar_tiles": [f"{t.phase}/{t.tilename}"
                                + (" (COPC)" if t.is_copc else " (plain LAZ)")
                                for t in tiles],
                "distortion_model": "none supplied by the EO table; imagery assumed vendor-rectified",
                "rotation_convention": "R_cw = diag(1,-1,-1) @ (Rx(omega) Ry(phi) Rz(kappa)).T",
                "principal_point_convention": "cx = w/2 + CamPpxMm/pitch; cy = h/2 - CamPpyMm/pitch",
            },
        )
        manifest.save_json(proj.output_dir / "source_manifest.json")
        print(f"[manifest] registered {len(manifest.assets)} assets with SHA-256")

    # 3. LiDAR and building geometry --------------------------------------
    lidar = proj.lidar
    lidar.save_laz(proj.building_dir / "lidar_local.laz", use_local_coords=True)
    print(f"[lidar] {len(lidar)} points in AOI -> lidar_local.laz")

    # A 15 m ceiling suits the single-storey pilot and silently excludes a
    # tower's roof entirely; "largest" then picks the podium. Both are
    # config-driven so a dense AOI can say what it actually is.
    roof_cfg = cfg["aoi"].get("roof", {})
    check_features, roof_xyz = roof_check_features(
        lidar,
        max_height_m=roof_cfg.get("max_height_m", 15.0),
        select=roof_cfg.get("select", "largest"),
    )
    if not check_features:
        print("[FAIL] could not isolate a roof surface from the LiDAR")
        return 1
    # Extent comes from the whole building surface -- every roof level plus its
    # facades -- not from the selected roof cluster. On a stepped tower that
    # cluster is the crown: 15.5 x 15.8 m against the building's 45.4 x 45.3 m.
    surface = building_surface(lidar)
    footprint = roof_footprint(surface)
    # The base is the measured ground under the footprint, not the roof minus a
    # constant. "roof - 8 m" encodes a single-storey building: on the pilot's
    # 1.3 m roof it reaches ground, and on a 160 m roof it produces a 16 m slab
    # floating at 150 m with the whole tower outside it.
    base_z = ground_elevation_under(
        lidar, np.column_stack([footprint,
                                np.full(len(footprint), float(np.median(roof_xyz[:, 2])))]))
    volume = BoundingVolume3D.from_footprint_and_elevation(
        footprint,
        min_z_m=float(base_z),
        max_z_m=float(max(roof_xyz[:, 2].max(), surface[:, 2].max())),
        pad_xy_m=6.0,
        pad_z_m=2.0,
    )
    sides = np.linalg.norm(np.diff(np.vstack([footprint, footprint[:1]]), axis=0), axis=1)
    print(f"[building] roof from {len(roof_xyz)} LiDAR points; "
          f"footprint {sides[0]:.1f} x {sides[1]:.1f} m "
          f"({sides[0] * sides[1]:.0f} sq m)")
    with open(proj.building_dir / "check_features.json", "w", encoding="utf-8") as f:
        json.dump({"features": [c.to_dict() for c in check_features],
                   "volume_local_m": volume.to_dict()}, f, indent=2)

    # 4. Frame selection by actual sensor coverage -------------------------
    frames_df = discover_candidate_frames(
        proj.eo_gpkg, proj.frames_gpkg, proj.aoi_bounds_ft_xy,
        flight_line=proj.flight_lines, camera_direction=proj.camera_directions,
        max_frames=None, require_containment=True,
    )
    all_cameras = [parse_camera_from_eo_record(dict(r), proj.coord_frame)
                   for _, r in frames_df.iterrows()]
    cameras, coverage = select_frames_by_coverage(all_cameras, volume)
    coverage.to_csv(proj.qa_dir / "frame_coverage.csv", index=False)
    print(f"[frames] {len(all_cameras)} candidate exposures over the AOI "
          f"({', '.join(proj.camera_directions)}); "
          f"{len(cameras)} fully contain the building volume")
    for _, row in coverage.iterrows():
        mark = "keep" if row["selected"] else "DROP"
        print(f"    {mark}  {Path(row['frame_id']).name}: "
              f"{row['corners_on_sensor']}/8 corners on sensor, "
              f"margin {row['margin_px']:.0f} px")
    if len(cameras) < qa_cfg["minimum_frame_count"]:
        print(f"[FAIL] only {len(cameras)} usable exposures, "
              f"need {qa_cfg['minimum_frame_count']}")
        return 1

    geometry = view_geometry_summary(cameras)
    print(f"[geometry] {geometry['n_exposures']} exposures, "
          f"{len(geometry['look_directions'])} look directions, "
          f"{len(geometry['flight_lines'])} flight lines")
    print(f"[geometry] max view-ray separation "
          f"{geometry['max_ray_separation_deg']:.1f} deg, "
          f"azimuth concentration R={geometry['azimuth_concentration']:.2f} "
          f"(1.00 = all from one bearing)")
    with open(proj.qa_dir / "view_geometry.json", "w", encoding="utf-8") as f:
        json.dump(geometry, f, indent=2)

    readers = readers_for_cameras(cameras)

    # 5. Measured cross-view reprojection QA -------------------------------
    anchors = build_planar_anchors(
        lidar, include_facades=cfg["qa"].get("include_facade_anchors", False))
    print(f"[qa] {len(anchors)} planar LiDAR anchors")
    windows = prefetch_windows(anchors, cameras, readers, PREFETCH_RADIUS_M)

    # Within-group: like compared with like, so this tests the camera model.
    print("[qa] within-group residuals (camera model check):")
    report = measure_within_group_residuals(anchors, cameras, readers,
                                            windows=windows, **MATCH_PARAMS)
    stats = report.stats()
    report.save_json(proj.qa_dir / "reprojection_report.json")
    report.to_dataframe().to_csv(proj.qa_dir / "cross_view_matches.csv", index=False)

    # Cross-direction: also carries line-to-line georeferencing bias, and is
    # harder to correlate, so it is reported rather than gated on.
    cross = measure_cross_view_residuals(
        anchors, cameras, readers, pair_mode="reference",
        reference=cfg["qa"].get("reference_frame", "Fwd_2034_201824"),
        windows=windows, progress=False,
        **{**MATCH_PARAMS, "search_radius_m": 6.0, "min_ncc": 0.7})
    cross_stats = cross.stats()
    cross.save_json(proj.qa_dir / "cross_direction_report.json")

    # Restricted to the building. The unrestricted number above is dominated by
    # anchors on ground and treeline, which is not what the building's walls
    # depend on.
    in_volume = {a.anchor_id for a in anchors_in_volume(anchors, volume)}
    cross_building = cross.filter_anchors(in_volume)
    cross_building_stats = cross_building.stats()
    cross_building.save_json(proj.qa_dir / "cross_direction_building_report.json")

    tri = triangulate_anchors(anchors, cameras, readers, windows=windows,
                              min_views=3,
                              **{**MATCH_PARAMS, "search_radius_m": 6.0, "min_ncc": 0.7})
    tri_stats = triangulation_summary(tri)
    with open(proj.qa_dir / "triangulation_report.json", "w", encoding="utf-8") as f:
        json.dump({"summary": tri_stats, "points": [t.to_dict() for t in tri]}, f, indent=2)

    print(f"[qa] within-group: accepted {stats['accepted']}/{stats['attempted']} "
          f"({100 * stats['acceptance_rate']:.0f}%), median "
          f"{stats['median_error_px']:.2f} px, p90 {stats['p90_error_px']:.2f} px")
    print(f"[qa] cross-direction, all anchors : median "
          f"{cross_stats['median_error_px']:.2f} px over "
          f"{cross_stats['accepted']} matches "
          f"({100 * cross_stats['acceptance_rate']:.0f}% accepted)")
    if cross_building_stats.get("accepted"):
        print(f"[qa] cross-direction, in volume  : median "
              f"{cross_building_stats['median_error_px']:.2f} px over "
              f"{cross_building_stats['accepted']} matches "
              f"({100 * cross_building_stats['acceptance_rate']:.0f}% accepted)"
              f"   <- the number reconstruction depends on")
    else:
        print("[qa] cross-direction, in volume  : no matches on the building")
    print(f"[qa] triangulated vs LiDAR: median 3D discrepancy "
          f"{tri_stats.get('median_3d_discrepancy_m', float('nan')):.2f} m "
          f"(median ray angle {tri_stats.get('median_max_ray_angle_deg', float('nan')):.0f} deg)")

    # 6. Crops -------------------------------------------------------------
    required = int(np.ceil(max(
        max(row["span_u_px"], row["span_v_px"])
        for _, row in coverage[coverage["selected"]].iterrows()
    ))) + 2 * int(cfg.get("crops", {}).get("padding_px", 120))
    crop_size = int(np.ceil(required / 16.0) * 16)
    print(f"[crops] sizing every crop at {crop_size}px to contain the widest projection")

    extractor = BuildingCropExtractor(proj.coord_frame)
    normalized: Dict[str, Any] = {}
    manifest_rows: List[Dict[str, Any]] = []
    crop_images: List[np.ndarray] = []
    crop_specs: List[Any] = []
    for idx, cam in enumerate(cameras, start=1):
        rgb, mask, spec = extractor.extract_crop(
            camera=cam, reader=readers[cam.frame_id], volume_3d=volume,
            padding_px=cfg.get("crops", {}).get("padding_px", 120),
            fixed_square_size=crop_size, allow_truncation=False,
        )
        img_name, mask_name = f"frame_{idx:03d}.png", f"mask_{idx:03d}.png"
        Image.fromarray(rgb).save(proj.building_dir / "images" / img_name)
        Image.fromarray(mask).save(proj.building_dir / "validity_masks" / mask_name)
        normalized[cam.frame_id] = {
            "frame_id": cam.frame_id, "crop_image": f"images/{img_name}",
            "validity_mask": f"validity_masks/{mask_name}",
            "width": spec.width_px, "height": spec.height_px,
            "fx": spec.fx_px, "fy": spec.fy_px, "cx": spec.cx_px, "cy": spec.cy_px,
            "R_cw": spec.R_cw.tolist(),
            "camera_center_local_m": spec.camera_center_local_m.tolist(),
            "sensor_coverage_fraction": spec.sensor_coverage_fraction,
            "volume_contained": spec.volume_contained,
        }
        manifest_rows.append(spec.to_dict())
        crop_images.append(rgb)
        crop_specs.append(spec)
        print(f"    {img_name}: {spec.width_px}x{spec.height_px} at "
              f"({spec.x0_px},{spec.y0_px}), sensor coverage "
              f"{100 * spec.sensor_coverage_fraction:.1f}%")

        plot_lidar_overlay(
            windows[cam.frame_id], cam, lidar.xyz_local_m, check_features,
            save_to=proj.qa_dir / "overlays" / f"overlay_frame_{idx:03d}.png",
        )

    with open(proj.building_dir / "cameras_normalized.json", "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)
    with open(proj.building_dir / "crop_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, indent=2)

    # 7. Figures -----------------------------------------------------------
    # The overlay draws the building's box, so it wants the extent lifted to
    # roof level -- not the check features, which a curved roof does not name
    # "RoofCorner" at all.
    roof_corners = np.column_stack(
        [footprint, np.full(len(footprint), float(np.median(roof_xyz[:, 2])))])
    plot_building_in_crops(
        crop_images, cameras, [(s.x0_px, s.y0_px) for s in crop_specs], roof_corners,
        ground_z_m=ground_elevation_under(lidar, roof_corners),
        labels=[Path(s.frame_id).name for s in crop_specs],
        ncols=6, save_to=proj.qa_dir / "building_in_crops.png")
    plot_frame_coverage(coverage, all_cameras, save_to=proj.qa_dir / "frame_coverage.png")
    plot_flight_geometry(cameras, footprint, save_to=proj.qa_dir / "flight_geometry.png")
    plot_residual_field(report, anchors, save_to=proj.qa_dir / "residual_field.png")
    plot_residual_histogram(report, target_px=qa_cfg["target_median_check_error_px"],
                            save_to=proj.qa_dir / "residual_histogram.png")

    # 8. Acceptance --------------------------------------------------------
    min_sep = qa_cfg.get("min_view_ray_separation_deg", 30.0)
    checks = {
        "usable exposures >= %d" % qa_cfg["minimum_frame_count"]:
            (len(cameras) >= qa_cfg["minimum_frame_count"], f"{len(cameras)} frames"),
        "within-group residual < %.1f px" % qa_cfg["target_median_check_error_px"]:
            (stats["median_error_px"] < qa_cfg["target_median_check_error_px"],
             f"{stats['median_error_px']:.2f} px over {stats['accepted']} matches"),
        "match acceptance rate >= 80%":
            (stats["acceptance_rate"] >= 0.80, f"{100 * stats['acceptance_rate']:.0f}%"),
        "view-ray separation >= %.0f deg" % min_sep:
            (geometry["max_ray_separation_deg"] >= min_sep,
             f"{geometry['max_ray_separation_deg']:.1f} deg across "
             f"{len(geometry['look_directions'])} look directions"),
        "every crop contains the building":
            (all(r["volume_contained"] for r in manifest_rows),
             f"{sum(r['volume_contained'] for r in manifest_rows)}/{len(manifest_rows)}"),
        "crop padding on sensor >= %.0f%%" % (100 * qa_cfg.get("min_crop_coverage", 0.75)):
            (all(r["sensor_coverage_fraction"] >= qa_cfg.get("min_crop_coverage", 0.75)
                 for r in manifest_rows),
             f"min {min(r['sensor_coverage_fraction'] for r in manifest_rows):.3f}; "
             f"{sum(r['sensor_coverage_fraction'] < 0.999 for r in manifest_rows)} "
             f"crop(s) padded at the sensor edge, marked in the validity mask"),
    }
    passed = all(ok for ok, _ in checks.values())

    findings = [
        "# Camera and imagery QA findings", "",
        f"- **Building**: {cfg['aoi']['building_name']} (`{proj.building_id}`)",
        f"- **Sensor family**: {cfg['selection'].get('sensor_family', 'n/a')}",
        f"- **Exposures used**: {len(cameras)} of {len(all_cameras)} covering the AOI",
        f"- **Look directions**: {', '.join(geometry['look_directions'])} "
        f"over flight lines {', '.join(geometry['flight_lines'])}",
        "", "## View geometry", "",
        f"- max view-ray separation: {geometry['max_ray_separation_deg']:.1f} deg",
        f"- elevation range: {geometry['elevation_min_deg']:.0f}-"
        f"{geometry['elevation_max_deg']:.0f} deg",
        f"- azimuth concentration R = {geometry['azimuth_concentration']:.2f} "
        f"(1.00 means every exposure looks from one bearing)",
        "", "## Measured reprojection", "",
        "Residuals are measured by rectifying image content onto each LiDAR anchor's",
        "local surface plane in two exposures and locating the offset by normalised",
        "cross-correlation. The imagery decides the result; no value is derived from",
        "the camera model being tested.", "",
        "**Within group** (same camera head and flight line) - the camera-model check:",
        "",
        f"- accepted matches: {stats['accepted']} of {stats['attempted']} "
        f"({100 * stats['acceptance_rate']:.0f}%)",
        f"- median {stats['median_error_px']:.2f} px, p90 {stats['p90_error_px']:.2f} px, "
        f"max {stats['max_error_px']:.2f} px",
        f"- median offset on the surface: {100 * stats['median_error_m']:.1f} cm",
        f"- systematic component: {100 * stats['systematic_offset_magnitude_m']:.1f} cm",
        "",
        f"**Across look directions** (reference {cfg['qa'].get('reference_frame')}) - "
        "also carries line-to-line georeferencing bias, and correlation is harder "
        "across wide view-angle changes, so this is reported rather than gated:",
        "",
        f"- accepted matches: {cross_stats['accepted']} of {cross_stats['attempted']} "
        f"({100 * cross_stats['acceptance_rate']:.0f}%)",
        f"- median {cross_stats['median_error_px']:.2f} px, "
        f"p90 {cross_stats['p90_error_px']:.2f} px",
        (f"- restricted to the building volume: "
         f"{cross_building_stats['median_error_px']:.2f} px over "
         f"{cross_building_stats['accepted']} matches "
         f"({100 * cross_building_stats['acceptance_rate']:.0f}% accepted)"
         if cross_building_stats.get("accepted")
         else "- restricted to the building volume: no matches"),
        "", "## Triangulation against LiDAR", "",
        f"- points triangulated: {tri_stats.get('count', 0)}",
        f"- median 3D discrepancy: {tri_stats.get('median_3d_discrepancy_m', float('nan')):.2f} m",
        f"- median horizontal: {tri_stats.get('median_horizontal_m', float('nan')):.2f} m; "
        f"vertical bias {tri_stats.get('median_vertical_bias_m', float('nan')):.2f} m "
        f"(sd {tri_stats.get('vertical_std_m', float('nan')):.2f} m)",
        f"- median max ray intersection angle: "
        f"{tri_stats.get('median_max_ray_angle_deg', float('nan')):.1f} deg",
        "", "## Acceptance", "",
        "| check | result | status |", "|---|---|---|",
    ]
    for name, (ok, detail) in checks.items():
        findings.append(f"| {name} | {detail} | {'**PASS**' if ok else '**FAIL**'} |")
    findings += [
        "", "## Scope of these numbers", "",
        "Residuals are relative: they measure agreement between exposures and with",
        "LiDAR. Absolute georeferencing accuracy is not established, because the",
        "dataset carries no surveyed ground control.",
    ]
    (proj.qa_dir / "camera_findings.md").write_text("\n".join(findings) + "\n", encoding="utf-8")

    print("=" * 70)
    for name, (ok, detail) in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print("=" * 70)
    print("All acceptance criteria passed." if passed else "ACCEPTANCE FAILED.")
    return 0 if passed else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-manifest", action="store_true",
                    help="skip asset hashing (fast iteration)")
    args = ap.parse_args()
    sys.exit(run_pipeline(args.config, skip_manifest=args.skip_manifest))
