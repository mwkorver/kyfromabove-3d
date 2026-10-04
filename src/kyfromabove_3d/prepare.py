"""Make a building's working directory from a footprint: the frame, the
LiDAR clip, the aerial cameras and their crops -- and nothing else.

``scripts/validate_building.py`` does this for one building from a
hand-written YAML, and spends three of its four and a half minutes on the
reprojection QA that the colourise and the mesh never read. A downtown of
375 footprints cannot be prepared by hand or wait for a QA per building, so
this is the same preparation driven by an Overture record (its GERS id, WKT,
and the LiDAR ground and height ``overture_boxes.py lidar-heights``
measured), with three things done once for the whole run rather than per
building:

* the **frame index** (a 1.07 GB GeoPackage read in place over ``/vsicurl``,
  14 s a query) is queried once over the whole area and cached as a local
  GeoPackage;
* the **building volume** that decides which frames see the building and how
  big the crops are is the footprint itself, lifted from the LiDAR ground to
  the LiDAR roof, not a footprint recovered from the LiDAR -- in a row of
  buildings the LiDAR clip holds the neighbours too, and a volume drawn round
  everything in it would be the block;
* the frames are **capped per camera head** (the best four by how far the
  building sits from the sensor's edge), because a downtown building lies in
  20-30 exposures and the crops are the cost of everything downstream.

What is written, in the same layout the three landmark buildings have, so
``colorize_building.py`` and ``build_mesh.py`` run on it unchanged:
``footprint.json`` (the record, its neighbours, its origin), ``coordinates.json``,
``lidar_local.laz``, ``check_features.json`` (the volume; no features),
``images/frame_NNN.png`` + ``validity_masks/mask_NNN.png``,
``cameras_normalized.json``, ``crop_manifest.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from kyfromabove_3d.coordinates import METERS_PER_US_SURVEY_FOOT, LocalCoordinateFrame

COLLECTION = "2024_Season1"
#: KyFromAbove's oblique collections, newest first: a tile takes the newest
#: that has frames over it (2024 Season 1 covers Louisville but not Hazard
#: or Somerset, September 2026).
COLLECTIONS = ("2024_Season1", "2023_Season2", "2023_Season1", "2022_Season2")
HEADS = ("Fwd", "Left", "Right", "Bwd", "Color")


# ------------------------------------------------------------------ the frame

def building_frame(rec: Dict[str, Any], ring_ft: np.ndarray,
                   fallback_ground_m: Optional[float] = None) -> LocalCoordinateFrame:
    """Origin at the footprint's centroid, ground at the LiDAR's ground under
    it, so local z = 0 is this building's own street level."""
    gz = rec.get("lidar_ground_z")
    if gz is None:
        gz = fallback_ground_m
    if gz is None:
        raise ValueError(f"{rec.get('id')}: no ground elevation")
    cx, cy = float(np.mean(ring_ft[:, 0])), float(np.mean(ring_ft[:, 1]))
    return LocalCoordinateFrame(origin_source_xyz=(cx, cy, gz / METERS_PER_US_SURVEY_FOOT))


def aoi_bounds_ft(ring_ft: np.ndarray, ground_ft: float, height_m: Optional[float],
                  pad_ft: float = 90.0) -> Tuple[float, float, float, float, float, float]:
    """(minx, maxx, miny, maxy, minz, maxz) in source feet: the footprint padded
    ``pad_ft`` (the landmarks used 90 ft), from 10 m below ground to 30 m
    above the roof -- enough for the profile's look above the roof line, not
    so much that a neighbouring tower's upper floors join the clip."""
    ft = METERS_PER_US_SURVEY_FOOT
    h = max(float(height_m or 0.0), 6.0)
    return (float(ring_ft[:, 0].min() - pad_ft), float(ring_ft[:, 0].max() + pad_ft),
            float(ring_ft[:, 1].min() - pad_ft), float(ring_ft[:, 1].max() + pad_ft),
            float(ground_ft - 10.0 / ft), float(ground_ft + (h + 30.0) / ft))


def building_volume(ring_m: np.ndarray, height_m: Optional[float], pad_xy_m: float = 3.0,
                    pad_z_m: float = 2.0):
    """The volume the frames must see: the footprint from ground to roof."""
    from kyfromabove_3d.crops import BoundingVolume3D

    h = max(float(height_m or 0.0), 6.0)
    return BoundingVolume3D.from_footprint_and_elevation(np.asarray(ring_m)[:, :2], 0.0, h,
                                                         pad_xy_m=pad_xy_m, pad_z_m=pad_z_m)


# ------------------------------------------------------------ the frame index

def load_frames_index(cache: Path, bounds_ft: Tuple[float, float, float, float],
                      collection: Optional[str] = None, heads: Sequence[str] = HEADS):
    """Every exposure of the collection that intersects ``bounds_ft``
    (minx, miny, maxx, maxy, EPSG:3089 ft), with its exterior orientation
    joined on: read once from the remote indexes and kept as a GeoPackage.
    With no ``collection``, the newest of :data:`COLLECTIONS` with frames
    there."""
    import geopandas as gpd

    cache = Path(cache)
    if cache.exists():
        return gpd.read_file(cache)
    from kyfromabove_3d.sources import discover_candidate_frames
    from kyfromabove_3d.urban import eo_index_url, frames_index_url

    df = None
    for c in ([collection] if collection else COLLECTIONS):
        df = discover_candidate_frames(eo_index_url(c), frames_index_url(c),
                                       tuple(bounds_ft), camera_direction=list(heads),
                                       max_frames=None, require_containment=False)
        if len(df):
            break
    if df is None or len(df) == 0:
        raise RuntimeError(f"no {collection or ', '.join(COLLECTIONS)} frames intersect {bounds_ft}")
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:3089")
    cache.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(cache, driver="GPKG")
    return gdf


def select_cameras(frames_gdf, ring_ft: np.ndarray, frame: LocalCoordinateFrame, volume,
                   max_per_head: int = 4, min_frames: int = 3,
                   corner_steps: Sequence[int] = (8, 6, 4)) -> Tuple[List[Any], Any]:
    """The cameras that see the building, from the cached index.

    Frames whose footprint contains the building's bbox are parsed into
    camera models and tested exactly -- all eight corners of the volume on
    the sensor -- then the best ``max_per_head`` per head by margin are kept.
    When fewer than ``min_frames`` pass, the corner test is relaxed to six,
    then four: a crop that runs off the sensor is padded and its mask says
    so, which is better than a building with no imagery at all.
    """
    from shapely.geometry import box

    from kyfromabove_3d.cameras import parse_camera_from_eo_record
    from kyfromabove_3d.sources import assess_frame_coverage

    b = box(ring_ft[:, 0].min(), ring_ft[:, 1].min(), ring_ft[:, 0].max(), ring_ft[:, 1].max())
    cand = frames_gdf[frames_gdf.contains(b)]
    if len(cand) == 0:
        cand = frames_gdf[frames_gdf.intersects(b)]
    cams = [parse_camera_from_eo_record(dict(r), frame) for _, r in cand.iterrows()]
    if not cams:
        return [], None
    table = assess_frame_coverage(cams, volume)
    chosen: List[Any] = []
    for need in corner_steps:
        ok = table["corners_on_sensor"] >= need
        chosen = []
        for head in sorted(set(table["frame_id"].str.rsplit("/", n=1).str[-1].str.split("_").str[0])):
            rows = table[ok & table["frame_id"].str.contains(f"/{head}_")].sort_values("margin_px", ascending=False)
            chosen += [cams[i] for i in rows.index[:max_per_head]]
        if len(chosen) >= min_frames:
            break
    ids = {c.frame_id for c in chosen}
    table = table.assign(selected=table["frame_id"].isin(ids))
    return chosen, table


# ------------------------------------------------------------------- the crops

#: The crops' PNG compression (``CROP_PNG_LEVEL`` overrides, 0-9). They are
#: read back by the colourise, the mesh and the check on the same machine and
#: never published, and PNG is lossless at every level: uncompressed, a
#: 24,000 m² warehouse's 16 crops of 4096 px were written in 2.0 s instead of
#: 11.0 s and read back in 2.1 s instead of 3.5 s, for 806 MB of disk instead
#: of 312 MB (the laptop, September 2026).
CROP_PNG_LEVEL = 0


def crop_png_level() -> int:
    import os

    return int(os.environ.get("CROP_PNG_LEVEL", CROP_PNG_LEVEL))


def write_crops(cameras: Sequence[Any], readers: Dict[str, Any], volume, frame: LocalCoordinateFrame,
                building_dir: Path, coverage=None, padding_px: int = 120,
                max_crop_px: int = 4096) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """One square crop per camera, all the same size: the widest projection
    of the volume plus the padding, rounded up to 16 px, capped at
    ``max_crop_px`` (a tower a few hundred metres from a nadir camera can ask
    for more; the mask records what the sensor held). Writes the images, the
    masks, ``cameras_normalized.json`` and ``crop_manifest.json``; returns
    both records. The same loop ``validate_building.py`` ran inline."""
    from PIL import Image

    from kyfromabove_3d.crops import BuildingCropExtractor
    from kyfromabove_3d.sources import assess_frame_coverage

    if coverage is None:
        coverage = assess_frame_coverage(cameras, volume)
    ids = {c.frame_id for c in cameras}
    sel = coverage[coverage["frame_id"].isin(ids)]
    span = float(np.nanmax(np.maximum(sel["span_u_px"], sel["span_v_px"]))) if len(sel) else 512.0
    # 32 px over the table's span: the extractor measures the window from
    # its own projection and can ask for a pixel more than the table said
    # (1249 against a 1248 crop stopped two buildings). Truncation is
    # allowed either way -- the validity mask records what the sensor held.
    required = int(np.ceil(span)) + 2 * int(padding_px) + 32
    crop_size = int(min(max_crop_px, np.ceil(required / 16.0) * 16))
    truncating = True

    (building_dir / "images").mkdir(parents=True, exist_ok=True)
    (building_dir / "validity_masks").mkdir(parents=True, exist_ok=True)
    extractor = BuildingCropExtractor(frame)
    level = crop_png_level()
    normalized: Dict[str, Any] = {}
    manifest: List[Dict[str, Any]] = []
    for idx, cam in enumerate(cameras, start=1):
        rgb, mask, spec = extractor.extract_crop(camera=cam, reader=readers[cam.frame_id], volume_3d=volume,
                                                 padding_px=padding_px, fixed_square_size=crop_size,
                                                 allow_truncation=truncating)
        img_name, mask_name = f"frame_{idx:03d}.png", f"mask_{idx:03d}.png"
        Image.fromarray(rgb).save(building_dir / "images" / img_name, compress_level=level)
        Image.fromarray(mask).save(building_dir / "validity_masks" / mask_name, compress_level=level)
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
        manifest.append(spec.to_dict())
    (building_dir / "cameras_normalized.json").write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    (building_dir / "crop_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return normalized, manifest


# ------------------------------------------------------------- the whole thing

def neighbours_of(rec: Dict[str, Any], records: Sequence[Dict[str, Any]], within_m: float = 1.0) -> List[Dict[str, Any]]:
    """The footprints within ``within_m`` of this one (party walls, mostly)."""
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform
    from shapely.wkt import loads as wkt_loads

    to_ft = Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True).transform
    me = shp_transform(to_ft, wkt_loads(rec["wkt"])).buffer(within_m / METERS_PER_US_SURVEY_FOOT)
    out = []
    for r in records:
        if r["id"] == rec["id"]:
            continue
        g = shp_transform(to_ft, wkt_loads(r["wkt"]))
        if g.intersects(me):
            out.append({"gers": r["id"], "name": r.get("name"), "wkt": r["wkt"],
                        "lidar_height_m": r.get("lidar_height_m"), "lidar_ground_z": r.get("lidar_ground_z")})
    return out


def prepare_building(rec: Dict[str, Any], out_dir: Path, frames_gdf, index, records: Sequence[Dict[str, Any]] = (),
                     pad_ft: float = 90.0, max_per_head: int = 4, min_frames: int = 3,
                     fallback_ground_m: Optional[float] = None, log=print) -> Dict[str, Any]:
    """Write the working directory for one Overture record; return what was
    found (counts, heads, crop size, timings) for the run's manifest: the
    piece (:func:`prepare_piece`), then the frame crops (:func:`cut_crops`)."""
    t0 = time.time()
    result, chosen = prepare_piece(rec, out_dir, frames_gdf, index, records, pad_ft, max_per_head, min_frames,
                                   fallback_ground_m, log)
    if chosen is not None:
        cut_crops(chosen, out_dir, result, log)
    result["t_total"] = round(time.time() - t0, 1)
    return result


def prepare_piece(rec: Dict[str, Any], out_dir: Path, frames_gdf, index, records: Sequence[Dict[str, Any]] = (),
                  pad_ft: float = 90.0, max_per_head: int = 4, min_frames: int = 3,
                  fallback_ground_m: Optional[float] = None, log=print):
    """A building's *piece*: everything the building step needs that comes
    from the LiDAR and the frame indexes, without the pixels --
    ``coordinates.json``, ``footprint.json``, the LiDAR clip
    (``lidar_local.laz``, ``clip.json``), ``check_features.json`` and the
    frames chosen (``frame_coverage.csv``). Returns (result, chosen): chosen
    is what :func:`cut_crops` needs, or None when no frame sees it."""
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform
    from shapely.wkt import loads as wkt_loads

    from kyfromabove_3d.footprints import largest_ring, ring_local_m
    from kyfromabove_3d.lidar import extract_lidar_for_aois

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    to_ft = Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True).transform
    poly_ft = largest_ring(shp_transform(to_ft, wkt_loads(rec["wkt"])))
    ring_ft = np.array(poly_ft.exterior.coords)
    frame = building_frame(rec, ring_ft, fallback_ground_m)
    ox, oy, oz = frame.origin_source_xyz
    ring_m = ring_local_m(ring_ft, ox, oy)
    height_m = rec.get("lidar_height_m")

    frame.save_json(out_dir / "coordinates.json")
    neighbours = neighbours_of(rec, records) if records else []
    (out_dir / "footprint.json").write_text(json.dumps({
        "gers": rec["id"], "name": rec.get("name") or rec["id"], "wkt": rec["wkt"],
        **({"parent_gers": rec["parent_gers"], "split": rec.get("split")} if rec.get("parent_gers") else {}),
        "height_m": height_m, "lidar_median_m": rec.get("lidar_median_m"), "lidar_n": rec.get("lidar_n"),
        "ground_z_m": oz * METERS_PER_US_SURVEY_FOOT, "overture_height_m": rec.get("height"),
        "num_floors": rec.get("num_floors"), "origin_source_xyz": list(frame.origin_source_xyz),
        "neighbours": neighbours,
    }, indent=1), encoding="utf-8")

    # LiDAR. The clip is reused only when it was cut for the same box
    # about the same origin (clip.json): a re-drawn or split footprint, a
    # new padding or a new origin cuts it again.
    t = time.time()
    laz_path = out_dir / "lidar_local.laz"
    key_path = out_dir / "clip.json"
    aoi = aoi_bounds_ft(ring_ft, oz, height_m, pad_ft)
    key = {"aoi_ft": [round(v, 3) for v in aoi], "origin_source_xyz": [round(float(v), 3) for v in (ox, oy, oz)]}
    from shapely import contains_xy
    from shapely.geometry import Polygon
    cached = (laz_path.exists() and key_path.exists()
              and json.loads(key_path.read_text(encoding="utf-8")) == key)
    if cached:
        from .clouds import read_cloud
        d = read_cloud(laz_path)
        n_aoi, xy = len(d["x"]), np.column_stack([d["x"], d["y"]])
    else:
        clouds = extract_lidar_for_aois({rec["id"]: aoi}, frame, index, progress=False)
        lidar = clouds[rec["id"]]
        lidar.save_laz(laz_path, use_local_coords=True)
        key_path.write_text(json.dumps(key), encoding="utf-8")
        (out_dir / "lidar_local.ply").unlink(missing_ok=True)      # an older run's copy
        n_aoi, xy = len(lidar), lidar.xyz_local_m[:, :2]
    n_on = int(contains_xy(Polygon(ring_m), xy[:, 0], xy[:, 1]).sum())
    t_lidar = time.time() - t
    log(f"  lidar: {n_aoi:,} returns in the clip{' (reused)' if cached else ''}, "
        f"{n_on:,} on the footprint ({t_lidar:.1f}s)")

    volume = building_volume(ring_m, height_m)
    (out_dir / "check_features.json").write_text(json.dumps(
        {"features": [], "volume_local_m": volume.to_dict()}, indent=2), encoding="utf-8")

    # Frames
    t = time.time()
    cams, table = select_cameras(frames_gdf, ring_ft, frame, volume, max_per_head, min_frames)
    heads = sorted({c.frame_id.rsplit("/", 1)[-1].split("_")[0] for c in cams})
    t_frames = time.time() - t
    log(f"  frames: {len(cams)} chosen ({', '.join(heads) or 'none'}) of {0 if table is None else len(table)} candidates ({t_frames:.1f}s)")
    if table is not None:
        table.to_csv(out_dir / "frame_coverage.csv", index=False)
    result = {"n_returns_aoi": n_aoi, "n_on_footprint": n_on, "n_frames": len(cams), "heads": heads,
              "crop_px": 0, "t_lidar": round(t_lidar, 1), "t_frames": round(t_frames, 1), "t_crops": 0.0}
    result["t_total"] = round(time.time() - t0, 1)
    if not cams:
        (out_dir / "cameras_normalized.json").write_text("{}", encoding="utf-8")
        return result, None
    return result, {"cams": cams, "volume": volume, "frame": frame, "table": table}


def cut_crops(chosen: Dict[str, Any], out_dir: Path, result: Dict[str, Any], log=print) -> None:
    """The frame crops of the cameras :func:`prepare_piece` chose, into
    ``out_dir`` (``images/``, ``validity_masks/``, ``cameras_normalized.json``)."""
    from kyfromabove_3d.imagery import readers_for_cameras

    t = time.time()
    readers = readers_for_cameras(chosen["cams"])
    normalized, _ = write_crops(chosen["cams"], readers, chosen["volume"], chosen["frame"], Path(out_dir),
                                coverage=chosen["table"])
    result["crop_px"] = int(next(iter(normalized.values()))["width"])
    result["t_crops"] = round(time.time() - t, 1)
    log(f"  crops: {len(normalized)} at {result['crop_px']} px ({result['t_crops']:.1f}s)")
