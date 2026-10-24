"""Data discovery, provenance tracking, and S3 ingestion for Kentucky From Above assets."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, box


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Compute SHA-256 checksum of a local file in 1MB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class AssetRecord:
    """Provenance and verification record for a source asset."""

    asset_id: str
    source_url_or_key: str
    product_identifier: str
    retrieval_date: str
    acquisition_date: Optional[str] = None
    size_bytes: int = 0
    etag: Optional[str] = None
    sha256: Optional[str] = None
    crs: Optional[str] = None
    vertical_datum: Optional[str] = None
    lineage_notes: Optional[str] = None
    local_path: Optional[str] = None


@dataclass
class SourceManifest:
    """Container for project source assets and metadata findings."""

    project_name: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    assets: Dict[str, AssetRecord] = field(default_factory=dict)
    metadata_notes: Dict[str, Any] = field(default_factory=dict)

    def add_asset(self, asset: AssetRecord) -> None:
        self.assets[asset.asset_id] = asset

    def save_json(self, filepath: Union[str, Path]) -> None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "project_name": self.project_name,
            "created_at": self.created_at,
            "assets": {k: asdict(v) for k, v in self.assets.items()},
            "metadata_notes": self.metadata_notes,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> SourceManifest:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        manifest = cls(
            project_name=data.get("project_name", "kyfromabove-3d"),
            created_at=data.get("created_at", ""),
            metadata_notes=data.get("metadata_notes", {}),
        )
        for k, v in data.get("assets", {}).items():
            manifest.assets[k] = AssetRecord(**v)
        return manifest


def download_s3_asset(
    s3_uri_or_url: str,
    local_dest: Union[str, Path],
    product_id: str = "kyfromabove_raw",
    crs: str = "EPSG:3089",
    vertical_datum: str = "NAVD88",
    acquisition_date: Optional[str] = None,
    lineage_notes: Optional[str] = None,
) -> AssetRecord:
    """Download an S3 asset (if not already local) and return its AssetRecord."""
    dest = Path(local_dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    s3_key = s3_uri_or_url
    if s3_key.startswith("https://kyfromabove.s3.amazonaws.com/"):
        s3_key = "s3://kyfromabove/" + s3_key.replace("https://kyfromabove.s3.amazonaws.com/", "")
    elif s3_key.startswith("https://kyfromabove.s3.us-west-2.amazonaws.com/"):
        s3_key = "s3://kyfromabove/" + s3_key.replace("https://kyfromabove.s3.us-west-2.amazonaws.com/", "")
    elif not s3_key.startswith("s3://"):
        s3_key = "s3://kyfromabove/" + s3_key.lstrip("/")

    if not dest.exists() or dest.stat().st_size == 0:
        cmd = ["aws", "s3", "cp", "--no-sign-request", s3_key, str(dest)]
        subprocess.run(cmd, check=True, capture_output=True)

    size = dest.stat().st_size
    sha256 = compute_file_sha256(dest)
    now_iso = datetime.now(timezone.utc).isoformat()

    return AssetRecord(
        asset_id=dest.stem,
        source_url_or_key=s3_uri_or_url,
        product_identifier=product_id,
        retrieval_date=now_iso,
        acquisition_date=acquisition_date,
        size_bytes=size,
        sha256=sha256,
        crs=crs,
        vertical_datum=vertical_datum,
        lineage_notes=lineage_notes or f"Fetched from {s3_uri_or_url}",
        local_path=str(dest),
    )


def discover_candidate_frames(
    eo_gpkg_path: Union[str, Path],
    frames_gpkg_path: Union[str, Path],
    aoi_point_or_bounds: Union[Tuple[float, float], Tuple[float, float, float, float], Polygon],
    flight_line: Optional[Union[str, Sequence[str]]] = None,
    camera_direction: Union[str, Sequence[str]] = "Fwd",
    max_frames: Optional[int] = None,
    require_containment: bool = True,
    eo_search_pad_ft: float = 12000.0,
) -> pd.DataFrame:
    """Find and rank candidate oblique exposures covering the given AOI.
    
    Args:
        eo_gpkg_path: Path to ImageFrameEO.gpkg
        frames_gpkg_path: Path to ImageFrames.gpkg
        aoi_point_or_bounds: Either (x, y) point, (minx, miny, maxx, maxy) bbox, or Polygon
        flight_line: Optional flight line filter; a single line ('2034') or a
            sequence of them. None accepts every line, which is what admits the
            side-looking cameras -- they fly on different lines from the Fwd
            exposures over the same ground.
        camera_direction: Camera identifier(s): 'Fwd', 'Bwd', 'Left', 'Right',
            'Color', or a sequence of them.
        max_frames: Maximum number of frames to return; None returns all.
        eo_search_pad_ft: Half-width of the box used to find EO records. The EO
            table is indexed on **camera position**, not on what the frame sees,
            so a box drawn around the AOI returns nothing -- the cameras are
            thousands of feet up and offset. 12000 ft comfortably covers the
            7709 ft maximum horizontal offset measured over downtown Louisville.
        require_containment: When the AOI is an extent (not a bare point), keep
            only exposures whose footprint fully contains it. Testing a single
            point accepts frames that see the AOI centre but clip the building
            off the sensor edge, which is how Phase 1 shipped two truncated
            crops. Footprint containment is a coarse screen only --
            ``assess_frame_coverage`` does the exact per-camera test.

    Returns:
        pd.DataFrame with merged EO and frame metadata.
    """
    if isinstance(aoi_point_or_bounds, tuple):
        if len(aoi_point_or_bounds) == 2:
            query_geom = Point(aoi_point_or_bounds[0], aoi_point_or_bounds[1])
        else:
            query_geom = box(*aoi_point_or_bounds)
    else:
        query_geom = aoi_point_or_bounds

    # Read only the relevant slice of the frames index. This is not an
    # optimisation: KY_KYAPED_2024_Season1_3IN's index is 1.07 GB and is read
    # in place over /vsicurl, where a full read is not viable. A GeoPackage is
    # SQLite with an RTree, so a bbox is served by range request.
    qminx, qminy, qmaxx, qmaxy = query_geom.bounds
    frames_gdf = gpd.read_file(
        str(frames_gpkg_path),
        bbox=(qminx - eo_search_pad_ft, qminy - eo_search_pad_ft,
              qmaxx + eo_search_pad_ft, qmaxy + eo_search_pad_ft),
    )

    # Filter by geometry. Containment is the meaningful test for an extent.
    if require_containment and not isinstance(query_geom, Point):
        covering = frames_gdf[frames_gdf.contains(query_geom)].copy()
        if len(covering) == 0:
            covering = frames_gdf[frames_gdf.intersects(query_geom)].copy()
    else:
        covering = frames_gdf[frames_gdf.intersects(query_geom)].copy()
    
    # Filter by camera direction
    directions = [camera_direction] if isinstance(camera_direction, str) else list(camera_direction)
    covering = covering[covering["CameraID"].isin(directions)]

    # Filter by flight line if specified
    if flight_line is not None:
        lines = [flight_line] if isinstance(flight_line, (str, int)) else list(flight_line)
        covering = covering[covering["FL"].astype(str).isin([str(x) for x in lines])]

    
    if len(covering) == 0:
        return pd.DataFrame()
        
    # Read EO spatially rather than by an attribute WHERE. A bbox uses the
    # RTree; `WHERE ID IN (...)` does not, and against a remote 553 MB table it
    # scans the whole file over HTTP and does not finish.
    #
    # The box is drawn around CAMERA positions, which is why it is padded so
    # far: over downtown Louisville the cameras sit 5,400 ft up and as much as
    # 7,709 ft away horizontally, so a box around the building returns zero.
    cminx, cminy, cmaxx, cmaxy = covering.total_bounds
    eo_df = gpd.read_file(
        str(eo_gpkg_path),
        bbox=(cminx - eo_search_pad_ft, cminy - eo_search_pad_ft,
              cmaxx + eo_search_pad_ft, cmaxy + eo_search_pad_ft),
    )
    if "geometry" in eo_df.columns:
        eo_df = pd.DataFrame(eo_df.drop(columns="geometry"))

    # IDs keep the collection prefix: 'KY_KYAPED_2024_Season1_3IN/Bwd_7021_47003'.
    # Matching on the bare shot id joins nothing, and does so silently.
    covering["ID"] = covering["Filename"].str.replace(".tif", "", regex=False)
    eo_df = eo_df[eo_df["ID"].isin(set(covering["ID"]))]
    merged = pd.merge(covering, eo_df, on="ID", suffixes=("_frame", ""))
    
    # Sort by camera then ShotID so ordering is stable across directions.
    merged = merged.sort_values(["CameraID", "ShotID"])
    if max_frames is not None:
        merged = merged.head(max_frames)
    return merged


def assess_frame_coverage(cameras: Sequence[Any], volume_3d: Any) -> pd.DataFrame:
    """Test, per camera, whether the whole building volume lands on the sensor.

    The GeoPackage footprint test is 2D and ground-referenced, so it cannot tell
    you that a 7 m roof projects off the top of the frame. This projects all
    eight volume corners and reports what actually fits.
    """
    corners = volume_3d.get_8_corners()
    rows = []
    for cam in cameras:
        uv, zc, _ = cam.project(corners)
        in_front = zc > 0.1
        inside = (
            in_front
            & (uv[:, 0] >= 0)
            & (uv[:, 0] <= cam.width_px)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] <= cam.height_px)
        )
        vis = uv[in_front]
        u0, u1 = (vis[:, 0].min(), vis[:, 0].max()) if len(vis) else (np.nan, np.nan)
        v0, v1 = (vis[:, 1].min(), vis[:, 1].max()) if len(vis) else (np.nan, np.nan)
        rows.append(
            {
                "frame_id": cam.frame_id,
                "shot_id": cam.shot_id,
                "corners_on_sensor": int(inside.sum()),
                "corners_total": len(corners),
                "fully_contained": bool(inside.all()),
                "u_min": float(u0),
                "u_max": float(u1),
                "v_min": float(v0),
                "v_max": float(v1),
                "span_u_px": float(u1 - u0),
                "span_v_px": float(v1 - v0),
                "margin_px": float(
                    min(u0, v0, cam.width_px - u1, cam.height_px - v1)
                ),
            }
        )
    return pd.DataFrame(rows)


def select_frames_by_coverage(
    cameras: Sequence[Any],
    volume_3d: Any,
    min_margin_px: float = 0.0,
    max_frames: Optional[int] = None,
) -> Tuple[List[Any], pd.DataFrame]:
    """Keep only exposures that fully contain the building volume.

    Returns ``(kept_cameras, coverage_table)``; the table retains the rejected
    rows so the caller can report what was dropped and why.
    """
    table = assess_frame_coverage(cameras, volume_3d)
    keep_mask = table["fully_contained"] & (table["margin_px"] >= min_margin_px)
    table = table.assign(selected=keep_mask)
    kept = [cam for cam, ok in zip(cameras, keep_mask.tolist()) if ok]
    if max_frames is not None:
        kept = kept[:max_frames]
        selected_ids = {c.frame_id for c in kept}
        table["selected"] = table["frame_id"].isin(selected_ids)
    return kept, table


def build_source_manifest(
    project_name: str,
    assets: Sequence[Tuple[Union[str, Path], str, str, str]],
    crs: str = "EPSG:3089",
    vertical_datum: str = "NAVD88",
    metadata_notes: Optional[Dict[str, Any]] = None,
    root: Optional[Union[str, Path]] = None,
) -> SourceManifest:
    """Register every asset with size and SHA-256 in one place.

    ``assets`` items are ``(local_path, source_url, product_id, notes)``. The
    previous notebook built a manifest, wrote a ``# Add primary source assets``
    comment, added nothing, and shipped ``"assets": {}`` behind a provenance
    PASS. Having one function do the registration removes the chance to forget.

    ``root`` makes each recorded ``local_path`` relative to it. The manifest is a
    committed provenance record, so an absolute path would bake one machine's
    directory layout into the repository and go stale the moment anyone clones
    it or renames the checkout.
    """
    manifest = SourceManifest(project_name=project_name)
    for local_path, url, product_id, notes in assets:
        record = download_s3_asset(
            s3_uri_or_url=url,
            local_dest=local_path,
            product_id=product_id,
            crs=crs,
            vertical_datum=vertical_datum,
            lineage_notes=notes,
        )
        if root is not None:
            try:
                record.local_path = str(Path(record.local_path).relative_to(Path(root)))
            except ValueError:
                pass  # asset lives outside the project; keep the absolute path
        manifest.add_asset(record)
    if metadata_notes:
        manifest.metadata_notes = dict(metadata_notes)
    return manifest
