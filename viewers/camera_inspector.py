"""3D Interactive / Matplotlib Camera Inspector for local building AOI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import yaml

from kyfromabove_3d.cameras import VirtualObliqueCamera, parse_camera_from_eo_record
from kyfromabove_3d.coordinates import LocalCoordinateFrame
from kyfromabove_3d.crops import BoundingVolume3D
from kyfromabove_3d.lidar import extract_lidar_aoi
from kyfromabove_3d.sources import discover_candidate_frames


def plot_3d_scene(
    lidar_pts_local_m: np.ndarray,
    cameras: List[VirtualObliqueCamera],
    volume_3d: BoundingVolume3D,
    save_path: Path = None,
) -> None:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Decimate lidar for plotting
    if len(lidar_pts_local_m) > 5000:
        step = len(lidar_pts_local_m) // 5000
        pts = lidar_pts_local_m[::step]
    else:
        pts = lidar_pts_local_m

    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap="turbo", s=1, alpha=0.5)
    fig.colorbar(sc, ax=ax, label="Local Z (m)", shrink=0.6)

    # Plot 3D Bounding Box
    c8 = volume_3d.get_8_corners()
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
        [0, 4], [1, 5], [2, 6], [3, 7],  # Verticals
    ]
    for e in edges:
        ax.plot(c8[e, 0], c8[e, 1], c8[e, 2], "r-", linewidth=1.5)

    # Plot camera frusta
    for idx, cam in enumerate(cameras):
        c_pos = cam.camera_center_local_m
        ax.scatter([c_pos[0]], [c_pos[1]], [c_pos[2]], color="blue", s=40)
        # Frustum rays
        ray_dir = cam.R_wc @ np.array([0, 0, 1.0])
        look_target = c_pos + ray_dir * 1500.0
        ax.plot([c_pos[0], look_target[0]], [c_pos[1], look_target[1]], [c_pos[2], look_target[2]], "b--", alpha=0.5)
        ax.text(c_pos[0], c_pos[1], c_pos[2] + 20, f"Cam {idx+1}", color="darkblue", fontsize=9)

    ax.set_xlabel("Local X (East, m)")
    ax.set_ylabel("Local Y (North, m)")
    ax.set_zlabel("Local Z (Up, m)")
    ax.set_title("3D Scene: Building Volume, LiDAR, and Camera Frusta")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved 3D inspection view to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="3D Camera and LiDAR Inspector")
    parser.add_argument("--config", default="configs/phase1.yaml", help="Path to config file")
    parser.add_argument("--output", default="outputs/phase1/qa/scene_3d.png", help="Output figure path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    origin_xyz = tuple(cfg["coordinates"]["local_origin_source_xyz"])
    coord_frame = LocalCoordinateFrame(origin_source_xyz=origin_xyz)

    raw_dir = Path(cfg["paths"]["raw"])
    eo_gpkg = raw_dir / "KY_KYAPED_2022_Season2_3IN_ImageFrameEO.gpkg"
    frames_gpkg = raw_dir / "KY_KYAPED_2022_Season2_3IN_ImageFrames.gpkg"
    laz_file = raw_dir / "N067E390.laz"

    candidate_df = discover_candidate_frames(
        eo_gpkg_path=eo_gpkg,
        frames_gpkg_path=frames_gpkg,
        aoi_point_or_bounds=(origin_xyz[0], origin_xyz[1]),
        flight_line=cfg["selection"]["flight_id"],
        camera_direction=cfg["selection"]["source_direction_label"],
        max_frames=cfg["selection"]["target_frame_count"],
    )

    cameras = [parse_camera_from_eo_record(dict(row), coord_frame) for _, row in candidate_df.iterrows()]

    aoi_bounds_ft = (
        cfg["aoi"]["bounds"][0],
        cfg["aoi"]["bounds"][2],
        cfg["aoi"]["bounds"][1],
        cfg["aoi"]["bounds"][3],
        550.0,
        750.0,
    )
    lidar_cloud = extract_lidar_aoi(laz_file, coord_frame, aoi_bounds_ft)

    volume_3d = BoundingVolume3D(
        min_x_m=-40.0, max_x_m=40.0, min_y_m=-40.0, max_y_m=40.0, min_z_m=-5.0, max_z_m=15.0
    )

    plot_3d_scene(lidar_cloud.xyz_local_m, cameras, volume_3d, Path(args.output))


if __name__ == "__main__":
    main()
