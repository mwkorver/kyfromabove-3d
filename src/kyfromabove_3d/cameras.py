"""Virtual oblique camera model and photogrammetric projections."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from .coordinates import LocalCoordinateFrame, us_survey_feet_to_meters


def rot_x(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rot_y(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rot_z(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


# OpenCV camera axes (X right, Y down, Z forward) relative to the photogrammetric
# photo frame (X right, Y up, Z out of the image toward the viewer).
_PHOTO_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


def photogrammetric_omega_phi_kappa_to_rcw(
    omega_deg: float, phi_deg: float, kappa_deg: float
) -> np.ndarray:
    """Convert aerial photogrammetric Omega, Phi, Kappa (deg) to OpenCV ``R_cw``.

    In the standard aerial convention the photo-to-ground (image-to-object)
    rotation is applied X, then Y, then Z::

        M_photo_to_ground = Rx(omega) @ Ry(phi) @ Rz(kappa)

    so the ground-to-photo rotation is its transpose, and the OpenCV
    world-to-camera matrix flips Y and Z::

        R_cw = diag(1, -1, -1) @ M_photo_to_ground.T
             = diag(1, -1, -1) @ Rz(-kappa) @ Ry(-phi) @ Rx(-omega)

    .. note::
       This previously read ``Rx(-omega) @ Ry(-phi) @ Rz(-kappa)``, which is the
       transpose of ``Rz(kappa) Ry(phi) Rx(omega)`` -- the same three rotations
       composed in the wrong order. Because omega is ~45 degrees for these
       obliques, the mistake is not small: measured against the imagery it put
       consecutive exposures 26 px (about 1.7 m on the ground) out of
       registration with each other. The correct order measures 0.9 px.

       Nothing in a self-consistency test can catch this. It was found by
       :func:`kyfromabove_3d.qa.measure_cross_view_residuals`, which locates the same
       physical surface in two exposures by correlation and so lets the imagery,
       rather than the model, decide whether the model is right.
    """
    w = np.radians(float(omega_deg))
    p = np.radians(float(phi_deg))
    k = np.radians(float(kappa_deg))

    m_photo_to_ground = rot_x(w) @ rot_y(p) @ rot_z(k)
    return (_PHOTO_TO_OPENCV @ m_photo_to_ground.T).astype(np.float64)


@dataclass
class VirtualObliqueCamera:
    """Virtual camera model for Kentucky From Above oblique imagery.
    
    Represents an exposure with known exterior orientation (C_local, R_cw)
    and interior orientation (fx, fy, cx, cy, distortion coefficients).
    """

    frame_id: str
    camera_name: str
    camera_center_local_m: np.ndarray  # (3,) in local metric frame
    R_cw: np.ndarray  # (3, 3) world-to-camera rotation matrix
    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    distortion_coeffs: np.ndarray = None  # (k1, k2, p1, p2, k3)
    s3_url: Optional[str] = None
    flight_line: Optional[str] = None
    shot_id: Optional[str] = None
    flight_date: Optional[str] = None
    flight_time: Optional[str] = None
    source_c_xyz_feet: Optional[Tuple[float, float, float]] = None
    source_omega_phi_kappa: Optional[Tuple[float, float, float]] = None

    def __post_init__(self):
        self.camera_center_local_m = np.asarray(self.camera_center_local_m, dtype=np.float64).reshape(3)
        self.R_cw = np.asarray(self.R_cw, dtype=np.float64).reshape(3, 3)
        if self.distortion_coeffs is None:
            self.distortion_coeffs = np.zeros(5, dtype=np.float64)
        else:
            self.distortion_coeffs = np.asarray(self.distortion_coeffs, dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        """3x3 Camera intrinsic matrix."""
        return np.array(
            [
                [self.fx_px, 0.0, self.cx_px],
                [0.0, self.fy_px, self.cy_px],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def R_wc(self) -> np.ndarray:
        """Camera-to-world rotation matrix (transpose of R_cw)."""
        return self.R_cw.T

    def project(
        self, xyz_local_m: np.ndarray, distorted: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project local 3D points (meters) into image pixels.
        
        Args:
            xyz_local_m: (N, 3) or (3,) array of points in local metric frame.
            distorted: Whether to apply lens distortion coefficients.
            
        Returns:
            uv: (N, 2) image pixel coordinates.
            depth_zc: (N,) depth along camera optical axis.
            valid: (N,) boolean mask indicating points in front of camera (zc > 0)
                   and within image bounds [0, width] x [0, height].
        """
        pts = np.asarray(xyz_local_m, dtype=np.float64)
        is_1d = pts.ndim == 1
        if is_1d:
            pts = pts.reshape(1, 3)

        dX = pts - self.camera_center_local_m
        pts_c = (self.R_cw @ dX.T).T  # (N, 3)

        xc = pts_c[:, 0]
        yc = pts_c[:, 1]
        zc = pts_c[:, 2]

        valid_depth = zc > 0.1
        safe_zc = np.where(valid_depth, zc, 1.0)

        xn = xc / safe_zc
        yn = yc / safe_zc

        if distorted and np.any(np.abs(self.distortion_coeffs) > 1e-9):
            r2 = xn * xn + yn * yn
            k1, k2, p1, p2, k3 = self.distortion_coeffs[:5]
            radial = 1.0 + k1 * r2 + k2 * (r2 * r2) + k3 * (r2 * r2 * r2)
            xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
            yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        else:
            xd = xn
            yd = yn

        u = self.fx_px * xd + self.cx_px
        v = self.fy_px * yd + self.cy_px

        uv = np.column_stack([u, v])

        in_bounds = (
            (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= float(self.width_px))
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= float(self.height_px))
        )
        valid = valid_depth & in_bounds

        if is_1d:
            return uv[0], zc[0], valid[0]
        return uv, zc, valid

    def unproject(
        self, uv: np.ndarray, distorted: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Unproject 2D image coordinates to 3D unit rays in the local world frame.
        
        Args:
            uv: (N, 2) or (2,) pixel coordinates.
            distorted: Whether input uv coordinates contain lens distortion.
            
        Returns:
            cam_center: (3,) camera origin in local world coordinates.
            ray_dirs: (N, 3) unit direction vectors in local world coordinates.
        """
        pts = np.asarray(uv, dtype=np.float64)
        is_1d = pts.ndim == 1
        if is_1d:
            pts = pts.reshape(1, 2)

        if distorted and np.any(np.abs(self.distortion_coeffs) > 1e-9):
            # Undistort normalized points with OpenCV
            pts_norm = cv2.undistortPoints(
                pts.reshape(-1, 1, 2), self.K, self.distortion_coeffs
            ).reshape(-1, 2)
            xn = pts_norm[:, 0]
            yn = pts_norm[:, 1]
        else:
            xn = (pts[:, 0] - self.cx_px) / self.fx_px
            yn = (pts[:, 1] - self.cy_px) / self.fy_px

        # Camera ray directions (xn, yn, 1.0)
        rays_c = np.column_stack([xn, yn, np.ones_like(xn)])
        norm = np.linalg.norm(rays_c, axis=1, keepdims=True)
        rays_c = rays_c / norm

        # Rotate to local world frame: d_w = R_wc @ d_c = R_cw.T @ d_c
        rays_w = (self.R_wc @ rays_c.T).T

        if is_1d:
            return self.camera_center_local_m.copy(), rays_w[0]
        return self.camera_center_local_m.copy(), rays_w

    def to_dict(self) -> Dict[str, Any]:
        """Serialize camera parameters to a JSON-serializable dictionary."""
        return {
            "frame_id": self.frame_id,
            "camera_name": self.camera_name,
            "camera_center_local_m": self.camera_center_local_m.tolist(),
            "R_cw": self.R_cw.tolist(),
            "width_px": int(self.width_px),
            "height_px": int(self.height_px),
            "fx_px": float(self.fx_px),
            "fy_px": float(self.fy_px),
            "cx_px": float(self.cx_px),
            "cy_px": float(self.cy_px),
            "distortion_coeffs": self.distortion_coeffs.tolist(),
            "s3_url": self.s3_url,
            "flight_line": self.flight_line,
            "shot_id": self.shot_id,
            "flight_date": self.flight_date,
            "flight_time": self.flight_time,
            "source_c_xyz_feet": list(self.source_c_xyz_feet) if self.source_c_xyz_feet else None,
            "source_omega_phi_kappa": list(self.source_omega_phi_kappa) if self.source_omega_phi_kappa else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> VirtualObliqueCamera:
        return cls(
            frame_id=d["frame_id"],
            camera_name=d["camera_name"],
            camera_center_local_m=np.array(d["camera_center_local_m"], dtype=np.float64),
            R_cw=np.array(d["R_cw"], dtype=np.float64),
            width_px=int(d["width_px"]),
            height_px=int(d["height_px"]),
            fx_px=float(d["fx_px"]),
            fy_px=float(d["fy_px"]),
            cx_px=float(d["cx_px"]),
            cy_px=float(d["cy_px"]),
            distortion_coeffs=np.array(d.get("distortion_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64),
            s3_url=d.get("s3_url"),
            flight_line=d.get("flight_line"),
            shot_id=d.get("shot_id"),
            flight_date=d.get("flight_date"),
            flight_time=d.get("flight_time"),
            source_c_xyz_feet=tuple(d["source_c_xyz_feet"]) if d.get("source_c_xyz_feet") else None,
            source_omega_phi_kappa=tuple(d["source_omega_phi_kappa"]) if d.get("source_omega_phi_kappa") else None,
        )


def parse_camera_from_eo_record(
    record: Dict[str, Any], coord_frame: LocalCoordinateFrame
) -> VirtualObliqueCamera:
    """Parse exterior and interior orientation from GPKG record into VirtualObliqueCamera.
    
    Args:
        record: Dictionary or Series with keys:
            ID, X, Y, Z, Omega, Phi, Kappa, CamName,
            CamWidthPx, CamHeighPx, CamFocalMm, CamCCDResU,
            CamPpxMm, CamPpyMm, S3URL (optional), etc.
        coord_frame: LocalCoordinateFrame to convert X,Y,Z into local meters.
    """
    frame_id = str(record["ID"])
    cam_name = str(record.get("CamName", "ObliqueCam"))

    # Source camera coordinates in EPSG:3089 feet
    c_source_ft = (float(record["X"]), float(record["Y"]), float(record["Z"]))
    c_local_m = coord_frame.to_local(c_source_ft)

    # Orientation angles in degrees
    omega_deg = float(record["Omega"])
    phi_deg = float(record["Phi"])
    kappa_deg = float(record["Kappa"])

    R_cw = photogrammetric_omega_phi_kappa_to_rcw(omega_deg, phi_deg, kappa_deg)

    # Sensor interior orientation
    width_px = int(record["CamWidthPx"])
    height_px = int(record["CamHeighPx"])
    focal_mm = float(record["CamFocalMm"])
    pitch_um = float(record["CamCCDResU"])  # micrometers
    pitch_mm = pitch_um * 1e-3  # mm

    fx_px = focal_mm / pitch_mm
    fy_px = focal_mm / pitch_mm

    # Principal point offset, in mm from the image centre in the *photo* frame,
    # where +y points up. Image rows increase downward, so the y offset is
    # subtracted. Getting this backwards moves the principal point by
    # 2 * 6.68 mm / 3.76 um = 3552 px on the Left and Right cameras.
    #
    # The sign was determined by measurement, not assumption: matching a Left
    # and a Right exposure against a validated Fwd exposure accepts 0 and 6
    # correspondences respectively with `+`, and 222 and 213 with `-`. The Bwd
    # camera carries no offset and is unaffected either way, which makes it a
    # clean control. See kyfromabove_3d.qa.measure_cross_view_residuals.
    ppx_mm = float(record.get("CamPpxMm", 0.0) or 0.0)
    ppy_mm = float(record.get("CamPpyMm", 0.0) or 0.0)

    cx_px = (width_px / 2.0) + (ppx_mm / pitch_mm)
    cy_px = (height_px / 2.0) - (ppy_mm / pitch_mm)

    return VirtualObliqueCamera(
        frame_id=frame_id,
        camera_name=cam_name,
        camera_center_local_m=c_local_m,
        R_cw=R_cw,
        width_px=width_px,
        height_px=height_px,
        fx_px=fx_px,
        fy_px=fy_px,
        cx_px=cx_px,
        cy_px=cy_px,
        s3_url=record.get("S3URL"),
        flight_line=str(record.get("FL", "")),
        shot_id=str(record.get("ShotID", "")),
        flight_date=str(record.get("FlightDate", "")),
        flight_time=str(record.get("FlightTime", "")),
        source_c_xyz_feet=c_source_ft,
        source_omega_phi_kappa=(omega_deg, phi_deg, kappa_deg),
    )
