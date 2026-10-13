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

    python scripts/colorize_building.py --building outputs/urban/building_002 \
        --case building_002
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
    facade_faces,
    split_surface_zones,
    wall_planes,
    verify_face_plane,
    infer_prism_faces,
    save_wall_planes,
    WallPlane,
    VerifiedWallPlane,
    rectify_footprint_corners,
)
from kyfromabove_3d.clouds import find_cloud, read_cloud
from kyfromabove_3d.project import repo_relative

ROOT = Path(__file__).resolve().parent.parent
#: every downtown footprint with its LiDAR ground and height: the neighbours
FOOTPRINTS = ROOT / "data" / "derived" / "louisville_downtown_overture_split.json"


def load_cameras(building: Path):
    """The aerial frames' camera models plus their crops, in the building's
    local metric frame. (Each carries ``is_photo`` False: the colour
    sampling, the coverage counts and the cloud's ``source`` property still
    have a slot for photographs, from when ground photographs were tried;
    only KyFromAbove's frames are used now.)"""
    from PIL import Image

    meta = json.loads((building / "cameras_normalized.json").read_text(encoding="utf-8"))
    cams = []
    for frame_id, c in meta.items():
        img_path = building / c["crop_image"]
        if not img_path.exists():
            continue
        img = np.asarray(Image.open(img_path).convert("RGB"))
        mask = None
        mpath = building / c.get("validity_mask", "")
        if c.get("validity_mask") and mpath.exists():
            mask = np.asarray(Image.open(mpath).convert("L")) > 127
        cams.append({
            "id": frame_id,
            "head": frame_id.rsplit("/", 1)[-1].split("_")[0],
            "K": np.array([[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]],
                           [0.0, 0.0, 1.0]]),
            "R_cw": np.array(c["R_cw"], dtype=np.float64),
            "C": np.array(c["camera_center_local_m"], dtype=np.float64),
            "img": img, "mask": mask,
            "wh": (int(c["width"]), int(c["height"])),
            "is_photo": False,
        })

    return cams


def scene_cloud(origin_ft, half_width_m, raw_dir=Path("data/raw")):
    """A wider LiDAR extent than the AOI, purely to occlude with.

    The AOI is drawn tight around the target so the QA measures the building and
    not its neighbours. That is right for everything else and wrong here: a
    depth map built from the AOI alone knows nothing about the tower across the
    street, so its facade gets painted onto this one's wall. On PNC Tower the
    podium level came out wearing the buildings opposite until this went in.
    """
    import laspy

    from kyfromabove_3d.lidar import LidarTileIndex, open_copc

    ox, oy, _ = origin_ft
    pad = half_width_m / METERS_PER_US_SURVEY_FOOT
    bounds = (ox - pad, oy - pad, ox + pad, oy + pad)
    tiles = LidarTileIndex(raw_dir).tiles_for_bounds(bounds, phase="phase2")
    out = []
    for t in tiles:
        try:
            reader = open_copc(t.url)
            hdr = reader.header
            pts = reader.query(laspy.copc.Bounds(
                mins=np.array([bounds[0], bounds[1], hdr.z_min]),
                maxs=np.array([bounds[2], bounds[3], hdr.z_max])))
        except Exception:                                   # noqa: BLE001
            continue
        cls = np.asarray(pts.classification)
        xyz = np.stack([pts.x, pts.y, pts.z], axis=1)[cls != 7]
        out.append(xyz)
    if not out:
        return None
    return np.vstack(out)


def scene_cloud_from_cache(path, origin_ft, half_width_m):
    """The occluder cloud from a cache written once for a whole area
    (``run_downtown.py`` does), windowed to ``half_width_m`` around this
    building's origin. Returned in source feet like :func:`scene_cloud`.
    A run of 375 buildings would otherwise range-read the same 250 m of
    downtown 375 times."""
    d = np.load(path)
    xyz_ft = d["xyz_ft"]
    ox, oy, _ = origin_ft
    pad = half_width_m / METERS_PER_US_SURVEY_FOOT
    m = (np.abs(xyz_ft[:, 0] - ox) <= pad) & (np.abs(xyz_ft[:, 1] - oy) <= pad)
    return xyz_ft[m].astype(np.float64) if m.any() else None


def may_land_on_crop(cam, xyz32, w, h, margin_px=32.0):
    """Which points could land on camera ``cam``'s ``w`` x ``h`` crop:
    a conservative cull, in single precision, ``margin_px`` wider than the
    crop on every side (single precision is off by far less than a pixel
    here), and in front of the camera. A point it drops is one the exact
    projection would have dropped."""
    A = cam["K"] @ cam["R_cw"]
    b = -(A @ cam["C"])
    p = xyz32 @ A.T.astype(np.float32) + b.astype(np.float32)
    z = p[:, 2]
    front = z > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = p[:, 0] / z, p[:, 1] / z
    return front & (u >= -margin_px) & (u < w + margin_px) & (v >= -margin_px) & (v < h + margin_px)


def depth_buffers(cams, scene_xyz, downscale=8, dilate=3):
    """A coarse depth map per camera, built from every LiDAR return in the AOI.

    Facing the right way is not the same as being visible. Without this, a wall
    plane is coloured from whatever pixel its ray lands on -- and where a
    neighbouring building stands in front of it, that pixel belongs to the
    neighbour, which gets painted onto the wall. On PNC Tower the whole podium
    level came out wearing the buildings across the street.

    Coarse on purpose. The LiDAR is sparse, so a full-resolution depth map is
    mostly holes; at 1/8 scale a return covers enough of the image to occlude
    reliably, and the resample pitch is far finer than the error that costs.
    """
    xyz32 = np.asarray(scene_xyz, dtype=np.float32)          # for the cull only
    for cam in cams:
        w, h = cam["wh"]
        dw, dh = w // downscale + 1, h // downscale + 1
        buf = np.full((dh, dw), np.inf)
        # Of the ~840,000 returns within 250 m, about 2% land on a crop.
        # Drop the rest first, in single precision with a margin, then
        # project the survivors exactly as before.
        near = scene_xyz[may_land_on_crop(cam, xyz32, w, h)]
        pc = (cam["R_cw"] @ (near - cam["C"]).T).T
        ok = pc[:, 2] > 1e-6
        if ok.any():
            proj = (cam["K"] @ pc[ok].T).T
            u = (proj[:, 0] / proj[:, 2] / downscale).astype(int)
            v = (proj[:, 1] / proj[:, 2] / downscale).astype(int)
            z = pc[ok, 2]
            m = (u >= 0) & (u < dw) & (v >= 0) & (v < dh)
            np.minimum.at(buf, (v[m], u[m]), z[m])
        # Close the holes. A sparse cloud leaves gaps between returns even at
        # 1/8 scale, and a gap passes every ray -- which is how a neighbouring
        # tower kept printing through in speckle after the depth test went in.
        # A minimum filter spreads each occluder slightly, which errs towards
        # rejecting a sample rather than colouring it through something.
        from scipy.ndimage import minimum_filter

        cam["depth"] = minimum_filter(buf, size=dilate, mode="nearest")
        cam["downscale"] = downscale
    return cams


def neighbour_prisms(b: Path, footprints: Path | None = FOOTPRINTS, radius_m: float = 250.0,
                     buildings_dir: Path | None = ROOT / "outputs" / "downtown" / "buildings"):
    """Every other building within ``radius_m`` of building dir ``b``, as solid
    prisms ``(ring (N, 2) closed, z_lo, z_hi)`` in ``b``'s local metres.

    A neighbour the colourise has seen (``buildings_dir/<id>/levels.json``)
    gives one prism per level -- a podium to its own top, the tower's traced
    outline above it -- so a podium does not stand at its tower's height;
    a level is capped at the footprint's LiDAR height.
    Any other stands as its Overture footprint from the LiDAR ground to its
    LiDAR height (the 99th percentile of its returns). A footprint whose
    inside overlaps ``b``'s by more than a square metre (the building
    itself, a split part's parent) is left out; one that only touches it
    (a neighbour across a party wall, a sibling part) stays.
    """
    if not footprints or not Path(footprints).exists():
        return []
    from shapely.geometry import Point
    from kyfromabove_3d.footprints import footprint_wgs84_to_local

    FT = METERS_PER_US_SURVEY_FOOT
    ox, oy, oz = json.loads((b / "coordinates.json").read_text(encoding="utf-8"))["origin_source_xyz"]
    recs = json.loads(Path(footprints).read_text(encoding="utf-8"))
    recs = recs if isinstance(recs, list) else recs.get("buildings", recs)
    if (b / "footprint.json").exists():
        wkt = json.loads((b / "footprint.json").read_text(encoding="utf-8"))["wkt"]
    else:                                         # a landmark's directory: find it by its GERS id
        ts = b / "3dtiles_mesh" / "tileset.json"
        gers = json.loads(ts.read_text(encoding="utf-8"))["extras"].get("gers") if ts.exists() else None
        wkt = next((r["wkt"] for r in recs if r["id"] == gers), None)
        if wkt is None:
            print(f"[warn] {b.name} is not in {Path(footprints).name}: no neighbour prisms")
            return []
    me = footprint_wgs84_to_local(wkt, ox, oy)
    out = []
    for r in recs:
        poly = footprint_wgs84_to_local(r["wkt"], ox, oy)
        if poly.distance(Point(0.0, 0.0)) > radius_m or poly.intersection(me).area > 1.0:
            continue
        nb = Path(buildings_dir) / r["id"] if buildings_dir else None
        if nb is not None and (nb / "levels.json").exists() and (nb / "coordinates.json").exists():
            nx, ny, nz = json.loads((nb / "coordinates.json").read_text(encoding="utf-8"))["origin_source_xyz"]
            shift = np.array([(nx - ox) * FT, (ny - oy) * FT])
            dz = (nz - oz) * FT
            # never taller than the footprint's own 99th percentile: a levels.json
            # written before the spill cap (362 of 381 downtown, September 2026)
            # can hold the tall building next door -- cc3f65d1, an 18 m
            # building beside Humana, stood at 116 m and hid Humana's walls
            cap = (float(r["lidar_ground_z"]) - oz * FT + float(r["lidar_height_m"])) \
                if r.get("lidar_ground_z") is not None and r.get("lidar_height_m") else np.inf
            for lv in json.loads((nb / "levels.json").read_text(encoding="utf-8")):
                ring = np.asarray(lv["ring"], dtype=float)[:, :2] + shift
                if not np.allclose(ring[0], ring[-1]):
                    ring = np.vstack([ring, ring[:1]])
                z_lo, z_hi = float(lv["z_lo"]) + dz, min(float(lv["z_hi"]) + dz, cap)
                if z_hi > z_lo:
                    out.append((ring, z_lo, z_hi))
        elif r.get("lidar_ground_z") is not None and (r.get("lidar_height_m") or r.get("height")):
            z_lo = float(r["lidar_ground_z"]) - oz * FT
            out.append((np.asarray(poly.exterior.coords)[:, :2], z_lo,
                        z_lo + float(r.get("lidar_height_m") or r.get("height"))))
    return out


def levels_dir(arg):
    """``--neighbour-levels`` as a folder, or None for 'none'."""
    return None if arg in (None, "", "none") else Path(arg)


#: Where a building's occlusion maps are kept for the next script to use.
OCCLUSION_FILE = "occlusion.npz"


def occlusion_maps(b: Path, cams, occ_xyz, prisms, downscale=8, dilate=3, log=print):
    """Each camera's depth map (:func:`depth_buffers`, then the neighbour
    prisms, :func:`add_prism_occluders`), made once per building and shared
    by the colourise, the mesh and the check, which each made the same maps.

    Kept in ``b/occlusion.npz`` under a fingerprint of everything that goes
    into them: each camera (its id, size, K, R and centre), the returns
    themselves, every prism, and the settings. A script uses the saved maps
    only when its own inputs give the same fingerprint, so a changed
    camera, cloud, neighbour or setting makes them again. Returns ``cams``."""
    import hashlib
    import os

    h = hashlib.sha256(f"occlusion-v1 {downscale} {dilate} {len(cams)}".encode())
    for c in cams:
        h.update(str(c["id"]).encode())
        h.update(np.asarray(c["wh"], dtype=np.int64).tobytes())
        for k in ("K", "R_cw", "C"):
            h.update(np.ascontiguousarray(c[k], dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(occ_xyz, dtype=np.float64).tobytes())
    h.update(str(len(prisms)).encode())
    for ring, z_lo, z_hi in prisms:
        h.update(np.ascontiguousarray(ring, dtype=np.float64).tobytes())
        h.update(np.array([z_lo, z_hi], dtype=np.float64).tobytes())
    key = h.hexdigest()
    path = Path(b) / OCCLUSION_FILE
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as z:
                if str(z["key"]) == key:
                    for i, c in enumerate(cams):
                        c["depth"], c["downscale"] = z[f"d{i}"], downscale
                    log(f"  occlusion maps: the saved ones ({path.name}, same inputs)")
                    return cams
        except (OSError, KeyError, ValueError):
            pass
    depth_buffers(cams, occ_xyz, downscale, dilate)
    if prisms:
        add_prism_occluders(cams, prisms)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, key=np.array(key), **{f"d{i}": c["depth"] for i, c in enumerate(cams)})
    os.replace(tmp, path)                        # a reader never sees half a file
    return cams


def occluder_cloud(b: Path, scene_cloud: Path | None, radius_m: float, fallback_xyz: np.ndarray):
    """The returns that can stand between a frame and building dir ``b``, in
    its local metres: the area's cloud within ``radius_m`` when there is one,
    else ``fallback_xyz`` (the building's own clip, which reaches 27 m past
    the footprint and misses the tower across the street)."""
    FT = METERS_PER_US_SURVEY_FOOT
    ox, oy, oz = json.loads((b / "coordinates.json").read_text(encoding="utf-8"))["origin_source_xyz"]
    wider = scene_cloud_from_cache(scene_cloud, (ox, oy, oz), radius_m) \
        if scene_cloud and Path(scene_cloud).exists() else None
    if wider is None:
        print("[warn] no scene cloud: occluding with the building's own clip, which misses farther neighbours")
        return fallback_xyz
    return np.column_stack([(wider[:, 0] - ox) * FT, (wider[:, 1] - oy) * FT, (wider[:, 2] - oz) * FT])


def add_prism_occluders(cams, prisms):
    """Add solid neighbours to each camera's depth map (from depth_buffers).

    The LiDAR is flown from above: a tower's walls hold few returns, so seen
    from the side its facade is mostly empty in a depth map built from
    points, and a ray to a building behind it passes through. Humana, behind
    PNC Tower in a Right-looking frame, was drawn -- and checked -- through
    PNC's 150 m of glass. ``prisms`` are ``(ring (N, 2) closed, z_lo, z_hi)``
    in the building's local metres: each neighbour as its footprint at its
    LiDAR height. Each depth-map cell keeps the nearest of its LiDAR depth
    and the prisms' walls and roofs along its ray.
    """
    from shapely import contains_xy
    from shapely.geometry import Polygon

    for cam in cams:
        D, ds = cam["depth"], cam["downscale"]
        dh, dw = D.shape
        jj, ii = np.meshgrid(np.arange(dw), np.arange(dh))
        pix = np.stack([(jj + 0.5) * ds, (ii + 0.5) * ds, np.ones_like(jj, dtype=float)], axis=-1)
        rays = pix @ np.linalg.inv(cam["K"]).T @ cam["R_cw"]         # world directions, z_cam = 1 per unit t
        C = cam["C"]
        for ring, z_lo, z_hi in prisms:
            ring = np.asarray(ring, dtype=float)
            box = np.array([[x, y, z] for x, y in (ring.min(0), ring.max(0), [ring[0, 0], ring.max(0)[1]],
                                                     [ring.max(0)[0], ring[0, 1]]) for z in (z_lo, z_hi)])
            pc = (cam["R_cw"] @ (box - C).T).T
            if (pc[:, 2] <= 1e-6).all():
                continue
            front = pc[pc[:, 2] > 1e-6]
            pr = (cam["K"] @ front.T).T
            u, v = pr[:, 0] / pr[:, 2] / ds, pr[:, 1] / pr[:, 2] / ds
            if (pc[:, 2] > 1e-6).all():
                j0, j1 = max(int(u.min()) - 1, 0), min(int(u.max()) + 2, dw)
                i0, i1 = max(int(v.min()) - 1, 0), min(int(v.max()) + 2, dh)
            else:                                                 # straddles the camera plane: test it all
                j0, j1, i0, i1 = 0, dw, 0, dh
            if j0 >= j1 or i0 >= i1:
                continue
            R = rays[i0:i1, j0:j1].reshape(-1, 3)
            best = np.full(len(R), np.inf)
            for a, b in zip(ring[:-1], ring[1:]):
                e = b - a
                L = float(np.hypot(*e))
                if L < 1e-6:
                    continue
                n = np.array([e[1], -e[0], 0.0]) / L
                den = R @ n
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = (n[:2] @ a - n @ C) / den
                X = C[None, :] + t[:, None] * R
                along = ((X[:, :2] - a) @ e) / L
                hit = (t > 0) & (along >= 0) & (along <= L) & (X[:, 2] >= z_lo) & (X[:, 2] <= z_hi)
                best[hit] = np.minimum(best[hit], t[hit])
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (z_hi - C[2]) / R[:, 2]
            X = C[None, :] + t[:, None] * R
            up = (t > 0) & np.isfinite(t)
            if up.any():
                roof = np.zeros(len(R), dtype=bool)
                roof[up] = contains_xy(Polygon(ring), X[up, 0], X[up, 1])
                best[roof] = np.minimum(best[roof], t[roof])
            sub = D[i0:i1, j0:j1]
            D[i0:i1, j0:j1] = np.minimum(sub, best.reshape(sub.shape))
    return cams


def project_visible(xyz, normals, cam, min_facing_cos=0.15, occlusion_slack_m=1.5):
    """Which of the points ``cam`` sees, and where: ``(idx, u, v, facing)``
    for the points in front of their surface as seen from the camera
    (``facing``, the cosine between the normal and the ray to the camera,
    at least ``min_facing_cos``), inside the crop and its validity mask,
    and not behind anything in the camera's depth buffer by more than
    ``occlusion_slack_m``. ``u``, ``v`` are integer pixel indices."""
    empty = (np.zeros(0, dtype=int),) * 3 + (np.zeros(0),)
    d = cam["C"][None, :] - xyz                           # point -> camera
    rng = np.linalg.norm(d, axis=1)
    ok = rng > 1e-6
    facing = np.zeros(len(xyz))
    facing[ok] = np.einsum("ij,ij->i", normals[ok], d[ok]) / rng[ok]
    visible = ok & (facing >= min_facing_cos)
    if not visible.any():
        return empty
    cam_pts = (cam["R_cw"] @ (xyz[visible] - cam["C"]).T).T
    infront = cam_pts[:, 2] > 1e-6
    idx = np.flatnonzero(visible)[infront]
    if not len(idx):
        return empty
    # Depth along the optical axis, which is what the buffer stores. Using
    # Euclidean range here instead makes every point read as farther than
    # the buffer and rejects the whole cloud.
    zdepth = cam_pts[infront, 2]
    proj = (cam["K"] @ cam_pts[infront].T).T
    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]
    w, h = cam["wh"]
    inside = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    idx, ui, vi, zdepth = idx[inside], u[inside].astype(int), v[inside].astype(int), zdepth[inside]
    if len(idx) and cam["mask"] is not None:
        keep = cam["mask"][vi, ui]
        idx, ui, vi, zdepth = idx[keep], ui[keep], vi[keep], zdepth[keep]
    if len(idx) and cam.get("depth") is not None:
        ds = cam["downscale"]
        near = cam["depth"][vi // ds, ui // ds]
        # Something closer than this point, by more than the slack, is
        # between it and the camera.
        vis = ~np.isfinite(near) | (zdepth <= near + occlusion_slack_m)
        idx, ui, vi = idx[vis], ui[vis], vi[vis]
    return idx, ui, vi, facing[idx]


def sample_colour(xyz, normals, cams, min_facing_cos=0.15, occlusion_slack_m=1.5,
                  chunk_size=200_000):
    """Sample RGB from the best-facing, unoccluded camera for each point.

    Two tests, and both are needed. A point is only coloured from a camera that
    is **in front of the surface** -- a wall faces one way, and 70% of
    camera-anchor pairs on the tower are behind it. And it is only coloured if
    nothing else stands in the way, tested against a depth map built from every
    return in the AOI: facing the right way is not the same as being visible.

    Among the cameras that pass, the most face-on wins: an oblique looking along
    a wall spreads one pixel over a metre of it.
    """
    n = len(xyz)
    if n > chunk_size:
        rgb_list, seen_list, best_list, photo_list = [], [], [], []
        for i in range(0, n, chunk_size):
            r, s, b, p = sample_colour(
                xyz[i:i + chunk_size], normals[i:i + chunk_size],
                cams, min_facing_cos, occlusion_slack_m, chunk_size=chunk_size)
            rgb_list.append(r)
            seen_list.append(s)
            best_list.append(b)
            photo_list.append(p)
        return (np.vstack(rgb_list), np.concatenate(seen_list),
                np.concatenate(best_list), np.concatenate(photo_list))

    rgb = np.zeros((n, 3), dtype=np.uint8)
    best = np.full(n, -1.0)
    seen = np.zeros(n, dtype=np.int32)
    from_photo = np.zeros(n, dtype=bool)

    for cam in cams:
        idx, ui, vi, facing = project_visible(xyz, normals, cam, min_facing_cos, occlusion_slack_m)
        if not len(idx):
            continue
        seen[idx] += 1
        better = facing > best[idx]
        take = idx[better]
        rgb[take] = cam["img"][vi[better], ui[better]]
        best[take] = facing[better]
        from_photo[take] = bool(cam.get("is_photo", False))
    return rgb, seen, best, from_photo


def resample_walls(xyz, wall_mask, ring_m, height_m, spacing_m, min_face_m,
                   max_offset_m, tile_m=12.0, max_rms_m=0.35, min_points=40, skip_faces=()):
    """New points on locally fitted wall planes, at imagery resolution.

    Depth comes from the planes the LiDAR defines, never from the imagery.

    **Fitted per tile, not per face.** One plane across a whole 60 m wall is too
    coarse a model for a real building: measured here, all four of the Humana
    Building's faces fit a single plane at 0.58-1.22 m, because Graves put
    setbacks, a portico and a cantilevered belvedere on them, and one of PNC
    Tower's two large faces missed a 0.5 m cut by 0.064 m. Fitting ``tile_m``
    squares instead lets the surface step where the building steps, and the
    residual then measures the wall's own texture rather than its shape.

    A tile whose returns still do not lie on a plane is skipped rather than
    extrapolated. Where a facade is genuinely not planar, this method has
    nothing to offer and should decline.
    """
    faces = facade_faces(ring_m, height_m=height_m, wall_xy=xyz[wall_mask][:, :2],
                         min_length_m=min_face_m, max_offset_m=max_offset_m)
    # an edge of a curved wall is profiled with its arc, not fitted as a plane
    faces = [f for f in faces if f.index not in set(skip_faces)]
    wall_xyz = xyz[wall_mask]
    v = np.asarray(ring_m, dtype=np.float64)[:, :2]
    if np.allclose(v[0], v[-1]):
        v = v[:-1]
    if np.sum((np.roll(v[:, 0], -1) - v[:, 0]) * (np.roll(v[:, 1], -1) + v[:, 1])) > 0:
        v = v[::-1]

    out_xyz, out_n, out_face, stats = [], [], [], []
    tiles_by_face: dict[int, list[dict]] = {}
    for f in faces:
        if not f.captured:
            continue
        p0, p1 = v[f.index], v[(f.index + 1) % len(v)]
        seg = p1 - p0
        L = np.hypot(*seg)
        t = np.clip(((wall_xyz[:, :2] - p0) @ seg) / max(L ** 2, 1e-9), 0.0, 1.0)
        closest = p0 + t[:, None] * seg
        pts = wall_xyz[np.linalg.norm(wall_xyz[:, :2] - closest, axis=1) <= max_offset_m]
        if height_m is not None and height_m > 0:
            pts = pts[pts[:, 2] <= (pts[:, 2].min() + height_m + 1.0)]
        if len(pts) < min_points:
            continue

        # Nominal frame for the face: along the wall, and up it.
        e1 = np.array([seg[0] / L, seg[1] / L, 0.0])
        e2 = np.array([0.0, 0.0, 1.0])
        origin = np.array([p0[0], p0[1], pts[:, 2].min()])
        rel = pts - origin
        a, b = rel @ e1, rel @ e2
        kept = skipped = 0
        for ai in np.arange(a.min(), a.max(), tile_m):
            for bi in np.arange(b.min(), b.max(), tile_m):
                m = (a >= ai) & (a < ai + tile_m) & (b >= bi) & (b < bi + tile_m)
                if m.sum() < min_points:
                    continue
                tp = pts[m]
                centroid = tp.mean(axis=0)
                normal = np.linalg.svd(tp - centroid, full_matrices=False)[2][2]
                rms = float(np.sqrt(np.mean(((tp - centroid) @ normal) ** 2)))
                if rms > max_rms_m:
                    skipped += 1
                    continue
                outward = np.array([np.sin(np.radians(f.bearing_deg)),
                                    np.cos(np.radians(f.bearing_deg)), 0.0])
                if normal @ outward < 0:
                    normal = -normal
                t1 = e1 - (e1 @ normal) * normal
                t1 /= np.linalg.norm(t1)
                t2 = np.cross(normal, t1)
                if t2[2] < 0:
                    t2 = -t2
                r = tp - centroid
                ga = np.arange((r @ t1).min(), (r @ t1).max() + spacing_m, spacing_m)
                gb = np.arange((r @ t2).min(), (r @ t2).max() + spacing_m, spacing_m)
                if len(ga) < 2 or len(gb) < 2 or len(ga) * len(gb) > 1_000_000:
                    continue
                # The tile as a rectangle in its own plane, for a mesh to use.
                tiles_by_face.setdefault(f.index, []).append({
                    "centroid": centroid, "normal": normal, "rms": rms,
                    "t1": t1, "t2": t2,
                    "a_range": (float(ga[0]), float(ga[-1])), "b_range": (float(gb[0]), float(gb[-1])),
                })
                A, B = np.meshgrid(ga, gb)
                grid = centroid + A.ravel()[:, None] * t1 + B.ravel()[:, None] * t2
                out_xyz.append(grid)
                out_n.append(np.repeat(normal[None, :], len(grid), axis=0))
                out_face.append(np.full(len(grid), f.index))
                kept += 1
        stats.append((f.index, f.length_m, len(pts), kept, skipped))

    if not out_xyz:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, dtype=int), faces, stats, tiles_by_face
    return (np.vstack(out_xyz), np.vstack(out_n),
            np.concatenate(out_face), faces, stats, tiles_by_face)


def extrude_verified_faces(
    nominal_planes: list[WallPlane],
    tiles_by_face: dict[int, list[dict]],
    spacing_m: float,
    tile_m: float = 12.0,
    max_offset_rms_m: float = 0.30,
    max_normal_dev_deg: float = 4.0,
    roof_z: Optional[float] = None,
    prism_completion: bool = True,
    min_prism_ratio: float = 0.60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, VerifiedWallPlane]]:
    """Extrude verified vertical wall planes and de-duplicate against fitted tiles.

    If ``prism_completion`` is True and verified faces account for at least
    ``min_prism_ratio`` of the nominal perimeter, unverified connecting walls
    are completed as planar prism faces bridging the verified walls.

    Returns:
        (ext_xyz, ext_n, ext_face, verified_planes)
    """
    ext_xyz, ext_n, ext_face = [], [], []
    verified_planes: dict[int, VerifiedWallPlane] = {}

    for plane in nominal_planes:
        face_idx = plane.face
        raw_tiles = tiles_by_face.get(face_idx, [])
        # Only consider tiles facing in the nominal outward wall direction,
        # filtering out horizontal ledges, canopies, or floor slabs captured near the facade.
        n_out_3d = np.array([plane.normal[0], plane.normal[1], 0.0])
        tiles = [t for t in raw_tiles if t["normal"] @ n_out_3d > 0.7]
        if len(tiles) < 2:
            continue
        centroids = np.array([t["centroid"] for t in tiles], dtype=np.float64)
        normals = np.array([t["normal"] for t in tiles], dtype=np.float64)

        verified = verify_face_plane(
            plane=plane,
            tile_centroids=centroids,
            tile_normals=normals,
            max_offset_rms_m=max_offset_rms_m,
            max_normal_dev_deg=max_normal_dev_deg,
            min_tiles=2,
            tile_m=tile_m,
            roof_z=roof_z,
        )
        if verified is not None:
            verified_planes[face_idx] = verified

    if prism_completion:
        inferred = infer_prism_faces(
            nominal_planes=nominal_planes,
            verified_planes=verified_planes,
            min_perimeter_ratio=min_prism_ratio,
            roof_z=roof_z,
        )
        verified_planes.update(inferred)

    for face_idx, verified in verified_planes.items():
        raw_tiles = tiles_by_face.get(face_idx, [])
        n_out_3d = np.array([verified.normal[0], verified.normal[1], 0.0])
        tiles = [t for t in raw_tiles if t["normal"] @ n_out_3d > 0.7]

        p0, p1 = verified.p0, verified.p1
        seg = p1 - p0
        L = float(np.hypot(seg[0], seg[1]))
        if L < 1e-6:
            continue
        e1 = seg / L
        n_out = verified.normal

        s_vals = np.arange(0.0, L, spacing_m)
        z_vals = np.arange(verified.z_lo, verified.z_hi, spacing_m)
        if len(s_vals) == 0 or len(z_vals) == 0:
            continue

        S, Z = np.meshgrid(s_vals, z_vals)
        S_flat = S.ravel()
        Z_flat = Z.ravel()

        keep = np.ones(len(S_flat), dtype=bool)
        half_tile = tile_m / 2.0
        for t in tiles:
            c_t = t["centroid"]
            s_t = float((c_t[:2] - p0) @ e1)
            z_t = float(c_t[2])
            in_tile = (np.abs(S_flat - s_t) < half_tile) & (np.abs(Z_flat - z_t) < half_tile)
            keep[in_tile] = False

        if not keep.any():
            continue

        S_keep = S_flat[keep]
        Z_keep = Z_flat[keep]

        pts_xy = p0[None, :] + S_keep[:, None] * e1[None, :]
        pts_3d = np.column_stack([pts_xy, Z_keep])
        pts_norm = np.tile(np.array([n_out[0], n_out[1], 0.0]), (len(pts_3d), 1))
        pts_face = np.full(len(pts_3d), face_idx, dtype=int)

        ext_xyz.append(pts_3d)
        ext_n.append(pts_norm)
        ext_face.append(pts_face)

    if not ext_xyz:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, dtype=int), verified_planes
    return (np.vstack(ext_xyz), np.vstack(ext_n),
            np.concatenate(ext_face), verified_planes)


def under_measured_edge(planes, is_arc, level_of, tower_levels, measured_edges, max_off_m: float = 1.5,
                        max_angle_deg: float = 10.0, min_cover: float = 0.6) -> set:
    """The faces standing under a roof edge the LiDAR measured: every face of
    a level whose outline is the LiDAR's tower edge, and the ground level's
    faces lying along (within ``max_off_m`` and ``max_angle_deg``) outline
    edges whose roof end it found, over at least ``min_cover`` of their length."""
    out = set()
    for pl in planes:
        if is_arc[pl.face]:
            continue
        lv = level_of[pl.face]
        if lv in tower_levels:
            out.add(pl.face)
            continue
        if lv != 0 or not measured_edges:
            continue
        L = pl.length_m
        e1 = (pl.p1 - pl.p0) / max(L, 1e-9)
        covered = []
        for q0, q1 in measured_edges:
            d = q1[:2] - q0[:2]
            Lq = float(np.hypot(*d))
            if Lq < 1e-6 or abs(float(np.dot(d / Lq, e1))) < np.cos(np.radians(max_angle_deg)):
                continue
            mid = 0.5 * (q0[:2] + q1[:2])
            if abs(float(np.dot(mid - pl.p0, pl.normal))) > max_off_m:
                continue
            a, b = sorted((float(np.dot(q0[:2] - pl.p0, e1)), float(np.dot(q1[:2] - pl.p0, e1))))
            if min(b, L) > max(a, 0.0):
                covered.append((max(a, 0.0), min(b, L)))
        covered.sort()
        tot, end = 0.0, 0.0
        for a, b in covered:
            if b > end:
                tot += b - max(a, end)
                end = b
        if tot >= min_cover * L:
            out.add(pl.face)
    return out


def write_ply(path: Path, xyz, rgb, extra=None, extras=None):
    """The coloured cloud: LAZ when the path says ``.laz`` (a quarter of the
    PLY), the binary PLY otherwise. ``clouds.read_cloud`` reads either."""
    if Path(path).suffix.lower() in (".laz", ".las"):
        from kyfromabove_3d.clouds import write_cloud
        cols = {"source": extra} if extra is not None else {}
        cols.update(extras or {})
        return write_cloud(Path(path), xyz, rgb=rgb, extras=cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
            "property float x", "property float y", "property float z",
            "property uchar red", "property uchar green", "property uchar blue"]
    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
          ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if extra is not None:
        head.append("property uchar source")
        dt.append(("source", "u1"))
    if extras is not None:
        for k in extras:
            head.append(f"property uchar {k}")
            dt.append((k, "u1"))
    head.append("end_header")
    arr = np.zeros(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if extra is not None:
        arr["source"] = extra
    if extras is not None:
        for k, v in extras.items():
            arr[k] = v
    with open(path, "wb") as fh:
        fh.write(("\n".join(head) + "\n").encode("ascii"))
        fh.write(arr.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--building", type=Path, required=True,
                    help="a pipeline output dir holding lidar_local.laz (or .ply), "
                         "cameras_normalized.json and images/")
    ap.add_argument("--case",
                    help="key in the JSON footprint cache, e.g. building_002")
    ap.add_argument("--gers",
                    help="GERS id, looked up in --footprint-csv instead")
    ap.add_argument("--footprint", type=Path, default=None,
                    help="a footprint.json written by prepare (the default when the building dir has one "
                         "and neither --case nor --gers is given)")
    ap.add_argument("--footprints", type=Path,
                    default=Path("outputs/survey/case_footprints.json"))
    ap.add_argument("--footprint-csv", type=Path,
                    default=Path("data/derived/ky_facade_tiled_wkt.csv"),
                    help="a fetch_overture_buildings list, for --gers")
    ap.add_argument("--spacing-m", type=float, default=0.15,
                    help="wall resample pitch. 3-inch imagery is 0.076 m; the "
                         "default is deliberately coarser, because the "
                         "reprojection is only good to 0.153 m and sampling "
                         "finer than that invents detail the geometry cannot "
                         "place")
    ap.add_argument("--min-face-m", type=float, default=4.0)
    ap.add_argument("--max-offset-m", type=float, default=2.0)
    ap.add_argument("--tile-m", type=float, default=12.0,
                    help="side of the square a plane is fitted to. One plane per "
                         "whole face rejects any building with setbacks")
    ap.add_argument("--max-plane-rms-m", type=float, default=0.35,
                    help="how far a tile's returns may sit off their own plane "
                         "before the tile is skipped rather than invented")
    ap.add_argument("--min-facing-cos", type=float, default=0.15)
    ap.add_argument("--occlusion-slack-m", type=float, default=1.5,
                    help="how much closer something must be to count as "
                         "blocking. Generous, because the depth map is built "
                         "from sparse LiDAR and a tight value rejects a wall "
                         "against its own returns")
    ap.add_argument("--scene-cloud", type=Path, default=None,
                    help="an .npz with xyz_ft (source feet) covering the area, written once by run_downtown.py; "
                         "windowed to --occluder-radius-m instead of range-reading the tiles")
    ap.add_argument("--drop-classes", type=lambda v: tuple(int(x) for x in v.split(",")) if v else (),
                    default=(3, 4, 5, 7, 9),
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
    ap.add_argument("--twin-colour", action="store_true",
                    help="colour a hidden face's unseen points from its symmetric twin (an invention; off in "
                         "the measured product)")
    ap.add_argument("--occluder-radius-m", type=float, default=250.0,
                    help="half-width of the LiDAR extent read purely to occlude "
                         "with. Must reach past the AOI: the neighbours that "
                         "block a wall are by definition outside the box drawn "
                         "around the target")
    ap.add_argument("--no-occlusion", action="store_true",
                    help="skip the visibility test, for comparison. The result "
                         "paints whatever is in front of a wall onto it")
    ap.add_argument("--extrude", choices=["profile", "prism", "verified", "off"], default="profile",
                    help="extrude planar-verified faces ('prism' completes connecting prism faces if >= 60%% perimeter verified; 'verified' only directly verified faces; 'off' disables extrusion). "
                         "'profile' (default) does 'prism' and then, for faces still unverified, extrudes band by "
                         "band wherever the returns near the face say where the wall is -- and bridges bands the "
                         "flight missed when the wall above and below agree. Humana's east wall stands 2 m outside "
                         "its footprint and steps back at the crown; its south wall was seen to 40 m and from 90 m; "
                         "neither is one plane, both are walls")
    ap.add_argument("--prism-min-ratio", type=float, default=0.60,
                    help="minimum fraction of nominal perimeter verified before inferring prism completion")
    ap.add_argument("--lidar-outline", action="store_true",
                    help="use the building's footprint_lidar.json (its outline from the LiDAR roof, refined in the "
                         "straight-down frames: scripts/lidar_outline.py) instead of Overture's footprint, when it has one")
    ap.add_argument("--match-walls", action="store_true",
                    help="give an unknown wall band the depth where the frames that see it agree (kyfromabove_3d.wall_match), "
                         "when they agree well enough; such a band is 'matched', at the matched depth less the building's "
                         "lean (from its measured bands, 0 with fewer than 3). Off by default; match_report.json says what it did")
    ap.add_argument("--profile-band-m", type=float, default=5.0,
                    help="height of the bands a profiled face is measured in")
    ap.add_argument("--extrude-max-rms-m", type=float, default=0.30,
                    help="maximum RMS distance of tile centroids to fitted face plane")
    ap.add_argument("--extrude-max-normal-deg", type=float, default=4.0,
                    help="maximum angular deviation of tile normals from face normal")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()


    started = time.time()
    coords = json.loads((args.building / "coordinates.json").read_text(encoding="utf-8"))
    ox, oy, oz = coords["origin_source_xyz"]

    lidar_path = find_cloud(args.building, "lidar_local")
    ply = read_cloud(lidar_path)
    xyz = np.column_stack([ply["x"], ply["y"], ply["z"]]).astype(np.float64)
    print(f"{len(xyz):,} LiDAR points from {lidar_path}")
    # the classification rides along with the points through every filter
    cls = np.asarray(ply["classification"]) if "classification" in ply else None
    if args.drop_classes and cls is not None:
        # vegetation, noise, water: a tree over a low roof is not its wall
        drop = np.isin(cls, args.drop_classes)
        if drop.any():
            print(f"  {drop.sum():,} returns of class {args.drop_classes} left out")
            xyz, cls = xyz[~drop], cls[~drop]

    cams = load_cameras(args.building)
    print(f"{len(cams)} cameras with crops: "
          f"{', '.join(sorted({c['head'] for c in cams}))}")
    if not cams:
        print("[FAIL] no cameras with images on disk")
        return 2

    # No fallback anywhere below. An earlier version quietly took the first
    # entry when the lookup missed, and coloured 400 West Market against Tractor
    # Supply's footprint without saying so.
    from kyfromabove_3d.footprints import resolve_footprint
    fp_case, fp_gers, fp_file = args.case, args.gers, args.footprint
    if args.lidar_outline and (args.building / "footprint_lidar.json").exists():
        # the outline from our own data; Overture's polygon stays in the record as outline.overture_wkt
        fp_case, fp_gers, fp_file = None, None, args.building / "footprint_lidar.json"
        print("footprint: the LiDAR outline (footprint_lidar.json), not Overture's")
    elif args.lidar_outline:
        print("[warn] --lidar-outline: no footprint_lidar.json here; Overture's footprint used")
    try:
        poly_local, entry = resolve_footprint(args.building, ox, oy, case=fp_case, gers=fp_gers,
                                              footprint=fp_file, footprints_json=args.footprints,
                                              footprint_csv=args.footprint_csv)
    except (KeyError, ValueError) as exc:
        print(f"[FAIL] {exc}")
        return 1
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
        from kyfromabove_3d.footprints import footprint_wgs84_to_local
        from shapely import contains_xy as _cxy
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

    # The zone split must run on THIS building, not the AOI. The AOI is 110 m
    # of downtown around the tower, so its modal elevation is the street, every
    # point lands above the roof line, and the whole cloud reads as roof --
    # which is the same way the tower's QA came to be measuring its neighbours.
    from shapely.geometry import Polygon
    from shapely import contains_xy as _contains
    poly_local = Polygon(ring_m)
    on_building = _contains(poly_local.buffer(args.max_offset_m),
                            xyz[:, 0], xyz[:, 1])
    print(f"{on_building.sum():,} of {len(xyz):,} points fall on the footprint")
    if on_building.sum() < 200:
        print("[FAIL] the footprint and the LiDAR do not overlap")
        return 2
    b_xyz = xyz[on_building]
    # The profile looks 3 m outside the footprint edge and 8 m inside; the
    # 2 m clip above would cut off a wall standing 2-3 m proud of a
    # footprint traced along its podium (Humana's south wall does).
    profile_xyz = xyz[_contains(poly_local.buffer(4.0), xyz[:, 0], xyz[:, 1])]
    zones = split_surface_zones(b_xyz)
    if zones is None:
        print("[FAIL] cannot split zones")
        return 2
    # Ground elevation in the local frame. In the downtown/prepared frame, 0 is
    # the frame's ground level. Standalone cases use a robust lower percentile
    # so noise returns in cellars, river banks or pits (-10 m for RIVUE) do not
    # sit below street level and cause parking lots or plazas to count as roof.
    if (args.building / "footprint.json").exists():
        ground_z = 0.0
    elif cls is not None and (cls[on_building] == 2).sum() >= 20:
        ground_z = float(np.median(b_xyz[cls[on_building] == 2, 2]))
    else:
        ground_z = float(np.percentile(b_xyz[:, 2], 1.0))

    from kyfromabove_3d.features import LEVEL_FACE_STRIDE, building_levels
    roof_z = float(np.percentile(b_xyz[:, 2], 99))
    levels = building_levels(b_xyz, ring_m, ground_z) if args.levels else \
        [{"level": 0, "ring": ring_m, "z_lo": ground_z, "z_hi": roof_z}]
    lidar_tower_levels: set[int] = set()                    # levels whose outline is the LiDAR's tower edge
    if args.lidar_outline and entry.get("towers_local") and len(levels) > 1:
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
    roof_z = max(float(lv["z_hi"]) for lv in levels)
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

    if not args.no_occlusion:
        if args.scene_cloud:
            wider = scene_cloud_from_cache(args.scene_cloud, (ox, oy, oz), args.occluder_radius_m)
        else:
            wider = scene_cloud((ox, oy, oz), args.occluder_radius_m)
        if wider is None:
            occ = xyz
            print("[warn] no wider tile; occluding with the AOI alone, which "
                  "cannot see the neighbours that block a wall")
        else:
            occ = np.column_stack([(wider[:, 0] - ox) * METERS_PER_US_SURVEY_FOOT,
                                   (wider[:, 1] - oy) * METERS_PER_US_SURVEY_FOOT,
                                   (wider[:, 2] - oz) * METERS_PER_US_SURVEY_FOOT])
        prisms = [] if args.no_prism_occluders else neighbour_prisms(
            args.building, args.neighbours, args.occluder_radius_m, levels_dir(args.neighbour_levels))
        occlusion_maps(args.building, cams, occ, prisms)
        print(f"depth maps from {len(occ):,} returns within "
              f"{args.occluder_radius_m:.0f} m ({len(xyz):,} in the AOI), so a "
              f"wall is not coloured through whatever stands in front of it")
        if prisms:
            print(f"  and {len(prisms)} neighbour prism(s): a tower's facade, which the LiDAR "
                  f"barely touches, occludes as a solid")
    rgb, seen, _, from_photo_orig = sample_colour(xyz, normals, cams, args.min_facing_cos,
                                                 args.occlusion_slack_m)
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
        b_xyz, zones.wall, ring_m, height_m, args.spacing_m,
        args.min_face_m, args.max_offset_m, tile_m=args.tile_m,
        max_rms_m=args.max_plane_rms_m, skip_faces=arc_edges)
    print(f"\n{len(faces)} wall faces, {sum(f.captured for f in faces)} reached")
    for i, length, n, kept, skipped in stats:
        print(f"    face {i}: {length:5.1f} m, {n:>6,} returns -> "
            f"{kept} tile(s) fitted, {skipped} too far from a plane")
    print(f"  resampled {len(new_xyz):,} new wall tile points at {args.spacing_m} m")

    # The bodies the footprint holds. Overture draws the ground floor's
    # outline; a tower on a podium has a second outline the LiDAR knows.
    # Each level gets its own faces (level k's numbered from k * LEVEL_FACE_STRIDE) over
    # its own height range, and everything below works per face as before.
    nominal_planes = []
    for lv in levels:
        # arcs only on the footprint's own ring: a traced tower outline has
        # rounded corners that are not curved walls
        edges = wall_planes(lv["ring"], lv["z_lo"], lv["z_hi"], min_length_m=args.min_face_m, arcs=(lv["level"] == 0))
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

    if args.extrude != "off":
        prism_comp = args.extrude in ("prism", "profile")
        all_ext_xyz, all_ext_n, all_ext_face = [], [], []
        for lv in levels:
            lv_planes = [pl for pl in nominal_planes if level_of[pl.face] == lv["level"] and not is_arc[pl.face]]
            if not lv_planes:
                continue
            ext_xyz, ext_n, ext_face, lv_verified = extrude_verified_faces(
                nominal_planes=lv_planes,
                tiles_by_face=tiles_by_face,
                spacing_m=args.spacing_m,
                tile_m=args.tile_m,
                max_offset_rms_m=args.extrude_max_rms_m,
                max_normal_dev_deg=args.extrude_max_normal_deg,
                roof_z=float(lv["z_hi"]),
                prism_completion=prism_comp,
                min_prism_ratio=args.prism_min_ratio,
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
    if args.corner_snap and len(levels) > 0 and level0_straight and cams:
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
            return lambda pts, nrm: sample_colour(pts, nrm, [cam], args.min_facing_cos, args.occlusion_slack_m)[:2]

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
    if args.extrude == "profile":
        from kyfromabove_3d.features import clip_profiles_by_neighbours, extrude_profile, face_profile, profile_keeps
        # Pass 1: every face's profile.
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
            bands = face_profile(plane, lv_xyz, band_m=args.profile_band_m)
            # A tower can stand well back from the edge its podium was traced
            # along: Humana's north wall is 25 m inside the Main Street edge,
            # seen by a thin column of returns the 8 m window never reaches.
            # When the standard window finds no wall above the podium, look
            # deeper -- up to 60% of the way to the far side -- with a lower
            # floor, and keep the result only if it finds a set-back wall.
            # (Only when the building is one level: with the tower split
            # out, its walls are its own faces.)
            top_seen = max((b.z_hi for b in bands if b.status in ("observed", "bridged", "extended")), default=0.0)
            if len(levels) == 1 and top_seen < 0.4 * (roof_z - ground_z) and not is_arc[plane.face]:
                depth = float(max(-((ring_m - plane.p0) @ plane.normal).min(), 8.0))
                deep = face_profile(plane, lv_xyz, band_m=args.profile_band_m,
                                    offset_window_m=(-0.6 * depth, 3.0), min_returns=20)
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
        if args.match_walls:
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
        # the faces under a roof edge the LiDAR measured (only with the LiDAR outline):
        # an unseen one takes the roof edge's word, not its twin's (pass 2a'')
        under = (under_measured_edge(nominal_planes, is_arc, level_of, lidar_tower_levels, measured_edges)
                 if args.lidar_outline else set())
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
        # gap. Only with the LiDAR outline: Overture's edge is not measured.
        if args.lidar_outline:
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
            pts = extrude_profile(plane, bands, args.spacing_m)
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
                                                             args.min_facing_cos,
                                                             args.occlusion_slack_m)
        # A face read from its twin for shape can take its colour from the
        # twin too, on the inferred bands and wherever no camera reached --
        # an invention, so only with --twin-colour: the measured product
        # (product 2) colours a texel only where a frame saw it, and the
        # completed product (3) makes the twin copy on its textures, flagged.
        colour_from_twin = np.zeros(len(new_xyz), dtype=bool)
        if args.extrude == "profile" and mirrored and args.twin_colour:
            from kyfromabove_3d.features import mirror_colour_from_twin
            colour_from_twin = mirror_colour_from_twin(nominal_planes, profiles, mirrored,
                                                       new_xyz, new_face, new_rgb, new_seen)
            for f, rec in sorted(mirrored.items()):
                n_f = int((colour_from_twin & (new_face == f)).sum())
                if n_f:
                    print(f"    face {f}: {n_f:,} points coloured from face {rec['from_face']} opposite")
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
            min_face_length_m=args.min_face_m,
            max_offset_m=args.max_offset_m,
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
