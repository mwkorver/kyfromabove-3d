"""What hides a building's walls from a frame: the LiDAR around it as depth maps, and its
neighbours standing as prisms at their measured heights (the colourise's occlusion)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kyfromabove_3d.coordinates import METERS_PER_US_SURVEY_FOOT

ROOT = Path(__file__).resolve().parents[3]        # the repository (src/kyfromabove_3d/colourise/..)

#: every downtown footprint with its LiDAR ground and height: the neighbours
FOOTPRINTS = ROOT / "data" / "derived" / "louisville_downtown_overture_split.json"


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


#: a neighbour's levels as the building step measured them (kyfromabove_3d.pipeline.MEASURED_LEVELS)
MEASURED_LEVELS = "levels_measured.json"


def neighbour_prisms(b: Path, footprints: Path | None = FOOTPRINTS, radius_m: float = 250.0,
                     buildings_dir: Path | None = ROOT / "outputs" / "downtown" / "buildings"):
    """Every other building within ``radius_m`` of building dir ``b``, as solid
    prisms ``(ring (N, 2) closed, z_lo, z_hi)`` in ``b``'s local metres.

    A neighbour the building step has measured (``buildings_dir/<id>/
    levels_measured.json``, kyfromabove_3d.pipeline: its levels as the LiDAR
    shows them, written before any building is built, so the same whatever
    order they are built in) gives one prism per level -- a podium to its own
    top, the tower's traced outline above it -- so a podium does not stand at
    its tower's height; a level is capped at the footprint's LiDAR height.
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
    wkt = json.loads((b / "footprint.json").read_text(encoding="utf-8"))["wkt"]
    me = footprint_wgs84_to_local(wkt, ox, oy)
    out = []
    for r in recs:
        poly = footprint_wgs84_to_local(r["wkt"], ox, oy)
        if poly.distance(Point(0.0, 0.0)) > radius_m or poly.intersection(me).area > 1.0:
            continue
        nb = Path(buildings_dir) / r["id"] if buildings_dir else None
        if nb is not None and (nb / MEASURED_LEVELS).exists() and (nb / "coordinates.json").exists():
            nx, ny, nz = json.loads((nb / "coordinates.json").read_text(encoding="utf-8"))["origin_source_xyz"]
            shift = np.array([(nx - ox) * FT, (ny - oy) * FT])
            dz = (nz - oz) * FT
            # never taller than the footprint's own 99th percentile: a levels.json
            # written before the spill cap (362 of 381 downtown, September 2026)
            # can hold the tall building next door -- cc3f65d1, an 18 m
            # building beside Humana, stood at 116 m and hid Humana's walls
            cap = (float(r["lidar_ground_z"]) - oz * FT + float(r["lidar_height_m"])) \
                if r.get("lidar_ground_z") is not None and r.get("lidar_height_m") else np.inf
            for lv in json.loads((nb / MEASURED_LEVELS).read_text(encoding="utf-8")):
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
