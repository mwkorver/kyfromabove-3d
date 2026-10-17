"""A building's points coloured from the frames: the cameras, which points each frame
sees, the colour sampled, and the coloured cloud written as PLY."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


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
