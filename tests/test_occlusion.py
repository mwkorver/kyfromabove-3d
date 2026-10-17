"""The occlusion maps' speed-ups, held to giving the same answers: a cull
before the projection, and maps made once and shared by the colourise, the
mesh and the check.

(The frame crops are not cut in threads: tried in September 2026, GDAL in
the Lambda image deadlocked inside a window read on the third try, each
thread holding only its own handles.)"""

from __future__ import annotations

import numpy as np
import pytest

from kyfromabove_3d.colourise import occlusion as C



def _cams(n=3, seed=0):
    """Oblique cameras about 1 km out and 800 m up, looking at the origin."""
    rng = np.random.default_rng(seed)
    cams = []
    for i in range(n):
        a = 2 * np.pi * i / n + rng.uniform(-0.2, 0.2)
        Cc = np.array([1000 * np.cos(a), 1000 * np.sin(a), 800.0])
        z = -Cc / np.linalg.norm(Cc)
        x = np.cross(z, [0.0, 0.0, 1.0]); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        cams.append({"id": f"cam{i}", "wh": (704, 704), "C": Cc, "R_cw": np.vstack([x, y, z]),
                     "K": np.array([[9000.0, 0, 352.0], [0, 9000.0, 352.0], [0, 0, 1.0]])})
    return cams


def _scene(n=200_000, seed=1):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-250, 250, n), rng.uniform(-250, 250, n), rng.uniform(-5, 60, n)])


def _exact_keep(cam, xyz, ds=8):
    """The points the original depth_buffers kept, by its own arithmetic."""
    w, h = cam["wh"]; dw, dh = w // ds + 1, h // ds + 1
    pc = (cam["R_cw"] @ (xyz - cam["C"]).T).T
    ok = pc[:, 2] > 1e-6
    proj = (cam["K"] @ pc[ok].T).T
    u = (proj[:, 0] / proj[:, 2] / ds).astype(int); v = (proj[:, 1] / proj[:, 2] / ds).astype(int)
    keep = np.zeros(len(xyz), bool)
    keep[np.flatnonzero(ok)[(u >= 0) & (u < dw) & (v >= 0) & (v < dh)]] = True
    return keep


def test_the_cull_never_drops_a_point_the_projection_keeps():
    xyz = _scene()
    for cam in _cams(6):
        # add points exactly on and just past the crop's edges, where a cull would err
        Kinv = np.linalg.inv(cam["K"])
        edge = []
        for u in (-8.0, -7.99, 0.0, 703.9, 704.0, 711.99, 712.0):
            for v in np.linspace(-8, 712, 40):
                ray = cam["R_cw"].T @ (Kinv @ np.array([u, v, 1.0]))
                edge.append(cam["C"] + ray * 1300.0)
        pts = np.vstack([xyz, np.array(edge)])
        keep = _exact_keep(cam, pts)
        cull = C.may_land_on_crop(cam, pts.astype(np.float32), *cam["wh"])
        assert keep.any() and not (keep & ~cull).any()        # everything kept survives the cull
        assert cull.mean() < 0.2                              # and the cull does drop most


def _original_depth_buffers(cams, scene_xyz, downscale=8, dilate=3):
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
    for cam in cams:
        w, h = cam["wh"]
        dw, dh = w // downscale + 1, h // downscale + 1
        buf = np.full((dh, dw), np.inf)
        pc = (cam["R_cw"] @ (scene_xyz - cam["C"]).T).T
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


def test_the_original_body_kept_here_is_the_one_replaced():
    a, b = _cams(3), _cams(3)
    xyz = _scene(50_000)
    _original_depth_buffers(a, xyz); C.depth_buffers(b, xyz)
    assert all(np.array_equal(x["depth"], y["depth"]) for x, y in zip(a, b))


@pytest.fixture
def counted(monkeypatch):
    calls = {"n": 0}
    real = C.depth_buffers

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(C, "depth_buffers", counting)
    return calls


def _prisms():
    return [(np.array([[20.0, 20], [40, 20], [40, 40], [20, 40], [20, 20]]), 0.0, 30.0)]


def test_saved_maps_are_reused_only_for_the_same_inputs(tmp_path, counted):
    xyz = _scene(30_000)
    first = C.occlusion_maps(tmp_path, _cams(3), xyz, _prisms(), log=lambda m: None)
    assert counted["n"] == 1 and (tmp_path / C.OCCLUSION_FILE).exists()
    again = C.occlusion_maps(tmp_path, _cams(3), xyz, _prisms(), log=lambda m: None)
    assert counted["n"] == 1                                              # the same inputs: loaded
    assert all(np.array_equal(x["depth"], y["depth"]) for x, y in zip(first, again))

    def changed(cams=None, pts=None, prisms=None, **kw):
        before = counted["n"]
        C.occlusion_maps(tmp_path, cams or _cams(3), xyz if pts is None else pts,
                         _prisms() if prisms is None else prisms, log=lambda m: None, **kw)
        return counted["n"] == before + 1

    moved = _cams(3); moved[1]["C"] = moved[1]["C"] + [0, 0, 0.001]
    assert changed(cams=moved)                                            # a camera 1 mm higher
    turned = _cams(3); turned[0]["R_cw"] = turned[0]["R_cw"] * (1 + 1e-12)
    assert changed(cams=turned)
    zoom = _cams(3); zoom[2]["K"] = zoom[2]["K"].copy(); zoom[2]["K"][0, 0] += 0.01
    assert changed(cams=zoom)
    size = _cams(3); size[0]["wh"] = (705, 704)
    assert changed(cams=size)
    renamed = _cams(3); renamed[0]["id"] = "another frame"
    assert changed(cams=renamed)
    assert changed(cams=_cams(2))                                         # a camera fewer
    one_point = xyz.copy(); one_point[123, 2] += 0.01
    assert changed(pts=one_point)                                         # one return 1 cm higher
    assert changed(prisms=[])                                             # no neighbours
    taller = [(r, lo, hi + 0.5) for r, lo, hi in _prisms()]
    assert changed(prisms=taller)
    assert changed(downscale=4)
    assert changed(dilate=5)


def test_a_damaged_saved_file_is_made_again(tmp_path, counted):
    (tmp_path / C.OCCLUSION_FILE).write_bytes(b"not an npz")
    C.occlusion_maps(tmp_path, _cams(2), _scene(10_000), [], log=lambda m: None)
    assert counted["n"] == 1
