"""The textures of the mesh: each face's image from the coloured cloud or the frames,
small holes filled, and the atlas the pages are packed into."""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: where a texel's colour came from -- the measured product keeps SEEN and
#: NEAR, and says PLAIN where nothing saw the wall; the completed product
#: fills PLAIN texels and records how (kyfromabove_3d.facade_fill)
from kyfromabove_3d.facade_fill import NEAR, PLAIN, SEEN
from kyfromabove_3d.features import (
    WALL_STATUSES,
    ArcWall,
    FaceBand,
    WallPlane,
)

from .walls import _orient, _wall_behind_band

TEXEL_M = 0.15            # texture resolution, metres per pixel

MAX_TEX_PX = 4096


def _mult4(n: int) -> int:
    """Block-compressed textures need sides that are multiples of 4."""
    return max(4, int(np.ceil(n / 4.0)) * 4)


def face_texture(plane: WallPlane, bands: list[FaceBand], extent, xyz: np.ndarray,
                 rgb: np.ndarray, texel_m: float = TEXEL_M, party: bool = False):
    """Rasterise the coloured cloud onto the face: a point counts where it
    lies on one of the face's band rectangles (within 0.6 m of the offset
    the profile gives at that height and position along the edge), or on a
    wall just behind a band that caught few points (``_wall_behind_band``;
    never on a ``party`` face)."""
    z0, z1, L = extent
    along, off = plane.local(xyz[:, :2])
    z = xyz[:, 2]
    near = (along >= -0.5) & (along <= L + 0.5) & (z >= z0 - 0.5) & (z <= z1 + 0.5)
    keep = np.zeros(len(xyz), dtype=bool)
    idx = np.flatnonzero(near)
    for b in bands:
        if b.status == "open" and b.back_offset_m is not None:
            b = FaceBand(b.z_lo, b.z_hi, b.back_offset_m, b.n_returns, 0.0, b.along_lo, b.along_hi, "observed")
        if b.status not in WALL_STATUSES or b.offset_m is None:
            continue
        inb = idx[(z[idx] >= b.z_lo - 0.2) & (z[idx] < b.z_hi + 0.2)]
        if not len(inb):
            continue
        expect = np.full(len(inb), b.offset_m)
        for a0, a1, oseg in b.segments:
            expect[(along[inb] >= a0) & (along[inb] < a1)] = oseg
        matched = np.abs(off[inb] - expect) < 0.6
        if not party:
            matched = _wall_behind_band(off[inb], expect, b.offset_m, float(getattr(plane, "offset_m", 0.0)), matched)
        keep[inb[matched]] = True
    W = max(2, min(MAX_TEX_PX, int(np.ceil(L / texel_m))))
    H = max(2, min(MAX_TEX_PX, int(np.ceil((z1 - z0) / texel_m))))
    sx, sy = W / L, H / (z1 - z0)
    img = np.zeros((H, W, 3), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.float64)
    k = np.flatnonzero(keep)
    coloured = k[rgb[k].sum(axis=1) > 15]
    if len(coloured):
        u = np.clip((along[coloured] * sx).astype(int), 0, W - 1)
        v = np.clip(((z[coloured] - z0) * sy).astype(int), 0, H - 1)
        np.add.at(img, (v, u), rgb[coloured].astype(np.float64))
        np.add.at(cnt, (v, u), 1.0)
    return _fill_holes(img, cnt, solid_px=_solid_px(texel_m)), int(len(coloured))


class Texture(np.ndarray):
    """A texture image that knows, per pixel, where its colour came from
    (``source``: SEEN -- a frame, or the cloud a frame coloured; NEAR -- the
    nearest seen pixel across a small gap; PLAIN -- the face's median, where
    nothing saw it). Rows run top-down, as the image does."""

    def __new__(cls, img: np.ndarray, source: np.ndarray):
        obj = np.asarray(img).view(cls)
        obj.source = source
        return obj

    def __array_finalize__(self, obj):
        self.source = getattr(obj, "source", None)


def _gap_between(filled: np.ndarray, max_px: int) -> np.ndarray:
    """Empty texels that lie *between* filled ones: a filled texel within
    ``max_px`` on both sides along one of four lines through it -- across,
    up, or either diagonal. A gap inside a sparse but regular raster of
    points is between; the ring round a lone point is not."""
    H, W = filled.shape

    def seen_towards(dy, dx):
        hit = np.zeros_like(filled)
        for k in range(1, max_px + 1):
            sy, sx = dy * k, dx * k
            if abs(sy) >= H or abs(sx) >= W:
                break                                   # past the texture's edge
            src = filled[max(sy, 0):H + min(sy, 0), max(sx, 0):W + min(sx, 0)]
            hit[max(-sy, 0):H + min(-sy, 0), max(-sx, 0):W + min(-sx, 0)] |= src
        return hit

    between = np.zeros_like(filled)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        between |= seen_towards(dy, dx) & seen_towards(-dy, -dx)
    return ~filled & between


def _solid_px(texel_m: float) -> int:
    """The window, about 1 m across and an odd number of texels, in which a
    seen texel must have seen neighbours to count on a wall."""
    return max(3, int(round(1.0 / texel_m)) | 1)


def _row_medians(out: np.ndarray, seen: np.ndarray, band_px: int, rows: np.ndarray,
                 min_seen: int = 30) -> np.ndarray:
    """Per row, the median seen colour within ``band_px`` rows of it -- the
    same floor, give or take. A floor too little seen takes the nearest floor
    that was; a face with none such, its own median."""
    H = len(seen)
    cum = np.concatenate([[0], np.cumsum(seen.sum(axis=1))])
    lo = np.maximum(np.arange(H) - band_px, 0); hi = np.minimum(np.arange(H) + band_px + 1, H)
    enough = np.flatnonzero(cum[hi] - cum[lo] >= min_seen)
    med = np.tile(np.median(out[seen], axis=0), (H, 1))
    if not len(enough):
        return med
    for r in np.flatnonzero(rows):
        k = enough[np.argmin(np.abs(enough - r))]
        med[r] = np.median(out[lo[k]:hi[k]][seen[lo[k]:hi[k]]], axis=0)
    return med


def _fill_holes(img: np.ndarray, cnt: np.ndarray, max_fill_px: int = 10, solid_px: int | None = None) -> Texture:
    """Mean per pixel; an empty pixel *between* filled ones -- a filled one
    within ``max_fill_px`` on both sides, across or up -- takes the nearest
    filled colour (a gap between returns); any other takes the face's median
    colour (a stretch of wall nothing saw), so unseen wall reads as plain
    surface. A lone point stays a point: nearest-fill in every direction had
    grown each one into a disc up to 3 m across, round dots on the walls. The
    result records which of the three each pixel is (:class:`Texture`).

    With ``solid_px`` (walls): a seen texel counts only where at least half
    the ``solid_px`` square round it was seen -- a scattered speck is dropped
    as unseen, since bridging specks drew lines and growing them drew dots --
    and colour then grows up to ``max_fill_px`` out from solid patches about
    2 m across or more (a smaller blob grew a ring: a dot again). Unseen wall
    takes the median colour of its own floor (the rows within ``max_fill_px``,
    specks included), or of the nearest floor seen enough -- not the whole
    face's, which shadow can darken."""
    from scipy.ndimage import (
        distance_transform_edt,
        label,
        maximum_filter,
        uniform_filter,
    )

    filled = cnt > 0
    if not filled.any():
        return Texture(np.full(img.shape, 128, dtype=np.uint8), np.full(cnt.shape, PLAIN, dtype=np.uint8))
    out = np.zeros_like(img)
    out[filled] = img[filled] / cnt[filled][:, None]
    if solid_px is None:
        plain = np.tile(np.median(out[filled], axis=0), (len(filled), 1))
        near = _gap_between(filled, max_fill_px)
    else:
        core = uniform_filter(filled.astype(np.float64), solid_px, mode="nearest") >= 0.5
        solid = filled & maximum_filter(core, solid_px)        # a core and the seen rim round it
        if not solid.any():                    # specks only: a plain face in their colour
            return Texture(np.broadcast_to(np.clip(np.median(out[filled], axis=0), 0, 255).astype(np.uint8),
                                           img.shape).copy(), np.full(cnt.shape, PLAIN, dtype=np.uint8))
        seen_any, filled = filled, solid
        lab, n = label(solid, structure=np.ones((3, 3)))
        big = np.flatnonzero(np.bincount(lab.ravel(), minlength=n + 1)[1:] >= (2 * solid_px) ** 2) + 1
        grows = np.isin(lab, big)              # patches about 2 m across or more: a blob grows no ring
        near = ~filled & (distance_transform_edt(~grows) <= max_fill_px) if grows.any() else np.zeros_like(filled)
        # a floor's colour from everything seen on it, specks too: untrusted one by one, fair together
        plain = _row_medians(out, seen_any, max_fill_px, (~filled & ~near).any(axis=1))
    _, (iy, ix) = distance_transform_edt(~filled, return_indices=True)
    out = out[iy, ix]
    blank = ~filled & ~near
    out[blank] = np.broadcast_to(plain[:, None, :], out.shape)[blank]
    source = np.where(filled, SEEN, np.where(near, NEAR, PLAIN)).astype(np.uint8)
    # image rows run top-down; our v runs bottom-up
    return Texture(np.clip(out[::-1], 0, 255).astype(np.uint8), source[::-1].copy())


def tile_geometry(building: Path, xyz: np.ndarray, poly, cxyz: np.ndarray, crgb: np.ndarray,
                  texel_m: float = TEXEL_M, max_offset_m: float = 2.0):
    """The locally fitted wall tiles the colourise resamples (12 m squares
    fitted to the wall returns, kept where they lie on a plane), each as a
    textured rectangle in its own plane. These are the wall where the LiDAR
    saw it directly; the profile bands stand in where it did not."""
    from shapely import contains_xy

    from kyfromabove_3d.colourise.walls import resample_walls
    from kyfromabove_3d.features import split_surface_zones

    ring = np.array(poly.exterior.coords)
    on_building = contains_xy(poly.buffer(max_offset_m), xyz[:, 0], xyz[:, 1])
    b_xyz = xyz[on_building]
    zones = split_surface_zones(b_xyz)
    if zones is None:
        return []
    height_m = float(np.percentile(b_xyz[:, 2], 99) - b_xyz[:, 2].min())
    _, _, _, _, _, tiles_by_face = resample_walls(b_xyz, zones.wall, ring, height_m, 0.15, 4.0, max_offset_m)
    out = []
    for face, tiles in tiles_by_face.items():
        for t in tiles:
            if "t1" not in t:
                continue
            c, n, t1, t2 = t["centroid"], t["normal"], t["t1"], t["t2"]
            (a0, a1), (b0, b1) = t["a_range"], t["b_range"]
            if a1 - a0 < 0.5 or b1 - b0 < 0.5:
                continue
            corners = np.array([c + a0 * t1 + b0 * t2, c + a1 * t1 + b0 * t2,
                                c + a1 * t1 + b1 * t2, c + a0 * t1 + b1 * t2])
            # texture: the coloured points within 0.3 m of the tile's plane
            rel = cxyz - c
            d = rel @ n
            ta, tb = rel @ t1, rel @ t2
            m = (np.abs(d) < 0.3) & (ta >= a0 - 0.2) & (ta <= a1 + 0.2) & (tb >= b0 - 0.2) & (tb <= b1 + 0.2)
            m &= crgb.sum(axis=1) > 15
            W = max(2, min(MAX_TEX_PX, int(np.ceil((a1 - a0) / texel_m))))
            H = max(2, min(MAX_TEX_PX, int(np.ceil((b1 - b0) / texel_m))))
            img = np.zeros((H, W, 3)); cnt = np.zeros((H, W))
            if m.any():
                u = np.clip(((ta[m] - a0) / (a1 - a0) * W).astype(int), 0, W - 1)
                v = np.clip(((tb[m] - b0) / (b1 - b0) * H).astype(int), 0, H - 1)
                np.add.at(img, (v, u), crgb[m].astype(np.float64)); np.add.at(cnt, (v, u), 1.0)
            tex = _fill_holes(img, cnt, solid_px=_solid_px(texel_m))
            uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
            out.append((_orient(corners, n), n, uv, tex, face, int(m.sum())))
    return out


def quad_texture_from_cameras(corners: np.ndarray, normal: np.ndarray, cams, texel_m: float = 0.3,
                              min_facing_cos: float = 0.15, arc: "ArcWall | None" = None):
    """A texture for one rectangle sampled straight from the aerial frames
    through their camera models (the same test as the roof), for a wall the
    profile did not model and so has no coloured points to rasterise. With
    ``arc`` the rectangle is in the arc's unrolled frame and every texel is
    sampled where it lands on the circle, facing outward there: one image
    for the whole curve rather than one per piece."""
    from kyfromabove_3d.colourise.colour import sample_colour

    e1 = corners[1] - corners[0]; e2 = corners[3] - corners[0]
    L1, L2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    if L1 < 1e-6 or L2 < 1e-6:
        return None, 0
    W = max(2, min(MAX_TEX_PX, int(np.ceil(L1 / texel_m))))
    H = max(2, min(MAX_TEX_PX, int(np.ceil(L2 / texel_m))))
    u = (np.arange(W) + 0.5) / W; v = (np.arange(H) + 0.5) / H
    U, V = (x.ravel()[:, None] for x in np.meshgrid(u, v))
    # each texel where the drawn face puts it: the four corners blended, as
    # roll_quads cuts the face into its strips. On sloping ground the face is
    # no parallelogram, and corners 0, 1 and 3 alone sampled its far top up
    # to the slope's drop off the wall
    c0, c1, c2, c3 = (c[None, :] for c in corners)
    pts = (1 - V) * ((1 - U) * c0 + U * c1) + V * ((1 - U) * c3 + U * c2)
    nrm = np.repeat(normal[None, :], len(pts), axis=0)
    if arc is not None:
        pts, nrm = arc.to_world_3d(pts, nrm)
    if not cams:
        return None, 0
    rgb, seen, _, _ = sample_colour(pts, nrm, cams, min_facing_cos=min_facing_cos, occlusion_slack_m=1.5)
    img = np.zeros((H * W, 3)); cnt = np.zeros(H * W)
    img[seen > 0] = rgb[seen > 0]; cnt[seen > 0] = 1
    if not cnt.any():
        return None, 0
    return _fill_holes(img.reshape(H, W, 3), cnt.reshape(H, W), max_fill_px=max(2, int(1.5 / texel_m)),
                       solid_px=_solid_px(texel_m)), int((seen > 0).sum())


def texel_sources(images) -> dict:
    """The share of texels by where their colour came from, over the images
    that record it (:class:`Texture`): ``{"seen", "near", "plain"}``, and the
    texel count. The measured product states its own coverage."""
    counts = np.zeros(4, dtype=np.int64)
    for img in images:
        src = getattr(img, "source", None)
        if src is not None:
            counts += np.bincount(src.ravel(), minlength=4)[:4]
    n = int(counts[1:].sum())
    return {"n": n, **{k: round(float(counts[v]) / max(n, 1), 4) for k, v in (("seen", SEEN), ("near", NEAR), ("plain", PLAIN))}}


class TextureAtlas:
    """Shelf-packs many small textures into a few pages, so a building is a
    handful of draw calls rather than one per face and tile. ``add`` takes an
    image and returns a handle; ``pack`` lays the pages out; ``uv`` maps a
    handle's local (u, v) in [0, 1] to page coordinates."""

    def __init__(self, page_px: int = 4096, gap_px: int = 2):
        self.page_px, self.gap = page_px, gap_px
        self.images = []
        self.meta = []
        self.places = {}
        self.pages = []

    def add(self, image: np.ndarray, meta: dict | None = None) -> int:
        """``meta``: what the image is of (the face, its extent), for the
        completed product; the image keeps its per-pixel source if it is a
        :class:`Texture`."""
        self.images.append(image)
        self.meta.append(meta or {})
        return len(self.images) - 1

    def pack(self):
        order = sorted(range(len(self.images)), key=lambda i: -self.images[i].shape[0])
        pages = []                                      # [ (x_cursor, y_cursor, shelf_h, canvas) ]
        for i in order:
            h, w = self.images[i].shape[:2]
            h, w = min(h, self.page_px), min(w, self.page_px)
            placed = False
            for pi, pg in enumerate(pages):
                x, y, sh, canvas = pg
                if x + w + self.gap <= self.page_px and y + h + self.gap <= self.page_px:
                    pass
                elif y + sh + self.gap + h + self.gap <= self.page_px:
                    x, y, sh = 0, y + sh + self.gap, 0
                else:
                    continue
                canvas[y:y + h, x:x + w] = self.images[i][:h, :w]
                self.places[i] = (pi, x, y, w, h)
                pages[pi] = (x + w + self.gap, y, max(sh, h), canvas)
                placed = True
                break
            if not placed:
                canvas = np.full((self.page_px, self.page_px, 3), 128, dtype=np.uint8)
                canvas[:h, :w] = self.images[i][:h, :w]
                self.places[i] = (len(pages), 0, 0, w, h)
                pages.append((w + self.gap, 0, h, canvas))
        # trim each page to what it used (to a multiple of 4: block compression)
        self.pages = []
        for pi, (x, y, sh, canvas) in enumerate(pages):
            used_w = _mult4(max(p[1] + p[3] for p in self.places.values() if p[0] == pi))
            used_h = _mult4(max(p[2] + p[4] for p in self.places.values() if p[0] == pi))
            self.pages.append(canvas[:used_h, :used_w].copy())
        return self.pages

    def uv(self, handle: int, uv_local: np.ndarray) -> tuple[int, np.ndarray]:
        """(page, uv on the page) for uv_local with v up from the image's bottom."""
        pi, x, y, w, h = self.places[handle]
        H, W = self.pages[pi].shape[:2]
        u = (x + uv_local[:, 0] * w) / W
        # the image row 0 is its top; v_local = 1 is the top
        v_img_top = (y + (1.0 - uv_local[:, 1]) * h) / H
        return pi, np.column_stack([u, 1.0 - v_img_top])
