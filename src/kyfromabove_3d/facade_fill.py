"""The completed product: fill what no frame saw, from the building itself.

The measured product (product 2) colours a texel only where a frame saw
it; a small gap takes the nearest seen colour, and a stretch of wall
nothing saw reads plain. The completed product (product 3) fills those
plain texels by *invention* -- but invention learned only from the same
building's own frames and LiDAR, never from outside images or a generic
prior. In order, per wall texture:

1. **twin** -- a hidden face the LiDAR profile found to be the mirror of
   the face opposite takes that face's seen texels, mirrored along the
   edge, at the same heights (off: it failed the held-out test, see
   :data:`POLICY`);
2. **rhythm** -- the facade's own period across (a bay, 1-8 m) and up (a
   storey, 2.5-6 m), found from the autocorrelation of its seen texels;
   a plain texel takes the seen texel a whole number of periods away;
3. **patch** -- what is left, near the seen part, takes blocks copied from
   the best-matching seen part of the same texture, a few metres in at
   most; beyond that it stays plain.

Every texel keeps its source (:data:`SEEN` ... :data:`PATCH`), so the
viewer can show what was invented and the tileset states the shares.
:func:`holdout` scores a method honestly: it hides seen texels, invents
them, and compares the invention with what the frames saw there.

Pure functions over numpy; images are (H, W, 3) uint8, rows top-down,
columns along the face; a texel is ``texel_m`` on a side.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

#: where a texel's colour came from
SEEN, NEAR, PLAIN, TWIN, RHYTHM, PATCH = 1, 2, 3, 4, 5, 6
NAMES = {SEEN: "seen", NEAR: "near", PLAIN: "plain", TWIN: "twin", RHYTHM: "rhythm", PATCH: "patch"}
KNOWN = (SEEN, NEAR)


def _luma(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float64)
    return f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114


def _highpass(g: np.ndarray, mask: np.ndarray, sigma_px: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    m = mask.astype(np.float64)
    num = gaussian_filter(np.where(mask, g, 0.0), sigma_px, mode="constant")
    den = gaussian_filter(m, sigma_px, mode="constant")
    return np.where(mask, g - num / np.maximum(den, 1e-6), 0.0)


def _shift_corr(hp: np.ndarray, mask: np.ndarray, k: int, axis: int) -> float:
    """Correlation of the texture with itself shifted ``k`` texels along
    ``axis`` (1 across, 0 up), over the texels known in both."""
    if axis == 1:
        a, b, ma, mb = hp[:, :-k], hp[:, k:], mask[:, :-k], mask[:, k:]
    else:
        a, b, ma, mb = hp[:-k], hp[k:], mask[:-k], mask[k:]
    m = ma & mb
    if m.sum() < 50:
        return float("nan")
    x, y = a[m], b[m]
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 1e-9 else float("nan")


def period(img: np.ndarray, known: np.ndarray, texel_m: float, axis: int,
           lo_m: float, hi_m: float, min_corr: float = 0.35) -> Optional[int]:
    """The facade's repeat along ``axis`` in texels -- the smallest shift
    between ``lo_m`` and ``hi_m`` at which the known texels correlate with
    themselves at least ``min_corr`` and nearly as well as at the best shift
    (a lattice also correlates at twice its period) -- or None."""
    hp = _highpass(_luma(img), known, sigma_px=max(2.0, 1.5 / texel_m))
    ks = range(max(int(round(lo_m / texel_m)), 2), int(round(hi_m / texel_m)) + 1)
    r = {k: _shift_corr(hp, known, k, axis) for k in ks if k < known.shape[axis] // 2}
    r = {k: v for k, v in r.items() if np.isfinite(v)}
    if not r:
        return None
    best = max(r.values())
    if best < min_corr:
        return None
    for k in sorted(r):
        left, right = r.get(k - 1, -1.0), r.get(k + 1, -1.0)
        if r[k] >= 0.85 * best and r[k] >= left and r[k] >= right:
            return k
    return None


def rhythm_fill(img: np.ndarray, source: np.ndarray, texel_m: float, reach: int = 6,
                bay_m=(1.0, 8.0), storey_m=(2.5, 6.0)) -> Tuple[np.ndarray, np.ndarray]:
    """Fill PLAIN texels from the known texel a whole number of periods away
    (the nearest such). Returns new (img, source); unchanged when the face
    shows no clear period either way."""
    known = np.isin(source, KNOWN)
    if known.sum() < 200 or not (source == PLAIN).any():
        return img, source
    ph = period(img, known, texel_m, 1, *bay_m)
    pv = period(img, known, texel_m, 0, *storey_m)
    if ph is None and pv is None:
        return img, source
    img, source = img.copy(), source.copy()
    H, W = source.shape
    offs = [(i * (pv or 0), j * (ph or 0)) for i in range(-reach, reach + 1) for j in range(-reach, reach + 1)
            if (i or j) and (pv or not i) and (ph or not j)]
    offs = sorted(set(offs), key=lambda o: o[0] ** 2 + o[1] ** 2)
    todo = source == PLAIN
    for dv, du in offs:
        if not todo.any():
            break
        ys, xs = np.nonzero(todo)
        sy, sx = ys + dv, xs + du
        ok = (sy >= 0) & (sy < H) & (sx >= 0) & (sx < W)
        ys, xs, sy, sx = ys[ok], xs[ok], sy[ok], sx[ok]
        take = known[sy, sx]
        img[ys[take], xs[take]] = img[sy[take], sx[take]]
        source[ys[take], xs[take]] = RHYTHM
        todo[ys[take], xs[take]] = False
    return img, source


def patch_fill(img: np.ndarray, source: np.ndarray, texel_m: float, block_m: float = 1.0,
               max_depth_m: float = 3.0, candidates: int = 400, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Fill PLAIN texels within ``max_depth_m`` of the known part with
    ``block_m`` blocks copied from the known part of the same image: each
    block on the edge of what is filled takes the known block whose margin
    (half a block all round) matches best where it overlaps known texels
    (sum of squared differences), among ``candidates`` sampled -- image
    quilting (Efros and Freeman, 2001). Deeper texels stay PLAIN."""
    from scipy.ndimage import distance_transform_edt
    rng = np.random.default_rng(seed)
    img, source = img.copy(), source.copy()
    have = np.isin(source, KNOWN)
    if have.sum() < 100 or not (source == PLAIN).any():
        return img, source
    H, W = source.shape
    b = max(int(round(block_m / texel_m)), 3)
    o = max(b // 2, 1)                         # the margin each block is matched on
    B = b + 2 * o
    depth = distance_transform_edt(~have)
    target = (source == PLAIN) & (depth * texel_m <= max_depth_m)
    # candidate windows (block plus margin): wholly known
    if H < B or W < B:
        return img, source
    ky, kx = np.nonzero(have[: H - B + 1, : W - B + 1])
    if not len(ky):
        return img, source
    pick = rng.choice(len(ky), size=min(candidates * 4, len(ky)), replace=False)
    cands = [(y, x) for y, x in zip(ky[pick], kx[pick]) if have[y:y + B, x:x + B].all()][:candidates]
    if not cands:
        return img, source
    f = img.astype(np.float64)
    stack = np.stack([f[y:y + B, x:x + B] for y, x in cands])          # (C, B, B, 3)
    for _ in range(int(np.ceil(max_depth_m / (b * texel_m))) + 2):
        filled_now = np.isin(source, KNOWN + (PATCH,))
        front = target & ~filled_now
        if not front.any():
            break
        # a block on the fill front: texels to fill in it, known texels in its margin
        for y0 in range(0, H, b):
            for x0 in range(0, W, b):
                blk = front[y0:y0 + b, x0:x0 + b]
                if not blk.any():
                    continue
                wy0, wx0 = max(y0 - o, 0), max(x0 - o, 0)
                wy1, wx1 = min(y0 + b + o, H), min(x0 + b + o, W)
                known_w = filled_now[wy0:wy1, wx0:wx1]
                if not known_w.any():
                    continue
                cy, cx = wy0 - (y0 - o), wx0 - (x0 - o)             # the window's place in a candidate
                cand_w = stack[:, cy:cy + (wy1 - wy0), cx:cx + (wx1 - wx0)]
                d = (((cand_w - f[wy0:wy1, wx0:wx1][None]) ** 2).sum(axis=-1) * known_w).sum(axis=(1, 2))
                k = int(np.argmin(d))
                h, w = blk.shape
                new = stack[k, o:o + h, o:o + w]
                f[y0:y0 + h, x0:x0 + w][blk] = new[blk]
                img[y0:y0 + h, x0:x0 + w][blk] = np.clip(new[blk], 0, 255).astype(img.dtype)
                source[y0:y0 + h, x0:x0 + w][blk] = PATCH
    return img, source


def twin_fill(img: np.ndarray, source: np.ndarray, meta: dict, twin_img: np.ndarray,
              twin_source: np.ndarray, twin_meta: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Fill PLAIN texels of a face from its mirror twin's known texels: the
    twin read backwards along its edge (the face opposite runs the other
    way round the building), at the same heights."""
    H, W = source.shape
    tH, tW = twin_source.shape
    z0, z1 = meta["z"]; t0, t1 = twin_meta["z"]
    zs = z1 - (np.arange(H) + 0.5) / H * (z1 - z0)               # row -> height (rows top-down)
    trow = np.round((t1 - zs) / (t1 - t0) * tH - 0.5).astype(int)
    tcol = np.round((1.0 - (np.arange(W) + 0.5) / W) * tW - 0.5).astype(int)  # mirrored along
    img, source = img.copy(), source.copy()
    ok_r = (trow >= 0) & (trow < tH)
    R, C = np.meshgrid(np.where(ok_r, trow, 0), np.clip(tcol, 0, tW - 1), indexing="ij")
    tknown = np.isin(twin_source, KNOWN)[R, C] & ok_r[:, None]
    take = (source == PLAIN) & tknown
    img[take] = twin_img[R[take], C[take]]
    source[take] = TWIN
    return img, source


#: which fills run on which kind of texture -- set by the held-out test
#: (scripts/eval_fill.py; all of downtown, 16,759 hidden rectangles of seen
#: facade on 342 buildings, September 2026): a method runs only where it beat
#: the plain fill. Pattern agreement (plain = 0) / colour closer than plain:
#:   tile     rhythm then patch  0.50 / 79%
#:   profile  rhythm then patch  0.16 / 78%   (rhythm alone 0.27 / 80%)
#:   prism    rhythm then patch  0.22 / 64%   (a first sample of 34 rectangles
#:            had rhythm below plain; the full one does not)
#:   roof     patch              0.05 / 67%   (rhythm 50%: no better than plain)
#:   twin     -0.03 / 13%: the face opposite is lit from the other side and its
#:            windows do not line up -- off everywhere
POLICY: Dict[str, Tuple[str, ...]] = {
    "tile": ("rhythm", "patch"),
    "profile": ("rhythm", "patch"),
    "prism": ("rhythm", "patch"),
    "roof": ("patch",),
}


def complete_textures(images: Sequence[np.ndarray], metas: Sequence[dict], texel_m: float,
                      twins: Optional[Dict[int, int]] = None,
                      policy: Optional[Dict[str, Tuple[str, ...]]] = None):
    """The completed copy of every wall texture (those carrying a ``source``):
    returns [(img, source)] in the same order. ``twins`` maps a face to the
    face it mirrors (the colourise's mirror report); ``policy`` says which
    fills run on which kind of texture (default :data:`POLICY`). Measured
    texels are never changed."""
    policy = POLICY if policy is None else policy
    by_face = {}
    for i, m in enumerate(metas):
        if m.get("kind") == "profile":
            by_face.setdefault(m["face"], i)
    out = []
    for img, meta in zip(images, metas):
        src = getattr(img, "source", None)
        im = np.asarray(img)
        if src is None:
            out.append((im, None)); continue
        s = src.copy()
        methods = policy.get(meta.get("kind"), ())
        if "twin" in methods and twins and meta.get("face") in twins:
            j = by_face.get(twins[meta["face"]])
            if j is not None and getattr(images[j], "source", None) is not None:
                im, s = twin_fill(im, s, meta, np.asarray(images[j]), images[j].source, metas[j])
        if "rhythm" in methods:
            im, s = rhythm_fill(im, s, texel_m)
        if "patch" in methods:
            im, s = patch_fill(im, s, texel_m)
        out.append((im, s))
    return out


def shares(sources: Sequence[Optional[np.ndarray]]) -> dict:
    """The share of texels by source over ``sources``, and their count."""
    counts = np.zeros(max(NAMES) + 1, dtype=np.int64)
    for s in sources:
        if s is not None:
            counts += np.bincount(s.ravel(), minlength=len(counts))[:len(counts)]
    n = int(counts[1:].sum())
    return {"n": n, **{NAMES[k]: round(float(counts[k]) / max(n, 1), 4) for k in NAMES}}


def holdout(img: np.ndarray, source: np.ndarray, texel_m: float, fill: Callable,
            gap_m: Tuple[float, float] = (3.0, 3.0), trials: int = 4, seed: int = 0):
    """Score a fill method where the truth is known: hide ``trials``
    rectangles of seen texels (``gap_m`` across and up), fill them with
    ``fill(img, source) -> (img, source)``, and compare with what was seen.
    Returns a list of {pattern agreement (high-passed brightness
    correlation), mean colour error, share filled} per trial, and the same
    for the plain fill (the face's median colour) it would replace."""
    from .texture_check import masked_ncc
    rng = np.random.default_rng(seed)
    H, W = source.shape
    h, w = max(int(round(gap_m[1] / texel_m)), 4), max(int(round(gap_m[0] / texel_m)), 4)
    seen = source == SEEN
    out = []
    for _ in range(trials * 10):
        if len(out) >= trials or H <= h + 2 or W <= w + 2:
            break
        y0, x0 = int(rng.integers(0, H - h)), int(rng.integers(0, W - w))
        box = np.zeros_like(seen); box[y0:y0 + h, x0:x0 + w] = True
        if not seen[box].all() or seen[~box].sum() < 400:
            continue
        s2 = source.copy(); s2[box] = PLAIN
        im2 = img.copy(); med = np.median(img[seen & ~box], axis=0); im2[box] = med
        filled, fs = fill(im2, s2)
        got = box & (fs != PLAIN)
        truth_hp = _highpass(_luma(img), np.ones_like(seen), 2.0)
        fill_hp = _highpass(_luma(filled), np.ones_like(seen), 2.0)
        out.append({
            "share_filled": float(got.sum() / box.sum()),
            "ncc": masked_ncc(truth_hp, fill_hp, got) if got.sum() > 20 else float("nan"),
            "colour_err": float(np.abs(filled[box].astype(float) - img[box].astype(float)).mean()),
            "plain_colour_err": float(np.abs(im2[box].astype(float) - img[box].astype(float)).mean()),
        })
    return out
