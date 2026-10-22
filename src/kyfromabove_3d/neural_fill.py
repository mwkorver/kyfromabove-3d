"""The fourth product: fill what no frame saw by borrowing from the walls
the frames did see, with numpy and scipy only.

1. **Context search** (:func:`retrieve_candidates`): around every ``PLAIN``
   region, the seen ring of texels around it (the halo) is compared in
   high-pass brightness against every position on the same texture and, when
   the same face has no strong match, on the building's other seen faces
   (``r.face != face``). The comparison is a masked normalised
   cross-correlation done with FFTs; each texture's transforms are computed
   once and reused for every hole. For short wall tiles where a whole block
   cannot shift by a full storey, a two-strip vertical split is also tried.
2. **Brightness matching**: each borrowed patch is brought to the hole's own
   light (mean and damped contrast over the shared halo), and a smooth
   illumination field (:func:`harmonic_base`) fills the smallest holes.
3. **Blend**: the three best patches are averaged, a better match counting
   for more. (The first version fed them to a small per-building neural
   network; on held-out walls the plain weighted average scored better --
   pattern agreement 0.56 against 0.43 on 51 buildings, 0.82 against 0.78 on
   the three landmarks -- so it was dropped, and PyTorch with it.)

Invented texels are marked :data:`NEURAL` (7) in ``source`` (the product's
name is kept); ``SEEN`` and ``NEAR`` texels are never modified.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import gaussian_filter, label

from .facade_fill import KNOWN, NEURAL, PLAIN, SEEN, _highpass, _luma, patch_fill, rhythm_fill


def harmonic_base(img: np.ndarray, known: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    """Smooth local illumination field interpolated from ``known`` texels."""
    m = known.astype(np.float64)
    den = np.maximum(gaussian_filter(m, sigma, mode="nearest"), 1e-6)
    base = np.zeros_like(img, dtype=np.float64)
    for c in range(3):
        num = gaussian_filter(np.where(known, img[..., c].astype(np.float64), 0.0), sigma, mode="nearest")
        base[..., c] = num / den
    return base


def _holdout_boxes(img: np.ndarray, source: np.ndarray, texel_m: float,
                   gap_m: Tuple[float, float] = (3.0, 3.0), trials: int = 3, seed: int = 0) -> List[Tuple[int, int, int, int]]:
    """Exact box coordinates hidden by :func:`kyfromabove_3d.facade_fill.holdout`,
    used when ``exclude_holdout=True`` so held-out test evaluations have zero
    leakage into the reference bank."""
    rng = np.random.default_rng(seed)
    H, W = source.shape
    h, w = max(int(round(gap_m[1] / texel_m)), 4), max(int(round(gap_m[0] / texel_m)), 4)
    seen = source == SEEN
    boxes: List[Tuple[int, int, int, int]] = []
    for _ in range(trials * 10):
        if len(boxes) >= trials or H <= h + 2 or W <= w + 2:
            break
        y0, x0 = int(rng.integers(0, H - h)), int(rng.integers(0, W - w))
        box = np.zeros_like(seen)
        box[y0 : y0 + h, x0 : x0 + w] = True
        if not seen[box].all() or seen[~box].sum() < 400:
            continue
        boxes.append((y0, y0 + h, x0, x0 + w))
    return boxes


class PreparedSource:
    """One facade texture ready to be searched: its high-pass brightness, its
    known mask, a summed-area table of the mask, and (per halo width) the
    Fourier transforms of the zero-padded maps the search correlates."""

    def __init__(self, img: np.ndarray, known: np.ndarray, face=None, kind: str = ""):
        self.img = np.asarray(img, dtype=np.float32)
        self.known = np.asarray(known, dtype=bool)
        self.face = face
        self.kind = kind
        self.H, self.W = self.known.shape
        self.hp = _highpass(_luma(self.img), self.known, sigma_px=4.0).astype(np.float32)
        k = self.known.astype(np.float64)
        self.sat = np.pad(k.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        self._ffts: Dict[int, tuple] = {}

    def spectra(self, pad: int):
        """``(shape, F[known], F[hp*known], F[hp^2*known])`` of the maps
        zero-padded by ``pad`` on every side, at a fast FFT size."""
        if pad not in self._ffts:
            hp = self.hp.astype(np.float64) * self.known
            maps = [self.known.astype(np.float64), hp, hp * self.hp]
            shape = (sfft.next_fast_len(self.H + 2 * pad, real=True), sfft.next_fast_len(self.W + 2 * pad, real=True))
            fs = [sfft.rfft2(np.pad(a, pad), s=shape) for a in maps]
            self._ffts[pad] = (shape, *fs)
        return self._ffts[pad]

    def box_count(self, rh: int, rw: int) -> np.ndarray:
        """Known texels in every ``rh x rw`` window, by top-left corner."""
        s = self.sat
        return s[rh:, rw:] - s[:-rh, rw:] - s[rh:, :-rw] + s[:-rh, :-rw]


def _correlate(spec: np.ndarray, kernel_spec: np.ndarray, shape, out_shape) -> np.ndarray:
    """``out[i, j] = sum(padded_map[i + a, j + b] * kernel[a, b])`` for the
    windows lying wholly inside the padded map (no wrap-around reaches them)."""
    return sfft.irfft2(spec * kernel_spec, s=shape)[: out_shape[0], : out_shape[1]]


def _retrieve_box_candidates(
    target_img: np.ndarray,
    target_known: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    self_src: PreparedSource,
    ref_sources: Sequence[PreparedSource],
    top_k: int = 3,
    pad: int = 10,
) -> List[Tuple[float, np.ndarray]]:
    H, W = target_known.shape
    rh, rw = y1 - y0, x1 - x0
    wh, ww = rh + 2 * pad, rw + 2 * pad

    ctx_mask = np.zeros((wh, ww), dtype=bool)
    t_hp = np.zeros((wh, ww), dtype=np.float32)
    t_rgb = np.zeros((wh, ww, 3), dtype=np.float32)

    wy0, wy1 = max(0, y0 - pad), min(H, y1 + pad)
    wx0, wx1 = max(0, x0 - pad), min(W, x1 + pad)
    dy0, dy1 = wy0 - (y0 - pad), wy1 - (y0 - pad)
    dx0, dx1 = wx0 - (x0 - pad), wx1 - (x0 - pad)

    ctx_mask[dy0:dy1, dx0:dx1] = target_known[wy0:wy1, wx0:wx1]
    ctx_mask[pad : pad + rh, pad : pad + rw] = False
    if ctx_mask.sum() < 12:
        med = np.median(target_img[target_known], axis=0) if target_known.any() else np.array([128.0, 128.0, 128.0])
        return [(0.0, np.broadcast_to(med, (rh, rw, 3)).copy())]

    t_hp[dy0:dy1, dx0:dx1] = self_src.hp[wy0:wy1, wx0:wx1]
    t_rgb[dy0:dy1, dx0:dx1] = target_img[wy0:wy1, wx0:wx1]

    t_hp_vals = t_hp[ctx_mask].astype(np.float64)
    t_hp_mean = float(t_hp_vals.mean())
    t_hp_norm = float(np.sqrt(((t_hp_vals - t_hp_mean) ** 2).sum())) + 1e-6
    w_hp = np.where(ctx_mask, (t_hp - t_hp_mean) / t_hp_norm, 0.0)
    m_f = ctx_mask.astype(np.float64)
    n_hole = rh * rw
    n_ctx = float(ctx_mask.sum())

    candidates: List[Tuple[float, np.ndarray]] = []

    def _search_one(src_obj: PreparedSource, is_self: bool):
        if src_obj.H < rh or src_obj.W < rw:
            return
        out_shape = (src_obj.H - rh + 1, src_obj.W - rw + 1)
        shape, f_k, f_hp, f_hp2 = src_obj.spectra(pad)
        f_m = np.conj(sfft.rfft2(m_f, s=shape))
        cnt_s = np.rint(_correlate(f_k, f_m, shape, out_shape))
        valid_hole = src_obj.box_count(rh, rw) >= n_hole
        min_ctx = max(16.0, (0.30 if is_self else 0.65) * n_ctx)
        valid = valid_hole & (cnt_s >= min_ctx)
        if is_self:
            vy0, vy1 = max(0, y0 - rh // 2), min(valid.shape[0], y0 + rh // 2 + 1)
            vx0, vx1 = max(0, x0 - rw // 2), min(valid.shape[1], x0 + rw // 2 + 1)
            valid[vy0:vy1, vx0:vx1] = False
        n_valid = int(valid.sum())
        if not n_valid:
            return

        cross = _correlate(f_hp, np.conj(sfft.rfft2(w_hp, s=shape)), shape, out_shape)
        sum_s2 = _correlate(f_hp2, f_m, shape, out_shape)
        sum_s = _correlate(f_hp, f_m, shape, out_shape)
        cnt_clamped = np.maximum(cnt_s, 1.0)
        var_s = np.maximum(sum_s2 - sum_s ** 2 / cnt_clamped, 1e-6)
        score_map = cross / np.sqrt(var_s) * np.sqrt(cnt_clamped / n_ctx)

        if is_self:
            if x0 < score_map.shape[1]:
                score_map[:, x0] += 0.18
            if y0 < score_map.shape[0]:
                score_map[y0, :] += 0.12
        else:
            score_map -= 0.08

        score_map[~valid] = -1e9
        flat = score_map.ravel()
        k_pick = min(top_k * 3 if is_self else 2, n_valid)
        top_idx = np.argpartition(-flat, k_pick - 1)[:k_pick]
        top_idx = top_idx[np.argsort(-flat[top_idx], kind="stable")]
        W_out = score_map.shape[1]

        picked_coords: List[Tuple[int, int]] = []
        for idx_flat in top_idx.tolist():
            val = float(flat[idx_flat])
            if val < -100:
                continue
            cy, cx = idx_flat // W_out, idx_flat % W_out
            if any(abs(cy - py) <= 3 and abs(cx - px) <= 3 for py, px in picked_coords):
                continue
            picked_coords.append((cy, cx))

            sy0, sy1 = max(0, cy - pad), min(src_obj.H, cy + rh + pad)
            sx0, sx1 = max(0, cx - pad), min(src_obj.W, cx + rw + pad)
            ks_y0, ks_y1 = sy0 - (cy - pad), sy1 - (cy - pad)
            ks_x0, ks_x1 = sx0 - (cx - pad), sx1 - (cx - pad)

            sub_ctx = ctx_mask[ks_y0:ks_y1, ks_x0:ks_x1] & src_obj.known[sy0:sy1, sx0:sx1]
            if sub_ctx.sum() < 10:
                continue
            s_halo = src_obj.img[sy0:sy1, sx0:sx1][sub_ctx]
            t_halo_sub = t_rgb[ks_y0:ks_y1, ks_x0:ks_x1][sub_ctx]

            s_mu = s_halo.mean(axis=0)
            s_std = np.clip(s_halo.std(axis=0), 2.0, None)
            t_mu_sub = t_halo_sub.mean(axis=0)
            t_std_sub = np.clip(t_halo_sub.std(axis=0), 2.0, None)

            gain = np.clip(t_std_sub / s_std, 0.7, 1.4)
            patch_raw = src_obj.img[cy : cy + rh, cx : cx + rw]
            patch_adj = (patch_raw - s_mu) * gain + t_mu_sub

            halo_adj = (s_halo - s_mu) * gain + t_mu_sub
            colour_mae = float(np.abs(halo_adj - t_halo_sub).mean())
            candidates.append((val - 0.008 * colour_mae, patch_adj))
            if len(picked_coords) >= (top_k if is_self else 2):
                break

    # 1. Search the target texture itself first
    _search_one(self_src, True)
    candidates.sort(key=lambda x: -x[0])

    # 2. Search the other faces unless the same face already matches strongly (>= 0.72)
    if not candidates or candidates[0][0] < 0.72:
        for r in ref_sources:
            if r.face != self_src.face or self_src.face is None:
                _search_one(r, False)
        candidates.sort(key=lambda x: -x[0])

    return candidates[:top_k]


def retrieve_candidates(
    target_img: np.ndarray,
    target_known: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    self_src: PreparedSource,
    ref_sources: Sequence[PreparedSource],
    top_k: int = 3,
    pad: int = 10,
) -> List[Tuple[float, np.ndarray]]:
    """Retrieve whole-box candidates, plus two-strip vertical split candidates on
    short wall tiles where ``H < rh + 28``."""
    cands = _retrieve_box_candidates(target_img, target_known, y0, y1, x0, x1, self_src, ref_sources, top_k=top_k, pad=pad)
    rh = y1 - y0
    if self_src.H < rh + 28 and rh >= 12:
        ymid = y0 + rh // 2
        top_c = _retrieve_box_candidates(target_img, target_known, y0, ymid, x0, x1, self_src, (), top_k=2, pad=8)
        bot_c = _retrieve_box_candidates(target_img, target_known, ymid, y1, x0, x1, self_src, (), top_k=2, pad=8)
        if top_c and bot_c:
            for k_idx in range(min(len(top_c), len(bot_c))):
                sc = 0.5 * (top_c[k_idx][0] + bot_c[k_idx][0]) + 0.04
                stitched = np.vstack([top_c[k_idx][1], bot_c[k_idx][1]])
                cands.append((sc, stitched))
        cands.sort(key=lambda x: -x[0])
    return cands[:top_k]


def reference_bank(
    images: Sequence[np.ndarray],
    metas: Sequence[dict],
    texel_m: float,
    exclude_holdout: bool = False,
    holdout_gap_m: Tuple[float, float] = (3.0, 3.0),
    holdout_trials: int = 3,
) -> List[PreparedSource]:
    """The building's seen profile walls (and their mirror images) that every
    hole may borrow from. With ``exclude_holdout`` the rectangles the held-out
    test will hide are hidden here too, so the test cannot see its answers."""
    refs: List[PreparedSource] = []
    for i, (raw_im, m) in enumerate(zip(images, metas)):
        src = getattr(raw_im, "source", None)
        if src is None or m.get("kind") != "profile":
            continue
        img = np.asarray(raw_im).copy()
        s = np.asarray(src).copy()
        if exclude_holdout and (s == SEEN).sum() * texel_m * texel_m >= 60.0:
            for by0, by1, bx0, bx1 in _holdout_boxes(img, s, texel_m, gap_m=holdout_gap_m, trials=holdout_trials, seed=i):
                s[by0:by1, bx0:bx1] = PLAIN
                img[by0:by1, bx0:bx1] = np.median(img[s == SEEN], axis=0)
        known = np.isin(s, KNOWN)
        if known.sum() >= 600:
            refs.append(PreparedSource(img, known, face=m.get("face"), kind="profile"))
            refs.append(PreparedSource(img[:, ::-1].copy(), known[:, ::-1].copy(), face=m.get("face"),
                                       kind="profile_mirror"))
    return refs


def neural_fill(
    img: np.ndarray,
    source: np.ndarray,
    texel_m: float,
    face=None,
    kind: str = "",
    ref_sources: Sequence[PreparedSource] = (),
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill ``PLAIN`` texels in ``(img, source)`` from the best-matching seen
    patches. Returns ``(new_img, new_source)`` with invented texels marked
    :data:`NEURAL` (7)."""
    known = np.isin(source, KNOWN)
    plain = source == PLAIN
    if not plain.any() or known.sum() < 50:
        return img, source

    if kind == "roof":
        r_img, r_src = patch_fill(img, source, texel_m)
        base_rgb = harmonic_base(img, known, sigma=max(3.0, 1.0 / texel_m)).astype(np.float32)
        out = img.astype(np.float32).copy()
        filled = (source == PLAIN) & (r_src != PLAIN)
        out[filled] = 0.65 * r_img[filled].astype(np.float32) + 0.35 * base_rgb[filled]
        out_src = source.copy()
        out_src[filled] = NEURAL
        return np.clip(out, 0, 255).astype(np.uint8), out_src

    out_img = img.astype(np.float32).copy()
    out_src = source.copy()
    base_rgb = harmonic_base(img, known, sigma=max(3.0, 1.2 / texel_m)).astype(np.float32)
    self_src = PreparedSource(img, known, face=face, kind=kind)

    regs, n_reg = label(plain)
    for r_id in range(1, n_reg + 1):
        rmask = regs == r_id
        if int(rmask.sum()) < 9:
            out_img[rmask] = np.clip(base_rgb[rmask], 0, 255)
            out_src[rmask] = NEURAL
            continue
        ys, xs = np.nonzero(rmask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        rh, rw = y1 - y0, x1 - x0
        if rh <= 128 and rw <= 128:
            cands = retrieve_candidates(img, known, y0, y1, x0, x1, self_src, ref_sources, top_k=3, pad=10)
            if not cands:
                continue
            while len(cands) < 3:
                cands.append(cands[-1])
            scores = np.array([c[0] for c in cands[:3]], dtype=np.float32)
            w = np.exp((scores - scores.max()) * 8.0)
            w /= w.sum()
            blended = sum(w[k] * cands[k][1] for k in range(3))

            sub_mask = rmask[y0:y1, x0:x1]
            out_img[y0:y1, x0:x1][sub_mask] = np.clip(blended[sub_mask], 0, 255)
            out_src[y0:y1, x0:x1][sub_mask] = NEURAL

    if (out_src == PLAIN).any():
        r_img, r_src = patch_fill(*rhythm_fill(out_img.astype(np.uint8), out_src, texel_m), texel_m)
        left = (out_src == PLAIN) & (r_src != PLAIN)
        # Harmonize fallback rhythm/patch with local harmonic illumination base
        out_img[left] = 0.80 * r_img[left].astype(np.float32) + 0.20 * base_rgb[left]
        out_src[left] = NEURAL

    return np.clip(out_img, 0, 255).astype(np.uint8), out_src


def complete_textures_neural(
    images: Sequence[np.ndarray],
    metas: Sequence[dict],
    texel_m: float,
    ref_sources: Optional[Sequence[PreparedSource]] = None,
) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Complete every wall/roof texture of a building with :func:`neural_fill`,
    borrowing from ``ref_sources`` (by default the building's own
    :func:`reference_bank`)."""
    if ref_sources is None:
        ref_sources = reference_bank(images, metas, texel_m)
    out: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
    for img, meta in zip(images, metas):
        src = getattr(img, "source", None)
        im = np.asarray(img)
        if src is None:
            out.append((im, None))
            continue
        im_filled, s_filled = neural_fill(im, src.copy(), texel_m, face=meta.get("face"),
                                          kind=meta.get("kind", ""), ref_sources=ref_sources)
        out.append((im_filled, s_filled))
    return out
