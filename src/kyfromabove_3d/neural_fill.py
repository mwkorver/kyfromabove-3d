"""The neural completed product: fill what no frame saw, using PyTorch
multi-view context matching, harmonic illumination adaptation, and a
building-focused residual ConvNet (:class:`FacadeRefineNet`) trained
self-supervised on the building's own seen views.

Why focused per-building learning works here:

1. **Padded 2D convolutional context matching** (:func:`retrieve_candidates`):
   around every ``PLAIN`` region, the known context halo is correlated in
   high-pass brightness space (``F.conv2d(..., padding=pad)``) against the
   rest of the same texture and, when same-face evidence is weak, against the
   building's other seen faces (``r.face != face``). For short wall tiles
   where a single block cannot shift by a full storey vertically, a two-strip
   vertical split is also evaluated.
2. **Radiometric / illumination transfer**: each retrieved candidate patch is
   adapted to the target hole's local boundary illumination (mean and damped
   contrast over the overlapping halo) plus a smooth harmonic illumination
   field (:func:`harmonic_base`), eliminating the sunlit-vs-shaded colour jump
   that broke raw pixel copying (:func:`kyfromabove_3d.facade_fill.twin_fill`).
3. **Self-supervised per-building ConvNet** (:class:`FacadeRefineNet`): a
   compact, static-shape-friendly PyTorch network trained on synthetic holes
   carved out of the building's own ``SEEN`` regions (optimizing high-pass
   pattern correlation + L1 colour loss) fuses the top candidate patches and
   harmonic base into the final fill. Marked as :data:`NEURAL` (7) in
   ``source``; ``SEEN`` and ``NEAR`` texels are never modified.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, label

from .facade_fill import KNOWN, NEURAL, PLAIN, SEEN, _highpass, _luma, patch_fill, rhythm_fill


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    leakage into reference banks or self-supervised training."""
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
    """Precomputed high-pass and mask tensors for one facade texture."""

    def __init__(self, img: np.ndarray, known: np.ndarray, face=None, kind: str = "",
                 device: Optional[torch.device] = None):
        self.device = device or default_device()
        self.img = np.asarray(img, dtype=np.float32)
        self.known = np.asarray(known, dtype=bool)
        self.face = face
        self.kind = kind
        self.H, self.W = self.known.shape
        luma = _luma(self.img)
        self.hp = _highpass(luma, self.known, sigma_px=4.0).astype(np.float32)
        self.sk_t = torch.from_numpy(self.known.astype(np.float32))[None, None].to(self.device)
        self.shp_t = torch.from_numpy((self.hp * self.known).astype(np.float32))[None, None].to(self.device)
        self.shp2_t = torch.from_numpy(((self.hp ** 2) * self.known).astype(np.float32))[None, None].to(self.device)


class FacadeRefineNet(nn.Module):
    """Compact residual ConvNet that fuses top-3 multi-view candidate patches
    with the local harmonic illumination base. Uses static-shape-compatible
    3x3 and 1x1 convolutions so it traces cleanly with ``torch.jit.trace`` or
    ``torch_neuronx.trace`` on AWS Inferentia2."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(12, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 7, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.0)
        with torch.no_grad():
            self.net[-1].bias[0] = 2.0
            self.net[-1].bias[1] = 0.8
            self.net[-1].bias[2] = 0.2
            self.net[-1].bias[3] = -0.5

    def forward(self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        x = torch.cat([(c1 - base) / 64.0, (c2 - base) / 64.0, (c3 - base) / 64.0, (base - 128.0) / 128.0], dim=1)
        out = self.net(x)
        w = F.softmax(out[:, :4], dim=1)
        res = torch.tanh(out[:, 4:]) * 20.0
        pred = w[:, 0:1] * c1 + w[:, 1:2] * c2 + w[:, 2:3] * c3 + w[:, 3:4] * base + res
        return torch.clamp(pred, 0.0, 255.0)


def _highpass_torch(rgb: torch.Tensor) -> torch.Tensor:
    luma = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    smooth = F.avg_pool2d(F.pad(luma, (2, 2, 2, 2), mode="replicate"), kernel_size=5, stride=1)
    return luma - smooth


def _ncc_loss_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    hp_p = _highpass_torch(pred).flatten(1)
    hp_t = _highpass_torch(target).flatten(1)
    hp_p = hp_p - hp_p.mean(dim=1, keepdim=True)
    hp_t = hp_t - hp_t.mean(dim=1, keepdim=True)
    num = (hp_p * hp_t).sum(dim=1)
    den = torch.sqrt((hp_p ** 2).sum(dim=1) * (hp_t ** 2).sum(dim=1) + 1e-6)
    return (1.0 - num / den).mean()


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
    device = self_src.device
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

    m_f = ctx_mask.astype(np.float32)
    t_hp_vals = t_hp[ctx_mask]
    t_hp_mean = float(t_hp_vals.mean())
    t_hp_norm = float(np.sqrt(((t_hp_vals - t_hp_mean) ** 2).sum())) + 1e-6
    w_hp = np.where(ctx_mask, (t_hp - t_hp_mean) / t_hp_norm, 0.0).astype(np.float32)

    wt_hp_t = torch.from_numpy(w_hp)[None, None].to(device)
    wt_m_t = torch.from_numpy(m_f)[None, None].to(device)
    wt_hole_t = torch.ones((1, 1, rh, rw), dtype=torch.float32, device=device)
    n_hole = float(rh * rw)
    n_ctx = float(ctx_mask.sum())

    candidates: List[Tuple[float, np.ndarray]] = []

    def _search_one(src_obj: PreparedSource, is_self: bool):
        if src_obj.H < rh or src_obj.W < rw:
            return
        valid_hole = F.conv2d(src_obj.sk_t, wt_hole_t)[0, 0] >= (n_hole - 0.5)
        cnt_s = F.conv2d(src_obj.sk_t, wt_m_t, padding=pad)[0, 0]
        min_ctx = max(16.0, (0.30 if is_self else 0.65) * n_ctx)
        valid = valid_hole & (cnt_s >= min_ctx)
        if is_self:
            vy0, vy1 = max(0, y0 - rh // 2), min(valid.shape[0], y0 + rh // 2 + 1)
            vx0, vx1 = max(0, x0 - rw // 2), min(valid.shape[1], x0 + rw // 2 + 1)
            valid[vy0:vy1, vx0:vx1] = False
        if not bool(valid.any().item()):
            return

        cross = F.conv2d(src_obj.shp_t, wt_hp_t, padding=pad)[0, 0]
        sum_s2 = F.conv2d(src_obj.shp2_t, wt_m_t, padding=pad)[0, 0]
        sum_s = F.conv2d(src_obj.shp_t, wt_m_t, padding=pad)[0, 0]
        cnt_clamped = torch.clamp(cnt_s, min=1.0)
        var_s = torch.clamp(sum_s2 - (sum_s ** 2) / cnt_clamped, min=1e-6)
        coverage = cnt_clamped / n_ctx
        ncc_map = (cross / torch.sqrt(var_s)) * torch.sqrt(coverage)

        score_map = ncc_map.clone()
        if is_self:
            if x0 < score_map.shape[1]:
                score_map[:, x0] += 0.18
            if y0 < score_map.shape[0]:
                score_map[y0, :] += 0.12
        else:
            score_map -= 0.08

        score_map[~valid] = -1e9
        flat_scores = score_map.view(-1)
        k_pick = min(top_k * 3 if is_self else 2, int(valid.sum().item()))
        top_vals, top_idx = torch.topk(flat_scores, k=k_pick)
        W_out = score_map.shape[1]

        picked_coords: List[Tuple[int, int]] = []
        for val, idx_flat in zip(top_vals.cpu().tolist(), top_idx.cpu().tolist()):
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
            combined_score = float(val) - 0.008 * colour_mae
            candidates.append((combined_score, patch_adj))
            if len(picked_coords) >= (top_k if is_self else 2):
                break

    # 1. Search the target texture itself first
    _search_one(self_src, True)
    candidates.sort(key=lambda x: -x[0])

    # 2. Search cross-face reference textures unless same-face match is already strong (>= 0.72)
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


def train_building_model(
    images: Sequence[np.ndarray],
    metas: Sequence[dict],
    texel_m: float,
    exclude_holdout: bool = False,
    holdout_gap_m: Tuple[float, float] = (3.0, 3.0),
    holdout_trials: int = 3,
    steps: int = 90,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[int, PreparedSource], List[PreparedSource], FacadeRefineNet]:
    """Build the building's multi-view reference bank and train a focused
    per-building :class:`FacadeRefineNet` on synthetic holes sampled from its
    ``SEEN`` texels."""
    device = device or default_device()
    clean_sources: Dict[int, PreparedSource] = {}
    ref_list: List[PreparedSource] = []

    for i, (raw_im, m) in enumerate(zip(images, metas)):
        src = getattr(raw_im, "source", None)
        if src is None:
            continue
        img = np.asarray(raw_im).copy()
        s = np.asarray(src).copy()
        if exclude_holdout and (s == SEEN).sum() * texel_m * texel_m >= 60.0:
            for by0, by1, bx0, bx1 in _holdout_boxes(img, s, texel_m, gap_m=holdout_gap_m, trials=holdout_trials, seed=i):
                s[by0:by1, bx0:bx1] = PLAIN
                med = np.median(img[s == SEEN], axis=0)
                img[by0:by1, bx0:bx1] = med

        known = np.isin(s, KNOWN)
        ps = PreparedSource(img, known, face=m.get("face"), kind=m.get("kind", ""), device=device)
        clean_sources[i] = ps
        if m.get("kind") == "profile" and known.sum() >= 600:
            ref_list.append(ps)
            ref_list.append(PreparedSource(img[:, ::-1].copy(), known[:, ::-1].copy(),
                                           face=m.get("face"), kind="profile_mirror", device=device))

    rng = np.random.default_rng(42)
    h = max(int(round(holdout_gap_m[1] / texel_m)), 4)
    w = max(int(round(holdout_gap_m[0] / texel_m)), 4)
    train_c1, train_c2, train_c3, train_base, train_gt = [], [], [], [], []

    eligible = [i for i, ps in clean_sources.items() if ps.H > h + 4 and ps.W > w + 4 and ps.known.sum() > 600 and ps.kind != "roof"]
    for _ in range(90):
        if len(train_gt) >= 56 or not eligible:
            break
        idx = int(rng.choice(eligible))
        ps = clean_sources[idx]
        y0 = int(rng.integers(1, ps.H - h - 1))
        x0 = int(rng.integers(1, ps.W - w - 1))
        y1, x1 = y0 + h, x0 + w
        if not ps.known[y0:y1, x0:x1].all():
            continue
        k_masked = ps.known.copy()
        k_masked[y0:y1, x0:x1] = False
        if k_masked.sum() < 400:
            continue
        im_masked = ps.img.copy()
        im_masked[y0:y1, x0:x1] = np.median(ps.img[k_masked], axis=0)
        b_full = harmonic_base(im_masked, k_masked, sigma=max(3.0, 1.2 / texel_m)).astype(np.float32)

        temp_self = PreparedSource(im_masked, k_masked, face=ps.face, kind=ps.kind, device=device)
        cands = retrieve_candidates(im_masked, k_masked, y0, y1, x0, x1, temp_self, ref_list, top_k=3, pad=10)
        if not cands:
            continue
        while len(cands) < 3:
            cands.append(cands[-1])
        train_c1.append(cands[0][1])
        train_c2.append(cands[1][1])
        train_c3.append(cands[2][1])
        train_base.append(b_full[y0:y1, x0:x1])
        train_gt.append(ps.img[y0:y1, x0:x1])

    torch.manual_seed(42)
    model = FacadeRefineNet(hidden=32).to(device)
    if len(train_gt) >= 8:
        t_c1 = torch.from_numpy(np.stack(train_c1).transpose(0, 3, 1, 2)).float().to(device)
        t_c2 = torch.from_numpy(np.stack(train_c2).transpose(0, 3, 1, 2)).float().to(device)
        t_c3 = torch.from_numpy(np.stack(train_c3).transpose(0, 3, 1, 2)).float().to(device)
        t_b = torch.from_numpy(np.stack(train_base).transpose(0, 3, 1, 2)).float().to(device)
        t_gt = torch.from_numpy(np.stack(train_gt).transpose(0, 3, 1, 2)).float().to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        for _ in range(steps):
            opt.zero_grad()
            pred = model(t_c1, t_c2, t_c3, t_b)
            loss = _ncc_loss_torch(pred, t_gt) + 0.04 * F.l1_loss(pred, t_gt)
            loss.backward()
            opt.step()
    model.eval()
    return clean_sources, ref_list, model


def neural_fill(
    img: np.ndarray,
    source: np.ndarray,
    texel_m: float,
    face=None,
    kind: str = "",
    ref_sources: Sequence[PreparedSource] = (),
    model: Optional[FacadeRefineNet] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill ``PLAIN`` texels in ``(img, source)`` using PyTorch multi-view
    context retrieval + :class:`FacadeRefineNet`. Returns ``(new_img, new_source)``
    with invented texels marked :data:`NEURAL` (7)."""
    device = device or (next(model.parameters()).device if model is not None else default_device())
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
    self_src = PreparedSource(img, known, face=face, kind=kind, device=device)

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
            b_patch = base_rgb[y0:y1, x0:x1]
            if model is not None:
                c1_t = torch.from_numpy(cands[0][1].transpose(2, 0, 1))[None].float().to(device)
                c2_t = torch.from_numpy(cands[1][1].transpose(2, 0, 1))[None].float().to(device)
                c3_t = torch.from_numpy(cands[2][1].transpose(2, 0, 1))[None].float().to(device)
                b_t = torch.from_numpy(b_patch.transpose(2, 0, 1))[None].float().to(device)
                with torch.no_grad():
                    pred = model(c1_t, c2_t, c3_t, b_t)[0].cpu().numpy().transpose(1, 2, 0)
                blended = pred
            else:
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
    model: Optional[FacadeRefineNet] = None,
    ref_sources: Optional[Sequence[PreparedSource]] = None,
) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Complete every wall/roof texture of a building using :func:`neural_fill`.
    If ``model`` or ``ref_sources`` is omitted, trains the building's model
    self-supervised on its own ``SEEN`` regions first."""
    if model is None or ref_sources is None:
        _, ref_sources, model = train_building_model(images, metas, texel_m, exclude_holdout=False)
    out: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
    for img, meta in zip(images, metas):
        src = getattr(img, "source", None)
        im = np.asarray(img)
        if src is None:
            out.append((im, None))
            continue
        im_filled, s_filled = neural_fill(
            im, src.copy(), texel_m, face=meta.get("face"), kind=meta.get("kind", ""),
            ref_sources=ref_sources, model=model,
        )
        out.append((im_filled, s_filled))
    return out
