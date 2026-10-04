"""Figures for QA.

Every function returns a matplotlib ``Figure`` and takes an optional
``save_to`` for the scripts that write them. Nothing here writes to disk as a
side effect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from .cameras import VirtualObliqueCamera
from .features import CheckFeature, PlanarAnchor
from .imagery import ImageWindow
from .qa import MatchResult, PlanePatch, ReprojectionReport, sample_plane_patch


def _finish(fig: plt.Figure, save_to: Optional[Union[str, Path]]) -> plt.Figure:
    fig.tight_layout()
    if save_to is not None:
        p = Path(save_to)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=140, bbox_inches="tight")
    return fig


def plot_match_pair(
    anchor: PlanarAnchor,
    ref_camera: VirtualObliqueCamera,
    ref_window: ImageWindow,
    tgt_camera: VirtualObliqueCamera,
    tgt_window: ImageWindow,
    match: MatchResult,
    patch_radius_m: float = 4.0,
    gsd_m: float = 0.08,
    save_to: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Show the two rectified patches and where correlation put the match.

    This is the figure to look at before believing any residual number: if the
    two panels obviously show the same thing at the same place, a large reported
    residual is a bug, and if they are visibly offset, it is not.
    """
    ref = sample_plane_patch(ref_window, ref_camera, anchor, patch_radius_m, gsd_m)
    tgt = sample_plane_patch(tgt_window, tgt_camera, anchor, patch_radius_m, gsd_m)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))

    for ax, patch, title in (
        (axes[0], ref, f"reference\n{Path(ref_camera.frame_id).name}"),
        (axes[1], tgt, f"target\n{Path(tgt_camera.frame_id).name}"),
    ):
        if patch is None:
            ax.text(0.5, 0.5, "unprojectable", ha="center")
            ax.axis("off")
            continue
        n = patch.patch.shape[0]
        extent = [-patch.radius_m, patch.radius_m, patch.radius_m, -patch.radius_m]
        ax.imshow(patch.patch, extent=extent)
        ax.plot(0, 0, "+", color="cyan", markersize=14, markeredgewidth=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("e1 (m)")
        ax.set_ylabel("e2 (m)")

    # Overlay: reference in red, target in cyan, at their rectified positions.
    if ref is not None and tgt is not None:
        comp = np.zeros((*ref.patch.shape[:2], 3), dtype=np.uint8)
        comp[..., 0] = np.asarray(ref.patch).mean(axis=2).astype(np.uint8)
        comp[..., 1] = np.asarray(tgt.patch).mean(axis=2).astype(np.uint8)
        comp[..., 2] = comp[..., 1]
        extent = [-ref.radius_m, ref.radius_m, ref.radius_m, -ref.radius_m]
        axes[2].imshow(comp, extent=extent)
        axes[2].plot(0, 0, "+", color="yellow", markersize=14, markeredgewidth=2)
        if match.accepted:
            axes[2].arrow(
                0, 0, match.offset_e1_m, match.offset_e2_m,
                color="yellow", width=0.03, head_width=0.18, length_includes_head=True,
            )
        axes[2].set_title(
            f"overlay (red=ref, cyan=target)\n"
            f"measured offset {match.residual_m:.2f} m / {match.residual_px:.1f} px, "
            f"NCC {match.ncc:.2f}",
            fontsize=10,
        )
        axes[2].set_xlabel("e1 (m)")

    fig.suptitle(f"Anchor {anchor.anchor_id} ({anchor.label})", fontsize=11)
    return _finish(fig, save_to)


def plot_residual_field(
    report: ReprojectionReport,
    anchors: Sequence[PlanarAnchor],
    save_to: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Quiver of residual vectors in plan view, per frame pair.

    Shape is the diagnosis: a coherent field means a pose/orientation error that
    a bundle adjustment can absorb; a random field means matching noise.
    """
    by_id = {a.anchor_id: a for a in anchors}
    pairs = sorted({(m.ref_frame, m.tgt_frame) for m in report.accepted})
    n = max(len(pairs), 1)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.4), squeeze=False)

    for ax, (ref, tgt) in zip(axes[0], pairs):
        sel = [m for m in report.accepted if m.ref_frame == ref and m.tgt_frame == tgt]
        xy = np.array([by_id[m.anchor_id].xyz_local_m[:2] for m in sel])
        # e1/e2 offsets expressed back in local world XY.
        duv = np.array(
            [
                m.offset_e1_m * by_id[m.anchor_id].e1[:2]
                + m.offset_e2_m * by_id[m.anchor_id].e2[:2]
                for m in sel
            ]
        )
        mag = np.hypot(duv[:, 0], duv[:, 1])
        q = ax.quiver(
            xy[:, 0], xy[:, 1], duv[:, 0], duv[:, 1], mag,
            cmap="viridis", angles="xy", scale_units="xy", scale=0.05, width=0.004,
        )
        plt.colorbar(q, ax=ax, fraction=0.046, label="residual (m)")
        ax.set_title(
            f"{Path(ref).name} -> {Path(tgt).name}\n"
            f"median {np.median(mag):.2f} m ({np.median([m.residual_px for m in sel]):.1f} px)",
            fontsize=10,
        )
        ax.set_xlabel("local X, East (m)")
        ax.set_ylabel("local Y, North (m)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3, linestyle=":")

    fig.suptitle("Cross-view residual field (arrows 20x)", fontsize=12)
    return _finish(fig, save_to)


def plot_residual_histogram(
    report: ReprojectionReport, target_px: float = 2.0,
    save_to: Optional[Union[str, Path]] = None
) -> plt.Figure:
    errs = report.residuals_px
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errs, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(float(np.median(errs)), color="crimson", linewidth=2,
               label=f"median {np.median(errs):.2f} px")
    ax.axvline(target_px, color="darkgreen", linestyle="--", linewidth=2,
               label=f"target < {target_px:g} px")
    ax.set_xlabel("cross-view residual (px)")
    ax.set_ylabel("matches")
    ax.set_title("Measured reprojection residuals")
    ax.legend()
    return _finish(fig, save_to)


def plot_lidar_overlay(
    window: ImageWindow,
    camera: VirtualObliqueCamera,
    lidar_xyz_local_m: np.ndarray,
    check_features: Sequence[CheckFeature] = (),
    point_size: float = 2.0,
    save_to: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """LiDAR points and check features drawn on a source window."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(window.rgb)

    uv, zc, _ = camera.project(lidar_xyz_local_m)
    local = window.to_local_xy(uv)
    inside = (
        (zc > 0.1)
        & (local[:, 0] >= 0) & (local[:, 0] < window.width)
        & (local[:, 1] >= 0) & (local[:, 1] < window.height)
    )
    sc = ax.scatter(
        local[inside, 0], local[inside, 1],
        c=lidar_xyz_local_m[inside, 2], s=point_size, cmap="turbo", alpha=0.6,
    )
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="local elevation (m)")

    for feat in check_features:
        uv_f, zc_f, _ = camera.project(np.array(feat.xyz_local_m))
        pt = window.to_local_xy(uv_f)
        if 0 <= pt[0] < window.width and 0 <= pt[1] < window.height:
            ax.plot(pt[0], pt[1], "+", color="red", markersize=14, markeredgewidth=2.5)
            ax.text(pt[0] + 10, pt[1] - 10, feat.feature_id, color="yellow",
                    fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7))

    ax.set_title(f"LiDAR overlay: {Path(camera.frame_id).name}", fontsize=12)
    ax.axis("off")
    return _finish(fig, save_to)


def plot_frame_coverage(
    coverage_table, cameras: Sequence[VirtualObliqueCamera],
    save_to: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Where the building volume lands on each sensor, and what falls off."""
    n = len(coverage_table)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), squeeze=False)
    for ax, (_, row) in zip(axes[0], coverage_table.iterrows()):
        cam = next(c for c in cameras if c.frame_id == row["frame_id"])
        ax.add_patch(plt.Rectangle((0, 0), cam.width_px, cam.height_px,
                                   fill=False, edgecolor="black", linewidth=1.5))
        ok = bool(row["fully_contained"])
        ax.add_patch(plt.Rectangle(
            (row["u_min"], row["v_min"]), row["span_u_px"], row["span_v_px"],
            fill=True, alpha=0.35, edgecolor="crimson" if not ok else "green",
            facecolor="crimson" if not ok else "green", linewidth=2))
        ax.set_xlim(min(-600, row["u_min"] - 200), max(cam.width_px + 600, row["u_max"] + 200))
        ax.set_ylim(max(cam.height_px + 600, row["v_max"] + 200), min(-600, row["v_min"] - 200))
        ax.set_title(f"{Path(row['frame_id']).name}\n"
                     f"{row['corners_on_sensor']}/8 corners on sensor",
                     fontsize=9, color="darkgreen" if ok else "crimson")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    fig.suptitle("Building volume vs sensor extent", fontsize=12)
    return _finish(fig, save_to)


def plot_flight_geometry(
    cameras: Sequence[VirtualObliqueCamera],
    footprint_local_xy: np.ndarray,
    save_to: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plan view of camera positions, look directions and the building."""
    fig, ax = plt.subplots(figsize=(8, 8))
    poly = np.vstack([footprint_local_xy, footprint_local_xy[0:1]])
    ax.plot(poly[:, 0], poly[:, 1], "r-", linewidth=2.5, label="building footprint (LiDAR)")
    ax.fill(poly[:, 0], poly[:, 1], "salmon", alpha=0.3)

    palette = {"Fwd": "#1f77b4", "Bwd": "#d62728", "Left": "#2ca02c",
               "Right": "#9467bd", "Color": "#ff7f0e"}
    by_dir = {}
    for c in cameras:
        by_dir.setdefault(c.camera_name.split("_")[0], []).append(c)

    for direction, members in sorted(by_dir.items()):
        colour = palette.get(direction, "#555555")
        cx = [c.camera_center_local_m[0] for c in members]
        cy = [c.camera_center_local_m[1] for c in members]
        ax.plot(cx, cy, "o", color=colour, markersize=8,
                label=f"{direction} ({len(members)})")
        for c in members:
            d = c.R_wc @ np.array([0, 0, 1.0])
            scale = 0.25 * float(np.linalg.norm(c.camera_center_local_m[:2]))
            ax.arrow(c.camera_center_local_m[0], c.camera_center_local_m[1],
                     d[0] * scale, d[1] * scale, head_width=25, head_length=35,
                     fc=colour, ec=colour, alpha=0.55, length_includes_head=True)
    ax.set_xlabel("local X, East (m)")
    ax.set_ylabel("local Y, North (m)")
    ax.set_title("Exposure geometry by look direction")
    ax.grid(alpha=0.4, linestyle=":")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="datalim")
    return _finish(fig, save_to)


def plot_building_in_crops(
    crops: Sequence[np.ndarray],
    cameras: Sequence[VirtualObliqueCamera],
    crop_offsets: Sequence[Tuple[int, int]],
    roof_corners_local_m: np.ndarray,
    ground_z_m: Optional[float] = None,
    labels: Optional[Sequence[str]] = None,
    ncols: Optional[int] = None,
    save_to: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Draw the delivered crops with the target building outlined on each.

    The roof polygon is projected solid and, when ``ground_z_m`` is given, the
    same footprint at ground level is projected dashed with vertical edges
    joining them. The gap between the two outlines is the building's apparent
    lean, which grows with obliquity and is a quick read on whether the pose is
    sane in each view.
    """
    n = len(crops)
    ncols = min(ncols or n, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 5.0 * nrows),
                                  squeeze=False)
    axes = [ax for row in axes_grid for ax in row]
    for ax in axes[n:]:
        ax.axis("off")

    roof = np.asarray(roof_corners_local_m, dtype=np.float64)
    roof_closed = np.vstack([roof, roof[:1]])
    if ground_z_m is not None:
        ground_closed = roof_closed.copy()
        ground_closed[:, 2] = float(ground_z_m)

    for i, (ax, img, cam, (x0, y0)) in enumerate(zip(axes, crops, cameras, crop_offsets)):
        ax.imshow(img)
        off = np.array([x0, y0], dtype=np.float64)

        uv_roof, _, _ = cam.project(roof_closed)
        uv_roof = uv_roof - off

        if ground_z_m is not None:
            uv_ground, _, _ = cam.project(ground_closed)
            uv_ground = uv_ground - off
            for a, b in zip(uv_roof[:-1], uv_ground[:-1]):
                ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="#ffd400",
                        linewidth=1.2, alpha=0.9)
            ax.plot(uv_ground[:, 0], uv_ground[:, 1], "--", color="#ffd400",
                    linewidth=1.8, label="footprint at ground")

        ax.plot(uv_roof[:, 0], uv_roof[:, 1], "-", color="#ff2d55",
                linewidth=2.4, label="roof outline (LiDAR)")
        ax.plot(uv_roof[:-1, 0], uv_roof[:-1, 1], "o", color="#ff2d55",
                markersize=6, markeredgecolor="white", markeredgewidth=1.2)

        title = labels[i] if labels else Path(cam.frame_id).name
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.axis("off")
        if i == 0:
            ax.legend(loc="lower left", fontsize=9, framealpha=0.85)

    fig.suptitle(f"Target building in all {n} usable exposures", fontsize=14)
    return _finish(fig, save_to)


def plot_building_profile(
    clouds: Dict[str, Any],
    save_to: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Side elevations of each AOI's LiDAR, with the bounding volume drawn on.

    The single most useful picture in this project, because it makes two things
    obvious that tables do not. The pilot and the tower differ by an order of
    magnitude in height and invert in aspect. And the volume that ``roof - 8 m``
    produced for the tower is visibly a slab near the roof with the entire
    building hanging below it, outside.

    ``clouds`` maps a label to ``{"xyz", "cls", "volume", "legacy_volume"}``.
    """
    n = len(clouds)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 6.4), squeeze=False)
    for ax, (label, d) in zip(axes[0], clouds.items()):
        xyz, cls = d["xyz"], d["cls"]
        # Project onto the x axis: a side elevation looking along y.
        ground = cls == 2
        ax.scatter(xyz[ground, 0], xyz[ground, 2], s=0.4, c="#8c8c8c",
                   linewidths=0, label="ground (class 2)")
        ax.scatter(xyz[~ground, 0], xyz[~ground, 2], s=0.4, c="#1f77b4",
                   linewidths=0, label="non-ground")

        for key, colour, style, name in (
            ("legacy_volume", "#d62728", "--", "volume from roof - 8 m"),
            ("volume", "#2ca02c", "-", "volume from measured ground"),
        ):
            v = d.get(key)
            if v is None:
                continue
            ax.add_patch(plt.Rectangle(
                (v["min_x_m"], v["min_z_m"]),
                v["max_x_m"] - v["min_x_m"], v["max_z_m"] - v["min_z_m"],
                fill=False, edgecolor=colour, linestyle=style, linewidth=2.2,
                label=f"{name}  ({v['max_z_m'] - v['min_z_m']:.0f} m tall)"))

        ax.set_xlabel("x, metres from AOI origin")
        ax.set_ylabel("elevation, metres")
        ax.set_title(f"{label}\n{d['title']}", fontsize=11)
        ax.grid(alpha=0.25)
        # Below the axes: the legacy volume sits near the roof on a tall
        # building, which is exactly where an in-axes legend would hide it.
        ax.legend(fontsize=8, markerscale=8, loc="upper center",
                  bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
        ax.set_aspect("equal", adjustable="datalim")
        for key, colour in (("legacy_volume", "#d62728"), ("volume", "#2ca02c")):
            v = d.get(key)
            if v is None:
                continue
            ax.annotate("", xy=(v["max_x_m"], v["min_z_m"]),
                        xytext=(v["max_x_m"], v["max_z_m"]),
                        arrowprops=dict(arrowstyle="<->", color=colour, lw=1.1))
            ax.text(v["max_x_m"] + 1.5, (v["min_z_m"] + v["max_z_m"]) / 2,
                    f"{v['max_z_m'] - v['min_z_m']:.0f} m", color=colour,
                    fontsize=9, va="center", fontweight="bold")
    fig.suptitle("Side elevation: what the bounding volume actually contains", fontsize=13)
    fig.tight_layout()
    return _finish(fig, save_to)


def plot_case_complexity(
    cases: Sequence[Dict[str, Any]],
    save_to: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Where each case building's "worth densifying" verdict comes from.

    The table of structure fractions says 400 West Market is twice PNC Tower.
    It does not say why, and the why turns out to matter: the top row shows the
    two towers get their roughness from different places, and the bottom row
    shows that once the walls are separated out, the wall rate is the same on
    both — a plain wall and an articulated one read alike, which is the height
    artefact rather than any structure.

    Each entry needs ``label``, ``xyz`` (metres, z above the lowest return),
    ``structure`` (a boolean mask), and ``plate_z_m``.
    """
    n = len(cases)
    fig = plt.figure(figsize=(4.6 * n, 9.0))
    gs = fig.add_gridspec(2, n, height_ratios=[2.5, 1.0], hspace=0.34, wspace=0.28)

    top = max(float(c["xyz"][:, 2].max()) for c in cases)
    for col, c in enumerate(cases):
        xyz, struct, plate = c["xyz"], np.asarray(c["structure"], dtype=bool), c["plate_z_m"]
        ax = fig.add_subplot(gs[0, col])
        # A side elevation, looking along y. Thinned only for file size; the
        # structure points are thinned less because they are the point.
        plain = xyz[~struct]
        ax.scatter(plain[::max(1, len(plain) // 9000), 0],
                   plain[::max(1, len(plain) // 9000), 2],
                   s=1.2, c="#b8c4d0", linewidths=0, rasterized=True,
                   label="LiDAR return")
        sx = xyz[struct]
        if len(sx):
            ax.scatter(sx[::max(1, len(sx) // 4000), 0], sx[::max(1, len(sx) // 4000), 2],
                       s=2.4, c="#d1495b", linewidths=0, rasterized=True,
                       label="counted as real structure")
        ax.axhline(plate - 2.0, color="#3d6b8c", lw=1.0, ls="--")
        ax.text(0.995, (plate - 2.0 + 4) / (top * 1.06), "roof level ",
                transform=ax.transAxes, fontsize=8, color="#3d6b8c",
                ha="right", va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
        ax.set_title(c["label"], fontsize=10.5, pad=8)
        ax.set_ylim(-4, top * 1.06)
        ax.set_aspect("equal")
        ax.set_xlabel("metres across the building", fontsize=9)
        if col == 0:
            ax.set_ylabel("metres above the lowest return", fontsize=9)
            ax.legend(fontsize=8, markerscale=5, loc="center left", frameon=False)
        ax.tick_params(labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        # Split the same points into walls and roof and rate each separately.
        axb = fig.add_subplot(gs[1, col])
        z = xyz[:, 2]
        zones = [("walls", (z >= 3.0) & (z < plate - 2.0)),
                 ("roof", z >= plate - 2.0)]
        # A zone with almost no returns has no rate worth drawing; the pilot's
        # walls are 40 points, and a bar there would invite a comparison the
        # data cannot support.
        names, rates, colours = [], [], []
        for (name, m), colour in zip(zones, ("#8fa3b8", "#d1495b")):
            names.append(name)
            rates.append(float(struct[m].mean()) if m.sum() >= 200 else np.nan)
            colours.append(colour)
        bars = axb.bar(names, np.nan_to_num(rates), color=colours, width=0.55)
        for (name, m), b, r in zip(zones, bars, rates):
            x = b.get_x() + b.get_width() / 2
            if np.isnan(r):
                axb.text(x, 0.02, f"barely any\n{name}\n({int(m.sum())} points)",
                         ha="center", va="bottom", fontsize=8.5, color="#777")
            else:
                axb.text(x, r + 0.012, f"{r:.0%}", ha="center", fontsize=10)
        axb.set_ylim(0, 0.45)
        axb.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
        axb.set_yticklabels(["0%", "10%", "20%", "30%", "40%"], fontsize=8)
        if col == 0:
            axb.set_ylabel("share of returns\ncounted as structure", fontsize=9)
        axb.tick_params(labelsize=9)
        for sp in ("top", "right"):
            axb.spines[sp].set_visible(False)

    fig.suptitle("Where the verdict comes from", fontsize=13.5, y=0.965)
    fig.text(0.5, 0.935,
             "Top: side view of the LiDAR. Bottom: the same points split into "
             "walls and roof, rated separately.",
             ha="center", fontsize=9.5, color="#444")
    return _finish(fig, save_to)


def plot_survey_zones(
    arms: Sequence[Dict[str, Any]],
    save_to: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """What rating roofs separately does to a survey's headline numbers.

    ``arms`` entries need ``label`` and ``results`` (a survey JSON's result
    list). The left panel is the finding -- ground and wall rate about three
    times the roof on both samples, so a whole-building average is pulled up by
    the ~7% of returns that are not roof. One panel per arm after that, because
    the thing anyone acts on is the share of buildings clearing a bar, and
    stacking both arms into one axis made it impossible to see which bar was
    which.
    """
    n = len(arms)
    fig, axes = plt.subplots(1, 1 + n, figsize=(5.0 + 4.2 * n, 5.0))

    ax = axes[0]
    zones = ("ground", "wall", "roof")
    colours = ("#c9a227", "#8fa3b8", "#d1495b")
    width = 0.8 / n
    for i, arm in enumerate(arms):
        vals = [np.nanmedian(np.array([r.get(f"{z}_rate") for r in arm["results"]],
                                      dtype=float)) for z in zones]
        x = np.arange(len(zones)) + (i - (n - 1) / 2) * width
        # Same colour per zone, faded for the second arm: the comparison the eye
        # should make first is zone-to-zone, not sample-to-sample.
        bars = ax.bar(x, vals, width=width * 0.9, color=colours,
                      alpha=1.0 if i == 0 else 0.45, edgecolor="white", linewidth=0.8,
                      label=arm["label"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.0%}",
                    ha="center", fontsize=9)
    ax.set_xticks(np.arange(len(zones)))
    ax.set_xticklabels(["ground around\nthe building", "walls", "roof"], fontsize=10)
    ax.set_ylabel("typical share of that zone's returns\nscored as real structure",
                  fontsize=9.5)
    ax.set_title("Where a building's score comes from", fontsize=11.5)
    ax.set_ylim(0, 0.235)
    ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
    ax.set_yticklabels(["0%", "5%", "10%", "15%", "20%"], fontsize=9)
    # Grey swatches: the legend distinguishes the two arms, and colour is
    # already carrying the zone.
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#555", alpha=1.0 if i == 0 else 0.45)
               for i in range(n)]
    ax.legend(handles, [a["label"] for a in arms], fontsize=8.5, frameon=False,
              loc="upper right")

    cuts = (0.05, 0.10, 0.20)
    for ax, arm in zip(axes[1:], arms):
        mixed = np.array([r["structure_frac"] for r in arm["results"]], dtype=float)
        roof = np.array([r.get("roof_rate") for r in arm["results"]], dtype=float)
        x = np.arange(len(cuts))
        for off, vals, colour, lab in ((-0.19, mixed, "#b8c4d0", "every return"),
                                       (0.19, roof, "#d1495b", "roof returns only")):
            v = [np.nanmean(vals > c) for c in cuts]
            ax.bar(x + off, v, width=0.34, color=colour, label=lab)
            for xi, vi in zip(x + off, v):
                ax.text(xi, vi + 0.012, f"{vi:.0%}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"more than {c:.0%}\nof the surface" for c in cuts],
                           fontsize=9.5)
        ax.set_ylim(0, 0.72)
        ax.set_yticks([0, 0.2, 0.4, 0.6])
        ax.set_yticklabels(["0%", "20%", "40%", "60%"], fontsize=9)
        ax.set_title(arm["label"], fontsize=11.5)
        ax.legend(fontsize=9, frameon=False, loc="upper right")
    axes[1].set_ylabel("share of buildings clearing the bar", fontsize=9.5)

    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("Rating roofs on their own, not mixed with the ground and walls "
                 "around them", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return _finish(fig, save_to)


def plot_roof_section(
    roof_xyz: np.ndarray,
    rect_corners: np.ndarray,
    features: Sequence[Dict[str, Any]],
    save_to: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Elevation against radius for one roof, with the rectangle corners marked.

    Shows why a domed crown cannot be described by four corners at one median
    elevation: the corners of a rectangle around a circular footprint sit at
    radius x sqrt(2), beyond where any roof points exist at all.
    """
    pts = np.asarray(roof_xyz, dtype=np.float64)
    cx, cy = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
    r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    median_z = float(np.median(pts[:, 2]))

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.scatter(r, pts[:, 2], s=4, c="#1f77b4", linewidths=0, label="LiDAR roof returns")
    ax.axhline(median_z, color="#7f7f7f", linestyle=":", linewidth=1.4,
               label=f"median elevation {median_z:.2f} m")

    rc = np.asarray(rect_corners, dtype=np.float64)
    rr = np.hypot(rc[:, 0] - cx, rc[:, 1] - cy)
    ax.scatter(rr, np.full(len(rr), median_z), s=170, marker="X", c="#d62728",
               zorder=5, label="minAreaRect corners at median z")
    ax.axvline(r.max(), color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7)
    lo_z, hi_z = pts[:, 2].min(), pts[:, 2].max()
    ax.annotate("last roof return", xy=(r.max(), lo_z),
                xytext=(r.max() * 0.55, lo_z + 0.25 * (hi_z - lo_z)),
                fontsize=9, color="#d62728", ha="center",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))
    ax.annotate(f"corners land {rr.mean() - r.max():.1f} m past the roof edge,\n"
                f"and {median_z - np.percentile(pts[r > 0.8 * r.max(), 2], 50):.1f} m "
                f"above the surface there",
                xy=(rr.mean(), median_z),
                xytext=(rr.mean(), median_z - 0.45 * (hi_z - lo_z)),
                fontsize=9, color="#d62728", ha="center",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))

    for f in features:
        x, y, z = f["xyz_local_m"]
        ax.scatter([np.hypot(x - cx, y - cy)], [z], s=140, marker="o",
                   facecolors="none", edgecolors="#2ca02c", linewidths=2.2, zorder=6)
    if features:
        ax.scatter([], [], s=140, marker="o", facecolors="none",
                   edgecolors="#2ca02c", linewidths=2.2,
                   label="surface-sampled check features")

    ax.set_xlabel("horizontal distance from roof centre, metres")
    ax.set_ylabel("elevation, metres")
    ax.set_title("A dome is not a rectangle: where the 'measured' corners actually land",
                 fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    return _finish(fig, save_to)


def plot_anchor_visibility(
    anchor_xy: Dict[str, Tuple[float, float]],
    acceptance: Dict[str, float],
    footprint: Optional[np.ndarray] = None,
    title: str = "",
    save_to: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plan view of anchors coloured by how often they matched across views.

    A wrong camera model degrades every anchor about equally, so its map is a
    uniform wash. Occlusion does not: it produces anchors that never match from
    any direction, clustered where other structures stand in the way. The
    histogram beside the map is the discriminating evidence -- a bimodal
    distribution with a spike at zero is a visibility problem, not a geometry one.
    """
    ids = [a for a in anchor_xy if a in acceptance]
    x = np.array([anchor_xy[a][0] for a in ids])
    y = np.array([anchor_xy[a][1] for a in ids])
    rate = np.array([acceptance[a] for a in ids])

    fig, (ax, hx) = plt.subplots(1, 2, figsize=(13.5, 6.0),
                                 gridspec_kw={"width_ratios": [1.55, 1]})
    sc = ax.scatter(x, y, c=rate, cmap="RdYlGn", vmin=0, vmax=1, s=34,
                    edgecolors="#333333", linewidths=0.3)
    never = rate == 0
    if never.any():
        ax.scatter(x[never], y[never], s=95, facecolors="none",
                   edgecolors="#000000", linewidths=1.1,
                   label=f"never matched ({int(never.sum())})")
        ax.legend(fontsize=9, loc="upper right")
    if footprint is not None:
        fp = np.asarray(footprint)
        ax.add_patch(plt.Polygon(fp, fill=False, edgecolor="#1f77b4",
                                 linewidth=2.0, linestyle="--"))
    fig.colorbar(sc, ax=ax, label="fraction of view pairs matched")
    ax.set_xlabel("x, metres")
    ax.set_ylabel("y, metres")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_title("Where matching succeeds", fontsize=11)

    hx.hist(rate, bins=20, color="#4c72b0", edgecolor="white")
    hx.set_xlabel("per-anchor acceptance rate")
    hx.set_ylabel("anchors")
    hx.grid(alpha=0.25, axis="y")
    hx.set_title(f"Bimodal, not uniform\n{int(never.sum())} of {len(rate)} "
                 f"never match from any direction", fontsize=11)
    fig.suptitle(title or "Anchor visibility", fontsize=13)
    fig.tight_layout()
    return _finish(fig, save_to)


def plot_residual_vs_distance(
    distances_m: np.ndarray,
    residuals_px: np.ndarray,
    edges_m: Sequence[float] = (0, 10, 20, 30, 40, 80),
    roof_half_extent_m: Optional[float] = None,
    save_to: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Cross-view residual against distance from the building centre.

    Written to test a hypothesis that turned out to be wrong: that residuals
    grow toward the crop edge. They do not -- they fall. The structure is a
    spike at the *building's* own edge, where a planar patch straddles roof,
    parapet and wall and the correspondence is genuinely ambiguous.
    """
    d = np.asarray(distances_m, dtype=np.float64)
    r = np.asarray(residuals_px, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.scatter(d, r, s=5, alpha=0.18, c="#4c72b0", linewidths=0)

    centres, medians, counts = [], [], []
    for lo, hi in zip(edges_m[:-1], edges_m[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() < 10:
            continue
        centres.append((lo + hi) / 2)
        medians.append(float(np.median(r[m])))
        counts.append(int(m.sum()))
    ax.plot(centres, medians, "o-", color="#d62728", linewidth=2.4, markersize=8,
            label="median per band")
    for cx, my, n in zip(centres, medians, counts):
        ax.annotate(f"{my:.1f} px\nn={n:,}", (cx, my), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8, color="#d62728")

    if roof_half_extent_m:
        ax.axvline(roof_half_extent_m, color="#2ca02c", linestyle="--", linewidth=1.6,
                   label=f"roof edge (~{roof_half_extent_m:.0f} m)")
    ax.set_yscale("log")
    ax.set_xlabel("anchor distance from building centre, metres")
    ax.set_ylabel("cross-view residual, px (log)")
    ax.set_title("Residuals spike at the building's edge, not the crop's edge",
                 fontsize=12)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _finish(fig, save_to)
