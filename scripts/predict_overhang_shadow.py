#!/usr/bin/env python3
"""Predict which walls an aerial survey can never reach, and check it.

An airborne scanner looks down. A wall set back beneath a projection -- a crown
wider than its shaft, a cornice, piers proud of a recessed curtain -- returns
nothing from any airborne angle. The projection itself is roof and crown, which
the scanner sees best, so the walls it hides can be predicted from returns the
survey already has, before anything is attempted on them.

This runs :func:`features.predict_overhang_shadow` over a building and, because
the facade capture pass measured where wall returns actually landed, checks the
prediction against the observation: cells predicted shadowed should be empty,
and cells predicted clear should not be.

It will not explain everything that is missing. A neighbour standing in the
line of sight also blanks a wall, and that is a different mechanism -- another
flight direction fixes it, where nothing fixes an overhang. The residual after
this prediction is the part another flight could recover.

Usage::

    python scripts/predict_overhang_shadow.py --building outputs/downtown/buildings/d2786314-4fb3-4da6-8f81-621873d40c08 \\
        --gers d2786314-4fb3-4da6-8f81-621873d40c08
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from kyfromabove_3d.coordinates import METERS_PER_US_SURVEY_FOOT
from kyfromabove_3d.features import fit_local_planes, predict_overhang_shadow
from kyfromabove_3d.clouds import find_cloud, read_cloud


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--building", type=Path, required=True)
    ap.add_argument("--gers", required=True)
    ap.add_argument("--footprint-csv", type=Path,
                    default=Path("data/derived/ky_facade_tiled_wkt.csv"))
    ap.add_argument("--max-off-nadir-deg", type=float, default=20.0,
                    help="steepest usable scan angle from vertical. Sets how far "
                         "below an overhang the shadow reaches: d / tan(angle)")
    ap.add_argument("--cell-m", type=float, default=3.0)
    ap.add_argument("--label", default=None)
    ap.add_argument("--colorized", type=Path, default=None,
                    help="path to colorized PLY to check for photo-illuminated overhang cells")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pyproj import Transformer
    from shapely import contains_xy
    from shapely.geometry import Polygon
    from shapely.ops import transform as shp_transform
    from shapely.wkt import loads as wkt_loads

    csv.field_size_limit(1 << 30)
    ox, oy, oz = json.loads((args.building / "coordinates.json").read_text())["origin_source_xyz"]
    ply = read_cloud(find_cloud(args.building, "lidar_local"))
    xyz = np.column_stack([ply["x"], ply["y"], ply["z"]]).astype(np.float64)
    wkt = next((r["wkt"] for r in csv.DictReader(open(args.footprint_csv, newline="",
                                                      encoding="utf-8"))
                if r["id"] == args.gers), None)
    if wkt is None:
        print(f"[FAIL] {args.gers} not in {args.footprint_csv}")
        return 1
    to_ft = Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True).transform
    ring = np.array(shp_transform(to_ft, wkt_loads(wkt)).exterior.coords)
    ring_m = np.column_stack([(ring[:, 0] - ox) * METERS_PER_US_SURVEY_FOOT,
                              (ring[:, 1] - oy) * METERS_PER_US_SURVEY_FOOT])
    # A generous clip: the crown that overhangs a wall sits outside the drawn
    # footprint by exactly the amount that matters.
    on = contains_xy(Polygon(ring_m).buffer(6.0), xyz[:, 0], xyz[:, 1])
    b = xyz[on]
    # A wall return is one whose local surface is near-vertical. Not the zone
    # split: that takes the modal elevation as the roof, and on a stepped
    # building the mode is whichever level holds the most returns -- Humana's
    # flips between its tower roof at 99 m and its podium at 35 m depending on
    # how much podium the clip takes in, and the "wall" goes with it.
    normals, _, used = fit_local_planes(b, k=8)
    wall = (used >= 5) & (np.abs(normals[:, 2]) < 0.35)
    height = float(np.percentile(b[:, 2], 99) - b[:, 2].min())
    shadows = predict_overhang_shadow(b, wall, ring_m, height,
                                      max_off_nadir_deg=args.max_off_nadir_deg,
                                      cell_m=args.cell_m)
    shadows = [s for s in shadows if s.n_returns.size]
    if not shadows:
        print("no faces")
        return 1

    label = args.label or args.building.name
    print(f"{label}: {len(b):,} returns on the building, height {height:.0f} m, "
          f"{len(shadows)} faces, cells {args.cell_m} m, "
          f"scan reach {args.max_off_nadir_deg:.0f} deg off nadir\n")
    print(f"{'face':>5}{'shadowed':>10}{'empty|shadowed':>16}{'empty|clear':>13}")
    tot_sh = tot_cells = 0
    all_sh, all_empty = [], []
    for s in shadows:
        # Only judge cells the wall plausibly spans: below the roof line.
        empty = s.n_returns == 0
        sh = s.shadowed
        p_es = empty[sh].mean() if sh.any() else float("nan")
        p_ec = empty[~sh].mean() if (~sh).any() else float("nan")
        print(f"{s.face:>5}{sh.mean():>10.0%}{p_es:>16.0%}{p_ec:>13.0%}")
        tot_sh += sh.sum(); tot_cells += sh.size
        all_sh.append(sh.ravel()); all_empty.append(empty.ravel())
    print(f"\n{tot_sh:,} of {tot_cells:,} wall cells ({tot_sh / tot_cells:.0%}) "
          f"predicted unreachable from the air")
    # Score only the faces the scanner reached at all. A face a neighbour
    # blocks is 99% empty whatever the prediction says, and it swamps the
    # faces where the prediction can be right or wrong.
    reached = [s for s in shadows if (s.n_returns > 0).mean() >= 0.15]
    if reached:
        sh = np.concatenate([s.shadowed.ravel() for s in reached])
        em = np.concatenate([(s.n_returns == 0).ravel() for s in reached])
        print(f"  on the {len(reached)} face(s) the scanner reached "
              f"(>=15% of cells have returns):")
        if sh.any() and (~sh).any():
            print(f"    of cells predicted shadowed, {em[sh].mean():.0%} have no returns")
            print(f"    of cells predicted clear,    {em[~sh].mean():.0%} have no returns")
            print(f"    -> shadowed cells are empty "
                  f"{em[sh].mean() / max(em[~sh].mean(), 1e-9):.1f}x as often as clear ones")
    else:
        sh = np.concatenate(all_sh); em = np.concatenate(all_empty)
        print("  no face has enough returns to test the prediction against")
    sh = np.concatenate(all_sh); em = np.concatenate(all_empty)
    unexplained = (em & ~sh).sum()
    print(f"  {unexplained:,} empty cells are NOT under an overhang -- those are "
          f"line-of-sight (a neighbour, or the flight direction), and another "
          f"pass could recover them")

    photo_pts = None
    if args.colorized and args.colorized.exists():
        cply = read_cloud(args.colorized)
        if "source" in cply:
            is_photo = cply["source"] == 2
            if is_photo.any():
                photo_pts = np.column_stack([cply["x"][is_photo],
                                              cply["y"][is_photo],
                                              cply["z"][is_photo]]).astype(float)
                print(f"Loaded {len(photo_pts):,} photo-coloured points from {args.colorized}")

    v = ring_m[:-1] if np.allclose(ring_m[0], ring_m[-1]) else ring_m
    if np.sum((np.roll(v[:, 0], -1) - v[:, 0]) * (np.roll(v[:, 1], -1) + v[:, 1])) > 0:
        v = v[::-1]

    big = sorted(shadows, key=lambda s: -s.n_returns.sum())[:2]
    fig, axes = plt.subplots(1, len(big), figsize=(7.4 * len(big), 7.6), squeeze=False)
    for ax, s in zip(axes[0], big):
        A, U = s.along_edges_m, s.up_edges_m
        dens = np.ma.masked_where(s.n_returns == 0, s.n_returns.astype(float))
        ax.pcolormesh(A, U, dens, cmap="Blues", shading="flat", vmin=0,
                      vmax=max(1, np.percentile(s.n_returns[s.n_returns > 0], 90))
                      if (s.n_returns > 0).any() else 1)
        # Predicted shadow as a red hatch over everything.
        shade = np.ma.masked_where(~s.shadowed, np.ones_like(s.shadowed, dtype=float))
        ax.pcolormesh(A, U, shade, cmap=matplotlib.colors.ListedColormap(["#d1495b"]),
                      alpha=0.45, shading="flat")

        # Overlay photo-illuminated cells in cyan/teal
        photo_filled_count = 0
        if photo_pts is not None and len(photo_pts) > 0:
            p0, p1 = v[s.face], v[(s.face + 1) % len(v)]
            seg = p1 - p0
            L = float(np.hypot(*seg))
            e1 = seg / L
            n_out = np.array([e1[1], -e1[0]])
            rel = photo_pts[:, :2] - p0
            along = rel @ e1
            o = rel @ n_out
            up = photo_pts[:, 2] - b[:, 2].min()
            on_f = (along >= 0) & (along <= L) & (np.abs(o) <= 4.0) & (up >= 0) & (up <= height)
            if on_f.any():
                na, nu = s.shadowed.shape[1], s.shadowed.shape[0]
                # The same edges the shadow grid was built on. L / na is not the
                # cell width: the edges run past L to a multiple of cell_m, so
                # dividing L by the count gives a narrower cell and a point near
                # the far end lands one column over from where the shadow is.
                ai = np.clip(np.searchsorted(s.along_edges_m, along[on_f], side="right") - 1,
                             0, na - 1)
                ui = np.clip(np.searchsorted(s.up_edges_m, up[on_f], side="right") - 1,
                             0, nu - 1)
                p_counts = np.zeros_like(s.shadowed, dtype=int)
                np.add.at(p_counts, (ui, ai), 1)
                recov = s.shadowed & (p_counts > 0)
                photo_filled_count = int(recov.sum())
                if photo_filled_count > 0:
                    shade_rec = np.ma.masked_where(~recov, np.ones_like(s.shadowed, dtype=float))
                    ax.pcolormesh(A, U, shade_rec, cmap=matplotlib.colors.ListedColormap(["#2a9d8f"]),
                                  alpha=0.75, shading="flat")
                    print(f"  face {s.face}: {photo_filled_count} of {s.shadowed.sum()} ({photo_filled_count / max(1, s.shadowed.sum()):.1%}) "
                          f"overhang shadow cells illuminated by registered photo")

        title_extra = f"\ncyan = {photo_filled_count} shadow cells illuminated by photo" if photo_filled_count > 0 else ""
        ax.set_title(f"face {s.face}: {s.shadowed_frac:.0%} predicted unreachable (red){title_extra}\n"
                     f"blue = wall returns scanner got",
                     fontsize=10.0)
        ax.set_xlabel("metres along the wall"); ax.set_ylabel("metres up the wall")
        ax.set_aspect("equal")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"{label}: walls predicted unreachable from the air, from the roof "
                 f"returns alone\nscanner reaches {args.max_off_nadir_deg:.0f} deg "
                 f"off vertical; a projection of d metres shadows "
                 f"{1 / np.tan(np.radians(args.max_off_nadir_deg)):.1f} d below it",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = args.out or (args.building / "overhang_shadow.png")
    fig.savefig(out, dpi=125, bbox_inches="tight")
    print(f"\nfigure: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
