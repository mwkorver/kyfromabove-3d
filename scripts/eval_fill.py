#!/usr/bin/env python3
"""Does inventing unseen texels beat leaving them plain? The held-out test.

For every wall texture in the buildings' ``3dtiles_complete/fill_faces.npz``
(written by ``build_mesh.py --complete``): hide rectangles of texels a frame
*did* see, fill them with each method, and compare with what the frame saw.
Two measures, per hidden rectangle:

* pattern agreement -- correlation of the high-passed brightness, 1 = the
  same windows and mullions in the same places, 0 = unrelated (a plain fill
  has no pattern: 0);
* colour error -- the mean difference per channel, 0-255, against the same
  error for the plain fill (the face's median) it would replace.

A method is worth shipping where it agrees in pattern and does not make the
colour worse than plain. Writes ``docs/figures/fill_holdout.png``.

Usage::

    python scripts/eval_fill.py --dirs outputs/downtown/buildings/<gers> ... --out docs/figures/fill_holdout.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent / "src"))

from kyfromabove_3d.facade_fill import SEEN, holdout, patch_fill, rhythm_fill, twin_fill   # noqa: E402
from kyfromabove_3d.mesh.textures import Texture                                            # noqa: E402

try:                                         # the neural method needs PyTorch (pip install .[neural])
    from kyfromabove_3d.neural_fill import neural_fill, train_building_model                # noqa: E402
except ImportError:
    neural_fill = train_building_model = None

METHODS = {
    "rhythm": lambda t: (lambda i, s: rhythm_fill(i, s, t)),
    "patch": lambda t: (lambda i, s: patch_fill(i, s, t)),
    "rhythm, then patch": lambda t: (lambda i, s: patch_fill(*rhythm_fill(i, s, t), t)),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--gap-m", type=float, nargs=2, default=(3.0, 3.0), help="the hidden rectangle, across and up")
    ap.add_argument("--trials", type=int, default=3, help="hidden rectangles per texture")
    ap.add_argument("--min-seen-m2", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/fill_holdout.png"))
    args = ap.parse_args()

    rows = []
    for d in args.dirs:
        p = d / "3dtiles_complete" / "fill_faces.npz"
        if not p.exists():
            print(f"[skip] {d.name}: no fill_faces.npz (build_mesh.py --complete)")
            continue
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["meta"])); t = float(z["texel_m"])
        twins = {int(k): v for k, v in json.loads(str(z["twins"])).items()} if "twins" in z else {}
        profile_of = {m["face"]: i for i, m in enumerate(meta) if m.get("kind") == "profile" and f"img_{i}" in z}
        if train_building_model is not None:
            tex_list = [Texture(z[f"img_{i}"], z[f"src_{i}"]) if f"img_{i}" in z
                        else np.zeros((1, 1, 3), dtype=np.uint8) for i in range(len(meta))]
            _, ref_list, model = train_building_model(tex_list, meta, t, exclude_holdout=True,
                                                      holdout_gap_m=tuple(args.gap_m), holdout_trials=args.trials)
        for i, m in enumerate(meta):
            if f"img_{i}" not in z:
                continue
            img, src = z[f"img_{i}"], z[f"src_{i}"]
            if (src == SEEN).sum() * t * t < args.min_seen_m2:
                continue
            methods = dict(METHODS)
            if train_building_model is not None:
                methods["neural"] = lambda t_, _m=m, _refs=ref_list, _mod=model: (
                    lambda i_, s_: neural_fill(i_, s_, t_, face=_m.get("face"), kind=_m.get("kind", ""),
                                               ref_sources=_refs, model=_mod))
            j = profile_of.get(twins.get(m.get("face"))) if m.get("kind") == "profile" else None
            if j is not None:                      # a face with a mirror twin: fill from it
                ti, ts, tm = z[f"img_{j}"], z[f"src_{j}"], meta[j]
                methods["twin"] = lambda t_, _ti=ti, _ts=ts, _tm=tm, _m=m: (
                    lambda i_, s_: twin_fill(i_, s_, _m, _ti, _ts, _tm))
            for name, make in methods.items():
                for r in holdout(img, src, t, make(t), gap_m=tuple(args.gap_m), trials=args.trials, seed=i):
                    rows.append({"building": d.name, "kind": m.get("kind"), "method": name, **r})
    if not rows:
        print("no textures to test"); return 1
    print(f"{len({r['building'] for r in rows})} buildings, {len(rows)} hidden rectangles")
    summary = {}
    all_methods = list(METHODS) + (["neural"] if train_building_model is not None else []) + ["twin"]
    for name in all_methods:
        rs = [r for r in rows if r["method"] == name]
        if not rs:
            continue
        filled = [r for r in rs if r["share_filled"] > 0.5]
        ncc = np.array([r["ncc"] for r in filled if np.isfinite(r["ncc"])])
        better = np.mean([r["colour_err"] < r["plain_colour_err"] for r in filled]) if filled else float("nan")
        summary[name] = {"tested": len(rs), "filled": len(filled),
                         "ncc_median": float(np.median(ncc)) if len(ncc) else None,
                         "colour_err_median": float(np.median([r["colour_err"] for r in filled])) if filled else None,
                         "plain_colour_err_median": float(np.median([r["plain_colour_err"] for r in filled])) if filled else None,
                         "share_colour_better": float(better)}
        s = summary[name]
        print(f"  {name:20s} filled {s['filled']:4d} of {s['tested']:4d}; pattern agreement median "
              f"{s['ncc_median'] if s['ncc_median'] is None else round(s['ncc_median'], 2)}; colour error "
              f"{s['colour_err_median'] and round(s['colour_err_median'], 1)} vs plain "
              f"{s['plain_colour_err_median'] and round(s['plain_colour_err_median'], 1)}; "
              f"colour better than plain in {s['share_colour_better']:.0%}")
    by_kind = {}
    for r in rows:
        if r["share_filled"] > 0.5 and np.isfinite(r["ncc"]):
            by_kind.setdefault((r["kind"], r["method"]), []).append(r)
    summary["by_kind"] = {}
    print("  by face type and method:")
    for (k, meth), v in sorted(by_kind.items()):
        ncc = float(np.median([r["ncc"] for r in v]))
        better = float(np.mean([r["colour_err"] < r["plain_colour_err"] for r in v]))
        summary["by_kind"][f"{k}/{meth}"] = {"n": len(v), "ncc_median": ncc, "share_colour_better": better}
        print(f"    {k:8s} {meth:20s} {len(v):4d} rectangles: pattern agreement median {ncc:.2f}, "
              f"colour closer than plain in {better:.0%}")
    (args.out.with_suffix(".json")).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=float),
                                               encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    names = list(METHODS) + ["neural"] + (["twin"] if any(r["method"] == "twin" for r in rows) else [])
    data = [[r["ncc"] for r in rows if r["method"] == n and r["share_filled"] > 0.5 and np.isfinite(r["ncc"])] for n in names]
    axs[0].boxplot(data, orientation="horizontal", tick_labels=names, showfliers=False)
    axs[0].axvline(0, color="#898781", lw=1, ls="--")
    axs[0].set_xlabel("pattern agreement with what the frame saw (1 = same, 0 = unrelated; plain = 0)")
    axs[0].set_title("hidden rectangles of seen facade, filled by each method", fontsize=10)
    for n, col in zip(names, ("#2a78d6", "#eb6834", "#1baf7a", "#8e44ad", "#e34948")):
        rs = [r for r in rows if r["method"] == n and r["share_filled"] > 0.5]
        axs[1].scatter([r["plain_colour_err"] for r in rs], [r["colour_err"] for r in rs], s=10, alpha=0.5, color=col, label=n)
    lim = max([r["plain_colour_err"] for r in rows] + [r["colour_err"] for r in rows] + [1.0])
    axs[1].plot([0, lim], [0, lim], color="#c3c2b7", lw=1)
    axs[1].set_xlabel("colour error of the plain fill (0-255)"); axs[1].set_ylabel("colour error of the method")
    axs[1].set_title("below the line: closer in colour than plain", fontsize=10)
    axs[1].legend(fontsize=8, frameon=False)
    for a in axs:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
