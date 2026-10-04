#!/usr/bin/env python3
"""Draw what rating roofs separately does to the complexity survey.

Usage::

    python scripts/plot_survey_zones.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kyfromabove_3d.plots import plot_survey_zones


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--random", type=Path,
                    default=Path("outputs/survey/complexity_random.json"))
    ap.add_argument("--dense", type=Path,
                    default=Path("outputs/survey/complexity_dense.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/survey/survey_zones.png"))
    args = ap.parse_args()

    arms = []
    for path, name in ((args.random, "statewide"), (args.dense, "urban")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        rows = blob["results"]
        if not any("roof_rate" in r for r in rows):
            print(f"[FAIL] {path} predates the wall/roof split; re-run the survey")
            return 1
        # The seed decides which tiles a random arm measures out of the 1,805
        # holding a building, and a different seed is a different answer, so it
        # belongs on the label.
        p = blob.get("params", {})
        seed = f", seed {p['seed']}" if p.get("sample") == "random" else ""
        arms.append({"label": f"{name} ({len(rows)} buildings{seed})",
                     "results": rows})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_survey_zones(arms, save_to=args.out)
    print(f"figure: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
