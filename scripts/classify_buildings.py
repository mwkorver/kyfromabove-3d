#!/usr/bin/env python3
"""Sort the downtown's buildings by size and architecture, from raw evidence.

The wall pipeline's settings were made on towers and fail small buildings
(fb099a5d, one storey, read its flat roof as colonnades); and the pipeline's
own archetype (``classify_building_structure``) is read off the pipeline's
results, so it inherits those mistakes -- 289 of the 401 downtown buildings
came out "sculpted_facade". This sorts each building before anything is
modelled, from the footprint and the LiDAR alone, so that each class can be
given its own strategy.

Measures, per building (the downtown frame: the ground at 0 m):

* ``height_m`` -- the 99th percentile of the returns on the footprint;
* ``area_m2`` and ``width_m`` -- the footprint's area, and its narrowest
  width (twice the largest circle that fits inside it);
* ``flat_share`` / ``pitched_share`` -- of the roof cells (0.5 m, highest
  return, 2 m or more above the ground), those sloping under 10 degrees, and
  between 15 and 60 degrees;
* ``tiers`` -- flat roof levels holding 10% or more of the roof each and 3 m
  or more apart; ``tier_span_m`` from the lowest to the highest;
* ``through_share`` -- of the returns 2 m or more inside the footprint, those
  2 m or more above the ground and 3 m or more under the highest return in
  their own 0.5 m cell: the laser getting in under a roof (open parking
  decks, canopies) rather than stopping at it, a lower roof tier included;
* ``party_share`` -- the share of the footprint's perimeter within 1 m of a
  neighbour's.

Classes: size by height (low under 10 m, mid 10-30 m, tall over 30 m) and by
footprint (small under 300 m2, medium, large over 2,000 m2; thin under 5 m
across; ``none`` for a footprint the LiDAR shows no building on: under 40 m2,
under 2.5 m tall, or under 30% roof); architecture, first match: ``tiered``
(stepped roof), ``pitched`` (sloping roof), ``row1`` / ``row2o`` / ``row2c`` /
``row3`` (one shared side; two facing, the middle of a row; two adjacent, a
corner; three or four, infill), ``box``. ``see_through`` (through_share
0.15 or more) is a hint at a glass facade, not a class: an open parking deck
does not show this way, its top deck hiding the ones below.

Usage::

    python scripts/classify_buildings.py measure     # -> outputs/classes/building_classes.json
    python scripts/classify_buildings.py sheet       # a contact sheet per class -> outputs/classes/
    python scripts/classify_buildings.py baseline    # every building audited as built now, summed per class
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BUILDINGS = ROOT / "outputs" / "downtown" / "buildings"
OUT = ROOT / "outputs" / "classes"
NOISE = (7, 18)


def measure_one(job) -> dict:
    import shapely
    from scipy.ndimage import median_filter

    from kyfromabove_3d.clouds import find_cloud, read_cloud
    from kyfromabove_3d.footprints import footprint_wgs84_to_local

    d, a = job
    rec = {"building": d.name}
    try:
        fp = json.loads((d / "footprint.json").read_text(encoding="utf-8"))
        ox, oy, _ = json.loads((d / "coordinates.json").read_text(encoding="utf-8"))["origin_source_xyz"]
        poly = footprint_wgs84_to_local(fp["wkt"], ox, oy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        L = read_cloud(find_cloud(d, "lidar_local"))
    except Exception as exc:                                    # noqa: BLE001
        return dict(rec, status=f"skipped: {exc}")
    rec["name"] = fp.get("name") if fp.get("name") and fp.get("name") != d.name else None
    xyz = np.column_stack([L["x"], L["y"], L["z"]]).astype(float)
    if "classification" in L:
        xyz = xyz[~np.isin(np.asarray(L["classification"]), NOISE)]
    on = shapely.contains_xy(poly, xyz[:, 0], xyz[:, 1])
    if on.sum() < 30:
        return dict(rec, status="too few returns on the footprint")
    z_on = xyz[on, 2]
    height = float(np.percentile(z_on, 99))
    rec["height_m"] = round(height, 1)
    rec["area_m2"] = round(float(poly.area), 1)
    mic = shapely.maximum_inscribed_circle(poly)
    rec["width_m"] = round(2.0 * float(mic.length), 1)
    # roof grid
    c = a.cell_m
    minx, miny, maxx, maxy = poly.bounds
    nx, ny = int((maxx - minx) / c) + 1, int((maxy - miny) / c) + 1
    ix = ((xyz[on, 0] - minx) / c).astype(int).clip(0, nx - 1)
    iy = ((xyz[on, 1] - miny) / c).astype(int).clip(0, ny - 1)
    g = np.full((ny, nx), -np.inf)
    np.maximum.at(g, (iy, ix), z_on)
    X, Y = np.meshgrid(minx + (np.arange(nx) + 0.5) * c, miny + (np.arange(ny) + 0.5) * c)
    inner = shapely.contains_xy(poly.buffer(-0.5), X, Y) if not poly.buffer(-0.5).is_empty else np.zeros_like(X, bool)
    roof = inner & np.isfinite(g) & (g >= a.min_roof_m)
    rec["roof_cover"] = round(float(roof.sum() / max(inner.sum(), 1)), 2)
    if roof.sum() >= 8:
        gs = median_filter(np.where(np.isfinite(g), g, 0.0), size=3)
        gy, gx = np.gradient(gs, c)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        ok = roof & median_filter(roof.astype(np.uint8), size=3).astype(bool)   # away from the roof's own edge
        n_ok = max(int(ok.sum()), 1)
        rec["flat_share"] = round(float((ok & (slope < 10)).sum() / n_ok), 2)
        rec["pitched_share"] = round(float((ok & (slope >= 15) & (slope <= 60)).sum() / n_ok), 2)
        flat_h = g[ok & (slope < 10)]
        tiers = []
        if len(flat_h) >= 8:
            hist, edges = np.histogram(flat_h, bins=np.arange(np.floor(flat_h.min()), np.ceil(flat_h.max()) + 1.01, 1.0))
            share = hist / hist.sum()
            for k in np.argsort(-share):
                if share[k] < 0.10:
                    break
                lvl = float(edges[k] + 0.5)
                if all(abs(lvl - t) >= 3.0 for t in tiers):
                    tiers.append(lvl)
        rec["tiers"] = len(tiers)
        rec["tier_span_m"] = round(max(tiers) - min(tiers), 1) if len(tiers) >= 2 else 0.0
    else:
        rec.update(flat_share=None, pitched_share=None, tiers=0, tier_span_m=0.0)
    # returns getting into the building
    core = poly.buffer(-2.0)
    if not core.is_empty and height > 5.0:
        in_core = shapely.contains_xy(core, xyz[on, 0], xyz[on, 1])
        top = g[iy[in_core], ix[in_core]]                       # the highest return in each return's own cell
        zc = z_on[in_core]
        under = (zc >= 2.0) & (zc <= top - 3.0)                 # got in under a roof, not a lower roof tier
        rec["through_share"] = round(float(under.sum() / max(len(zc), 1)), 3)
    else:
        rec["through_share"] = None
    # party walls
    nbs = []
    for n in fp.get("neighbours") or []:
        try:
            q = footprint_wgs84_to_local(n["wkt"], ox, oy)
            nbs.append(q if q.is_valid else q.buffer(0))
        except Exception:                                        # noqa: BLE001
            pass
    if nbs:
        near = shapely.union_all(nbs).buffer(1.0)
        rec["party_share"] = round(float(poly.exterior.intersection(near).length / max(poly.exterior.length, 1e-9)), 2)
        rec["shared_sides"], rec["shared_layout"] = shared_sides(poly, near, a.side_share)
    else:
        rec["party_share"], rec["shared_sides"], rec["shared_layout"] = 0.0, 0, "none"
    rec["split_part"] = d.name[-2:] in ("-a", "-b", "-c", "-d", "-e")
    rec["status"] = "ok"
    rec.update(classify(rec, a))
    # the class beside the building, for the build to choose its strategy (configs/class_strategies.yaml)
    (d / "building_class.json").write_text(json.dumps({k: rec[k] for k in (
        "class", "architecture", "height_class", "footprint_class", "see_through", "height_m", "area_m2",
        "width_m", "party_share", "shared_sides", "shared_layout") if k in rec}, indent=1), encoding="utf-8")
    return rec


def shared_sides(poly, near, min_share: float):
    """How many of the building's four sides stand against a neighbour: the
    outline's edges grouped by which way they face, in the building's own
    frame (its main direction, from the edges, modulo 90 degrees), a side
    shared when ``min_share`` of its length is within 1 m of a neighbour.
    Returns (count, layout): layout "opposite" for two facing sides (a
    building in the middle of a row), "corner" for two adjacent ones."""
    P = np.asarray(poly.exterior.coords)
    d = np.diff(P, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    ang = np.arctan2(d[:, 1], d[:, 0])
    theta = float(np.angle(np.sum(L * np.exp(4j * ang))) / 4.0)
    area2 = np.sum(P[:-1, 0] * P[1:, 1] - P[1:, 0] * P[:-1, 1])
    out_ang = ang - np.pi / 2 if area2 > 0 else ang + np.pi / 2          # each edge's outward normal
    side = (np.round(((out_ang - theta) % (2 * np.pi)) / (np.pi / 2)).astype(int)) % 4
    from shapely.geometry import LineString
    total, shared = np.zeros(4), np.zeros(4)
    for k in range(len(d)):
        if L[k] < 1e-6:
            continue
        seg = LineString([P[k], P[k + 1]])
        total[side[k]] += L[k]
        shared[side[k]] += seg.intersection(near).length
    on = [i for i in range(4) if total[i] > 0 and shared[i] / total[i] >= min_share]
    if len(on) == 2:
        layout = "opposite" if (on[1] - on[0]) % 2 == 0 else "corner"
    else:
        layout = {0: "none", 1: "one", 3: "three", 4: "four"}.get(len(on), str(len(on)))
    return len(on), layout


def classify(r: dict, a) -> dict:
    h, area, w = r["height_m"], r["area_m2"], r["width_m"]
    height_class = "low" if h < 10 else ("mid" if h < 30 else "tall")
    if area < a.tiny_m2 or h < a.min_height_m or (r.get("roof_cover") or 0) < a.min_roof_cover:
        # a footprint with no building the LiDAR can see on it: a kiosk, a
        # stair head, grass where a building was
        return {"height_class": height_class, "footprint_class": "tiny", "architecture": "none",
                "see_through": False, "class": "none"}
    size = "thin" if w < a.thin_m else ("small" if area < 300 else ("medium" if area < 2000 else "large"))
    if r.get("tiers", 0) >= 2 and r.get("tier_span_m", 0) >= max(4.0, 0.2 * h):
        arch = "tiered"
    elif (r.get("pitched_share") or 0) >= a.pitched_share:
        arch = "pitched"
    elif r.get("shared_sides", 0) >= 1:
        # by how many sides stand against a neighbour: one (the end of a row,
        # or a building backing onto another), two facing (the middle of a
        # row), two adjacent (a corner of a block), three or four (infill)
        arch = {1: "row1", 2: "row2" + ("o" if r.get("shared_layout") == "opposite" else "c")}.get(r["shared_sides"], "row3")
    else:
        arch = "box"
    # the laser getting in under the roof: through glass on a tower (Truist,
    # RIVUE, LG&E, Waterfront Plaza) -- a hint at the facade, not a class yet;
    # a parking deck's lower decks are hidden by its top one and do not show
    glassy = r.get("through_share") is not None and r["through_share"] >= a.open_share
    return {"height_class": height_class, "footprint_class": size, "architecture": arch,
            "see_through": bool(glassy), "class": f"{arch}/{height_class}"}


def cmd_measure(a) -> int:
    import multiprocessing as mp
    dirs = sorted(d for d in a.buildings.iterdir() if (d / "footprint.json").exists())
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        recs = list(pool.imap_unordered(measure_one, [(d, a) for d in dirs]))
    recs.sort(key=lambda r: r["building"])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "building_classes.json").write_text(json.dumps(recs, indent=1), encoding="utf-8")
    ok = [r for r in recs if r["status"] == "ok"]
    print(f"{len(ok)} of {len(recs)} buildings measured ({time.time() - t0:.0f} s); "
          f"skipped: {[(r['building'][:8], r['status']) for r in recs if r['status'] != 'ok']}")
    from collections import Counter
    cnt = Counter((r["architecture"], r["height_class"]) for r in ok)
    archs = ["box", "row1", "row2o", "row2c", "row3", "pitched", "tiered"]
    print(f"not a building the LiDAR can see (tiny, no roof): {sum(1 for r in ok if r['architecture'] == 'none')}")
    print(f"{'':10s}" + "".join(f"{h:>8s}" for h in ("low", "mid", "tall")))
    for ar in archs:
        print(f"{ar:10s}" + "".join(f"{cnt.get((ar, h), 0):8d}" for h in ("low", "mid", "tall")))
    print(f"-> {a.out / 'building_classes.json'}")
    return 0


def thumbnail(d: Path, size_px: int = 220, azimuth_deg: float = 210.0, elevation_deg: float = 35.0,
              margin_m: float = 6.0, height_m: Optional[float] = None):
    """The building from the side and above: its returns within ``margin_m``
    of the footprint, coloured from the aerial frames (colorized_aerial.laz),
    drawn far to near, and the footprint in red at the roof's height."""
    import shapely
    from kyfromabove_3d.clouds import read_cloud
    from kyfromabove_3d.footprints import footprint_wgs84_to_local
    p = d / "colorized_aerial.laz"
    if not p.exists():
        return None
    L = read_cloud(p)
    if not all(k in L for k in ("red", "green", "blue")):
        return None
    xyz = np.column_stack([L["x"], L["y"], L["z"]]).astype(float)
    rgb = np.column_stack([L["red"], L["green"], L["blue"]]).astype(float)
    if rgb.max() > 255:
        rgb = rgb / 256.0
    fp = json.loads((d / "footprint.json").read_text(encoding="utf-8"))
    ox, oy, _ = json.loads((d / "coordinates.json").read_text(encoding="utf-8"))["origin_source_xyz"]
    poly = footprint_wgs84_to_local(fp["wkt"], ox, oy)
    keep = shapely.contains_xy(poly.buffer(margin_m), xyz[:, 0], xyz[:, 1])
    xyz, rgb = xyz[keep], rgb[keep]
    if len(xyz) < 10:
        return None
    if len(xyz) > 150000:
        k = np.random.default_rng(0).choice(len(xyz), 150000, replace=False)
        xyz, rgb = xyz[k], rgb[k]
    ring = np.asarray(poly.exterior.coords)
    ring = np.column_stack([ring, np.full(len(ring), height_m if height_m is not None else np.percentile(xyz[:, 2], 99))])
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    right = np.array([np.cos(az), -np.sin(az), 0.0])
    fwd = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), -np.sin(el)])
    up = np.cross(right, fwd)
    c0 = xyz.mean(axis=0)
    u, v, depth = (xyz - c0) @ right, (xyz - c0) @ up, (xyz - c0) @ fwd
    span = max(np.ptp(u), np.ptp(v), 1.0)
    u0, v0 = u.min(), v.min()
    to_px = lambda uu, vv: ((uu - u0) / span * (size_px - 12) + 6, size_px - 6 - (vv - v0) / span * (size_px - 12))
    px, py = (np.asarray(t).astype(int) for t in to_px(u, v))
    img = np.full((size_px, size_px, 3), 245, dtype=np.uint8)
    order = np.argsort(-depth)                                   # far first
    for dx in (0, 1):
        for dy in (0, 1):
            img[(py[order] + dy).clip(0, size_px - 1), (px[order] + dx).clip(0, size_px - 1)] = rgb[order].clip(0, 255).astype(np.uint8)
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    rx, ry = to_px((ring - c0) @ right, (ring - c0) @ up)
    ImageDraw.Draw(im).line(list(zip(rx.tolist(), ry.tolist())), fill=(220, 30, 30), width=2)
    return np.asarray(im)


def cmd_sheet(a) -> int:
    from PIL import Image, ImageDraw, ImageFont
    recs = [r for r in json.loads((a.out / "building_classes.json").read_text(encoding="utf-8")) if r["status"] == "ok"]
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except OSError:
        font = big = ImageFont.load_default()
    by = {}
    for r in recs:
        by.setdefault(r["class"], []).append(r)
    T, cols = 220, a.columns
    index = []
    for cls, rs in sorted(by.items()):
        rng = np.random.default_rng(0)
        pick = [rs[i] for i in sorted(rng.choice(len(rs), min(len(rs), a.per_class), replace=False))]
        rows = (len(pick) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * (T + 8) + 8, rows * (T + 40) + 50), "white")
        dr = ImageDraw.Draw(sheet)
        dr.text((10, 12), f"{cls}: {len(rs)} buildings (showing {len(pick)})", font=big, fill="black")
        for k, r in enumerate(pick):
            x, y = 8 + (k % cols) * (T + 8), 46 + (k // cols) * (T + 40)
            img = thumbnail(a.buildings / r["building"], T, height_m=r["height_m"])
            if img is not None:
                sheet.paste(Image.fromarray(img), (x, y))
            label = (r.get("name") or r["building"][:8])[:30]
            dr.text((x + 2, y + T + 2), label, font=font, fill="black")
            dr.text((x + 2, y + T + 16), f"{r['height_m']:.0f} m, {r['area_m2']:.0f} m2, party {r['party_share']:.0%}, "
                                         f"through {r['through_share'] if r['through_share'] is not None else '-'}", font=font, fill="#555")
        path = a.out / f"sheet_{cls.replace('/', '_')}.jpg"
        sheet.save(path, quality=85)
        index.append((cls, len(rs), path.name))
        print(f"  {cls}: {len(rs)} -> {path.name}")
    return 0


def audit_one(job) -> dict:
    """One building's audit as built now (audit_building.audit, against the
    LiDAR solid within its footprint), with its wall area for scale."""
    d, outline = job
    sys.path.insert(0, str(ROOT / "scripts"))
    from audit_building import audit
    try:
        r = audit(d, outline=outline)
    except Exception as exc:                                    # noqa: BLE001
        return {"building": d.name, "status": f"audit failed: {exc}"}
    h = r["holes"]
    return {"building": d.name, "status": "ok",
            "holes_views_m2": round(sum(v.get("hole_m2", 0) or 0 for v in h.get("views", [])), 1),
            "inside_m2": (r.get("inside_walls") or {}).get("m2"),
            "outside_m2": (r.get("stray_walls") or {}).get("outside_m2"),
            "above_m2": (r.get("stray_walls") or {}).get("above_m2"),
            "blank_share": (r.get("texture") or {}).get("blank_share"),
            "roof_gap_m2": (r.get("roof_gap") or {}).get("m2"), "roof_gap_share": (r.get("roof_gap") or {}).get("share")}


def cmd_baseline(a) -> int:
    """Every classified building audited as built now, and per class: holes
    (over the nine views, per 100 m2 of the building's wall), wall drawn
    inside the building and wall drawn where the LiDAR has none (each as a
    share of its wall), and blank wall -- so a one-storey building counts as
    much as a tower."""
    import multiprocessing as mp
    import shapely
    from kyfromabove_3d.footprints import footprint_wgs84_to_local
    recs = {r["building"]: r for r in json.loads((a.out / "building_classes.json").read_text(encoding="utf-8")) if r["status"] == "ok"}
    keys = sorted(k for k, r in recs.items() if r["architecture"] != "none" and (a.buildings / k / "3dtiles_mesh" / "building.glb").exists())
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        res = list(pool.imap_unordered(audit_one, [(a.buildings / k, a.audit_outline) for k in keys]))
    rows = []
    for r in res:
        if r["status"] != "ok":
            continue
        c = recs[r["building"]]
        d = a.buildings / r["building"]
        fp = json.loads((d / "footprint.json").read_text(encoding="utf-8"))
        ox, oy, _ = json.loads((d / "coordinates.json").read_text(encoding="utf-8"))["origin_source_xyz"]
        poly = footprint_wgs84_to_local(fp["wkt"], ox, oy)
        wall = max(float(poly.exterior.length) * c["height_m"] * (1.0 - c.get("party_share", 0.0)), 1.0)   # wall the street sees
        rows.append({**r, "class": c["class"], "architecture": c["architecture"], "height_class": c["height_class"],
                     "name": c.get("name"), "wall_m2": round(wall, 1)})
    (a.out / "baseline.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{len(rows)} buildings audited ({time.time() - t0:.0f} s); failed: {[r['building'][:8] for r in res if r['status'] != 'ok']}")
    by = {}
    for r in rows:
        by.setdefault(r["class"], []).append(r)
    print(f"{'class':14s} {'n':>4s} {'holes/100m2':>12s} {'inside %':>9s} {'outside %':>10s} {'blank %':>8s}   (medians over the class's buildings)")
    summary = []
    for cls, rs in sorted(by.items()):
        med = lambda f: float(np.median([f(r) for r in rs]))
        row = dict(cls=cls, n=len(rs),
                   holes=med(lambda r: 100 * r["holes_views_m2"] / r["wall_m2"]),
                   inside=med(lambda r: 100 * (r["inside_m2"] or 0) / r["wall_m2"]),
                   outside=med(lambda r: 100 * (r["outside_m2"] or 0) / r["wall_m2"]),
                   blank=med(lambda r: 100 * (r["blank_share"] or 0)))
        summary.append(row)
        print(f"{cls:14s} {row['n']:4d} {row['holes']:12.1f} {row['inside']:9.1f} {row['outside']:10.1f} {row['blank']:8.0f}")
    (a.out / "baseline_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("measure", "sheet", "baseline"):
        p = sub.add_parser(name)
        p.add_argument("--buildings", type=Path, default=BUILDINGS)
        p.add_argument("--out", type=Path, default=OUT)
        p.add_argument("--cell-m", type=float, default=0.5)
        p.add_argument("--min-roof-m", type=float, default=2.0, help="a roof cell is at least this far above the ground")
        p.add_argument("--thin-m", type=float, default=5.0, help="narrower than this across: a thin part")
        p.add_argument("--open-share", type=float, default=0.15, help="returns in under the roof for the see-through (glass) hint")
        p.add_argument("--pitched-share", type=float, default=0.35, help="sloping roof cells for 'pitched'")
        p.add_argument("--side-share", type=float, default=0.4, help="share of a side's length against a neighbour to count it shared")
        p.add_argument("--tiny-m2", type=float, default=40.0, help="a footprint smaller than this is not a building to model")
        p.add_argument("--min-height-m", type=float, default=2.5, help="... nor one lower than this")
        p.add_argument("--min-roof-cover", type=float, default=0.3, help="... nor one with less roof than this share of it")
        p.add_argument("--workers", type=int, default=6)
        p.add_argument("--per-class", type=int, default=24)
        p.add_argument("--columns", type=int, default=6)
        p.add_argument("--audit-outline", choices=("footprint", "lidar"), default="footprint",
                       help="the LiDAR solid within Overture's footprint (as built now) or within the LiDAR outline")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = parse_args(argv)
    return dict(measure=cmd_measure, sheet=cmd_sheet, baseline=cmd_baseline)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
