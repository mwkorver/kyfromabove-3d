"""A building's class, by size and architecture, from its footprint and its
LiDAR alone -- before anything is modelled -- so each class can be built
with its own strategy (configs/class_strategies.yaml, kyfromabove_3d.settings).

Measures, per building (its own frame: the ground at 0 m):

* ``height_m`` -- the 99th percentile of the returns on the footprint;
* ``area_m2`` and ``width_m`` -- the footprint's area, and its narrowest
  width (twice the largest circle that fits inside it);
* ``flat_share`` / ``pitched_share`` -- of the roof cells (0.5 m, highest
  return, 2 m or more above the ground), those sloping under 10 degrees, and
  between 15 and 60 degrees;
* ``tiers`` -- flat roof levels holding 10% or more of the roof each and 3 m
  or more apart; ``tier_span_m`` from the lowest to the highest; and
  ``lower_share``, the roof standing well below its main height;
* ``through_share`` -- the laser getting in under a roof;
* ``party_share`` and ``shared_sides`` -- how much of the outline, and how
  many of its four sides, stand against a neighbour.

Classes: height low (under 10 m), mid (10-30 m), tall; architecture, first
match: ``tiered``, ``pitched``, ``row1`` / ``row2o`` / ``row2c`` / ``row3``
(one shared side; two facing; two adjacent; three or four), ``box``; and
``none`` for a footprint the LiDAR shows no building on. Moved here from
scripts/classify_buildings.py (2026-10-14) so the building step classifies
each building itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .lidar import VEGETATION
from .settings import ClassifySettings

NOISE = (7, 18)
#: what building_class.json holds
CLASS_FIELDS = ("class", "architecture", "height_class", "footprint_class", "see_through", "height_m", "area_m2",
                "width_m", "party_share", "shared_sides", "shared_layout")


def measure(d: Path, a: ClassifySettings) -> dict:
    """The building's measures (module docstring) and its class; nothing written."""
    import shapely
    from scipy.ndimage import median_filter

    from kyfromabove_3d.clouds import find_cloud, read_cloud
    from kyfromabove_3d.footprints import footprint_wgs84_to_local

    d = Path(d)
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
        xyz = xyz[~np.isin(np.asarray(L["classification"]), NOISE + VEGETATION)]
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
    # the roof away from its own edge; a roof too small to have any such cell
    # (a 6 m shed: 3a7c18d7) is measured as one with too few cells
    ok = roof & median_filter(roof.astype(np.uint8), size=3).astype(bool) if roof.sum() >= 8 else np.zeros_like(roof)
    if ok.any():
        gs = median_filter(np.where(np.isfinite(g), g, 0.0), size=3)
        gy, gx = np.gradient(gs, c)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        n_ok = int(ok.sum())
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
        # A lower part of the roof spread over several heights (the Galt House's
        # wings at 12-24 m beside a 50 m block) makes no single 1 m level of 10%,
        # and the tiers above missed it: the share of the roof's cells standing
        # well below its main height -- 4 m, or a fifth of the building's height,
        # whichever is more -- counts too (not on a pitched roof, whose eaves
        # stand below its ridge; the wings' roofs are not flat enough to count
        # flat cells alone: 4.5% of the Galt House's roof).
        hist_all, e_all = np.histogram(g[ok], bins=np.arange(np.floor(g[ok].min()), np.ceil(g[ok].max()) + 1.01, 1.0))
        main = float(e_all[int(np.argmax(hist_all))] + 0.5)
        low = ok & (g >= a.min_roof_m) & (g <= main - max(4.0, 0.2 * height))
        rec["main_roof_m"] = round(main, 1)
        rec["lower_share"] = round(float(low.sum() / n_ok), 3)
    else:
        rec.update(flat_share=None, pitched_share=None, tiers=0, tier_span_m=0.0, main_roof_m=None, lower_share=None)
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


def classify(r: dict, a: ClassifySettings) -> dict:
    """The class from a building's measures."""
    h, area, w = r["height_m"], r["area_m2"], r["width_m"]
    height_class = "low" if h < 10 else ("mid" if h < 30 else "tall")
    if area < a.tiny_m2 or h < a.min_height_m or (r.get("roof_cover") or 0) < a.min_roof_cover:
        # a footprint with no building the LiDAR can see on it: a kiosk, a
        # stair head, grass where a building was
        return {"height_class": height_class, "footprint_class": "tiny", "architecture": "none",
                "see_through": False, "class": "none"}
    size = "thin" if w < a.thin_m else ("small" if area < 300 else ("medium" if area < 2000 else "large"))
    if (r.get("tiers", 0) >= 2 and r.get("tier_span_m", 0) >= max(4.0, 0.2 * h)) or \
            ((r.get("lower_share") or 0) >= a.lower_share and (r.get("pitched_share") or 0) < a.pitched_share):
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


def write_building_class(d: Path, rec: dict) -> None:
    """The class beside the building (building_class.json), for the build to
    choose its strategy."""
    (Path(d) / "building_class.json").write_text(json.dumps({k: rec[k] for k in CLASS_FIELDS if k in rec}, indent=1),
                                                 encoding="utf-8")


def classify_building(d: Path, settings: Optional[ClassifySettings] = None) -> dict:
    """Measure, classify and write building_class.json (when it measured);
    returns the record."""
    rec = measure(d, settings or ClassifySettings())
    if rec.get("status") == "ok":
        write_building_class(d, rec)
    return rec
