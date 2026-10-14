"""Straight outlines from traced roof masks.

A roof traced on a grid has a stepped, wobbling edge; buildings mostly have
straight walls at right angles. ``regularize`` turns a traced outline into
straight edges snapped to the building's own directions, each placed through
the traced edge (scripts/lidar_outline.py's tower outlines first, then the
colourise's upper tiers, building_levels).
"""

from __future__ import annotations

import numpy as np


def main_directions(ring) -> float:
    """The building's main direction (radians, 0-90 degrees): the length-weighted
    mean of its outline's edge directions taken modulo 90 degrees."""
    P = np.asarray(ring, float)
    d = np.diff(P, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    a = np.arctan2(d[:, 1], d[:, 0])
    return float(np.angle(np.sum(L * np.exp(4j * a))) / 4.0)


def regularize(poly, theta: float, snap_deg: float = 20.0, min_edge_m: float = 2.0, simplify_m: float = 1.0):
    """A traced roof outline as straight edges: simplified by ``simplify_m``,
    each run snapped to the building's directions (``theta`` and at right
    angles) when within ``snap_deg``, neighbouring runs in one direction
    merged, runs under ``min_edge_m`` folded into their longer neighbour, and
    each edge then placed through the traced boundary's points nearest it
    (direction kept, position their median), corners re-intersected. Falls
    back to the simplified outline when the straightened one is not a simple
    polygon. Returns a closed ring, or None."""
    from shapely.geometry import LineString, Polygon
    g = poly.simplify(simplify_m, preserve_topology=True)
    if g.geom_type != "Polygon" or len(g.exterior.coords) < 4:
        return None
    ext = g.exterior if g.exterior.is_ccw else LineString(list(g.exterior.coords)[::-1])
    P = np.asarray(ext.coords)[:-1]
    bd = poly.exterior                                                     # the traced boundary, densely
    B = np.array([bd.interpolate(t).coords[0] for t in np.arange(0, bd.length, 0.25)])
    runs = []
    for k in range(len(P)):
        a_, b_ = P[k], P[(k + 1) % len(P)]
        v = b_ - a_
        L = float(np.hypot(*v))
        if L < 1e-6:
            continue
        ang = float(np.arctan2(v[1], v[0]))
        snapped = theta + round((ang - theta) / (np.pi / 2)) * np.pi / 2
        if abs(np.angle(np.exp(1j * (ang - snapped)))) <= np.radians(snap_deg):
            ang = snapped
        runs.append(dict(ang=ang, a=a_, b=b_, L=L))
    def same(r, q):
        return abs(np.angle(np.exp(1j * (r["ang"] - q["ang"])))) < 1e-6
    merged = []
    for r in runs:
        if merged and same(merged[-1], r):
            merged[-1].update(b=r["b"], L=merged[-1]["L"] + r["L"])
        else:
            merged.append(dict(r))
    if len(merged) > 1 and same(merged[0], merged[-1]):
        merged[-1].update(b=merged[0]["b"], L=merged[-1]["L"] + merged[0]["L"]); merged.pop(0)
    while len(merged) > 4 and min(m["L"] for m in merged) < min_edge_m:
        k = int(np.argmin([m["L"] for m in merged]))
        prv, nxt = merged[k - 1], merged[(k + 1) % len(merged)]
        if prv["L"] >= nxt["L"]:
            prv.update(b=merged[k]["b"], L=prv["L"] + merged[k]["L"])
        else:
            nxt.update(a=merged[k]["a"], L=nxt["L"] + merged[k]["L"])
        merged.pop(k)
        i = 0                                                              # folding can leave two runs in one direction side by side
        while i < len(merged) and len(merged) > 1:
            j = (i + 1) % len(merged)
            if same(merged[i], merged[j]) and i != j:
                merged[i].update(b=merged[j]["b"], L=merged[i]["L"] + merged[j]["L"]); merged.pop(j)
            else:
                i += 1
    if len(merged) < 3:
        return None
    # each run's line through the traced boundary points nearest it
    segs = [LineString([m["a"], m["b"]]) for m in merged]
    from shapely.geometry import Point
    owner = np.array([int(np.argmin([sg.distance(Point(*q)) for sg in segs])) for q in B])
    lines = []
    for k, m in enumerate(merged):
        d = np.array([np.cos(m["ang"]), np.sin(m["ang"])])
        nrm = np.array([d[1], -d[0]])
        pts = B[owner == k]
        c = float(np.median(pts @ nrm)) if len(pts) >= 3 else float(np.asarray(m["a"]) @ nrm)
        lines.append((nrm, c))
    out = []
    for k in range(len(lines)):
        (n0, c0), (n1, c1) = lines[k - 1], lines[k]
        A = np.array([n0, n1])
        if abs(np.linalg.det(A)) < 0.17:                                   # under about 10 degrees apart: keep the simplified corner
            out.append(np.asarray(merged[k]["a"]))
        else:
            out.append(np.linalg.solve(A, [c0, c1]))
    ring = np.vstack([out, out[:1]])
    if Polygon(ring).is_valid and Polygon(ring).area > 0.5 * poly.area:
        return ring
    return np.vstack([P, P[:1]])
