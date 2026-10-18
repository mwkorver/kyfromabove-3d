"""Party walls: the edges a building shares with a neighbour, and where the shared wall
steps (the fallback mesh's walls)."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------- glTF

# ------------------------------------------------------------- the fallback

def edge_top_z(xs, ys, Z, valid, p0: np.ndarray, p1: np.ndarray, within_m: float = 2.0,
               default: float = 6.0) -> float:
    """The roof height along one footprint edge: the median of the valid
    height-field cells within ``within_m`` of the edge. A podium's edge is
    not extruded to the tower's top this way."""
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X[valid], Y[valid]])
    zs = Z[valid]
    if not len(pts):
        return default
    d = p1 - p0
    L2 = float(d @ d) + 1e-9
    t = np.clip(((pts - p0) @ d) / L2, 0.0, 1.0)
    dist = np.linalg.norm(pts - (p0 + t[:, None] * d), axis=1)
    near = zs[dist <= within_m]
    return float(np.median(near)) if len(near) >= 3 else default


def party_edges(ring_m: np.ndarray, neighbours: list, faces, within_m: float = 0.5,
                min_run_m: float = 1.0, tops: list | None = None) -> dict:
    """Where the faces are party walls: the stretches of each edge that lie
    within ``within_m`` of a neighbouring footprint. Nothing sees a shared
    wall -- no return, no camera -- so it is drawn plain rather than painted
    with whatever stands behind it. Only up to the neighbour's roof, though:
    above it the wall stands in the open and the cameras see it. Returns
    ``{face: [(along_lo, along_hi, the neighbour's roof as local z, or None
    when unknown), ...]}`` -- one stretch per neighbour the edge touches, so
    an edge shared with a tall building on one half and a low one on the
    other is plain to a different height on each."""
    from shapely.geometry import LineString

    if not neighbours:
        return {}
    out: dict = {}
    tops = list(tops) if tops is not None else [None] * len(neighbours)
    zones = [g.buffer(within_m) for g in neighbours]
    for f in faces:
        edge = LineString(f.world_line())
        if edge.length <= 0:
            continue
        stretches = []
        for zone, t in zip(zones, tops):
            inter = edge.intersection(zone)
            pieces = list(inter.geoms) if hasattr(inter, "geoms") else [inter]
            for piece in pieces:
                if piece.is_empty or piece.geom_type != "LineString" or piece.length < min_run_m:
                    continue
                along, _ = f.local(np.array(piece.coords))
                s0, s1 = float(along.min()), float(along.max())
                if s1 - s0 >= min_run_m:
                    stretches.append((s0, s1, None if t is None else float(t)))
        if stretches:
            # by where they lie, an unknown roof (None) before a known one: two
            # neighbours on the same stretch, one with no roof height, could not
            # be compared (ce8b50e8)
            out[f.face] = sorted(stretches, key=lambda s: (s[0], s[1], -np.inf if s[2] is None else s[2]))
    return out


def party_steps(stretches, a: float, b: float, z_top: float, min_open_m: float = 0.5):
    """Cut one prism run ``[a, b]`` up to ``z_top`` by the party stretches:
    ``(c0, c1, plain_top)`` pieces, ``plain_top`` the height the wall is
    plain up to (``None`` where the wall is in the open the whole way)."""
    cuts = {a, b}
    for s0, s1, _ in stretches:
        for c in (s0, s1):
            if a < c < b:
                cuts.add(c)
    cuts = sorted(cuts)
    out = []
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        mid = 0.5 * (c0 + c1)
        covering = [t for s0, s1, t in stretches if s0 - 1e-6 <= mid <= s1 + 1e-6]
        if not covering:
            plain = None
        elif any(t is None for t in covering):
            plain = z_top
        else:
            plain = min(max(covering), z_top)
            if plain >= z_top - min_open_m:
                plain = z_top
        if out and out[-1][2] == plain:
            out[-1] = (out[-1][0], c1, plain)
        else:
            out.append((c0, c1, plain))
    return out
