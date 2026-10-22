#!/usr/bin/env python3
"""Overture buildings as extruded boxes, for comparison on the globe.

What Cesium OSM Buildings and the like draw: the footprint pulled up to the
height the data carries. Overture's downtown Louisville buildings have
OpenStreetMap footprints, and heights that mostly come from Microsoft's
building-height model (``sources[].dataset = 'Microsoft ML Buildings'``);
a building with only a floor count is drawn at 3.5 m a floor, and one with
neither at a default, so the guessing shows.

The colour says where the height came from: grey for a measured/tagged
height, blue-grey for floors x 3.5, orange for the default. The three
buildings this project models are drawn brighter.

Input is the JSON ``fetch`` writes (one query of the Overture release over a
bounding box); output is one glTF of every box in the local frame of a
project building, and a tileset placing it.

Usage::

    python scripts/overture_boxes.py fetch --bbox -85.764 38.25 -85.75 38.26 --out data/derived/louisville_downtown_overture.json
    python scripts/overture_boxes.py lidar-heights --buildings data/derived/louisville_downtown_overture.json \\
        --frame outputs/downtown/buildings/d2786314-4fb3-4da6-8f81-621873d40c08 --out data/derived/louisville_downtown_overture_lidar.json
    python scripts/overture_boxes.py build --buildings data/derived/louisville_downtown_overture_lidar.json \\
        --frame outputs/downtown/buildings/d2786314-4fb3-4da6-8f81-621873d40c08 --out outputs/overture_boxes/downtown_lidar
    python scripts/overture_boxes.py split --buildings data/derived/louisville_downtown_overture_lidar.json \\
        --frame outputs/downtown/buildings/d2786314-4fb3-4da6-8f81-621873d40c08 --out data/derived/louisville_downtown_overture_split.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kyfromabove_3d.coordinates import METERS_PER_US_SURVEY_FOOT       # noqa: E402
# the measuring and splitting live in the package, for the cloud stage too
from kyfromabove_3d.product1 import (_fit_seam, _one_patch, area_cloud_local, cut_into_tiers,  # noqa: E402,F401
                                     local_polygons_about, measure_heights, min_corner_deg, roof_returns,
                                     roof_step_split, roof_tiers, split_records)

OVERTURE_RELEASE = "2026-10-21.0"
FLOOR_M = 3.5
DEFAULT_HEIGHT_M = 6.0
COLOURS = {"lidar": (185, 190, 180), "height": (190, 190, 185), "floors": (150, 165, 190), "default": (225, 150, 80)}
HIGHLIGHT = {"lidar": (235, 240, 225), "height": (235, 235, 225), "floors": (190, 205, 235), "default": (250, 190, 120)}


def fetch(bbox, out: Path) -> int:
    import duckdb

    lon0, lat0, lon1, lat1 = bbox
    src = f"s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}/theme=buildings/type=building/*"
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial; SET s3_region='us-west-2';")
    rows = con.execute(f"""
        SELECT id, names.primary, height, num_floors, roof_shape, class, subtype, ST_AsText(geometry),
               list_transform(sources, s -> s.dataset), list_transform(sources, s -> s.property)
        FROM read_parquet('{src}', filename=false, hive_partitioning=1)
        WHERE bbox.xmin BETWEEN {lon0} AND {lon1} AND bbox.ymin BETWEEN {lat0} AND {lat1}
    """).fetchall()
    recs = [{"id": r[0], "name": r[1], "height": r[2], "num_floors": r[3], "roof_shape": r[4],
             "class": r[5], "subtype": r[6], "wkt": r[7], "datasets": r[8], "props": r[9]} for r in rows]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"release": OVERTURE_RELEASE, "bbox": list(bbox), "buildings": recs}, indent=0),
                   encoding="utf-8")
    n_h = sum(1 for r in recs if r["height"] is not None)
    n_ms = sum(1 for r in recs if any(d == "Microsoft ML Buildings" and p and "height" in p
                                      for d, p in zip(r["datasets"], r["props"])))
    print(f"{len(recs)} buildings; {n_h} with a height ({n_ms} of them from Microsoft ML Buildings), "
          f"{sum(1 for r in recs if r['height'] is None and r['num_floors'])} with floors only -> {out}")
    return 0


def local_polygons(recs, frame: Path):
    """Every record's footprint in the frame building's local metres, and the frame origin."""
    coords = json.loads((frame / "coordinates.json").read_text(encoding="utf-8"))
    ox, oy, _ = coords["origin_source_xyz"]
    return local_polygons_about(recs, ox, oy), ox, oy


def area_cloud(recs, frame: Path, resolution_m=None):
    """The LiDAR over every footprint (plus 20 m), by COPC range read, in the
    frame's local metres: vegetation, noise and water left out."""
    polys, ox, oy = local_polygons(recs, frame)
    xyz, cls = area_cloud_local(polys, ox, oy, resolution_m)
    return polys, xyz, cls, ox, oy


def split(buildings: Path, frame: Path, out: Path, resolution_m=None, min_returns: int = 20,
          min_corner_deg_limit: float = 30.0) -> int:
    """Cut every footprint that :func:`roof_tiers` finds two or three
    buildings in, and write the list with the parts in its place:
    ``<gers>-a`` (the tallest), ``<gers>-b``, ``<gers>-c``, each with
    ``parent_gers`` and its own LiDAR
    height, ground and return count. A cut that would leave a corner
    sharper than ``min_corner_deg_limit`` is refused. Everything else
    passes through."""

    data = json.loads(buildings.read_text(encoding="utf-8"))
    recs = data["buildings"]
    polys, xyz, cls, ox, oy = area_cloud(recs, frame, resolution_m)
    out_recs, splits = split_records(recs, polys, xyz, ox, oy, str(frame), min_returns, min_corner_deg_limit)
    data["buildings"] = out_recs
    data["split"] = {"rule": "two or three roof heights meeting on straight lines inside the outline",
                     "n": len(splits), "buildings": splits}
    out.write_text(json.dumps(data, indent=0), encoding="utf-8")
    print(f"{len(splits)} of {len(recs)} footprints split -> {len(out_recs)} records -> {out}")
    return 0


def lidar_heights(buildings: Path, frame: Path, out: Path, resolution_m=None,
                  min_returns: int = 20) -> int:
    """Measure every footprint's height from the LiDAR: the roof is the 99th
    percentile of the non-ground returns inside the footprint, 1.5 m in from
    its edge (:func:`roof_returns`: a tall neighbour's wall spills over the
    edge) (above the
    ground by more than 2 m, so a parked car or a stoop is not a roof;
    vegetation, noise and water classes left out), the
    ground is the median of the ground-classified returns within 15 m of it,
    and the height is the difference. Written back as ``lidar_height_m`` on
    each record, with the ground, the return count and the roof's median
    height too (a tower on a podium has a median well below its top)."""

    data = json.loads(buildings.read_text(encoding="utf-8"))
    recs = data["buildings"]
    polys, xyz, cls, ox, oy = area_cloud(recs, frame, resolution_m)
    n_ok = measure_heights(recs, polys, xyz, cls, min_returns)
    data["lidar"] = {"phase": "phase2", "resolution_m": resolution_m or "full",
                     "roof": "p99 of returns > ground + 2 m, 1.5 m inside the outline (the whole of a small one)",
                     "ground": "median of class-2 returns within 15 m"}
    out.write_text(json.dumps(data, indent=0), encoding="utf-8")
    both = [(r["height"], r["lidar_height_m"]) for r in recs if r.get("height") is not None and r.get("lidar_height_m")]
    d = np.array([b - a for a, b in both])
    print(f"{n_ok} of {len(recs)} measured; {len(recs) - n_ok} had too few returns. "
          f"Against Overture's {len(both)} heights: median difference {np.median(d):+.1f} m, "
          f"half within {np.percentile(np.abs(d), 50):.1f} m, 90% within {np.percentile(np.abs(d), 90):.1f} m -> {out}")
    return 0


def height_of(r: dict, prefer_lidar: bool = True) -> tuple[float, str]:
    if prefer_lidar and r.get("lidar_height_m") is not None:
        return float(r["lidar_height_m"]), "lidar"
    if r.get("height") is not None:
        return float(r["height"]), "height"
    if r.get("num_floors"):
        return float(r["num_floors"]) * FLOOR_M, "floors"
    return DEFAULT_HEIGHT_M, "default"


def height_source(r: dict) -> str:
    for d, p in zip(r.get("datasets") or [], r.get("props") or []):
        if p and "height" in p:
            return d
    return "OpenStreetMap" if r.get("height") is not None else "none"


def extrude(ring_m: np.ndarray, z0: float, h: float):
    """Triangles of a prism: walls as quads, the top by Delaunay clipped to
    the polygon."""
    from shapely import delaunay_triangles
    from shapely.geometry import Polygon

    poly = Polygon(ring_m)
    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.geom_type != "Polygon":
            poly = max(poly.geoms, key=lambda g: g.area)
    v = np.array(poly.exterior.coords)[:-1]
    if np.sum((np.roll(v[:, 0], -1) - v[:, 0]) * (np.roll(v[:, 1], -1) + v[:, 1])) > 0:
        v = v[::-1]                                        # counter-clockwise
    P, N, I = [], [], []
    for a, b in zip(v, np.roll(v, -1, axis=0)):
        n = np.array([b[1] - a[1], -(b[0] - a[0]), 0.0]); n /= np.linalg.norm(n) + 1e-9
        k = len(P)
        P += [[a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z0 + h], [a[0], a[1], z0 + h]]
        N += [n.tolist()] * 4
        I += [k, k + 1, k + 2, k, k + 2, k + 3]
    for tri in delaunay_triangles(poly).geoms:
        if not poly.contains(tri.representative_point()):
            continue
        c = np.array(tri.exterior.coords)[:3]
        d1, d2 = c[1] - c[0], c[2] - c[0]
        if d1[0] * d2[1] - d1[1] * d2[0] < 0:
            c = c[::-1]
        k = len(P)
        P += [[x, y, z0 + h] for x, y in c]
        N += [[0.0, 0.0, 1.0]] * 3
        I += [k, k + 1, k + 2]
    return np.array(P), np.array(N), np.array(I)


def build(buildings: Path, frame: Path, out: Path, highlight: list[str], ground_z: float | None,
          heights: str = "lidar") -> int:
    from pyproj import Transformer
    from shapely.wkt import loads as wkt_loads

    from build_mesh import GlbWriter, enu_transform
    from kyfromabove_3d.features import load_wall_planes

    data = json.loads(buildings.read_text(encoding="utf-8"))
    recs = data["buildings"]
    coords = json.loads((frame / "coordinates.json").read_text(encoding="utf-8"))
    ox, oy, oz = coords["origin_source_xyz"]
    origin_z_m = oz * METERS_PER_US_SURVEY_FOOT
    if ground_z is None:
        planes = load_wall_planes(frame / "wall_planes.json")
        ground_z = min(p.z_lo for p in planes)
    to_ft = Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True).transform
    glb = GlbWriter("png")   # flat colour only: no texture encoder needed
    mats = {}
    counts = {"lidar": 0, "height": 0, "floors": 0, "default": 0}
    by_source = {}
    extras_list = []
    for r in recs:
        geom = wkt_loads(r["wkt"])
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)
        x, y = to_ft(*np.array(geom.exterior.coords).T)
        ring = np.column_stack([(np.asarray(x) - ox) * METERS_PER_US_SURVEY_FOOT,
                                (np.asarray(y) - oy) * METERS_PER_US_SURVEY_FOOT])
        h, kind = height_of(r, prefer_lidar=(heights == "lidar"))
        counts[kind] += 1
        # lidar_ground_z is an absolute NAVD88 elevation; the frame's z is
        # measured from its origin (455 ft up, downtown). Forgetting the
        # origin floated every box 140 m into the air.
        z0 = (ground_z if (heights != "lidar" or r.get("lidar_ground_z") is None)
              else r["lidar_ground_z"] - origin_z_m)
        src = height_source(r)
        by_source[src] = by_source.get(src, 0) + 1
        hl = r["id"] in highlight
        key = (kind, hl)
        if key not in mats:
            c = (HIGHLIGHT if hl else COLOURS)[kind]
            mats[key] = glb.material_flat(c, f"{kind}{'_highlight' if hl else ''}", unlit=False)
        P, N, I = extrude(ring, z0, h)
        glb.primitive(P, N, None, I, mats[key])
        extras_list.append({"id": r["id"], "name": r["name"], "height_m": h, "height_from": kind,
                            "height_source": src, "overture_height_m": r.get("height"),
                            "lidar_height_m": r.get("lidar_height_m")})
    out.mkdir(parents=True, exist_ok=True)
    glb.write(out / "boxes.glb", {"name": "Overture buildings", "release": data["release"],
                                  "buildings": extras_list})
    transform, origin_geo = enu_transform(coords)
    # bounding sphere from the position extents the writer recorded
    mins = np.array([a["min"] for a in glb.accessors if "min" in a]); maxs = np.array([a["max"] for a in glb.accessors if "max" in a])
    lo, hi = mins.min(axis=0), maxs.max(axis=0)                   # glTF y-up
    centre = 0.5 * (lo + hi); centre = np.array([centre[0], -centre[2], centre[1]])   # back to z-up
    radius = float(np.linalg.norm(hi - lo)) / 2
    tileset = {
        "asset": {"version": "1.1", "generator": "kyfromabove-3d overture_boxes.py"},
        "geometricError": 500.0,
        "root": {"transform": transform, "boundingVolume": {"sphere": centre.tolist() + [radius]},
                 "geometricError": 0.0, "refine": "REPLACE", "content": {"uri": "boxes.glb"}},
        "extras": {"name": f"Overture {data['release']} buildings" + (", LiDAR heights" if heights == "lidar" else ""),
                   "kind": "boxes", "heights_from": heights,
                   "buildings": len(recs), "heights": counts, "height_sources": by_source,
                   "origin_lon_lat_h": origin_geo},
    }
    (out / "tileset.json").write_text(json.dumps(tileset, indent=1), encoding="utf-8")
    print(f"{len(recs)} boxes: {counts['lidar']} at the LiDAR's height, {counts['height']} at Overture's ({by_source}), "
          f"{counts['floors']} from floors x {FLOOR_M} m, {counts['default']} at the {DEFAULT_HEIGHT_M} m default -> {out / 'tileset.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("LON0", "LAT0", "LON1", "LAT1"))
    f.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("build"); b.add_argument("--buildings", type=Path, required=True)
    b.add_argument("--frame", type=Path, required=True, help="a project building dir whose local frame to use")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--highlight", nargs="*", default=[], help="GERS ids to draw brighter")
    b.add_argument("--ground-z", type=float, default=None, help="ground in the frame (default: the frame building's)")
    b.add_argument("--heights", choices=["lidar", "overture"], default="lidar",
                   help="lidar: the measured height where there is one, Overture's otherwise; overture: Overture's only")
    l = sub.add_parser("lidar-heights"); l.add_argument("--buildings", type=Path, required=True)
    l.add_argument("--frame", type=Path, required=True); l.add_argument("--out", type=Path, required=True)
    l.add_argument("--resolution-m", type=float, default=None, help="cap the COPC read (default: full resolution)")
    sp = sub.add_parser("split", help="cut the footprints that hold two buildings side by side at their roof step")
    sp.add_argument("--buildings", type=Path, required=True, help="the lidar-heights output")
    sp.add_argument("--frame", type=Path, required=True); sp.add_argument("--out", type=Path, required=True)
    sp.add_argument("--resolution-m", type=float, default=None, help="cap the COPC read (default: full resolution)")
    args = ap.parse_args()
    if args.cmd == "fetch":
        return fetch(tuple(args.bbox), args.out)
    if args.cmd == "lidar-heights":
        return lidar_heights(args.buildings, args.frame, args.out, args.resolution_m)
    if args.cmd == "split":
        return split(args.buildings, args.frame, args.out, args.resolution_m)
    return build(args.buildings, args.frame, args.out, args.highlight, args.ground_z, args.heights)


if __name__ == "__main__":
    sys.exit(main())
