#!/usr/bin/env python3
"""Rebuild the complexity survey's building list from Overture.

The list is Kentucky buildings above an area floor, each tagged with the
KyFromAbove 5k tile that covers it, so :mod:`survey_building_complexity` can
batch its COPC reads by tile.

**The original list did not mean what it said.** Its ``area_m2`` column was the
axis-aligned *bounding box*, not the footprint, so "12,363 buildings over
5,000 m^2" was really 12,363 buildings whose bounding box cleared 5,000 -- a
median true footprint of **2,180 m^2**, with some under 700. Measured on the
polygon, only **4,708** clear 5,000 m^2. Both areas are now written to every
row, and ``--area-basis`` selects which one the floor applies to.

It was originally built ad hoc and lived only in a scratch directory, which
meant every survey number in the repository rested on a file the repository did
not contain. This regenerates it.

**The Overture release is part of the answer, not an implementation detail.**
Footprints are revised between releases and the footprint clip is the dominant
error term in the survey, so the release is pinned here and matches the one
``measure_case_complexity`` uses. A different release is a different population.

Usage::

    python scripts/fetch_overture_buildings.py --out data/derived/ky_large_tiled_wkt.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

#: Same release as scripts/measure_case_complexity.py. Keep them together.
OVERTURE_RELEASE = "2026-10-21.0"
OVERTURE_BUILDINGS = (
    f"s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}"
    "/theme=buildings/type=building/*"
)
#: Kentucky, generously bounded. Trimmed to the LiDAR grid by the tile join
#: below, which is the real state boundary as far as this project is concerned.
KY_BBOX = (-89.6, 36.4, -81.9, 39.2)


def report_bands() -> int:
    """Overture attribute coverage by **true footprint** area band.

    The original version measured bounding boxes. Re-run on real footprints
    (Overture 2026-08-19.0, Kentucky bbox):

    =============  ===========  ==========  ========  =======
    band (m2)        buildings  roof_shape    height    class
    =============  ===========  ==========  ========  =======
    <200             3,676,650        0.1%     65.7%    10.0%
    200-1k           1,643,934        0.1%     85.1%     9.9%
    1k-2k               41,075        0.4%     82.6%    12.5%
    2k-5k               19,931        0.4%     80.8%    17.0%
    **5k-10k**           5,209        0.5%     78.9%  **31.2%**
    10k-20k              2,290        0.3%     77.1%    36.2%
    >=20k                1,122        0.4%     53.5%    38.9%
    =============  ===========  ==========  ========  =======

    Two corrections fall out. **``roof_shape`` is useless in Kentucky** -- under
    0.5% at every size, never the 22% an earlier note attributed to the large
    bands. That note had read the ``class`` column: class coverage is what
    roughly doubles at 5,000 m^2, 17.0% to 31.2%, and it is the only attribute
    that responds to size at all. So the floor stands, for a different reason
    than it was given.

    **``height`` is worst where it would matter most**, falling to 53.5% above
    20,000 m^2 against 80% in the middle bands. Another reason the facade term
    takes its height from the LiDAR's roof plate.

    Area is computed in DuckDB here rather than pyproj, because this aggregates
    over every building in the bbox and never pulls a geometry back. ``always_xy``
    is passed explicitly: DuckDB honours the authority axis order, so EPSG:4326
    is read as (lat, lon) without it, and it returns coordinates two orders of
    magnitude wrong rather than raising.
    """
    import duckdb

    lon0, lat0, lon1, lat1 = KY_BBOX
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;"
                " SET s3_region='us-west-2';")
    ft2 = 0.3048006096012192 ** 2
    print(f"Overture {OVERTURE_RELEASE}, Kentucky bbox (takes in some of the "
          f"neighbouring states)\ncoverage by TRUE footprint area:\n")
    rows = con.execute(f"""
        WITH b AS (
          SELECT ST_Area(ST_Transform(geometry, 'EPSG:4326', 'EPSG:3089', true))
                 * {ft2} AS area_m2,
                 roof_shape, height, class
          FROM read_parquet('{OVERTURE_BUILDINGS}', filename=false, hive_partitioning=1)
          WHERE bbox.xmin BETWEEN {lon0} AND {lon1}
            AND bbox.ymin BETWEEN {lat0} AND {lat1}
        )
        SELECT CASE WHEN area_m2 < 200 THEN 'a. <200'
                    WHEN area_m2 < 1000 THEN 'b. 200-1k'
                    WHEN area_m2 < 2000 THEN 'c. 1k-2k'
                    WHEN area_m2 < 5000 THEN 'd. 2k-5k'
                    WHEN area_m2 < 10000 THEN 'e. 5k-10k'
                    WHEN area_m2 < 20000 THEN 'f. 10k-20k'
                    ELSE 'g. >=20k' END AS band,
               count(*) AS n,
               avg(CASE WHEN roof_shape IS NOT NULL THEN 1.0 ELSE 0.0 END) AS roof,
               avg(CASE WHEN height IS NOT NULL THEN 1.0 ELSE 0.0 END) AS h,
               avg(CASE WHEN class IS NOT NULL THEN 1.0 ELSE 0.0 END) AS cls
        FROM b GROUP BY band ORDER BY band
    """).fetchall()
    print(f"{'band (m2)':<14}{'buildings':>12}{'roof_shape':>12}{'height':>9}{'class':>8}")
    for band, n, roof, h, cls in rows:
        print(f"{band:<14}{n:>12,}{roof:>12.1%}{h:>9.1%}{cls:>8.1%}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=Path("data/derived/ky_large_tiled_wkt.csv"))
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"),
                    help="holds the KyFromAbove 5k grid GeoPackages")
    ap.add_argument("--min-area-m2", type=float, default=5000.0,
                    help="area floor, applied to --area-basis. 5,000 m2 is where "
                         "Overture's class coverage roughly doubles, 17%% to 31%% "
                         "-- see --bands. NOT roof_shape, which an earlier note "
                         "claimed and which is under 0.5%% at every size in "
                         "Kentucky")
    ap.add_argument("--area-basis", choices=["polygon", "bbox"], default="polygon",
                    help="'polygon' is the building's actual footprint. 'bbox' is "
                         "its axis-aligned bounding box, which is what the "
                         "original 12,363-building list used -- it admits a "
                         "diagonal or L-shaped building at well under the stated "
                         "floor, and its median true footprint is 2,180 m2, not "
                         "5,000. Use it only to reproduce that population")
    ap.add_argument("--select", choices=["area", "facade"], default="area",
                    help="'area' lists buildings whose ROOF may be worth "
                         "densifying, filtered on footprint. 'facade' lists "
                         "buildings whose WALLS may be worth imaging, filtered "
                         "on wall area against roof area. The two select in "
                         "opposite directions and share almost nothing: a "
                         "5,000 m2 footprint is wide and low by construction, "
                         "and has a facade-to-roof ratio around 0.4")
    ap.add_argument("--min-height-m", type=float, default=12.0,
                    help="facade mode: floor on height, about four storeys. Also "
                         "what makes the query tractable -- without it every "
                         "row house in Kentucky is a candidate")
    ap.add_argument("--min-facade-ratio", type=float, default=2.0,
                    help="facade mode: wall area / roof area. 2.0 means the wall "
                         "is the dominant surface, the same cut the survey "
                         "summary uses. The pilot is 0.5; PNC Tower is 16")
    ap.add_argument("--min-facade-m2", type=float, default=500.0,
                    help="facade mode: absolute floor, so a small tall object "
                         "with a high ratio is not selected on ratio alone")
    ap.add_argument("--floor-height-m", type=float, default=3.5,
                    help="facade mode: metres per storey, used only where "
                         "Overture gives num_floors and no height. Deliberately "
                         "under the 3.9-4.8 m the two Louisville towers actually "
                         "have, so the estimate under-selects rather than over-")
    ap.add_argument("--bands", action="store_true",
                    help="print Overture attribute coverage by true footprint "
                         "band and exit, without writing a list. The 5,000 m2 "
                         "floor was justified by a jump in roof_shape coverage, "
                         "and that analysis was run on bounding boxes")
    args = ap.parse_args()

    if args.bands:
        return report_bands()

    import duckdb
    import geopandas as gpd
    import numpy as np
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform
    from shapely.wkt import loads as wkt_loads

    from kyfromabove_3d.coordinates import METERS_PER_US_SURVEY_FOOT
    from kyfromabove_3d.lidar import LidarTileIndex

    lon0, lat0, lon1, lat1 = KY_BBOX
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;"
                " SET s3_region='us-west-2';")

    # A bounding box is never smaller than the polygon inside it, so a cheap
    # test on the bbox's own extent in degrees drops the great majority of
    # Kentucky's ~5M buildings before anything is reprojected. Converted at the
    # southern edge of the state, where a degree of longitude is longest and the
    # threshold therefore smallest, then halved -- this filter must never
    # exclude a building the exact test would keep.
    deg2_floor = args.min_area_m2 / (89_600.0 * 110_900.0) / 2.0
    if args.select == "area":
        # A bounding box is never smaller than the polygon inside it, so this
        # cheap degree-space test drops most of Kentucky's ~5M buildings before
        # anything is reprojected.
        where = (f"(bbox.xmax - bbox.xmin) * (bbox.ymax - bbox.ymin) "
                 f">= {deg2_floor}")
    else:
        # Height cannot be pre-filtered geometrically, so the height floor is
        # the selective clause. Buildings with neither height nor num_floors are
        # invisible here; how many is reported below, because it is the main
        # limitation of this list.
        where = (f"(height >= {args.min_height_m} OR "
                 f"num_floors >= {args.min_height_m / args.floor_height_m})")
    print(f"querying Overture {OVERTURE_RELEASE}, select={args.select} ...")
    rows = con.execute(f"""
        SELECT id, height, class,
               bbox.xmin AS xmin, bbox.xmax AS xmax,
               bbox.ymin AS ymin, bbox.ymax AS ymax,
               ST_AsText(geometry) AS wkt, num_floors
        FROM read_parquet('{OVERTURE_BUILDINGS}', filename=false, hive_partitioning=1)
        WHERE bbox.xmin BETWEEN {lon0} AND {lon1}
          AND bbox.ymin BETWEEN {lat0} AND {lat1}
          AND {where}
    """).fetchall()
    print(f"  {len(rows):,} candidates")

    # Reproject and take the exact area in Python, with pyproj, the way
    # survey_building_complexity and measure_case_complexity do. Not in DuckDB:
    # its ST_Transform honours the authority's axis order, so EPSG:4326 is read
    # as (lat, lon) unless always_xy is passed, and it returns coordinates two
    # orders of magnitude wrong rather than raising. Doing it here keeps one
    # transform path across the project instead of two that can disagree.
    to_ft = Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True).transform
    kept, polys, extra = [], [], []
    for r in rows:
        try:
            poly = shp_transform(to_ft, wkt_loads(r[7]))
        except Exception:                                   # noqa: BLE001
            continue
        area = poly.area * METERS_PER_US_SURVEY_FOOT ** 2
        x0, y0, x1, y1 = poly.bounds
        bbox_area = (x1 - x0) * (y1 - y0) * METERS_PER_US_SURVEY_FOOT ** 2

        if args.select == "area":
            basis = area if args.area_basis == "polygon" else bbox_area
            if basis < args.min_area_m2:
                continue
            extra.append({})
        else:
            height = r[1]
            if height is None and r[8] is not None:
                height = float(r[8]) * args.floor_height_m
            if height is None:                      # unreachable via the SQL
                continue
            perim = poly.length * METERS_PER_US_SURVEY_FOOT
            facade = perim * float(height)
            ratio = facade / area if area > 0 else 0.0
            if (float(height) < args.min_height_m
                    or facade < args.min_facade_m2
                    or ratio < args.min_facade_ratio):
                continue
            extra.append({"perimeter_m": perim, "est_height_m": float(height),
                          "height_source": "overture" if r[1] is not None else "floors",
                          "est_facade_m2": facade, "est_facade_to_roof": ratio})
        kept.append((r, area, bbox_area))
        polys.append(poly)
    if args.select == "area":
        print(f"  {len(kept):,} buildings with {args.area_basis} area over "
              f"{args.min_area_m2:,.0f} m2")
    else:
        print(f"  {len(kept):,} buildings with wall area >= "
              f"{args.min_facade_ratio:g}x roof and >= {args.min_facade_m2:,.0f} m2")
        # Counted separately: the SQL already requires a height, so buildings
        # with none never reach the loop above. They are the main gap in this
        # list and cannot be recovered from Overture's geometry the way a
        # footprint area can.
        blind = con.execute(f"""
            SELECT count(*) FROM read_parquet('{OVERTURE_BUILDINGS}',
                                              filename=false, hive_partitioning=1)
            WHERE bbox.xmin BETWEEN {lon0} AND {lon1}
              AND bbox.ymin BETWEEN {lat0} AND {lat1}
              AND height IS NULL AND num_floors IS NULL
        """).fetchone()[0]
        print(f"  {blind:,} buildings in the bbox have neither height nor "
              f"num_floors and are invisible to this list")

    # Tag each with its 5k tile. A spatial join against the grid, not a
    # per-building lookup: the grid is 46,167 rows and this is tens of thousands
    # of points.
    index = LidarTileIndex(args.raw_dir)
    grid = index.grids.get("phase2")
    if grid is None:
        print(f"[FAIL] no phase2 grid under {args.raw_dir}")
        return 1
    pts = gpd.GeoDataFrame(
        {"i": np.arange(len(polys))},
        geometry=[p.centroid for p in polys],
        crs="EPSG:3089",
    ).to_crs(index.GRID_CRS)
    joined = gpd.sjoin(pts, grid[["tilename", "geometry"]], how="inner",
                       predicate="within")
    tile_for = dict(zip(joined["i"], joined["tilename"].astype(str)))
    print(f"  {len(tile_for):,} fall inside a Phase 2 tile "
          f"({len(kept) - len(tile_for):,} outside Kentucky's grid, dropped)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Both areas are written. The original list carried one column named
    # area_m2 that was in fact the bounding box, and every document describing
    # the survey said "footprint over 5,000 m2" on the strength of it. Naming
    # both makes that impossible to repeat.
    field = ["id", "area_m2", "bbox_area_m2", "area_basis", "height", "class",
             "tilename", "xmin", "xmax", "ymin", "ymax", "wkt"]
    if args.select == "facade":
        field += ["perimeter_m", "est_height_m", "height_source",
                  "est_facade_m2", "est_facade_to_roof"]
    n = 0
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=field)
        w.writeheader()
        for i, (r, area, bbox_area) in enumerate(kept):
            tile = tile_for.get(i)
            if tile is None:
                continue
            w.writerow({
                "id": r[0], "area_m2": area, "bbox_area_m2": bbox_area,
                "area_basis": args.area_basis,
                "height": "" if r[1] is None else r[1],
                "class": r[2] or "", "tilename": tile,
                "xmin": r[3], "xmax": r[4], "ymin": r[5], "ymax": r[6],
                "wkt": r[7], **extra[i],
            })
            n += 1
    tiles = len(set(tile_for.values()))
    print(f"\n{n:,} buildings over {tiles:,} tiles -> {args.out}")
    print(f"  area_basis={args.area_basis}; area_m2 is the footprint polygon and "
          f"bbox_area_m2 its bounding box, always both")
    if args.area_basis == "bbox":
        print("  reproduces the original 12,363-building population, whose median "
              "true footprint is 2,180 m2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
