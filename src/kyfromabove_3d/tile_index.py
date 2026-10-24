"""One map file of the published tiles: ``kentucky/tiles.parquet``, a row per
LiDAR tile with its square, its buildings, the LiDAR and photos it was built
from, and what each product holds there. It is GeoParquet with Parquet's own
GEOMETRY type (GeoParquet 2.0), so DuckDB, GDAL or anything that reads Parquet
can answer "which tiles ..." without walking the tile trees::

    INSTALL spatial; LOAD spatial;
    SELECT tile, gap_months FROM 'tiles.parquet' WHERE gap_months > 6;
    SELECT tile FROM 'tiles.parquet'
     WHERE ST_Intersects(geometry, ST_Point(-84.5085, 39.0866));

Longitude and latitude (OGC:CRS84); times in UTC. Every column but ``tile`` and
``geometry`` can be null: a tile published before a field existed, or a
product not published on it. Written by DuckDB, which the laptop has (the
``overture`` extra); the cloud image does not yet.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

#: the tile's sources, the same for every building on it (``pieces.PROVENANCE``)
SOURCES = (("lidar_phase", "VARCHAR"), ("lidar_date", "DATE"), ("lidar_points_per_m2", "DOUBLE"),
           ("oblique_collection", "VARCHAR"), ("oblique_date", "DATE"), ("gap_months", "DOUBLE"))
#: what each product's tileset says of the tile (``publish.publish_tile``'s summary)
PER_PRODUCT = (("meshed", "INTEGER"), ("boxed", "INTEGER"), ("mb", "DOUBLE"))
INDEX_KEY = "kentucky/tiles.parquet"


def _common(recs: Sequence[Dict[str, Any]], field: str):
    """The value most of the tile's buildings carry (one, unless a tile was
    measured in parts by different code)."""
    seen = Counter(r.get(field) for r in recs if r.get(field) is not None)
    return seen.most_common(1)[0][0] if seen else None


def tile_rows(store, grid, products: Sequence[str], tiles: Optional[Sequence[str]] = None,
              tiles_prefix: str = "tiles", log: Callable = print) -> List[Dict[str, Any]]:
    """A row per tile that has a ``buildings.json``: read from ``store`` 32
    at a time (the reads wait on the network). ``grid`` is
    ``pieces.phase2_grid``; a tile not in it gets no geometry and is left out,
    with a warning."""
    from concurrent.futures import ThreadPoolExecutor

    from pyproj import Transformer
    from shapely.ops import transform

    if tiles is None:
        tiles = store.list_dirs(f"{tiles_prefix}/")
    squares = dict(zip(grid["tilename"], grid.geometry))
    to_lonlat = Transformer.from_crs(grid.crs, "OGC:CRS84", always_xy=True).transform

    def read(t):
        try:
            b = store.read_json(f"{tiles_prefix}/{t}/buildings.json")
        except FileNotFoundError:
            return None
        recs = b.get("buildings", [])
        row: Dict[str, Any] = {"tile": t, "buildings": len(recs), "code_version": b.get("code_version")}
        for name, _ in SOURCES:
            row[name] = _common(recs, name)
        try:
            row["tile_step_at"] = store.read_json(f"{tiles_prefix}/{t}/summary.json").get("written_at")
        except FileNotFoundError:
            row["tile_step_at"] = None
        for p in products:
            try:
                ex = store.read_json(f"{tiles_prefix}/{t}/3dtiles/{p}/tileset.json").get("extras", {})
            except FileNotFoundError:
                ex = None
            row[f"{p}_meshed"] = ex.get("meshed") if ex else None
            row[f"{p}_boxed"] = ex.get("boxed") if ex else None
            row[f"{p}_mb"] = round(ex["area_bytes"] / 1e6, 3) if ex and "area_bytes" in ex else None
        return row

    rows = []
    with ThreadPoolExecutor(32) as ex:
        for row in ex.map(read, sorted(tiles)):
            if row is None:
                continue
            square = squares.get(row["tile"])
            if square is None:
                log(f"[warn] {row['tile']}: not in the Phase 2 grid, left out of the index")
                continue
            if square.geom_type == "MultiPolygon" and len(square.geoms) == 1:
                square = square.geoms[0]                                # the grid stores each square as a one-part multipolygon
            row["geometry"] = transform(to_lonlat, square).wkb
            rows.append(row)
    return rows


def write_tile_index(rows: Sequence[Dict[str, Any]], path: Path, products: Sequence[str]) -> Path:
    """``rows`` as GeoParquet 2.0 at ``path``, sorted by tile name, the
    geometry stored as Parquet's GEOMETRY type with a bounding box per row
    group."""
    import duckdb

    cols = ([("tile", "VARCHAR"), ("buildings", "INTEGER")] + list(SOURCES)
            + [("code_version", "VARCHAR"), ("tile_step_at", "TIMESTAMP")]
            + [(f"{p}_{k}", t) for p in products for k, t in PER_PRODUCT])
    con = duckdb.connect()
    con.execute("LOAD spatial")
    con.execute("CREATE TABLE tiles (" + ", ".join(f'"{c}" {t}' for c, t in cols) + ", wkb BLOB)")
    con.executemany(f"INSERT INTO tiles VALUES ({', '.join('?' * (len(cols) + 1))})",
                    [[r.get(c) for c, _ in cols] + [r["geometry"]] for r in rows])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""COPY (SELECT * EXCLUDE (wkb), ST_GeomFromWKB(wkb) AS geometry FROM tiles ORDER BY tile)
                    TO '{path}' (FORMAT parquet, GEOPARQUET_VERSION 'V2', COMPRESSION zstd)""")
    return path


def publish_tile_index(store, grid, products: Sequence[str], work: Path, out_store=None,
                       tiles: Optional[Sequence[str]] = None, log: Callable = print) -> Dict[str, Any]:
    """The index over every tile in ``store``, written to ``out_store``
    (default ``store``) at :data:`INDEX_KEY`. Returns its row and building
    counts."""
    rows = tile_rows(store, grid, products, tiles=tiles, log=log)
    path = write_tile_index(rows, Path(work) / "tiles.parquet", products)
    (out_store or store).put_file(INDEX_KEY, path)
    summary = {"tiles": len(rows), "buildings": sum(r["buildings"] for r in rows), "bytes": path.stat().st_size}
    log(f"tile index: {summary['tiles']} tiles, {summary['buildings']:,} buildings -> {INDEX_KEY}")
    return summary
