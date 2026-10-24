"""The map file of the published tiles (tile_index): one row a tile, Parquet's
own GEOMETRY type, readable by DuckDB. Needs DuckDB's spatial extension
(installed once with ``INSTALL spatial``); skipped without it."""

from __future__ import annotations

import json

import pytest

duckdb = pytest.importorskip("duckdb")

from kyfromabove_3d import tile_index as TI                                  # noqa: E402
from kyfromabove_3d.store import LocalStore                                   # noqa: E402


def _spatial():
    con = duckdb.connect()
    try:
        con.execute("LOAD spatial")
    except duckdb.Error:
        pytest.skip("DuckDB's spatial extension is not installed")
    return con


def _grid():
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, box

    # two squares about 1.5 km across, in web mercator like the Phase 2 grid,
    # each a one-part multipolygon as the grid stores them
    return gpd.GeoDataFrame({"tilename": ["N077E228", "N015E299"]},
                            geometry=[MultiPolygon([box(-9547000, 4508000, -9545500, 4509500)]),
                                      MultiPolygon([box(-9407500, 4734000, -9406000, 4735500)])],
                            crs="EPSG:3857")


SOURCES = {"lidar_phase": "phase2", "lidar_date": "2022-12-12", "lidar_points_per_m2": 3.1,
           "oblique_collection": "2024_Season1", "oblique_date": "2024-03-20", "gap_months": 15.2}


def _store(tmp_path):
    return _fill(LocalStore(tmp_path / "store"))


def _fill(st):
    st.put_json("tiles/N077E228/buildings.json",
                {"tile": "N077E228", "code_version": "stage1-2", "buildings": [dict(SOURCES, id=i) for i in "abc"]})
    st.put_json("tiles/N077E228/summary.json", {"written_at": "2026-10-22T08:58:43+00:00"})
    st.put_json("tiles/N077E228/3dtiles/measured/tileset.json",
                {"extras": {"meshed": 2, "boxed": 1, "area_bytes": 1_500_000}})
    st.put_json("tiles/N077E228/3dtiles/boxes/tileset.json", {"extras": {"meshed": 0, "boxed": 3}})
    st.put_json("tiles/N015E299/buildings.json", {"tile": "N015E299", "buildings": [dict(SOURCES, id="d")]})
    st.put_json("tiles/N999E999/buildings.json", {"tile": "N999E999", "buildings": []})   # not in the grid
    return st


def test_the_index_is_geoparquet_2_with_a_row_per_tile_and_finds_a_tile_by_a_point(tmp_path):
    con = _spatial()
    st = _store(tmp_path)
    warnings = []
    s = TI.publish_tile_index(st, _grid(), ["boxes", "measured"], tmp_path / "work", log=warnings.append)
    assert s["tiles"] == 2 and s["buildings"] == 4
    assert any("N999E999" in w for w in warnings)

    p = tmp_path / "store" / TI.INDEX_KEY
    logical = dict(con.sql(f"SELECT name, logical_type FROM parquet_schema('{p}')").fetchall())
    assert logical["geometry"].startswith("GeometryType")                 # Parquet's own type, not a blob
    geo = json.loads(con.sql(f"SELECT decode(value) FROM parquet_kv_metadata('{p}') "
                             "WHERE decode(key) = 'geo'").fetchone()[0])
    assert geo["version"] == "2.0.0" and geo["columns"]["geometry"]["geometry_types"] == ["Polygon"]

    rows = {r[0]: r[1:] for r in con.sql(
        f"SELECT tile, buildings, lidar_date::VARCHAR, gap_months, tile_step_at::VARCHAR, measured_meshed, "
        f"measured_boxed, measured_mb, boxes_boxed, code_version FROM '{p}' ORDER BY tile").fetchall()}
    assert rows["N077E228"] == (3, "2022-12-12", 15.2, "2026-10-22 08:58:43", 2, 1, 1.5, 3, "stage1-2")
    assert rows["N015E299"] == (1, "2022-12-12", 15.2, None, None, None, None, None, None)

    # the squares come back in longitude and latitude (in Kentucky), and a
    # point inside one finds that one
    lon, lat = con.sql(f"SELECT ST_X(ST_Centroid(geometry)), ST_Y(ST_Centroid(geometry)) FROM '{p}' "
                       "WHERE tile = 'N077E228'").fetchone()
    assert -89.6 < lon < -81.9 and 36.4 < lat < 39.2
    hit = con.sql(f"SELECT tile FROM '{p}' WHERE ST_Intersects(geometry, ST_Point({lon}, {lat}))").fetchall()
    assert hit == [("N077E228",)]


def test_the_cloud_tree_job_writes_the_index_to_the_bucket(tmp_path, monkeypatch):
    """The index job (cloud_jobs.tree_job) on a fake bucket: after the trees
    it writes ``kentucky/tiles.parquet`` over every tile in the store."""
    _spatial()
    moto = pytest.importorskip("moto")
    import boto3

    from kyfromabove_3d import cloud_jobs, pieces, publish
    from kyfromabove_3d.store import S3Store

    monkeypatch.setattr(pieces, "phase2_grid", lambda raw: _grid())
    # the trees and the terrain have their own tests: here only what they hand on
    monkeypatch.setattr(publish, "publish_trees",
                        lambda store, grid, products: {"measured": {"tile_names": ["N077E228"]}})
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    monkeypatch.setenv("STORE", "s3://kfa3d/runs/test")
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-west-2")
        s3.create_bucket(Bucket="kfa3d", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
        _fill(S3Store("kfa3d", "runs/test", client=s3))
        res = cloud_jobs.run_job({"kind": "tree", "run": "r1", "terrain": False})
        assert res["tile_index"]["tiles"] == 2 and res["tile_index"]["buildings"] == 4
        body = s3.get_object(Bucket="kfa3d", Key="runs/test/" + TI.INDEX_KEY)["Body"].read()
    p = tmp_path / "tiles.parquet"
    p.write_bytes(body)
    con = _spatial()
    assert con.sql(f"SELECT list(tile ORDER BY tile) FROM '{p}'").fetchone()[0] == ["N015E299", "N077E228"]
    assert list(tmp_path.iterdir()) == [p]                              # the job's working folder is gone
