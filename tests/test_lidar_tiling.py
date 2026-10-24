"""Tile resolution and multi-tile AOI handling."""

import numpy as np
import pytest

from kyfromabove_3d.lidar import LidarCloud, LidarTileIndex, concat_clouds


def _cloud(n, x0=0.0):
    xyz = np.column_stack([np.full(n, x0), np.arange(n, dtype=float), np.zeros(n)])
    z = np.zeros(n)
    return LidarCloud(xyz, xyz.copy(), z, z, z, z)


def test_concat_preserves_every_point():
    """An AOI split across tiles must come back whole.

    Taking only the first covering tile returned a truncated cloud that looks
    downstream like a sparse building rather than an error.
    """
    merged = concat_clouds([_cloud(10, 0.0), _cloud(7, 100.0), _cloud(3, 200.0)])
    assert len(merged) == 20
    assert merged.xyz_source_ft.shape == (20, 3)
    assert merged.intensity.shape == (20,)
    assert set(np.unique(merged.xyz_source_ft[:, 0])) == {0.0, 100.0, 200.0}


def test_concat_handles_empty_and_single():
    assert len(concat_clouds([])) == 0
    assert len(concat_clouds([_cloud(0), _cloud(5)])) == 5
    single = _cloud(4)
    assert concat_clouds([single]) is single


def test_copc_detection_accepts_both_spellings():
    """KyFromAbove writes COPC as .copz.laz in some index rows, .copc.laz in others."""
    for key, expected in [
        ("elevation/PointCloud/Phase2/N067E390_LAS_Phase2.copz.laz", True),
        ("elevation/PointCloud/Phase3/N016E301_LAS_Phase3.copc.laz", True),
        ("elevation/PointCloud/Phase1/N067E390.laz", False),
    ]:
        assert (".copc." in key or ".copz." in key) is expected


def test_tile_index_reprojects_from_source_crs(tmp_path):
    """The grids are EPSG:3857 and Phase 2 ships with no CRS at all.

    Querying them with EPSG:3089 coordinates directly returns no tiles silently,
    so the index must reproject rather than compare raw numbers.
    """
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box
    from shapely.ops import transform
    from pyproj import Transformer

    # A 5k-style tile around the Grayson AOI, written in EPSG:3857 with no CRS.
    aoi_3089 = (5725000.0, 4025400.0, 5725350.0, 4025800.0)
    to3857 = Transformer.from_crs("EPSG:3089", "EPSG:3857", always_xy=True).transform
    tile_geom = transform(to3857, box(*aoi_3089)).buffer(500)
    gdf = gpd.GeoDataFrame(
        {"tilename": ["T0001"], "key": ["elevation/PointCloud/Phase2/T0001.copz.laz"],
         "aws_url": ["https://example.invalid/T0001.copz.laz"]},
        geometry=[tile_geom], crs=None,
    )
    raw = tmp_path
    # Writing without a CRS is the point of this fixture: it reproduces the
    # Phase 2 grid exactly. pyogrio warns, correctly, so silence it here only.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        gdf.to_file(raw / "kyfromabove_phase2_pointcloud_5k_grid.gpkg", driver="GPKG")

    index = LidarTileIndex(raw, phases=("phase2",))
    assert index.phases == ["phase2"]
    tiles = index.tiles_for_bounds(aoi_3089, source_crs="EPSG:3089")
    assert len(tiles) == 1 and tiles[0].tilename == "T0001"
    assert tiles[0].is_copc

    # A far-away AOI must resolve to nothing rather than to this tile.
    assert index.tiles_for_bounds((4930000.0, 3300000.0, 4930350.0, 3300400.0)) == []


def test_open_copc_falls_back_to_the_other_spelling(monkeypatch):
    """~90 Phase 2 tiles are published under a filename that does not exist.

    All of them exist under the other COPC spelling, so a hard failure here is
    a lost tile rather than a real absence.
    """
    import laspy

    from kyfromabove_3d import lidar as lidar_mod

    tried = []

    def fake_open(url):
        tried.append(url)
        if ".copz." in url:
            raise OSError("404 Client Error: Not Found")
        return f"reader:{url}"

    monkeypatch.setattr(laspy.CopcReader, "open", staticmethod(fake_open))
    got = lidar_mod.open_copc("https://x/N072E392_LAS_Phase2.copz.laz")
    assert got.endswith(".copc.laz")
    assert len(tried) == 2, "the listed spelling is tried first"

    # A url that works is opened once; no speculative second request.
    tried.clear()
    lidar_mod.open_copc("https://x/N067E390_LAS_Phase2.copc.laz")
    assert len(tried) == 1

    # Both spellings failing is a real absence and must raise, not return None.
    def always_fail(url):
        raise OSError("404")

    monkeypatch.setattr(laspy.CopcReader, "open", staticmethod(always_fail))
    with pytest.raises(FileNotFoundError, match="no COPC readable"):
        lidar_mod.open_copc("https://x/N000E000_LAS_Phase2.copz.laz")
