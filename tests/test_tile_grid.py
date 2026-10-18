"""LiDAR tile lists with broken shapes: the index refuses one, and the repair
gives each tile its shape back by name (KyFromAbove's Phase 3 list of
2026-09-15 stores every shape at infinity)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import Polygon                                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import repair_tile_grid as R                                                # noqa: E402

from kyfromabove_3d.lidar import LidarTileIndex                            # noqa: E402

INF = Polygon([(np.inf, np.inf)] * 4)


def _phase2(names):
    """A reference list of full tiles placed by the name rule."""
    a0, b0 = 3775000.0, 4365000.0
    return gpd.GeoDataFrame({"tilename": names, "key": [f"{n}.copc.laz" for n in names],
                             "aws_url": [f"https://x/{n}.copc.laz" for n in names]},
                            geometry=[R.shape_from_name(n, a0, b0) for n in names], crs="EPSG:3857")


def test_the_index_refuses_a_list_whose_shapes_are_not_finite(tmp_path):
    g = _phase2(["N078E248", "N078E249"])
    g.to_file(tmp_path / "kyfromabove_phase2_pointcloud_5k_grid.gpkg", driver="GPKG")
    bad = g.copy().set_geometry([INF, INF], crs="EPSG:3857")
    bad.to_file(tmp_path / "kyfromabove_phase3_pointcloud_5k_grid.gpkg", driver="GPKG")
    assert LidarTileIndex(tmp_path, phases=("phase2",)).phases == ["phase2"]
    with pytest.raises(ValueError, match="not finite"):
        LidarTileIndex(tmp_path, phases=("phase3",))


def test_the_repair_takes_shapes_by_name_and_makes_the_rest_from_the_name():
    ref = _phase2(["N078E247", "N078E248", "N078E249", "N079E248", "N079E249", "N077E248"])
    broken = gpd.GeoDataFrame({"tilename": ["N078E248", "N078E250"], "key": ["a", "b"], "aws_url": ["a", "b"]},
                              geometry=[INF, INF], crs="EPSG:3857")
    fixed, n = R.repair(broken, ref)
    assert n["from_reference"] == 1 and n["from_name"] == 1 and n["name_rule_error_ft"] < 0.5
    assert fixed.geometry.iloc[0].equals(ref.set_index("tilename").geometry["N078E248"])
    # N078E250 is not in the reference: made from its name, beside N078E249
    right_of_249 = ref.set_index("tilename").geometry["N078E249"]
    new = fixed.geometry.iloc[1]
    assert new.boundary.intersection(right_of_249.buffer(0.01)).length > 1000        # shares its 1.5 km edge
    assert new.intersection(right_of_249).area < 1.0                                # does not overlap
    assert R.finite(fixed.geometry).all()
