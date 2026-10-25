"""The VC Map oblique collection's map tiles (``scripts/oblique_collection.py``):
a photo is listed in the web-mercator tile holding its ground centre, numbered
as OpenLayers numbers them (y counted down from the top), or VC Map looks for
it in the wrong file."""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import oblique_collection as oc  # noqa: E402


def test_a_photo_is_listed_in_the_tile_openlayers_looks_in():
    from pyproj import Transformer

    lon, lat = -85.7585, 38.2527                                       # downtown Louisville
    e, n = Transformer.from_crs("EPSG:4326", oc.UTM16N, always_xy=True).transform(lon, lat)
    k = 2 ** 14                                                         # the slippy-map tile numbers
    x = int((lon + 180) / 360 * k)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * k)
    assert oc.tile_of([e, n, 140.0], 14) == f"14/{x}/{y}"
