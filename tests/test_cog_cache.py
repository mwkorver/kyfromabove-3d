"""The frame cache switch: with COG_CACHE_URL set, KyFromAbove's frames are
read through it (deploy/cog_cache), and nothing else changes."""

from kyfromabove_3d.imagery import cog_url

URL = "https://kyfromabove.s3.amazonaws.com/imagery/obliques/Phase3/KY_KYAPED_2024_Season1_3IN/Bwd_7022_46837.tif"


def test_frames_go_through_the_cache_only_when_it_is_set(monkeypatch):
    monkeypatch.delenv("COG_CACHE_URL", raising=False)
    assert cog_url(URL) == URL
    monkeypatch.setenv("COG_CACHE_URL", "http://127.0.0.1:8001/")
    assert cog_url(URL) == ("http://127.0.0.1:8001/imagery/obliques/Phase3/"
                            "KY_KYAPED_2024_Season1_3IN/Bwd_7022_46837.tif")
    assert cog_url("/data/local/frame.tif") == "/data/local/frame.tif"     # a local copy stays local
    assert cog_url("https://elsewhere.example/x.tif") == "https://elsewhere.example/x.tif"
