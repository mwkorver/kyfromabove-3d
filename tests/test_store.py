"""The two stores behave alike: a folder, and S3 (mocked by moto)."""

from __future__ import annotations

import pytest

from kyfromabove_3d.store import LocalStore, S3Store, store_from


def _exercise(store, tmp_path):
    store.put_json("pieces/a/footprint.json", {"gers": "a"})
    f = tmp_path / "clip.laz"
    f.write_bytes(b"\x00\x01\x02")
    store.put_file("pieces/a/lidar_local.laz", f)
    store.put_bytes("pieces/b/piece.json", b"{}")
    assert store.exists("pieces/a/footprint.json") and not store.exists("pieces/a/nothing.json")
    assert store.read_json("pieces/a/footprint.json") == {"gers": "a"}
    assert store.get_bytes("pieces/a/lidar_local.laz") == b"\x00\x01\x02"
    assert sorted(store.list("pieces/a/")) == ["pieces/a/footprint.json", "pieces/a/lidar_local.laz"]
    assert len(list(store.list("pieces/"))) == 3
    back = store.get_file("pieces/a/lidar_local.laz", tmp_path / "back" / "x.laz")
    assert back.read_bytes() == b"\x00\x01\x02"
    glb = b"glTF" + bytes(4000)                                    # compressible, as geometry is
    store.put_bytes("tiles/t/3dtiles/measured/boxes.glb", glb, "model/gltf-binary", content_encoding="gzip")
    assert store.get_bytes("tiles/t/3dtiles/measured/boxes.glb") == glb   # read back as it was written


def test_a_local_store(tmp_path):
    _exercise(LocalStore(tmp_path / "store"), tmp_path)
    with pytest.raises(ValueError):
        LocalStore(tmp_path / "store").put_bytes("../escape.txt", b"x")


def test_an_s3_store_behaves_the_same(tmp_path):
    moto = pytest.importorskip("moto")
    import boto3
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-west-2")
        s3.create_bucket(Bucket="kfa3d", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
        _exercise(S3Store("kfa3d", "runs/test", client=s3), tmp_path)
        assert s3.head_object(Bucket="kfa3d", Key="runs/test/pieces/a/footprint.json")
        glb = "runs/test/" + "tiles/t/3dtiles/measured/boxes.glb"          # split: git-secrets reads 40 chars as a key
        head = s3.head_object(Bucket="kfa3d", Key=glb)
        assert head["ContentEncoding"] == "gzip" and head["ContentLength"] < 200   # stored gzipped, says so


def test_store_from_a_spec(tmp_path):
    assert isinstance(store_from(str(tmp_path)), LocalStore)
