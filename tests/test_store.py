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
    with pytest.raises(FileNotFoundError):                         # a missing object: the same error from both
        store.get_bytes("pieces/a/nothing.json")
    assert store.list_dirs("pieces/") == ["a", "b"] and store.list_dirs("tiles/") == ["t"]   # one level down
    assert store.list_dirs("nothing/") == []
    # a few bytes out of a stored object, and its length, without reading it whole
    store.put_bytes("pieces/c/piece.zip", bytes(range(200)))
    assert store.size("pieces/c/piece.zip") == 200
    assert store.get_range("pieces/c/piece.zip", 10, 14) == bytes([10, 11, 12, 13])
    assert store.get_range("pieces/c/piece.zip", 190, 400) == bytes(range(190, 200))


def test_a_zip_is_read_member_by_member_with_ranged_reads(tmp_path):
    from kyfromabove_3d.archive import read_zip_members, zip_bytes

    st = LocalStore(tmp_path)
    big = bytes(range(256)) * 4000                                 # a megabyte at the end, as a piece's LiDAR is
    st.put_bytes("p/piece.zip", zip_bytes({"frames.json": b"[1, 2]", "lidar_local.laz": big}))
    reads = []
    real = st.get_range
    st.get_range = lambda k, a, b: reads.append(b - a) or real(k, a, b)
    got = read_zip_members(st, "p/piece.zip", ("frames.json", "missing.json"))
    assert got == {"frames.json": b"[1, 2]"}
    assert sum(reads) < len(big) // 4                              # the LiDAR stayed unread


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
