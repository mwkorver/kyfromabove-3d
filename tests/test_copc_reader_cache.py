"""The per-process COPC reader cache: a worker that touches more than 16
LiDAR tiles evicts the oldest reader (laspy's CopcReader has no ``close``;
seen September 27 as 984 failed buildings on a long-lived Spot worker)."""

from kyfromabove_3d import lidar


class FakeReader:
    """Like laspy 2.7's CopcReader: closes through ``__exit__`` only."""

    def __init__(self, url):
        self.url, self.closed = url, False

    def __exit__(self, *exc):
        self.closed = True


def test_eviction_closes_the_oldest(monkeypatch):
    monkeypatch.setattr(lidar, "_READERS", {})
    monkeypatch.setattr(lidar, "open_copc", FakeReader)
    first = lidar.copc_reader("u0")
    for i in range(1, 20):
        lidar.copc_reader(f"u{i}")
    assert len(lidar._READERS) == 16
    assert first.closed and "u0" not in lidar._READERS
    assert lidar.copc_reader("u19") is lidar._READERS["u19"]


def test_failed_read_drops_and_closes_the_reader(monkeypatch):
    class Broken(FakeReader):
        def query(self, bounds, **kw):
            raise OSError("mid-read")

    monkeypatch.setattr(lidar, "_READERS", {})
    monkeypatch.setattr(lidar, "open_copc", Broken)
    monkeypatch.setattr(lidar, "with_retries", lambda f: f())
    r = lidar.copc_reader("u")
    try:
        lidar.copc_query("u", (0, 0, 1, 1))
    except OSError:
        pass
    assert r.closed and "u" not in lidar._READERS
