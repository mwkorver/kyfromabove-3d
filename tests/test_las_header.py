"""LAS public header parsing, hand-rolled against the specification.

``read_las_header`` exists so that comparing 46,000 tiles costs one range
request each instead of opening 50 MB files. That only works if the parser is
right, and a byte-offset parser is exactly the kind of code that returns
plausible numbers when it is wrong -- a misread point count looks like a sparse
tile, not like a bug. So the tests build headers to the spec and read them back.
"""

import struct

import pytest

from kyfromabove_3d.lidar import read_las_header


def build_las_header(version=(1, 4), n_points=1_000_000, legacy_n=None,
                     point_format=6, compressed=True, day=342, year=2024,
                     bounds=(0.0, 5000.0, 0.0, 5000.0, 500.0, 900.0)):
    """A LAS public header block, built at the offsets the specification names."""
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    h = bytearray(375)
    h[0:4] = b"LASF"
    h[24], h[25] = version
    struct.pack_into("<HH", h, 90, day, year)
    struct.pack_into("<H", h, 94, 375)
    fmt_byte = point_format | (0x80 if compressed else 0x00)
    h[104] = fmt_byte
    struct.pack_into("<H", h, 105, 30)
    # LAS 1.4 zeroes the legacy count for point formats above 5.
    struct.pack_into("<I", h, 107, legacy_n if legacy_n is not None else 0)
    struct.pack_into("<6d", h, 179, max_x, min_x, max_y, min_y, max_z, min_z)
    if version[1] >= 4:
        struct.pack_into("<Q", h, 247, n_points)
    return bytes(h)


def write(tmp_path, name="t.laz", **kw):
    p = tmp_path / name
    p.write_bytes(build_las_header(**kw))
    return p


def test_reads_las_14_point_count_from_the_64_bit_field(tmp_path):
    """LAS 1.4 moved the count; falling back to the legacy field reports zero."""
    h = read_las_header(write(tmp_path, n_points=8_759_469, legacy_n=0))
    assert h.n_points == 8_759_469
    assert h.version == "1.4"


def test_falls_back_to_the_legacy_count_for_las_12(tmp_path):
    h = read_las_header(write(tmp_path, version=(1, 2), n_points=0,
                              legacy_n=7_321_833, point_format=1))
    assert h.n_points == 7_321_833
    assert h.version == "1.2"


def test_compression_flag_is_split_from_the_point_format(tmp_path):
    """The stored byte is format | 0x80; reporting 134 instead of 6 is the bug."""
    laz = read_las_header(write(tmp_path, point_format=6, compressed=True))
    assert (laz.point_format, laz.compressed) == (6, True)
    las = read_las_header(write(tmp_path, "u.las", point_format=6, compressed=False))
    assert (las.point_format, las.compressed) == (6, False)


def test_creation_date_is_day_of_year_not_month(tmp_path):
    assert read_las_header(write(tmp_path, day=342, year=2024)).created == "2024-12-07"
    assert read_las_header(write(tmp_path, "a.laz", day=1, year=2013)).created == "2013-01-01"
    # A zero year means unset, and must not raise or invent a date.
    assert read_las_header(write(tmp_path, "b.laz", day=0, year=0)).created == ""


def test_density_uses_us_survey_feet_and_z_span_detects_blunders(tmp_path):
    h = read_las_header(write(tmp_path, n_points=8_759_469,
                              bounds=(0.0, 5000.0, 0.0, 5000.0, 544.0, 872.9)))
    # 5000 US survey ft = 1524.003 m; 8.76e6 / 1524.003^2 ~= 3.77 pts/m^2.
    assert h.density_per_m2() == pytest.approx(3.77, abs=0.02)
    assert h.z_span == pytest.approx(328.9)
    assert (h.width, h.height) == (5000.0, 5000.0)


def test_malformed_headers_fail_loudly(tmp_path):
    not_las = tmp_path / "x.laz"
    not_las.write_bytes(b"NOPE" + bytes(400))
    with pytest.raises(ValueError, match="not a LAS/LAZ file"):
        read_las_header(not_las)

    short = tmp_path / "s.laz"
    short.write_bytes(build_las_header()[:200])
    with pytest.raises(ValueError, match="truncated"):
        read_las_header(short)
