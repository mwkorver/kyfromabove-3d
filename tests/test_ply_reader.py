"""The PLY reader that the LiDAR clips and coloured clouds of older runs
still come in (new runs write LAZ; ``clouds.read_cloud`` reads either)."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from kyfromabove_3d.geometry import read_ply


def write_binary_ply(path, xyz, n_extra=56):
    """A binary PLY with many float properties per vertex (the layout of
    the colourised clouds, which the reader was first built for)."""
    n = len(xyz)
    names = ["x", "y", "z"] + [f"p_{i}" for i in range(n_extra)]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property float {nm}" for nm in names] + ["end_header", ""]
    body = bytearray()
    for i in range(n):
        body += struct.pack("<%df" % len(names), *xyz[i], *([0.0] * n_extra))
    path.write_bytes("\n".join(header).encode("ascii") + bytes(body))
    return path


def write_lidar_ply(path, xyz, classification=None):
    n = len(xyz)
    classification = np.full(n, 2) if classification is None else np.asarray(classification)
    lines = ["ply", "format ascii 1.0", f"element vertex {n}",
             "property float x", "property float y", "property float z",
             "property uchar intensity", "property uchar classification", "end_header"]
    lines += [f"{x:.4f} {y:.4f} {z:.4f} 0 {int(c)}"
              for (x, y, z), c in zip(xyz, classification)]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_binary_ply_round_trips_through_a_wide_layout(tmp_path):
    xyz = np.array([[1.0, 2.0, 3.0], [-4.0, 5.5, 0.25]])
    got = read_ply(write_binary_ply(tmp_path / "s.ply", xyz))
    assert len(got) == 59
    assert np.allclose(np.stack([got["x"], got["y"], got["z"]], axis=1), xyz)


def test_ascii_ply_reads_and_keeps_classification(tmp_path):
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0]])
    got = read_ply(write_lidar_ply(tmp_path / "l.ply", xyz, [2, 1, 7]))
    assert np.allclose(np.stack([got["x"], got["y"], got["z"]], axis=1), xyz)
    assert got["classification"].astype(int).tolist() == [2, 1, 7]


def test_malformed_ply_fails_loudly(tmp_path):
    bad = tmp_path / "bad.ply"
    bad.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 1\n"
                    b"property list uchar int vertex_index\nend_header\n")
    with pytest.raises(ValueError, match="list properties"):
        read_ply(bad)

    truncated = tmp_path / "trunc.ply"
    write_binary_ply(truncated, np.zeros((4, 3)))
    raw = truncated.read_bytes()
    truncated.write_bytes(raw[: len(raw) - 300])
    with pytest.raises(ValueError, match="truncated"):
        read_ply(truncated)
