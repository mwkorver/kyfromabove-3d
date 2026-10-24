"""A reader for the PLY point clouds of older runs.

New runs write their clouds as LAZ (``clouds.py``); the 378 downtown
directories and the landmarks from before that still hold ``lidar_local.ply``
(ASCII) and ``colorized_aerial.ply`` (binary), and ``clouds.read_cloud``
comes here for those.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

_PLY_DTYPES = {
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
    "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4",
}


def read_ply(path: Path) -> Dict[str, np.ndarray]:
    """Read a PLY's vertex element into a dict of columns.

    Handles the two formats in play here: the ascii LiDAR export and the
    binary_little_endian clouds the colour step writes. Deliberately not a general PLY library -- list
    properties and big-endian are rejected loudly rather than mis-parsed.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt = fh.readline().decode("ascii", "replace").strip()
        names, types, count, in_vertex = [], [], None, False
        while True:
            raw = fh.readline()
            if not raw:
                raise ValueError(f"{path}: header ended without end_header")
            line = raw.decode("ascii", "replace").strip()
            if line == "end_header":
                break
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError(f"{path}: list properties are not supported")
                if parts[1] not in _PLY_DTYPES:
                    raise ValueError(f"{path}: unsupported property type {parts[1]}")
                types.append(_PLY_DTYPES[parts[1]])
                names.append(parts[2])
        if count is None:
            raise ValueError(f"{path}: no vertex element")
        body = fh.read()

    if fmt.startswith("format binary_little_endian"):
        dtype = np.dtype([(n, "<" + t) for n, t in zip(names, types)])
        need = dtype.itemsize * count
        if len(body) < need:
            raise ValueError(f"{path}: truncated, wanted {need} bytes, got {len(body)}")
        arr = np.frombuffer(body[:need], dtype=dtype)
        return {n: np.asarray(arr[n]) for n in names}

    if fmt.startswith("format ascii"):
        rows = [ln.split() for ln in body.decode("ascii", "replace").splitlines() if ln.strip()]
        if len(rows) < count:
            raise ValueError(f"{path}: wanted {count} vertices, found {len(rows)}")
        table = np.array(rows[:count], dtype=np.float64)
        return {n: table[:, i].astype(t if t.startswith("f") else "f8")
                for i, (n, t) in enumerate(zip(names, types))}

    raise ValueError(f"{path}: unsupported PLY format line {fmt!r}")
