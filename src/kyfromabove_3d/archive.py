"""Zip archives a building's files travel in.

S3 charges for every upload, and most of a building's files are a few KB,
so a building is stored as a few zips rather than about twenty files
(:func:`kyfromabove_3d.building.pack_building`). The zips are written the
same way every time -- members sorted, one fixed timestamp, fixed
permissions -- so the same files give the same bytes, and a rebuild that
changed nothing uploads nothing new. Files that are already compressed
(glb, png, npz, ktx2, laz) are stored as they are; text is deflated.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Mapping, Union

#: 1980-01-01, the earliest time a zip can record: the same in every archive.
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
#: Already compressed: deflating them again costs time and saves nothing.
STORED = (".glb", ".png", ".jpg", ".jpeg", ".webp", ".ktx2", ".npz", ".laz", ".zip")


def zip_bytes(members: Mapping[str, Union[Path, bytes]]) -> bytes:
    """``{name in the zip: a file or its bytes}`` as one deterministic zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name in sorted(members):
            src = members[name]
            data = Path(src).read_bytes() if isinstance(src, Path) else bytes(src)
            info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
            info.compress_type = (zipfile.ZIP_STORED if name.lower().endswith(STORED)
                                  else zipfile.ZIP_DEFLATED)
            info.external_attr = 0o644 << 16
            info.create_system = 3                         # unix, whatever made it
            z.writestr(info, data, compresslevel=6 if info.compress_type == zipfile.ZIP_DEFLATED else None)
    return buf.getvalue()


def read_zip(store, key: str) -> zipfile.ZipFile:
    """The zip at ``key`` in ``store``, read with one request; its members
    come back with ``.read(name)`` and ``.namelist()``."""
    return zipfile.ZipFile(io.BytesIO(store.get_bytes(key)))
