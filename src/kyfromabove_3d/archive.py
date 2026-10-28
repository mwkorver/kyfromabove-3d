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


class _Ranged(io.RawIOBase):
    """A stored object as a seekable file, read in ``block``-sized ranges as
    they are touched: a zip's directory at its end and a few small members
    at its start cost two reads, not the whole archive."""

    def __init__(self, store, key: str, block: int = 64 * 1024):
        self.store, self.key, self.block = store, key, block
        self.length = store.size(key)
        self.pos = 0
        self.blocks: dict = {}

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.length + offset}[whence]
        return self.pos

    def readinto(self, b) -> int:
        n = min(len(b), max(0, self.length - self.pos))
        out = bytearray()
        while len(out) < n:
            i, at = divmod(self.pos + len(out), self.block)
            if i not in self.blocks:
                self.blocks[i] = self.store.get_range(self.key, i * self.block, min(self.length, (i + 1) * self.block))
            out += self.blocks[i][at:at + n - len(out)]
        b[:n] = out
        self.pos += n
        return n


def read_zip_members(store, key: str, names) -> dict:
    """``{name: bytes}`` for those of ``names`` the zip at ``key`` holds,
    read with ranged requests (:class:`_Ranged`) instead of the whole zip."""
    with zipfile.ZipFile(io.BufferedReader(_Ranged(store, key))) as z:
        have = set(z.namelist())
        return {n: z.read(n) for n in names if n in have}
