"""Where a stage's results go: a folder on this machine or a prefix in S3.

The pipeline writes a building's files into a local folder, as it always
has; publishing that folder is the one step that knows about storage. A
``Store`` is the few operations that step needs -- put a file or bytes, read
bytes back, ask whether a key exists, list a prefix -- so the same stage code
runs on the laptop (``LocalStore``) and in Lambda (``S3Store``), and the
tests run against either (``S3Store`` over moto, a mocked S3).

Keys are always forward-slash paths relative to the store's root, e.g.
``pieces/<gers>/footprint.json``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterator, Optional


class Store:
    """The operations a stage needs from storage."""

    def put_bytes(self, key: str, data: bytes, content_type: Optional[str] = None,
                  content_encoding: Optional[str] = None) -> None:
        """``content_encoding="gzip"``: a store that serves files over HTTP
        keeps them gzipped and says so (browsers unpack them; reads here
        give the original bytes back); a folder keeps them as they are."""
        raise NotImplementedError

    def put_file(self, key: str, path: Path) -> None:
        raise NotImplementedError

    def get_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def list(self, prefix: str) -> Iterator[str]:
        raise NotImplementedError

    def get_file(self, key: str, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.get_bytes(key))
        return path

    def put_json(self, key: str, obj: Any) -> None:
        self.put_bytes(key, json.dumps(obj, indent=1, default=float).encode("utf-8"), "application/json")

    def read_json(self, key: str) -> Any:
        return json.loads(self.get_bytes(key).decode("utf-8"))


class LocalStore(Store):
    """A folder: key ``a/b.json`` is ``<root>/a/b.json``."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents and p != self.root.resolve():
            raise ValueError(f"key {key!r} leaves the store's root")
        return p

    def put_bytes(self, key: str, data: bytes, content_type: Optional[str] = None,
                  content_encoding: Optional[str] = None) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(p)                                   # a reader never sees half a file

    def put_file(self, key: str, path: Path) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part")
        shutil.copyfile(path, tmp)
        tmp.replace(p)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list(self, prefix: str) -> Iterator[str]:
        base = self._path(prefix) if prefix else self.root
        start = base if base.is_dir() else base.parent
        if not start.exists():
            return
        for p in sorted(start.rglob("*")):
            if p.is_file() and not p.name.endswith(".part"):
                key = p.relative_to(self.root.resolve()).as_posix()   # _path resolves: so must the root
                if key.startswith(prefix):
                    yield key


class S3Store(Store):
    """A bucket and a prefix. ``endpoint_url`` points it at LocalStack (or
    any S3-compatible store) instead of AWS."""

    def __init__(self, bucket: str, prefix: str = "", endpoint_url: Optional[str] = None,
                 region: Optional[str] = None, client: Any = None):
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.s3 = client or boto3.client("s3", endpoint_url=endpoint_url, region_name=region)

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_bytes(self, key: str, data: bytes, content_type: Optional[str] = None,
                  content_encoding: Optional[str] = None) -> None:
        extra = {"ContentType": content_type} if content_type else {}
        if content_encoding == "gzip":
            import gzip

            data = gzip.compress(data, compresslevel=9, mtime=0)
            extra["ContentEncoding"] = "gzip"
        elif content_encoding:
            raise ValueError(f"content encoding {content_encoding!r}: only gzip")
        self.s3.put_object(Bucket=self.bucket, Key=self._key(key), Body=data, **extra)

    def put_file(self, key: str, path: Path) -> None:
        self.s3.upload_file(str(path), self.bucket, self._key(key))

    def get_bytes(self, key: str) -> bytes:
        r = self.s3.get_object(Bucket=self.bucket, Key=self._key(key))
        data = r["Body"].read()
        if r.get("ContentEncoding") == "gzip":
            import gzip

            data = gzip.decompress(data)
        return data

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list(self, prefix: str) -> Iterator[str]:
        cut = len(self.prefix) + 1 if self.prefix else 0
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=self._key(prefix)):
            for o in page.get("Contents", []):
                yield o["Key"][cut:]


def store_from(spec: str, endpoint_url: Optional[str] = None) -> Store:
    """``s3://bucket/prefix`` or a local folder."""
    if spec.startswith("s3://"):
        bucket, _, prefix = spec[5:].partition("/")
        return S3Store(bucket, prefix, endpoint_url=endpoint_url)
    return LocalStore(Path(spec))
