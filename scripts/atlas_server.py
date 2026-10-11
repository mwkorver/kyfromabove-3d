#!/usr/bin/env python3
"""Serve the facade atlas on this machine and keep the verdicts marked on it.

The published atlas keeps your marks in its claude.ai store; a copy opened
from this machine has none, so this server serves the atlas folder and adds
one small endpoint the page uses instead:

  GET  /api/verdicts   every face's verdict so far (a JSON object, face key -> verdict)
  POST /api/verdicts   a JSON object of face key -> verdict; each replaces the stored
                       one unless the stored one is newer (by its ``at`` time)

Verdicts are written to ``verdicts.json`` in the served folder, where
``face_match.py`` and Claude read them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
LOCK = threading.Lock()


def merge(stored: dict, incoming: dict) -> dict:
    """``stored`` with each incoming verdict that is at least as new as the one it replaces."""
    out = dict(stored)
    for k, v in incoming.items():
        if isinstance(v, dict) and (k not in out or str(v.get("at", "")) >= str(out[k].get("at", ""))):
            out[k] = v
    return out


class Handler(SimpleHTTPRequestHandler):
    store: Path

    def _json(self, code: int, body) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read(self) -> dict:
        try:
            return json.loads(self.store.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def do_GET(self):
        if self.path.split("?")[0] == "/api/verdicts":
            with LOCK:
                return self._json(200, self._read())
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?")[0] != "/api/verdicts":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            incoming = json.loads(self.rfile.read(min(n, 2_000_000)) or b"{}")
            if not isinstance(incoming, dict):
                raise ValueError("expected an object")
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        with LOCK:
            merged = merge(self._read(), incoming)
            tmp = self.store.with_suffix(".tmp")
            tmp.write_text(json.dumps(merged, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.store)
        return self._json(200, {"saved": len(incoming), "total": len(merged)})


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--directory", type=Path, default=ROOT / "outputs" / "wall_visibility" / "reference")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = parse_args(argv)
    Handler.store = a.directory / "verdicts.json"
    server = ThreadingHTTPServer((a.bind, a.port), partial(Handler, directory=str(a.directory)))
    print(f"atlas on http://{a.bind}:{a.port}/facade-atlas.html; verdicts -> {Handler.store}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
