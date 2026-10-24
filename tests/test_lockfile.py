"""The cloud image and the laptop run the same libraries: every pin in
deploy/requirements-lambda.txt is the version uv.lock records. (The image
installs from that file; the laptop and CI from the lock.)"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: the image's name for a library the project names otherwise
SAME_AS = {"opencv-python-headless": "opencv-python"}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def test_the_image_pins_are_the_locked_versions():
    lock = {p["name"]: p.get("version") for p in tomllib.loads((ROOT / "uv.lock").read_text())["package"]}
    pins = {}
    for line in (ROOT / "deploy" / "requirements-lambda.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            name, _, version = line.partition("==")
            pins[_norm(name)] = version
    assert pins, "no pins read"
    wrong = {n: (v, lock.get(SAME_AS.get(n, n))) for n, v in pins.items() if lock.get(SAME_AS.get(n, n)) != v}
    assert not wrong, f"image pin vs uv.lock: {wrong}"
