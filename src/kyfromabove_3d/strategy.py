"""The strategy for a building's class.

Each downtown building is classed by size and architecture before anything is
modelled (scripts/classify_buildings.py writes ``building_class.json`` beside
it), and ``configs/class_strategies.yaml`` gives each class its settings. A
building with no class, or a class with no entry, is built as before classes.
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "class_strategies.yaml"


def class_strategy(building, config: Path = CONFIG) -> dict:
    """The building's class's settings (``_class`` holds the class), or {}."""
    import yaml
    p = Path(building) / "building_class.json"
    if not p.exists() or not Path(config).exists():
        return {}
    cls = json.loads(p.read_text(encoding="utf-8")).get("class")
    entry = (yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}).get(cls)
    return {**entry, "_class": cls} if entry else {}
