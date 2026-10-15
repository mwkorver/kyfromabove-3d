"""The settings a building is built with: one record per stage, typed.

Each stage's method settings live here with their defaults, and nowhere else:
the scripts take their command-line defaults from these records, and a
class's settings (``configs/class_strategies.yaml``, keyed by the class
``classify_buildings.py`` writes to ``building_class.json``) are checked
against them when the file is read -- a misspelt setting, a value of the
wrong kind or one that is not allowed stops the build there, instead of
being ignored. A building's settings are resolved in one order: the
defaults, then its class's, then what the command line changed.

Paths, which building, where the output goes: those stay the scripts' own
arguments. Only what changes how a building is modelled is a setting.
"""

from __future__ import annotations

import dataclasses
import json
import os
import typing
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

CLASS_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "class_strategies.yaml"


class SettingsError(ValueError):
    """A setting that does not exist, or a value it cannot take."""


@dataclass(frozen=True)
class LevelSettings:
    """How a footprint's bodies (a tower on its podium) are found from the
    roof returns: ``features.building_levels``'s parameters, with its
    defaults, and ``fallback_only``: the class's way is tried only where
    the defaults find one level."""
    cell_m: float = 1.0
    upper_fraction: float = 0.4
    min_cover: float = 0.05
    max_cover: float = 0.7
    min_step_m: float = 12.0
    min_area_m2: float = 150.0
    max_vertices: int = 16
    from_tiers: bool = False
    min_tier_share: float = 0.10
    below_main: bool = False
    straighten: bool = False
    fine_cell_m: float = 0.5
    smooth_m: float = 1.0
    fallback_only: bool = False

    def kwargs(self) -> Dict[str, Any]:
        """The arguments for ``building_levels``."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "fallback_only"}


@dataclass(frozen=True)
class ColouriseSettings:
    """The colourise (``colorize_building.py``): the wall planes and their
    bands, the levels, and the colour from the frames."""
    spacing_m: float = 0.15
    min_face_m: float = 4.0
    look_in_share: float = 0.45
    max_offset_m: float = 2.0
    tile_m: float = 12.0
    max_plane_rms_m: float = 0.35
    min_facing_cos: float = 0.15
    occlusion_slack_m: float = 1.5
    drop_classes: typing.Tuple[int, ...] = (3, 4, 5, 7, 9)
    corner_snap: bool = True
    find_levels: bool = True
    prism_occluders: bool = True
    twin_colour: bool = False
    occluder_radius_m: float = 250.0
    occlusion: bool = True
    extrude: Literal["profile", "prism", "verified", "off"] = "profile"
    prism_min_ratio: float = 0.60
    outline: Literal["overture", "lidar"] = "overture"
    match_walls: bool = False
    profile_band_m: float = 5.0
    extrude_max_rms_m: float = 0.30
    extrude_max_normal_deg: float = 4.0
    # every face of an upper level is under a measured roof edge: "twin" (an
    # unseen band takes its twin's depth) or "roof_edge" (it takes the edge's)
    upper_levels: Literal["twin", "roof_edge"] = "twin"
    levels: LevelSettings = field(default_factory=LevelSettings)


@dataclass(frozen=True)
class MeshSettings:
    """The mesh (``build_mesh.py``) and the completed product."""
    texel_m: float = 0.15
    roof_tidy: bool = True
    wall_cap: bool = True
    roof_edge: Literal["outline", "grid"] = "outline"
    roof_cell_m: float = 0.0
    texture: Literal["ktx2", "webp", "jpeg", "png"] = "ktx2"
    # a face that is a party wall over half its length: "seen" (only what the
    # colourise saw) or "whole" (outline to roof edge, plain to the neighbour's roof)
    party_walls: Literal["seen", "whole"] = "seen"
    # an upper level's lowest wall: "level" (from its own foot) or
    # "down_to_roof" (to the roof beside it where that is lower)
    tower_foot: Literal["level", "down_to_roof"] = "level"
    # "absent": a footprint the LiDAR shows no building on is a box at its
    # recorded height in the completed product, from absent_min_area_m2 up
    lidar: Literal["present", "absent"] = "present"
    absent_min_area_m2: float = 0.0


@dataclass(frozen=True)
class BuildingSettings:
    """Everything a building is modelled with, and the class it came from."""
    colourise: ColouriseSettings = field(default_factory=ColouriseSettings)
    mesh: MeshSettings = field(default_factory=MeshSettings)
    building_class: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


#: Where each setting of configs/class_strategies.yaml goes: (stage, field).
CLASS_KEYS = {
    "outline": ("colourise", "outline"),
    "upper_levels": ("colourise", "upper_levels"),
    "levels": ("colourise", "levels"),
    "party_walls": ("mesh", "party_walls"),
    "tower_foot": ("mesh", "tower_foot"),
    "lidar": ("mesh", "lidar"),
    "min_area_m2": ("mesh", "absent_min_area_m2"),
}
STAGES = {"colourise": ColouriseSettings, "mesh": MeshSettings}


def _check(cls, name: str, value: Any, where: str) -> Any:
    """``value`` for field ``name`` of dataclass ``cls``, checked (and a
    nested record built from a mapping); raises SettingsError."""
    hints = typing.get_type_hints(cls)
    if name not in hints:
        known = ", ".join(f.name for f in fields(cls))
        raise SettingsError(f"{where}: no setting {name!r} (known: {known})")
    t = hints[name]
    if dataclasses.is_dataclass(t):
        if not isinstance(value, Mapping):
            raise SettingsError(f"{where}: {name} takes a mapping of settings, not {value!r}")
        return replace(t(), **{k: _check(t, k, v, f"{where}.{name}") for k, v in value.items()})
    if typing.get_origin(t) is Literal:
        if value not in typing.get_args(t):
            raise SettingsError(f"{where}: {name} is one of {typing.get_args(t)}, not {value!r}")
        return value
    if t is bool:
        if not isinstance(value, bool):
            raise SettingsError(f"{where}: {name} is true or false, not {value!r}")
        return value
    if t is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingsError(f"{where}: {name} is a number, not {value!r}")
        return float(value)
    if t is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SettingsError(f"{where}: {name} is a whole number, not {value!r}")
        return value
    if typing.get_origin(t) is tuple:
        if not isinstance(value, (list, tuple)) or not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            raise SettingsError(f"{where}: {name} is a list of whole numbers, not {value!r}")
        return tuple(value)
    return value


def class_overrides(entry: Mapping[str, Any], where: str) -> Dict[str, Dict[str, Any]]:
    """One class's entry as {stage: {field: checked value}}."""
    out: Dict[str, Dict[str, Any]] = {s: {} for s in STAGES}
    for key, value in (entry or {}).items():
        if key not in CLASS_KEYS:
            raise SettingsError(f"{where}: no class setting {key!r} (known: {', '.join(CLASS_KEYS)})")
        stage, name = CLASS_KEYS[key]
        out[stage][name] = _check(STAGES[stage], name, value, where)
    return out


def load_class_settings(path: Optional[Path] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Every class's checked settings: {class: {stage: {field: value}}}.
    ``KFA_CLASS_STRATEGIES`` names another file (an A/B build)."""
    import yaml
    path = Path(path or os.environ.get("KFA_CLASS_STRATEGIES") or CLASS_CONFIG)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {cls: class_overrides(entry, f"{path.name}, class {cls}") for cls, entry in raw.items()}


def building_class(building: Path) -> Optional[str]:
    p = Path(building) / "building_class.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("class")


def resolve(building: Optional[Path] = None, colourise: Optional[Mapping[str, Any]] = None,
            mesh: Optional[Mapping[str, Any]] = None, class_name: Optional[str] = None,
            config: Optional[Path] = None) -> BuildingSettings:
    """A building's settings: the defaults, then its class's (from
    ``building_class.json`` in ``building``, or ``class_name``), then the
    given changes per stage (what the command line changed)."""
    cls = class_name if class_name is not None else (building_class(building) if building is not None else None)
    over = load_class_settings(config).get(cls, {}) if cls else {}
    stages = {}
    for stage, rec in STAGES.items():
        s = replace(rec(), **over.get(stage, {}))
        given = (colourise if stage == "colourise" else mesh) or {}
        changes = {k: _check(rec, k, v, f"{stage} settings") for k, v in given.items()}
        stages[stage] = replace(s, **changes)
    return BuildingSettings(colourise=stages["colourise"], mesh=stages["mesh"], building_class=cls)


def changed_from_defaults(values: Mapping[str, Any], rec) -> Dict[str, Any]:
    """The values that differ from ``rec``'s defaults (what the command line changed)."""
    d = rec()
    return {k: v for k, v in values.items() if hasattr(d, k) and getattr(d, k) != v}
