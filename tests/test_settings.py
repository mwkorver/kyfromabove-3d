"""The typed settings (kyfromabove_3d.settings): one place for the defaults,
the class file checked when it is read, one order of resolution."""

from __future__ import annotations

import dataclasses
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kyfromabove_3d import settings as S  # noqa: E402


def test_level_defaults_are_the_level_finders_own():
    from kyfromabove_3d.features import building_levels
    sig = {k: p.default for k, p in inspect.signature(building_levels).parameters.items()
           if p.default is not inspect.Parameter.empty}
    assert S.LevelSettings().kwargs() == sig


def test_the_class_file_reads_and_every_class_resolves():
    classes = S.load_class_settings(S.CLASS_CONFIG)
    assert classes, "configs/class_strategies.yaml has no classes"
    for name in classes:
        b = S.resolve(class_name=name, config=S.CLASS_CONFIG)
        assert b.building_class == name


@pytest.mark.parametrize("entry, says", [
    ({"party_wall": "whole"}, "no class setting 'party_wall'"),
    ({"tower_foot": "down"}, "tower_foot is one of"),
    ({"levels": {"min_stepm": 4}}, "no setting 'min_stepm'"),
    ({"levels": {"straighten": "yes"}}, "straighten is true or false"),
    ({"levels": 4}, "takes a mapping"),
    ({"min_area_m2": "big"}, "is a number"),
])
def test_a_mistake_in_a_class_stops_the_build(entry, says):
    with pytest.raises(S.SettingsError, match=says):
        S.class_overrides(entry, "test")


def test_defaults_then_class_then_command_line(tmp_path):
    cfg = tmp_path / "classes.yaml"
    cfg.write_text("tiered/mid:\n  party_walls: whole\n  levels:\n    min_step_m: 4.0\n", encoding="utf-8")
    b = S.resolve(class_name="tiered/mid", config=cfg)
    assert b.mesh.party_walls == "whole" and b.colourise.levels.min_step_m == 4.0
    assert b.colourise.levels.min_area_m2 == S.LevelSettings().min_area_m2      # the rest stays default
    b = S.resolve(class_name="tiered/mid", config=cfg, mesh={"party_walls": "seen", "wall_cap": False})
    assert b.mesh.party_walls == "seen" and b.mesh.wall_cap is False
    assert S.resolve(class_name="box/low", config=cfg) == S.BuildingSettings(building_class="box/low")


@pytest.mark.parametrize("script, stage", [("colorize_building", S.ColouriseSettings), ("build_mesh", S.MeshSettings)])
def test_a_scripts_defaults_are_the_settings_defaults(script, stage):
    mod = __import__(script)
    args = mod.parser().parse_args(["--building", "x"])
    got = mod.settings_from_args(args)
    assert got == stage(), {f.name: (getattr(got, f.name), getattr(stage(), f.name))
                            for f in dataclasses.fields(stage) if getattr(got, f.name) != getattr(stage(), f.name)}


def test_the_ab_switches_reach_their_settings():
    import build_mesh
    import colorize_building
    m = build_mesh.settings_from_args(build_mesh.parser().parse_args(
        ["--building", "x", "--no-wall-cap", "--no-roof-tidy", "--roof-edge", "grid"]))
    assert (m.wall_cap, m.roof_tidy, m.roof_edge) == (False, False, "grid")
    c = colorize_building.settings_from_args(colorize_building.parser().parse_args(
        ["--building", "x", "--lidar-outline", "--no-prism-occluders", "--no-levels", "--drop-classes", ""]))
    assert (c.outline, c.prism_occluders, c.find_levels, c.drop_classes) == ("lidar", False, False, ())
    # what the command line changed wins over the class
    b = S.resolve(class_name="box/tall", config=S.CLASS_CONFIG, mesh=S.changed_from_defaults({"wall_cap": False}, S.MeshSettings))
    assert b.mesh.wall_cap is False and b.colourise.outline == "lidar"


def test_the_building_lists_read():
    import yaml
    for f in ("safety_net.yaml", "audit_reference.yaml", "landmarks.yaml", "class_strategies.yaml"):
        assert yaml.safe_load((ROOT / "configs" / f).read_text(encoding="utf-8"))


def test_no_class_settings_stops_the_build(tmp_path):
    with pytest.raises(S.SettingsError, match="no class settings"):
        S.load_class_settings(tmp_path / "missing.yaml")
