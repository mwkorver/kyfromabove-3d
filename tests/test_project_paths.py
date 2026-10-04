"""Where a project writes, and why it matters that two buildings differ.

``qa_dir`` used to be ``output_dir / "qa"``. Two buildings sharing a
``paths.output`` then wrote their reports over each other with nothing said:
running PNC Tower into ``outputs/urban`` destroyed ``building_002``'s
reprojection report, which was the file the facade evidence came from. These
tests exist so that cannot come back.
"""

import textwrap

import pytest

from kyfromabove_3d.project import Project, repo_relative


def write_config(tmp_path, building_id, output="outputs/shared"):
    cfg = tmp_path / f"{building_id}.yaml"
    cfg.write_text(textwrap.dedent(f"""
        project:
          name: test
          phase: 1
        source:
          bucket: kyfromabove
          region: us-west-2
        aoi:
          horizontal_crs: "EPSG:3089"
          bounds: [5725000.0, 4025400.0, 5725350.0, 4025800.0]
          building_ids:
            - "{building_id}"
          pad_ft_us: 30.0
        selection:
          flight_ids: null
          camera_directions: ["Fwd"]
          sensor_family: "x"
        coordinates:
          local_origin_source_xyz: [5725184.6, 4025570.9, 600.0]
          horizontal_unit: "US survey foot"
          vertical_unit: "US survey foot"
          vertical_datum: "NAVD88"
        paths:
          raw: data/raw
          derived: data/derived
          output: {output}
        qa:
          target_median_check_error_px: 2.0
          minimum_frame_count: 3
    """).lstrip(), encoding="utf-8")
    return cfg


def test_qa_dir_is_under_the_building(tmp_path):
    proj = Project(config_path=str(write_config(tmp_path, "building_001")))
    assert proj.qa_dir.parent == proj.building_dir, (
        "qa_dir must sit under the building, not the output tree")
    assert proj.qa_dir.name == "qa"


def test_two_buildings_sharing_an_output_tree_do_not_collide(tmp_path):
    """The specific accident this replaced: one building overwriting another."""
    a = Project(config_path=str(write_config(tmp_path, "building_001")))
    b = Project(config_path=str(write_config(tmp_path, "pnc_tower")))

    assert a.output_dir == b.output_dir, "the fixture must share an output tree"
    assert a.qa_dir != b.qa_dir
    assert a.building_dir != b.building_dir
    # Every report name is the same for both, so a shared directory means a
    # silent overwrite rather than a clash anyone would notice.
    for name in ("reprojection_report.json", "cross_direction_report.json",
                 "triangulation_report.json", "frame_coverage.csv"):
        assert a.qa_dir / name != b.qa_dir / name


def test_repo_relative_hides_the_machine_a_report_was_made_on(tmp_path):
    inside = tmp_path / "outputs" / "urban" / "building_002"
    inside.mkdir(parents=True)
    assert repo_relative(inside, root=tmp_path) == "outputs/urban/building_002"
    # a path outside the repository is kept as given, not guessed at
    assert repo_relative("/elsewhere/b", root=tmp_path) == "/elsewhere/b"
