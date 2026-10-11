"""Every pipeline script's option parser builds: ``--help`` exits 0.

A duplicated option is only caught when the parser is built, which no other
test does -- a second ``--footprints`` in colorize_building.py made every
colourise exit 1 and every building of a downtown run fall back to a box."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ["colorize_building.py", "build_mesh.py", "check_building.py", "run_downtown.py",
           "merge_downtown_tileset.py", "rebuild_downtown_meshes.py", "plot_texture_check.py",
           "plot_corner_snap.py", "export_3dtiles.py", "eval_fill.py", "render_flyaround.py",
           "score_trajectory.py", "register_video.py", "video_poses.py", "wall_visibility.py", "wall_match.py",
           "face_match.py", "atlas_server.py"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_the_option_parser_builds(script):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--help"], cwd=str(ROOT),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-800:]
