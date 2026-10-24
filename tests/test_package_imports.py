"""The package stands alone: no module in ``src/kyfromabove_3d`` imports one
from ``scripts/``. Such an import works wherever ``scripts/`` happens to be on
the path (a test, a command started from ``scripts/``) and fails anywhere else
(a cloud job started otherwise), so it is caught here, not in a run."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_package_module_imports_a_script():
    scripts = {p.stem for p in (ROOT / "scripts").glob("*.py")}
    found = []
    for path in sorted((ROOT / "src" / "kyfromabove_3d").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            found += [f"{path.relative_to(ROOT)}:{node.lineno} {n}" for n in names if n.split(".")[0] in scripts]
    assert not found, "the package imports from scripts/: " + ", ".join(found)
