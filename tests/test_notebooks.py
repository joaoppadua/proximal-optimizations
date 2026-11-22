"""Every notebook runs end to end as a script; notebooks 00 to 03 never touch OLMo."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.py"))


def olmo_uses(source: str) -> list[str]:
    """Every way `source` reaches OLMo: the module `proxopt.olmo` imported in any form
    (`import proxopt.olmo`, `from proxopt.olmo import ...`, `from proxopt import olmo`),
    the text `proxopt.olmo` anywhere, or the path `data/olmo`."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.startswith("proxopt.olmo")]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("proxopt.olmo"):
                found.append(f"from {node.module}")
            elif node.module == "proxopt":
                found += [f"from proxopt import {a.name}" for a in node.names if a.name == "olmo"]
    found += [s for s in ("proxopt.olmo", "data/olmo") if s in source]
    return found


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_runs_as_a_script(path):
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        timeout=300,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]


def test_early_notebooks_do_not_use_olmo():
    early = [p for p in NOTEBOOKS if p.name[:3] in {"00_", "01_", "02_", "03_"}]
    for path in early:
        assert olmo_uses(path.read_text()) == [], path.name
