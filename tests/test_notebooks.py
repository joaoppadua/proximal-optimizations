"""Every notebook runs end to end as a script; notebooks 00 to 03 never touch OLMo."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.py"))


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
        text = path.read_text()
        assert "proxopt.olmo" not in text, path.name
        assert "data/olmo" not in text, path.name
