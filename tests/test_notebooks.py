"""Every notebook runs end to end as a script; notebooks 00 to 03 never touch OLMo."""

import ast
import json
import os
import shutil
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


SYNTHESIS = ROOT / "notebooks" / "04_synthesis.py"

# Runs the notebook's app in a fresh process and prints each cell's rendered output (not its
# source) as plain text, one per line, so a test can check what a reader would see. With a
# second argument "live", the live comparison's button counts as pressed and its result is
# the committed seed-0 runs (read from data/synthesis.json directly), so the live figures
# are drawn without training.
_RENDER = """
import importlib.util, json, re, sys
from pathlib import Path
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location("nb", sys.argv[1])
nb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nb)
defs = None
if sys.argv[2:] == ["live"]:
    from proxopt import synthesis
    saved = synthesis.load(Path("data/synthesis.json"))
    def fake_compare(beta, tau, seed, rates):
        i, n = saved["seeds"].index(seed), len(saved["seeds"])
        return {"beta": beta, "tau": tau, "seeds": [seed],
                "runs": {k: [v[i]] for k, v in saved["runs"].items()},
                "points": {k: v[i::n] for k, v in saved["points"].items()},
                "reward_model": [r for r in saved["reward_model"] if r["seed"] == seed]}
    defs = {"live_button": SimpleNamespace(value=True), "beta_live": SimpleNamespace(value=0.1),
            "tau_live": SimpleNamespace(value=0.25), "seed_live": SimpleNamespace(value=0),
            "live_compare": fake_compare}
outputs, _ = nb.app.run(defs=defs)
texts = []
for o in outputs:
    if o is None:
        continue
    t = getattr(o, "text", None) or str(o)
    texts.append(re.sub(r"\\s+", " ", re.sub(r"<[^>]+>", " ", t)).strip())
print(json.dumps(texts))
"""


def rendered_outputs(env_overrides: dict[str, str], live: bool = False) -> list[str]:
    """Notebook 04's cell outputs as text, in cell order, run with `env_overrides`; with
    `live`, as if the live comparison's button had been pressed (see `_RENDER`)."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("PROXOPT_OLMO_DIR", "PROXOPT_SYNTHESIS_FILE")}
    env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, "-c", _RENDER, str(SYNTHESIS), *(["live"] if live else [])],
        cwd=ROOT, env=env, timeout=300, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def _section(outputs: list[str], start: str, stop: str) -> list[str]:
    """The outputs from the first one containing `start` up to (not including) the first
    later one containing `stop`."""
    i = next(k for k, t in enumerate(outputs) if start in t)
    j = next(k for k in range(i + 1, len(outputs)) if stop in outputs[k])
    return outputs[i:j]


# Text that only the gated cells produce (or that is shown only when the data are loaded);
# every phrase appears in normal mode. Matching is case-insensitive.
COMPARISON_ONLY = (
    "the committed comparison",
    "from the committed comparison",
    "tuned for it on a small grid",
    "the comparison below measures",
    "pairs in the comparison",
    "committed seed-0",
    "at the rule's own rate",
    "does a better reward model give ppo a lower gap",
)
# The caveat that goes with every comparison figure: shown whenever comparison figures are
# drawn (the committed ones, or a live run's), absent when none are.
CAVEAT = ("do not rank the algorithms", "what carries over")
OLMO_ONLY = ("what dpo learned", "next-token kl", "what this section does",
             "the checkpoints measured here", "where instruct keeps dpo's margin")


def _lower_text(outputs: list[str]) -> str:
    return " ".join(outputs).lower()


def _olmo_without_meta(tmp_path: Path) -> Path:
    for name in ("margins.jsonl", "drift.jsonl"):
        shutil.copy(ROOT / "data" / "olmo" / name, tmp_path / name)
    return tmp_path


@pytest.mark.slow
def test_synthesis_notebook_renders_every_section_on_the_committed_data():
    text = _lower_text(rendered_outputs({}))
    for s in COMPARISON_ONLY + CAVEAT + OLMO_ONLY:
        assert s in text, s
    assert "make synthesis" not in text and "make olmo" not in text


@pytest.mark.slow
@pytest.mark.parametrize("case", ["empty_olmo_dir", "olmo_dir_without_meta"])
def test_synthesis_notebook_without_olmo_data_shows_only_the_notice(case, tmp_path):
    """Without the OLMo files (or without `meta.json`, whose settings the section states),
    the OLMo section is the notice naming `make olmo` and nothing else."""
    root = tmp_path if case == "empty_olmo_dir" else _olmo_without_meta(tmp_path)
    outputs = rendered_outputs({"PROXOPT_OLMO_DIR": str(root)})
    section = _section(outputs, "The OLMo ladder", "References")
    assert len(section) == 1 and "make olmo" in section[0], section
    text = _lower_text(outputs)
    for s in OLMO_ONLY:
        assert s not in text, s
    assert "the committed comparison" in text


@pytest.mark.slow
def test_synthesis_notebook_without_the_comparison_shows_the_notice_and_the_live_cell(tmp_path):
    """Without the precomputed comparison, the notebook names `make synthesis`, keeps the
    live cell, and says nothing about a comparison it does not have; with no figure
    drawn, the figures' caveat is absent too."""
    outputs = rendered_outputs({"PROXOPT_SYNTHESIS_FILE": str(tmp_path / "missing.json")})
    text = _lower_text(outputs)
    assert "make synthesis" in text and "other settings, live" in text
    for s in COMPARISON_ONLY + CAVEAT:
        assert s not in text, s
    assert "what dpo learned" in text


@pytest.mark.slow
def test_synthesis_notebook_live_run_without_the_comparison_shows_the_caveat(tmp_path):
    """Without the precomputed comparison but with the live run drawn, the figures come
    with the caveat (no ranking, what carries over), and still nothing is said about the
    comparison that is not loaded."""
    outputs = rendered_outputs({"PROXOPT_SYNTHESIS_FILE": str(tmp_path / "missing.json")},
                               live=True)
    text = _lower_text(outputs)
    assert "start gap" in text  # the live run's table was drawn
    for s in CAVEAT:
        assert s in text, s
    for s in COMPARISON_ONLY:
        assert s not in text, s


def test_early_notebooks_do_not_use_olmo():
    early = [p for p in NOTEBOOKS if p.name[:3] in {"00_", "01_", "02_", "03_"}]
    for path in early:
        assert olmo_uses(path.read_text()) == [], path.name
