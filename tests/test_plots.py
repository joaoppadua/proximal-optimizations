import warnings

import matplotlib

matplotlib.use("Agg")

import pytest
import torch
from matplotlib.figure import Figure

from proxopt import nbtools, plots
from proxopt.plots import (
    COLORS,
    distribution_bars,
    extras_curves,
    feedback_points,
    frontier_plot,
    ratio_hist,
    scatter_true_vs_estimate,
    training_curves,
)
from proxopt.train import Run


def fake_run(offset: float = 0.0, extras: dict | None = None) -> Run:
    n = 10
    return Run(
        policy=None,
        config="fake",
        target_beta=0.1,
        steps=list(range(0, 100, 10)),
        feedback=[64 * i for i in range(n)],
        metrics={
            "reward": [0.1 * i + offset for i in range(n)],
            "kl": [0.05 * i for i in range(n)],
            "kl_rev": [0.04 * i for i in range(n)],
            "objective": [0.05 * i + offset for i in range(n)],
            "gap": [1.0 / (i + 1) + offset for i in range(n)],
        },
        extras=extras or {},
    )


@pytest.fixture(autouse=True)
def no_figure_leak():
    yield
    import matplotlib.pyplot as plt

    assert plt.get_fignums() == []


@pytest.fixture(autouse=True)
def errors_on_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def test_colors_keys():
    assert set(COLORS) == {
        "ref", "optimum", "reinforce", "ppo", "grpo", "grpo_ppo_like", "dpo", "ppo_rm"
    }
    assert len(set(COLORS.values())) == len(COLORS)


def test_training_curves_panels_and_colors():
    run, run2 = fake_run(), fake_run(0.2)
    fig = training_curves({"ppo": run, "grpo": [run, run2]})
    assert isinstance(fig, Figure) and len(fig.axes) == 3
    assert len(training_curves({"ppo": run}, keys=("gap",)).axes) == 1
    ax = fig.axes[0]
    assert {line.get_color() for line in ax.get_lines()} >= {COLORS["ppo"], COLORS["grpo"]}


def test_training_curves_free_labels_avoid_palette_and_feedback_axis():
    fig = training_curves({f"β = {b}": fake_run(b) for b in range(8)}, x="feedback")
    colors = {line.get_color() for line in fig.axes[0].get_lines()}
    assert not colors & set(COLORS.values())
    assert len(colors) >= 6


def test_training_curves_works_after_json_roundtrip():
    run = Run.from_json_dict(fake_run().to_json_dict())
    assert isinstance(training_curves({"ppo": run}), Figure)


def test_frontier_plot_with_ceiling_and_targets():
    frontier = [(0.1 * i, 1 - 1 / (i + 1)) for i in range(1, 8)]
    fig = frontier_plot(
        frontier, {"ppo": fake_run()}, targets={"π*": (0.3, 0.5)}, ceiling=1.5
    )
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert any(list(line.get_xdata()) == [1.5, 1.5] or set(line.get_xdata()) == {1.5}
               for line in ax.get_lines())
    assert "ceiling" in " ".join(t.get_text() for t in fig.legends[0].get_texts()).lower()
    assert isinstance(frontier_plot(frontier, {"ppo": [fake_run(), fake_run(0.1)]}), Figure)


def test_ratio_hist_title_has_percent():
    fig = ratio_hist(torch.rand(500) + 0.5, 0.2)
    assert isinstance(fig, Figure) and "%" in fig.axes[0].get_title()


def test_distribution_bars():
    dists = {"ref": torch.tensor([0.5, 0.3, 0.2]), "ppo": torch.tensor([0.1, 0.3, 0.6])}
    fig = distribution_bars(dists, torch.tensor([0.0, 0.5, 1.0]), labels=["a", "b", "c"])
    assert isinstance(fig, Figure) and len(fig.axes) == 2
    assert isinstance(distribution_bars(dists, torch.tensor([0.0, 0.5, 1.0])), Figure)


def test_scatter_true_vs_estimate():
    est, true = torch.randn(50), torch.randn(50)
    assert isinstance(scatter_true_vs_estimate(est, true), Figure)
    mask = torch.rand(50) > 0.5
    fig = scatter_true_vs_estimate(est, true, mask, xlabel="x", ylabel="y")
    assert isinstance(fig, Figure)
    texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "covered" in texts and "not covered" in texts


def test_feedback_points():
    fig = feedback_points(
        {"ppo": fake_run(), "grpo": [fake_run(), fake_run(0.1)]},
        {"dpo": [(100, 0.5), (1000, 0.2)]},
    )
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_xscale() == "log"


def test_extras_curves():
    run = fake_run(extras={"train_loss": [1.0 / (i + 1) for i in range(10)],
                           "holdout_loss": [1.0 / (i + 1) + 0.1 for i in range(10)]})
    fig = extras_curves(run, ["train_loss", "holdout_loss", "missing"], ylabel="loss")
    assert isinstance(fig, Figure) and len(fig.axes) == 1
    assert len(fig.axes[0].get_lines()) == 2
    assert len(extras_curves(run, ["train_loss", "missing"]).axes[0].get_lines()) == 1
    with pytest.raises(ValueError, match="nope.*also_nope"):
        extras_curves(run, ["nope", "also_nope"])


def test_source_md():
    md = nbtools.source_md(plots.training_curves)
    assert md.startswith("```python") and "def training_curves" in md
    assert md.rstrip().endswith("```")


def test_reveal_contains_question_and_answer():
    html = nbtools.reveal("What is q?", "The answer is a.")
    assert "What is q?" in html.text
    assert "The answer is a." in html.text
