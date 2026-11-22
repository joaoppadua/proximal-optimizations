import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
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
    assert len(colors) == len(plots._CYCLE)


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


# --- multi-seed alignment and aggregation ---------------------------------------


def seed_runs() -> list[Run]:
    return [fake_run(0.0), fake_run(0.3), fake_run(-0.2)]


def _band_edges(ax) -> dict[float, tuple[float, float]]:
    """x -> (lower, upper) read back from the first fill_between polygon."""
    verts = ax.collections[0].get_paths()[0].vertices
    edges: dict[float, tuple[float, float]] = {}
    for x in sorted(set(np.round(verts[:, 0], 9))):
        ys = verts[np.isclose(verts[:, 0], x), 1]
        edges[x] = (ys.min(), ys.max())
    return edges


def test_training_curves_mean_and_band_match_per_step_stats():
    runs = seed_runs()
    fig = training_curves({"grpo": runs}, keys=("reward", "gap"))
    for ax, key in zip(fig.axes, ("reward", "gap")):
        ys = np.array([r.metrics[key] for r in runs])
        np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), ys.mean(0))
        np.testing.assert_allclose(ax.get_lines()[0].get_xdata(), runs[0].steps)
        edges = _band_edges(ax)
        np.testing.assert_allclose([e[0] for e in edges.values()], ys.min(0))
        np.testing.assert_allclose([e[1] for e in edges.values()], ys.max(0))


def test_training_curves_feedback_axis_uses_feedback():
    runs = seed_runs()
    fig = training_curves({"grpo": runs}, keys=("reward",), x="feedback")
    np.testing.assert_allclose(fig.axes[0].get_lines()[0].get_xdata(), runs[0].feedback)


def test_frontier_plot_mean_and_band_both_coordinates():
    runs = seed_runs()
    fig = frontier_plot([(0.1, 0.5), (0.5, 0.9)], {"grpo": runs})
    ax = fig.axes[0]
    trajectory = next(ln for ln in ax.get_lines() if ln.get_label() == "grpo")
    np.testing.assert_allclose(trajectory.get_xdata(), np.mean([r.metrics["kl"] for r in runs], 0))
    np.testing.assert_allclose(
        trajectory.get_ydata(), np.mean([r.metrics["reward"] for r in runs], 0)
    )


def test_frontier_plot_single_start_marker_with_several_runs():
    fig = frontier_plot([(0.1, 0.5), (0.5, 0.9)], {"a": fake_run(), "b": fake_run(0.1)})
    hollow = [ln for ln in fig.axes[0].get_lines() if ln.get_marker() == "o"
              and ln.get_markerfacecolor() == "none"]
    assert len(hollow) == 1


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(lambda: [fake_run(), truncated(fake_run())], id="unequal-length"),
        pytest.param(lambda: [fake_run(), shifted(fake_run())], id="same-length-other-grid"),
        pytest.param(list, id="empty"),
    ],
)
@pytest.mark.parametrize("which", ["training_curves", "frontier_plot", "feedback_points"])
def test_multi_seed_misalignment_raises(bad, which):
    group = bad()
    if which == "training_curves":
        call = lambda: training_curves({"grpo": group})
    elif which == "frontier_plot":
        call = lambda: frontier_plot([(0.1, 0.5)], {"grpo": group})
    else:
        call = lambda: feedback_points({"grpo": group}, {})
    with pytest.raises(ValueError, match="grpo"):
        call()


def test_misalignment_message_names_first_differing_run():
    good, other = fake_run(), fake_run(0.1)
    bad = shifted(fake_run())
    with pytest.raises(ValueError, match=r"run 2"):
        training_curves({"grpo": [good, other, bad]})


def test_feedback_axis_checks_feedback_not_steps():
    a, b = fake_run(), fake_run(0.1)
    b.feedback = [f + 1 for f in b.feedback]
    training_curves({"grpo": [a, b]})  # same steps: fine on the steps axis
    with pytest.raises(ValueError, match="feedback"):
        training_curves({"grpo": [a, b]}, x="feedback")


def test_many_free_labels_cycle_without_error():
    fig = training_curves({f"run {i}": fake_run(i) for i in range(15)}, keys=("reward",))
    assert len(fig.axes[0].get_lines()) == 15


def test_ratio_hist_range_covers_data_and_clip_lines():
    r = torch.cat([torch.full((50,), 0.3), torch.full((50,), 1.9)])
    ax = ratio_hist(r, 0.2).axes[0]
    lo, hi = ax.get_xlim()
    assert lo <= 0.3 and hi >= 1.9
    lo, hi = ratio_hist(torch.full((10,), 1.0), 0.2).axes[0].get_xlim()
    assert lo <= 0.8 and hi >= 1.2


def test_free_label_cycle_avoids_blue_first():
    assert plots._CYCLE[0] != COLORS["ppo"]
    assert not set(plots._CYCLE) & set(COLORS.values())


def truncated(run: Run) -> Run:
    run.steps, run.feedback = run.steps[:-1], run.feedback[:-1]
    run.metrics = {k: v[:-1] for k, v in run.metrics.items()}
    return run


def shifted(run: Run) -> Run:
    run.steps = [s + 1 for s in run.steps]
    run.feedback = [f + 1 for f in run.feedback]
    return run


def test_styled_figure_applies_the_module_style_inside_the_block():
    with plots.styled_figure(2, width=3.0, height=2.0) as (fig, axes):
        assert isinstance(fig, Figure) and len(axes) == 2
        (line,) = axes[0].plot([0, 1], [0, 1])
        label = axes[0].set_xlabel("x")
    assert tuple(fig.get_size_inches()) == (6.0, 2.0)
    assert line.get_linewidth() == 2.0  # the module's lines.linewidth, not matplotlib's 1.5
    assert matplotlib.colors.to_hex(label.get_color()) == "#8a8a8a"
    assert axes[0].get_facecolor()[3] == 0.0  # transparent background
    assert not axes[0].spines["top"].get_visible()
