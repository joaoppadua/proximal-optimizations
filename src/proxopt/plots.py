"""Shared figures. Every function builds its own `Figure` and returns it.

Nothing here touches pyplot, so no figure outlives its caller and `plt.show` is never
called. Figures have a transparent background and mid-grey text and axes, which read on
both light and dark notebook themes; every series colour has a contrast of about 3:1 or
better against both a white (#fcfcfb) and a dark (#1a1a19) surface. Colours are never the only cue: curves carry a legend,
trajectory ends carry markers.

Axis notation is fixed: drift is KL(π ‖ π_ref), the gap is β·KL(π ‖ π*), reward is E[r].
"""

import itertools
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Literal

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from torch import Tensor

from proxopt.train import Run

COLORS: dict[str, str] = {
    "ref": "#8a8a8a",  # neutral grey: the thing everything is measured from
    "optimum": "#199e70",  # teal
    "reinforce": "#d55181",  # magenta
    "ppo": "#3987e5",  # blue
    "grpo": "#d95926",  # orange
    "grpo_ppo_like": "#c98500",  # amber: warm like grpo, lighter and yellower
    "dpo": "#9085e9",  # violet
    "ppo_rm": "#e66767",  # red
}

# For free-form labels: none of these is a COLORS value, the first ones are far from
# the ppo blue, and each holds >= 3:1 on both a light and a dark surface.
_CYCLE = ["#2a9d3f", "#b34fb3", "#8f8f1f", "#2a9cb0", "#c07a45"]
_BAND_ALPHA = 0.4
_TARGET_MARKERS = ["D", "s", "^", "v", "P", "X"]

_INK = "#8a8a8a"
_GRID = "#8a8a8a"

DRIFT = "KL(π ‖ π_ref)"
GAP = "β·KL(π ‖ π*)"
REWARD = "E[r]"
_LABELS = {
    "reward": REWARD,
    "kl": DRIFT,
    "kl_rev": "KL(π_ref ‖ π)",
    "objective": "E[r] − β·KL(π ‖ π_ref)",
    "gap": GAP,
}

RunLike = Run | list[Run]


@contextmanager
def _style() -> Iterator[None]:
    rc = {
        "text.color": _INK,
        "axes.labelcolor": _INK,
        "axes.edgecolor": _INK,
        "axes.titlecolor": _INK,
        "xtick.color": _INK,
        "ytick.color": _INK,
        "axes.facecolor": "none",
        "figure.facecolor": "none",
        "savefig.facecolor": "none",
        "axes.grid": True,
        "grid.color": _GRID,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
    }
    with mpl.rc_context(rc):
        yield


def _figure(ncols: int = 1, width: float = 4.6, height: float = 3.6) -> tuple[Figure, list]:
    fig = Figure(figsize=(width * ncols, height), layout="constrained")
    axes = fig.subplots(1, ncols, squeeze=False)[0]
    return fig, list(axes)


@contextmanager
def styled_figure(
    ncols: int = 1, width: float = 4.6, height: float = 3.6
) -> Iterator[tuple[Figure, list]]:
    """A figure with one row of `ncols` axes, each `width` × `height` inches, in this
    module's style. Draw inside the `with` block, which is where the style applies:

        with plots.styled_figure(2) as (fig, axes):
            axes[0].plot(...)

    Notebooks use it for one-off figures that no function here provides, so that they
    look like the shared ones."""
    with _style():
        yield _figure(ncols, width, height)


def _colors(labels: Sequence[str]) -> dict[str, str]:
    """COLORS for known keys; free-form ones take the default cycle in order."""
    free = itertools.cycle(_CYCLE)
    return {k: COLORS[k] if k in COLORS else next(free) for k in labels}


def _as_list(run: RunLike) -> list[Run]:
    return list(run) if isinstance(run, list) else [run]


def _group(name: str, entry: RunLike, x: str) -> tuple[list[Run], np.ndarray]:
    """The runs behind one label and their shared x values. Seeds are only averaged when
    they recorded at exactly the same x values; nothing is truncated or intersected."""
    group = _as_list(entry)
    if not group:
        raise ValueError(f"{name!r}: the list of runs is empty")
    xs = list(getattr(group[0], x))
    for i, run in enumerate(group[1:], start=1):
        if list(getattr(run, x)) != xs:
            raise ValueError(
                f"{name!r}: run {i} recorded different {x} values from run 0, "
                "so the seeds cannot be averaged point by point"
            )
    return group, np.array(xs, dtype=float)


def _stack(name: str, runs: list[Run], getter: Callable[[Run], Sequence[float]]) -> np.ndarray:
    """[n_runs, n_points] array of one per-point quantity."""
    rows = [list(getter(r)) for r in runs]
    if len({len(row) for row in rows}) != 1:
        raise ValueError(f"{name!r}: the runs record different numbers of points")
    return np.array(rows, dtype=float)


def _legend(fig: Figure, ax_with_handles) -> None:
    handles, labels = ax_with_handles.get_legend_handles_labels()
    if len(labels) >= 2:
        fig.legend(handles, labels, loc="outside lower center", ncols=min(len(labels), 4))


def training_curves(
    runs: dict[str, RunLike],
    keys: Sequence[str] = ("reward", "kl", "gap"),
    x: Literal["steps", "feedback"] = "steps",
) -> Figure:
    """One panel per metric. A list of runs is its mean line with a min-to-max band."""
    colors = _colors(list(runs))
    with _style():
        fig, axes = _figure(len(keys), width=4.2)
        for ax, key in zip(axes, keys):
            for name, entry in runs.items():
                group, xs = _group(name, entry, x)
                ys = _stack(name, group, lambda r, key=key: r.metrics[key])
                ax.plot(xs, ys.mean(0), color=colors[name], label=name)
                if len(group) > 1:
                    ax.fill_between(xs, ys.min(0), ys.max(0), color=colors[name], alpha=_BAND_ALPHA, lw=0)
            ax.set_xlabel("gradient steps" if x == "steps" else "units of feedback")
            ax.set_ylabel(_LABELS.get(key, key))
            if key == "gap":
                ax.set_ylim(bottom=0)
        _legend(fig, axes[0])
    return fig


def frontier_plot(
    frontier: list[tuple[float, float]],
    runs: dict[str, RunLike],
    targets: dict[str, tuple[float, float]] | None = None,
    ceiling: float | None = None,
) -> Figure:
    """Reward against drift: the exact frontier, each run's trajectory, target points."""
    colors = _colors(list(runs))
    with _style():
        fig, (ax,) = _figure(width=6.0, height=4.2)
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, color=COLORS["optimum"], ls="--", label="frontier π*_β")
        starts: list[tuple[float, float]] = []
        for name, entry in runs.items():
            group, _ = _group(name, entry, "steps")
            xs = _stack(name, group, lambda r: r.metrics["kl"]).mean(0)
            ys = _stack(name, group, lambda r: r.metrics["reward"]).mean(0)
            ax.plot(xs, ys, color=colors[name], label=name, lw=1.6)
            ax.plot(xs[-1], ys[-1], "o", color=colors[name], ms=8, mec="none")
            starts.append((xs[0], ys[0]))
        if starts:  # every run starts from the reference policy: one marker
            ax.plot(*starts[0], "o", mfc="none", color=COLORS["ref"], ms=7, mew=1.6, label="start")
        markers = itertools.cycle(_TARGET_MARKERS)  # several targets differ by shape, not colour
        for name, (tx, ty) in (targets or {}).items():
            # labelled through the legend, which sits outside the axes and covers nothing
            ax.plot(tx, ty, next(markers), color=COLORS["optimum"], ms=7, mec="none", ls="none",
                    label=name)
        if ceiling is not None:
            ax.axvline(ceiling, color=_INK, ls=":", lw=1.4, label=f"drift ceiling ({ceiling:.2f})")
        ax.set_xlabel(DRIFT)
        ax.set_ylabel(REWARD)
        ax.set_xlim(left=0)
        fig.legend(*ax.get_legend_handles_labels(), loc="outside right center")
    return fig


def ratio_hist(ratios: Tensor, clip: float) -> Figure:
    """Histogram of π_θ/π_old with the clip range marked."""
    r = ratios.detach().flatten().cpu().numpy()
    lo, hi = 1 - clip, 1 + clip
    outside = float(((r < lo) | (r > hi)).mean()) * 100
    left, right = min(r.min(), lo), max(r.max(), hi)
    pad = 0.05 * (right - left)
    with _style():
        fig, (ax,) = _figure(width=5.0, height=3.4)
        ax.hist(r, bins=50, color=COLORS["ppo"], alpha=0.85)
        for v, name, side in ((lo, "1 − ε", 1), (hi, "1 + ε", -1)):
            ax.axvline(v, color=_INK, ls="--", lw=1.4)
            ax.annotate(
                name, (v, 1), xycoords=("data", "axes fraction"), xytext=(4 * side, -12),
                textcoords="offset points", fontsize=9, ha="left" if side > 0 else "right",
            )
        ax.set_xlim(left - pad, right + pad)
        ax.set_xlabel("ratio π_θ / π_old")
        ax.set_ylabel("count")
        ax.set_title(f"{outside:.1f}% of ratios outside [{lo:.2f}, {hi:.2f}]")
    return fig


def distribution_bars(
    dists: dict[str, Tensor], rewards: Tensor, labels: Sequence[str] | None = None
) -> Figure:
    """Probabilities of a few responses under each distribution, and their rewards."""
    n = len(rewards)
    names = list(labels) if labels is not None else [f"y{i}" for i in range(n)]
    colors = _colors(list(dists))
    width = 0.8 / len(dists)
    pos = np.arange(n)
    with _style():
        fig, (ax, ax_r) = _figure(2, width=max(4.6, 0.6 * n * len(dists) / 2 + 3))
        for i, (name, p) in enumerate(dists.items()):
            ax.bar(pos + (i - (len(dists) - 1) / 2) * width, p.detach().numpy(), width,
                   color=colors[name], label=name)
        ax_r.bar(pos, rewards.detach().numpy(), 0.6, color=_INK, alpha=0.7)
        for a, ylabel in ((ax, "probability"), (ax_r, "reward r")):
            a.set_xticks(pos, names, rotation=45 if n > 6 else 0)
            a.set_ylabel(ylabel)
            a.set_xlabel("response")
            a.grid(axis="x", visible=False)
        _legend(fig, ax)
    return fig


def scatter_true_vs_estimate(
    estimate: Tensor,
    true: Tensor,
    mask: Tensor | None = None,
    mask_labels: tuple[str, str] = ("covered", "not covered"),
    xlabel: str = "",
    ylabel: str = "",
) -> Figure:
    """Truth on x, estimate on y, with the identity line; two colours when `mask` is given."""
    e = estimate.detach().flatten().numpy()
    t = true.detach().flatten().numpy()
    with _style():
        fig, (ax,) = _figure(width=4.6, height=4.2)
        if mask is None:
            ax.scatter(t, e, s=14, alpha=0.6, color=_CYCLE[1], edgecolors="none")
        else:
            m = mask.detach().flatten().numpy().astype(bool)
            # the masked group (usually the few covered points) is drawn on top
            ax.scatter(t[m], e[m], s=14, alpha=0.6, color=_CYCLE[1], edgecolors="none",
                       label=mask_labels[0], zorder=3)
            ax.scatter(t[~m], e[~m], s=14, alpha=0.6, color=_CYCLE[0], edgecolors="none",
                       label=mask_labels[1])
            ax.legend(loc="upper left")
        both = np.concatenate([e, t])
        lo, hi = both.min(), both.max()
        ax.plot([lo, hi], [lo, hi], color=_INK, ls="--", lw=1.2, zorder=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    return fig


def feedback_points(
    lines: dict[str, RunLike], points: dict[str, list[tuple[int, float]]]
) -> Figure:
    """Gap against units of feedback: trajectories for sampling methods, markers for
    methods trained on a fixed dataset. Where a name has several points at one x (one per
    seed), it gets one marker at their mean with a min-to-max bar. Log x-axis."""
    colors = _colors(list(lines) + list(points))
    with _style():
        fig, (ax,) = _figure(width=6.0, height=4.0)
        for name, entry in lines.items():
            group, xs = _group(name, entry, "feedback")
            ys = _stack(name, group, lambda r: r.metrics["gap"])
            keep = xs > 0  # the log axis has no zero
            ax.plot(xs[keep], ys.mean(0)[keep], color=colors[name], label=name)
            if len(group) > 1:
                ax.fill_between(xs[keep], ys.min(0)[keep], ys.max(0)[keep],
                                color=colors[name], alpha=_BAND_ALPHA, lw=0)
        for name, pts in points.items():
            by_x: dict[float, list[float]] = {}
            for px, py in pts:
                by_x.setdefault(px, []).append(py)
            px = sorted(by_x)
            if all(len(v) == 1 for v in by_x.values()):
                py = [by_x[v][0] for v in px]
                ax.plot(px, py, "o", color=colors[name], ms=8, mec="none", label=name)
                continue
            ys = [np.array(by_x[v], dtype=float) for v in px]
            mean = np.array([y.mean() for y in ys])
            err = [mean - [y.min() for y in ys], [y.max() for y in ys] - mean]
            ax.errorbar(px, mean, yerr=err, fmt="o", color=colors[name], ms=8, mec="none",
                        elinewidth=1.5, capsize=0, label=name)
        ax.set_xscale("log")
        ax.set_xlabel("units of feedback")
        ax.set_ylabel(f"final {GAP}")
        ax.set_ylim(bottom=0)
        fig.legend(*ax.get_legend_handles_labels(), loc="outside right center")
    return fig


def extras_curves(run: Run, keys: Sequence[str], ylabel: str = "") -> Figure:
    """One panel with `run.extras[key]` against steps for each key that is present."""
    present = [k for k in keys if k in run.extras]
    if not present:
        raise ValueError(f"none of the keys {list(keys)} is in run.extras")
    colors = _colors(present)
    with _style():
        fig, (ax,) = _figure(width=5.0, height=3.6)
        for key in present:
            ax.plot(run.steps, run.extras[key], color=colors[key], label=key)
        ax.set_xlabel("gradient steps")
        ax.set_ylabel(ylabel)
        _legend(fig, ax)
    return fig
