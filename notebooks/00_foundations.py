# marimo cells end in a bare `return`; that is the file format, not dead code.
# ruff: noqa: PLR1711
import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch

    from proxopt import exact, plots
    from proxopt.algos import reinforce
    from proxopt.algos.reinforce import ReinforceConfig, reinforce_loss
    from proxopt.nbtools import reveal, source_md
    from proxopt.reference import load_reference
    from proxopt.refs import cite
    from proxopt.train import scorer
    from proxopt.world import DEFAULT_WORLD, World

    return (
        DEFAULT_WORLD,
        ReinforceConfig,
        World,
        cite,
        exact,
        load_reference,
        mo,
        np,
        plots,
        reinforce,
        reinforce_loss,
        reveal,
        scorer,
        source_md,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 00 · Foundations: the objective, its exact optimum, and REINFORCE

    ## Where we are

    This is the first of five notebooks on three algorithms used to post-train language
    models: **PPO** (notebook 01), **GRPO** (notebook 02, framed as a diff against PPO) and
    **DPO** (notebook 03, the one real conceptual jump). Notebook 04 puts all three on one
    world against one yardstick. This notebook fixes what every later one reuses: the toy
    world, the notation, the objective all three are measured against, its exact optimum,
    and the plainest algorithm that climbs it, REINFORCE.

    | symbol | meaning |
    |---|---|
    | $x$ | prompt |
    | $y = (y_1, \dots, y_T)$ | response; $y_t$ its $t$-th token |
    | $\pi_\theta(y \mid x)$ | the policy being trained, $\prod_t \pi_\theta(y_t \mid x, y_{<t})$ |
    | $\pi_{\text{ref}}$ | the reference policy (stand-in for the SFT model) |
    | $r(x, y)$ | reward of a whole response |
    | $\beta$ | strength of the KL penalty |
    | $A$ | advantage |
    | $\tau$ | preference noise (notebook 03) |
    | $G$ | group size (notebook 02) |
    | $\varepsilon$ | clip range (notebooks 01, 02) |

    The notation never changes from here on.
    """)
    return


@app.cell
def _(DEFAULT_WORLD, cite, mo):
    _w = DEFAULT_WORLD
    _before = mo.md(
        rf"""
        ## Definition

        **The world.** A prompt $x$ is a target number, one of {list(_w.targets)}. A response
        $y$ is {_w.length} tokens, each a digit from 0 to {_w.vocab_size - 1}, so there are
        ${_w.vocab_size}^{_w.length} = {_w.n_responses:,}$ responses per prompt. The reward is a
        programmatic check on the whole response. In the default, *graded* mode,
        $r(x, y) = 1 - |\textstyle\sum_t y_t - \text{{target}}(x)| / {_w.max_sum}$, which lies in
        $[0, 1]$ and is 1 only when the digits sum to the target. (The *verifiable* mode, 1 if
        the sum is exact and 0 otherwise, appears in notebook 02.)

        **The policy** $\pi_\theta$ is a small GRU that emits one token at a time: at step $t$
        it reads $x$ and $y_{{t-1}}$ and outputs $\pi_\theta(y_t \mid x, y_{{<t}})$. So
        $\log \pi_\theta(y \mid x) = \sum_t \log \pi_\theta(y_t \mid x, y_{{<t}})$, exactly as
        in an LLM. Because every response can be enumerated, every quantity below is computed
        exactly, not estimated.

        **The reference policy** $\pi_{{\text{{ref}}}}$ is the same network trained by maximum
        likelihood on a synthetic corpus with a style (each token tends to repeat or increment
        the previous one) that ignores the prompt. It is the stand-in for an SFT model: fluent
        in its own way, mediocre at the task, and every response has non-zero probability. All
        training starts from it.

        **The objective.** For $\beta > 0$,

        $$
        J_\beta(\pi) = \mathbb{{E}}_x \Big[ \mathbb{{E}}_{{y \sim \pi(\cdot \mid x)}}[r(x, y)]
        - \beta \, \mathrm{{KL}}\big(\pi(\cdot \mid x) \,\|\, \pi_{{\text{{ref}}}}(\cdot \mid x)\big) \Big].
        $$

        The KL term is why the trained model stays close to the SFT model: without it the
        maximiser puts all its mass on whatever the reward function happens to score highest,
        including its blind spots. This is the objective of RLHF as set out in:
        """
    )
    _after = mo.md(
        r"""
        **Its exact optimum.** Define $Z_\beta(x) = \sum_y \pi_{\text{ref}}(y \mid x)\, e^{r(x, y)/\beta}$ and

        $$
        \pi^*_\beta(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x)\, \exp\!\big(r(x, y)/\beta\big)}{Z_\beta(x)}.
        $$

        *Derivation.* For one prompt, put the reward inside the log:
        $\mathbb{E}_\pi[r] - \beta\,\mathrm{KL}(\pi \| \pi_{\text{ref}})
        = -\beta \sum_y \pi(y) \log \frac{\pi(y)}{\pi_{\text{ref}}(y)\, e^{r(y)/\beta}}
        = -\beta \sum_y \pi(y) \log \frac{\pi(y)}{\pi^*_\beta(y)} + \beta \log Z_\beta$.
        So the per-prompt objective is $\beta \log Z_\beta(x) - \beta\,\mathrm{KL}(\pi \| \pi^*_\beta)$;
        the KL is $\ge 0$ and is 0 only at $\pi = \pi^*_\beta$, which is therefore the maximiser.

        **Its value.**

        $$
        J_\beta(\pi^*_\beta) = \beta \, \mathbb{E}_x\big[\log Z_\beta(x)\big].
        $$

        *Derivation.* Set $\pi = \pi^*_\beta$ in the per-prompt identity, which makes its KL
        term vanish, and average over $x$.

        **The gap identity**, the yardstick of every notebook:

        $$
        J_\beta(\pi^*_\beta) - J_\beta(\pi) = \beta \, \mathbb{E}_x\Big[\mathrm{KL}\big(\pi(\cdot \mid x) \,\|\, \pi^*_\beta(\cdot \mid x)\big)\Big].
        $$

        *Derivation.* Subtract the per-prompt identity for $\pi$ from the one for $\pi^*_\beta$:
        the $\beta \log Z_\beta(x)$ terms cancel and $\beta\,\mathrm{KL}(\pi \| \pi^*_\beta)$
        is left; average over $x$. (In the figures the gap is written
        $\beta \cdot \mathrm{KL}(\pi \| \pi^*)$, with the mean over prompts understood.)
        """
    )
    mo.vstack([_before, mo.md(cite("ziegler2019", "ouyang2022")), _after])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Intuition

    ### Part 1: one choice

    Start with responses of length 1: eight possible answers to one prompt, whose target
    is 5. Here everything is a table; there is no network. The reference below is written by
    hand. It prefers small digits, so it is clearly not the reward's favourite. Move
    $\beta$ and watch $\pi^*_\beta$ interpolate between $\pi_{\text{ref}}$ (large $\beta$)
    and the reward's argmax (small $\beta$).
    """)
    return


@app.cell
def _(mo, np):
    beta_view = mo.ui.slider(
        steps=[float(f"{v:.3g}") for v in 10 ** np.linspace(-1.7, 0.3, 41)],
        value=0.1,
        label=r"$\beta$ (log scale)",
        show_value=True,
    )
    return (beta_view,)


@app.cell
def _(World, beta_view, exact, mo, plots, torch):
    one_world = World(vocab_size=8, length=1, targets=(5,))
    one_reward = one_world.reward_table()  # [1, 8]
    one_ref = torch.tensor([[0.30, 0.22, 0.15, 0.10, 0.08, 0.05, 0.04, 0.06]])
    _logp_ref = one_ref.log()
    _logp_opt = exact.optimum(_logp_ref, one_reward, beta_view.value)
    _ev = exact.evaluate(_logp_opt, _logp_ref, one_reward, beta_view.value)
    mo.vstack(
        [
            beta_view,
            plots.distribution_bars(
                {"ref": one_ref[0], "optimum": _logp_opt.exp()[0]},
                one_reward[0],
                labels=[str(i) for i in range(8)],
            ),
            mo.md(
                rf"""
                "ref" is $\pi_{{\text{{ref}}}}$ and "optimum" is $\pi^*_\beta$ at
                $\beta = {beta_view.value}$. Under $\pi^*_\beta$: $\mathbb{{E}}[r] = {_ev['reward']:.3f}$
                (under $\pi_{{\text{{ref}}}}$: {(one_ref * one_reward).sum().item():.3f}),
                drift $\mathrm{{KL}}(\pi^*_\beta \| \pi_{{\text{{ref}}}}) = {_ev['kl']:.3f}$.
                Each bar is the reference bar multiplied by $e^{{r/\beta}}$ and renormalised:
                rewards only ever *reweight* what the reference already does, so a response
                with $\pi_{{\text{{ref}}}}(y) = 0$ would stay at 0 at every $\beta$.
                """
            ),
        ]
    )
    return


@app.cell
def _(DEFAULT_WORLD, load_reference, mo, scorer, torch):
    ref = load_reference(DEFAULT_WORLD)  # outside no_grad: the first call trains it
    with torch.no_grad():
        logp_ref = scorer(ref).all_logp()  # [P, N]
    reward = DEFAULT_WORLD.reward_table()
    prompt_pick = mo.ui.dropdown(
        {str(t): i for i, t in enumerate(DEFAULT_WORLD.targets)},
        value=str(DEFAULT_WORLD.targets[1]),
        label="target of the prompt",
    )
    return logp_ref, prompt_pick, ref, reward


@app.cell
def _(DEFAULT_WORLD, mo, prompt_pick, ref, scorer, torch):
    _g = torch.Generator().manual_seed(0)
    _x = torch.full((10,), prompt_pick.value)
    _y = ref.sample(_x, _g)
    _r = DEFAULT_WORLD.reward(_x, _y)
    with torch.no_grad():
        _lp = scorer(ref).token_logp(_x, _y).sum(1)
    _rows = "\n".join(
        f"| {' '.join(str(int(t)) for t in _y[i])} | {int(_y[i].sum())} | {_r[i]:.3f} "
        f"| {_lp[i]:.2f} |"
        for i in range(10)
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ### Part 2: five tokens

                Now the default world. Ten responses sampled from $\pi_{\text{ref}}$ for one
                prompt (sampling seed 0). Note the style: runs and increments, whatever the
                target.
                """
            ),
            prompt_pick,
            mo.md(
                "| response $y$ | sum | $r(x, y)$ | $\\log \\pi_{\\text{ref}}(y \\mid x)$ |\n"
                "|---|---|---|---|\n" + _rows
            ),
        ]
    )
    return


@app.cell
def _(beta_view, exact, logp_ref, mo, np, plots, reward):
    frontier_betas = [float(b) for b in np.logspace(-2, 1, 40)]
    frontier = exact.frontier(logp_ref, reward, frontier_betas)
    ceiling = exact.drift_ceiling(logp_ref, reward).item()
    _opt = exact.optimum(logp_ref, reward, beta_view.value)
    _point = (exact.kl(_opt, logp_ref).item(), exact.expected_reward(_opt, reward).item())
    mo.vstack(
        [
            mo.md(
                r"""
                The **reward–drift frontier**: each point is
                $\big(\mathrm{KL}(\pi^*_\beta \| \pi_{\text{ref}}),\ \mathbb{E}_{\pi^*_\beta}[r]\big)$
                for one $\beta$ (40 values from 0.01 to 10, log-spaced), averaged over prompts.
                No policy lies above it: at a given drift, $\pi^*_\beta$ has the most reward.
                The diamond is $\pi^*_\beta$ at the slider's $\beta$ (the same slider as in
                part 1). The dotted line is the **drift ceiling**: the largest drift any
                optimum can have, reached as $\beta \to 0$, when $\pi^*_\beta$ becomes
                $\pi_{\text{ref}}$ restricted to the responses of maximal reward, so its drift
                is $-\mathbb{E}_x \log \pi_{\text{ref}}(\text{best responses} \mid x)$.
                """
            ),
            beta_view,
            plots.frontier_plot(
                frontier,
                {},
                targets={f"π*_β at β = {beta_view.value}": _point},
                ceiling=ceiling,
            ),
            mo.md(
                f"At β = {beta_view.value}: drift {_point[0]:.3f}, expected reward "
                f"{_point[1]:.3f}; π_ref itself sits at (0, "
                f"{exact.expected_reward(logp_ref, reward).item():.3f})."
            ),
        ]
    )
    return ceiling, frontier


@app.cell
def _(cite, mo, reinforce_loss, source_md):
    _equation = mo.md(
        r"""
        ## The code

        **REINFORCE** on $J_\beta$. Since $\nabla_\theta \pi = \pi \nabla_\theta \log \pi$, and
        $\mathbb{E}_{y \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(y \mid x)] = 0$,

        $$
        \nabla_\theta J_\beta = \mathbb{E}_x \mathbb{E}_{y \sim \pi_\theta}\Big[ R(x, y)\, \nabla_\theta \log \pi_\theta(y \mid x) \Big],
        \qquad
        R(x, y) = r(x, y) - \beta \sum_t \big( \log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_{\text{ref}}(y_t \mid x, y_{<t}) \big),
        $$

        with $\nabla_\theta \log \pi_\theta(y \mid x) = \sum_t \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t})$.
        (Differentiating the KL term also gives $-\beta\, \mathbb{E}[\nabla_\theta \log \pi_\theta]$,
        which is 0.) $R$ is the KL-regularised return, treated as a constant (detached). With
        a batch of $B$ prompts drawn uniformly and one sample $y_i \sim \pi_\theta(\cdot \mid x_i)$
        each, the loss whose gradient is the estimate is

        $$
        L(\theta) = -\frac{1}{B} \sum_{i=1}^{B} \big( R_i - b \big) \sum_t \log \pi_\theta(y_{i,t} \mid x_i, y_{i,<t}).
        $$
        """
    )
    _baseline = mo.md(
        r"""
        **The baseline $b$.** With $b = 0$ ("none") the estimate is unbiased for
        $\nabla_\theta J_\beta$. With $b = \frac{1}{B} \sum_j R_j$ ("mean") the baseline
        contains $R_i$ itself, so the expected update is
        $(1 - 1/B)\, \nabla_\theta J_\beta$: the same direction scaled down (0.94 at $B = 16$,
        0.98 at $B = 64$), absorbed by the learning rate; it is not the baseline of Williams'
        theorem, which must not depend on the sample's own output (a leave-one-out mean would
        be). Its point is variance: it centres the returns, so a sample is pushed up only if
        it beat its batch.

        The REINFORCE estimator is due to:
        """
    )
    _after = mo.md(
        r"""
        The KL-regularised return $R$ is this project's application of it to $J_\beta$, not
        Williams' own setting.
        """
    )
    mo.vstack(
        [
            _equation,
            mo.md(source_md(reinforce_loss)),
            _baseline,
            mo.md(cite("williams1992")),
            _after,
        ]
    )
    return


@app.cell
def _(mo):
    beta_run = mo.ui.slider(
        steps=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0], value=0.1, label="β", show_value=True
    )
    batch_run = mo.ui.slider(
        steps=[16, 32, 64, 128, 256], value=64, label="batch size B", show_value=True
    )
    steps_run = mo.ui.slider(
        steps=[100, 200, 400, 800], value=200, label="gradient steps", show_value=True
    )
    baseline_run = mo.ui.radio(["none", "mean"], value="mean", label="baseline b", inline=True)
    seed_run = mo.ui.number(start=0, stop=99, step=1, value=0, label="seed")
    return baseline_run, batch_run, beta_run, seed_run, steps_run


@app.cell
def _(DEFAULT_WORLD, load_reference, mo, reinforce):
    @mo.cache
    def train_reinforce(cfg):
        """One REINFORCE run on the default world, cached by its (frozen) config."""
        return reinforce.train(load_reference(DEFAULT_WORLD), DEFAULT_WORLD, cfg)

    return (train_reinforce,)


@app.cell
def _(
    ReinforceConfig,
    baseline_run,
    batch_run,
    beta_run,
    mo,
    seed_run,
    steps_run,
    train_reinforce,
):
    run_cfg = ReinforceConfig(
        beta=beta_run.value,
        batch_size=batch_run.value,
        steps=steps_run.value,
        baseline=baseline_run.value,
        seed=int(seed_run.value),
    )
    run = train_reinforce(run_cfg)
    mo.vstack(
        [
            mo.md(
                r"""
                ## Simulation

                REINFORCE from $\pi_{\text{ref}}$, Adam, learning rate 3e-3. Each control
                change trains one run (about a second at the default; cached, so going back
                to a setting is instant). Reward, drift and gap are exact, computed over all
                responses every 10 steps.
                """
            ),
            mo.hstack([beta_run, batch_run, steps_run], justify="start"),
            mo.hstack([baseline_run, seed_run], justify="start"),
        ]
    )
    return run, run_cfg


@app.cell
def _(ceiling, exact, frontier, logp_ref, mo, plots, reward, run, run_cfg):
    _opt = exact.optimum(logp_ref, reward, run_cfg.beta)
    _target = (exact.kl(_opt, logp_ref).item(), exact.expected_reward(_opt, reward).item())
    _m = run.metrics
    mo.vstack(
        [
            plots.training_curves({"reinforce": run}),
            plots.frontier_plot(
                frontier,
                {"reinforce": run},
                targets={f"π*_β, β = {run_cfg.beta}": _target},
                ceiling=ceiling,
            ),
            mo.md(
                f"After {run.steps[-1]} steps ({run.feedback[-1]:,} reward calls): "
                f"E[r] = {_m['reward'][-1]:.3f} (target {_target[1]:.3f}), "
                f"KL(π ‖ π_ref) = {_m['kl'][-1]:.3f} (target {_target[0]:.3f}), "
                f"gap = {_m['gap'][-1]:.4f} (start {_m['gap'][0]:.4f})."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What carries over to real LLMs, and what does not.** The *shape* carries over: the
    loss is a sum of per-token log-probabilities weighted by a sequence-level return,
    the KL to the reference is a per-token log-ratio summed along the response, and the
    baseline matters for the same reason. What does not: here the response space is
    enumerable, so the expected reward, the drift and the gap to $\pi^*_\beta$ are computed
    exactly. For an LLM, the reward of a sampled response can be computed, but expected
    reward and drift can only be estimated from samples, and the optimum and the gap are
    unknown. The numbers (how many steps, what batch size, how much spread across seeds)
    are facts about this toy world, not about LLMs. So is the explanation offered for item 1
    below (without a baseline every sample is pushed up). It relies on a reward that lies in
    $[0, 1]$ and is always positive, and reward-model scores need not have either property.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to try

    Make a prediction before opening each answer. The answers report what was measured
    on the machine this notebook was written on (seeds 0, 1, 2; default learning rate).
    """)
    return


@app.cell
def _(mo, reveal):
    seeds_baseline = mo.ui.run_button(label="Run 3 seeds × 2 baselines at B = 16")
    mo.vstack(
        [
            reveal(
                "**1. Baseline.** Set the batch size to 16 and compare baseline *none* with "
                "*mean*, over seeds 0, 1 and 2 each (or press the button below). Which one "
                "spreads more across seeds, and does each reach the target?",
                "Measured over seeds 0–2, 200 steps, B = 16, β = 0.1. **Mean baseline:** the "
                "gap falls from 0.102 to 0.019–0.023, drift ends at 0.73–0.77 (target 0.64), "
                "and the seeds stay within 0.003 of each other on average over training "
                "(0.004 at the end). **No baseline:** the spread is about ten times wider "
                "(0.027 on average, 0.045 at the end), and, more than the theory's prediction "
                "of noise alone, the gap does **not** fall: it ends at 0.100–0.145, no better "
                "than at the start. Expected reward still rises (0.745 to 0.86–0.87), but drift "
                "overshoots to 1.19–1.54, about twice the optimum's 0.64. A likely reading "
                "(not separately measured): every reward here is positive, about 0.75 on "
                "average, so without a baseline every sampled response is pushed up, and with "
                "16 samples the −β·log-ratio part of the return is too noisy to pull the "
                "policy back. The two estimates point the same way in expectation; they "
                "differ in variance, and at this batch size and learning rate that decides "
                "the outcome.",
            ),
            seeds_baseline,
        ]
    )
    return (seeds_baseline,)


@app.cell
def _(ReinforceConfig, mo, plots, seeds_baseline, train_reinforce):
    mo.stop(not seeds_baseline.value, mo.md("*Press the button to train the six runs (about 3 s).*"))
    _groups = {
        f"baseline {_b}": [
            train_reinforce(ReinforceConfig(batch_size=16, baseline=_b, seed=_s))
            for _s in (0, 1, 2)
        ]
        for _b in ("none", "mean")
    }
    plots.training_curves(_groups)
    return


@app.cell
def _(reveal):
    reveal(
        "**2. Small β.** Set β to 0.02, in part 2 of the intuition and in the simulation. "
        "Where does the optimum move, and what happens to drift?",
        "Measured. The optimum concentrates on the highest-reward responses: π*_0.02 puts 61% "
        "of its mass on responses of maximal reward (π*_0.1: 15%; π_ref: 3.3%), with expected "
        "reward 0.985 (π*_0.1: 0.912). Its drift is 2.21, against 0.64 at β = 0.1, and "
        "approaches the drift ceiling of 3.44 as β falls further. In the simulation (B = 64, "
        "200 steps, seeds 0–2) the gap falls from 0.196 to 0.017, the run reaches E[r] = "
        "0.970–0.972 and drift 2.31–2.35: slightly *more* drift than the optimum for less "
        "reward, so the run ends below the frontier, not on it. A small β buys reward with "
        "drift.",
    )
    return


@app.cell
def _(mo, reveal):
    seeds_batch = mo.ui.run_button(label="Run 3 seeds × 2 settings")
    mo.vstack(
        [
            reveal(
                "**3. Same feedback, smaller batches.** Compare B = 64 for 200 steps with "
                "B = 32 for 400 steps: the same 12,800 reward calls (or press the button, "
                "which plots both against units of feedback). Which path is noisier, and which "
                "ends closer to the optimum?",
                "Measured over seeds 0–2. B = 64 is evaluated every 640 reward calls and "
                "B = 32 every 320; the comparisons below use the shared grid of every 640 "
                "calls, with the gap averaged over seeds unless a range is given. "
                "**A slightly faster start:** B = 32 is ahead at 640, 1,280 and 1,920 calls "
                "(0.031, 0.025, 0.021 against 0.036, 0.027, 0.022). On its own finer grid its "
                "mean gap is below 0.02 at 2,240 calls; B = 64 is below 0.02 by 2,560, its "
                "first evaluation after 1,920. **A slightly worse finish:** from 2,560 calls "
                "on, B = 64 is ahead, and at 12,800 calls the gaps are 0.0105–0.0112 (B = 64) "
                "against 0.0127–0.0129 (B = 32). **Noise:** a curve's wobble is measured as "
                "its distance from a 3-point moving average, averaged over seeds and points. "
                "After the initial drop (from 1,920 calls on) B = 32 wobbles 1.6 times as "
                "much as B = 64, but only 1.2 times as much if the drop is included (from 640 "
                "calls on). The seed-to-seed range (max − min), averaged over the grid from "
                "640 calls on, is 0.0018 for B = 32 and 0.0011 for B = 64. At the last point "
                "this reverses: 0.0002 for B = 32 against 0.0007. In short, more and smaller "
                "steps on the same budget give noisier estimates, a slightly faster start "
                "and a slightly worse finish.",
            ),
            seeds_batch,
        ]
    )
    return (seeds_batch,)


@app.cell
def _(ReinforceConfig, mo, plots, seeds_batch, train_reinforce):
    mo.stop(not seeds_batch.value, mo.md("*Press the button to train the six runs (about 6 s).*"))
    _groups = {
        "B = 64, 200 steps": [train_reinforce(ReinforceConfig(seed=_s)) for _s in (0, 1, 2)],
        "B = 32, 400 steps": [
            train_reinforce(ReinforceConfig(batch_size=32, steps=400, seed=_s))
            for _s in (0, 1, 2)
        ],
    }
    plots.training_curves(_groups, x="feedback")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Two weaknesses** come out of these experiments, and each later notebook answers
    one. First, **noise**: a single sampled return per response is a high-variance signal,
    and how well a baseline tames it decides whether the run reaches the optimum at all.
    Second, **each batch of samples is used for a single gradient step** and then thrown
    away, although sampling (for an LLM, generation) is the expensive part.

    ## Where it went next

    PPO (notebook 01) reuses each batch for several epochs and keeps the policy from
    moving too far on stale samples with a clipped ratio; it replaces the batch-mean
    baseline with a learned critic and per-token advantages. GRPO (notebook 02) drops the
    critic again and uses the mean of a group of samples for the same prompt as the
    baseline, standardised by the group's spread. DPO (notebook 03) uses the closed form of
    $\pi^*_\beta$ derived above to do without on-policy sampling and without an explicit
    reward model. It still needs data: pairs of sampled responses, each labelled with which
    one is preferred, and it fits the policy to those pairs directly.
    """)
    return


@app.cell
def _(cite, mo):
    mo.md("## References\n\n" + cite("williams1992", "ziegler2019", "ouyang2022"))
    return


if __name__ == "__main__":
    app.run()
