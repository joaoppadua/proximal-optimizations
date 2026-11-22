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

    from proxopt import exact, olmo, plots, prefs, synthesis
    from proxopt.nbtools import reveal, source_md
    from proxopt.reference import load_reference
    from proxopt.refs import REFS, cite
    from proxopt.train import scorer
    from proxopt.world import DEFAULT_WORLD

    return (
        DEFAULT_WORLD,
        REFS,
        cite,
        exact,
        load_reference,
        mo,
        np,
        olmo,
        plots,
        prefs,
        reveal,
        scorer,
        source_md,
        synthesis,
        torch,
    )


@app.cell
def _(DEFAULT_WORLD, exact, load_reference, np, scorer, synthesis, torch):
    syn = synthesis.load()  # None when data/synthesis.json (or $PROXOPT_SYNTHESIS_FILE) is missing
    ref = load_reference(DEFAULT_WORLD)  # outside no_grad: the first call trains it
    with torch.no_grad():
        logp_ref = scorer(ref).all_logp()  # [P, N]
    reward = DEFAULT_WORLD.reward_table()  # the graded reward, as in the comparison
    frontier = exact.frontier(logp_ref, reward, [float(b) for b in np.logspace(-2, 1, 40)])
    ceiling = exact.drift_ceiling(logp_ref, reward).item()

    def target_point(beta):
        """(drift, expected reward) of π*_β."""
        opt = exact.optimum(logp_ref, reward, beta)
        return exact.kl(opt, logp_ref).item(), exact.expected_reward(opt, reward).item()

    def span(values, fmt=".4f"):
        """min–max of `values`, or one number when both round to the same text."""
        lo, hi = format(min(values), fmt), format(max(values), fmt)
        return lo if lo == hi else f"{lo}–{hi}"

    def rate(x):
        """A learning rate as 1e-4, 3e-4, ..."""
        return f"{x:.0e}".replace("e-0", "e-")

    labels = {
        "ppo": "PPO",
        "grpo": "GRPO as published",
        "grpo_ppo_like": "GRPO, PPO-like",
        "dpo": "DPO (β/τ)",
        "ppo_rm": "PPO on a reward model (β/τ)",
    }
    sampling = ("ppo", "grpo", "grpo_ppo_like")
    return ceiling, frontier, labels, rate, sampling, span, syn, target_point


@app.cell
def _(labels, mo, sampling, synthesis, syn):
    def _cost(name):
        if syn is None:
            return "(run `make synthesis`)"
        _r = syn["runs"][name][0]
        _b = syn["method"]["budget"][name]
        if name in sampling:
            return f"{_r.steps[-1]:,} gradient steps, {_r.feedback[-1]:,} reward calls"
        if name == "dpo":
            return f"{_r.steps[-1]:,} gradient steps on {_b['n_pairs']:,} labelled pairs"
        return (
            f"{_b['reward_model_steps']:,} reward-model steps on {_b['n_pairs']:,} labelled "
            f"pairs, then {_r.steps[-1]:,} PPO steps and {_r.feedback[-1]:,} reward-model calls"
        )

    _rows = {
        "reward": ("the true reward (in RLHF, a reward model)", "the true reward",
                   "the true reward", "none: only which of two responses won",
                   "a reward model fitted to the pairs"),
        "critic": ("a learned $V$", "none (the group mean)", "none (the group mean)", "none",
                   "a learned $V$"),
        "preference data": ("no", "no", "no", "yes, a fixed set", "yes, to fit the reward model"),
        "fresh samples from the policy": ("yes, a batch per iteration", "yes, a group per prompt",
                                          "yes, a group per prompt", "no", "yes, a batch per iteration"),
        "aims at": (r"$\pi^*_\beta$ (notebook 01)",
                    r"no fixed target: it does not maximise $J_\beta$ (notebook 02)",
                    r"$\pi^*_\beta$ is a fixed point (notebook 02)",
                    r"$\pi^*_{\beta_{\text{DPO}}\tau}$, here $\pi^*_\beta$ (notebook 03)",
                    r"$\pi^*_{\beta_{\text{PPO}}\tau}$ if the model recovers $r/\tau$; here $\pi^*_\beta$"),
        "cost in this notebook's runs": tuple(_cost(n) for n in synthesis.NAMES),
    }
    _head = "| | " + " | ".join(labels[n] for n in synthesis.NAMES) + " |\n"
    _head += "|---" * (len(synthesis.NAMES) + 1) + "|\n"
    _table = _head + "\n".join(f"| {k} | " + " | ".join(v) + " |" for k, v in _rows.items())
    _plan = (
        f", each at a learning rate tuned for it on a small grid, over {len(syn['seeds'])} seeds"
        if syn is not None
        else "; the precomputed runs are not loaded (run `make synthesis`), and a live run for "
        "one seed is available below"
    )
    _costs = (" The costs are per run, from the committed comparison; they are budgets, not "
              "results." if syn is not None else "")
    mo.vstack(
        [
            mo.md(rf"""
    # 04 · Synthesis: three algorithms, one world, one target

    ## Where we are

    Notebooks 01 to 03 each built one algorithm on the toy world of notebook 00 and
    measured it against the exact optimum $\pi^*_\beta$. Each was compared with its
    predecessor at that notebook's own defaults. This notebook puts them side by side:
    PPO, GRPO as published (the paper's algorithm, but here with $\beta = 0.1$ and 4 epochs
    per batch, not the paper's 0.04 and single update; notebook 02 explains both and
    measures the paper's settings too), GRPO with both switches PPO-like, DPO, and PPO on a reward model
    fitted to the same pairs DPO uses, all on one world, aimed at (or measured against) one
    target{_plan}. Then it
    looks back at the history, and ends with a reality check on three real checkpoints of
    OLMo 2 1B.
    """),
            mo.md(_table),
            mo.md(
                "PPO-like GRPO is GRPO with the division by the group's standard deviation "
                "off and the KL in the reward (notebook 02's two switches)." + _costs
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Definition

    **One target.** Every gap in this notebook is notebook 00's
    $\text{gap} = \beta\,\mathrm{KL}(\pi \,\|\, \pi^*_\beta)$, with the true graded reward
    and the same $\beta$ for all five runs. What each run *aims at* differs:

    - **PPO** in its RLHF form maximises $J_\beta$, whose maximiser is $\pi^*_\beta$
      (notebook 01).
    - **GRPO as published** does not maximise $J_\beta$ (notebook 02). Its KL estimator has
      the right expected value, but on fixed samples its expected gradient, for one
      next-token distribution, is that of $\mathrm{KL}(\pi_{\text{ref}}(\cdot \mid s_t) \,\|\, \pi_\theta(\cdot \mid s_t))$,
      the other direction; summed over a response it is the gradient of neither
      sequence-level KL. And dividing the advantage by the group's standard deviation
      $\sigma$ (here, as in notebook 02, $\sigma$ is that standard deviation; in the
      reward model's loss under "The code", as in notebook 03, it is the logistic
      function) makes the effective KL weight $\beta\sigma$, different for every group.
      Its gap is measured against $\pi^*_\beta$, but $\pi^*_\beta$ is not its target.
    - **GRPO, PPO-like** (no division, KL in the reward): every advantage is zero at
      $\pi^*_\beta$, so $\pi^*_\beta$ is a fixed point (notebook 02, (c)).
    - **DPO** with parameter $\beta_{\text{DPO}}$ on preferences drawn with noise $\tau$
      aims at $\pi^*_{\beta_{\text{DPO}}\tau}$, given enough pairs covering every response
      (notebook 03, step 7). The target depends on $\beta_{\text{DPO}}$ and $\tau$ only
      through their product; the loss does not.
    - **PPO on a reward model.** A Bradley–Terry reward model fitted to the same pairs
      learns, at best, $r/\tau$ up to a constant per prompt (the argument of notebook 03,
      step 7, applied to a free-standing model). A constant per prompt does not move the
      optimum, so PPO with $\beta_{\text{PPO}}$ on that reward aims at
      $\pi_{\text{ref}}\, e^{(r/\tau)/\beta_{\text{PPO}}}/Z = \pi^*_{\beta_{\text{PPO}}\tau}$.

    So that the two methods that learn from pairs aim at the same point as PPO, both run
    with $\beta/\tau$ in the place of $\beta$:

    $$
    \beta_{\text{DPO}} = \beta_{\text{PPO}} = \frac{\beta}{\tau}
    \quad\Longrightarrow\quad
    \pi^*_{(\beta/\tau)\,\tau} = \pi^*_\beta .
    $$

    Their gaps are then measured against the same $\pi^*_\beta$ as PPO's, with the true
    reward, which neither of them ever sees.
    """)
    return


@app.cell
def _(mo, syn):
    _measured = " (the comparison below measures this)" if syn is not None else ""
    mo.md(rf"""
    ## Intuition

    **What each method spends.** PPO and GRPO spend *reward calls*: every sampled response
    is scored by the reward function, and every batch is fresh. DPO spends *labelled pairs*:
    a fixed set, collected once from $\pi_{{\text{{ref}}}}$ and reused at every step. PPO on a
    reward model acquires the same pairs once; fitting the model draws many minibatches
    from them, and PPO then makes as many reward-model calls as it likes, which cost no
    further labels.

    **Which comparison is fair.** PPO and GRPO here are scored by the true reward, exactly,
    on every sample: information that the two pair-based methods never have. Comparing DPO
    with PPO on the true reward would compare two different amounts of information. The
    fair comparison is DPO against PPO on a reward model fitted to *the same pairs*: both see exactly the same labels, and differ
    only in how they use them.

    **What a gap means for a run that aims elsewhere.** For a run whose target is
    $\pi^*_\beta$, the gap says how far it got. For published GRPO, which has no fixed
    target, the gap says only how far from $\pi^*_\beta$ it happens to be at the end of
    a budget; it can fall and then rise, and how far it gets depends on the learning
    rate{_measured}.
    """)
    return


@app.cell
def _(cite, mo, prefs, source_md, synthesis, syn):
    if syn is not None:
        _b = syn["method"]["budget"]["ppo_rm"]
        _fit = (f"{_b['reward_model_steps']} steps of {_b['reward_model_batch_size']} pairs "
                "in the comparison")
    else:
        _fit = f"`synthesis.run_one` uses {synthesis.RM_STEPS} steps of {synthesis.RM_BATCH} pairs"
    mo.vstack(
        [
            mo.md(
                r"""
                ## The code

                **One config per name.** `synthesis.configs` builds the five runs at their
                modules' default budgets; DPO and PPO-on-reward-model get
                $\beta/\tau$ (the equation above), the others $\beta$:
                """
            ),
            mo.md(source_md(synthesis.configs)),
            mo.md(
                rf"""
                **The reward model.** A small GRU that reads the whole response, fitted to
                the pairs by the Bradley–Terry likelihood (the DPO paper's eq. 1, citing
                Bradley and Terry),
                $L_{{\text{{RM}}}} = -\,\mathbb{{E}}_{{(x, y_w, y_l)}}\big[\log \sigma\big(\hat r(x, y_w) - \hat r(x, y_l)\big)\big]$,
                where $\sigma(z) = 1/(1 + e^{{-z}})$ is the logistic function of notebook 03,
                not the group standard deviation of the Definition, with Adam on minibatches
                ({_fit}):
                """
            ),
            mo.md(source_md(prefs.fit_reward_model)),
            mo.md(cite("rafailov2023", "bradley1952")),
            mo.md(
                r"""
                `synthesis.run_one` trains one name: PPO and GRPO with the true reward;
                DPO on pairs drawn from $\pi_{\text{ref}}$ with the run's seed; PPO on a
                reward model fitted to those same pairs (same seed), with its gap computed
                from the true reward against $\pi^*_\beta$. `synthesis.tune` runs every
                name at every rate of the grid over the seeds and keeps the rate with the
                lowest mean final gap; `scripts/synthesis.py` does that once and writes
                `data/synthesis.json`, which this notebook reads.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Simulation
    """)
    return


@app.cell
def _(mo, syn):
    mo.stop(
        syn is None,
        mo.md(
            "**No precomputed comparison.** `data/synthesis.json` (or the file named by "
            "`$PROXOPT_SYNTHESIS_FILE`) was not found. Run `make synthesis` to write it. "
            "Until then only the live cell below works."
        ),
    )
    comparison = syn
    return (comparison,)


@app.cell
def _(comparison, labels, mo, rate, span, synthesis):
    _m = comparison["method"]
    _grid = comparison["grid"]
    _rates = _m["lr_grid"]
    _rule, _used = _m["rule_lr"], _m["comparison_lr"]
    _by = {(r["name"], r["lr"]): r for r in _grid}

    def _cell(name, lr):
        r = _by[(name, lr)]
        text = f"{r['mean_final_gap']:.4f} ({span(r['final_gaps'])})"
        return f"**{text}**" if lr == _rule[name] else text

    _edge = [n for n, e in _m["grid_edge"].items() if e["at_grid_edge"]]
    _edge_text = ", ".join(
        f"{labels[n]} ({rate(_rule[n])}, the {_m['grid_edge'][n]['edge']} rate)" for n in _edge
    )
    _table = (
        "| | " + " | ".join(f"lr {rate(x)}" for x in _rates) + " | rule's choice | runs at |\n"
        + "|---" * (len(_rates) + 3) + "|\n"
        + "\n".join(
            f"| {labels[n]} | " + " | ".join(_cell(n, x) for x in _rates)
            + f" | {rate(_rule[n])} | {rate(_used[n])} |"
            for n in synthesis.NAMES
        )
    )
    _seeds = ", ".join(str(s) for s in _m["tuning_seeds"])
    _same = _m["same_seeds_for_tuning_and_comparison"]
    mo.vstack(
        [
            mo.md(
                rf"""
                **The committed comparison**: $\beta$ = {comparison['beta']:g},
                $\tau$ = {comparison['tau']:g}, the default world and its graded reward,
                pairs drawn from $\pi_{{\text{{ref}}}}$, seeds
                {", ".join(str(s) for s in comparison['seeds'])}. Every run starts at
                $\pi_{{\text{{ref}}}}$ and runs its module's default budget (table above).

                **How "tuned" is defined.** For each name, every learning rate of the grid
                {", ".join(rate(x) for x in _rates)} was run over seeds {_seeds}, and the
                rule keeps the rate with the lowest *mean final gap* against
                $\pi^*_\beta$ (a tie goes to the lower rate). Only the learning rate was
                tuned, at {_m['budget']['dpo']['n_pairs']:,} pairs; every other setting is
                its module's default, and the reward model's own fit was not tuned.
                {"Tuning and the comparison use the same seeds, so the comparison's means are the ones the rule minimised: they are slightly optimistic." if _same else "Tuning and the comparison use different seeds."}
                The rule's choice sits at an edge of the grid for: {_edge_text or "none"}.
                A better rate may lie beyond that edge; it was not tried.

                Mean final gap over the seeds (min–max in brackets), the rule's choice in
                bold:
                """
            ),
            mo.md(_table),
        ]
    )
    return


@app.cell
def _(comparison, mo, rate):
    _x = comparison["method"]["grpo_rate_exception"]
    _start = _x["start_gap"]
    _rows = _x["mean_final_gap_by_lr"]
    _table = (
        "| shared lr | GRPO as published | GRPO, PPO-like | difference | published ends |\n"
        "|---|---|---|---|---|\n"
        + "\n".join(
            f"| {rate(r['lr'])} | {r['grpo']:.4f} | {r['grpo_ppo_like']:.4f} "
            f"| {r['grpo'] - r['grpo_ppo_like']:.4f} "
            f"| {'below' if r['grpo'] < _start else 'above'} its start |"
            for r in _rows
        )
    )
    _below = [rate(r["lr"]) for r in _rows if r["grpo"] < _start]
    _above = [rate(r["lr"]) for r in _rows if r["grpo"] >= _start]
    _mn, _st, _fin = _x["min_gap_during_training"], _x["min_gap_step"], _x["final_gap"]
    _steps = comparison["method"]["budget"]["grpo"]["gradient_steps"]
    mo.vstack(
        [
            mo.md(
                rf"""
                **One exception: published GRPO.** It runs at {rate(_x['used_lr'])}, the
                rate chosen for its PPO-like twin, not at {rate(_x['rule_lr'])}, the rate
                the rule alone would pick for it, so that the two GRPOs differ only in the
                two switches. The rule does not suit it: published GRPO does not aim at
                $\pi^*_\beta$, so the rule rewards whichever rate keeps it from going far
                within the budget. Both variants' mean final gap at every rate of the grid
                (every run starts at a gap of {_start:.4f}):
                """
            ),
            mo.md(_table),
            mo.md(
                rf"""
                At the shared rate, {rate(_x['used_lr'])}, published GRPO first gets close:
                its gap falls to {_mn['min']:.4f}–{_mn['max']:.4f} at gradient steps
                {_st['min']}–{_st['max']}, then moves away and ends at
                {_fin['min']:.4f}–{_fin['max']:.4f}, {_fin['min'] / _start:.1f} to
                {_fin['max'] / _start:.1f} times its start. The size of the difference
                between the two variants depends on the shared rate (last-but-one column),
                and whether published GRPO ends above or below its start depends on the
                rate and the budget: within {_steps:,} gradient steps it ends below at
                {", ".join(_below) or "no rate"} and above at {", ".join(_above) or "no rate"}.
                So its result here is a result at one rate and one budget, not a property
                of the algorithm. Notebook 02 measures it at that notebook's default rate,
                with each switch on its own, with the paper's $\beta = 0.04$ and single
                update per batch ("Does the collapse depend on reusing the batch?"), and at
                a ten times lower rate, 1e-4, run for 1,500 iterations ("Does it depend on
                the learning rate?"): there the same fall and rise come later.
                """
            ),
        ]
    )
    return


@app.cell
def _(ceiling, comparison, frontier, labels, mo, plots, rate, span, synthesis, target_point):
    _beta = comparison["beta"]
    _runs = comparison["runs"]
    _target = target_point(_beta)
    _start = _runs["ppo"][0].metrics["gap"][0]

    def _row(name):
        group = _runs[name]

        def last(k):
            return [r.metrics[k][-1] for r in group]

        unit = {"dpo": "pairs", "ppo_rm": "reward-model calls"}.get(name, "reward calls")
        return (
            f"| {labels[name]} | {rate(comparison['lr'][name])} | {group[0].steps[-1]:,} "
            f"| {group[0].feedback[-1]:,} {unit} | {span(last('reward'), '.3f')} "
            f"| {span(last('kl'), '.3f')} | {span(last('gap'))} "
            f"| {span([g / _start for g in last('gap')], '.0%')} |"
        )

    mo.vstack(
        [
            mo.md(
                rf"""
                **All five in the reward–drift plane**, mean trajectory over the seeds, with
                the exact frontier and the single point every run is measured against,
                $\pi^*_\beta$ at $\beta$ = {_beta:g}:
                """
            ),
            plots.frontier_plot(
                frontier,
                _runs,
                targets={f"π*_β, β = {_beta:g} (the one target)": _target},
                ceiling=ceiling,
            ),
            mo.md(
                "**Against gradient steps**, mean line and min-to-max band over the seeds. "
                "DPO takes its steps on minibatches of pairs, the others on sampled batches."
            ),
            plots.training_curves(_runs, keys=("reward", "kl", "gap")),
            mo.md(
                "| run | lr | gradient steps | feedback | E[r] | KL(π ‖ π_ref) | gap "
                "| gap / start |\n|---|---|---|---|---|---|---|---|\n"
                + "\n".join(_row(n) for n in synthesis.NAMES)
                + f"\n| π*_β, β = {_beta:g} | | | | {_target[1]:.3f} | {_target[0]:.3f} | 0 | 0 |"
                + f"\n\nValues at the last step, min–max over the seeds; every run starts at "
                f"a gap of {_start:.4f}. Feedback is counted in calls to the true reward for "
                "the three sampling runs, in labelled pairs for DPO, and in calls to the fitted "
                "reward model for PPO on it (which was fitted to the same pairs as DPO)."
            ),
        ]
    )
    return


@app.cell
def _(comparison, mo, plots, rate, sampling):
    _m = comparison["method"]
    _b = _m["budget"]
    _uses = _b["dpo"]["steps"] * min(_b["dpo"]["batch_size"], _b["dpo"]["n_pairs"]) / _b["dpo"]["n_pairs"]
    _counts = sorted({int(n) for n, _ in comparison["points"]["dpo"]})
    mo.vstack(
        [
            mo.md(
                rf"""
                ### Feedback

                The gap against units of feedback. What one unit is differs by name:

                - PPO and both GRPOs: one unit is one call to the true reward function,
                  one per sampled response; their curves run over the whole budget.
                - DPO: one unit is one labelled preference pair; the set is fixed before
                  training, so a run is one point at its pair count. At
                  {_b['dpo']['n_pairs']:,} pairs each pair is used {_uses:.1f} times on
                  average.
                - PPO on the reward model: also drawn at its pair count, the labels it
                  consumed. It then makes {_b['ppo_rm']['iterations'] * _b['ppo_rm']['batch_size']:,}
                  reward-model calls, which cost no labels.

                Points are the final gap at {", ".join(f"{c:,}" for c in _counts)} pairs:
                a marker at the mean of the seeds, a bar from min to max. Every run at
                every pair count uses the rate tuned at {_b['dpo']['n_pairs']:,} pairs
                (DPO {rate(comparison['lr']['dpo'])}, PPO on the reward model
                {rate(comparison['lr']['ppo_rm'])}).

                **A reward call and a labelled pair are different things.** A reward call
                is a program run on one response; a labelled pair is a noisy judgement
                between two. This plot puts both on one axis only to show where each
                method spends its budget; there is no exchange rate between the two units,
                and **these plots do not rank the algorithms.** Each runs at its own tuned
                rate and its module's default budget, and a different budget would move
                every curve and point.
                """
            ),
            plots.feedback_points(
                {n: comparison["runs"][n] for n in sampling},
                {n: [tuple(p) for p in pts] for n, pts in comparison["points"].items()},
            ),
        ]
    )
    return


@app.cell
def _(comparison, mo, np, plots, rate, span):
    _pts = comparison["points"]
    _rm = comparison["reward_model"]
    _counts = sorted({int(n) for n, _ in _pts["dpo"]})
    _start = comparison["runs"]["dpo"][0].metrics["gap"][0]

    def _at(name, n):
        return [g for c, g in _pts[name] if int(c) == n]

    def _rmv(key, n):
        return [r[key] for r in _rm if r["n_pairs"] == n]

    _rows = "\n".join(
        f"| {n:,} | {np.mean(_at('dpo', n)):.4f} ({span(_at('dpo', n))}) "
        f"| {np.mean(_at('ppo_rm', n)):.4f} ({span(_at('ppo_rm', n))}) "
        f"| {span(_rmv('corr_all', n), '.2f')} | {span(_rmv('corr_covered', n), '.2f')} "
        f"| {span(_rmv('covered_mass_ref', n), '.0%')} |"
        for n in _counts
    )
    _dpo_ahead = [n for n in _counts if max(_at("dpo", n)) < min(_at("ppo_rm", n))]
    _overlap = [n for n in _counts if n not in _dpo_ahead]
    _rm_above = [n for n in _counts if min(_at("ppo_rm", n)) > _start]
    _ahead_text = (
        "in every seed at every pair count"
        if len(_dpo_ahead) == len(_counts)
        else f"in every seed at {', '.join(f'{n:,}' for n in _dpo_ahead) or 'no pair count'}; "
        f"the seeds overlap at {', '.join(f'{n:,}' for n in _overlap)}"
    )
    _edge = comparison["method"]["grid_edge"]["ppo_rm"]
    _cov_better = all(r["corr_covered"] > r["corr_all"] for r in _rm)
    _mean_all = [np.mean(_rmv("corr_all", n)) for n in _counts]
    _mean_mass = [np.mean(_rmv("covered_mass_ref", n)) for n in _counts]
    _rises = bool(np.all(np.diff(_mean_all) > 0))
    _mass_rises = bool(np.all(np.diff(_mean_mass) > 0))
    _acc_text = (
        ("Its mean correlation over all responses rises at every step up in pairs"
         if _rises else "Its mean correlation over all responses does not rise at every step up in pairs")
        + f" ({', '.join(f'{v:.2f}' for v in _mean_all)}). "
        + ("In every row (every seed at every pair count) it is higher on covered responses "
           "than on all responses." if _cov_better
           else "It is not higher on covered responses than on all responses in every row.")
        + (" The pairs cover more of π_ref's mass as their number grows "
           if _mass_rises else " The covered share of π_ref's mass does not grow at every step ")
        + f"({', '.join(f'{v:.0%}' for v in _mean_mass)})."
    )

    with plots.styled_figure(2, width=4.4, height=3.4) as (_fig, _axes):
        for _name in ("dpo", "ppo_rm"):
            _mean = [np.mean(_at(_name, n)) for n in _counts]
            _lo = [_mu - min(_at(_name, n)) for _mu, n in zip(_mean, _counts)]
            _hi = [max(_at(_name, n)) - _mu for _mu, n in zip(_mean, _counts)]
            _axes[0].errorbar(_counts, _mean, yerr=[_lo, _hi], fmt="o-", color=plots.COLORS[_name],
                              label=_name, capsize=0, ms=6)
        _axes[0].axhline(_start, color=plots.COLORS["ref"], ls=":", lw=1.2, label="start (π_ref)")
        _axes[0].set_ylabel("final β·KL(π ‖ π*)")
        _axes[0].set_ylim(bottom=0)
        _axes[0].legend(loc="upper right", fontsize=8)
        for _key, _ls, _lab in (("corr_all", "-", "corr., all responses"),
                                ("corr_covered", "--", "corr., covered responses"),
                                ("covered_mass_ref", ":", "π_ref mass covered")):
            _axes[1].plot(_counts, [np.mean(_rmv(_key, n)) for n in _counts], _ls, marker="o",
                          color=plots.COLORS["ppo_rm"] if _key != "covered_mass_ref" else plots.COLORS["ref"],
                          label=_lab, ms=5)
        _axes[1].set_ylim(0, 1)
        _axes[1].legend(loc="lower right", fontsize=8)
        for _ax in _axes:
            _ax.set_xscale("log")
            _ax.set_xlabel("labelled pairs")

    mo.vstack(
        [
            mo.md(
                r"""
                ### The fair comparison: DPO against PPO on a reward model, on the same pairs

                For each seed, both runs train on the same pairs (drawn from
                $\pi_{\text{ref}}$, labelled with noise $\tau$). The reward model's accuracy
                is measured, not assumed: the Pearson correlation of its scores with
                $r/\tau$, each centred on its prompt's mean (so a constant per prompt does
                not count), pooled over the prompts, once over all responses alike and once
                over the responses that appear in a pair ("covered"); and the probability
                mass of $\pi_{\text{ref}}$ on covered responses, averaged over prompts.
                The correlations weigh every response alike, not by probability, so they
                are not what the gap weighs.
                """
            ),
            mo.md(
                "| pairs | DPO final gap | PPO-on-RM final gap | RM corr., all | RM corr., "
                "covered | π_ref mass covered |\n|---|---|---|---|---|---|\n" + _rows
                + "\n\nGaps: mean over the seeds (min–max); reward-model columns: min–max."
            ),
            _fig,
            mo.md(
                rf"""
                **What the file shows.** DPO ends with the lower gap {_ahead_text}. PPO on
                the reward model ends *above* its starting gap ({_start:.4f}) in every seed
                at {", ".join(f"{n:,}" for n in _rm_above) or "no pair count"} pairs. The
                reward model: {_acc_text}

                **A reading, not measured here.** With few pairs the reward model is
                wrong on most responses, and PPO samples fresh responses wherever its
                policy goes and climbs the model's scores there, errors included, so it can
                end further from $\pi^*_\beta$ than it started. DPO moves the policy only
                through the log-ratios of the responses in its pairs (and whatever the
                shared network generalises). The file records the reward model's accuracy
                and the final gaps; it does not record where PPO's policy went or how the
                model scored there, so it does not settle this.

                **What these numbers depend on.** Both fixed-dataset methods' results depend
                on the step budget: notebook 03 measured that DPO's gap on these pairs
                turns and rises when it trains longer (its item 5). PPO on the reward model
                runs at {rate(comparison['lr']['ppo_rm'])}, which is
                {"the " + _edge["edge"] + " rate of the grid: a lower one might do better and was not tried" if _edge["at_grid_edge"] else "inside the grid"}.
                Both rates were tuned at one pair count and reused at the others.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    beta_live = mo.ui.slider(steps=[0.05, 0.1, 0.2, 0.5], value=0.1, label="β", show_value=True)
    tau_live = mo.ui.slider(steps=[0.1, 0.25, 0.5, 1.0], value=0.25, label="preference noise τ",
                            show_value=True)
    seed_live = mo.ui.number(start=0, stop=99, step=1, value=0, label="seed")
    live_button = mo.ui.run_button(label="Run the comparison for one seed (about 40 s)")
    return beta_live, live_button, seed_live, tau_live


@app.cell
def _(beta_live, live_button, mo, rate, seed_live, synthesis, syn, tau_live):
    live_lr = dict(syn["lr"]) if syn is not None else dict(synthesis.DEFAULT_LR)
    _which = ("the committed rates" if syn is not None
              else "each module's default rate (no committed rates were found)")
    _repro = (" At the defaults and seed 0 it reproduces the committed seed-0 runs."
              if syn is not None else "")
    _tuned = (f"The rates were tuned at $\\beta$ = {syn['beta']:g} and $\\tau$ = {syn['tau']:g} only, "
              "and one seed" if syn is not None else "One seed")
    mo.vstack(
        [
            mo.md(
                rf"""
                **Other settings, live.** Pick $\beta$, $\tau$ and a seed and press the
                button: `synthesis.compare` trains all five for that one seed, plus DPO and
                PPO on the reward model at every pair count, with {_which}
                ({", ".join(f"{k} {rate(v)}" for k, v in live_lr.items())}). It takes about
                40 s: measured at 40 and 43 s on the machine this set was built on (an
                Apple M3). Results are cached by their arguments, so returning to a setting
                is instant. {_tuned} shows no spread, so a live run is a look, not a
                measurement.{_repro}
                """
            ),
            mo.hstack([beta_live, tau_live, seed_live, live_button], justify="start"),
        ]
    )
    return (live_lr,)


@app.cell
def _(mo, synthesis):
    @mo.cache
    def live_compare(beta, tau, seed, rates):
        """`synthesis.compare` for one seed, cached by its arguments."""
        return synthesis.compare(beta, tau, (seed,), dict(rates))

    return (live_compare,)


@app.cell
def _(
    beta_live,
    ceiling,
    frontier,
    labels,
    live_button,
    live_compare,
    live_lr,
    mo,
    plots,
    sampling,
    seed_live,
    synthesis,
    tau_live,
    target_point,
):
    mo.stop(not live_button.value, mo.md("*Press the button to train the runs (about 40 s).*"))
    _res = live_compare(beta_live.value, tau_live.value, int(seed_live.value),
                        tuple(sorted(live_lr.items())))
    _b = _res["beta"]
    _rows = "\n".join(
        f"| {labels[n]} | {_res['runs'][n][0].metrics['gap'][0]:.4f} "
        f"| {_res['runs'][n][0].metrics['gap'][-1]:.4f} |"
        for n in synthesis.NAMES
    )
    _rm = "\n".join(
        f"| {r['n_pairs']:,} | {r['corr_all']:.2f} | {r['corr_covered']:.2f} "
        f"| {r['covered_mass_ref']:.0%} |"
        for r in _res["reward_model"]
    )
    mo.vstack(
        [
            plots.frontier_plot(frontier, _res["runs"],
                                targets={f"π*_β, β = {_b:g}": target_point(_b)}, ceiling=ceiling),
            plots.training_curves(_res["runs"], keys=("reward", "kl", "gap")),
            plots.feedback_points({n: _res["runs"][n] for n in sampling},
                                  {n: list(p) for n, p in _res["points"].items()}),
            mo.md(
                f"β = {_b:g}, τ = {_res['tau']:g} (DPO and PPO on the reward model use "
                f"β/τ = {_b / _res['tau']:g}), seed {_res['seeds'][0]}.\n\n"
                "| run | start gap | final gap |\n|---|---|---|\n" + _rows
                + "\n\n| pairs | RM corr., all | RM corr., covered | π_ref mass covered |\n"
                "|---|---|---|---|\n" + _rm
            ),
        ]
    )
    return


@app.cell
def _(DEFAULT_WORLD, live_button, mo, syn):
    # shown whenever comparison figures are drawn: the committed ones, or the live run's
    mo.stop(syn is None and not live_button.value)
    _w = DEFAULT_WORLD
    _setup = (f"this grid and {len(syn['seeds'])} seeds" if syn is not None
              else "each module's default rate and one seed")
    mo.md(rf"""
    **What carries over to real LLMs, and what does not.** What carries over is the logic of
    the comparison. Algorithms that aim at different points cannot be compared by one
    number until they aim at the same one, which is why DPO and PPO on a reward model run
    with $\beta/\tau$ here. Information matters: a method scored by the true reward is not
    comparable with one that sees only pairs. A tuned rate is a choice of rule and grid,
    and the tuning seeds should be stated. What does not carry over: here the optimum, the
    gap, the reward model's correlation with the truth and the coverage of the pairs are
    exact, and the true $\tau$ is known, so $\beta/\tau$ can be set; for an LLM none of
    these is available. Which method ends with the lower gap, by how much, and at which
    rate, are facts about this world ({_w.n_prompts} prompts, {_w.n_responses:,} responses
    per prompt, a small GRU, these budgets, {_setup}), not about LLMs,
    and the plots above do not rank the algorithms in general.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to try

    Make a prediction before opening each answer.
    """)
    return


@app.cell
def _(mo, syn):
    mo.stop(syn is None, mo.md("*Items 1 to 3 are computed from the precomputed runs, which are not loaded: run `make synthesis`.*"))
    _seeds = ", ".join(str(x) for x in syn["seeds"])
    mo.md(f"Answers 1 to 3 are computed from the committed comparison (seeds {_seeds}), so "
          "they agree with the tables above.")
    return


@app.cell
def _(comparison, np, rate, reveal):
    _x = comparison["method"]["grpo_rate_exception"]
    _start = _x["start_gap"]
    _at = {r["lr"]: r for r in _x["mean_final_gap_by_lr"]}
    _low, _used = _at[_x["rule_lr"]], _at[_x["used_lr"]]
    _best = min(_x["mean_final_gap_by_lr"], key=lambda r: r["grpo"])
    _grid = [r["lr"] for r in _x["mean_final_gap_by_lr"]]
    _where = ("the grid's lowest rate" if _best["lr"] == min(_grid)
              else "the grid's highest rate" if _best["lr"] == max(_grid) else "inside the grid")
    _pts = comparison["points"]
    pair_counts = sorted({int(n) for n, _ in _pts["dpo"]})
    pair_diff = {n: np.mean([g for c, g in _pts["ppo_rm"] if int(c) == n])
                 - np.mean([g for c, g in _pts["dpo"] if int(c) == n]) for n in pair_counts}
    _rm = comparison["reward_model"]
    rm_corr = {n: np.mean([r["corr_all"] for r in _rm if r["n_pairs"] == n]) for n in pair_counts}
    reveal(
        f"**1. Published GRPO at the rule's own rate.** The rule would have run published "
        f"GRPO at {rate(_x['rule_lr'])} instead of {rate(_x['used_lr'])}. Predict: does it end "
        "above or below its starting gap there, and is it then closer to its PPO-like twin?",
        f"From the grid. At {rate(_x['rule_lr'])} published GRPO's mean final gap is "
        f"{_low['grpo']:.4f}, {'below' if _low['grpo'] < _start else 'above'} its start of "
        f"{_start:.4f}; at {rate(_x['used_lr'])} it is {_used['grpo']:.4f}. The PPO-like twin "
        f"ends at {_low['grpo_ppo_like']:.4f} and {_used['grpo_ppo_like']:.4f}, so the "
        f"difference between the two is {_low['grpo'] - _low['grpo_ppo_like']:.4f} at the "
        f"lower rate and {_used['grpo'] - _used['grpo_ppo_like']:.4f} at the shared one. "
        f"Its lowest final gap on the grid is at {rate(_best['lr'])}, {_where}. "
        "The file keeps only the final gaps at the other rates; what happens later at a "
        "lower rate was measured in notebook 02 (\"Does it depend on the learning "
        "rate?\"): published GRPO at 1e-4, 4 epochs per batch, run for 1,500 iterations "
        "instead of 150, seeds 0–2, has its lowest gap at iterations 64–73, is back above "
        "its start by iterations 159–174, and ends at 3.8 to 4.2 times its start (at "
        "notebook 02's default rate, 1e-3, after 150 iterations: 3.6 to 4.3). So at 1e-4 "
        "the 150-iteration budget ends after the gap's lowest point and before it is back "
        "above its start, and the rule's choice says little about the algorithm.",
    )
    return pair_counts, pair_diff, rm_corr


@app.cell
def _(pair_counts, pair_diff, reveal, rm_corr):
    reveal(
        "**2. More pairs for the reward model.** As the pair count grows from the smallest "
        "to the largest, does PPO on the reward model catch up with DPO?",
        "From the points. PPO-on-RM's mean final gap minus DPO's is "
        + ", ".join(f"{pair_diff[n]:.4f} at {n:,} pairs" for n in pair_counts)
        + "; the reward model's mean correlation with r/τ over all responses is "
        + ", ".join(f"{rm_corr[n]:.2f}" for n in pair_counts)
        + ". "
        + ("The difference shrinks at every step up in pairs"
           if all(pair_diff[pair_counts[i]] > pair_diff[pair_counts[i + 1]]
                  for i in range(len(pair_counts) - 1))
           else "The difference does not shrink at every step up in pairs")
        + (", and it has not closed at the largest count" if pair_diff[pair_counts[-1]] > 0
           else ", and it has closed by the largest count")
        + ". This is at fixed budgets, and at rates tuned at one pair count.",
    )
    return


@app.cell
def _(comparison, np, pair_counts, reveal):
    _seeds = list(comparison["seeds"])
    _rows = []  # (pair count, reward model's corr. over all responses, PPO-on-RM final gap)
    for _n in pair_counts:
        _gaps = [g for c, g in comparison["points"]["ppo_rm"] if int(c) == _n]  # seed order
        _rm = sorted((r for r in comparison["reward_model"] if r["n_pairs"] == _n),
                     key=lambda r: _seeds.index(r["seed"]))
        _rows += [(_n, r["corr_all"], g) for r, g in zip(_rm, _gaps)]
    _corr = np.corrcoef([r[1] for r in _rows], [r[2] for r in _rows])[0, 1]
    _match = [n for n in pair_counts
              if max((r for r in _rows if r[0] == n), key=lambda r: r[1])
              == min((r for r in _rows if r[0] == n), key=lambda r: r[2])]
    reveal(
        "**3. Does a better reward model give PPO a lower gap?** Across seeds and pair "
        "counts, predict: are the runs whose reward model correlates best with r/τ (over all "
        "responses) the runs where PPO on it ends closest to π*_β?",
        f"From the committed rows ({len(_rows)}: {len(_seeds)} seeds at each of "
        f"{len(pair_counts)} pair counts). The Pearson correlation between the reward "
        f"model's correlation with r/τ and PPO's final gap is {_corr:.2f}"
        + (" (a better model goes with a lower gap)" if _corr < 0 else
           " (a better model does not go with a lower gap)")
        + ". Both change with the pair count, so this mixes the effect of more pairs with "
        "that of a better fit. Within one pair count, the seed with the best-correlated "
        "reward model is also the seed with the lowest PPO gap at "
        + (", ".join(f"{n:,}" for n in _match) + " pairs" if _match else "no pair count")
        + f" ({len(_match)} of {len(pair_counts)}). The correlation weighs every response "
        "alike, while the gap weighs responses by probability, so the two need not agree.",
    )
    return


@app.cell
def _(margins, np, olmo, reveal):
    _dpo = np.array(olmo.implicit_margins(margins, "dpo", per_token=True))
    _ins = np.array(olmo.implicit_margins(margins, "instruct", per_token=True))
    _keep = ~((_dpo == 0) & (_ins == 0))
    _dpo, _ins = _dpo[_keep], _ins[_keep]
    _q1, _q3 = np.quantile(np.abs(_dpo), [0.25, 0.75])
    _same = np.sign(_dpo) == np.sign(_ins)
    _big, _small = np.abs(_dpo) >= _q3, np.abs(_dpo) <= _q1
    reveal(
        "**4. Where Instruct keeps DPO's margin.** On the OLMo preference pairs, sort the "
        "pairs by the size of the DPO checkpoint's per-token margin against SFT. Predict: "
        "after the two RL stages, does the Instruct checkpoint keep the margin's sign more "
        "often on the pairs DPO separated most, or about as often as on the others?",
        f"From `data/olmo/margins.jsonl` ({int(_keep.sum())} pairs, leaving out those whose "
        "margin is exactly zero under both checkpoints). Instruct's per-token margin has "
        f"the sign of DPO's on {_same[_big].mean():.0%} of the quarter of pairs with the "
        f"largest |DPO margin| (at least {_q3:.3f}), on {_same[_small].mean():.0%} of the "
        f"quarter with the smallest (at most {_q1:.3f}), and on {_same.mean():.0%} of all. "
        "These are the DPO stage's own training pairs, so this is about what the RL "
        "stages kept of what DPO fitted, not about generalisation.",
    )
    return


@app.cell
def _(cite, mo):
    mo.vstack(
        [
            mo.md(r"""
    ## Where it went next

    In two of the recipes recorded below (Tülu 3 and OLMo 2) the methods were not used as
    alternatives: a model first goes through supervised fine-tuning (SFT), then is trained
    on preferences with DPO, and then with RL on rewards a program can check. The history
    explains how that came about, and the OLMo ladder looks at one model trained that way.

    ### History

    **1992.** REINFORCE: the score-function gradient that notebook 00 climbs $J_\beta$ with.
    """),
            mo.md(cite("williams1992")),
            mo.md(r"""
    **2015 to 2017.** TRPO constrains the KL between successive policies; GAE estimates
    per-step advantages from a learned value function; PPO replaces TRPO's constraint with a
    clipped ratio and reuses each batch for several epochs (notebook 01).
    """),
            mo.md(cite("schulman2015trpo", "schulman2016gae", "schulman2017ppo")),
            mo.md(r"""
    **2017 to 2022: RLHF.** Learning from human preferences instead of a hand-written reward:
    humans compare pairs of outputs, a reward model is fitted to the comparisons, and RL
    maximises it. Ziegler et al. applied this to language models, with the KL-penalised
    reward $r - \beta \log(\pi/\pi_{\text{ref}})$ and the closed form
    $\pi^* \propto \pi_{\text{ref}}\, e^{r/\beta}$ that notebook 00 uses. Stiennon et al.
    trained a value function with parameters separate from the policy's, initialised from
    the reward model. InstructGPT (Ouyang et al.) paid the KL penalty token by token, the
    form notebook 01 builds.
    """),
            mo.md(cite("christiano2017", "ziegler2019", "stiennon2020", "ouyang2022")),
            mo.md(r"""
    **2023: DPO.** Starting from the same closed form, and the Bradley–Terry model of
    preferences, Rafailov et al. turned the two RLHF stages into one classification loss on
    the pairs: no reward model, no sampling, no RL step (notebook 03).
    """),
            mo.md(cite("bradley1952", "rafailov2023")),
            mo.md(r"""
    **2024: GRPO.** DeepSeekMath replaced PPO's critic with the mean reward of a group of
    samples for the same prompt, and moved the KL term from the reward into the loss, with
    a coefficient of 0.04 (notebook 02).
    """),
            mo.md(cite("shao2024")),
            mo.md(r"""
    **2024: RL with verifiable rewards.** Tülu 3 introduced RLVR "for training language
    models on tasks with verifiable outcomes such as mathematical problem-solving and
    instruction following", trained "following preference finetuning", with PPO; its
    authors also report that "length-normalized DPO works best".
    """),
            mo.md(cite("lambert2024")),
            mo.md(r"""
    **2025.** DeepSeek-R1-Zero applied GRPO directly to a base model, with no supervised
    fine-tuning first, and with a rule-based reward for accuracy and format; its authors
    write that "neural reward models are susceptible to reward hacking during large-scale
    RL".
    """),
            mo.md(cite("guo2025")),
            mo.md(r"""
    **2025: OLMo 2 1B.** The OLMo 2 1B checkpoints are dated April 2025 (`OLMo-2-0425-1B`).
    The OLMo 2 paper's reference year, 2024, is that of its first submission; the version
    read for this set (arXiv v3) covers the 1B model and reports RLVR on its 1B and 32B
    models "with Group Relative Policy Optimization (GRPO) …, which forgoes the need for a
    reward model" (its 7B and 13B models used PPO, with the value function initialised from
    the reward models).
    """),
            mo.md(cite("olmo2024")),
            mo.md(r"""
    **Why RL came back after DPO.** What the sources recorded here say: Tülu 3 and OLMo 2
    did not replace DPO with RL; they added an RL stage after it: RLVR, which Tülu 3
    describes as training "on tasks with verifiable outcomes". OLMo 2 used GRPO for its 1B
    model, describing it as a method that "forgoes the need for a reward model". DeepSeek-R1 used rule-based rewards, writing that
    "neural reward models are susceptible to reward hacking during large-scale RL".
    *This notebook's reading*, which none of those sources states in these words:
    a reward a program can check needs no annotator per sample, so an RL method can score
    as many fresh samples as compute allows; and an RL method samples from its own current
    policy, so it is not confined to the responses of a fixed dataset, a limit notebook
    03 measured for DPO (its pairs cover only part of $\pi_{\text{ref}}$'s probability,
    and the implicit reward of responses outside them moves only a little towards the
    truth).
    """),
        ]
    )
    return


@app.cell
def _(mo, olmo):
    margins = olmo.load_margins()
    drift = olmo.load_drift()
    meta = olmo.load_meta()
    _missing = margins is None or drift is None or meta is None
    mo.stop(
        _missing,
        mo.md(
            "### The OLMo ladder: a reality check\n\n**No OLMo measurements.** "
            "`data/olmo/margins.jsonl`, `data/olmo/drift.jsonl` or `data/olmo/meta.json` (or "
            "the directory named by `$PROXOPT_OLMO_DIR`) was not found. Run `make olmo` to "
            "measure them."
        ),
    )
    return drift, margins, meta


@app.cell
def _(cite, margins, meta, mo):
    _rev = meta.get("revisions", {})
    _names = " → ".join(f"`{_rev[m]['repo']}`" for m in ("sft", "dpo", "instruct") if m in _rev)
    mo.vstack(
        [
            mo.md(
                rf"""
                ### The OLMo ladder: a reality check

                **The recipe** of OLMo 2 1B (April 2025), as recorded in
                `docs/olmo-facts.md`, which quotes each source:

                1. **SFT**, the starting checkpoint.
                2. **DPO**, length-normalised, with $\beta = 5$, on
                   `allenai/olmo-2-0425-1b-preference-mix`. Source: the training command
                   the model's authors published in their open-instruct repository
                   (`--dpo_loss_type dpo_norm --dpo_beta 5`); the dataset is also named on
                   the DPO model card. The OLMo 2 paper and the model cards name neither
                   the variant nor $\beta$. Length-normalised means each response enters
                   through its *per-token mean* log-probability (Tülu 3, eq. 6), so this
                   $\beta$ is not comparable with a $\beta$ on summed log-ratios.
                3. **RL with verifiable rewards, two GRPO stages in sequence**: first on
                   MATH, GSM8K and instruction-following constraints (giving the RLVR1
                   checkpoint), then on MATH only (giving Instruct). Sources: the OLMo 2
                   paper (§5: RLVR on the 1B model with GRPO); the open-instruct
                   documentation of the 1B runs (`--beta 0.01`, a KL coefficient of 0.01, in
                   both stages; the second stage's quoted command also sets
                   `--kl_estimator kl3`); the GRPO script `docs/olmo-facts.md` names as the
                   one those runs used, in which `kl3` is the estimator of notebook 02, added
                   to the loss, and advantages are standardised over groups of
                   `number_samples_per_prompt = 16`; and the
                   Instruct model card ("Finetuned from model: allenai/OLMo-2-0425-1B-RLVR1").
                4. **Instruct**, the released model.

                The checkpoints measured here are {_names}, at the revisions in
                `data/olmo/meta.json`; RLVR1 is not measured, so the step DPO → Instruct
                covers both RL stages. Every number below is a forward pass on these
                checkpoints in {meta.get('dtype', 'half precision')} on
                {meta.get('device', 'the local device')}, computed by
                `scripts/olmo_measure.py`; nothing was generated for this notebook.
                """
            ),
            mo.md(cite("olmo2024", "lambert2024")),
        ]
    )
    return


@app.cell
def _(margins, meta, mo, np, olmo, plots, torch):
    def _g(k):
        return np.array([r[k] for r in margins], dtype=float)

    _nw, _nl = _g("n_w"), _g("n_l")
    _s = olmo.summarize_margins(margins)
    _mm = {(m, v): np.array(olmo.implicit_margins(margins, m, per_token=v == "mean"))
           for m in ("dpo", "instruct") for v in ("mean", "sum")}

    def _line(model, version):
        d = _s[model][version]
        return (f"mean {d['mean_margin']:.3f}, median {np.median(_mm[(model, version)]):.3f}, "
                f"positive in {d['share_positive']:.0%} of pairs")

    _zero = {v: (_mm[("dpo", v)] == 0) & (_mm[("instruct", v)] == 0) for v in ("mean", "sum")}
    _agree = {v: (np.sign(_mm[("dpo", v)]) == np.sign(_mm[("instruct", v)]))[~_zero[v]].mean()
              for v in ("mean", "sum")}
    _n_zero = int(_zero["mean"].sum())
    _corr = {v: np.corrcoef(_mm[("dpo", v)], _mm[("instruct", v)])[0, 1] for v in ("mean", "sum")}
    _dw = (_g("lp_w_dpo") - _g("lp_w_sft")) / _nw
    _dl = (_g("lp_l_dpo") - _g("lp_l_sft")) / _nl
    _mg = meta.get("margins", {})
    if ((_dw < 0) & (_dl < 0)).mean() > 0.5 and _dl.mean() < _dw.mean():
        _how = ("So on most of these pairs DPO lowered both responses, and the rejected one "
                "by more on average.")
    elif _dw.mean() > 0 and _dl.mean() < 0:
        _how = "So on average DPO raised the chosen response and lowered the rejected one."
    else:
        _how = "The data show no single pattern."

    with plots.styled_figure(2, width=4.4, height=3.3) as (_hist, _axes):
        for _ax, _v, _title in ((_axes[0], "mean", "per-token mean (what DPO trained on)"),
                                (_axes[1], "sum", "summed")):
            for _m, _name in (("dpo", "DPO vs SFT"), ("instruct", "Instruct vs SFT")):
                _ax.hist(_mm[(_m, _v)], bins=40, histtype="step", lw=1.8,
                         color=plots.COLORS["dpo" if _m == "dpo" else "ppo"], label=_name)
            _ax.axvline(0, color=plots.COLORS["ref"], lw=1.0)
            _ax.set_title(_title)
            _ax.set_xlabel("implicit reward margin (log-prob units)")
            _ax.set_ylabel("pairs")
        _axes[0].legend(loc="upper right", fontsize=8)

    _pair = plots.scatter_true_vs_estimate(
        estimate=torch.tensor(_mm[("instruct", "mean")]), true=torch.tensor(_mm[("dpo", "mean")]),
        xlabel="DPO vs SFT, per-token margin", ylabel="Instruct vs SFT, per-token margin")
    _cr = plots.scatter_true_vs_estimate(
        estimate=torch.tensor(_dw), true=torch.tensor(_dl),
        xlabel="Δ per-token log-prob of the rejected response",
        ylabel="Δ per-token log-prob of the chosen response")

    mo.vstack(
        [
            mo.md(
                rf"""
                **What DPO learned.** For each of {_s['n']} preference pairs from the DPO
                stage's own data, the implicit reward margin of a checkpoint against SFT,
                $\log\frac{{\pi(y_w)}}{{\pi_{{\text{{SFT}}}}(y_w)}} - \log\frac{{\pi(y_l)}}{{\pi_{{\text{{SFT}}}}(y_l)}}$
                (notebook 03's $h/\beta$, with SFT in the place of $\pi_{{\text{{ref}}}}$),
                in two versions: with each log-ratio divided by its response's token count
                first (the per-token mean, the quantity length-normalised DPO trains on,
                so it comes first), and summed.

                - DPO against SFT, per-token mean: {_line('dpo', 'mean')}.
                - DPO against SFT, summed: {_line('dpo', 'sum')}.
                - Instruct against SFT, per-token mean: {_line('instruct', 'mean')}.
                - Instruct against SFT, summed: {_line('instruct', 'sum')}.
                """
            ),
            _hist,
            mo.md(
                rf"""
                **Does Instruct keep the DPO margin, pair by pair?** Leaving out the
                {_n_zero} pairs whose margin is exactly zero under both models (per token),
                the two margins have the same sign in {_agree['mean']:.0%} of pairs per
                token ({_agree['sum']:.0%} summed, leaving out {int(_zero['sum'].sum())}), with a correlation of {_corr['mean']:.2f}
                per token ({_corr['sum']:.2f} summed). Below, each point is one pair; on the
                dashed line Instruct's margin equals DPO's.
                """
            ),
            _pair,
            mo.md(
                rf"""
                **How the margin grew.** A margin can grow by raising the chosen response or
                by lowering the rejected one faster. Per token, from SFT to DPO, the chosen
                response's log-probability changed by {_dw.mean():.3f} on average (median
                {np.median(_dw):.3f}) and fell in {(_dw < 0).mean():.0%} of pairs; the
                rejected one's by {_dl.mean():.3f} (median {np.median(_dl):.3f}), falling in
                {(_dl < 0).mean():.0%}. Both fell in {((_dw < 0) & (_dl < 0)).mean():.0%} of
                pairs. {_how} Points above the dashed line have a positive margin.
                """
            ),
            _cr,
            mo.md(
                rf"""
                **Caveats.** These are the DPO stage's own training pairs, so this shows
                what DPO fitted, not how it generalises. Pairs were taken in order from the
                dataset shuffled with seed {_mg.get('shuffle_seed')} and kept only when the
                prompt plus either response was at most {_mg.get('max_tokens')} tokens: of
                {_mg.get('scanned')} pairs scanned, {_mg.get('too_long')} were too long and
                {_mg.get('prefix_mismatch')} could not be split at the prompt, so the
                sample leans towards short conversations ({_mg.get('identical_responses')}
                of the {_s['n']} kept have identical chosen and rejected text). The
                log-probabilities are from {meta.get('dtype', 'half-precision')} forward
                passes.
                """
            ),
        ]
    )
    return


@app.cell
def _(drift, meta, mo, np, olmo, plots):
    _gen = meta.get("generation", {})

    def _setting(mode):
        if mode == "greedy":
            return f"greedy, up to {_gen.get('max_new_tokens')} new tokens"
        if mode == "sample":
            s = _gen.get("samples", {})
            return (f"sampled at temperature {s.get('temperature')} (seeds "
                    f"{', '.join(str(x) for x in s.get('seeds', []))}), up to "
                    f"{_gen.get('max_new_tokens')} new tokens")
        if mode == "neither":
            s = _gen.get("neither", {})
            return f"sampled at temperature {s.get('temperature')}, up to {s.get('max_new_tokens')} new tokens"
        return mode

    _stats, _tops = {}, {}
    for _step in olmo.STEPS:
        _rows = [r for r in drift if r["step"] == _step]
        _n = np.array([r["n_tokens"] for r in _rows], dtype=float)
        for _key in ("kl_new_old", "kl_old_new"):
            _x = np.array([r[_key] for r in _rows])
            _stats[(_step, _key)] = {"n": len(_rows), "mean": _x.mean(), "median": np.median(_x),
                                     "token": (_x * _n).sum() / _n.sum()}
            _order = np.argsort(-_x)[:3]
            _tops[(_step, _key)] = [(float(_x[i]), int(_n[i])) for i in _order]
    _ones, _no_one = {}, {}
    for _step in olmo.STEPS:
        _long = [r["kl_new_old"] for r in drift if r["step"] == _step and r["n_tokens"] > 1]
        _ones[_step] = sum(1 for r in drift if r["step"] == _step and r["n_tokens"] == 1)
        _no_one[_step] = float(np.mean(_long))
    _modes = {}
    for _r in drift:
        if _r["step"] == olmo.STEPS[0]:
            _modes[_r["mode"]] = _modes.get(_r["mode"], 0) + 1
    _names = {"kl_new_old": "KL(later ‖ earlier)", "kl_old_new": "KL(earlier ‖ later)"}
    _table = (
        "| step | direction | texts | per-text mean | median | token-weighted mean |\n"
        "|---|---|---|---|---|---|\n"
        + "\n".join(
            f"| {s} | {_names[k]} | {_stats[(s, k)]['n']} | {_stats[(s, k)]['mean']:.4f} "
            f"| {_stats[(s, k)]['median']:.4f} | {_stats[(s, k)]['token']:.4f} |"
            for s in olmo.STEPS for k in _names
        )
    )
    _s1, _s2 = olmo.STEPS
    _ids = {st: {r["item_id"] for r in drift if r["step"] == st} for st in olmo.STEPS}
    _shared = len(_ids[_s1] & _ids[_s2])
    _ratio = {k: {st: _stats[(_s1, k)][st] / _stats[(_s2, k)][st] for st in ("mean", "median", "token")}
              for k in _names}
    _top_text = "; ".join(
        f"{s}, {_names[k]}: " + ", ".join(f"{v:.3f} ({n} token{'s' if n != 1 else ''})"
                                          for v, n in _tops[(s, k)])
        for s in olmo.STEPS for k in _names
    )
    _max_one = all(_tops[(s, k)][0][1] == 1 for s in olmo.STEPS for k in _names)
    _checks = meta.get("drift", {}).get("steps", {})
    _greedy = "; ".join(
        f"{s}: {c['greedy_argmax_agree']:,} of {c['greedy_positions']:,} tokens"
        for s, c in _checks.items() if "greedy_positions" in c
    )

    with plots.styled_figure(2, width=4.4, height=3.3) as (_fig, _axes):
        for _ax, _step in zip(_axes, olmo.STEPS):
            _rows = [r for r in drift if r["step"] == _step]
            _ax.scatter([r["n_tokens"] for r in _rows], [r["kl_new_old"] for r in _rows], s=14,
                        alpha=0.7, color=plots.COLORS["dpo" if _step == _s1 else "grpo"],
                        edgecolors="none")
            _ax.set_title(_step)
            _ax.set_xscale("log")
            _ax.set_xlabel("response tokens")
            _ax.set_ylabel("per-text mean KL(later ‖ earlier)")

    mo.vstack(
        [
            mo.md(
                rf"""
                **How far each stage moved the model: next-token KL along the later
                checkpoint's own responses.** For each step, texts the later checkpoint
                wrote (from the neighbouring repository's response files, templated,
                English): {", ".join(f"{c} {_setting(m)}" for m, c in _modes.items())}.
                At every token position of each text, the exact KL between the two
                checkpoints' next-token distributions, in both directions, averaged over
                the text's positions. These texts are not exact samples from either model
                (they were generated at the settings above and capped at
                {_gen.get('max_new_tokens')} new tokens), so
                this is **not the KL between the two models**; it is the next-token KL
                along these texts. As a check that the texts align with the later
                checkpoint, its argmax reproduces its own greedy texts on {_greedy}.

                The per-text mean gives a one-token text the same weight as a long one, so
                the median and a token-weighted mean (each text weighted by its token count)
                are shown with it:
                """
            ),
            mo.md(_table),
            _fig,
            mo.md(
                rf"""
                **Where the largest values come from.** The three largest per-text values,
                with their texts' token counts: {_top_text}.
                {"In every step and direction the largest comes from a one-token text." if _max_one else "Not every largest value comes from a one-token text."}
                Without the one-token texts ({_ones[_s1]} and {_ones[_s2]} of them), the
                per-text mean of KL(later ‖ earlier) is {_no_one[_s1]:.4f} in the first step
                (with them, {_stats[(_s1, 'kl_new_old')]['mean']:.4f}) and
                {_no_one[_s2]:.4f} in the second (with them,
                {_stats[(_s2, 'kl_new_old')]['mean']:.4f}).

                **The two steps compared.** In the direction KL(later ‖ earlier), the first
                step's value is {_ratio['kl_new_old']['mean']:.1f} times the second's by the
                per-text mean, {_ratio['kl_new_old']['median']:.1f} times by the median and
                {_ratio['kl_new_old']['token']:.1f} times by the token-weighted mean (in the
                other direction: {_ratio['kl_old_new']['mean']:.1f}, {_ratio['kl_old_new']['median']:.1f}
                and {_ratio['kl_old_new']['token']:.1f}). The two steps are measured on
                different texts: the DPO checkpoint's own responses for the first, the
                Instruct checkpoint's for the second; {_shared} item ids appear in both
                steps ({len(_ids[_s1])} and {len(_ids[_s2])} in each). Along these texts, the two RL stages
                together moved the next-token distributions much less than the DPO stage
                did. This measurement does not say why. What differs between the stages, as
                `docs/olmo-facts.md` records it: the DPO stage is length-normalised DPO with
                $\beta = 5$ on a preference mix; the RL stages are two GRPO runs, both with a KL
                coefficient of 0.01 (the second stage's command names the `kl3` estimator,
                which the script adds to the loss; the script's groups are of 16), on
                mathematics and instruction-following data. The texts here answer the
                neighbouring repository's general prompts, not those training data, so the
                measurement says nothing about how far the model moved on mathematics or
                instruction following.
                """
            ),
            mo.md(r"""
    **What this section does and does not connect to the toy world.** It connects through
    two quantities. The implicit reward margin is what DPO's loss pushes up (notebook 03),
    measured on a real model's own training pairs. The next-token KL is the per-token
    quantity whose sampled estimates notebooks 01 and 02 penalise, measured exactly along
    real texts. It does not
    connect through anything the toy world computes exactly: there is no $\pi^*_\beta$, no
    gap and no frontier for OLMo; its $\beta = 5$ multiplies per-token mean log-ratios and
    cannot be compared with the toy world's $\beta$; and nothing here tests whether
    published GRPO's behaviour on the toy world (notebook 02) occurs in OLMo's RL stages.
    """),
        ]
    )
    return


@app.cell
def _(REFS, cite, mo):
    mo.md("## References\n\n" + cite(*REFS))
    return


if __name__ == "__main__":
    app.run()
