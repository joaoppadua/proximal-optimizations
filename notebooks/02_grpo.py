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
    from proxopt.algos import grpo, ppo
    from proxopt.algos.grpo import (
        GRPOConfig,
        group_advantages,
        grpo_loss,
        kl_estimator,
        shaped_rewards,
    )
    from proxopt.algos.ppo import PPOConfig
    from proxopt.nbtools import reveal, source_md
    from proxopt.reference import load_reference
    from proxopt.refs import cite
    from proxopt.train import scorer
    from proxopt.world import DEFAULT_WORLD

    return (
        DEFAULT_WORLD,
        GRPOConfig,
        PPOConfig,
        cite,
        exact,
        group_advantages,
        grpo,
        grpo_loss,
        kl_estimator,
        load_reference,
        mo,
        np,
        plots,
        ppo,
        reveal,
        scorer,
        shaped_rewards,
        source_md,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 02 · GRPO, as a diff against PPO

    ## Where we are

    Notebook 01 built PPO in its RLHF form and found that it aims at $\pi^*_\beta$: its
    return is notebook 00's $J_\beta$, so the gap $\beta\,\mathrm{KL}(\pi \,\|\, \pi^*_\beta)$
    is the right yardstick for it. Its price is the critic, in RLHF a second large network.
    GRPO keeps PPO's sampling loop, ratio and clip, and changes three things:

    | | PPO (notebook 01) | GRPO |
    |---|---|---|
    | baseline | a learned critic $V(s_t)$ | the mean reward of a group of $G$ responses to the same prompt |
    | advantage | one per token, from GAE | one per response, standardised within its group, copied to each of its tokens |
    | KL to $\pi_{\text{ref}}$ | folded into the reward, $-\beta \log(\pi_{\text{old}}/\pi_{\text{ref}})$ per token | a separate term in the loss, with its own estimator |

    The first change is a cheaper way to do the same thing. The other two are not mere
    rearrangements: this notebook shows, by derivation and by exact measurement, that
    GRPO as published does not maximise $J_\beta$, and on this world drifts further from
    $\pi_{\text{ref}}$ than any $\pi^*_\beta$ does. Two switches undo the two departures
    one at a time, so each one's effect can be seen. Notation is notebook 00's.
    """)
    return


@app.cell
def _(cite, mo):
    _def = mo.md(
        r"""
        ## Definition

        **Groups.** For each prompt $x$, sample $G$ responses
        $y_1, \dots, y_G \sim \pi_{\text{old}}(\cdot \mid x)$ and score them,
        $r_i = r(x, y_i)$. As in PPO, $\pi_{\text{old}}$ is the policy that drew the batch,
        refreshed at every iteration and fixed while the batch is reused.

        **The advantage.** Each response gets one number, its reward standardised within
        its group, and every token of $y_i$ gets that same number:

        $$
        \hat A_{i,t} = \hat A_i = \frac{r_i - \mathrm{mean}(r_1, \dots, r_G)}{\mathrm{std}(r_1, \dots, r_G)}.
        $$

        The code uses the sample standard deviation and adds $10^{-6}$ to it, so a group
        whose rewards all tie gets $\hat A = 0$ (the paper specifies neither). The group mean
        includes $r_i$ itself, so, without the division, the expected update from the
        advantage is $(G - 1)/G$ times the policy gradient of what the group averages
        ($\mathbb{E}[r]$, or $J_\beta$ with the KL in the reward; 0.875 at $G = 8$), the
        analogue of the $(1 - 1/B)$ factor notebook 00 states for REINFORCE's batch mean.

        **The objective.** With the per-token ratio
        $\rho_{i,t} = \pi_\theta(y_{i,t} \mid x, y_{i,<t}) / \pi_{\text{old}}(y_{i,t} \mid x, y_{i,<t})$,
        GRPO maximises

        $$
        J_{\text{GRPO}}(\theta) = \mathbb{E}\Big[\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|}
        \Big\{ \min\big(\rho_{i,t}\, \hat A_{i,t},\ \mathrm{clip}(\rho_{i,t}, 1 - \varepsilon, 1 + \varepsilon)\, \hat A_{i,t}\big)
        - \beta\, \mathbb{D}_{\text{KL}}\big[\pi_\theta \,\|\, \pi_{\text{ref}}\big] \Big\}\Big],
        $$

        where the KL is not computed but estimated on each sampled token,

        $$
        \mathbb{D}_{\text{KL}}\big[\pi_\theta \,\|\, \pi_{\text{ref}}\big]
        = \frac{\pi_{\text{ref}}(y_{i,t} \mid x, y_{i,<t})}{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}
        - \log \frac{\pi_{\text{ref}}(y_{i,t} \mid x, y_{i,<t})}{\pi_\theta(y_{i,t} \mid x, y_{i,<t})} - 1.
        $$

        Every response here has $|y_i| = T = 5$ tokens, so the double average is the mean
        over all $P \cdot G \cdot T$ tokens of the batch ($P = 4$ prompts, one group each).
        There is no value loss: nothing else is trained.

        **"As published", and where this notebook differs.** The simulation's defaults are
        the paper's algorithm (division on, KL in the loss) with two differences. The paper
        reports a KL coefficient of 0.04 (the OLMo 2 1B GRPO runs used 0.01); the default here
        is $\beta = 0.1$, the $\beta$ notebooks 00 and 01 use, so that the gap is measured
        against the same $\pi^*_{0.1}$. And the paper's run takes a single policy update per
        sampling stage ("the policy model only has a single update following each
        exploration stage", §4.2); the default here is 4 epochs per batch, PPO's default, for
        comparability with notebook 01. Both are controls below, and the measurements under
        "What GRPO aims at" are given at 1 epoch as well as at 4. With one update the
        ratio is exactly 1 at the only step taken, so the clip never binds: GRPO is then a
        policy gradient with group-standardised advantages plus the KL estimator's gradient.
        The objective, advantage, estimator, $\beta = 0.04$ and the single update are from
        the first reference; the OLMo 2 runs' $\beta$ from the second's training commands:
        """
    )
    mo.vstack([_def, mo.md(cite("shao2024", "olmo2024"))])
    return


@app.cell
def _(cite, mo):
    _a = mo.md(
        r"""
        ### What GRPO aims at

        **(a) The estimator's value and its gradient point at different KLs.** Fix one state
        $s_t = (x, y_{<t})$ and write $u(a) = \pi_{\text{ref}}(a \mid s_t)/\pi_\theta(a \mid s_t)$.
        Its *value* is right: $\mathbb{E}_{a \sim \pi_\theta}[u - \log u - 1]
        = \sum_a \pi_{\text{ref}}(a) - 1 + \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})
        = \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$, the direction $J_\beta$ penalises.
        But in the loss the token $a$ is a fixed sample, and only $\theta$ moves. Since
        $\nabla u = -u\, \nabla \log \pi_\theta(a)$:

        $$
        \begin{aligned}
        \nabla \big(u - \log u - 1\big) &= \nabla u - \frac{\nabla u}{u}
          = \big(1 - u(a)\big)\, \nabla \log \pi_\theta(a), \\
        \mathbb{E}_{a \sim \pi_\theta}\Big[\big(1 - u(a)\big) \nabla \log \pi_\theta(a)\Big]
          &= \sum_a \nabla \pi_\theta(a) - \sum_a \pi_{\text{ref}}(a)\, \nabla \log \pi_\theta(a) \\
          &= 0 + \nabla\, \mathrm{KL}\big(\pi_{\text{ref}} \,\|\, \pi_\theta\big).
        \end{aligned}
        $$

        So on one next-token distribution the term descends $\mathrm{KL}(\pi_{\text{ref}} \,\|\, \pi_\theta)$,
        the *other* direction. The samples come from $\pi_{\text{old}}$, not $\pi_\theta$, so
        the expectation above is exact at the first epoch on each batch, where
        $\pi_{\text{old}} = \pi_\theta$ (at every step when there is one epoch per batch);
        later epochs add the usual off-policy mismatch. The test suite checks both statements exactly, by enumeration
        over one distribution (`tests/test_grpo.py`). The paper's appendix gives the same
        gradient coefficient, $\hat A_{i,t} + \beta\,(\pi_{\text{ref}}/\pi_\theta - 1)$, without
        drawing this conclusion.

        Over a whole response there is one more step. Summed over its tokens, the expected
        gradient (again at $\pi_{\text{old}} = \pi_\theta$, with $\nabla$ acting on the
        conditionals $\pi_\theta(\cdot \mid s_t)$ and the distribution of the states held
        fixed, since the samples are fixed) is $\sum_t \mathbb{E}_{s_t \sim \pi_\theta}\big[\nabla\, \mathrm{KL}(\pi_{\text{ref}}(\cdot \mid s_t) \,\|\, \pi_\theta(\cdot \mid s_t))\big]$:
        the reverse KL at each state, weighted by how often *$\pi_\theta$* reaches that state.
        The gradient of the sequence-level $\mathrm{KL}(\pi_{\text{ref}} \,\|\, \pi_\theta)$
        weights the states by $\pi_{\text{ref}}$ instead. So for responses longer than one
        token, the term is the gradient of neither sequence-level KL. Measured exactly, by
        enumerating all 32,768 responses of each prompt, at the final policy of the default
        PPO run (seed 0), the cosine between the term's expected gradient and the gradient of
        $\mathrm{KL}(\pi_{\text{ref}} \,\|\, \pi_\theta)$ is 0.87, and with that of
        $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$ 0.92; at the final policy of
        published GRPO (seed 0), 0.73 and 0.68. Its expected *value* is still exactly
        $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$ of the whole response (0.6217 and
        5.7746 at those two policies, equal to the exact drift to four decimals).

        **(b) The division changes the balance between reward and KL, group by group.**
        With the KL outside the advantage, dividing the advantage by the group's standard
        deviation $\sigma$ scales the reward term by $1/\sigma$ and leaves the KL term alone.
        Multiplying the whole per-group objective by $\sigma$ shows the balance: reward
        differences $r_i - \mathrm{mean}(r)$ against a KL weight of $\beta\sigma$. So the
        effective $\beta$ is $\beta\sigma$, different for every prompt and every batch, and it
        falls as the policy improves, because a group of good responses has a small spread.
        Measured in published GRPO (a copy of `grpo.train` that logs $\sigma$ and reproduces
        its metrics exactly; seeds 0–2): the median $\sigma$ over groups with a non-zero
        spread is 0.12–0.14 in iterations 1–10 and 0.020–0.022 in iterations 141–150, so the
        effective $\beta$ goes from about 0.013 to about 0.002 while the nominal one is 0.1.
        At 1 epoch the median $\sigma$ is 0.15 in iterations 1–10 and 0.030–0.032 in
        iterations 141–150 (effective $\beta$ about 0.015, then 0.003).
        """
    )
    _c = mo.md(
        r"""
        **(c) Consequence: GRPO as published does not maximise $J_\beta$.** The theory for
        each switch:

        - *KL in the reward* (as PPO), with or without the division: the regularised return
          $R_i = r_i - \beta \log(\pi_{\text{old}}(y_i)/\pi_{\text{ref}}(y_i))$ is, at
          $\pi^*_\beta$, the same number $\beta \log Z_\beta(x)$ for every response
          (notebook 00's closed form), so every advantage is zero whatever it is divided by.
          $\pi^*_\beta$ is a fixed point, and away from it the division only rescales each
          group's step by a positive number (only roughly, since $\sigma$ is estimated from
          the same samples). Expected: the gap closes, as for PPO.
        - *KL in the loss, no division*: the reward term's expected gradient is
          $(G - 1)/G \cdot \nabla \mathbb{E}[r]$ and the KL term's is the per-state reverse
          KL of (a). That is not $\nabla J_\beta$, so its fixed point is not $\pi^*_\beta$.
          Expected: the gap levels off above zero.
        - *As published* (both): there is not even a fixed objective, since the balance of
          (b) moves with every batch. The expectation this project started from: the gap
          levels off above zero.

        Measured on the default world: $\beta = 0.1$, graded reward, $G = 8$, $\varepsilon = 0.2$,
        150 iterations, seeds 0–2, gap and drift at the last iteration as min–max over the
        seeds. The gap starts at 0.1022; $\pi^*_{0.1}$ has drift 0.642; the drift ceiling
        (notebook 00) is 3.44.

        | division | KL | gap, 4 epochs | drift, 4 epochs | gap, 1 epoch | drift, 1 epoch |
        |---|---|---|---|---|---|
        | on | in the loss (**published**) | **0.373–0.437** | **5.12–5.77** | **0.225–0.230** | **3.49–3.53** |
        | on | in the reward | 0.0198–0.0258 | 0.71–0.76 | 0.0137–0.0140 | 0.57–0.62 |
        | off | in the loss | 0.0341–0.0392 | 0.49–0.61 | 0.0236–0.0249 | 0.41–0.46 |
        | off | in the reward (PPO-like) | 0.0175–0.0192 | 0.70–0.75 | 0.0134–0.0139 | 0.55–0.56 |
        | PPO (notebook 01, 4 epochs) | | 0.0145–0.0157 | 0.62–0.63 | | |

        The two "KL in the reward" rows close the gap, as predicted: with the division on,
        to 19–25% of its start at 4 epochs (seed 2 ends just above a quarter) and 13–14% at
        1 epoch. "KL in the loss" without the division levels off above them, as
        predicted: at 4 epochs its gap is at 0.030–0.032 by its minimum and ends at
        0.034–0.039. **As published, the prediction fails, and in the worse direction.** The
        gap does not level off: it falls to 0.063–0.064 (iterations 4–5), is back above its
        start by iteration 12–13 and ends at 3.6–4.3 times its start. The drift crosses the
        drift ceiling at iterations 24–37 and ends at 5.1–5.8. Every $\pi^*_\beta$, for every
        $\beta$, has drift below the ceiling, so no regularised optimum, however small its
        $\beta$, is where this run ends; "the division makes $\beta$ smaller", which (b) shows
        is true, cannot by itself explain it. The final policy puts 60–85% of its mass on the
        maximum-reward responses but has an exponentiated entropy of only 2.8–11.9 responses
        per prompt, where those responses number 330 to 2,010 per prompt and $\pi^*_{0.1}$
        spreads over 430–850. It has piled onto a few of the best responses. Only the two
        departures together do this: with either switch alone the run ends with drift below 0.8
        and an exponentiated entropy above 300.

        **Does the collapse depend on reusing the batch?** No. With one update per batch, as
        in the paper, published GRPO still falls and turns: its gap reaches 0.043–0.051 at
        iterations 22–28, is back above its start at iterations 65–71, and its drift crosses
        the ceiling at iterations 145–146, still rising at the end. At 4 epochs the ceiling
        is crossed after 96–148 gradient steps, at 1 epoch after 145–146: per gradient step
        the pace is similar, and reuse only packs four steps into each batch. Run on to 600
        iterations at 1 epoch (600 gradient steps, the same as the default's), it ends at a
        gap of 0.33–0.44 and drift 4.70–5.83, near the 4-epoch results. The paper's own
        $\beta = 0.04$ with one update also passes the ceiling: drift 3.99–4.31 after 150
        iterations, gap 0.086–0.100 against $\pi^*_{0.04}$'s start of 0.160, and rising
        again after its minimum.

        **The KL term is there, with the right sign.** Sweeping $\beta$ on published GRPO
        (4 epochs, seeds 0–2), the final drift falls as $\beta$ rises: 6.9–7.8 at
        $\beta = 0.01$, 6.7–7.4 at 0.04, 5.1–5.8 at 0.1, 2.2–2.3 at 0.3, 0.42–0.52 at 1. But
        at every $\beta$ it is far above that $\beta$'s optimum (2.98, 1.46, 0.64, 0.13, 0.014),
        and from $\beta = 0.04$ up the gap ends above its start: 0.160 to 0.180–0.205 at
        0.04, 0.102 to 0.373–0.437 at 0.1, 0.044 to 0.49–0.54 at 0.3, 0.014 to 0.33–0.41 at 1.
        At $\beta = 0.01$ it ends below its start, 0.221 to 0.037–0.047. A likely reading,
        not separately measured: $\pi^*_{0.01}$ itself has drift 2.98, close to the ceiling,
        so a policy piled onto maximum-reward responses is not far from it. Group size does not remove it either: final drift
        3.1–3.8 at $G = 2$, 4.8–5.4 at 4, 4.8–6.5 at 16, 4.9–6.1 at 32.

        **A hypothesis, not a measured mechanism.** Near the top, graded rewards differ in
        steps of 1/35. The division turns a difference of one step into an advantage of order
        one, which pushes up whichever of the many maximum-reward responses happened to be
        sampled; the KL term weighs against it only as much as a reward difference of
        $\beta\sigma$, about 0.002 late in the run. Nothing in the reward prefers one
        maximum-reward response over another, so the policy can keep sharpening onto a few
        of them at no cost in reward. What supports this: each switch alone removes the
        collapse; the effective $\beta$ falls as in (b); a larger $\beta$ restrains it. What
        it is not: it is not batch reuse (1 epoch collapses too) and not small groups (up to
        $G = 32$ collapses too). None of these measurements isolates the sharpening step
        itself.
        """
    )
    mo.vstack([_a, mo.md(cite("shao2024")), _c])
    return


@app.cell
def _(DEFAULT_WORLD, exact, load_reference, np, scorer, torch):
    ref = load_reference(DEFAULT_WORLD)  # outside no_grad: the first call trains it
    with torch.no_grad():
        logp_ref = scorer(ref).all_logp()  # [P, N]
    rewards_by_mode = {m: DEFAULT_WORLD.reward_table(m) for m in ("graded", "verifiable")}
    _betas = [float(b) for b in np.logspace(-2, 1, 40)]
    frontiers = {m: exact.frontier(logp_ref, r, _betas) for m, r in rewards_by_mode.items()}
    ceilings = {m: exact.drift_ceiling(logp_ref, r).item() for m, r in rewards_by_mode.items()}
    return ceilings, frontiers, logp_ref, ref, rewards_by_mode


@app.cell
def _(DEFAULT_WORLD, grpo, mo, ppo, ref):
    @mo.cache
    def train_grpo(cfg):
        """One GRPO run on the default world, cached by its (frozen) config."""
        return grpo.train(ref, DEFAULT_WORLD, cfg)

    @mo.cache
    def train_ppo(cfg):
        """One PPO run on the default world, cached by its (frozen) config."""
        return ppo.train(ref, DEFAULT_WORLD, cfg)

    return train_grpo, train_ppo


@app.cell
def _(mo):
    g_view = mo.ui.slider(steps=[2, 4, 8, 16, 32], value=8, label="group size G",
                          show_value=True)
    return (g_view,)


@app.cell
def _(DEFAULT_WORLD, PPOConfig, g_view, grpo, mo, plots, ref, train_ppo):
    _ppo_cfg = PPOConfig()
    _policy = train_ppo(_ppo_cfg).policy
    _cmp = grpo.baseline_comparison(_policy, ref, DEFAULT_WORLD, g_view.value, _ppo_cfg.beta,
                                    "graded", seed=0)
    _rmse = {k: ((_cmp[k] - _cmp["exact"]) ** 2).mean().sqrt().item() for k in ("group", "critic")}
    _true = "exact advantage R − J_β(x)"
    mo.vstack(
        [
            mo.md(
                r"""
                ## Intuition

                **The group is a cheap, critic-free baseline.** An advantage is a return
                minus a baseline, and the usual choice of baseline for a response to $x$ is
                the expected return of $x$ under the current policy, $J_\beta$ of that prompt
                (the value function; it is not exactly the variance-minimising baseline, but
                it is what a critic learns).
                PPO learns it with a critic. GRPO estimates it by sampling $G$ responses to
                the same prompt and averaging: no extra network, at the price of $G$ samples
                per prompt and an estimate whose error shrinks only like $1/\sqrt{G}$.

                Below, both are compared with the truth on the same samples. The policy is the
                final policy of the default PPO run of notebook 01 (`PPOConfig()`: $\beta = 0.1$,
                graded reward, 150 iterations, seed 0), whose critic started at zero and was
                trained by that run; it is the same run as the PPO comparison in the simulation
                at its defaults. One group of $G$ responses per prompt is drawn from it
                (sampling seed 0). For each response, $R = r - \beta \log(\pi/\pi_{\text{ref}})$
                minus: the exact $J_\beta$ of its prompt (horizontal axis), its group's mean
                $R$ (left), and the critic's $V$ at the first position (right). Points on the
                dashed line are exact. No standard-deviation division here: this is only the
                baseline.
                """
            ),
            g_view,
            mo.hstack(
                [
                    plots.scatter_true_vs_estimate(
                        _cmp["exact"], _cmp["group"], xlabel=_true,
                        ylabel="R − group mean of R",
                    ),
                    plots.scatter_true_vs_estimate(
                        _cmp["exact"], _cmp["critic"], xlabel=_true,
                        ylabel="R − critic V(x)",
                    ),
                ],
                justify="start",
            ),
            mo.md(
                f"At G = {g_view.value}, on these samples: root-mean-square error against the "
                f"exact advantage {_rmse['group']:.4f} for the group mean, {_rmse['critic']:.4f} "
                "for the critic."
            ),
            mo.md(
                r"""
                Measured over 20 sampling seeds (0–19), the mean root-mean-square error of the
                group baseline is 0.037 at $G = 2$, 0.023 at 4, 0.015 at 8, 0.012 at 16 and
                0.0085 at 32; the critic's is 0.0060 at every $G$ and seed, since its error is
                one fixed number per prompt (its $V$ is 0.803–0.856 against true values
                0.811–0.855). The exact advantages themselves have a standard deviation of
                about 0.03–0.06. So, averaged over 20 sampling seeds, this critic, trained for
                150 iterations on a toy world, is more accurate than the group at every $G$ up
                to 32; single draws of a group can still beat it (RMSE as low as 0.0043 at
                $G = 4$ and 0.0044 at $G = 32$). The group mean wins on cost, not accuracy. A critic for an LLM is a second network the size of the policy, and
                its accuracy is not known in advance.
                """
            ),
        ]
    )
    return


@app.cell
def _(group_advantages, grpo_loss, kl_estimator, mo, shaped_rewards, source_md):
    mo.vstack(
        [
            mo.md(
                r"""
                ## The code

                Four functions are the whole algorithm; `grpo.train` samples one group per
                prompt, calls them, and steps Adam.

                **The group advantage**,
                $\hat A_i = (r_i - \mathrm{mean}(r)) / (\mathrm{std}(r) + 10^{-6})$, or
                $r_i - \mathrm{mean}(r)$ with the division switched off (`rewards` is
                $[P, G]$, one row per group):
                """
            ),
            mo.md(source_md(group_advantages)),
            mo.md(
                r"""
                **The KL estimator**, per token,
                $\pi_{\text{ref}}/\pi_\theta - \log(\pi_{\text{ref}}/\pi_\theta) - 1$, computed
                from the log-ratio $d = \log \pi_{\text{ref}} - \log \pi_\theta$ as
                $e^d - d - 1$:
                """
            ),
            mo.md(source_md(kl_estimator)),
            mo.md(
                r"""
                **The loss**, $-J_{\text{GRPO}}$ on one batch: the clipped surrogate with each
                response's advantage on all its tokens, minus $\beta$ times the estimator, both
                averaged over all $P \cdot G \cdot T$ tokens:
                """
            ),
            mo.md(source_md(grpo_loss)),
            mo.md(
                r"""
                **The KL switch.** With the KL "in the reward", the group sees PPO's
                regularised return $r - \beta \sum_t (\log \pi_{\text{old}}(y_t) - \log \pi_{\text{ref}}(y_t))$
                (notebook 01's per-token rewards, summed) and the loss has no KL term; "in the
                loss" leaves $r$ alone:
                """
            ),
            mo.md(source_md(shaped_rewards)),
            mo.md(
                r"""
                A unit test checks that a group whose rewards all tie gets advantages of
                exactly zero, so it contributes no gradient through the surrogate; with the KL
                in the loss, its responses still feel the KL term. `extras["dead_groups"]`
                in the simulation counts such groups, by their task rewards.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    g_run = mo.ui.slider(steps=[2, 4, 8, 16, 32], value=8, label="group size G",
                         show_value=True)
    clip_run = mo.ui.slider(steps=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0], value=0.2, label="clip ε",
                            show_value=True)
    beta_run = mo.ui.slider(steps=[0.01, 0.02, 0.04, 0.1, 0.2, 0.5, 1.0], value=0.1,
                            label="β", show_value=True)
    epochs_run = mo.ui.slider(start=1, stop=8, step=1, value=4, label="epochs per batch",
                              show_value=True)
    mode_run = mo.ui.radio(["graded", "verifiable"], value="graded", label="reward",
                           inline=True)
    std_run = mo.ui.checkbox(value=True, label="divide by the group standard deviation")
    kl_run = mo.ui.radio(
        {"in the loss (paper)": "loss", "in the reward (as PPO)": "reward"},
        value="in the loss (paper)",
        label="KL term",
        inline=True,
    )
    seed_run = mo.ui.number(start=0, stop=99, step=1, value=0, label="seed")
    return beta_run, clip_run, epochs_run, g_run, kl_run, mode_run, seed_run, std_run


@app.cell
def _(
    GRPOConfig,
    PPOConfig,
    beta_run,
    clip_run,
    epochs_run,
    g_run,
    kl_run,
    mo,
    mode_run,
    seed_run,
    std_run,
    train_grpo,
    train_ppo,
):
    grpo_cfg = GRPOConfig(
        beta=beta_run.value,
        clip=clip_run.value,
        group_size=g_run.value,
        epochs=epochs_run.value,
        std_normalize=std_run.value,
        kl_in=kl_run.value,
        reward_mode=mode_run.value,
        seed=int(seed_run.value),
    )
    ppo_cfg = PPOConfig(beta=grpo_cfg.beta, reward_mode=grpo_cfg.reward_mode, seed=grpo_cfg.seed)
    grpo_run = train_grpo(grpo_cfg)
    ppo_run = train_ppo(ppo_cfg)
    mo.vstack(
        [
            mo.md(
                rf"""
                ## Simulation

                GRPO from $\pi_{{\text{{ref}}}}$: {grpo_cfg.iterations} iterations, each one
                group of $G$ responses for each of the 4 prompts, reused for `epochs`
                full-batch Adam steps (learning rate {grpo_cfg.lr:g}). The defaults are GRPO as
                published, except $\beta$ and the epochs (see "Definition"). Next to it, PPO
                from notebook 01 at the same $\beta$, reward and seed, with its own defaults
                otherwise (batch of {ppo_cfg.batch_size}, {ppo_cfg.epochs} epochs, ε =
                {ppo_cfg.clip}). A control change retrains GRPO (2 to 4 s; slowest at
                $G = 32$ with 8 epochs) and, if $\beta$, the reward or the seed changed, PPO
                (about 3 s); both are cached, so returning to a setting is instant. Reward,
                both KL directions and the gap are exact, after every iteration.
                """
            ),
            mo.hstack([g_run, clip_run, beta_run], justify="start"),
            mo.hstack([epochs_run, seed_run], justify="start"),
            mo.hstack([std_run, kl_run], justify="start"),
            mode_run,
        ]
    )
    return grpo_cfg, grpo_run, ppo_cfg, ppo_run


@app.cell
def _(
    DEFAULT_WORLD,
    ceilings,
    exact,
    frontiers,
    grpo_cfg,
    grpo_run,
    logp_ref,
    mo,
    plots,
    ppo_run,
    rewards_by_mode,
):
    _reward = rewards_by_mode[grpo_cfg.reward_mode]
    _opt = exact.optimum(logp_ref, _reward, grpo_cfg.beta)
    _target = (exact.kl(_opt, logp_ref).item(), exact.expected_reward(_opt, _reward).item())
    _ceiling = ceilings[grpo_cfg.reward_mode]
    _runs = {"grpo": grpo_run, "ppo": ppo_run}
    _dead = grpo_run.extras["dead_groups"]

    def _summary(name, run):
        m = run.metrics
        return (
            f"| {name} | {run.steps[-1]:,} | {run.feedback[-1]:,} | {m['reward'][-1]:.3f} "
            f"| {m['kl'][-1]:.3f} | {m['kl_rev'][-1]:.3f} "
            f"| {m['gap'][0]:.4f} → {m['gap'][-1]:.4f} |"
        )

    with plots.styled_figure(width=5.0, height=3.2) as (_dead_fig, (_ax,)):
        _ax.plot(range(1, len(_dead) + 1), _dead, color=plots.COLORS["grpo"], lw=1.4)
        _ax.set_ylim(-0.2, DEFAULT_WORLD.n_prompts + 0.2)
        _ax.set_xlabel("iteration")
        _ax.set_ylabel("dead groups (of 4)")

    _kl_end = grpo_run.metrics["kl"][-1]
    _where = (
        f"above the drift ceiling, by {_kl_end - _ceiling:.2f}: no π*_β of any β is there"
        if _kl_end > _ceiling
        else f"below the drift ceiling ({_kl_end / _ceiling:.0%} of it)"
    )
    mo.vstack(
        [
            mo.md(
                r"""
                **Against gradient steps**, with both KL directions: $\mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})$
                is the drift $J_\beta$ penalises, $\mathrm{KL}(\pi_{\text{ref}} \,\|\, \pi)$ the
                direction the estimator's gradient leans towards (a).
                """
            ),
            plots.training_curves(_runs, keys=("reward", "kl", "kl_rev", "gap")),
            mo.md(
                rf"""
                **Against units of feedback** (one reward call per sampled response). GRPO
                draws $4G$ = {4 * grpo_cfg.group_size} responses per iteration,
                {grpo_run.feedback[-1]:,} in all; PPO draws 64, {ppo_run.feedback[-1]:,} in all.
                """
            ),
            plots.training_curves(_runs, keys=("reward", "kl", "kl_rev", "gap"), x="feedback"),
            mo.md(
                "| run | gradient steps | reward calls | E[r] | KL(π ‖ π_ref) | KL(π_ref ‖ π) | gap |\n"
                "|---|---|---|---|---|---|---|\n"
                + _summary("grpo", grpo_run) + "\n" + _summary("ppo", ppo_run) + "\n"
                + f"| π*_β, β = {grpo_cfg.beta} | | | {_target[1]:.3f} | {_target[0]:.3f} | | 0 |"
            ),
            mo.md(
                f"""
                **Dead groups.** Groups whose {grpo_cfg.group_size} task rewards all
                tie. With the KL in the loss their advantages are 0, so their samples teach
                the surrogate nothing (only the KL term acts on them); with the KL in the
                reward the shaped returns of such a group still differ by their KL terms,
                so it is not dead to the surrogate after the first iteration.
                {sum(_dead)} of {len(_dead) * DEFAULT_WORLD.n_prompts} groups in this run
                ({sum(_dead) / (len(_dead) * DEFAULT_WORLD.n_prompts):.0%}).
                """
            ),
            _dead_fig,
            mo.md(
                r"""
                **The ratios at the last gradient step** (a snapshot of the last batch at its
                last epoch, not the whole run). With one epoch they are all exactly 1 and the
                clip is inert.
                """
            ),
            plots.ratio_hist(grpo_run.extras["ratios"], grpo_cfg.clip),
            plots.frontier_plot(
                frontiers[grpo_cfg.reward_mode],
                _runs,
                targets={f"π*_β, β = {grpo_cfg.beta}": _target},
                ceiling=_ceiling,
            ),
            mo.md(
                f"The frontier of the {grpo_cfg.reward_mode} reward; the dotted line is its "
                f"drift ceiling, the largest drift of any π*_β. This GRPO run ends at drift "
                f"{_kl_end:.2f}, {_where}. At the defaults (published, β = 0.1, seeds 0–2) it "
                "ends at 5.1–5.8, past the ceiling and below the frontier: more drift than any "
                "optimum, for less reward than the frontier's maximum (the reward at the ceiling)."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    side_button = mo.ui.run_button(label="Run 3 seeds × (GRPO published, GRPO PPO-like, PPO)")
    mo.vstack(
        [
            mo.md(
                r"""
                **Side by side.** GRPO as published, GRPO with both switches PPO-like
                (no division, KL in the reward), and PPO, three seeds each (0, 1, 2), all at
                their defaults ($\beta = 0.1$, graded reward, 4 epochs; GRPO with $G = 8$),
                whatever the controls above say. Bands are min to max over the seeds.
                """
            ),
            side_button,
        ]
    )
    return (side_button,)


@app.cell
def _(GRPOConfig, PPOConfig, mo, plots, side_button, train_grpo, train_ppo):
    mo.stop(not side_button.value, mo.md("*Press the button to train the nine runs (about 25 s).*"))
    _seeds = (0, 1, 2)
    _groups = {
        "grpo": [train_grpo(GRPOConfig(seed=_s)) for _s in _seeds],
        "grpo_ppo_like": [
            train_grpo(GRPOConfig(std_normalize=False, kl_in="reward", seed=_s)) for _s in _seeds
        ],
        "ppo": [train_ppo(PPOConfig(seed=_s)) for _s in _seeds],
    }
    _keys = ("reward", "kl", "kl_rev", "gap")
    mo.vstack(
        [
            mo.md("Against gradient steps:"),
            plots.training_curves(_groups, keys=_keys),
            mo.md("Against units of feedback:"),
            plots.training_curves(_groups, keys=_keys, x="feedback"),
        ]
    )
    return


@app.cell
def _(cite, mo):
    _note = mo.md(r"""
    **What carries over to real LLMs, and what does not.** The *shape* carries over: one
    group of samples per prompt, one standardised advantage per response copied to its
    tokens, a per-token ratio and clip, and the KL estimator added to the loss; the
    derivation in (a) holds for any autoregressive policy. The settings do not: DeepSeekMath
    used $\beta = 0.04$ and 64 samples per question with a single update per batch,
    DeepSeek-R1-Zero $\beta = 0.001$ and 16 samples per question, with its reference
    model replaced by the latest policy every 400 steps (so its KL term holds the policy
    near a recent copy of itself, not near the starting model as here), the OLMo 2 1B runs
    $\beta = 0.01$ and 16. What does not carry over: here the response space is enumerable, so
    the drift, the gap and the drift ceiling are exact, and the collapse onto a few
    maximum-reward responses can be seen; for an LLM none of these can be computed. The
    collapse itself is a measured fact about this world: a reward with a large set of tied
    best responses (330 to 2,010 per prompt) and steps of 1/35 near the top, a small GRU,
    and these learning rates. It is not a claim that GRPO collapses LLMs. What does carry
    over is the reason to look: GRPO as published is not maximising $J_\beta$, so a low KL
    coefficient in its loss is not a promise about how far the policy will drift.
    """)
    mo.vstack([_note, mo.md(cite("shao2024", "guo2025", "olmo2024"))])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to try

    Make a prediction before opening each answer. The answers report what was measured on
    the machine this notebook was written on: seeds 0, 1 and 2, all controls at their
    defaults except the ones named, values read at the last iteration and given as the
    range over the three seeds (min–max) unless stated otherwise.
    """)
    return


@app.cell
def _(mo, reveal):
    switch_button = mo.ui.run_button(label="Run 3 seeds × 4 switch settings")
    switch_epochs = mo.ui.radio({"4 epochs": 4, "1 epoch": 1}, value="4 epochs",
                                label="epochs", inline=True)
    mo.vstack(
        [
            reveal(
                "**1. One switch at a time.** Turn off the division alone, then put the KL "
                "in the reward alone, then both. Which settings close the gap, and does "
                "anything change with one epoch per batch?",
                "Measured, 150 iterations, gap at the last iteration (start 0.1022; PPO "
                "0.0145–0.0157). **4 epochs:** published 0.373–0.437 (drift 5.12–5.77); "
                "division on, KL in the reward 0.0198–0.0258 (0.71–0.76); division off, KL in "
                "the loss 0.0341–0.0392 (0.49–0.61); both PPO-like 0.0175–0.0192 (0.70–0.75). "
                "**1 epoch:** 0.225–0.230 (3.49–3.53); 0.0137–0.0140 (0.57–0.62); "
                "0.0236–0.0249 (0.41–0.46); 0.0134–0.0139 (0.55–0.56). The theory's ranking "
                "holds in both: KL in the reward closes the gap with or without the division "
                "(though with it on, seed 2 at 4 epochs ends at 25% of the start, not below "
                "it); KL in the loss without the division levels off above them. Published "
                "GRPO is the exception the theory did not predict: its gap rises past its start "
                "and its drift past the ceiling (3.44) at both epoch settings. At 1 epoch it "
                "is slower per batch, not different in kind (see \"Does the collapse depend on "
                "reusing the batch?\" above). One result goes the other way from notebook 01: "
                "for the three non-collapsing settings, 1 epoch ends *better* than 4 after "
                "150 iterations, and the two KL-in-the-reward settings at 1 epoch end below "
                "PPO's 4-epoch gap, with half its reward calls (4,800 against 9,600). Nothing "
                "was tuned for this; both algorithms use their default learning rate, 1e-3.",
            ),
            mo.hstack([switch_epochs, switch_button], justify="start"),
        ]
    )
    return switch_button, switch_epochs


@app.cell
def _(GRPOConfig, mo, plots, switch_button, switch_epochs, train_grpo):
    mo.stop(not switch_button.value, mo.md("*Press the button to train the twelve runs (about 30 s).*"))
    _settings = {
        "published (÷σ, KL in loss)": {"std_normalize": True, "kl_in": "loss"},
        "÷σ, KL in reward": {"std_normalize": True, "kl_in": "reward"},
        "no ÷σ, KL in loss": {"std_normalize": False, "kl_in": "loss"},
        "PPO-like (no ÷σ, KL in reward)": {"std_normalize": False, "kl_in": "reward"},
    }
    plots.training_curves(
        {
            _name: [
                train_grpo(GRPOConfig(epochs=switch_epochs.value, seed=_s, **_kw))
                for _s in (0, 1, 2)
            ]
            for _name, _kw in _settings.items()
        },
        keys=("reward", "kl", "gap"),
    )
    return


@app.cell
def _(mo, reveal):
    dead_button = mo.ui.run_button(label="Run 3 seeds × G ∈ {2, 16}, verifiable reward")
    mo.vstack(
        [
            reveal(
                "**2. Dead groups.** Switch the reward to *verifiable* (1 for an exact sum, "
                "0 otherwise; π_ref scores 1 on 3.3% of samples) and compare G = 2 with "
                "G = 16. How many groups are dead, and what does that cost?",
                "Measured, published GRPO, 150 iterations. **G = 2:** 70–89% of all groups "
                "over the run are dead (85–93% in the first 10 iterations, close to the "
                "chance that two samples from π_ref both score 0, 0.967² ≈ 0.94). Most "
                "samples teach nothing, and learning barely starts: expected reward ends at "
                "0.08–0.41 (from 0.033) and the gap at 0.51–0.65 (start 0.623), after 1,200 "
                "reward calls. **G = 16:** 10–12% dead over the run (30–53% in the first 10 "
                "iterations); expected reward ends at 0.86–0.88 and the gap at 0.26–0.31, "
                "after 9,600 calls, with drift 4.80–5.22, past the ceiling again. **At equal "
                "feedback** the comparison is closer than the dead fractions suggest: at "
                "1,152 calls the gap is 0.51–0.66 (G = 2, iteration 144) against 0.44–0.51 "
                "(G = 16, iteration 18). Run on to 1,200 iterations (9,600 calls, set in "
                "code; not reachable from the controls), G = 2 is still 58–70% dead and ends "
                "at a gap of 0.43–0.45, against 0.26–0.31 for G = 16. So the expected "
                "picture holds: small groups under a rare 0/1 reward waste most of their "
                "samples, and at equal feedback they end further from the optimum.",
            ),
            dead_button,
        ]
    )
    return (dead_button,)


@app.cell
def _(GRPOConfig, dead_button, mo, plots, train_grpo):
    mo.stop(not dead_button.value, mo.md("*Press the button to train the six runs (about 17 s).*"))
    plots.training_curves(
        {
            f"G = {_g}": [
                train_grpo(GRPOConfig(group_size=_g, reward_mode="verifiable", seed=_s))
                for _s in (0, 1, 2)
            ]
            for _g in (2, 16)
        },
        x="feedback",
    )
    return


@app.cell
def _(reveal):
    reveal(
        "**3. Pairs under the graded reward.** With G = 2 and the division on, what is the "
        "advantage of each response in a pair, as a function of how far apart their rewards "
        "are?",
        "Measured with `group_advantages` on pairs sampled from π_ref (200 batches of one "
        "pair per prompt, sampling seed 0). Every pair whose rewards differ gets advantages "
        "of ±0.7071 (0.70707 to 0.70711, the spread coming from the 10⁻⁶ added to the "
        "standard deviation), whether the rewards differ by 0.029, the smallest step, or by "
        "0.69, the largest seen. Pairs that tie (5% of them) get 0. This is algebra: with "
        "two values, (r₁ − mean) = ±|r₁ − r₂|/2 and the sample standard deviation is "
        "|r₁ − r₂|/√2, so the ratio is ±1/√2 whatever the difference. At G = 2 the "
        "division throws away the size of the reward difference and keeps only its sign. "
        "In the simulation at G = 2 (graded, published, seeds 0–2), the drift still ends "
        "at 3.1–3.8, near or past the ceiling.",
    )
    return


@app.cell
def _(reveal):
    reveal(
        "**4. Against PPO at equal feedback.** Press the side-by-side button in the "
        "simulation and read the feedback plots. GRPO uses 32 reward calls per batch, PPO "
        "64. Which is ahead at equal feedback?",
        "Measured, defaults, seeds 0–2 compared seed by seed (GRPO seed s against PPO seed "
        "s), gap given as the min–max range over seeds. **Against reward calls** (read every "
        "640 calls, the grid both runs share, and at 4,800, where GRPO ends): PPO-like GRPO "
        "is clearly ahead in every seed up to 1,920 calls: 0.028–0.030 against 0.066–0.088 "
        "at 640, 0.023–0.026 against 0.039–0.048 at 1,280, 0.024–0.025 against 0.029–0.032 "
        "at 1,920. From 2,560 calls on the two are level within the seed spread: in seed 2 "
        "PPO is ahead or level at 2,560 to 3,840 calls, and at 4,800 PPO is ahead in seeds "
        "0 and 2 (0.0175–0.0192 against 0.0184–0.0207). PPO goes on to 0.0145–0.0157 at 9,600 calls. **Against "
        "gradient steps** (read every 40 steps): GRPO is ahead in every seed up to 120 "
        "steps, the order differs by seed from 160 to 280 steps, and from 320 steps on PPO "
        "is ahead in every seed, ending at 0.0145–0.0157 against 0.0175–0.0192 at 600. Both "
        "algorithms take 4 steps per batch and GRPO's batch is half the size, so per reward "
        "call it takes twice the steps; the comparison does not separate the group baseline "
        "from that. **Published GRPO** is behind PPO in every seed at every number of reward "
        "calls read: 0.136–0.204 at 640 calls and 0.373–0.437 at 4,800.",
    )
    return


@app.cell
def _(cite, mo):
    _r1 = mo.md(r"""
    ## Where it went next

    DeepSeek-R1-Zero applied GRPO (standardised advantage, the same KL estimator in the
    loss) directly to a base model, with no supervised fine-tuning first and a rule-based
    reward (accuracy and format) instead of a neural reward model, citing reward hacking.
    Two settings differ from this notebook's: each rollout was split into 16 minibatches
    for "a single inner epoch", so the policy takes several steps per batch, and "every 400
    steps, we replace the reference model with the latest policy model", so $\pi_{\text{ref}}$
    is not fixed.
    """)
    _tulu = mo.md(r"""
    Tülu 3 introduced Reinforcement Learning with Verifiable Rewards (RLVR), training on a
    programmatic check of the answer, and optimised it with PPO, not GRPO. The OLMo 2 1B
    model's RLVR stage used GRPO ($\beta = 0.01$, the same estimator in the loss); notebook
    04 looks at its checkpoints.
    """)
    mo.vstack(
        [
            _r1,
            mo.md(cite("guo2025")),
            _tulu,
            mo.md(cite("lambert2024", "olmo2024")),
        ]
    )
    return


@app.cell
def _(cite, mo):
    mo.md(
        "## References\n\n"
        + cite("shao2024", "guo2025", "lambert2024", "olmo2024", "schulman2017ppo")
    )
    return


if __name__ == "__main__":
    app.run()
