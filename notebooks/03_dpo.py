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

    from proxopt import exact, plots, prefs
    from proxopt.algos import dpo
    from proxopt.algos.dpo import DPOConfig, dpo_loss, implicit_reward
    from proxopt.nbtools import reveal, source_md
    from proxopt.reference import load_reference
    from proxopt.refs import cite
    from proxopt.train import scorer
    from proxopt.world import DEFAULT_WORLD

    return (
        DEFAULT_WORLD,
        DPOConfig,
        cite,
        dpo,
        dpo_loss,
        exact,
        implicit_reward,
        load_reference,
        mo,
        np,
        plots,
        prefs,
        reveal,
        scorer,
        source_md,
        torch,
    )


@app.cell
def _(cite, mo):
    _text = mo.md(r"""
    # 03 · DPO: preferences without a reward model

    ## Where we are

    PPO (notebook 01) and GRPO (notebook 02) both climb notebook 00's objective
    $J_\beta$ by sampling responses from the policy as it trains and scoring them with a
    reward function. DPO changes the inputs, not just the update:

    | | PPO (notebook 01) | GRPO (notebook 02) | DPO |
    |---|---|---|---|
    | samples from the policy during training | yes, a batch per iteration | yes, a group per prompt | **no** |
    | reward | a reward function (in RLHF, a reward model) | a reward function | **none**: only which of two responses won |
    | critic | a learned $V$ | none (the group mean) | **none** |
    | input | prompts | prompts | a **fixed set** of triples $(x, y_w, y_l)$: prompt, preferred, rejected |
    | KL to $\pi_{\text{ref}}$ | per token, in the reward | an estimator, in the loss | inside the loss, through the log-ratios $\log(\pi_\theta/\pi_{\text{ref}})$ |

    Notebook 01 recalled the two-stage RLHF pipeline: humans compare pairs of responses,
    a reward model is fitted to the comparisons, and PPO then maximises $J_\beta$ with that
    reward model in the place of $r$. DPO collapses both stages into one loss on the pairs.
    It is the one real conceptual jump in this set, so the derivation below is given in
    full, one step at a time. It uses notebook 00's objective, optimum and notation, and
    one new symbol, $\tau$, the noise of the preferences.
    """)
    mo.vstack([_text, mo.md(cite("christiano2017", "ouyang2022", "rafailov2023"))])
    return


@app.cell
def _(cite, mo):
    _step = mo.md(r"""
    ## Definition

    **Step 1. The objective and its optimum** (notebook 00). For each prompt $x$,
    $\mathbb{E}_\pi[r] - \beta\,\mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})
    = \beta \log Z_\beta(x) - \beta\,\mathrm{KL}(\pi \,\|\, \pi^*_\beta)$, so the maximiser
    is

    $$
    \pi^*_\beta(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x)\, \exp\!\big(r(x, y)/\beta\big)}{Z_\beta(x)},
    \qquad Z_\beta(x) = \sum_y \pi_{\text{ref}}(y \mid x)\, e^{r(x, y)/\beta}.
    $$

    This is the point PPO aims at in notebook 01; DPO starts from the same formula
    (its eq. 4):
    """)
    mo.vstack([_step, mo.md(cite("rafailov2023"))])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Step 2. Solve for the reward.** Take the log of step 1,
    $\log \pi^*_\beta = \log \pi_{\text{ref}} + r/\beta - \log Z_\beta$, and rearrange:

    $$
    r(x, y) = \beta \log \frac{\pi^*_\beta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z_\beta(x).
    $$

    A reward can be read off its own optimum, up to the term $\beta \log Z_\beta(x)$, which
    depends on the prompt but not on the response (the DPO paper's eq. 5).
    """)
    return


@app.cell
def _(cite, mo):
    _step = mo.md(r"""
    **Step 3. The Bradley–Terry model of preferences.** Given two responses to the same
    prompt, the first is preferred with probability

    $$
    p(y_1 \succ y_2 \mid x) = \sigma\big(r(x, y_1) - r(x, y_2)\big),
    \qquad \sigma(z) = \frac{1}{1 + e^{-z}}.
    $$

    A preference depends only on the *difference* between the two rewards. This is the
    form the DPO paper gives (its eq. 1, citing Bradley and Terry), and the loss notebook
    04's reward model is fitted with:
    """)
    mo.vstack([_step, mo.md(cite("bradley1952", "rafailov2023"))])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Step 4. Substitute step 2 into step 3.** Both responses answer the same prompt, so
    both carry the same $\beta \log Z_\beta(x)$, and it cancels in the difference:

    $$
    p(y_1 \succ y_2 \mid x) = \sigma\!\Big(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}
    - \beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)}\Big).
    $$

    $Z_\beta(x)$ is the one quantity in step 1 that needs a sum over every response; after
    the cancellation, the probability of a preference is written with two policies and
    nothing else (the DPO paper's eq. 6).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Step 5. Maximum likelihood, with $\pi_\theta$ in the place of $\pi^*$.** The data
    say which response won. Fit the policy so that the model of step 4, built from *its*
    log-ratios, gives the observed winners high probability:

    $$
    L_{\text{DPO}}(\theta) = -\,\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\Big(
    \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}
    - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big)\Big].
    $$

    This is a Bradley–Terry reward model whose reward is $\beta \log(\pi_\theta/\pi_{\text{ref}})$:
    fitting it fits the policy, so there is no separate reward model and no RL step
    (the DPO paper's eq. 7). The expectation is the mean over the dataset, and
    $\log \pi_\theta(y \mid x)$ is the sum of the token log-probabilities, as everywhere in
    this set.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Step 6. The gradient.** Write $\hat r(x, y) = \beta \log(\pi_\theta(y \mid x)/\pi_{\text{ref}}(y \mid x))$
    and $h = \hat r(x, y_w) - \hat r(x, y_l)$, so the loss of one pair is $-\log \sigma(h)$.
    Since $\frac{d}{dh}\log \sigma(h) = \sigma(-h)$ and $\pi_{\text{ref}}$ does not depend on
    $\theta$, $\nabla h = \beta\,(\nabla \log \pi_\theta(y_w \mid x) - \nabla \log \pi_\theta(y_l \mid x))$ and

    $$
    \nabla_\theta L_{\text{DPO}} = -\beta\, \mathbb{E}\Big[\sigma\big(\hat r(x, y_l) - \hat r(x, y_w)\big)
    \big(\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)\big)\Big].
    $$

    A descent step raises the winner's log-probability and lowers the loser's, weighted by
    $\sigma(\hat r_l - \hat r_w)$: close to 1 for a pair the current $\hat r$ ranks the wrong
    way, close to 0 for a pair it already ranks right by a wide margin, so those
    contribute little. This is the form in the DPO paper's §4. Its Appendix A.4 writes the
    argument of $\sigma$ the other way round, $\sigma(\hat r_w - \hat r_l)$; that is a slip,
    and autograd on the loss below gives the §4 form.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Step 7. Noise: what DPO aims at depends on $\tau$.** The preferences in this
    notebook are drawn with noise $\tau$: the first response wins with probability
    $\sigma\big((r(x, y_1) - r(x, y_2))/\tau\big)$. (This noise model is this project's,
    not the paper's.) Small $\tau$ means sharp annotators, large $\tau$ noisy ones. Since
    $(r_1 - r_2)/\tau = r_1/\tau - r_2/\tau$, these are Bradley–Terry preferences for the
    reward $r/\tau$. Run steps 1 to 5 with $r/\tau$ in the place of $r$: the policy whose
    preference probabilities match the annotators' is

    $$
    \pi_{\text{ref}}(y \mid x)\, \exp\!\Big(\frac{r(x, y)/\tau}{\beta}\Big) \Big/ Z
    = \pi_{\text{ref}}(y \mid x)\, \exp\!\Big(\frac{r(x, y)}{\beta\tau}\Big) \Big/ Z
    = \pi^*_{\beta\tau}(y \mid x).
    $$

    So with enough pairs, covering every response, DPO with parameter $\beta$ approaches
    $\pi^*_{\beta\tau}$, not $\pi^*_\beta$. The target depends on $\beta$ and $\tau$ only
    through their product (the loss does not: $\beta$ scales every margin, $\tau$ sets the
    labels): $\beta = 0.4$ on $\tau = 0.25$, $\beta = 0.2$ on $\tau = 0.5$ and $\beta = 0.1$
    on $\tau = 1$ all aim at $\pi^*_{0.1}$, and the data cannot tell DPO's $\beta$ from the
    sharpness of the annotators. DPO aims at notebook 00's yardstick $\pi^*_\beta$ only
    when $\tau = 1$. Every gap in this notebook is therefore measured against
    $\pi^*_{\beta\tau}$: $\text{gap} = \beta\tau \cdot \mathrm{KL}(\pi \,\|\, \pi^*_{\beta\tau})$.
    A unit test checks the convergence on a 9-response world with 50,000 pairs drawn
    uniformly (`tests/test_dpo.py`): the gap against $\pi^*_{\beta\tau}$ falls below 10% of
    its start, and the final policy is closer to $\pi^*_{\beta\tau}$ than to $\pi^*_\beta$.
    "Enough pairs, covering every response" is a real condition: the simulation below has
    neither, and shows what that costs.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Intuition

    **A classifier on pairs.** Steps 5 and 6 are logistic regression. Each pair is one
    example, its label is "the first response won", and its logit is
    $h = \hat r(x, y_w) - \hat r(x, y_l)$, the difference of the **implicit rewards**
    $\hat r = \beta \log(\pi_\theta/\pi_{\text{ref}})$. By step 2, $\hat r$ is the reward
    for which $\pi_\theta$ would itself be the optimum $\pi^*_\beta$: every policy defines
    one. Training moves the policy until its implicit reward classifies the pairs well.

    **Defined only up to a constant per prompt.** Add any $c(x)$ to a reward: $\pi^*_\beta$
    does not change (the factor $e^{c(x)/\beta}$ cancels against $Z_\beta(x)$), and no
    preference probability changes (it depends on differences within a prompt). So nothing
    in the data or in the policy can fix the level of the reward of a prompt, only the
    differences between its responses. The most DPO can recover is $r/\tau$ plus an unknown
    constant for each prompt; `implicit_reward` centres $\hat r$ on each prompt's mean over
    its responses, and the plots compare it with $r/\tau$ centred the same way.

    **Which errors the gap counts.** Write the error of the implicit reward as
    $e(x, y) = \hat r(x, y) - r(x, y)/\tau$. Since $\pi_\theta = \pi_{\text{ref}}\, e^{\hat r/\beta}$
    and $\pi^*_{\beta\tau} = \pi_{\text{ref}}\, e^{r/(\beta\tau)}/Z$, one line of algebra
    gives, for each prompt,

    $$
    \mathrm{KL}(\pi_\theta \,\|\, \pi^*_{\beta\tau})
    = \frac{1}{\beta}\,\mathbb{E}_{\pi_\theta}[e] - \log \mathbb{E}_{\pi^*_{\beta\tau}}\big[e^{e/\beta}\big],
    $$

    which is unchanged by adding a constant to $e$. A response that both $\pi_\theta$ and
    $\pi^*_{\beta\tau}$ give almost no probability adds almost nothing to either term,
    whatever its error. So an implicit reward can be far off on most of the 32,768
    responses of a prompt and the gap still be small, and the other way round; the
    simulation measures agreement both ways.

    **A pair seen once has no best margin.** For a pair whose winner is fixed,
    $-\log \sigma(h)$ decreases for ever as $h$ grows (left, below): the loss keeps asking
    for a larger margin. If the same pair were labelled many times, with winners in the
    proportion $p = \sigma(\Delta r/\tau)$, the average loss
    $-p \log \sigma(h) - (1 - p) \log \sigma(-h)$ would have its minimum at
    $h = \Delta r/\tau$ (right), the margin $\pi^*_{\beta\tau}$ gives. On the default data,
    98.5–98.8% of the 4,000 pairs occur exactly once (seeds 0–2). Taken as a function of
    the implicit rewards (free up to a constant per prompt, since any policy normalises),
    the dataset's loss then has no finite minimiser: a response that only ever appears as
    a winner always gains from having its $\hat r$ raised further.
    """)
    return


@app.cell
def _(mo, np, plots):
    _h = np.linspace(-6, 6, 401)
    _logsig = -np.logaddexp(0.0, -_h)  # log σ(h)
    _logsig_neg = -np.logaddexp(0.0, _h)  # log σ(−h)
    _ink = plots.COLORS["ref"]
    with plots.styled_figure(2, width=4.2, height=3.2) as (loss_fig, _axes):
        _axes[0].plot(_h, -_logsig, color=plots.COLORS["dpo"], lw=2.2, label="−log σ(h)")
        _axes[0].plot(_h, 1 / (1 + np.exp(_h)), color=_ink, ls=":", lw=1.6,
                      label="weight σ(−h) in the gradient")
        _axes[0].set_title("one pair, winner fixed")
        _axes[0].legend(loc="upper right", fontsize=8)
        for _p, _ls in ((0.6, "--"), (0.8, "-")):
            _avg = -_p * _logsig - (1 - _p) * _logsig_neg
            _axes[1].plot(_h, _avg, color=plots.COLORS["dpo"], ls=_ls, lw=2.0, label=f"p = {_p}")
            _axes[1].axvline(np.log(_p / (1 - _p)), color=_ink, ls=_ls, lw=0.9)
        _axes[1].set_title("same pair, many labels")
        _axes[1].legend(loc="upper right", fontsize=8)
        for _ax in _axes:
            _ax.set_xlabel("margin h = r̂(y_w) − r̂(y_l)")
            _ax.set_ylabel("loss")
    mo.vstack(
        [
            loss_fig,
            mo.md(
                "Left: the loss of one pair keeps falling as its margin grows, while the "
                "weight σ(−h) its gradient carries (step 6) shrinks but never reaches 0. "
                "Right: the loss averaged over labels drawn with probability p for the "
                "first response; the vertical lines mark its minimum, h = log(p/(1 − p))."
            ),
        ]
    )
    return


@app.cell
def _(dpo_loss, mo, source_md):
    mo.vstack(
        [
            mo.md(
                r"""
                ## The code

                **The loss**, step 5,
                $-\mathrm{mean}\big(\log \sigma(\beta\,[(\log \pi_\theta(y_w) - \log \pi_{\text{ref}}(y_w)) - (\log \pi_\theta(y_l) - \log \pi_{\text{ref}}(y_l))])\big)$,
                on sequence log-probabilities (sums over the 5 tokens), one entry per pair
                of the batch:
                """
            ),
            mo.md(source_md(dpo_loss)),
            mo.md(
                r"""
                The reference log-probabilities of every pair are computed once, before
                training, with a frozen copy of $\pi_{\text{ref}}$; only $\pi_\theta$ is
                scored at each step. `dpo.train` draws minibatches of 256 pairs with
                replacement from the fixed dataset and steps Adam; it never samples a
                response. A unit test checks the loss at known values; autograd on it gives
                step 6's gradient, with $\sigma(\hat r_l - \hat r_w)$ (checked numerically
                when the equations were logged against the paper).

                **The pairs.** `prefs.sample_pairs` draws a prompt uniformly and two
                different responses to it, either both from $\pi_{\text{ref}}$ or both
                uniformly from all 32,768. The first wins with probability
                $\sigma((r_1 - r_2)/\tau)$, using the graded reward.
                """
            ),
        ]
    )
    return


@app.cell
def _(DEFAULT_WORLD, exact, load_reference, np, scorer, torch):
    ref = load_reference(DEFAULT_WORLD)  # outside no_grad: the first call trains it
    with torch.no_grad():
        logp_ref = scorer(ref).all_logp()  # [P, N]
    reward = DEFAULT_WORLD.reward_table()  # pairs are always labelled with the graded reward
    _betas = [float(b) for b in np.logspace(-3, 1, 50)]
    frontier = exact.frontier(logp_ref, reward, _betas)
    ceiling = exact.drift_ceiling(logp_ref, reward).item()
    holdout_pairs = 4000

    def target_point(beta):
        """(drift, expected reward) of π*_β."""
        opt = exact.optimum(logp_ref, reward, beta)
        return exact.kl(opt, logp_ref).item(), exact.expected_reward(opt, reward).item()

    def optimum_loss(data, tau):
        """The DPO loss of π*_{β·τ} on `data`: its β·margin is exactly (r_w − r_l)/τ."""
        r_w = DEFAULT_WORLD.reward(data.prompts, data.winners)
        r_l = DEFAULT_WORLD.reward(data.prompts, data.losers)
        return -torch.nn.functional.logsigmoid((r_w - r_l) / tau).mean().item()

    def line_distance(est, true, weights):
        """Root-mean-square distance of the points (true, est) from the line est = true + c,
        with the best constant c for each prompt, under `weights` ([P, N]); mean over the
        prompts that have any weight."""
        keep = weights.sum(1) > 0
        est, true, weights = est[keep], true[keep], weights[keep]
        w = weights / weights.sum(1, keepdim=True)
        diff = est - true
        diff = diff - (w * diff).sum(1, keepdim=True)
        return (w * diff**2).sum(1).sqrt().mean().item()

    return ceiling, frontier, holdout_pairs, line_distance, logp_ref, optimum_loss, ref, reward, target_point


@app.cell
def _(DEFAULT_WORLD, dpo, holdout_pairs, mo, prefs, ref):
    @mo.cache
    def pairs(n_pairs, tau, source, seed):
        """Preference pairs from the default world, cached by their arguments."""
        return prefs.sample_pairs(DEFAULT_WORLD, ref, n_pairs, tau, source, seed=seed)

    @mo.cache
    def train_dpo(cfg, tau, n_pairs, source):
        """One DPO run, cached by its (frozen) config and its data. The training pairs use
        the config's seed, the held-out pairs (fresh, same source and τ) that seed + 100."""
        return dpo.train(
            ref,
            DEFAULT_WORLD,
            pairs(n_pairs, tau, source, cfg.seed),
            cfg,
            tau,
            holdout=pairs(holdout_pairs, tau, source, cfg.seed + 100),
        )

    return pairs, train_dpo


@app.cell
def _(mo):
    beta_run = mo.ui.slider(steps=[0.05, 0.1, 0.2, 0.4, 1.0, 2.0], value=0.4, label="β",
                            show_value=True)
    tau_run = mo.ui.slider(steps=[0.05, 0.1, 0.25, 0.5, 1.0], value=0.25,
                           label="preference noise τ", show_value=True)
    pairs_run = mo.ui.slider(steps=[100, 300, 1000, 2000, 4000, 8000, 16000], value=4000,
                             label="number of pairs", show_value=True)
    source_run = mo.ui.radio({"π_ref": "ref", "uniform": "uniform"}, value="π_ref",
                             label="pairs drawn from", inline=True)
    lr_run = mo.ui.dropdown({"3e-4 (default)": 3e-4, "3e-3": 3e-3}, value="3e-4 (default)",
                            label="learning rate")
    steps_run = mo.ui.radio({"300": 300, "1,200": 1200}, value="300",
                            label="gradient steps (1,200: 13 to 21 s per change)",
                            inline=True)
    seed_run = mo.ui.number(start=0, stop=99, step=1, value=0, label="seed")
    return beta_run, lr_run, pairs_run, seed_run, source_run, steps_run, tau_run


@app.cell
def _(
    DPOConfig,
    beta_run,
    holdout_pairs,
    lr_run,
    mo,
    pairs_run,
    seed_run,
    source_run,
    steps_run,
    tau_run,
    train_dpo,
):
    dpo_cfg = DPOConfig(
        beta=beta_run.value, lr=lr_run.value, steps=steps_run.value, seed=int(seed_run.value)
    )
    tau = tau_run.value
    n_pairs = pairs_run.value
    source = source_run.value
    dpo_run = train_dpo(dpo_cfg, tau, n_pairs, source)
    mo.vstack(
        [
            mo.md(
                rf"""
                ## Simulation

                DPO from $\pi_{{\text{{ref}}}}$: {dpo_cfg.steps} Adam steps, each on a
                minibatch of {dpo_cfg.batch_size} pairs drawn with replacement from a fixed
                set of labelled pairs. No response is sampled during training. Reward,
                drift and the gap against $\pi^*_{{\beta\tau}}$ are exact, every
                {dpo_cfg.eval_every} steps. The DPO loss is recorded at the same steps over
                the whole training set and over {holdout_pairs:,} *held-out* pairs: fresh
                pairs from the same source and $\tau$, drawn with seed + 100, never trained
                on. A control change resamples the pairs, retrains and redraws: at 300
                steps about 4 s at 4,000 pairs and 6 s at 16,000, the top of the range;
                **at 1,200 steps about 13 s at 4,000 pairs and 21 s at 16,000.** Runs are
                cached, so returning to a setting is instant.
                """
            ),
            mo.hstack([beta_run, tau_run, pairs_run], justify="start"),
            mo.hstack([source_run, lr_run, steps_run, seed_run], justify="start"),
        ]
    )
    return dpo_cfg, dpo_run, n_pairs, source, tau


@app.cell
def _(
    ceiling,
    dpo_cfg,
    dpo_run,
    frontier,
    holdout_pairs,
    mo,
    n_pairs,
    optimum_loss,
    pairs,
    plots,
    source,
    tau,
    target_point,
):
    _bt = dpo_cfg.beta * tau
    _m = dpo_run.metrics
    _g = _m["gap"]
    _imin = min(range(len(_g)), key=_g.__getitem__)
    _train_loss = dpo_run.extras["train_loss"]
    _hold_loss = dpo_run.extras["holdout_loss"]
    _hmin = min(range(len(_hold_loss)), key=_hold_loss.__getitem__)
    _floor_train = optimum_loss(pairs(n_pairs, tau, source, dpo_cfg.seed), tau)
    _floor_hold = optimum_loss(pairs(holdout_pairs, tau, source, dpo_cfg.seed + 100), tau)
    _target = target_point(_bt)
    if tau == 1.0:
        _targets = {f"π*_β = π*_βτ, β = {dpo_cfg.beta:g} (τ = 1)": _target}
    else:
        _targets = {
            f"π*_β, β = {dpo_cfg.beta:g} (the yardstick)": target_point(dpo_cfg.beta),
            f"π*_βτ, βτ = {_bt:g} (what DPO aims at)": _target,
        }
    mo.vstack(
        [
            mo.md(
                rf"""
                **Reward, drift and gap.** In the gap panel, $\beta \cdot \mathrm{{KL}}(\pi \,\|\, \pi^*)$
                stands for $\beta\tau \cdot \mathrm{{KL}}(\pi \,\|\, \pi^*_{{\beta\tau}})$, here
                with $\beta\tau$ = {_bt:g}.
                """
            ),
            plots.training_curves({"dpo": dpo_run}),
            mo.md(
                rf"""
                **The loss it minimises.** Both curves start at $\log 2 = 0.693$ (at
                $\pi_\theta = \pi_{{\text{{ref}}}}$ every margin is 0). For comparison,
                $\pi^*_{{\beta\tau}}$ itself scores {_floor_train:.4f} on the training pairs and
                {_floor_hold:.4f} on the held-out pairs: a training loss below the first
                number fits these particular labels better than the true preference model
                does.
                """
            ),
            plots.extras_curves(dpo_run, ("train_loss", "holdout_loss"), "DPO loss"),
            mo.md(
                "| | E[r] | KL(π ‖ π_ref) | gap | training loss | held-out loss |\n"
                "|---|---|---|---|---|---|\n"
                f"| start (π_ref) | {_m['reward'][0]:.3f} | 0 | {_g[0]:.4f} | "
                f"{_train_loss[0]:.4f} | {_hold_loss[0]:.4f} |\n"
                f"| lowest gap, step {dpo_run.steps[_imin]} | {_m['reward'][_imin]:.3f} "
                f"| {_m['kl'][_imin]:.3f} | {_g[_imin]:.4f} | {_train_loss[_imin]:.4f} "
                f"| {_hold_loss[_imin]:.4f} |\n"
                f"| end, step {dpo_run.steps[-1]} | {_m['reward'][-1]:.3f} | {_m['kl'][-1]:.3f} "
                f"| {_g[-1]:.4f} | {_train_loss[-1]:.4f} | {_hold_loss[-1]:.4f} |\n"
                f"| π*_βτ, βτ = {_bt:g} | {_target[1]:.3f} | {_target[0]:.3f} | 0 "
                f"| {_floor_train:.4f} | {_floor_hold:.4f} |\n\n"
                f"The held-out loss is lowest at step {dpo_run.steps[_hmin]} "
                f"({_hold_loss[_hmin]:.4f}). Feedback: {n_pairs:,} labelled pairs, used "
                f"{dpo_cfg.steps * min(dpo_cfg.batch_size, n_pairs) / n_pairs:.1f} times each "
                "on average."
            ),
            plots.frontier_plot(frontier, {"dpo": dpo_run}, targets=_targets, ceiling=ceiling),
            mo.md(
                "The frontier of the graded reward, with both targets: notebook 00's "
                "yardstick π*_β at the slider's β, and π*_βτ, the point DPO aims at with "
                "this τ. Move τ and the second one moves along the frontier; at τ = 1 "
                "they coincide."
            ),
        ]
    )
    return


@app.cell
def _(
    DEFAULT_WORLD,
    dpo_cfg,
    dpo_run,
    exact,
    implicit_reward,
    line_distance,
    logp_ref,
    mo,
    n_pairs,
    pairs,
    plots,
    prefs,
    ref,
    reward,
    source,
    tau,
    torch,
):
    _data = pairs(n_pairs, tau, source, dpo_cfg.seed)
    _est = implicit_reward(dpo_run.policy, ref, dpo_cfg.beta)  # [P, N], centred per prompt
    _true = reward / tau
    _true = _true - _true.mean(1, keepdim=True)
    _cov = prefs.coverage_mask(DEFAULT_WORLD, _data)
    _opt = exact.optimum(logp_ref, reward, dpo_cfg.beta * tau)
    with torch.no_grad():
        _logp = dpo_run.policy.all_logp()
    # the responses that hold 90% of π*_βτ's probability, prompt by prompt
    _p_sorted, _order = _opt.exp().sort(1, descending=True)
    _keep_sorted = (_p_sorted.cumsum(1) - _p_sorted) < 0.9
    _top = torch.zeros_like(_cov).scatter(1, _order, _keep_sorted)
    _labels = ("covered by a pair", "not covered")
    _weights = {
        "every response alike": torch.ones_like(_est),
        "covered responses only, alike": _cov.float(),
        "uncovered responses only, alike": (~_cov).float(),
        "π_ref": logp_ref.exp(),
        "π*_βτ": _opt.exp(),
        "the trained π_θ": _logp.exp(),
    }
    _share = {
        "every response alike": _cov.float().mean().item(),
        "π_ref": prefs.covered_mass(DEFAULT_WORLD, _data, logp_ref),
        "π*_βτ": prefs.covered_mass(DEFAULT_WORLD, _data, _opt),
        "the trained π_θ": prefs.covered_mass(DEFAULT_WORLD, _data, _logp),
    }
    _rows = "\n".join(
        f"| {k} | {line_distance(torch.zeros_like(_true), _true, w):.3f} "
        f"| {line_distance(_est, _true, w):.3f} | "
        + (f"{_share[k]:.1%}" if k in _share else "")
        + " |"
        for k, w in _weights.items()
    )
    mo.vstack(
        [
            mo.md(
                r"""
                **The implicit reward against the truth.** Each point is one response to
                one prompt: horizontal, its $r/\tau$; vertical, the trained policy's
                $\hat r = \beta \log(\pi_\theta/\pi_{\text{ref}})$; both centred on their
                prompt's mean (so a per-prompt constant cannot show). Points on the dashed
                line are recovered exactly. Left, all $4 \times 32{,}768$ responses, every
                one drawn with the same weight, although most of them have almost no
                probability under $\pi_{\text{ref}}$ or $\pi^*_{\beta\tau}$. Right, only the
                responses that hold 90% of $\pi^*_{\beta\tau}$'s probability, prompt by
                prompt: one of the two places the gap is decided (the other is wherever
                the trained $\pi_\theta$ puts its probability; see the Intuition).
                """
            ),
            mo.hstack(
                [
                    plots.scatter_true_vs_estimate(
                        true=_true, estimate=_est, mask=_cov, mask_labels=_labels,
                        xlabel="r/τ, centred", ylabel="implicit reward r̂, centred",
                    ),
                    plots.scatter_true_vs_estimate(
                        true=_true[_top], estimate=_est[_top], mask=_cov[_top], mask_labels=_labels,
                        xlabel="r/τ, centred (90% of π*_βτ)", ylabel="implicit reward r̂, centred",
                    ),
                ],
                justify="start",
            ),
            mo.md(
                "Distance from the line, as a root-mean-square of r̂ − r/τ after removing the "
                "best constant for each prompt (so it measures exactly what DPO can "
                "recover), with each response weighted as named, at the start (π_θ = π_ref, "
                "so r̂ = 0) and at the end of the run; and the share of that same weight "
                "that falls on responses covered by a pair (the first row: the fraction of "
                "responses covered).\n\n"
                "| responses weighted by | distance at the start | at the end | share covered |\n"
                "|---|---|---|---|\n" + _rows
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What carries over to real LLMs, and what does not.** The *shape* carries over: a
    fixed dataset of preference triples, a loss on summed sequence log-probabilities, the
    reference's log-probabilities computed once by a frozen copy, minibatches and no
    sampling. So does the algebra, which uses nothing particular to this world (only a
    reference that gives every response some probability): for any reward and any
    Bradley–Terry annotators, the policy DPO aims at depends on its $\beta$ and on the
    scale of the annotators' reward only through their product, and the implicit reward is
    defined only up to a constant per prompt. What does not carry over: here every response can be enumerated, so the gap,
    the coverage and the mass a dataset covers are exact, and the true reward and the true
    $\tau$ are known. For an LLM none of these can be computed; only the losses can, and
    the measurements below show that a falling training loss does not mean a falling
    gap, and that a held-out loss says roughly when to stop but not how far from its
    target the policy is. Which source of
    pairs works better, where the gap turns, and at which learning rate, are facts about
    this world (4 prompts, 32,768 responses, a reference that ignores the prompt, a small
    GRU), not about LLM datasets.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to try

    Make a prediction before opening each answer. The answers report what was measured on
    the machine this notebook was written on: seeds 0, 1 and 2 (each seed sets both the
    pairs and the minibatch order), 300 steps, all controls at their defaults (β = 0.4,
    τ = 0.25, 4,000 pairs from π_ref, learning rate 3e-4) except the ones named. Values are
    read at the last step and given as the range over the three seeds (min–max) unless
    stated otherwise. The gap is always against π*_βτ.
    """)
    return


@app.cell
def _(mo, reveal):
    tau_button = mo.ui.run_button(label="Run 3 seeds × τ ∈ {0.1, 0.25, 0.5, 1}")
    mo.vstack(
        [
            reveal(
                "**1. Move τ with β fixed.** Keep β = 0.4 and move τ from 0.05 to 1. Where "
                "does the target go, and does the end of the run go with it?",
                "Measured, β = 0.4, 4,000 pairs from π_ref, 300 steps. **The target moves along "
                "the frontier**, from π*_0.02 at τ = 0.05 (drift 2.21, E[r] 0.985) through "
                "π*_0.1 at τ = 0.25 (0.64, 0.912) to π*_0.4 = π*_β at τ = 1 (0.077, 0.810). "
                "**The end of the run moves the same way, but does not reach it:** drift "
                "1.43–1.50 at τ = 0.05, 1.07–1.15 at 0.1, 0.58–0.61 at 0.25, 0.30–0.35 at 0.5 "
                "and 0.20–0.24 at 1 (expected reward 0.929–0.932, 0.914–0.918, 0.873–0.876, "
                "0.823–0.833, 0.787–0.796). With sharp labels it falls short of its target; "
                "with noisy ones it overshoots. As a fraction of its start (which differs with "
                "the target), the gap ends at 20–21% for τ = 0.05, 21–22% for 0.1, 30–32% for "
                "0.25 and 69–73% for 0.5 (lowest 0.030–0.031 at steps 80–120, then rising). "
                "**At τ = 1 the gap ends at 2.1 to 2.3 times its start** (0.034 to 0.070–0.078; "
                "lowest 0.029–0.033 at steps 10–90) and the held-out loss never gets "
                "meaningfully below log 2. That is the one setting where DPO aims at notebook "
                "00's yardstick, and with 4,000 pairs it learns almost nothing there: at τ = 1 "
                "a reward difference of 0.1 moves the winning probability only from 0.5 to "
                "0.525, so the labels are close to coin flips and the run fits their noise. "
                "**KL is a strict judge of a concentrated target.** At τ = 0.05, "
                "KL(π ‖ π*_βτ) ends at 1.99–2.03, larger than KL(π ‖ π*_β) = 1.05–1.12, and "
                "at τ = 0.1 too (0.85–0.87 against 0.73–0.80); at 0.25 and 0.5 the two are "
                "close (0.30–0.33 against 0.34–0.36; 0.21–0.23 against 0.19–0.22). π*_0.02 "
                "and π*_0.04 give almost no probability to responses the run still gives "
                "some, and KL(π ‖ π*) charges for each of them. So the expected picture "
                "(target and endpoint move together) holds for the direction in the "
                "reward–drift plane, not for the distance, at this budget.",
            ),
            tau_button,
        ]
    )
    return (tau_button,)


@app.cell
def _(DPOConfig, frontier, ceiling, mo, plots, tau_button, target_point, train_dpo):
    mo.stop(not tau_button.value, mo.md("*Press the button to train the twelve runs (about 40 s).*"))
    _taus = (0.1, 0.25, 0.5, 1.0)
    plots.frontier_plot(
        frontier,
        {f"τ = {_t:g}": [train_dpo(DPOConfig(seed=_s), _t, 4000, "ref") for _s in (0, 1, 2)]
         for _t in _taus},
        targets={f"π*_βτ, τ = {_t:g}": target_point(0.4 * _t) for _t in _taus},
        ceiling=ceiling,
    )
    return


@app.cell
def _(mo, reveal):
    product_button = mo.ui.run_button(label="Run 3 seeds × (β, τ) ∈ {(0.4, 0.25), (0.2, 0.5)}")
    mo.vstack(
        [
            reveal(
                "**2. Halve β and double τ.** Compare β = 0.4, τ = 0.25 with β = 0.2, "
                "τ = 0.5. Step 7 says both aim at π*_0.1. Do they end in the same place?",
                "Measured, 4,000 pairs from π_ref, 300 steps. Both settings have the same "
                "target, π*_0.1, and the same starting gap, 0.1022. **They do not end in the "
                "same place.** β = 0.4, τ = 0.25 ends at a gap of 0.030–0.033 (drift 0.58–0.61, "
                "E[r] 0.873–0.876); β = 0.2, τ = 0.5 at 0.050–0.054 (drift 0.57–0.66, E[r] "
                "0.851–0.864), its lowest 0.044–0.046 at steps 130–190. Seed by seed, the KL "
                "between the two final policies is 0.11–0.13. A third pair with the same "
                "product, β = 1, τ = 0.1, ends at 0.021–0.022. So step 7 holds for what the "
                "runs aim at, and the gap of every one of them falls, but at a finite budget "
                "they end in the order (β, τ) = (1, 0.1), (0.4, 0.25), (0.2, 0.5): sharper "
                "labels together with a larger β. Two things change together along that "
                "order, the noise of the labels and the β inside the loss (and with it the "
                "scale of every margin), and these runs do not separate them. The losses of the two settings are not comparable with each "
                "other: their pairs carry different labels.",
            ),
            product_button,
        ]
    )
    return (product_button,)


@app.cell
def _(DPOConfig, mo, plots, product_button, train_dpo):
    mo.stop(not product_button.value, mo.md("*Press the button to train the six runs (about 15 s).*"))
    plots.training_curves(
        {
            f"β = {_b:g}, τ = {_t:g}": [
                train_dpo(DPOConfig(beta=_b, seed=_s), _t, 4000, "ref") for _s in (0, 1, 2)
            ]
            for _b, _t in ((0.4, 0.25), (0.2, 0.5))
        }
    )
    return


@app.cell
def _(mo, reveal):
    source_button = mo.ui.run_button(label="Run 3 seeds × 2 sources at 4,000 pairs")
    mo.vstack(
        [
            reveal(
                "**3. Pairs from π_ref against uniform pairs.** Switch the source to "
                "*uniform* and compare. Uniform pairs spread over all 32,768 responses of a "
                "prompt; pairs from π_ref repeat the responses π_ref likes. Which gives the "
                "lower gap, and what do the scatter and the table show?",
                "Measured, 300 steps unless stated. **Pairs from π_ref end with the lower gap "
                "at every size from 1,000 pairs up:** 0.063–0.079 against 0.088–0.093 for "
                "uniform pairs at 1,000, 0.030–0.033 against 0.055–0.060 at 4,000 and "
                "0.021–0.023 against 0.044–0.047 at 16,000. (At 100 pairs both end above the "
                "start, 0.19–0.23 against 0.16–0.20; see item 4.) At 1,200 steps (the steps "
                "control) the order holds: lowest gap 0.029–0.031 against 0.054–0.058 at 4,000 "
                "pairs, 0.020–0.022 against 0.037–0.041 at 16,000. **Coverage by count is not "
                "coverage by probability.** At 4,000 pairs uniform pairs touch more responses "
                "(5.9% of them, against 2.6–2.7%), but those responses carry 4.9–5.9% of "
                "π_ref's probability, against 70–71% for pairs from π_ref, and 4.4–5.6% of "
                "π*_0.1's, against 69–70% (at 16,000 pairs: 21.7% against 6.3% of responses, "
                "21–22% against 86% of π_ref's mass). The gap is an expectation under the "
                "trained policy, which starts at π_ref, so it is decided where π_ref and π*_βτ "
                "put their probability. Uniform pairs also move the policy less: drift "
                "0.32–0.38 against 0.58–0.61 at 4,000 pairs, 0.59–0.70 against 0.82–0.95 at "
                "1,000. **The scatter and the table** at 4,000 pairs, distance from the line "
                "at the start (r̂ = 0) → at step 300, pairs from π_ref, then uniform pairs. "
                "Every response alike: 0.495 → 0.415–0.455, and 0.495 → 0.373–0.388, so "
                "uniform pairs end closer on the plain scatter. Covered responses only: "
                "0.626–0.629 → 0.354–0.382, and 0.491–0.496 → 0.368–0.387. Uncovered only: "
                "0.490 → 0.415–0.456, and 0.495 → 0.374–0.388. Weighted by π_ref: 0.670 → "
                "0.328–0.355, and 0.670 → 0.490–0.520. By π*_βτ: 0.341 → 0.307–0.318, and "
                "0.341 → 0.335–0.380. So with pairs from π_ref the covered responses move a "
                "long way towards the line and the uncovered ones only a little (0.490 to "
                "0.415–0.456), which is close to the expectation that uncovered responses are "
                "left unconstrained. With uniform pairs, covered and uncovered responses move "
                "by the same amount, and end level with the π_ref pairs' covered ones; what "
                "this measurement does not show is why. Uniform pairs come out ahead on the "
                "uncovered majority, which carries almost no probability, and behind wherever "
                "the probability is. **Not label noise:** with each pair's drawn label "
                "replaced by its probability σ((r_w − r_l)/τ) (information no real dataset "
                "has; see item 5 and `scripts/dpo_label_diagnostic.py`), the same 4,000 pairs "
                "reach 0.0123–0.0125 from π_ref against 0.032–0.033 uniform at step 300. So "
                "uniform pairs are worse here at every size and budget measured, because of "
                "where they fall. They are not useless: a unit test shows that on a 9-response "
                "world 50,000 uniform pairs, which cover every response, bring DPO to its "
                "target. That test does not compare sources.",
            ),
            source_button,
        ]
    )
    return (source_button,)


@app.cell
def _(DPOConfig, mo, plots, source_button, train_dpo):
    mo.stop(not source_button.value, mo.md("*Press the button to train the six runs (about 15 s).*"))
    plots.training_curves(
        {
            f"pairs from {_name}": [
                train_dpo(DPOConfig(seed=_s), 0.25, 4000, _src) for _s in (0, 1, 2)
            ]
            for _name, _src in (("π_ref", "ref"), ("uniform", "uniform"))
        }
    )
    return


@app.cell
def _(reveal):
    reveal(
        "**4. 100 pairs.** Set the number of pairs to 100 and watch the two loss curves and "
        "the gap. Then try 1,000.",
        "Measured, pairs from π_ref, 300 steps. **100 pairs: the training loss falls, "
        "the gap rises.** The training loss goes from 0.693 to 0.126–0.150. The gap is "
        "lowest at steps 10–20, barely below its start (0.097–0.098 against 0.1022), "
        "then rises and ends at 0.19–0.23, about twice the start, with drift 1.36–1.92 "
        "against the target's 0.64. The held-out loss is lowest at step 10 "
        "(0.689–0.691) and ends at 0.84–0.93, above log 2: on fresh pairs the trained "
        "policy does worse than a 50/50 guess. With uniform pairs the gap ends at "
        "0.16–0.20, and in seed 2 it never goes below its start. The covered responses, "
        "pushed hardest, end furthest from the line (distance 1.33–1.53 against "
        "1.00–1.11 for the uncovered ones). The expected picture holds, more strongly "
        "than expected: the gap does not just fail to fall, it gets worse than not "
        "training at all. **1,000 pairs overfit within 300 steps for both sources.** "
        "From π_ref: gap lowest 0.048–0.058 at steps 90–120, ending at 0.063–0.079; "
        "held-out loss lowest at steps 100–110, ending 0.012–0.016 higher. Uniform: gap "
        "lowest 0.073–0.078 at steps 60–140, ending at 0.088–0.093; held-out loss "
        "lowest at steps 40–50 and ending at 0.709–0.723, above log 2, while the gap is "
        "still below its start: the two measures do not turn together.",
    )
    return


@app.cell
def _(mo, reveal):
    lr_button = mo.ui.run_button(label="Run 3 seeds × 2 learning rates")
    mo.vstack(
        [
            reveal(
                "**5. A larger learning rate.** Set the learning rate to 3e-3, ten times the "
                "default. What happens to the gap, and to the two losses? Does the default "
                "rate escape what you see?",
                "Measured, 4,000 pairs from π_ref. **At 3e-3** the gap falls fast, to "
                "0.042–0.046 at steps 10–20, then rises while the training loss keeps falling "
                "(to 0.497–0.503 at step 300): it ends at 0.093–0.108, above its start of "
                "0.1022 in seeds 0 and 1, with drift 1.39–1.46 against the target's 0.64. The "
                "held-out loss turns with it: lowest 0.640–0.644 at steps 10–30, then up to "
                "0.677–0.696. **At the default 3e-4** the same thing happens later and more "
                "slowly: the gap is lowest at 0.029–0.031 at steps 140–210 and ends at "
                "0.030–0.033; the held-out loss is lowest at steps 150–210 and has risen by "
                "only 0.0010–0.0063 at step 300, too little to see. Set the steps to 1,200 and "
                "the turn is plain: the gap is 0.030–0.033, 0.036–0.041, 0.043–0.048 and "
                "0.050–0.056 at steps 300, 600, 900 and 1,200, the held-out loss ends at "
                "0.651–0.657 (lowest 0.628–0.636), and the drift at 0.91–0.93. So the default "
                "rate does not escape it; 300 steps at 3e-4 behave like stopping early. One "
                "sign was visible all along: at the gap's lowest point the training loss "
                "(0.588–0.596) is already below the 0.602–0.611 that π*_0.1 itself scores on "
                "the same pairs. **The labels, not the coverage.** 98.5–98.8% of these pairs "
                "occur once, each with one drawn winner. Two diagnostic runs "
                "(`scripts/dpo_label_diagnostic.py`, which has its own copy of the training "
                "loop) keep the same pairs, and so the same coverage, and the same minibatch "
                "order, and change only the label each pair carries. With its probability "
                "σ((r_w − r_l)/τ), computed from the true reward (an oracle, not an "
                "algorithm), every pair has a finite best margin and there is no label noise, "
                "and there is no turn: at 3e-3 the gap is 0.0076–0.0083 at step 300 and still "
                "falling; at 3e-4 it is 0.0123–0.0125 at step 300 and 0.0078–0.0080 at step "
                "1,200. With a noisy label that still has a finite best margin, (k + 0.5)/5 "
                "for k wins in 4 fresh draws, the gap turns again, less: at 3e-3 it is lowest "
                "at 0.026–0.028 at steps 80–130 and ends at 0.034–0.038 at step 300; at 3e-4 "
                "lowest at 0.020–0.024 at steps 220–420 and 0.023–0.028 at step 1,200. "
                "So the turn comes from the drawn labels, not from which responses the pairs "
                "cover, and most of the default run's gap at step 300 (0.030–0.033, against "
                "0.012 with oracle labels on the same pairs) does too. What these runs do not "
                "separate is the two things a single drawn label brings: noise, and a best "
                "margin that is infinite (the left plot in the Intuition). A finite best "
                "margin alone does not remove the turn.",
            ),
            lr_button,
        ]
    )
    return (lr_button,)


@app.cell
def _(DPOConfig, lr_button, mo, plots, train_dpo):
    mo.stop(not lr_button.value, mo.md("*Press the button to train the six runs (about 15 s).*"))
    plots.training_curves(
        {
            f"lr = {_lr:g}": [
                train_dpo(DPOConfig(lr=_lr, seed=_s), 0.25, 4000, "ref") for _s in (0, 1, 2)
            ]
            for _lr in (3e-4, 3e-3)
        }
    )
    return


@app.cell
def _(cite, mo):
    _norm = mo.md(r"""
    ## Where it went next

    **Length-normalised DPO.** The Tülu 3 recipe divides each log-ratio by the length of
    its response, $\frac{\beta}{|y_w|} \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}
    - \frac{\beta}{|y_l|} \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$
    (its eq. 6), so that each response enters through its mean per-token log-ratio; its
    authors report that "length-normalized DPO works best". The DPO stage of OLMo 2 1B,
    whose checkpoints notebook 04 looks at, used this length-normalised loss with
    $\beta = 5$, according to the training command its authors published in their
    open-instruct repository (the OLMo 2 paper names neither the variant nor $\beta$).
    That $\beta$ multiplies per-token mean log-ratios, so it is not comparable with a
    $\beta$ on summed log-ratios like this notebook's. Here every response has 5 tokens, so the
    length-normalised loss with $\beta$ is exactly the summed one with $\beta/5$.
    """)
    _next = mo.md(r"""
    DPO did not end the use of RL. Tülu 3 introduced Reinforcement Learning with
    Verifiable Rewards (RLVR), for "tasks with verifiable outcomes such as mathematical
    problem-solving and instruction following", and its authors "train models with RLVR
    following preference finetuning", using PPO. OLMo 2 1B was post-trained with SFT,
    then DPO, then RLVR with GRPO. Notebook 04 runs PPO, GRPO and DPO on the toy world of
    notebooks 00 to 03, against the same exact optimum, and adds the comparison this
    notebook cannot make: a Bradley–Terry reward model fitted to the same pairs DPO uses,
    with PPO run on it, next to DPO.
    """)
    mo.vstack([_norm, mo.md(cite("lambert2024")), _next, mo.md(cite("lambert2024", "olmo2024"))])
    return


@app.cell
def _(cite, mo):
    mo.md(
        "## References\n\n"
        + cite("rafailov2023", "bradley1952", "christiano2017", "ouyang2022", "lambert2024",
               "olmo2024")
    )
    return


if __name__ == "__main__":
    app.run()
