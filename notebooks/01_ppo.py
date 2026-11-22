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
    from proxopt.algos import ppo, reinforce
    from proxopt.algos.ppo import PPOConfig, gae, ppo_loss, token_rewards
    from proxopt.algos.reinforce import ReinforceConfig
    from proxopt.nbtools import reveal, source_md
    from proxopt.reference import load_reference
    from proxopt.refs import cite
    from proxopt.train import scorer
    from proxopt.world import DEFAULT_WORLD

    return (
        DEFAULT_WORLD,
        PPOConfig,
        ReinforceConfig,
        cite,
        exact,
        gae,
        load_reference,
        mo,
        np,
        plots,
        ppo,
        ppo_loss,
        reinforce,
        reveal,
        scorer,
        source_md,
        token_rewards,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 01 · PPO, in its RLHF form

    ## Where we are

    Notebook 00 set up the objective
    $J_\beta(\pi) = \mathbb{E}_x\big[\mathbb{E}_{y \sim \pi}[r(x, y)] - \beta\,\mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})\big]$,
    its exact optimum $\pi^*_\beta \propto \pi_{\text{ref}}\, e^{r/\beta}$, the
    reward–drift frontier, the drift ceiling, and the yardstick
    $\text{gap} = \beta\,\mathrm{KL}(\pi \,\|\, \pi^*_\beta)$. It climbed $J_\beta$ with
    REINFORCE and found two weaknesses: a single sampled return per response is a **noisy**
    signal, and **each batch is used for one gradient step** and thrown away.

    PPO answers both. Against the noise, a **learned critic** $V$ predicts the return from
    each prefix, so every token gets its own advantage instead of sharing one
    sequence-level return. Against the waste, a **clipped objective** lets the same batch
    be reused for several gradient steps without the policy running away from the policy
    that sampled it. This notebook is the reference PPO that notebooks 02 (GRPO) and 03
    (DPO) are compared against. The notation is notebook 00's.
    """)
    return


@app.cell
def _(cite, mo):
    _mdp = mo.md(
        r"""
        ## Definition

        **Tokens as actions.** The state before token $t$ is $s_t = (x, y_{<t})$, the action
        is $y_t$, and the episode ends after $y_T$ ($T = 5$ here). Samples come from
        $\pi_{\text{old}}$, the policy as it was when the batch was drawn; it stays fixed
        while that batch is reused.

        **The per-token reward.** The task reward arrives once, on the last token; the KL
        penalty to $\pi_{\text{ref}}$ is paid on every token:

        $$
        r_t = -\beta \log \frac{\pi_{\text{old}}(y_t \mid s_t)}{\pi_{\text{ref}}(y_t \mid s_t)}
        + \mathbb{1}[t = T]\, r(x, y).
        $$

        Without discounting, the tokens' rewards add up to

        $$
        \sum_{t=1}^{T} r_t = r(x, y) - \beta \log \frac{\pi_{\text{old}}(y \mid x)}{\pi_{\text{ref}}(y \mid x)},
        $$

        whose expectation under $y \sim \pi_{\text{old}}(\cdot \mid x)$ is
        $\mathbb{E}[r] - \beta\,\mathrm{KL}(\pi_{\text{old}} \,\|\, \pi_{\text{ref}})$, the
        per-prompt $J_\beta$. So the return this PPO maximises is notebook 00's objective,
        and **this PPO aims at $\pi^*_\beta$**: the gap is the right yardstick for it. The
        KL-penalised reward of a whole response, $r - \beta \log(\pi / \pi_{\text{ref}})$, is
        Ziegler et al.'s; paying the penalty token by token ("a per-token KL penalty from the
        SFT model at each token") is Ouyang et al.'s:
        """
    )
    _critic = mo.md(
        r"""
        **The critic.** $V(s_t)$ predicts the return still to come from state $s_t$,
        $\mathbb{E}\big[\sum_{l \ge t} r_l \mid s_t\big]$, with $V(s_{T+1}) = 0$. Here it is
        a second linear head on the policy's GRU, read from the same hidden state as the
        token logits. **It starts at zero**: the head's weight and bias are initialised to
        0, so an untrained critic predicts $V = 0$ everywhere, a neutral baseline. A randomly
        initialised head is not neutral: its noise enters the advantages below as if it were
        signal. Measured here, with a randomly drawn value head (three draws) and a reward of
        exactly 0, the policy drifted to $\mathrm{KL}(\pi \,\|\, \pi_{\text{ref}}) =$
        0.13–0.18 in 10 iterations, moved by nothing but the critic's noise. A value head
        freshly added to a pretrained LLM has the same problem, and that is the motivation
        for the choice here. (The RLHF papers below do something else: they initialise the
        value function from the reward model; see "Where it went next".)

        **Advantages (GAE).** With TD errors

        $$
        \delta_t = r_t + V(s_{t+1}) - V(s_t),
        $$

        the generalised advantage estimate, with discount $\gamma = 1$, is

        $$
        \hat A_t = \sum_{l=0}^{T-t} \lambda^l\, \delta_{t+l}.
        $$

        At $\lambda = 1$ the sum telescopes to $\sum_{l \ge t} r_l - V(s_t)$: the Monte Carlo
        return with the critic as baseline, unbiased whatever $V$ is. At $\lambda = 0$ it is
        $\delta_t$ alone, which trusts $V(s_{t+1})$ for everything after token $t$: lower
        variance, biased by whatever the critic gets wrong. $\hat A_t$ is computed once per
        batch, with $\pi_{\text{old}}$ and the critic as it was, and then held fixed.

        **The clipped objective.** With the per-token probability ratio

        $$
        \rho_t(\theta) = \frac{\pi_\theta(y_t \mid s_t)}{\pi_{\text{old}}(y_t \mid s_t)},
        $$

        PPO maximises

        $$
        L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t(\theta)\, \hat A_t,\
        \mathrm{clip}(\rho_t(\theta), 1 - \varepsilon, 1 + \varepsilon)\, \hat A_t\big)\Big],
        $$

        where $\mathbb{E}_t$ is the mean over all $B \cdot T$ tokens of the batch.

        **The value loss.** The critic regresses on the targets
        $V^{\text{targ}}_t = \hat A_t + V_{\text{old}}(s_t)$:

        $$
        L^{\text{VF}}(\theta) = \mathbb{E}_t\big[(V_\theta(s_t) - V^{\text{targ}}_t)^2\big].
        $$

        The loss minimised, for `epochs` full-batch gradient steps on each batch, is
        $-L^{\text{CLIP}} + c_1 L^{\text{VF}}$ with $c_1 = 0.5$. Two simplifications against
        the PPO paper: no entropy bonus (its $c_2 = 0$), and each epoch is one step on the
        whole batch rather than several minibatches. Clipping, GAE and the combined loss are
        from:
        """
    )
    _pipeline = mo.md(
        r"""
        **The two-stage pipeline.** In RLHF, $r$ is not a program. Humans compare pairs of
        responses, a reward model is fitted to those comparisons, and PPO then maximises
        $J_\beta$ with the reward model in the place of $r$; the KL penalty keeps the policy
        where the reward model was trained, close to the SFT model. Here the first stage is
        skipped: $r$ is the world's exact programmatic reward. Notebook 03 is DPO, which
        learns from preference pairs without a reward model; notebook 04 fits a
        Bradley–Terry reward model to the same kind of pairs and runs PPO on it, next to DPO.
        """
    )
    mo.vstack(
        [
            _mdp,
            mo.md(cite("ziegler2019", "ouyang2022")),
            _critic,
            mo.md(cite("schulman2017ppo", "schulman2016gae")),
            _pipeline,
            mo.md(cite("christiano2017", "ziegler2019", "stiennon2020", "ouyang2022")),
        ]
    )
    return


@app.cell
def _(cite, mo):
    eps_view = mo.ui.slider(
        steps=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0], value=0.2, label="ε", show_value=True
    )
    _text = mo.md(
        r"""
        ## Intuition

        **Clipping is a trust region on each token's probability.** Take one token with
        advantage $\hat A > 0$. Raising its probability raises $\rho \hat A$, but only
        until $\rho = 1 + \varepsilon$: past that the clipped term is constant, the min picks
        it, and the gradient for this token is zero. With $\hat A < 0$ the same happens
        below $1 - \varepsilon$. So on each token the objective stops rewarding a move once
        that token's probability has changed by a factor $1 \pm \varepsilon$ in the
        direction its advantage asks for. The min also makes the objective pessimistic: if
        a token's ratio has moved the *wrong* way ($\hat A > 0$, $\rho < 1 - \varepsilon$),
        the unclipped term is the smaller one and the gradient that would bring it back is
        kept.

        Two things this is not. It is not a constraint: a clipped token gets no gradient of
        its own, but the parameters it shares with other tokens keep moving it, so ratios
        do leave the band (the histogram in the simulation shows by how much at the run's
        last gradient step; item 1 of "Things to try" gives the extremes over a run). And it is
        not the KL penalty: the clip keeps $\pi_\theta$ near $\pi_{\text{old}}$ *within one
        batch*, while $\beta$ keeps the policy near $\pi_{\text{ref}}$ over the whole run.
        The clip stands in for TRPO's explicit constraint on
        $\mathrm{KL}(\pi_{\text{old}} \,\|\, \pi_\theta)$ between successive policies:
        """
    )
    mo.vstack([_text, mo.md(cite("schulman2015trpo")), eps_view])
    return (eps_view,)


@app.cell
def _(eps_view, mo, np, plots):
    _eps = eps_view.value
    _rho = np.linspace(0.0, 2.5, 501)
    _ink = plots.COLORS["ref"]
    with plots.styled_figure(2, width=4.2, height=3.2) as (clip_fig, _axes):
        for _ax, _adv in zip(_axes, (1.0, -1.0)):
            _clipped = np.minimum(_rho * _adv, np.clip(_rho, 1 - _eps, 1 + _eps) * _adv)
            _ax.plot(_rho, _rho * _adv, color=_ink, ls=":", lw=1.4, label="unclipped ρ·Â")
            _ax.plot(_rho, _clipped, color=plots.COLORS["ppo"], lw=2.2, label="L^CLIP")
            for _v in (1 - _eps, 1 + _eps):
                _ax.axvline(_v, color=_ink, ls="--", lw=1.0)
            _ax.axvline(1.0, color=_ink, lw=0.8)
            _ax.set_title(f"Â = {_adv:+.0f}")
            _ax.set_xlabel("ratio ρ = π_θ / π_old")
            _ax.set_ylabel("objective for one token")
        clip_fig.legend(*_ax.get_legend_handles_labels(), loc="outside lower center", ncols=2)
    mo.vstack(
        [
            clip_fig,
            mo.md(
                f"Dashed lines at 1 ± ε = {1 - _eps:.2f} and {1 + _eps:.2f}. Where the solid "
                "line is flat, the token's gradient is zero."
            ),
        ]
    )
    return


@app.cell
def _(gae, mo, ppo_loss, source_md, token_rewards):
    mo.vstack(
        [
            mo.md(
                r"""
                ## The code

                The three functions below are the whole algorithm; `ppo.train` only samples,
                calls them, and steps Adam.

                **Per-token rewards**,
                $r_t = -\beta \log \frac{\pi_{\text{old}}(y_t \mid s_t)}{\pi_{\text{ref}}(y_t \mid s_t)} + \mathbb{1}[t = T]\, r(x, y)$:
                """
            ),
            mo.md(source_md(token_rewards)),
            mo.md(
                r"""
                With $\pi_{\text{old}} = \pi_{\text{ref}}$, as at the start, the penalty is 0 on
                every token. So with a task reward of exactly 0 and a critic that starts at 0,
                every $r_t$, every $\delta_t$ and every $\hat A_t$ is 0 and the policy never
                moves. That is a trivial fixed point, not a sign of the penalty pulling the
                policy back. The penalty at work shows with the true reward: after 20
                iterations (seeds 0–2, everything else default) the drift is 0.074–0.078 at
                $\beta = 1$ and 0.71–0.79 at $\beta = 0.05$. Against their optima's drifts
                (0.014 and 1.24), the $\beta = 1$ run has overshot, by about five times, and the
                $\beta = 0.05$ run has not yet got there.

                **GAE**, $\delta_t = r_t + V(s_{t+1}) - V(s_t)$,
                $\hat A_t = \sum_{l=0}^{T-t} \lambda^l \delta_{t+l}$, $V(s_{T+1}) = 0$, and the
                value targets $V^{\text{targ}}_t = \hat A_t + V_{\text{old}}(s_t)$, computed by
                the backward recursion $\hat A_t = \delta_t + \lambda \hat A_{t+1}$
                (`values[:, t]` is $V(s_t)$, the value of the state before $y_t$):
                """
            ),
            mo.md(source_md(gae)),
            mo.md(
                r"""
                **The loss**, $-L^{\text{CLIP}} + c_1 L^{\text{VF}}$, with
                $\rho_t = \exp(\log \pi_\theta(y_t \mid s_t) - \log \pi_{\text{old}}(y_t \mid s_t))$
                and both means over all $B \cdot T$ tokens:
                """
            ),
            mo.md(source_md(ppo_loss)),
            mo.md(
                r"""
                At the first epoch on a batch $\pi_\theta = \pi_{\text{old}}$, so every
                $\rho_t = 1$, the clip cannot bind, and the policy gradient is the plain
                policy gradient with per-token advantages $\hat A_t$. A unit test checks that,
                with the same advantage on every token, it equals REINFORCE's gradient divided
                by $T$, for any $\varepsilon$; the factor $1/T$ is there because the loss averages
                over all $B \cdot T$ tokens, not over the $B$ responses.
                The clip only matters from the second epoch on.
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
    rewards_by_mode = {m: DEFAULT_WORLD.reward_table(m) for m in ("graded", "verifiable")}
    _betas = [float(b) for b in np.logspace(-2, 1, 40)]
    frontiers = {m: exact.frontier(logp_ref, r, _betas) for m, r in rewards_by_mode.items()}
    ceilings = {m: exact.drift_ceiling(logp_ref, r).item() for m, r in rewards_by_mode.items()}
    return ceilings, frontiers, logp_ref, ref, rewards_by_mode


@app.cell
def _(DEFAULT_WORLD, mo, ppo, reinforce, ref):
    @mo.cache
    def train_ppo(cfg):
        """One PPO run on the default world, cached by its (frozen) config."""
        return ppo.train(ref, DEFAULT_WORLD, cfg)

    @mo.cache
    def train_reinforce(cfg):
        """One REINFORCE run on the default world, cached by its (frozen) config."""
        return reinforce.train(ref, DEFAULT_WORLD, cfg)

    return train_ppo, train_reinforce


@app.cell
def _(mo):
    clip_run = mo.ui.slider(
        steps=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0], value=0.2, label="clip ε", show_value=True
    )
    beta_run = mo.ui.slider(
        steps=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0], value=0.1, label="β", show_value=True
    )
    epochs_run = mo.ui.slider(start=1, stop=8, step=1, value=4, label="epochs per batch",
                              show_value=True)
    lam_run = mo.ui.slider(
        steps=[0.0, 0.5, 0.8, 0.9, 0.95, 1.0], value=0.95, label="GAE λ", show_value=True
    )
    batch_run = mo.ui.slider(steps=[16, 32, 64, 128], value=64, label="batch size B",
                             show_value=True)
    mode_run = mo.ui.radio(["graded", "verifiable"], value="graded", label="reward",
                           inline=True)
    seed_run = mo.ui.number(start=0, stop=99, step=1, value=0, label="seed")
    return batch_run, beta_run, clip_run, epochs_run, lam_run, mode_run, seed_run


@app.cell
def _(
    PPOConfig,
    ReinforceConfig,
    batch_run,
    beta_run,
    clip_run,
    epochs_run,
    lam_run,
    mo,
    mode_run,
    seed_run,
    train_ppo,
    train_reinforce,
):
    ppo_cfg = PPOConfig(
        beta=beta_run.value,
        clip=clip_run.value,
        epochs=epochs_run.value,
        lam=lam_run.value,
        batch_size=batch_run.value,
        reward_mode=mode_run.value,
        seed=int(seed_run.value),
    )
    rf_cfg = ReinforceConfig(
        beta=ppo_cfg.beta,
        batch_size=ppo_cfg.batch_size,
        reward_mode=ppo_cfg.reward_mode,
        seed=ppo_cfg.seed,
    )
    ppo_run = train_ppo(ppo_cfg)
    rf_run = train_reinforce(rf_cfg)
    mo.vstack(
        [
            mo.md(
                rf"""
                ## Simulation

                PPO from $\pi_{{\text{{ref}}}}$: {ppo_cfg.iterations} iterations, each one
                batch of $B$ sampled responses reused for `epochs` full-batch Adam steps
                (learning rate {ppo_cfg.lr:g}, $c_1 = {ppo_cfg.vf_coef}$). Next to it,
                REINFORCE with the batch-mean baseline from notebook 00, at the same $\beta$,
                batch size, reward and seed, with its own defaults otherwise
                ({rf_cfg.steps} steps, learning rate {rf_cfg.lr:g}). A control change trains
                PPO (about 2 to 5 s) and, if $\beta$, $B$, the reward or the seed changed,
                REINFORCE (under a second); both are cached, so returning to a setting is
                instant. Reward, drift and gap are exact, computed over all responses after
                every iteration (PPO) or every 10 steps (REINFORCE).
                """
            ),
            mo.hstack([clip_run, beta_run, epochs_run], justify="start"),
            mo.hstack([lam_run, batch_run], justify="start"),
            mo.hstack([mode_run, seed_run], justify="start"),
        ]
    )
    return ppo_cfg, ppo_run, rf_cfg, rf_run


@app.cell
def _(
    ceilings,
    exact,
    frontiers,
    logp_ref,
    mo,
    plots,
    ppo_cfg,
    ppo_run,
    rewards_by_mode,
    rf_cfg,
    rf_run,
):
    _reward = rewards_by_mode[ppo_cfg.reward_mode]
    _opt = exact.optimum(logp_ref, _reward, ppo_cfg.beta)
    _target = (exact.kl(_opt, logp_ref).item(), exact.expected_reward(_opt, _reward).item())
    _runs = {"ppo": ppo_run, "reinforce": rf_run}

    def _summary(name, run):
        m = run.metrics
        return (
            f"| {name} | {run.steps[-1]:,} | {run.feedback[-1]:,} | {m['reward'][-1]:.3f} "
            f"| {m['kl'][-1]:.3f} | {m['gap'][0]:.4f} → {m['gap'][-1]:.4f} |"
        )

    mo.vstack(
        [
            mo.md(
                rf"""
                **Against gradient steps.** One step is one Adam update. PPO takes
                {ppo_cfg.epochs} per batch, so {ppo_cfg.iterations} batches make
                {ppo_cfg.iterations * ppo_cfg.epochs:,} steps; REINFORCE takes one per batch,
                {rf_cfg.steps:,} in all.
                """
            ),
            plots.training_curves(_runs, x="steps"),
            mo.md(
                rf"""
                **Against units of feedback.** One unit is one call to the reward function,
                one per sampled response. PPO uses $B$ = {ppo_cfg.batch_size} per batch,
                {ppo_run.feedback[-1]:,} in all; REINFORCE {rf_run.feedback[-1]:,}. Epochs
                cost steps but no feedback. The two algorithms run at their own learning
                rates, so neither plot ranks them; they show where each spends its budget.
                """
            ),
            plots.training_curves(_runs, x="feedback"),
            mo.md(
                "| run | gradient steps | reward calls | E[r] | KL(π ‖ π_ref) | gap |\n"
                "|---|---|---|---|---|---|\n"
                + _summary("ppo", ppo_run) + "\n" + _summary("reinforce", rf_run) + "\n"
                + f"| π*_β, β = {ppo_cfg.beta} | | | {_target[1]:.3f} | {_target[0]:.3f} "
                "| 0 |"
            ),
            mo.md(
                r"""
                **What the clip cut, at one moment.** The ratios $\rho_t$ of every token of
                the last batch at the run's last gradient step (the last epoch of the last
                iteration), against the clip range. It is a snapshot, not the whole run: at
                other steps of the run, ratios go further from 1 (item 1 below gives the
                extremes over a run). With one epoch they are all exactly 1.
                """
            ),
            plots.ratio_hist(ppo_run.extras["ratios"], ppo_cfg.clip),
            plots.frontier_plot(
                frontiers[ppo_cfg.reward_mode],
                _runs,
                targets={f"π*_β, β = {ppo_cfg.beta}": _target},
                ceiling=ceilings[ppo_cfg.reward_mode],
            ),
            mo.md(
                f"The frontier is that of the {ppo_cfg.reward_mode} reward; the dotted line is "
                "its drift ceiling (notebook 00). A policy can pass the ceiling: the ceiling "
                "is the largest drift of an *optimum*, which spreads over all the best "
                "responses as π_ref does, and a policy that piles onto fewer of them drifts "
                "further for no more reward."
            ),
        ]
    )
    return


@app.cell
def _(cite, mo):
    _note = mo.md(r"""
    **What carries over to real LLMs, and what does not.** The *shape* carries over: the
    reward is a per-token KL penalty plus one task reward on the last token, advantages
    come from GAE over the tokens of a response, the ratio and its clip are per token,
    and the critic is a value head trained on the same batch. The defaults are the values
    Stiennon et al. report ($\gamma = 1$, $\lambda = 0.95$, 4 epochs per batch; their
    Appendix B.1). They are not universal: InstructGPT split each batch of 512 into 8
    minibatches of 64 and trained on it "for only a single inner epoch" (Ouyang et al.,
    Appendix C.4). Its policy still takes 8 steps per batch, and from the second
    minibatch on its ratios can leave 1 and the clip can bind; here each epoch is one step
    on the whole batch. What does not carry over: the response space here is enumerable, so
    expected reward, drift and gap are exact, while for an LLM they can only be estimated
    from samples and the optimum is unknown. Here sampling a batch costs little; for an LLM, generating the batch is the expensive part, which is the case for reusing
    it, and that case does not show up in these numbers (at the defaults, REINFORCE has a
    lower gap than PPO at equal reward calls). The critic here shares the policy's network
    and costs almost nothing; in RLHF it is a second large model. Every number below (how
    much clipping cuts, how much λ matters, how seeds spread) is a fact about this toy world
    and these learning rates, not about LLMs.
    """)
    mo.vstack([_note, mo.md(cite("stiennon2020", "ouyang2022"))])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to try

    Make a prediction before opening each answer. The answers report what was measured on
    the machine this notebook was written on: seeds 0, 1 and 2, all controls at their
    defaults except the ones named, gap read at the iteration named and given as the range
    over the three seeds (min–max) unless a mean is stated.
    """)
    return


@app.cell
def _(mo, reveal):
    seeds_clip = mo.ui.run_button(label="Run 3 seeds × 2 clip ranges at 8 epochs")
    mo.vstack(
        [
            reveal(
                "**1. Reuse without a brake.** Set epochs to 8 and compare ε = 0.2 with "
                "ε = 1. With ε = 1 the band is [0, 2]: the clip still acts, but only on "
                "ratios above 2 (a ratio cannot fall below 0). Over seeds 0–2 (or press the "
                "button): how far do the ratios move, and what happens to the gap?",
                "Measured, epochs 8, 150 iterations. **The ratios over the whole run** (every "
                "token at every one of the 1,200 gradient steps, read in a copy of the "
                "training loop that logs them; its last step matches `extras[\"ratios\"]` "
                "exactly): with ε = 1 they reach 2.52–3.20 at the top and 0.29–0.40 at the "
                "bottom; with ε = 0.2, 1.98–2.78 and 0.44–0.54. So ε = 1 lets them go "
                "further, and its clip does act, but rarely: 0.01–0.02% of token-steps are "
                "above 2. With ε = 0.2, 1.9–2.1% of token-steps are outside [0.8, 1.2], and "
                "some tokens still reach twice their probability under π_old: the clip "
                "brakes the ratios, it does not bound them. (At the default 4 epochs: "
                "0.67–0.74 to 1.53–1.58, 0.08% outside the band.) **The snapshot** in the "
                "histogram, the last batch at its last step, shows much less: maximum "
                "1.37–1.49 (ε = 0.2) against 1.49–1.76 (ε = 1), with 3.8–10.3% against "
                "4.4–13.1% of tokens outside [0.8, 1.2]. Read alone, it would suggest the clip "
                "makes little difference. **The gap:** ε = 1 ends worse, 0.0209–0.0268 "
                "against 0.0183–0.0199 for ε = 0.2, and its curve is rougher: after "
                "iteration 20 its largest rise from one iteration to the next is 0.005–0.012, "
                "against 0.003–0.004 for ε = 0.2 (and 0.001 at 4 epochs). Both end worse "
                "than 4 epochs (0.0145–0.0157), so reusing the batch 8 times costs something "
                "even with the clip, and the clip reduces that cost. The expected picture "
                "holds: without an effective clip the ratios move further from 1 and the gap "
                "is rougher and ends worse. It is a degradation, not a breakdown.",
            ),
            seeds_clip,
        ]
    )
    return (seeds_clip,)


@app.cell
def _(PPOConfig, mo, plots, seeds_clip, train_ppo):
    mo.stop(not seeds_clip.value, mo.md("*Press the button to train the six runs (about 20 s).*"))
    plots.training_curves(
        {
            f"ε = {_c:g}": [train_ppo(PPOConfig(epochs=8, clip=_c, seed=_s)) for _s in (0, 1, 2)]
            for _c in (0.2, 1.0)
        }
    )
    return


@app.cell
def _(mo, reveal):
    seeds_epochs = mo.ui.run_button(label="Run 3 seeds × (PPO 1 epoch, REINFORCE)")
    mo.vstack(
        [
            reveal(
                "**2. One epoch.** Set epochs to 1. The ratio is then 1 on every token, so "
                "the clip never binds and PPO is a policy gradient with a critic and GAE "
                "advantages. How does it compare with REINFORCE, against units of feedback?",
                "Measured. The ratio histogram is a single bar at 1 (every ratio exactly "
                "1.000), as the algebra says. But PPO with one epoch is **not** close to "
                "REINFORCE here; it is clearly slower. At equal reward calls the gaps are "
                "0.082–0.088 against 0.035–0.037 at 640 calls, 0.049–0.053 against "
                "0.016–0.018 at 3,200, and 0.0215–0.0232 against 0.012–0.013 at 9,600, the "
                "end of the PPO run (REINFORCE goes on to 0.0105–0.0112 at 12,800). One "
                "epoch is also slower than PPO's default 4 epochs at equal feedback "
                "(0.0145–0.0157 at 9,600), which is what the reuse buys. Part of the "
                "difference from REINFORCE is the learning rate (1e-3 for PPO, 3e-3 for "
                "REINFORCE, each tuned for its own default): in a separate run with the "
                "learning rate set to 3e-3 in code, one-epoch PPO ends at 0.0151–0.0173, "
                "nearer REINFORCE but still behind it. The rest was not separately measured; "
                "candidates are a critic that starts at zero and must be learned, and a value "
                "loss that shares the policy's network. Note that this one-epoch setting is "
                "not InstructGPT's \"single inner epoch\": there each batch was split into 8 "
                "minibatches (Ouyang et al., Appendix C.4), so the policy moved between them "
                "and the clip could bind; here one epoch is one full-batch step and every "
                "ratio is 1.",
            ),
            seeds_epochs,
        ]
    )
    return (seeds_epochs,)


@app.cell
def _(PPOConfig, ReinforceConfig, mo, plots, seeds_epochs, train_ppo, train_reinforce):
    mo.stop(not seeds_epochs.value, mo.md("*Press the button to train the six runs (about 8 s).*"))
    plots.training_curves(
        {
            "ppo, 1 epoch": [train_ppo(PPOConfig(epochs=1, seed=_s)) for _s in (0, 1, 2)],
            "reinforce": [train_reinforce(ReinforceConfig(seed=_s)) for _s in (0, 1, 2)],
        },
        x="feedback",
    )
    return


@app.cell
def _(mo, reveal):
    seeds_lam = mo.ui.run_button(label="Run 3 seeds × λ ∈ {0, 1}")
    mo.vstack(
        [
            reveal(
                "**3. λ = 0 against λ = 1, early in training.** The critic starts at zero, "
                "so early on it knows nothing. Which λ suffers from that, and how?",
                "Measured. With the critic at zero, λ = 0 gives the last token the advantage "
                "r(x, y) − 0, which is always positive (about 0.75 on average), and every "
                "earlier token only its KL term, which is 0 at the start; the task reward "
                "reaches earlier tokens only as fast as the critic learns to carry it back. "
                "λ = 1 hands every token the whole return from the first batch on, with no "
                "baseline yet. **λ = 0 goes the wrong way first:** its gap *rises* from "
                "0.102 to a peak of 0.170–0.183 at iterations 9–11, because drift grows "
                "(0.56–0.74 at iteration 10) while expected reward does not (0.727–0.753, "
                "from 0.745). Only then does it fall: 0.106–0.128 at iteration 20, "
                "0.053–0.059 at 50, ending at 0.030–0.034. **λ = 1** never rises more than "
                "0.0004 above its start of 0.1022, but its first iterations differ by seed: "
                "seed 2 falls from the first iteration, seed 0 dips to 0.088 at iteration 2 "
                "and is back at 0.101 by iteration 6, and seed 1 stays at 0.100–0.103 until "
                "iteration 7. All three are falling by iteration 8: 0.066–0.086 at iteration 10, "
                "0.041–0.049 at 20, ending at 0.0141–0.0156, "
                "no different from the default λ = 0.95 (0.0145–0.0157). A likely reading of "
                "the rise (not separately measured): an always-positive advantage on the last "
                "token pushes up whichever last token was sampled, which piles mass on a few "
                "of them without raising reward, the same failure as REINFORCE without a "
                "baseline in notebook 00. So λ = 0 does lean on a bad critic, as the theory "
                "predicts; with a critic that starts at zero, \"bad\" means no baseline and no "
                "credit for any token but the last.",
            ),
            seeds_lam,
        ]
    )
    return (seeds_lam,)


@app.cell
def _(PPOConfig, mo, plots, seeds_lam, train_ppo):
    mo.stop(not seeds_lam.value, mo.md("*Press the button to train the six runs (about 17 s).*"))
    plots.training_curves(
        {
            f"λ = {_l:g}": [train_ppo(PPOConfig(lam=_l, seed=_s)) for _s in (0, 1, 2)]
            for _l in (0.0, 1.0)
        }
    )
    return


@app.cell
def _(reveal):
    reveal(
        "**4. Verifiable reward.** Switch the reward to *verifiable*: 1 if the digits sum "
        "exactly to the target, 0 otherwise. Under π_ref only 3.3% of samples score 1. What "
        "happens to the start of training?",
        "Measured. Under the verifiable reward the target π*_0.1 is far from π_ref (drift "
        "3.42, near the ceiling of 3.44; expected reward 0.9986), so the gap starts at 0.623 "
        "rather than 0.102; the comparison below is of the gap as a fraction of its start. "
        "**A slower start, as expected:** after 10 iterations (640 reward calls) the gap is "
        "still 97–98% of its start (graded: 64–86%), after 20 it is 79–87% (graded: "
        "39–47%), and expected reward passes 0.5 only at iterations 26–40. **Then it "
        "climbs:** expected reward ends at 0.951–0.974 (from 0.033), the gap at "
        "0.178–0.194, 28–31% of its start. Drift ends at 4.89–5.09, *past* the drift "
        "ceiling: the run piles onto a few exact-sum responses where the optimum spreads "
        "over all of them, so it ends below the frontier. REINFORCE (the comparison in the "
        "simulation) is slower here per reward call: at 2,560 calls its gap is 60–79% of "
        "the start against 41–65% for PPO, and at 9,600 calls 0.199–0.224 against "
        "0.178–0.194. That is the reverse of the graded reward, where REINFORCE is ahead "
        "at equal reward calls.",
    )
    return


@app.cell
def _(cite, mo):
    _next = mo.md(r"""
    ## Where it went next

    PPO on a learned reward model, with the per-token KL penalty, is the RL stage of
    InstructGPT (Ouyang et al. 2022), which also tried mixing pretraining gradients into the
    update ("PPO-ptx"). Its price is the critic. Stiennon et al. (2020, §3.4) use "a
    Transformer with completely separate parameters from the policy", initialise it "to the
    parameters of the reward model", and note that "the reward model, policy, and value
    function are the same size"; their ablation (Appendix G.1) found separate networks
    better than a shared one, at the cost of memory. Ouyang et al. (§3.5, Appendix C.4)
    initialise "a 6B value function" from the 6B reward model for policies of every size.
    Either way, PPO keeps four networks in memory (policy, reference, reward model,
    critic). GRPO (notebook 02) removes the critic and takes the baseline from a group of
    samples for the same prompt.
    """)
    mo.vstack([_next, mo.md(cite("ouyang2022", "stiennon2020"))])
    return


@app.cell
def _(cite, mo):
    mo.md(
        "## References\n\n"
        + cite(
            "schulman2015trpo",
            "schulman2016gae",
            "schulman2017ppo",
            "christiano2017",
            "ziegler2019",
            "stiennon2020",
            "ouyang2022",
        )
    )
    return


if __name__ == "__main__":
    app.run()
