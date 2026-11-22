"""The numbers notebook 02 quotes under "What GRPO aims at", (a) to (c), and in its
paragraph "Does the collapse depend on reusing the batch?" (including the learning rate).

Default world, graded reward, beta = 0.1, G = 8, eps = 0.2, 150 iterations, seeds 0, 1, 2,
ranges min–max over the seeds, unless a block says otherwise.

(a) The KL estimator's expected gradient, by exact enumeration of all 32,768 responses of
    each prompt: the gradient of E_{y ~ pi_theta}[sum_t k3_t] with the samples' weights
    pi_theta(y | x) held fixed (what the loss term's gradient is in expectation at
    pi_old = pi_theta), against the gradients of the two sequence-level KLs; cosines, and
    the expected value against the exact drift. At the final policies of the default PPO
    run and of published GRPO, seed 0.
(b) The group standard deviation sigma of the task rewards, logged by this script's own
    copy of `grpo.train` (checked to give `grpo.train`'s metrics exactly): the median over
    the groups with a non-zero spread in iterations 1–10 and 141–150, per seed; effective
    beta = 0.1 * sigma. At 4 epochs and at 1 epoch.
(c) The final gap and drift of the four switch settings at 4 epochs and 1 epoch, and of
    PPO; for published GRPO the trajectory (lowest gap and its iteration, the first
    iteration after it with the gap back above its start, the first iteration with drift
    above the drift ceiling); the final policies' mass on the maximum-reward responses and
    their exponentiated entropy (per prompt, range over seeds and prompts), against
    pi*_0.1's.
(d) Published GRPO at a ten times lower learning rate, 1e-4, 4 epochs, 1,500 iterations
    (6,000 gradient steps), the same trajectory statistics.
(e) The rest of the paragraphs on what the collapse depends on: published GRPO at 1 epoch
    for 600 iterations; at the paper's beta = 0.04 with 1 epoch (gap against pi*_0.04);
    a sweep of beta at 4 epochs (final drift, and gap from its start to its end, each
    against that beta's optimum); a sweep of the group size G (final drift).

Run with `uv run python scripts/measure_nb02.py` (about seven minutes).
"""

import time

import numpy as np
import torch

from proxopt import exact
from proxopt.algos import grpo, ppo
from proxopt.algos.grpo import (
    GRPOConfig,
    _groups,
    group_advantages,
    grpo_loss,
    kl_estimator,
    shaped_rewards,
)
from proxopt.algos.ppo import PPOConfig
from proxopt.reference import load_reference
from proxopt.train import Recorder, Run, scorer
from proxopt.world import DEFAULT_WORLD as WORLD

SEEDS = (0, 1, 2)
SWITCHES = (("published (÷σ, KL in loss)", True, "loss"), ("÷σ, KL in reward", True, "reward"),
            ("no ÷σ, KL in loss", False, "loss"), ("PPO-like (no ÷σ, KL in reward)", False, "reward"))


def span(values, fmt=".4f") -> str:
    lo, hi = format(min(values), fmt), format(max(values), fmt)
    return lo if lo == hi else f"{lo}–{hi}"


def train_logged(ref, cfg: GRPOConfig) -> tuple[Run, list[list[float]]]:
    """`grpo.train`'s loop, also returning the group std of the rewards the advantage
    sees (the task rewards when the KL is in the loss), per iteration and group."""
    policy = ref.clone()
    ref_scorer = scorer(ref)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    rec = Recorder(WORLD, ref, cfg.reward_mode, cfg.beta)
    rec.record(0, 0, policy)
    x = _groups(WORLD, cfg.group_size)
    stds = []
    for it in range(1, cfg.iterations + 1):
        y = policy.sample(x, gen)
        with torch.no_grad():
            logp_old = policy.token_logp(x, y)
            ref_logp = ref_scorer.token_logp(x, y)
            rewards = shaped_rewards(WORLD.reward(x, y, cfg.reward_mode), logp_old, ref_logp,
                                     cfg.beta, cfg.kl_in)
            groups = rewards.view(WORLD.n_prompts, cfg.group_size)
            stds.append(groups.std(1).tolist())
            advantages = group_advantages(groups, cfg.std_normalize).flatten()
        for _ in range(cfg.epochs):
            loss, _ = grpo_loss(policy.token_logp(x, y), logp_old, ref_logp, advantages,
                                cfg.clip, cfg.beta, kl_in_loss=cfg.kl_in == "loss")
            opt.zero_grad()
            loss.backward()
            opt.step()
        rec.record(it * cfg.epochs, it * len(x), policy)
    return rec.finish(policy, cfg), stds


def trajectory(run: Run, ceiling: float) -> dict:
    """Lowest gap and its iteration; first iteration after it with the gap above its
    start; first iteration with drift above the ceiling (None if never)."""
    gap, kl = run.metrics["gap"], run.metrics["kl"]
    low = min(range(len(gap)), key=gap.__getitem__)
    back = next((i for i in range(low, len(gap)) if gap[i] > gap[0]), None)
    cross = next((i for i, v in enumerate(kl) if v > ceiling), None)
    return {"start": gap[0], "low": gap[low], "low_it": low, "back": back, "cross": cross,
            "gap": gap[-1], "kl": kl[-1]}


def print_trajectory(label: str, rows: list[dict], epochs: int) -> None:
    def col(k):
        return [r[k] for r in rows]

    def iterations(values):
        if None in values:  # not in every seed: list them, seed by seed
            return "in seeds " + ", ".join("never" if v is None else str(v) for v in values)
        return f"at iterations {span(values, 'd')}"

    cross = col("cross")
    cross_text = iterations(cross) + ("" if None in cross else
                                      f" (gradient steps {span([c * epochs for c in cross], 'd')})")
    print(f"  {label}: start gap {span(col('start'))}; lowest gap {span(col('low'))} at "
          f"iterations {span(col('low_it'), 'd')}; back above its start "
          f"{iterations(col('back'))}; drift above the ceiling {cross_text}; ends at gap "
          f"{span(col('gap'))}, drift {span(col('kl'), '.2f')}")


def concentration(logp: torch.Tensor, reward: torch.Tensor) -> tuple[list, list]:
    """Per prompt: probability mass on the maximum-reward responses, and exp(entropy)."""
    best = reward == reward.max(1, keepdim=True).values
    p = logp.exp()
    return (p * best).sum(1).tolist(), (-(p * logp).sum(1)).exp().tolist()


def kl_gradients(policy, ref) -> dict:
    """(a): cosines of the k3 term's expected gradient with the two sequence-level KL
    gradients, by enumeration, and its expected value against the exact drift."""
    y_all = WORLD.all_responses()
    n = len(y_all)
    x = torch.arange(WORLD.n_prompts).repeat_interleave(n)
    y = y_all.repeat(WORLD.n_prompts, 1)
    params = list(policy.parameters())
    token_logp = policy.token_logp(x, y)
    with torch.no_grad():
        ref_token_logp = scorer(ref).token_logp(x, y)
    seq, ref_seq = token_logp.sum(1), ref_token_logp.sum(1)
    weight = seq.exp().detach()  # pi_theta(y | x), the sampling distribution, held fixed
    k3 = (weight * kl_estimator(token_logp, ref_token_logp).sum(1)).sum() / WORLD.n_prompts
    rev = (ref_seq.exp() * (ref_seq - seq)).sum() / WORLD.n_prompts
    fwd = (seq.exp() * (seq - ref_seq)).sum() / WORLD.n_prompts

    def grad(scalar):
        g = torch.autograd.grad(scalar, params, allow_unused=True, retain_graph=True)
        return torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                          for gi, p in zip(g, params)])

    g_k3, g_rev, g_fwd = grad(k3), grad(rev), grad(fwd)

    def cos(a, b):
        return (a @ b / (a.norm() * b.norm())).item()

    return {"cos_rev": cos(g_k3, g_rev), "cos_fwd": cos(g_k3, g_fwd), "value": k3.item(),
            "drift": fwd.item()}


def main() -> None:
    start = time.perf_counter()
    ref = load_reference(WORLD)
    reward = WORLD.reward_table()
    with torch.no_grad():
        logp_ref = scorer(ref).all_logp()
    ceiling = exact.drift_ceiling(logp_ref, reward).item()
    opt = exact.optimum(logp_ref, reward, 0.1)
    print(f"drift ceiling {ceiling:.2f}; pi*_0.1 drift {exact.kl(opt, logp_ref).item():.3f}")

    runs = {}
    for epochs in (4, 1):
        for name, std, kl_in in SWITCHES:
            runs[(epochs, name)] = [
                grpo.train(ref, WORLD, GRPOConfig(std_normalize=std, kl_in=kl_in,
                                                  epochs=epochs, seed=s))
                for s in SEEDS
            ]
    ppo_runs = [ppo.train(ref, WORLD, PPOConfig(seed=s)) for s in SEEDS]
    print(f"start gap {ppo_runs[0].metrics['gap'][0]:.4f}")

    print("\n(a) the KL estimator's expected gradient, by enumeration, seed 0:")
    for label, policy in (("final policy of the default PPO run", ppo_runs[0].policy),
                          ("final policy of published GRPO", runs[(4, SWITCHES[0][0])][0].policy)):
        a = kl_gradients(policy, ref)
        print(f"  {label}: cosine with grad KL(pi_ref || pi) {a['cos_rev']:.2f}, with grad "
              f"KL(pi || pi_ref) {a['cos_fwd']:.2f}; expected value {a['value']:.4f}, exact "
              f"drift {a['drift']:.4f}")

    print("\n(b) median group std sigma over groups with a non-zero spread, published GRPO:")
    for epochs in (4, 1):
        early, late = [], []
        for s in SEEDS:
            cfg = GRPOConfig(epochs=epochs, seed=s)
            run, stds = train_logged(ref, cfg)
            assert run.metrics == runs[(epochs, SWITCHES[0][0])][s].metrics, "copy differs"
            st = np.array(stds)
            early.append(float(np.median(st[0:10][st[0:10] > 0])))
            late.append(float(np.median(st[140:150][st[140:150] > 0])))
        print(f"  {epochs} epoch(s): iterations 1–10 {span(early, '.3f')}, 141–150 "
              f"{span(late, '.3f')}; effective beta {span([0.1 * v for v in early], '.3f')}, "
              f"then {span([0.1 * v for v in late], '.3f')}")

    print("\n(c) final gap and drift at iteration 150:")
    print("  | division | KL | gap, 4 epochs | drift, 4 epochs | gap, 1 epoch | drift, 1 epoch |")
    for name, _, _ in SWITCHES:
        cells = []
        for epochs in (4, 1):
            group = runs[(epochs, name)]
            cells += [span([r.metrics["gap"][-1] for r in group]),
                      span([r.metrics["kl"][-1] for r in group], ".2f")]
        print(f"  | {name} | " + " | ".join(cells) + " |")
    print(f"  | PPO, 4 epochs | {span([r.metrics['gap'][-1] for r in ppo_runs])} | "
          f"{span([r.metrics['kl'][-1] for r in ppo_runs], '.2f')} | | |")
    print("  published GRPO trajectory:")
    for epochs in (4, 1):
        print_trajectory(f"{epochs} epoch(s)",
                         [trajectory(r, ceiling) for r in runs[(epochs, SWITCHES[0][0])]], epochs)

    best_count = (reward == reward.max(1, keepdim=True).values).sum(1).tolist()
    _, opt_eff = concentration(opt, reward)
    print(f"  maximum-reward responses per prompt: {span(best_count, 'd')}; pi*_0.1's "
          f"exponentiated entropy per prompt {span(opt_eff, '.0f')}")
    for epochs in (4, 1):
        for name, _, _ in SWITCHES:
            mass, eff = [], []
            for r in runs[(epochs, name)]:
                with torch.no_grad():
                    m, e = concentration(r.policy.all_logp(), reward)
                mass += m
                eff += e
            print(f"  {epochs} epoch(s), {name}: mass on maximum-reward responses "
                  f"{span([100 * v for v in mass], '.0f')}%, exponentiated entropy "
                  f"{span(eff, '.1f')}")

    print("\n(d) published GRPO at lr 1e-4, 4 epochs, 1,500 iterations:")
    long_runs = [grpo.train(ref, WORLD, GRPOConfig(lr=1e-4, iterations=1500, seed=s))
                 for s in SEEDS]
    print_trajectory("lr 1e-4", [trajectory(r, ceiling) for r in long_runs], 4)
    print(f"  gap at iteration 150: {span([r.metrics['gap'][150] for r in long_runs])}")

    print("\n(e) published GRPO, other settings:")
    long_1 = [grpo.train(ref, WORLD, GRPOConfig(epochs=1, iterations=600, seed=s)) for s in SEEDS]
    print_trajectory("1 epoch, 600 iterations", [trajectory(r, ceiling) for r in long_1], 1)
    paper = [grpo.train(ref, WORLD, GRPOConfig(beta=0.04, epochs=1, seed=s)) for s in SEEDS]
    print_trajectory("beta = 0.04, 1 epoch", [trajectory(r, ceiling) for r in paper], 1)
    for beta in (0.01, 0.04, 0.1, 0.3, 1.0):
        group = runs[(4, SWITCHES[0][0])] if beta == 0.1 else [
            grpo.train(ref, WORLD, GRPOConfig(beta=beta, seed=s)) for s in SEEDS]
        drift_opt = exact.kl(exact.optimum(logp_ref, reward, beta), logp_ref).item()
        print(f"  beta = {beta:g}, 4 epochs: final drift {span([r.metrics['kl'][-1] for r in group], '.2f')} "
              f"(optimum {drift_opt:.3g}); gap {group[0].metrics['gap'][0]:.3f} at the start, "
              f"{span([r.metrics['gap'][-1] for r in group], '.3f')} at the end")
    for g_size in (2, 4, 16, 32):
        group = [grpo.train(ref, WORLD, GRPOConfig(group_size=g_size, seed=s)) for s in SEEDS]
        print(f"  G = {g_size}: final drift {span([r.metrics['kl'][-1] for r in group], '.2f')}")
    print(f"\ntotal {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
