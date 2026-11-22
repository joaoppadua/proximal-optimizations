"""The numbers notebook 01 quotes in item 1 of "Things to try": reuse without a brake.

PPO at its defaults except the epochs and the clip, seeds 0, 1, 2, 150 iterations:
8 epochs with eps = 0.2 and eps = 1, and the default 4 epochs with eps = 0.2. `ppo.train`
keeps only the ratios of its last gradient step, so this script has its own copy of the
training loop that also looks at the ratios of every token at every gradient step; the
copy is checked to give `ppo.train`'s metrics and last-step ratios exactly.

Over the whole run (every token at every gradient step):
- the largest and smallest ratio;
- the share of token-steps with a ratio above 2, and of those with a ratio above 2 *and*
  a positive advantage (only those lose their gradient to the clip at eps = 1);
- the share of token-steps outside [0.8, 1.2].
At the last gradient step (the histogram's snapshot): the largest ratio and the share
outside [0.8, 1.2]. The gap at the last iteration, and the largest rise of the gap from
one iteration to the next after iteration 20. Ranges are min–max over the seeds.

Run with `uv run python scripts/measure_nb01.py` (about two minutes).
"""

import itertools
import time

import torch

from proxopt.algos import ppo
from proxopt.algos.ppo import PPOConfig, _logp_and_values, gae, ppo_loss, token_rewards
from proxopt.reference import load_reference
from proxopt.train import Recorder, Run, scorer
from proxopt.world import DEFAULT_WORLD as WORLD

SEEDS = (0, 1, 2)
SETTINGS = (("8 epochs, eps = 0.2", 8, 0.2), ("8 epochs, eps = 1", 8, 1.0),
            ("4 epochs, eps = 0.2", 4, 0.2))


def train_logged(ref, cfg: PPOConfig) -> tuple[Run, dict]:
    """`ppo.train`'s loop, with statistics of the ratios at every gradient step."""
    policy = ref.clone()
    ref_scorer = scorer(ref)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    rec = Recorder(WORLD, ref, cfg.reward_mode, cfg.beta)
    rec.record(0, 0, policy)
    stats = {"max": 0.0, "min": float("inf"), "n": 0, "above_2": 0, "above_2_pos": 0,
             "outside_band": 0}
    ratios = torch.ones(0)
    for it in range(1, cfg.iterations + 1):
        x = torch.randint(WORLD.n_prompts, (cfg.batch_size,), generator=gen)
        y = policy.sample(x, gen)
        with torch.no_grad():
            logp_old, values_old = _logp_and_values(policy, x, y)
            ref_logp = ref_scorer.token_logp(x, y)
            rewards = token_rewards(WORLD.reward(x, y, cfg.reward_mode), logp_old, ref_logp,
                                    cfg.beta)
            advantages, returns = gae(rewards, values_old, cfg.lam)
        for _ in range(cfg.epochs):
            logp_new, values_new = _logp_and_values(policy, x, y)
            loss, ratios = ppo_loss(
                logp_new, logp_old, advantages, values_new, returns, cfg.clip, cfg.vf_coef
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            stats["max"] = max(stats["max"], ratios.max().item())
            stats["min"] = min(stats["min"], ratios.min().item())
            stats["n"] += ratios.numel()
            stats["above_2"] += int((ratios > 2).sum())
            stats["above_2_pos"] += int(((ratios > 2) & (advantages > 0)).sum())
            stats["outside_band"] += int(((ratios < 0.8) | (ratios > 1.2)).sum())
        rec.record(it * cfg.epochs, it * cfg.batch_size, policy)
    return rec.finish(policy, cfg, ratios=ratios.flatten()), stats


def span(values, fmt) -> str:
    return f"{format(min(values), fmt)}–{format(max(values), fmt)}"


def main() -> None:
    start = time.perf_counter()
    ref = load_reference(WORLD)
    for name, epochs, clip in SETTINGS:
        rows = []
        for seed in SEEDS:
            cfg = PPOConfig(epochs=epochs, clip=clip, seed=seed)
            run, stats = train_logged(ref, cfg)
            check = ppo.train(ref, WORLD, cfg)
            assert run.metrics == check.metrics, "the logging copy differs from ppo.train"
            assert torch.equal(run.extras["ratios"], check.extras["ratios"])
            last = run.extras["ratios"]
            gap = run.metrics["gap"]
            rows.append({
                "max": stats["max"], "min": stats["min"],
                "above_2": 100 * stats["above_2"] / stats["n"],
                "above_2_pos": 100 * stats["above_2_pos"] / stats["n"],
                "outside": 100 * stats["outside_band"] / stats["n"],
                "n_above_2": stats["above_2"], "n_above_2_pos": stats["above_2_pos"],
                "last_max": last.max().item(),
                "last_outside": 100 * ((last < 0.8) | (last > 1.2)).float().mean().item(),
                "gap": gap[-1],
                "rise": max(b - a for a, b in itertools.pairwise(gap[20:])),
            })

        def col(key, rows=rows):
            return [r[key] for r in rows]

        print(f"{name} (seeds {SEEDS[0]}–{SEEDS[-1]}):")
        print(f"  over the run: max ratio {span(col('max'), '.2f')}, "
              f"min ratio {span(col('min'), '.2f')} (to 4 decimals: "
              f"{span(col('max'), '.4f')}, {span(col('min'), '.4f')})")
        print(f"  token-steps above 2: {span(col('above_2'), '.3f')}%; above 2 with a positive "
              f"advantage: {span(col('above_2_pos'), '.3f')}% ({sum(col('n_above_2_pos'))} of "
              f"{sum(col('n_above_2'))} over the three seeds)")
        print(f"  token-steps outside [0.8, 1.2]: {span(col('outside'), '.2f')}%")
        print(f"  last step: max ratio {span(col('last_max'), '.2f')}, outside [0.8, 1.2] "
              f"{span(col('last_outside'), '.1f')}%")
        print(f"  final gap {span(col('gap'), '.4f')}; largest rise after iteration 20 "
              f"{span(col('rise'), '.3f')}")
    print(f"\ntotal {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
