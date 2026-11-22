"""What drives the turn of DPO's gap: the drawn labels, or which responses the pairs cover?

Notebook 03, reveals 3 and 5. On the default data (4,000 pairs from pi_ref, tau = 0.25,
beta = 0.4), three runs share the same pairs and the same minibatch order and differ only
in the label each pair carries:

- hard:  the ordinary DPO run (`dpo.train`): the drawn winner, probability 1;
- soft:  the true preference probability p = sigma((r_w - r_l) / tau), an oracle no real
         dataset has; every pair has a finite best margin and no label noise;
- noisy: (k + 0.5) / 5, where k counts wins of y_w in 4 fresh draws with probability p;
         every pair has a finite best margin, but the label is noisy.

A pair with label q has loss -q log sigma(h) - (1 - q) log sigma(-h); q = 1 is DPO's.
The last block repeats hard and soft on 4,000 uniform pairs (300 steps, lr 3e-4).
Run with `uv run python scripts/dpo_label_diagnostic.py` (about two minutes).
"""

import time

import torch
import torch.nn.functional as F

from proxopt.algos import dpo
from proxopt.algos.dpo import DPOConfig
from proxopt.prefs import PrefData, sample_pairs
from proxopt.reference import load_reference
from proxopt.train import Recorder, Run, scorer
from proxopt.world import DEFAULT_WORLD as WORLD

TAU, BETA, N_PAIRS = 0.25, 0.4, 4000
SETTINGS = ((3e-3, 300), (3e-4, 1200))  # (learning rate, steps)


def train_labelled(ref, data: PrefData, q: torch.Tensor, cfg: DPOConfig) -> Run:
    """`dpo.train`'s loop (same minibatch generator) with label q in [0, 1] per pair."""
    policy = ref.clone()
    with torch.no_grad():
        ref_scorer = scorer(ref)
        ref_w = ref_scorer.token_logp(data.prompts, data.winners).sum(1)
        ref_l = ref_scorer.token_logp(data.prompts, data.losers).sum(1)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    rec = Recorder(WORLD, ref, "graded", cfg.beta * TAU)
    rec.record(0, len(data), policy)
    for step in range(1, cfg.steps + 1):
        i = torch.randint(len(data), (min(cfg.batch_size, len(data)),), generator=gen)
        x = data.prompts[i]
        h = cfg.beta * (
            (policy.token_logp(x, data.winners[i]).sum(1) - ref_w[i])
            - (policy.token_logp(x, data.losers[i]).sum(1) - ref_l[i])
        )
        loss = -(q[i] * F.logsigmoid(h) + (1 - q[i]) * F.logsigmoid(-h)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            rec.record(step, len(data), policy)
    return rec.finish(policy, cfg)


def summary(run: Run) -> str:
    gap = run.metrics["gap"]
    i = min(range(len(gap)), key=gap.__getitem__)
    at = {s: g for s, g in zip(run.steps, gap)}
    marks = " ".join(f"{s}:{at[s]:.4f}" for s in (100, 300, 600, 1200) if s in at)
    return f"min {gap[i]:.4f} (step {run.steps[i]:4d})  last {gap[-1]:.4f}  [{marks}]"


def main() -> None:
    start = time.perf_counter()
    ref = load_reference(WORLD)
    for seed in (0, 1, 2):
        data = sample_pairs(WORLD, ref, N_PAIRS, TAU, "ref", seed=seed)
        r_w = WORLD.reward(data.prompts, data.winners)
        r_l = WORLD.reward(data.prompts, data.losers)
        p = torch.sigmoid((r_w - r_l) / TAU)
        g = torch.Generator().manual_seed(1000 + seed)
        k = (torch.rand(len(data), 4, generator=g) < p.unsqueeze(1)).sum(1)
        labels = {"soft": p, "noisy": (k + 0.5) / 5}
        for lr, steps in SETTINGS:
            cfg = DPOConfig(beta=BETA, lr=lr, steps=steps, seed=seed)
            print(f"seed {seed}, lr {lr:g}, {steps} steps")
            print(f"  hard   {summary(dpo.train(ref, WORLD, data, cfg, TAU))}")
            for name, q in labels.items():
                print(f"  {name:6s} {summary(train_labelled(ref, data, q, cfg))}")
    for seed in (0, 1, 2):  # reveal 3: uniform pairs, soft labels against hard
        data = sample_pairs(WORLD, ref, N_PAIRS, TAU, "uniform", seed=seed)
        p = torch.sigmoid(
            (WORLD.reward(data.prompts, data.winners) - WORLD.reward(data.prompts, data.losers))
            / TAU
        )
        cfg = DPOConfig(beta=BETA, seed=seed)
        print(f"seed {seed}, uniform pairs, lr {cfg.lr:g}, {cfg.steps} steps")
        print(f"  hard   {summary(dpo.train(ref, WORLD, data, cfg, TAU))}")
        print(f"  soft   {summary(train_labelled(ref, data, p, cfg))}")
    print(f"done in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
