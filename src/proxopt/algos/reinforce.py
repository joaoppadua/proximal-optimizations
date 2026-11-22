"""REINFORCE on the KL-regularised objective, with an optional mean baseline."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from proxopt.policy import Policy
from proxopt.train import Recorder, Run
from proxopt.world import World


@dataclass(frozen=True)
class ReinforceConfig:
    beta: float = 0.1
    lr: float = 3e-3
    batch_size: int = 64
    steps: int = 200
    baseline: Literal["none", "mean"] = "mean"
    reward_mode: str = "graded"
    seed: int = 0
    eval_every: int = 10


def reinforce_loss(
    logp_tokens: Tensor, ref_logp_tokens: Tensor, rewards: Tensor, beta: float, baseline: str
) -> Tensor:
    """-mean((R - b) * sum_t logp_t), R = r - beta * sum_t (logp_t - ref_logp_t) detached,
    b the batch mean of R ("mean") or 0 ("none")."""
    logp = logp_tokens.sum(1)
    ret = (rewards - beta * (logp_tokens.detach() - ref_logp_tokens).sum(1)).detach()
    if baseline == "mean":
        ret = ret - ret.mean()
    elif baseline != "none":
        raise ValueError(f"unknown baseline: {baseline!r}")
    return -(ret * logp).mean()


def train(ref: Policy, world: World, cfg: ReinforceConfig) -> Run:
    policy = ref.clone()
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    rec = Recorder(world, ref, cfg.reward_mode, cfg.beta)
    rec.record(0, 0, policy)
    for step in range(1, cfg.steps + 1):
        x = torch.randint(world.n_prompts, (cfg.batch_size,), generator=gen)
        y = policy.sample(x, gen)
        with torch.no_grad():
            ref_logp = ref.token_logp(x, y)
        loss = reinforce_loss(
            policy.token_logp(x, y), ref_logp, world.reward(x, y, cfg.reward_mode), cfg.beta,
            cfg.baseline,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            rec.record(step, step * cfg.batch_size, policy)
    return rec.finish(policy, cfg)
