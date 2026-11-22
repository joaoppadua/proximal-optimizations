"""GRPO (Shao et al. 2024) as published: no value head, group-standardised advantages,
and a KL term in the loss with the paper's estimator. Two switches undo each departure
from PPO: `std_normalize=False` and `kl_in="reward"`."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from proxopt import exact
from proxopt.policy import Policy
from proxopt.train import Recorder, Run, scorer
from proxopt.world import World


@dataclass(frozen=True)
class GRPOConfig:
    beta: float = 0.1
    clip: float = 0.2
    group_size: int = 8
    epochs: int = 4
    iterations: int = 150
    lr: float = 1e-3
    std_normalize: bool = True
    kl_in: Literal["loss", "reward"] = "loss"
    reward_mode: str = "graded"
    seed: int = 0


def group_advantages(rewards: Tensor, std_normalize: bool, eps: float = 1e-6) -> Tensor:
    """A = r - mean of its group, divided by (group std + eps) when `std_normalize`.
    rewards, [P, G], one row per group. A group whose rewards are all equal gets A = 0."""
    if rewards.shape[1] < 2:
        raise ValueError(f"group size must be at least 2, got {rewards.shape[1]}")
    advantages = rewards - rewards.mean(1, keepdim=True)
    if std_normalize:
        advantages = advantages / (rewards.std(1, keepdim=True) + eps)
    return advantages


def kl_estimator(logp_new: Tensor, ref_logp: Tensor) -> Tensor:
    """pi_ref / pi_theta - log(pi_ref / pi_theta) - 1, per token."""
    log_ratio = ref_logp - logp_new
    return torch.exp(log_ratio) - log_ratio - 1


def grpo_loss(
    logp_new: Tensor,
    logp_old: Tensor,
    ref_logp: Tensor,
    advantages: Tensor,
    clip: float,
    beta: float,
    kl_in_loss: bool,
) -> tuple[Tensor, Tensor]:
    """-mean(min(rho * A, clip(rho, 1 - eps, 1 + eps) * A) - beta * kl_estimator),
    rho_t = exp(logp_new_t - logp_old_t), A the response's advantage on each of its
    tokens, means over all B * T tokens; the KL term only when `kl_in_loss`.
    logp_*, [B, T]; advantages, [B]. Returns the loss and the ratios rho (detached)."""
    ratios = torch.exp(logp_new - logp_old)
    a = advantages[:, None]
    per_token = torch.min(ratios * a, ratios.clamp(1 - clip, 1 + clip) * a)
    if kl_in_loss:
        per_token = per_token - beta * kl_estimator(logp_new, ref_logp)
    return -per_token.mean(), ratios.detach()


def shaped_rewards(
    rewards: Tensor, logp_old: Tensor, ref_logp: Tensor, beta: float, kl_in: str
) -> Tensor:
    """The rewards the group baseline sees: the task reward r, [B], minus
    beta * sum_t (logp_old_t - ref_logp_t) when `kl_in == "reward"` (as in PPO);
    r unchanged when `kl_in == "loss"` (the KL term then goes into `grpo_loss`)."""
    if kl_in == "reward":
        return rewards - beta * (logp_old - ref_logp).sum(1)
    if kl_in == "loss":
        return rewards
    raise ValueError(f"unknown kl_in: {kl_in!r}")


def _groups(world: World, group_size: int) -> Tensor:
    """Prompts for one group per prompt, in order: [0] * G, [1] * G, ...; shape [P * G]."""
    return torch.arange(world.n_prompts).repeat_interleave(group_size)


def train(ref: Policy, world: World, cfg: GRPOConfig) -> Run:
    """Each iteration samples a group of `cfg.group_size` responses for every prompt,
    fixes their advantages, takes `cfg.epochs` gradient steps, then records.
    `extras["dead_groups"]` counts, per iteration, the groups whose task rewards were all
    equal (before any KL term is folded in)."""
    if cfg.kl_in not in ("loss", "reward"):
        raise ValueError(f"unknown kl_in: {cfg.kl_in!r}")
    policy = ref.clone()
    ref_scorer = scorer(ref)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    rec = Recorder(world, ref, cfg.reward_mode, cfg.beta)
    rec.record(0, 0, policy)
    x = _groups(world, cfg.group_size)
    ratios = torch.ones(0)
    dead_groups = []
    for it in range(1, cfg.iterations + 1):
        y = policy.sample(x, gen)
        with torch.no_grad():
            logp_old = policy.token_logp(x, y)
            ref_logp = ref_scorer.token_logp(x, y)
            rewards = world.reward(x, y, cfg.reward_mode)
            groups = rewards.view(world.n_prompts, cfg.group_size)
            dead_groups.append(int((groups == groups[:, :1]).all(1).sum()))
            rewards = shaped_rewards(rewards, logp_old, ref_logp, cfg.beta, cfg.kl_in)
            advantages = group_advantages(
                rewards.view(world.n_prompts, cfg.group_size), cfg.std_normalize
            ).flatten()
        for _ in range(cfg.epochs):
            loss, ratios = grpo_loss(
                policy.token_logp(x, y), logp_old, ref_logp, advantages, cfg.clip, cfg.beta,
                kl_in_loss=cfg.kl_in == "loss",
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        rec.record(it * cfg.epochs, it * len(x), policy)
    return rec.finish(policy, cfg, ratios=ratios.flatten(), dead_groups=dead_groups)


def baseline_comparison(
    policy: Policy,
    ref: Policy,
    world: World,
    group_size: int,
    beta: float,
    reward_mode: str,
    seed: int,
) -> dict[str, Tensor]:
    """Samples one group per prompt from `policy`. For each response (flattened, [P * G]),
    its regularised return R = r - beta * sum_t (logp_t - ref_logp_t) minus three baselines:
    "exact", the true J_beta of its prompt under `policy`; "group", the mean R of its group;
    "critic", the policy's value head at the first position."""
    x = _groups(world, group_size)
    y = policy.sample(x, torch.Generator().manual_seed(seed))
    with torch.no_grad():
        logits, values = policy(x, y)
        logp = torch.log_softmax(logits, dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        ref_scorer = scorer(ref)
        ref_logp = ref_scorer.token_logp(x, y)
        ret = world.reward(x, y, reward_mode) - beta * (logp - ref_logp).sum(1)
        true_values = exact.prompt_values(
            policy.all_logp(), ref_scorer.all_logp(), world.reward_table(reward_mode), beta
        )
        group_mean = ret.view(world.n_prompts, group_size).mean(1).repeat_interleave(group_size)
    return {"exact": ret - true_values[x], "group": ret - group_mean, "critic": ret - values[:, 0]}
