"""PPO in its RLHF form: per-token KL penalty to pi_ref, GAE, clipped surrogate."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import LongTensor, Tensor

from proxopt.policy import Policy
from proxopt.train import Recorder, Run, scorer
from proxopt.world import World


@dataclass(frozen=True)
class PPOConfig:
    beta: float = 0.1
    clip: float = 0.2
    epochs: int = 4
    lam: float = 0.95
    batch_size: int = 64
    iterations: int = 150
    lr: float = 1e-3
    vf_coef: float = 0.5
    reward_mode: str = "graded"
    seed: int = 0


def token_rewards(rewards: Tensor, logp_old: Tensor, ref_logp: Tensor, beta: float) -> Tensor:
    """r_t = -beta * (logp_old_t - ref_logp_t) on every token, plus the task reward r
    on the last token. rewards, [B]; logp_old and ref_logp, [B, T]. Returns [B, T]."""
    out = -beta * (logp_old - ref_logp)
    out[:, -1] += rewards
    return out


def gae(rewards: Tensor, values: Tensor, lam: float) -> tuple[Tensor, Tensor]:
    """Generalised advantage estimation with discount 1 and V = 0 after the last token.
    delta_t = r_t + V_{t+1} - V_t, A_t = sum_l lam^l delta_{t+l}, returns = A + V."""
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[:, 0])
    next_value = torch.zeros_like(values[:, 0])
    for t in reversed(range(rewards.shape[1])):
        delta = rewards[:, t] + next_value - values[:, t]
        running = delta + lam * running
        advantages[:, t] = running
        next_value = values[:, t]
    return advantages, advantages + values


def ppo_loss(
    logp_new: Tensor,
    logp_old: Tensor,
    advantages: Tensor,
    values_new: Tensor,
    returns: Tensor,
    clip: float,
    vf_coef: float,
) -> tuple[Tensor, Tensor]:
    """-mean(min(rho * A, clip(rho, 1 - eps, 1 + eps) * A)) + vf_coef * mean((V - returns)^2),
    rho_t = exp(logp_new_t - logp_old_t), means over all B * T tokens.
    Returns the loss and the ratios rho (detached), [B, T]."""
    ratios = torch.exp(logp_new - logp_old)
    surrogate = torch.min(ratios * advantages, ratios.clamp(1 - clip, 1 + clip) * advantages)
    value_loss = ((values_new - returns) ** 2).mean()
    return -surrogate.mean() + vf_coef * value_loss, ratios.detach()


def _logp_and_values(policy: Policy, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """log pi(y_t | x, y_<t) and V(x, y_<t), both [B, T], from one forward pass."""
    logits, values = policy(x, y)
    logp = torch.log_softmax(logits, dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return logp, values


def train(
    ref: Policy,
    world: World,
    cfg: PPOConfig,
    reward_fn: Callable[[LongTensor, LongTensor], Tensor] | None = None,
    target_beta: float | None = None,
) -> Run:
    """Each iteration samples a batch from the current policy, fixes its advantages and
    returns, takes `cfg.epochs` full-batch gradient steps, then records.
    `reward_fn` defaults to the world's reward in `cfg.reward_mode`; `target_beta`
    (the beta the Recorder measures against) defaults to `cfg.beta`."""
    if reward_fn is None:
        def reward_fn(x, y):
            return world.reward(x, y, cfg.reward_mode)

    policy = ref.clone()
    ref_scorer = scorer(ref)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    rec = Recorder(world, ref, cfg.reward_mode, cfg.beta if target_beta is None else target_beta)
    rec.record(0, 0, policy)
    ratios = torch.ones(0)
    for it in range(1, cfg.iterations + 1):
        x = torch.randint(world.n_prompts, (cfg.batch_size,), generator=gen)
        y = policy.sample(x, gen)
        with torch.no_grad():
            logp_old, values_old = _logp_and_values(policy, x, y)
            ref_logp = ref_scorer.token_logp(x, y)
            rewards = token_rewards(reward_fn(x, y), logp_old, ref_logp, cfg.beta)
            advantages, returns = gae(rewards, values_old, cfg.lam)
        for _ in range(cfg.epochs):
            logp_new, values_new = _logp_and_values(policy, x, y)
            loss, ratios = ppo_loss(
                logp_new, logp_old, advantages, values_new, returns, cfg.clip, cfg.vf_coef
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        rec.record(it * cfg.epochs, it * cfg.batch_size, policy)
    return rec.finish(policy, cfg, ratios=ratios.flatten())
