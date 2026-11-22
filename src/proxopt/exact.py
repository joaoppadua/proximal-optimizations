"""Exact quantities over enumerated responses (spec section 5.3).

Every function takes plain tables of shape ``[P, N]`` (P prompts, N responses):
``logp`` and ``logp_ref`` hold log-probabilities (rows are normalised) and
``reward`` holds ``r(x, y)``. Scalars are means over prompts. All functions
are differentiable with respect to ``logp`` and work in log space.
"""

from collections.abc import Sequence

import torch
from torch import Tensor


def _prompt_kl(logp_a: Tensor, logp_b: Tensor) -> Tensor:
    """Per-prompt KL(a || b), with 0 * log 0 taken as 0."""
    p = logp_a.exp()
    diff = torch.where(p > 0, logp_a - logp_b, torch.zeros_like(logp_a))
    return (p * diff).sum(1)


def _prompt_reward(logp: Tensor, reward: Tensor) -> Tensor:
    return (logp.exp() * reward).sum(1)


def expected_reward(logp: Tensor, reward: Tensor) -> Tensor:
    """E_pi[r], averaged over prompts."""
    return _prompt_reward(logp, reward).mean()


def kl(logp_a: Tensor, logp_b: Tensor) -> Tensor:
    """KL(a || b), averaged over prompts."""
    return _prompt_kl(logp_a, logp_b).mean()


def prompt_values(logp: Tensor, logp_ref: Tensor, reward: Tensor, beta: float) -> Tensor:
    """Per-prompt J_beta(pi) = E_pi[r] - beta * KL(pi || pi_ref); shape [P]."""
    return _prompt_reward(logp, reward) - beta * _prompt_kl(logp, logp_ref)


def objective(logp: Tensor, logp_ref: Tensor, reward: Tensor, beta: float) -> Tensor:
    """J_beta(pi), averaged over prompts."""
    return prompt_values(logp, logp_ref, reward, beta).mean()


def optimum(logp_ref: Tensor, reward: Tensor, beta: float) -> Tensor:
    """log pi*_beta, with pi*_beta proportional to pi_ref * exp(r / beta).

    Normalised with ``log_softmax`` (logsumexp), so tiny beta stays finite.
    """
    return (logp_ref + reward / beta).log_softmax(1)


def gap(logp: Tensor, logp_ref: Tensor, reward: Tensor, beta: float) -> Tensor:
    """J_beta(pi*_beta) - J_beta(pi) = beta * KL(pi || pi*_beta)."""
    return beta * kl(logp, optimum(logp_ref, reward, beta))


def drift_ceiling(logp_ref: Tensor, reward: Tensor) -> Tensor:
    """The beta -> 0 limit of KL(pi*_beta || pi_ref), averaged over prompts.

    As beta -> 0, pi*_beta becomes pi_ref restricted to the responses of maximal reward,
    so the limit is -log pi_ref(argmax_y r(x, y)) per prompt. No pi*_beta drifts further."""
    best = reward == reward.max(1, keepdim=True).values
    masked = torch.where(best, logp_ref, torch.full_like(logp_ref, -torch.inf))
    return -masked.logsumexp(1).mean()


def frontier(
    logp_ref: Tensor, reward: Tensor, betas: Sequence[float]
) -> list[tuple[float, float]]:
    """(KL(pi*_beta || pi_ref), E_{pi*_beta}[r]) for each beta."""
    points = []
    for beta in betas:
        opt = optimum(logp_ref, reward, beta)
        points.append((kl(opt, logp_ref).item(), expected_reward(opt, reward).item()))
    return points


def evaluate(
    logp: Tensor, logp_ref: Tensor, reward: Tensor, beta: float
) -> dict[str, float]:
    """All yardstick quantities for one policy, as plain floats."""
    return {
        "reward": expected_reward(logp, reward).item(),
        "kl": kl(logp, logp_ref).item(),
        "kl_rev": kl(logp_ref, logp).item(),
        "objective": objective(logp, logp_ref, reward, beta).item(),
        "gap": gap(logp, logp_ref, reward, beta).item(),
    }
