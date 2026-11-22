"""The reference policy: a stand-in for an SFT model, trained by maximum likelihood
on a synthetic corpus that has a style but knows nothing about the reward."""

from functools import cache

import torch

from proxopt.policy import Policy
from proxopt.world import DEFAULT_WORLD, World


def sample_corpus(world: World, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Prompts [n] and responses [n, T]. Prompts are uniform and the response does not
    depend on them. The first token is uniform; each later token repeats the previous
    one (probability 0.4), is the previous plus 1 modulo V (0.4), or is uniform (0.2)."""
    g = torch.Generator().manual_seed(seed)
    v, length = world.vocab_size, world.length
    prompts = torch.randint(world.n_prompts, (n,), generator=g)
    tokens = [torch.randint(v, (n,), generator=g)]
    for _ in range(length - 1):
        prev = tokens[-1]
        u = torch.rand(n, generator=g)
        uniform = torch.randint(v, (n,), generator=g)
        tokens.append(torch.where(u < 0.4, prev, torch.where(u < 0.8, (prev + 1) % v, uniform)))
    return prompts, torch.stack(tokens, dim=1)


def train_reference(
    world: World = DEFAULT_WORLD,
    seed: int = 0,
    n: int = 4096,
    steps: int = 100,
    lr: float = 1e-2,
) -> Policy:
    """Maximum-likelihood training on `sample_corpus`. Deterministic given the arguments;
    the caller's global torch random state is restored afterwards. Returns a frozen policy.
    Training runs with gradients enabled, so a first call inside `torch.no_grad()` works."""
    prompts, responses = sample_corpus(world, n, seed)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        policy = Policy(world)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            loss = -policy.token_logp(prompts, responses).sum(dim=-1).mean()
            loss.backward()
            opt.step()
    return policy.requires_grad_(False)


@cache
def _cached_reference(world: World, seed: int) -> Policy:
    return train_reference(world, seed)


def load_reference(world: World = DEFAULT_WORLD, seed: int = 0) -> Policy:
    """Cached `train_reference`; each call returns a fresh frozen clone."""
    return _cached_reference(world, seed).clone().requires_grad_(False)
