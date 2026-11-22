"""Bradley-Terry preference pairs and a reward model fitted to them."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from proxopt.policy import Policy
from proxopt.world import World


@dataclass(frozen=True)
class PrefData:
    """M preference pairs: for prompt `prompts[i]`, `winners[i]` was preferred to `losers[i]`."""

    prompts: torch.Tensor  # [M]
    winners: torch.Tensor  # [M, T]
    losers: torch.Tensor  # [M, T]

    def __len__(self) -> int:
        return self.prompts.shape[0]


def sample_pairs(
    world: World,
    ref: Policy,
    n_pairs: int,
    tau: float,
    source: Literal["ref", "uniform"],
    reward_mode: str = "graded",
    seed: int = 0,
) -> PrefData:
    """Uniform prompts; two different responses drawn from `source`; the first wins with
    probability sigma((r_1 - r_2) / tau). All randomness comes from one generator."""
    g = torch.Generator().manual_seed(seed)
    prompts = torch.randint(world.n_prompts, (n_pairs,), generator=g)
    table = world.all_responses()

    def draw(p: torch.Tensor) -> torch.Tensor:
        if source == "ref":
            return ref.sample(p, generator=g)
        if source == "uniform":
            return table[torch.randint(world.n_responses, (p.shape[0],), generator=g)]
        raise ValueError(f"unknown source: {source!r}")

    y1, y2 = draw(prompts), draw(prompts)
    tied = (y1 == y2).all(dim=1)
    while tied.any():
        y2[tied] = draw(prompts[tied])
        tied = (y1 == y2).all(dim=1)

    gap = world.reward(prompts, y1, reward_mode) - world.reward(prompts, y2, reward_mode)
    first_wins = torch.rand(n_pairs, generator=g) < torch.sigmoid(gap / tau)
    w = first_wins.unsqueeze(1)
    return PrefData(prompts, torch.where(w, y1, y2), torch.where(w, y2, y1))


def coverage_mask(world: World, data: PrefData) -> torch.Tensor:
    """[P, N] bool: true where the response (in `World.all_responses()` order) appears,
    as winner or loser, in a pair for that prompt."""
    powers = world.vocab_size ** torch.arange(world.length - 1, -1, -1)
    mask = torch.zeros(world.n_prompts, world.n_responses, dtype=torch.bool)
    for y in (data.winners, data.losers):
        mask[data.prompts, (y * powers).sum(dim=-1)] = True
    return mask


def covered_mass(world: World, data: PrefData, logp: torch.Tensor) -> float:
    """Mean over prompts of the probability mass `logp.exp()` ([P, N]) puts on the
    responses that appear in `data` for that prompt."""
    return (logp.exp() * coverage_mask(world, data)).sum(1).mean().item()


class RewardModel(nn.Module):
    """Scalar score r_hat(x, y). A GRU reads the whole response, seeded with the prompt
    embedding at every step; a linear head reads its final state."""

    def __init__(self, world: World, hidden: int = 64):
        super().__init__()
        self.world = world
        self.token_emb = nn.Embedding(world.vocab_size, hidden)
        self.prompt_emb = nn.Embedding(world.n_prompts, hidden)
        self.gru = nn.GRU(2 * hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, prompts: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        px = self.prompt_emb(prompts).unsqueeze(1).expand(-1, responses.shape[1], -1)
        out, _ = self.gru(torch.cat([self.token_emb(responses), px], dim=-1))
        return self.head(out[:, -1]).squeeze(-1)


def fit_reward_model(
    world: World,
    data: PrefData,
    steps: int = 500,
    lr: float = 3e-3,
    batch_size: int = 256,
    seed: int = 0,
) -> RewardModel:
    """Minimise -log sigma(r_hat(winner) - r_hat(loser)) with Adam on minibatches. Learns
    r / tau up to a constant per prompt. Deterministic given the arguments; the caller's
    global torch random state is restored afterwards."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = RewardModel(world)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        i = torch.randint(len(data), (min(batch_size, len(data)),), generator=g)
        p, w, l = data.prompts[i], data.winners[i], data.losers[i]
        opt.zero_grad()
        loss = -nn.functional.logsigmoid(model(p, w) - model(p, l)).mean()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def reward_model_table(model: RewardModel, world: World) -> torch.Tensor:
    """r_hat for every prompt and response, [P, N], in `World.all_responses()` order."""
    y = world.all_responses()
    return torch.stack(
        [model(torch.full((world.n_responses,), p), y) for p in range(world.n_prompts)]
    )
