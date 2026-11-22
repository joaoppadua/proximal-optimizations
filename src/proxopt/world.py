"""The toy world: a prompt is a target sum, a response is a token sequence."""

from dataclasses import dataclass
from functools import cache
from typing import Literal

import torch


@cache
def _enumerate(vocab_size: int, length: int) -> torch.Tensor:
    n = vocab_size**length
    idx = torch.arange(n).unsqueeze(1)
    powers = vocab_size ** torch.arange(length - 1, -1, -1)
    return (idx // powers) % vocab_size


@dataclass(frozen=True)
class World:
    """Prompt x is an index into `targets`; response y has `length` tokens in [0, vocab_size)."""

    vocab_size: int = 8
    length: int = 5
    targets: tuple[int, ...] = (7, 14, 21, 28)

    def __post_init__(self):
        for t in self.targets:
            if not 0 <= t <= self.max_sum:
                raise ValueError(f"target {t} is unreachable: sums lie in [0, {self.max_sum}]")

    @property
    def n_prompts(self) -> int:
        return len(self.targets)

    @property
    def n_responses(self) -> int:
        return self.vocab_size**self.length

    @property
    def max_sum(self) -> int:
        return (self.vocab_size - 1) * self.length

    def all_responses(self) -> torch.Tensor:
        """All responses in lexicographic order, shape [N, T]. Cached; do not mutate."""
        return _enumerate(self.vocab_size, self.length)

    def reward(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        mode: Literal["graded", "verifiable"] = "graded",
    ) -> torch.Tensor:
        target = torch.tensor(self.targets)[prompts]
        gap = (responses.sum(dim=-1) - target).abs()
        if mode == "graded":
            return 1.0 - gap.float() / self.max_sum
        if mode == "verifiable":
            return (gap == 0).float()
        raise ValueError(f"unknown reward mode: {mode!r}")

    def reward_table(self, mode: Literal["graded", "verifiable"] = "graded") -> torch.Tensor:
        """Reward of every response under every prompt, shape [P, N]."""
        y = self.all_responses()
        return torch.stack(
            [
                self.reward(torch.full((self.n_responses,), p), y, mode)
                for p in range(self.n_prompts)
            ]
        )


DEFAULT_WORLD = World()
TINY_WORLD = World(vocab_size=3, length=2, targets=(1, 3))
