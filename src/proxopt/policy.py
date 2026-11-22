"""Autoregressive policy pi_theta(y | x) with a value head, for the toy world."""

import copy

import torch
from torch import nn

from proxopt.world import World


class Policy(nn.Module):
    """One-layer GRU. At step t it reads the previous token (a start symbol at t = 0)
    and a prompt embedding, and emits logits for y_t and a value for the state before y_t."""

    def __init__(self, world: World, hidden: int = 64):
        super().__init__()
        self.world = world
        self.hidden = hidden
        v = world.vocab_size
        self.token_emb = nn.Embedding(v + 1, hidden)  # index v is the start symbol
        self.prompt_emb = nn.Embedding(world.n_prompts, hidden)
        self.gru = nn.GRU(2 * hidden, hidden, batch_first=True)
        self.policy_head = nn.Linear(hidden, v)
        self.value_head = nn.Linear(hidden, 1)

    def _step(self, prev: torch.Tensor, px: torch.Tensor, h: torch.Tensor):
        """One GRU step. prev, [B]; px (prompt embedding), [B, H]; h, [B, H].
        Returns (logits [B, V], value [B], new h [B, H])."""
        inp = torch.cat([self.token_emb(prev), px], dim=-1).unsqueeze(1)
        out, h_new = self.gru(inp, h.unsqueeze(0).contiguous())
        out = out[:, 0]
        return self.policy_head(out), self.value_head(out).squeeze(-1), h_new[0]

    def forward(self, prompts: torch.Tensor, responses: torch.Tensor):
        """Teacher-forced. Returns logits [B, T, V] and values [B, T]; entry t
        depends on x and y_<t only."""
        b = responses.shape[0]
        start = torch.full((b, 1), self.world.vocab_size, dtype=torch.long)
        prev = torch.cat([start, responses[:, :-1]], dim=1)
        px = self.prompt_emb(prompts).unsqueeze(1).expand(-1, prev.shape[1], -1)
        out, _ = self.gru(torch.cat([self.token_emb(prev), px], dim=-1))
        return self.policy_head(out), self.value_head(out).squeeze(-1)

    def token_logp(self, prompts: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """log pi(y_t | x, y_<t) for each given token, [B, T]."""
        logits, _ = self(prompts, responses)
        logp = torch.log_softmax(logits, dim=-1)
        return logp.gather(-1, responses.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def sample(
        self, prompts: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Ancestral samples, [B, T], no gradient."""
        b = prompts.shape[0]
        px = self.prompt_emb(prompts)
        prev = torch.full((b,), self.world.vocab_size, dtype=torch.long)
        h = torch.zeros(b, self.hidden)
        tokens = []
        for _ in range(self.world.length):
            logits, _, h = self._step(prev, px, h)
            prev = torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)
            tokens.append(prev)
        return torch.stack(tokens, dim=1)

    def all_logp(self) -> torch.Tensor:
        """log pi(y | x) for every prompt and every response, [P, N], differentiable.

        Walks the tree of shared prefixes level by level, so each prefix is
        evaluated once. Responses come out in the lexicographic order of
        `World.all_responses()`."""
        w = self.world
        p, v = w.n_prompts, w.vocab_size
        px = self.prompt_emb.weight  # [P, H]
        # state per (prompt, prefix): flattened as [P * V^t]
        h = torch.zeros(p, self.hidden)
        prev = torch.full((p,), v, dtype=torch.long)
        pxs = px
        logp = torch.zeros(p, 1)  # [P, V^t]
        for t in range(w.length):
            logits, _, h = self._step(prev, pxs, h)  # batch P * V^t
            step = torch.log_softmax(logits, dim=-1).view(p, -1, v)
            logp = (logp.unsqueeze(-1) + step).reshape(p, -1)  # [P, V^(t+1)]
            if t + 1 < w.length:
                n = step.shape[1]
                h = h.repeat_interleave(v, dim=0)
                pxs = px.repeat_interleave(n * v, dim=0)
                prev = torch.arange(v).repeat(p * n)
        return logp

    def clone(self) -> "Policy":
        """Independent deep copy with trainable parameters."""
        return copy.deepcopy(self)
