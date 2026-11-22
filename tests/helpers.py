"""Shared test helpers."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def flat_grad(scalar: Tensor, params: nn.Module | Sequence[Tensor]) -> Tensor:
    """Gradient of `scalar` with respect to a module's parameters (or the given
    tensors), flattened into one vector. Parameters unused by `scalar` are skipped.
    The graph is kept, so several scalars that share intermediate tensors can each be
    differentiated."""
    tensors = list(params.parameters()) if isinstance(params, nn.Module) else list(params)
    grads = torch.autograd.grad(scalar, tensors, allow_unused=True, retain_graph=True)
    return torch.cat([g.reshape(-1) for g in grads if g is not None])
