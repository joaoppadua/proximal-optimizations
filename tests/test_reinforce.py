import json

import pytest
import torch

from proxopt import exact
from proxopt.algos import reinforce
from proxopt.algos.reinforce import ReinforceConfig, reinforce_loss
from proxopt.reference import load_reference, train_reference
from proxopt.train import Run
from proxopt.world import DEFAULT_WORLD, TINY_WORLD
from tests.helpers import flat_grad


def test_gradient_estimates_the_gradient_of_J():
    w = TINY_WORLD
    ref = train_reference(w, seed=0, n=512, steps=100)
    pol = ref.clone()
    beta = 0.3
    with torch.no_grad():  # move away from ref so the KL term matters
        for p in pol.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=torch.Generator().manual_seed(0)))
    r = w.reward_table("graded")
    g_exact = flat_grad(-exact.objective(pol.all_logp(), ref.all_logp(), r, beta), pol)
    g = torch.Generator().manual_seed(0)
    x = torch.randint(w.n_prompts, (200_000,), generator=g)
    y = pol.sample(x, g)
    loss = reinforce_loss(pol.token_logp(x, y), ref.token_logp(x, y), w.reward(x, y), beta, "none")
    g_est = flat_grad(loss, pol)
    assert torch.cosine_similarity(g_exact, g_est, dim=0) > 0.98


def test_run_record_and_determinism():
    ref = load_reference(DEFAULT_WORLD)
    cfg = ReinforceConfig(steps=40)
    a, b = reinforce.train(ref, DEFAULT_WORLD, cfg), reinforce.train(ref, DEFAULT_WORLD, cfg)
    assert a.metrics == b.metrics
    assert a.steps[0] == 0 and a.steps[-1] == 40 and a.feedback[-1] == 40 * cfg.batch_size
    assert set(a.metrics) == {"reward", "kl", "kl_rev", "objective", "gap"}
    assert a.target_beta == cfg.beta and abs(a.metrics["kl"][0]) < 1e-6
    back = Run.from_json_dict(json.loads(json.dumps(a.to_json_dict())))
    assert back.policy is None and (back.steps, back.feedback, back.metrics) == (
        a.steps,
        a.feedback,
        a.metrics,
    )


@pytest.mark.slow
def test_reinforce_reduces_gap():
    run = reinforce.train(load_reference(DEFAULT_WORLD), DEFAULT_WORLD, ReinforceConfig())
    assert run.metrics["gap"][-1] < 0.5 * run.metrics["gap"][0]
