import math

import pytest
import torch

from proxopt.algos import grpo
from proxopt.algos.grpo import GRPOConfig, group_advantages, grpo_loss, kl_estimator
from proxopt.reference import load_reference
from proxopt.world import DEFAULT_WORLD
from tests.helpers import flat_grad


def test_group_advantages():
    r = torch.rand(4, 8, generator=torch.Generator().manual_seed(0))
    for std in (True, False):
        assert group_advantages(r, std).mean(1).abs().max() < 1e-5
    assert torch.allclose(group_advantages(r, False), r - r.mean(1, keepdim=True))
    assert (group_advantages(r, True).std(1) - 1).abs().max() < 1e-3


def test_identical_group_gives_zero_not_nan():  # Review Focus 2, spec §9
    for std in (True, False):
        assert torch.equal(group_advantages(torch.ones(3, 8), std), torch.zeros(3, 8))
    with pytest.raises(ValueError):
        group_advantages(torch.rand(4, 1), True)


def test_zero_advantage_gives_zero_policy_gradient():
    w = DEFAULT_WORLD
    pol = load_reference(w).clone()
    x = torch.zeros(8, dtype=torch.long)
    y = pol.sample(x, torch.Generator().manual_seed(0))
    old = pol.token_logp(x, y).detach()
    loss, _ = grpo_loss(pol.token_logp(x, y), old, old, torch.zeros(8), 0.2, 0.0, True)
    assert flat_grad(loss, pol).abs().max() == 0


def test_kl_estimator_value_and_gradient_direction():  # spec §6.2 and §9
    # one next-token distribution, far from the reference
    logits = torch.tensor([2.0, 0.0, -1.0, 0.5], requires_grad=True)
    lt, lr_ = logits.log_softmax(0), torch.tensor([-1.0, 1.5, 0.0, 0.3]).log_softmax(0)
    kl_theta_ref = (lt.exp() * (lt - lr_)).sum()
    kl_ref_theta = (lr_.exp() * (lr_ - lt)).sum()
    # expectation over y ~ π_θ, samples held fixed
    est = (lt.exp().detach() * kl_estimator(lt, lr_)).sum()
    assert torch.allclose(est, kl_theta_ref, atol=1e-6)
    g_est, g_rt, g_tr = (flat_grad(s, [logits]) for s in (est, kl_ref_theta, kl_theta_ref))
    assert torch.allclose(g_est, g_rt, atol=1e-6)
    assert (g_est - g_tr).abs().max() > 0.05


def test_verifiable_reward_with_pairs_stays_finite():  # Review Focus 2
    cfg = GRPOConfig(reward_mode="verifiable", group_size=2, iterations=8)
    run = grpo.train(load_reference(DEFAULT_WORLD), DEFAULT_WORLD, cfg)
    assert all(math.isfinite(v) for vs in run.metrics.values() for v in vs)
    assert len(run.extras["dead_groups"]) == 8
    assert max(run.extras["dead_groups"]) <= DEFAULT_WORLD.n_prompts


def test_baseline_comparison_keys():
    ref = load_reference(DEFAULT_WORLD)
    out = grpo.baseline_comparison(ref.clone(), ref, DEFAULT_WORLD, 8, 0.1, "graded", 0)
    assert set(out) == {"exact", "group", "critic"} and all(v.shape == (32,) for v in out.values())


@pytest.mark.slow
def test_published_grpo_raises_reward_and_ppo_like_grpo_closes_the_gap():  # spec §9
    ref = load_reference(DEFAULT_WORLD)
    cfg = GRPOConfig()
    pub = grpo.train(ref, DEFAULT_WORLD, cfg)
    assert pub.metrics["reward"][-1] > pub.metrics["reward"][0] + 0.1
    like = grpo.train(ref, DEFAULT_WORLD, GRPOConfig(std_normalize=False, kl_in="reward"))
    assert like.metrics["gap"][-1] < 0.25 * like.metrics["gap"][0]
    assert pub.feedback[-1] == cfg.iterations * DEFAULT_WORLD.n_prompts * cfg.group_size
    assert pub.metrics == grpo.train(ref, DEFAULT_WORLD, cfg).metrics
