import math

import pytest
import torch

from proxopt.algos import ppo
from proxopt.algos.ppo import PPOConfig, gae, ppo_loss, token_rewards
from proxopt.reference import load_reference, train_reference
from proxopt.world import DEFAULT_WORLD, TINY_WORLD
from tests.helpers import flat_grad


def test_gae_limits():
    g = torch.Generator().manual_seed(0)
    r = torch.randn(6, 5, generator=g)
    v = torch.randn(6, 5, generator=g)
    adv1, ret1 = gae(r, v, 1.0)
    to_go = r.flip(1).cumsum(1).flip(1)
    assert torch.allclose(adv1, to_go - v, atol=1e-6) and torch.allclose(ret1, to_go, atol=1e-6)
    adv0, _ = gae(r, v, 0.0)
    v_next = torch.cat([v[:, 1:], torch.zeros(6, 1)], 1)
    assert torch.allclose(adv0, r + v_next - v, atol=1e-6)


def test_token_rewards():
    g = torch.Generator().manual_seed(0)
    lo = torch.randn(4, 5, generator=g)
    lr_ = torch.randn(4, 5, generator=g)
    r = torch.rand(4, generator=g)
    tr = token_rewards(r, lo, lr_, 0.2)
    assert torch.allclose(tr.sum(1), r - 0.2 * (lo - lr_).sum(1), atol=1e-6)
    assert torch.allclose(tr[:, :-1], -0.2 * (lo - lr_)[:, :-1], atol=1e-6)


def test_first_step_gradient_equals_reinforce_for_any_clip():  # spec §9
    w = TINY_WORLD
    pol = train_reference(w, seed=0, n=512, steps=100).clone()
    g = torch.Generator().manual_seed(0)
    x = torch.randint(w.n_prompts, (256,), generator=g)
    y = pol.sample(x, g)
    A = torch.randn(256, generator=g)
    old = pol.token_logp(x, y).detach()
    z = torch.zeros(256, w.length)
    g_rf = flat_grad(-(A * pol.token_logp(x, y).sum(1)).mean() / w.length, pol)
    for clip in (0.05, 0.2, 10.0):
        loss, ratios = ppo_loss(
            pol.token_logp(x, y), old, A[:, None].expand(256, w.length), z, z, clip, 0.0
        )
        assert torch.allclose(ratios, torch.ones_like(ratios), atol=1e-6)
        assert torch.allclose(flat_grad(loss, pol), g_rf, atol=1e-6)


def test_zero_reward_keeps_the_policy_at_the_reference():
    ref = load_reference(DEFAULT_WORLD)
    run = ppo.train(
        ref, DEFAULT_WORLD, PPOConfig(iterations=10), reward_fn=lambda x, y: torch.zeros(len(x))
    )
    assert run.metrics["kl"][-1] < 0.02


def test_target_beta_defaults_to_cfg_beta_and_can_be_overridden():
    ref = load_reference(DEFAULT_WORLD)
    cfg = PPOConfig(beta=0.05, iterations=1)
    assert ppo.train(ref, DEFAULT_WORLD, cfg).target_beta == 0.05
    assert ppo.train(ref, DEFAULT_WORLD, cfg, target_beta=0.3).target_beta == 0.3


@pytest.mark.parametrize(
    "kw",
    [{"clip": 0.01}, {"clip": 10.0, "epochs": 8}, {"epochs": 1}, {"beta": 0.02}, {"lam": 0.0}],
)
def test_extreme_controls_stay_finite(kw):  # Review Focus 4
    run = ppo.train(load_reference(DEFAULT_WORLD), DEFAULT_WORLD, PPOConfig(iterations=8, **kw))
    assert all(math.isfinite(v) for vs in run.metrics.values() for v in vs)
    assert torch.isfinite(run.extras["ratios"]).all()


@pytest.mark.slow
def test_ppo_reduces_gap_and_is_deterministic():
    ref = load_reference(DEFAULT_WORLD)
    cfg = PPOConfig()
    a = ppo.train(ref, DEFAULT_WORLD, cfg)
    assert a.metrics["gap"][-1] < 0.25 * a.metrics["gap"][0]
    assert a.metrics == ppo.train(ref, DEFAULT_WORLD, cfg).metrics
    assert a.feedback[-1] == cfg.iterations * cfg.batch_size
    assert a.steps[-1] == cfg.iterations * cfg.epochs


def test_larger_beta_stays_closer_to_the_reference():
    ref = load_reference(DEFAULT_WORLD)
    kl = {
        beta: ppo.train(ref, DEFAULT_WORLD, PPOConfig(beta=beta, iterations=20)).metrics["kl"][-1]
        for beta in (0.05, 1.0)
    }
    assert kl[1.0] < kl[0.05]
