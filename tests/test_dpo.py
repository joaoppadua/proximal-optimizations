import math

import pytest
import torch

from proxopt import exact
from proxopt.algos import dpo
from proxopt.algos.dpo import DPOConfig, dpo_loss, implicit_reward
from proxopt.prefs import sample_pairs
from proxopt.reference import load_reference, train_reference
from proxopt.world import DEFAULT_WORLD
from proxopt.world import TINY_WORLD as W

REF = train_reference(W, seed=0, n=512, steps=100)


def test_loss_values():
    z = torch.zeros(4)
    assert torch.allclose(dpo_loss(z, z, z, z, 0.5), torch.tensor(math.log(2)))

    def t(v):
        return torch.tensor([v])

    assert torch.allclose(
        dpo_loss(t(-1.0), t(-3.0), t(-2.0), t(-2.0), 0.5),
        -torch.nn.functional.logsigmoid(torch.tensor(1.0)),
    )


def test_dpo_approaches_the_optimum_for_beta_times_tau():  # spec §6.3 and §9
    tau, beta = 0.5, 0.4
    data = sample_pairs(W, REF, 50_000, tau, "uniform", seed=0)
    run = dpo.train(REF, W, data, DPOConfig(beta=beta, steps=1500), tau)
    assert run.target_beta == pytest.approx(0.2)
    assert run.metrics["gap"][-1] < 0.1 * run.metrics["gap"][0]
    lp, ref_lp, r = run.policy.all_logp().detach(), REF.all_logp(), W.reward_table()
    assert exact.kl(lp, exact.optimum(ref_lp, r, beta * tau)) < exact.kl(
        lp, exact.optimum(ref_lp, r, beta)
    )
    est = implicit_reward(run.policy, REF, beta).detach()
    true = W.reward_table() / tau
    true = true - true.mean(1, keepdim=True)
    assert est.mean(1).abs().max() < 1e-5
    assert torch.corrcoef(torch.stack([est.flatten(), true.flatten()]))[0, 1] > 0.95


def test_fewer_pairs_than_a_batch():  # Review Focus 3
    data = sample_pairs(W, REF, 8, 0.5, "uniform", seed=0)
    run = dpo.train(REF, W, data, DPOConfig(steps=30), 0.5)
    assert all(math.isfinite(v) for vs in run.metrics.values() for v in vs)
    assert run.feedback[-1] == 8


@pytest.mark.slow
def test_dpo_reduces_gap_on_the_default_world_and_is_deterministic():
    ref = load_reference(DEFAULT_WORLD)
    data = sample_pairs(DEFAULT_WORLD, ref, 4000, 0.25, "ref", seed=0)
    a = dpo.train(ref, DEFAULT_WORLD, data, DPOConfig(), 0.25)
    assert a.metrics["gap"][-1] < 0.5 * a.metrics["gap"][0]
    assert a.metrics == dpo.train(ref, DEFAULT_WORLD, data, DPOConfig(), 0.25).metrics
