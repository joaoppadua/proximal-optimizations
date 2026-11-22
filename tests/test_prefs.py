import torch

from proxopt.prefs import (
    PrefData,
    coverage_mask,
    covered_mass,
    fit_reward_model,
    reward_model_table,
    sample_pairs,
)
from proxopt.reference import load_reference, train_reference
from proxopt.world import DEFAULT_WORLD
from proxopt.world import TINY_WORLD as W

REF = train_reference(W, seed=0, n=512, steps=100)


def idx(y):
    return y[:, 0] * 3 + y[:, 1]


def centre(t):
    return t - t.mean(1, keepdim=True)


def test_pairs_are_distinct_and_deterministic():
    for world, ref, src in (
        (W, REF, "ref"),
        (DEFAULT_WORLD, load_reference(DEFAULT_WORLD), "ref"),
        (W, REF, "uniform"),
    ):
        d = sample_pairs(world, ref, 500, 0.25, src, seed=0)
        assert len(d) == 500 and d.winners.shape == (500, world.length)
        assert (d.winners != d.losers).any(1).all()
        assert torch.equal(d.winners, sample_pairs(world, ref, 500, 0.25, src, seed=0).winners)


def test_bradley_terry_frequencies():
    tau = 0.5
    d = sample_pairs(W, REF, 40_000, tau, "uniform", seed=0)
    rw, rl = W.reward(d.prompts, d.winners), W.reward(d.prompts, d.losers)
    for gap_ in (0.25, 0.5):  # reward differences that occur in TINY_WORLD
        hi, lo = (rw - rl - gap_).abs() < 1e-6, (rl - rw - gap_).abs() < 1e-6
        share = hi.sum() / (hi.sum() + lo.sum())
        assert abs(share - torch.sigmoid(torch.tensor(gap_ / tau))) < 0.03


def test_sharp_preferences_always_pick_the_better_response():
    d = sample_pairs(W, REF, 2000, 1e-3, "uniform", seed=0)
    assert (W.reward(d.prompts, d.winners) >= W.reward(d.prompts, d.losers)).all()


def test_coverage_mask():
    d = sample_pairs(W, REF, 3, 0.25, "uniform", seed=0)
    m = coverage_mask(W, d)
    assert m.shape == (2, 9) and m.sum() <= 6
    assert m[d.prompts, idx(d.winners)].all() and m[d.prompts, idx(d.losers)].all()


def test_covered_mass_by_hand():
    # prompt 0 compares responses 0 and 1, prompt 1 compares responses 8 and 3
    d = PrefData(
        torch.tensor([0, 1]), torch.tensor([[0, 0], [2, 2]]), torch.tensor([[0, 1], [1, 0]])
    )
    probs = torch.stack([torch.arange(1.0, 10.0) / 45, torch.full((9,), 1 / 9)])
    expected = (3 / 45 + 2 / 9) / 2
    assert abs(covered_mass(W, d, probs.log()) - expected) < 1e-6


def test_full_coverage_gives_mass_one():
    y = W.all_responses()
    d = PrefData(torch.tensor([0] * 9 + [1] * 9), torch.cat([y, y]), torch.cat([y, y]).roll(1, 0))
    assert abs(covered_mass(W, d, REF.all_logp().detach()) - 1.0) < 1e-6


def test_reward_model_learns_r_over_tau_up_to_a_constant():
    tau = 0.5
    d = sample_pairs(W, REF, 20_000, tau, "uniform", seed=0)
    est = centre(reward_model_table(fit_reward_model(W, d, seed=0), W).detach())
    true = centre(W.reward_table() / tau)
    slope = (est * true).sum() / (true * true).sum()
    corr = torch.corrcoef(torch.stack([est.flatten(), true.flatten()]))[0, 1]
    assert corr > 0.95 and 0.75 < slope < 1.25
