import torch

from proxopt import exact
from proxopt.reference import load_reference, sample_corpus, train_reference
from proxopt.world import DEFAULT_WORLD as W
from proxopt.world import TINY_WORLD


def test_corpus_style():
    x, y = sample_corpus(W, 4096, seed=0)
    assert x.shape == (4096,) and y.shape == (4096, 5)
    step = (y[:, 1:] - y[:, :-1]) % 8
    assert ((step == 0) | (step == 1)).float().mean() > 0.8  # expected 0.85


def test_reference_is_deterministic_frozen_and_full_support():
    a, b = train_reference(W, seed=0), train_reference(W, seed=0)
    assert torch.equal(a.all_logp(), b.all_logp())
    assert torch.isfinite(a.all_logp()).all()
    assert not any(p.requires_grad for p in a.parameters())


def test_reference_has_the_style_and_is_mediocre():
    ref = load_reference(W)
    lp = ref.all_logp()
    r = W.reward_table("graded")
    y = ref.sample(torch.zeros(4000, dtype=torch.long), torch.Generator().manual_seed(0))
    step = (y[:, 1:] - y[:, :-1]) % 8
    assert ((step == 0) | (step == 1)).float().mean() > 0.6
    opt = exact.optimum(lp, r, 0.1)
    assert exact.expected_reward(lp, r) < exact.expected_reward(opt, r) - 0.1
    assert exact.gap(lp, lp, r, 0.1) > 0.05


def test_load_reference_returns_fresh_copies():
    assert load_reference(W) is not load_reference(W)


def test_train_reference_works_inside_no_grad():
    with torch.no_grad():
        inside = train_reference(TINY_WORLD, seed=0, steps=5)
    outside = train_reference(TINY_WORLD, seed=0, steps=5)
    assert torch.equal(inside.all_logp(), outside.all_logp())
