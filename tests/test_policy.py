import torch

from proxopt.policy import Policy
from proxopt.world import TINY_WORLD, World

ONE_STEP = World(vocab_size=8, length=1, targets=(3,))


def test_all_logp_is_a_distribution():
    for w in (TINY_WORLD, ONE_STEP):
        torch.manual_seed(0)
        lp = Policy(w).all_logp()
        assert lp.shape == (w.n_prompts, w.n_responses)
        assert torch.allclose(lp.logsumexp(1), torch.zeros(w.n_prompts), atol=1e-5)


def test_token_logp_sums_to_all_logp():
    w = TINY_WORLD
    torch.manual_seed(0)
    p = Policy(w)
    x = torch.zeros(w.n_responses, dtype=torch.long)
    assert torch.allclose(p.token_logp(x, w.all_responses()).sum(1), p.all_logp()[0], atol=1e-5)


def test_logits_do_not_see_the_token_they_predict():
    w = World()
    torch.manual_seed(0)
    p = Policy(w)
    x = torch.tensor([0, 0])
    y = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0]])
    logits, values = p(x, y)
    assert values.shape == (2, 5)
    assert torch.allclose(logits[0], logits[1]) and torch.allclose(values[0], values[1])


def test_sample_matches_distribution():
    w = TINY_WORLD
    torch.manual_seed(0)
    p = Policy(w)
    g = torch.Generator().manual_seed(0)
    y = p.sample(torch.zeros(20000, dtype=torch.long), generator=g)
    idx = y[:, 0] * 3 + y[:, 1]
    freq = torch.bincount(idx, minlength=9) / 20000
    assert (freq - p.all_logp()[0].exp()).abs().max() < 0.02


def test_clone_is_independent():
    w = TINY_WORLD
    torch.manual_seed(0)
    p = Policy(w)
    q = p.clone()
    assert torch.equal(p.all_logp(), q.all_logp())
    with torch.no_grad():
        for prm in q.parameters():
            prm.add_(1.0)
    assert not torch.equal(p.all_logp(), q.all_logp())
    assert all(prm.requires_grad for prm in q.parameters())


def test_fresh_policy_values_are_zero():
    pol = Policy(TINY_WORLD)
    x = torch.arange(TINY_WORLD.n_prompts)
    y = pol.sample(x, torch.Generator().manual_seed(0))
    _, values = pol(x, y)
    assert torch.equal(values, torch.zeros_like(values))
