import itertools

import torch

from proxopt import exact


def tables(seed=0, P=3, N=50):
    g = torch.Generator().manual_seed(seed)
    logp = torch.randn(P, N, generator=g).log_softmax(1)
    ref = torch.randn(P, N, generator=g).log_softmax(1)
    return logp, ref, torch.rand(P, N, generator=g)


def test_gap_identity():  # spec §5.3
    logp, ref, r = tables()
    beta = 0.3
    opt = exact.optimum(ref, r, beta)
    lhs = exact.objective(opt, ref, r, beta) - exact.objective(logp, ref, r, beta)
    assert torch.allclose(lhs, beta * exact.kl(logp, opt), atol=1e-5)
    assert torch.allclose(lhs, exact.gap(logp, ref, r, beta), atol=1e-5)


def test_optimum_beats_perturbations():
    _, ref, r = tables()
    beta = 0.3
    opt = exact.optimum(ref, r, beta)
    best = exact.objective(opt, ref, r, beta)
    g = torch.Generator().manual_seed(1)
    for _ in range(20):
        other = (opt + 0.1 * torch.randn(opt.shape, generator=g)).log_softmax(1)
        assert exact.objective(other, ref, r, beta) <= best + 1e-6


def test_tiny_beta_stays_finite():  # Review Focus 1
    _, ref, r = tables()
    opt = exact.optimum(ref, r, 1e-3)
    assert torch.isfinite(opt).all()
    assert (opt.exp().sum(1) - 1).abs().max() < 1e-5
    assert all(
        map(
            torch.isfinite,
            (exact.gap(ref, ref, r, 1e-3), exact.objective(opt, ref, r, 1e-3)),
        )
    )


def test_frontier_is_monotone():
    _, ref, r = tables()
    pts = exact.frontier(ref, r, [10.0, 1.0, 0.1, 0.01])
    assert all(a[0] < b[0] and a[1] < b[1] for a, b in itertools.pairwise(pts))


def test_evaluate():
    logp, ref, r = tables()
    beta = 0.3
    m = exact.evaluate(ref, ref, r, beta)
    assert set(m) == {"reward", "kl", "kl_rev", "objective", "gap"}
    assert abs(m["kl"]) < 1e-6 and abs(m["kl_rev"]) < 1e-6 and m["gap"] > 0
    assert abs(exact.evaluate(exact.optimum(ref, r, beta), ref, r, beta)["gap"]) < 1e-5
    assert torch.allclose(
        exact.prompt_values(logp, ref, r, beta).mean(),
        exact.objective(logp, ref, r, beta),
    )
    # an asymmetric case: the two KL directions must not be swapped
    m = exact.evaluate(logp, ref, r, beta)
    assert abs(m["kl"] - exact.kl(logp, ref).item()) < 1e-6
    assert abs(m["kl_rev"] - exact.kl(ref, logp).item()) < 1e-6
    assert abs(m["kl"] - m["kl_rev"]) > 1e-3


def test_drift_ceiling_is_the_small_beta_limit_and_bounds_the_frontier():
    for seed in (0, 1):
        _, ref, r = tables(seed)
        for rewards in (r, (4 * r).round() / 4):  # unique maxima, then tied maxima
            ceiling = exact.drift_ceiling(ref, rewards)
            limit = exact.kl(exact.optimum(ref, rewards, 1e-4), ref)
            assert (ceiling - limit).abs() < 1e-3
            pts = exact.frontier(ref, rewards, [10.0, 1.0, 0.1, 0.01, 1e-3])
            assert all(drift <= ceiling + 1e-6 for drift, _ in pts)
