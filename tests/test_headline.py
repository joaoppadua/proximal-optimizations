"""Headline results at seed 0, pinned against silent drift of the defaults."""

import pytest

from proxopt.algos import dpo, grpo, ppo
from proxopt.algos.dpo import DPOConfig
from proxopt.algos.grpo import GRPOConfig
from proxopt.algos.ppo import PPOConfig
from proxopt.prefs import sample_pairs
from proxopt.reference import load_reference
from proxopt.world import DEFAULT_WORLD


@pytest.mark.slow
def test_headline_final_gaps_at_seed_0():
    """Final gaps of the default runs the notebooks quote, to 4 decimals (measured
    2026-11-22). A change here means the numbers quoted in the notebooks (and in
    scripts/measure_nb0*.py's output) must be re-measured, not that this test needs new
    values."""
    ref = load_reference(DEFAULT_WORLD)
    pairs = sample_pairs(DEFAULT_WORLD, ref, 4000, 0.25, "ref", seed=0)
    final = {
        "ppo": ppo.train(ref, DEFAULT_WORLD, PPOConfig()),
        "grpo": grpo.train(ref, DEFAULT_WORLD, GRPOConfig()),
        "grpo_ppo_like": grpo.train(
            ref, DEFAULT_WORLD, GRPOConfig(std_normalize=False, kl_in="reward")
        ),
        "dpo": dpo.train(ref, DEFAULT_WORLD, pairs, DPOConfig(), tau=0.25),
    }
    expected = {"ppo": 0.0145, "grpo": 0.4370, "grpo_ppo_like": 0.0189, "dpo": 0.0321}
    got = {name: round(run.metrics["gap"][-1], 4) for name, run in final.items()}
    assert got == expected
