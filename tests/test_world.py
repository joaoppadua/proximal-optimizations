import pytest
import torch

from proxopt.world import DEFAULT_WORLD, TINY_WORLD, World


def test_default_sizes():
    w = World()
    assert (w.vocab_size, w.length, w.targets) == (8, 5, (7, 14, 21, 28))
    assert (w.n_prompts, w.n_responses, w.max_sum) == (4, 32768, 35)
    a = w.all_responses()
    assert a.shape == (32768, 5) and a[0].tolist() == [0] * 5 and a[-1].tolist() == [7] * 5
    assert a[1].tolist() == [0, 0, 0, 0, 1]

def test_reward_modes():
    w = World()
    x = torch.tensor([1, 1])                      # target 14
    y = torch.tensor([[2, 3, 3, 3, 3], [0, 0, 0, 0, 0]])
    assert w.reward(x, y, "graded").tolist() == pytest.approx([1.0, 1 - 14 / 35])
    assert w.reward(x, y, "verifiable").tolist() == [1.0, 0.0]

def test_reward_table_matches_reward():
    w = TINY_WORLD
    for mode in ("graded", "verifiable"):
        table = w.reward_table(mode)
        assert table.shape == (2, 9) and table.min() >= 0 and table.max() <= 1
        for p in range(w.n_prompts):
            x = torch.full((w.n_responses,), p)
            assert torch.equal(table[p], w.reward(x, w.all_responses(), mode))

def test_tiny_world():
    assert TINY_WORLD == World(vocab_size=3, length=2, targets=(1, 3))
    assert DEFAULT_WORLD == World()

def test_rejects_unreachable_target():
    with pytest.raises(ValueError):
        World(targets=(99,))
