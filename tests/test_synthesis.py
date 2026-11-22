import json
import re
from pathlib import Path

import pytest
import torch

from proxopt import synthesis
from proxopt.reference import load_reference
from proxopt.train import Run
from proxopt.world import DEFAULT_WORLD


def test_configs_share_one_target():
    c = synthesis.configs(beta=0.1, tau=0.25, seed=3, lr={n: 3e-3 for n in synthesis.NAMES})
    assert set(c) == set(synthesis.NAMES)
    assert c["ppo"].beta == c["grpo"].beta == c["grpo_ppo_like"].beta == 0.1
    assert c["dpo"].beta == c["ppo_rm"].beta == pytest.approx(0.4)
    assert (c["grpo"].std_normalize, c["grpo"].kl_in) == (True, "loss")
    assert (c["grpo_ppo_like"].std_normalize, c["grpo_ppo_like"].kl_in) == (False, "reward")
    assert all(cfg.seed == 3 for cfg in c.values())
    assert all(cfg.lr == 3e-3 for cfg in c.values())


def test_lr_grid():
    assert synthesis.LR_GRID == (1e-4, 3e-4, 1e-3, 3e-3)


RULE = {"ppo": 1e-3, "grpo": 1e-4, "grpo_ppo_like": 3e-4, "dpo": 3e-3, "ppo_rm": 3e-4}


def _grid(rule):
    """Grid rows whose lowest mean final gap falls at `rule[name]` for each name."""
    return [{"name": n, "lr": r, "seeds": [0, 1, 2], "final_gaps": [0.1] * 3,
             "mean_final_gap": 0.01 if r == rule[n] else 0.1 + r}
            for n in synthesis.NAMES for r in synthesis.LR_GRID]


def test_choose_takes_lowest_mean_final_gap():
    assert synthesis.choose(_grid(RULE)) == RULE


def test_choose_breaks_ties_with_the_lowest_rate():
    rows = [{"name": "ppo", "lr": r, "mean_final_gap": 0.05} for r in reversed(synthesis.LR_GRID)]
    assert synthesis.choose(rows) == {"ppo": min(synthesis.LR_GRID)}


def _fake_run(gaps, steps):
    return Run(policy=None, config="fake", target_beta=0.1, steps=steps,
               feedback=list(steps), metrics={"gap": gaps})


def test_comparison_rates_run_grpo_at_grpo_ppo_like_rate():
    rates = synthesis.comparison_rates(RULE)
    assert rates["grpo"] == RULE["grpo_ppo_like"] == 3e-4
    assert {n: r for n, r in rates.items() if n != "grpo"} == {
        n: r for n, r in RULE.items() if n != "grpo"
    }
    assert RULE["grpo"] == 1e-4  # the argument is not modified


def test_method_flags_grid_edges_and_shared_seeds():
    grid = _grid(RULE)
    lr = synthesis.comparison_rates(RULE)
    runs = {n: [_fake_run([0.1, 0.05, 0.08], [0, 10, 20])] * 3 for n in synthesis.NAMES}
    runs["grpo"] = [_fake_run([0.1, 0.04, 0.2], [0, 10, 20]),
                    _fake_run([0.1, 0.03, 0.2], [0, 20, 30]),
                    _fake_run([0.1, 0.05, 0.3], [0, 10, 20])]
    m = synthesis.method(grid, {"beta": 0.1, "tau": 0.25, "lr": lr, "seeds": [0, 1, 2],
                                "runs": runs})
    assert "lowest rate wins" in m["tie_break"]
    edge = m["grid_edge"]  # describes what the rule alone picked
    assert edge["ppo"] == {"at_grid_edge": False, "edge": None}
    assert edge["grpo"] == {"at_grid_edge": True, "edge": "lowest"}
    assert edge["dpo"] == {"at_grid_edge": True, "edge": "highest"}
    assert edge["grpo_ppo_like"]["at_grid_edge"] is False
    assert m["rule_lr"] == RULE and m["comparison_lr"] == lr
    assert m["comparison_seeds"] == [0, 1, 2] and m["tuning_seeds"] == [0, 1, 2]
    assert m["same_seeds_for_tuning_and_comparison"] is True
    exc = m["grpo_rate_exception"]
    assert exc["rule_lr"] == 1e-4 and exc["used_lr"] == 3e-4
    assert "grpo_ppo_like" in exc["reason"]
    assert [row["lr"] for row in exc["mean_final_gap_by_lr"]] == list(synthesis.LR_GRID)
    assert exc["mean_final_gap_by_lr"][0]["mean_final_gap"] == 0.01
    both = exc["mean_final_gap_by_lr"]
    assert [row["grpo_ppo_like"] for row in both] == [
        r["mean_final_gap"] for r in grid if r["name"] == "grpo_ppo_like"
    ]
    assert [row["grpo"] for row in both] == [row["mean_final_gap"] for row in both]
    assert exc["start_gap"] == 0.1
    assert exc["min_gap_during_training"] == {"min": 0.03, "max": 0.05}
    assert exc["min_gap_step"] == {"min": 10, "max": 20}
    assert exc["final_gap"] == {"min": 0.2, "max": 0.3}
    assert "moves least far from pi_ref" in exc["reason"]
    assert "depends on the shared rate" in exc["rate_dependence"]
    assert "above or below" in exc["rate_dependence"]


class _Stop(Exception):
    pass


def test_dpo_and_ppo_rm_receive_identical_pairs(monkeypatch):
    ref = load_reference(DEFAULT_WORLD)
    seen = {}

    def capture(name):
        def f(*args, **kwargs):
            seen[name] = args[2] if name == "dpo" else args[1]
            raise _Stop
        return f

    monkeypatch.setattr(synthesis.dpo, "train", capture("dpo"))
    monkeypatch.setattr(synthesis.prefs, "fit_reward_model", capture("ppo_rm"))
    for name in ("dpo", "ppo_rm"):
        with pytest.raises(_Stop):
            synthesis.run_one(name, ref, DEFAULT_WORLD, 0.1, 0.25, 2, 3e-4, n_pairs=300)
    expected = synthesis.pairs(DEFAULT_WORLD, ref, 300, 0.25, seed=2)
    for field in ("prompts", "winners", "losers"):
        assert torch.equal(getattr(seen["dpo"], field), getattr(seen["ppo_rm"], field))
        assert torch.equal(getattr(seen["dpo"], field), getattr(expected, field))


def test_load_returns_none_when_missing(tmp_path):
    assert synthesis.load(tmp_path / "nope.json") is None


def test_load_honours_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("PROXOPT_SYNTHESIS_FILE", str(tmp_path / "missing.json"))
    assert synthesis.load() is None


@pytest.mark.slow
def test_every_run_targets_beta_and_round_trips(tmp_path):
    ref = load_reference(DEFAULT_WORLD)
    runs = {
        n: [synthesis.run_one(n, ref, DEFAULT_WORLD, 0.1, 0.25, 0, 3e-3, n_pairs=500)]
        for n in synthesis.NAMES
    }
    assert all(r[0].target_beta == pytest.approx(0.1) for r in runs.values())
    # ppo_rm measures against pi*_beta from step 0, not only on the returned run
    assert runs["ppo_rm"][0].metrics["gap"][0] == pytest.approx(runs["ppo"][0].metrics["gap"][0])
    acc = runs["ppo_rm"][0].extras["reward_model"]
    assert set(acc) == {"corr_all", "corr_covered", "covered_mass_ref"}
    assert -1 <= acc["corr_all"] <= 1 and -1 <= acc["corr_covered"] <= 1
    assert 0 < acc["covered_mass_ref"] <= 1
    result = {
        "beta": 0.1,
        "tau": 0.25,
        "seeds": [0],
        "lr": {n: 3e-3 for n in synthesis.NAMES},
        "runs": runs,
        "points": {"dpo": [(500, 0.1)], "ppo_rm": [(500, 0.2)]},
        "reward_model": [{"seed": 0, "n_pairs": 500, **acc}],
    }
    grid = [{"name": "ppo", "lr": 3e-3, "seeds": [0], "final_gaps": [0.01], "mean_final_gap": 0.01}]
    synthesis.save(tmp_path / "s.json", grid, result)
    back = synthesis.load(tmp_path / "s.json")
    assert set(back["runs"]) == set(synthesis.NAMES)
    assert back["runs"]["ppo"][0].metrics == runs["ppo"][0].metrics
    assert back["grid"][0]["name"] == "ppo" and [tuple(p) for p in back["points"]["dpo"]] == [(500, 0.1)]
    assert back["reward_model"][0]["n_pairs"] == 500
    assert set(back["method"]["budget"]) == set(synthesis.NAMES)
    assert set(back["method"]["feedback_units"]) == set(synthesis.NAMES)
    # the same inputs give the same bytes
    synthesis.save(tmp_path / "t.json", grid, result)
    assert (tmp_path / "s.json").read_bytes() == (tmp_path / "t.json").read_bytes()


def test_committed_file_is_complete():
    data = synthesis.load()
    assert data is not None and set(data["runs"]) == set(synthesis.NAMES)
    assert all(len(v) == 3 for v in data["runs"].values())
    assert all(r.target_beta == pytest.approx(data["beta"]) for v in data["runs"].values() for r in v)
    for name in ("dpo", "ppo_rm"):
        counts = [p[0] for p in data["points"][name]]
        assert sorted(set(counts)) == list(synthesis.PAIR_COUNTS)
        assert all(counts.count(n) == 3 for n in synthesis.PAIR_COUNTS)
        assert counts == sorted(counts)
    assert len(data["grid"]) == len(synthesis.NAMES) * len(synthesis.LR_GRID)
    rows = data["reward_model"]
    assert sorted((r["n_pairs"], r["seed"]) for r in rows) == [
        (n, s) for n in synthesis.PAIR_COUNTS for s in (0, 1, 2)
    ]
    assert set(data["method"]) >= {"selection", "budget", "feedback_units", "grid_edge",
                                   "same_seeds_for_tuning_and_comparison", "grpo_rate_exception"}
    # grpo runs at grpo_ppo_like's rate, and its grid rows are all still there
    assert data["lr"]["grpo"] == data["lr"]["grpo_ppo_like"]
    lr_of = lambda r: float(re.search(r"\blr=([0-9.e-]+)", r.config).group(1))
    assert {lr_of(r) for r in data["runs"]["grpo"]} == {lr_of(r) for r in data["runs"]["grpo_ppo_like"]}
    assert {lr_of(r) for r in data["runs"]["grpo"]} == {data["lr"]["grpo_ppo_like"]}
    for name in synthesis.NAMES:
        assert sorted(r["lr"] for r in data["grid"] if r["name"] == name) == list(synthesis.LR_GRID)
        assert {lr_of(r) for r in data["runs"][name]} == {data["lr"][name]}
    assert synthesis.choose(data["grid"]) == data["method"]["rule_lr"]
    assert data["lr"] == synthesis.comparison_rates(data["method"]["rule_lr"])


def test_committed_file_has_no_machine_details():
    path = synthesis.DEFAULT_PATH
    if not path.exists():
        pytest.skip("data/synthesis.json not written yet")
    text = path.read_text()
    data = json.loads(text)
    assert not ({"time", "wall_time", "timestamp", "date"} & set(data))
    home = Path.home()
    fragments = ["/Users/", "/home/", "C:\\", str(home), home.name, str(Path.cwd())]

    def strings(x):
        if isinstance(x, dict):
            for k, v in x.items():
                yield k
                yield from strings(v)
        elif isinstance(x, list):
            for v in x:
                yield from strings(v)
        elif isinstance(x, str):
            yield x

    for value in strings(data):
        for frag in fragments:
            assert frag not in value, (frag, value[:80])
