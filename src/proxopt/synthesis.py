"""The comparison notebook 04 draws: every algorithm on one world, aimed at one target
π*_β, over several seeds. `scripts/synthesis.py` computes it once and writes
`data/synthesis.json`; `load` reads it back (and returns `None` when it is missing).

Fixed for the committed file: `DEFAULT_WORLD`, the graded reward, pairs drawn from π_ref.
DPO and PPO-on-a-reward-model see pairs labelled with noise τ, so both learn r/τ and run
with β/τ in place of β; their target is still π*_β."""

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from proxopt import prefs
from proxopt.algos import dpo, grpo, ppo
from proxopt.policy import Policy
from proxopt.reference import load_reference
from proxopt.train import Run, scorer
from proxopt.world import DEFAULT_WORLD, World

NAMES = ("ppo", "grpo", "grpo_ppo_like", "dpo", "ppo_rm")
PAIR_COUNTS = (250, 1000, 4000, 16000)
LR_GRID = (1e-4, 3e-4, 1e-3, 3e-3)
SEEDS = (0, 1, 2)
N_PAIRS = 4000  # the pair count of the comparison runs and of tuning
REWARD_MODE = "graded"
PAIR_SOURCE = "ref"
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "synthesis.json"
FIXED_DATASET = ("dpo", "ppo_rm")
RM_STEPS, RM_BATCH = 500, 256  # reward-model fit
DEFAULT_LR = {
    "ppo": ppo.PPOConfig.lr,
    "grpo": grpo.GRPOConfig.lr,
    "grpo_ppo_like": grpo.GRPOConfig.lr,
    "dpo": dpo.DPOConfig.lr,
    "ppo_rm": ppo.PPOConfig.lr,
}


def configs(beta: float, tau: float, seed: int, lr: dict[str, float]) -> dict[str, Any]:
    """One config per name, each at its module's default budget."""
    return {
        "ppo": ppo.PPOConfig(beta=beta, lr=lr["ppo"], seed=seed),
        "grpo": grpo.GRPOConfig(beta=beta, lr=lr["grpo"], seed=seed),
        "grpo_ppo_like": grpo.GRPOConfig(
            beta=beta, lr=lr["grpo_ppo_like"], seed=seed, std_normalize=False, kl_in="reward"
        ),
        "dpo": dpo.DPOConfig(beta=beta / tau, lr=lr["dpo"], seed=seed),
        "ppo_rm": ppo.PPOConfig(beta=beta / tau, lr=lr["ppo_rm"], seed=seed),
    }


def _centred(a: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """`a` minus its per-prompt mean over the entries where `mask` holds; 0 elsewhere."""
    m = mask.float()
    mean = (a * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
    return (a - mean) * m


def _corr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    """Pearson correlation, pooled over prompts, of `a` and `b` each centred per prompt
    over the entries in `mask` (so a per-prompt constant in either does not count)."""
    ca, cb = _centred(a, mask), _centred(b, mask)
    return float((ca * cb).sum() / ((ca**2).sum() * (cb**2).sum()).sqrt())


def reward_model_accuracy(
    model: prefs.RewardModel, world: World, ref: Policy, data: prefs.PrefData, tau: float
) -> dict[str, float]:
    """How well a fitted reward model recovers r/τ: correlations over all responses and over
    the responses that appear in `data` for their prompt, and the π_ref mass of the latter."""
    fitted = prefs.reward_model_table(model, world)
    true = world.reward_table(REWARD_MODE) / tau
    covered = prefs.coverage_mask(world, data)
    with torch.no_grad():
        logp_ref = scorer(ref).all_logp()
    return {
        "corr_all": _corr(fitted, true, torch.ones_like(covered)),
        "corr_covered": _corr(fitted, true, covered),
        "covered_mass_ref": prefs.covered_mass(world, data, logp_ref),
    }


def run_one(
    name: str,
    ref: Policy,
    world: World,
    beta: float,
    tau: float,
    seed: int,
    lr: float,
    n_pairs: int = N_PAIRS,
) -> Run:
    """One run of `name`, measured against π*_β. `dpo` and `ppo_rm` draw `n_pairs` pairs
    from π_ref with `seed`; `ppo_rm` fits a reward model to them (same seed), trains PPO on
    it, and puts the model's accuracy in `extras["reward_model"]`."""
    cfg = configs(beta, tau, seed, {n: lr for n in NAMES})[name]
    if name in ("ppo", "grpo", "grpo_ppo_like"):
        return (ppo.train if name == "ppo" else grpo.train)(ref, world, cfg)
    data = prefs.sample_pairs(world, ref, n_pairs, tau, PAIR_SOURCE, REWARD_MODE, seed=seed)
    if name == "dpo":
        return dpo.train(ref, world, data, cfg, tau, REWARD_MODE)
    if name == "ppo_rm":
        model = prefs.fit_reward_model(world, data, RM_STEPS, batch_size=RM_BATCH, seed=seed)
        run = ppo.train(ref, world, cfg, reward_fn=model, target_beta=beta)
        run.extras["reward_model"] = reward_model_accuracy(model, world, ref, data, tau)
        return run
    raise ValueError(f"unknown name: {name!r}")


def _final_gap(run: Run) -> float:
    return run.metrics["gap"][-1]


def choose(grid: list[dict]) -> dict[str, float]:
    """The lowest-final-gap rule: per name in `grid`, the rate with the lowest mean final gap."""
    names = [n for n in NAMES if any(r["name"] == n for r in grid)]
    return {
        n: min((r for r in grid if r["name"] == n), key=lambda r: r["mean_final_gap"])["lr"]
        for n in names
    }


def comparison_rates(rule: dict[str, float]) -> dict[str, float]:
    """The rates the comparison runs at: the rule's choice for every name, except that
    published GRPO runs at the rate chosen for `grpo_ppo_like`. Published GRPO does not aim
    at π*_β, so the rule would pick whichever rate gets least far within the budget; at one
    shared rate the two GRPOs differ only in their two switches."""
    rates = dict(rule)
    rates["grpo"] = rule["grpo_ppo_like"]
    return rates


def tune(
    beta: float = 0.1, tau: float = 0.25, seeds: Sequence[int] = SEEDS
) -> tuple[list[dict], dict[str, float]]:
    """For each name and each rate in `LR_GRID`, the final gap of each seed and their mean,
    at the default budget; and the rule's choice per name (`choose`)."""
    ref = load_reference(DEFAULT_WORLD)
    rows = []
    for name in NAMES:
        for lr in LR_GRID:
            gaps = [_final_gap(run_one(name, ref, DEFAULT_WORLD, beta, tau, s, lr)) for s in seeds]
            rows.append({"name": name, "lr": lr, "seeds": list(seeds), "final_gaps": gaps,
                         "mean_final_gap": sum(gaps) / len(gaps)})
    return rows, choose(rows)


def compare(
    beta: float = 0.1,
    tau: float = 0.25,
    seeds: Sequence[int] = SEEDS,
    lr: dict[str, float] | None = None,
) -> dict:
    """Every name over `seeds` (pairs: `N_PAIRS`), then, for `dpo` and `ppo_rm`, the final
    gap at every pair count in `PAIR_COUNTS` for every seed, ordered by pair count then seed,
    and the reward model's accuracy for every `ppo_rm` run."""
    lr = dict(DEFAULT_LR if lr is None else lr)
    ref = load_reference(DEFAULT_WORLD)
    done: dict[tuple[str, int, int], Run] = {}

    def get(name: str, seed: int, n_pairs: int = N_PAIRS) -> Run:
        key = (name, seed, n_pairs if name in FIXED_DATASET else 0)
        if key not in done:
            done[key] = run_one(name, ref, DEFAULT_WORLD, beta, tau, seed, lr[name], n_pairs)
        return done[key]

    runs = {name: [get(name, s) for s in seeds] for name in NAMES}
    points = {
        name: [(n, _final_gap(get(name, s, n))) for n in PAIR_COUNTS for s in seeds]
        for name in FIXED_DATASET
    }
    reward_model = [
        {"seed": s, "n_pairs": n, **get("ppo_rm", s, n).extras["reward_model"]}
        for n in PAIR_COUNTS for s in seeds
    ]
    return {"beta": beta, "tau": tau, "seeds": list(seeds), "lr": lr, "runs": runs,
            "points": points, "reward_model": reward_model}


def method(grid: list[dict], result: dict) -> dict:
    """What the file is: how rates were chosen, each name's budget, its feedback units."""
    c = configs(result["beta"], result["tau"], 0, result["lr"])
    p, g, d = c["ppo"], c["grpo"], c["dpo"]
    ppo_budget = {"iterations": p.iterations, "epochs": p.epochs, "batch_size": p.batch_size,
                  "gradient_steps": p.iterations * p.epochs}
    grpo_budget = {"iterations": g.iterations, "epochs": g.epochs, "group_size": g.group_size,
                   "groups_per_iteration": "one per prompt", "gradient_steps": g.iterations * g.epochs}
    tuning_seeds = sorted({s for row in grid for s in row.get("seeds", [])})
    comparison_seeds = list(result["seeds"])
    rule = choose(grid)

    def edge(rate: float) -> dict:
        at = "lowest" if rate == min(LR_GRID) else "highest" if rate == max(LR_GRID) else None
        return {"at_grid_edge": at is not None, "edge": at}

    return {
        "selection": (
            "per algorithm, the learning rate in lr_grid with the lowest mean final gap "
            f"over seeds {tuning_seeds} at that algorithm's default budget, against the "
            "common target pi*_beta (rule_lr); the comparison runs at comparison_lr, which "
            "is rule_lr except for grpo (see grpo_rate_exception)"
        ),
        "lr_grid": list(LR_GRID),
        "tuning_seeds": tuning_seeds,
        "comparison_seeds": comparison_seeds,
        "same_seeds_for_tuning_and_comparison": tuning_seeds == sorted(comparison_seeds),
        "seed_note": (
            "when the seeds are the same, the comparison's reported means are those the "
            "selection minimised, so they are slightly optimistic"
        ),
        "rule_lr": rule,
        "comparison_lr": dict(result["lr"]),
        "grid_edge": {name: edge(rate) for name, rate in rule.items()},
        "grid_edge_note": "grid_edge describes the rate the lowest-final-gap rule picked",
        "target": f"pi*_beta, beta = {result['beta']}",
        "world": "DEFAULT_WORLD",
        "reward_mode": REWARD_MODE,
        "pair_source": PAIR_SOURCE,
        "budget": {
            "ppo": ppo_budget,
            "grpo": grpo_budget,
            "grpo_ppo_like": grpo_budget,
            "dpo": {"steps": d.steps, "batch_size": d.batch_size, "n_pairs": N_PAIRS,
                    "beta_used": d.beta},
            "ppo_rm": {**ppo_budget, "n_pairs": N_PAIRS, "beta_used": c["ppo_rm"].beta,
                       "reward_model_steps": RM_STEPS, "reward_model_batch_size": RM_BATCH},
        },
        "feedback_units": {
            "ppo": "reward calls (true reward)",
            "grpo": "reward calls (true reward)",
            "grpo_ppo_like": "reward calls (true reward)",
            "dpo": "labelled preference pairs (constant over a run)",
            "ppo_rm": "reward-model calls; the model was fitted on n_pairs labelled pairs",
        },
        **_grpo_exception(grid, rule, result["lr"]),
        "points": "final gap of dpo and ppo_rm at each n_pairs in PAIR_COUNTS, one entry per seed",
    }


def _grpo_exception(grid: list[dict], rule: dict, used: dict) -> dict:
    if "grpo" not in rule or "grpo" not in used:
        return {}
    return {"grpo_rate_exception": {
        "rule_lr": rule["grpo"],
        "used_lr": used["grpo"],
        "reason": (
            f"published grpo runs at {used['grpo']:g}, the rate chosen for grpo_ppo_like, "
            f"not at {rule['grpo']:g}, the rate the lowest-final-gap rule picks for it. "
            "Published GRPO does not aim at pi*_beta, so the rule picks whichever rate gets "
            "least far within the budget; at one shared rate the two GRPO variants differ "
            "only in their two switches (std_normalize, kl_in)"
        ),
        "mean_final_gap_by_lr": [
            {"lr": r["lr"], "mean_final_gap": r["mean_final_gap"]}
            for r in grid if r["name"] == "grpo"
        ],
    }}


def save(path: Path, grid: list[dict], result: dict) -> None:
    out = {
        "beta": result["beta"],
        "tau": result["tau"],
        "seeds": result["seeds"],
        "lr": result["lr"],
        "method": method(grid, result),
        "grid": grid,
        "runs": {name: [r.to_json_dict() for r in runs] for name, runs in result["runs"].items()},
        "points": {name: [list(p) for p in pts] for name, pts in result["points"].items()},
        "reward_model": result["reward_model"],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")


def _path(path: Path | None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get("PROXOPT_SYNTHESIS_FILE")
    return Path(env) if env else DEFAULT_PATH


def load(path: Path | None = None) -> dict | None:
    """The saved comparison with its runs rebuilt as `Run`s, or `None` if the file is
    missing. `path` defaults to `$PROXOPT_SYNTHESIS_FILE`, else `data/synthesis.json`."""
    path = _path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    data["runs"] = {n: [Run.from_json_dict(r) for r in runs] for n, runs in data["runs"].items()}
    return data
