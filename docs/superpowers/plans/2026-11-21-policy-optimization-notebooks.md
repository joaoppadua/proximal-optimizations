# PPO, GRPO and DPO Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Five marimo notebooks that teach PPO, GRPO and DPO on one shared, exactly solvable toy language model, with a reality check on real OLMo checkpoints.

**Architecture:** A small tested package, `proxopt`, holds the toy world, the policy network, exact enumeration-based metrics, the four algorithms (REINFORCE, PPO, GRPO, DPO), preference data and plots. Notebooks import it, show each loss function's source next to its equation, and run live simulations. OLMo measurements are precomputed by a script into small committed files.

**Tech Stack:** Python ≥ 3.12, uv, torch (CPU), matplotlib, marimo, pytest, ruff. Optional extra `olmo`: transformers ≥ 4.48, datasets, accelerate.

**Spec:** `docs/superpowers/specs/2026-11-21-policy-optimization-notebooks-design.md`. Read it before any task. Section numbers below (§) refer to it.

## Global Constraints

- Work on branch `build-notebooks`, created from `main` in Task 1. Every commit message ends with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.
- Machine: Apple M3, 16 GB. The toy world trains on CPU. Default simulations finish in about five seconds each.
- Notebooks are marimo `.py` files under `notebooks/`, opened with `uv run marimo edit notebooks/NN_name.py`, and each must also run as a script (`uv run python notebooks/NN_name.py`) with exit code 0.
- No new model downloads and no text generation with OLMo. Only the SFT, DPO and Instruct checkpoints, at the revisions pinned in `../gricean-devices/configs/models.yaml`, forward passes only. Never write to `../gricean-devices`.
- Notebooks 00 to 03 must not import `proxopt.olmo` or read `data/olmo/`.
- Everything is seeded: the same config gives the same numbers.
- Notation never changes: `x` prompt, `y` response, `y_t` token, `π_θ`, `π_ref`, `r`, `β`, `A` advantage, `τ` preference noise, `G` group size, `ε` clip range.
- Language: English.
- **Math:** every equation written into a notebook is checked against its source paper and logged in `docs/equation-check.md` (Task 10). If the paper differs from this plan, the paper wins; say so in the task report.
- **Citations:** notebooks cite only through `proxopt.refs.cite(...)`. No citation is typed by hand in a notebook.
- **Honesty of simulations:** the "things to try" reveal texts describe what the simulation actually shows on this machine. If a run contradicts the outcome this plan expects, stop and report; do not adjust the text or the seed to fit.
- **Thresholds:** numeric thresholds in "reduces the gap" tests are this plan's estimates. Tuning default hyperparameters to meet them is expected. Changing a threshold is allowed only with a line in the task report saying what and why.
- Configs are frozen dataclasses (hashable), so notebooks can cache runs by config.
- If the `dataviz` skill is available, load it before writing `plots.py`.

## Review Focus

1. **Extreme β.** A β slider at its smallest value makes `exp(r/β)` overflow if computed naively. Expected: all exact quantities stay finite. Test in Task 3.
2. **Groups with no spread, and tiny groups.** Under the verifiable reward a whole group often scores the same (standard deviation 0); a group of one has no spread at all. Expected: zero advantage, no NaN; a group of one is rejected with a clear error. Tests in Task 7.
3. **Degenerate preference data.** Fewer pairs than the batch size, or two identical responses drawn as a pair. Expected: training runs; a pair never has winner equal to loser. Tests in Tasks 8 and 9.
4. **Extreme PPO controls.** Clip range 0.01 or 10, one epoch or eight, λ = 0. Expected: finite metrics, no crash. Test in Task 6.
5. **Missing OLMo files.** Notebook 04 opened in a clone where `data/olmo/` is absent or partial. Expected: a notice in place of the OLMo section, not a traceback. Tests in Tasks 16 and 18.

---

## File map

```
pyproject.toml  Makefile  .gitignore  README.md
src/proxopt/
  __init__.py
  world.py        World, DEFAULT_WORLD, TINY_WORLD
  policy.py       Policy
  exact.py        expected_reward, kl, objective, optimum, gap, frontier, evaluate, prompt_values
  reference.py    sample_corpus, train_reference, load_reference
  train.py        Run, Recorder
  prefs.py        PrefData, sample_pairs, coverage_mask, RewardModel, fit_reward_model, reward_model_table
  algos/__init__.py
  algos/reinforce.py  ReinforceConfig, reinforce_loss, train
  algos/ppo.py        PPOConfig, token_rewards, gae, ppo_loss, train
  algos/grpo.py       GRPOConfig, group_advantages, kl_estimator, grpo_loss, train, baseline_comparison
  algos/dpo.py        DPOConfig, dpo_loss, implicit_reward, train
  refs.py         REFS, cite
  plots.py        COLORS and figure functions
  nbtools.py      source_md, reveal
  olmo.py         response_logprob, next_token_kl, summarize_margins, load_margins, load_drift, load_meta
  synthesis.py    configs, run_one, compare, tune, save, load
notebooks/        00_foundations.py 01_ppo.py 02_grpo.py 03_dpo.py 04_synthesis.py
scripts/          olmo_measure.py synthesis.py
data/olmo/        margins.jsonl drift.jsonl meta.json
data/synthesis.json
docs/             equation-check.md olmo-facts.md
tests/            __init__.py, helpers.py, test_<module>.py, test_notebooks.py
```

Shapes used throughout: `P` prompts, `N` responses (`vocab_size ** length`), `T` response length, `V` vocabulary size, `B` batch. A "table" is a float tensor `[P, N]` indexed by prompt and by response in the order of `World.all_responses()`.

---

### Task 1: Project scaffold and the world

**Files:**
- Create: `pyproject.toml`, `Makefile`, `.gitignore`, `README.md` (two lines for now), `src/proxopt/__init__.py`, `src/proxopt/world.py`
- Test: `tests/test_world.py`

**Interfaces:**
- Produces:
  - `World(vocab_size: int = 8, length: int = 5, targets: tuple[int, ...] = (7, 14, 21, 28))`, a frozen dataclass. A prompt is an index into `targets`.
  - `World.n_prompts: int`, `World.n_responses: int`, `World.max_sum: int` (`(vocab_size − 1) * length`)
  - `World.all_responses() -> LongTensor[N, T]` in lexicographic order, cached
  - `World.reward(prompts: LongTensor[B], responses: LongTensor[B, T], mode: Literal["graded", "verifiable"] = "graded") -> FloatTensor[B]`
  - `World.reward_table(mode=...) -> FloatTensor[P, N]`
  - `DEFAULT_WORLD = World()`, `TINY_WORLD = World(vocab_size=3, length=2, targets=(1, 3))`

- [ ] **Step 1: Scaffold.** `git checkout -b build-notebooks`. Write `pyproject.toml`: project `proxopt` 0.1.0, `requires-python = ">=3.12"`, hatchling with `packages = ["src/proxopt"]`; dependencies `torch>=2.4`, `numpy`, `matplotlib>=3.9`, `marimo>=0.25`; extras `dev = ["pytest>=8", "pytest-timeout>=2", "ruff"]` and `olmo = ["transformers>=4.48", "datasets>=3.0", "accelerate>=1.0", "pyyaml>=6"]`; pytest config `testpaths = ["tests"]`, `markers = ["slow: trains default configs, runs notebooks, or needs a model"]`, `addopts = "-m 'not slow'"`. `Makefile` targets: `install` (`uv sync --extra dev`), `test` (`uv run pytest`), `test-all` (`uv run pytest -m ""`), `lint` (`uv run ruff check .`), `olmo` (`uv sync --extra dev --extra olmo && uv run python scripts/olmo_measure.py all`). `.gitignore`: `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `__marimo__/`. Run `make install`; expected: `.venv` created, `uv.lock` written.

- [ ] **Step 2: Write the failing tests**

```python
import pytest, torch
from proxopt.world import World, DEFAULT_WORLD, TINY_WORLD

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
```

- [ ] **Step 3: Run** `uv run pytest tests/test_world.py -v`. Expected: FAIL, `proxopt.world` not found.
- [ ] **Step 4: Implement `world.py`.** Graded reward is `1 − |sum(y) − target| / max_sum`. A target outside `[0, max_sum]` raises `ValueError`.
- [ ] **Step 5: Run** the same command. Expected: 5 passed.
- [ ] **Step 6: Commit** the scaffold, `world.py`, its test, and the two documents already in `docs/superpowers/` (`feat: project scaffold and toy world`).

---

### Task 2: The policy network

**Files:**
- Create: `src/proxopt/policy.py`
- Test: `tests/test_policy.py`

**Interfaces:**
- Consumes: `World`.
- Produces: `Policy(world: World, hidden: int = 64)`, an `nn.Module`:
  - `forward(prompts: LongTensor[B], responses: LongTensor[B, T]) -> tuple[Tensor[B, T, V], Tensor[B, T]]`: teacher-forced. `logits[:, t]` is the distribution of `y_t` given `x` and `y_<t`; `values[:, t]` is the value head's estimate at that same state (before `y_t` is chosen).
  - `token_logp(prompts, responses) -> Tensor[B, T]`: log-probability of each given token.
  - `sample(prompts: LongTensor[B], generator: torch.Generator | None = None) -> LongTensor[B, T]`, no gradient.
  - `all_logp() -> Tensor[P, N]`: `log π(y | x)` for every prompt and response, differentiable.
  - `clone() -> Policy`: an independent deep copy with trainable parameters.
  - `world: World` attribute.

- [ ] **Step 1: Write the failing tests**

```python
import torch
from proxopt.world import World, TINY_WORLD
from proxopt.policy import Policy

ONE_STEP = World(vocab_size=8, length=1, targets=(3,))

def test_all_logp_is_a_distribution():
    for w in (TINY_WORLD, ONE_STEP):
        torch.manual_seed(0)
        lp = Policy(w).all_logp()
        assert lp.shape == (w.n_prompts, w.n_responses)
        assert torch.allclose(lp.logsumexp(1), torch.zeros(w.n_prompts), atol=1e-5)

def test_token_logp_sums_to_all_logp():
    w = TINY_WORLD; torch.manual_seed(0); p = Policy(w)
    x = torch.zeros(w.n_responses, dtype=torch.long)
    assert torch.allclose(p.token_logp(x, w.all_responses()).sum(1), p.all_logp()[0], atol=1e-5)

def test_logits_do_not_see_the_token_they_predict():
    w = World(); torch.manual_seed(0); p = Policy(w)
    x = torch.tensor([0, 0]); y = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0]])
    logits, values = p(x, y)
    assert values.shape == (2, 5)
    assert torch.allclose(logits[0], logits[1]) and torch.allclose(values[0], values[1])

def test_sample_matches_distribution():
    w = TINY_WORLD; torch.manual_seed(0); p = Policy(w)
    g = torch.Generator().manual_seed(0)
    y = p.sample(torch.zeros(20000, dtype=torch.long), generator=g)
    idx = y[:, 0] * 3 + y[:, 1]
    freq = torch.bincount(idx, minlength=9) / 20000
    assert (freq - p.all_logp()[0].exp()).abs().max() < 0.02

def test_clone_is_independent():
    w = TINY_WORLD; torch.manual_seed(0); p = Policy(w); q = p.clone()
    assert torch.equal(p.all_logp(), q.all_logp())
    with torch.no_grad():
        for prm in q.parameters(): prm.add_(1.0)
    assert not torch.equal(p.all_logp(), q.all_logp())
    assert all(prm.requires_grad for prm in q.parameters())
```

- [ ] **Step 2: Run** `uv run pytest tests/test_policy.py -v`. Expected: FAIL, module not found.
- [ ] **Step 3: Implement `policy.py`.** One-layer GRU, hidden 64. Input at step `t` is the embedding of the previous token (a start symbol at `t = 0`) concatenated with a prompt embedding. Two linear heads on the GRU output: policy (`V` logits) and value (scalar). `all_logp` is called at every recorded training step, so on `DEFAULT_WORLD` under `torch.no_grad()` it must take under 50 ms; evaluate each shared prefix once (about 4,700 states per prompt) instead of all 32,768 responses token by token.
- [ ] **Step 4: Run** the same command. Expected: 5 passed. Report the measured time of `all_logp` on `DEFAULT_WORLD`.
- [ ] **Step 5: Commit** (`feat: autoregressive policy with value head`).

---

### Task 3: Exact quantities

**Files:**
- Create: `src/proxopt/exact.py`
- Test: `tests/test_exact.py`

**Interfaces:**
- Consumes: tables only (no `Policy`), so everything here is plain tensor math.
- Produces (all tables `[P, N]`; scalars are means over prompts):
  - `expected_reward(logp, reward) -> Tensor[]`
  - `kl(logp_a, logp_b) -> Tensor[]`: `KL(a ‖ b)`
  - `objective(logp, logp_ref, reward, beta: float) -> Tensor[]`: `J_β`
  - `optimum(logp_ref, reward, beta: float) -> Tensor[P, N]`: `log π*_β`
  - `gap(logp, logp_ref, reward, beta: float) -> Tensor[]`: `β · KL(π ‖ π*_β)`
  - `frontier(logp_ref, reward, betas: Sequence[float]) -> list[tuple[float, float]]`: `(KL(π*_β ‖ π_ref), E_{π*_β}[r])` per β
  - `prompt_values(logp, logp_ref, reward, beta: float) -> Tensor[P]`: per-prompt `J_β` (the exact baseline, used in Task 7)
  - `evaluate(logp, logp_ref, reward, beta: float) -> dict[str, float]` with keys exactly `reward`, `kl` (`KL(π ‖ π_ref)`), `kl_rev` (`KL(π_ref ‖ π)`), `objective`, `gap`

- [ ] **Step 1: Write the failing tests**

```python
import torch
from proxopt import exact

def tables(seed=0, P=3, N=50):
    g = torch.Generator().manual_seed(seed)
    logp = torch.randn(P, N, generator=g).log_softmax(1)
    ref = torch.randn(P, N, generator=g).log_softmax(1)
    return logp, ref, torch.rand(P, N, generator=g)

def test_gap_identity():                                   # spec §5.3
    logp, ref, r = tables(); beta = 0.3
    opt = exact.optimum(ref, r, beta)
    lhs = exact.objective(opt, ref, r, beta) - exact.objective(logp, ref, r, beta)
    assert torch.allclose(lhs, beta * exact.kl(logp, opt), atol=1e-5)
    assert torch.allclose(lhs, exact.gap(logp, ref, r, beta), atol=1e-5)

def test_optimum_beats_perturbations():
    _, ref, r = tables(); beta = 0.3
    opt = exact.optimum(ref, r, beta); best = exact.objective(opt, ref, r, beta)
    g = torch.Generator().manual_seed(1)
    for _ in range(20):
        other = (opt + 0.1 * torch.randn(opt.shape, generator=g)).log_softmax(1)
        assert exact.objective(other, ref, r, beta) <= best + 1e-6

def test_tiny_beta_stays_finite():                         # Review Focus 1
    _, ref, r = tables()
    opt = exact.optimum(ref, r, 1e-3)
    assert torch.isfinite(opt).all()
    assert (opt.exp().sum(1) - 1).abs().max() < 1e-5
    assert all(map(torch.isfinite, (exact.gap(ref, ref, r, 1e-3), exact.objective(opt, ref, r, 1e-3))))

def test_frontier_is_monotone():
    _, ref, r = tables()
    pts = exact.frontier(ref, r, [10.0, 1.0, 0.1, 0.01])
    assert all(a[0] < b[0] and a[1] < b[1] for a, b in zip(pts, pts[1:]))

def test_evaluate():
    logp, ref, r = tables(); beta = 0.3
    m = exact.evaluate(ref, ref, r, beta)
    assert set(m) == {"reward", "kl", "kl_rev", "objective", "gap"}
    assert abs(m["kl"]) < 1e-6 and abs(m["kl_rev"]) < 1e-6 and m["gap"] > 0
    assert abs(exact.evaluate(exact.optimum(ref, r, beta), ref, r, beta)["gap"]) < 1e-5
    assert torch.allclose(exact.prompt_values(logp, ref, r, beta).mean(), exact.objective(logp, ref, r, beta))
```

- [ ] **Step 2: Run** `uv run pytest tests/test_exact.py -v`. Expected: FAIL, module not found.
- [ ] **Step 3: Implement `exact.py`** in log space throughout (`logsumexp`, never `exp(r/β)` directly). Treat `0 · log 0` as 0 in `kl`.
- [ ] **Step 4: Run** the same command. Expected: 5 passed.
- [ ] **Step 5: Commit** (`feat: exact objective, optimum and frontier`).

---

### Task 4: The reference policy

**Files:**
- Create: `src/proxopt/reference.py`
- Test: `tests/test_reference.py`

**Interfaces:**
- Consumes: `World`, `Policy`, `exact`.
- Produces:
  - `sample_corpus(world: World, n: int, seed: int) -> tuple[LongTensor[n], LongTensor[n, T]]`: prompts uniform; first token uniform; each later token repeats the previous one with probability 0.4, is the previous one plus 1 modulo `V` with probability 0.4, and is uniform otherwise. The corpus does not depend on the prompt.
  - `train_reference(world: World = DEFAULT_WORLD, seed: int = 0, n: int = 4096, steps: int = 300, lr: float = 3e-3) -> Policy`: maximum-likelihood training on that corpus; returns the policy with `requires_grad_(False)`.
  - `load_reference(world: World = DEFAULT_WORLD, seed: int = 0) -> Policy`: cached `train_reference`, returning a fresh frozen clone each call.

- [ ] **Step 1: Write the failing tests**

```python
import torch
from proxopt import exact
from proxopt.world import DEFAULT_WORLD as W
from proxopt.reference import sample_corpus, train_reference, load_reference

def test_corpus_style():
    x, y = sample_corpus(W, 4096, seed=0)
    assert x.shape == (4096,) and y.shape == (4096, 5)
    step = (y[:, 1:] - y[:, :-1]) % 8
    assert ((step == 0) | (step == 1)).float().mean() > 0.8        # expected 0.85

def test_reference_is_deterministic_frozen_and_full_support():
    a, b = train_reference(W, seed=0), train_reference(W, seed=0)
    assert torch.equal(a.all_logp(), b.all_logp())
    assert torch.isfinite(a.all_logp()).all()
    assert not any(p.requires_grad for p in a.parameters())

def test_reference_has_the_style_and_is_mediocre():
    ref = load_reference(W); lp = ref.all_logp(); r = W.reward_table("graded")
    y = ref.sample(torch.zeros(4000, dtype=torch.long), torch.Generator().manual_seed(0))
    step = (y[:, 1:] - y[:, :-1]) % 8
    assert ((step == 0) | (step == 1)).float().mean() > 0.6
    opt = exact.optimum(lp, r, 0.1)
    assert exact.expected_reward(lp, r) < exact.expected_reward(opt, r) - 0.1
    assert exact.gap(lp, lp, r, 0.1) > 0.05

def test_load_reference_returns_fresh_copies():
    assert load_reference(W) is not load_reference(W)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_reference.py -v`. Expected: FAIL, module not found.
- [ ] **Step 3: Implement `reference.py`.** If the last two thresholds fail, tune `n`, `steps`, `lr` (not the corpus rule). `train_reference` must take under 5 seconds on this machine; report the measured time.
- [ ] **Step 4: Run** the same command. Expected: 4 passed.
- [ ] **Step 5: Commit** (`feat: synthetic corpus and reference policy`).

---

### Task 5: Run records and REINFORCE

**Files:**
- Create: `src/proxopt/train.py`, `src/proxopt/algos/__init__.py`, `src/proxopt/algos/reinforce.py`, `tests/__init__.py`, `tests/helpers.py`
- Test: `tests/test_reinforce.py`

**Interfaces:**
- Consumes: `World`, `Policy`, `exact.evaluate`.
- Produces:
  - `tests/helpers.py`: `flat_grad(scalar: Tensor, params: Module | Sequence[Tensor]) -> Tensor`, the gradient of a scalar with respect to a module's parameters (or the given tensors), flattened into one vector, unused parameters skipped. Later test files import it with `from tests.helpers import flat_grad`.
  - `Run` dataclass: `policy: Policy | None`, `config: Any`, `target_beta: float`, `steps: list[int]` (gradient steps so far), `feedback: list[int]` (cumulative units of feedback: reward calls, or labelled pairs for DPO), `metrics: dict[str, list[float]]` (the five `evaluate` keys), `extras: dict[str, Any]`. `Run.to_json_dict() -> dict` keeps `target_beta`, `steps`, `feedback`, `metrics` and `repr(config)`; `Run.from_json_dict(d: dict) -> Run` rebuilds a run with `policy=None` and empty `extras`. (Used by Task 17 to commit precomputed runs.)
  - `Recorder(world: World, ref: Policy, reward_mode: str, target_beta: float)` with `record(step: int, feedback: int, policy: Policy) -> None` and `finish(policy: Policy, config: Any, **extras) -> Run`. It always measures against the **true** reward table and `π*_{target_beta}`. It records the starting point too (step 0).
  - `ReinforceConfig` (frozen): `beta: float = 0.1`, `lr: float = 3e-3`, `batch_size: int = 64`, `steps: int = 200`, `baseline: Literal["none", "mean"] = "mean"`, `reward_mode: str = "graded"`, `seed: int = 0`, `eval_every: int = 10`.
  - `reinforce_loss(logp_tokens: Tensor[B, T], ref_logp_tokens: Tensor[B, T], rewards: Tensor[B], beta: float, baseline: str) -> Tensor[]`
  - `reinforce.train(ref: Policy, world: World, cfg: ReinforceConfig) -> Run`
  - Convention for every `train` in `algos/`: start from `ref.clone()`, Adam, all sampling through one `torch.Generator` seeded with `cfg.seed`, prompts drawn uniformly.

The loss implements `∇J_β = E[(r − β·log(π_θ(y|x)/π_ref(y|x))) · ∇log π_θ(y|x)]`: with `R = r − β·Σ_t(logp_t − ref_logp_t)` detached and `b` the batch mean of `R` (or 0), the loss is `−mean((R − b) · Σ_t logp_t)`.

- [ ] **Step 1: Write the failing tests**

```python
import json, pytest, torch
from proxopt import exact
from proxopt.train import Run
from proxopt.world import DEFAULT_WORLD, TINY_WORLD
from proxopt.reference import load_reference, train_reference
from proxopt.algos import reinforce
from proxopt.algos.reinforce import ReinforceConfig, reinforce_loss
from tests.helpers import flat_grad

def test_gradient_estimates_the_gradient_of_J():
    w = TINY_WORLD; ref = train_reference(w, seed=0, n=512, steps=100); pol = ref.clone(); beta = 0.3
    with torch.no_grad():                                  # move away from ref so the KL term matters
        for p in pol.parameters(): p.add_(0.05 * torch.randn(p.shape, generator=torch.Generator().manual_seed(0)))
    r = w.reward_table("graded")
    g_exact = flat_grad(-exact.objective(pol.all_logp(), ref.all_logp(), r, beta), pol)
    g = torch.Generator().manual_seed(0)
    x = torch.randint(w.n_prompts, (200_000,), generator=g); y = pol.sample(x, g)
    loss = reinforce_loss(pol.token_logp(x, y), ref.token_logp(x, y), w.reward(x, y), beta, "none")
    g_est = flat_grad(loss, pol)
    assert torch.cosine_similarity(g_exact, g_est, dim=0) > 0.98

def test_run_record_and_determinism():
    ref = load_reference(DEFAULT_WORLD); cfg = ReinforceConfig(steps=40)
    a, b = reinforce.train(ref, DEFAULT_WORLD, cfg), reinforce.train(ref, DEFAULT_WORLD, cfg)
    assert a.metrics == b.metrics
    assert a.steps[0] == 0 and a.steps[-1] == 40 and a.feedback[-1] == 40 * cfg.batch_size
    assert set(a.metrics) == {"reward", "kl", "kl_rev", "objective", "gap"}
    assert a.target_beta == cfg.beta and abs(a.metrics["kl"][0]) < 1e-6
    back = Run.from_json_dict(json.loads(json.dumps(a.to_json_dict())))
    assert back.policy is None and (back.steps, back.feedback, back.metrics) == (a.steps, a.feedback, a.metrics)

@pytest.mark.slow
def test_reinforce_reduces_gap():
    run = reinforce.train(load_reference(DEFAULT_WORLD), DEFAULT_WORLD, ReinforceConfig())
    assert run.metrics["gap"][-1] < 0.5 * run.metrics["gap"][0]
```

- [ ] **Step 2: Run** `uv run pytest tests/test_reinforce.py -v -m ""`. Expected: FAIL, modules not found.
- [ ] **Step 3: Implement** `train.py` and `algos/reinforce.py`.
- [ ] **Step 4: Run** the same command. Expected: 3 passed. Report the wall time of the default `train`.
- [ ] **Step 5: Commit** (`feat: run records and REINFORCE baseline`).

---

### Task 6: PPO

**Files:**
- Create: `src/proxopt/algos/ppo.py`
- Test: `tests/test_ppo.py`

**Interfaces:**
- Consumes: `Run`, `Recorder`, `Policy`, `World`.
- Produces:
  - `PPOConfig` (frozen): `beta: float = 0.1`, `clip: float = 0.2`, `epochs: int = 4`, `lam: float = 0.95`, `batch_size: int = 64`, `iterations: int = 50`, `lr: float = 3e-3`, `vf_coef: float = 0.5`, `reward_mode: str = "graded"`, `seed: int = 0`.
  - `token_rewards(rewards: Tensor[B], logp_old: Tensor[B, T], ref_logp: Tensor[B, T], beta: float) -> Tensor[B, T]`: `−β·(logp_old_t − ref_logp_t)` on every token, plus the task reward on the last token. (The KL penalty is per token; this is the plan's decision on the point §10 left open.)
  - `gae(rewards: Tensor[B, T], values: Tensor[B, T], lam: float) -> tuple[Tensor[B, T], Tensor[B, T]]`: advantages and returns, with discount 1 and value 0 after the last token. `δ_t = r_t + V_{t+1} − V_t`, `Â_t = Σ_l λ^l δ_{t+l}`, `returns = Â + V`.
  - `ppo_loss(logp_new, logp_old, advantages, values_new, returns, clip: float, vf_coef: float) -> tuple[Tensor[], Tensor[B, T]]` (all inputs `[B, T]`): returns the loss and the ratios `ρ_t = exp(logp_new − logp_old)`. Loss is `−mean(min(ρ·Â, clip(ρ, 1−ε, 1+ε)·Â)) + vf_coef · mean((values_new − returns)²)`, means over all `B·T` tokens.
  - `ppo.train(ref: Policy, world: World, cfg: PPOConfig, reward_fn: Callable[[LongTensor, LongTensor], Tensor] | None = None) -> Run`. Each iteration: sample `batch_size` responses from the current policy, compute rewards with `reward_fn` (default: the world's reward in `cfg.reward_mode`), then take `epochs` full-batch gradient steps on `ppo_loss`, then record. `feedback` grows by `batch_size` per iteration. `extras["ratios"]`: the flattened ratios of the last gradient step of the last iteration.

- [ ] **Step 1: Write the failing tests**

```python
import math, pytest, torch
from proxopt.world import DEFAULT_WORLD, TINY_WORLD
from proxopt.reference import load_reference, train_reference
from proxopt.algos import ppo
from proxopt.algos.ppo import PPOConfig, gae, ppo_loss, token_rewards
from tests.helpers import flat_grad

def test_gae_limits():
    g = torch.Generator().manual_seed(0); r = torch.randn(6, 5, generator=g); v = torch.randn(6, 5, generator=g)
    adv1, ret1 = gae(r, v, 1.0)
    to_go = r.flip(1).cumsum(1).flip(1)
    assert torch.allclose(adv1, to_go - v, atol=1e-6) and torch.allclose(ret1, to_go, atol=1e-6)
    adv0, _ = gae(r, v, 0.0)
    v_next = torch.cat([v[:, 1:], torch.zeros(6, 1)], 1)
    assert torch.allclose(adv0, r + v_next - v, atol=1e-6)

def test_token_rewards():
    g = torch.Generator().manual_seed(0); lo = torch.randn(4, 5, generator=g); lr_ = torch.randn(4, 5, generator=g)
    r = torch.rand(4, generator=g); tr = token_rewards(r, lo, lr_, 0.2)
    assert torch.allclose(tr.sum(1), r - 0.2 * (lo - lr_).sum(1), atol=1e-6)
    assert torch.allclose(tr[:, :-1], -0.2 * (lo - lr_)[:, :-1], atol=1e-6)

def test_first_step_gradient_equals_reinforce_for_any_clip():      # spec §9
    w = TINY_WORLD; pol = train_reference(w, seed=0, n=512, steps=100).clone()
    g = torch.Generator().manual_seed(0)
    x = torch.randint(w.n_prompts, (256,), generator=g); y = pol.sample(x, g); A = torch.randn(256, generator=g)
    old = pol.token_logp(x, y).detach(); z = torch.zeros(256, w.length)
    g_rf = flat_grad(-(A * pol.token_logp(x, y).sum(1)).mean() / w.length, pol)
    for clip in (0.05, 0.2, 10.0):
        loss, ratios = ppo_loss(pol.token_logp(x, y), old, A[:, None].expand(256, w.length), z, z, clip, 0.0)
        assert torch.allclose(ratios, torch.ones_like(ratios), atol=1e-6)
        assert torch.allclose(flat_grad(loss, pol), g_rf, atol=1e-6)

def test_zero_reward_keeps_the_policy_at_the_reference():
    ref = load_reference(DEFAULT_WORLD)
    run = ppo.train(ref, DEFAULT_WORLD, PPOConfig(iterations=10), reward_fn=lambda x, y: torch.zeros(len(x)))
    assert run.metrics["kl"][-1] < 0.02

@pytest.mark.parametrize("kw", [dict(clip=0.01), dict(clip=10.0, epochs=8), dict(epochs=1), dict(beta=0.02), dict(lam=0.0)])
def test_extreme_controls_stay_finite(kw):                          # Review Focus 4
    run = ppo.train(load_reference(DEFAULT_WORLD), DEFAULT_WORLD, PPOConfig(iterations=8, **kw))
    assert all(math.isfinite(v) for vs in run.metrics.values() for v in vs)
    assert torch.isfinite(run.extras["ratios"]).all()

@pytest.mark.slow
def test_ppo_reduces_gap_and_is_deterministic():
    ref = load_reference(DEFAULT_WORLD); cfg = PPOConfig(); a = ppo.train(ref, DEFAULT_WORLD, cfg)
    assert a.metrics["gap"][-1] < 0.25 * a.metrics["gap"][0]
    assert a.metrics == ppo.train(ref, DEFAULT_WORLD, cfg).metrics
    assert a.feedback[-1] == cfg.iterations * cfg.batch_size and a.steps[-1] == cfg.iterations * cfg.epochs
```

- [ ] **Step 2: Run** `uv run pytest tests/test_ppo.py -v -m ""`. Expected: FAIL, module not found.
- [ ] **Step 3: Implement `algos/ppo.py`.** Advantages and returns are computed once per iteration from the sampling policy and held fixed across epochs. No advantage normalisation. Rewards, whether from the world or from `reward_fn`, are constants: compute them under `torch.no_grad()`.
- [ ] **Step 4: Run** the same command. Expected: all pass. Report the wall time of the default `train`.
- [ ] **Step 5: Commit** (`feat: PPO in its RLHF form`).

---

### Task 7: GRPO

**Files:**
- Create: `src/proxopt/algos/grpo.py`
- Test: `tests/test_grpo.py`

**Interfaces:**
- Consumes: `Run`, `Recorder`, `Policy`, `World`, `exact.prompt_values`.
- Produces:
  - `GRPOConfig` (frozen): `beta: float = 0.1`, `clip: float = 0.2`, `group_size: int = 8`, `epochs: int = 4`, `iterations: int = 50`, `lr: float = 3e-3`, `std_normalize: bool = True`, `kl_in: Literal["loss", "reward"] = "loss"`, `reward_mode: str = "graded"`, `seed: int = 0`. The defaults are GRPO as published.
  - `group_advantages(rewards: Tensor[P, G], std_normalize: bool, eps: float = 1e-6) -> Tensor[P, G]`: reward minus group mean, divided by `(group std + eps)` when `std_normalize`. Raises `ValueError` if `G < 2`.
  - `kl_estimator(logp_new: Tensor, ref_logp: Tensor) -> Tensor` (same shape): `π_ref/π_θ − log(π_ref/π_θ) − 1` per token.
  - `grpo_loss(logp_new: Tensor[B, T], logp_old: Tensor[B, T], ref_logp: Tensor[B, T], advantages: Tensor[B], clip: float, beta: float, kl_in_loss: bool) -> tuple[Tensor[], Tensor[B, T]]`: `−mean(min(ρ·Â, clip(ρ)·Â) − [β · kl_estimator if kl_in_loss])` over all tokens, with the response's advantage on each of its tokens; also returns the ratios.
  - `grpo.train(ref: Policy, world: World, cfg: GRPOConfig) -> Run`. Each iteration: for **every** prompt sample `group_size` responses from the current policy; compute rewards; if `kl_in == "reward"`, subtract `β·Σ_t(logp_old_t − ref_logp_t)` from each response's reward before `group_advantages` and pass `kl_in_loss=False`; take `epochs` gradient steps; record. `feedback` grows by `P · group_size`. `extras["ratios"]` as in PPO; `extras["dead_groups"]`: one integer per iteration, the number of groups whose rewards were all equal.
  - `baseline_comparison(policy: Policy, ref: Policy, world: World, group_size: int, beta: float, reward_mode: str, seed: int) -> dict[str, Tensor]`: samples one group per prompt from `policy` and returns, per sampled response (flattened), `"exact"` (regularised return minus `exact.prompt_values`), `"group"` (regularised return minus its group mean) and `"critic"` (regularised return minus the policy's value head at the first position). Used by notebook 02 with a PPO-trained policy.

- [ ] **Step 1: Write the failing tests**

```python
import math, pytest, torch
from proxopt.world import DEFAULT_WORLD
from proxopt.reference import load_reference
from proxopt.algos import grpo
from proxopt.algos.grpo import GRPOConfig, group_advantages, kl_estimator, grpo_loss
from tests.helpers import flat_grad

def test_group_advantages():
    r = torch.rand(4, 8, generator=torch.Generator().manual_seed(0))
    for std in (True, False):
        assert group_advantages(r, std).mean(1).abs().max() < 1e-5
    assert torch.allclose(group_advantages(r, False), r - r.mean(1, keepdim=True))
    assert (group_advantages(r, True).std(1) - 1).abs().max() < 1e-3

def test_identical_group_gives_zero_not_nan():                       # Review Focus 2, spec §9
    for std in (True, False):
        assert torch.equal(group_advantages(torch.ones(3, 8), std), torch.zeros(3, 8))
    with pytest.raises(ValueError):
        group_advantages(torch.rand(4, 1), True)

def test_zero_advantage_gives_zero_policy_gradient():
    w = DEFAULT_WORLD; pol = load_reference(w).clone(); x = torch.zeros(8, dtype=torch.long)
    y = pol.sample(x, torch.Generator().manual_seed(0)); old = pol.token_logp(x, y).detach()
    loss, _ = grpo_loss(pol.token_logp(x, y), old, old, torch.zeros(8), 0.2, 0.0, True)
    assert flat_grad(loss, pol).abs().max() == 0

def test_kl_estimator_value_and_gradient_direction():               # spec §6.2 and §9
    logits = torch.tensor([2.0, 0.0, -1.0, 0.5], requires_grad=True)   # one next-token distribution, far from the reference
    lt, lr_ = logits.log_softmax(0), torch.tensor([-1.0, 1.5, 0.0, 0.3]).log_softmax(0)
    kl_theta_ref = (lt.exp() * (lt - lr_)).sum(); kl_ref_theta = (lr_.exp() * (lr_ - lt)).sum()
    est = (lt.exp().detach() * kl_estimator(lt, lr_)).sum()         # expectation over y ~ π_θ, samples held fixed
    assert torch.allclose(est, kl_theta_ref, atol=1e-6)
    g_est, g_rt, g_tr = (flat_grad(s, [logits]) for s in (est, kl_ref_theta, kl_theta_ref))
    assert torch.allclose(g_est, g_rt, atol=1e-6)
    assert (g_est - g_tr).abs().max() > 0.05

def test_verifiable_reward_with_pairs_stays_finite():               # Review Focus 2
    cfg = GRPOConfig(reward_mode="verifiable", group_size=2, iterations=8)
    run = grpo.train(load_reference(DEFAULT_WORLD), DEFAULT_WORLD, cfg)
    assert all(math.isfinite(v) for vs in run.metrics.values() for v in vs)
    assert len(run.extras["dead_groups"]) == 8 and max(run.extras["dead_groups"]) <= DEFAULT_WORLD.n_prompts

def test_baseline_comparison_keys():
    ref = load_reference(DEFAULT_WORLD)
    out = grpo.baseline_comparison(ref.clone(), ref, DEFAULT_WORLD, 8, 0.1, "graded", 0)
    assert set(out) == {"exact", "group", "critic"} and all(v.shape == (32,) for v in out.values())

@pytest.mark.slow
def test_published_grpo_raises_reward_and_ppo_like_grpo_closes_the_gap():   # spec §9
    ref = load_reference(DEFAULT_WORLD); cfg = GRPOConfig()
    pub = grpo.train(ref, DEFAULT_WORLD, cfg)
    assert pub.metrics["reward"][-1] > pub.metrics["reward"][0] + 0.1
    like = grpo.train(ref, DEFAULT_WORLD, GRPOConfig(std_normalize=False, kl_in="reward"))
    assert like.metrics["gap"][-1] < 0.25 * like.metrics["gap"][0]
    assert pub.feedback[-1] == cfg.iterations * DEFAULT_WORLD.n_prompts * cfg.group_size
    assert pub.metrics == grpo.train(ref, DEFAULT_WORLD, cfg).metrics
```

- [ ] **Step 2: Run** `uv run pytest tests/test_grpo.py -v -m ""`. Expected: FAIL, module not found.
- [ ] **Step 3: Implement `algos/grpo.py`.** Use the same standard-deviation convention in `group_advantages` and in the test (`Tensor.std`, the unbiased one).
- [ ] **Step 4: Run** the same command. Expected: all pass. Report wall time, and report the final `gap` of the four switch combinations (`std_normalize` × `kl_in`); notebook 02 needs them.
- [ ] **Step 5: Commit** (`feat: GRPO as published, with switches back to PPO-like choices`).

---

### Task 8: Preference pairs and the reward model

**Files:**
- Create: `src/proxopt/prefs.py`
- Test: `tests/test_prefs.py`

**Interfaces:**
- Consumes: `World`, `Policy`.
- Produces:
  - `PrefData` (frozen dataclass): `prompts: LongTensor[M]`, `winners: LongTensor[M, T]`, `losers: LongTensor[M, T]`; `len(data) == M`.
  - `sample_pairs(world: World, ref: Policy, n_pairs: int, tau: float, source: Literal["ref", "uniform"], reward_mode: str = "graded", seed: int = 0) -> PrefData`: prompts uniform; two **different** responses from the source (redraw on a tie); the first wins with probability `σ((r_1 − r_2)/τ)`.
  - `coverage_mask(world: World, data: PrefData) -> BoolTensor[P, N]`: true where that response appears (as winner or loser) for that prompt.
  - `RewardModel(world: World, hidden: int = 64)`, an `nn.Module` with `forward(prompts: LongTensor[B], responses: LongTensor[B, T]) -> Tensor[B]`.
  - `fit_reward_model(world: World, data: PrefData, steps: int = 500, lr: float = 3e-3, batch_size: int = 256, seed: int = 0) -> RewardModel`: minimises `−log σ(r̂(winner) − r̂(loser))`. It therefore learns `r/τ` up to a constant per prompt.
  - `reward_model_table(model: RewardModel, world: World) -> Tensor[P, N]`

- [ ] **Step 1: Write the failing tests**

```python
import torch
from proxopt.world import DEFAULT_WORLD, TINY_WORLD as W
from proxopt.reference import load_reference, train_reference
from proxopt.prefs import sample_pairs, coverage_mask, fit_reward_model, reward_model_table

REF = train_reference(W, seed=0, n=512, steps=100)
def idx(y): return y[:, 0] * 3 + y[:, 1]
def centre(t): return t - t.mean(1, keepdim=True)

def test_pairs_are_distinct_and_deterministic():                    # Review Focus 3
    for world, ref, src in ((W, REF, "ref"), (DEFAULT_WORLD, load_reference(DEFAULT_WORLD), "ref"), (W, REF, "uniform")):
        d = sample_pairs(world, ref, 500, 0.25, src, seed=0)
        assert len(d) == 500 and d.winners.shape == (500, world.length)
        assert (d.winners != d.losers).any(1).all()
        assert torch.equal(d.winners, sample_pairs(world, ref, 500, 0.25, src, seed=0).winners)

def test_bradley_terry_frequencies():
    tau = 0.5; d = sample_pairs(W, REF, 40_000, tau, "uniform", seed=0)
    rw, rl = W.reward(d.prompts, d.winners), W.reward(d.prompts, d.losers)
    for gap_ in (0.25, 0.5):                                         # reward differences that occur in TINY_WORLD
        hi, lo = (rw - rl - gap_).abs() < 1e-6, (rl - rw - gap_).abs() < 1e-6
        share = hi.sum() / (hi.sum() + lo.sum())
        assert abs(share - torch.sigmoid(torch.tensor(gap_ / tau))) < 0.03

def test_sharp_preferences_always_pick_the_better_response():
    d = sample_pairs(W, REF, 2000, 1e-3, "uniform", seed=0)
    assert (W.reward(d.prompts, d.winners) >= W.reward(d.prompts, d.losers)).all()

def test_coverage_mask():
    d = sample_pairs(W, REF, 3, 0.25, "uniform", seed=0); m = coverage_mask(W, d)
    assert m.shape == (2, 9) and m.sum() <= 6
    assert m[d.prompts, idx(d.winners)].all() and m[d.prompts, idx(d.losers)].all()

def test_reward_model_learns_r_over_tau_up_to_a_constant():
    tau = 0.5; d = sample_pairs(W, REF, 20_000, tau, "uniform", seed=0)
    est = centre(reward_model_table(fit_reward_model(W, d, seed=0), W).detach()); true = centre(W.reward_table() / tau)
    slope = (est * true).sum() / (true * true).sum()
    assert torch.corrcoef(torch.stack([est.flatten(), true.flatten()]))[0, 1] > 0.95 and 0.75 < slope < 1.25
```

- [ ] **Step 2: Run** `uv run pytest tests/test_prefs.py -v`. Expected: FAIL, module not found.
- [ ] **Step 3: Implement `prefs.py`.** The reward model may reuse the policy's sequence encoder design; it reads the whole response.
- [ ] **Step 4: Run** the same command. Expected: 5 passed.
- [ ] **Step 5: Commit** (`feat: Bradley-Terry preference pairs and reward model`).

---

### Task 9: DPO

**Files:**
- Create: `src/proxopt/algos/dpo.py`
- Test: `tests/test_dpo.py`

**Interfaces:**
- Consumes: `PrefData`, `sample_pairs`, `Run`, `Recorder`, `exact`.
- Produces:
  - `DPOConfig` (frozen): `beta: float = 0.4`, `steps: int = 300`, `batch_size: int = 256`, `lr: float = 3e-3`, `seed: int = 0`, `eval_every: int = 10`.
  - `dpo_loss(logp_w: Tensor[B], logp_l: Tensor[B], ref_logp_w: Tensor[B], ref_logp_l: Tensor[B], beta: float) -> Tensor[]`: `−mean(log σ(β·[(logp_w − ref_logp_w) − (logp_l − ref_logp_l)]))`, with sequence log-probabilities (sums over tokens).
  - `implicit_reward(policy: Policy, ref: Policy, beta: float) -> Tensor[P, N]`: `β·(log π_θ − log π_ref)`, centred per prompt (each row has mean 0).
  - `dpo.train(ref: Policy, world: World, data: PrefData, cfg: DPOConfig, tau: float, reward_mode: str = "graded") -> Run`: minibatches drawn with replacement from `data`; `target_beta = cfg.beta * tau`, so `metrics["gap"]` is measured against `π*_{β·τ}`; `feedback` is `len(data)` at every record.

- [ ] **Step 1: Write the failing tests**

```python
import math, pytest, torch
from proxopt import exact
from proxopt.world import DEFAULT_WORLD, TINY_WORLD as W
from proxopt.reference import load_reference, train_reference
from proxopt.prefs import sample_pairs
from proxopt.algos import dpo
from proxopt.algos.dpo import DPOConfig, dpo_loss, implicit_reward

REF = train_reference(W, seed=0, n=512, steps=100)

def test_loss_values():
    z = torch.zeros(4)
    assert torch.allclose(dpo_loss(z, z, z, z, 0.5), torch.tensor(math.log(2)))
    t = lambda v: torch.tensor([v])
    assert torch.allclose(dpo_loss(t(-1.0), t(-3.0), t(-2.0), t(-2.0), 0.5), -torch.nn.functional.logsigmoid(torch.tensor(1.0)))

def test_dpo_approaches_the_optimum_for_beta_times_tau():           # spec §6.3 and §9
    tau, beta = 0.5, 0.4
    data = sample_pairs(W, REF, 50_000, tau, "uniform", seed=0)
    run = dpo.train(REF, W, data, DPOConfig(beta=beta, steps=1500), tau)
    assert run.target_beta == pytest.approx(0.2)
    assert run.metrics["gap"][-1] < 0.1 * run.metrics["gap"][0]
    lp, ref_lp, r = run.policy.all_logp().detach(), REF.all_logp(), W.reward_table()
    assert exact.kl(lp, exact.optimum(ref_lp, r, beta * tau)) < exact.kl(lp, exact.optimum(ref_lp, r, beta))
    est = implicit_reward(run.policy, REF, beta).detach(); true = W.reward_table() / tau
    true = true - true.mean(1, keepdim=True)
    assert est.mean(1).abs().max() < 1e-5
    assert torch.corrcoef(torch.stack([est.flatten(), true.flatten()]))[0, 1] > 0.95

def test_fewer_pairs_than_a_batch():                                # Review Focus 3
    data = sample_pairs(W, REF, 8, 0.5, "uniform", seed=0)
    run = dpo.train(REF, W, data, DPOConfig(steps=30), 0.5)
    assert all(math.isfinite(v) for vs in run.metrics.values() for v in vs) and run.feedback[-1] == 8

@pytest.mark.slow
def test_dpo_reduces_gap_on_the_default_world_and_is_deterministic():
    ref = load_reference(DEFAULT_WORLD); data = sample_pairs(DEFAULT_WORLD, ref, 4000, 0.25, "ref", seed=0)
    a = dpo.train(ref, DEFAULT_WORLD, data, DPOConfig(), 0.25)
    assert a.metrics["gap"][-1] < 0.5 * a.metrics["gap"][0]
    assert a.metrics == dpo.train(ref, DEFAULT_WORLD, data, DPOConfig(), 0.25).metrics
```

- [ ] **Step 2: Run** `uv run pytest tests/test_dpo.py -v -m ""`. Expected: FAIL, module not found.
- [ ] **Step 3: Implement `algos/dpo.py`.** The reference log-probabilities of the whole dataset can be computed once.
- [ ] **Step 4: Run** the same command. Expected: 4 passed. Report wall time of the default-world run, and the final gap with `source="ref"` against `source="uniform"`; notebook 03 needs both.
- [ ] **Step 5: Commit** (`feat: DPO`).

---

### Task 10: Verified references, equation check, OLMo facts

This task is research, not code. It needs web access (arXiv, the open-instruct repository, the OLMo 2 paper). Nothing about the user or this project goes into any request.

**Files:**
- Create: `src/proxopt/refs.py`, `docs/equation-check.md`, `docs/olmo-facts.md`
- Test: `tests/test_refs.py`

**Interfaces:**
- Produces:
  - `REFS: dict[str, Ref]` with exactly these keys: `williams1992`, `schulman2015trpo`, `schulman2016gae`, `schulman2017ppo`, `christiano2017`, `ziegler2019`, `stiennon2020`, `ouyang2022`, `bradley1952`, `rafailov2023`, `shao2024`, `guo2025`, `lambert2024`, `olmo2024`. `Ref` is a frozen dataclass: `authors: str`, `year: int`, `title: str`, `venue: str`, `url: str`.
  - `cite(*keys: str) -> str`: a markdown bulleted list, one formatted reference per key, in the order given; unknown key raises `KeyError`.
  - `docs/olmo-facts.md`: answers to two questions, each with a verbatim quote and its URL, or the words "not confirmed": (1) did the last post-training stage of OLMo 2 1B (April 2025) use GRPO; (2) did its DPO stage use the length-normalised variant, and with which β. End the file with one line each: `GRPO_CONFIRMED: yes|no` and `DPO_VARIANT: standard|length-normalised|unknown`.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from proxopt.refs import REFS, cite

KEYS = {"williams1992", "schulman2015trpo", "schulman2016gae", "schulman2017ppo", "christiano2017", "ziegler2019",
        "stiennon2020", "ouyang2022", "bradley1952", "rafailov2023", "shao2024", "guo2025", "lambert2024", "olmo2024"}

def test_refs_are_complete():
    assert set(REFS) == KEYS
    for r in REFS.values():
        assert r.authors and r.title and r.venue and r.url.startswith("https://") and 1950 <= r.year <= 2026

def test_cite():
    out = cite("rafailov2023", "shao2024")
    assert out.count("\n- ") + out.startswith("- ") == 2 and REFS["rafailov2023"].title in out
    assert out.index(REFS["rafailov2023"].title) < out.index(REFS["shao2024"].title)
    with pytest.raises(KeyError):
        cite("nope")
```

- [ ] **Step 2: Run** `uv run pytest tests/test_refs.py -v`. Expected: FAIL, module not found.
- [ ] **Step 3: Verify each reference at its source** (arXiv abstract page or publisher page) and write `refs.py`. Copy authors, title, year and venue from the page; do not write any field from memory. List every author up to six, then "et al.".
- [ ] **Step 4: Write `docs/equation-check.md`.** One row per equation this plan uses, with the paper, the equation number in the paper, and "matches" or the exact difference: the REINFORCE gradient (Task 5); the PPO clipped objective and the GAE recursion (Task 6); the RLHF reward with the KL penalty (Ziegler et al., Ouyang et al.); the GRPO objective, its advantage and its KL estimator (Task 7); the DPO optimum, the reward rewritten in terms of the policy, the Bradley–Terry model and the DPO loss and its gradient (Tasks 9 and 15). Where a paper differs from the plan, write the difference at the top of the file under "Differences to resolve".
- [ ] **Step 5: Write `docs/olmo-facts.md`** as specified above.
- [ ] **Step 6: Run** the test. Expected: 2 passed.
- [ ] **Step 7: Commit** (`docs: verified references, equation check, OLMo facts`).

---

### Task 11: Plots and notebook helpers

**Files:**
- Create: `src/proxopt/plots.py`, `src/proxopt/nbtools.py`
- Test: `tests/test_plots.py`

**Interfaces:**
- Consumes: `Run`.
- Produces in `plots.py` (each returns a `matplotlib.figure.Figure`; no function calls `plt.show`):
  - `COLORS: dict[str, str]` with keys `ref`, `optimum`, `reinforce`, `ppo`, `grpo`, `dpo`, `ppo_rm`. Every figure in every notebook uses these.
  - `training_curves(runs: dict[str, Run | list[Run]], keys: Sequence[str] = ("reward", "kl", "gap"), x: Literal["steps", "feedback"] = "steps") -> Figure`: one panel per key. A list of runs (different seeds) is drawn as its mean line with a min-to-max band. Dict keys that are in `COLORS` take that colour.
  - `frontier_plot(frontier: list[tuple[float, float]], runs: dict[str, Run | list[Run]], targets: dict[str, tuple[float, float]] | None = None) -> Figure`: drift on x, reward on y; the frontier as a curve, each run as a trajectory with its end marked, each target as a labelled point.
  - `ratio_hist(ratios: Tensor, clip: float) -> Figure`: histogram with vertical lines at `1 − clip` and `1 + clip`, and the share of ratios outside them in the title.
  - `distribution_bars(dists: dict[str, Tensor], rewards: Tensor, labels: Sequence[str] | None = None) -> Figure`: grouped bars of probabilities for a small set of responses, rewards on a second panel. Used for one-step worlds.
  - `scatter_true_vs_estimate(estimate: Tensor, true: Tensor, mask: Tensor | None = None, mask_labels: tuple[str, str] = ("covered", "not covered"), xlabel: str = "", ylabel: str = "") -> Figure`: scatter with the identity line; two colours when a mask is given.
  - `feedback_points(lines: dict[str, Run | list[Run]], points: dict[str, list[tuple[int, float]]]) -> Figure`: final-gap comparison against units of feedback; trajectories for sampling algorithms, points for fixed-dataset ones; log x-axis.
- Produces in `nbtools.py`:
  - `source_md(fn: Callable) -> str`: the function's source in a fenced Python block.
  - `reveal(question: str, answer: str) -> mo.Html`: the question as text and the answer inside a collapsed `mo.accordion`.

- [ ] **Step 1: Write the failing tests.** Use `matplotlib.use("Agg")`. Build two fake `Run`s by hand (ten points each, any numbers). Assert: `set(COLORS) == {"ref", "optimum", "reinforce", "ppo", "grpo", "dpo", "ppo_rm"}`; `training_curves({"ppo": run, "grpo": [run, run2]})` returns a `Figure` with 3 axes, and with `keys=("gap",)` 1 axis; `frontier_plot`, `ratio_hist(torch.rand(500) + 0.5, 0.2)`, `distribution_bars`, `scatter_true_vs_estimate` (with and without a mask) and `feedback_points` each return a `Figure`; `ratio_hist`'s title contains a percent sign; `source_md(training_curves)` starts with "```python" and contains `def training_curves`; `reveal("q", "a")` returns without error.
- [ ] **Step 2: Run** `uv run pytest tests/test_plots.py -v`. Expected: FAIL, module not found.
- [ ] **Step 3: Implement** both modules.
- [ ] **Step 4: Run** the same command. Expected: all pass.
- [ ] **Step 5: Commit** (`feat: shared figures and notebook helpers`).

---

### Notebook tasks 12 to 15 and 18: common rules

- First add `tests/test_notebooks.py` (in Task 12; later tasks only rely on it): parametrised over `sorted(Path("notebooks").glob("*.py"))`, marked `slow`, each case runs `subprocess.run([sys.executable, str(path)], timeout=300)` from the repo root and asserts return code 0. One more test, not slow: no file among `00_*.py` to `03_*.py` contains the strings `proxopt.olmo` or `data/olmo`.
- Every notebook has these sections in this order, as markdown headings: **Where we are**, **Definition**, **Intuition**, **The code**, **Simulation**, **Things to try**, **Where it went next** (one short paragraph), **References** (`cite(...)` only).
- "The code" shows `nbtools.source_md(<loss function>)` directly under the equation it implements.
- Controls are marimo UI elements; training cells are cached by config (`mo.cache` or equivalent) so returning to a setting is instant. Each notebook must open (all cells run) in well under a minute.
- Every simulation has a seed control and plots `training_curves` with `reward`, `kl`, `gap`.
- A change to a control retrains at most two runs (the algorithm, plus one comparison run that is cached when its own config did not change). Anything that needs several seeds or several algorithms at once sits in its own cell behind a `mo.ui.run_button`.
- Each "things to try" item uses `nbtools.reveal`. Write the reveal after running the experiment; the "expected" notes below are what the theory predicts, to be confirmed or reported. Reveals hold text and measured numbers, not live figures.
- Each notebook ends its Simulation section with a short "what carries over to real LLMs, and what does not" note (§2 anti-goals).
- Check with `uv run marimo check notebooks/NN_name.py` if this marimo version has that command, and by running the notebook as a script.
- Commit each notebook on its own.

---

### Task 12: Notebook `00_foundations`

**Files:**
- Create: `notebooks/00_foundations.py`, `tests/test_notebooks.py`

**Interfaces:**
- Consumes: `World`, `load_reference`, `train_reference`, `exact`, `reinforce`, `plots`, `nbtools`, `refs`.

- [ ] **Step 1: Write `tests/test_notebooks.py`** as described above; run it; expected: the "no OLMo import" test passes vacuously, no notebooks yet.
- [ ] **Step 2: Write the notebook.** Content:
  - **Where we are:** the three algorithms, the order of the set, and the notation table (Global Constraints).
  - **Definition:** the world (§5.1), the policy as a token-by-token model, the reference policy as the stand-in for an SFT model. Then
    `J_β(π) = E_x E_{y∼π(·|x)}[r(x, y)] − β · KL(π(·|x) ‖ π_ref(·|x))`,
    `π*_β(y|x) = π_ref(y|x) · exp(r(x, y)/β) / Z_β(x)`, `J_β(π*_β) = β · E_x[log Z_β(x)]`,
    and the gap identity `J_β(π*_β) − J_β(π) = β · KL(π ‖ π*_β)`, each with a two-line derivation.
  - **Intuition, part 1, one choice:** a one-step world `World(vocab_size=8, length=1, targets=(5,))`, handled as tables only (no network): its `reward_table()` and a hand-written, clearly non-uniform `π_ref` over the 8 responses. A β slider (log scale, 0.02 to 2, default 0.1) drives `distribution_bars` of `π_ref`, `π*_β` and the rewards.
  - **Intuition, part 2, five tokens:** the default world. A table of ten responses sampled from `π_ref` with their rewards for a chosen prompt; the frontier (`exact.frontier` over 40 values of β from 0.01 to 10, log-spaced) with the point for the slider's β marked.
  - **The code / Simulation:** REINFORCE (cite `williams1992`). The gradient from Task 5, `reinforce_loss` source, then controls: β, batch size (16 to 256), baseline (none / mean), seed. Plot `training_curves`, and a `frontier_plot` with the run's trajectory and the target point.
  - **Things to try:** (1) baseline none against mean at batch size 16, three seeds each: expected, visibly wider spread without the baseline. (2) β at 0.02: expected, the optimum concentrates on the highest-reward responses and drift is large. (3) Halve the batch size and double the steps: expected, similar feedback, noisier path. Close with the two weaknesses that motivate what follows: noise, and each batch of samples used for a single step.
  - **References:** `williams1992`, `ziegler2019`, `ouyang2022`.
- [ ] **Step 3: Run** `uv run python notebooks/00_foundations.py` and `uv run pytest tests/test_notebooks.py -m "" -v`. Expected: exit code 0; tests pass. Report the time to run the notebook.
- [ ] **Step 4: Commit** (`feat: notebook 00, foundations`).

---

### Task 13: Notebook `01_ppo`

**Files:**
- Create: `notebooks/01_ppo.py`

**Interfaces:**
- Consumes: `ppo`, `reinforce`, `exact`, `plots`, `nbtools`, `refs`.

- [ ] **Step 1: Write the notebook.** Compact: the reader knows PPO. Content:
  - **Where we are:** REINFORCE's two weaknesses; PPO's answers (a learned critic for the noise, a clipped objective so one batch can be reused).
  - **Definition**, in the RLHF form: state `s_t = (x, y_<t)`, action `y_t`; token reward `r_t = −β·log(π_old(y_t|s_t)/π_ref(y_t|s_t)) + 1[t = T]·r(x, y)`; `δ_t = r_t + V(s_{t+1}) − V(s_t)`, `Â_t = Σ_l λ^l δ_{t+l}`; `ρ_t(θ) = π_θ(y_t|s_t)/π_old(y_t|s_t)`; `L^CLIP(θ) = E[min(ρ_t·Â_t, clip(ρ_t, 1−ε, 1+ε)·Â_t)]`; the value loss. State that the sum of the token rewards is `r − β·log(π_old(y|x)/π_ref(y|x))`, so this PPO aims at `π*_β`. One paragraph on the two-stage RLHF pipeline (a reward model fitted to human preferences, then PPO), which notebooks 03 and 04 return to.
  - **Intuition:** clipping as a trust region on each token's probability; a small plot of the clipped objective against ρ for a positive and a negative advantage.
  - **The code:** `token_rewards`, `gae`, `ppo_loss` sources under their equations.
  - **Simulation:** controls: ε (0.01 to 1, default 0.2), β, epochs per batch (1 to 8, default 4), λ (0 to 1, default 0.95), batch size, reward mode, seed. Plots: `training_curves` with REINFORCE at the same β and batch size alongside, against steps and against feedback; `ratio_hist` of `extras["ratios"]`; `frontier_plot`.
  - **Things to try:** (1) epochs 8 with ε = 0.2 against ε = 1: expected, without effective clipping the reused batch pushes ratios far from 1 and the gap curve becomes erratic or worse. (2) epochs 1: expected, close to REINFORCE with a critic. (3) λ = 0 against λ = 1 early in training, when the critic is untrained: expected, λ = 0 leans on a bad critic. (4) verifiable reward: expected, slower start because most samples score 0.
  - **Where it went next:** one paragraph (PPO in InstructGPT; its cost: a critic the size of the policy).
  - **References:** `schulman2015trpo`, `schulman2016gae`, `schulman2017ppo`, `christiano2017`, `ziegler2019`, `stiennon2020`, `ouyang2022`.
- [ ] **Step 2: Run** the notebook as a script and `uv run pytest tests/test_notebooks.py -m "" -v`. Expected: exit code 0; tests pass.
- [ ] **Step 3: Commit** (`feat: notebook 01, PPO`).

---

### Task 14: Notebook `02_grpo`

**Files:**
- Create: `notebooks/02_grpo.py`

**Interfaces:**
- Consumes: `grpo`, `ppo`, `exact`, `plots`, `nbtools`, `refs`; the four final gaps reported in Task 7.

- [ ] **Step 1: Write the notebook.** Content:
  - **Where we are:** a three-row diff table against PPO: baseline (critic → group mean), advantage (per token → one per response, standardised in its group), KL (in the reward → a separate loss term).
  - **Definition:** for each prompt, responses `y_1 … y_G ∼ π_old(·|x)`; `Â_i = (r_i − mean(r_1..G)) / std(r_1..G)`;
    `J_GRPO(θ) = E[ (1/G) Σ_i (1/|y_i|) Σ_t { min(ρ_{i,t}·Â_i, clip(ρ_{i,t}, 1−ε, 1+ε)·Â_i) − β·D_KL[π_θ ‖ π_ref] } ]`,
    with the estimator `D_KL = π_ref(y_{i,t}|·)/π_θ(y_{i,t}|·) − log(π_ref(y_{i,t}|·)/π_θ(y_{i,t}|·)) − 1`.
  - **What GRPO aims at** (its own subsection, §6.2): (a) the estimator's expected value is `KL(π_θ ‖ π_ref)`, but with samples held fixed `∇D_KL = (1 − π_ref/π_θ)·∇log π_θ`, whose expectation under `π_θ` is `∇KL(π_ref ‖ π_θ)`; show the three-line derivation and say that the test suite checks it. (b) With the KL term outside the advantage, dividing by the group standard deviation changes the balance between reward and KL separately for each prompt and each batch, so the effective β is not constant. (c) Consequence: GRPO as published does not maximise `J_β`, and its gap to `π*_β` levels off above zero. State the four measured final gaps for the two switches. Theory predicts the gap closes whenever the KL is in the reward, with or without the standard-deviation division (dividing a correct advantage by a positive number only changes step sizes); confirm this against the numbers from Task 7 and report if it does not hold.
  - **Intuition:** the group as a cheap, critic-free baseline; `baseline_comparison` on a PPO-trained policy: two `scatter_true_vs_estimate` panels, group against exact and critic against exact, with a G slider.
  - **The code:** `group_advantages`, `kl_estimator`, `grpo_loss` sources under their equations.
  - **Simulation:** controls: G (2 to 32, default 8), ε, β, reward mode, the two switches (defaults: as published), seed. Plots: `training_curves` (keys `reward`, `kl`, `kl_rev`, `gap`) with one PPO run at the same β and seed alongside; dead groups per iteration; `frontier_plot`. Then a side-by-side cell behind a run button: GRPO as published, GRPO with both switches PPO-like, and PPO, three seeds each drawn as bands, against steps and against feedback.
  - **Things to try:** (1) flip each switch alone, then both: expected as in (c). (2) verifiable reward with G = 2 against G = 16: expected, many dead groups at G = 2, hence wasted samples. (3) G = 2 with graded reward and standardisation on: expected, every advantage is ±0.707 whatever the size of the reward difference (state the number the notebook actually shows). (4) compare with PPO at equal feedback.
  - **Where it went next:** one paragraph (DeepSeek-R1; RL with verifiable rewards in Tülu 3; variants that drop the standard-deviation division or the KL term).
  - **References:** `shao2024`, `guo2025`, `lambert2024`, `schulman2017ppo`.
- [ ] **Step 2: Run** the notebook as a script and the notebook tests. Expected: exit code 0; tests pass.
- [ ] **Step 3: Commit** (`feat: notebook 02, GRPO`).

---

### Task 15: Notebook `03_dpo`

**Files:**
- Create: `notebooks/03_dpo.py`

**Interfaces:**
- Consumes: `dpo`, `prefs`, `exact`, `plots`, `nbtools`, `refs`; the two final gaps reported in Task 9.

- [ ] **Step 1: Write the notebook.** The reader knows little about DPO, so the derivation is in full, one numbered step per cell, each with a sentence of explanation. Content:
  - **Where we are:** a diff table against PPO and GRPO: no sampling from the policy during training, no reward function, no critic; input is a fixed set of (prompt, preferred, rejected) triples. Recall the two-stage RLHF pipeline from notebook 01: DPO collapses both stages into one loss.
  - **Definition (the derivation):**
    1. The objective and its optimum from notebook 00: `π*_β(y|x) = π_ref(y|x)·exp(r(x,y)/β)/Z_β(x)`.
    2. Solve for the reward: `r(x, y) = β·log(π*_β(y|x)/π_ref(y|x)) + β·log Z_β(x)`.
    3. The Bradley–Terry model: `p(y_1 ≻ y_2 | x) = σ(r(x, y_1) − r(x, y_2))`.
    4. Substitute 2 into 3: the `β·log Z_β(x)` terms cancel, leaving `p(y_1 ≻ y_2 | x) = σ(β·log(π*(y_1|x)/π_ref(y_1|x)) − β·log(π*(y_2|x)/π_ref(y_2|x)))`.
    5. Maximum likelihood over the data with `π_θ` in place of `π*`: `L_DPO(θ) = −E_{(x, y_w, y_l)}[log σ(β·log(π_θ(y_w|x)/π_ref(y_w|x)) − β·log(π_θ(y_l|x)/π_ref(y_l|x)))]`.
    6. The gradient: `∇L_DPO = −β·E[σ(r̂(x, y_l) − r̂(x, y_w))·(∇log π_θ(y_w|x) − ∇log π_θ(y_l|x))]` with `r̂ = β·log(π_θ/π_ref)`: pairs the model already ranks correctly contribute little.
    7. Noise: if preferences follow `σ((r_1 − r_2)/τ)`, they are Bradley–Terry preferences for the reward `r/τ`, so DPO approaches `π_ref·exp(r/(β·τ)) = π*_{β·τ}`. β and τ enter only through their product.
  - **Intuition:** a classifier on pairs whose logit is the difference of implicit rewards; what "implicit reward" means and why it is defined only up to a constant per prompt.
  - **The code:** `dpo_loss` source; the pair sampler described in two sentences.
  - **Simulation:** controls: β (default 0.4), τ (0.05 to 1, default 0.25), number of pairs (100 to 20,000, default 4000), source of pairs (`π_ref` / uniform), seed. Plots: `training_curves` with the gap measured against `π*_{β·τ}`; `frontier_plot` with two targets marked, `π*_β` and `π*_{β·τ}`; `scatter_true_vs_estimate` of `implicit_reward` against centred `r/τ`, coloured by `coverage_mask`.
  - **Things to try:** (1) move τ with β fixed: expected, the target and the endpoint move together. (2) halve β and double τ: expected, same target. (3) pairs from `π_ref` against uniform: expected, on uncovered responses the implicit reward is unconstrained and scatters away from the line; report the two gaps from Task 9. (4) 100 pairs: expected, the loss falls while the gap does not.
  - Then one paragraph on the length-normalised variant used in the Tülu 3 recipe (cite `lambert2024`), stating only what `docs/olmo-facts.md` and the paper confirm.
  - **Where it went next:** one paragraph.
  - **References:** `rafailov2023`, `bradley1952`, `christiano2017`, `ouyang2022`, `lambert2024`.
- [ ] **Step 2: Run** the notebook as a script and the notebook tests. Expected: exit code 0; tests pass.
- [ ] **Step 3: Commit** (`feat: notebook 03, DPO`).

---

### Task 16: OLMo measurements

**Files:**
- Create: `src/proxopt/olmo.py`, `scripts/olmo_measure.py`, `data/olmo/margins.jsonl`, `data/olmo/drift.jsonl`, `data/olmo/meta.json`
- Test: `tests/test_olmo.py`

**Interfaces:**
- Consumes: `../gricean-devices/configs/models.yaml` (entries `olmo_sft`, `olmo_dpo`, `olmo_instruct`: `repo`, `revision`), `../gricean-devices/data/responses/olmo_{dpo,instruct}__tmpl.jsonl`, the dataset `allenai/olmo-2-0425-1b-preference-mix`, `docs/olmo-facts.md`.
- Produces in `olmo.py`:
  - `response_logprob(logits: Tensor[L, V], input_ids: LongTensor[L], response_start: int) -> tuple[float, int]`: summed log-probability of tokens `input_ids[response_start:]` (token `i` is predicted by `logits[i − 1]`) and their count.
  - `next_token_kl(logits_a: Tensor[L, V], logits_b: Tensor[L, V], response_start: int) -> tuple[float, float]`: mean over the positions that predict response tokens of `KL(a ‖ b)` and of `KL(b ‖ a)`, computed in float32.
  - `summarize_margins(rows: list[dict]) -> dict`: for `dpo` and `instruct`, each against `sft`, and for both the summed and the per-token-mean versions: the mean margin and the share of pairs with a positive margin, plus `n`.
  - `load_margins(root: Path | None = None) -> list[dict] | None`, `load_drift(...) -> list[dict] | None`, `load_meta(...) -> dict | None`: `None` when the file is missing. Default root is the environment variable `PROXOPT_OLMO_DIR` when set, otherwise `data/olmo/` located from the package file, not from the working directory.
- File formats:
  - `margins.jsonl`, one row per pair: `pair` (running index), `n_w`, `n_l` (response token counts), and `lp_w_<m>`, `lp_l_<m>` (summed log-probabilities) for `m` in `sft`, `dpo`, `instruct`. No text from the dataset is stored.
  - `drift.jsonl`, one row per text and step: `step` (`"sft->dpo"` or `"dpo->instruct"`), `item_id`, `mode`, `seed`, `n_tokens`, `kl_new_old`, `kl_old_new`.
  - `meta.json`: date, the three revisions, dataset name and revision, counts, device, dtype, the dataset shuffle seed, and the generation settings of the sister repo's samples (read from `../gricean-devices/configs/generation.yaml`).

- [ ] **Step 1: Write the failing tests**

```python
import json, pytest, torch
from proxopt import olmo

def test_response_logprob_alignment():
    g = torch.Generator().manual_seed(0); logits = torch.randn(6, 11, generator=g); ids = torch.tensor([3, 1, 4, 1, 5, 9])
    lp = logits.log_softmax(-1); want = lp[2, 1] + lp[3, 5] + lp[4, 9]            # tokens 3, 4, 5
    got, n = olmo.response_logprob(logits, ids, 3)
    assert n == 3 and got == pytest.approx(want.item(), abs=1e-5)

def test_next_token_kl_both_directions():
    g = torch.Generator().manual_seed(0); a = torch.randn(6, 11, generator=g); b = torch.randn(6, 11, generator=g)
    la, lb = a.log_softmax(-1)[2:5], b.log_softmax(-1)[2:5]
    ab, ba = olmo.next_token_kl(a, b, 3)
    assert ab == pytest.approx((la.exp() * (la - lb)).sum(-1).mean().item(), abs=1e-5)
    assert ba == pytest.approx((lb.exp() * (lb - la)).sum(-1).mean().item(), abs=1e-5)
    assert olmo.next_token_kl(a, a, 3) == pytest.approx((0.0, 0.0), abs=1e-6)

def test_summarize_margins():
    rows = [dict(pair=0, n_w=2, n_l=4, lp_w_sft=-4, lp_l_sft=-4, lp_w_dpo=-2, lp_l_dpo=-8, lp_w_instruct=-4, lp_l_instruct=-4),
            dict(pair=1, n_w=2, n_l=2, lp_w_sft=-4, lp_l_sft=-4, lp_w_dpo=-6, lp_l_dpo=-4, lp_w_instruct=-3, lp_l_instruct=-4)]
    s = olmo.summarize_margins(rows)
    assert s["n"] == 2
    assert s["dpo"]["sum"]["mean_margin"] == pytest.approx((6 + -2) / 2) and s["dpo"]["sum"]["share_positive"] == 0.5
    assert s["dpo"]["mean"]["mean_margin"] == pytest.approx(((1 - (-1)) + (-1 - 0)) / 2)
    assert s["instruct"]["sum"]["share_positive"] == 0.5

def test_loaders_return_none_when_files_are_missing(tmp_path):                    # Review Focus 5
    assert olmo.load_margins(tmp_path) is None and olmo.load_drift(tmp_path) is None and olmo.load_meta(tmp_path) is None
    (tmp_path / "margins.jsonl").write_text(json.dumps({"pair": 0}) + "\n")
    assert olmo.load_margins(tmp_path) == [{"pair": 0}] and olmo.load_drift(tmp_path) is None

def test_default_root_honours_the_environment(tmp_path, monkeypatch):
    (tmp_path / "meta.json").write_text(json.dumps({"n": 1}))
    monkeypatch.setenv("PROXOPT_OLMO_DIR", str(tmp_path))
    assert olmo.load_meta() == {"n": 1} and olmo.load_margins() is None
```

(The margin for a model `m` is `(lp_w_m − lp_w_sft) − (lp_l_m − lp_l_sft)`; the per-token-mean version divides each log-probability by its token count first.)

- [ ] **Step 2: Run** `uv run pytest tests/test_olmo.py -v`. Expected: FAIL, module not found.
- [ ] **Step 3: Implement `olmo.py`**; run the test; expected: 5 passed.
- [ ] **Step 4: Read how the sister repo loads and prompts these models** (`../gricean-devices/src/gricean/`, `configs/generation.yaml`): device, dtype, how the chat template was applied to produce the `__tmpl` responses, and what `prompt_text` contains. Tokenise exactly as it did, so that `response_start` is the first token of the response.
- [ ] **Step 5: Write `scripts/olmo_measure.py`** with subcommands `margins [--n 300]`, `drift [--n 120]`, `all`. It refuses to run if a checkpoint revision is not already in the local Hugging Face cache (use `local_files_only=True` for models), so it can never download a model.
  - `margins`: stream the dataset, shuffle with a fixed seed and a 10,000-row buffer, keep single-turn pairs whose prompt plus either response is at most 512 tokens, take the first `n`. Load one checkpoint at a time, score every pair, free it, load the next.
  - `drift`: for each step, take up to `n` English rows of the later checkpoint's `__tmpl` responses file (all modes; skip empty responses), load the two checkpoints of that step together in half precision (about 6 GB), and compute `next_token_kl`. This departs from the spec's "one checkpoint at a time" because comparing full next-token distributions needs both; if memory does not allow it, stop and report rather than change the measurement.
  - Write the three files; print `summarize_margins` and the mean of each KL column per step.
- [ ] **Step 6: Run** `make olmo`. Expected: three files in `data/olmo/`; margins rows equal to `--n`; no network access to model repositories. Report the printed summary and the run time. Sanity check to report: the share of positive margins for `dpo` should be well above 0.5 (these are DPO's training pairs); if it is not, suspect the tokenisation or the template before anything else, and say so.
- [ ] **Step 7: Commit** code, tests and `data/olmo/` (`feat: OLMo measurements`).

---

### Task 17: The synthesis comparison, precomputed

Notebook 04 compares five runs over three seeds plus eight fixed-dataset points. That is too slow to compute when the notebook opens, so the default comparison is computed once by a script and committed.

**Files:**
- Create: `src/proxopt/synthesis.py`, `scripts/synthesis.py`, `data/synthesis.json`
- Modify: `Makefile` (target `synthesis`: `uv run python scripts/synthesis.py`)
- Test: `tests/test_synthesis.py`

**Interfaces:**
- Consumes: all four algorithms, `prefs`, `Run.to_json_dict`, `Run.from_json_dict`.
- Produces in `src/proxopt/synthesis.py`:
  - `NAMES = ("ppo", "grpo", "grpo_ppo_like", "dpo", "ppo_rm")`, `PAIR_COUNTS = (250, 1000, 4000, 16000)`, `LR_GRID = (1e-3, 3e-3, 1e-2)`.
  - `configs(beta: float, tau: float, seed: int, lr: dict[str, float]) -> dict[str, Any]`: one config per name. PPO and both GRPOs use `beta`; DPO and PPO-on-reward-model use `beta / tau`; `grpo_ppo_like` is `GRPOConfig(std_normalize=False, kl_in="reward")`.
  - `run_one(name: str, ref: Policy, world: World, beta: float, tau: float, seed: int, lr: float, n_pairs: int = 4000) -> Run`: runs that algorithm. `dpo` and `ppo_rm` draw `sample_pairs(world, ref, n_pairs, tau, "ref", seed=seed)`; `ppo_rm` fits `fit_reward_model` on those pairs and calls `ppo.train(..., reward_fn=model)`. Every run's `target_beta` must equal `beta` (for `ppo_rm`, set it on the returned run: its PPO config carries `beta / tau`, but its target is `π*_β`).
  - `compare(beta: float = 0.1, tau: float = 0.25, seeds: Sequence[int] = (0, 1, 2), lr: dict[str, float] | None = None) -> dict`: `{"beta", "tau", "runs": {name: [Run, ...]}, "points": {"dpo": [(n_pairs, final_gap), ...], "ppo_rm": [...]}}`, points from seed 0 over `PAIR_COUNTS`.
  - `tune(beta: float = 0.1, tau: float = 0.25, seeds: Sequence[int] = (0, 1, 2)) -> tuple[list[dict], dict[str, float]]`: for each name and each learning rate in `LR_GRID`, the mean final gap over seeds; returns the table rows (`name`, `lr`, `mean_final_gap`) and the best learning rate per name.
  - `save(path: Path, grid: list[dict], result: dict) -> None` and `load(path: Path | None = None) -> dict | None` (`None` when the file is missing; runs come back through `Run.from_json_dict`; default path `data/synthesis.json` located from the package file).

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from proxopt import synthesis
from proxopt.world import DEFAULT_WORLD
from proxopt.reference import load_reference

def test_configs_share_one_target():
    c = synthesis.configs(beta=0.1, tau=0.25, seed=3, lr={n: 3e-3 for n in synthesis.NAMES})
    assert set(c) == set(synthesis.NAMES)
    assert c["ppo"].beta == c["grpo"].beta == c["grpo_ppo_like"].beta == 0.1
    assert c["dpo"].beta == c["ppo_rm"].beta == pytest.approx(0.4)
    assert (c["grpo"].std_normalize, c["grpo"].kl_in) == (True, "loss")
    assert (c["grpo_ppo_like"].std_normalize, c["grpo_ppo_like"].kl_in) == (False, "reward")
    assert all(cfg.seed == 3 for cfg in c.values())

def test_load_returns_none_when_missing(tmp_path):
    assert synthesis.load(tmp_path / "nope.json") is None

@pytest.mark.slow
def test_every_run_targets_beta_and_round_trips(tmp_path):
    ref = load_reference(DEFAULT_WORLD)
    runs = {n: [synthesis.run_one(n, ref, DEFAULT_WORLD, 0.1, 0.25, 0, 3e-3, n_pairs=500)] for n in synthesis.NAMES}
    assert all(r[0].target_beta == pytest.approx(0.1) for r in runs.values())
    result = {"beta": 0.1, "tau": 0.25, "runs": runs, "points": {"dpo": [(500, 0.1)], "ppo_rm": [(500, 0.2)]}}
    synthesis.save(tmp_path / "s.json", [{"name": "ppo", "lr": 3e-3, "mean_final_gap": 0.01}], result)
    back = synthesis.load(tmp_path / "s.json")
    assert set(back["runs"]) == set(synthesis.NAMES) and back["runs"]["ppo"][0].metrics == runs["ppo"][0].metrics
    assert back["grid"][0]["name"] == "ppo" and [tuple(p) for p in back["points"]["dpo"]] == [(500, 0.1)]

@pytest.mark.slow
def test_committed_file_is_complete():
    data = synthesis.load()
    assert data is not None and set(data["runs"]) == set(synthesis.NAMES)
    assert all(len(v) == 3 for v in data["runs"].values())
    assert [p[0] for p in data["points"]["dpo"]] == list(synthesis.PAIR_COUNTS)
    assert len(data["grid"]) == len(synthesis.NAMES) * len(synthesis.LR_GRID)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_synthesis.py -v -m ""`. Expected: FAIL, module not found.
- [ ] **Step 3: Implement `src/proxopt/synthesis.py` and `scripts/synthesis.py`** (the script calls `tune`, then `compare` with the best learning rates, then `save`).
- [ ] **Step 4: Run** `make synthesis`, then the tests again. Expected: `data/synthesis.json` written (it holds only lists of numbers; if it exceeds about 1 MB, record fewer points per run); all tests pass. Report the grid table, the chosen learning rates, the run time, and each algorithm's mean final gap.
- [ ] **Step 5: Commit** code, tests and `data/synthesis.json` (`feat: precomputed synthesis comparison`).

---

### Task 18: Notebook `04_synthesis`, README

**Files:**
- Create: `notebooks/04_synthesis.py`
- Modify: `README.md`, `tests/test_notebooks.py`

**Interfaces:**
- Consumes: `synthesis.load`, `synthesis.compare`, `olmo.load_*`, `olmo.summarize_margins`, `exact.frontier`, `plots`, `refs`; `docs/olmo-facts.md` (its two last lines decide the wording of the OLMo section).

- [ ] **Step 1: Add the failing test** to `tests/test_notebooks.py`, marked `slow`: run `notebooks/04_synthesis.py` with the environment variable `PROXOPT_OLMO_DIR` pointing at an empty temporary directory; assert return code 0. Review Focus 5.
- [ ] **Step 2: Write the notebook.** Content:
  - **Where we are:** one paragraph and the table of what each algorithm needs (reward function, critic, preference data, fresh samples), what it aims at (`π*_β`; not `J_β`, see notebook 02; `π*_{β·τ}`), and its measured cost in this notebook's runs.
  - **One world, one target:** the committed comparison from `synthesis.load()` at β = 0.1 and τ = 0.25: PPO; GRPO as published; GRPO with both switches PPO-like; DPO with `β/τ` on pairs from `π_ref`; PPO on a reward model fitted to the same pairs (colour `ppo_rm`); three seeds each. Plots: `frontier_plot` with all trajectories and the single target `π*_β`; `training_curves` of the gap against gradient steps. Below it, β, τ and seed controls and a run button that recomputes the comparison live with `synthesis.compare` for other settings (one seed), cached by its arguments. If `synthesis.load()` is `None`, show a notice naming `make synthesis` and leave only the live cell.
  - **Feedback:** `feedback_points`: PPO and GRPO as trajectories against cumulative reward calls; DPO and PPO-on-reward-model as points at the four pair counts. State in the text that a reward call and a labelled pair are different things, that the learning rate was tuned per algorithm on a small grid (show the committed grid and the chosen values in a table), and that these plots do not rank the algorithms.
  - **The fair comparison:** DPO against PPO-on-reward-model on the same pairs, with what the simulation shows about where each loses accuracy (reward-model error against coverage).
  - **History:** PPO 2017, RLHF 2019 to 2022, DPO 2023, GRPO 2024, RL with verifiable rewards; why RL returned after DPO (rewards a program can check need no annotators; exploration beyond a fixed dataset).
  - **The OLMo ladder:** if `load_margins()` or `load_drift()` is `None`, show a notice naming `make olmo` and nothing else. Otherwise: the recipe as the model cards state it, with the GRPO attribution worded according to `GRPO_CONFIRMED`; the distribution of implicit-reward margins of DPO against SFT with the share positive, summed and per-token (lead with the one matching `DPO_VARIANT`), and the same for Instruct against SFT, with the caveat that these are training pairs; the mean next-token KL for the two steps in both directions, labelled exactly as "mean next-token KL along the later checkpoint's own responses (sampled at the settings in `meta.json`)", never as the KL between the models.
  - **References:** all keys.
- [ ] **Step 3: Write `README.md`:** what the set is, how to install (`make install`), how to open each notebook, the order to read them, what `make olmo` and `make synthesis` do and that both are optional (their outputs are committed), how to run tests (`make test`, `make test-all`), and a link to the spec.
- [ ] **Step 4: Run** `make test-all` and `make lint`. Expected: all tests pass, including every notebook as a script and the empty-OLMo-directory case; no lint errors. Report the run time of each notebook.
- [ ] **Step 5: Commit** (`feat: notebook 04, synthesis; README`).
