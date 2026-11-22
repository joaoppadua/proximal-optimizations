# PPO, GRPO and DPO in five marimo notebooks: design

Date: 2026-11-21. Status: reviewed by a builder check; awaiting João's approval.

## 1. Intent

João wants general mastery of three algorithms used to post-train LLMs:

- **PPO**, Proximal Policy Optimization (Schulman et al. 2017), as used in RLHF (Ouyang et al. 2022). He knows this one fairly well.
- **GRPO**, Group Relative Policy Optimization (Shao et al. 2024). He knows it less well.
- **DPO**, Direct Preference Optimization (Rafailov et al. 2023). He knows it very little.

The deliverable is a set of marimo notebooks that give, for each algorithm, its definition and formalization, the intuition, and a simulation of how it works. The notebooks build on each other and make the differences between the algorithms explicit. They cite the seminal work for each.

This is a study tool for one reader who already knows RL vocabulary. It is not a companion to the `gricean-devices` paper. The OLMo checkpoints from that sister repo appear only as a reality check.

## 2. Goals, non-goals, anti-goals

**Goals**

- After working through the set, the reader could implement each algorithm on a real LLM. The simulation code therefore has the same shape as real LLM training code: per-token log-probabilities, a value head, group sampling per prompt, summed sequence log-probabilities.
- Every algorithm is measured against the same exact yardstick (section 5.3), on the same world, on the same axes. Not every algorithm aims at that yardstick: where one does not, the notebook says why, and the remaining gap is part of the lesson (sections 6.2 and 6.3).
- Each notebook states what changed relative to the previous one.
- Simulations are live: changing a control retrains in a few seconds.

**Non-goals**

- Training a real LLM with any of the three.
- Claims about which algorithm is better in general.
- Covering variants (TRPO in depth, IPO, KTO, SimPO, DAPO, Dr. GRPO and so on). One short "where it went next" paragraph per notebook is enough.
- Portuguese versions.

**Anti-goals** (these make the project a failure even if everything runs)

- Math that looks right and is wrong. Every equation is checked against its source paper, and where possible against a unit test.
- Citations written from memory. Every reference is verified against the paper itself (authors, year, title, arXiv id or venue) before it goes into a notebook.
- Toy results presented as facts about LLMs. Each simulation says what carries over and what does not.
- Conclusions that depend on one lucky seed. Simulations expose the seed and show run-to-run spread where a comparison is drawn.
- Notebooks that take long to open or to react.

## 3. Constraints

- Machine: Apple M3, 16 GB. The toy world trains on CPU.
- marimo notebooks stored as `.py` files, opened with `uv run marimo edit notebooks/NN_name.py`. marimo is not installed globally; it is a project dependency.
- Project managed with `uv`, Python 3.12 or later, same conventions as `gricean-devices` (a `src/` package, `tests/` with pytest, a `Makefile`).
- No new model downloads and no new text generation with OLMo. Only three cached OLMo 2 1B checkpoints are used (SFT, DPO, Instruct), at the revisions pinned in `../gricean-devices/configs/models.yaml`, and only for forward passes.
- Notebooks must run with no OLMo model in memory. OLMo measurements are precomputed by a script and committed as small files.
- Everything is seeded and reproducible.
- Language: English.

## 4. Approaches considered

**A. One-step choice among fixed responses.** Each prompt has about a dozen candidate responses and the policy is a table of probabilities. Instant and exactly plottable, but there are no tokens: PPO's critic and advantage estimation collapse to almost nothing, and GRPO's one-reward-per-sequence step cannot be shown.

**B. Tiny token-by-token language model (chosen by João).** A small vocabulary, short responses generated autoregressively by a small network that is first trained on a synthetic corpus. The code has the shape of real LLM training, it trains in seconds, and the space of responses is small enough to enumerate, so exact answers exist. Costs more code and a torch dependency. With response length 1 it reduces to A.

**C. A real small LM** (SmolLM2-360M, or LoRA on OLMo 1B). The real thing, but each run takes minutes, results are noisy, and there is no exact answer to compare against.

B was picked because it is the only one that is both faithful to the LLM setting and exactly checkable.

## 5. The shared world

All five notebooks import the same world from a package, so they differ only in the algorithm.

### 5.1 Prompts, responses, reward

- Vocabulary of 8 tokens (the digits 0 to 7). A response `y` is exactly 5 tokens. There are 8^5 = 32,768 possible responses.
- A small fixed set of prompts `x` (about 4 to 8). Each prompt is a target number.
- Reward `r(x, y)` is a programmatic check on the whole response, in [0, 1], with two modes selectable in the notebooks:
  - **graded** (default): `1 − |sum(y) − target(x)| / D`, with `D` chosen so the range is [0, 1].
  - **verifiable**: 1 if `sum(y) == target(x)`, else 0. This is the stand-in for rewards checked by a program, and it exposes what happens when every sample in a group gets the same reward.
- The world's sizes are parameters, so tests can use a much smaller world (for example 3 tokens, length 2) and notebook 00 can start with length 1.

### 5.2 Policy and reference policy

- The policy `π_θ(y | x)` is a small causal sequence model in torch with a policy head and a value head (the value head is used only by PPO). It must support: sampling responses, per-token log-probabilities of given responses, and the full distribution over all responses for a prompt.
- The **reference policy** `π_ref` is the stand-in for the SFT model. It is that same network trained by maximum likelihood on a synthetic corpus that has a recognisable style (for example, each token tends to repeat or increment the previous one) and knows little about the target. So the reference is "fluent" but mediocre at the task. Every algorithm starts from `π_ref` and is regularised towards it.
- Training the reference is deterministic given a seed and fast enough to run when a notebook opens, or it is cached; the builder decides.

### 5.3 Exact quantities (the common yardstick)

Because all responses can be enumerated, the package computes exactly, per prompt and averaged over prompts:

- expected reward `E_π[r]`
- drift from the reference, `KL(π ‖ π_ref)`, and also the other direction, `KL(π_ref ‖ π)` (needed in 6.2)
- the regularised objective `J_β(π) = E_π[r] − β · KL(π ‖ π_ref)`
- its exact optimum `π*_β(y | x) ∝ π_ref(y | x) · exp(r(x, y) / β)`
- the gap to the optimum, using the identity `J_β(π*_β) − J_β(π) = β · KL(π ‖ π*_β)`
- the reward–drift frontier: the curve of (`KL(π*_β ‖ π_ref)`, `E_{π*_β}[r]`) as β sweeps

Every simulation plots reward, drift and gap over training. Notebook 04 plots each algorithm's trajectory in the reward–drift plane against the frontier.

## 6. The notebooks

Order follows what João already knows: PPO, then GRPO as a small step from it, then DPO as the one real conceptual jump. (My call; João accepted it.)

Every notebook has the same sections: where we are and what is new; definition and formalization; intuition; the code (the loss function's source shown next to its equation); simulation; things to try (three or four guided experiments, each asking for a prediction before revealing the outcome); references. Notation is fixed in notebook 00 and never changes: `x` prompt, `y` response, `y_t` token, `π_θ`, `π_ref`, `r`, `β`, `A` advantage.

### 6.0 `00_foundations`

- The world, shown concretely: sample responses from `π_ref`, their rewards.
- The objective `J_β(π)` and why the KL term is there.
- The exact optimum `π*_β` and the frontier, with a β slider.
- REINFORCE (Williams 1992) on this objective as the plain baseline: it works, but it is noisy and each batch of samples is used once. That motivates everything after.
- Starts with response length 1 (a single choice) before moving to length 5.

### 6.1 `01_ppo`

- Compact, since João knows PPO. This notebook is the reference implementation that later notebooks are compared against.
- Formalization in the RLHF form: tokens as actions, reward at the end of the response with the KL penalty folded into the reward (Ziegler et al. 2019; Ouyang et al. 2022), value head, generalised advantage estimation (Schulman et al. 2016), the clipped surrogate (Schulman et al. 2017), several epochs over one batch of samples.
- PPO in this form does aim at `π*_β`.
- Simulation controls: clip range ε, β, epochs per batch, GAE λ, batch size, reward mode, seed.
- Extra plot: histogram of probability ratios against the clip bounds, to show what clipping actually cuts.

### 6.2 `02_grpo`

- Framed as a diff against PPO: no value head; for each prompt sample a group of G responses; the advantage of a response is its reward standardised within its group (minus the group mean, divided by the group standard deviation) and is the same for all its tokens; the KL term is a separate term in the loss, using the per-token estimator given in Shao et al. 2024, `π_ref/π_θ − log(π_ref/π_θ) − 1`.
- **GRPO as published does not maximise `J_β`, and the notebook must say so and show it.** Two of the changes are not just rearrangements:
  - Dividing by the group standard deviation rescales the reward differently for each prompt and each batch.
  - The estimator's value is an unbiased estimate of `KL(π_θ ‖ π_ref)`, but used as a loss term on fixed samples its expected gradient is that of `KL(π_ref ‖ π_θ)`, the other direction. (Derivation in the notebook; checked by a test, section 9.)
  So GRPO's gap to `π*_β` levels off above zero. That is taught as a finding, not hidden.
- The default is GRPO exactly as in the paper. Two switches let the reader undo each departure separately: "divide by group standard deviation" (on/off) and "KL term" (in the loss with the paper's estimator / in the reward as in PPO). With both set to the PPO-like choice, GRPO aims at `π*_β` again. (My call: faithful by default, with the switches, because seeing each departure's effect is the point.)
- Other simulation controls: group size G, ε, β, reward mode, seed.
- Extra plots: advantage estimates from the group baseline against PPO's critic on the same samples; what happens under the verifiable reward when a whole group scores the same; both KL directions over training.
- Side-by-side run against PPO on the same world.

### 6.3 `03_dpo`

- The full derivation, step by step, from the same objective as notebook 00: the closed-form optimum; solving it for the reward; the Bradley–Terry model of preferences (Bradley and Terry 1952); the cancellation of the normalising constant; the resulting loss, `−log σ(β · [log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x)])`.
- Framed as a diff against both earlier notebooks: no sampling from the policy during training, no reward function, no critic; a fixed dataset of (prompt, preferred, rejected) triples.
- The preference dataset is generated from the world: pairs of responses, with the winner drawn with probability `σ((r_w − r_l)/τ)`, where τ is the preference noise.
- **What DPO aims at depends on τ.** With many pairs and full coverage, DPO with parameter β approaches `π*_{β·τ}`, not `π*_β`. It matches the yardstick only when τ = 1. The notebook derives this, draws the target `π*_{β·τ}` in its plots, and moves it when the noise slider moves. The lesson: DPO's β and the sharpness of the annotators cannot be told apart from the data.
- Simulation controls: β, number of pairs, preference noise τ, where the pairs come from (sampled from `π_ref`, or uniform over all responses), seed.
- Extra plot: the implicit reward `β · log(π_θ/π_ref)` against the true reward over all responses, split by whether a response was covered by the data. The implicit reward recovers `r/τ` only up to a constant per prompt, so it is centred per prompt before plotting.
- One paragraph on the length-normalised variant of DPO used in the Tülu 3 recipe, since it bears on section 7.

### 6.4 `04_synthesis`

- All three algorithms on one world against the exact frontier: trajectories in the reward–drift plane, and the gap to the optimum over training.
- So that all runs aim at the same point, DPO is run with `β/τ` in place of β. The same applies to the next item.
- The fair comparison from the DPO paper: PPO run on a reward model fitted (Bradley–Terry loss) to the same preference pairs DPO uses, next to DPO. (My call to include this; it is what makes PPO-based RLHF and DPO comparable.)
- The gap over training is drawn against two x-axes: gradient steps, and units of feedback consumed (reward calls for PPO and GRPO, labelled pairs for DPO). The notebook says plainly that these units are not commensurable and that the plots do not rank the algorithms. Hyperparameters are tuned per algorithm on a small grid, and the grid is shown. (My call.)
- A table of what each algorithm needs (a reward function, a critic, preference data, fresh samples), what it aims at, and what it cost in the simulations.
- The history in chronological order (PPO 2017, DPO 2023, GRPO 2024) and why RL came back after DPO.
- The OLMo ladder (section 7).

## 7. OLMo reality check

Two facts need confirming before a notebook states them. If either cannot be confirmed from the OLMo 2 paper or the open-instruct repository, the notebook reports only what the model cards say.

- **The last stage used GRPO.** The OLMo 2 1B base model card lists that model's post-training as "SFT+DPO+GRPO" and the 7B and 13B models' as "SFT + DPO + PPO". The cards for the 1B Instruct checkpoint describe its last stage as RL with verifiable rewards and do not name the algorithm. The lineage on the cards is SFT → DPO → RLVR1 → Instruct; RLVR1 is not cached and is not used.
- **Which DPO variant was used.** The Tülu 3 recipe uses length-normalised DPO. If OLMo 2 1B did too, the quantity DPO trained on is the per-token mean log-probability margin, not the summed one.

A script, `scripts/olmo_measure.py`, loads one checkpoint at a time and writes small result files under `data/olmo/`, which are committed. It uses the chat template throughout and computes two things:

1. **What DPO learned.** On a few hundred pairs streamed from the 1B preference dataset named on the DPO model card (`allenai/olmo-2-0425-1b-preference-mix`), the log-probability of the chosen and the rejected response under the SFT, DPO and Instruct checkpoints, both summed and as a per-token mean. From these: the implicit reward margin `log π_DPO(y_w)/π_SFT(y_w) − log π_DPO(y_l)/π_SFT(y_l)` per pair (both versions), the share of pairs where it is positive, and the same quantity for Instruct against SFT. The notebook states that these are training pairs, so this shows what DPO fitted, not how it generalises.
2. **How far each stage moved the model.** For the two steps SFT → DPO and DPO → Instruct: at every token position of a fixed set of texts, the exact KL divergence between the two checkpoints' next-token distributions, in both directions, averaged over positions. The texts are the later checkpoint's own responses already on disk in `../gricean-devices/data/responses/` (templated). This needs forward passes only. It is reported as "mean next-token KL along these texts", never as the KL between the two models: the sister repo's samples were drawn at temperature 0.7 and cut at 150 tokens, so they are not exact samples from either model. (My call, replacing an earlier plan that needed temperature-1 samples.)

The sister repo is read only by this script. The notebooks read only `data/olmo/`.

## 8. Code layout

```
pyproject.toml, uv.lock, Makefile, README.md
src/proxopt/
  world.py        vocabulary, prompts, reward modes, enumeration
  policy.py       the small network, sampling, log-probabilities, full distribution
  reference.py    synthetic corpus and training of π_ref
  exact.py        section 5.3
  prefs.py        preference pairs, Bradley–Terry reward model
  algos/          reinforce.py, ppo.py, grpo.py, dpo.py
  plots.py        shared figures
  olmo.py         readers for data/olmo/
notebooks/        00_foundations.py … 04_synthesis.py
scripts/          olmo_measure.py
data/olmo/        committed measurement files
tests/
```

- The package name `proxopt` and the use of matplotlib for figures are my calls.
- Each algorithm module exposes its loss as one readable function. Notebooks display that function's source next to the equation it implements, so what is shown is what runs.
- Unlike `gricean-devices`, notebooks here contain explanatory prose and math. Logic still lives in the package.

## 9. Testing

- Exact quantities: distributions sum to 1; the identity `J_β(π*_β) − J_β(π) = β · KL(π ‖ π*_β)` holds numerically; `π*_β` beats random perturbations of itself on `J_β`.
- PPO: at the first gradient step (θ equal to the sampling policy), with the same advantage on every token of a response and the same loss reduction, the PPO policy gradient equals REINFORCE's, for any clip range.
- GRPO: advantages are zero-mean within each group; a group with identical rewards gives zero policy gradient from the advantage term. On a tiny world, computed exactly by enumeration: the paper's KL estimator has expected value `KL(π_θ ‖ π_ref)`, and the expected gradient of that term equals the gradient of `KL(π_ref ‖ π_θ)`.
- DPO: on a tiny world with full coverage and many pairs drawn from the Bradley–Terry model with noise τ, DPO with parameter β approaches `π*_{β·τ}`. (Not "noiseless" preferences: if the better response always wins, the loss has no finite minimum.)
- On the default world with a fixed seed: PPO, DPO (against `π*_{β·τ}`) and GRPO with both switches set to the PPO-like choice reduce the gap to their target; GRPO as published raises expected reward.
- Every notebook runs end to end without error from the command line.

## 10. Left to the builder

- The synthetic corpus generator, the network architecture and size, default hyperparameters (including default τ and number of pairs: at τ = 1 with rewards in [0, 1] most pairs are near coin flips, so a smaller default τ is likely), and the number of prompts, subject to: `π_ref` has clearly sub-optimal reward and gives non-zero probability to every response; default simulations finish in about five seconds on the M3.
- Whether the KL penalty in PPO's reward is applied per token or once per response.
- Whether simulations rerun on every control change or on a run button.
- Figure styling, exercise content, sample sizes for the OLMo measurements.
- The build order, with one constraint: notebooks 00 to 03 must not depend on the OLMo measurements.

## 11. References to verify and cite

Williams 1992 (REINFORCE). Schulman et al. 2015 (TRPO). Schulman et al. 2016 (GAE). Schulman et al. 2017 (PPO). Christiano et al. 2017 (RL from human preferences). Ziegler et al. 2019 (fine-tuning language models from human preferences). Stiennon et al. 2020 (learning to summarize from human feedback). Ouyang et al. 2022 (InstructGPT). Bradley and Terry 1952. Rafailov et al. 2023 (DPO). Shao et al. 2024 (DeepSeekMath, GRPO). Guo et al. 2025 (DeepSeek-R1). Lambert et al. 2024 (Tülu 3, RL with verifiable rewards). OLMo et al. 2024 (2 OLMo 2 Furious).

## Errata (found during the build)

- Sections 6.2 and 9 state that the KL estimator's expected gradient on fixed samples is that of `KL(π_ref ‖ π_θ)` without saying "per next-token distribution". Corrected: for one state `s_t` it is `∇KL(π_ref(·|s_t) ‖ π_θ(·|s_t))`; summed over a response sampled from `π_θ`, it is `Σ_t E_{s_t∼π_θ}[∇KL(π_ref(·|s_t) ‖ π_θ(·|s_t))]` with `∇` on the conditionals only, which is the gradient of neither sequence-level KL. Notebook 02, "What GRPO aims at" (a), gives the derivation and the measured cosines.
