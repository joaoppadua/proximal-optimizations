# proxopt

Five [marimo](https://marimo.io) notebooks on three algorithms used to post-train language
models: PPO, GRPO and DPO. Each notebook gives the algorithm's definition and derivation,
the intuition, the code (the same shape as real LLM training code: per-token
log-probabilities, a value head, group sampling, summed sequence log-probabilities) and a
live simulation on a small toy world where every response can be enumerated, so the optimum
and the distance to it are computed exactly rather than estimated.

The set is a study tool for one reader who already knows RL vocabulary and wants to master
these three algorithms well enough to implement them on a real model. It makes no claim
about which algorithm is better in general, and every notebook says which of its results
carry over to LLMs and which are facts about the toy world only.

## Reading order

1. `notebooks/00_foundations.py`: the toy world, the notation, the KL-regularised objective
   `J_β`, its exact optimum `π*_β`, the gap that every later notebook is measured by, and
   REINFORCE.
2. `notebooks/01_ppo.py`: PPO in its RLHF form (per-token KL penalty, a critic, GAE, the
   clipped ratio), the reference the next two are compared with.
3. `notebooks/02_grpo.py`: GRPO as a diff against PPO, and why GRPO as published does not
   maximise `J_β`; two switches undo its two departures one at a time.
4. `notebooks/03_dpo.py`: DPO, derived step by step; what it aims at under noisy
   preferences, and what a fixed dataset of pairs costs.
5. `notebooks/04_synthesis.py`: all of them on one world against one target, a fair
   comparison of DPO with PPO on a reward model fitted to the same pairs, the history, and
   a reality check on three real checkpoints of OLMo 2 1B.

## Install and open

Prerequisites: [uv](https://docs.astral.sh/uv/) and Python 3.12 or later. The measurements
quoted in the notebooks were made on an Apple M3, and every timing given here and in the
notebooks is for that machine.

```sh
make install                                  # uv sync --extra dev
uv run marimo edit notebooks/00_foundations.py
```

Replace the file name to open another notebook (`notebooks/NN_name.py`). Each notebook also
runs as a script, `uv run python notebooks/NN_name.py`.

The notebooks train a small toy model live, on the CPU: a control change retrains in a few
seconds, and runs are cached, so returning to a setting is instant. Anything that trains
several seeds or several algorithms at once sits behind a button. Everything is seeded: the
same settings give the same numbers.

## Precomputed data (optional)

Two commands regenerate the data notebook 04 reads. Both are optional: their outputs are
committed, and notebook 04 shows a notice instead of the affected section when a file is
missing.

- `make synthesis` runs `scripts/synthesis.py`: it tunes each algorithm's learning rate on a
  small grid over three seeds, runs the comparison at the chosen rates, and writes
  `data/synthesis.json`. It trains every run of the grid and of the comparison, so it takes
  much longer than opening a notebook.
- `make olmo` installs the optional `olmo` extra and runs `scripts/olmo_measure.py all`,
  which writes `data/olmo/margins.jsonl`, `data/olmo/drift.jsonl` and `data/olmo/meta.json`.
  It needs:
  - the SFT, DPO and Instruct checkpoints of OLMo 2 1B (April 2025) already in the local
    Hugging Face cache, at the revisions pinned in `../gricean-devices/configs/models.yaml`.
    It never downloads a model (it checks the cache first and loads with
    `local_files_only=True`) and never generates text; it streams the DPO stage's preference
    dataset from the Hugging Face Hub;
  - the neighbouring repository `../gricean-devices`, whose response files
    (`data/responses/`) supply the texts for the drift measurement; it is only read;
  - about 20 minutes and several gigabytes of free memory on an Apple M3 (two half-precision
    checkpoints are loaded together for the drift measurement).

  It leaves a git-ignored working folder, `data/olmo/.work/`, with partial results (a rerun
  resumes from them) and the token ids of the selected dataset conversations, which decode
  back to the dataset's text. The committed files hold numbers and identifiers only.

  The committed OLMo measurements were computed from three checkpoints on the Hugging Face
  Hub, `allenai/OLMo-2-0425-1B-SFT`, `allenai/OLMo-2-0425-1B-DPO` and
  `allenai/OLMo-2-0425-1B-Instruct`, and from the preference dataset
  `allenai/olmo-2-0425-1b-preference-mix`, at the revisions `data/olmo/meta.json` records. The files under `data/olmo/` hold numeric measurements
  (log-probabilities, token counts, KL values), the settings of the measurement, and
  identifiers only (of the checkpoints, the dataset, and the neighbouring repository's
  response items); they contain no model weights and no text from the dataset or from the
  models.

## Measurement scripts

Each prints numbers a notebook quotes, with the definitions the notebook states; run any of
them with `uv run python scripts/<name>.py`. Times are on an Apple M3.

- `scripts/measure_nb00.py`: notebook 00, item 3 (the same feedback with smaller batches:
  gaps on the shared grid, the wobble measure, the seed-to-seed range); about 10 s.
- `scripts/measure_nb01.py`: notebook 01, item 1 (the ratios over every gradient step, read
  in a logging copy of the training loop, the snapshot, the gaps); about 1 minute.
- `scripts/measure_nb02.py`: notebook 02, "What GRPO aims at" (a) to (c) and the paragraphs
  on what the collapse depends on, including the run at a lower learning rate; about
  seven minutes.
- `scripts/dpo_label_diagnostic.py`: notebook 03, items 3 and 5: it reruns DPO on the same
  pairs with each pair's drawn label replaced by its true preference probability, or by a
  noisy label with a finite best margin; about two minutes.

## Tests and lint

```sh
make test       # the fast suite
make test-all   # everything, including the slow tests that train default configs and run
                # every notebook as a script
make lint       # ruff
```

## Design and checks

- `docs/superpowers/specs/2026-11-21-policy-optimization-notebooks-design.md`: the design,
  with its errata at the end.
- `docs/equation-check.md`: every equation the notebooks use, checked against its source
  paper, and the facts recorded from the papers.
- `docs/olmo-facts.md`: how OLMo 2 1B was post-trained, each fact quoted from its source.
