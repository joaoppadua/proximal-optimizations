"""Measure the real OLMo 2 1B checkpoints (SFT, DPO, Instruct) with forward passes only.

    uv run python scripts/olmo_measure.py margins [--n 300]
    uv run python scripts/olmo_measure.py drift [--n 120]
    uv run python scripts/olmo_measure.py all

`margins`: log-probabilities of the chosen and rejected responses of `--n` single-turn pairs
streamed from the DPO stage's preference data, under each checkpoint in turn.
`drift`: exact next-token KL, both directions, between the two checkpoints of each step
(SFT -> DPO, DPO -> Instruct), along the later checkpoint's own templated responses from
`../gricean-devices/data/responses/`.

Never downloads a model (`local_files_only=True`, and a cache check before any load) and never
generates. Partial results go to `<out>/.work/` (git-ignored, not committed), so a rerun
resumes. `.work/pairs_n<N>.json` holds the token ids of the selected dataset conversations, which
decode back to the dataset's text; it stays on disk only. The committed files hold numbers and
identifiers only, no text."""

import argparse
import datetime
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

from proxopt import olmo

ROOT = Path(__file__).resolve().parents[1]
SISTER = ROOT.parent / "gricean-devices"
DATASET = "allenai/olmo-2-0425-1b-preference-mix"
SHUFFLE_SEED, SHUFFLE_BUFFER, MAX_TOKENS = 0, 10_000, 512
DRIFT_SEED = 0
REGISTRY = {"sft": "olmo_sft", "dpo": "olmo_dpo", "instruct": "olmo_instruct"}
STEP_MODELS = {"sft->dpo": ("sft", "dpo"), "dpo->instruct": ("dpo", "instruct")}
MIN_FREE_GB_FOR_TWO = 6.5          # two fp16 checkpoints (~3 GB each) plus activations


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- models

def specs() -> dict[str, tuple[str, str]]:
    models = yaml.safe_load((SISTER / "configs" / "models.yaml").read_text())["models"]
    return {m: (models[name]["repo"], str(models[name]["revision"])) for m, name in REGISTRY.items()}


def require_cached(repo: str, revision: str) -> bool:
    """Exit unless the snapshot is cached; return whether its weights are safetensors
    (the SFT and DPO snapshots at their pinned revisions hold `pytorch_model.bin`)."""
    from huggingface_hub import constants
    snap = Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    safetensors = bool(list(snap.glob("*.safetensors")))
    if not (snap / "config.json").exists() or not (safetensors or list(snap.glob("pytorch_model*.bin"))):
        sys.exit(f"{repo}@{revision} is not in the local Hugging Face cache ({snap}); refusing to download it.")
    return safetensors


def device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.backends.mps.is_available():          # as the sister repo: fp16 off the CPU
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def load_tokenizer(m: str):
    from transformers import AutoTokenizer
    repo, rev = specs()[m]
    require_cached(repo, rev)
    return AutoTokenizer.from_pretrained(repo, revision=rev, local_files_only=True)


def load_model(m: str):
    from transformers import AutoModelForCausalLM
    repo, rev = specs()[m]
    safetensors = require_cached(repo, rev)
    device, dtype = device_and_dtype()
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=rev, local_files_only=True, use_safetensors=safetensors,
        dtype=dtype, device_map={"": device}).eval()
    log(f"loaded {m} ({repo}@{rev[:8]}) on {device} {dtype} in {time.time() - t0:.1f}s")
    return model


def in_child(kind: str, what: str, n: int, out: Path) -> None:
    """Run one model-holding job in a fresh process. Deleting a model, emptying the MPS cache
    and collecting garbage did not hand its memory back to the system inside one process
    (measured: 8.4 GB available before the first drift step, 2.4 GB after it); a process exit does."""
    subprocess.run([sys.executable, __file__, "_child", kind, what, "--n", str(n), "--out", str(out)], check=True)


def available_gb() -> float:
    """Free + inactive + speculative + purgeable pages (macOS `vm_stat`)."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    pages = {k: int(v) for k, v in re.findall(r"Pages (\w+):\s+(\d+)", out)}
    return sum(pages.get(k, 0) for k in ("free", "inactive", "speculative", "purgeable")) * page / 1e9


@torch.inference_mode()
def logits_of(model, ids: list[int]) -> torch.Tensor:
    x = torch.tensor([ids], device=model.device)
    return model(x).logits[0]


# ---------------------------------------------------------------- files

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def update_meta(out: Path, section: str, values: dict) -> None:
    import transformers
    path = out / "meta.json"
    meta = json.loads(path.read_text()) if path.exists() else {}
    device, dtype = device_and_dtype()
    meta.update({
        "date": datetime.date.today().isoformat(),
        "revisions": {m: {"repo": r, "revision": v} for m, (r, v) in specs().items()},
        "device": device, "dtype": str(dtype).removeprefix("torch."),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "generation": yaml.safe_load((SISTER / "configs" / "generation.yaml").read_text()),
    })
    meta[section] = values
    path.write_text(json.dumps(meta, indent=2) + "\n")


# ---------------------------------------------------------------- margins

def single_turn(row: dict) -> bool:
    w, l = row["chosen"], row["rejected"]
    return (len(w) == 2 and len(l) == 2 and [m["role"] for m in w] == ["user", "assistant"]
            and [m["role"] for m in l] == ["user", "assistant"] and w[0]["content"] == l[0]["content"])


def select_pairs(n: int, tok) -> tuple[list[dict], dict]:
    """Token ids of the first `n` eligible pairs of the shuffled stream, and how they were found.

    The prompt is the user turn rendered with the generation prompt; the response is the rest of
    the full conversation's rendering (the assistant text and its end-of-sequence token), as
    the DPO stage's loss mask did."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision = HfApi().dataset_info(DATASET).sha
    ds = load_dataset(DATASET, split="train", streaming=True, revision=revision)
    ds = ds.shuffle(seed=SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER)
    stats = {"scanned": 0, "not_single_turn": 0, "too_long": 0, "prefix_mismatch": 0, "identical_responses": 0}
    pairs = []
    for row in ds:
        stats["scanned"] += 1
        if not single_turn(row):
            stats["not_single_turn"] += 1
            continue
        prompt = tok.apply_chat_template(row["chosen"][:1], tokenize=False, add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        full = {}
        for side in ("chosen", "rejected"):
            text = tok.apply_chat_template(row[side], tokenize=False)
            full[side] = tok(text, add_special_tokens=False).input_ids
        if max(len(v) for v in full.values()) > MAX_TOKENS:
            stats["too_long"] += 1
            continue
        if any(v[: len(p_ids)] != p_ids for v in full.values()):
            stats["prefix_mismatch"] += 1          # prompt tokens would straddle the boundary
            continue
        if row["chosen"][1]["content"] == row["rejected"][1]["content"]:
            stats["identical_responses"] += 1
        pairs.append({"ids_w": full["chosen"], "ids_l": full["rejected"], "start": len(p_ids)})
        if len(pairs) == n:
            break
    return pairs, {"dataset": DATASET, "dataset_revision": revision, **stats}


def score_margins(m: str, n: int, out: Path) -> None:
    """Score the cached pairs under checkpoint `m`, appending to (and resuming) its part file."""
    work = out / ".work"
    pairs = json.loads((work / f"pairs_n{n}.json").read_text())["pairs"]
    part = work / f"margins_n{n}_{m}.jsonl"
    done = read_jsonl(part)
    model = load_model(m)
    tm = time.time()
    with part.open("a") as f:
        for i in range(len(done), len(pairs)):
            p = pairs[i]
            lp_w, n_w = olmo.response_logprob(logits_of(model, p["ids_w"]), torch.tensor(p["ids_w"]), p["start"])
            lp_l, n_l = olmo.response_logprob(logits_of(model, p["ids_l"]), torch.tensor(p["ids_l"]), p["start"])
            f.write(json.dumps({"pair": i, "n_w": n_w, "n_l": n_l, "lp_w": lp_w, "lp_l": lp_l}) + "\n")
            f.flush()
            if i < 5 or (i + 1) % 50 == 0:
                log(f"{m}: pair {i + 1}/{len(pairs)}, {(time.time() - tm) / (i + 1 - len(done)):.2f}s/pair")


def cmd_margins(n: int, out: Path) -> None:
    t0 = time.time()
    work = out / ".work"
    work.mkdir(parents=True, exist_ok=True)
    cache = work / f"pairs_n{n}.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        pairs, stats = saved["pairs"], saved["stats"]
        log(f"reusing {len(pairs)} selected pairs from {cache}")
    else:
        tokenizers = {m: load_tokenizer(m) for m in olmo.MODELS}
        ref = tokenizers["sft"]
        same = all(t.chat_template == ref.chat_template and t.get_vocab() == ref.get_vocab()
                   for t in tokenizers.values())
        if not same:
            sys.exit("the three checkpoints' tokenizers or chat templates differ; per-model rendering is not implemented")
        log(f"streaming {DATASET} (shuffle seed {SHUFFLE_SEED}, buffer {SHUFFLE_BUFFER})")
        pairs, stats = select_pairs(n, ref)
        stats["templates_identical"] = same
        cache.write_text(json.dumps({"pairs": pairs, "stats": stats}))
        log(f"selected {len(pairs)} pairs: {stats}")
    if len(pairs) < n:
        log(f"only {len(pairs)} eligible pairs found")
    for m in olmo.MODELS:
        if len(read_jsonl(work / f"margins_n{n}_{m}.jsonl")) == len(pairs):
            log(f"{m}: already scored")
            continue
        in_child("margins", m, n, out)
    parts = {m: read_jsonl(work / f"margins_n{n}_{m}.jsonl") for m in olmo.MODELS}
    rows = []
    for i in range(len(pairs)):
        r = {"pair": i, "n_w": parts["sft"][i]["n_w"], "n_l": parts["sft"][i]["n_l"]}
        for m in olmo.MODELS:
            r[f"lp_w_{m}"], r[f"lp_l_{m}"] = parts[m][i]["lp_w"], parts[m][i]["lp_l"]
        rows.append(r)
    write_jsonl(out / "margins.jsonl", rows)
    update_meta(out, "margins", {"n_requested": n, "n": len(rows), "shuffle_seed": SHUFFLE_SEED,
                                 "shuffle_buffer": SHUFFLE_BUFFER, "max_tokens": MAX_TOKENS,
                                 "response_tokens": "assistant text and its end-of-sequence token",
                                 "wall_time_s": round(time.time() - t0, 1), **stats})
    log(f"margins: {len(rows)} rows, {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- drift

def drift_texts(later: str, n: int) -> list[dict]:
    """Up to `n` non-empty English templated responses of the later checkpoint, chosen with a
    fixed seed and kept in file order."""
    path = SISTER / "data" / "responses" / f"{REGISTRY[later]}__tmpl.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    ok = [i for i, r in enumerate(rows) if r["language"] == "en" and r["template"] and r["response"].strip()]
    keep = sorted(random.Random(DRIFT_SEED).sample(ok, min(n, len(ok))))
    return [rows[i] for i in keep]


def score_drift(step: str, n: int, out: Path) -> None:
    """Next-token KL along the step's texts, both checkpoints loaded together, appending to (and
    resuming) the step's part file."""
    old, new = STEP_MODELS[step]
    tok = load_tokenizer("sft")
    texts = drift_texts(new, n)
    part = out / ".work" / f"drift_n{n}_{step.replace('->', '_to_')}.jsonl"
    done = read_jsonl(part)
    m_old, m_new = load_model(old), load_model(new)
    tm = time.time()
    with part.open("a") as f:
        for i in range(len(done), len(texts)):
            r = texts[i]
            # As the sister repo built it: the rendered prompt with special tokens, then the
            # response tokenised on its own (its continuation_logprob does the same).
            p_ids = tok(r["prompt_text"], add_special_tokens=True).input_ids
            c_ids = tok(r["response"], add_special_tokens=False).input_ids
            ids, start = p_ids + c_ids, len(p_ids)
            lo, ln = logits_of(m_old, ids), logits_of(m_new, ids)
            kl_no, kl_on = olmo.next_token_kl(ln, lo, start)
            row = {"step": step, "item_id": r["item_id"], "version": r["version"], "mode": r["mode"],
                   "seed": r["seed"], "n_tokens": len(c_ids), "kl_new_old": kl_no, "kl_old_new": kl_on,
                   "decode_ok": tok.decode(c_ids, skip_special_tokens=True) == r["response"]}
            if r["mode"] == "greedy":       # the later checkpoint's own greedy text: argmax must reproduce it
                am = ln[start - 1 : -1].argmax(-1).cpu()
                row["greedy_agree"] = int((am == torch.tensor(c_ids)).sum())
                row["greedy_first"] = bool(am[0] == c_ids[0])
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i < 5 or (i + 1) % 20 == 0:
                log(f"{step}: text {i + 1}/{len(texts)}, {(time.time() - tm) / (i + 1 - len(done)):.2f}s/text")


def cmd_drift(n: int, out: Path) -> None:
    t0 = time.time()
    work = out / ".work"
    work.mkdir(parents=True, exist_ok=True)
    info, all_rows = {}, []
    for step, (_, new) in STEP_MODELS.items():
        ts = time.time()
        texts = drift_texts(new, n)
        part = work / f"drift_n{n}_{step.replace('->', '_to_')}.jsonl"
        done = read_jsonl(part)
        checks = {"decode_mismatch": 0, "greedy_rows": 0, "greedy_first_token_agree": 0,
                  "greedy_positions": 0, "greedy_argmax_agree": 0}
        if len(done) < len(texts):
            gb = available_gb()
            log(f"{step}: {gb:.1f} GB available before loading two checkpoints")
            if gb < MIN_FREE_GB_FOR_TWO:
                sys.exit(f"only {gb:.1f} GB available, need {MIN_FREE_GB_FOR_TWO}; stopping rather than swapping")
            in_child("drift", step, n, out)
        rows = read_jsonl(part)
        for row in rows:
            checks["decode_mismatch"] += not row.pop("decode_ok")
            if "greedy_agree" in row:
                checks["greedy_rows"] += 1
                checks["greedy_first_token_agree"] += row.pop("greedy_first")
                checks["greedy_positions"] += row["n_tokens"]
                checks["greedy_argmax_agree"] += row.pop("greedy_agree")
        info[step] = {"n": len(rows), "source": f"{REGISTRY[new]}__tmpl.jsonl", "wall_time_s": round(time.time() - ts, 1),
                      **checks}
        all_rows += rows
    write_jsonl(out / "drift.jsonl", all_rows)
    update_meta(out, "drift", {"n_requested": n, "selection_seed": DRIFT_SEED,
                               "texts": "English, templated, non-empty responses, all modes",
                               "wall_time_s": round(time.time() - t0, 1), "steps": info})
    log(f"drift: {len(all_rows)} rows, {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- summary

def summary(out: Path) -> None:
    margins, drift = olmo.load_margins(out), olmo.load_drift(out)
    if margins:
        print(json.dumps(olmo.summarize_margins(margins), indent=2))
    for step in olmo.STEPS:
        rows = [r for r in drift or [] if r["step"] == step]
        if rows:
            kn = sum(r["kl_new_old"] for r in rows) / len(rows)
            ko = sum(r["kl_old_new"] for r in rows) / len(rows)
            print(f"{step}: n={len(rows)}  mean KL(new||old)={kn:.4f}  mean KL(old||new)={ko:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["margins", "drift", "all", "_child"])
    ap.add_argument("child", nargs="*", help=argparse.SUPPRESS)
    ap.add_argument("--n", type=int, default=None, help="pairs (margins, default 300) or texts per step (drift, default 120)")
    ap.add_argument("--n-drift", type=int, default=120, help="texts per step for `all`")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "olmo")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.command == "_child":
        kind, what = a.child
        (score_margins if kind == "margins" else score_drift)(what, a.n, a.out)
        return
    if a.command in ("margins", "all"):
        cmd_margins(a.n or 300, a.out)
    if a.command in ("drift", "all"):
        cmd_drift(a.n if a.command == "drift" and a.n else a.n_drift, a.out)
    summary(a.out)


if __name__ == "__main__":
    main()
