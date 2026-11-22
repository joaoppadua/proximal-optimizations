"""Measurements on the real OLMo 2 1B checkpoints (SFT, DPO, Instruct), and loaders for the
small result files `scripts/olmo_measure.py` writes under `data/olmo/`.

Needs only torch and the standard library: the notebooks import this module without the
`olmo` extra, and only the script loads models."""

import json
import os
from pathlib import Path

import torch

MODELS = ("sft", "dpo", "instruct")
STEPS = ("sft->dpo", "dpo->instruct")
_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "olmo"


def response_logprob(logits: torch.Tensor, input_ids: torch.Tensor, response_start: int) -> tuple[float, int]:
    """Summed log-probability of `input_ids[response_start:]` and their count.

    `logits[i - 1]` predicts token `i`, so the response is scored by rows
    `response_start - 1` to `L - 2`."""
    logp = logits[response_start - 1 : -1].float().log_softmax(-1)
    targets = input_ids[response_start:].to(logp.device)
    total = logp.gather(1, targets.unsqueeze(1)).sum()
    return float(total), int(targets.numel())


def next_token_kl(logits_a: torch.Tensor, logits_b: torch.Tensor, response_start: int) -> tuple[float, float]:
    """Mean of KL(a ‖ b) and of KL(b ‖ a) over the positions that predict response tokens,
    computed exactly over the vocabulary in float32."""
    la = logits_a[response_start - 1 : -1].float().log_softmax(-1)
    lb = logits_b[response_start - 1 : -1].float().log_softmax(-1).to(la.device)
    ab = (la.exp() * (la - lb)).sum(-1).mean()
    ba = (lb.exp() * (lb - la)).sum(-1).mean()
    return float(ab), float(ba)


def implicit_margins(rows: list[dict], model: str, per_token: bool) -> list[float]:
    """Per pair, the implicit reward margin of `model` against `sft`:
    `(lp_w_m − lp_w_sft) − (lp_l_m − lp_l_sft)`, with each log-probability divided by its
    token count first when `per_token` (the quantity length-normalised DPO trains on)."""
    out = []
    for r in rows:
        nw, nl = (r["n_w"], r["n_l"]) if per_token else (1, 1)
        out.append((r[f"lp_w_{model}"] - r["lp_w_sft"]) / nw - (r[f"lp_l_{model}"] - r["lp_l_sft"]) / nl)
    return out


def summarize_margins(rows: list[dict]) -> dict:
    """Implicit reward margin of `dpo` and of `instruct`, each against `sft`:
    `(lp_w_m − lp_w_sft) − (lp_l_m − lp_l_sft)`, summed (`sum`) and with each log-probability
    divided by its token count first (`mean`, the quantity length-normalised DPO trains on)."""
    out: dict = {"n": len(rows)}
    for model in ("dpo", "instruct"):
        out[model] = {}
        for version, per_token in (("sum", False), ("mean", True)):
            m = implicit_margins(rows, model, per_token)
            out[model][version] = {
                "mean_margin": sum(m) / len(m) if m else float("nan"),
                "share_positive": sum(x > 0 for x in m) / len(m) if m else float("nan"),
            }
    return out


def _root(root: Path | None) -> Path:
    if root is not None:
        return Path(root)
    env = os.environ.get("PROXOPT_OLMO_DIR")
    return Path(env) if env else _DEFAULT_ROOT


def _read_jsonl(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_margins(root: Path | None = None) -> list[dict] | None:
    return _read_jsonl(_root(root) / "margins.jsonl")


def load_drift(root: Path | None = None) -> list[dict] | None:
    return _read_jsonl(_root(root) / "drift.jsonl")


def load_meta(root: Path | None = None) -> dict | None:
    path = _root(root) / "meta.json"
    return json.loads(path.read_text()) if path.exists() else None
