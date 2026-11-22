"""DPO: the KL-regularised objective fitted directly to preference pairs, no reward model."""

from dataclasses import dataclass

import torch
from torch import Tensor

from proxopt.policy import Policy
from proxopt.prefs import PrefData
from proxopt.train import Recorder, Run, scorer
from proxopt.world import World


@dataclass(frozen=True)
class DPOConfig:
    beta: float = 0.4
    steps: int = 300
    batch_size: int = 256
    lr: float = 3e-4
    seed: int = 0
    eval_every: int = 10


def dpo_loss(
    logp_w: Tensor, logp_l: Tensor, ref_logp_w: Tensor, ref_logp_l: Tensor, beta: float
) -> Tensor:
    """-mean(log sigma(beta * [(logp_w - ref_logp_w) - (logp_l - ref_logp_l)])).
    All four are sequence log-probabilities (sums over tokens), [B]."""
    margin = (logp_w - ref_logp_w) - (logp_l - ref_logp_l)
    return -torch.nn.functional.logsigmoid(beta * margin).mean()


def _seq_logp(policy: Policy, data: PrefData) -> tuple[Tensor, Tensor]:
    """Sequence log-probabilities of every winner and every loser in `data`, each [M]."""
    return (
        policy.token_logp(data.prompts, data.winners).sum(1),
        policy.token_logp(data.prompts, data.losers).sum(1),
    )


@torch.no_grad()
def _loss_on(policy: Policy, data: PrefData, ref_seq: tuple[Tensor, Tensor], beta: float) -> float:
    return dpo_loss(*_seq_logp(policy, data), *ref_seq, beta).item()


@torch.no_grad()
def pair_loss(policy: Policy, ref: Policy, data: PrefData, beta: float) -> float:
    """The DPO loss over the whole of `data`, reference scored through `scorer(ref)`."""
    return _loss_on(policy, data, _seq_logp(scorer(ref), data), beta)


@torch.no_grad()
def implicit_reward(policy: Policy, ref: Policy, beta: float) -> Tensor:
    """beta * (log pi_theta - log pi_ref) for every prompt and response, [P, N],
    centred per prompt: DPO recovers r / tau only up to a constant per prompt."""
    r = beta * (policy.all_logp() - scorer(ref).all_logp())
    return r - r.mean(1, keepdim=True)


def train(
    ref: Policy,
    world: World,
    data: PrefData,
    cfg: DPOConfig,
    tau: float,
    reward_mode: str = "graded",
    holdout: PrefData | None = None,
) -> Run:
    """Minibatches drawn with replacement from `data`; no sampling from the policy.
    DPO with parameter beta on pairs of noise tau aims at pi*_{beta * tau}, so that is
    the Recorder's target. `feedback` is the number of labelled pairs, len(data).
    At every record, the DPO loss over all of `data` goes to extras["train_loss"] and,
    if `holdout` is given, over all of `holdout` to extras["holdout_loss"]."""
    policy = ref.clone()
    with torch.no_grad():
        ref_scorer = scorer(ref)
        ref_w, ref_l = _seq_logp(ref_scorer, data)
        ref_holdout = None if holdout is None else _seq_logp(ref_scorer, holdout)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    rec = Recorder(world, ref, reward_mode, cfg.beta * tau)
    losses: dict[str, list[float]] = {"train_loss": []}
    if holdout is not None:
        losses["holdout_loss"] = []

    def record(step: int) -> None:
        rec.record(step, len(data), policy)
        losses["train_loss"].append(_loss_on(policy, data, (ref_w, ref_l), cfg.beta))
        if holdout is not None:
            losses["holdout_loss"].append(_loss_on(policy, holdout, ref_holdout, cfg.beta))

    record(0)
    for step in range(1, cfg.steps + 1):
        i = torch.randint(len(data), (min(cfg.batch_size, len(data)),), generator=gen)
        x = data.prompts[i]
        loss = dpo_loss(
            policy.token_logp(x, data.winners[i]).sum(1),
            policy.token_logp(x, data.losers[i]).sum(1),
            ref_w[i],
            ref_l[i],
            cfg.beta,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            record(step)
    return rec.finish(policy, cfg, **losses)
