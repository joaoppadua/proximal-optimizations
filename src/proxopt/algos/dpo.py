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
) -> Run:
    """Minibatches drawn with replacement from `data`; no sampling from the policy.
    DPO with parameter beta on pairs of noise tau aims at pi*_{beta * tau}, so that is
    the Recorder's target. `feedback` is the number of labelled pairs, len(data)."""
    policy = ref.clone()
    with torch.no_grad():
        ref_scorer = scorer(ref)
        ref_w = ref_scorer.token_logp(data.prompts, data.winners).sum(1)
        ref_l = ref_scorer.token_logp(data.prompts, data.losers).sum(1)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    rec = Recorder(world, ref, reward_mode, cfg.beta * tau)
    rec.record(0, len(data), policy)
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
            rec.record(step, len(data), policy)
    return rec.finish(policy, cfg)
