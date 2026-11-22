"""Run records shared by every training algorithm in `proxopt.algos`.

Convention for every `train(ref, world, cfg) -> Run`: start from `ref.clone()`,
use Adam, draw all samples through one `torch.Generator` seeded with `cfg.seed`,
draw prompts uniformly, and call `Recorder.record` at step 0 and every
`cfg.eval_every` gradient steps (the last step is always recorded).
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from proxopt import exact
from proxopt.policy import Policy
from proxopt.world import World


@dataclass
class Run:
    policy: Policy | None
    config: Any
    target_beta: float
    steps: list[int]  # gradient steps so far
    feedback: list[int]  # cumulative reward calls (or labelled pairs for DPO)
    metrics: dict[str, list[float]]  # reward, kl, kl_rev, objective, gap
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        return {
            "target_beta": self.target_beta,
            "steps": self.steps,
            "feedback": self.feedback,
            "metrics": self.metrics,
            "config": repr(self.config),
        }

    @classmethod
    def from_json_dict(cls, d: dict) -> "Run":
        return cls(
            policy=None,
            config=d["config"],
            target_beta=d["target_beta"],
            steps=list(d["steps"]),
            feedback=list(d["feedback"]),
            metrics={k: list(v) for k, v in d["metrics"].items()},
            extras={},
        )


class Recorder:
    """Measures a policy against the true reward table and pi*_{target_beta}."""

    def __init__(self, world: World, ref: Policy, reward_mode: str, target_beta: float):
        self.target_beta = target_beta
        self.reward = world.reward_table(reward_mode)
        with torch.no_grad():
            self.logp_ref = ref.all_logp()
        self.steps: list[int] = []
        self.feedback: list[int] = []
        self.metrics: dict[str, list[float]] = {}

    def record(self, step: int, feedback: int, policy: Policy) -> None:
        with torch.no_grad():
            values = exact.evaluate(policy.all_logp(), self.logp_ref, self.reward, self.target_beta)
        self.steps.append(step)
        self.feedback.append(feedback)
        for key, value in values.items():
            self.metrics.setdefault(key, []).append(value)

    def finish(self, policy: Policy, config: Any, **extras) -> Run:
        return Run(
            policy=policy,
            config=config,
            target_beta=self.target_beta,
            steps=self.steps,
            feedback=self.feedback,
            metrics=self.metrics,
            extras=extras,
        )
