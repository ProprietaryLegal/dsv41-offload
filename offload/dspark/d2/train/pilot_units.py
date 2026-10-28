"""Matched pilot work units, including masked tails and end-to-end estimates."""
from __future__ import annotations

import math
from typing import Protocol, TypedDict

from .loop import BatchPlan, Run, lr_multiplier

QDQ_SEMANTICS = "official act_qdq before dense/expert, output dtype rounding, identity STE backward; additive FP32 LoRA before output rounding; raw uncached taps"


class TokenCounts(Protocol):
    def __len__(self) -> int: ...
    def supervised_tokens(self, index: int) -> int: ...


class Units(TypedDict):
    blocks: int
    valid_supervised_tokens: int
    optimizer_updates: int
    accumulation: int
    batch_ceiling: int
    lr_schedule: str
    learning_rates: list[float]
    qdq_semantics: str


def work_units(data: TokenCounts, plan: BatchPlan, run: Run, step: int | None = None) -> Units:
    stop = len(plan.batches) if step is None else step
    if not 0 < stop <= len(plan.batches):
        raise ValueError("invalid matched-unit checkpoint")
    counts = [data.supervised_tokens(i) for batch in plan.batches[:stop] for i in batch]
    if any(type(n) is not int or not 1 <= n <= 5 for n in counts):
        raise ValueError("invalid supervised position count")
    return {"blocks": len(counts), "valid_supervised_tokens": sum(counts), "optimizer_updates": stop,
            "accumulation": 1, "batch_ceiling": run.batch_size,
            "lr_schedule": ("linear-warmup-10pct-then-cosine; " if run.warmup_fraction else "")+"cosine-per-optimizer-update; initial lr then scheduler.step after update; total="+str(len(plan.batches)),
            "learning_rates": [run.lr*lr_multiplier(run, s, len(plan.batches)) for s in range(stop)],
            "qdq_semantics": QDQ_SEMANTICS}


def require_matched(arms: dict[str, dict[str, Units]]) -> None:
    if set(arms) != {"REF", "FAST", "FAST-B"} or not arms["REF"]:
        raise ValueError("matched units require all arms/checkpoints")
    if arms["REF"] != arms["FAST"] or arms["REF"] != arms["FAST-B"]:
        raise ValueError("pilot blocks/tokens/updates/accumulation/LR/QDQ units differ")


class Estimate(TypedDict):
    label: str
    training_gpu_hours: float
    evaluation_gpu_hours: float
    replay_gpu_hours: float
    serving_gpu_hours: float
    total_gpu_hours: float
    ideal_two_solo_wall_hours: float
    training_blocks: int
    optimizer_updates: int
    valid_supervised_tokens: int


def estimate(units: Units, blocks_per_second: dict[str, float], evaluation_rounds: int,
             eval_seconds_per_round: float, replay_gpu_seconds: float, serving_gpu_seconds: float) -> Estimate:
    costs = [*blocks_per_second.values(), eval_seconds_per_round, replay_gpu_seconds, serving_gpu_seconds]
    if (set(blocks_per_second) != {"REF", "FAST", "FAST-B"} or any(not math.isfinite(x) or x <= 0 for x in costs)
            or evaluation_rounds < 1):
        raise ValueError("positive measured/declared costs required including replay and serving qualification")
    training = sum(units["blocks"]/rate for rate in blocks_per_second.values())/3600
    evaluation = 13*evaluation_rounds*eval_seconds_per_round/3600
    replay, serving = replay_gpu_seconds/3600, serving_gpu_seconds/3600
    total = training+evaluation+replay+serving
    return {"label": "ESTIMATED; additive costs, perfect two-solo balance lower bound; excludes load/I/O/queue overhead",
            "training_gpu_hours": training, "evaluation_gpu_hours": evaluation, "replay_gpu_hours": replay,
            "serving_gpu_hours": serving, "total_gpu_hours": total, "ideal_two_solo_wall_hours": total/2,
            "training_blocks": 3*units["blocks"], "optimizer_updates": 3*units["optimizer_updates"],
            "valid_supervised_tokens": 3*units["valid_supervised_tokens"]}
