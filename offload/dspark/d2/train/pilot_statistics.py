"""Paired family bootstrap of G2 accepted-prefix estimates; bonus excluded.

Whole requests stay intact. Each sampled family occurrence resamples requests
within that family; the identical draw is applied to both arms. Round totals
retain the pilot's round-weighted estimand rather than weighting short requests
as heavily as long requests. IDs are hashes; no tokens or generated text here.
"""
from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
import json
import math
import random
import statistics
from typing import TypedDict, cast
from collections.abc import Iterable

BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 20261024
DEV_SPLITS = {"acap": "dev_acap", "heldout_legal": "dev_legal"}
LIVE_DEV_SPLITS = {"dev": "dev", "heldout_legal": "dev_legal"}


def split_roles(keys: Iterable[str]) -> dict[str, str]:
    if set(keys) == set(LIVE_DEV_SPLITS):
        return LIVE_DEV_SPLITS
    if set(keys) == set(DEV_SPLITS):
        return DEV_SPLITS
    raise ValueError("missing/mismatched pilot selection splits")


CLUSTER_KEY = "within-split union-find source_docs components; empty source_docs clustered by prompt_sha256"
FEW_CLUSTERS = "Fewer than 5 clusters: this interval cannot establish between-family uncertainty."


class RequestTotal(TypedDict):
    request_sha256: str
    family: str
    task_label: str
    rounds: int
    accepted_sum: float


class Interval(TypedDict):
    difference: float
    lower: float
    upper: float
    families: int
    requests: int
    resamples: int
    seed: int


def paired_interval(left: list[RequestTotal], right: list[RequestTotal]) -> Interval:
    # Content-keyed cache only; mutated evidence always changes the key. Return
    # a copy so receipt edits cannot alter the cached statistical computation.
    return cast(Interval, dict(_cached_interval(json.dumps([left, right], sort_keys=True, allow_nan=False))))


@lru_cache(maxsize=512)
def _cached_interval(serialized: str) -> Interval:
    left, right = cast(tuple[list[RequestTotal], list[RequestTotal]], json.loads(serialized))
    def indexed(rows: list[RequestTotal]) -> dict[str, RequestTotal]:
        result = {r["request_sha256"]: r for r in rows}
        if not rows or len(rows) != len(result):
            raise ValueError("missing/duplicate pilot request aggregates")
        for row in rows:
            if (not row["family"] or type(row["rounds"]) is not int or row["rounds"] < 1
                    or not math.isfinite(row["accepted_sum"])
                    or not 0 <= row["accepted_sum"] <= 5*row["rounds"]):
                raise ValueError("invalid pilot request aggregate")
        return result
    a, b = indexed(left), indexed(right)
    if a.keys() != b.keys():
        raise ValueError("unpaired pilot request population")
    families: dict[str, list[str]] = {}
    for key in sorted(a):
        if (a[key]["family"], a[key]["task_label"], a[key]["rounds"]) != (b[key]["family"], b[key]["task_label"], b[key]["rounds"]):
            raise ValueError("unpaired pilot family/round coverage")
        families.setdefault(a[key]["family"], []).append(key)
    names = sorted(families)
    def estimate(keys: list[str]) -> float:
        return float(sum((Decimal(str(a[k]["accepted_sum"]))-Decimal(str(b[k]["accepted_sum"])) for k in keys), Decimal(0))/sum(a[k]["rounds"] for k in keys))
    rng = random.Random(BOOTSTRAP_SEED)
    values = []
    for _ in range(BOOTSTRAPS):
        keys = []
        for family in rng.choices(names, k=len(names)):
            group = families[family]
            keys.extend(rng.choices(group, k=len(group)))
        values.append(estimate(keys))
    # Inclusive linear percentile, pinned independent of numpy versions.
    quantiles = statistics.quantiles(sorted(values), n=100, method="inclusive")
    return {"difference": estimate(sorted(a)), "lower": quantiles[4], "upper": quantiles[94],
            "families": len(families), "requests": len(a), "resamples": BOOTSTRAPS, "seed": BOOTSTRAP_SEED}
