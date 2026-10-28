from __future__ import annotations

import copy

import pytest
import torch

from offload.dspark.d2.train.agreement import cosine, population_gates
from offload.dspark.d2.train.fast_policy import tensor_groups


def test_zero_and_missing_gradients_are_explicit() -> None:
    z = torch.zeros(4)
    assert cosine(z, z) == {"pass": True, "cosine": 1., "both_zero": True}
    assert not cosine(z, torch.ones(4))["pass"]
    assert not cosine(None, None)["pass"]
    assert not cosine(torch.ones(4), -torch.ones(4))["pass"]


def test_lora_pair_group_includes_both_factors_and_zero_only_groups_fail() -> None:
    assert tensor_groups({"x.a", "x.b", "norm"}) == {"norm": ["norm"], "x.lora": ["x.a", "x.b"]}
    row = {"position1": True, "exact": True, "kl": [0.] * 5, "loss_relative_error": 0.}
    zero = cosine(torch.zeros(4), torch.zeros(4))
    assert population_gates([row], {"zero_only": zero})["gates"]["gradients"] is False


@pytest.mark.parametrize("failure", ("drafts", "kl", "loss", "gradients"))
def test_fixed_agreement_thresholds_refuse_each_failure(failure: str) -> None:
    rows = [{"position1": True, "exact": True, "kl": [0.] * 5, "loss_relative_error": 0.} for _ in range(50)]
    gradients = {"norm": {"pass": True, "cosine": 1.}}
    assert population_gates(rows, gradients)["pass"]
    broken = copy.deepcopy(rows)
    if failure == "drafts":
        broken[0]["position1"] = False
    elif failure == "kl":
        for row in broken:
            row["kl"] = [.0011] * 5
    elif failure == "loss":
        broken[0]["loss_relative_error"] = .0051
    else:
        gradients["norm"] = {"pass": False, "cosine": .9989}
    result = population_gates(broken, gradients)
    assert result["gates"][failure] is False
    assert result["pass"] is False
