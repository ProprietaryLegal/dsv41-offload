"""S25 item 4: vocab_audit against an independent log-domain dense oracle; two-pass contract on both providers."""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from offload.dspark.d2.train.vocab_audit import audit_logits

HERE = Path(__file__).resolve().parent


def test_dense_oracle_with_hard_masks_padding_and_true_infinities() -> None:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
    out = subprocess.run([sys.executable, "-I", str(HERE / "s25_vocab_check.py"), str(HERE.parents[4]), "dense"],
                         check=True, env=env, capture_output=True, text=True).stdout
    result = ast.literal_eval(out.strip().splitlines()[-1])
    assert result["cases"] >= 150 and result["worst_rel_err"] <= 1e-11 and result["rows_with_inf_full_kl"] > 0


@pytest.mark.parametrize("who", ("teacher", "student"))
@pytest.mark.parametrize("which_chunk", (0, 3))
def test_each_chunk_is_read_exactly_twice_and_any_change_is_refused(who: str, which_chunk: int) -> None:
    z = torch.randn(2, 10, dtype=torch.float64)
    calls: Counter[tuple[str, int]] = Counter()

    def provider(name: str, mutate: bool) -> Callable[[int, int], torch.Tensor]:
        def f(lo: int, hi: int) -> torch.Tensor:
            calls[(name, lo)] += 1
            v = z[:, lo:hi].clone()
            if mutate and calls[(name, lo)] == 2 and lo == which_chunk * 3:
                v[0, 0] = torch.nextafter(v[0, 0], torch.tensor(torch.inf, dtype=torch.float64))  # one ulp
            return v
        return f

    ids = torch.tensor([[0, 4], [9, -1]])
    lab = torch.tensor([1, 2])
    audit_logits(provider("teacher", False), provider("student", False), ids, lab, lab, 10, 3)
    assert set(calls.values()) == {2} and len(calls) == 8
    calls.clear()
    with pytest.raises(ValueError, match="changed between audit passes"):
        audit_logits(provider("teacher", who == "teacher"), provider("student", who == "student"), ids, lab, lab, 10, 3)
