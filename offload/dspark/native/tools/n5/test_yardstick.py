"""CPU checks for signed quantizer distances and faithful dump representation."""
from __future__ import annotations

import pytest
import torch

from offload.dspark.native.tools.n5.yardstick import bf16_steps, code_steps, upper_median


def test_fp8_signed_steps_merge_zero_and_cross_sign() -> None:
    # Keep a common scale of 1 and include E4M3's subnormal boundary.
    x = torch.zeros(32)
    y = x.clone()
    x[0] = y[0] = 448
    x[1:6] = torch.tensor([1.0, -1.0, -0.0, -2**-9, -2**-9])
    y[1:6] = torch.tensor([1.125, -1.125, 0.0, 0.0, 2**-9])
    assert code_steps(x, y)[1:6].tolist() == [1, 1, 0, 1, 2]


def test_fp8_multistep_changes_and_block_scales() -> None:
    x = torch.zeros(64)
    y = x.clone()
    x[0] = y[0] = 448
    x[32] = y[32] = 224
    x[1] = 1
    y[1] = 1.5  # four codes
    x[33] = 0.5
    y[33] = 0.5625  # one code, half-sized block scale
    steps = code_steps(x, y)
    assert int(steps[1]) == 4
    assert int(steps[33]) == 1


def test_fp8_rejects_inexact_reconstruction_and_nan() -> None:
    x = torch.zeros(32)
    x[0] = 448
    y = x.clone()
    y[1] = 0.123456
    with pytest.raises(ValueError, match="reconstruct exactly"):
        code_steps(x, y)
    y[1] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        code_steps(x, y)


def test_bf16_widened_dumps_and_signed_zero() -> None:
    x = torch.tensor([1, -1, -0.0], dtype=torch.bfloat16).float()
    y = torch.tensor([1.0078125, -1.0078125, 0.0], dtype=torch.bfloat16)
    assert bf16_steps(x, y).tolist() == [1, 1, 0]
    with pytest.raises(ValueError, match="not exactly representable"):
        bf16_steps(torch.tensor([0.123456]), torch.tensor([0.125]))


def test_v1_upper_median() -> None:
    assert upper_median([0, 1, 2, 10]) == 2
    with pytest.raises(ValueError, match="empty"):
        upper_median([])
