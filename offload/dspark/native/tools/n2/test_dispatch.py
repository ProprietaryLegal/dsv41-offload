"""Pin the integration to the lane's qualified measured profile."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from offload.dspark.kernels.fusedmoe import select_variant
from offload.dspark.native.tp2_native import ARMS


def test_published_fused_selector() -> None:
    path = Path(__file__).resolve().parents[3] / 'kernels/fusedmoe/dispatch.json'
    profile = json.loads(path.read_text())
    for stage in range(3):
        for rows in range(1, 21):
            expected = next((entry['variant'] for entry in profile['shape_overrides']
                             if (entry['stage'], entry['rows']) == (stage, rows)), profile['default_variant'])
            assert select_variant(stage, rows) == expected


@pytest.mark.parametrize('stage,rows', [(-1, 5), (3, 5), (0, 0), (1, 21)])
def test_invalid_fused_shape(stage: int, rows: int) -> None:
    with pytest.raises(ValueError):
        select_variant(stage, rows)


def test_kx2_retains_native_contract() -> None:
    arm, previous = ARMS['native-hybrid-kx2'], ARMS['native-hybrid-kx']
    arm.check()
    for key in ('experts', 'dense', 'head', 'combine', 'kv_qdq', 'wo_a_qdq', 'expanded_stages', 'markov'):
        assert getattr(arm, key) == getattr(previous, key)
    assert arm.expert_kernel == 'kx-fused'
    assert arm.dense_kernel == 'kx-selective'
