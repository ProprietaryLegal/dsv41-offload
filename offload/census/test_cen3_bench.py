"""CEN3 in-image driver, CPU parts: exact smoe bytes per launch, mrow groups, metric choice, bandwidth arithmetic."""
from __future__ import annotations

import json
import random

import pytest

from offload.census import cen3_bench as B
from offload.kl1 import shapes as S
from offload.kl1 import smoe_bench as KB


def bits(seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    return [(k, k) for k in (rng.choice((2, 3, 4, 5)) for _ in range(S.E))]


@pytest.mark.parametrize('group', ['smoe-b1', 'smoe-b4'])
def test_smoe_parts_split_kl1_layer_bytes_by_launch(group: str) -> None:
    b = bits(3)
    cases = B.smoe_cases(group)
    assert len(cases) == 8
    for case in cases:
        ids = KB.routing(case, 0)
        # `unique` is per request (KL1 routing); four independent requests cover more experts together
        assert KB.distinct(ids) == case.unique if case.requests == 1 else KB.distinct(ids) >= case.unique
        gu, down = B.smoe_parts(ids, b)
        assert gu + down == pytest.approx(KB.layer_bytes(ids, b))
        assert gu > down > 0, 'gate|up streams two matrices, down one'


def test_smoe_case_axes() -> None:
    b1 = {(c.rows, c.unique) for c in B.smoe_cases('smoe-b1')}
    assert (6, 13) in b1 and (3, 24) not in b1, '3 rows cannot cover 24 experts'
    assert {(c.rows, c.unique, c.core) for c in B.smoe_cases('smoe-b4')} >= {(16, 12, 0), (20, 24, 12)}
    with pytest.raises(ValueError):
        B.smoe_cases('dense')


def test_mrow_groups_match_kl1_dense_shapes() -> None:
    widths = {g.name: g.width for g in B.DENSE_GROUPS}
    heads_local = S.HEADS * S.HEAD_DIM // S.TP
    wo_in = S.HEADS * S.HEAD_DIM // S.O_GROUPS
    groups_local = S.O_GROUPS // S.TP
    assert widths == {'wq_a+wkv': (S.H, S.Q_LORA + S.KV_DIM), 'wq_b': (S.Q_LORA, heads_local),
                      'wo_a': (groups_local * wo_in, groups_local * S.O_LORA),
                      'wo_b': (S.O_GROUPS * S.O_LORA // S.TP, S.H)}      # smoe_bench.gpu_dense widths
    wo_a = next(g for g in B.DENSE_GROUPS if g.name == 'wo_a')
    assert wo_a.offsets == [(j * wo_in, j * S.O_LORA) for j in range(groups_local)]
    qkv = B.DENSE_GROUPS[0]
    assert qkv.offsets == [(0, 0), (0, S.Q_LORA)]
    total = sum(g.bytes for g in B.DENSE_GROUPS)
    assert total == pytest.approx(sum(lin.bytes for lin in S.DECODE_DENSE) + 2 * S.H), \
        'per-linear scales: wq_a and wkv each carry their own input scale vector'
    assert B.HEAD_BYTES == S.LM_HEAD.bytes
    dense = B.dense_cases('dense')
    assert len(dense) == 12 and {c[2] for c in dense} == {'mrow'}
    assert [c[2] for c in B.dense_cases('head')] == ['mrow_head', 'mrow_head']
    assert B.dense_cases('smoke') == [('smoke-wq_b-r4', 4, 'mrow', B.DENSE_GROUPS[1].bytes)]


def test_metric_choice_prefers_dram_and_says_when_it_falls_back() -> None:
    dram = 'Metric Name,Type\ndram__bytes_read,Counter\ndram__bytes_write,Counter\ngpu__time_duration,Counter\n'
    picked = B.pick_metrics(dram)
    assert picked.metrics == ['dram__bytes_read.sum', 'dram__bytes_write.sum', 'gpu__time_duration.sum']
    assert picked.source == 'DRAM counters'
    l2 = '"lts__t_sectors_op_read","Counter"\n"lts__t_sectors_op_write","Counter"\n'
    picked = B.pick_metrics(l2)
    assert picked.read == 'lts__t_sectors_op_read.sum' and picked.time is None
    assert picked.source.startswith('L2 sector counters')
    json.dumps(picked.doc())
    with pytest.raises(ValueError, match='neither'):
        B.pick_metrics('sm__cycles_elapsed\n')


def test_bandwidth_summary() -> None:
    row = B.bw_summary('copy', 2**31, [20.0, 16.0, 18.0])
    assert row['bytes_moved'] == 2**32 and row['median_ms'] == 18.0
    assert row['gbps_max'] == pytest.approx(2**32 / 16.0 / 1e6)
    assert B.bw_bytes('read', 10) == 10 and B.bw_bytes('write', 10) == 10
    with pytest.raises(ValueError):
        B.bw_summary('read', 1, [])
    json.dumps(B.plan())
