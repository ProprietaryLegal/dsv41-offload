from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from offload.prefill_cause.localize import HIGH, LOW, chunk_rows, classify, localize, main
from offload.prefill_maxlen.evidence import Json, number, record

W2 = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/PF1-W2-20261026T010854Z/traces')


def chunk(rank: int, cap: int, prefix: int, layers: int, ffn_extra: float, req: str = 'big',
          new: int = 4092) -> list[dict[str, Json]]:
    members: list[Json] = [{'request': req, 'chunk': f'{req}:{prefix}:{new}:0', 'new_tokens': new, 'prefix_tokens': prefix}]
    base: dict[str, Json] = {'rank': rank, 'max_model_len': cap, 'members': members}
    rows: list[dict[str, Json]] = [{**base, 'phase': 'chunk.begin'}]
    event = 0

    def add(phase: str, ms: float, **extra: Json) -> None:
        nonlocal event
        rows.append({**base, 'phase': phase, 'event': event, 'cuda_ms': ms, 'wall_ms': 0.1, **extra})
        event += 1
    add('chunk.total', 1000.0 + layers * ffn_extra)
    add('metadata.prepare', 1.0)
    add('collective.all_reduce', 2.0, input__shape=f'{new}x5120')
    for _ in range(layers):
        add('sparse.dispatch.flashinfer', 5.0)
        add('collective.all_reduce', 2.0, input__shape=f'{new}x5120')
        add('collective.all_reduce', 2.0 + ffn_extra, input__shape=f'{new}x5120')
    rows.append({**base, 'phase': 'chunk.end'})
    return rows


def window(layers: int = 3, extra: float = 30.0) -> list[dict[str, Json]]:
    out: list[dict[str, Json]] = []
    for cap in (LOW, HIGH):
        for rank in (0, 1):
            for prefix in (0, 4092, 8184, 12276):
                # rank 1's FFN half is slower at the high cap, so rank 0 waits at the FFN all-reduce
                out += chunk(rank, cap, prefix, layers, extra if (rank == 0 and cap == HIGH) else 0.0)
    return out


def test_ffn_wait_is_attributed_to_the_ffn_all_reduces() -> None:
    result = localize(window())
    r0, r1 = record(result['rank0']), record(result['rank1'])
    groups = record(r0['groups'])
    assert number(record(groups['ffn:4092'])['sum_of_median_delta_ms']) == pytest.approx(90.0)
    assert number(record(groups['attn:4092'])['sum_of_median_delta_ms']) == 0.0
    assert number(record(groups['embed:4092'])['all_reduces']) == 1
    assert all(number(record(g)['sum_of_median_delta_ms']) == 0.0 for g in record(r1['groups']).values())
    kinds = [record(a)['kind'] for a in cast(list[Json], r0['all_reduces'])]
    assert kinds == ['embed', 'attn', 'ffn', 'attn', 'ffn', 'attn', 'ffn']
    assert number(r0['chunk_total_median_delta_ms']) == pytest.approx(90.0)


def test_largest_request_is_selected_and_sizes_can_be_forced() -> None:
    rows = window()
    for cap in (LOW, HIGH):
        for rank in (0, 1):
            rows += chunk(rank, cap, 0, 3, 0.0, req='warm')  # a one-chunk warm-up request
    data = chunk_rows(rows)
    assert {k[2] for k in data} == {0, 4092, 8184, 12276}
    with pytest.raises(ValueError, match='need one selected request'):
        chunk_rows(rows, request_size=999)


def test_mismatched_stack_and_incomplete_events_refuse() -> None:
    rows = window()
    broken = [r for r in rows if not (r['max_model_len'] == HIGH and r.get('event') == 4
                                      and record(cast(list[Json], r['members'])[0])['prefix_tokens'] == 4092)]
    with pytest.raises(ValueError, match='sequence differs'):
        localize(broken)
    rows = window()
    for r in rows:
        if r.get('phase') == 'collective.all_reduce':
            del r['cuda_ms']
            break
    with pytest.raises(ValueError, match='incomplete all-reduce'):
        localize(rows)
    with pytest.raises(ValueError, match='>=3 matched'):
        localize([r for r in window() if record(cast(list[Json], r['members'])[0])['prefix_tokens'] in (0, 4092)])


def test_classify_marks_unexplained_reduces_other() -> None:
    rows = chunk(0, LOW, 0, 1, 0.0)
    rows.insert(-1, {**rows[-2], 'event': 99})  # a third consecutive all-reduce in the layer
    assert [k for k, _, _ in classify(rows)] == ['embed', 'attn', 'ffn', 'other']


def test_cli_refuses_existing_output(tmp_path: Path) -> None:
    trace = tmp_path / 't.jsonl'
    trace.write_text(''.join(json.dumps(r) + '\n' for r in window()))
    out = tmp_path / 'out.json'
    import sys
    argv = sys.argv
    try:
        sys.argv = ['localize', '--traces', str(trace), '--output', str(out)]
        main()
        assert record(json.loads(out.read_text()))['caps'] == [LOW, HIGH]
        with pytest.raises(FileExistsError):
            main()
    finally:
        sys.argv = argv


@pytest.mark.skipif(not W2.is_dir(), reason='PF1 window-2 traces not on this host')
def test_window2_measured_localization() -> None:
    from offload.prefill_cause.localize import read
    result = localize(read(sorted(W2.glob('*/*/*.jsonl'))))
    r0, r1 = record(result['rank0']), record(result['rank1'])
    g0 = record(r0['groups'])
    ffn = number(record(g0['ffn:4092'])['sum_of_median_delta_ms'])
    assert 590 < ffn < 630 and number(r0['chunk_total_median_delta_ms']) > 600
    assert abs(number(record(g0['ffn:128'])['sum_of_median_delta_ms'])) < 5
    assert abs(number(record(g0['attn:4092'])['sum_of_median_delta_ms'])) < 20
    assert all(abs(number(record(g)['sum_of_median_delta_ms'])) < 5 for g in record(r1['groups']).values())
    per_layer = [number(record(a)['median_delta_ms']) for a in cast(list[Json], r0['all_reduces'])
                 if record(a)['kind'] == 'ffn' and record(a)['rows'] == 4092]
    assert len(per_layer) == 20 and min(per_layer) > 25 and max(per_layer) < 37


def chunk_with_moe(rank: int, cap: int, prefix: int, layers: int, slow: float) -> list[dict[str, Json]]:
    """Window rows as the PF1-CAUSE overlay writes them (event numbers in span start order): rank 1's MoE runs
    `slow` ms longer at the high cap; rank 0's FFN all-reduce absorbs the wait."""
    new = 4092
    members: list[Json] = [{'request': 'big', 'chunk': f'big:{prefix}:{new}:0', 'new_tokens': new, 'prefix_tokens': prefix}]
    base: dict[str, Json] = {'rank': rank, 'max_model_len': cap, 'members': members}
    rows: list[dict[str, Json]] = [{**base, 'phase': 'chunk.begin'}]
    event = 0
    extra_wait = slow if (rank == 0 and cap == HIGH) else 0.0
    extra_work = slow if (rank == 1 and cap == HIGH) else 0.0

    def add(phase: str, ms: float, **more: Json) -> None:
        nonlocal event
        rows.append({**base, 'phase': phase, 'event': event, 'cuda_ms': ms, 'wall_ms': 0.1, **more})
        event += 1
    add('chunk.total', 1000.0 + layers * slow)
    add('collective.all_reduce', 2.0, input__shape=f'{new}x5120')
    for layer in range(layers):
        add('attn.total', 20.0, layer=layer, rows=new)
        add('sparse.dispatch.flashinfer', 5.0)
        add('collective.all_reduce', 2.0, input__shape=f'{new}x5120')
        add('mhc.ffn_pre', 1.0, layer=layer, rows=new)
        add('moe.ffn', 50.0 + extra_work + extra_wait + 2.0, layer=layer, rows=new)
        add('moe.routed.tf', 40.0 + extra_work, rows=new, slots=6)
        add('tf.grouped_mma.gateup', 25.0 + extra_work, rows=new, slots_total=6 * new)
        add('collective.all_reduce', 2.0 + extra_wait, input__shape=f'{new}x5120')
    rows.append({**base, 'phase': 'chunk.end'})
    return rows


def test_cause_spans_give_each_rank_its_own_moe_time() -> None:
    rows = [r for cap in (LOW, HIGH) for rank in (0, 1) for p in (0, 4092, 8184)
            for r in chunk_with_moe(rank, cap, p, 2, 31.0)]
    result = localize(rows)
    r0, r1 = record(result['rank0']), record(result['rank1'])
    assert [record(a)['kind'] for a in cast(list[Json], r0['all_reduces'])] == ['embed', 'attn', 'ffn', 'attn', 'ffn']
    l0, l1 = record(r0['cause_labels']), record(r1['cause_labels'])
    assert number(record(l1['moe.local:4092'])['median_delta_ms']) == pytest.approx(62.0)
    assert number(record(l0['moe.local:4092'])['median_delta_ms']) == pytest.approx(0.0)
    assert number(record(l1['tf.grouped_mma.gateup:4092'])['median_delta_ms']) == pytest.approx(62.0)
    assert number(record(l0['moe.ffn:4092'])['median_delta_ms']) == pytest.approx(62.0)   # wait included
    assert number(record(l1['attn.total:4092'])['median_delta_ms']) == 0.0
    broken = [r for r in rows if not (r.get('phase') == 'moe.ffn' and r['rank'] == 1 and r['max_model_len'] == LOW
                                      and record(cast(list[Json], r['members'])[0])['prefix_tokens'] == 0
                                      and r.get('layer') == 1)]
    with pytest.raises(ValueError, match='moe.ffn spans but'):
        localize(broken)
