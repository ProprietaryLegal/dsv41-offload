from __future__ import annotations

import json
import time
import subprocess
from pathlib import Path
from typing import cast

import pytest

from offload.prefill_maxlen.evidence import Json, record
from offload.prefill_trace.fit_cost import fit, match_by_size
from offload.prefill_trace.window_requests import HEADER, cache_salt, exact_prompt, stream_summary


def test_prompt_family_exact_prefix_and_refusal() -> None:
    materials = HEADER + 'public corpus ' * 1000
    assert exact_prompt(materials, 800, len) == materials[:800]
    assert exact_prompt(materials, 1000, len).startswith(exact_prompt(materials, 800, len))
    with pytest.raises(ValueError, match='too short'):
        exact_prompt(materials, 1000000, len)


def test_sse_summary_never_retains_text() -> None:
    events = [b'data: ' + json.dumps(obj).encode() for obj in [
        {'id': 'chatcmpl-id', 'choices': [{'delta': {'reasoning_content': 'private-reason'}}]},
        {'id': 'chatcmpl-id', 'choices': [{'delta': {'content': 'private-answer'}, 'finish_reason': 'length'}],
         'usage': {'prompt_tokens': 500000, 'completion_tokens': 100,
                   'completion_tokens_details': {'reasoning_tokens': 90}}}]] + [b'data: [DONE]']
    result = stream_summary(events, time.monotonic())
    assert result['reasoning_tokens'] == 90 and result['answer_and_control_tokens'] == 10
    assert 'private' not in str(result)
    with pytest.raises(RuntimeError, match='incomplete'):
        stream_summary(events[:-1], time.monotonic())


def fake(chunk: int = 4096, remainder: int = 0) -> tuple[list[dict[str, Json]], list[dict[str, Json]]]:
    traces: list[dict[str, Json]] = []
    requests: list[dict[str, Json]] = []
    for cap in (327680, 524288):
        rid = str(cap)
        requests.append({'kind': 'cold', 'trace_request': rid})
        sizes = [(prefix, chunk) for prefix in range(0, 4 * chunk, chunk)] + ([(4 * chunk, remainder)] if remainder else [])
        for prefix, new in sizes:
            for rank in (0, 1):
                row: dict[str, Json] = {'rank': rank, 'max_model_len': cap,
                    'members': [{'request': rid, 'chunk': f'{rid}:{prefix}', 'new_tokens': new, 'prefix_tokens': prefix}]}
                traces.append({**row, 'phase': 'chunk.begin'})
                cost = 100 + rank + 2 * prefix / 100000 + (10 + 3 * prefix / 100000 if cap == 524288 else 0)
                traces.append({**row, 'phase': 'chunk.total', 'cuda_ms': cost, 'wall_ms': cost + 1})
    return traces, requests


def test_fit_separates_prefix_and_cap_and_rejects_missing_rank() -> None:
    traces, requests = fake()
    result = fit(traces, requests)
    ranks = cast(list[Json], result['ranks'])
    models = record(record(ranks[0])['models_common_support'])
    cuda = record(models['cuda_ms'])
    assert cuda['high_cap_extra_ms'] == pytest.approx(10)
    assert cuda['prefix_ms_per_100K'] == pytest.approx(2)
    assert cuda['high_cap_prefix_extra_ms_per_100K'] == pytest.approx(3)
    with pytest.raises(ValueError, match='both rank'):
        fit([r for r in traces if r['rank'] == 0], requests)
    with pytest.raises(ValueError, match='missing begin'):
        fit(traces[:-1], requests)


def test_span_deltas_name_the_label_that_carries_the_cap_cost() -> None:
    traces, requests = fake()
    extra: list[dict[str, Json]] = []
    for row in traces:
        if row['phase'] != 'chunk.begin':
            continue
        high = row['max_model_len'] == 524288
        for layer in (0, 1):    # per-layer rows are summed within the chunk
            extra.append({**row, 'phase': 'indexer.mqa', 'cuda_ms': 20.0 + (4 if high else 0), 'layer': layer})
            extra.append({**row, 'phase': 'collective.all_reduce', 'cuda_ms': 5.0})
        extra.append({**row, 'phase': 'indexer.plan', 'workspace_exceeded': high, 'workspace_group_split': False,
                      'query_subchunked': False})
        extra.append({**row, 'phase': 'sparse.dispatch.triton' if high else 'sparse.dispatch.flashinfer', 'cuda_ms': 1.0})
    spans = record(fit(traces + extra, requests)['spans'])
    rank0 = record(spans['rank0'])
    labels = record(rank0['labels'])
    assert rank0['chunk_total_median_delta_ms'] == pytest.approx(10 + 3 * 6144 / 100000)
    mqa = record(labels['indexer.mqa'])
    assert mqa['median_delta_ms'] == pytest.approx(8) and mqa['matched_prefixes'] == 4
    assert mqa['share_of_chunk_total_delta'] == pytest.approx(8 / (10 + 3 * 6144 / 100000))
    assert record(labels['collective.all_reduce'])['median_delta_ms'] == pytest.approx(0)
    assert record(labels['sparse.dispatch.triton']) == {'matched_prefixes': 0, 'only_at_cap': [524288]}
    assert record(rank0['indexer_plan']) == {'327680:query_subchunked': 0, '327680:rows': 4, '327680:workspace_exceeded': 0,
        '327680:workspace_group_split': 0, '524288:query_subchunked': 0, '524288:rows': 4,
        '524288:workspace_exceeded': 4, '524288:workspace_group_split': 0}


def test_match_by_size_maps_sse_rows_to_the_unique_traced_request() -> None:
    traces, _ = fake()
    sse: list[dict[str, Json]] = [{'kind': 'warmup', 'cap': 327680, 'target_ctx': 32000, 'trace_request': 'w'}]
    sse += [{'kind': 'cold', 'cap': cap, 'target_ctx': 16384, 'trace_request': f'sse-{cap}'} for cap in (327680, 524288)]
    with pytest.raises(ValueError, match='no cold requests|both rank|no single-request scheduler chunks'):
        fit(traces, sse)    # SSE hashes never match the engine hashes: the plain fit refuses
    mapped = match_by_size(traces, sse)
    assert [r['trace_request'] for r in mapped] == ['w', '327680', '524288'] and mapped[0] is sse[0]
    assert record(record(record(cast(list[Json], fit(traces, mapped)['ranks'])[0])['models_common_support'])['cuda_ms'])[
        'high_cap_extra_ms'] == pytest.approx(10)
    with pytest.raises(ValueError, match='one cold request per cap'):
        match_by_size(traces, sse + [sse[1]])
    with pytest.raises(ValueError, match='exactly one traced request'):
        match_by_size([r for r in traces if r['rank'] == 0], sse)
    twin: list[dict[str, Json]] = [{**r, 'members': [{**record(m), 'request': 'twin'} for m in cast(list[Json], r['members'])]}
            for r in traces if r['max_model_len'] == 327680]
    with pytest.raises(ValueError, match='exactly one traced request'):
        match_by_size(traces + twin, sse)


def test_plan_command_makes_no_network_calls(tmp_path: Path) -> None:
    import sys
    output = tmp_path / 'must-not-exist.jsonl'
    result = subprocess.run([sys.executable, '-m', 'offload.prefill_trace.window_requests', '--cap', '327680',
        '--contexts', '150000', '300000', '--trace-run', 'cpu', '--output', str(output)],
        text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)['executed'] is False
    assert not output.exists()


def test_runtime_verifier_covers_composed_targets() -> None:
    import ast
    from offload.prefill_trace.emit_verify import render
    plain, enabled = render(False), render(True)
    compile(plain, 'off-verifier', 'exec')
    compile(enabled, 'on-verifier', 'exec')
    def pins(text: str) -> dict[str, str]:
        assign = cast(ast.Assign, ast.parse(text).body[1])
        return cast(dict[str, str], ast.literal_eval(assign.value))
    low, high = pins(plain), pins(enabled)
    assert len(low) == 13 and len(high) == 17
    assert high['/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py'] != low['/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py']


def test_paired_cache_salts_reproduce_and_separate_cells() -> None:
    assert cache_salt('up2-window1', 150000, 0, 'main') == cache_salt('up2-window1', 150000, 0, 'main')
    cells = {cache_salt('up2-window1', n, rep, mode) for n in (150000, 300000)
             for rep in range(3) for mode in ('main', 'partial')}
    assert len(cells) == 12
    assert cache_salt(None, 150000, 0, 'main') != cache_salt(None, 150000, 0, 'main')
    with pytest.raises(ValueError, match='invalid fixed'):
        cache_salt('', 150000, 0, 'main')


def test_fit_uses_the_measured_full_chunk_size_and_excludes_the_remainder() -> None:
    """S40 round 3: real GB10 full chunks are 4,092 tokens (PF1 window 1), not 4,096; the last chunk is smaller."""
    traces, requests = fake(chunk=4092, remainder=1000)
    result = fit(traces, requests)
    assert result['full_chunk_tokens'] == 4092
    cuda = record(record(record(cast(list[Json], result['ranks'])[0])['models_common_support'])['cuda_ms'])
    assert cuda['high_cap_extra_ms'] == pytest.approx(10)
    assert result['full_chunk_rank_observations'] == 2 * 4 * 2       # remainder chunks never enter the fit
