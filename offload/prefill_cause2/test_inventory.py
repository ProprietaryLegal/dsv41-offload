"""PF1-CAUSE2 item 1 inventory: the cause overlay's edits are exactly imports + span wrappers; host-lead arithmetic."""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from offload.prefill_cause import build_overlay as bo
from offload.prefill_cause2 import inventory as inv


def test_cause_overlay_edits_are_imports_and_span_wrappers_only() -> None:
    out = inv.diff()
    files = cast(dict[str, dict[str, object]], out['files'])
    assert set(files) == set(bo.FILES)
    assert {p: f['imports_added'] for p, f in files.items()} == {
        'vllm/models/deepseek_v41/nvidia/model.py': ['from vllm.v1.worker.gpu import dsv41_prefill_trace as _pf'],
        'vllm_exl3_moe/tf_prompt.py': ['from vllm.v1.worker.gpu import dsv41_prefill_trace as _pf',
                                       'from vllm.v1.worker.gpu import dsv41_prefill_cause as _pc'],
        'vllm_exl3_moe/tf_experts.py': ['from vllm.v1.worker.gpu import dsv41_prefill_trace as _pf'],
    }
    spans = {p: len(cast(list[object], f['spans'])) for p, f in files.items()}
    assert spans == {'vllm/models/deepseek_v41/nvidia/model.py': 4, 'vllm_exl3_moe/tf_prompt.py': 2,
                     'vllm_exl3_moe/tf_experts.py': 19}
    pins = {path: digest for digest, path in (line.split() for line in (bo.HERE / 'OVERLAY.sha256').read_text().splitlines())}
    assert all(f['overlay_sha256'] == pins[p] for p, f in files.items())
    meta = {c for f in files.values() for s in cast(list[dict[str, list[str]]], f['spans']) for c in s['metadata_calls']}
    assert meta == {'_pc.alloc', 's.xd.data_ptr', 's.xg.data_ptr', 'out.data_ptr'}
    helper = cast(dict[str, object], out['helper'])
    assert helper['calls'] == ['STATS.items', 'int', 'stats.get', 'torch.cuda.memory_stats']


def test_census_refuses_a_behavioural_change_outside_the_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    real = bo.instrument

    def sneaky(text: str, path: str) -> str:
        out = real(text, path)
        if path.endswith('tf_prompt.py'):
            out = out.replace('    ex = tables(bank, mcg, mul1)\n', '    ex = tables(bank, mcg, mul1)\n    torch.cuda.synchronize()\n', 1)
        return out
    monkeypatch.setattr(bo, 'instrument', sneaky)
    with pytest.raises(ValueError, match='behaviour-relevant calls differ'):
        inv.diff()


def _trace(path: Path, chunks: list[tuple[float, list[float], int]]) -> None:
    """chunks: (chunk.total cuda_ms, 81 all-reduce host offsets in ms, extra timed spans)."""
    rows: list[dict[str, object]] = []
    for n, (gpu, offsets, extra) in enumerate(chunks):
        members = [{'chunk': f'r:{n}', 'new_tokens': 4092, 'prefix_tokens': n * 4092, 'request': 'r'}]
        begin = 10**12 + n * 10**10
        rows.append({'phase': 'chunk.begin', 'members': members, 'host_start_ns': begin})
        rows.append({'phase': 'chunk.end', 'members': members, 'host_end_ns': begin + int(gpu * 1e6)})
        event = 0
        for off in offsets:
            rows.append({'phase': 'collective.all_reduce', 'members': members, 'event': event,
                         'host_start_ns': begin + int(off * 1e6), 'cuda_ms': 1.0})
            event += 1
        for _ in range(extra):
            rows.append({'phase': 'tf.rot_in', 'members': members, 'event': event, 'host_start_ns': begin, 'cuda_ms': .1})
            event += 1
        rows.append({'phase': 'chunk.total', 'members': members, 'event': event, 'host_start_ns': begin, 'cuda_ms': gpu})
    # a small chunk (warm-up tail) is ignored
    rows.append({'phase': 'chunk.begin', 'members': [{'chunk': 's', 'new_tokens': 7, 'prefix_tokens': 0, 'request': 's'}],
                 'host_start_ns': 1})
    path.write_text(''.join(json.dumps(r) + '\n' for r in reversed(rows)))


def test_lead_reports_queued_gpu_time_and_refuses_incomplete(tmp_path: Path) -> None:
    offsets = [float(i) * 10 for i in range(81)]
    _trace(tmp_path / 'a.jsonl', [(1800.0, offsets, 0), (1900.0, offsets, 0), (2000.0, offsets, 0)])
    _trace(tmp_path / 'b.jsonl', [(1800.0, offsets, 480)])
    out = cast(dict[str, dict[str, object]], inv.lead([tmp_path / 'a.jsonl', tmp_path / 'b.jsonl']))
    a, b = out[str(tmp_path / 'a.jsonl')], out[str(tmp_path / 'b.jsonl')]
    assert a['full_chunks'] == 3 and a['chunk_gpu_ms'] == 1900.0 and a['event_pairs_per_chunk'] == 82
    assert b['event_pairs_per_chunk'] == 82 + 480
    assert cast(dict[str, float], a['ar_host_offset_ms'])['layer19_ffn'] == pytest.approx(400.0)
    assert a['gpu_ms_queued_at_layer19_ffn'] == pytest.approx(1500.0)
    assert a['host_ms_per_layer_before_backpressure'] == pytest.approx((200.0 - 20.0) / 9)
    _trace(tmp_path / 'c.jsonl', [(1800.0, offsets[:80], 0)])
    with pytest.raises(ValueError, match='80 all-reduces'):
        inv.lead([tmp_path / 'c.jsonl'])
