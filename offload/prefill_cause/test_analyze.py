"""CPU synthetic span evidence; no model, network, service or GPU."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import cast

import pytest

from offload.prefill_cause import analyze as a
from offload.prefill_cause.test_localize import chunk_with_moe
from offload.prefill_cause.localize import LOW, HIGH, read
from offload.launch.windows.tests.test_pf1_window import request_rows, off_cells
from offload.launch.windows.pf1_window import CAUSE_ARMS, CAUSE_T0
from offload.prefill_maxlen.evidence import Json, record, number


def save_summary(root: Path, summary: dict[str, Json]) -> None:
    summary['artifact_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.jsonl')}
    (root / 'summary.json').write_text(json.dumps(summary))


def evidence(root: Path, t0: float | None = 70.0) -> tuple[Path, Path]:
    off, cause = root / 'off', root / 'cause'
    off.mkdir(); cause.mkdir()
    arms = off_cells()
    for name, off_rows in arms.items():
        (off / f'{name}-requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in off_rows))
    save_summary(off, {'gate': {'status': 'REPRODUCED'}, 'restore': {s: 'PASS' for s in ('start', 'check', 'close')},
                       'arms': {n: {'cold': cast(Json, rows)} for n, rows in arms.items()}})
    found: dict[str, Json] = {}
    for arm in (*CAUSE_ARMS, *((CAUSE_T0,) if t0 is not None else ())):
        run = 'test-' + arm.name
        requests = request_rows(arm, run, (150000,), 1)
        requests[2]['ttft_s'] = t0 if arm == CAUSE_T0 else 70.0 if arm.cap == LOW else 93.0
        (cause / f'{arm.name}-requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in requests))
        found[arm.name] = {'cold': [requests[2]]}
        for rank, host in enumerate(('sparkhead', 'sparkworker-wired')):
            rows: list[dict[str, Json]] = []
            for request in requests[1:]:
                size = int(number(request['target_ctx']))
                full = 4092
                for prefix in range(0, size, full):
                    new = min(full, size - prefix)
                    chunk = chunk_with_moe(rank, arm.cap, prefix, 40, 30.0 if arm.transient else 0.0)
                    for row in chunk:
                        row['run'] = run
                        member = record(cast(list[Json], row['members'])[0])
                        member.update({'new_tokens': new, 'request': request['trace_request'],
                                       'chunk': f'{request["trace_request"]}:{prefix}:{new}:0'})
                        if row.get('phase') == 'chunk.total':
                            row['cuda_ms'] = 1800.0 + prefix / 1000 + (600 if arm.cap == HIGH and arm.transient else 0)
                    rows.extend(chunk)
            path = cause / 'traces' / run / host / f'prefill-r{rank}-p1.jsonl'
            path.parent.mkdir(parents=True)
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    save_summary(cause, {'phase': 'cause', 'status': 'COMPLETE', 'arms': found,
                         'restore': {s: 'PASS' for s in ('start', 'check', 'close')}})
    return cause, off


@pytest.mark.parametrize('ttft,within,reading', [(70, True, 'H1 CONFIRMED'), (70, False, 'AMBIGUOUS'),
    (73.5, True, 'H1 CONFIRMED'), (93, True, 'H1 RULED OUT'), (80, True, 'AMBIGUOUS')])
def test_preregistered_decision(ttft: float, within: bool, reading: str) -> None:
    assert a.t0_reading(ttft, 70, 93, within) == reading


@pytest.mark.parametrize('t0', [70.0, None])
def test_full_analysis_and_layer_wait_subtraction(tmp_path: Path, t0: float | None) -> None:
    cause, off = evidence(tmp_path, t0)
    result = a.analyze(cause, off)
    layer_rows = record(result['per_layer'])
    r1 = [record(r) for r in cast(list[Json], layer_rows['rank1']) if record(r)['phase'] == 'moe.local']
    r0 = [record(r) for r in cast(list[Json], layer_rows['rank0']) if record(r)['phase'] == 'moe.local']
    assert len(r1) == 40 and all(r['median_delta_ms'] == 30 for r in r1)
    assert len(r0) == 40 and all(r['median_delta_ms'] == 0 for r in r0)
    assert record(result['t0'])['reading'] == ('H1 CONFIRMED' if t0 else 'NOT RUN')
    assert str(result['off_chunk_totals']).startswith('UNAVAILABLE')
    assert result['input_sha256']
    if t0:
        chunk = record(record(record(result['t0'])['per_chunk'])['rank1'])
        assert chunk['t0_median_within_cause327_range'] is True


def test_missing_span_and_tampered_evidence_fail(tmp_path: Path) -> None:
    cause, off = evidence(tmp_path, None)
    path = next((cause / 'traces/test-cause-327/sparkhead').glob('*.jsonl'))
    rows = read([path])
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows if not (r.get('phase') == 'moe.ffn' and r.get('layer') == 3)))
    with pytest.raises(ValueError, match='moe.ffn spans|40 cause'):
        a.analyze(cause, off)
    req = cause / 'cause-524-requests.jsonl'
    req.write_text(req.read_text() + '\n')
    with pytest.raises(ValueError, match='evidence changed'):
        a.analyze(cause, off)


def test_cli_writes_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cause, off = evidence(tmp_path, None)
    output = tmp_path / 'analysis.json'
    monkeypatch.setattr(sys, 'argv', ['analyze', '--cause-window', str(cause), '--off-window', str(off), '--output', str(output)])
    a.main()
    assert json.loads(output.read_text())['classification'].startswith('MEASURED')
    with pytest.raises(FileExistsError):
        a.main()



def test_whole_missing_chunk_is_not_hidden_by_size_mapping(tmp_path: Path) -> None:
    cause, off = evidence(tmp_path, None)
    path = next((cause / 'traces/test-cause-327/sparkhead').glob('*.jsonl'))
    rows = read([path])
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows if not (
        record(cast(list[Json], r['members'])[0])['prefix_tokens'] == 4092)))
    with pytest.raises(ValueError, match='missing/duplicate cold chunk prefix'):
        a.analyze(cause, off)
