"""Round6 scheduling and admission receipts, all CPU fakes in private scratch."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from offload.launch.windows import pf1_window as pf
from offload.launch.windows import pp4_budget as b
from offload.launch.windows import pp4_window as w
from offload.launch.windows.tests import test_pp4_admission as admission
from offload.launch.windows.tests.test_pp4_window import (
    WORKER,
    Fake,
    block_cell,
    complete,
    counter_stats,
)
from offload.prefill_maxlen.evidence import record


def evidence(root: Path, seconds: float = 120.) -> Path:
    session, _result = admission.run(admission.Fake(root))
    assert session.data['verdict'] == 'QUALIFIED'
    path = root/'physical-pair-qualification.json'
    physical = record(json.loads(path.read_text()))
    physical['seconds'] = seconds
    path.write_text(json.dumps(physical))
    data = session.data
    record(data['physical_pair_evidence'])['sha256'] = pf.digest(path)
    record(data['artifact_sha256'])[path.name] = pf.digest(path)
    receipt = root/'admission-receipt.json'
    receipt.write_text(json.dumps(data))
    return receipt


@pytest.mark.parametrize('seconds,reserve,cells', [(1., 180, b.ALL_CELLS), (120., 180, b.ALL_CELLS),
    (120.1, 181, b.ALL_CELLS), (218., 327, b.ALL_CELLS), (218.01, 328, b.B1_CELLS),
    (400., 600, b.B1_CELLS), (638., 957, b.B1_CELLS)])
def test_admission_measured_pair_reserve_and_deterministic_matrix(tmp_path: Path, seconds: float, reserve: int,
        cells: tuple[str, ...]) -> None:
    receipt = evidence(tmp_path, seconds)
    pair = b.pair_reserve(receipt, WORKER.commit)
    assert pair['seconds'] == reserve and pair['measured_driver_s'] == seconds
    assert pair['admission_receipt_sha256'] == pf.digest(receipt)
    assert pair['sha256'] == pf.digest(tmp_path/'physical-pair-qualification.json')
    registered = b.admitted(receipt, WORKER.commit)
    assert registered['cells'] == list(cells)
    assert registered['total_s'] == (4773 if cells == b.ALL_CELLS else 4143)+reserve
    assert registered['fits'] is True and registered['profile_per_start_s'] == 0
    assert registered['restore_s'] == 720 and registered['restore_deadline_s'] == 85*60
    assert registered['abort_s'] == 73*60


def test_even_two_b1_cells_over85_refuse_without_shortening(tmp_path: Path) -> None:
    receipt = evidence(tmp_path, 638.01)
    with pytest.raises(ValueError, match='exceeds85min after B4 omission'):
        b.admitted(receipt, WORKER.commit)


@pytest.mark.parametrize('bad', ['verdict', 'stop', 'restore', 'deadline', 'watch', 'source', 'run', 'artifact',
                                'hash', 'missing', 'duration', 'nan', 'pair-only', 'solo', 'sources'])
def test_pair_duration_requires_own_complete_hashed_admission(tmp_path: Path, bad: str) -> None:
    path = evidence(tmp_path)
    data = record(json.loads(path.read_text()))
    primary = tmp_path/'physical-pair-qualification.json'
    if bad == 'verdict': data['verdict'] = 'NOT QUALIFIED'
    if bad == 'stop': data['stop'] = 'failed probe'
    if bad == 'restore': record(data['restore'])['status'] = 'FAILED'
    if bad == 'deadline': record(data['restore'])['restore_deadline_met'] = False
    if bad == 'watch': record(data['post_close_watch'])['status'] = 'FAIL'
    if bad == 'source': data['source_commit'] = 'f'*40
    if bad == 'run': record(data['physical_pair_evidence'])['run'] = 'other-hold'
    if bad == 'artifact': record(data['artifact_sha256']).pop(primary.name)
    if bad == 'hash': primary.write_text('{}')
    if bad == 'missing': primary.unlink()
    if bad == 'pair-only': data['schema'] = 'pp4-physical-pair-v1'
    if bad in ('duration', 'nan', 'solo', 'sources'):
        physical = record(json.loads(primary.read_text()))
        if bad == 'duration': physical['seconds'] = 0
        if bad == 'nan': physical['seconds'] = float('nan')
        if bad == 'solo': physical['classification'] = 'MEASURED synthetic solo'
        if bad == 'sources': physical['source_sha256'] = {'stale': 'a'*64}
        primary.write_text(json.dumps(physical))
        record(data['physical_pair_evidence'])['sha256'] = pf.digest(primary)
        record(data['artifact_sha256'])[primary.name] = pf.digest(primary)
    path.write_text(json.dumps(data))
    with pytest.raises((ValueError, KeyError, OSError)):
        b.admitted(path, WORKER.commit)


def test_missing_physical_step_prevents_qualified_admission(tmp_path: Path) -> None:
    session, result = admission.run(admission.Fake(tmp_path))
    (tmp_path/'physical-pair-qualification.json').unlink()
    session.finish(result)
    assert session.data['verdict'] == 'NOT QUALIFIED'
    assert session.data['failing_gate'] == 'physical_pair'


def test_b1_fallback_rehearses_full_aba_and_cannot_silently_drop_cells() -> None:
    fake = Fake()
    result = w.window(fake, lambda row: None, measured_cells=b.B1_CELLS)
    assert result['status'] == 'COMPLETE'
    assert record(result['decision'])['status'] == 'CANDIDATE FOR REVIEW'
    assert result['measurement_cells'] == list(b.B1_CELLS)
    assert not any('32k-b4' in event for event in fake.events)
    assert [e for e in fake.events if e in [a.name for a in w.ARMS]] == ['s1-A', 's2-B', 's3-A']
    assert record(result['decision'])['measured_cells'] == list(b.B1_CELLS)
    # Incomplete all-cell results do not become B1 evidence unless registered.
    with pytest.raises(ValueError, match='arm/cell set'):
        w.decision(record(result['starts']))
    bad = Fake()
    refused = w.window(bad, lambda row: None, measured_cells=('short-b1',))
    assert not bad.events and 'stop' in refused


def test_measurement_neither_runs_nor_requires_profile_companions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live = w.Live(tmp_path, pf.Supervisor(tmp_path), native=WORKER, remote='experiment', expected=WORKER.commit,
                  data_dir=tmp_path, tag='owned', admission={'verifier_sha256': {}})
    live.submit = time.time()+3600
    monkeypatch.setattr(live, 'stats', lambda label: counter_stats())
    live.begin_measurement(w.ARMS[0])
    count = iter((1, 2))
    monkeypatch.setattr(live, 'probe_worker', lambda label, arm: {'steps': next(count)})
    before, after = counter_stats(), counter_stats()
    record(after['counters'])['rounds_ok'] = 101
    stats = iter((before, after))
    monkeypatch.setattr(live, 'measured_stats', lambda arm, label: next(stats))
    def call(label: str, argv: list[str], **kwargs: object) -> str:
        output = Path(argv[argv.index('--output')+1])
        row = block_cell('short-b1')
        row.pop('p95_round_ms'); row.pop('complete_round_profile')
        output.write_text(json.dumps(row))
        return ''
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError('measurement must not submit profile companions')
    monkeypatch.setattr(live, 'call', call)
    monkeypatch.setattr(live, 'profile_cell', forbidden)
    try:
        row = live.cell(w.ARMS[0], 'short-b1')
        assert 'p95_round_ms' not in row and 'complete_round_profile' not in row
        assert 'omitted' in str(row['full_round_p95_guard'])
    finally:
        live.sup.close()
    starts = complete()
    for value in starts.values():
        for cell in record(record(value)['cells']).values():
            record(cell).pop('p95_round_ms'); record(cell).pop('complete_round_profile')
    assert w.decision(starts)['status'] == 'CANDIDATE FOR REVIEW'
