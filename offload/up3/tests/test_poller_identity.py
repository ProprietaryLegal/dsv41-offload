"""Receipt/census agreement, actual collector/runner refusal and restoration."""
from __future__ import annotations

import io
import json
import subprocess
from typing import cast
from unittest.mock import patch

import pytest

from offload.launch.windows import up3_window as w, pf1_window as pf
from offload.prefill_maxlen.evidence import Json, record
from offload.up3 import host_gates as h, poller_identity as proof, proc_census, cells
from offload.up3.tests.pin_fixtures import evidence
from offload.up3.tests.test_proc_census import ProcFixture
from offload.up3.tests.test_up3_host_gates import serve_script
from offload.up3.tests.test_up3_window import Remote, rehearsal as rehearsal


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('field,value', [
    ('pid', 102), ('tid', 101), ('tid', 999), ('tid', True), ('pid_starttime', '99'),
    ('tid_starttime', '99'), ('schema', 'old'), ('new_comm', 'VLLM::Worker_TP'),
    ('old_comm', ''), ('old_comm', 'exl3-rdma-prx'), ('old_comm', 'x' * 16),
    ('affinity', [5]), ('affinity', [14.0]), ('monotonic_ns', 0), ('monotonic_ns', True),
    ('clock', 'CLOCK_REALTIME'),
])
def test_every_receipt_field_is_checked_against_current_census(rank: int, field: str, value: Json) -> None:
    row = evidence(rank)
    process = record(cast(list[Json], record(row['affinity'])['processes'])[0])
    record(process['pin_receipt'])[field] = value
    with pytest.raises(ValueError, match='proven RDMA poller identity'):
        h.validate(row, 'cpu-pin', rank, 'fixture')


@pytest.mark.parametrize('fault', ['missing', 'extra', 'name-only', 'mask-only', 'duplicate-name', 'wrong-census-mask',
    'unrelated-cpu14', 'proxy-impersonates'])
def test_name_or_cpu14_alone_never_proves_poller(fault: str) -> None:
    row = evidence(0)
    processes = [record(item) for item in cast(list[Json], record(row['affinity'])['processes'])]
    worker = processes[0]
    threads = cast(list[Json], worker['threads'])
    poller = record(threads[1])
    if fault == 'missing': worker.pop('pin_receipt')
    elif fault == 'extra': record(worker['pin_receipt'])['unregistered'] = True
    elif fault == 'name-only': poller['tid'] = 999
    elif fault == 'mask-only': poller['comm'] = 'VLLM::Worker_TP'
    elif fault == 'duplicate-name': threads.append({**poller, 'tid': 999})
    elif fault == 'wrong-census-mask': poller['cpus'] = cast(Json, h.CPUS)
    elif fault == 'unrelated-cpu14': threads.append({**poller, 'tid': 999, 'comm': 'worker'})
    else: processes[1]['pin_receipt'] = worker['pin_receipt']
    with pytest.raises(ValueError, match='proven RDMA poller identity'):
        h.validate(row, 'cpu-pin', 0, 'fixture')


def test_receipt_decoder_refuses_duplicate_or_truncated_json() -> None:
    for text in ('{"tid":1,"tid":2}', '{', '[]', 'null'):
        with pytest.raises(ValueError): proof.decode(text)


def test_receipt_read_is_bracketed_by_both_identity_sweeps() -> None:
    fixture = ProcFixture(0, pinned=True)
    def read(pid: int, starttime: str) -> str:
        fixture.fault = 'reuse'
        return fixture.receipt(pid, starttime)
    with pytest.raises(ValueError, match='reused'):
        proc_census.collect(0, fixture.cat, receipts=read)


def test_receipt_transport_is_finite_local_experiment_cat_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    def output(argv: list[str], **kwargs: object) -> str:
        calls.append(argv)
        assert kwargs['timeout'] == 15 and kwargs['errors'] == 'strict'
        return '{}'
    monkeypatch.setattr(subprocess, 'check_output', output)
    assert h.ExperimentReads().pin_receipt(101, '123') == '{}'
    assert calls == [['docker', 'exec', 'dsv41-serve', 'cat', '/run/up3-rdma-pin/101-123.json']]
    for pid, start in ((0, '123'), (proc_census.MAX_PID + 1, '123'), (101, '../123')):
        with pytest.raises(ValueError): h.ExperimentReads().pin_receipt(pid, start)
    with pytest.raises(ValueError): h.ExperimentReads('sparkhead').pin_receipt(101, '123')
    with pytest.raises(ValueError): h.ProductionReads().output(calls[0])
    assert len(calls) == 1


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('fault', ['none', 'missing', 'tid-reused', 'comm-differs', 'wrong-affinity', 'duplicate-receipt-field'])
def test_actual_window_collects_receipt_and_never_times_unproved_poller(rank: int, fault: str,
        rehearsal: tuple[w.Live, Remote], monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake = rehearsal
    live.group = 'cpu-pin'
    original = fake.remote
    receipts_read: list[int] = []

    class Logs:
        def __init__(self) -> None:
            self.stdout = io.StringIO('')  # S73 suppressed pin-init producer.

        def wait(self) -> int:
            return 0

    def remote(label: str, host: str, command: str, stdin: str | None) -> str:
        if '-host-gate-' not in label:
            return original(label, host, command, stdin)
        current_rank = pf.HOSTS.index(host)
        fixture = ProcFixture(current_rank, pinned=True)
        if current_rank == rank:
            if fault == 'tid-reused': fixture.proof['tid_starttime'] = '99'
            elif fault == 'wrong-affinity': fixture.proof['affinity'] = [5]
            elif fault == 'comm-differs':
                for row in fixture.rows:
                    for thread in cast(list[dict[str, Json]], row['threads']):
                        if thread['Name'] == 'exl3-rdma-prx': thread['Name'] = 'VLLM::Worker_TP'

        def output(argv: list[str], **kwargs: object) -> str:
            if argv == ['docker', 'inspect', 'dsv41-serve']:
                return json.dumps([{'Id': live.ids[current_rank], 'State': {'Running': True}}])
            if argv == ['docker', 'exec', 'dsv41-serve', 'cat', '/serve.sh']:
                return serve_script()
            assert argv == ['docker', 'exec', 'dsv41-serve', 'cat',
                f'/run/up3-rdma-pin/{fixture.worker}-{fixture.worker + 1000}.json']
            receipts_read.append(current_rank)
            if current_rank == rank and fault == 'missing': raise FileNotFoundError('lazy ring: no pin receipt')
            text = fixture.receipt(fixture.worker, str(fixture.worker + 1000))
            return text[:-1] + ',"tid":999}' if current_rank == rank and fault == 'duplicate-receipt-field' else text

        with patch.object(subprocess, 'check_output', output), patch.object(subprocess, 'Popen', lambda *a, **k: Logs()), \
                patch.object(h.ProductionReads, 'proc', lambda self, paths, sparse=False: fixture.cat(paths, sparse)):
            return json.dumps(h.collect('cpu-pin', current_rank, live.ids[current_rank]))

    monkeypatch.setattr(fake, 'remote', remote)
    checkpoints: list[dict[str, Json]] = []
    result = w.window(live, 'cpu-pin', checkpoints.append)
    assert rank in receipts_read
    starts = record(result['starts'])
    if fault == 'none':
        assert result['status'] == 'COMPLETE', result
        assert record(result['decision'])['status'] == 'CANDIDATE FOR REVIEW'
    else:
        assert result['status'] == 'INCOMPLETE', result
        assert record(result['decision'])['status'] == 'NOT A CANDIDATE'
        assert record(starts['s2-cpu-pin'])['cells'] == {}
        assert not any('s2-cpu-pin-' + name in fake.commands for name in cells.CELLS)
        assert {'arm': 's2-cpu-pin', 'gates': 'PASS'} not in checkpoints
        assert 's3-control' not in starts
    assert str(record(result['restore'])['status']).startswith('RESTORED'), result
    assert fake.restores == ['start', 'check', 'close'] and not fake.hold
