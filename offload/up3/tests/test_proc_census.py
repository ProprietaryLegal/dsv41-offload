"""Replay captured process shapes; no live Docker, SSH, model or GPU operation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import pytest

from offload.prefill_maxlen.evidence import Json, record
from offload.up3 import host_gates as h, proc_census as p
from offload.up3.tests.pin_fixtures import receipt
from offload.up3.rdma_pin import COMM

FIXTURE = Path(__file__).with_name('fixtures')/'process-census-shapes.json'


class ProcFixture:
    """Captured argv/comm/masks/counts, projected status, synthetic stat births."""

    def __init__(self, rank: int, *, pinned: bool = False) -> None:
        self.rank = rank
        self.rows: list[dict[str, Json]] = copy.deepcopy(json.loads(FIXTURE.read_text())['hosts'][('sparkhead', 'sparkworker-wired')[rank]])
        self.calls: list[tuple[list[str], bool]] = []
        self.sweeps = 0
        self.stat_reads = 0
        self.fault = ''
        self.worker = next(int(str(row['Pid'])) for row in self.rows if row['argv'] == [f'VLLM::Worker_TP{rank}'])
        self.proof: dict[str, Json] = {}
        if pinned:
            for row in self.rows:
                for thread in cast(list[dict[str, Json]], row['threads']):
                    if row['Pid'] == str(self.worker) and thread['Cpus_allowed_list'] == '14':
                        tid = int(str(thread['Pid']))
                        thread['Name'] = COMM
                        self.proof = receipt(self.worker, tid, str(self.worker + 1000), str(tid + 1000))
                        continue
                    thread['Cpus_allowed_list'] = '5-9,15-19'

    def receipt(self, pid: int, starttime: str) -> str:
        assert pid == self.worker and starttime == str(pid + 1000)
        return json.dumps(self.proof)

    @staticmethod
    def status(thread: dict[str, Json]) -> str:
        keys = ('Name', 'State', 'Tgid', 'Pid', 'PPid', 'Threads', 'Cpus_allowed_list')
        return ''.join(f'{key}:\t{thread[key]}\n' for key in keys)

    def cat(self, paths: list[str], sparse: bool = False) -> str:
        self.calls.append((paths, sparse))
        assert all(p.proc_path(path) for path in paths)
        if paths == ['/proc/sys/kernel/ns_last_pid']:
            self.sweeps += 1
            if self.fault == 'overflow':
                return str(p.MAX_PID+1)+'\n'
            return str(max(int(str(t['Pid'])) for row in self.rows for t in cast(list[dict[str, Json]], row['threads'])))+'\n'
        out = ''
        for path in paths:
            parts = path.split('/')
            tid = int(parts[-2])
            thread = next((t for row in self.rows for t in cast(list[dict[str, Json]], row['threads']) if t['Pid'] == str(tid)), None)
            if self.fault == 'unreadable' and tid == self.worker:
                raise PermissionError('required worker unreadable')
            if thread is None:
                assert sparse, path
                continue
            if self.fault == 'thread-disappears' and self.sweeps == 2 and tid != self.worker and thread['Tgid'] == str(self.worker):
                continue
            if parts[-1] == 'status':
                out += self.status(thread)
            elif parts[-1] == 'cmdline':
                row = next(row for row in self.rows if row['Pid'] == str(tid))
                out += '\0'.join(cast(list[str], row['argv']))+'\0'
            elif parts[-1] == 'stat':
                # Field22 starttime is synthetic. Fault changes it after the second sweep.
                birth = str(tid+1000+(self.fault == 'reuse' and self.sweeps == 2))
                out += f'{tid} ({thread["Name"]}) S ' + ' '.join(['0']*18+[birth])+'\n'
        return out


@pytest.mark.parametrize('rank', [0, 1])
def test_captured_real_tree_argv_and_every_thread_without_any_log(rank: int) -> None:
    fixture = ProcFixture(rank)
    result = p.collect(rank, fixture.cat)
    processes = [record(row) for row in cast(list[Json], result['processes'])]
    assert {row['role'] for row in processes} == ({'launcher', 'engine', 'worker', 'proxy'} if rank == 0 else {'launcher', 'worker'})
    assert fixture.worker in cast(list[Json], result['pinned_pids'])
    assert len(cast(list[Json], result['proxy_pids'])) == (1 if rank == 0 else 0)
    assert all(len(cast(list[Json], row['threads'])) == row['thread_count'] for row in processes)
    assert fixture.sweeps == 2
    assert 'no unique' in str(result['poller_identity'])
    worker = next(row for row in processes if row['role'] == 'worker')
    threads = [record(t) for t in cast(list[Json], worker['threads'])]
    cpu14 = [t for t in threads if t['cpus'] == [14]]
    assert len(cpu14) == 1 and cpu14[0]['tid'] != worker['pid']
    assert cpu14[0]['comm'] == 'VLLM::Worker_TP'
    assert sum(t['comm'] == cpu14[0]['comm'] for t in threads) > 100


@pytest.mark.parametrize('rank', [0, 1])
def test_exact_thread_masks_and_proven_poller_qualify_without_pin_or_proxy_logs(rank: int) -> None:
    fixture = ProcFixture(rank, pinned=True)
    affinity = p.collect(rank, fixture.cat, receipts=fixture.receipt)
    result: dict[str, Json] = {'arm': 'cpu-pin', 'rank': rank, 'container': 'fixture', 'affinity': affinity}
    h.validate(result, 'cpu-pin', rank, 'fixture')
    assert affinity['status'] == 'OBSERVED'


@pytest.mark.parametrize('rank', [0, 1])
def test_actual_shared_worker_comm_does_not_prove_poller_exception(rank: int) -> None:
    fixture = ProcFixture(rank, pinned=True)
    worker = next(row for row in fixture.rows if row['Pid'] == str(fixture.worker))
    thread = next(t for t in cast(list[dict[str, Json]], worker['threads']) if t['Pid'] != worker['Pid'] and t['Name'] == 'VLLM::Worker_TP')
    thread['Cpus_allowed_list'] = '14'
    affinity = p.collect(rank, fixture.cat, receipts=fixture.receipt)
    with pytest.raises(ValueError, match='proven RDMA poller identity'):
        h.validate({'arm': 'cpu-pin', 'rank': rank, 'container': 'fixture', 'affinity': affinity}, 'cpu-pin', rank, 'fixture')


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('fault', ['unreadable', 'reuse', 'thread-disappears', 'overflow', 'missing-worker', 'wrong-parent', 'wrong-argv', 'incomplete-threads'])
def test_incomplete_or_unidentified_required_process_always_refuses(rank: int, fault: str) -> None:
    fixture = ProcFixture(rank)
    fixture.fault = fault
    worker = next(row for row in fixture.rows if row['Pid'] == str(fixture.worker))
    if fault == 'missing-worker': fixture.rows.remove(worker)
    elif fault == 'wrong-parent':
        for thread in cast(list[dict[str, Json]], worker['threads']): thread['PPid'] = '9999'
    elif fault == 'wrong-argv': worker['argv'] = ['unknown-worker']
    elif fault == 'incomplete-threads':
        cast(list[dict[str, Json]], worker['threads']).pop()
    with pytest.raises((ValueError, PermissionError)):
        p.collect(rank, fixture.cat)


def test_proc_paths_are_finite_and_cannot_be_shell_or_other_files() -> None:
    assert p.proc_path('/proc/1/task/2/stat')
    for path in ('/etc/passwd', '/proc/self/status', '/proc/1/environ', '/proc/1/mem', '/proc/1/../2/status', '/proc/1/stat;true', '/proc/0/status', f'/proc/{p.MAX_PID+1}/status'):
        assert not p.proc_path(path)


def test_cpu_list_and_stat_parse_fail_closed() -> None:
    assert p.cpu_list('5-9,15-19') == h.CPUS
    assert p.stats('42 (worker (name)) S ' + ' '.join(['0']*18+['123'])+'\n') == {42: '123'}
    for text in ('', '9-5', '-1', '0-999999', 'x'):
        with pytest.raises(ValueError): p.cpu_list(text)
    with pytest.raises(ValueError): p.stats('42 (broken) S 1 2\n')


def test_excessive_leaders_refuse_before_argv_reads() -> None:
    def denied(paths: list[str], sparse: bool) -> str:
        raise AssertionError('unbounded argv read')
    with pytest.raises(ValueError, match='excessive process leaders'):
        p.identify({pid: {'pid': pid} for pid in range(1, 10)}, 0, denied)
