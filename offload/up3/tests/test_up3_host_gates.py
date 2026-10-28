"""Round4 config-only policy and read-only evidence, all CPU fixtures."""
from __future__ import annotations

import copy
import json
import io
import os
import subprocess
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from offload.launch.windows import up3_window as w, pf1_window as pf
from offload.prefill_maxlen.evidence import Json, record
from offload.up3 import host_gates as h, cells
from offload.up3.tests.test_up3_window import Remote, rehearsal as rehearsal
from offload.up3.tests.test_proc_census import ProcFixture
from offload.up3.tests.pin_fixtures import evidence as pin_evidence

FIXTURES = Path(__file__).with_name('fixtures')


def serve_script(port: str = '8001') -> str:
    # Captured by docker exec cat /serve.sh on sparkhead, 2026-10-27.
    return (FIXTURES/'production-serve-rank0.sh').read_text().replace('--port 8000', '--port ' + port)


def ring(rank: int = 0) -> dict[str, Json]:
    return {'arm': 'dual-ring', 'rank': rank, 'container': 'fixture',
        'rdma': [rdma_line(rank)],
        'gids': [{'device': dev, 'index': 3, 'type': 'RoCE v2', 'ipv4': f'10.202.{26+i}.1', 'netdev': 'eth'+str(i)}
                 for i, dev in enumerate(h.DEVICES)]}


def rdma_line(rank: int = 0) -> str:
    templates = json.loads((FIXTURES/'logger-templates.json').read_text())
    return str(templates['rdma']['template']) % (rank, 2, ','.join(h.DEVICES), 3, 14, 256, 3072, 'up to 3072 KB')


def pin(rank: int = 0, container: str = 'fixture') -> dict[str, Json]:
    return pin_evidence(rank, container)


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('fault', ['empty', 'missing-pinned', 'omitted-process', 'unreadable', 'reused'])
def test_incomplete_cpu_pin_refuses_before_candidate_timing_and_restores(rank: int, fault: str,
        rehearsal: tuple[w.Live, Remote], monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake = rehearsal
    live.group = 'cpu-pin'
    original = fake.remote
    evidence: dict[str, Json] = {}
    def remote(label: str, host: str, command: str, stdin: str | None) -> str:
        if '-host-gate-' not in label or host != pf.HOSTS[rank]:
            return original(label, host, command, stdin)
        evidence.update(pin(rank, live.ids[rank]))
        affinity = record(evidence['affinity'])
        processes = cast(list[Json], affinity['processes'])
        if fault == 'empty':
            affinity.update(pinned_pids=[], proxy_pids=[], processes=[])
        elif fault == 'missing-pinned':
            affinity['pinned_pids'] = []
            affinity['processes'] = processes[1:]
        elif fault == 'omitted-process':
            processes.pop(0)
        else:
            index = -1 if fault == 'unreadable' else 0
            processes[index] = {'pid': record(processes[index])['pid'], 'status': 'UNMEASURED',
                'reason': 'ProcessLookupError' if fault == 'reused' else 'PermissionError'}
        return json.dumps(evidence)
    monkeypatch.setattr(fake, 'remote', remote)
    checkpoints: list[dict[str, Json]] = []
    result = w.window(live, 'cpu-pin', checkpoints.append)
    assert result['status'] == 'INCOMPLETE', result
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'
    assert 'incomplete required CPU-pin affinity census' in str(result['stop'])
    assert 'PARTIAL' in str(result['stop'])
    if fault in ('unreadable', 'reused'):
        assert ('ProcessLookupError' if fault == 'reused' else 'PermissionError') in str(result['stop'])
    starts = record(result['starts'])
    assert record(starts['s1-control'])['status'] == 'COMPLETE'
    assert record(starts['s2-cpu-pin'])['cells'] == {}
    assert not any('s2-cpu-pin-' + cell in fake.commands for cell in cells.CELLS)
    assert {'arm': 's2-cpu-pin', 'gates': 'PASS'} not in checkpoints
    assert 's3-control' not in starts
    assert str(record(result['restore'])['status']).startswith('RESTORED'), result
    assert fake.restores == ['start', 'check', 'close']
    assert not fake.hold


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('fault', ['leader14', 'missing-leader'])
def test_cpu_pin_leader_refuses_before_candidate_timing_and_restores(rank: int, fault: str,
        rehearsal: tuple[w.Live, Remote], monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake = rehearsal
    live.group = 'cpu-pin'
    original = fake.remote
    def remote(label: str, host: str, command: str, stdin: str | None) -> str:
        if '-host-gate-' not in label or host != pf.HOSTS[rank]:
            return original(label, host, command, stdin)
        evidence = pin(rank, live.ids[rank])
        process = record(cast(list[Json], record(evidence['affinity'])['processes'])[0])
        thread = record(cast(list[Json], process['threads'])[0])
        if fault == 'leader14':
            thread['cpus'] = [14]
        else:
            thread['tid'] = 999
        return json.dumps(evidence)
    monkeypatch.setattr(fake, 'remote', remote)
    checkpoints: list[dict[str, Json]] = []
    result = w.window(live, 'cpu-pin', checkpoints.append)
    assert result['status'] == 'INCOMPLETE', result
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'
    assert 'process leader' in str(result['stop'])
    starts = record(result['starts'])
    assert record(starts['s1-control'])['status'] == 'COMPLETE'
    assert record(starts['s2-cpu-pin'])['cells'] == {}
    assert not any('s2-cpu-pin-' + cell in fake.commands for cell in cells.CELLS)
    assert {'arm': 's2-cpu-pin', 'gates': 'PASS'} not in checkpoints
    assert 's3-control' not in starts
    assert str(record(result['restore'])['status']).startswith('RESTORED'), result
    assert fake.restores == ['start', 'check', 'close']
    assert not fake.hold


@pytest.mark.parametrize('pid', [101, 102])
@pytest.mark.parametrize('fault', ['leader14', 'missing-leader', 'wrong-mask'])
def test_each_required_process_leader_is_present_and_has_exact_mask(pid: int, fault: str) -> None:
    row = pin()
    processes = cast(list[Json], record(row['affinity'])['processes'])
    process = next(record(p) for p in processes if record(p)['pid'] == pid)
    thread = record(cast(list[Json], process['threads'])[0])
    if fault == 'missing-leader':
        thread['tid'] = 999
    else:
        thread['cpus'] = [14] if fault == 'leader14' else [0]
    with pytest.raises(ValueError, match='process leader'):
        h.validate(row, 'cpu-pin', 0, 'fixture')


@pytest.mark.parametrize('rank', [0, 1])
def test_nonleader14_without_unique_poller_identity_refuses(rank: int) -> None:
    row = pin(rank)
    for process in cast(list[Json], record(row['affinity'])['processes']):
        threads = cast(list[Json], record(process)['threads'])
        threads.append({'tid': 999, 'starttime': '13', 'cpus': [14], 'comm': 'VLLM::Worker_TP'})
    with pytest.raises(ValueError, match='proven RDMA poller identity'):
        h.validate(row, 'cpu-pin', rank, 'fixture')


@pytest.mark.parametrize('arm', ['split', 'vor'])
def test_code_arms_refuse_before_any_command_preflight_or_output(arm: str, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def denied(*args: object, **kwargs: object) -> None:
        raise AssertionError('parked arm reached a command')
    monkeypatch.setattr(subprocess, 'run', denied)
    monkeypatch.setattr(subprocess, 'check_output', denied)
    monkeypatch.setattr(w.Live, 'preflight', denied)
    assert w.main(['--execute', '--group', arm, '--out', str(tmp_path/'absent')]) == 1
    assert 'BEFORE HOLD' in capsys.readouterr().err
    assert not (tmp_path/'absent').exists()


@pytest.mark.parametrize('group', [g for g in h.CONFIG_ARMS if g in w.GROUPS])
def test_config_arms_run_without_any_resident_collector(group: str, rehearsal: tuple[w.Live, Remote],
        monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake = rehearsal
    live.group = group
    def denied(*args: object, **kwargs: object) -> None:
        raise AssertionError('config arm attempted resident collection')
    from offload.up3 import qualify
    monkeypatch.setattr(qualify, 'collect', denied)
    result = w.window(live, group, lambda row: None)
    assert result['status'] == 'COMPLETE', result
    assert str(record(result['restore'])['status']).startswith('RESTORED')
    assert not any('offload/up3/qualify.py' in c for c in fake.commands)
    if group == 'cpu-pin':
        assert record(result['decision'])['status'] == 'CANDIDATE FOR REVIEW'
        for host in pf.HOSTS:
            evidence = record(record(record(record(result['starts'])['s2-cpu-pin'])['gates'])[host])
            assert record(evidence['affinity'])['status'] == 'OBSERVED'
            assert 'UNMEASURED' in str(record(evidence['affinity'])['native_tid_births'])


@pytest.mark.parametrize('fault', ['absent', 'single', 'rank', 'gid', 'fallback', 'duplicate', 'device', 'fabric', 'v1', 'container'])
def test_dual_ring_requires_both_exact_logs_and_gid3(fault: str) -> None:
    row = ring()
    h.validate(copy.deepcopy(row), 'dual-ring', 0, 'fixture')
    if fault == 'absent': row['rdma'] = []
    elif fault == 'single': row['rdma'] = [str(cast(list[Json], row['rdma'])[0]).replace(',roceP2p1s0f1', '')]
    elif fault == 'rank': row['rank'] = 1
    elif fault == 'gid': record(cast(list[Json], row['gids'])[1])['index'] = 0
    elif fault == 'fallback': row['rdma'] = ['RDMA all-reduce off; NCCL stays']
    elif fault == 'duplicate': cast(list[Json], row['rdma']).append(cast(list[Json], row['rdma'])[0])
    elif fault == 'device': cast(list[Json], row['gids']).pop()
    elif fault == 'fabric': record(cast(list[Json], row['gids'])[1])['ipv4'] = '10.202.28.1'
    elif fault == 'v1': record(cast(list[Json], row['gids'])[0])['type'] = 'RoCE v1'
    else: row['container'] = 'other'
    with pytest.raises(ValueError):
        h.validate(row, 'dual-ring', 0, 'fixture')


def test_gid_reads_actual_sysfs_fixture(tmp_path: Path) -> None:
    port = tmp_path/h.DEVICES[0]/'ports/1'
    for path, text in {'gids/3': '::ffff:10.202.26.1', 'gid_attrs/types/3': 'RoCE v2', 'gid_attrs/ndevs/3': 'eth0'}.items():
        target = port/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    assert h.gid(tmp_path, h.DEVICES[0])['ipv4'] == '10.202.26.1'


def test_inaccessible_affinity_refuses_and_retains_partial_evidence() -> None:
    row = pin()
    cast(list[Json], record(row['affinity'])['processes'])[1] = {'pid': 102, 'role': 'proxy',
        'status': 'UNMEASURED', 'reason': 'PermissionError'}
    with pytest.raises(ValueError, match='incomplete required CPU-pin affinity census.*PermissionError'):
        h.validate(row, 'cpu-pin', 0, 'fixture')
    assert record(row['affinity'])['status'] == 'PARTIAL'
    assert record(row['affinity'])['unmeasured'] == [{'pid': 102, 'role': 'proxy', 'status': 'UNMEASURED', 'reason': 'PermissionError'}]


def test_missing_rank0_proxy_refuses_and_retains_reason() -> None:
    row = pin()
    evidence = record(row['affinity'])
    evidence['proxy_pids'] = []
    cast(list[Json], evidence['processes']).pop()
    with pytest.raises(ValueError, match='incomplete required CPU-pin affinity census.*proxy PID unavailable'):
        h.validate(row, 'cpu-pin', 0, 'fixture')
    assert evidence['status'] == 'PARTIAL'


def test_observed_affinity_mismatch_refuses() -> None:
    row = pin()
    record(cast(list[Json], record(cast(list[Json], record(row['affinity'])['processes'])[0])['threads'])[0])['cpus'] = [0]
    with pytest.raises(ValueError, match='affinity differs'):
        h.validate(row, 'cpu-pin', 0, 'fixture')


@pytest.mark.parametrize('port', ['8000', '8001'])
def test_actual_collector_uses_mounted_serve_port_and_ignores_auxiliary_rdma_line(port: str, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    docker = tmp_path/'docker'
    log = str(cast(list[Json], ring()['rdma'])[0])
    docker.write_text('#!/bin/bash\ncase "$1" in\ninspect) cat <<\'JSON\'\n' + json.dumps([
        {'Id': 'fixture', 'State': {'Running': True},
         'Config': {'Cmd': ['/serve.sh'], 'Entrypoint': ['/bin/bash']}}]) +
        '\nJSON\n;;\nexec) [ "$2 $3 $4" = "dsv41-serve cat /serve.sh" ] || exit 98\ncat <<\'SERVE\'\n' +
        serve_script(port) + '\nSERVE\n;;\nlogs) cat <<\'LOG\'\n' + log +
        "\nEXL3 RDMA all-reduce: vLLM's TP all-reduce routed for bf16 tensors up to 64 KB\nLOG\n;;\n*) exit 98;;\nesac\n")
    docker.chmod(0o700)
    monkeypatch.setenv('PATH', str(tmp_path) + ':' + os.environ['PATH'])
    gids = {dev: record(cast(list[Json], ring()['gids'])[i]) for i, dev in enumerate(h.DEVICES)}
    monkeypatch.setattr(h, 'gid', lambda root, dev: gids[dev])
    if port == '8000':
        with pytest.raises(ValueError, match='requires experiment8001'):
            h.collect('dual-ring', 0, 'fixture')
        return
    evidence = h.collect('dual-ring', 0, 'fixture')
    h.validate(evidence, 'dual-ring', 0, 'fixture')
    assert evidence['rdma'] == [log]


@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'dangling', 'equals', 'variable', 'second-launch', 'other-command'])
def test_serve_port_refuses_ambiguous_or_nonliteral_launcher(fault: str) -> None:
    script = serve_script()
    if fault == 'duplicate': script += '\n# next line is an extra port\nextra --port 8001\n'
    elif fault == 'missing': script = script.replace('--port 8001', '')
    elif fault == 'dangling': script = 'exec vllm serve model --port'
    elif fault == 'equals': script = script.replace('--port 8001', '--port=8001')
    elif fault == 'variable': script = script.replace('--port 8001', '--port "$PORT"')
    elif fault == 'second-launch': script += '\nexec vllm serve model\n'
    else: script = script.replace('exec vllm serve', 'echo vllm serve')
    with pytest.raises(ValueError, match='host gate requires'):
        h.serve_port(script)


def test_serve_port_joins_continuations_and_ignores_comments() -> None:
    assert h.serve_port(serve_script().replace('--port 8001', '--port \\\n  8001') + '\n# --port 8000\n') == '8001'


@pytest.mark.parametrize('rank,host', [(0, 'sparkhead'), (1, 'sparkworker-wired')])
def test_captured_host_shapes_and_production_log_bodies(rank: int, host: str, tmp_path: Path) -> None:
    shapes = json.loads((FIXTURES/'production-host-shapes.json').read_text())[host]
    assert {key: shapes['inspect']['Config'][key] for key in ('Cmd', 'Entrypoint')} == {
        'Cmd': ['/serve.sh'], 'Entrypoint': ['/bin/bash']}
    assert shapes['inspect']['State'] == {'Running': True, 'Paused': False, 'Status': 'running'}
    assert shapes['inspect']['serve_mount']['Destination'] == '/serve.sh'
    assert shapes['inspect']['serve_mount']['RW'] is False
    root = tmp_path/'infiniband'
    for name, value in shapes['sysfs'].items():
        path = root/str(name).removeprefix('/sys/class/infiniband/')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(value) + '\n')
    evidence = ring(rank)
    evidence['gids'] = [h.gid(root, dev) for dev in h.DEVICES]
    h.validate(evidence, 'dual-ring', rank, 'fixture')
    for index, row in enumerate(cast(list[Json], evidence['gids'])):
        assert record(row)['ipv4'] == f'10.202.{26+index}.{2 if rank == 0 else 1}'
    logs = h.log_evidence((FIXTURES/f'production-logs-{host}.txt').read_text().splitlines())
    assert logs['rdma'] == [rdma_line(rank).replace(',roceP2p1s0f1', '')]
    assert logs['rdma_routing_lines'] == (2 if rank == 0 else 1)
    assert logs['pinned_pids'] == []  # EXL3_CPU_PIN is OFF in production.
    assert logs['proxy_pids'] == ([462] if rank == 0 else [])


@pytest.mark.parametrize('rank', [0, 1])
def test_exact_source_logger_templates_parse_with_real_logging_prefixes(rank: int) -> None:
    templates = json.loads((FIXTURES/'logger-templates.json').read_text())
    # Arm messages are source-rendered, not claimed captured from production.
    prefix = f'(VllmWorker-{rank} pid=244) INFO 10-27 06:07:50 [rdma.py:108] '
    pin_line = str(templates['pin']['template']) % (244, 19, h.CPUS)
    proxy_line = str(templates['proxy']['template']) % (
        'capacity', 'o1rdma://10.202.28.1:18620?bind=10.202.28.2', 462, 45.1, 30.1)
    logs = h.log_evidence([prefix + rdma_line(rank), prefix + pin_line, prefix + proxy_line])
    assert logs['rdma'] == [rdma_line(rank)]
    assert logs['pinned_pids'] == [244]
    assert logs['proxy_pids'] == [462]


@pytest.mark.parametrize('line', [
    'EXL3 RDMA all-reduce: rank 0 of 2 on rocep1s0f1,roceP2p1s0f1, gid 3, proxy on cpu 14, fixture',
    'vllm_exl3_moe: pid 244, 19 threads pinned to CPUs nonsense (EXL3_CPU_PIN)',
    'through proxy process pid 462',
    'EXL3 RDMA all-reduce off for tp, NCCL stays: fixture',
])
def test_source_log_parsers_refuse_partial_invented_or_fallback_messages(line: str) -> None:
    with pytest.raises(ValueError, match='malformed|NCCL-only'):
        h.log_evidence([line])


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('fault', ['none', 'unreadable', 'missing-worker', 'shared-comm14'])
def test_actual_window_direct_collector_needs_no_pin_log_and_restores_on_census_failure(
        rank: int, fault: str, rehearsal: tuple[w.Live, Remote], monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake = rehearsal
    live.group = 'cpu-pin'
    original = fake.remote

    class Logs:
        def __init__(self) -> None:
            self.stdout = io.StringIO('')  # Actual first-registration seam: NO pin log.

        def wait(self) -> int:
            return 0

    def remote(label: str, host: str, command: str, stdin: str | None) -> str:
        if '-host-gate-' not in label:
            return original(label, host, command, stdin)
        current_rank = pf.HOSTS.index(host)
        fixture = ProcFixture(current_rank, pinned=True)
        if current_rank == rank:
            fixture.fault = fault
            worker = next(row for row in fixture.rows if row['Pid'] == str(fixture.worker))
            if fault == 'missing-worker': fixture.rows.remove(worker)
            elif fault == 'shared-comm14':
                thread = next(t for t in cast(list[dict[str, Json]], worker['threads'])
                    if t['Pid'] != worker['Pid'] and t['Name'] == 'VLLM::Worker_TP')
                thread['Cpus_allowed_list'] = '14'

        def output(argv: list[str], **kwargs: object) -> str:
            if argv == ['docker', 'inspect', 'dsv41-serve']:
                return json.dumps([{'Id': live.ids[current_rank], 'State': {'Running': True}}])
            if argv[:4] == ['docker', 'exec', 'dsv41-serve', 'cat'] and argv[4].startswith('/run/up3-rdma-pin/'):
                assert kwargs['timeout'] == 15
                return fixture.receipt(fixture.worker, str(fixture.worker + 1000))
            assert argv == ['docker', 'exec', 'dsv41-serve', 'cat', '/serve.sh']
            return serve_script()

        with patch.object(subprocess, 'check_output', output), patch.object(subprocess, 'Popen', lambda *a, **k: Logs()), \
                patch.object(h.ProductionReads, 'proc', lambda self, paths, sparse=False: fixture.cat(paths, sparse)):
            return json.dumps(h.collect('cpu-pin', current_rank, live.ids[current_rank]))

    monkeypatch.setattr(fake, 'remote', remote)
    result = w.window(live, 'cpu-pin', lambda row: None)
    starts = record(result['starts'])
    if fault == 'none':
        assert result['status'] == 'COMPLETE', result
        assert record(result['decision'])['status'] == 'CANDIDATE FOR REVIEW'
        for host in pf.HOSTS:
            gates = record(record(record(starts['s2-cpu-pin'])['gates'])[host])
            assert record(record(gates['affinity'])['log_corroboration']) == {'pinned_pids': [], 'proxy_pids': []}
    else:
        assert result['status'] == 'INCOMPLETE', result
        assert record(result['decision'])['status'] == 'NOT A CANDIDATE'
        assert record(starts['s2-cpu-pin'])['cells'] == {}
        assert not any('s2-cpu-pin-'+cell in fake.commands for cell in cells.CELLS)
        assert 's3-control' not in starts
        if fault == 'shared-comm14': assert 'proven RDMA poller identity' in str(result['stop'])
    assert str(record(result['restore'])['status']).startswith('RESTORED'), result
    assert fake.restores == ['start', 'check', 'close'] and not fake.hold
