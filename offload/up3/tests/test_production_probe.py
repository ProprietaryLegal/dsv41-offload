"""Real production probe plumbing over an inert, command-recording SSH transport."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import cast

import pytest

from offload.prefill_maxlen.evidence import Json, record
from offload.up3 import host_gates as h
from offload.up3.tests.test_proc_census import ProcFixture

FIXTURES = Path(__file__).with_name('fixtures')


@pytest.fixture
def transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, dict[str, Json]]]:
    shapes = json.loads((FIXTURES/'production-host-shapes.json').read_text())
    absence = json.loads((FIXTURES/'production-rdma-absence.json').read_text())
    responses: dict[str, dict[str, Json]] = {}
    for rank, host in enumerate(('sparkhead', 'sparkworker-wired')):
        # Explicit combination of prior captured launcher/process shapes and
        # Round8's captured mount destinations; not a single live receipt.
        shapes[host]['inspect']['Mounts'] = [{'Destination': name} for name in absence[host]['mount_destinations']]
        responses[host] = {
            'docker inspect dsv41-serve': json.dumps([shapes[host]['inspect']]),
            'docker exec dsv41-serve cat /serve.sh': (FIXTURES/f'production-serve-rank{rank}.sh').read_text(),
            'docker logs dsv41-serve': (FIXTURES/f'production-logs-{host}.txt').read_text(),
            'docker exec dsv41-serve cat ' + h.RDMA_SOURCE: Path('/opt/dsv41/data/src-image/vllm_exl3_moe/rdma.py').read_text(),
            **{'cat ' + path: value + '\n' for path, value in shapes[host]['sysfs'].items()}}
    mapping = tmp_path/'responses.json'
    mapping.write_text(json.dumps(responses))
    log = tmp_path/'calls.jsonl'
    ssh = tmp_path/'ssh'
    ssh.write_text('#!/home/operator/miniconda3/bin/python3\n' + '''import json, os, sys
from pathlib import Path
root=Path(os.environ['UP3_PROBE_FAKE'])
assert sys.argv[1:5] == ['-o','BatchMode=yes','-o','ConnectTimeout=10']
host, command=sys.argv[5:]
with (root/'calls.jsonl').open('a') as handle: handle.write(json.dumps([host,command])+'\\n')
import shlex
parts=shlex.split(command)
if parts[:4]==['docker','exec','dsv41-serve','cat'] and all(path.startswith('/proc/') for path in parts[4:]):
 from offload.up3.tests.test_proc_census import ProcFixture
 fixture=ProcFixture(('sparkhead','sparkworker-wired').index(host))
 print(fixture.cat(parts[4:],all(path.endswith('/status') for path in parts[4:])),end='')
 raise SystemExit(0)
response=json.loads((root/'responses.json').read_text())[host][command]
if isinstance(response,dict):
 print(response['error']); raise SystemExit(17)
# An optional second inspect response tests an ID change during collection.
if isinstance(response,list):
 calls=[json.loads(line) for line in (root/'calls.jsonl').read_text().splitlines()]
 response=response[min(sum(row==[host,command] for row in calls)-1,len(response)-1)]
print(response,end='')
''')
    ssh.chmod(0o700)
    monkeypatch.setenv('PATH', str(tmp_path) + ':' + os.environ['PATH'])
    monkeypatch.setenv('UP3_PROBE_FAKE', str(tmp_path))
    return log, responses


def publish(log: Path, responses: dict[str, dict[str, Json]]) -> None:
    (log.parent/'responses.json').write_text(json.dumps(responses))


@pytest.mark.parametrize('rank,host', [(0, 'sparkhead'), (1, 'sparkworker-wired')])
def test_full_probe_real_plumbing_is_only_allowed_reads(rank: int, host: str,
        transport: tuple[Path, dict[str, dict[str, Json]]]) -> None:
    log, _ = transport
    result = h.probe_production(rank, h.ProductionReads(host))
    assert result['status'] == 'PARSE COMPLETE'
    assert 'NOT QUALIFIED' in str(result['qualification'])
    assert record(result['serve']) == {'status': 'PARSED', 'port': '8000', 'node_rank': rank}
    assert record(result['inspect'])['identity_rechecked'] is True
    assert str(record(result['startup_overlay'])['status']).startswith('ABSENT:')
    assert record(result['startup_overlay'])['overlay_mounts'] == []
    assert record(result['logs'])['proxy_pids'] == ([462] if rank == 0 else [])
    assert 'EXPECTED ABSENT' in str(record(result['arm_lines'])['cpu_pin'])
    assert record(result['arm_lines'])['rdma_init'] == 'OBSERVED'
    assert 'EXPECTED ABSENT' in str(record(result['arm_lines'])['dual_ring'])
    assert 'two bounded PID-namespace sweeps' in str(record(result['affinity'])['discovery'])
    assert 'no unique source-defined name' in str(record(result['affinity'])['poller_identity'])
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) > 10
    commands = [command for called_host, command in calls if called_host == host]
    assert len(commands) == len(calls)
    assert commands.count('docker inspect dsv41-serve') == 2
    assert commands.count('docker logs dsv41-serve') == 1
    assert commands.count('docker exec dsv41-serve cat /serve.sh') == 1
    assert commands.count('docker exec dsv41-serve cat ' + h.RDMA_SOURCE) == 1
    assert not any('/run/up3-rdma-pin/' in command for command in commands)
    assert all(shlex.split(command)[0] in ('cat', 'docker') for command in commands)
    assert len(cast(list[Json], result['gids'])) == 2


@pytest.mark.parametrize('fault', ['experiment-port', 'wrapper', 'container-changed', 'stopped', 'fallback', 'malformed-env'])
def test_probe_refuses_wrong_identity_or_logs(fault: str, transport: tuple[Path, dict[str, dict[str, Json]]]) -> None:
    log, responses = transport
    row = responses['sparkhead']
    if fault == 'experiment-port':
        row['docker exec dsv41-serve cat /serve.sh'] = str(row['docker exec dsv41-serve cat /serve.sh']).replace('--port 8000', '--port 8001')
    elif fault == 'fallback': row['docker logs dsv41-serve'] = 'RDMA all-reduce off, NCCL stays\n'
    else:
        info = json.loads(str(row['docker inspect dsv41-serve']))
        if fault == 'wrapper': info[0]['Config']['Cmd'] = ['--port', '8000']
        elif fault == 'stopped': info[0]['State']['Running'] = False
        elif fault == 'malformed-env': info[0]['Config']['Env'] = ['BAD']
        else: info[0]['Id'] = 'changed'
        row['docker inspect dsv41-serve'] = ([row['docker inspect dsv41-serve'], json.dumps(info)]
            if fault == 'container-changed' else json.dumps(info))
    publish(log, responses)
    with pytest.raises(ValueError):
        h.probe_production(0, h.ProductionReads('sparkhead'))
    assert '8001/health' not in log.read_text()


@pytest.mark.parametrize('fault', ['missing', 'fabric'])
def test_probe_reports_failed_gid_field_and_cli_exits_nonzero(fault: str,
        transport: tuple[Path, dict[str, dict[str, Json]]], capsys: pytest.CaptureFixture[str]) -> None:
    log, responses = transport
    key = '/sys/class/infiniband/roceP2p1s0f1/ports/1/gids/3'
    responses['sparkhead']['cat ' + key] = ({'error': 'missing sysfs file'} if fault == 'missing'
        else '0000:0000:0000:0000:0000:ffff:0aca:1d02\n')
    publish(log, responses)
    with pytest.raises(SystemExit) as exc:
        h.main(['--probe-production', '--rank', '0', '--host', 'sparkhead'])
    assert exc.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'PARSE FAILURE'
    assert result['errors'][0]['device'] == 'roceP2p1s0f1'
    assert 'NOT QUALIFIED' in result['qualification']
    assert len(result['gids']) == 1


@pytest.mark.parametrize('argv', [
    ['--probe-production', '--rank', '0', '--arm', 'cpu-pin'],
    ['--probe-production', '--rank', '0', '--container', 'fixture'],
    ['--arm', 'dual-ring', '--rank', '0', '--container', 'fixture', '--host', 'sparkhead'],
])
def test_cli_modes_cannot_bypass_experiment_gate(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*args: object, **kwargs: object) -> str:
        raise AssertionError('invalid mode reached transport')
    monkeypatch.setattr(subprocess, 'check_output', denied)
    with pytest.raises(SystemExit):
        h.main(argv)


@pytest.mark.parametrize('argv', [
    ['docker', 'exec', 'dsv41-serve', 'python3', '-c', 'print(1)'],
    ['docker', 'stop', 'dsv41-serve'], ['curl', 'http://localhost:8000/health'],
    ['cat', '/proc/1/stat'], ['cat', '/sys/class/infiniband/rocep1s0f0/ports/1/gids/3'],
])
def test_read_allowlist_denies_every_other_command_before_transport(argv: list[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*args: object, **kwargs: object) -> str:
        raise AssertionError('unsafe command reached transport')
    monkeypatch.setattr(subprocess, 'check_output', denied)
    with pytest.raises(ValueError, match='allowlist'):
        h.ProductionReads('sparkhead').output(argv)


def test_host_rank_mismatch_refuses_before_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*args: object, **kwargs: object) -> str:
        raise AssertionError('wrong rank reached transport')
    monkeypatch.setattr(subprocess, 'check_output', denied)
    with pytest.raises(ValueError, match='host/rank'):
        h.probe_production(0, h.ProductionReads('sparkworker-wired'))
    with pytest.raises(ValueError, match='registered Spark'):
        h.ProductionReads('x86host')


@pytest.mark.parametrize('field', ['none', 'cpu-pin', 'dual-ring'])
def test_missing_arm_init_is_reported_without_claiming_qualification(field: str,
        transport: tuple[Path, dict[str, dict[str, Json]]]) -> None:
    log, responses = transport
    row = responses['sparkhead']
    row['docker logs dsv41-serve'] = '\n'.join(line for line in str(row['docker logs dsv41-serve']).splitlines()
        if 'EXL3 RDMA all-reduce: rank' not in line)
    info = json.loads(str(row['docker inspect dsv41-serve']))
    if field == 'cpu-pin': info[0]['Config']['Env'].append('EXL3_CPU_PIN=5-9,15-19')
    elif field == 'dual-ring': info[0]['Config']['Env'] = ['EXL3_RDMA_DEVICES=' + ','.join(h.DEVICES)]
    row['docker inspect dsv41-serve'] = json.dumps(info)
    publish(log, responses)
    result = h.probe_production(0, h.ProductionReads('sparkhead'))
    assert result['status'] == 'PARSE COMPLETE'
    assert 'NOT QUALIFIED' in str(result['qualification'])
    assert str(record(result['arm_lines'])['rdma_init']).startswith('ABSENT:')
    if field != 'none':
        key = 'cpu_pin' if field == 'cpu-pin' else 'dual_ring'
        assert str(record(result['arm_lines'])[key]).startswith('ABSENT:')
        assert 'configured' in str(record(result['arm_lines'])[key])
    with pytest.raises(ValueError, match='arm/rank/container'):
        h.validate(result, 'dual-ring', 0, str(result['container']))


def test_local_spark_mode_uses_same_reads_without_ssh(transport: tuple[Path, dict[str, dict[str, Json]]],
        monkeypatch: pytest.MonkeyPatch) -> None:
    _, responses = transport
    calls: list[list[str]] = []
    def output(argv: list[str], **kwargs: object) -> str:
        calls.append(argv)
        assert kwargs['timeout'] == 15
        return str(responses['sparkhead'][shlex.join(argv)])
    monkeypatch.setattr(subprocess, 'check_output', output)
    fixture = ProcFixture(0)
    monkeypatch.setattr(h.ProductionReads, 'proc', lambda self, paths, sparse=False: fixture.cat(paths, sparse))
    result = h.probe_production(0, h.ProductionReads())
    assert result['status'] == 'PARSE COMPLETE' and result['host'] == 'local Spark'
    assert len(calls) == 11 and all(call[0] != 'ssh' for call in calls)


def test_disappearing_container_read_stops_before_other_probes(transport: tuple[Path, dict[str, dict[str, Json]]]) -> None:
    log, responses = transport
    responses['sparkhead']['docker inspect dsv41-serve'] = {'error': 'No such object: dsv41-serve'}
    publish(log, responses)
    with pytest.raises(subprocess.CalledProcessError):
        h.probe_production(0, h.ProductionReads('sparkhead'))
    assert [json.loads(line) for line in log.read_text().splitlines()] == [['sparkhead', 'docker inspect dsv41-serve']]


@pytest.mark.parametrize('fault', ['changed-source', 'rdma-mount', 'helper-mount', 'parent-mount', 'receipt-mount', 'missing-mounts'])
def test_probe_proves_experiment_startup_overlay_absent(fault: str,
        transport: tuple[Path, dict[str, dict[str, Json]]]) -> None:
    log, responses = transport
    row = responses['sparkhead']
    if fault == 'changed-source':
        row['docker exec dsv41-serve cat ' + h.RDMA_SOURCE] = 'experiment source\n'
    else:
        info = json.loads(str(row['docker inspect dsv41-serve']))
        if fault == 'missing-mounts': info[0].pop('Mounts')
        else:
            destination = {'rdma-mount': h.RDMA_SOURCE,
                'helper-mount': str(Path(h.RDMA_SOURCE).with_name('up3_rdma_pin.py')),
                'parent-mount': str(Path(h.RDMA_SOURCE).parent), 'receipt-mount': '/run'}[fault]
            info[0]['Mounts'].append({'Destination': destination})
        row['docker inspect dsv41-serve'] = json.dumps(info)
    publish(log, responses)
    with pytest.raises((ValueError, KeyError)):
        h.probe_production(0, h.ProductionReads('sparkhead'))
    assert '/run/up3-rdma-pin/' not in log.read_text()
    assert 'docker logs dsv41-serve' not in log.read_text()


@pytest.mark.parametrize('host', [None, 'sparkhead', 'sparkworker-wired'])
@pytest.mark.parametrize('fault', ['none', 'missing', 'permission', 'ssh', 'unexpected-error'])
def test_proc_cat_transport_accepts_only_expected_discovery_gaps(host: str | None, fault: str,
        monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        assert kwargs == {'capture_output': True, 'text': True, 'errors': 'strict', 'timeout': 15}
        rc, err = (0, '') if fault == 'none' else (1, 'cat: /proc/2/status: No such file or directory\n')
        if fault == 'permission': err = 'cat: /proc/2/status: Permission denied\n'
        elif fault == 'ssh': rc = 255
        elif fault == 'unexpected-error': err = 'cat: /etc/passwd: No such file or directory\n'
        return subprocess.CompletedProcess(argv, rc, 'projected-status', err)
    monkeypatch.setattr(subprocess, 'run', run)
    if fault in ('none', 'missing'):
        assert h.ProductionReads(host).proc(['/proc/1/status', '/proc/2/status'], True) == 'projected-status'
    else:
        with pytest.raises(ValueError, match='required /proc read failed'):
            h.ProductionReads(host).proc(['/proc/1/status', '/proc/2/status'], True)
    actual = calls[0] if host is None else shlex.split(calls[0][-1])
    assert actual == ['docker', 'exec', 'dsv41-serve', 'cat', '/proc/1/status', '/proc/2/status']


def test_required_missing_file_cannot_use_sparse_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, 'run', lambda argv, **kwargs: subprocess.CompletedProcess(
        argv, 1, '', 'cat: /proc/1/task/1/stat: No such file or directory\n'))
    with pytest.raises(ValueError, match='required /proc read failed'):
        h.ProductionReads().proc(['/proc/1/task/1/stat'])


@pytest.mark.parametrize('paths,sparse', [(['/etc/passwd'], False), (['/proc/1/status;true'], False),
    (['/proc/1/environ'], False), (['/proc/1/task/1/stat'], True), ([], False)])
def test_proc_allowlist_refuses_before_transport(paths: list[str], sparse: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*args: object, **kwargs: object) -> None:
        raise AssertionError('disallowed /proc command reached transport')
    monkeypatch.setattr(subprocess, 'run', denied)
    with pytest.raises(ValueError):
        h.ProductionReads().proc(paths, sparse)


def test_production_census_failure_is_reported_as_parse_failure(transport: tuple[Path, dict[str, dict[str, Json]]],
        monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*args: object, **kwargs: object) -> str:
        raise PermissionError('required worker unreadable')
    monkeypatch.setattr(h.ProductionReads, 'proc', denied)
    result = h.probe_production(0, h.ProductionReads('sparkhead'))
    assert result['status'] == 'PARSE FAILURE'
    assert record(result['affinity'])['status'] == 'UNMEASURED'
    assert 'PermissionError' in str(result['errors'])
    assert 'NOT QUALIFIED' in str(result['qualification'])
