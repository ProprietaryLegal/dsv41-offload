"""Replay exact captured Docker/launcher shapes through actual finite readers."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shlex
import subprocess
from typing import cast

import pytest

from offload.pc1 import build as b, host_gates as h
from offload.prefill_maxlen.evidence import Json, record

FIXTURES = Path(__file__).parent / 'fixtures'
SHAPES = cast(dict[str, dict[str, Json]], json.loads((FIXTURES / 'production-host-shapes.json').read_text()))


@pytest.mark.parametrize('rank', [0, 1])
def test_production_probe_real_wrapper_and_settings(monkeypatch: pytest.MonkeyPatch, rank: int) -> None:
    host = ('sparkhead', 'sparkworker-wired')[rank]
    calls: list[list[str]] = []
    def output(argv: list[str], *, stderr: int, text: bool, errors: str, timeout: int) -> str:
        assert (stderr, text, errors, timeout) == (subprocess.STDOUT, True, 'replace', 15)
        assert argv[:6] == ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host]
        command = shlex.split(argv[-1])
        calls.append(command)
        if command == ['docker', 'inspect', 'dsv41-serve']:
            return json.dumps([SHAPES[host]['inspect']])
        assert command == ['docker', 'exec', 'dsv41-serve', 'cat', '/serve.sh']
        return (FIXTURES / f'production-serve-rank{rank}.sh').read_text()
    monkeypatch.setattr(subprocess, 'check_output', output)
    result = h.collect('control', rank, reads=h.ProductionReads(host), production=True)
    assert result['status'] == 'PARSE COMPLETE' and 'NOT QUALIFIED' in str(result['qualification'])
    assert result['container'] == record(SHAPES[host]['inspect'])['Id']
    assert calls == [['docker', 'inspect', 'dsv41-serve'],
        ['docker', 'exec', 'dsv41-serve', 'cat', '/serve.sh'], ['docker', 'inspect', 'dsv41-serve']]
    with pytest.raises(ValueError, match='invalid experiment'):
        h.validate(result, 'control', rank, str(result['container']))


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('arm', b.ARMS)
def test_actual_mounted_pairs_parse(rank: int, arm: str) -> None:
    script = (FIXTURES / f'production-serve-rank{rank}.sh').read_text().replace('--port 8000', '--port 8001')
    for key, old, value in zip(('--max-num-batched-tokens', '--long-prefill-token-threshold'),
                               b.SETTINGS['control'], b.SETTINGS[arm]):
        script = script.replace(f'{key} {old}', f'{key} {value}')
    result = h.settings(script, rank, '8001', arm)
    assert result['max-num-batched-tokens'] == str(b.SETTINGS[arm][0])


@pytest.mark.parametrize('fault', ['port8000', 'variable', 'equals', 'duplicate', 'missing', 'second', 'chunk8192'])
def test_malformed_or_wrong_mounted_flags_refuse(fault: str) -> None:
    script = (FIXTURES / 'production-serve-rank0.sh').read_text().replace('--port 8000', '--port 8001')
    if fault == 'port8000':
        script = script.replace('--port 8001', '--port 8000')
    elif fault == 'variable':
        script = script.replace('--port 8001', '--port "$PORT"')
    elif fault == 'equals':
        script = script.replace('--port 8001', '--port=8001')
    elif fault == 'duplicate':
        script = script.replace('--port 8001', '--port 8001 --port 8001')
    elif fault == 'missing':
        script = script.replace('--long-prefill-token-threshold 1024', '')
    elif fault == 'second':
        script += '\nexec vllm serve other --port 8001\n'
    else:
        script = script.replace('--max-num-batched-tokens 4096', '--max-num-batched-tokens 8192')
    with pytest.raises(ValueError):
        h.settings(script, 0, '8001', 'control')


@pytest.mark.parametrize('fault', ['env_duplicate', 'env_bad', 'stopped', 'wrong_id', 'wrapper'])
def test_real_inspect_failures_refuse(fault: str) -> None:
    row = copy.deepcopy(record(SHAPES['sparkhead']['inspect']))
    config = record(row['Config'])
    if fault == 'env_duplicate':
        config['Env'] = ['A=1', 'A=2']
    elif fault == 'env_bad':
        config['Env'] = ['broken']
    elif fault == 'stopped':
        record(row['State'])['Running'] = False
    elif fault == 'wrong_id':
        row['Id'] = 'changed'
    else:
        config['Cmd'] = ['bash', '/serve.sh']
    with pytest.raises(ValueError):
        h.inspect_running(json.dumps([row]), str(record(SHAPES['sparkhead']['inspect'])['Id']))


@pytest.mark.parametrize('command', [['docker', 'exec', 'dsv41-serve', 'cat', '/serve.sh; touch /x'],
    ['docker', 'logs', 'dsv41-serve'], ['docker', 'inspect', 'other'], ['curl', 'http://localhost:8000'],
    ['docker', 'exec', 'dsv41-serve', 'python3', '-c', 'print(1)'], ['nvidia-smi']])
def test_allowlist_denies_before_transport(monkeypatch: pytest.MonkeyPatch, command: list[str]) -> None:
    def denied(*args: object, **kwargs: object) -> str:
        raise AssertionError('transport reached')
    monkeypatch.setattr(subprocess, 'check_output', denied)
    with pytest.raises(ValueError, match='allowlist'):
        h.ProductionReads('sparkhead').output(command)


@pytest.mark.parametrize('fault', ['changed-id', 'disappeared', 'stopped', 'experiment-port'])
def test_actual_production_sequence_refuses_changed_state(fault: str) -> None:
    class Reads(h.ProductionReads):
        count = 0

        def output(self, argv: list[str]) -> str:
            if argv == ['docker', 'exec', 'dsv41-serve', 'cat', '/serve.sh']:
                script = (FIXTURES / 'production-serve-rank0.sh').read_text()
                return script.replace('--port 8000', '--port 8001') if fault == 'experiment-port' else script
            assert argv == ['docker', 'inspect', 'dsv41-serve']
            self.count += 1
            info = copy.deepcopy(record(SHAPES['sparkhead']['inspect']))
            if self.count == 2:
                if fault == 'changed-id':
                    info['Id'] = 'new-container'
                elif fault == 'disappeared':
                    raise subprocess.CalledProcessError(1, argv, 'no such container')
                else:
                    record(info['State'])['Running'] = False
            return json.dumps([info])
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        h.collect('control', 0, reads=Reads('sparkhead'), production=True)
