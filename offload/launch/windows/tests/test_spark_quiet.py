"""S54 FIX1: file/in-memory Docker and GPU fixtures; no real daemon or CUDA."""
from __future__ import annotations

import json
import subprocess
import time
from typing import cast

import pytest

from offload.launch.windows import spark_quiet as q
from offload.launch.windows.tests.test_post_close_watch import Clock

CID = 'a' * 64


@pytest.mark.parametrize('marker', ['NBW EXPERIMENT ABSENT', 'WATCH SPARK QUIET'])
@pytest.mark.parametrize('command', [
    "from vllm_exl3_moe import tf_experts as t, mrow; t._ext(); mrow.ext()",
    "import vllm.models.deepseek_v41.common.engram as e; e._load_engram_remote_ext()",
])
@pytest.mark.parametrize('pause', ['import', 'build-lock', 'active-CUDA'])
def test_anonymous_jit_waits_until_original_finishes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    marker: str, command: str, pause: str,
) -> None:
    clock = Clock()
    monkeypatch.setattr(time, 'time', clock)
    monkeypatch.setattr(time, 'sleep', clock.sleep)
    calls: list[list[str]] = []

    def checked(argv: list[str], timeout: float) -> str:
        calls.append(argv)
        assert timeout > 0
        if argv[1] == 'ps':
            return CID if clock() < 10 else ''
        if argv[1] == 'inspect':
            assert argv[2] == CID
            return json.dumps([{'Id': CID, 'Name': '/patient_babbage', 'Path': 'python3',
                'Args': ['-c', command], 'Config': {'Cmd': ['-c', command]},
                'State': {'Status': 'running'}}])
        assert argv[0] == 'nvidia-smi'
        return '123, python3' if pause == 'active-CUDA' and clock() < 10 else ''

    monkeypatch.setattr(q, 'checked', checked)
    q.wait(20, marker)
    assert clock() == 10
    assert capsys.readouterr().out.strip() == marker
    assert all(argv[1] in ('ps', 'inspect', '--query-compute-apps=pid,process_name') for argv in calls)


@pytest.mark.parametrize('marker', ['NBW EXPERIMENT ABSENT', 'WATCH SPARK QUIET'])
@pytest.mark.parametrize('failure', ['enumerate', 'inspect', 'gpu', 'malformed', 'identity'])
def test_unproved_docker_or_gpu_state_keeps_hold(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], marker: str, failure: str,
) -> None:
    def checked(argv: list[str], timeout: float) -> str:
        if argv[1] == 'ps':
            if failure == 'enumerate':
                raise subprocess.CalledProcessError(1, argv)
            return CID
        if argv[1] == 'inspect':
            if failure == 'inspect':
                raise subprocess.CalledProcessError(1, argv)
            if failure == 'malformed':
                return '{}'
            return json.dumps([{'Id': 'b' * 64 if failure == 'identity' else CID,
                'Name': '/unrelated', 'Path': 'sleep', 'Args': ['1'], 'Config': {},
                'State': {'Status': 'running'}}])
        raise subprocess.CalledProcessError(1, argv)
    monkeypatch.setattr(q, 'checked', checked)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        q.wait(time.time() + 10, marker)
    assert not capsys.readouterr().out


def test_unrelated_container_untouched_and_compiler_never_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    monkeypatch.setattr(time, 'time', clock)
    monkeypatch.setattr(time, 'sleep', clock.sleep)
    row: dict[str, object] = {'Id': CID, 'Name': '/unrelated', 'Path': 'sleep', 'Args': ['300'],
                              'Config': {}, 'State': {'Status': 'running'}}
    def checked(argv: list[str], timeout: float) -> str:
        if argv[1] == 'ps':
            return CID
        if argv[1] == 'inspect':
            return json.dumps([row])
        assert argv[0] == 'nvidia-smi'
        return ''
    monkeypatch.setattr(q, 'checked', checked)
    assert q.snapshot(10)
    cast(dict[str, object], row['Config'])['Env'] = ['DSV41_ENGRAM_OVERLAY=/reviewed/engram.py']
    with pytest.raises(ValueError, match='exhausted'):
        q.wait(12, 'must not pass')
    assert clock() == 12 and row['State'] == {'Status': 'running'}


@pytest.mark.parametrize('status,quiet', [('exited', True), ('dead', True), ('running', False), ('created', False),
                                          ('paused', False), ('restarting', False)])
@pytest.mark.parametrize('name', ['/dsv41-serve-stopped-20261022', '/dsv41-serve', '/dsv41-exp-nbw'])
def test_only_live_experiment_or_serve_containers_block_restore(
    monkeypatch: pytest.MonkeyPatch, status: str, quiet: bool, name: str,
) -> None:
    # DEP1-W-20261027T020407Z: an exited 10-22 production container blocked W8 and left production down.
    def checked(argv: list[str], timeout: float) -> str:
        if argv[1] == 'ps':
            return CID
        if argv[1] == 'inspect':
            return json.dumps([{'Id': CID, 'Name': name, 'Path': 'python3', 'Args': ['-m', 'vllm'],
                                'Config': {}, 'State': {'Status': status}}])
        return ''
    monkeypatch.setattr(q, 'checked', checked)
    if quiet:
        assert q.snapshot(10)
    else:
        with pytest.raises(ValueError, match='experiment/serve container remains'):
            q.snapshot(10)

