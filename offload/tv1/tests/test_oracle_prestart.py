"""Run the real receiving script and start hook against CPU command fixtures."""
from __future__ import annotations

import json
import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import cast
from types import SimpleNamespace

import pytest

from offload.prefill_maxlen.evidence import Json, record
from offload.tv1 import build_overlay, oracle_prestart as op
from offload.tv1.candidate_cases import EXCLUSIONS, case_matrix


def oracle() -> dict[str, Json]:
    return {'type': 'tv1-installed-oracle-v2', 'classification': 'MEASURED SM121',
            'all_identical': True, 'capability': [12, 1], 'exclusions': cast(Json, EXCLUSIONS),
            'cases': cast(Json, [{**s, 'eager_identical': True, 'replay_identical': True}
                                for s in case_matrix()])}


def packet() -> dict[str, Json]:
    return {'type': 'tv1-prestart-oracles-v1', 'run': 'tag',
            'ranks': {h: {'image': op.IMAGE, 'server_absent_before': True,
                          'compute_absent_after': True, 'oracle': oracle()}
                      for h in ('sparkhead', 'sparkworker-wired')}}


@pytest.fixture
def commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    binary = tmp_path / 'bin'
    binary.mkdir()
    code = '#!' + sys.executable + '''
import json,os,sys
from pathlib import Path
root=Path(os.environ['TV1_ORACLE_FIXTURE'])
args=sys.argv[1:]
name=Path(sys.argv[0]).name
with (root/'calls').open('a') as f: f.write(json.dumps([name,*args])+'\\n')
mode=os.environ.get('TV1_ORACLE_MODE','')
if name=='nvidia-smi':
 if mode=='busy': print('12345')
elif args[0]=='ps':
 if mode=='server' or (mode=='leftover' and (root/'ran').exists()): print('owned-container')
elif args[:2]==['image','inspect']:
 print('wrong' if mode=='image' else os.environ['TV1_ORACLE_IMAGE'])
elif args[0]=='run':
 (root/'ran').touch()
 (root/'stdin').write_text(sys.stdin.read())
 if mode=='cuda-failure':
  print('public CUDA allocation sentinel',file=sys.stderr)
  raise SystemExit(1)
 print((root/'oracle.json').read_text())
else: raise SystemExit(9)
'''
    for name in ('docker', 'nvidia-smi'):
        path = binary / name
        path.write_text(code)
        path.chmod(0o700)
    (tmp_path / 'oracle.json').write_text(json.dumps(oracle()))
    monkeypatch.setenv('PATH', str(binary) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('TV1_ORACLE_FIXTURE', str(tmp_path))
    monkeypatch.setenv('TV1_ORACLE_IMAGE', op.IMAGE)
    return tmp_path


def receiving() -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, '-c', op.remote_script(
        '-v $HOME/overlay/file:/engine/file:ro -v $HOME/profile:/sp1-profile:rw',
        'VLLM_TV1_CANDIDATE=1 VLLM_TV1_COMPACT=0')], input='public installed-code sentinel',
        capture_output=True, text=True, timeout=10)


def test_receiving_isolated_image_mounts_and_exit_proof(commands: Path) -> None:
    result = receiving()
    assert result.returncode == 0, result.stderr
    rank = json.loads(result.stdout)
    assert rank == record(packet()['ranks'])['sparkhead']
    calls = [json.loads(x) for x in (commands / 'calls').read_text().splitlines()]
    assert [c[0] for c in calls] == ['docker', 'nvidia-smi', 'docker', 'docker', 'docker', 'nvidia-smi']
    argv = calls[3]
    assert argv[:12] == ['docker', 'run', '--rm', '--name', 'dsv41-serve', '--gpus', 'all',
                         '--ipc', 'host', '--network', 'none', '-i']
    assert str(commands / 'overlay/file') + ':/engine/file:ro' in argv
    assert argv[-4:] == ['--entrypoint', 'python3', op.IMAGE, '-']
    assert (commands / 'stdin').read_text() == 'public installed-code sentinel'


@pytest.mark.parametrize('mode', ['server', 'busy', 'image', 'cuda-failure', 'leftover'])
def test_receiving_refuses_competition_failure_or_leftover(
        commands: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setenv('TV1_ORACLE_MODE', mode)
    result = receiving()
    assert result.returncode != 0
    assert (commands / 'ran').exists() == (mode in ('cuda-failure', 'leftover'))
    if mode == 'cuda-failure':
        assert 'public CUDA allocation sentinel' in result.stderr


@pytest.mark.parametrize('change', ['rank', 'run', 'image', 'absence', 'identity', 'case'])
def test_both_rank_receipt_cannot_waive_existing_exact_gate(change: str) -> None:
    value = packet()
    ranks = record(value['ranks'])
    rank = record(ranks['sparkhead'])
    if change == 'rank':
        del ranks['sparkworker-wired']
    elif change == 'run':
        value['run'] = 'stale'
    elif change == 'image':
        rank['image'] = 'other'
    elif change == 'absence':
        rank['compute_absent_after'] = False
    else:
        rows = cast(list[Json], record(rank['oracle'])['cases'])
        if change == 'identity':
            record(rows[0])['eager_identical'] = False
        else:
            rows.pop()
    with pytest.raises(ValueError, match='TV1'):
        op.validate(value, 'tag')


@pytest.mark.parametrize('role,failed', [('candidate', False), ('candidate', True), ('off', False)])
def test_actual_launcher_hook_after_stop_staging_before_serve(
        tmp_path: Path, role: str, failed: bool) -> None:
    root = tmp_path / 'checkout'
    launch = root / 'offload/launch'
    launch.mkdir(parents=True)
    (launch / 'start_exp.sh').write_bytes((build_overlay.ROOT / 'offload/launch/start_exp.sh').read_bytes())
    (tmp_path / 'dsv41-deploy').mkdir()
    (tmp_path / 'dsv41-deploy/HOLD').touch()
    (root / 'dsv41.env').write_text('HEAD=sparkhead\nWORKER=sparkworker\nIMAGE=public\nRECIPE=public\nMODEL_DIR=/srv/dsv41/legal-v2a\n')
    (root / 'config').write_text(f'TV1_ROLE={role}\nOVERLAYS="overlay=/engine/file"\n')
    (root / 'overlay').write_text('public bytes')
    for path, body in ((root / 'stop.sh', 'echo stop >> "$HOME/events"'),
                       (root / 'firewall.sh', ':'),
                       (launch / 'serve_exp.sh', 'echo serve >> "$HOME/events"')):
        path.write_text('#!/bin/bash\n' + body + '\n')
        path.chmod(0o700)
    binary = tmp_path / 'bin'
    binary.mkdir()
    for name, body in {'ssh': ':', 'scp': 'echo stage >> "$HOME/events"', 'sudo': ':',
        'python3': 'echo oracle >> "$HOME/events"\nexit ' + str(int(failed))}.items():
        path = binary / name
        path.write_text('#!/bin/bash\n' + body + '\n')
        path.chmod(0o700)
    result = subprocess.run(['bash', str(launch / 'start_exp.sh'), 'config'],
        env={**os.environ, 'HOME': str(tmp_path), 'PATH': str(binary) + os.pathsep + os.environ['PATH']},
        capture_output=True, text=True, timeout=10)
    events = (tmp_path / 'events').read_text().splitlines()
    assert events == ['stop', 'stage', 'stage'] + (['oracle'] if role == 'candidate' else []) + ([] if failed else ['serve']), result.stderr
    assert result.returncode == int(failed), result.stderr


def test_oracle_high_offset_arithmetic_and_installed_imports_unchanged() -> None:
    source = (build_overlay.HERE / 'gb10_oracle.py').read_text()
    spec = case_matrix()[0]
    count = (spec['width'] + spec['page'] - 1) // spec['page']
    span = (spec['high_page'] + count) * spec['stride'] + spec['cache_offset'] + spec['page'] * 132
    assert span == 2401735488
    assert spec['high_page'] * spec['stride'] + spec['cache_offset'] > 2**31
    assert 'from vllm_exl3_moe.tv1_candidate_metadata import score_decode' in source
    assert 'get_device_capability() != (12, 1)' in source
    assert 'torch.equal(a, b)' in source
    assert 'free_bytes < storage_bytes + 512 * 1024**2' in source


@pytest.mark.parametrize('failure', ['', 'free', 'allocation'])
def test_actual_oracle_storage_precheck_and_clear_error(failure: str) -> None:
    tree = ast.parse((build_overlay.HERE / 'gb10_oracle.py').read_text())
    main = next(s for s in tree.body if isinstance(s, ast.FunctionDef) and s.name == 'main')
    start = next(i for i, s in enumerate(main.body) if isinstance(s, ast.Assign)
                 and isinstance(s.targets[0], ast.Name) and s.targets[0].id == 'matrix')
    end = next(i for i, s in enumerate(main.body) if isinstance(s, ast.Assign)
               and isinstance(s.targets[0], ast.Name) and s.targets[0].id == 'cache')
    code = compile(ast.Module(body=main.body[start:end], type_ignores=[]), '<real-oracle-allocation>', 'exec')
    calls: list[int] = []
    class AcceleratorError(RuntimeError):
        pass
    class OutOfMemoryError(AcceleratorError):
        pass
    def empty(size: int, *, device: str, dtype: object) -> object:
        assert device == 'cuda'
        calls.append(size)
        if failure == 'allocation':
            raise AcceleratorError('original allocation sentinel')
        return object()
    fake = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (
        1 if failure == 'free' else 4 * 1024**3, 128 * 1024**3)), empty=empty, uint8=object(),
        OutOfMemoryError=OutOfMemoryError, AcceleratorError=AcceleratorError)
    namespace = {'torch': fake, 'case_matrix': case_matrix}
    if failure:
        with pytest.raises(ValueError, match='TV1.*(2401735488|contiguous).*') as caught:
            exec(code, namespace)
        if failure == 'allocation':
            assert isinstance(caught.value.__cause__, AcceleratorError)
    else:
        exec(code, namespace)
    assert calls == ([] if failure == 'free' else [2401735488])


@pytest.mark.parametrize('failed_rank', ['', 'sparkworker-wired'])
def test_prestart_main_hashes_installed_bytes_and_requires_both_before_receipt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_rank: str) -> None:
    tree = tmp_path / 'candidate'
    manifest = build_overlay.build(tree, ('candidate',))
    tree.with_suffix('.manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(build_overlay, 'ROOT', tmp_path)
    monkeypatch.setenv('HOME', str(tmp_path))
    (tmp_path / 'dsv41-offload-run').mkdir()
    for key, value in {'TV1_TREE': 'candidate', 'TV1_ROLE': 'candidate', 'TV1_ORACLE_RUN': 'tag',
                       'SCOPED_REMOTE_DEADLINE': '2000000000', 'EXTRA_DOCKER_ARGS': '',
                       'EXP_DOCKER_ENV': 'VLLM_TV1_CANDIDATE=1 VLLM_TV1_COMPACT=0'}.items():
        monkeypatch.setenv(key, value)
    hosts: list[str] = []
    def fake(argv: list[str], *, input: str, capture_output: bool, text: bool,
             timeout: int) -> subprocess.CompletedProcess[str]:
        assert argv[:3] == ['ssh', '-o', 'BatchMode=yes'] and timeout == 120
        assert capture_output and text
        host = argv[3]
        hosts.append(host)
        compile(input, '<installed-oracle-payload>', 'exec')
        for entry in manifest['files']:
            assert entry['sha256'] in input and build_overlay.PREFIX + entry['path'] in input
        assert input.index('runtime source hash mismatch') < input.index('import torch')
        assert 'from vllm_exl3_moe.tv1_candidate_metadata import score_decode' in input
        rank = record(record(packet()['ranks'])[host])
        return subprocess.CompletedProcess(argv, 1 if host == failed_rank else 0, json.dumps(rank), 'sentinel')
    monkeypatch.setattr(subprocess, 'run', fake)
    if failed_rank:
        with pytest.raises(ValueError, match='sparkworker-wired rc1'):
            op.main()
        assert not op.receipt_path('tag').exists()
    else:
        op.main()
        op.validate(json.loads(op.receipt_path('tag').read_text()), 'tag')
        with pytest.raises(ValueError, match='stale/repeated'):
            op.main()
    assert hosts == ['sparkhead', 'sparkworker-wired']
