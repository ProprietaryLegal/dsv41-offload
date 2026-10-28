"""Execute emitted prerequisite shells on a private CPU-only fake X86host; no network, unit or GPU action."""
from __future__ import annotations

import ast
import fcntl
import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from offload.launch.windows import nbw_prerequisites as p
from offload.launch.windows import nbw_window as n


def run(script: str, home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(['bash', '-c', script, 'nbw-test', *args], text=True, capture_output=True,
                          env={**os.environ, 'HOME': str(home), 'PATH': str(home / 'bin') + ':' + os.environ['PATH'],
                               'WINDOW_COORDINATOR': 'Deepseek/coordinator',
                               'NBW_REVIEWED_CACHE_ALTERNATIVE': 'E-TP2-937e9633', 'CUDA_VISIBLE_DEVICES': ''}, timeout=20)


@pytest.fixture
def node(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    home = tmp_path / 'x86host'
    (home / '.config/ndm').mkdir(parents=True)
    (home / '.config/ndm/paths.sh').write_text(':\n')
    (home / 'bin').mkdir()
    # Executing production/GPU/network commands is always a test failure. Solo overrides ONLY nvidia-smi below.
    for name in ('ssh', 'scp', 'curl', 'systemctl', 'nvidia-smi'):
        path = home / 'bin' / name
        path.write_text('#!/bin/sh\necho "FORBIDDEN CPU TEST COMMAND" >&2\nexit 97\n')
        path.chmod(0o700)
    repo = tmp_path / 'bundle-source'
    repo.mkdir()
    git = ['git', '-C', str(repo), '-c', 'user.name=NBW-test', '-c', 'user.email=test@invalid']
    subprocess.run([*git, 'init', '-q'], check=True)
    (repo / 'source.py').write_text('# public synthetic source only\n')
    subprocess.run([*git, 'add', 'source.py'], check=True)
    subprocess.run([*git, 'commit', '-qm', 'synthetic fixture'], check=True)
    commit = subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip()
    bundle = tmp_path / 'reviewed.bundle'
    subprocess.run([*git, 'bundle', 'create', str(bundle), 'HEAD'], check=True)
    monkeypatch.setattr(p, 'NATIVE_COMMIT', commit)
    cache = b'public synthetic cache bytes\n'
    monkeypatch.setattr(n, 'TUNE_SHA256', hashlib.sha256(cache).hexdigest())
    for rel in ('e/run/exl3_tune_cache.bin', 'prod/run/exl3_tune_cache_tp2.bin'):
        path = home / 'dsv41-offload' / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(cache)
    binaries: dict[str, str] = {}
    for rel in p.BINARIES:
        path = home / 'dsv41-offload/n4' / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        data = ('public synthetic binary ' + rel).encode()
        path.write_bytes(data)
        binaries[rel] = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(p, 'BINARIES', binaries)
    (home / 'dsv41-offload/n4/triton').mkdir()
    (home / 'dsv41-offload/locks').mkdir()
    executable = home / 'dsv41-offload/e/venv/bin/python'
    executable.parent.mkdir(parents=True)
    executable.write_text('#!/bin/sh\necho "unexpected GPU startup" >&2\nexit 98\n')
    executable.chmod(0o700)
    return home, bundle


def test_stage_actual_bundle_and_durable_source_receipts(node: tuple[Path, Path]) -> None:
    home, bundle = node
    result = run(p.stage_script(), home, str(bundle))
    assert result.returncode == 0, result.stderr + result.stdout
    source = home / p.NATIVE_TREE
    assert (source / 'SOURCE_COMMIT').read_text().strip() == p.NATIVE_COMMIT
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == p.NATIVE_COMMIT
    assert (home / n.TUNE_CACHE).read_bytes() == (home / 'dsv41-offload/e/run/exl3_tune_cache.bin').read_bytes()
    assert 'W8 provenance unavailable' in (source.parent / 'PREREQUISITES.txt').read_text()
    repeated = run(p.stage_script(), home, str(bundle))
    assert repeated.returncode != 0, 'existing source must not be overwritten'


@pytest.mark.parametrize('failure', ['review', 'coordinator', 'tune', 'production-cache', 'binary', 'commit', 'symlink'])
def test_stage_refuses_before_durable_writes(node: tuple[Path, Path], failure: str) -> None:
    home, bundle = node
    script = p.stage_script()
    if failure == 'review':
        script = 'unset NBW_REVIEWED_CACHE_ALTERNATIVE\n' + script
    elif failure == 'coordinator':
        script = 'unset WINDOW_COORDINATOR\n' + script
    elif failure == 'tune':
        (home / 'dsv41-offload/e/run/exl3_tune_cache.bin').write_bytes(b'changed')
    elif failure == 'production-cache':
        (home / 'dsv41-offload/prod/run/exl3_tune_cache_tp2.bin').write_bytes(b'changed')
    elif failure == 'binary':
        (home / 'dsv41-offload/n4/lib/libofl.so').write_bytes(b'changed')
    elif failure == 'commit':
        script = script.replace(p.NATIVE_COMMIT, '0' * 40)
    elif failure == 'symlink':
        (home / 'dsv41-offload/nbw-source').symlink_to(home / 'missing')
    result = run(script, home, str(bundle))
    assert result.returncode != 0, result.stdout
    assert not (home / n.TUNE_CACHE).exists()
    assert not (home / p.NATIVE_TREE / 'SOURCE_COMMIT').exists()
    # On a missing commit, clone is retained for inspection; there is no successful receipt.
    assert not (home / 'dsv41-offload/nbw-source/PREREQUISITES.txt').exists()


def solo_fake(home: Path) -> None:
    smi = home / 'bin/nvidia-smi'
    smi.write_text('#!/bin/sh\nexit 0\n')
    worker = home / 'dsv41-offload/e/venv/bin/python'
    worker.write_text(f'#!{sys.executable}\n' + '''import fcntl, os
from pathlib import Path
root = Path.home()
lock = root / 'dsv41-offload/locks/gpu-bc62df2a.lock'
a, b = os.fstat(9), lock.stat()
assert (a.st_ino, a.st_dev) == (b.st_ino, b.st_dev)
with lock.open('a') as probe:
    try: fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: pass
    else: raise AssertionError('solo process did not inherit exclusive fd9')
assert os.environ['DSPARK_NATIVE_BATCH_INVARIANT'] == '2'
assert os.environ['DSPARK_WORKER_TP_EMULATE'] == '1'
assert os.environ['KX_FP8DENSE_LOCK_FD'] == '9'
assert os.environ['CUDA_VISIBLE_DEVICES'].startswith('GPU-bc62df2a-')
(root / 'solo-started').touch()
print('FAKE CPU SOLO STARTUP; NO GPU IMPORT')
''')


@pytest.mark.parametrize('busy', [False, True])
def test_solo_command_inherits_real_nonblocking_fd9(node: tuple[Path, Path], busy: bool) -> None:
    home, bundle = node
    staged = run(p.stage_script(), home, str(bundle))
    assert staged.returncode == 0, staged.stderr
    solo_fake(home)
    with (home / 'dsv41-offload/locks/gpu-bc62df2a.lock').open('a') as lock:
        if busy:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run(p.solo_script(p.SOLO_GPU), home)
    if busy:
        assert result.returncode == 73 and 'owned/busy' in result.stderr
        assert not (home / 'solo-started').exists()
    else:
        assert result.returncode == 0 and 'FAKE CPU SOLO' in result.stdout, result.stderr
        assert (home / 'solo-started').exists()


def test_scripts_pin_real_prerequisites_and_parse_without_execution() -> None:
    assert p.NATIVE_COMMIT == 'dc6de03a8601acba039e972e69c8cee54560f4d7'
    assert n.TUNE_SHA256 == '937e9633334be0a5a7a97557e83f962bfbc33dce84a76b93e331b8017b0d405e'
    for script in (p.stage_script(), p.solo_script(p.SOLO_GPU)):
        subprocess.run(['bash', '-n', '-c', script], check=True)
        assert 'systemctl' not in script and 'prod_window.sh' not in script
    bootstrap = p.solo_bootstrap()
    ast.parse(bootstrap)
    assert "graphs=True, tp=2, max_model_len=1048576" in bootstrap
    assert "runpy.run_module('offload.dspark.worker.server'" not in bootstrap
    assert 'read-only Triton cache miss' in bootstrap and 'NBW_SOLO_RECEIPT' in bootstrap
    words = shlex.split(p.solo_script(p.SOLO_GPU))
    assert bootstrap in words
    with pytest.raises(ValueError, match='another solo UUID needs'):
        p.solo_script('GPU-240876ad-ea42-47de-1b8d-f6153810b7e1')


def test_cli_only_emits_and_requires_explicit_gpu(capsys: pytest.CaptureFixture[str]) -> None:
    assert p.main([]) == 0 and capsys.readouterr().out.startswith('DRY PLAN')
    assert p.main(['--emit', 'stage']) == 0 and capsys.readouterr().out.startswith('#!/usr/bin/env bash')
    with pytest.raises(SystemExit) as missing:
        p.main(['--emit', 'solo'])
    assert missing.value.code == 2
    assert p.main(['--emit', 'solo', '--solo-gpu', p.SOLO_GPU]) == 0
    assert p.SOLO_GPU in capsys.readouterr().out


@pytest.mark.parametrize('query', ['failure', 'active'])
def test_solo_gpu_query_error_or_compute_activity_refuses(node: tuple[Path, Path], query: str) -> None:
    home, bundle = node
    staged = run(p.stage_script(), home, str(bundle))
    assert staged.returncode == 0, staged.stderr
    solo_fake(home)
    (home / 'bin/nvidia-smi').write_text('#!/bin/sh\n' + ('exit 9\n' if query == 'failure' else 'echo 12345\n'))
    result = run(p.solo_script(p.SOLO_GPU), home)
    assert result.returncode == (9 if query == 'failure' else 74), result.stderr
    assert not (home / 'solo-started').exists()
