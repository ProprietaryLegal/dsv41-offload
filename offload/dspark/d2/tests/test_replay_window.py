"""No Spark access: synthetic workflow, parsers and owned-child lifecycle."""
from __future__ import annotations

import json
import os
import ast
import hashlib
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
from typing import Callable, cast

import pytest

from d2 import replay_window as w
from d2.window_checks import equiv_complete, responses, response_policy

REPO = Path(__file__).resolve().parents[4]


def test_dry_run_exercises_all_checks_without_subprocess_network_or_sleep() -> None:
    io = w.DryIO()
    with patch('subprocess.Popen', side_effect=AssertionError('process')), patch(
            'urllib.request.urlopen', side_effect=AssertionError('network')), patch(
                'time.sleep', side_effect=AssertionError('real sleep')):
        w.run(io, 'synthetic', REPO)
    assert io.elapsed == 300
    assert 'response-check' in io.calls and 'equiv-complete' in io.calls
    assert 'identity-sparkhead' in io.calls and 'identity-sparkworker-wired' in io.calls
    assert io.calls[-1] == 'equiv-gate'
    assert 'experiment-stop' in io.calls
    assert not any('restore' in step for step in io.calls)


@pytest.mark.parametrize('step', ['preflight', 'source-pins', 'window-open', 'start', 'startup',
    'mount-owner-sparkhead', 'mount-owner-sparkworker-wired', 'memory-sparkhead', 'memory-sparkworker-wired',
    'identity-sparkhead', 'identity-sparkworker-wired', 'hashes-sparkhead', 'hashes-sparkworker-wired',
    'replay', 'response-check', 'copy', 'copy-check'])
def test_every_window_command_failure_stops_without_restoring_or_stopping_foreign_engine(step: str) -> None:
    io = w.DryIO(step)
    with pytest.raises(ValueError, match='synthetic'):
        w.run(io, 'synthetic', REPO)
    assert 'experiment-stop' not in io.calls
    assert not any(c.startswith('equiv-') for c in io.calls)


@pytest.mark.parametrize('step', ['equiv-replay-check', 'equiv-taps', 'equiv-drafts', 'equiv-complete',
    'equiv-report', 'equiv-gate'])
def test_equivalence_runs_after_the_sparks_are_released_and_its_failure_stops(step: str) -> None:
    io = w.DryIO(step)
    with pytest.raises(ValueError, match='synthetic'):
        w.run(io, 'synthetic', REPO)
    # The experiment was stopped (Sparks released for production restoration) before any CPU equivalence step.
    assert 'experiment-stop' in io.calls
    assert io.calls.index('experiment-stop') < min(i for i, c in enumerate(io.calls) if c.startswith('equiv-'))


def test_equivalence_phase_contacts_x86host_only() -> None:
    io = w.DryIO()
    w.run_equiv(io, 'synthetic', REPO)
    assert set(io.hosts) == {'x86host'}
    assert not any(c.startswith('memory-') for c in io.calls)
    assert io.calls[-1] == 'equiv-gate'


def test_window_phase_ends_with_experiment_stop() -> None:
    io = w.DryIO()
    w.run_window(io, 'synthetic', REPO)
    assert 'experiment-stop' in io.calls and not any(c.startswith('equiv-') for c in io.calls)


def test_busy_resets_five_minute_clock() -> None:
    class Busy(w.DryIO):
        def metrics(self) -> str:
            if 295 <= self.elapsed < 300:
                return 'vllm:num_requests_running{} 1\nvllm:num_requests_waiting{} 0\n'
            return super().metrics()
    io = Busy()
    w.idle_five_minutes(io)
    assert io.elapsed == 600
    for text in ('', 'vllm:num_requests_running{} 0', 'vllm:num_requests_running{} NaN\nvllm:num_requests_waiting{} 0'):
        with pytest.raises(ValueError):
            w.gauges(text)


@pytest.mark.parametrize('text', ['MemAvailable: 3145727 kB', '', 'MemAvailable: nan kB'])
def test_memory_stop(text: str) -> None:
    with pytest.raises(ValueError):
        w.memory(text)
    assert w.memory('MemAvailable: 3145728 kB') == w.MEM_KB


def test_bad_startup_copy_and_equivalence_stop() -> None:
    for text in ('', 'D2 replay taps on, rank 0 root-owned', 'D2 replay taps on, rank 0 refused'):
        with pytest.raises(ValueError):
            w.startup(text)
    with pytest.raises(ValueError):
        w.complete_check(json.dumps({'planned': 128, 'logprobs': 128, 'taps': 127,
                                    'missing_logprobs': [], 'missing_taps': ['missing'], 'bad': []}))
    with pytest.raises(ValueError):
        w.equivalence(json.dumps({'schema': 'd2.equiv.gate.v1', 'gate': {'pass': False, 'verdict': 'fail'}}))


@pytest.mark.parametrize('response,error', [
    ({'error': {'message': 'synthetic refusal'}, 'choices': [{}]}, ValueError),
    ({'choices': [{'refusal': 'synthetic refusal'}]}, ValueError),
    ({'choices': [{'finish_reason': 'content_filter'}]}, ValueError),
    ({'choices': [{'finish_reason': 'error'}]}, ValueError),
    ({'choices': [{'finish_reason': 'abort'}]}, ValueError),
    ({'choices': [{}], 'usage': {'prompt_tokens_details': {'cached_tokens': 1}}}, RuntimeError),
])
def test_http_200_refused_or_cached_responses_stop_before_array_writes(
        response: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error, match='refused|cached'):
        response_policy(response)


def test_nested_real_equivalence_schema_and_runtime_identities() -> None:
    w.equivalence(json.dumps({'schema': 'd2.equiv.gate.v1', 'gate': {'pass': True, 'verdict': 'pass_bf16_noise'},
                             'replay_binding': {'synthetic': {}}}))
    io = w.DryIO()
    text = io.call('identity-sparkhead', 'synthetic', '')
    assert w.identity(text) == 'a'*64
    with pytest.raises(ValueError, match='image'):
        w.identity(text.replace(w.IMAGE, 'different'))
    hashes = io.call('hashes-sparkhead', 'synthetic', '')
    w.runtime_hashes(hashes, REPO)
    with pytest.raises(ValueError, match='hash'):
        w.runtime_hashes(hashes.replace(w.TOKENIZER, '0'*64), REPO)


def test_incomplete_and_cached_response_metadata_refused(tmp_path: Path) -> None:
    entry = dict(seq_id='synthetic', request_id='d2r.synthetic', L=12, gen_start=8,
                 tokens_sha256='synthetic', **{'from': 0})
    (tmp_path/'plan.jsonl').write_text(json.dumps(entry)+'\n')
    (tmp_path/'logprobs').mkdir()
    with pytest.raises(FileNotFoundError):
        responses(tmp_path)
    meta = dict(entry, usage={'prompt_tokens_details': {'cached_tokens': 1}})
    (tmp_path/'logprobs/synthetic.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='cached'):
        responses(tmp_path)


def test_incomplete_equivalence_shard_refused(tmp_path: Path) -> None:
    (tmp_path/'drafts-0.log').write_text('181 blocks in shard 0/3, 181 to go\ndone 181 blocks in 12 s\n')
    (tmp_path/'drafts-0of3.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='partial'):
        equiv_complete(tmp_path)


def test_owned_remote_lease_eof_stops_only_its_child(tmp_path: Path) -> None:
    helper = REPO/'offload/dspark/d2/owned_command.py'
    marker = tmp_path/'started'
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    child = subprocess.Popen([sys.executable, str(helper), '--timeout', '20', '--', sys.executable, '-c',
                              f'from pathlib import Path; import time; Path({str(marker)!r}).touch(); time.sleep(30)'],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for _ in range(100):
            if marker.exists():
                break
            time.sleep(.02)
        assert marker.exists()
        assert child.stdin is not None
        child.stdin.close()
        child.wait(timeout=8)
        assert child.returncode != 0
        assert unrelated.poll() is None
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=8)
        unrelated.terminate()
        unrelated.wait()


def test_live_supervisor_with_fake_ssh_aborts_on_low_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path/'ssh'
    count, marker = tmp_path/'polls', tmp_path/'child.pid'
    fake.write_text('#!'+sys.executable+'\n'+'''import os, pathlib, subprocess, sys
command=sys.argv[-1]
if command.endswith('cat /proc/meminfo'):
 p=pathlib.Path(os.environ['FAKE_POLLS']); n=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(n))
 print('MemAvailable: '+str(3145727 if n>4 else 6291456)+' kB'); sys.exit(0)
os.execvp('bash', ['bash', '-c', command])
''')
    fake.chmod(0o700)
    monkeypatch.setenv('PATH', str(tmp_path)+':'+os.environ['PATH'])
    monkeypatch.setenv('FAKE_POLLS', str(count))
    io = w.LiveIO(tmp_path, 'synthetic')
    script = f'import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)'
    import shlex
    with pytest.raises(ValueError, match='MemAvailable'):
        io.call('owned-work', 'fake-host', shlex.join([sys.executable, '-c', script]), timeout=15, supervised=True)
    assert marker.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)


def test_window_capture_hashes_small_unaligned_files_with_direct_io(tmp_path: Path) -> None:
    import re
    script = (REPO/'offload/dspark/d2/capture_window_identity.sh').read_text()
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", script, re.S)
    assert len(snippets) == 2
    for snippet in snippets:
        ast.parse(snippet)
    digest = next(n for n in ast.parse(snippets[1]).body if isinstance(n, ast.FunctionDef) and n.name == 'digest')
    namespace: dict[str, object] = {'hashlib': hashlib, 'subprocess': subprocess}
    exec(compile(ast.Module(body=[digest], type_ignores=[]), '<synthetic-direct-io-probe>', 'exec'), namespace)
    path = tmp_path/'synthetic-metadata'
    path.write_bytes(b'synthetic byte identity\n'*179)
    assert cast(Callable[[Path], str], namespace['digest'])(path) == hashlib.sha256(path.read_bytes()).hexdigest()
