"""Execute the exact remote log reader with a public CPU Docker fixture."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from offload.tv1 import engine_logs as logs


@pytest.fixture
def docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    binary = tmp_path / 'bin'
    binary.mkdir()
    script = binary / 'docker'
    script.write_text('#!' + sys.executable + '''
import json,os,sys,time
from pathlib import Path
root=Path(os.environ['TV1_LOG_FIXTURE'])
with (root/'calls').open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1]=='inspect':
 if os.environ.get('TV1_LOG_MODE')=='inspect-error': raise SystemExit(1)
 print('a'*64)
 print(json.dumps({'Status':'exited','ExitCode':1}))
 print(json.dumps([] if os.environ.get('TV1_LOG_MODE')=='production' else [{'Destination':'/sp1-profile'}]))
else:
 assert sys.argv[1:]==['logs','--timestamps','a'*64]
 sys.stdout.buffer.write((root/'payload').read_bytes());sys.stdout.flush()
 if os.environ.get('TV1_LOG_MODE')=='timeout': time.sleep(2)
 if os.environ.get('TV1_LOG_MODE')=='logs-error': raise SystemExit(3)
''')
    script.chmod(0o700)
    monkeypatch.setenv('PATH', str(binary) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('TV1_LOG_FIXTURE', str(tmp_path))
    payload = tmp_path / 'payload'
    payload.write_bytes(b'root cause before tail40\n' + b'public startup\n'*80)
    return payload


def run(seconds: float = 2., limit: int = logs.LIMIT) -> str:
    result = subprocess.run([sys.executable, '-c', logs.script(limit=limit, seconds=seconds)],
                            capture_output=True, text=True, timeout=6, check=True)
    return result.stdout


def test_complete_log_and_hash(docker: Path) -> None:
    output = docker.parent / 'saved.log'
    receipt = logs.save(run(), output)
    assert receipt['complete'] is True and output.read_bytes() == docker.read_bytes()
    assert receipt['stream_sha256'] == receipt['stored_sha256'] == hashlib.sha256(docker.read_bytes()).hexdigest()
    assert output.with_suffix('.json').exists()
    assert 'data_b64' not in receipt


def test_bound_retains_head_tail_and_hashes_entire_stream(docker: Path) -> None:
    output = docker.parent / 'bounded.log'
    receipt = logs.save(run(limit=128), output)
    data = docker.read_bytes()
    assert output.read_bytes() == data[:64] + data[-64:]
    assert receipt['truncated'] is True and receipt['complete'] is False
    assert receipt['stored_bytes'] == 128 and receipt['total_bytes'] == len(data)
    assert receipt['stream_sha256'] == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize('mode', ['production', 'inspect-error', 'logs-error', 'timeout'])
def test_errors_are_explicit_and_never_export_production_logs(
        docker: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setenv('TV1_LOG_MODE', mode)
    receipt = logs.save(run(seconds=.1), docker.parent / 'error.log')
    assert receipt.get('complete') is not True and receipt.get('error')
    calls = [json.loads(line) for line in (docker.parent / 'calls').read_text().splitlines()]
    if mode in ('production', 'inspect-error'):
        assert len(calls) == 1


def test_tampered_or_false_complete_payload_refuses(docker: Path) -> None:
    packet = json.loads(run())
    packet['stored_sha256'] = '0'*64
    with pytest.raises(ValueError, match='hash mismatch'):
        logs.save(json.dumps(packet), docker.parent / 'bad.log')
    packet = json.loads(run(limit=128))
    packet['complete'] = True
    with pytest.raises(ValueError, match='falsely complete'):
        logs.save(json.dumps(packet), docker.parent / 'false.log')
