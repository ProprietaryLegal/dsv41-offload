"""S33 probes of the decode_close.sh `pause` handshake with REAL tmux, bash and the pinned generator.

A COPY of the campaign script runs NON-dry (C= and STATE= rewritten into tmp). tmux is the real binary, but on a
PRIVATE server (TMUX unset, TMUX_TMPDIR in tmp), so `tmux ... -t =d2-gen2` can never reach the coordinator's server or
its d2-gen2 session. The generator is the REAL `python3 -m d2.generate run` from a `git clone --shared` of 1b8fc1c
(the pinned pass-2 revision), launched with the exact decode_gen.sh tmux command shape against the fake vLLM server.
ssh and curl are local stubs; nothing can reach a host.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

SCRIPT = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/decode_close.sh')
SCRIPT_SHA = 'a30a7b66b8a5bd4776022aa02327d941fc55a26c2ce8df899ffafb0659e27cfc'
REPO = Path(__file__).resolve().parents[4]
GEN_COMMIT = '1b8fc1c944cb2d7e23aeff9042c3a5915203746b'
GEN_SHA = '56c2fd7152e8dd8a8ccbce2a2bf79ad1382a9f68d03f343fba25116c76ddfd7a'  # W12 w12.log GEN: line
PY = '/home/operator/miniconda3/bin/python3'
ERR_FILE = '/home/rank0user/dsv41-offload-run/d2-decode-d2pass2-r1/ERROR_COUNT.json'

pytestmark = pytest.mark.skipif(not SCRIPT.exists() or shutil.which('tmux') is None or not Path(PY).exists(),
                                reason='campaign script, tmux or the generator python absent')

FAKE_SERVER = r'''
import sys, time
sys.path.insert(0, sys.argv[1])
from d2.tests.fakevllm import Fake
with Fake(token_delay=0.05) as f:
    open(sys.argv[2], 'w').write(f.url)
    while True:
        time.sleep(3600)
'''

GEN_SSH = r'''#!/bin/bash
# generator-side ssh: HOLD probe answers "no HOLD"; the capture error read returns a clean count.
cmd="${@: -1}"
case "$cmd" in
  "cat -- "*) printf '{"schema": "d2.decode-error-count.v1", "errors": 0, "global_refusal": false}' ;;
  "test -e "*) exit 1 ;;
  *) echo "unexpected ssh $*" >&2; exit 97 ;;
esac
'''

CURL = r'''#!/bin/bash
# idle production metrics for the close script's idle() gate
printf 'vllm:num_requests_running{engine="0",model_name="m"} 0.0\nvllm:num_requests_waiting{engine="0",model_name="m"} 0.0\n'
'''


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exe(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o700)


def clean_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GIT_', 'TMUX'))}
    env.update(extra)
    return env


DELIVERED_SIGINT = 'tmux send-keys -t =d2-gen2: C-c'   # corrected in the delivered script (S33 FIX 1)
BROKEN_SIGINT = 'tmux send-keys -t =d2-gen2 C-c'      # what R2 delivered: not a pane target on tmux 3.6
CORRECTED_SIGINT = 'tmux send-keys -t =d2-gen2: C-c'   # S33 FIX: a pane target needs the session-name colon


class Rig:
    def __init__(self, tmp: Path, corrected: bool = False) -> None:
        assert sha(SCRIPT) == SCRIPT_SHA
        self.t = tmp
        self.c = tmp/'C'
        (self.c/'W12').mkdir(parents=True)
        self.state = tmp/'state'
        # tmux's socket path must fit sun_path (108 bytes): a short private dir, never the default server
        self.tmux_dir = Path(tempfile.mkdtemp(prefix='s33t', dir=os.environ.get('S33_SHORT_DIR', str(tmp))))
        self.bin = tmp/'bin'
        self.bin.mkdir()
        exe(self.bin/'ssh', GEN_SSH)
        exe(self.bin/'curl', CURL)
        text = SCRIPT.read_text()
        old_c = 'C=/opt/dsv41/campaigns/DSV41-OFFLOAD\n'
        old_s = 'STATE=/opt/dsv41/data/d2/gen-decode-07fc7d6-$RUN\n'
        assert text.count(old_c) == 1 and text.count(old_s) == 1
        assert text.count(DELIVERED_SIGINT) == 1
        if corrected:
            text = text.replace(DELIVERED_SIGINT, CORRECTED_SIGINT)
        self.script = self.c/'W12/decode_close.sh'
        exe(self.script, text.replace(old_c, f'C={self.c}\n').replace(old_s, f'STATE={self.state}\n'))
        self.env = clean_env(PATH=f'{self.bin}:/usr/bin:/bin', TMUX_TMPDIR=str(self.tmux_dir), SHELL='/bin/bash',
                             D2_COORDINATOR='Deepseek/coordinator', LC_ALL='C')
        self.server: subprocess.Popen[bytes] | None = None

    def tmux(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(['tmux', '-f', '/dev/null', *args], env=self.env, text=True, capture_output=True)

    def gen_clone(self) -> Path:
        g = self.t/'gen'
        env = clean_env()
        subprocess.run(['git', 'clone', '-q', '--shared', '--no-checkout', str(REPO), str(g)], env=env, check=True)
        subprocess.run(['git', '-C', str(g), '-c', 'advice.detachedHead=false', 'checkout', '-q', '--detach',
                        GEN_COMMIT], env=env, check=True)
        assert sha(g/'offload/dspark/d2/generate.py') == GEN_SHA
        return g/'offload/dspark'

    def start_generator(self, extra: str = '') -> None:
        code = self.gen_clone()
        url_file = self.t/'url'
        self.server = subprocess.Popen([PY, '-B', '-c', FAKE_SERVER, str(code), str(url_file)], env=clean_env())
        for _ in range(100):
            if url_file.exists() and url_file.read_text():
                break
            time.sleep(0.1)
        url = url_file.read_text()
        prompts = self.t/'prompts.jsonl'
        prompts.write_text(''.join(json.dumps({
            'pid': f'p{i:04d}', 'split': 'train', 'family': 'f', 'prompt_sha256': f'h{i}',
            'messages': [{'role': 'user', 'content': f'synthetic prompt {i}'}],
            'chat_template_kwargs': {'thinking': True, 'reasoning_effort': 'high'}})+'\n' for i in range(400)))
        log = self.c/'D2-GEN2.out.log'
        # Exact decode_gen.sh shape: `cd X && PY -m d2.generate run ... >> LOG 2>&1; echo D2-GEN2-EXIT=$? >> LOG`
        command = (f'cd {code} && PYTHONDONTWRITEBYTECODE=1 {PY} -m d2.generate run --prompts {prompts} '
                   f'--state {self.state} --server {url} --model m --spark-hours 1.0 --streams 3 --top20 generation '
                   f'--pause-file {self.c}/D2-GEN2.PAUSE --capture-error-host sparkhead --capture-error-file {ERR_FILE} '
                   f'--idle-s 0.5 --backoff-s 1 --no-local-check{extra} >> {log} 2>&1; '
                   f'echo D2-GEN2-EXIT=$? >> {log}')
        r = self.tmux('new-session', '-d', '-s', 'd2-gen2', command)
        assert r.returncode == 0, r.stderr
        time.sleep(4)
        assert self.has_session(), log.read_text()

    def has_session(self) -> bool:
        return self.tmux('has-session', '-t', '=d2-gen2').returncode == 0

    def pause(self) -> tuple[subprocess.CompletedProcess[str], float]:
        start = time.monotonic()
        r = subprocess.run(['/bin/bash', str(self.script), '--nonce', 'n1', 'pause'], env=self.env, text=True,
                           capture_output=True, timeout=300)
        return r, time.monotonic()-start

    def close(self) -> None:
        self.tmux('kill-server')
        shutil.rmtree(self.tmux_dir, ignore_errors=True)
        if self.server is not None:
            self.server.terminate()
            self.server.wait()


@pytest.fixture
def rig(tmp_path: Path):
    r = Rig(tmp_path)
    try:
        yield r
    finally:
        r.close()


@pytest.fixture
def corrected_rig(tmp_path: Path):
    r = Rig(tmp_path, corrected=True)
    try:
        yield r
    finally:
        r.close()


def seqs(state: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((state/'seqs').glob('*.json'))]


def test_broken_target_is_gone_and_still_rejected_by_tmux(rig: Rig) -> None:
    """S33 FIX 1: tmux 3.6 rejects `send-keys -t =d2-gen2` ("can't find pane"); the delivered script uses `=d2-gen2:`."""
    assert BROKEN_SIGINT not in rig.script.read_text() and DELIVERED_SIGINT in rig.script.read_text()
    rig.start_generator()
    time.sleep(3)
    r = rig.tmux('send-keys', '-t', '=d2-gen2', 'C-c')
    assert r.returncode != 0 and "can't find pane: =d2-gen2" in r.stderr
    r, _ = rig.pause()
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert not rig.has_session()
    assert 'D2-GEN2-EXIT=0' in (rig.c/'D2-GEN2.out.log').read_text()


def test_corrected_target_sigints_real_generator_which_saves_and_exits(corrected_rig: Rig) -> None:
    rig = corrected_rig
    rig.start_generator()
    time.sleep(6)                                       # streams running and checkpointed
    assert any(s['gen_ids'] for s in seqs(rig.state))
    r, elapsed = rig.pause()
    assert r.returncode == 0, (r.stdout, r.stderr, (rig.c/'D2-GEN2.out.log').read_text())
    assert 'NEXT: flush' in r.stdout
    assert not rig.has_session()
    log = (rig.c/'D2-GEN2.out.log').read_text()
    assert 'D2-GEN2-EXIT=0' in log                       # bash survived C-c and wrote the marker after python
    events = [json.loads(l) for l in (rig.state/'events.jsonl').read_text().splitlines()]
    kinds = [e.get('kind', e.get('event')) for e in events]
    assert 'stop' in kinds and 'pause' in kinds
    assert not any(s['status'] == 'running' for s in seqs(rig.state))
    assert json.loads((rig.state/'budget.json').read_text())['busy_s'] > 0
    assert (rig.c/'W12/private/close-versions/evidence-d2pass2-r1/pause.ok').read_text() == 'n1\n'
    # 10 s natural-exit interval + SIGINT exit; well inside the 120 s bound
    assert elapsed < 30, elapsed
    print('MEASURED pause phase wall seconds', round(elapsed, 2))


def test_pause_without_session_but_with_marker_passes(rig: Rig) -> None:
    rig.state.mkdir()
    (rig.c/'D2-GEN2.out.log').write_text('D2-GEN2-EXIT=0\n')
    r, elapsed = rig.pause()
    assert r.returncode == 0, r.stderr
    assert elapsed < 15


def test_pause_without_session_and_without_marker_refuses_untouched(rig: Rig) -> None:
    rig.state.mkdir()
    (rig.c/'D2-GEN2.out.log').write_text('')
    r, elapsed = rig.pause()
    assert r.returncode != 0
    assert 'generator shutdown not verified after SIGINT' in r.stderr
    assert 'STOP: phase pause failed' in r.stderr and 'Production untouched' in r.stderr
    assert (rig.c/'D2-GEN2.PAUSE').exists()
    print('MEASURED refusal after', round(elapsed, 1), 's')
