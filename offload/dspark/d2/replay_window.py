"""Coordinator replay window; default is an entirely offline synthetic rehearsal.

No Spark access or subprocess is possible through DryIO. LiveIO is created only
by --execute plus the explicit coordinator/source pin checks. Restoration stays
with W8/restore.sh and is never called here.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time
from typing import Callable, Protocol
import urllib.request

from .window_checks import response_entry, shard_complete

ROOT = '/opt/dsv41/x86host-data/d2'
PYTHON = '$HOME/dsv41-offload/o2/venv/bin/python'
MEM_KB = 3 * 1024 * 1024
METRICS = 'http://HEAD_IP:8000/metrics'
EXP = 'http://HEAD_IP:8001'
CAMPAIGN = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD')
PLAN_SHA = 'cd95f20702607d9d62349b0d9e7a57e7de4e76c5d5139dd584b9e390cadb4e53'
V1_SHA = '17109d736c217ea9c459be1f7851a2b7cfe057b762ae6b60ee86a7dc7f063684'
IMAGE = 'sha256:435a5236ce3a7db9d45f5a2c483817d60ea629c4ea0a0b3af33da2ef8122334b'
ENCODER = 'abfe2a1aa76d235bad273fe144d18919d8ef77c3709857bb523ef69f75cdfc6b'
TOKENIZER = 'c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def gauges(text: str) -> bool:
    values: dict[str, list[float]] = {'running': [], 'waiting': []}
    for line in text.splitlines():
        m = re.fullmatch(r'vllm:num_requests_(running|waiting)(?:\{[^}]*\})?\s+(\S+)', line)
        if m:
            value = float(m[2])
            require(math.isfinite(value) and value >= 0, 'invalid request gauge')
            values[m[1]].append(value)
    require(all(values.values()), 'missing running/waiting metrics')
    return all(value == 0 for group in values.values() for value in group)


def memory(text: str) -> int:
    m = re.search(r'^MemAvailable:\s*(\d+)\s+kB\s*$', text, re.M)
    require(m is not None, 'missing MemAvailable telemetry')
    kb = int(m[1]) if m else 0
    require(kb >= MEM_KB, 'MemAvailable below 3 GiB')
    return kb


def startup(text: str) -> None:
    require('D2 replay taps on, rank 0' in text, 'replay overlay inactive')
    require(not re.search(r'root-owned|refus|Traceback|CUDA error', text, re.I), 'startup refusal/root ownership/error')


def complete_check(text: str) -> None:
    d = json.loads(text)
    require(d.get('planned', 0) > 0, 'empty replay plan')
    require(d.get('logprobs') == d['planned'] == d.get('taps'), 'incomplete copied replay')
    require(all(d.get(k) == [] for k in ('missing_logprobs', 'missing_taps', 'bad')), 'bad/missing copied replay')


def equivalence(text: str) -> None:
    d = json.loads(text)
    gate = d.get('gate', {})
    require(d.get('schema') == 'd2.equiv.gate.v1' and gate.get('pass') is True
            and gate.get('verdict') in ('pass', 'pass_bf16_noise'), 'equivalence gate failed')
    require(d.get('replay_binding') is not None, 'missing equivalence replay binding')


def identity(text: str) -> str:
    records = [json.loads(line) for line in text.splitlines()]
    require(len(records) == 2, 'incomplete both-record runtime identity capture')
    container, files = records
    require(container.get('schema') == 'd2.window-container.v1' and container.get('state') == 'running'
            and container.get('image') == IMAGE, 'container image/state mismatch')
    require(any(m['Destination'] == '/d2-replay' and m['RW'] is True for m in container['mounts']),
            'replay capture mount missing')
    require(any(m['Destination'] == '/srv/dsv41/legal-v2a' and m['RW'] is False for m in container['mounts']),
            'read-only legal-v2a model mount missing')
    require(files.get('schema') == 'd2.window-files.v1' and files.get('engine') ==
            '0.31.1.dev0+g6e517b15.dsv41gb10', 'engine identity mismatch')
    require(files.get('generation_config') == {'exists': False, 'sha256': None}, 'generation defaults changed')
    cid = container['id']
    require(isinstance(cid, str) and re.fullmatch('[0-9a-f]{64}', cid) is not None, 'invalid container id')
    if not isinstance(cid, str):
        raise ValueError('container id is not a string')
    return cid


def expected_hashes(repo: Path) -> dict[str, str]:
    base = '/usr/local/lib/python3.12/dist-packages/vllm'
    return {base+'/v1/worker/gpu/spec_decode/dspark/speculator.py': hashlib.sha256(
                (repo/'offload/dspark/capture/speculator_replay.py').read_bytes()).hexdigest(),
            base+'/tokenizers/deepseek_v41_encoding.py': ENCODER,
            '/srv/dsv41/legal-v2a/tokenizer.json': TOKENIZER,
            '/srv/dsv41/legal-v2a/CANDIDATE_MANIFEST.json':
                '7549bb956f6f660d7932bcd5112501f78a2c660b8bcbcbc8905c2673f5ae05ae',
            '/srv/dsv41/legal-v2a/exl3_moe_plan.json':
                'ebf315efa28868de91fb95f5782fbc3a4a75623e0e0cd8a7fd3bae0b1a01830d'}


def runtime_hashes(text: str, repo: Path) -> None:
    actual = {line.split(maxsplit=1)[1].strip(): line.split(maxsplit=1)[0]
              for line in text.splitlines() if len(line.split(maxsplit=1)) == 2}
    require(all(actual.get(path) == digest for path, digest in expected_hashes(repo).items()),
            'served overlay/renderer hash mismatch')


def source_pin_command(repo: Path, source: str) -> str:
    dspark = repo/'offload/dspark'
    pins = {str(p.relative_to(dspark)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in dspark.rglob('*') if p.is_file()
            and p.suffix in ('.py', '.sh', '.cpp', '.cu', '.h', '.hpp', '.cc', '.json')
            and '__pycache__' not in p.parts}
    code = f'{ROOT}/source/{source}/repo/offload/dspark'
    script = ('import hashlib,json,pathlib; root=pathlib.Path('+repr(code)+'); pins=json.loads('+repr(json.dumps(pins))+'); '
              'assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in pins.items()), "source snapshot byte mismatch"')
    return shlex.join(['python3', '-c', script])


class IO(Protocol):
    sparks: bool  # False in the equivalence phase: production is back, so no Spark memory stop applies
    def now(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...
    def metrics(self) -> str: ...
    def call(self, name: str, host: str, command: str, timeout: int = 60,
             supervised: bool = False) -> str: ...


class DryIO:
    """Synthetic branch coverage, not hardware evidence; every validator still runs."""
    def __init__(self, fail: str = '') -> None:
        self.elapsed = 0.0
        self.fail = fail
        self.calls: list[str] = []
        self.hosts: list[str] = []
        self.sparks = True

    def now(self) -> float:
        return self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds

    def metrics(self) -> str:
        if self.fail == 'metrics':
            return ''
        return 'vllm:num_requests_running{} 0\nvllm:num_requests_waiting{} 0\n'

    def call(self, name: str, host: str, command: str, timeout: int = 60,
             supervised: bool = False) -> str:
        self.calls.append(name)
        self.hosts.append(host)
        print(f'DRY-RUN synthetic check {name}: ssh {host} {command}', flush=True)
        if self.fail == name:
            raise ValueError('synthetic command refusal: '+name)
        if supervised and self.sparks:
            for node in ('sparkhead', 'sparkworker-wired'):
                memory(self.call('memory-'+node, node, 'cat /proc/meminfo'))
        if name.startswith('memory-'):
            return 'MemAvailable: 6291456 kB\n'
        if name == 'startup':
            return 'DSV41-OFFLOAD: D2 replay taps on, rank 0\n'
        if name.startswith('identity-'):
            return json.dumps({'schema': 'd2.window-container.v1', 'state': 'running', 'image': IMAGE,
                               'id': 'a'*64, 'mounts': [{'Destination': '/d2-replay', 'RW': True},
                                                        {'Destination': '/srv/dsv41/legal-v2a', 'RW': False}]})+'\n'+json.dumps({
                                   'schema': 'd2.window-files.v1', 'engine': '0.31.1.dev0+g6e517b15.dsv41gb10',
                                   'generation_config': {'exists': False, 'sha256': None}})
        if name.startswith('hashes-'):
            repo = Path(__file__).resolve().parents[3]
            return '\n'.join(digest+'  '+path for path, digest in expected_hashes(repo).items())
        if name == 'response-check':
            e: dict[str, object] = {'seq_id': 'synthetic', 'request_id': 'synthetic', 'L': 12,
                                    'gen_start': 8, 'from': 0, 'tokens_sha256': 'synthetic'}
            response_entry(dict(e, usage={'prompt_tokens_details': {'cached_tokens': 0}}), e)
        if name == 'equiv-complete':
            for k, count in enumerate((181, 181, 180)):
                shard_complete(f'{count} blocks in shard {k}/3, {count} to go\ndone {count} blocks in 1 s\n', count, k)
        if name in ('copy-check', 'equiv-replay-check'):
            return json.dumps({'planned': 128, 'logprobs': 128, 'taps': 128,
                               'missing_logprobs': [], 'missing_taps': [], 'bad': []})
        if name == 'equiv-gate':
            return json.dumps({'schema': 'd2.equiv.gate.v1', 'gate': {'pass': True, 'verdict': 'pass'}, 'replay_binding': {}})
        return 'synthetic success (not served evidence)\n'


class LiveIO:
    def __init__(self, out: Path, source: str) -> None:
        self.out, self.source = out, source
        self.index = 0
        self.sparks = True

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def metrics(self) -> str:
        with urllib.request.urlopen(METRICS, timeout=3) as r:
            text = r.read().decode()
        if not isinstance(text, str):
            raise ValueError('metrics is not text')
        with (self.out/'idle-metrics.log').open('a') as f:
            f.write(f'{time.time()}\n{text}\n')
        return text

    def memcheck(self) -> None:
        require(not (self.out/'STOP').exists(), 'coordinator STOP file present')
        if not self.sparks:
            return
        for node in ('sparkhead', 'sparkworker-wired'):
            memory(self.call('memory-'+node, node, 'cat /proc/meminfo', timeout=8))

    def call(self, name: str, host: str, command: str, timeout: int = 60,
             supervised: bool = False) -> str:
        self.index += 1
        path = self.out/f'{self.index:05d}-{name}.log'
        if supervised:
            self.memcheck()
            # Every active remote step gets an EOF lease. No arbitrary PID/pkill cleanup.
            # The lease helper is fed inline so Spark hosts need no copied file.
            code = Path(__file__).with_name('owned_command.py').read_text()
            command = shlex.join(['python3', '-c', code, '--timeout', str(timeout), '--', 'bash', '-lc', command])
        command = 'export PYTHONDONTWRITEBYTECODE=1; umask 077; '+command
        argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', host, command]
        with path.open('w') as f:
            child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=f, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                deadline = self.now() + timeout + (15 if supervised else 0)
                while child.poll() is None:
                    require(self.now() < deadline, f'{name} timeout')
                    if supervised:
                        self.memcheck()
                    self.sleep(2 if supervised else .1)
                require(child.returncode == 0, f'{name} failed ({child.returncode}); inspect {path}')
            finally:
                if child.stdin:
                    child.stdin.close()  # EOF tells remote wrapper to stop its own child group.
                if child.poll() is None:
                    try:
                        child.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait(timeout=5)
        return path.read_text()


def idle_five_minutes(io: IO) -> None:
    since: float | None = None
    deadline = io.now() + 3600
    while io.now() < deadline:
        # Missing/stale/invalid metrics fail closed before opening, never count as idle.
        if gauges(io.metrics()):
            since = io.now() if since is None else since
            if io.now() - since >= 300:
                return
        else:
            since = None
        io.sleep(5)
    raise ValueError('no continuous five-minute production idle interval within one hour')


def d2_command(source: str) -> Callable[[str], str]:
    code = f'{ROOT}/source/{source}/repo/offload/dspark'
    base = f'export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1; cd {code}; '
    def d2(command: str) -> str:
        return base+f'{PYTHON} -u -m d2.'+command
    return d2


def preflight(io: IO, source: str, repo: Path) -> None:
    io.call('preflight', 'x86host', f"set -euo pipefail; cd {ROOT}; sha256sum -c INPUTS.SHA256 >/dev/null; "
            f"test \"$(cat source/{source}/SOURCE_COMMIT)\" = {source}; "
            f"echo '{PLAN_SHA}  replay/plan.jsonl' | sha256sum -c -; cd v1; "
            f"echo '{V1_SHA}  V1.SHA256' | sha256sum -c -; sha256sum -c V1.SHA256 >/dev/null")
    io.call('source-pins', 'x86host', source_pin_command(repo, source))


def run(io: IO, source: str, repo: Path) -> None:
    """Both phases in order (dry-run coverage). Live execution runs them as separate --phase calls with production
    restored in between, because equivalence is x86host CPU work (~90 min) that does not need the Sparks."""
    run_window(io, source, repo)
    run_equiv(io, source, repo)


def run_window(io: IO, source: str, repo: Path) -> None:
    d2 = d2_command(source)
    # Source/input/plan pins checked before any production request or mutation.
    preflight(io, source, repo)
    io.call('no-generation', 'sparkhead', 'test ! -e ~/dsv41-deploy/HOLD')
    idle_five_minutes(io)
    # Check again at the opening boundary; window.sh also rechecks production idle.
    require(gauges(io.metrics()), 'production became busy before open')
    for node in ('sparkhead', 'sparkworker-wired'):
        memory(io.call('memory-'+node, node, 'cat /proc/meminfo'))
        io.call('mount-owner-'+node, node, 'set -eu; umask 077; mkdir -p ~/dsv41-offload-run/d2-replay; '
                'test "$(stat -c %u ~/dsv41-offload-run/d2-replay)" = "$(id -u)"; '
                'test "$(stat -c %u ~/dsv41-offload-run/d2-replay)" != 0')
    io.call('window-open', 'sparkhead', 'cd ~/dsv41-deploy-offload && offload/launch/window.sh open "D2 replay W9"',
            timeout=3700, supervised=True)
    io.call('start', 'sparkhead', 'cd ~/dsv41-deploy-offload && '
            'offload/launch/start_exp.sh offload/launch/configs/A-replay.env', timeout=1800, supervised=True)
    startup(io.call('startup', 'sparkhead', 'docker logs dsv41-serve 2>&1', supervised=True))
    capture = (repo/'offload/dspark/d2/capture_window_identity.sh').read_text()
    container_ids: dict[str, str] = {}
    for node in ('sparkhead', 'sparkworker-wired'):
        # Capture commands are concrete reads on both ranks. Exact byte identities are mandatory.
        cid = identity(io.call('identity-'+node, node, shlex.join(['bash', '-c', capture]), supervised=True))
        container_ids[node] = cid
        hash_script = ('set -euo pipefail; for f in '+shlex.join(list(expected_hashes(repo)))+
                       '; do hash=$(dd if="$f" iflag=direct bs=4096 status=none | sha256sum); '
                       'printf "%s  %s\\n" "${hash%% *}" "$f"; done')
        probe = shlex.join(['docker', 'exec', 'dsv41-serve', 'bash', '-c', hash_script])
        hashes = io.call('hashes-'+node, node, probe, supervised=True)
        runtime_hashes(hashes, repo)
    io.call('replay', 'x86host', d2(f'replay run --out {ROOT}/replay --server {EXP} --inflight 2 --check-remote'),
            timeout=7200, supervised=True)
    io.call('response-check', 'x86host', d2(f'window_checks responses {ROOT}/replay'), timeout=600, supervised=True)
    io.call('copy', 'x86host', d2(f'replay copy --out {ROOT}/replay --metrics {EXP}/metrics'), timeout=3600, supervised=True)
    complete_check(io.call('copy-check', 'x86host', d2(f'replay check --out {ROOT}/replay'), timeout=600, supervised=True))
    for node, cid in container_ids.items():
        io.call('stop-identity-'+node, node,
                'test "$(docker inspect --format \'{{.Id}}\' dsv41-serve)" = '+shlex.quote(cid), supervised=True)
    io.call('experiment-stop', 'sparkhead', 'set -eu; test "$(docker inspect --format \'{{.Id}}\' dsv41-serve)" = '
            +shlex.quote(container_ids['sparkhead'])+'; cd ~/dsv41-deploy-offload && ./stop.sh', timeout=180, supervised=True)
    print('Replay window complete; the Sparks are released. HOLD remains. Coordinator: W8/restore.sh start, check, close; '
          'then run --phase equiv (x86host CPU only).', flush=True)


def run_equiv(io: IO, source: str, repo: Path) -> None:
    d2 = d2_command(source)
    io.sparks = False
    preflight(io, source, repo)
    complete_check(io.call('equiv-replay-check', 'x86host', d2(f'replay check --out {ROOT}/replay'), timeout=600,
                           supervised=True))
    cap = '/opt/dsv41/x86host-data/dspark/captures/A-capture-20261022/dspark-captures'
    shared = f'--v1 {ROOT}/v1 --captures {cap} --acap {ROOT}/acap --replay {ROOT}/replay'
    eq = ROOT+'/equiv-w9-'+source[:12]
    io.call('equiv-fresh', 'x86host', f'test ! -e {eq} && mkdir -m 700 {eq}', supervised=True)
    io.call('equiv-taps', 'x86host', d2(f'equiv taps {shared} --out {eq}/taps.json'), timeout=1800, supervised=True)
    # One shell owns and waits for all three children; lease loss kills only this shell's group.
    drafts = d2(f'equiv drafts {shared} --ckpt /opt/dsv41/x86host-data/native-1017 '
                f'--official /opt/dsv41/x86host-data/native-1017/inference --work {eq} '
                '--shard "$k/3" --threads 8 --every 1')
    io.call('equiv-drafts', 'x86host', 'set -euo pipefail; pids=(); for k in 0 1 2; do ('+drafts+
            f') > {eq}/drafts-$k.log 2>&1 & pids+=("$!"); done; '
            'for pid in "${pids[@]}"; do wait "$pid"; done; '
            f'for k in 0 1 2; do tail -1 {eq}/drafts-$k.log | grep -Eq "^done [0-9]+ blocks in "; done',
            timeout=14400, supervised=True)
    io.call('equiv-complete', 'x86host', d2(f'window_checks equiv-complete {eq}'), supervised=True)
    io.call('equiv-report', 'x86host', d2(f'equiv report --work {eq} --taps {eq}/taps.json --out {eq}/gate.json'),
            timeout=600, supervised=True)
    equivalence(io.call('equiv-gate', 'x86host', f'cat {eq}/gate.json', supervised=True))
    print('Equivalence gate passed (x86host CPU; the Sparks were not used in this phase).', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--execute', action='store_true')
    ap.add_argument('--source', default='DRY_RUN_SOURCE')
    ap.add_argument('--phase', choices=('window', 'equiv'), default=None,
                    help='live: window (Sparks: replay, copy, stop the experiment) then, after production is restored, '
                         'equiv (x86host CPU); dry-run without --phase covers both')
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[3]
    io: IO
    if a.execute:
        require(os.environ.get('D2_COORDINATOR') == 'Deepseek/coordinator', 'coordinator execution only')
        require(re.fullmatch('[0-9a-f]{40}', a.source) is not None, 'full immutable source commit required')
        actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        require(actual == a.source, 'local runner HEAD differs from source pin')
        require(a.phase is not None, '--phase window or --phase equiv is required with --execute')
        require(a.out is not None and a.out.resolve().is_relative_to(CAMPAIGN), 'new campaign output directory required')
        require((CAMPAIGN/'D2-GEN.PAUSE').exists() or subprocess.run(
            ['tmux', 'has-session', '-t', '=d2-gen'], capture_output=True).returncode != 0, 'generation still running')
        os.umask(0o077)
        lock = (CAMPAIGN/f'W9/replay-{a.phase}.LOCK').open('a')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.out.mkdir(mode=0o700, parents=True, exist_ok=False)
        io = LiveIO(a.out, a.source)
    else:
        io = DryIO()
    try:
        if a.phase == 'window':
            run_window(io, a.source, repo)
        elif a.phase == 'equiv':
            run_equiv(io, a.source, repo)
        else:
            run(io, a.source, repo)
    except BaseException:
        if a.execute and a.out:
            (a.out/'STOP').write_text('Aborted. Inspect owned experiment; production restoration remains coordinator-owned.\n')
        print('STOP. Production restoration is separate: inspect, then W8/restore.sh start/check/close.', flush=True)
        raise


if __name__ == '__main__':
    main()
