#!/usr/bin/env python3
"""PP4-PROMOTE production switch: byte-exact Spark env transaction, running verification, x86host drop-in text.

Runs from the PRODUCTION checkout on sparkhead (system python3, stdlib + K1 manager only) for `fold-clock` (once, after
the staging commit is pushed), `spark-apply`, `spark-restore`, `verify-running` and `active` (the production hooks'
validated selection check); `render-dropin` is pure text and runs anywhere. It never stops or starts a
service: the promotion runner owns HOLD, stop/start and rollback ordering.

The K1 start guard hashes dsv41.env (logs/k1-promotion/files.json `promoted`). Apply therefore rebases exactly that
one hash, like PROMO1's switch_profile.py, and keeps an exclusive snapshot so restore returns the installed state
byte for byte (never the pre-K1 manager backup).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from types import ModuleType

PP4_COMMIT = '90f0278e900c5446d627fd5f42db15246cd544b5'
PP4_ID = PP4_COMMIT[:12]
DIST = '/usr/local/lib/python3.12/dist-packages'
RUNNER = 'vllm/v1/worker/gpu/model_runner.py'
UTILS = 'vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py'
TRACE = 'vllm/v1/worker/gpu/dsv41_pp4_round_trace.py'
REMOTE = 'vllm/v1/worker/gpu/spec_decode/remote_dspark'
# Ordinary (measurement) verifier from the qualified admission seed `verifier_sha256`; the admission variant is never used.
VERIFIER_SHA256 = {RUNNER: 'e0a11b89a99db93e34f586c03162bb9a4fb3325b6572e88b76f7b14f1a837a97',
                   UTILS: '1468ec723b116013912911625c6d489acc69aedc7a0dffc540d9dd42605c58de',
                   TRACE: '5b9165adfa6a065497aa3d1fe78f468484d25ecab22230dd6aaa09cf3a44a4d3'}
STAGE = '$HOME/dsv41-offload-prod/pp4/' + PP4_ID   # per node; dsv41.env stores it as \$HOME (sourced, not expanded)
# PP4-sampled.env / pp4_window.COMMON_ENV, minus the experiment-only round directory; identities stay production's.
PP4_ENV = (('VLLM_DSV41_ADAPTIVE_K', '0'), ('VLLM_DSPARK_DEPTH_POLICY', 'fixed:5'), ('VLLM_DSPARK_DEPTH_FILE', ''),
           ('VLLM_DSPARK_LOOKUP', ''), ('VLLM_DSPARK_REMOTE_WANT_MARGIN', '0'), ('VLLM_DSPARK_SAMPLED_TOP32', '1'))
PP4_KEYS = frozenset(k for k, _ in PP4_ENV) | {'VLLM_PP4_ROUND_DIR', 'VLLM_PP4_ADMISSION_RUN'}
SPEC = '{"method":"dspark","num_speculative_tokens":5}'
STATE_FILES = ('dsv41.env', 'logs/k1-promotion/files.json')
BACKUP = 'logs/pp4-switch'
# Repo plan Phase 0 (coordinator 2026-10-27): the production commit carries 420be24's clock fold, cherry-picked on top of
# production before the PP4 hooks. `fold-clock` rebases the K1 record's dsv41.env hash for exactly that one line.
CLOCK_FOLD_COMMIT = '420be244a524b34a904a0d7cca0ec954057af00b'
CLOCK_PRE = "GPU_CLOCK_CAP=${GPU_CLOCK_CAP-2150}         # nvidia-smi -lgc 0,$GPU_CLOCK_CAP on both nodes at start (non-persistent; resets on reboot). Empty = stock. Author's fallback: 2000" + '\n'
CLOCK_FOLDED = "GPU_CLOCK_CAP=${GPU_CLOCK_CAP-2300}         # NDM 2026-10-27 persistent 2300 MHz (was a /etc/environment override; dial-back writes an explicit 2150 there). nvidia-smi -lgc 0,$GPU_CLOCK_CAP on both nodes at start (non-persistent; resets on reboot). Empty = stock. Author's fallback: 2000" + '\n'
CLOCK_DEFAULT = 'GPU_CLOCK_CAP=${GPU_CLOCK_CAP-2300}'
FOLD = 'logs/clock-fold'
RESTORED = 'restored.json'   # written only after a byte-exact, K1-verified restore; disarms the K1-fallback delegation
MARKER = '# PP4-PROMOTE ' + PP4_ID
# X86host layout (offload/dspark/worker/run/prod_tp2.sh stage names a release by its 7-character short commit).
PYTHON = '/home/operator/miniconda3/bin/python3'
RELEASE = '%h/dsv41-offload/prod/dspark-worker/' + PP4_COMMIT[:7]
RUNTIME = '%h/dsv41-offload/prod/pp4-runtime/' + PP4_ID
DROPIN_DIR = '.config/systemd/user/dsv41-dspark-worker-tp2-prod.service.d'
DROPIN = DROPIN_DIR + '/50-pp4.conf'
UNIT = 'dsv41-dspark-worker-tp2-prod.service'
SO_SHA256 = {'stock/exllamav3_ext.cpython-313-x86_64-linux-gnu.so':
                 '6c77786a3755fa60f9548b86fa0650c4c47100dc3728cda9567658580ccd653e',
             'facc/exllamav3_ext_facc.cpython-313-x86_64-linux-gnu.so':
                 '2582a363ef76b3338b80dfdd4296025fb5c1743c610748bd0e7e5b75d8b0d61e',
             'p2p/08e1903b0a43feff/dspark_p2p.so':
                 '92d1b8c003df23b871b33217931c89e4487e6027fd705d46621ebf838f625cf6'}
TUNE = 'exl3_tune_cache-reviewed.bin'
TUNE_SHA256 = '937e9633334be0a5a7a97557e83f962bfbc33dce84a76b93e331b8017b0d405e'


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def k1_manager(root: Path) -> ModuleType:
    """The INSTALLED production K1 manager, loaded from its own file.

    596ba1c's manage.py imports `profiles` as a script sibling (not importable as a package module); PROMO1's version
    uses package imports. Loading by path, with its directory first on sys.path, works for both.
    """
    k1 = (root / 'offload/promote/k1').resolve()
    for entry in (str(k1), str(root.resolve())):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location('pp4_installed_k1_manage', k1 / 'manage.py')
    if spec is None or spec.loader is None:
        raise ValueError('installed K1 manage.py not loadable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assignment(lines: list[str], name: str) -> int:
    """Index of the ONE active single-line `NAME="..."` assignment; comments and look-alikes never match."""
    hits = [i for i, line in enumerate(lines) if line.startswith(name + '="') and line.rstrip('\n').endswith('"')]
    require(len(hits) == 1, f'{name}: expected exactly one active single-line assignment, found {len(hits)}')
    return hits[0]


def tokens(line: str, name: str) -> list[str]:
    value = line.rstrip('\n')[len(name) + 2:-1]
    require('"' not in value and '\n' not in value, f'{name}: unexpected quoting')
    return value.split()


def edit_env(text: str) -> str:
    """Production dsv41.env text -> PP4 production text. Refuses anything it cannot place exactly."""
    require(MARKER not in text and 'PROD_PP4=' not in text, 'PP4 switch already present in dsv41.env')
    lines = text.splitlines(keepends=True)
    require(bool(lines) and lines[-1].endswith('\n'), 'dsv41.env must end with a newline')
    for name in ('OFFLOAD_DOCKER_ENV', 'OFFLOAD_DSPARK_ENV'):
        i = assignment(lines, name)
        kept = [t for t in tokens(lines[i], name) if t.split('=', 1)[0] not in PP4_KEYS]
        if name == 'OFFLOAD_DSPARK_ENV':
            require(any(t.startswith('VLLM_DSPARK_REMOTE=') for t in kept), 'production remote DSpark absent')
            kept += [f'{k}={v}' for k, v in PP4_ENV]
        lines[i] = f'{name}="{" ".join(kept)}"\n'
    i = assignment(lines, 'OFFLOAD_DOCKER_ARGS')
    args = tokens(lines[i], 'OFFLOAD_DOCKER_ARGS')
    require(len(args) % 2 == 0 and all(a == '-v' for a in args[::2]), 'OFFLOAD_DOCKER_ARGS: -v pairs required')
    mounts = args[1::2]
    stage = STAGE.replace('$HOME', '\\$HOME')
    wanted = {f'{DIST}/{REMOTE}': f'{stage}/remote_dspark', f'{DIST}/{RUNNER}': f'{stage}/verifier/{RUNNER}'}
    seen: set[str] = set()
    out: list[str] = []
    for mount in mounts:
        require(mount.endswith(':ro') and mount.count(':') == 2, f'unexpected mount {mount}')
        source, target, _ = mount.split(':')
        require(target not in (f'{DIST}/{UTILS}', f'{DIST}/{TRACE}'), f'{target} already mounted')
        if target in wanted:
            require(target not in seen, f'duplicate mount {target}')
            seen.add(target)
            source = wanted[target]
        out.append(f'{source}:{target}:ro')
    require(seen == set(wanted), 'production remote_dspark/model_runner.py mounts not found exactly once')
    out += [f'{stage}/verifier/{name}:{DIST}/{name}:ro' for name in (UTILS, TRACE)]
    lines[i] = 'OFFLOAD_DOCKER_ARGS="' + ' '.join(f'-v {m}' for m in out) + '"\n'
    lines += [MARKER + ': sampled K2 drafting; rollback restores this file byte-exact (logs/pp4-switch).\n',
              f'PROD_PP4={PP4_ID}\n', f"PROD_SPEC_OVERRIDE='{SPEC}'\n",
              f'PROD_PP4_RUNNER_SHA256={VERIFIER_SHA256[RUNNER]}\n']
    return ''.join(lines)


def replace(path: Path, data: bytes) -> None:
    temporary = path.with_name('.' + path.name + '.pp4')
    with temporary.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    shutil.copymode(path, temporary)
    os.replace(temporary, path)


def current_profile(root: Path, manage: ModuleType) -> str:
    value = manage.read_object(root / 'logs/k1-promotion/files.json').get('profile')
    require(isinstance(value, str), 'promotion record has no explicit profile')
    return str(value)


def apply(root: Path) -> None:
    manage = k1_manager(root)
    profile = current_profile(root, manage)
    manage.verify(root, profile)                       # the installed (possibly PROMO1-low) state, unchanged
    snapshot = root / BACKUP
    snapshot.mkdir(parents=True, exist_ok=False)       # exclusive: a second apply or a stale snapshot refuses
    originals = {name: manage.sha(root / name) for name in STATE_FILES}
    for name in STATE_FILES:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    wanted = edit_env((root / 'dsv41.env').read_text()).encode()
    record = manage.read_object(root / 'logs/k1-promotion/files.json')
    hashes = manage.hashes(record, 'promoted')
    hashes['dsv41.env'] = sha(wanted)
    record['promoted'] = hashes
    intended = {'dsv41.env': wanted, 'logs/k1-promotion/files.json': (json.dumps(record, indent=2) + '\n').encode()}
    ledger = {'pp4': PP4_COMMIT, 'profile': profile, 'original': originals,
              'promoted': {name: sha(data) for name, data in intended.items()}}
    # Intent precedes mutation: a partial switch is always restorable, a foreign later edit always refused.
    (snapshot / 'transaction.json').write_text(json.dumps(ledger, indent=2) + '\n')
    for name, data in intended.items():
        replace(root / name, data)
    manage.verify(root, profile)
    print(f'PP4 SWITCH APPLIED {PP4_ID} on profile {profile}')


def fold_clock(root: Path) -> None:
    """After the staging commit (with the 420be24 fold) is pushed: rebase the K1 record's dsv41.env hash, proving that the
    env differs from the K1-verified one by exactly the clock line. Idempotent; the ledger and a files.json copy stay."""
    manage = k1_manager(root)
    profile = current_profile(root, manage)
    env = (root / 'dsv41.env').read_bytes()
    text = env.decode()
    require(text.count(CLOCK_FOLDED) == 1 and CLOCK_PRE not in text,
            'dsv41.env lacks exactly one folded 2300 clock line; cherry-pick 420be24 onto production first')
    record_path = root / 'logs/k1-promotion/files.json'
    record = manage.read_object(record_path)
    hashes = manage.hashes(record, 'promoted')
    ledger = root / FOLD / 'transaction.json'
    if hashes['dsv41.env'] == sha(env):
        require(ledger.is_file() and json.loads(ledger.read_text()).get('env') == sha(env),
                'K1 record already names this env without this clock-fold ledger; refused')
        manage.verify(root, profile)
        print('CLOCK FOLD ALREADY RECORDED 2300')
        return
    require(sha(text.replace(CLOCK_FOLDED, CLOCK_PRE).encode()) == hashes['dsv41.env'],
            'dsv41.env differs from the K1-verified env beyond the clock line; refused')
    (root / FOLD).mkdir(parents=True, exist_ok=False)
    before = record_path.read_bytes()
    (root / FOLD / 'files.json').write_bytes(before)
    hashes['dsv41.env'] = sha(env)
    record['promoted'] = hashes
    after = (json.dumps(record, indent=2) + '\n').encode()
    ledger.write_text(json.dumps({'fold': CLOCK_FOLD_COMMIT, 'profile': profile, 'env': sha(env),
                                  'files_json_before': sha(before), 'files_json_after': sha(after)}, indent=2) + '\n')
    replace(record_path, after)
    manage.verify(root, profile)
    print(f'CLOCK FOLD RECORDED 2300 on profile {profile}')


def restore(root: Path) -> None:
    manage = k1_manager(root)
    snapshot = root / BACKUP
    ledger = json.loads((snapshot / 'transaction.json').read_text())
    require(isinstance(ledger, dict) and ledger.get('pp4') == PP4_COMMIT, 'invalid PP4 transaction')
    original, promoted = ledger.get('original'), ledger.get('promoted')
    require(isinstance(original, dict) and isinstance(promoted, dict)
            and set(original) == set(promoted) == set(STATE_FILES), 'invalid PP4 transaction hashes')
    for name in STATE_FILES:
        require(manage.sha(snapshot / name) == original[name], f'PP4 backup damaged: {name}; HOLD retained')
        require(manage.sha(root / name) in (original[name], promoted[name]), f'later edit of {name}; HOLD retained')
    for name in STATE_FILES:
        if manage.sha(root / name) != original[name]:
            replace(root / name, (snapshot / name).read_bytes())
        require(manage.sha(root / name) == original[name], f'byte-exact restore failed: {name}')
    manage.verify(root, str(ledger.get('profile')))
    # The snapshot is retained (evidence); a later apply needs the coordinator to archive it first. The marker turns the
    # retained transaction into evidence only: the K1 fallback and every override hook treat it as no longer active.
    marker = snapshot / RESTORED
    if marker.exists():
        require(json.loads(marker.read_text()).get('original') == original, 'foreign PP4 restore marker; HOLD retained')
    else:
        temporary = marker.with_name('.' + RESTORED + '.pp4')
        with temporary.open('x') as handle:
            handle.write(json.dumps({'pp4': PP4_COMMIT, 'original': original, 'restored': 'byte-exact, K1 verified'},
                                    indent=2) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker)
    print(f'PP4 SWITCH RESTORED {PP4_ID}; installed state byte-exact')


def env_value(lines: list[str], name: str) -> str:
    hits = [line[len(name) + 1:] for line in lines if line.startswith(name + '=')]
    require(len(hits) == 1, f'switched dsv41.env must set {name} exactly once')
    return hits[0]


def active_state(root: Path) -> dict[str, str]:
    """The validated ACTIVE selection: an open (not restored) PP4 transaction whose switched bytes are installed."""
    snapshot = root / BACKUP
    ledger = json.loads((snapshot / 'transaction.json').read_text())
    require(isinstance(ledger, dict) and ledger.get('pp4') == PP4_COMMIT, 'no PP4 switch transaction for ' + PP4_ID)
    require(not (snapshot / RESTORED).exists(), 'PP4 switch was restored; not active')
    promoted = ledger.get('promoted')
    require(isinstance(promoted, dict) and set(promoted) == set(STATE_FILES), 'invalid PP4 transaction hashes')
    for name in STATE_FILES:
        require(sha((root / name).read_bytes()) == promoted[name], f'{name} is not the switched PP4 state')
    lines = (root / 'dsv41.env').read_text().splitlines()
    require(env_value(lines, 'PROD_PP4') == PP4_ID, 'switched dsv41.env selects another PP4')
    return {'spec': env_value(lines, 'PROD_SPEC_OVERRIDE').removeprefix("'").removesuffix("'"),
            'runner': env_value(lines, 'PROD_PP4_RUNNER_SHA256')}


def check_active(root: Path, spec: str | None, runner: str | None) -> None:
    state = active_state(root)
    require(spec is None or spec == state['spec'] == SPEC, 'PROD_SPEC_OVERRIDE differs from the switched fixed5 window')
    require(runner is None or runner == state['runner'], 'declared model_runner.py sha differs from the switched one')
    print(f'PP4 ACTIVE {PP4_ID}')


def running_probe() -> str:
    """One running container, bracketed by its id: bytes, effective env and the launched argv (/serve.sh)."""
    paths = ' '.join(f'{DIST}/{name}' for name in VERIFIER_SHA256)
    keys = json.dumps(sorted(PP4_KEYS | {'VLLM_DSPARK_REMOTE_IDENTITY', 'VLLM_DSPARK_REMOTE_MARKOV_IDENTITY'}))
    code = f'import json, os; print(json.dumps({{k: os.environ.get(k) for k in {keys}}}))'
    container = f'docker inspect -f {shlex.quote("{{.Id}}")} dsv41-serve | sed "s/^/PP4-ID /"'
    return (f'{container} && docker exec dsv41-serve sha256sum {paths} && '
            f'docker exec dsv41-serve sh -c {shlex.quote(f"cd {DIST}/{REMOTE} && find . -name \'*.py\' -not -path \'*/__pycache__/*\' | LC_ALL=C sort | xargs sha256sum")} && '
            f'cat {STAGE}/SHA256SUMS && echo PP4-ENV && docker exec dsv41-serve python3 -c {shlex.quote(code)} && '
            f'echo PP4-ARGV && docker exec dsv41-serve cat /serve.sh && echo && {container}')


def check_running(node: str, text: str) -> str:
    """Validate one node's probe; returns the container id the whole probe observed."""
    lines = text.rstrip('\n').splitlines()
    require(len(lines) >= 2 and lines[0].startswith('PP4-ID ') and lines[-1] == lines[0]
            and re.fullmatch('[0-9a-f]{64}', lines[0][len('PP4-ID '):]) is not None,
            f'{node}: probe not bound to one unchanged dsv41-serve container')
    container = lines[0][len('PP4-ID '):]
    body = '\n'.join(lines[1:-1]) + '\n'
    head, _, rest = body.partition('PP4-ENV\n')
    env_text, _, argv_text = rest.partition('PP4-ARGV\n')
    require(bool(env_text) and bool(argv_text.strip()), f'{node}: PP4 env/argv probe missing')
    got: dict[str, str] = {}
    staged: dict[str, str] = {}
    for line in head.splitlines():
        digest, _, path = line.partition('  ')
        require(re.fullmatch('[0-9a-f]{64}', digest) is not None, f'{node}: unexpected probe line {line!r}')
        if path.startswith(DIST + '/'):
            got[path[len(DIST) + 1:]] = digest
        elif path.startswith('./remote_dspark/'):
            staged['remote/' + path[len('./remote_dspark/'):]] = digest
        elif path.startswith('./verifier/'):
            staged['verifier/' + path[len('./verifier/'):]] = digest
        elif path.startswith('./'):
            got['remote/' + path[2:]] = digest
        else:
            raise ValueError(f'{node}: unexpected probe path {path!r}')
    for name, want in VERIFIER_SHA256.items():
        require(got.get(name) == want and staged.get('verifier/' + name) == want, f'{node}: {name} not the qualified PP4 bytes')
    remote = {k: v for k, v in got.items() if k.startswith('remote/')}
    require(bool(remote) and remote == {k: v for k, v in staged.items() if k.startswith('remote/')},
            f'{node}: running remote_dspark differs from the staged PP4 package')
    env = json.loads(env_text.strip().splitlines()[-1])
    require(isinstance(env, dict), f'{node}: env probe malformed')
    for key, want in PP4_ENV:
        require(env.get(key) == want, f'{node}: effective {key}={env.get(key)!r}, expected {want!r}')
    require(env.get('VLLM_PP4_ROUND_DIR') is None and env.get('VLLM_PP4_ADMISSION_RUN') is None,
            f'{node}: experiment-only PP4 settings present in production')
    argv = shlex.split(argv_text.replace('\\\n', ' '))
    require(argv.count('--speculative-config') == 1, f'{node}: launched argv lacks exactly one speculative config')
    require(argv[argv.index('--speculative-config') + 1] == SPEC, f'{node}: launched verify window is not fixed5')
    print(f'PP4 RUNNING VERIFIED {node} container {container}')
    return container


def verify_running(root: Path) -> None:
    env = subprocess.run(['bash', '-c', 'set -a; . ./dsv41.env >/dev/null; echo "$HEAD $WORKER $PROD_PP4"'],
                         cwd=root, check=True, capture_output=True, text=True).stdout.split()
    require(len(env) == 3 and env[2] == PP4_ID, 'dsv41.env does not select this PP4 production')
    for node in env[:2]:
        text = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', node, running_probe()],
                              check=True, capture_output=True, text=True, timeout=120).stdout
        check_running(node, text)
    print('PP4 RUNNING VERIFIED')


def render_dropin() -> str:
    """systemd drop-in for the existing TP2 production unit: everything not named stays the unit's own."""
    attest = f'{RUNTIME}/attest.sh'
    locks = '%h/dsv41-offload/locks/gpu-c4e7a371.lock %h/dsv41-offload/locks/gpu-7744b44b.lock'
    return '\n'.join((
        f'# PP4-PROMOTE {PP4_ID}: sampled K2 drafting from qualified bytes (pp4_promote_switch.py render-dropin).',
        '# Rollback: remove exactly this file (hash-checked), daemon-reload, restart through prod_window.sh.',
        '[Service]',
        'Environment=PYTHONDONTWRITEBYTECODE=1',
        f'Environment=PYTHONPATH={RUNTIME}/stock:{RUNTIME}/facc:{RELEASE}/offload/dspark',
        f'Environment=DSPARK_P2P_PREBUILT={RUNTIME}/p2p',
        f'Environment=EXLLAMAV3_TUNE_CACHE={RUNTIME}/{TUNE}',
        'Environment=DSPARK_WORKER_ARM=k2-tp2',
        'Environment=DSPARK_SAMPLED_TOP32=1',
        f'Environment=DSPARK_SOURCE_COMMIT={PP4_COMMIT}',
        f'WorkingDirectory={RELEASE}/offload/dspark',
        f'ExecStartPre=/bin/sh {attest}',
        'ExecStart=',
        f'ExecStart=/bin/sh {RELEASE}/offload/dspark/worker/run/gpu_locked.sh {locks} -- {PYTHON} -u -m worker.server '
        '--tp 2 --root /opt/dsv41/x86host-data/dspark --ofl 10.202.28.1:18620 --max-model-len 1048576 '
        '--status-file %h/dsv41-offload/prod/run/dspark-worker.status.json',
        ''))


def render_attest() -> str:
    """Runtime attestation the unit runs before every start: drift refuses a start rather than changing the drafter."""
    return '\n'.join((
        '#!/bin/sh',
        f'# PP4-PROMOTE {PP4_ID} runtime attestation (pp4_promote_switch.py render-attest).',
        'set -eu',
        'cd "$(dirname "$0")"',
        'sha256sum -c --quiet SHA256SUMS',
        *(f'test "$(sha256sum {name} | cut -d" " -f1)" = {digest}' for name, digest in sorted(SO_SHA256.items())),
        f"{PYTHON} -I -c 'import sys, torch; assert sys.version_info[:2] == (3, 13); "
        "assert torch.__version__ == \"2.12.1+cu130\"; assert torch.version.cuda == \"13.0\"'",
        'echo "PP4 RUNTIME ATTESTED"',
        ''))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('spark-apply', 'spark-restore', 'verify-running', 'render-dropin',
                                           'render-attest', 'edit-env', 'active', 'fold-clock'))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--env', type=Path, help='edit-env: print the PP4 text for this dsv41.env (no write)')
    parser.add_argument('--spec', help='active: the override the caller is about to apply')
    parser.add_argument('--runner-sha256', help='active: the model_runner.py replacement the caller declares')
    args = parser.parse_args()
    try:
        if args.action == 'spark-apply':
            apply(args.root)
        elif args.action == 'spark-restore':
            restore(args.root)
        elif args.action == 'verify-running':
            verify_running(args.root)
        elif args.action == 'active':
            check_active(args.root, args.spec, args.runner_sha256)
        elif args.action == 'fold-clock':
            fold_clock(args.root)
        elif args.action == 'render-dropin':
            sys.stdout.write(render_dropin())
        elif args.action == 'render-attest':
            sys.stdout.write(render_attest())
        else:
            require(args.env is not None, '--env required')
            sys.stdout.write(edit_env(Path(args.env).read_text()))
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f'PP4 SWITCH REFUSED: {exc}\n')


if __name__ == '__main__':
    main()
