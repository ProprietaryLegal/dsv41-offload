#!/usr/bin/env python3
"""UP2-PROMOTE production switch: 0075 replay skip ON through the exact mounts/flag the qualified arm used.

Runs from the PRODUCTION checkout on sparkhead (system python3, stdlib + the installed K1 manager) for `spark-apply`,
`spark-restore` and `verify-running`; `edit-env` and `base` are pure text. It never stops or starts a service: the
promotion runner owns HOLD, stop/start and rollback ordering (offload/promote/up2/promote_window.py).

The switch adds `VLLM_DSV41_REPLAY_SKIP=1` to OFFLOAD_DOCKER_ENV, points production's ONE remote_dspark directory mount
at the staged 0075 package and mounts the four 0075 vLLM files from the same staged directory
(`~/dsv41-offload-prod/up2/<id>/`, stage.sh, immutable). The K1 start guard hashes dsv41.env, so apply rebases exactly
that hash (PROMO1/PP4 pattern) and keeps an exclusive snapshot: restore returns the installed state byte for byte.

Bases (pinned by the runner's preflight, detected here from the installed state; anything else refuses):
  ops1        K1 profile 327680, production remote package   -> package `qualified`
  promo1-low  K1 profile 327680-low (PROMO1), same mounts     -> package `qualified` (0075 shares no file with PROMO1)
  pp4         PROD_PP4 present (PP4-PROMOTE)                  -> package `pp4-composed`, refused until measured
On a PP4 production this transaction stacks on PP4's own: restore UP2 first, then PP4.
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
from typing import cast

HERE = Path(__file__).resolve().parent
DIST = '/usr/local/lib/python3.12/dist-packages'
REMOTE = 'vllm/v1/worker/gpu/spec_decode/remote_dspark'
VLLM_FILES = ('vllm/v1/core/sched/scheduler.py', 'vllm/v1/worker/gpu/input_batch.py',
              'vllm/models/deepseek_v41/nvidia/model_state.py', 'vllm/models/deepseek_v41/replay_skip.py')
SWITCH_ENV = ('VLLM_DSV41_REPLAY_SKIP', '1')
STATE_FILES = ('dsv41.env', 'logs/k1-promotion/files.json')
BACKUP = 'logs/up2-switch'
STAGE = '$HOME/dsv41-offload-prod/up2/'          # + package id; dsv41.env stores \$HOME (sourced, not expanded)
PRODUCTION_REMOTE = '\\$HOME/dsv41-offload-prod/dspark/remote_dspark'
PP4_REMOTE = '\\$HOME/dsv41-offload-prod/pp4/941a3c3fbe6f/remote_dspark'
PACKAGE_FOR = {'ops1': 'qualified', 'promo1-low': 'qualified', 'pp4': 'pp4-composed'}
REMOTE_SOURCE = {'ops1': PRODUCTION_REMOTE, 'promo1-low': PRODUCTION_REMOTE, 'pp4': PP4_REMOTE}
PROFILE = {'ops1': '327680', 'promo1-low': '327680-low'}
MARKER = '# UP2-PROMOTE '


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def package(name: str, here: Path = HERE) -> dict[str, object]:
    value: object = json.loads((here / 'packages' / (name + '.json')).read_text())
    require(isinstance(value, dict) and value.get('schema') == 'up2-promote-package-v1', f'{name}: invalid package')
    manifest = cast(dict[str, object], value)
    files = manifest.get('files')
    require(isinstance(files, dict) and bool(files) and isinstance(manifest.get('id'), str), f'{name}: invalid files/id')
    lines = ''.join(f'{digest}  {path}\n' for path, digest in sorted(cast(dict[str, str], files).items()))
    require(sha(lines.encode())[:12] == manifest['id'], f'{name}: package id does not match its files')
    return manifest


def k1_manager(root: Path) -> ModuleType:
    """The INSTALLED production K1 manager, loaded from its own file (596ba1c imports `profiles` script-style)."""
    k1 = (root / 'offload/promote/k1').resolve()
    for entry in (str(k1), str(root.resolve())):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location('up2_installed_k1_manage', k1 / 'manage.py')
    if spec is None or spec.loader is None:
        raise ValueError('installed K1 manage.py not loadable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assignment(lines: list[str], name: str) -> int:
    """Index of the ONE active single-line `NAME="..."` assignment; comments and look-alikes never match."""
    hits = [i for i, line in enumerate(lines) if line.startswith(name + '="') and line.rstrip('\n').endswith('"')]
    require(len(hits) == 1, f'{name}: expected exactly one active single-line assignment, found {len(hits)}')
    return hits[0]


def tokens(line: str, name: str) -> list[str]:
    value = line.rstrip('\n')[len(name) + 2:-1]
    require('"' not in value and '\n' not in value, f'{name}: unexpected quoting')
    return value.split()


def base_of(text: str, profile: str) -> str:
    """The installed production this switch may stack on; refuses an applied, unknown or half-switched state."""
    require(MARKER not in text and not re.search(r'^PROD_UP2=', text, re.M), 'UP2 switch already present in dsv41.env')
    if re.search(r'^PROD_PP4=', text, re.M):
        require(profile in PROFILE.values(), f'PP4 production on unknown K1 profile {profile!r}')
        return 'pp4'
    for name, wanted in PROFILE.items():
        if profile == wanted:
            return name
    raise ValueError(f'unknown production: K1 profile {profile!r}; only ops1, promo1-low and pp4 compose with 0075')


def edit_env(text: str, base: str, manifest: dict[str, object]) -> str:
    """Installed dsv41.env text -> 0075-ON text for `base`. Refuses anything it cannot place exactly."""
    lines = text.splitlines(keepends=True)
    require(bool(lines) and lines[-1].endswith('\n'), 'dsv41.env must end with a newline')
    key, value = SWITCH_ENV
    for name in ('OFFLOAD_DOCKER_ENV', 'OFFLOAD_DSPARK_ENV'):
        require(not any(t.split('=', 1)[0] == key for t in tokens(lines[assignment(lines, name)], name)),
                f'{key} already set in {name}')
    i = assignment(lines, 'OFFLOAD_DOCKER_ENV')
    lines[i] = 'OFFLOAD_DOCKER_ENV="' + ' '.join([*tokens(lines[i], 'OFFLOAD_DOCKER_ENV'), f'{key}={value}']) + '"\n'
    i = assignment(lines, 'OFFLOAD_DOCKER_ARGS')
    args = tokens(lines[i], 'OFFLOAD_DOCKER_ARGS')
    require(len(args) % 2 == 0 and all(a == '-v' for a in args[::2]), 'OFFLOAD_DOCKER_ARGS: -v pairs required')
    stage = (STAGE + str(manifest['id'])).replace('$HOME', '\\$HOME')
    out: list[str] = []
    replaced = 0
    for mount in args[1::2]:
        require(mount.endswith(':ro') and mount.count(':') == 2, f'unexpected mount {mount}')
        source, target, _ = mount.split(':')
        require(target not in (f'{DIST}/{name}' for name in VLLM_FILES), f'{target} already mounted; unknown combination')
        if target == f'{DIST}/{REMOTE}':
            require(source == REMOTE_SOURCE[base], f'{base}: remote_dspark source {source} is not the expected '
                    f'{REMOTE_SOURCE[base]}; unknown combination')
            source = f'{stage}/remote_dspark'
            replaced += 1
        out.append(f'{source}:{target}:ro')
    require(replaced == 1, f'production remote_dspark mount found {replaced} times, expected exactly once')
    out += [f'{stage}/{name}:{DIST}/{name}:ro' for name in VLLM_FILES]
    lines[i] = 'OFFLOAD_DOCKER_ARGS="' + ' '.join(f'-v {m}' for m in out) + '"\n'
    lines += [f"{MARKER}{manifest['id']}: 0075 replay skip ({manifest['name']}); rollback restores this file "
              'byte-exact (logs/up2-switch).\n', f"PROD_UP2={manifest['id']}\n"]
    return ''.join(lines)


def replace(path: Path, data: bytes) -> None:
    temporary = path.with_name('.' + path.name + '.up2')
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


def selected(root: Path, manage: ModuleType) -> tuple[str, dict[str, object]]:
    base = base_of((root / 'dsv41.env').read_text(), current_profile(root, manage))
    manifest = package(PACKAGE_FOR[base], root / 'offload/promote/up2')
    require(base in cast(list[str], manifest['bases']), f'package {manifest["name"]} does not list base {base}')
    require(manifest.get('arm') is not None, f'{base}: package {manifest["name"]} {manifest["id"]} was never measured; '
            'unknown combination refused (run a qualification window on this exact package first)')
    return base, manifest


def apply(root: Path) -> None:
    manage = k1_manager(root)
    profile = current_profile(root, manage)
    manage.verify(root, profile)                       # the installed (possibly PROMO1-low or PP4) state, unchanged
    base, manifest = selected(root, manage)
    snapshot = root / BACKUP
    snapshot.mkdir(parents=True, exist_ok=False)       # exclusive: a second apply or a stale snapshot refuses
    originals = {name: manage.sha(root / name) for name in STATE_FILES}
    for name in STATE_FILES:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    wanted = edit_env((root / 'dsv41.env').read_text(), base, manifest).encode()
    record = manage.read_object(root / 'logs/k1-promotion/files.json')
    hashes = manage.hashes(record, 'promoted')
    hashes['dsv41.env'] = sha(wanted)
    record['promoted'] = hashes
    intended = {'dsv41.env': wanted, 'logs/k1-promotion/files.json': (json.dumps(record, indent=2) + '\n').encode()}
    ledger = {'package': manifest['id'], 'name': manifest['name'], 'base': base, 'profile': profile,
              'original': originals, 'promoted': {name: sha(data) for name, data in intended.items()}}
    # Intent precedes mutation: a partial switch is always restorable, a foreign later edit always refused.
    (snapshot / 'transaction.json').write_text(json.dumps(ledger, indent=2) + '\n')
    for name, data in intended.items():
        replace(root / name, data)
    manage.verify(root, profile)
    print(f"UP2 SWITCH APPLIED {manifest['id']} on base {base} (profile {profile})")


def restore(root: Path) -> None:
    manage = k1_manager(root)
    snapshot = root / BACKUP
    ledger = json.loads((snapshot / 'transaction.json').read_text())
    require(isinstance(ledger, dict) and isinstance(ledger.get('package'), str), 'invalid UP2 transaction')
    original, promoted = ledger.get('original'), ledger.get('promoted')
    require(isinstance(original, dict) and isinstance(promoted, dict)
            and set(original) == set(promoted) == set(STATE_FILES), 'invalid UP2 transaction hashes')
    for name in STATE_FILES:
        require(manage.sha(snapshot / name) == original[name], f'UP2 backup damaged: {name}; HOLD retained')
        require(manage.sha(root / name) in (original[name], promoted[name]), f'later edit of {name}; HOLD retained')
    for name in STATE_FILES:
        if manage.sha(root / name) != original[name]:
            replace(root / name, (snapshot / name).read_bytes())
        require(manage.sha(root / name) == original[name], f'byte-exact restore failed: {name}')
    manage.verify(root, str(ledger.get('profile')))
    # The snapshot is retained (evidence); a later apply needs the coordinator to archive it first.
    print(f"UP2 SWITCH RESTORED {ledger['package']}; installed state byte-exact")


def running_probe(package_id: str) -> str:
    paths = ' '.join(f'{DIST}/{name}' for name in VLLM_FILES)
    code = f"import json, os; print(json.dumps({{'{SWITCH_ENV[0]}': os.environ.get('{SWITCH_ENV[0]}')}}))"
    listing = f"cd {DIST}/{REMOTE} && find . -name '*.py' -not -path '*/__pycache__/*' | LC_ALL=C sort | xargs sha256sum"
    return (f'docker exec dsv41-serve sha256sum {paths} && docker exec dsv41-serve sh -c {shlex.quote(listing)} && '
            f'echo UP2-STAGED && cat {STAGE}{package_id}/SHA256SUMS && echo UP2-ENV && '
            f'docker exec dsv41-serve python3 -c {shlex.quote(code)}')


def check_running(node: str, text: str, manifest: dict[str, object]) -> None:
    head, _, rest = text.partition('UP2-STAGED\n')
    staged_text, _, env_text = rest.partition('UP2-ENV\n')
    require(bool(env_text), f'{node}: UP2 probe incomplete')
    files = cast(dict[str, str], manifest['files'])
    running: dict[str, str] = {}
    for line in head.splitlines():
        digest, _, path = line.partition('  ')
        require(re.fullmatch('[0-9a-f]{64}', digest) is not None, f'{node}: unexpected probe line {line!r}')
        if path.startswith(DIST + '/'):
            running[path[len(DIST) + 1:]] = digest
        elif path.startswith('./') and '/' not in path[2:]:
            running['remote_dspark/' + path[2:]] = digest
        else:
            raise ValueError(f'{node}: unexpected probe path {path!r}')
    staged = {path: digest for digest, _, path in (line.partition('  ') for line in staged_text.splitlines())}
    require(staged == files, f'{node}: staged package differs from {manifest["id"]}')
    require(running == files, f'{node}: running 0075 files/remote_dspark package differ from {manifest["id"]}')
    env = json.loads(env_text.strip().splitlines()[-1])
    require(isinstance(env, dict) and env.get(SWITCH_ENV[0]) == SWITCH_ENV[1],
            f'{node}: effective {SWITCH_ENV[0]}={env.get(SWITCH_ENV[0]) if isinstance(env, dict) else env!r}')
    print(f'UP2 RUNNING VERIFIED {node}')


def verify_running(root: Path) -> None:
    env = subprocess.run(['bash', '-c', 'set -a; . ./dsv41.env >/dev/null; echo "$HEAD $WORKER $PROD_UP2"'],
                         cwd=root, check=True, capture_output=True, text=True).stdout.split()
    require(len(env) == 3, 'dsv41.env does not select a UP2 production')
    ledger = json.loads((root / BACKUP / 'transaction.json').read_text())
    manifest = package(str(ledger['name']), root / 'offload/promote/up2')
    require(env[2] == manifest['id'] == ledger['package'], 'dsv41.env PROD_UP2 differs from the applied package')
    for node in env[:2]:
        text = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', node, running_probe(env[2])],
                              check=True, capture_output=True, text=True, timeout=120).stdout
        check_running(node, text, manifest)
    print('UP2 RUNNING VERIFIED')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('spark-apply', 'spark-restore', 'verify-running', 'edit-env', 'base'))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--env', type=Path, help='edit-env/base: this dsv41.env text (no write)')
    parser.add_argument('--profile', help='edit-env/base: the installed K1 profile')
    args = parser.parse_args()
    try:
        if args.action == 'spark-apply':
            apply(args.root)
        elif args.action == 'spark-restore':
            restore(args.root)
        elif args.action == 'verify-running':
            verify_running(args.root)
        else:
            require(args.env is not None and args.profile is not None, '--env and --profile required')
            text = Path(args.env).read_text()
            base = base_of(text, args.profile)
            if args.action == 'base':
                print(base)
            else:
                manifest = package(PACKAGE_FOR[base])
                sys.stdout.write(edit_env(text, base, manifest))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f'UP2 SWITCH REFUSED: {exc}\n')


if __name__ == '__main__':
    main()
