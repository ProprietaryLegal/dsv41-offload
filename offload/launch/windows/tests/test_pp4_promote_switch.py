"""PP4-PROMOTE switch + package: real production env/tree bytes, byte-exact transaction, hook behaviour."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import sys
import types

import pytest

from offload.launch.windows import pp4_promote_package as package
from offload.launch.windows import pp4_promote_switch as switch

ROOT = Path(__file__).resolve().parents[4]
PRODUCTION = '596ba1cf4e3b11be29b41a57e7f69a5d8efa4799'
PROMO1_STAGED = 'af5eaf6'          # coordinator's PROMO1 production staging commit (596ba1c + PROMO1 package)
PROMO1 = '4740ce9f2a3ca610dfc8621e01d49e0772fb987f'


def show(commit: str, path: str) -> str:
    return subprocess.run(['git', '-C', str(ROOT), 'show', f'{commit}:{path}'], check=True,
                          capture_output=True, text=True).stdout


def production_env(low: bool, tmp_path: Path) -> str:
    text = show(PRODUCTION, 'dsv41.env')
    if not low:
        return text
    env = tmp_path / 'low.env'
    env.write_text(text)
    patch = tmp_path / 'low.patch'
    patch.write_text(show(PROMO1, 'offload/promote/k1/dsv41.env.327680-low.patch'))
    subprocess.run(['patch', '-s', str(env), str(patch)], check=True)
    return env.read_text()


def sourced(text: str, tmp_path: Path, *names: str) -> list[str]:
    env = tmp_path / 'sourced.env'
    env.write_text(text)
    script = 'set -a; . "$1" >/dev/null; ' + '; '.join(f'printf "%s\\n" "${name}"' for name in names)
    return subprocess.run(['bash', '-c', script, '_', str(env)], check=True, capture_output=True, text=True).stdout.splitlines()


@pytest.mark.parametrize('low', [False, True])
def test_edit_env_registers_exactly_the_pp4_switch(tmp_path: Path, low: bool) -> None:
    before = production_env(low, tmp_path)
    after = switch.edit_env(before)
    old = before.splitlines()
    new = after.splitlines()
    changed = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
    assert [old[i].split('=', 1)[0] for i in changed] == ['OFFLOAD_DSPARK_ENV', 'OFFLOAD_DOCKER_ARGS']
    assert new[len(old):] == [switch.MARKER + ': sampled K2 drafting; rollback restores this file byte-exact (logs/pp4-switch).',
                              f'PROD_PP4={switch.PP4_ID}', f"PROD_SPEC_OVERRIDE='{switch.SPEC}'",
                              f'PROD_PP4_RUNNER_SHA256={switch.VERIFIER_SHA256[switch.RUNNER]}']
    dspark, docker, args, spec, profile, kv = sourced(after, tmp_path, 'OFFLOAD_DSPARK_ENV', 'OFFLOAD_DOCKER_ENV',
                                                      'OFFLOAD_DOCKER_ARGS', 'PROD_SPEC_OVERRIDE', 'PROD_K1_PROFILE', 'PROD_KV_BYTES')
    tokens = dspark.split()
    assert tokens[-6:] == [f'{k}={v}' for k, v in switch.PP4_ENV]
    assert all(t.split('=', 1)[0] not in switch.PP4_KEYS for t in tokens[:-6] + docker.split())
    old_dspark, old_args, old_profile, old_kv = sourced(before, tmp_path, 'OFFLOAD_DSPARK_ENV', 'OFFLOAD_DOCKER_ARGS',
                                                        'PROD_K1_PROFILE', 'PROD_KV_BYTES')
    assert tokens[:-6] == [t for t in old_dspark.split() if t.split('=', 1)[0] not in switch.PP4_KEYS]
    assert (profile, kv) == (old_profile, old_kv) and spec == switch.SPEC
    mounts, old_mounts = args.split()[1::2], old_args.split()[1::2]
    assert len(mounts) == len(old_mounts) + 2
    stage = switch.STAGE                      # sourced literally (\$HOME); docker's shell expands it per node
    target = {m.split(':')[1]: m.split(':')[0] for m in mounts}
    assert target[f'{switch.DIST}/{switch.REMOTE}'] == f'{stage}/remote_dspark'
    for name in switch.VERIFIER_SHA256:
        assert target[f'{switch.DIST}/{name}'] == f'{stage}/verifier/{name}'
    unchanged = {m for m in old_mounts if not m.split(':')[1].endswith(('/remote_dspark', '/model_runner.py'))}
    assert unchanged <= set(mounts)


@pytest.mark.parametrize('change', ['twice', 'no-remote', 'duplicate-args', 'utils-mounted', 'comment-only', 'no-newline'])
def test_edit_env_refuses_anything_it_cannot_place(tmp_path: Path, change: str) -> None:
    text = production_env(False, tmp_path)
    if change == 'twice':
        text = switch.edit_env(text)
    elif change == 'no-remote':
        text = text.replace('spec_decode/remote_dspark:ro', 'spec_decode/other:ro')
    elif change == 'duplicate-args':
        line = next(x for x in text.splitlines(True) if x.startswith('OFFLOAD_DOCKER_ARGS="'))
        text += line
    elif change == 'utils-mounted':
        text = text.replace('spec_decode/__init__.py:ro', 'spec_decode/__init__.py:ro -v /x:' + switch.DIST + '/' + switch.UTILS + ':ro')
    elif change == 'comment-only':
        text = ''.join('#' + x if x.startswith('OFFLOAD_DSPARK_ENV="') else x for x in text.splitlines(True))
    else:
        text = text.rstrip('\n')
    with pytest.raises(ValueError):
        switch.edit_env(text)


def fold(env: Path) -> None:
    """What `git cherry-pick 420be24` does to a production dsv41.env."""
    text = env.read_text()
    assert text.count(switch.CLOCK_PRE) == 1
    env.write_text(text.replace(switch.CLOCK_PRE, switch.CLOCK_FOLDED))


def test_clock_lines_are_the_production_and_420be24_bytes() -> None:
    assert switch.CLOCK_PRE in show(PRODUCTION, 'dsv41.env') and switch.CLOCK_PRE in show(PROMO1_STAGED, 'dsv41.env')
    assert switch.CLOCK_FOLDED in show(switch.CLOCK_FOLD_COMMIT, 'dsv41.env')
    assert switch.CLOCK_DEFAULT in switch.CLOCK_FOLDED and switch.CLOCK_DEFAULT not in switch.CLOCK_PRE
    diff = subprocess.run(['git', '-C', str(ROOT), 'diff', '--name-only', PRODUCTION, switch.CLOCK_FOLD_COMMIT],
                          check=True, capture_output=True, text=True).stdout.split()
    assert diff == ['dsv41.env']


def installed_tree(tmp_path: Path, base: str = PRODUCTION) -> Path:
    """A production checkout's K1 package (its real manage.py/profiles.py) plus an installed K1 record."""
    root = tmp_path / 'prod'
    state = root / 'logs/k1-promotion'
    state.mkdir(parents=True)
    archive = subprocess.run(['git', '-C', str(ROOT), 'archive', base, 'offload/promote/k1'], check=True,
                             capture_output=True).stdout
    subprocess.run(['tar', '-x', '-C', str(root)], input=archive, check=True)
    manage = switch.k1_manager(root)
    env = root / 'dsv41.env'
    env.write_text(production_env(False, tmp_path))
    (root / 'start.sh').write_text('#!/bin/sh\n')
    for name in manage.FILES:
        shutil.copy2(root / name, state / name)
    digest = {name: manage.sha(root / name) for name in manage.FILES}
    (state / 'files.json').write_text(json.dumps({'profile': '327680', 'original': digest, 'promoted': digest}, indent=2) + '\n')
    return root


@pytest.fixture
def verified(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, str | None]]:
    """The installed manager with only verify() narrowed to its promoted-hash part (S18/manifest are production-only)."""
    calls: list[tuple[Path, str | None]] = []
    real = switch.k1_manager

    def loaded(root: Path) -> types.ModuleType:
        manage = real(root)

        def verify(root: Path, selected: str | None = None) -> None:
            _, _, promoted = manage.saved(root)
            for name in manage.FILES:
                if manage.sha(root / name) != promoted[name]:
                    raise ValueError(f'promoted {name} changed since apply; refusing start')
            calls.append((root, selected))
        setattr(manage, 'verify', verify)
        return manage
    monkeypatch.setattr(switch, 'k1_manager', loaded)
    return calls


@pytest.mark.parametrize('base', [PRODUCTION, PROMO1_STAGED])
def test_installed_manager_loads_in_a_clean_production_process(tmp_path: Path, base: str) -> None:
    """596ba1c's manage.py imports `profiles` as a script sibling; PROMO1's imports the package. On sparkhead the switch
    runs in a fresh interpreter from the production checkout: reproduce exactly that, isolated (-I) from this tree."""
    root = tmp_path / 'prod'
    root.mkdir()
    archive = subprocess.run(['git', '-C', str(ROOT), 'archive', base, 'offload/promote/k1', 'offload/kvpack'], check=True,
                             capture_output=True).stdout
    subprocess.run(['tar', '-x', '-C', str(root)], input=archive, check=True)
    target = root / package.SWITCH
    target.parent.mkdir(parents=True)
    shutil.copy2(ROOT / package.SWITCH, target)
    code = ('import sys; from pathlib import Path; sys.path.insert(0, sys.argv[2]); import pp4_promote_switch as s; '
            'm = s.k1_manager(Path(sys.argv[1])); '
            "print(m.MANIFEST_SHA, m.FILES, all(callable(getattr(m, n)) for n in ('hashes', 'saved', 'sha', 'read_object', 'verify')))")
    out = subprocess.run([sys.executable, '-I', '-c', code, str(root), str(target.parent)], cwd=root, check=True,
                         capture_output=True, text=True).stdout.split()
    assert out == ['559c5867ba42aa7daa7b4a7c8231d15ee1d35be40ae5b715b013f973193c58de', "('dsv41.env',", "'start.sh')", 'True']


def test_apply_rebases_only_the_env_hash_and_restore_is_byte_exact(tmp_path: Path, verified: list[tuple[Path, str | None]]) -> None:
    root = installed_tree(tmp_path)
    before = {name: (root / name).read_bytes() for name in switch.STATE_FILES}
    switch.apply(root)
    after_env = (root / 'dsv41.env').read_bytes()
    assert after_env.decode() == switch.edit_env(before['dsv41.env'].decode())
    record = json.loads((root / 'logs/k1-promotion/files.json').read_text())
    old = json.loads(before['logs/k1-promotion/files.json'])
    assert record['promoted'] == {**old['promoted'], 'dsv41.env': hashlib.sha256(after_env).hexdigest()}
    assert record['original'] == old['original'] and record['profile'] == '327680'
    ledger = json.loads((root / switch.BACKUP / 'transaction.json').read_text())
    assert ledger['pp4'] == switch.PP4_COMMIT and ledger['profile'] == '327680'
    with pytest.raises(FileExistsError):
        switch.apply(root)                                  # exclusive snapshot: never a second apply
    switch.restore(root)
    assert {name: (root / name).read_bytes() for name in switch.STATE_FILES} == before
    assert [s for _, s in verified] == ['327680'] * 4   # apply, apply-after, refused re-apply, restore


@pytest.mark.parametrize('damage', ['later-edit', 'backup'])
def test_restore_refuses_foreign_edits(tmp_path: Path, verified: list[tuple[Path, str | None]], damage: str) -> None:
    root = installed_tree(tmp_path)
    switch.apply(root)
    target = root / 'dsv41.env' if damage == 'later-edit' else root / switch.BACKUP / 'dsv41.env'
    target.write_text(target.read_text() + '# foreign\n')
    with pytest.raises(ValueError):
        switch.restore(root)


def test_restore_completes_a_partial_switch(tmp_path: Path, verified: list[tuple[Path, str | None]]) -> None:
    root = installed_tree(tmp_path)
    before = {name: (root / name).read_bytes() for name in switch.STATE_FILES}
    switch.apply(root)
    (root / 'logs/k1-promotion/files.json').write_bytes(before['logs/k1-promotion/files.json'])  # lost second write
    switch.restore(root)
    assert {name: (root / name).read_bytes() for name in switch.STATE_FILES} == before


FIXED5_ARGV = f"#!/bin/bash\nexec vllm serve /srv/dsv41/model --speculative-config '{switch.SPEC}' \\\n --port 8000\n"


def probe_text(env: dict[str, str | None], runner: str | None = None, remote_drift: bool = False,
               argv: str = FIXED5_ARGV, ids: tuple[str, str] = ('d' * 64, 'd' * 64)) -> str:
    lines = [f'PP4-ID {ids[0]}', f'{runner or switch.VERIFIER_SHA256[switch.RUNNER]}  {switch.DIST}/{switch.RUNNER}',
             f'{switch.VERIFIER_SHA256[switch.UTILS]}  {switch.DIST}/{switch.UTILS}',
             f'{switch.VERIFIER_SHA256[switch.TRACE]}  {switch.DIST}/{switch.TRACE}',
             f'{"a" * 64}  ./wire.py', f'{"b" * 64}  ./sampled_wire.py',
             f'{("c" if remote_drift else "a") * 64}  ./remote_dspark/wire.py', f'{"b" * 64}  ./remote_dspark/sampled_wire.py']
    lines += [f'{digest}  ./verifier/{name}' for name, digest in switch.VERIFIER_SHA256.items()]
    return '\n'.join(lines) + '\nPP4-ENV\n' + json.dumps(env) + '\nPP4-ARGV\n' + argv + '\n' + f'PP4-ID {ids[1]}\n'


def test_running_check_accepts_only_the_registered_runtime() -> None:
    good: dict[str, str | None] = {**dict(switch.PP4_ENV), 'VLLM_PP4_ROUND_DIR': None, 'VLLM_PP4_ADMISSION_RUN': None}
    assert switch.check_running('spark', probe_text(good)) == 'd' * 64
    adaptive = FIXED5_ARGV.replace(switch.SPEC, '{"method":"dspark","num_speculative_tokens":5,'
                                                '"num_speculative_tokens_per_batch_size":[[1,2,5],[3,4,4]]}')
    for bad in (probe_text({**good, 'VLLM_DSV41_ADAPTIVE_K': '1'}), probe_text({**good, 'VLLM_PP4_ROUND_DIR': '/pp4-round'}),
                probe_text(good, runner='0' * 64), probe_text(good, remote_drift=True), probe_text(good).replace('PP4-ENV\n', ''),
                probe_text(good, argv=adaptive), probe_text(good, argv='#!/bin/bash\n'), probe_text(good, ids=('d' * 64, 'e' * 64)),
                probe_text(good).replace('PP4-ARGV\n', ''), probe_text(good).split('\n', 1)[1]):
        with pytest.raises(ValueError):
            switch.check_running('spark', bad)


def test_dropin_keeps_the_unit_boundary_and_names_qualified_bytes() -> None:
    text = switch.render_dropin()
    exec_lines = [x for x in text.splitlines() if x.startswith('ExecStart=')]
    assert exec_lines[0] == 'ExecStart=' and len(exec_lines) == 2
    start = exec_lines[1]
    assert 'gpu_locked.sh %h/dsv41-offload/locks/gpu-c4e7a371.lock %h/dsv41-offload/locks/gpu-7744b44b.lock -- ' in start
    assert f'{switch.PYTHON} -u -m worker.server --tp 2 --root /opt/dsv41/x86host-data/dspark --ofl 10.202.28.1:18620 ' \
           '--max-model-len 1048576 --status-file %h/dsv41-offload/prod/run/dspark-worker.status.json' in start
    assert 'Environment=DSPARK_SAMPLED_TOP32=1' in text and f'Environment=DSPARK_SOURCE_COMMIT={switch.PP4_COMMIT}' in text
    assert 'CUDA_VISIBLE_DEVICES' not in text and 'MemoryMax' not in text and 'Restart' not in text
    attest = switch.render_attest()
    for name, digest in switch.SO_SHA256.items():
        assert f'test "$(sha256sum {name} | cut -d" " -f1)" = {digest}' in attest
    assert subprocess.run(['sh', '-n'], input=attest, text=True).returncode == 0


def verify_with(tree: Path, root: Path, runner_sha: str | None, env_file: Path) -> int:
    env = {**os.environ, 'KVPACK_TEST_RUNNING_ROOT': str(root), 'KVPACK_TEST_RUNNING_ENV': str(env_file)}
    if runner_sha is not None:
        env['KVPACK_PP4_RUNNER_SHA256'] = runner_sha
    return subprocess.run(['bash', 'offload/kvpack/stage_overlay.sh', 'verify-running', 'local'], cwd=tree, env=env,
                          capture_output=True).returncode


def activate(tree: Path, runner_sha: str) -> None:
    """An open, unrestored PP4 transaction whose switched bytes are installed in `tree` (what spark-apply leaves)."""
    env = tree / 'dsv41.env'
    env.write_text(f"PROD_PP4={switch.PP4_ID}\nPROD_SPEC_OVERRIDE='{switch.SPEC}'\nPROD_PP4_RUNNER_SHA256={runner_sha}\n")
    files = tree / 'logs/k1-promotion/files.json'
    files.parent.mkdir(parents=True, exist_ok=True)
    files.write_text('{}\n')
    ledger = tree / switch.BACKUP / 'transaction.json'
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({'pp4': switch.PP4_COMMIT, 'promoted': {
        name: hashlib.sha256((tree / name).read_bytes()).hexdigest() for name in switch.STATE_FILES}}))


@pytest.mark.parametrize('base', [PRODUCTION, PROMO1_STAGED])
def test_package_hooks_apply_once_and_k1_running_check_obeys_them(tmp_path: Path, base: str) -> None:
    tree = tmp_path / 'tree'
    tree.mkdir()
    archive = subprocess.run(['git', '-C', str(ROOT), 'archive', base, 'serve.sh', 'dsv41.env', 'offload/promote/k1',
                              'offload/kvpack/stage_overlay.sh', 'offload/kvpack/MANIFEST.json', 'offload/kvpack/overlay'],
                             check=True, capture_output=True).stdout
    subprocess.run(['tar', '-x', '-C', str(tree)], input=archive, check=True)
    with pytest.raises(ValueError, match='clock fold'):
        package.plan(tree)                                  # the 420be24 cherry-pick comes before the hooks
    fold(tree / 'dsv41.env')
    assert package.apply(tree) == [*package.HOOKS_PATHS, package.SWITCH]
    for path in package.HOOKS_PATHS:
        assert subprocess.run(['bash', '-n', str(tree / path)]).returncode == 0
    with pytest.raises(ValueError):
        package.plan(tree)
    start_checked = (tree / 'offload/promote/k1/start_checked.sh').read_text()
    assert start_checked.index('logs/pp4-switch/transaction.json') < start_checked.index('manage.py restore')
    running = tmp_path / 'running'
    shutil.copytree(tree / 'offload/kvpack/overlay', running)
    (running / switch.RUNNER).write_text('pp4 runner\n')
    pp4 = hashlib.sha256(b'pp4 runner\n').hexdigest()
    env_file = tmp_path / 'env'
    env_file.write_text('')
    assert verify_with(tree, running, pp4, env_file) != 0          # S91 #4: a declared replacement needs an active switch
    activate(tree, pp4)

    def verify(root: Path, runner_sha: str | None) -> int:
        return verify_with(tree, root, runner_sha, env_file)
    assert verify(running, pp4) == 0
    assert verify(running, None) != 0 and verify(running, '0' * 64) != 0
    assert verify(tree / 'offload/kvpack/overlay', None) == 0 and verify(tree / 'offload/kvpack/overlay', pp4) != 0
    (tree / switch.BACKUP / switch.RESTORED).write_text('{}\n')       # restored: evidence only, never active again
    assert verify(running, pp4) != 0 and verify(tree / 'offload/kvpack/overlay', None) == 0
    serve = (tree / 'serve.sh').read_text()
    assert serve.index(package.SPEC_LINE) < serve.index('PROD_SPEC_OVERRIDE')


def test_fold_clock_rebases_only_the_clock_line_and_is_idempotent(tmp_path: Path, verified: list[tuple[Path, str | None]],
                                                                  capsys: pytest.CaptureFixture[str]) -> None:
    root = installed_tree(tmp_path)                   # K1 record names the pre-fold 2150 env
    with pytest.raises(ValueError, match='cherry-pick 420be24'):
        switch.fold_clock(root)
    fold(root / 'dsv41.env')                          # the pushed staging commit
    record = root / 'logs/k1-promotion/files.json'
    before = record.read_bytes()
    switch.fold_clock(root)
    promoted = json.loads(record.read_text())['promoted']
    assert promoted['dsv41.env'] == hashlib.sha256((root / 'dsv41.env').read_bytes()).hexdigest()
    assert (root / switch.FOLD / 'files.json').read_bytes() == before
    switch.fold_clock(root)                           # idempotent
    assert 'CLOCK FOLD ALREADY RECORDED 2300' in capsys.readouterr().out
    switch.apply(root)                                # the PP4 snapshot is the folded production
    assert switch.CLOCK_FOLDED in (root / switch.BACKUP / 'dsv41.env').read_text()
    switch.restore(root)
    assert switch.CLOCK_FOLDED in (root / 'dsv41.env').read_text()


def test_fold_clock_refuses_any_other_env_difference(tmp_path: Path, verified: list[tuple[Path, str | None]]) -> None:
    root = installed_tree(tmp_path)
    fold(root / 'dsv41.env')
    (root / 'dsv41.env').write_text((root / 'dsv41.env').read_text() + 'PROD_KV_BYTES=1\n')
    with pytest.raises(ValueError, match='beyond the clock line'):
        switch.fold_clock(root)
    assert not (root / switch.FOLD).exists()
