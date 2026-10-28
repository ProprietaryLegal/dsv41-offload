"""UP2-PROMOTE package, switch and hooks: real production env/tree bytes, byte-exact transaction, composition refusals."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types
from typing import cast

import pytest

from offload.promote.up2 import hooks, package, promote_window, switch
from offload.up3 import build as b

ROOT = Path(__file__).resolve().parents[4]
PRODUCTION = '596ba1cf4e3b11be29b41a57e7f69a5d8efa4799'
PROMO1_STAGED = 'af5eaf6'          # coordinator's PROMO1 production staging commit (596ba1c + PROMO1 package)
PROMO1 = '4740ce9f2a3ca610dfc8621e01d49e0772fb987f'
PP4_PACKAGE = '95e9d67'            # PP4-PROMOTE item 2: its switch and production hooks


def show(commit: str, path: str) -> str:
    return subprocess.run(['git', '-C', str(ROOT), 'show', f'{commit}:{path}'], check=True,
                          capture_output=True, text=True).stdout


def pp4_module(name: str) -> types.ModuleType:
    """PP4-PROMOTE's own module from its commit (never copied into this branch)."""
    spec = importlib.util.spec_from_loader('pp4_' + name, loader=None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(ROOT / f'offload/launch/windows/{name}.py')   # its ROOT; only pure functions are used
    sys.modules[module.__name__] = module      # dataclasses resolve annotations through sys.modules at class creation
    try:
        exec(compile(show(PP4_PACKAGE, f'offload/launch/windows/{name}.py'), name, 'exec'), module.__dict__)
    finally:
        del sys.modules[module.__name__]
    return module


def production_env(base: str, tmp_path: Path) -> str:
    text = show(PRODUCTION, 'dsv41.env')
    if base in ('promo1-low', 'pp4-low'):
        env = tmp_path / 'low.env'
        env.write_text(text)
        patch = tmp_path / 'low.patch'
        patch.write_text(show(PROMO1, 'offload/promote/k1/dsv41.env.327680-low.patch'))
        subprocess.run(['patch', '-s', str(env), str(patch)], check=True)
        text = env.read_text()
    if base.startswith('pp4'):
        text = str(pp4_module('pp4_promote_switch').edit_env(text))
    return text


PROFILE = {'ops1': '327680', 'promo1-low': '327680-low', 'pp4': '327680', 'pp4-low': '327680-low'}


def sourced(text: str, tmp_path: Path, *names: str) -> list[str]:
    env = tmp_path / 'sourced.env'
    env.write_text(text)
    script = 'set -a; . "$1" >/dev/null; ' + '; '.join(f'printf "%s\\n" "${name}"' for name in names)
    return subprocess.run(['bash', '-c', script, '_', str(env)], check=True, capture_output=True, text=True).stdout.splitlines()


def test_packages_rebuild_exactly_and_are_content_addressed() -> None:
    packages = package.verify()
    qualified, composed = packages['qualified'], packages['pp4-composed']
    assert qualified['arm'] == 'up3-replay' and composed['arm'] is None
    assert qualified['bases'] == ['ops1', 'promo1-low'] and composed['bases'] == ['pp4']
    for manifest in (qualified, composed):
        lines = ''.join(f'{v}  {k}\n' for k, v in sorted(manifest['files'].items()))
        assert hashlib.sha256(lines.encode()).hexdigest()[:12] == manifest['id']
        assert switch.package(manifest['name'], package.HERE) == json.loads(json.dumps(manifest))
    # The qualified package IS the measured UP3 replay arm, renamed to the staged layout.
    replay = b.verify('replay')['files']
    assert qualified['files'] == {('remote_dspark/' + k.removeprefix(b.REMOTE) if k.startswith(b.REMOTE) else k): v
                                  for k, v in replay.items()}
    # PP4 composition: same four vLLM files; the shared remote files carry both changes (differ from each parent).
    pp4 = {Path(p).name for p in subprocess.check_output(['git', 'ls-tree', '--name-only', package.PP4_COMMIT,
           'offload/dspark/spark/remote_dspark/'], cwd=ROOT, text=True).splitlines() if p.endswith('.py')}
    assert set(composed['files']) == {*package.VLLM_FILES, *('remote_dspark/' + n for n in pp4)}
    assert all(composed['files'][n] == qualified['files'][n] for n in package.VLLM_FILES)
    for name in package.SHARED_WITH_PP4:
        key = 'remote_dspark/' + name
        parent = hashlib.sha256(subprocess.check_output(['git', 'show',
            f'{package.PP4_COMMIT}:offload/dspark/spark/remote_dspark/{name}'], cwd=ROOT)).hexdigest()
        if name == 'suffix_provenance.py':
            assert composed['files'][key] == qualified['files'][key] == parent
        else:
            assert composed['files'][key] not in (parent, qualified['files'][key]), name


def test_pp4_suffix_port_changes_exactly_the_reviewed_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    def changed(text: str) -> list[str]:
        return sorted(line for line in text.splitlines() if line[:1] in ('+', '-') and line[:3] not in ('+++', '---'))
    original = (ROOT / 'offload/upstream_adopt/patches/0075-suffix.patch').read_text()
    port = (package.HERE / package.PP4_SUFFIX_PORT).read_text()
    assert changed(port) == changed(original)
    # The unported patch conflicts with PP4's round.py: that is why the port exists.
    monkeypatch.setattr(package, 'PP4_SUFFIX_PORT', '../../upstream_adopt/patches/0075-suffix.patch')
    with pytest.raises(ValueError, match='not exact'):
        package.pp4_files()


def test_composition_table_matches_the_production_layers() -> None:
    k1 = json.loads(b.production_file('offload/kvpack/MANIFEST.json'))
    assert set(package.COMPOSITION['k1']) == {row['path'] for row in k1['files']}
    ours = set(package.COMPOSITION['0075'])
    assert not ours & set(package.COMPOSITION['k1']) and not package.COMPOSITION['promo1-low']
    assert ours & set(package.COMPOSITION['pp4']) == {'vllm/v1/worker/gpu/spec_decode/remote_dspark/'}
    assert package.VLLM_FILES == switch.VLLM_FILES


@pytest.mark.parametrize('base', ['ops1', 'promo1-low', 'pp4', 'pp4-low'])
def test_base_detection_on_real_production_envs(tmp_path: Path, base: str) -> None:
    text = production_env(base, tmp_path)
    assert switch.base_of(text, PROFILE[base]) == ('pp4' if base.startswith('pp4') else base)


@pytest.mark.parametrize('base', ['ops1', 'promo1-low'])
def test_edit_env_registers_exactly_the_qualified_arm(tmp_path: Path, base: str) -> None:
    before = production_env(base, tmp_path)
    manifest = switch.package('qualified', package.HERE)
    after = switch.edit_env(before, base, manifest)
    old, new = before.splitlines(), after.splitlines()
    changed = [i for i, (a, c) in enumerate(zip(old, new)) if a != c]
    assert [old[i].split('=', 1)[0] for i in changed] == ['OFFLOAD_DOCKER_ENV', 'OFFLOAD_DOCKER_ARGS']
    assert new[len(old):] == [f"{switch.MARKER}{manifest['id']}: 0075 replay skip (qualified); rollback restores this "
                              'file byte-exact (logs/up2-switch).', f"PROD_UP2={manifest['id']}"]
    docker, args, profile, kv = sourced(after, tmp_path, 'OFFLOAD_DOCKER_ENV', 'OFFLOAD_DOCKER_ARGS', 'PROD_K1_PROFILE',
                                        'PROD_KV_BYTES')
    old_docker, old_args, old_profile, old_kv = sourced(before, tmp_path, 'OFFLOAD_DOCKER_ENV', 'OFFLOAD_DOCKER_ARGS',
                                                        'PROD_K1_PROFILE', 'PROD_KV_BYTES')
    assert docker.split() == [*old_docker.split(), 'VLLM_DSV41_REPLAY_SKIP=1'] and (profile, kv) == (old_profile, old_kv)
    mounts, old_mounts = args.split()[1::2], old_args.split()[1::2]
    stage = switch.STAGE + str(manifest['id'])
    target = {m.split(':')[1]: m.split(':')[0] for m in mounts}
    assert len(mounts) == len(old_mounts) + 4
    assert target[f'{switch.DIST}/{switch.REMOTE}'] == f'{stage}/remote_dspark'
    for name in switch.VLLM_FILES:
        assert target[f'{switch.DIST}/{name}'] == f'{stage}/{name}'
    assert {m for m in old_mounts if not m.split(':')[1].endswith('/remote_dspark')} <= set(mounts)
    # The same flag and container targets the qualified UP3 replay arm used (window.env; UP3 pins ops1's KV, so ops1 only).
    if base == 'ops1':
        (tmp_path / 'ops1.env').write_text(before)
        arm = subprocess.check_output(['bash', '-eu', '-c', 'UP3_ARM=replay UP3_PROD_ENV=$1; source offload/up3/window.env; '
            'printf "%s\\n" "$EXP_DOCKER_ENV" "$OVERLAYS"', 'up3', str(tmp_path / 'ops1.env')], cwd=ROOT, text=True)
        flags, overlays = arm.splitlines()
        assert flags.split()[-1] == 'VLLM_DSV41_REPLAY_SKIP=1'
        assert {p.split('=', 1)[1] for p in overlays.split()} == {f'{switch.DIST}/{n}' for n in (*switch.VLLM_FILES, switch.REMOTE)}


@pytest.mark.parametrize('change', ['twice', 'no-remote', 'pp4-remote-on-ops1', 'file-mounted', 'flag-set', 'no-newline'])
def test_edit_env_refuses_anything_it_cannot_place(tmp_path: Path, change: str) -> None:
    text = production_env('ops1', tmp_path)
    manifest = switch.package('qualified', package.HERE)
    if change == 'twice':
        text = switch.edit_env(text, 'ops1', manifest)
        with pytest.raises(ValueError, match='already present'):
            switch.base_of(text, '327680')
    elif change == 'no-remote':
        text = text.replace('spec_decode/remote_dspark:ro', 'spec_decode/other:ro')
    elif change == 'pp4-remote-on-ops1':
        text = text.replace(switch.PRODUCTION_REMOTE, switch.PP4_REMOTE)
    elif change == 'file-mounted':
        text = text.replace('spec_decode/__init__.py:ro', f'spec_decode/__init__.py:ro -v /x:{switch.DIST}/{switch.VLLM_FILES[0]}:ro')
    elif change == 'flag-set':
        text = text.replace('VLLM_DSV41_KV_PAIR=3', 'VLLM_DSV41_KV_PAIR=3 VLLM_DSV41_REPLAY_SKIP=0')
    else:
        text = text.rstrip('\n')
    with pytest.raises(ValueError):
        switch.edit_env(text, 'ops1', manifest)


def test_pp4_base_composes_but_is_refused_until_measured(tmp_path: Path) -> None:
    text = production_env('pp4', tmp_path)
    composed = switch.package('pp4-composed', package.HERE)
    after = switch.edit_env(text, 'pp4', composed)       # the bytes compose mechanically ...
    args, = sourced(after, tmp_path, 'OFFLOAD_DOCKER_ARGS')
    target = {m.split(':')[1]: m.split(':')[0] for m in args.split()[1::2]}
    assert target[f'{switch.DIST}/{switch.REMOTE}'] == f"{switch.STAGE}{composed['id']}/remote_dspark"
    with pytest.raises(ValueError, match='unknown combination'):
        switch.edit_env(text, 'ops1', composed)            # PP4's remote source is not production's
    with pytest.raises(ValueError, match='unknown production'):
        switch.base_of(show(PRODUCTION, 'dsv41.env'), '524288')


def installed_tree(tmp_path: Path, base: str, commit: str = PRODUCTION) -> Path:
    """A production checkout's K1 package (its real manage.py/profiles.py), the UP2 tool and an installed K1 record."""
    root = tmp_path / 'prod'
    state = root / 'logs/k1-promotion'
    state.mkdir(parents=True)
    archive = subprocess.run(['git', '-C', str(ROOT), 'archive', commit, 'offload/promote/k1'], check=True,
                             capture_output=True).stdout
    subprocess.run(['tar', '-x', '-C', str(root)], input=archive, check=True)
    for path in (hooks.SWITCH, *hooks.PACKAGES):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, root / path)
    (root / 'dsv41.env').write_text(production_env(base, tmp_path))
    (root / 'start.sh').write_text('#!/bin/sh\n')
    # Both managers record exactly these two files (never loaded in-process here: PROMO1's imports its own package).
    for name in ('dsv41.env', 'start.sh'):
        shutil.copy2(root / name, state / name)
    digest = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ('dsv41.env', 'start.sh')}
    (state / 'files.json').write_text(json.dumps({'profile': PROFILE[base], 'original': digest, 'promoted': digest},
                                                 indent=2) + '\n')
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


@pytest.mark.parametrize('base', ['ops1', 'promo1-low'])
def test_apply_rebases_only_the_env_hash_and_restore_is_byte_exact(tmp_path: Path, verified: list[tuple[Path, str | None]],
                                                                   base: str) -> None:
    root = installed_tree(tmp_path, base)
    before = {name: (root / name).read_bytes() for name in switch.STATE_FILES}
    switch.apply(root)
    after_env = (root / 'dsv41.env').read_bytes()
    manifest = switch.package('qualified', package.HERE)
    assert after_env.decode() == switch.edit_env(before['dsv41.env'].decode(), base, manifest)
    record = json.loads((root / 'logs/k1-promotion/files.json').read_text())
    old = json.loads(before['logs/k1-promotion/files.json'])
    assert record['promoted'] == {**old['promoted'], 'dsv41.env': hashlib.sha256(after_env).hexdigest()}
    assert record['original'] == old['original'] and record['profile'] == PROFILE[base]
    ledger = json.loads((root / switch.BACKUP / 'transaction.json').read_text())
    assert (ledger['package'], ledger['base'], ledger['name']) == (manifest['id'], base, 'qualified')
    with pytest.raises(ValueError, match='already present'):
        switch.apply(root)                                  # an applied switch never stacks on itself
    switch.restore(root)
    assert {name: (root / name).read_bytes() for name in switch.STATE_FILES} == before
    assert [s for _, s in verified] == [PROFILE[base]] * 4  # apply, apply-after, refused re-apply, restore


def test_apply_on_pp4_production_refuses_before_any_write(tmp_path: Path, verified: list[tuple[Path, str | None]]) -> None:
    root = installed_tree(tmp_path, 'pp4')
    before = {name: (root / name).read_bytes() for name in switch.STATE_FILES}
    with pytest.raises(ValueError, match='never measured'):
        switch.apply(root)
    assert not (root / switch.BACKUP).exists()
    assert {name: (root / name).read_bytes() for name in switch.STATE_FILES} == before


@pytest.mark.parametrize('damage', ['later-edit', 'backup'])
def test_restore_refuses_foreign_edits(tmp_path: Path, verified: list[tuple[Path, str | None]], damage: str) -> None:
    root = installed_tree(tmp_path, 'ops1')
    switch.apply(root)
    target = root / 'dsv41.env' if damage == 'later-edit' else root / switch.BACKUP / 'dsv41.env'
    target.write_text(target.read_text() + '# foreign\n')
    with pytest.raises(ValueError):
        switch.restore(root)


def test_restore_completes_a_partial_switch(tmp_path: Path, verified: list[tuple[Path, str | None]]) -> None:
    root = installed_tree(tmp_path, 'ops1')
    before = {name: (root / name).read_bytes() for name in switch.STATE_FILES}
    switch.apply(root)
    (root / 'logs/k1-promotion/files.json').write_bytes(before['logs/k1-promotion/files.json'])  # lost second write
    switch.restore(root)
    assert {name: (root / name).read_bytes() for name in switch.STATE_FILES} == before


@pytest.mark.parametrize('commit', [PRODUCTION, PROMO1_STAGED])
def test_switch_loads_the_installed_manager_in_a_clean_production_process(tmp_path: Path, commit: str) -> None:
    """On sparkhead the switch runs in a fresh system interpreter from the production checkout, stdlib only (-I)."""
    root = installed_tree(tmp_path, 'ops1', commit)
    code = ('import sys; from pathlib import Path; sys.path.insert(0, sys.argv[2]); import switch as s; '
            'm = s.k1_manager(Path(sys.argv[1])); p = s.package("qualified", Path(sys.argv[2])); '
            "print(m.FILES, p['id'], s.base_of(Path(sys.argv[1], 'dsv41.env').read_text(), '327680'))")
    out = subprocess.run([sys.executable, '-I', '-c', code, str(root), str(root / 'offload/promote/up2')], cwd=root,
                         check=True, capture_output=True, text=True).stdout.split()
    assert out == ["('dsv41.env',", "'start.sh')", str(switch.package('qualified', package.HERE)['id']), 'ops1']


def probe_text(manifest: dict[str, object], *, flag: str | None = '1', drift: str = '') -> str:
    files = dict(cast(dict[str, str], manifest['files']))
    running = {**files, **({drift: '0' * 64} if drift else {})}
    lines = [f'{running[n]}  {switch.DIST}/{n}' for n in switch.VLLM_FILES]
    lines += [f'{running[k]}  ./{k.removeprefix("remote_dspark/")}' for k in sorted(running) if k.startswith('remote_dspark/')]
    staged = [f'{v}  {k}' for k, v in sorted(files.items())]
    return '\n'.join(lines) + '\nUP2-STAGED\n' + '\n'.join(staged) + '\nUP2-ENV\n' + \
        json.dumps({switch.SWITCH_ENV[0]: flag}) + '\n'


def test_running_check_accepts_only_the_staged_package_and_flag() -> None:
    manifest = switch.package('qualified', package.HERE)
    switch.check_running('spark', probe_text(manifest), manifest)
    for bad in (probe_text(manifest, flag=None), probe_text(manifest, flag='0'),
                probe_text(manifest, drift='remote_dspark/round.py'), probe_text(manifest, drift=switch.VLLM_FILES[0]),
                probe_text(manifest).replace('UP2-ENV\n', ''),
                probe_text(manifest).replace('  ./round.py\n', '  ./round.py\n' + '1' * 64 + '  ./extra.py\n')):
        with pytest.raises(ValueError):
            switch.check_running('spark', bad, manifest)
    assert 'sha256sum' in switch.running_probe(str(manifest['id'])) and str(manifest['id']) in switch.running_probe(str(manifest['id']))


@pytest.mark.parametrize('kind', ['ops1', 'promo1', 'pp4'])
def test_hooks_apply_once_on_every_supported_tree(tmp_path: Path, kind: str) -> None:
    tree = tmp_path / 'tree'
    tree.mkdir()
    commit = PROMO1_STAGED if kind == 'promo1' else PRODUCTION
    archive = subprocess.run(['git', '-C', str(ROOT), 'archive', commit, 'serve.sh', 'offload/promote/k1',
                              'offload/kvpack/stage_overlay.sh'], check=True, capture_output=True).stdout
    subprocess.run(['tar', '-x', '-C', str(tree)], input=archive, check=True)
    if kind == 'pp4':
        pp4 = pp4_module('pp4_promote_package')
        for path, text in pp4.plan(tree).items():
            (tree / path).write_text(text)
    assert hooks.apply(tree) == [*hooks.HOOKS_PATHS, hooks.SWITCH, *hooks.PACKAGES]
    for path in hooks.HOOKS_PATHS:
        assert subprocess.run(['bash', '-n', str(tree / path)]).returncode == 0
    with pytest.raises(ValueError):
        hooks.plan(tree)
    start_checked = (tree / 'offload/promote/k1/start_checked.sh').read_text()
    assert start_checked.index('logs/up2-switch/transaction.json') < start_checked.index('manage.py restore')
    verify = (tree / 'offload/promote/k1/verify.sh').read_text()
    assert verify.index('stage_overlay.sh verify-running') < verify.index('offload/promote/up2/switch.py verify-running')
    for path in (hooks.SWITCH, *hooks.PACKAGES):
        assert (tree / path).read_bytes() == (ROOT / path).read_bytes()


def test_scripts_parse_and_pin_the_package() -> None:
    manifest = switch.package('qualified', package.HERE)
    for script in ('stage.sh', 'restore-up2.sh'):
        assert subprocess.run(['bash', '-n', str(package.HERE / script)]).returncode == 0
    restore = (package.HERE / 'restore-up2.sh').read_text()
    assert f"ID={manifest['id']}\n" in restore and 'greedy K1-post vs restored: ' in restore
    assert "grep -qx 'UP2 RUNNING VERIFIED'" in restore and "! grep -q '^PROD_PP4=' dsv41.env" in restore


CLOCK_FOLD = '420be244a524b34a904a0d7cca0ec954057af00b'
FOLD_CLOCK_REUSED = '8548a7c'      # PP4-PROMOTE's `fold-clock`, reused byte for byte (never re-implemented)


def test_clock_fold_reuses_pp4_fold_clock_then_the_0075_switch_stacks_on_it(
        tmp_path: Path, verified: list[tuple[Path, str | None]], monkeypatch: pytest.MonkeyPatch) -> None:
    """0075-first order: 420be24 -> hooks -> push sparkhead -> fold-clock -> W8 start -> (HOLD) spark-apply."""
    tool = ROOT / 'offload/launch/windows/pp4_promote_switch.py'
    assert tool.read_text() == show(FOLD_CLOCK_REUSED, 'offload/launch/windows/pp4_promote_switch.py')
    root = installed_tree(tmp_path, 'ops1')                      # K1 record names the pre-fold 596ba1c env
    folded = show(CLOCK_FOLD, 'dsv41.env')
    assert [a for a, c in zip(show(PRODUCTION, 'dsv41.env').splitlines(), folded.splitlines()) if a != c] == [
        line for line in show(PRODUCTION, 'dsv41.env').splitlines() if line.startswith('GPU_CLOCK_CAP=')]
    check = ['bash', '-c', 'cd "$1" && ' + promote_window.CLOCK_FOLD_CHECK, '_', str(root)]
    assert subprocess.run(check).returncode != 0                 # unfolded production: the runner preflight refuses
    (root / 'dsv41.env').write_text(folded)                      # the pushed staging commit (fold + hooks)
    with pytest.raises(ValueError, match='promoted dsv41.env changed'):
        switch.apply(root)                                       # before fold-clock the K1 guard refuses every start
    spec = importlib.util.spec_from_file_location('pp4_fold_clock_tool', tool)
    assert spec is not None and spec.loader is not None
    pp4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp4)
    monkeypatch.setattr(pp4, 'k1_manager', switch.k1_manager)   # the same narrowed installed manager (fixture)
    pp4.fold_clock(root)
    pp4.fold_clock(root)                                         # idempotent: ALREADY RECORDED
    ledger = json.loads((root / 'logs/clock-fold/transaction.json').read_text())
    assert ledger['fold'] == CLOCK_FOLD and ledger['env'] == hashlib.sha256(folded.encode()).hexdigest()
    assert subprocess.run(check).returncode == 0                 # the runner preflight's fold check passes
    switch.apply(root)                                           # the 0075 switch stacks on the folded production
    env = (root / 'dsv41.env').read_text()
    assert 'GPU_CLOCK_CAP=${GPU_CLOCK_CAP-2300}' in env and 'VLLM_DSV41_REPLAY_SKIP=1' in env
    switch.restore(root)                                         # rollback returns the FOLDED production, byte-exact
    assert (root / 'dsv41.env').read_text() == folded


def test_promo1_low_base_is_refused_because_it_can_never_pass_the_clock_fold_preflight(
        tmp_path: Path, verified: list[tuple[Path, str | None]], monkeypatch: pytest.MonkeyPatch) -> None:
    """S97 CX3 (as a regression): CLOCK_FOLD_CHECK needs the fold ledger, which PP4's fold-clock refuses to write on
    PROMO1's low env (its 2300 line is PROMO1's text, not 420be24's). So the runner offers ops1 only (FIX 4)."""
    assert promote_window.BASES == ('ops1',)
    root = installed_tree(tmp_path, 'promo1-low')
    check = ['bash', '-c', 'cd "$1" && ' + promote_window.CLOCK_FOLD_CHECK, '_', str(root)]
    assert subprocess.run(check).returncode != 0                       # the preflight refuses the PROMO1-low production
    spec = importlib.util.spec_from_file_location('pp4_fold_clock_promo1', ROOT / 'offload/launch/windows/pp4_promote_switch.py')
    assert spec is not None and spec.loader is not None
    pp4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp4)
    monkeypatch.setattr(pp4, 'k1_manager', switch.k1_manager)
    with pytest.raises(ValueError, match='lacks exactly one folded 2300 clock line'):
        pp4.fold_clock(root)                                           # and fold-clock cannot make it pass
    assert not (root / 'logs/clock-fold').exists()


def test_fold_step_verifies_the_staged_tool_bytes_before_it_writes_the_k1_record(tmp_path: Path) -> None:
    """S97 FIX 5: fold-clock writes production's K1 record, so README / UP2-REPIN run it only after sha256sum proves the
    staged copy is the reviewed tool (blob 9dcb9eb). A tampered staged copy never reaches the fold."""
    import shlex
    tool = ROOT / promote_window.FOLD_TOOL
    assert hashlib.sha256(tool.read_bytes()).hexdigest() == promote_window.FOLD_TOOL_SHA256
    for doc in ('README.md', 'UP2-REPIN.md'):
        text = (ROOT / 'offload/promote/up2' / doc).read_text()
        assert [line for line in text.splitlines() if 'fold-clock --root' in line] != []
        assert all(promote_window.FOLD_COMMAND in line for line in text.splitlines() if 'fold-clock --root' in line), doc
    remote = shlex.split(promote_window.FOLD_COMMAND)
    assert remote[:2] == ['ssh', 'sparkhead'] and len(remote) == 3
    # The remote script against a stand-in staged checkout: the fold itself is replaced by a marker echo.
    script = remote[2].replace('cd ~/dsv41-deploy-up2a', 'cd "$1"').replace(
        f'python3 {promote_window.FOLD_TOOL} fold-clock --root ~/dsv41-deploy', 'echo FOLD-RAN')
    assert script != remote[2] and 'FOLD-RAN' in script
    staged = tmp_path / 'staged' / promote_window.FOLD_TOOL
    staged.parent.mkdir(parents=True)
    staged.write_bytes(tool.read_bytes())
    ok = subprocess.run(['bash', '-c', script, '_', str(tmp_path / 'staged')], capture_output=True, text=True)
    assert ok.returncode == 0 and ok.stdout.strip() == 'FOLD-RAN'
    staged.write_bytes(tool.read_bytes() + b'# drift\n')
    bad = subprocess.run(['bash', '-c', script, '_', str(tmp_path / 'staged')], capture_output=True, text=True)
    assert bad.returncode != 0 and 'FOLD-RAN' not in bad.stdout
