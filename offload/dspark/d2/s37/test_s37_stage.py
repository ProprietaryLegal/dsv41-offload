"""S37 item 3: full_stage. Dry by default; execute refusals as strict as pilot_stage; leases; namespaces; real dry plan.

Never executes a job: tmux/git/admission are stubbed or run against a scratch git repo; no GPU, host or network access.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Callable

import pytest

from offload.dspark.d2.train import full_stage as S
from offload.dspark.d2.train import pilot_stage as P
from offload.dspark.d2.train.safety import GPU, SOLO_B

REPO = Path(__file__).resolve().parents[4]
HOME = Path('/home/x86host-user')  # rendered paths only; nothing is created there
PRODUCTION_PAIR = ('7744b44b', 'c4e7a371')


def runbook_args(commit: str) -> list[str]:
    """The exact full_args array of FULL-RUN.md, with $HOME/$PWD/commit substituted."""
    text = (REPO/'offload/dspark/d2/train/FULL-RUN.md').read_text()
    block = re.search(r'full_args=\((.*?)\)\n', text, re.S)
    assert block is not None
    body = block.group(1).replace('REVIEWED_FULL_40_HEX_COMMIT', commit)
    body = body.replace('"$HOME', '"'+str(HOME)).replace('"$PWD"', shlex.quote(str(REPO)))
    return shlex.split(body)


def head() -> str:
    from offload.dspark.d2.train.tests.snapshot_git import git
    return git(REPO, 'rev-parse', 'HEAD').stdout.decode().strip()


def render(*extra: str) -> dict[str, object]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'GIT_'))}
    env.update(HOME=str(HOME), PYTHONPATH=f'{REPO}:{REPO}/offload/dspark', CUDA_VISIBLE_DEVICES='',
               PYTHONDONTWRITEBYTECODE='1')
    done = subprocess.run([sys.executable, '-m', 'offload.dspark.d2.train.full_stage', *runbook_args(head()), *extra],
                          cwd=REPO, env=env, check=True, capture_output=True, text=True)
    return dict(json.loads(done.stdout))


@pytest.mark.parametrize('resume', [False, True])
def test_real_runbook_dry_plan_renders_and_every_body_parses(resume: bool, tmp_path: Path) -> None:
    plan = render(*(['--resume'] if resume else []))
    assert plan['dry'] is True and plan['order'] == list(S.PHASES)
    commands = plan['commands']
    assert isinstance(commands, dict)
    names = {phase: [name for name, _ in jobs] for phase, jobs in commands.items()}
    assert names == {'full-train': ['full-fast'], 'full-g2': ['g2-dev', 'g2-legal', 'g2-general'],
                     'finish': ['g2-finish']}
    (tmp_path/f'dry-plan{"-resume" if resume else ""}.json').write_text(json.dumps(plan, indent=2))
    for phase, jobs in commands.items():
        for name, body in jobs:
            subprocess.run(['bash', '-n'], input=body, text=True, check=True)
            uuids = set(re.findall(r'GPU-[0-9a-f-]{36}', body))
            assert uuids <= {GPU, SOLO_B}, (name, uuids)
            assert not any(p in body for p in PRODUCTION_PAIR)
            assert 'ssh ' not in body and 'curl' not in body and ':8000' not in body and ':8001' not in body
            assert re.search(r'^export .*CUDA_VISIBLE_DEVICES=$', body, re.M) and body.startswith('set -euo pipefail')
            # Every GPU command goes through the unchanged nonblocking-flock lease wrapper with ONE solo UUID.
            gpu_lines = [l for l in body.splitlines() if 'D2_PILOT_GPU=' in l]
            assert len(gpu_lines) == (0 if phase == 'finish' else 1)
            for line in gpu_lines:
                assert 'run_pilot_locked.sh' in line and len(uuids) == 1
            assert ('--resume' in body) == (resume and phase != 'finish')
    train = commands['full-train'][0][1]
    g2 = dict(commands['full-g2'])
    assert GPU in train and SOLO_B not in train
    assert GPU in g2['g2-dev'] and SOLO_B in g2['g2-legal'] and SOLO_B in g2['g2-general']


def lease_lines(plan: dict[str, object]) -> list[list[str]]:
    rows = []
    for jobs in dict(plan['commands']).values():  # type: ignore[call-overload]
        for _, body in jobs:
            for line in body.splitlines():
                if 'run_pilot_locked.sh' in line:
                    rows.append(shlex.split(line))
    return rows


def test_rendered_python_arguments_parse_with_the_real_clis(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from offload.dspark.d2.train import full_eval, full_train, loop, pilot, safety
    plan = render()
    parsed: list[str] = []
    for argv in lease_lines(plan):
        module = argv[argv.index('-m')+1]
        args = argv[argv.index('-m')+2:]
        monkeypatch.setattr(sys, 'argv', [module, *args])
        if module.endswith('full_train'):
            seen: list[tuple[object, ...]] = []
            monkeypatch.setattr(safety, 'assert_gpu_lease', lambda: None)
            monkeypatch.setattr(pilot, 'validate_live_g1', lambda *a: seen.append(a))

            class Stop(Exception):
                pass
            def load(*a: object, **k: object) -> None:
                raise Stop()
            monkeypatch.setattr(loop, 'require_full_pilot', lambda *a: 'hash')
            monkeypatch.setattr(loop, 'pilot_recipe', lambda *a: 'ace')
            original_read = Path.read_text
            monkeypatch.setattr(Path, 'read_text', lambda p, *a, **k: json.dumps({'plan': {'frozen_identity': 'base'}}) if p.name == 'receipt.json' else original_read(p, *a, **k))
            monkeypatch.setattr(loop, 'load', load)
            with pytest.raises(Stop):
                full_train.main()
            assert seen and seen[0][0] == Path('/opt/dsv41/x86host-data/d2/g1-d2pass2-r1/dev.json')
        else:
            got: list[argparse.Namespace] = []
            monkeypatch.setattr(full_eval, 'evaluate', got.append)
            full_eval.main()
            ns = got[0]
            assert ns.adapter.name == 'full-fast.pt' and ns.out.name == ns.split+'.json' and not ns.resume
            assert str(ns.pilot_source).endswith('/d2/pilot/decode-d2pass2-r1-p2-ace/repo')
        parsed.append(module)
    assert parsed == ['offload.dspark.d2.train.full_train'] + ['offload.dspark.d2.train.full_eval']*3


def test_default_namespaces_never_overlap_the_running_pilot_or_assembly() -> None:
    args = runbook_args('a'*40)
    def value(flag: str) -> Path:
        return Path(args[args.index(flag)+1])
    full = [value('--snapshot'), value('--output')]
    pilot = [value('--pilot-source').parent, value('--pilot-receipt').parent, P.ASSEMBLY]
    for a in full:
        for b in pilot:
            assert not a.is_relative_to(b) and not b.is_relative_to(a), (a, b)
    assert P.namespace('ace', P.FIRST_NAME) == 'decode-d2pass2-r1-p2-ace'
    assert value('--output').name == value('--pilot-receipt').parent.name+'-full'


def settings(tmp: Path) -> S.Full:
    base = P.Settings(source=tmp/'source', commit='a'*40, snapshot=tmp/'full-snap', output=tmp/'full-out',
                      assembly=P.ASSEMBLY, parity=tmp/'parity.json', g1=tmp/'dev.json', k2_dev=tmp/'dev.k2',
                      k2_legal=tmp/'legal.k2', ckpt=tmp/'native', python=tmp/'python')
    return S.Full(base, tmp/'pilot-snap/repo', tmp/'pilot-out/receipt.json', tmp/'general.k2')


@pytest.mark.parametrize('field,where', [
    ('output', 'pilot-out'), ('output', 'pilot-out/sub'), ('snapshot', 'pilot-snap'), ('snapshot', 'pilot-snap/x'),
    ('output', ''), ('snapshot', 'source/x'), ('output', 'native/x'), ('output', 'general.k2')])
def test_overlaps_refused(tmp_path: Path, field: str, where: str) -> None:
    s = settings(tmp_path)
    bad = replace(s, settings=replace(s.settings, **{field: tmp_path/where if where else tmp_path}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='overlap'):
        S.validate(bad)


def test_full_output_inside_assembly_refused(tmp_path: Path) -> None:
    s = settings(tmp_path)
    with pytest.raises(ValueError, match='overlap'):
        S.validate(replace(s, settings=replace(s.settings, output=P.ASSEMBLY/'full')))


@pytest.mark.parametrize('pair', [('ref_gpu', 'GPU-7744b44b-0000-0000-0000-000000000000'), ('fast_gpu', GPU)])
def test_settings_refuse_other_cards(tmp_path: Path, pair: tuple[str, str]) -> None:
    s = settings(tmp_path)
    with pytest.raises(ValueError, match='solo UUIDs'):
        S.validate(replace(s, settings=replace(s.settings, **{pair[0]: pair[1]})))  # type: ignore[arg-type]


def test_lock_script_refuses_any_other_uuid(tmp_path: Path) -> None:
    script = REPO/'offload/dspark/d2/train/run_pilot_locked.sh'
    home = tmp_path/'home'
    (home/'.config/ndm').mkdir(parents=True)
    (home/'.config/ndm/paths.sh').write_text('')
    for uuid in ('GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc', ''):
        env = {'PATH': os.environ['PATH'], 'HOME': str(home), 'D2_TRAIN_ROOT': str(tmp_path), 'D2_PILOT_GPU': uuid}
        done = subprocess.run(['bash', str(script), 'x', '-c', 'pass'], env=env, capture_output=True, text=True)
        assert done.returncode != 0
    assert 'flock -n 9' in script.read_text() and "[[ \"$task_limit\" == '225.00' ]]" in script.read_text()


# ---------------------------------------------------------------- execute refusals (nothing launched)


@pytest.fixture
def no_launch(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []
    real = subprocess.run

    def run(cmd: list[str], *a: object, **k: object) -> object:
        if cmd[0] == 'tmux':
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 1 if cmd[1] == 'has-session' else 0, b'', b'')
        return real(cmd, *a, **k)  # type: ignore[call-overload]
    monkeypatch.setattr(subprocess, 'run', run)
    return calls


def test_execute_refuses_before_any_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                         no_launch: list[list[str]]) -> None:
    s = settings(tmp_path)
    touched: list[str] = []
    monkeypatch.setattr(S, 'admission', lambda *a: touched.append('admission'))
    monkeypatch.delenv('D2_COORDINATOR', raising=False)
    monkeypatch.setattr('socket.gethostname', lambda: 'x86host')
    with pytest.raises(ValueError, match='coordinator'):
        S.execute(s, 'stage', None, False)
    monkeypatch.setenv('D2_COORDINATOR', 'someone-else')
    with pytest.raises(ValueError, match='coordinator'):
        S.execute(s, 'stage', None, False)
    monkeypatch.setenv('D2_COORDINATOR', P.COORDINATOR)
    monkeypatch.setattr('socket.gethostname', lambda: 'Controller')
    with pytest.raises(ValueError, match='coordinator'):
        S.execute(s, 'stage', None, False)
    monkeypatch.setattr('socket.gethostname', lambda: 'x86host')
    with pytest.raises(ValueError, match='-c2 assembly'):
        S.execute(replace(s, settings=replace(s.settings, assembly=tmp_path/'other')), 'stage', None, False)
    with pytest.raises(ValueError, match='attempt'):
        S.execute(s, 'stage', None, False, attempt=0)
    assert touched == [] and no_launch == [] and not list(tmp_path.iterdir())


def scratch_repo(tmp_path: Path) -> tuple[Path, str]:
    src = tmp_path/'source'
    src.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t', GIT_COMMITTER_NAME='t', GIT_COMMITTER_EMAIL='t@t')
    for cmd in (['init', '-q'], ['add', '.'], ['commit', '-q', '-m', 'x']):
        if cmd[0] == 'add':
            (src/'a.py').write_text('x = 1\n')
        subprocess.run(['git', '-C', str(src), *cmd], env=env, check=True)
    commit = subprocess.run(['git', '-C', str(src), 'rev-parse', 'HEAD'], env=env, check=True, capture_output=True,
                            text=True).stdout.strip()
    return src, commit


@pytest.fixture
def staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
           no_launch: list[list[str]]) -> tuple[S.Full, Callable[..., None], list[list[str]]]:
    src, commit = scratch_repo(tmp_path)
    s = settings(tmp_path)
    s = replace(s, settings=replace(s.settings, source=src, commit=commit))
    monkeypatch.setenv('D2_COORDINATOR', P.COORDINATOR)
    monkeypatch.setattr('socket.gethostname', lambda: 'x86host')
    inputs = {'manifest': 'm', 'pilot': 'p', 'recipe': 'ace'}
    monkeypatch.setattr(S, 'pilot_recipe', lambda *a: 'ace')
    monkeypatch.setattr(S, 'admission', lambda *a: dict(inputs))
    S.execute(s, 'stage', None, False)
    return s, lambda phase, job=None, resume=False, attempt=1: S.execute(s, phase, job, resume, attempt), no_launch


def test_stage_requires_new_namespaces_and_no_job(staged: tuple[S.Full, Callable[..., None], list[list[str]]]) -> None:
    s, run, launched = staged
    assert (s.settings.snapshot/'repo/a.py').exists() and launched == []
    with pytest.raises(FileExistsError):
        run('stage')
    for kw in ({'job': 'full-fast'}, {'resume': True}):
        with pytest.raises(FileExistsError):
            run('stage', **kw)


def test_dirty_or_moved_source_refused(staged: tuple[S.Full, Callable[..., None], list[list[str]]]) -> None:
    s, run, launched = staged
    (s.settings.source/'a.py').write_text('x = 2\n')
    with pytest.raises(ValueError, match='clean reviewed source'):
        run('full-train', 'full-fast')
    assert launched == []


def test_staged_tree_and_contract_changes_refused(staged: tuple[S.Full, Callable[..., None], list[list[str]]],
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    s, run, launched = staged
    repo = s.settings.snapshot/'repo'
    (repo/'extra.py').write_text('')
    with pytest.raises(ValueError, match='file set'):
        run('full-train', 'full-fast')
    (repo/'extra.py').unlink()
    (repo/'a.py').write_text('x = 3\n')
    with pytest.raises(ValueError, match='bytes changed'):
        run('full-train', 'full-fast')
    (repo/'a.py').write_text('x = 1\n')
    monkeypatch.setattr(S, 'admission', lambda *a: {'manifest': 'm', 'pilot': 'OTHER'})
    with pytest.raises(ValueError, match='contract changed'):
        run('full-train', 'full-fast')
    assert launched == []


def test_job_launch_rules(staged: tuple[S.Full, Callable[..., None], list[list[str]]],
                          monkeypatch: pytest.MonkeyPatch) -> None:
    s, run, launched = staged
    with pytest.raises(ValueError, match='choose exactly one'):
        run('full-train')
    with pytest.raises(ValueError, match='choose exactly one'):
        run('full-train', 'g2-dev')
    from offload.dspark.d2.train import full_eval
    def incomplete(*a: object) -> str:
        raise ValueError('full G2 requires the completed three-epoch qualified FAST checkpoint')
    monkeypatch.setattr(full_eval, 'completed', incomplete)
    with pytest.raises(ValueError, match='three-epoch'):
        run('full-g2', 'g2-dev')
    assert launched == []
    run('full-train', 'full-fast')
    assert [c[1] for c in launched] == ['has-session', 'new-session']
    assert launched[-1][4] == 'd2full-full-out-full-fast'
    with pytest.raises(FileExistsError):  # script name is exclusive: never relaunch a job
        run('full-train', 'full-fast')
    run('full-train', 'full-fast', resume=True, attempt=1)
    assert launched[-1][4] == 'd2full-full-out-full-fast-resume-1'
    body = (s.settings.snapshot/'full-fast-resume-1.sh').read_text()
    assert '--resume' in body and GPU in body


def test_existing_session_refused(staged: tuple[S.Full, Callable[..., None], list[list[str]]],
                                  monkeypatch: pytest.MonkeyPatch) -> None:
    s, run, launched = staged
    monkeypatch.setattr(subprocess, 'run', lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, b'', b'')
                        if cmd[0] == 'tmux' else pytest.fail(str(cmd)))

    def git(settings: object, *args: str) -> str:
        return s.settings.commit if args[0] == 'rev-parse' else '' if args[0] == 'status' else 'a.py'
    monkeypatch.setattr(S, 'git', git)
    with pytest.raises(ValueError, match='session already exists'):
        run('full-train', 'full-fast')
    assert not (s.settings.snapshot/'full-fast.sh').exists()
