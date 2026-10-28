"""S37 item 1: full-run FAST admission (loop.py G1 swap, full_admission.require_full_pilot).

Real-receipt probes read the S36-inputs copies (read only) and a `git archive` of the reviewed pilot-2 commit into
pytest's tmp dir; they are skipped when either is absent. Paths recorded in the real receipts point at x86host
(/mnt/nvme/...); `x86host_paths` redirects ONLY those reads to the byte-identical S36-inputs copies, never rewriting a receipt.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Iterator, cast

import pytest

from offload.dspark.d2.train import full_admission as A
from offload.dspark.d2.train import loop, pilot
from offload.dspark.d2.train.pilot_verdict import RULE, RULE_VERSION

REPO = Path(__file__).resolve().parents[4]
INPUTS = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/S36-inputs')
HEADER = INPUTS/'pilot-plan-header.json'
MANIFEST = INPUTS/'shards/MANIFEST.json'
X86HOST_G1 = '/opt/dsv41/x86host-data/d2/g1-d2pass2-r1/'
PILOT_COMMIT = A.PILOT_COMMIT
needs_real = pytest.mark.skipif(not (HEADER.exists() and MANIFEST.exists() and (INPUTS/'dev.json').exists()),
                                reason='S36-inputs receipt copies absent')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> subprocess.CompletedProcess[bytes]:
    from offload.dspark.d2.train.tests.snapshot_git import git as scratch_git
    return scratch_git(REPO, *args)


@pytest.fixture(scope='module')
def pristine(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """offload/dspark of the commit the running pilot was staged from (its snapshot is a git archive of it)."""
    try:
        archive = git('archive', PILOT_COMMIT, 'offload/dspark', 'bench.py', 'functest.py', 'quality.py', 'report_tables.py', 'verify_hashes.py').stdout
    except subprocess.CalledProcessError:
        pytest.skip('pilot commit not in this clone')
    root = tmp_path_factory.mktemp('pilot-snapshot')/'repo'
    root.mkdir()
    subprocess.run(['tar', '-x', '-C', str(root)], input=archive, check=True)
    return root


@pytest.fixture
def snapshot(pristine: Path, tmp_path: Path) -> Path:
    root = tmp_path/'repo'
    shutil.copytree(pristine, root, symlinks=True)
    return root


def header_sources() -> dict[str, str]:
    return dict(A.PILOT_SOURCES)


@pytest.fixture
def x86host_paths(monkeypatch: pytest.MonkeyPatch) -> Callable[[Path], None]:
    """Serve reads of the real receipts' x86host paths from a local directory (default: S36-inputs)."""
    base = {'root': INPUTS}
    text, data = Path.read_text, Path.read_bytes

    def mapped(p: Path) -> Path:
        s = str(p)
        return base['root']/s[len(X86HOST_G1):] if s.startswith(X86HOST_G1) else p

    monkeypatch.setattr(Path, 'read_text', lambda p, *a, **k: text(mapped(p), *a, **k))
    monkeypatch.setattr(Path, 'read_bytes', lambda p: data(mapped(p)))
    return lambda root: base.__setitem__('root', root)


# ---------------------------------------------------------------- source lineage


@needs_real
def test_pins_match_real_pilot_lineage_and_candidate(pristine: Path) -> None:
    src = header_sources()
    current = pilot.sources()
    assert src['d2/train/loop.py'] == A.PILOT_LOOP == sha(pristine/'offload/dspark/d2/train/loop.py')
    assert current['d2/train/loop.py'] == A.FULL_LOOP
    # The complete pilot-2 snapshot already contains the original full modules; the merged tree adds only the
    # byte-pinned read-only D2-DIAG module (FULL_EXTRAS, S41 FIX 2).
    assert set(src) <= set(current) and set(current) - set(src) == set(A.FULL_EXTRAS)
    assert {k: current[k] for k in A.FULL_EXTRAS} == A.FULL_EXTRAS
    assert {k for k in src if src[k] != current[k]} == {
        'd2/train/loop.py', 'd2/train/full_admission.py', 'd2/train/full_eval.py', 'd2/train/full_stage.py'}
    assert {k: current[k] for k in A.FULL_PATCHES} == A.FULL_PATCHES
    A.verify_pilot_source(pristine, src)


@needs_real
@pytest.mark.parametrize('mutation', ['extra_train_py', 'changed_g1', 'changed_lock_script', 'changed_native',
                                      'outside_symlink', 'missing_file'])
def test_snapshot_mutations_refused(snapshot: Path, tmp_path: Path, mutation: str) -> None:
    d = snapshot/'offload/dspark'
    if mutation == 'extra_train_py':
        (d/'d2/train/unreviewed.py').write_text('')
    elif mutation == 'changed_g1':
        (d/'d2/g1.py').write_text((d/'d2/g1.py').read_text()+'#\n')
    elif mutation == 'changed_lock_script':
        (d/'d2/train/run_pilot_locked.sh').write_text((d/'d2/train/run_pilot_locked.sh').read_text()+'#\n')
    elif mutation == 'changed_native':
        (d/'native/official.py').write_text((d/'native/official.py').read_text()+'#\n')
    elif mutation == 'outside_symlink':
        outside = tmp_path/'loss.py'
        shutil.copyfile(d/'d2/train/loss.py', outside)
        (d/'d2/train/loss.py').unlink()
        (d/'d2/train/loss.py').symlink_to(outside)  # identical bytes, but outside the snapshot
    else:
        (d/'d2/train/pilot_units.py').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        A.verify_pilot_source(snapshot, header_sources())


@needs_real
def test_inside_symlink_with_identical_bytes_is_admitted(snapshot: Path) -> None:
    """Observation only: a content-identical symlink INSIDE the tree passes (bytes, not inodes, are bound)."""
    d = snapshot/'offload/dspark/d2'
    shutil.copyfile(d/'train/loss.py', d/'train/tests/loss_copy.py')
    (d/'train/loss.py').unlink()
    (d/'train/loss.py').symlink_to(d/'train/tests/loss_copy.py')
    A.verify_pilot_source(snapshot, header_sources())


@needs_real
@pytest.mark.parametrize('name', ['d2/train/model.py', 'd2/live_reference.py', 'd2/train/pilot_verdict.py',
                                  'd2/train/loss.py', 'd2/train/fast_loss.py', 'd2/train/pilot.py',
                                  'd2/train/full_eval.py', 'd2/train/full_stage.py', 'd2/train/full_train.py'])
def test_changed_current_shared_module_refused(pristine: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    changed = dict(pilot.sources(), **{name: '0'*64})
    monkeypatch.setattr(pilot, 'sources', lambda: changed)
    with pytest.raises(ValueError, match='differs from qualified pilot source'):
        A.verify_pilot_source(pristine, header_sources())


@needs_real
def test_changed_current_loop_refused(pristine: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    changed = dict(pilot.sources(), **{'d2/train/loop.py': '1'*64})
    monkeypatch.setattr(pilot, 'sources', lambda: changed)
    with pytest.raises(ValueError, match='full loop changed'):
        A.verify_pilot_source(pristine, header_sources())


@needs_real
def test_other_lineage_refused(pristine: Path) -> None:
    src = header_sources()
    with pytest.raises(ValueError, match='lineage'):
        A.verify_pilot_source(pristine, dict(src, **{'d2/train/loop.py': A.FULL_LOOP}))
    with pytest.raises(ValueError, match='lineage'):
        A.verify_pilot_source(pristine, {})


# ---------------------------------------------------------------- the CPU child


def receipt(tmp_path: Path, verdict: str = 'PASS', synthetic: bool = False, **extra: object) -> Path:
    path = tmp_path/'receipt.json'
    body = {'schema': 'dsv41.d2.pilot-receipt.v1', 'passed': verdict == 'PASS', 'synthetic': synthetic,
            'rule': RULE, 'rule_version': RULE_VERSION, 'verdict': verdict,
            'plan': dict(json.loads(HEADER.read_text()), source_sha256=header_sources()), **extra}
    path.write_text(json.dumps(body))
    return path


def admit(path: Path, root: Path | None) -> str:
    return A.require_full_pilot(path, 'base', MANIFEST, path.parent/'parity', path.parent/'g1', 128, 8192, 32, root)


@needs_real
@pytest.mark.parametrize('verdict,synthetic,extra', [('FAIL', False, {}), ('INCONCLUSIVE', False, {}),
                                                     ('PASS', True, {}), ('PASS', False, {'rule_version': 'old'})])
def test_child_refuses_unqualified_real_lineage_receipts(snapshot: Path, tmp_path: Path, verdict: str,
                                                         synthetic: bool, extra: dict[str, object]) -> None:
    before = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='(?s)original pilot verifier refused.*PASSING real pilot receipt'):
        admit(receipt(tmp_path, verdict, synthetic, **extra), snapshot)
    assert {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*') if p.is_file()} == before


@needs_real
def test_child_refuses_pass_labelled_receipt_without_evidence(snapshot: Path, tmp_path: Path) -> None:
    # A receipt can claim PASS; the unchanged verifier still has to recompute it from plan/evaluation/arms.
    with pytest.raises(ValueError, match='original pilot verifier refused'):
        admit(receipt(tmp_path), snapshot)


@needs_real
def test_parent_python_env_does_not_leak_into_child(snapshot: Path, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    evil = tmp_path/'evil'
    fake = evil/'offload/dspark/d2/train'
    fake.mkdir(parents=True)
    (fake/'fast_policy.py').write_text(
        'import hashlib\ndef require_pilot(p, *a):\n    return hashlib.sha256(p.read_bytes()).hexdigest()\n')
    (evil/'sitecustomize.py').write_text('import os\nos._exit(0)\n')
    (evil/'startup.py').write_text('raise SystemExit(0)\n')
    for k, v in {'PYTHONPATH': str(evil), 'PYTHONSTARTUP': str(evil/'startup.py'), 'PYTHONINSPECT': '',
                 'PYTHONUSERBASE': str(evil)}.items():
        monkeypatch.setenv(k, v)
    seen: dict[str, str] = {}
    real = subprocess.run

    def spy(*args: object, **kwargs: object) -> object:
        seen.update(cast(dict[str, str], kwargs['env']))
        return real(*args, **kwargs)  # type: ignore[call-overload]
    monkeypatch.setattr(subprocess, 'run', spy)
    with pytest.raises(ValueError, match='PASSING real pilot receipt'):
        admit(receipt(tmp_path, 'FAIL'), snapshot)
    assert seen['PYTHONPATH'] == f'{snapshot}:{snapshot}/offload/dspark' and seen['CUDA_VISIBLE_DEVICES'] == ''
    assert not {k for k in seen if k.startswith('PYTHON')} - {'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE'}


@needs_real
def test_receipt_swapped_during_child_refused(snapshot: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = receipt(tmp_path)
    digest = sha(path)

    def child(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        path.write_text(path.read_text()+' ')
        return subprocess.CompletedProcess([], 0, digest+'\n', '')
    monkeypatch.setattr(subprocess, 'run', child)
    with pytest.raises(ValueError, match='changed during original-source verification'):
        admit(path, snapshot)


@needs_real
def test_child_wrong_hash_or_nonzero_exit_refused(snapshot: Path, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    path = receipt(tmp_path)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess([], 0, '0'*64, ''))
    with pytest.raises(ValueError, match='changed during'):
        admit(path, snapshot)

    def fails(*a: object, **k: object) -> None:
        raise subprocess.CalledProcessError(1, 'child', sha(path), 'boom')
    monkeypatch.setattr(subprocess, 'run', fails)
    with pytest.raises(ValueError, match='refused: boom'):
        admit(path, snapshot)


@needs_real
@pytest.mark.parametrize('where', ['repo_root', 'dspark_root'])
def test_unhashed_startup_file_in_snapshot_is_refused(snapshot: Path, tmp_path: Path, where: str) -> None:
    """A planted startup hook cannot admit a FAIL by printing its receipt hash."""
    target = snapshot if where == 'repo_root' else snapshot/'offload/dspark'
    (target/'sitecustomize.py').write_text(
        'import hashlib, os, sys\n'
        'if len(sys.argv) > 1 and sys.argv[1].endswith("receipt.json"):\n'
        '    print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()); sys.stdout.flush(); os._exit(0)\n')
    path = receipt(tmp_path, 'FAIL')
    with pytest.raises(ValueError, match='unbound pilot import source'):
        A.verify_pilot_source(snapshot, header_sources())
    with pytest.raises(ValueError, match='unbound pilot import source'):
        admit(path, snapshot)


@needs_real
def test_no_pilot_source_fails_closed_for_the_real_pilot(tmp_path: Path) -> None:
    # Same-source policy: the candidate tree's sources() (new loop + full_*) never equal the real plan's.
    with pytest.raises(ValueError, match='pilot source identity mismatch'):
        admit(receipt(tmp_path), None)


@needs_real
def test_real_pilot_snapshot_tree_has_no_stdlib_shadowing(pristine: Path) -> None:
    """The child's '' (=snapshot) and offload/dspark roots precede the stdlib; the real tree shadows nothing."""
    names = {p.stem for root in (pristine, pristine/'offload/dspark') for p in root.iterdir()
             if p.suffix == '.py' or (p.is_dir() and not p.name.startswith('.'))}
    stdlib = set(sys.stdlib_module_names) | {'sitecustomize', 'usercustomize', 'torch', 'safetensors', 'numpy'}
    assert names & stdlib == set()


# ---------------------------------------------------------------- loop.py G1 admission on the real receipts


def g1_copy(tmp_path: Path, **change: object) -> Path:
    gate = json.loads((INPUTS/'dev.json').read_text())
    for key, value in change.items():
        if key.startswith('g1.'):
            gate['g1'][key[3:]] = value
        else:
            gate[key] = value
    path = tmp_path/'g1.json'
    path.write_text(json.dumps(gate))
    return path


@needs_real
def test_real_dev_g1_admitted(x86host_paths: Callable[[Path], None]) -> None:
    live = pilot.validate_live_g1(INPUTS/'dev.json', MANIFEST)
    assert live['status'] == 'VALID' and live['sha256'].startswith('af6ad0d3')


@needs_real
@pytest.mark.parametrize('case', ['heldout_legal', 'heldout_general', 'acap', 'synthetic', 'tripwire', 'fail',
                                  'status', 'amendment7', 'fraction', 'statistic', 'score', 'manifest', 'live_hash',
                                  'live_missing'])
def test_g1_refusals(x86host_paths: Callable[[Path], None], tmp_path: Path, case: str) -> None:
    manifest = MANIFEST
    if case in ('heldout_legal', 'heldout_general'):
        path = INPUTS/f'{case}.json'
    elif case == 'acap':  # the historical D2-DATA receipt shape the old loop accepted
        path = tmp_path/'acap.json'
        path.write_text(json.dumps({'split': 'acap', 'g1': {'binding': True, 'pass': True}}))
    elif case == 'synthetic':
        path = g1_copy(tmp_path, synthetic=True)
    elif case == 'tripwire':
        path = g1_copy(tmp_path, **{'g1.tripwire_passed': False})
    elif case == 'fail':
        path = g1_copy(tmp_path, **{'g1.pass': False, 'g1.status': 'FAIL'})
    elif case == 'status':
        path = g1_copy(tmp_path, **{'g1.status': 'INCONCLUSIVE'})
    elif case == 'amendment7':
        path = g1_copy(tmp_path, **{'g1.amendment': '7'})
    elif case == 'fraction':
        path = g1_copy(tmp_path, **{'g1.minimum_live_fraction': .4})
    elif case == 'statistic':
        path = g1_copy(tmp_path, **{'g1.statistic': 'accepted_per_round'})
    elif case == 'score':  # verdict recomputed from the score, not trusted
        path = g1_copy(tmp_path, position0_match_rate=.3)
    elif case == 'manifest':
        path = INPUTS/'dev.json'
        manifest = tmp_path/'MANIFEST.json'
        manifest.write_text(MANIFEST.read_text()+' ')
    elif case == 'live_hash':
        path = g1_copy(tmp_path, **{'g1.live_reference_sha256': '0'*64})
    else:
        path = INPUTS/'dev.json'
        x86host_paths(tmp_path/'nowhere')
    with pytest.raises((ValueError, FileNotFoundError)):
        pilot.validate_live_g1(path, manifest)


@needs_real
def test_invalid_live_reference_refused(x86host_paths: Callable[[Path], None], tmp_path: Path) -> None:
    local = tmp_path/'live'
    shutil.copytree(INPUTS, local, ignore=shutil.ignore_patterns('shards', 'decode-plan', 'COPY_RECEIPT.json'))
    os.chmod(local, 0o700)
    for p in [local/'live-reference.json', local/'live-sources', *(local/'live-sources').iterdir()]:
        os.chmod(p, 0o700 if p.is_dir() else 0o600)
    ref = json.loads((local/'live-reference.json').read_text())
    ref['status'] = 'INCONCLUSIVE'
    (local/'live-reference.json').write_text(json.dumps(ref))
    x86host_paths(local)
    with pytest.raises(ValueError, match='live reference receipt or snapshot hash changed'):
        pilot.validate_live_g1(INPUTS/'dev.json', MANIFEST)
    rate = local/'live-sources/rate-20261025T061849Z.json'
    (local/'live-reference.json').write_text((INPUTS/'live-reference.json').read_text())
    row = json.loads(rate.read_text())
    row['gen_tokens'] -= 20000  # foreign share > 2 %  -> INCONCLUSIVE, hash no longer matches
    rate.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        pilot.validate_live_g1(INPUTS/'dev.json', MANIFEST)


class Loaded(Exception):
    pass


@pytest.fixture
def stop_at_load(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[bool]]:
    calls: list[bool] = []
    monkeypatch.setattr(loop, 'require_full_pilot', lambda *a: 'hash')
    monkeypatch.setattr(loop, 'pilot_recipe', lambda *a: 'ace')
    original_read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda p, *a, **k: json.dumps({'plan': {'frozen_identity': 'base'}}) if p.name == 'r.json' else original_read(p, *a, **k))

    def load(ckpt: str, device: str, rank: int, reference_math: bool) -> None:
        calls.append(reference_math)
        raise Loaded()
    monkeypatch.setattr(loop, 'load', load)
    yield calls


def argv(tmp_path: Path, mode: str, g1: Path) -> list[str]:
    extra = ['--pilot-receipt', str(tmp_path/'r.json')] if mode == '--fast-training' else []
    return ['loop', '--manifest', str(MANIFEST), '--ckpt', 'x', '--output', str(tmp_path/'o.pt'), mode,
            '--parity-receipt', str(tmp_path/'p'), '--g1-receipt', str(g1), '--batch-size', '128', '--chunk', '8192', *extra]


@needs_real
@pytest.mark.parametrize('mode', ['--reference-math', '--fast-training'])
def test_loop_admits_real_dev_g1_for_both_modes(x86host_paths: Callable[[Path], None], stop_at_load: list[bool],
                                                tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setattr(sys, 'argv', argv(tmp_path, mode, INPUTS/'dev.json'))
    with pytest.raises(Loaded):
        loop.main()
    assert stop_at_load == [mode == '--reference-math']


@needs_real
@pytest.mark.parametrize('mode', ['--reference-math', '--fast-training'])
def test_loop_refuses_acap_and_heldout_before_load(x86host_paths: Callable[[Path], None], stop_at_load: list[bool],
                                                   tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    acap = tmp_path/'acap.json'
    acap.write_text(json.dumps({'split': 'acap', 'g1': {'binding': True, 'pass': True}}))
    for g1 in (acap, INPUTS/'heldout_legal.json'):
        monkeypatch.setattr(sys, 'argv', argv(tmp_path, mode, g1))
        with pytest.raises(ValueError, match='amendment-7b'):
            loop.main()
    assert stop_at_load == []


# ---------------------------------------------------------------- positive path through the real child

INJECT = r'''
import runpy, sys
from pathlib import Path
# Test harness ONLY: the snapshot's own test fixture uses a 23,000-block stand-in dataset instead of real shards.
from offload.dspark.d2.train import data, pilot_eval
from offload.dspark.d2.train.tests.test_pilot_policy import Population
data.D2Blocks = lambda *_: Population()
while sys.argv[1] in ('-I', '-B'): del sys.argv[1]
import json
spec = json.loads(Path(sys.argv[3]).read_text())['plan']
pilot_eval.D2Blocks = lambda _, split: split
pilot_eval.complete_rounds = lambda split, _: list(enumerate(spec['sample']['keys'][split]))
assert sys.argv[1] == '-c'
code = sys.argv[2]
sys.argv = ['-c', *sys.argv[3:]]
exec(compile(code, '<child>', 'exec'), {'__name__': '__main__'})
'''


def pass_receipt(snapshot: Path, tmp_path: Path, recipe: pilot.Recipe = 'kl', *, tamper_sample: bool = False) -> list[Path]:
    """Genuine-format non-synthetic PASS receipt built by the PILOT's own code (pilot_verdict.build) in the snapshot."""
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
    env.update(PYTHONPATH=f'{snapshot}:{snapshot}/offload/dspark', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    code = r"""
import sys
from pathlib import Path
from _pytest.monkeypatch import MonkeyPatch
from offload.dspark.d2.train import pilot
from offload.dspark.d2.train.tests import test_pilot_policy as fixture
from offload.dspark.d2.train.data import sha256
recipe = sys.argv[2]
base = Path(sys.argv[1])
for split in ('dev', 'heldout_legal'):
    (base/(split+'.k2_drafts.jsonl')).write_text('CPU fixture '+split)
original_evidence = fixture.evidence
def evidence():
    spec, ev = original_evidence()
    ev['k2_sha256'] = {s: sha256(base/(s+'.k2_drafts.jsonl')) for s in ('dev', 'heldout_legal')}
    return spec, ev
fixture.evidence = evidence
original_write, original_run = pilot.write_json, pilot.pilot_run
original_schedule, original_units = pilot.schedule, pilot.matched_units
pilot.pilot_run = lambda r=recipe: original_run(r)
pilot.schedule = lambda length, points, arm, r=recipe: original_schedule(length, points, arm, r)
pilot.matched_units = lambda data, points, r=recipe: original_units(data, points, r)
def write(p, v):
    if p.name in ('plan.json', 'arm.json'): v['recipe'] = recipe
    if p.name == 'plan.json' and sys.argv[3] == 'tamper':
        v['sample']['population_sha256']['dev'] = '0'*64
    original_write(p, v)
pilot.write_json = write
print(*fixture.receipt_fixture(base, MonkeyPatch()))
"""
    out = tmp_path/'pilot-out'
    out.mkdir()
    done = subprocess.run([sys.executable, '-c', code, str(out), recipe, 'tamper' if tamper_sample else 'clean'], cwd=snapshot, env=env, check=True,
                          capture_output=True, text=True)
    return [Path(p) for p in done.stdout.strip().splitlines()[-1].split()]


@needs_real
@pytest.mark.parametrize('recipe', pilot.RECIPES)
def test_genuine_format_pass_admitted_through_child_and_tamper_refused(snapshot: Path, tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch, recipe: pilot.Recipe) -> None:
    path, manifest, parity, g1 = pass_receipt(snapshot, tmp_path, recipe)
    plan = json.loads(path.read_text())['plan']
    assert plan['source_sha256'] == header_sources()  # same lineage as the real running pilot
    wrapper = tmp_path/'python-with-population'
    wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} {tmp_path/"inject.py"} "$@"\n')
    wrapper.chmod(0o700)
    (tmp_path/'inject.py').write_text(INJECT)
    monkeypatch.setattr(sys, 'executable', str(wrapper))
    d = path.parent
    args = (path, 'base', manifest, parity, g1, 128, 8192, 32, snapshot)
    assert A.require_full_pilot(*args) == sha(path)
    assert A.pilot_recipe(path, sha(path)) == recipe
    # The same receipt is NOT admitted by the candidate's same-source policy (fails closed without --pilot-source).
    with pytest.raises(ValueError, match='pilot source identity mismatch'):
        A.require_full_pilot(*args[:-1], None)
    for bad in ((path, 'other-base', *args[2:]), (*args[:5], 64, 8192, 32, snapshot)):
        with pytest.raises(ValueError, match='original pilot verifier refused'):
            A.require_full_pilot(*bad)
    # Tamper with one archived FAST checkpoint: the original verifier recomputes and refuses.
    ckpt = next((d/'FAST/latest.checkpoints').iterdir())
    ckpt.write_bytes(ckpt.read_bytes()+b'\0')
    with pytest.raises(ValueError, match='(?s)original pilot verifier refused.*checkpoint'):
        A.require_full_pilot(*args)


@needs_real
def test_first_pilot_lineage_now_refuses(pristine: Path) -> None:
    first = dict(json.loads(HEADER.read_text())['source_sha256'])
    assert first != header_sources()
    with pytest.raises(ValueError, match='unsupported pilot source lineage'):
        A.verify_pilot_source(pristine, first)


@needs_real
@pytest.mark.parametrize('name', ['loop.py', 'loss.py', 'fast_loss.py', 'pilot.py', 'pilot_eval.py',
                                  'pilot_stage.py', 'pilot_units.py', 'full_eval.py', 'full_stage.py', 'full_admission.py'])
def test_pilot2_original_reviewed_files_are_immutable(snapshot: Path, name: str) -> None:
    path = snapshot/'offload/dspark/d2/train'/name
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match='immutable pilot source changed'):
        A.verify_pilot_source(snapshot, header_sources())


@needs_real
@pytest.mark.parametrize('defect', ['missing_recipe', 'borrowed_recipe', 'sample_redraw', 'wrong_k2'])
def test_pilot2_recipe_and_sample_refusals(snapshot: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                         defect: str) -> None:
    path, manifest, parity, g1 = pass_receipt(snapshot, tmp_path, 'ace', tamper_sample=defect == 'sample_redraw')
    wrapper = tmp_path/'python-with-population'
    wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} {tmp_path/"inject.py"} "$@"\n')
    wrapper.chmod(0o700)
    (tmp_path/'inject.py').write_text(INJECT)
    monkeypatch.setattr(sys, 'executable', str(wrapper))
    if defect in ('missing_recipe', 'borrowed_recipe'):
        body = json.loads(path.read_text())
        if defect == 'missing_recipe':
            del body['plan']['recipe']
        else:
            body['plan']['recipe'] = 'lr'
        path.write_text(json.dumps(body))
    elif defect == 'wrong_k2':
        (g1.parent/'dev.k2_drafts.jsonl').write_text('changed CPU fixture')
    with pytest.raises(ValueError, match='original pilot verifier refused') as error:
        A.require_full_pilot(path, 'base', manifest, parity, g1, 128, 8192, 32, snapshot)
    if defect == 'sample_redraw':
        assert 'seed-20261024 population draw' in str(error.value)
    if defect == 'wrong_k2':
        assert 'K2 sources differ' in str(error.value)


@needs_real
@pytest.mark.parametrize('mutation', ['native_init', 'cached_code', 'extension', 'stdlib_package', 'native_package'])
def test_unbound_original_import_implementations_refused(snapshot: Path, mutation: str) -> None:
    if mutation == 'native_init':
        path = snapshot/'offload/dspark/native/__init__.py'
        path.write_text(path.read_text()+'\n')
    elif mutation == 'cached_code':
        (snapshot/'offload/dspark/d2/train/fast_policy.pyc').write_bytes(b'unbound CPU fixture')
    elif mutation == 'extension':
        (snapshot/'offload/dspark/d2/train/fast_policy.so').write_bytes(b'unbound CPU fixture')
    else:
        shadow = 'json' if mutation == 'stdlib_package' else 'native'
        (snapshot/shadow).mkdir()
        (snapshot/shadow/'__init__.py').write_text('raise RuntimeError("unbound import")')
    with pytest.raises(ValueError, match='immutable pilot package initializer changed|unbound compiled pilot import source|unbound pilot import shadow'):
        A.verify_pilot_source(snapshot, header_sources())
