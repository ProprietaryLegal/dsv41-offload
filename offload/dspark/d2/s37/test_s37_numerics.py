"""S37 item 2: the full loop's arithmetic, cancellation, checkpointing and resume equal the pilot's loop.

Static: compare every definition to immutable pilot-2 d1dd41a. Only the CLI and explicit full-recipe checkpoint
metadata/resume guard differ. Dynamic: every recipe trains the same tiny CPU model with interruption/resume in
separate original-source and candidate interpreters; checkpoint state and timing-free curves compare bit for bit.
"""
from __future__ import annotations

import ast
from dataclasses import asdict
import difflib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import cast

import pytest
import torch

from offload.dspark.d2.train import full_train, loop, pilot

REPO = Path(__file__).resolve().parents[4]
LOOP = 'offload/dspark/d2/train/loop.py'


def old_loop() -> str:
    from offload.dspark.d2.train.tests.snapshot_git import git
    try:
        return git(REPO, 'show', 'd1dd41a:'+LOOP).stdout.decode()
    except subprocess.CalledProcessError:
        pytest.skip('pilot commit absent')


def test_only_main_and_imports_differ_from_pilot_loop() -> None:
    old, new = ast.parse(old_loop()), ast.parse((REPO/LOOP).read_text())
    def defs(tree: ast.Module) -> dict[str, str]:
        return {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    a, b = defs(old), defs(new)
    assert set(a) == set(b)
    assert {n for n in a if a[n] != b[n]} == {'main', 'snapshot', '_train'}
    # Arithmetic stays identical. These two changes only bind full recipe metadata.
    def source(text: str, name: str) -> list[str]:
        n = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
        return (ast.get_source_segment(text, n) or '').splitlines()
    for name, expected in {
        'snapshot': ['-    return {"schema": "dsv41.d2.train-checkpoint.v1", "run": asdict(run), "dataset": data.fingerprint,',
                     '+    return {**({"full_recipe": model.full_recipe} if hasattr(model, "full_recipe") else {}),',
                     '+            "schema": "dsv41.d2.train-checkpoint.v1", "run": asdict(run), "dataset": data.fingerprint,'],
        '_train': ['+        if state.get("full_recipe") != getattr(model, "full_recipe", None):',
                   '+            raise ValueError("resume full recipe changed")']}.items():
        diff = [l for l in difflib.unified_diff(source(old_loop(), name), source((REPO/LOOP).read_text(), name), n=0, lineterm='')
                if l[:1] in '+-' and not l.startswith(('+++', '---'))]
        assert diff == expected


def test_main_diff_is_admission_only() -> None:
    from offload.dspark.d2.train.full_admission import FULL_LOOP
    from offload.dspark.d2.train.data import sha256
    assert sha256(REPO/LOOP) == FULL_LOOP
    text = (REPO/LOOP).read_text()
    assert 'qualified_run = full_run(recipe)' in text
    assert text.index('pilot_hash = require_full_pilot(') < text.index('model = load(a.ckpt')
    assert 'initialization(model) != spec["initialization_sha256"]' in text


def test_full_run_settings_equal_pilot_except_epochs() -> None:
    full = asdict(loop.Run(epochs=3, batch_size=128, chunk=8192, rank=32))  # exactly what loop.main builds
    pilot_run = asdict(pilot.pilot_run())
    assert {k for k in full if full[k] != pilot_run[k]} == {'epochs'} and full['epochs'] == 3


TRAIN = r'''
import json, sys, threading, torch
from pathlib import Path
from offload.dspark.d2.train.data import SyntheticBlocks
from offload.dspark.d2.train.loop import Run, train
from offload.dspark.d2.train.pilot import pilot_run
from dataclasses import replace
from offload.dspark.d2.train.model import Drafter
from offload.dspark.d2.train.synthetic import SyntheticSource, tiny_config
import offload.dspark.d2.train.loop as L
torch.set_num_threads(2)
out = Path(sys.argv[1]); out.mkdir(parents=True)
def model():
    torch.manual_seed(23)
    c = tiny_config()
    return Drafter(SyntheticSource(c), c, rank=32)
data = SyntheticBlocks(tiny_config(), count=5)
run = replace(pilot_run(sys.argv[2]), epochs=2, chunk=32, batch_size=2)
train(model(), data, run, out/'whole.pt')
train(model(), data, run, out/'cut.pt', max_steps=3)
stop = threading.Event(); stop.set()
r = train(model(), data, run, out/'cut.pt', stop, resume=True)   # cancelled resume: boundary only
train(model(), data, run, out/'cut.pt', resume=True)
print(json.dumps({'file': L.__file__, 'cancelled_step': r['step']}))
'''


def run_loop(root: Path, out: Path, recipe: pilot.Recipe = 'kl') -> dict[str, object]:
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
    env.update(PYTHONPATH=f'{root}:{root}/offload/dspark', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    done = subprocess.run([sys.executable, '-c', TRAIN, str(out), recipe], cwd=root, env=env, check=True,
                          capture_output=True, text=True)
    return dict(json.loads(done.stdout.strip().splitlines()[-1]))


def load(path: Path) -> dict[str, object]:
    return dict(torch.load(path, map_location='cpu', weights_only=False))


def same(a: object, b: object) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return bool(a == b)


@pytest.mark.parametrize('recipe', pilot.RECIPES)
def test_pilot_loop_and_full_loop_train_identically_with_interrupt_and_resume(tmp_path: Path, recipe: pilot.Recipe) -> None:
    from offload.dspark.d2.train.tests.snapshot_git import git
    old = tmp_path/'old'
    old.mkdir()
    archive = git(REPO, 'archive', 'd1dd41a', 'offload/dspark/d2', 'offload/dspark/native').stdout
    subprocess.run(['tar', '-x', '-C', str(old)], input=archive, check=True)
    a, b = run_loop(old, tmp_path/'a', recipe), run_loop(REPO, tmp_path/'b', recipe)
    assert str(a['file']).startswith(str(old)) and str(b['file']).startswith(str(REPO))
    assert a['cancelled_step'] == b['cancelled_step'] == 3
    for name in ('whole.pt', 'cut.pt'):
        sa, sb = load(tmp_path/'a'/name), load(tmp_path/'b'/name)
        sa.pop('curve'), sb.pop('curve')  # wall-clock seconds differ; losses compared below
        assert same(sa, sb), name
    def curve(x: str) -> list[dict[str, object]]:
        rows = cast(list[dict[str, object]], load(tmp_path/x/'whole.pt')['curve'])
        return [{k: v for k, v in r.items() if k != 'seconds'} for r in rows]
    ca, cb = curve('a'), curve('b')
    assert ca == cb
    # Exact resume holds inside each implementation too (interrupted == uninterrupted trainables/optimizer).
    for x in 'ab':
        w, c = load(tmp_path/x/'whole.pt'), load(tmp_path/x/'cut.pt')
        assert same(w['trainable'], c['trainable']) and same(w['optimizer'], c['optimizer'])
    archived = sorted(p.name for p in (tmp_path/'b/cut.checkpoints').iterdir())
    assert archived == sorted(p.name for p in (tmp_path/'a/cut.checkpoints').iterdir())


class Constructed(Exception):
    pass


def rng_at_load(monkeypatch: pytest.MonkeyPatch, module: object) -> list[tuple[torch.Tensor, tuple[object, ...]]]:
    seen: list[tuple[torch.Tensor, tuple[object, ...]]] = []

    def load(*args: object, **kwargs: object) -> None:
        seen.append((torch.get_rng_state(), (*args, *sorted(kwargs.items()))))
        raise Constructed()
    monkeypatch.setattr(module, 'load', load)
    return seen


def test_full_train_seeds_construction_like_the_pilot_fast_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Global RNG at model construction == fresh manual_seed(23) in BOTH paths, with the same load() call."""
    from offload.dspark.d2.train import safety
    monkeypatch.setattr(safety, 'assert_gpu_lease', lambda: None)
    # Stubbed here; test_real_g1_admission_consumes_no_rng shows the real check draws nothing between seed and load.
    monkeypatch.setattr(pilot, 'validate_live_g1', lambda *a: None)
    full = rng_at_load(monkeypatch, loop)
    monkeypatch.setattr(sys, 'argv', ['full_train', '--manifest', 'm', '--ckpt', 'ck', '--output', str(tmp_path/'o.pt'),
        '--device', 'cuda', '--fast-training', '--epochs', '3', '--batch-size', '128', '--chunk', '8192', '--rank', '32',
        '--pilot-receipt', 'r', '--pilot-source', 's', '--parity-receipt', 'p', '--g1-receipt', 'g'])
    monkeypatch.setattr(loop, 'require_full_pilot', lambda *a: 'hash')
    monkeypatch.setattr(loop, 'pilot_recipe', lambda *a: 'ace')
    original_read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda p, *a, **k: json.dumps({'plan': {'frozen_identity': 'base'}}) if str(p) == 'r' else original_read(p, *a, **k))
    torch.manual_seed(999)
    with pytest.raises(Constructed):
        full_train.main()
    # The pilot FAST arm path (pilot.arm) up to construction, inputs stubbed.
    import argparse
    from offload.dspark.d2.train import data as D
    arm = rng_at_load(monkeypatch, pilot)
    spec = {'checkpoints': [1, 2, 3, 4], 'population_sha256': 'x', 'arm_orders': {'FAST': 'y'}}
    monkeypatch.setattr(pilot, 'read_spec', lambda *a, **k: spec)
    monkeypatch.setattr(pilot, 'validate_inputs', lambda *a: None)
    monkeypatch.setattr(pilot, 'D2Blocks', lambda *a: list(range(8)))
    monkeypatch.setattr(pilot, 'verify_units', lambda *a: None)
    monkeypatch.setattr(pilot, 'schedule', lambda *a: (None, [], 'y'))
    monkeypatch.setattr(pilot, 'digest', lambda *a: 'x')
    torch.manual_seed(999)
    with pytest.raises(Constructed):
        pilot.arm(argparse.Namespace(plan='p', manifest='m', parity_receipt='p', g1_receipt='g', ckpt='ck', arm='FAST'))
    torch.manual_seed(pilot.SEED)
    fresh = torch.get_rng_state()
    assert torch.equal(full[0][0], fresh) and torch.equal(arm[0][0], fresh)
    assert full[0][1] == ('ck', 'cuda', 32, ('reference_math', False))
    assert arm[0][1] == ('ck', 'cuda', ('rank', 32), ('reference_math', False))
    del D


def test_full_epoch0_starts_with_the_pilot_population_order() -> None:
    """Information: epoch 0 uses randperm(seed 23) = the pilot's 20K population order (REF/FAST arms)."""
    n = 488031
    epoch0 = torch.randperm(n, generator=torch.Generator().manual_seed(23)).tolist()
    population = torch.randperm(n, generator=torch.Generator().manual_seed(pilot.SEED)).tolist()[:20000]
    assert epoch0[:20000] == population


def test_real_g1_admission_consumes_no_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    """full_train seeds BEFORE loop.main's admission; only RNG-free work may sit between seed and load().
    (Nothing re-checks the pilot's initialization_sha256 at full-run time, so this is the only guard.)"""
    from offload.dspark.d2.s37.test_s37_admission import X86HOST_G1, INPUTS, MANIFEST, needs_real
    if not (INPUTS/'dev.json').exists():
        pytest.skip('S36-inputs absent')
    text, data = Path.read_text, Path.read_bytes

    def mapped(p: Path) -> Path:
        return INPUTS/str(p)[len(X86HOST_G1):] if str(p).startswith(X86HOST_G1) else p

    def read_text(p: Path, *a: object, **k: object) -> str:
        return text(mapped(p), *a, **k)  # type: ignore[arg-type]

    def read_bytes(p: Path) -> bytes:
        return data(mapped(p))
    monkeypatch.setattr(Path, 'read_text', read_text)
    monkeypatch.setattr(Path, 'read_bytes', read_bytes)
    torch.manual_seed(23)
    before = torch.get_rng_state()
    pilot.validate_live_g1(INPUTS/'dev.json', MANIFEST)
    assert torch.equal(torch.get_rng_state(), before)
    del needs_real
