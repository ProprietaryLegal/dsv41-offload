"""CPU recipe identity and exact full-run/resume arithmetic; no real qualification."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import threading

import pytest
import torch

from offload.dspark.d2.train import full_admission as A, pilot
from offload.dspark.d2.train.data import SyntheticBlocks, sha256
from offload.dspark.d2.train.loop import lr_multiplier, train
from offload.dspark.d2.train.model import Drafter
from offload.dspark.d2.train.tests.test_loop import model


@pytest.mark.parametrize('recipe', pilot.RECIPES)
def test_full_recipe_exact_schedule_and_resume(tmp_path: Path, recipe: pilot.Recipe) -> None:
    run = A.full_run(recipe)
    assert run == replace(pilot.pilot_run(recipe), epochs=3)
    # Tiny full epochs use the full total, never the pilot's literal 159 rates.
    run = replace(run, chunk=32, batch_size=2)
    data = SyntheticBlocks(model().c, count=3)
    def fresh() -> Drafter:
        m = model()
        setattr(m, 'full_recipe', recipe)
        return m
    train(fresh(), data, run, tmp_path/'whole.pt')
    train(fresh(), data, run, tmp_path/'cut.pt', max_steps=2)
    archive = tmp_path/'cut.checkpoints/step-000000002.pt'
    digest = sha256(archive)
    stop = threading.Event()
    stop.set()
    assert train(fresh(), data, run, tmp_path/'cut.pt', stop, resume=True)['step'] == 2
    train(fresh(), data, run, tmp_path/'cut.pt', resume=True)
    whole = torch.load(tmp_path/'whole.pt', weights_only=False)
    cut = torch.load(tmp_path/'cut.pt', weights_only=False)
    for name in whole['trainable']:
        assert torch.equal(whole['trainable'][name], cut['trainable'][name])
    assert whole['scheduler'] == cut['scheduler']
    assert whole['full_recipe'] == cut['full_recipe'] == recipe
    assert [r['applied_lr'] for r in whole['curve']] == [run.lr*lr_multiplier(run, i, 6) for i in range(6)]
    assert sha256(archive) == digest
    with pytest.raises(ValueError, match='resume full recipe changed'):
        train(model(), data, run, tmp_path/'cut.pt', resume=True)
    with pytest.raises(ValueError, match='resume contract changed'):
        train(fresh(), data, replace(run, lr=run.lr/2), tmp_path/'cut.pt', resume=True)


@pytest.mark.parametrize('recipe', pilot.RECIPES)
def test_explicit_receipt_recipe_and_hash(tmp_path: Path, recipe: pilot.Recipe) -> None:
    path = tmp_path/'receipt.json'
    spec = {'recipe': recipe, 'run': asdict(pilot.pilot_run(recipe))}
    path.write_text(json.dumps({'plan': spec}))
    digest = sha256(path)
    assert A.pilot_recipe(path, digest) == recipe
    spec['recipe'] = 'lr' if recipe != 'lr' else 'ace'
    path.write_text(json.dumps({'plan': spec}))
    with pytest.raises(ValueError, match='recipe and matched run differ'):
        A.pilot_recipe(path)
    del spec['recipe']
    path.write_text(json.dumps({'plan': spec}))
    with pytest.raises(ValueError, match='explicit qualified pilot recipe'):
        A.pilot_recipe(path)
    spec['recipe'] = recipe
    path.write_text(json.dumps({'plan': spec})+' ')
    with pytest.raises(ValueError, match='changed while reading recipe'):
        A.pilot_recipe(path, digest)


@pytest.mark.parametrize('recipe', pilot.RECIPES)
def test_full_cli_uses_admitted_recipe_before_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  recipe: pilot.Recipe) -> None:
    import sys
    from offload.dspark.d2.train import loop
    m = model()
    receipt = tmp_path/'receipt.json'
    receipt.write_text(json.dumps({'plan': {'recipe': recipe, 'run': asdict(pilot.pilot_run(recipe)),
        'frozen_identity': m.frozen_identity, 'initialization_sha256': pilot.initialization(m)}}))
    parity = tmp_path/'parity.json'
    parity.write_text(json.dumps({'pass': True, 'samples': 50, 'reference_math': True,
                                 'frozen_identity': m.frozen_identity}))
    order: list[str] = []
    monkeypatch.setattr(pilot, 'validate_live_g1', lambda *a: None)
    def admit(*args: object) -> str:
        order.append('admit')
        return sha256(receipt)
    def load(*args: object, **kwargs: object) -> Drafter:
        order.append('load')
        return m
    monkeypatch.setattr(loop, 'require_full_pilot', admit)
    monkeypatch.setattr(loop, 'load', load)
    monkeypatch.setattr(loop, 'D2Blocks', lambda *a: SyntheticBlocks(m.c, count=3))
    seen: list[object] = []
    def training(*args: object) -> dict[str, object]:
        seen.append(args[2])
        return {'complete': True}
    monkeypatch.setattr(loop, 'train', training)
    monkeypatch.setattr(m, 'enable_fast_training', lambda: None)
    argv = ['full_train', '--manifest', str(tmp_path/'manifest'), '--ckpt', 'native',
        '--pilot-receipt', str(receipt), '--pilot-source', str(tmp_path/'source'),
        '--parity-receipt', str(parity), '--g1-receipt', str(tmp_path/'g1'),
        '--fast-training', '--batch-size', '128', '--chunk', '8192', '--output', str(tmp_path/'full.pt')]
    monkeypatch.setattr(sys, 'argv', argv)
    loop.main()
    assert order == ['admit', 'load']
    assert seen == [A.full_run(recipe)]
    assert getattr(m, 'full_recipe') == recipe
    assert getattr(m, 'fast_pilot_sha256') == sha256(receipt)
    # A CLI objective override or a pilot adapter must refuse before model load.
    for extra in (['--divergence', 'tvd'], ['--initial-adapter', 'pilot.pt'], ['--epochs', '1']):
        order.clear()
        monkeypatch.setattr(sys, 'argv', argv+extra)
        with pytest.raises(ValueError, match='fresh initialization and fixed qualified settings'):
            loop.main()
        assert order == []


def test_g2_parts_require_the_same_explicit_recipe() -> None:
    from offload.dspark.d2.train.full_eval import verdict
    from offload.dspark.d2.train.tests.test_full_stage import parts
    for recipe in pilot.RECIPES:
        p = parts()
        for row in p:
            row['recipe'] = recipe
        assert verdict(p)['recipe'] == recipe
        p[1]['recipe'] = 'lr' if recipe != 'lr' else 'ace'
        with pytest.raises(ValueError, match='identity mismatch'):
            verdict(p)
    p = parts()
    for row in p:
        del row['recipe']
    with pytest.raises(ValueError):
        verdict(p)


@pytest.mark.parametrize('recipe', pilot.RECIPES)
def test_stage_binds_recipe_and_rejects_wrong_namespaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                        recipe: pilot.Recipe) -> None:
    from offload.dspark.d2.train import full_stage as S, data, pilot_sample
    from offload.dspark.d2.train.pilot_stage import Settings
    name = 'cpu-p2-'+recipe
    path = tmp_path/'pilot-output'/name/'receipt.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'plan': {'frozen_identity': 'base', 'recipe': recipe,
                                       'run': asdict(pilot.pilot_run(recipe))}}))
    settings = Settings(source=tmp_path/'source', commit='a'*40, snapshot=tmp_path/'snapshot'/(name+'-full'),
        output=tmp_path/'output'/(name+'-full'), assembly=tmp_path/'assembly', parity=tmp_path/'parity',
        g1=tmp_path/'g1', k2_dev=tmp_path/'dev', k2_legal=tmp_path/'legal', ckpt=tmp_path/'native', python=tmp_path/'python')
    config = S.Full(settings, tmp_path/'pilot-snapshot'/name/'repo', path, tmp_path/'general')
    config.k2_general.write_text('CPU K2 fixture')
    monkeypatch.setattr(S, 'check_inputs', lambda *a: {})
    monkeypatch.setattr(S, 'require_full_pilot', lambda *a: sha256(path))
    monkeypatch.setattr(data, 'D2Blocks', lambda *a: object())
    monkeypatch.setattr(pilot_sample, 'complete_rounds', lambda *a: [object()])
    result = S.admission(config)
    assert result['recipe'] == recipe and result['pilot'] == sha256(path)
    for bad in (replace(config, settings=replace(settings, output=tmp_path/'output'/'old-full')),
                replace(config, settings=replace(settings, snapshot=tmp_path/'snapshot'/'wrong')),
                replace(config, pilot_source=tmp_path/'pilot-snapshot'/'wrong'/'repo')):
        with pytest.raises(ValueError, match='namespace'):
            S.admission(bad)


@pytest.mark.parametrize('flag', ['--k2-dev', '--k2-legal'])
def test_partial_k2_pair_refuses_before_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str) -> None:
    import sys
    from offload.dspark.d2.train import loop, full_eval
    monkeypatch.setattr(loop, 'load', lambda *a, **k: pytest.fail('partial K2 reached model load'))
    monkeypatch.setattr(sys, 'argv', ['loop', '--output', str(tmp_path/'out'), flag, str(tmp_path/'k2')])
    with pytest.raises(SystemExit) as error:
        loop.main()
    assert error.value.code == 2
    args = ['full_eval', 'score']
    for key in ('manifest', 'adapter', 'ckpt', 'pilot-receipt', 'pilot-source', 'parity-receipt', 'g1-receipt', 'k2', 'out'):
        args.extend(['--'+key, str(tmp_path/key)])
    monkeypatch.setattr(full_eval, 'evaluate', lambda *a: pytest.fail('partial K2 reached evaluator'))
    monkeypatch.setattr(sys, 'argv', [*args, '--split', 'dev', flag, str(tmp_path/'extra-k2')])
    with pytest.raises(SystemExit) as error:
        full_eval.main()
    assert error.value.code == 2


@pytest.mark.parametrize('recipe', pilot.RECIPES)
def test_completed_g2_checkpoint_binds_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             recipe: pilot.Recipe) -> None:
    from offload.dspark.d2.train import full_eval as F
    from offload.dspark.d2.train.fast_policy import BACKEND, numeric_sources
    state: dict[str, object] = {'full_recipe': recipe, 'run': asdict(A.full_run(recipe)), 'dataset': 'data',
        'step': 3, 'fast_training': True, 'fast_pilot_sha256': 'pilot', 'fast_backend': BACKEND,
        'fast_numeric_sources': numeric_sources(), 'reference_math': False, 'curve': [{'loss': 1.}]*3,
        'trainable': {'synthetic': torch.ones(1)}}
    class Data:
        fingerprint = 'data'
        def __len__(self) -> int:
            return 128
    monkeypatch.setattr(F, 'D2Blocks', lambda *a: Data())
    monkeypatch.setattr(F, 'read_checkpoint', lambda *a: (state, 'checkpoint-hash'))
    args = (tmp_path/'adapter', tmp_path/'manifest', 'pilot', recipe)
    assert F.completed(*args) == 'checkpoint-hash'
    state['full_recipe'] = 'ace' if recipe != 'ace' else 'lr'
    with pytest.raises(ValueError, match='completed three-epoch'):
        F.completed(*args)
    del state['full_recipe']
    with pytest.raises(ValueError, match='completed three-epoch'):
        F.completed(*args)


@pytest.mark.parametrize('defect', ['base', 'initialization'])
def test_full_cli_refuses_changed_fresh_base_before_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            defect: str) -> None:
    import sys
    from offload.dspark.d2.train import loop
    m = model()
    spec = {'frozen_identity': m.frozen_identity, 'initialization_sha256': pilot.initialization(m),
            'recipe': 'ace', 'run': asdict(pilot.pilot_run('ace'))}
    spec['frozen_identity' if defect == 'base' else 'initialization_sha256'] = 'changed'
    receipt, parity = tmp_path/'receipt', tmp_path/'parity'
    receipt.write_text(json.dumps({'plan': spec}))
    parity.write_text(json.dumps({'pass': True, 'samples': 50, 'reference_math': True,
                                 'frozen_identity': m.frozen_identity}))
    monkeypatch.setattr(pilot, 'validate_live_g1', lambda *a: None)
    monkeypatch.setattr(loop, 'require_full_pilot', lambda *a: sha256(receipt))
    monkeypatch.setattr(loop, 'load', lambda *a, **k: m)
    monkeypatch.setattr(loop, 'train', lambda *a, **k: pytest.fail('changed base reached training'))
    monkeypatch.setattr(sys, 'argv', ['loop', '--manifest', str(tmp_path/'manifest'), '--ckpt', 'native',
        '--output', str(tmp_path/'output'), '--fast-training', '--pilot-receipt', str(receipt),
        '--pilot-source', str(tmp_path/'source'), '--parity-receipt', str(parity), '--g1-receipt', str(tmp_path/'g1'),
        '--batch-size', '128', '--chunk', '8192'])
    with pytest.raises(ValueError, match='full FAST base or admitted receipt changed|fresh initialization differs'):
        loop.main()
    assert not (tmp_path/'output').exists()
