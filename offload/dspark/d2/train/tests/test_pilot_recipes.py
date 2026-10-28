from __future__ import annotations

from dataclasses import asdict, replace
import importlib.util
import os
from pathlib import Path
import subprocess
from types import ModuleType
from typing import Literal, cast

import pytest
import torch

from offload.dspark.d2.train import loop, pilot
from offload.dspark.d2.train.data import SyntheticBlocks
from offload.dspark.d2.train.fast_loss import bucket_loss
from offload.dspark.d2.train.loss import POSITION_WEIGHTS, distillation_loss, teacher_argmax
from offload.dspark.d2.train.pilot_units import work_units
from offload.dspark.d2.train.tests.test_loop import model
from offload.dspark.d2.train.tests.test_loss import Logits
from offload.dspark.d2.train.tests.test_pilot_sample import sample_fixture


@pytest.mark.parametrize('recipe', pilot.RECIPES)
def test_recipe_changes_only_registered_setting(recipe: pilot.Recipe) -> None:
    base = pilot.pilot_run()
    run = pilot.pilot_run(recipe)
    expected = replace(base, divergence=recipe) if recipe in ('tvd', 'ace') else base
    if recipe == 'lr':
        expected = replace(base, lr=5e-5, warmup_fraction=.1)
    assert run == expected
    schedules = [pilot.schedule(23000, list(pilot.CHECKPOINTS), 'REF', r) for r in ('kl', recipe)]
    assert schedules[0][0].batches == schedules[1][0].batches
    assert schedules[0][1:] == schedules[1][1:]


@pytest.mark.parametrize('recipe', pilot.RECIPES)
def test_recipe_bound_in_plan_and_matched_units(tmp_path: Path, recipe: pilot.Recipe) -> None:
    plan, _, _ = sample_fixture(tmp_path)
    spec = pilot.read_spec(plan)
    spec['recipe'] = recipe
    spec['run'] = dict(asdict(pilot.pilot_run(recipe)))
    # A metadata-only counter population is enough to verify recipe-bound units.
    class Counts:
        def __len__(self) -> int:
            return 23000
        def supervised_tokens(self, index: int) -> int:
            return 5
    spec['matched_units'] = pilot.matched_units(Counts(), spec['checkpoints'], recipe)
    pilot.validate_spec(spec)
    other: pilot.Recipe = 'lr' if recipe != 'lr' else 'ace'
    spec['recipe'] = other
    with pytest.raises(ValueError, match='settings mismatch'):
        pilot.validate_spec(spec)


def test_ace_dense_oracle_ties_padding_masks_and_empty_rows() -> None:
    torch.manual_seed(45)
    z = torch.randn(2, 5, 32, requires_grad=True)
    ids = torch.tensor([7, 3, 9, -1]).expand(2, 5, -1).clone()
    logp = torch.tensor([.4, .4, .1, 0.]).log().expand_as(ids).clone()
    present = torch.ones(2, 5, dtype=torch.bool)
    mask = present.clone()
    ids[0, 2] = -1  # even a marked-present empty row is skipped in ACE
    present[1, 1] = False
    mask[1, -1] = False
    hidden = torch.zeros(2, 5, 1)
    prev = torch.zeros(2, 5, dtype=torch.long)
    picks = teacher_argmax(ids, logp, 32)
    assert picks[0, 0] == 3
    loss, per = distillation_loss(Logits(z), hidden, prev, ids, logp, torch.full((2, 5), .1),
                                  prev, present, mask, chunk=7, divergence='ace')
    active = mask & present & (ids >= 0).any(-1)
    expected = -z.log_softmax(-1)[..., 3]
    weights = z.new_tensor(POSITION_WEIGHTS) * active
    want = (expected*weights).sum()/weights.sum()
    torch.testing.assert_close(loss, want, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(per, expected.detach().masked_fill(~active, 0), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.autograd.grad(loss, z, retain_graph=True)[0],
                               torch.autograd.grad(want, z)[0], atol=1e-7, rtol=2e-5)
    with pytest.raises(ValueError, match='no scored positions'):
        distillation_loss(Logits(z), hidden, prev, ids, logp, torch.full((2, 5), .1),
                          prev, present, torch.zeros_like(mask), chunk=7, divergence='ace')


@pytest.mark.parametrize('chunk', [1, 17, 128])
def test_ace_fast_loss_and_all_head_gradients_match_reference(chunk: int) -> None:
    torch.manual_seed(91)
    m = model()
    h = torch.randn(2, 5, m.c.dim, requires_grad=True)
    previous = torch.tensor([[3, 3, 7, 8, 9], [9, 1, 5, 5, 6]])
    top, ids = torch.randn(2, 5, m.c.vocab).softmax(-1).topk(20, -1)
    # Force a tie with reversed token ids; two missing forms and padding.
    top[..., :2] = top[..., :2].mean(-1, keepdim=True)
    present = torch.ones(2, 5, dtype=torch.bool)
    mask = present.clone()
    mask[0, -1] = False
    present[1, 2] = False
    ids[1, 2] = -1
    top[1, 2] = 0
    ids[0, 1, -1] = -1
    top[0, 1, -1] = 0
    tail = 1-top.sum(-1)
    args = (previous, ids, top.log(), tail, previous, present, mask)
    expected, per = distillation_loss(m, h, *args, chunk=chunk, divergence='ace')
    params = (h, m.last('markov_head.embed.weight'), m.last('markov_head.head.weight'))
    gradients = torch.autograd.grad(expected, params)
    got, fast_per = bucket_loss(h, m.t['head.weight'], params[1], params[2], *args, chunk, 'ace')
    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fast_per, per, rtol=1e-6, atol=1e-6)
    for a, b in zip(torch.autograd.grad(got, params), gradients):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-7)


def test_lr_warmup_applied_rates_and_exact_resume(tmp_path: Path) -> None:
    torch.set_num_threads(2)
    run = replace(pilot.pilot_run('lr'), batch_size=1, chunk=32)
    data = SyntheticBlocks(model().c, 20)
    plan = loop.BatchPlan(tuple((i,) for i in range(20)), (20,), 'warmup')
    class Counts:
        def __len__(self) -> int:
            return 20
        def supervised_tokens(self, index: int) -> int:
            return 5
    expected = work_units(Counts(), plan, run)['learning_rates']
    assert expected[:3] == [2.5e-5, 5e-5, 5e-5]
    assert loop.lr_multiplier(run, 20, 20) == 0
    full, partial = tmp_path/'full.pt', tmp_path/'part.pt'
    loop.train(model(), data, run, full, plan=plan)
    loop.train(model(), data, run, partial, plan=plan, max_steps=1)
    loop.train(model(), data, run, partial, plan=plan, resume=True)
    a, b = [torch.load(p, weights_only=False) for p in (full, partial)]
    assert [r['applied_lr'] for r in a['curve']] == expected
    assert a['scheduler'] == b['scheduler']
    for n in a['trainable']:
        assert torch.equal(a['trainable'][n], b['trainable'][n])


def historical(tmp_path: Path, name: str) -> ModuleType:
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    repo = tmp_path/'baseline-git'
    if not repo.exists():
        subprocess.run(['git', 'clone', '--shared', '--no-checkout', str(Path(__file__).resolve().parents[5]),
                        str(repo)], cwd=tmp_path, env=env, check=True, capture_output=True)
    source = subprocess.run(['git', '-C', str(repo), 'show', 'b6f79e9:offload/dspark/d2/train/'+name+'.py'],
                            cwd=tmp_path, env=env, check=True, capture_output=True).stdout
    path = tmp_path/(name+'.py')
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location('offload.dspark.d2.train._baseline_'+name, path)
    assert spec is not None and spec.loader is not None
    import sys
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_kl_bit_identical_to_unchanged_b6f79e9_loss_and_training(tmp_path: Path) -> None:
    torch.set_num_threads(2)
    old_loss, old_loop = historical(tmp_path, 'loss'), historical(tmp_path, 'loop')
    setattr(old_loop, 'distillation_loss', old_loss.distillation_loss)
    data = SyntheticBlocks(model().c, 3)
    run = replace(pilot.pilot_run('kl'), batch_size=1, chunk=32)
    old_run = old_loop.Run(**{k: v for k, v in asdict(run).items() if k != 'warmup_fraction'})
    old_loop.train(model(), data, old_run, tmp_path/'old.pt')
    loop.train(model(), data, run, tmp_path/'new.pt')
    old, new = [torch.load(tmp_path/n, weights_only=False) for n in ('old.pt', 'new.pt')]
    for field in ('loss', 'applied_lr', 'lr'):
        assert [r[field] for r in old['curve']] == [r[field] for r in new['curve']]
    for n in old['trainable']:
        assert torch.equal(old['trainable'][n], new['trainable'][n])
    for i, state in old['optimizer']['state'].items():
        for n, tensor in state.items():
            assert torch.equal(tensor, new['optimizer']['state'][i][n])
    assert cast(dict[str, object], old['scheduler']) == new['scheduler']


def test_every_arm_receipt_rejects_foreign_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from offload.dspark.d2.train import pilot_eval
    from offload.dspark.d2.train.tests.test_pilot_policy import receipt_fixture
    import json
    receipt_fixture(tmp_path, monkeypatch)
    spec = pilot.read_spec(tmp_path/'plan.json', sampled=True)
    for arm in pilot.ARMS:
        path = tmp_path/arm/'arm.json'
        original = path.read_bytes()
        value = json.loads(original)
        value['recipe'] = 'tvd'
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match='arm receipt mismatch'):
            pilot_eval.read_arm(tmp_path, arm, spec, tmp_path/'plan.json', 23000)
        path.write_bytes(original)


def test_fast_kl_bit_identical_to_unchanged_b6f79e9(tmp_path: Path) -> None:
    old = historical(tmp_path, 'fast_loss')
    torch.manual_seed(91)
    m = model()
    h = torch.randn(1, 5, m.c.dim, requires_grad=True)
    block = SyntheticBlocks(m.c, 1)[0]['targets']
    divergence: Literal['kl'] = 'kl'
    args = (h, m.t['head.weight'], m.last('markov_head.embed.weight'),
            m.last('markov_head.head.weight'), block['previous'], block['top_ids'],
            block['top_logp'], block['tail_mass'], block['labels'], block['present'],
            block['position_mask'], 17, divergence)
    before, old_per = old.bucket_loss(*args)
    after, new_per = bucket_loss(*args)
    assert torch.equal(before, after) and torch.equal(old_per, new_per)
    params = (h, m.last('markov_head.embed.weight'), m.last('markov_head.head.weight'))
    for a, b in zip(torch.autograd.grad(before, params), torch.autograd.grad(after, params)):
        assert torch.equal(a, b)


def test_new_plan_requires_explicit_recipe_in_every_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from offload.dspark.d2.train import pilot_eval
    plan, _, _ = sample_fixture(tmp_path)
    spec = pilot.read_spec(plan)
    spec['recipe'] = 'kl'
    class CheckpointReached(Exception):
        pass
    def checkpoint(path: Path) -> None:
        raise CheckpointReached()
    monkeypatch.setattr(pilot_eval, 'read_checkpoint', checkpoint)
    for arm in pilot.ARMS:
        batch, _, _ = pilot.schedule(23000, spec['checkpoints'], arm, 'kl')
        value: dict[str, object] = {
            'schema': 'dsv41.d2.pilot-arm.v1', 'arm': arm, 'complete': True,
            'recipe': 'kl', 'plan_sha256': 'plan-hash', 'synthetic': spec['synthetic'],
            'batch_plan_identity': pilot.digest({'plan': spec, 'arm': arm, 'schedule': batch.identity}),
            'step_count': len(batch.batches), 'units': spec['matched_units'][arm],
            'checkpoints': {str(n): {'path': '/unread', 'step': step, 'sha256': 'unused'}
                            for n, step in zip(spec['checkpoints'], batch.checkpoints)}}
        path = tmp_path/arm/'arm.json'
        path.parent.mkdir()
        path.write_text(json.dumps(value))
        with pytest.raises(CheckpointReached):
            pilot_eval.read_arm(tmp_path, arm, spec, 'plan-hash', 23000)
        del value['recipe']
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match='arm receipt mismatch'):
            pilot_eval.read_arm(tmp_path, arm, spec, 'plan-hash', 23000)
