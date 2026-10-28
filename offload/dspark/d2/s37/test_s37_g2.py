"""S37 item 4: full_eval / full G2 on a tiny REAL-arithmetic fixture (tiny Drafter, reference math, real score()).

Only production admission (pilot child, completed checkpoint), lease, model load, dataset and K2 file reading are
replaced; every metric below is computed by the real pilot_eval.score / full_eval.summarize / evaluate.clustered.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Callable, cast

import pytest
import torch

from offload.dspark.d2.train import full_eval as F, pilot, pilot_eval, replay, safety
from offload.dspark.d2.train.data import Block, D2Blocks, SyntheticBlocks, sha256
from offload.dspark.d2.train.evaluate import Score, clustered, match_length
from offload.dspark.d2.train.model import Drafter
from offload.dspark.d2.train.pilot_sample import RoundKey, key_hash
from offload.dspark.d2.train.tests.test_loop import model

REQUESTS = {'r0': [0, 2], 'r1': [4, 5, 7], 'r2': [9, 11]}   # block indices of each request's complete K2 rounds


class Held(SyntheticBlocks):
    request_families = {'r0': 'f0', 'r1': 'f1', 'r2': 'f1'}
    request_tasks = {'r0': 'synthetic', 'r1': 'synthetic', 'r2': 'synthetic'}

    def __getitem__(self, index: int) -> Block:
        b = super().__getitem__(index)
        b['request'] = 'r0' if index < 4 else 'r1' if index < 8 else 'r2'
        return b


def tiny() -> Drafter:
    m = model()
    m.reference_math = True
    return m.eval()


def rounds() -> list[tuple[int, RoundKey]]:
    return [(i, {'request': r, 'P': 12+i}) for r, idx in REQUESTS.items() for i in idx]


def k2_drafts(data: Held) -> dict[str, dict[int, list[int]]]:
    """K2 proposals = the truth with one wrong token at position i % 5 (round i accepts i % 5 under K2)."""
    out: dict[str, dict[int, list[int]]] = {}
    for i, key in rounds():
        labels = data[i]['targets']['labels'][0].tolist()
        labels[i % 5] = (labels[i % 5]+1) % data.c.vocab
        out.setdefault(key['request'], {})[key['P']] = labels
    return out


def candidate_state(m: Drafter, scale: float = .05) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(5)
    state = {}
    for n, p in m.named_parameters():
        t = p.detach().clone()
        if n.endswith('.b') or 'markov_head' in n:
            t += scale*torch.randn(t.shape, generator=g, dtype=t.dtype)
        state[n] = t
    return state


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('CPU fixture touched CUDA'))
    m = tiny()
    data = Held(m.c, 12)
    monkeypatch.setattr(F, 'D2Blocks', lambda *_: cast(D2Blocks, data))
    monkeypatch.setattr(F, 'complete_rounds', lambda *_: rounds())
    monkeypatch.setattr(pilot_eval, 'complete_rounds', lambda *_: rounds())
    monkeypatch.setattr(safety, 'assert_gpu_lease', lambda: None)
    pilot_hash = {'value': 'pilot-receipt-hash'}
    monkeypatch.setattr(F, 'require_full_pilot', lambda *a: pilot_hash['value'])
    monkeypatch.setattr(F, 'pilot_recipe', lambda *a: 'kl')
    monkeypatch.setattr(F, 'completed', lambda adapter, *_: sha256(adapter))
    monkeypatch.setattr(F, 'load', lambda *a, **k: tiny())
    monkeypatch.setattr(replay, 'g1_drafts', lambda *_: k2_drafts(data))
    state = {'frozen_identity': m.frozen_identity, 'model_config': asdict(m.c), 'trainable': candidate_state(m)}
    adapter = tmp_path/'full-fast.pt'
    torch.save(state, adapter)
    for name, body in (('manifest', '{}'), ('k2', 'k2 v1'), ('pilot', json.dumps({'plan': {
            'frozen_identity': m.frozen_identity,
            'sample': {'keys': {'dev': [{'request': 'r0', 'P': 12}]}}}}))):  # an amendment-4-style sample, unused
        (tmp_path/name).write_text(body)
    a = argparse.Namespace(pilot_receipt=tmp_path/'pilot', manifest=tmp_path/'manifest', parity_receipt=tmp_path/'p',
                           g1_receipt=tmp_path/'g1', pilot_source=tmp_path/'old', adapter=adapter, split='dev',
                           k2=tmp_path/'k2', out=tmp_path/'dev.json', resume=False, ckpt=tmp_path/'native')
    return {'a': a, 'data': data, 'state': state, 'pilot_hash': pilot_hash, 'tmp': tmp_path}


def run(fx: dict[str, Any]) -> dict[str, Any]:
    F.evaluate(fx['a'])
    return dict(json.loads(fx['a'].out.read_text()))


def independent(fx: dict[str, Any]) -> dict[str, Any]:
    """The pilot evaluator over the whole population, plus per-round rows for the ORIGINAL G2 bootstrap."""
    data, keys = fx['data'], [k for _, k in rounds()]
    base, cand = tiny(), tiny()
    replay.apply_trainables(cand, fx['state']['trainable'])
    whole = {'baseline': pilot_eval.score(base, data, Path('k2'), keys),
             'candidate': pilot_eval.score(cand, data, Path('k2'), keys)}
    drafts = k2_drafts(data)
    rows: list[Score] = []
    for i, key in rounds():
        one_c = pilot_eval.score(cand, data, Path('k2'), [key])
        one_b = pilot_eval.score(base, data, Path('k2'), [key])
        k2 = match_length(drafts[key['request']][key['P']], data[i]['targets']['labels'][0].tolist())
        assert k2 == i % 5
        rows.append({'request': key['request'],  # raw id, as score_dumps rows; same sort order as full_eval
                      'candidate': one_c['accepted_length'], 'k2': float(k2),
                     'kl_candidate': list(one_c['kl_by_position']), 'kl_baseline': list(one_b['kl_by_position'])})
    return {'whole': whole, 'original_g2': clustered(rows), 'rows': rows}


def test_full_eval_reproduces_pilot_evaluator_and_original_g2_bootstrap(fixture: dict[str, Any]) -> None:
    got = run(fixture)
    m = got['metrics']
    ref = independent(fixture)
    whole, orig = ref['whole'], ref['original_g2']
    # Full population, independent of the candidate and of any pilot sample.
    assert got['rounds'] == m['rounds'] == whole['candidate']['rounds'] == 7 and m['requests'] == 3
    assert got['population_sha256'] == key_hash([k for _, k in rounds()])
    # E[A] exactly as the pilot evaluator computes it on the same population.
    assert m['candidate'] == whole['candidate']['accepted_length']
    assert math.isclose(m['native_gain'], whole['candidate']['accepted_length'] - whole['baseline']['accepted_length'],
                        abs_tol=1e-15)
    for i in range(5):
        assert math.isclose(m['kl_candidate'][i], whole['candidate']['kl_by_position'][i], rel_tol=1e-12)
        assert math.isclose(m['kl_baseline'][i], whole['baseline']['kl_by_position'][i], rel_tol=1e-12)
    # Gain/CI/worst-request bit-identical to evaluate.clustered (the original G2) on per-round rows.
    for key in ('rounds', 'requests', 'candidate', 'k2', 'gain', 'ci95_request_cluster', 'worst_request_gain'):
        assert m[key] == orig[key], key
    for key in ('kl_candidate', 'kl_baseline'):
        assert all(math.isclose(x, y, rel_tol=1e-12) for x, y in zip(m[key], orig[key]))
    # The fixture is non-degenerate: candidate differs from baseline, K2 differs from both.
    assert whole['candidate']['kl_by_position'] != whole['baseline']['kl_by_position']
    assert m['k2'] == sum(i % 5 for _, idx in REQUESTS.items() for i in idx)/7


def test_population_never_depends_on_the_candidate(fixture: dict[str, Any], tmp_path: Path) -> None:
    first = run(fixture)
    other = tiny()
    state = dict(fixture['state'], trainable=candidate_state(other, scale=2.))
    torch.save(state, fixture['a'].adapter)
    fixture['a'].out = tmp_path/'other.json'
    second = run(fixture)
    assert first['population_sha256'] == second['population_sha256'] and first['rounds'] == second['rounds']
    assert first['metrics']['candidate'] != second['metrics']['candidate'] or \
        first['metrics']['kl_candidate'] != second['metrics']['kl_candidate']


def test_interrupted_request_is_redone_and_result_identical(fixture: dict[str, Any], tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    whole = run(fixture)
    fixture['a'].out = tmp_path/'cut.json'
    real, calls = pilot_eval.score, {'n': 0}

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        calls['n'] += 1
        if calls['n'] == 4:  # inside request 2's candidate pass
            raise KeyboardInterrupt('supervised stop')
        return real(*args, **kwargs)
    monkeypatch.setattr(F, 'score', interrupted)
    with pytest.raises(KeyboardInterrupt):
        F.evaluate(fixture['a'])
    journal = tmp_path/'cut.requests'
    assert sorted(p.name for p in journal.iterdir() if not p.name.startswith('.')) == sorted(
        ['contract.json', pilot.digest('r0')+'.json'])
    assert not fixture['a'].out.exists()
    with pytest.raises(FileExistsError, match='use --resume'):
        F.evaluate(fixture['a'])
    monkeypatch.setattr(F, 'score', real)
    fixture['a'].resume = True
    resumed = run(fixture)
    assert resumed['metrics'] == whole['metrics']
    with pytest.raises(FileExistsError, match='already complete'):
        F.evaluate(fixture['a'])


def interrupt_once(fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    real, calls = pilot_eval.score, {'n': 0}

    def stop(*args: Any, **kwargs: Any) -> Any:
        calls['n'] += 1
        if calls['n'] == 3:
            raise KeyboardInterrupt()
        return real(*args, **kwargs)
    monkeypatch.setattr(F, 'score', stop)
    with pytest.raises(KeyboardInterrupt):
        F.evaluate(fixture['a'])
    monkeypatch.setattr(F, 'score', real)
    fixture['a'].resume = True


@pytest.mark.parametrize('change', ['adapter', 'k2', 'pilot', 'source', 'split_population'])
def test_stale_identity_refused_on_resume(fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    interrupt_once(fixture, monkeypatch)
    a = fixture['a']
    if change == 'adapter':
        state = dict(fixture['state'], trainable=candidate_state(tiny(), scale=.5))
        torch.save(state, a.adapter)
    elif change == 'k2':
        a.k2.write_text('k2 v2')
    elif change == 'pilot':
        fixture['pilot_hash']['value'] = 'another-pilot'
    elif change == 'source':
        changed = dict(pilot.sources(), **{'d2/train/pilot_eval.py': '0'*64})
        monkeypatch.setattr(pilot, 'sources', lambda: changed)
    else:
        monkeypatch.setattr(F, 'complete_rounds', lambda *_: rounds()[:-1])
    with pytest.raises(ValueError, match='resume contract changed'):
        F.evaluate(a)


def test_tampered_or_foreign_request_part_refused(fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    interrupt_once(fixture, monkeypatch)
    part = fixture['a'].out.with_suffix('.requests')/(pilot.digest('r0')+'.json')
    pair = json.loads(part.read_text())
    pair['candidate']['population_sha256'] = key_hash([{'request': 'r0', 'P': 12}])
    part.chmod(0o600)
    part.write_text(json.dumps(pair))
    with pytest.raises(ValueError, match='coverage changed'):
        F.evaluate(fixture['a'])


def test_stray_duplicate_part_ignored_and_duplicate_pairs_refused(fixture: dict[str, Any],
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    whole = run(fixture)
    journal = fixture['a'].out.with_suffix('.requests')
    (journal/'copy-of-r0.json').write_text((journal/(pilot.digest('r0')+'.json')).read_text())
    fixture['a'].out.unlink()
    fixture['a'].resume = True
    assert run(fixture)['metrics'] == whole['metrics']
    pair = json.loads((journal/(pilot.digest('r0')+'.json')).read_text())
    with pytest.raises(ValueError, match='duplicate'):
        F.summarize([pair, deepcopy(pair)])


def test_adapter_changed_during_scoring_refused(fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    real, calls = pilot_eval.score, {'n': 0}

    def swap(*args: Any, **kwargs: Any) -> Any:
        calls['n'] += 1
        if calls['n'] == 2:
            with fixture['a'].adapter.open('ab') as f:
                f.write(b'\0')
        return real(*args, **kwargs)
    monkeypatch.setattr(F, 'score', swap)
    with pytest.raises(ValueError, match='adapter changed during scoring'):
        F.evaluate(fixture['a'])
    assert not fixture['a'].out.exists()


def test_nonfinite_score_refused(fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    state = dict(fixture['state'])
    state['trainable'] = {n: (t*float('nan') if 'markov_head' in n else t) for n, t in state['trainable'].items()}
    torch.save(state, fixture['a'].adapter)
    with pytest.raises((FloatingPointError, ValueError)):
        F.evaluate(fixture['a'])
    assert not fixture['a'].out.exists()


# ---------------------------------------------------------------- gate thresholds (D2-TRAIN brief G2 + amendment 6)


def parts(dev: int = 30, legal: int = 24) -> list[dict[str, object]]:
    def metrics(request: str) -> dict[str, Any]:
        return {'requests': 1, 'rounds': 2, 'population_sha256': request, 'loss': 1., 'accepted_length': 1.,
                'kl_by_position': [1.]*5, 'kl_median_by_position': [1.]*5,
                'request_totals': [{'request_sha256': request, 'family': request, 'task_label': 't', 'rounds': 2,
                                    'accepted_sum': 2.}]}
    out: list[dict[str, object]] = []
    for split, n in (('dev', dev), ('heldout_legal', legal), ('heldout_general', 12)):
        pairs: list[F.Pair] = []
        for i in range(n):
            b, c = metrics(f'{split}{i}'), metrics(f'{split}{i}')
            c['accepted_length'], c['request_totals'][0]['accepted_sum'] = 1.5, 3.
            c['kl_by_position'] = [.5]*5
            pairs.append(cast(F.Pair, {'baseline': b, 'candidate': c, 'k2_sum': 2}))
        s = F.summarize(pairs)
        out.append({'schema': 'dsv41.d2.full-g2-part.v1', 'split': split, 'manifest_sha256': 'm', 'adapter_sha256': 'a',
                    'recipe': 'kl', 'pilot_sha256': 'p', 'source_sha256': {}, 'rounds': s['rounds'], 'metrics': s})
    return out


def with_metric(split: int, key: str, value: object) -> list[dict[str, object]]:
    p = parts()
    cast(dict[str, object], p[split]['metrics'])[key] = value
    return p


DEV, LEGAL, GENERAL = 0, 1, 2


def below(x: float) -> float:
    return math.nextafter(x, -math.inf)


@pytest.mark.parametrize('split,key,value,passes', [
    (LEGAL, 'gain', .15, True), (LEGAL, 'gain', below(.15), False),                  # legal >= +0.15
    (DEV, 'gain', -1., True),                                                       # dev gain: no floor in plan
    (LEGAL, 'ci95_request_cluster', [0., 1.], False), (LEGAL, 'ci95_request_cluster', [1e-12, 1.], True),
    (LEGAL, 'worst_request_gain', -.2, True), (LEGAL, 'worst_request_gain', below(-.2), False),
    (DEV, 'worst_request_gain', -.2, True), (DEV, 'worst_request_gain', below(-.2), False),
    (LEGAL, 'native_gain', -.01, True), (LEGAL, 'native_gain', below(-.01), False),
    (DEV, 'native_gain', -.01, True), (DEV, 'native_gain', below(-.01), False),
    (LEGAL, 'kl_candidate', [.5, .5, .5, .5, 1.], False), (DEV, 'kl_candidate', [1., .5, .5, .5, .5], False),
    # Amendment 4 keeps the D2-TRAIN G2 general gates (coordinator fix after S37 observation 4).
    (GENERAL, 'gain', -.03, True), (GENERAL, 'gain', below(-.03), False),
    (GENERAL, 'worst_request_gain', -.2, True), (GENERAL, 'worst_request_gain', below(-.2), False),
    (GENERAL, 'native_gain', -5., True),                                              # no native floor on general
    (GENERAL, 'kl_candidate', [1.]*5, False), (GENERAL, 'kl_candidate', [.5, .5, .5, .5, below(1.)], True),
    (GENERAL, 'requests', 9, False), (GENERAL, 'requests', 10, True)])
def test_gate_boundaries(split: int, key: str, value: object, passes: bool) -> None:
    assert (F.verdict(with_metric(split, key, value))['verdict'] == 'PASS') == passes


@pytest.mark.parametrize('dev,legal,passes', [(30, 24, True), (29, 24, False), (31, 24, False), (30, 20, True),
                                              (30, 19, False)])
def test_population_request_counts(dev: int, legal: int, passes: bool) -> None:
    assert (F.verdict(parts(dev, legal))['verdict'] == 'PASS') == passes


@pytest.mark.parametrize('split', [DEV, LEGAL, GENERAL])
@pytest.mark.parametrize('defect', ['nan', 'inf', 'kl4', 'ci1', 'rounds', 'zero_rounds'])
def test_nonfinite_or_incomplete_summaries_refused(split: int, defect: str) -> None:
    p = parts()
    m = cast(dict[str, Any], p[split]['metrics'])
    if defect == 'nan':
        m['kl_baseline'] = [float('nan')]*5
    elif defect == 'inf':
        m['gain'] = float('inf')
    elif defect == 'kl4':
        m['kl_candidate'] = [.5]*4
    elif defect == 'ci1':
        m['ci95_request_cluster'] = [.1]
    elif defect == 'rounds':
        p[split]['rounds'] = cast(int, p[split]['rounds'])+1
    else:
        m['rounds'] = p[split]['rounds'] = 0
    with pytest.raises(ValueError):
        F.verdict(p)


def test_completed_requires_exact_three_epoch_step_count_on_real_train_length(monkeypatch: pytest.MonkeyPatch,
                                                                            tmp_path: Path) -> None:
    from offload.dspark.d2.train.fast_policy import BACKEND, numeric_sources
    from offload.dspark.d2.train.loop import Run
    steps = 3*math.ceil(488031/128)
    assert steps == 11439
    state: dict[str, object] = {'full_recipe': 'kl', 'run': asdict(Run(epochs=3, batch_size=128, chunk=8192, rank=32)), 'dataset': 'd',
        'step': steps, 'fast_training': True, 'fast_pilot_sha256': 'p', 'fast_backend': BACKEND,
        'fast_numeric_sources': numeric_sources(), 'reference_math': False, 'curve': [{'loss': 1.}]*steps,
        'trainable': {'x': torch.ones(1)}}

    class Data:
        fingerprint = 'd'
        def __len__(self) -> int:
            return 488031
    monkeypatch.setattr(F, 'D2Blocks', lambda *_: Data())
    monkeypatch.setattr(F, 'read_checkpoint', lambda *_: (state, 'h'))
    assert F.completed(tmp_path, tmp_path, 'p') == 'h'
    for bad in ({'step': steps-1}, {'curve': [{'loss': 1.}]*(steps-1)}, {'curve': [{'loss': float('nan')}]*steps}):
        monkeypatch.setattr(F, 'read_checkpoint', lambda *_, b=bad: (dict(state, **b), 'h'))
        with pytest.raises(ValueError):
            F.completed(tmp_path, tmp_path, 'p')



def test_reference_g2_arithmetic_and_gates_equal_a5d527a() -> None:
    import ast
    from offload.dspark.d2.train.tests.snapshot_git import git
    repo = Path(__file__).resolve().parents[4]
    original = ast.parse(git(repo, 'show', 'a5d527a:offload/dspark/d2/train/full_eval.py').stdout.decode())
    current = ast.parse((repo/'offload/dspark/d2/train/full_eval.py').read_text())
    def function(tree: ast.Module, name: str) -> ast.FunctionDef:
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    assert ast.dump(function(original, 'summarize')) == ast.dump(function(current, 'summarize'))
    def gates(tree: ast.Module) -> str:
        return ast.dump(next(n for n in function(tree, 'verdict').body if isinstance(n, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == 'checks' for t in n.targets)))
    assert gates(original) == gates(current)
