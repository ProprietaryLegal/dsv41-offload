"""Synthetic CPU admissions; no coordinator execution, GPU or network."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from offload.dspark.d2.train import full_stage as S
from offload.dspark.d2.train.full_eval import Pair, summarize, verdict
from offload.dspark.d2.train.evaluate import clustered, Score
from offload.dspark.d2.train.pilot_eval import Metrics
from offload.dspark.d2.train.pilot_stage import Settings
from offload.dspark.d2.train.safety import GPU, SOLO_B


def config(tmp: Path) -> S.Full:
    base = Settings(source=tmp/'source', commit='a'*40, snapshot=tmp/'snapshot',
        output=tmp/'output', assembly=tmp/'assembly', parity=tmp/'parity.json',
        g1=tmp/'dev.json', k2_dev=tmp/'dev.k2', k2_legal=tmp/'legal.k2',
        ckpt=tmp/'native', python=tmp/'venv/bin/python')
    return S.Full(base, tmp/'pilot/repo', tmp/'pilot-output/receipt.json', tmp/'general.k2')


def test_dry_has_no_reads_execution_or_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('dry default accessed inputs or execution')
    monkeypatch.setattr(Path, 'read_text', forbidden)
    monkeypatch.setattr(subprocess, 'run', forbidden)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(sys, 'argv', ['full_stage', '--commit', 'a'*40, '--g1', '/absent/dev',
        '--k2-dev', '/absent/dev.k2', '--k2-legal', '/absent/legal.k2', '--k2-general', '/absent/general.k2'])
    S.main()
    result = json.loads(capsys.readouterr().out)
    assert result['dry'] is True
    assert result['order'] == list(S.PHASES)
    assert list(tmp_path.iterdir()) == []


def test_render_full_sets_leases_and_resume(tmp_path: Path) -> None:
    s = config(tmp_path/'spaces $literal')
    jobs = S.commands(s)
    train = jobs['full-train'][0][1]
    assert '--fast-training' in train and '--epochs 3' in train
    assert '--pilot-source' in train and '--pilot-receipt' in train
    assert GPU in train and SOLO_B not in train
    for _, body in jobs['full-g2']:
        assert 'full_eval score' in body and '--sample' not in body and '--limit' not in body
        assert '--adapter' in body and 'full-fast.pt' in body
    assert '--split dev' in jobs['full-g2'][0][1]
    assert '--split heldout_legal' in jobs['full-g2'][1][1]
    assert '--split heldout_general' in jobs['full-g2'][2][1]
    for phase in S.commands(s, resume=True).values():
        for _, body in phase:
            subprocess.run(['bash', '-n'], input=body, text=True, check=True)
            if 'full_eval merge' not in body:
                assert '--resume' in body
            assert 'ssh ' not in body and 'nvidia-smi' not in body


@pytest.mark.parametrize('target', ['pilot_source', 'pilot_receipt'])
def test_pilot_namespace_protected(tmp_path: Path, target: str) -> None:
    s = config(tmp_path)
    bad = replace(s.settings, output=getattr(s, target).parent/'child')
    with pytest.raises(ValueError, match='overlaps pilot'):
        S.commands(replace(s, settings=bad))


def metrics(request: str, values: list[int], kl: float) -> Metrics:
    return {'requests': 1, 'rounds': len(values), 'population_sha256': request, 'loss': kl,
        'kl_by_position': [kl]*5, 'kl_median_by_position': [kl]*5,
        'accepted_length': sum(values)/len(values),
        'request_totals': [{'request_sha256': request, 'family': request, 'task_label': 'synthetic',
            'rounds': len(values), 'accepted_sum': float(sum(values))}]}


def test_request_sufficient_totals_equal_original_g2_bootstrap() -> None:
    pairs: list[Pair] = []
    rows: list[Score] = []
    for request, actual, k2 in [('a', [2, 3], [1, 2]), ('b', [1, 4, 4], [1, 3, 3])]:
        pairs.append({'baseline': metrics(request, k2, 1.), 'candidate': metrics(request, actual, .5), 'k2_sum': sum(k2)})
        rows += [{'request': request, 'candidate': a, 'k2': b, 'kl_candidate': [.5]*5, 'kl_baseline': [1.]*5}
                 for a, b in zip(actual, k2)]
    new, old = summarize(pairs), clustered(rows)
    for key in old:
        assert new[key] == pytest.approx(old[key])


def parts() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for split, n in [('dev', 30), ('heldout_legal', 24), ('heldout_general', 12)]:
        m = summarize([{'baseline': metrics(str(i), [1, 1], 1.),
            'candidate': metrics(str(i), [2, 2], .5), 'k2_sum': 2} for i in range(n)])
        result.append({'schema': 'dsv41.d2.full-g2-part.v1', 'split': split, 'manifest_sha256': 'manifest',
            'adapter_sha256': 'adapter', 'recipe': 'kl', 'pilot_sha256': 'pilot', 'source_sha256': {}, 'rounds': m['rounds'], 'metrics': m})
    return result


def test_g2_gates_bind_legal_and_general() -> None:
    p = parts()
    assert verdict(p)['verdict'] == 'PASS'
    for key, value in [('gain', .149), ('ci95_request_cluster', [0., 1.]),
        ('worst_request_gain', -.201), ('kl_candidate', [1.]*5), ('native_gain', -.011)]:
        bad = deepcopy(p)
        bad[1]['metrics'][key] = value  # type: ignore[index]
        assert verdict(bad)['verdict'] == 'FAIL'
    # Amendment 4 keeps the D2-TRAIN G2 general gates; general has no native-regression floor (amendment 6 is dev).
    for key, value, passes in [('gain', -.03, True), ('gain', -.031, False), ('worst_request_gain', -.201, False),
        ('kl_candidate', [1.]*5, False), ('requests', 9, False), ('requests', 10, True), ('native_gain', -1., True)]:
        general = deepcopy(p)
        general[2]['metrics'][key] = value  # type: ignore[index]
        assert (verdict(general)['verdict'] == 'PASS') == passes


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'identity', 'nan', 'schema'])
def test_g2_malformed_parts_refuse(defect: str) -> None:
    p = parts()
    if defect == 'missing':
        p.pop()
    elif defect == 'duplicate':
        p[1]['split'] = 'dev'
    elif defect == 'identity':
        p[1]['adapter_sha256'] = 'other'
    elif defect == 'schema':
        p[1]['schema'] = 'pilot'
    else:
        p[0]['metrics']['gain'] = float('nan')  # type: ignore[index]
    with pytest.raises(ValueError):
        verdict(p)


def test_reference_full_population_request_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real tiny reference arithmetic; ONLY production admission/load replaced."""
    import argparse
    from typing import cast
    from offload.dspark.d2.train import full_eval as F, pilot_eval, replay, safety
    from offload.dspark.d2.train.data import Block, D2Blocks, SyntheticBlocks, sha256
    from offload.dspark.d2.train.tests.test_loop import model
    from offload.dspark.d2.train.pilot_sample import RoundKey
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('CPU rehearsal touched CUDA'))

    class Held(SyntheticBlocks):
        request_families = {'r0': 'f0', 'r1': 'f1'}
        request_tasks = {'r0': 'synthetic', 'r1': 'synthetic'}
        def __getitem__(self, index: int) -> Block:
            b = super().__getitem__(index)
            b['request'] = 'r0' if index < 4 else 'r1'
            return b

    from offload.dspark.d2.train.model import Drafter
    def tiny() -> Drafter:
        m = model()
        m.reference_math = True
        return m

    m = tiny()
    data = Held(m.c, 8)
    keys: list[RoundKey] = [{'request': 'r0', 'P': 12}, {'request': 'r1', 'P': 18}]
    rounds = [(0, keys[0]), (6, keys[1])]
    monkeypatch.setattr(F, 'D2Blocks', lambda *_: cast(D2Blocks, data))
    monkeypatch.setattr(F, 'complete_rounds', lambda *_: rounds)
    monkeypatch.setattr(pilot_eval, 'complete_rounds', lambda *_: rounds)
    monkeypatch.setattr(safety, 'assert_gpu_lease', lambda: None)
    monkeypatch.setattr(F, 'require_full_pilot', lambda *args: 'test-only')
    monkeypatch.setattr(F, 'pilot_recipe', lambda *a: 'kl')
    monkeypatch.setattr(F, 'completed', lambda adapter, *_: sha256(adapter))
    monkeypatch.setattr(F, 'load', lambda *args, **kwargs: tiny())
    monkeypatch.setattr(replay, 'g1_drafts', lambda *_: {
        k['request']: {k['P']: data[i]['targets']['labels'][0].tolist()} for i, k in rounds})
    from dataclasses import asdict
    state = {'frozen_identity': m.frozen_identity, 'model_config': asdict(m.c), 'trainable': {n: p.detach().clone() for n, p in m.named_parameters()}}
    adapter = tmp_path/'adapter.pt'
    torch.save(state, adapter)
    manifest, k2, pilot = [tmp_path/n for n in ('manifest', 'k2', 'pilot')]
    manifest.write_text('{}')
    k2.write_text('synthetic test only')
    pilot.write_text(json.dumps({'plan': {'frozen_identity': m.frozen_identity}}))
    a = argparse.Namespace(pilot_receipt=pilot, manifest=manifest, parity_receipt=tmp_path/'parity',
        g1_receipt=tmp_path/'g1', pilot_source=tmp_path/'old', adapter=adapter, split='dev',
        k2=k2, out=tmp_path/'dev.json', resume=False, ckpt=tmp_path/'native')
    real = pilot_eval.score
    calls = 0
    def interrupted(*args: object, **kwargs: object) -> Metrics:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('synthetic interruption after first request')
        return real(*args, **kwargs)  # type: ignore[arg-type]
    monkeypatch.setattr(F, 'score', interrupted)
    with pytest.raises(RuntimeError, match='synthetic interruption'):
        F.evaluate(a)
    assert len(list((tmp_path/'dev.requests').glob('*.json'))) == 2  # contract + completed request
    monkeypatch.setattr(F, 'score', real)
    a.resume = True
    F.evaluate(a)
    result = json.loads(a.out.read_text())
    assert result['rounds'] == result['metrics']['rounds'] == 2
    assert result['metrics']['requests'] == 2
    assert result['metrics']['k2'] == 5.
    with pytest.raises(FileExistsError, match='already complete'):
        F.evaluate(a)


def test_full_initialization_seed_precedes_model_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    from offload.dspark.d2.train import full_train
    from offload.dspark.d2.train.pilot import SEED
    samples: list[torch.Tensor] = []
    monkeypatch.setattr(full_train, 'loop_main', lambda: samples.append(torch.rand(4)))
    full_train.main()
    torch.manual_seed(999)
    full_train.main()
    torch.manual_seed(SEED)
    expected = torch.rand(4)
    for sample in samples:
        torch.testing.assert_close(sample, expected, rtol=0, atol=0)


@pytest.mark.parametrize('defect', ['incomplete', 'wrong_run', 'pilot', 'backend', 'numerics', 'nonfinite'])
def test_full_g2_refuses_bad_training_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    from dataclasses import asdict
    from offload.dspark.d2.train import full_eval as F
    from offload.dspark.d2.train.fast_policy import BACKEND, numeric_sources
    from offload.dspark.d2.train.loop import Run
    run = Run(epochs=3, batch_size=128, chunk=8192, rank=32)
    state: dict[str, object] = {'full_recipe': 'kl', 'run': asdict(run), 'dataset': 'data', 'step': 3,
        'fast_training': True, 'fast_pilot_sha256': 'pilot', 'fast_backend': BACKEND,
        'fast_numeric_sources': numeric_sources(), 'reference_math': False,
        'curve': [{'loss': 1.}]*3, 'trainable': {'synthetic': torch.ones(1)}}
    if defect == 'incomplete':
        state['step'] = 2
    elif defect == 'wrong_run':
        state['run'] = asdict(Run(epochs=1))
    elif defect == 'pilot':
        state['fast_pilot_sha256'] = 'other'
    elif defect == 'backend':
        state['fast_backend'] = 'unqualified'
    elif defect == 'numerics':
        state['fast_numeric_sources'] = {}
    else:
        state['trainable'] = {'synthetic': torch.tensor([float('nan')])}
    class Data:
        fingerprint = 'data'
        def __len__(self) -> int:
            return 128
    monkeypatch.setattr(F, 'D2Blocks', lambda *_: Data())
    monkeypatch.setattr(F, 'read_checkpoint', lambda *_: (state, 'hash'))
    with pytest.raises(ValueError, match='completed three-epoch'):
        F.completed(tmp_path/'checkpoint', tmp_path/'manifest', 'pilot')


def test_full_receipt_atomic_exclusive_and_no_partial_json(tmp_path: Path) -> None:
    from offload.dspark.d2.train.full_eval import write_json
    path = tmp_path/'receipt.json'
    write_json(path, {'complete': True})
    with pytest.raises(FileExistsError):
        write_json(path, {'complete': False})
    assert json.loads(path.read_text()) == {'complete': True}
    with pytest.raises(ValueError):
        write_json(tmp_path/'nan.json', {'bad': float('nan')})
    assert not (tmp_path/'nan.json').exists()
    assert list(tmp_path.iterdir()) == [path]
