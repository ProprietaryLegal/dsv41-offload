"""CPU amendment-7 fixtures; invented counters never establish a real G1 PASS."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from offload.dspark.d2 import g1
from offload.dspark.d2.live_reference import (
    PREFIX, RESTART, build_reference, file_hash, parse_metrics, timestamp, verify_reference,
)
from offload.dspark.d2.train.pilot import validate_live_g1


def metrics(rounds: int = 100, total: int = 250, *, start: float | None = None) -> str:
    labels = 'engine="0",model_name="DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2"'
    created = timestamp(RESTART)+12.32255 if start is None else start
    counts = {'num_drafts': rounds, 'num_draft_tokens': 4*rounds,
              'num_accepted_tokens': sum([rounds*.7, rounds*.5, rounds*.2, rounds*.1, 0])}
    lines = ['# HELP process_start_time_seconds Start time.', 'process_start_time_seconds 1.79289657188e+09']
    for name, value in counts.items():
        lines += [f'{PREFIX}{name}_total{{{labels}}} {value:.1f}',
                  f'{PREFIX}{name}_created{{{labels}}} {created:.8e}']
    for pos, fraction in enumerate((.7, .5, .2, .1, 0)):
        lines += [f'{PREFIX}num_accepted_tokens_per_pos_total{{{labels},position="{pos}"}} {rounds*fraction:.1f}',
                  f'{PREFIX}num_accepted_tokens_per_pos_created{{{labels},position="{pos}"}} {created:.8e}']
    lines += [f'vllm:generation_tokens_total{{{labels}}} {total:.1f}']
    return '\n'.join(lines)+'\n'


def reference_fixture(root: Path, *, gen: int = 248) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    first, final = [root/f'metrics-20261025T{t}Z.prom' for t in ('043300', '050000')]
    first.write_text(metrics(80, 200))
    final.write_text(metrics())
    rate = root/'rate-20261025T050000Z.json'
    rate.write_text(json.dumps({'t': '2026-10-25T05:00:00Z', 'gen_tokens': gen}))
    result = build_reference([first, final], [rate], '2026-10-25T04:59:59Z', '2026-10-25T05:01:00Z')
    path = root/'live-reference.json'
    path.write_text(json.dumps(result))
    return path


def gate_fixture(root: Path, manifest: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    live_path = reference_fixture(root/'live')
    live = verify_reference(live_path)
    path = root/'dev-g1.json'
    path.write_text(json.dumps({'split': 'dev', 'position0_match_rate': .7,
        'g1': g1.g1_verdict(.7, 'dev', live_reference=live),
        'manifest_sha256': file_hash(manifest), 'k2_sha256': 'test-only-not-a-stage-receipt',
        'live_reference_path': str(live_path)}))
    return path


def test_real_format_cumulative_reference_and_hash_binding(tmp_path: Path) -> None:
    path = reference_fixture(tmp_path)
    live = verify_reference(path)
    assert live['status'] == 'VALID'
    assert live['position0_acceptance'] == .7  # cumulative, not snapshot delta
    assert live['per_position_acceptance'] == [.7, .5, .2, .1, 0]
    assert live['accepted_per_round'] == 1.5 and live['mean_depth'] == 4
    assert live['foreign_share'] == .008
    assert len(live['sources']) == 3
    assert len(parse_metrics(Path(live['snapshots'][0]))['counters']) == 9
    final = Path(live['snapshots'][-1])
    final.write_text(final.read_text()+'# changed bytes\n')
    with pytest.raises(ValueError, match='hash changed'):
        verify_reference(path)


def test_counter_reset_and_engine_restart_refuse(tmp_path: Path) -> None:
    path = reference_fixture(tmp_path)
    live = verify_reference(path)
    final = Path(live['snapshots'][-1])
    final.write_text(metrics(70))
    with pytest.raises(ValueError, match='counters reset'):
        verify_reference(path)
    final.write_text(metrics(start=timestamp('2026-10-25T05:01:00Z')))
    with pytest.raises(ValueError, match='rollback restart forbidden'):
        verify_reference(path)


@pytest.mark.parametrize('paused,rollback', [
    ('2026-10-25T05:00:01Z', '2026-10-25T05:01:00Z'),
    ('2026-10-25T04:59:59Z', '2026-10-25T05:00:00Z'),
])
def test_final_snapshot_must_be_paused_and_pre_rollback(tmp_path: Path, paused: str, rollback: str) -> None:
    live = verify_reference(reference_fixture(tmp_path))
    with pytest.raises(ValueError, match='after pause and before rollback'):
        build_reference([Path(p) for p in live['snapshots']], [Path(p) for p in live['rates']], paused, rollback)


def test_foreign_share_above_two_percent_is_inconclusive(tmp_path: Path) -> None:
    live = verify_reference(reference_fixture(tmp_path, gen=244))
    assert live['foreign_share'] == .024 and live['status'] == 'INCONCLUSIVE'
    assert g1.g1_verdict(.7, 'dev', live_reference=live)['pass'] is None


@pytest.mark.parametrize('proxy,passed', [(.35, True), (.3499, False), (.67, True), (.7301, True), (1., True), (0., False)])
def test_tripwire_is_position_zero_on_dev(tmp_path: Path, proxy: float, passed: bool) -> None:
    live = verify_reference(reference_fixture(tmp_path))
    result = g1.g1_verdict(proxy, 'dev', live_reference=live)
    assert result['pass'] is passed
    assert result['statistic'] == 'position0_acceptance'
    for split in ('heldout_legal', 'heldout_general', 'acap'):
        assert g1.g1_verdict(proxy, split, live_reference=live)['pass'] is None


def test_historical_mode_explicit_and_no_default_reference() -> None:
    with pytest.raises(ValueError, match='receipt required'):
        g1.g1_verdict(1.82, 'acap')
    assert g1.g1_verdict(1.82, 'acap', mode='historical-acap')['pass'] is True


@pytest.mark.parametrize('defect', ['legacy', 'wrong_split', 'missing_live', 'changed_snapshot', 'synthetic', 'changed_live_hash'])
def test_real_pilot_admission_refusals(tmp_path: Path, defect: str) -> None:
    manifest = tmp_path/'MANIFEST.json'
    manifest.write_text('{}')
    path = gate_fixture(tmp_path, manifest)
    validate_live_g1(path, manifest)
    gate = json.loads(path.read_text())
    if defect == 'legacy':
        gate['g1'] = g1.g1_verdict(1.82, 'acap', mode='historical-acap')
        gate['split'] = 'acap'
    elif defect == 'wrong_split':
        gate['split'] = 'heldout_legal'
    elif defect == 'missing_live':
        del gate['live_reference_path']
    elif defect == 'synthetic':
        gate['synthetic'] = True
    elif defect == 'changed_live_hash':
        gate['g1']['live_reference_sha256'] = '0'*64
    else:
        live = json.loads(Path(gate['live_reference_path']).read_text())
        final = Path(live['snapshots'][-1])
        final.write_text(final.read_text()+'# modified\n')
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError):
        validate_live_g1(path, manifest)


def test_coordinator_procedure_is_dry_and_quotes_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    import torch
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('dry procedure attempted host/file/CUDA work')
    monkeypatch.setattr(Path, 'read_text', forbidden)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    for gpu in g1.GPU_ALLOWED:
        plan = g1.coordinator_procedure(tmp_path/'shards $literal', tmp_path/'work space', tmp_path/'live.json', gpu)
        assert plan['dry'] is True and plan['deadline_s'] == 780 and plan['hold_limit_s'] == 840
        assert 'gpu-'+gpu[4:12]+'.lock' in plan['lock']
        for body in [*plan['holds'].values(), *plan['scores'].values()]:
            subprocess.run(['bash', '-n'], input=body, text=True, check=True)
            assert 'ssh ' not in body
        assert "CUDA_VISIBLE_DEVICES=''" in plan['scores']['dev']
        assert '--split dev' in plan['holds']['dev']
        assert 'flock -n' in plan['holds']['dev']
    with pytest.raises(ValueError, match='GPU_ALLOWED'):
        g1.coordinator_procedure(tmp_path, tmp_path, tmp_path, 'GPU-production')


def test_proxy_statistic_is_first_match_not_mean_accepted(tmp_path: Path) -> None:
    live = verify_reference(reference_fixture(tmp_path))
    tokens = list(range(100, 130))
    seqs = {'s': {'tokens': tokens, 'gen_start': 10, 'L': 30, 'family': 'test'}}
    drafts = {'s': {p: tokens[p+2:p+7]+[0]*(5-len(tokens[p+2:p+7])) for p in range(9, 28)}}
    result = g1.score(seqs, drafts, 'dev', live_reference=live)
    assert result['position0_match_rate'] == 1
    assert result['k2_accepted_per_round'] == 5
    assert result['g1']['pass'] is True
    assert result['g1']['reference'] == .7
    assert result['g1']['minimum_proxy_rate'] == .35
    assert result['g1']['reference'] == .7


@pytest.mark.parametrize('defect', ['old_rule', 'failed_tripwire', 'forged_pass'])
def test_admission_requires_7b_tripwire(tmp_path: Path, defect: str) -> None:
    manifest = tmp_path/'MANIFEST.json'
    manifest.write_text('{}')
    path = gate_fixture(tmp_path, manifest)
    gate = json.loads(path.read_text())
    if defect == 'old_rule':
        gate['g1'].update(amendment=7, tolerance=.03)
    else:
        gate['position0_match_rate'] = .3499
        gate['g1'] = g1.g1_verdict(.3499, 'dev', live_reference=verify_reference(Path(gate['live_reference_path'])))
        if defect == 'forged_pass':
            gate['g1'].update(tripwire_passed=True, status='PASS', **{'pass': True})
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError):
        validate_live_g1(path, manifest)
