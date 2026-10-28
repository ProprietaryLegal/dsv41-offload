"""Budget arithmetic/admission use only offline constants and fake clocks."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import cast

import pytest

from offload.pc1 import window as w
from offload.prefill_maxlen.evidence import Json, record
from offload.pc1 import cells, build
from offload.pc1.tests.test_window import Remote, rehearsal as rehearsal


def test_budget_reconciles_measurements_and_complete_recovery_reserve() -> None:
    data = cast(dict[str, Json], json.loads((build.HERE/'budget.json').read_text()))
    # Rounds 1-4 (pre-WIN2) registration: every cell x2 in every start. Kept as recorded history.
    historical = w.PREPARE_S + sum(cells.ESTIMATES[n] * cells.REPEATS[n] for n in cells.CELLS)
    assert historical == data['control_block_s'] == 1214
    assert historical + w.QUALIFY_S == data['candidate_block_s'] == 1244
    assert data['hold_estimate_minutes'] == 75.2
    assert w.RESTORE_S - w.ABORT_S == w.RECOVERY_S == 720
    assert w.SUBMIT_S == w.ABORT_S == 73*60
    assert data['startup_observations_s'] == [388, 404]
    assert w.STARTUP_S == 404
    assert w.GREEDY_S == data['greedy_allowance_per_start_s'] == 60
    assert w.QUALIFY_S == 30
    assert [a.role for a in w.arms('chunk2048')] == ['control','chunk2048','control']
    for name in cells.CELLS:
        row = record(record(data['cells'])[name])
        assert row['allowance_per_group_s'] == cells.ESTIMATES[name]
        assert row['groups_per_start'] == cells.REPEATS[name]
        assert row['requests_per_start'] == cells.batch_for(name)*cells.REPEATS[name] >= 2
        if name in ('short-b1','32k-b1'):
            observed = [record(v) for v in cast(list[Json], row['observations'])]
            scaled = max(float(str(r['max_ttft_s'])) for r in observed) + max(
                float(str(r['max_decode_span_s'])) for r in observed)*511/1499 + 7
            assert row['scaled_512_estimate_s'] == pytest.approx(scaled)
            assert scaled <= cells.ESTIMATES[name]
    assert sum(cells.batch_for(n)*cells.REPEATS[n] for n in cells.CELLS) == 12


EXPECTED_HOLD = {'threshold2048': 76.7, 'threshold512': 76.7, 'chunk2048': 73.7, 'chunk8192': 76.7, 'chunk16384': 76.7}


@pytest.mark.parametrize('group', sorted(EXPECTED_HOLD))
def test_win2_budget_reconciles_per_group_plan(group: str) -> None:
    win2 = record(record(cast(Json, json.loads((build.HERE/'budget.json').read_text())))['win2'])
    assert (win2['decision_repeats'], win2['reported_repeats']) == (cells.DECISION_REPEATS, cells.REPORTED_REPEATS) == (3, 1)
    row = record(record(win2['groups'])[group])
    plan = cells.plan_for(group)
    assert row['decision'] == list(plan.decision) and row['reported'] == list(plan.reported)
    block = w.PREPARE_S + sum(cells.ESTIMATES[n] * plan.repeats(n) for n in plan.cells)
    assert row['control_block_s'] == w.block_seconds(group) == w.arm_seconds(group, 'control') == block
    extra = 180 if group in build.LARGE_ARMS else 0
    assert row['candidate_block_s'] == w.arm_seconds(group, group) == block + w.QUALIFY_S + extra
    assert w.budget_minutes(group) == row['hold_estimate_minutes'] == EXPECTED_HOLD[group] <= 85
    submit = (w.SETUP_S + sum(w.arm_seconds(group, a.role) for a in w.arms(group))) / 60
    assert row['submit_minutes'] == round(submit, 2) and submit * 60 <= w.SUBMIT_S


def test_overbudget_execute_refuses_before_any_command_or_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    def denied(*args: object, **kwargs: object) -> None:
        raise AssertionError('overbudget plan reached an external command')
    monkeypatch.setattr(subprocess,'run',denied)
    monkeypatch.setattr(subprocess,'check_output',denied)
    monkeypatch.setattr(w,'budget_minutes',lambda group='chunk2048':85.01)
    assert w.main(['--execute','--group','chunk2048','--out',str(tmp_path/'absent')]) == 1
    assert 'before any host access' in capsys.readouterr().err
    assert not (tmp_path/'absent').exists()


def test_admission_reserves_all_remaining_starts(rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = rehearsal
    planned = w.arms('chunk2048')
    total = sum(w.arm_seconds('chunk2048', a.role) for a in planned)
    live.submit = fake.clock.now + total - .01
    assert not live.fits(planned[0])
    live.submit += .01
    assert live.fits(planned[0])
    live.submit = fake.clock.now + w.arm_seconds('chunk2048', 'control')
    assert live.fits(planned[-1])
    assert not live.fits(planned[1])


def test_pre_measurement_qualification_budget_refuses_even_a_control(rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = rehearsal
    live.started_at = fake.clock.now - w.PREPARE_S
    with pytest.raises(ValueError,match='qualification budget exhausted'):
        live.qualify(w.arms('chunk2048')[0])


def test_observed_150k_prefill_is_never_scaled_like_decode() -> None:
    data = record(record(json.loads((build.HERE / 'budget.json').read_text())['cells'])['150k-b1'])
    observations = [record(v) for v in cast(list[Json], data['observations'])]
    from offload.prefill_maxlen.evidence import number
    estimate = max(number(r['ttft_s']) for r in observations) + max(
        number(r['decode_span_s']) for r in observations) * 511 / 1499 + 7
    assert 85 < estimate < cells.ESTIMATES['150k-b1'] == 100


def test_win2_allowances_cover_every_measured_start_block() -> None:
    # MEASURED per-repeat maxima are re-derived from the cited windows' checkpoints, then every planned start
    # block's allowance must cover its measured block (the mixed cell alone runs over its 145 s allowance).
    from pathlib import Path
    from collections import defaultdict
    win2 = record(record(cast(Json, json.loads((build.HERE/'budget.json').read_text())))['win2'])
    basis = record(win2['measured_basis'])
    cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    prepare: dict[str, list[float]] = defaultdict(list)
    for name in cast(list[str], basis['sources']):
        rows = [record(cast(Json, json.loads(line))) for line in
                (Path('/opt/dsv41/campaigns/DSV41-OFFLOAD') / name / 'checkpoints.jsonl').read_text().splitlines()]
        start = previous = 0.
        role = ''
        for row in rows:
            now = float(str(row['time']))
            if row.get('status') == 'STARTED':
                start, previous = now, 0.
                arm = str(row['arm'])
                role = 'control' if arm.endswith('-control') else arm.split('-', 1)[1]
            if row.get('gates') == 'PASS':
                prepare[role].append(now - start)
                previous = now
            elif 'cell' in row and previous:
                cell[(role, str(row['cell']))].append((now - previous) / 2)
                previous = now
    assert {k: round(max(v), 1) for k, v in prepare.items()} == basis['prepare_s_max']
    per_repeat = record(basis['cell_s_per_repeat_max'])
    for (role, name), values in cell.items():
        assert record(per_repeat[role])[name] == round(max(values), 1)
    assert max(float(str(v)) for v in record(basis['prepare_s_max']).values()) < w.PREPARE_S
    for group, item in record(basis['projection']).items():
        projection = record(item)
        projected = [float(str(v)) for v in cast(list[Json], projection['blocks_s'])]
        allowed = [w.arm_seconds(group, a.role) for a in w.arms(group)]
        assert projection['allowance_blocks_s'] == allowed
        assert all(m < a for m, a in zip(projected, allowed)), group
        assert float(str(projection['hold_minutes'])) < w.budget_minutes(group) <= 85
