"""Budget arithmetic/admission use only offline constants and fake clocks."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import up3_window as w
from offload.prefill_maxlen.evidence import Json, record
from offload.up3 import cells, build
from offload.up3.tests.test_up3_window import Remote, rehearsal as rehearsal


def test_budget_reconciles_measurements_and_complete_recovery_reserve() -> None:
    data = cast(dict[str, Json], json.loads((build.HERE/'budget.json').read_text()))
    assert w.budget_minutes() == 83.95 <= 85
    assert w.BLOCK_S == data['control_block_s'] == 1389
    assert w.BUDGET['split'] == data['candidate_block_s'] == 1419
    assert w.RESTORE_S - w.ABORT_S == w.RECOVERY_S == 720
    assert w.SUBMIT_S == w.ABORT_S == 73*60
    assert data['startup_observations_s'] == [388, 404]
    assert w.STARTUP_S == 404
    assert w.GREEDY_S == data['greedy_allowance_per_start_s'] == 60
    assert w.QUALIFY_S == 30
    assert data['hold_estimate_minutes'] == w.budget_minutes()
    assert [a.role for a in w.arms('split')] == ['control','split','control']
    for name in cells.CELLS:
        row = record(record(data['cells'])[name])
        assert row['allowance_per_group_s'] == cells.ESTIMATES[name]
        assert row['groups_per_start'] == cells.REPEATS[name]
        assert row['requests_per_start'] == cells.batch_for(name)*cells.REPEATS[name] >= 4
        if name in ('short-b1','32k-b1','32k-b3'):
            observed = [record(v) for v in cast(list[Json], row['observations'])]
            scaled = max(float(str(r['max_ttft_s'])) for r in observed) + max(
                float(str(r['max_decode_span_s'])) for r in observed)*511/1499 + 7
            assert row['scaled_512_estimate_s'] == pytest.approx(scaled)
            assert scaled <= cells.ESTIMATES[name]
    assert sum(cells.batch_for(n)*cells.REPEATS[n] for n in cells.CELLS) == 24


def test_overbudget_execute_refuses_before_any_command_or_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    def denied(*args: object, **kwargs: object) -> None:
        raise AssertionError('overbudget plan reached an external command')
    monkeypatch.setattr(subprocess,'run',denied)
    monkeypatch.setattr(subprocess,'check_output',denied)
    monkeypatch.setattr(w,'budget_minutes',lambda group='split':85.01)
    assert w.main(['--execute','--group','k5','--out',str(tmp_path/'absent')]) == 1
    assert 'before any host access' in capsys.readouterr().err
    assert not (tmp_path/'absent').exists()


def test_admission_reserves_all_remaining_starts(rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = rehearsal
    planned = w.arms('k5')
    total = sum(w.BUDGET[a.role] for a in planned)
    live.submit = fake.clock.now + total - .01
    assert not live.fits(planned[0])
    live.submit += .01
    assert live.fits(planned[0])
    live.submit = fake.clock.now + w.BUDGET['control']
    assert live.fits(planned[-1])
    assert not live.fits(planned[1])


def test_pre_measurement_qualification_budget_refuses_even_a_control(rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = rehearsal
    live.started_at = fake.clock.now - w.PREPARE_S
    with pytest.raises(ValueError,match='qualification budget exhausted'):
        live.qualify(w.arms('k5')[0])
