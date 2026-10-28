from __future__ import annotations

import pytest

from offload.tv1 import cells, schedule as s


@pytest.mark.parametrize(('feature', 'expected', 'five'), [
    ('candidate', 5055, 8230), ('compact', 3420, 5180)])
def test_full_estimate_keeps_recovery_and_all_qualification(feature: str, expected: int, five: int) -> None:
    assert s.total_s(feature) == expected <= s.WINDOW_S
    assert s.total_s(feature, five=True) == five > s.WINDOW_S
    assert s.roles(feature) == ('off', feature, 'off')
    assert s.block_s('candidate', 'candidate') - s.block_s('candidate', 'off') == 735 + 240
    assert cells.order(feature) == ('warmup', 'ctl-short-b1',
        'ctl-'+cells.controls(feature)[1], 'prof-short-b1', 'prof-'+cells.controls(feature)[1])
    assert sum(cells.CELLS[n].batch*cells.repetitions(cells.CELLS[n])
               for n in cells.order(feature) if n.startswith('ctl-')) == 8


@pytest.mark.parametrize('feature', ['candidate', 'compact'])
def test_no_start_can_borrow_the_closing_control_or_recovery_budget(feature: str) -> None:
    for index in range(3):
        latest = 100 + s.WINDOW_S - s.remaining_s(feature, index)
        s.admit(feature, index, opened=100, now=latest)
        with pytest.raises(ValueError, match='remaining starts plus recovery'):
            s.admit(feature, index, opened=100, now=latest+.01)


@pytest.mark.parametrize('feature', ['candidate', 'compact'])
def test_fake_clock_full_window_fits_and_startup_overrun_refuses(feature: str) -> None:
    now = float(s.SETUP_S)
    for index, role in enumerate(s.roles(feature)):
        s.admit(feature, index, opened=0, now=now)
        now += s.block_s(feature, role)
        if index == 0:
            now += s.BUILD_S
    assert now + s.RECOVERY_S == s.total_s(feature) <= s.WINDOW_S
    # Only one remaining block would fit. Retaining the whole unrun schedule
    # correctly refuses; it never truncates to C/arm or skips the closing C.
    late = s.WINDOW_S - s.RECOVERY_S - s.block_s(feature, 'off')
    with pytest.raises(ValueError, match='over-budget start'):
        s.admit(feature, 0, opened=0, now=late)


def test_first_start_reserves_prompt_build_that_has_not_run_yet() -> None:
    without_build = s.remaining_s('candidate', 0) - s.BUILD_S
    now = s.WINDOW_S - without_build
    with pytest.raises(ValueError, match='over-budget start'):
        s.admit('candidate', 0, opened=0, now=now)


@pytest.mark.parametrize('now', [float('nan'), float('inf'), -1])
def test_bad_clock_cannot_admit(now: float) -> None:
    with pytest.raises(ValueError, match='refuse over-budget'):
        s.admit('candidate', 0, opened=0, now=now)
