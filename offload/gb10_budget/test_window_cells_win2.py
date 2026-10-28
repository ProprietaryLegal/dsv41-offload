"""WIN2 shared cells: late-arrival mixed 4-user cell and 5,000-token answer cell (CPU only, fake server)."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from offload.gb10_budget import window_cells as wc
from offload.gb10_budget.test_window_cells import FakeApi, FakeClock, plan, texts
from offload.prefill_maxlen.evidence import Json


class RunningApi(FakeApi):
    """FakeApi whose running gauge counts streams that have started and not finished."""
    def __init__(self, clock: FakeClock, **kw: Any) -> None:
        super().__init__(clock, **kw)
        self.active = 0
        self.max_active = 0

    def metrics(self) -> str:
        base = super().metrics().replace('vllm:num_requests_running{m="x"} 0', f'vllm:num_requests_running{{m="x"}} {self.active}')
        return base

    def stream(self, body: dict[str, Json], first: threading.Event) -> dict[str, Json]:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            return super().stream(body, first)
        finally:
            with self.lock:
                self.active -= 1


def test_existing_order_unchanged_and_new_cells_opt_in() -> None:
    assert set(wc.WIN2_CELLS) <= set(wc.CELLS) and not set(wc.WIN2_CELLS) & set(wc.ORDER)
    mixed, long = wc.CELLS['ctl-mixed150k-4user'], wc.CELLS['ctl-long-5k']
    assert mixed.batch == 4 and mixed.late == 1 and mixed.prompts[-1] == '150k' and not mixed.captures
    assert long.max_tokens == 5000 and long.batch == 1 and long.late == 0


@pytest.mark.parametrize('kw', [{'late': 1, 'prompts': ('150k',)}, {'late': 1, 'captures': ('decode',)},
                                {'late': -1}, {'late': 1, 'warmup': True}])
def test_invalid_late_cells_refused(kw: dict[str, Any]) -> None:
    args: dict[str, Any] = {'name': 'x', 'prompts': ('32k-1', '150k'), **kw}
    with pytest.raises(ValueError, match='late arrivals'):
        wc.Cell(**args)


def test_late_arrival_waits_for_decoders_and_proves_overlap(tmp_path: Path) -> None:
    clock = FakeClock()
    api = RunningApi(clock, ttft=1.0, decode=30.0)
    result = wc.run_cell(api, wc.CELLS['ctl-mixed150k-4user'], texts(), plan(tmp_path), sleep=clock.sleep, clock=clock)
    streams = result['streams']
    assert isinstance(streams, list) and len(streams) == 4
    late = result['late_arrival']
    assert isinstance(late, dict) and late['overlap'] is True
    sent = [float(str(s['submitted_rel_s'])) for s in streams if isinstance(s, dict)]
    assert max(sent[:3]) < 1.0 <= sent[3]          # the 150K was sent only after the decoders' first tokens
    assert api.max_active == 4


def test_decoders_finishing_first_refuse_the_cell(tmp_path: Path) -> None:
    clock = FakeClock()
    api = RunningApi(clock, ttft=1.0, decode=1.0)    # decoders end inside the settle period
    with pytest.raises(RuntimeError, match='finished before the late arrival'):
        wc.run_cell(api, wc.CELLS['ctl-mixed150k-4user'], texts(), plan(tmp_path), sleep=clock.sleep, clock=clock)


def test_long_answer_cell_requests_5000_tokens(tmp_path: Path) -> None:
    clock = FakeClock()
    api = RunningApi(clock, ttft=.5, decode=2.0)
    result = wc.run_cell(api, wc.CELLS['ctl-long-5k'], texts(), plan(tmp_path), sleep=clock.sleep, clock=clock)
    streams = result['streams']
    assert isinstance(streams, list) and isinstance(streams[0], dict)
    sampling = streams[0]['sampling']
    assert isinstance(sampling, dict) and sampling['max_tokens'] == 5000 and result['late'] == 0


def test_running_gauge_parser() -> None:
    assert wc.running('vllm:num_requests_running{a="b"} 3\nvllm:num_requests_waiting 1\n') == 3
    with pytest.raises(ValueError, match='num_requests_running'):
        wc.running('vllm:num_requests_waiting 1\n')
