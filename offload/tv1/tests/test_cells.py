from __future__ import annotations

import threading
from typing import cast

import pytest

from offload.gb10_budget import window_cells as wc
from offload.prefill_maxlen.evidence import Json
from offload.tv1 import cells


def test_timed_protocol_forces_full_width_without_changing_reference_bodies(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, Json]] = []
    def stream(self: wc.Http, body: dict[str, Json], first: threading.Event) -> dict[str, Json]:
        sent.append(body)
        return {}
    monkeypatch.setattr(wc.Http, 'stream', stream)
    api = cells.Http(wc.URL)
    api.stream({'max_tokens':512}, threading.Event())
    assert sent == [{'max_tokens':512, 'ignore_eos':True}]
    assert all(not c.captures or c.captures == ('decode',) for c in cells.CELLS.values())
    assert all(c.max_tokens == (64 if c.warmup else 512) for c in cells.CELLS.values())


@pytest.mark.parametrize('name', ['ctl-short-b1', 'ctl-32k-b1', 'ctl-150k-b1', 'ctl-32k-b4'])
def test_controls_complete_four_requests_with_independent_salts(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import time
    from offload.gb10_budget.test_window_cells import FakeApi, FakeClock
    from offload.tv1 import metrics
    from offload.prefill_maxlen.evidence import record
    class Api(FakeApi):
        completed = 0
        def metrics(self) -> str:
            return (super().metrics()+f'vllm:request_decode_time_seconds_sum {self.rounds*.04}\n'
                    +f'vllm:request_decode_time_seconds_count {self.completed}\n'
                    +f'vllm:request_success_total {self.completed}\n')
    salts: list[str] = []
    def run(api: wc.Api, cell: wc.Cell, texts: dict[str, tuple[str,int]], plan: wc.Plan) -> dict[str, Json]:
        assert isinstance(api, Api)
        api.rounds += cell.batch*100
        api.completed += cell.batch
        salts.append(plan.salt_prefix)
        return {'cell':cell.name,'batch':cell.batch,'profiled':False,'streams':[
            {'finish':'length','usage':{'completion_tokens':512}} for _ in cell.prompts], 'captures':[]}
    monkeypatch.setattr(wc,'run_cell',run)
    result = cells.measure(Api(FakeClock()),cells.CELLS[name],{},wc.Plan(time.time()+10000,0,[],'paired'))
    trials = cast(list[dict[str, Json]],result['trials'])
    assert len(salts) == len(set(salts)) == (1 if name.endswith('b4') else 4)
    assert result['request_count'] == 4 and metrics.pool(trials)['requests'] == 4
    assert record(result['pooled'])['server_decode_ms_per_spec_request_round'] == 40


def test_whole_repeated_cell_must_fit_before_first_request(monkeypatch: pytest.MonkeyPatch) -> None:
    import time
    from offload.gb10_budget.test_window_cells import FakeApi, FakeClock
    with pytest.raises(ValueError,match='complete repeated cell'):
        cells.measure(FakeApi(FakeClock()),cells.CELLS['ctl-150k-b1'],{},
                      wc.Plan(time.time()+3*cells.CELLS['ctl-150k-b1'].estimate_s,0,[],'paired'))
