"""S54 FIX2: actual health/watch adapters with in-memory queued HTTP traffic."""
from __future__ import annotations

import json
from typing import cast
import urllib.request

import pytest

from offload.launch.windows import production_health as h, post_close_watch as w
from offload.launch.windows.tests.test_post_close_watch import Clock, Recovery

MODELS = {'data': [{'root': h.MODEL_ROOT, 'id': 'healthy-model', 'max_model_len': 327680}]}
COMPLETION = {'choices': [{'text': 'O', 'finish_reason': 'length'}], 'usage': {'completion_tokens': 1}}


@pytest.mark.parametrize('gauges', ['4\n', '0\n'])
def test_http_saturation_or_idle_hang_over_several_polls(
    monkeypatch: pytest.MonkeyPatch, gauges: str,
) -> None:
    clock, recovery = Clock(), Recovery()
    class Response:
        status = 200
        def __init__(self, data: bytes) -> None:
            self.data = data
        def __enter__(self) -> Response:
            return self
        def __exit__(self, *args: object) -> None:
            pass
        def read(self, size: int) -> bytes:
            return self.data
    def urlopen(req: urllib.request.Request, timeout: float) -> Response:
        assert timeout == 10
        if req.full_url.endswith('/metrics'):
            return Response(('vllm:num_requests_running ' + gauges + 'vllm:num_requests_waiting 0\n').encode())
        if req.data is None:
            return Response(json.dumps(MODELS).encode())
        assert isinstance(req.data, bytes)
        body = json.loads(req.data)
        assert body['priority'] == 20 and body['max_tokens'] == 1
        clock.sleep(timeout)
        raise TimeoutError('probe queued' if gauges == '4\n' else 'idle engine hung')
    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    logs: list[str] = []
    result = w.watch(h.ProductionHealth(sleep=clock.sleep), recovery, log=logs.append,
                     clock=clock, sleep=clock.sleep, duration=180)
    if gauges == '4\n':
        assert result.recoveries == result.failures == 0 and not recovery.calls
        assert result.polls >= 4 and any('BUSY/INCONCLUSIVE' in row for row in logs)
    else:
        assert result.recoveries == len(recovery.calls) == 1
        assert result.status.startswith('FAILED: recurrence')
    assert clock() <= 180 + 75  # a bounded final confirmation, never an unbounded queue wait


@pytest.mark.parametrize('state', ['healthy', 'busy', 'unknown', 'recovered', 'idle-hung', 'unreachable'])
def test_immediate_pre_stop_revalidation(state: str) -> None:
    clock = Clock()
    completions = 0
    def request(path: str, body: dict[str, object] | None) -> object:
        nonlocal completions
        if state == 'unreachable':
            raise OSError('no connection')
        if path == '/metrics':
            if state == 'unknown':
                return 'vllm:num_requests_running 0\n'  # missing waiting gauge cannot prove idle
            return 'vllm:num_requests_running ' + ('4' if state == 'busy' else '0') + '\nvllm:num_requests_waiting 0\n'
        if body is None:
            return MODELS
        completions += 1
        if state == 'healthy' or state == 'recovered' and completions >= 2:
            return COMPLETION
        raise TimeoutError('delayed completion')
    health = h.ProductionHealth(request=request, sleep=clock.sleep)
    stops: list[str] = []
    def stop() -> int:
        stops.append('stopped')
        return 0
    if state in ('idle-hung', 'unreachable'):
        assert h.guarded_stop(health, stop) == 0 and stops == ['stopped']
    else:
        with pytest.raises(h.BusyHealth):
            h.guarded_stop(health, stop)
        assert not stops
    assert clock() <= h.CONFIRM_S


def test_busy_resets_a_failure_streak_without_consuming_recovery() -> None:
    clock, recovery = Clock(), Recovery()
    outcomes = iter(['failed', 'busy', 'failed', 'healthy'])
    class Health:
        def probe(self) -> None:
            value = next(outcomes)
            if value == 'busy':
                raise h.BusyHealth('active gauges')
            if value == 'failed':
                raise h.WatchError('confirmed idle failure')
    result = w.watch(Health(), recovery, log=lambda _: None, clock=clock, sleep=clock.sleep, duration=120)
    assert result.recoveries == 0 and result.failures == 2 and not recovery.calls


@pytest.mark.parametrize('metrics', ['', 'vllm:num_requests_running nan\nvllm:num_requests_waiting 0',
                                     'vllm:num_requests_running -1\nvllm:num_requests_waiting 0'])
def test_metrics_never_invents_idle(metrics: str) -> None:
    with pytest.raises((h.WatchError, ValueError)):
        h.requests(cast(object, metrics))
