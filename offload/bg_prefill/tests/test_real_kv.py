"""S19 exact trace function with SO2/D2 loader substituted only at setup.

Native KVCacheManager/queues/readiness/completion execute. Transfer notifications
and model completion are synthetic. No GPU tensor, connector transport or socket.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import hashlib
import inspect
import json
import sys
from types import MethodType, SimpleNamespace
from typing import cast

import pytest

from offload.bg_prefill.cpu import engine as so2_engine
from offload.bg_prefill.cpu import production_speculation
from offload.dspark.d2.scheduler_cpu import load_engine, make_scheduler


def setup(native: bool, asynchronous: bool, order: str) -> tuple[SimpleNamespace, SimpleNamespace]:
    if native:
        e = load_engine()
        e.scheduler._DSV41_THROTTLE_MIN_PRIO = 5
        s = make_scheduler(e, asynchronous)
        production_speculation(s)
    else:
        e, s = so2_engine(asynchronous=asynchronous)
    s.requires_kv_delivery = False
    s.encoder_cache_manager.free = lambda req: None
    return e, s


def trace(scope: dict[str, object], native: bool, asynchronous: bool,
          remote_completed: bool) -> dict[str, object]:
    engine, sched = setup(native, asynchronous, 'fcfs')
    create = cast(Callable[..., SimpleNamespace], scope['KVCacheManager'])
    manager = create(scope['kvc_s'], max_model_len=scope['MAX_LEN'],
        scheduler_block_size=scope['sched_bs'], hash_block_size=scope['hash_bs'],
        max_in_flight_tokens=8192, enable_caching=True, use_eagle=True, log_stats=True)
    sched.kv_cache_manager = manager
    sched._get_local_prefix_cache_hit = lambda req: (*manager.get_computed_blocks(req), False)
    # Undo the supplied fixture's queue/readiness shims: execute native methods.
    sched._holds_kv_blocks = engine.scheduler.Scheduler._holds_kv_blocks
    sched._handle_blocked_waiting_request = MethodType(
        engine.scheduler.Scheduler._handle_blocked_waiting_request, sched)
    make = cast(Callable[[list[int], str], SimpleNamespace], scope['req'])
    prompt = cast(Callable[[int, int], list[int]], scope['prompt'])
    holder = make(prompt(32000, 1), 'holder')
    holder.request_id = 'holder'
    holder.priority = 5
    frontier = 3840 if remote_completed else 4096
    assert manager.allocate_slots(holder, 0 if remote_completed else 4096,
                                  num_external_computed_tokens=frontier if remote_completed else 0,
                                  delay_cache_blocks=remote_completed,
                                  full_sequence_must_fit=True, has_scheduled_reqs=False) is not None
    holder.num_computed_tokens = frontier
    holder.status = (engine.request.RequestStatus.WAITING_FOR_REMOTE_KVS if remote_completed
                     else engine.request.RequestStatus.WAITING)
    sched.requests['holder'] = holder
    # Use native enqueue: positive computed position puts it in kv_holding_waiting.
    sched._enqueue_waiting_request(holder)
    assert list(sched.kv_holding_waiting) == [holder]
    sched.finished_recving_kv_req_ids = {'holder'}
    sched.failed_recving_kv_req_ids = set()
    sched.needs_kv_cache_zeroing = False
    # Actual post-transfer cache/replay update and readiness promotion. Only the
    # notification that the transfer completed is synthetic; no GPU bytes exist.
    def finish_transfer(req: SimpleNamespace) -> None:
        # Supply the connector identity assertion only during this native
        # completion callback. Other schedule paths use the no-connector arm.
        sched.connector = SimpleNamespace()
        try:
            engine.scheduler.Scheduler._update_waiting_for_remote_kv(sched, req)
        finally:
            sched.connector = None

    sched._update_waiting_for_remote_kv = finish_transfer
    if remote_completed:
        # Two other loaded prefixes supply ordinary KV pressure. All external
        # frontiers are aligned to 384 and within RP1's default 29568-token cap.
        # External loads allocate only live SWA tails, so a lone 4K load cannot
        # reproduce the refusal. All three holders have completion notifications.
        for index in range(2):
            pressure = make(prompt(96000, 3 + index), f'loaded-pressure-{index}')
            pressure.request_id = f'loaded-pressure-{index}'
            pressure.priority = 10
            assert manager.allocate_slots(pressure, 0, num_external_computed_tokens=29568,
                                          delay_cache_blocks=True, full_sequence_must_fit=True,
                                          has_scheduled_reqs=True) is not None
            pressure.num_computed_tokens = 29568
            pressure.status = engine.request.RequestStatus.WAITING_FOR_REMOTE_KVS
            sched.finished_recving_kv_req_ids.add(pressure.request_id)
            sched.requests[pressure.request_id] = pressure
            sched._enqueue_waiting_request(pressure)
    winner = make(prompt(300000, 2), 'winner')
    winner.request_id = 'winner'
    winner.priority = 0
    sched.requests['winner'] = winner
    sched._enqueue_waiting_request(winner)
    assert manager.allocate_slots(winner, 4096, full_sequence_must_fit=True,
                                  has_scheduled_reqs=True) is None
    outputs: list[dict[str, int]] = []
    for _ in range(8):
        outputs.append(sched.schedule(throttle_prefills=False).num_scheduled_tokens)
        for req in sched.running:
            req.num_in_flight_tokens = 0
    return {'native': native, 'async': asynchronous, 'remote_completed': remote_completed,
            'holder_advanced': holder.num_computed_tokens - frontier,
            'winner_computed': winner.num_computed_tokens, 'trace': outputs,
            'holding_waiting': [r.request_id for r in sched.kv_holding_waiting],
            'waiting': [r.request_id for r in sched.waiting],
            'priorities': [holder.priority, winner.priority]}


@pytest.fixture(scope='module')
def kv_scope() -> dict[str, object]:
    fixture = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/S8R2-evidence/kvsim.py')
    image = Path('/opt/dsv41/data/src-image')
    pins = json.loads(Path(__file__).with_name('regression-pins.json').read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == pins['kvsim_sha256']
    assert hashlib.sha256(inspect.getsource(trace).rstrip().encode()).hexdigest() == pins['s19_trace_sha256']
    old_argv = sys.argv
    scope: dict[str, object] = {}
    try:
        sys.argv = ['so2', str(image), '1342177280', 'bootstrap']
        source = fixture.read_text().split('if scen == "needles":')[0]
        exec(compile(source, str(fixture) + ':bootstrap', 'exec'), scope)
    finally:
        sys.argv = old_argv
    return scope


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('remote_completed', [False, True])
def test_s19_real_holder_drain(kv_scope: dict[str, object], asynchronous: bool,
                              remote_completed: bool) -> None:
    control = trace(kv_scope, True, asynchronous, remote_completed)
    candidate = trace(kv_scope, False, asynchronous, remote_completed)
    assert control['trace'] == candidate['trace']
    assert cast(int, candidate['holder_advanced']) == 8192
    if remote_completed:
        assert candidate['trace'] == [
            {'holder': 1024, 'loaded-pressure-0': 1024, 'loaded-pressure-1': 1024}] * 8
        assert candidate['winner_computed'] == 0


@pytest.mark.parametrize('asynchronous', [False, True])
def test_s19_priority20_holders_keep_same_progress(kv_scope: dict[str, object], asynchronous: bool,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    original_setup = setup

    def setup_background(native: bool, asynchronous: bool, order: str) -> tuple[SimpleNamespace, SimpleNamespace]:
        e, s = original_setup(native, asynchronous, order)
        enqueue = s._enqueue_waiting_request

        def enqueue_background(r: SimpleNamespace) -> None:
            if r.request_id == 'holder' or r.request_id.startswith('loaded-pressure-'):
                r.priority = 20
            enqueue(r)

        s._enqueue_waiting_request = enqueue_background
        return e, s

    monkeypatch.setattr(sys.modules[__name__], 'setup', setup_background)
    control = trace(kv_scope, True, asynchronous, True)
    candidate = trace(kv_scope, False, asynchronous, True)
    assert candidate['trace'] == control['trace']
    assert candidate['holder_advanced'] == 8192
    assert candidate['priorities'] == [20, 0]
