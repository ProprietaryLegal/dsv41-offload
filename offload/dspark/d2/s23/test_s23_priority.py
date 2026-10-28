"""S23 independent probes of D2-FIX1 item 1 on the image scheduler (CPU, offload/d2-fix1 harness reused).

What D2-FIX1's own tests do not cover: KV-pressure preemption victim choice (the place where numeric priority
decides who LOSES work), waiting order when the auxiliary request arrived EARLIER, the deployed throttle floor
coming from the real env-var statement, and the slot count when legal work arrives while three D2 streams decode.
"""
from __future__ import annotations

import importlib
import os
from types import SimpleNamespace

import pytest

from d2.scheduler_cpu import Blocks, FakeKV, load_engine, make_request, make_scheduler


class TightKV(FakeKV):
    """Token-capacity KV: allocation fails once the capacity is used; free() returns a request's tokens."""

    def __init__(self, capacity: int) -> None:
        self.capacity, self.used = capacity, {}

    def allocate_slots(self, request, num_new_tokens, *a, **k):
        need = request.num_computed_tokens + num_new_tokens
        extra = need - self.used.get(request.request_id, 0)
        if sum(self.used.values()) + max(0, extra) > self.capacity:
            return None
        self.used[request.request_id] = max(need, self.used.get(request.request_id, 0))
        return Blocks()

    def free(self, request) -> None:
        self.used.pop(request.request_id, None)


def engine(floor: int = 5) -> SimpleNamespace:
    e = load_engine()
    e.scheduler._DSV41_THROTTLE_MIN_PRIO = floor
    return e


def decoding(e, s, rid: str, priority: int, arrival: float, prompt: int = 100):
    r = make_request(e, rid, prompt, priority=priority, arrival=arrival)
    r.status = e.request.RequestStatus.RUNNING
    r.num_computed_tokens = prompt
    r.append_output_token_ids(7)          # one sampled token pending: a decode step needs 1 new token
    s.requests[rid] = r
    s.running.append(r)
    return r


def test_env_floor_statement_is_the_deployed_5(monkeypatch: pytest.MonkeyPatch) -> None:
    """The module-level statement itself (not a test override) gives 5 with serve.sh's env value."""
    monkeypatch.setenv("VLLM_PREFILL_THROTTLE_MIN_PRIO", "5")
    e = load_engine()
    assert e.scheduler._DSV41_THROTTLE_MIN_PRIO == 5


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("aux,victim", [(20, "d2"), (5, "legal")])
def test_kv_pressure_preempts_auxiliary_not_legal(asynchronous: bool, aux: int, victim: str) -> None:
    """Under KV pressure the priority policy preempts max(priority, arrival). With D2 at 20 the D2 stream is the
    victim even though it arrived first; with the former 5, legal 10 is preempted in favour of D2."""
    e = engine()
    s = make_scheduler(e, asynchronous)
    s.encoder_cache_manager.free = lambda r: None
    s.requires_kv_delivery, s.processed_step_seq = False, 0     # real __init__ values without a KV connector
    s.kv_cache_manager = TightKV(capacity=200 + 1)        # both prompts resident + room for ONE decode token
    s.kv_cache_manager.used = {"d2": 100, "legal": 100}
    decoding(e, s, "d2", aux, arrival=0)
    decoding(e, s, "legal", 10, arrival=1)
    out = s.schedule()
    preempted = sorted(set(s.requests) - {r.request_id for r in s.running})
    assert preempted == [victim]
    assert victim not in out.num_scheduled_tokens


@pytest.mark.parametrize("asynchronous", [False, True])
def test_ndm0_is_never_the_victim(asynchronous: bool) -> None:
    e = engine()
    s = make_scheduler(e, asynchronous)
    s.encoder_cache_manager.free = lambda r: None
    s.requires_kv_delivery, s.processed_step_seq = False, 0     # real __init__ values without a KV connector
    s.kv_cache_manager = TightKV(capacity=300 + 1)
    s.kv_cache_manager.used = {"ndm": 100, "legal": 100, "d2": 100}
    decoding(e, s, "ndm", 0, arrival=2)
    decoding(e, s, "legal", 10, arrival=1)
    decoding(e, s, "d2", 20, arrival=0)
    s.schedule()
    assert sorted(r.request_id for r in s.running) == ["legal", "ndm"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_earlier_auxiliary_still_waits_behind_later_legal(asynchronous: bool) -> None:
    """D2's own test uses equal arrival times; here D2 arrives first and still goes behind legal in the queue."""
    e = engine()
    s = make_scheduler(e, asynchronous)
    s.max_num_active_reqs = s.max_num_running_reqs = 1    # one free slot: who gets it
    for rid, pr, t in [("d2", 20, 0.0), ("legal", 10, 5.0)]:
        r = make_request(e, rid, 100, priority=pr, arrival=t)
        s.requests[rid] = r
        s.waiting.add_request(r)
    assert list(s.schedule().num_scheduled_tokens) == ["legal"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_legal_arriving_during_three_decoding_d2_streams_is_admitted_same_step(asynchronous: bool) -> None:
    """Steady state of the real run is decode, not prefill: three decoding D2 streams with K=5 drafting use
    3 x (1 + 4) input tokens; legal prefill is admitted in the same step with the 1024-token chunk cap."""
    e = engine()
    s = make_scheduler(e, asynchronous)
    s.vllm_config.speculative_config = SimpleNamespace(max_num_new_slots_for_drafting=4)
    for j in range(3):
        decoding(e, s, f"d2-{j}", 20, arrival=j)
    legal = make_request(e, "legal", 32000, priority=10, arrival=9)
    s.requests["legal"] = legal
    s.waiting.add_request(legal)
    counts = s.schedule().num_scheduled_tokens
    assert counts["legal"] == 1024 and all(counts[f"d2-{j}"] == 1 for j in range(3))
    assert len(s.running) == 4
