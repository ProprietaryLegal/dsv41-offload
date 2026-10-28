"""CPU execution of the installed image classes (including 0031c), without SO1.

Only dependency imports, GPU construction, KV allocation and delivery are faked.
No scheduler statement, Request body or priority queue is replaced.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from d2.scheduler_cpu import load_engine, make_request, make_scheduler


def engine() -> SimpleNamespace:
    result = load_engine()
    result.scheduler._DSV41_THROTTLE_MIN_PRIO = 5
    return result


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('aux_priority,expected', [(5, ['ndm', 'd2', 'legal']),
                                                  (20, ['ndm', 'legal', 'd2'])])
def test_actual_waiting_admission_order(asynchronous: bool, aux_priority: int,
                                       expected: list[str]) -> None:
    e = engine()
    s = make_scheduler(e, asynchronous)
    for rid, priority in [('d2', aux_priority), ('legal', 10), ('ndm', 0)]:
        r = make_request(e, rid, 100, priority=priority)
        s.requests[rid] = r
        s.waiting.add_request(r)
    assert list(s.schedule().num_scheduled_tokens) == expected
    assert [(r.request_id, r.priority) for r in s.running] == [
        (rid, {'ndm': 0, 'legal': 10, 'd2': aux_priority}[rid]) for rid in expected]


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('foreground,throttled', [(0, True), (10, False)])
def test_0031_does_not_protect_priority10_from_auxiliary(asynchronous: bool,
                                                       foreground: int, throttled: bool) -> None:
    e = engine()
    s = make_scheduler(e, asynchronous)
    for rid, priority in [('d2', 20), ('legal', foreground)]:
        r = make_request(e, rid, 32000 if rid == 'd2' else 256, priority=priority)
        r.status = e.request.RequestStatus.RUNNING
        r.is_prefill_chunk = True
        s.requests[rid] = r
        s.running.append(r)
    before = {r.request_id: r.num_computed_tokens for r in s.running}
    counts = s.schedule(throttle_prefills=True).num_scheduled_tokens
    if throttled:
        assert counts == {'legal': 256}
        assert s.requests['d2'].num_computed_tokens == before['d2']
    else:
        assert counts == {'d2': 1024, 'legal': 256}
    # Release and capacity escape preserve progress; priority20 is no pause rule.
    assert s.schedule(throttle_prefills=False).num_scheduled_tokens['d2'] == 1024
    s.prefill_capacity_bound = True
    assert s.schedule(throttle_prefills=True).num_scheduled_tokens['d2'] == 1024


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('draft_slots', [0, 4])
def test_three_auxiliary_prefills_leave_admission_budget(asynchronous: bool, draft_slots: int) -> None:
    e = engine()
    s = make_scheduler(e, asynchronous)
    s.vllm_config.speculative_config = SimpleNamespace(max_num_new_slots_for_drafting=draft_slots)
    for j in range(3):
        r = make_request(e, f'd2-{j}', 32000, priority=20)
        s.requests[r.request_id] = r
        s.waiting.add_request(r)
    s.schedule()
    legal = make_request(e, 'legal', 32000, priority=10, arrival=1)
    s.requests['legal'] = legal
    s.waiting.add_request(legal)
    counts = s.schedule(throttle_prefills=True).num_scheduled_tokens
    assert counts == {'d2-0': 1024, 'd2-1': 1024, 'd2-2': 1024, 'legal': 1024 - 4 * draft_slots}
    assert len(s.running) == 4 and not list(s.waiting)
