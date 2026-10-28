"""Actual composed sync/async scheduler statements; KV/model execution are fake.

Persistent 64/128 *leftovers* are adversarial competing work, not a physically
impossible total engine capacity. Reservations must reduce earlier extras.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from offload.dspark.d2.scheduler_cpu import FakeKV, load_engine, make_request, make_scheduler

pytestmark = pytest.mark.parametrize('draft_k', [0, 5])

SOURCE = Path(__file__).resolve().parents[1] / 'overlay/vllm/v1/core/sched/scheduler.py'


def setup(asynchronous: bool = False, enabled: bool = True, k: int = 0) -> tuple[SimpleNamespace, SimpleNamespace]:
    e = load_engine(SOURCE.read_text())
    e.scheduler._DSV41_REPLAY_SKIP = enabled
    e.scheduler._DSV41_THROTTLE_MIN_PRIO = 5
    s = make_scheduler(e, asynchronous)
    if k:
        s.vllm_config.speculative_config = SimpleNamespace(max_num_new_slots_for_drafting=k - 1)
        s.num_spec_tokens = s.num_lookahead_tokens = k
        s.num_prefill_lookahead = 1
        if asynchronous:
            s._spec_token_placeholders = [-1] * k
    s.prefix_replay_tokens = 128
    s.requires_kv_delivery = False
    s._free_request_blocks = lambda r: None
    s.encoder_cache_manager.free = lambda r: None
    return e, s


def running(e: SimpleNamespace, s: SimpleNamespace, rid: str, tokens: int,
            priority: int = 10) -> SimpleNamespace:
    r = make_request(e, rid, tokens, priority=priority)
    r.status = e.request.RequestStatus.RUNNING
    r.is_prefill_chunk = True
    s.requests[rid] = r
    s.running.append(r)
    return r


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('leftover', [64, 128])
@pytest.mark.parametrize('path', ['running', 'waiting'])
def test_persistent_small_leftovers_finish(draft_k: int, asynchronous: bool, leftover: int, path: str) -> None:
    e, s = setup(asynchronous, k=draft_k)
    s.scheduler_config.long_prefill_token_threshold = 0
    # This earlier priority-0 request would leave exactly leftover rows forever.
    s.max_model_len = 1000000
    peer = running(e, s, 'earlier', 1000000, priority=0)
    target = make_request(e, 'target', 2048 + 129)
    s.requests['target'] = target
    if path == 'waiting':
        s.waiting.add_request(target)
    else:
        target.status = e.request.RequestStatus.RUNNING
        target.is_prefill_chunk = True
        s.running.append(target)
    prior_skip = False
    for step in range(30):
        # Apply the same offered work each turn; installed budget statements cap it.
        peer.num_computed_tokens = peer.num_tokens - (4096 - leftover)
        peer.is_prefill_chunk = True
        before = target.num_computed_tokens
        counts = s.schedule().num_scheduled_tokens
        q = counts.get('target', 0)
        assert q > 0
        if prior_skip:
            assert q > 128
        assert sum(counts.values()) <= 4096
        if step > 0 or path == 'running':
            assert q >= min(129, target.num_tokens - before)
            assert counts['earlier'] <= 4096 - q
        prior_skip = q > 128 and target.num_tokens - before - q > 128
        assert getattr(target, '_dsv41_replay_end', None) == (before + q if prior_skip else None)
        if target.num_computed_tokens == target.num_tokens:
            assert not prior_skip
            break
    else:
        pytest.fail('no bounded successor progress')


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('remainder', [1, 127, 128, 129, 256, 257])
def test_final_remainders_with_obligation(draft_k: int, asynchronous: bool, remainder: int) -> None:
    e, s = setup(asynchronous, k=draft_k)
    s.scheduler_config.long_prefill_token_threshold = 0
    target = running(e, s, 'target', 4096 - max(0, draft_k - 1) + remainder)
    first = s.schedule().num_scheduled_tokens['target']
    assert first == 4096 - max(0, draft_k - 1)
    promised = getattr(target, '_dsv41_replay_end', None)
    assert (promised is not None) == (remainder > 128)
    q = s.schedule().num_scheduled_tokens['target']
    assert q == remainder
    assert getattr(target, '_dsv41_replay_end', None) is None


@pytest.mark.parametrize('asynchronous', [False, True])
def test_obligation_overrides_cadence_but_preserves_decode(draft_k: int, asynchronous: bool) -> None:
    e, s = setup(asynchronous, k=draft_k)
    target = running(e, s, 'target', 32000, priority=20)
    s.schedule()
    assert target._dsv41_replay_end is not None
    peer = running(e, s, 'legal', 256, priority=0)
    peer.num_computed_tokens = 255
    peer.is_prefill_chunk = False
    peer.spec_token_ids = [7] * draft_k
    counts = s.schedule(throttle_prefills=True).num_scheduled_tokens
    assert counts['target'] >= 129 and counts['legal'] == 1 + draft_k


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('backtrack', [1, 127, 128, 129, 2048])
def test_backtrack_uses_installed_preempt_then_real_prefix_replay(draft_k: int, asynchronous: bool, backtrack: int) -> None:
    e, s = setup(asynchronous, k=draft_k)
    target = running(e, s, 'target', 9000)
    s.schedule()
    target.num_in_flight_tokens = 0  # fake successful model completion, before KV loss
    target.num_computed_tokens -= backtrack
    original = target.num_computed_tokens
    # Actual _preempt_request executes; only its external KV release is faked.
    assert s.schedule().num_scheduled_tokens == {}
    assert target.status == e.request.RequestStatus.PREEMPTED
    assert target.num_computed_tokens == 0
    s._get_local_prefix_cache_hit = lambda r: (s.kv_cache_manager.empty_kv_cache_blocks, original, 0, False)
    # Use the installed _mark_prefix_replay, not make_scheduler's test stub.
    s._mark_prefix_replay = e.scheduler.Scheduler._mark_prefix_replay.__get__(s)
    counts = s.schedule().num_scheduled_tokens
    assert counts['target'] > 128
    assert target.replay_start == max(0, original - 128)
    assert target.num_computed_tokens == target.replay_start + counts['target']


@pytest.mark.parametrize('asynchronous', [False, True])
def test_kv_failure_recovers_without_silent_zero(draft_k: int, asynchronous: bool) -> None:
    e, s = setup(asynchronous, k=draft_k)
    target = running(e, s, 'target', 9000)
    s.schedule()
    target.num_in_flight_tokens = 0  # complete the predecessor before allocation fails
    s.kv_cache_manager.allocate_slots = lambda *args, **kwargs: None
    s._request_blocks_can_be_freed = lambda r: True
    assert s.schedule().num_scheduled_tokens == {}
    assert target.status == e.request.RequestStatus.PREEMPTED
    assert target._dsv41_replay_end is not None  # not erased on failure
    s.kv_cache_manager = FakeKV()
    counts = s.schedule().num_scheduled_tokens
    assert counts['target'] > 128 and target.num_computed_tokens == counts['target']


@pytest.mark.parametrize('asynchronous', [False, True])
def test_waiting_restore_rebuilds_suffix_and_blocked_peer_does_not_freeze(draft_k: int, asynchronous: bool) -> None:
    e, s = setup(asynchronous, k=draft_k)
    target = make_request(e, 'restored', 5000)
    target.num_computed_tokens = 4900
    target._dsv41_replay_end = 4900
    s.requests[target.request_id] = target
    s.waiting.add_request(target)
    s._mark_prefix_replay = e.scheduler.Scheduler._mark_prefix_replay.__get__(s)
    s._handle_blocked_waiting_request = lambda r: r is not target
    peer = running(e, s, 'peer', 5000)
    assert s.schedule().num_scheduled_tokens['peer'] > 0
    s._handle_blocked_waiting_request = lambda r: True
    counts = s.schedule().num_scheduled_tokens
    assert counts['restored'] == 228
    assert target.replay_start == 4772
    assert target.num_computed_tokens == 5000
    assert target._dsv41_replay_end is None


def test_cancel_and_reuse_have_no_inherited_obligation(draft_k: int, ) -> None:
    e, s = setup(k=draft_k)
    old = running(e, s, 'same-id', 9000)
    s.schedule()
    assert old._dsv41_replay_end is not None
    s.running.remove(old)
    del s.requests[old.request_id]
    new = make_request(e, 'same-id', 100)
    s.requests[new.request_id] = new
    s.waiting.add_request(new)
    assert getattr(new, '_dsv41_replay_end', None) is None
    assert s.schedule().num_scheduled_tokens == {'same-id': 100}


def test_impossible_reservation_refuses_before_skip_and_off_is_unchanged(draft_k: int, ) -> None:
    e, s = setup(k=draft_k)
    s.max_num_scheduled_tokens = 128
    r = running(e, s, 'a', 5000)
    with pytest.raises(RuntimeError, match='reservation unavailable'):
        s.schedule()
    assert r.num_computed_tokens == 0
    e.scheduler._DSV41_REPLAY_SKIP = False
    assert s.schedule().num_scheduled_tokens == {'a': 128}


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('restored', [False, True])
def test_ready_waiting_owner_has_floor_ahead_of_greedy_running_and_new_waiter(draft_k: int, 
    asynchronous: bool, restored: bool,
) -> None:
    e, s = setup(asynchronous, k=draft_k)
    s.scheduler_config.long_prefill_token_threshold = 0
    greedy = running(e, s, 'greedy', 32000, priority=0)
    target = make_request(e, 'resumed', 5000, priority=20)
    target.num_computed_tokens = 4900 if restored else 0
    target._dsv41_replay_end = 4900
    s.requests[target.request_id] = target
    s.waiting.add_request(target)
    s._mark_prefix_replay = e.scheduler.Scheduler._mark_prefix_replay.__get__(s)
    new = make_request(e, 'new', 32000, priority=0)
    s.requests[new.request_id] = new
    s.waiting.add_request(new)
    counts = s.schedule().num_scheduled_tokens
    assert counts['resumed'] == 129
    assert counts['greedy'] == 4096 - 129 - 2 * max(0, draft_k - 1)
    assert 'new' not in counts
    # Restored suffix replay begins at 4772; a cold preempt rebuild starts at 0.
    assert target.num_computed_tokens == (4772 if restored else 0) + 129
    assert greedy.num_computed_tokens == 3967 - 2 * max(0, draft_k - 1)


@pytest.mark.parametrize('asynchronous', [False, True])
def test_backtrack_waits_for_connector_fence_while_ready_peer_progresses(draft_k: int, asynchronous: bool) -> None:
    e, s = setup(asynchronous, k=draft_k)
    target = running(e, s, 'target', 9000)
    s.schedule()
    target.num_computed_tokens -= 1
    peer = running(e, s, 'peer', 100, priority=0)
    s._request_blocks_can_be_freed = lambda r: r is not target
    counts = s.schedule().num_scheduled_tokens
    assert counts == {'peer': 100}
    assert target.status == e.request.RequestStatus.RUNNING
    assert target.num_computed_tokens == 4095 - max(0, draft_k - 1)
    target.num_in_flight_tokens = 0
    s._request_blocks_can_be_freed = lambda r: True
    s.schedule()
    assert target.status == e.request.RequestStatus.PREEMPTED


def test_unadmittable_new_waiter_cannot_hide_reserved_resumed_owner(draft_k: int, ) -> None:
    e, s = setup(k=draft_k)
    s.scheduler_config.long_prefill_token_threshold = 0
    peer = running(e, s, 'decoder', 100, priority=0)
    peer.num_computed_tokens = 99
    peer.is_prefill_chunk = False
    peer.spec_token_ids = [7] * draft_k
    target = make_request(e, 'resumed', 5000, priority=20)
    target._dsv41_replay_end = 4096
    new = make_request(e, 'unadmittable', 32000, priority=0)
    for r in (new, target):
        s.requests[r.request_id] = r
        s.waiting.add_request(r)
    allocate = s.kv_cache_manager.allocate_slots
    s.kv_cache_manager.allocate_slots = lambda r, *a, **kw: None if r is new else allocate(r, *a, **kw)
    counts = s.schedule().num_scheduled_tokens
    assert counts['decoder'] == 1 + draft_k and counts['resumed'] >= 129
    assert 'unadmittable' not in counts
    assert new in list(s.waiting)


def test_actual_async_restore_readiness_then_reserved_rebuild(draft_k: int, ) -> None:
    e, s = setup(k=draft_k)
    target = make_request(e, 'async-restored', 5000)
    target.status = e.request.RequestStatus.WAITING_FOR_REMOTE_KVS
    target.num_computed_tokens = 4900
    target._dsv41_replay_end = 4900
    s._mark_prefix_replay = e.scheduler.Scheduler._mark_prefix_replay.__get__(s)
    s._update_waiting_for_remote_kv = e.scheduler.Scheduler._update_waiting_for_remote_kv.__get__(s)
    s._handle_blocked_waiting_request = e.scheduler.Scheduler._handle_blocked_waiting_request.__get__(s)
    s.connector = object()
    s.failed_recving_kv_req_ids = set()
    s.finished_recving_kv_req_ids = set()
    s.kv_cache_manager.cache_blocks = lambda r, n: None
    assert not s._handle_blocked_waiting_request(target)
    assert target.num_computed_tokens == 4900
    s.finished_recving_kv_req_ids.add(target.request_id)
    assert s._handle_blocked_waiting_request(target)
    assert target.status == e.request.RequestStatus.WAITING
    assert target.num_computed_tokens == target.replay_start == 4772
    s.connector = None  # fake external transfer is complete; target scheduler stays real
    s.requests[target.request_id] = target
    s.waiting.add_request(target)
    s.scheduler_config.long_prefill_token_threshold = 0
    running(e, s, 'greedy', 32000, priority=0)
    counts = s.schedule().num_scheduled_tokens
    assert counts['async-restored'] == 129
    assert counts['greedy'] == 3967 - 2 * max(0, draft_k - 1)
    assert target.replay_start == 4644


def test_one_free_slot_unadmittable_owed_waiter_yields_to_ready_owed_peer(draft_k: int, ) -> None:
    e, s = setup(k=draft_k)
    s.max_num_active_reqs = 2
    s.scheduler_config.long_prefill_token_threshold = 0
    running(e, s, 'greedy', 32000, priority=0)
    missing = make_request(e, 'missing-kv', 5000, priority=10)
    healthy = make_request(e, 'healthy', 5000, priority=20)
    for r in (missing, healthy):
        r._dsv41_replay_end = 4900
        s.requests[r.request_id] = r
        s.waiting.add_request(r)
    allocate = s.kv_cache_manager.allocate_slots
    s.kv_cache_manager.allocate_slots = lambda r, *a, **kw: None if r is missing else allocate(r, *a, **kw)
    counts = s.schedule().num_scheduled_tokens
    assert counts == {'greedy': 3967 - 2 * max(0, draft_k - 1), 'healthy': 129}
    assert missing in list(s.waiting)


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('k', [4, 5])
def test_decode_near_max_model_len_with_drafts(draft_k: int, asynchronous: bool, k: int) -> None:
    """S27 regression: replay preserves the installed near-cap decode clamp."""
    results: dict[bool, int] = {}
    for enabled in (False, True):
        e, s = setup(asynchronous, enabled, k=draft_k)
        s.max_model_len = 1000
        s.vllm_config.speculative_config = SimpleNamespace(max_num_new_slots_for_drafting=k - 1)
        s.num_spec_tokens = s.num_lookahead_tokens = k
        s.num_prefill_lookahead = 1
        if asynchronous:
            s._spec_token_placeholders = [-1] * k
        d = running(e, s, 'tail', 995)
        d.max_tokens = 5
        d.append_output_token_ids([7])
        d.num_computed_tokens = d.num_tokens - 1
        d.is_prefill_chunk = False
        d.spec_token_ids = [7] * k
        results[enabled] = s.schedule().num_scheduled_tokens['tail']
    assert results[False] == 1000 - 995 - 1
    assert results[True] == results[False]
