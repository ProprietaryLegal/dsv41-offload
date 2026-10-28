"""Execute pinned Scheduler/AsyncScheduler statements; KV/model delivery faked here."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from offload.bg_prefill.cpu import engine
from offload.bg_prefill.policy import Options
from offload.dspark.d2.scheduler_cpu import Blocks, make_request


def add(e: SimpleNamespace, s: SimpleNamespace, rid: str, priority: int,
        tokens: int = 60000, decode: bool = False, waiting: bool = False) -> SimpleNamespace:
    r = make_request(e, rid, tokens, priority=priority)
    s.requests[rid] = r
    if decode:
        r.num_computed_tokens = tokens
        r.append_output_token_ids(7)
    if waiting:
        s.waiting.add_request(r)
    else:
        r.status = e.request.RequestStatus.RUNNING
        r.is_prefill_chunk = not decode
        s.running.append(r)
    return r


def deliver(e: SimpleNamespace, s: SimpleNamespace, counts: dict[str, int]) -> None:
    for rid in counts:
        r = s.requests[rid]
        r.num_in_flight_tokens = 0
        r.num_output_placeholders = 0
        # Synthetic completion discards the next draft; AsyncScheduler otherwise
        # leaves its speculative placeholders pending across fixture deliveries.
        r.spec_token_ids = []
        if r.num_computed_tokens >= r.num_prompt_tokens:
            r.is_prefill_chunk = False
            # Synthetic delivery keeps one pending decode token; never used as output evidence.
            r.append_output_token_ids(7)


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('priority', [0, 10, 19])
@pytest.mark.parametrize('backgrounds', [1, 2, 3])
@pytest.mark.parametrize('bg_decode', [False, True])
def test_bounded_service_and_foreground_first(asynchronous: bool, priority: int,
                                             backgrounds: int, bg_decode: bool) -> None:
    e, s = engine(asynchronous=asynchronous)
    for j in range(backgrounds):
        add(e, s, f'bg{j}', 20 + j, decode=bg_decode)
    add(e, s, 'fg', priority, decode=True)
    original = [(r, r.priority, r.sampling_params, list(r.prompt_token_ids)) for r in s.running]
    seen: dict[str, list[int]] = {f'bg{j}': [] for j in range(backgrounds)}
    assert s.num_prefill_lookahead == 1 and s.num_spec_tokens == 5
    assert s.vllm_config.speculative_config.method == 'dspark'
    assert s.vllm_config.speculative_config.max_num_new_slots_for_drafting == 4
    assert s.dynamic_sd_lookup == [0, 5, 5, 4, 4]
    for step in range(1, 97):
        counts = dict(s.schedule(throttle_prefills=step % 24 != 0).num_scheduled_tokens)
        assert list(counts)[0] == 'fg'
        assert counts['fg'] == 1
        assert sum(counts.values()) + len(counts) * 4 <= 4096
        for rid in seen:
            if step % 24 == 0:
                assert counts[rid] == (1 if bg_decode else 32)
                seen[rid].append(step)
            else:
                assert rid not in counts
        deliver(e, s, counts)
    assert all(steps == [24, 48, 72, 96] for steps in seen.values())
    assert s.so2_stats['mixed'] == 96 and s.so2_stats['maintenance'] == 4
    assert [(r.priority, r.sampling_params, r.prompt_token_ids) for r, *_ in original] == [
        (p, sampling, prompt) for _, p, sampling, prompt in original]


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('foreground', [0, 10])
def test_waiting_foreground_admission_leaves_budget_without_excluding_background(
        asynchronous: bool, foreground: int) -> None:
    e, s = engine(asynchronous=asynchronous)
    for j in range(3):
        add(e, s, f'bg{j}', 20)
    add(e, s, 'fg', foreground, waiting=True)
    counts = s.schedule(throttle_prefills=True).num_scheduled_tokens
    assert all(counts[f'bg{j}'] == 32 for j in range(3))
    assert counts['fg'] >= 1024 and s.so2_stats['admission'] == 1
    assert s.requests['fg'].priority == foreground


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('priority', [10, 20])
@pytest.mark.parametrize('throttle', [False, True])
def test_s15_unadmittable_waiter_cannot_stall_running_prefill(
        asynchronous: bool, priority: int, throttle: bool) -> None:
    e, s = engine(asynchronous=asynchronous)
    r = add(e, s, 'running', priority, tokens=32000)
    r.num_computed_tokens = 4096
    add(e, s, 'infeasible', 0, tokens=300000, waiting=True)
    allocate = s.kv_cache_manager.allocate_slots

    def bounded(req: SimpleNamespace, *args: object, **kwargs: object) -> Blocks | None:
        return None if req.request_id == 'infeasible' else allocate(req, *args, **kwargs)

    s.kv_cache_manager.allocate_slots = bounded
    for _ in range(8):
        counts = s.schedule(throttle_prefills=throttle).num_scheduled_tokens
        assert counts.get('running', 0) > 0
        assert 'infeasible' not in counts
    assert r.num_computed_tokens > 4096
    if priority == 10 and not throttle:
        assert r.num_computed_tokens == 4096 + 8192  # exact S15 release-step regression


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('priority', [0, 10, 20])
@pytest.mark.parametrize('waiting', [False, True])
def test_solo_cap_applies_only_to_background(asynchronous: bool, priority: int, waiting: bool) -> None:
    e, s = engine(asynchronous=asynchronous)
    add(e, s, 'solo', priority, waiting=waiting)
    counts = s.schedule().num_scheduled_tokens
    # Native foreground consumes the input budget less four reserved draft slots.
    assert counts == {'solo': 1024 if priority >= 20 else 4092}


@pytest.mark.parametrize('asynchronous', [False, True])
def test_native_capacity_escape_preserved(asynchronous: bool) -> None:
    e, s = engine(asynchronous=asynchronous)
    add(e, s, 'bg', 20)
    add(e, s, 'fg', 0, decode=True)
    s.prefill_capacity_bound = True
    counts = s.schedule(throttle_prefills=True).num_scheduled_tokens
    assert counts['bg'] == 1024 and counts['fg'] == 1
    assert s.so2_stats['native'] == 1


@pytest.mark.parametrize('asynchronous', [False, True])
def test_off_matches_native_counts_and_has_no_overlay_state(asynchronous: bool) -> None:
    e, s = engine(enabled=False, asynchronous=asynchronous)
    add(e, s, 'bg', 20)
    add(e, s, 'fg', 10, decode=True)
    assert s.schedule(throttle_prefills=True).num_scheduled_tokens == {'bg': 1024, 'fg': 1}
    assert not hasattr(s, 'so2_stats') and not hasattr(s, 'so2_step')


def test_default_off_and_invalid_env_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ('VLLM_SO2_BG_PREFILL', 'VLLM_SO2_PERIOD', 'VLLM_SO2_QUANTUM'):
        monkeypatch.delenv(name, raising=False)
    assert Options.from_env() == Options(False, 24, 32)
    for key, value in [('VLLM_SO2_BG_PREFILL', 'yes'), ('VLLM_SO2_PERIOD', '0'),
                       ('VLLM_SO2_QUANTUM', '0')]:
        with monkeypatch.context() as m:
            m.setenv(key, value)
            with pytest.raises(ValueError):
                Options.from_env()


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('dependency', ['connector', 'holding', 'blocked', 'large', 'capacity', 'budget', 'mtp'])
def test_unsupported_state_never_pauses_background(asynchronous: bool, dependency: str) -> None:
    e, s = engine(asynchronous=asynchronous)
    add(e, s, 'bg', 20)
    fg = add(e, s, 'fg', 10, decode=True)
    if dependency == 'connector':
        # The pure decision refuses connector states; do not invent its native I/O methods.
        from offload.bg_prefill.policy import Work, decide
        decision = decide(Options(True), 1, [Work('fg', 10, 1, False, True),
                                            Work('bg', 20, 60000, True, True)],
                          waiting_foreground=False, has_waiters=False, dependency=True,
                          input_budget=4096, token_budget=4096, draft_slots=4, max_sequences=4)
        assert decision.mode == 'native'
        return
    if dependency == 'holding':
        holder = make_request(e, 'holder', 60000, priority=0)
        s.requests['holder'] = holder
        s.kv_holding_waiting.add_request(holder)
    elif dependency == 'blocked':
        fg.next_decode_eligible_step = 100
    elif dependency == 'large':
        fg.num_computed_tokens -= 50
    elif dependency == 'capacity':
        s.prefill_capacity_bound = True
    elif dependency == 'budget':
        s.max_num_scheduled_tokens = 2048
    elif dependency == 'mtp':
        s.num_prefill_lookahead = 5
    counts = s.schedule().num_scheduled_tokens
    assert counts.get('bg', 0) > 0
    assert s.so2_stats['native'] == 1 and s.so2_stats['pauses'] == 0


@pytest.mark.parametrize('asynchronous', [False, True])
def test_cancellation_and_resume_keep_native_lifecycle(asynchronous: bool) -> None:
    e, s = engine(asynchronous=asynchronous)
    bg = add(e, s, 'bg', 20)
    add(e, s, 'fg', 10, decode=True)
    s.requires_kv_delivery = False
    s.encoder_cache_manager.free = lambda req: None
    s.kv_cache_manager.free = lambda req: None
    s._free_request = lambda req, **kw: s.requests.pop(req.request_id)
    assert s.schedule().num_scheduled_tokens == {'fg': 1}
    s.finish_requests('fg', e.request.RequestStatus.FINISHED_ABORTED)
    counts = s.schedule().num_scheduled_tokens
    assert counts == {'bg': 1024} and bg.priority == 20


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('due', [False, True])
def test_kv_pressure_preempts_background_never_ndm_or_legal(asynchronous: bool, due: bool) -> None:
    e, s = engine(asynchronous=asynchronous)
    s.so2_step = 23 if due else 0
    used: dict[str, int] = {'bg': 100, 'legal': 100, 'ndm': 100}

    def allocate(r: SimpleNamespace, tokens: int, *args: object, **kw: object) -> Blocks | None:
        needed = r.num_computed_tokens + tokens
        extra = max(0, needed - used.get(r.request_id, 0))
        if sum(used.values()) + extra > 302:
            return None
        used[r.request_id] = max(needed, used.get(r.request_id, 0))
        return Blocks()

    s.kv_cache_manager.allocate_slots = allocate
    s.kv_cache_manager.free = lambda r: used.pop(r.request_id, None)
    s.encoder_cache_manager.free = lambda r: None
    s.requires_kv_delivery, s.processed_step_seq = False, 0
    for rid, priority in [('bg', 20), ('legal', 10), ('ndm', 0)]:
        add(e, s, rid, priority, tokens=100, decode=True)
    # Force one additional allocation row beyond available capacity.
    s.requests['ndm'].append_output_token_ids(7)
    s.schedule()
    assert {r.request_id for r in s.running} == {'ndm', 'legal'}
