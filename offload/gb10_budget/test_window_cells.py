"""CPU-only checks of the SP1-WINDOW cell driver: prompt cuts, capture lifecycle, memory floor, privacy. No network."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from offload.gb10_budget import window_cells as wc
from offload.prefill_maxlen.evidence import Json
from offload.prefill_trace.window_requests import HEADER

SENTINEL = 'PRIVATE_PROMPT_SENTINEL'


class FakeClock:
    """Monotonic time advanced only by the driver's sleeps; fake streams follow it."""

    def __init__(self) -> None:
        self.t = 0.0
        self.lock = threading.Lock()

    def __call__(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        with self.lock:
            self.t += dt
        time.sleep(0.0005)

    def wait_until(self, t: float) -> None:
        while self.t < t:
            time.sleep(0.0005)


class FakeApi:
    def __init__(self, clock: FakeClock, *, ttft: float = 10.0, decode: float = 30.0, fail: bool = False,
                 cached: int = 0, chars_per_token: float = 3.1) -> None:
        self.clock, self.ttft, self.decode, self.fail, self.cached = clock, ttft, decode, fail, cached
        self.ratio = chars_per_token
        self.calls: list[tuple[str, float, int]] = []      # (verb, fake time, streams emitted so far)
        self.emitted = 0
        self.rounds = 0.0
        self.count_calls = 0
        self.lock = threading.Lock()

    def tokens(self, text: str) -> int:
        self.count_calls += 1
        return int(len(text) / self.ratio)

    def metrics(self) -> str:
        return ('vllm:num_requests_running{m="x"} 0\nvllm:num_requests_waiting{m="x"} 0\n'
                f'vllm:spec_decode_num_drafts_total{{m="x"}} {self.rounds}\n'
                f'vllm:spec_decode_num_draft_tokens_total{{m="x"}} {self.rounds * 5}\n'
                f'vllm:spec_decode_num_accepted_tokens_total{{m="x"}} {self.rounds * 1.5}\n')

    def model(self) -> dict[str, Json]:
        return {'id': 'fixture', 'max_model_len': wc.CAP}

    def stream(self, body: dict[str, Json], first: threading.Event) -> dict[str, Json]:
        start = self.clock()
        messages: Any = body['messages']
        text = str(messages[0]['content'])
        self.clock.wait_until(start + self.ttft)
        if self.fail:
            raise RuntimeError('stream fixture failure')
        first.set()
        with self.lock:
            self.emitted += 1
        self.clock.wait_until(start + self.ttft + self.decode)
        with self.lock:
            self.rounds += 100
        tokens = int(len(text) / self.ratio)
        out = int(str(body['max_tokens']))
        return {'response_id': 'chatcmpl-x', 'finish': 'length' if body.get('ignore_eos') else 'stop',
                'usage': {'prompt_tokens': tokens, 'completion_tokens': out,
                          'prompt_tokens_details': {'cached_tokens': self.cached}},
                'ttft_s': self.ttft, 'total_s': self.ttft + self.decode, 'first_by_kind_s': {}, 'last_by_kind_s': {},
                'reasoning_tokens': out, 'answer_and_control_tokens': 0, 'sse_times_ms': [1.0, 2.0],
                'answer_text_should_never_be_here': None}

    def profile(self, verb: str) -> None:
        with self.lock:
            self.calls.append((verb, self.clock(), self.emitted))


def telemetry(tmp_path: Path, kb: int) -> list[Path]:
    paths = []
    for host in ('sparkhead', 'sparkworker-wired'):
        path = tmp_path / f'telemetry-{host}.jsonl'
        path.write_text(json.dumps({'kind': 'memory', 'utc': time.time(), 'kb': {'MemAvailable': kb, 'SwapFree': 1}}) + '\n'
                        + json.dumps({'kind': 'gpu', 'utc': time.time(), 'csv': 'x'}) + '\n')
        paths.append(path)
    return paths


def texts(ratio: float = 3.1) -> dict[str, tuple[str, int]]:
    out = {}
    for key in wc.PROMPTS:
        text = SENTINEL + key + 'x' * 300
        out[key] = (text, int(len(text) / ratio))
    return out


def plan(tmp_path: Path, kb: int = 6 * 1024**2, deadline: float = 0) -> wc.Plan:
    return wc.Plan(deadline or time.time() + 3600, int(4.5 * 1024**2), telemetry(tmp_path, kb), 'tag', tmp_path / 'ledger.jsonl')


def run(tmp_path: Path, name: str, api: FakeApi, clock: FakeClock, **kw: int) -> Any:
    """The cell artifact as plain JSON (what the runner reads back)."""
    result = wc.run_cell(api, wc.CELLS[name], texts(), plan(tmp_path, **kw), sleep=clock.sleep, clock=clock)
    return json.loads(json.dumps(result))


def test_cut_needs_at_most_four_tokenize_calls_and_lands_in_band() -> None:
    materials = HEADER + 'Public opinion text. ' * 700_000
    for ratio in (2.6, 3.1, 3.7, 4.4):
        api = FakeApi(FakeClock(), chars_per_token=ratio)
        for prompt in wc.PROMPTS.values():
            before = api.count_calls
            chars, tokens = wc.cut(materials, prompt, api.tokens)
            assert api.count_calls - before <= wc.MAX_COUNT_CALLS
            assert prompt.target * (1 - wc.TOLERANCE) <= tokens <= prompt.target
            assert tokens == api.tokens(wc.render(materials, prompt.offset, chars))


def test_cut_refuses_instead_of_bisecting_forever() -> None:
    materials = HEADER + 'y' * 12_000_000
    with pytest.raises(ValueError, match='within 4 /tokenize calls'):
        wc.cut(materials, wc.PROMPTS['short'], lambda text: 2 * wc.PROMPTS['short'].target)   # never in band
    with pytest.raises(ValueError, match='outside the public corpus'):
        wc.render(materials, 11_999_990, 100)


def test_build_records_numbers_and_hashes_only_and_rebuild_refuses_drift() -> None:
    materials = HEADER + ('opinion ' + SENTINEL + ' ') * 400_000
    hashes: dict[str, Json] = {'opinions_cl.jsonl': 'a' * 64}
    cuts: Any = wc.build(FakeApi(FakeClock()), materials, hashes)
    assert SENTINEL not in json.dumps(cuts) and set(cuts['prompts']) == set(wc.PROMPTS)
    rebuilt = wc.texts_from(cuts, materials, hashes)
    assert {k: v[1] for k, v in rebuilt.items()} == {k: v['tokens'] for k, v in cuts['prompts'].items()}
    with pytest.raises(ValueError, match='corpus changed'):
        wc.texts_from(cuts, materials, {'opinions_cl.jsonl': 'b' * 64})
    with pytest.raises(ValueError, match='differs from its recorded hash'):
        wc.texts_from(cuts, materials.replace('opinion', 'Opinion', 1), hashes)


def test_control_cell_never_touches_the_profiler_and_writes_no_text(tmp_path: Path) -> None:
    clock = FakeClock()
    api = FakeApi(clock)
    result = run(tmp_path, 'ctl-32k-b4', api, clock)
    assert api.calls == []
    assert len(result['streams']) == 4 and result['captures'] == []
    assert SENTINEL not in json.dumps(result) and 'chatcmpl' not in json.dumps(result)
    spec = result['spec']
    assert spec['rounds'] == 400 and spec['mean_accepted_per_round'] == 1.5
    assert len({s['cache_salt_sha256'] for s in result['streams']}) == 4


def test_32k_profiled_cell_captures_early_prefill_then_decode(tmp_path: Path) -> None:
    clock = FakeClock()
    api = FakeApi(clock, ttft=14.5, decode=24)
    result = run(tmp_path, 'prof-32k-b1', api, clock)
    verbs = [v for v, _, _ in api.calls]
    assert verbs == ['start_profile', 'stop_profile', 'start_profile', 'stop_profile']
    (_, t0, e0), (_, t1, _), (_, t2, e2), (_, t3, _) = api.calls
    assert e0 == 0 and t1 - t0 >= wc.PREFILL_S - .1            # recording began before any token
    assert e2 == 1 and t2 >= 14.5 + wc.SETTLE_S - .1           # decode only after the first token plus settle
    assert t3 - t2 <= wc.DECODE_S + .2
    assert [c['kind'] for c in result['captures']] == ['prefill-early', 'decode']
    assert [c['status'] for c in result['captures']] == ['TAKEN', 'TAKEN']


def test_b4_decode_capture_waits_for_every_stream(tmp_path: Path) -> None:
    clock = FakeClock()
    api = FakeApi(clock, ttft=60, decode=50)
    run(tmp_path, 'prof-32k-b4', api, clock)
    assert [(v, e) for v, _, e in api.calls] == [('start_profile', 4), ('stop_profile', 4)]


def test_late_prefill_refuses_a_request_that_already_decodes(tmp_path: Path) -> None:
    clock = FakeClock()
    api = FakeApi(clock, ttft=20, decode=60)
    with pytest.raises(RuntimeError, match='mislabelled|ended before the late'):
        run(tmp_path, 'prof-150k-b1', api, clock)
    assert api.calls == []


def test_late_prefill_then_decode_on_one_150k_request(tmp_path: Path) -> None:
    clock = FakeClock()
    api = FakeApi(clock, ttft=69, decode=28)
    result = run(tmp_path, 'prof-150k-b1', api, clock)
    (_, t0, e0), _, (_, t2, e2), _ = api.calls
    assert t0 >= wc.LATE_DELAY_S - .1 and e0 == 0 and e2 == 1 and t2 > 69
    assert [c['status'] for c in result['captures']] == ['TAKEN', 'TAKEN']


def test_memory_floor_skips_captures_but_keeps_the_requests(tmp_path: Path) -> None:
    clock = FakeClock()
    api = FakeApi(clock, ttft=14.5, decode=24)
    result = run(tmp_path, 'prof-32k-b1', api, clock, kb=4 * 1024**2 + 1)
    assert api.calls == []
    assert [c['status'] for c in result['captures']] == ['SKIPPED_MEMORY', 'SKIPPED_MEMORY']
    assert len(result['streams']) == 1


def test_stale_telemetry_refuses_before_any_start(tmp_path: Path) -> None:
    clock = FakeClock()
    api = FakeApi(clock)
    p = plan(tmp_path)
    for path in p.telemetry:
        old = time.time() - 60
        import os
        os.utime(path, (old, old))
    with pytest.raises(RuntimeError, match='telemetry older'):
        wc.run_cell(api, wc.CELLS['prof-short-b1'], texts(), p, sleep=clock.sleep, clock=clock)
    assert api.calls == []


def test_stream_failure_still_stops_the_profiler(tmp_path: Path) -> None:
    clock = FakeClock()
    api = FakeApi(clock, ttft=1, fail=True)
    with pytest.raises(RuntimeError, match='stream fixture failure'):
        run(tmp_path, 'prof-32k-b1', api, clock)
    assert [v for v, _, _ in api.calls] == ['start_profile', 'stop_profile']


def test_cache_hit_deadline_and_cap_refusals(tmp_path: Path) -> None:
    clock = FakeClock()
    with pytest.raises(ValueError, match='hit the cache'):
        run(tmp_path, 'ctl-short-b1', FakeApi(clock, ttft=.1, decode=.1, cached=64), clock)
    with pytest.raises(RuntimeError, match='window submission deadline reached'):
        wc.run_cell(FakeApi(clock), wc.CELLS['ctl-150k-b1'], texts(), plan(tmp_path, deadline=time.time() + 10),
                    sleep=clock.sleep, clock=clock)


def test_warmup_forces_its_length(tmp_path: Path) -> None:
    clock = FakeClock()
    api = FakeApi(clock, ttft=.5, decode=.5)
    result = run(tmp_path, 'warmup', api, clock)
    assert result['warmup'] is True and api.calls == []
    assert result['streams'][0]['usage']['completion_tokens'] == 64


def test_cells_cover_the_brief() -> None:
    profiled = {n: c for n, c in wc.CELLS.items() if c.captures}
    assert {(c.prompts[0][:3], c.batch) for c in profiled.values()} == {('sho', 1), ('32k', 1), ('32k', 4), ('150', 1)}
    assert sorted(k for c in profiled.values() for k in c.captures) == ['decode'] * 4 + ['prefill-early', 'prefill-late']
    for name, cell in profiled.items():
        assert wc.CELLS['ctl' + name[4:]].prompts == cell.prompts          # matched never-started control
    assert wc.ORDER.index('ctl-150k-b1') < min(wc.ORDER.index(n) for n in profiled)
    offsets = sorted((p.offset, p.offset + p.target * 5) for p in wc.PROMPTS.values())
    assert all(a[1] <= b[0] for a, b in zip(offsets, offsets[1:]))         # no shared text between prompts
