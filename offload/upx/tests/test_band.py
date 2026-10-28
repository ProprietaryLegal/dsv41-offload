"""CPU-only: band arithmetic, KL screen, token greedy and the band cell driver over fake servers (no host)."""
from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import greedy_kl as gk
from offload.prefill_maxlen.evidence import Json, record
from offload.upx import band as bd

GREEDY_POST = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/K1-production/greedy-post.jsonl')


def test_every_short_prompt_step_is_in_the_0064_band_whatever_the_decoders_k() -> None:
    assert bd.BAND == (33, 63)
    for length in bd.TINY_LENGTHS:
        low, high = bd.tiny_rows(length)
        assert bd.in_band(length) and bd.in_band(low) and bd.in_band(high), length
    assert bd.DECODE_ROWS == (3, 15)


def test_probe_lengths_hit_the_band_edges_and_the_unchanged_controls() -> None:
    finals = {n: bd.probe_final_rows(n) for n in bd.PROBE_LENGTHS}
    assert [finals[n] for n in bd.PROBE_LENGTHS] == [32, 33, 42, 43, 63, 64, 40, 40]
    assert {n for n, rows in finals.items() if not bd.in_band(rows)} == set(bd.CONTROL_LENGTHS)
    assert {33, 42, 43, 63} <= set(finals.values()), 'both exl3_smoe chunk regimes and both band edges'
    assert all(bd.in_band(n) for n in bd.GREEDY_LENGTHS)
    assert any(n <= bd.SMOE_CHUNK for n in bd.GREEDY_LENGTHS) and any(n > bd.SMOE_CHUNK for n in bd.GREEDY_LENGTHS)


def test_standard_greedy_prompts_never_reach_the_band() -> None:
    """MEASURED: the K1 reference usage of the six W5 greedy prompts; B1 decode verifies at most 6 rows."""
    if not GREEDY_POST.is_file():
        pytest.skip(f'{GREEDY_POST} unavailable')
    prompts = [json.loads(line)['usage']['prompt_tokens'] for line in GREEDY_POST.read_text().splitlines()]
    assert prompts == [20, 17, 24, 20, 27, 23]
    assert max(prompts) < bd.BAND[0] and 1 + 5 < bd.BAND[0]


def test_regions_are_disjoint_from_each_other_and_from_the_shared_cell_prompts() -> None:
    spans = [(o, o + c) for o, c, _ in bd.REGIONS.values()]
    # window_cells prompts: generous 4 characters per token over their targets
    spans += [(p.offset, p.offset + 4 * p.target) for p in wc.PROMPTS.values()]
    spans.sort()
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 <= b0, ((a0, a1), (b0, b1))
    assert bd.REGIONS['tiny'][2] == bd.TINY_SPACING * bd.TINY_SLOTS
    assert max(bd.TINY_LENGTHS) <= bd.TINY_SPACING


def test_tiny_prompts_cycle_lengths_and_never_overlap() -> None:
    tiny = list(range(bd.TINY_SPACING * bd.TINY_SLOTS))
    seen: set[int] = set()
    for j in range(bd.TINY_SLOTS):
        p = bd.tiny_prompt(tiny, j)
        assert len(p) == bd.TINY_LENGTHS[j % len(bd.TINY_LENGTHS)]
        assert not seen & set(p)
        seen |= set(p)
    assert bd.tiny_prompt(tiny, bd.TINY_SLOTS) == bd.tiny_prompt(tiny, 0)


def test_split_engagement_line() -> None:
    line = ('(Worker_TP0 pid=77) INFO 10-27 20:01:02 [forward.py:110] EXL3 routed experts: a 47-row call runs exl3_smoe '
            'on row chunks of <= 42 rows (dsv41 patch 0064, VLLM_EXL3_MOE_SPLIT_MAX_TOKENS=63)')
    assert bd.split_engaged('x\n' + line + '\n') == 47
    assert bd.split_engaged('EXL3 RDMA all-reduce: rank 0 of 2') is None
    with pytest.raises(ValueError, match='outside'):
        bd.split_engaged(line.replace('a 47-row', 'a 70-row'))


def top(values: Mapping[int, float]) -> dict[str, Json]:
    """A 20-token top-20 dict from a few given logprobs plus filler (all as token_id:N keys)."""
    out: dict[str, Json] = {f'token_id:{k}': v for k, v in values.items()}
    filler = 1000
    while len(out) < 20:
        out[f'token_id:{filler}'] = -12.0 - filler / 1e4
        filler += 1
    return out


def screen_inputs(shift: float) -> tuple[dict[str, Json], dict[str, Json], dict[str, Json]]:
    base = top({1: math.log(.6), 2: math.log(.3), 3: math.log(.09)})
    aa = top({1: math.log(.6), 2: math.log(.3), 3: math.log(.09)})
    x = top({1: math.log(.6 - shift), 2: math.log(.3 + shift), 3: math.log(.09)})
    c1: dict[str, Json] = {str(n): base for n in bd.PROBE_LENGTHS}
    c2: dict[str, Json] = {str(n): aa for n in bd.PROBE_LENGTHS}
    xs: dict[str, Json] = {str(n): x for n in bd.PROBE_LENGTHS}
    return c1, xs, c2


def test_kl_screen_passes_small_differences_and_refuses_large_ones() -> None:
    ok = bd.kl_screen(*screen_inputs(.01))
    assert ok['pass'] is True
    row = record(record(ok['rows'])['43'])
    assert row['band'] is True and row['bound'] == bd.KL_FLOOR and row['identical_top20'] is False
    assert record(record(ok['rows'])['64'])['band'] is False
    bad = bd.kl_screen(*screen_inputs(.2))
    assert bad['pass'] is False


class FakeHttp(wc.Http):
    """Completions server: greedy continuation = a per-prompt fixed list unless `flip` changes one position."""

    def __init__(self, flip: dict[int, tuple[int, int]] | None = None, tie_nats: float = .3) -> None:
        super().__init__('fake://')
        self.flip = flip or {}
        self.tie_nats = tie_nats
        self.calls: list[dict[str, Json]] = []

    @staticmethod
    def step(token: int) -> int:
        return (token * 7 + 3) % 4999            # never 4999, the fake's runner-up token

    def continuation(self, prompt: list[int]) -> list[int]:
        out = [self.step(prompt[-1])]
        while len(out) < bd.GREEDY_MAX_TOKENS:
            out.append(self.step(out[-1]))
        return out

    def post(self, endpoint: str, body: dict[str, Json]) -> dict[str, Json]:
        self.calls.append(body)
        assert endpoint == '/v1/completions'
        raw = body['prompt']
        prompt = [t for t in raw if isinstance(t, int)] if isinstance(raw, list) else []
        if body['max_tokens'] == 1:     # greedy_kl.probe: top-20 at the prefix; the greedy next token is step(last)
            a = self.step(prompt[-1])
            return {'choices': [{'logprobs': {'top_logprobs': [top({a: -.9, 4999: -.9 - self.tie_nats})]}}]}
        out = self.continuation(prompt)
        if len(prompt) in self.flip:
            j, token = self.flip[len(prompt)]
            out[j] = token
        return {'choices': [{'finish_reason': 'length',
                             'logprobs': {'tokens': [f'token_id:{t}' for t in out]}}]}


def test_token_greedy_records_then_compares_and_passes_registered_near_ties() -> None:
    region = list(range(1, 2000))
    prompts = bd.greedy_prompts(region)
    reference = bd.references(bd.token_greedy(FakeHttp(), prompts, None, 'w:c1'))
    assert [len(r) for r in reference] == [bd.GREEDY_MAX_TOKENS] * len(prompts)
    same = bd.token_greedy(FakeHttp(), prompts, reference, 'w:x')
    assert same['identical'] == len(prompts)
    # prompt of 43 tokens diverges at token 5 to 4999, which sits 0.3 nats below the reference token: near-tie
    ref43 = reference[bd.GREEDY_LENGTHS.index(43)]
    result = bd.token_greedy(FakeHttp(flip={43: (5, 4999)}), prompts, reference, 'w:x')
    checks = result['checks']
    assert isinstance(checks, list)
    row = record([r for r in checks if record(r)['prompt_tokens'] == 43][0])
    assert result['identical'] == len(prompts) - 1
    assert row['identical'] is False and row['token_index'] == 5 and row['near_tie'] is True
    assert row['reference_token_id'] == ref43[5] and row['experiment_token_id'] == 4999
    assert row['probe_prefix_token_ids'] == [*prompts[bd.GREEDY_LENGTHS.index(43)], *ref43[:5]]
    assert gk.top20_kl(record(row['top20_logprobs']), record(row['top20_logprobs']))['kl_nats'] == 0


def test_token_greedy_refuses_a_divergence_beyond_the_registration() -> None:
    prompts = bd.greedy_prompts(list(range(1, 2000)))
    reference = bd.references(bd.token_greedy(FakeHttp(), prompts, None, 'w:c1'))
    with pytest.raises(ValueError, match='not a 0.901-nat near-tie'):
        bd.token_greedy(FakeHttp(flip={52: (9, 4999)}, tie_nats=1.5), prompts, reference, 'w:x')


def test_generate_refuses_short_or_malformed_outputs() -> None:
    class Short(FakeHttp):
        def post(self, endpoint: str, body: dict[str, Json]) -> dict[str, Json]:
            return {'choices': [{'finish_reason': 'stop', 'logprobs': {'tokens': ['token_id:1']}}]}
    with pytest.raises(ValueError, match='fixed length'):
        bd.generate(Short(), [1, 2, 3], 's')


# --- band cell driver over a fake server ----------------------------------------------------------------------------

class Counts:
    def parse_metrics(self, text: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for line in text.splitlines():
            name, value = line.rsplit(' ', 1)
            out[name] = float(value)
        return out


class FakeServer:
    def __init__(self, short_needed: int, rounds_per_decoder: int = 100, cached: int = 0) -> None:
        self.short_needed = short_needed
        self.rounds_per_decoder = rounds_per_decoder
        self.cached = cached
        self.lock = threading.Lock()
        self.release = threading.Event()
        self.shorts = 0
        self.drafts = 0.0
        self.accepted = 0.0
        self.decode_s = 0.0
        self.order: list[str] = []
        self.finished = 0
        self.all_done = threading.Event()

    def model(self) -> dict[str, Json]:
        return {'id': bd.MODEL, 'max_model_len': wc.CAP}

    def metrics(self) -> str:
        with self.lock:
            return ('vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n'
                    f'vllm:spec_decode_num_drafts_total {self.drafts}\n'
                    f'vllm:spec_decode_num_accepted_tokens_total {self.accepted}\n'
                    f'vllm:request_decode_time_seconds_sum {self.decode_s}')

    def stream(self, body: dict[str, Json], first: threading.Event) -> dict[str, Json]:
        assert body['ignore_eos'] is True and body['priority'] == 10 and body['stream'] is True
        messages = body['messages']
        assert isinstance(messages, list)
        content = str(record(messages[0])['content'])
        with self.lock:
            self.order.append(content)
        first.set()
        assert self.release.wait(10), 'the fake decoders were never released'
        with self.lock:
            self.drafts += self.rounds_per_decoder
            self.accepted += 3 * self.rounds_per_decoder
            self.decode_s += .08 * self.rounds_per_decoder
            self.finished += 1
            if self.finished == bd.DECODERS:
                self.all_done.set()
        return {'usage': {'prompt_tokens': int(content.split(':')[1]), 'completion_tokens': body['max_tokens'],
                          'prompt_tokens_details': {'cached_tokens': 0}}, 'finish': 'length', 'ttft_s': 1.0}

    def post(self, endpoint: str, body: dict[str, Json]) -> dict[str, Json]:
        assert endpoint == '/v1/completions' and body['max_tokens'] == 1 and body['priority'] == 10
        prompt = body['prompt']
        assert isinstance(prompt, list)
        with self.lock:
            self.shorts += 1
            if self.shorts >= self.short_needed:
                self.release.set()
        if self.release.is_set():
            # the released decoders finish before this short prompt's response arrives (as when their last round
            # lands first); the pause lets the executor mark their futures done
            assert self.all_done.wait(10)
            time.sleep(.05)
        return {'usage': {'prompt_tokens': len(prompt), 'completion_tokens': 1,
                          'prompt_tokens_details': {'cached_tokens': self.cached}}}


def texts() -> dict[str, tuple[str, int]]:
    return {k: (f'{k}:{p.target - 100}', p.target - 100) for k, p in wc.PROMPTS.items()}


@pytest.mark.parametrize('name', list(bd.BAND_CELLS))
def test_band_cell_counts_band_steps_and_reports_the_pooled_round_metric(name: str) -> None:
    server = FakeServer(short_needed=80)
    tiny = list(range(bd.TINY_SPACING * bd.TINY_SLOTS))
    result = bd.run_band_cell(server, Counts(), bd.BAND_CELLS[name], texts(), tiny, bd.Plan(1e12, 'W:c1'),
                              sleep=lambda s: None)
    assert result['band_ok'] is True and result['band_steps'] == 79, 'the 80th finished beside no decoder'
    assert result['short_prompts'] == 80
    assert result['ms_per_round'] == pytest.approx(80.0)
    assert result['accepted_per_round'] == pytest.approx(3.0)
    assert result['band_fraction'] == pytest.approx(int(str(result['band_steps'])) / 100)
    assert result['max_tokens'] == max(bd.BAND_CELLS[name].max_tokens)
    if bd.BAND_CELLS[name].staged:
        assert server.order[0].startswith('150k:'), 'the 150K document goes first, alone'
    json.dumps(result, allow_nan=False)


def test_band_cell_refuses_below_the_band_registration() -> None:
    server = FakeServer(short_needed=10)
    tiny = list(range(bd.TINY_SPACING * bd.TINY_SLOTS))
    with pytest.raises(ValueError, match='band census'):
        bd.run_band_cell(server, Counts(), bd.BAND_CELLS['band-32k-4user'], texts(), tiny, bd.Plan(1e12, 'W:c1'),
                         sleep=lambda s: None)


def test_band_cell_refuses_a_cached_short_prompt() -> None:
    server = FakeServer(short_needed=40, cached=32)
    tiny = list(range(bd.TINY_SPACING * bd.TINY_SLOTS))
    with pytest.raises(ValueError, match='cached'):
        bd.run_band_cell(server, Counts(), bd.BAND_CELLS['band-32k-4user'], texts(), tiny, bd.Plan(1e12, 'W:c1'),
                         sleep=lambda s: None)


def test_band_cell_refuses_after_the_deadline() -> None:
    with pytest.raises(RuntimeError, match='deadline'):
        bd.run_band_cell(FakeServer(1), Counts(), bd.BAND_CELLS['band-32k-4user'], texts(), [], bd.Plan(0, 'W'))


def test_region_pins_detect_drift() -> None:
    built = {'tiny': [1, 2, 3], 'probe': [4, 5]}
    pins = bd.region_pins(built)
    bd.verify_regions(built, pins)
    with pytest.raises(ValueError, match='differ'):
        bd.verify_regions({'tiny': [1, 2, 4], 'probe': [4, 5]}, pins)
