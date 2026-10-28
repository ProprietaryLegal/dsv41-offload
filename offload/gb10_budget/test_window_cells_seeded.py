"""WIN2-R3 seeded paired cells: driver, per-request counters, cold/warm prefix proof, digests (CPU only, fake server)."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from offload.gb10_budget import window_cells as wc
from offload.prefill_maxlen.evidence import Json, record


def texts() -> dict[str, tuple[str, int]]:
    return {key: (f'PUBLIC {key} ' * 10, 300 + i) for i, key in enumerate(wc.PROMPTS)}


def register_fake_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixture texts stand in for the registered prompts: register THEIR hashes (window_cells pins every seeded
    prompt by sha256 and refuses any other text)."""
    monkeypatch.setattr(wc, 'PROMPT_SHA256', {**wc.PROMPT_SHA256, **{k: hashlib.sha256(t.encode()).hexdigest()
                                                                       for k, (t, _) in texts().items()}})


@pytest.fixture(autouse=True)
def _registered(monkeypatch: pytest.MonkeyPatch) -> None:
    register_fake_texts(monkeypatch)


def plan(tmp_path: Path) -> wc.Plan:
    return wc.Plan(deadline=4e9, floor_kb=4 * 1024**2, telemetry=[], salt_prefix='W-test', ledger=None)


class SeededFake:
    """Fake target: deterministic per seed (output digest and rounds depend only on the seed and `shift`), counters
    published at request end; `foreign` adds one unrelated finished request; `warm` controls the prefix cache."""
    def __init__(self, *, shift: int = 0, foreign: bool = False, cold: bool = True, warm: bool = True,
                 speed: float = 1.0) -> None:
        self.shift, self.foreign, self.cold, self.warm, self.speed = shift, foreign, cold, warm, speed
        self.counters = {wc.DRAFTS: 0.0, wc.ACCEPTED: 0.0, wc.DECODE_SECONDS: 0.0, wc.SUCCESS: 0.0}
        self.bodies: list[dict[str, Json]] = []
        self.seen_salts: set[str] = set()
        self.lock = threading.Lock()

    def model(self) -> dict[str, Json]:
        return {'id': 'fixture', 'max_model_len': wc.CAP}

    def metrics(self) -> str:
        with self.lock:
            lines = ['vllm:num_requests_running{m="x"} 0', 'vllm:num_requests_waiting{m="x"} 0']
            lines += [f'{k}{{m="x"}} {v}' for k, v in self.counters.items()]
        return '\n'.join(lines) + '\n'

    def seeded(self, body: dict[str, Json]) -> dict[str, Json]:
        seed, out = int(str(body['seed'])), int(str(body['max_tokens']))
        salt = str(body['cache_salt'])
        with self.lock:
            self.bodies.append(body)
            first_time = salt + str(body['messages']) not in self.seen_salts
            self.seen_salts.add(salt + str(body['messages']))
            rounds = 100 + (seed + self.shift) % 7 if out > 1 else 0
            self.counters[wc.DRAFTS] += rounds
            self.counters[wc.ACCEPTED] += rounds * 1.4
            self.counters[wc.DECODE_SECONDS] += rounds * 0.04 / self.speed if out > 1 else 0.0
            self.counters[wc.SUCCESS] += 1 + (1 if self.foreign and out > 1 else 0)
        cached = 0 if (first_time and self.cold) else (250 if self.warm else 0)
        span = (out - 1) / (55.0 * self.speed) if out > 1 else 0.0
        return {'response_id': 'x', 'finish': 'length', 'ttft_s': 0.3,
                'usage': {'prompt_tokens': self._tokens(body),
                          'completion_tokens': out, 'prompt_tokens_details': {'cached_tokens': cached}},
                'first_by_kind_s': {'reasoning': 0.3}, 'last_by_kind_s': {'reasoning': 0.3 + span},
                'output_sha256': hashlib.sha256(f'{seed + self.shift}:{out}'.encode()).hexdigest()}

    @staticmethod
    def _tokens(body: dict[str, Json]) -> int:
        messages = body['messages']
        assert isinstance(messages, list)
        return {t[0]: t[1] for t in texts().values()}[str(record(messages[0])['content'])]


def test_registered_cells_profiles_and_seeds() -> None:
    cells = wc.SEEDED_CELLS
    assert cells['seeded-short-b1'].profile == 'single' and cells['seeded-32k-b3'].profile == 'multi'
    bases = [c.seed_base for c in cells.values()]
    assert len(set(bases)) == len(bases)
    b3 = cells['seeded-32k-b3']
    assert b3.seeds(2) == (104_020, 104_021, 104_022) and b3.batch == 3
    with pytest.raises(ValueError, match='n >= 6'):
        wc.SeededCell('x', ('short',), seed_base=1, pairs=5)
    with pytest.raises(ValueError, match='distinct registered prompt'):
        wc.SeededCell('x', ('short', 'short'), seed_base=1)
    assert cells['seeded-32k-b1'].with_pairs(12).pairs == 12 and cells['seeded-short-b1'].estimate_s < 45


def test_b1_cell_scrapes_counters_per_request(tmp_path: Path) -> None:
    api = SeededFake()
    r = wc.run_seeded_cell(api, wc.SEEDED_CELLS['seeded-short-b1'], texts(), plan(tmp_path), sleep=lambda s: None)
    assert isinstance(r['pairs'], list)
    pairs = [record(p) for p in r['pairs']]
    assert r['profile'] == 'single' and len(pairs) == 6
    for i, p in enumerate(pairs):
        seed = 101_000 + 10 * i
        assert p['seeds'] == [seed] and p['rounds'] == 100 + seed % 7
        assert abs(float(str(p['ms_per_round'])) - 40.0) < 1e-9 and abs(float(str(p['accepted_per_round'])) - 1.4) < 1e-9
        assert abs(float(str(p['decode_tps'])) - 55.0) < 1e-9
    measured = [b for b in api.bodies if b['max_tokens'] == wc.SEEDED_MAX_TOKENS]
    assert [b['seed'] for b in measured] == [101_000 + 10 * i for i in range(6)]
    assert all(b['temperature'] == 1.0 and b['ignore_eos'] is True and b['cache_salt'] == 'W-test:seeded-short-b1:0'
               for b in api.bodies)
    assert 'text' not in json.dumps(r) and 'PUBLIC' not in json.dumps(r)          # numbers and digests only


def test_b3_cell_pairs_groups_of_concurrent_requests(tmp_path: Path) -> None:
    api = SeededFake()
    r = wc.run_seeded_cell(api, wc.SEEDED_CELLS['seeded-32k-b3'], texts(), plan(tmp_path), sleep=lambda s: None)
    assert r['profile'] == 'multi' and r['batch'] == 3 and isinstance(r['pairs'], list)
    first = record(r['pairs'][0])
    assert first['seeds'] == [104_000, 104_001, 104_002] and isinstance(first['output_sha256'], list)
    assert isinstance(r['prefill'], list) and len(first['output_sha256']) == 3 and len(r['prefill']) == 3


def test_two_starts_of_a_deterministic_server_are_identical(tmp_path: Path) -> None:
    cell = wc.SEEDED_CELLS['seeded-32k-b1']
    a = wc.run_seeded_cell(SeededFake(), cell, texts(), plan(tmp_path), sleep=lambda s: None)
    b = wc.run_seeded_cell(SeededFake(speed=1.1), cell, texts(), plan(tmp_path), sleep=lambda s: None)
    assert wc.streams_identical(a, b)                               # same tokens and rounds; speed may differ
    c = wc.run_seeded_cell(SeededFake(shift=1), cell, texts(), plan(tmp_path), sleep=lambda s: None)
    assert not wc.streams_identical(a, c)
    assert wc.seeded_values(b, 'decode_tps')[0] > wc.seeded_values(a, 'decode_tps')[0]


@pytest.mark.parametrize('fake,match', [(SeededFake(foreign=True), 'foreign requests'),
                                        (SeededFake(cold=False), 'not cold'),
                                        (SeededFake(warm=False), 'warm prefix')])
def test_refusals(tmp_path: Path, fake: SeededFake, match: str) -> None:
    with pytest.raises((ValueError, RuntimeError), match=match):
        wc.run_seeded_cell(fake, wc.SEEDED_CELLS['seeded-short-b1'], texts(), plan(tmp_path), sleep=lambda s: None)


def test_seeded_cells_pin_their_prompts_by_sha256(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DP-2: a seeded cell refuses a text that is not its registered prompt and records the prompt identity per pair."""
    r = wc.run_seeded_cell(SeededFake(), wc.SEEDED_CELLS['seeded-32k-b3'], texts(), plan(tmp_path), sleep=lambda s: None)
    members = [hashlib.sha256(texts()[k][0].encode()).hexdigest() for k in ('32k-1', '32k-2', '32k-3')]
    identity = hashlib.sha256('|'.join(sorted(members)).encode()).hexdigest()
    assert r['prompt_sha256'] == identity and r['prompt_members'] == members and r['report_only'] is False
    assert all(record(p)['prompt_sha256'] == identity for p in cast_list(r['pairs']))
    changed = {**texts(), 'short': ('PUBLIC short, edited', 300)}
    api = SeededFake()
    with pytest.raises(ValueError, match='not its registered'):
        wc.run_seeded_cell(api, wc.SEEDED_CELLS['seeded-short-b1'], changed, plan(tmp_path), sleep=lambda s: None)
    assert api.bodies == []                                         # refused before any request
    monkeypatch.undo()                                              # the real registry: fixture texts are not the prompts
    with pytest.raises(ValueError, match='not its registered'):
        wc.run_seeded_cell(SeededFake(), wc.SEEDED_CELLS['seeded-short-b1'], texts(), plan(tmp_path), sleep=lambda s: None)
    assert wc.PROMPT_SHA256['short'].startswith('932db59f8e4c') and wc.PROMPT_SHA256['32k-0'].startswith('0ac74617c153')


def test_report_only_cells_take_three_pairs_and_bench2_prompts() -> None:
    cell = wc.SeededCell('bridge', ('bench2-legal-short',), seed_base=7, pairs=3, max_tokens=1500, report_only=True)
    assert cell.pairs == 3 and cell.report_only
    with pytest.raises(ValueError, match='n >= 3'):
        wc.SeededCell('bridge', ('short',), seed_base=7, pairs=2, report_only=True)
    with pytest.raises(ValueError, match='distinct registered prompt'):
        wc.SeededCell('x', ('bench2-legal-short',), seed_base=7)            # decision cells: corpus prompts only
    with pytest.raises(ValueError, match='n >= 6'):
        wc.SeededCell('x', ('short',), seed_base=7, pairs=3)


def cast_list(value: Json) -> list[Json]:
    assert isinstance(value, list)
    return value


def test_counters_refuse_missing_series() -> None:
    with pytest.raises(ValueError, match='lack the seeded-cell counters'):
        wc.counters('vllm:spec_decode_num_drafts_total 1\n')


def test_output_digest_hashes_pieces_in_order_and_passes_lines_through() -> None:
    def sse(**delta: Any) -> bytes:
        return b'data: ' + json.dumps({'choices': [{'delta': delta}]}).encode()
    lines = [sse(reasoning_content='ab'), sse(content='c'), b'data: [DONE]']
    a, b = hashlib.sha256(), hashlib.sha256()
    assert list(wc.output_digest(lines, a)) == lines
    list(wc.output_digest([sse(reasoning_content='a'), sse(reasoning_content='b'), sse(content='c'), b'data: [DONE]'], b))
    assert a.hexdigest() != b.hexdigest()               # piece boundaries are part of the identity
    c = hashlib.sha256()
    list(wc.output_digest(lines, c))
    assert a.hexdigest() == c.hexdigest()
