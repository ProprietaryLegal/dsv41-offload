"""Real hash-pinned public corpus coverage; CPU only, no tokenizer server or skip."""
from __future__ import annotations

import copy
import hashlib
import threading
import time
from itertools import pairwise
from pathlib import Path
from typing import cast

import pytest

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import pp4_cells as c
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record

DATA = Path('/opt/dsv41/data/bench-data')
# Independent pins: changing implementation constants must not repin this test.
PINS = {
    'opinions': ('opinions_cl.jsonl', '80461203f88a6b837eed8fdf9b7e77a40660acb36dfa9887e00a0ae37483b6a1', 10_195_729),
    'filings': ('filings_recap.jsonl', 'b1f2744634cddb0716074c451df5f46e2da3d851324ab794f0caff05ab7202a9', 1_592_473)}


class Api:
    """Count surrogate and API transport only; real corpus/render/request paths."""
    def __init__(self) -> None:
        self.completed = 0
        self.sent: list[dict[str, Json]] = []
        self.lock = threading.Lock()

    def tokens(self, text: str) -> int:
        return len(text)//4

    def metrics(self) -> str:
        with self.lock:
            n = self.completed
        return ('vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n'
            f'vllm:spec_decode_num_drafts_total {100*n}\n'
            f'vllm:spec_decode_num_draft_tokens_total {500*n}\n'
            f'vllm:spec_decode_num_accepted_tokens_total {120*n}\n'
            f'vllm:request_decode_time_seconds_sum {2*n}\n'
            f'vllm:request_decode_time_seconds_count {n}\n')

    def model(self) -> dict[str, Json]:
        return {'id': 'CPU-public-fixture', 'max_model_len': wc.CAP}

    def profile(self, verb: str) -> None:
        raise AssertionError('unexpected profiler: '+verb)

    def stream(self, body: dict[str, Json], first: threading.Event) -> dict[str, Json]:
        cache = body['cache_salt']
        assert isinstance(cache, str) and cache.strip()
        text = str(ck.rows(body['messages'])[0]['content'])
        with self.lock:
            assert cache not in [row['cache_salt'] for row in self.sent]
            self.sent.append(body)
            self.completed += 1
        first.set()
        return {'response_id': 'CPU-fixture', 'finish': 'length', 'usage': {
            'prompt_tokens': self.tokens(text), 'completion_tokens': body['max_tokens'],
            'prompt_tokens_details': {'cached_tokens': 0}}, 'ttft_s': 1.,
            'first_by_kind_s': {'reasoning': 1.}, 'last_by_kind_s': {'reasoning': 3.}}


@pytest.mark.parametrize('family', c.FAMILIES)
def test_every_real_corpus_cut_has_space_for_conservative_bound(family: str) -> None:
    filename, digest, expected_chars = PINS[family]
    # Required real files; missing or changed input is a failure, never a skip.
    assert hashlib.sha256((DATA/filename).read_bytes()).hexdigest() == digest
    materials, hashes = c.family_corpus(DATA, family)
    body = wc.body(materials)
    assert len(body) == expected_chars
    assert hashes == {filename: digest}
    prompts, plan = c.family_prompts(materials, hashes)
    assert set(prompts) == {'short', *(f'short-{i}' for i in range(1, 4)), *(f'32k-{i}' for i in range(20))}
    assert c.CHARS_PER_TOKEN_BOUND == 8
    spans = [(p.offset, p.offset + p.target*8) for p in prompts.values()]
    assert all(0 <= start < end <= len(body) for start, end in spans)
    assert all(b[0]-a[0] >= 4096 for a, b in pairwise(spans))
    for p in prompts.values():
        # Also cover the production token-count search's first rendering.
        for chars in (int(p.target*wc.FIRST_RATIO), p.target*8):
            assert len(wc.body(wc.render(materials, p.offset, chars))) == chars
    total_reserved = sum(end-start for start, end in spans)
    assert plan['overlap_required'] == (total_reserved > len(body))
    if total_reserved <= len(body):
        assert all(a[1] <= b[0] for a, b in pairwise(spans))
        assert prompts['32k-19'].offset == 4_896_768
    else:
        assert family == 'filings' and total_reserved == 5_152_768
        assert prompts['32k-19'].offset == 1_336_461
    # Independently built admission/measurement cuts have identical hashes/layout.
    admission = c.build(Api(), materials, hashes)
    measurement = c.build(Api(), materials, hashes)
    assert admission == measurement and admission['plan'] == plan
    texts = c.texts_from(admission, materials, hashes)
    assert len({hashlib.sha256(text.encode()).hexdigest() for text, _ in texts.values()}) == 24
    for key, (text, tokens) in texts.items():
        row = record(record(admission['prompts'])[key])
        assert row['sha256'] == hashlib.sha256(text.encode()).hexdigest()
        assert .98*prompts[key].target <= tokens <= prompts[key].target


@pytest.mark.parametrize('change', ['hash', 'body', 'registration', 'offset', 'reserve', 'prompt-hash'])
def test_frozen_real_cut_drift_refuses(change: str) -> None:
    materials, hashes = c.family_corpus(DATA, 'filings')
    cuts = c.build(Api(), materials, hashes)
    if change == 'hash':
        hashes = {c.CORPUS_FILES['filings']: '0'*64}
    if change == 'body': materials = materials[:-1]+'x'
    if change == 'registration': record(cuts['plan'])['registration'] = 'old-combined-offsets'
    row = record(record(cuts['prompts'])['32k-19'])
    if change == 'offset': row['offset'] = 5_250_000
    if change == 'reserve': row['chars'] = 32_000*8+1
    if change == 'prompt-hash': row['sha256'] = '0'*64
    with pytest.raises(ValueError): c.texts_from(cuts, materials, hashes)


def test_changed_real_family_file_refuses_before_tokenizer(tmp_path: Path) -> None:
    filename = c.CORPUS_FILES['filings']
    (tmp_path/filename).write_bytes((DATA/filename).read_bytes()+b'\n')
    with pytest.raises(ValueError, match='frozen public corpus changed'):
        c.family_corpus(tmp_path, 'filings')


def test_tokenizer_cannot_exceed_reserved_span() -> None:
    materials, hashes = c.family_corpus(DATA, 'filings')
    # A counter demanding >8 chars/token must stop, not extend into another cut.
    class TooSparse(Api):
        def tokens(self, text: str) -> int:
            return len(text)//20
    with pytest.raises(ValueError, match='outside the public corpus'):
        c.build(TooSparse(), materials, hashes)


@pytest.mark.parametrize('salt', [None, '', '  ', 42])
@pytest.mark.parametrize('sender', ['http', 'seeded'])
def test_missing_salt_refuses_before_any_transport(salt: Json, sender: str) -> None:
    body: dict[str, Json] = {} if salt is None else {'cache_salt': salt}
    api = Api()
    with pytest.raises(ValueError, match='cold per-stream cache_salt required'):
        if sender == 'http': c.Http('http://never-contacted').stream(body, threading.Event())
        else: c.SeededApi(api, 42).stream(body, threading.Event())
    assert not api.sent


def test_actual_request_groups_have_cold_salts_on_every_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api = Api()
    cuts: dict[str, Json] = {f: c.build(api, *c.family_corpus(DATA, f)) for f in c.FAMILIES}
    monkeypatch.setattr(time, 'sleep', lambda seconds: None)
    all_salt_hashes: list[str] = []
    for start in ('OFF-before', 'ON', 'OFF-after'):
        for name in c.CELLS:
            plan = wc.Plan(9_999_999_999., 0, [], 'owned-window-'+start, None)
            result = c.repeated_cell(api, name, copy.deepcopy(cuts), DATA, plan, tmp_path/(start+'-'+name+'.json'))
            for wave in ck.rows(result['waves']):
                for stream in ck.rows(wave['streams']):
                    cache = wc.salt(plan.salt_prefix+'-wave'+str(wave['wave']),
                                    name+'-wave'+str(wave['wave']), int(number(stream['stream'])))
                    assert stream['cache_salt_sha256'] == hashlib.sha256(cache.encode()).hexdigest()
                    all_salt_hashes.append(str(stream['cache_salt_sha256']))
    # Warmup is the same real shared request path with the HTTP salt guard.
    materials, hashes = c.family_corpus(DATA, 'opinions')
    wc.run_cell(api, wc.Cell('warmup', ('32k-0',), max_tokens=64, warmup=True),
        c.texts_from(record(cuts['opinions']), materials, hashes),
        wc.Plan(9_999_999_999., 0, [], 'owned-window-warm', None))
    assert len(api.sent) == 49
    assert len(all_salt_hashes) == len(set(all_salt_hashes)) == 48
    assert len({cast(str, b['cache_salt']) for b in api.sent}) == 49
