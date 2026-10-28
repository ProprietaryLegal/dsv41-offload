from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest

from offload.prefill_maxlen.evidence import Json, record, number
from offload.pc1 import cells


def test_delivery_p99_uses_continuous_actual_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Json]] = []
    def data(count: int, content: str, finish: Json = None) -> bytes:
        return ('data: ' + json.dumps({'id': 'public', 'choices': [{'delta': {'content': content}, 'finish_reason': finish}],
                                      'usage': {'completion_tokens': count, 'prompt_tokens': 10}}) + '\n').encode()
    @contextmanager
    def urlopen(request: urllib.request.Request, timeout: float) -> Iterator[list[bytes]]:
        seen.append(json.loads(cast(bytes, request.data)))
        yield [data(1, 'a'), data(3, 'bc'), data(3, '', 'length'), b'data: [DONE]\n']
    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    first = threading.Event()
    row = cells.Http('http://fixture').stream({}, first)
    assert first.is_set()
    assert row['delivery_gap_samples'] == 2
    assert number(row['p99_inter_token_delivery_gap_ms']) >= 0
    assert record(seen[0]['stream_options'])['continuous_usage_stats'] is True
    assert cells.percentile([0.] * 99 + [100.], .99) == 0
    assert cells.percentile([0.] * 98 + [100., 100.], .99) == 100


def test_missing_continuous_usage_refuses_p99(monkeypatch: pytest.MonkeyPatch) -> None:
    @contextmanager
    def urlopen(request: urllib.request.Request, timeout: float) -> Iterator[list[bytes]]:
        yield [b'data: {"id":"public","choices":[{"delta":{"content":"abc"},"finish_reason":"length"}],"usage":{"completion_tokens":3}}\n',
               b'data: [DONE]\n']
    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    with pytest.raises(ValueError, match='per-token delivery timing unavailable'):
        cells.Http('http://fixture').stream({}, threading.Event())


class Api:
    def __init__(self) -> None:
        self.requests: list[dict[str, Json]] = []
        self.stamps: list[float] = []
        self.active = 0
        self.completed = 0
        self.lock = threading.Lock()

    def model(self) -> dict[str, Json]:
        return {'id': 'fixture', 'max_model_len': 327680}

    def tokens(self, text: str) -> int:
        return 100

    def metrics(self) -> str:
        return (f'vllm:num_requests_running {self.active}\nvllm:num_requests_waiting 0\n'
                f'vllm:spec_decode_num_drafts_total {100*self.completed}\n'
                f'vllm:spec_decode_num_accepted_tokens_total {300*self.completed}\n'
                f'vllm:request_decode_time_seconds_sum {4*self.completed}\n'
                f'vllm:request_decode_time_seconds_count {self.completed}\n')

    def profile(self, verb: str) -> None:
        raise AssertionError('PC1 never profiles')

    def stream(self, body: dict[str, Json], first: threading.Event) -> dict[str, Json]:
        with self.lock:
            self.active += 1
            self.requests.append(body)
            self.stamps.append(time.monotonic())
        first.set()
        time.sleep(.15)
        with self.lock:
            self.active -= 1
            self.completed += 1
        n = 150000 if record(cast(list[Json], body['messages'])[0])['content'] == 'long' else 32000
        return {'usage': {'completion_tokens': body['max_tokens'], 'prompt_tokens': n, 'prompt_tokens_details': {'cached_tokens': 0}},
            'finish': 'length', 'ttft_s': .01, 'first_by_kind_s': {'reasoning': .01},
            'last_by_kind_s': {'reasoning': .3}, 'p95_inter_token_delivery_gap_ms': 20., 'p99_inter_token_delivery_gap_ms': 20.}


def test_real_mixed_submission_and_fixed_matched_work(tmp_path: Path) -> None:
    api = Api()
    prompts = {key: ('long' if key == '150k' else 'public', 150000 if key == '150k' else 32000)
               for key in cells.KEYS}
    result = cells.run(api, cells.CELLS[-1], prompts, deadline=time.time() + 500, salt='scope', ledger=tmp_path / 'ledger')
    assert result['batch'] == 3
    assert result['mixed_three_active_overlap'] is True
    assert [r['max_tokens'] for r in api.requests] == [512, 512, 512]
    assert api.stamps[2] >= max(api.stamps[:2])
    assert len({r['cache_salt'] for r in api.requests}) == 3
    assert all(r['temperature'] == 1 and r['seed'] == 42 and r['priority'] == 10 for r in api.requests)
    assert number(result['acceptance_per_round']) == 3
