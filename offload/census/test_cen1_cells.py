"""CEN1 request driver, CPU only: window/live rows, one batch over a fake server, refusals."""
from __future__ import annotations

import json
import threading
from typing import cast

import pytest

from offload.census import cen1_cells as c
from offload.gb10_budget import window_cells as wc
from offload.prefill_maxlen.evidence import Json, record

TEXTS = {k: (f'public {k}', 1000 + i) for i, k in enumerate(('short', '32k-0', '32k-1', '32k-2', '32k-3'))}


class FakeApi:
    def __init__(self, *, busy: bool = False, tokens: int = 0, finish: str = 'length') -> None:
        self.busy, self.extra_tokens, self.finish = busy, tokens, finish
        self.bodies: list[dict[str, Json]] = []
        self.lock = threading.Lock()

    def model(self) -> dict[str, Json]:
        return {'id': 'm', 'max_model_len': wc.CAP}

    def metrics(self) -> str:
        return f'vllm:num_requests_running {int(self.busy)}\nvllm:num_requests_waiting 0\n'

    def tokens(self, text: str) -> int:
        return 0

    def profile(self, verb: str) -> None:
        raise AssertionError('CEN1 never profiles')

    def stream(self, body: dict[str, Json], first: threading.Event) -> dict[str, Json]:
        with self.lock:
            self.bodies.append(body)
        text = str(record(cast(list[Json], body['messages'])[0])['content'])
        prompt = next(n for k, (t, n) in TEXTS.items() if t == text)
        return {'usage': {'prompt_tokens': prompt + self.extra_tokens, 'completion_tokens': c.MAX_TOKENS,
                          'prompt_tokens_details': {'cached_tokens': 512}}, 'finish': self.finish, 'response_id': 'r'}


def test_rows_follow_the_speculation_schedule() -> None:
    assert [c.speculative_k(b) for b in (1, 2, 3, 4)] == [5, 5, 4, 4]
    assert c.window_rows(1) == 6 and c.window_rows(4) == 20
    assert [c.live_rows(1, d) for d in c.DEPTHS['short-b1']] == [1, 2, 3, 4, 5, 6]
    assert [c.live_rows(4, d) for d in c.DEPTHS['32k-b4']] == [4, 8, 12, 16, 20]
    with pytest.raises(ValueError):
        c.live_rows(4, 5)
    plan = c.plan()
    assert plan['order'] == ['short-b1', '32k-b1', '32k-b4']
    json.dumps(plan)


def test_one_batch_sends_every_stream_with_one_salt_and_fresh_seeds() -> None:
    api = FakeApi()
    out = c.batch(api, '32k-b4', TEXTS, 3, 'tag', 1e12, wall=lambda: 0.0)
    assert out['batch'] == 4 and len(cast(list[Json], out['streams'])) == 4
    salts = {str(b['cache_salt']) for b in api.bodies}
    assert salts == {'tag:32k-b4:0'}, 'one salt per cell keeps the prefix cached after the first batch'
    assert sorted(int(cast(int, b['seed'])) for b in api.bodies) == [48, 49, 50, 51]
    assert all(b['ignore_eos'] is True and b['max_tokens'] == c.MAX_TOKENS for b in api.bodies)
    text = json.dumps(out)
    assert 'public' not in text, 'no prompt text in the artifact'


@pytest.mark.parametrize('api,match', [(FakeApi(busy=True), 'idle'), (FakeApi(tokens=1), 'built prompt'),
                                       (FakeApi(finish='stop'), 'MAX_TOKENS')])
def test_batch_refusals(api: FakeApi, match: str) -> None:
    with pytest.raises((ValueError, RuntimeError), match=match):
        c.batch(api, 'short-b1', TEXTS, 1, 'tag', 1e12, wall=lambda: 0.0)


def test_batch_refuses_the_deadline() -> None:
    with pytest.raises(RuntimeError, match='deadline'):
        c.batch(FakeApi(), 'short-b1', TEXTS, 1, 'tag', 100.0, wall=lambda: 0.0)
