"""The new tag changes only request identity on the reviewed generator."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from d2 import generate as G


def test_decode_tag_uses_current_prompt_tail():
    server = G.Server('http://127.0.0.1:9', 'test')
    first = G.Stream(server, 'pA', 0, list(range(200)), 20)
    resume = G.Stream(server, 'pA', 1, list(range(210)), 10)
    short = G.Stream(server, 'pB', 0, [1], 20)
    assert first.request_id == 'd2c.pA.0.72.200'
    assert resume.request_id == 'd2c.pA.1.82.210'
    assert short.request_id == 'd2c.pB.0.0.1'
    assert first.body['priority'] == 20 and G.MAX_STREAMS == 3
    assert first.body['temperature'] == first.body['top_p'] == 1.0
    assert 'logprobs' not in first.body and G.GENERATOR_REVISION == 'd2-decode-capture-option-ii-v2'


def test_historical_state_refused_before_any_io():
    with pytest.raises(SystemExit, match='NEW second-pass'):
        G.cmd_run(SimpleNamespace(state='/opt/dsv41/data/d2/gen'))
