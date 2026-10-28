"""S32 independent check of offload/d2-opt2 @ ed34558 (option-ii generation top20). Synthetic rows only.

FIX-1 probes (fail at ed34558): a real-shaped completions row whose top20 mass is a hair above 1 (MEASURED in the
real R6 image response: 20/128 rows, max excess 1.24e-7; 3/128 rows with the sampled LP exactly 0.0) passes the
online validator but is saved with resid = 1 - mass < 0. replay.py clips resid to [0, 1]; shards.check_record only
compares against the clipped value with atol 1e-5, so the negative value reaches the shards, and the trainer
(train/loss.py distillation_loss) and post_replay.aggregate refuse any tail mass below 0.
"""
from __future__ import annotations

import http.server
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from d2 import generate as G
from d2 import generation_export as E
from d2 import generation_top20 as T

REPO = Path(__file__).resolve().parents[4]


def wire(own_lp: float, others: list[float], token: int = 7) -> dict[str, object]:
    """One completions logprobs object in the pinned image format (sampled entry first, token_id keys)."""
    ids = [token] + list(range(1000, 1000 + len(others)))
    top = {f'token_id:{k}': v for k, v in zip(ids, [own_lp] + others)}
    return {'tokens': [f'token_id:{token}'], 'token_logprobs': [own_lp], 'top_logprobs': [top], 'text_offset': [0]}


# own LP exactly 0.0 (R6: 3/128 rows) and a near-one top token whose neighbours push the mass just above 1 (R6: 17 more)
REAL_SHAPED = [(0.0, [-17.0] * 19), (-1e-8, [-18.9] * 19)]


@pytest.mark.parametrize('own,others', REAL_SHAPED)
def test_fix1_real_shaped_row_is_accepted_but_resid_must_not_be_negative(own: float, others: list[float]) -> None:
    r = T.rows([7], wire(own, others))[0]       # accepted online: 1 < mass <= 1 + 1e-6
    mass = math.fsum(math.exp(v) for v in r['top_lp'])
    assert 1 < mass <= 1 + 1e-6
    assert r['resid'] >= 0, f"saved resid {r['resid']:.3e} < 0 (replay clips to [0, 1])"


def test_fix1_exported_targets_are_refused_by_the_trainer() -> None:
    import importlib
    sys.path.insert(0, str(REPO))                # the trainer imports as offload.dspark.d2.train (relative natives)
    L = importlib.import_module('offload.dspark.d2.train.loss')
    rows = [T.rows([7], wire(own, others))[0] for own, others in REAL_SHAPED]
    state = {'top20': 'generation', 'gen_ids': [7, 7], 'logprob_rows': rows}
    arrays = E.saved_arrays(state)               # the exporter revalidates and writes these arrays to the npz
    tail = torch.from_numpy(arrays['resid'])[None]
    top_lp = torch.from_numpy(arrays['top_lp'])[None]
    top_ids = torch.from_numpy(arrays['top_ids']).long()[None]
    present = torch.ones_like(tail, dtype=torch.bool)
    try:
        L.distillation_loss(None, None, None, top_ids, top_lp, tail, torch.full_like(tail, 7).long(),  # type: ignore[arg-type]
                            present, present)
    except ValueError as exc:
        assert 'tail mass' not in str(exc) and 'sum to one' not in str(exc), exc
    except (AttributeError, TypeError):
        pass                                     # past the target checks: the stand-in model is None


def test_multi_token_chunk_with_one_bad_row_commits_nothing_from_that_chunk() -> None:
    """Drafting emits several tokens per chunk; one bad row rejects the whole chunk (prefix of earlier chunks kept)."""
    good = wire(-0.5, [-5.0 - j * 0.1 for j in range(19)])
    T.rows([7], good)
    bad = wire(-0.5, [-5.0 - j * 0.1 for j in range(18)])       # 19 entries
    chunk = {k: good[k] + bad[k] for k in ('tokens', 'token_logprobs', 'top_logprobs')}  # type: ignore[operator]
    with pytest.raises(ValueError):
        T.rows([7, 7], chunk)


# ------------------------------------------------------------------ off mode: request bytes identical to 8174771
def _old_generate():
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    src = subprocess.run(['git', 'show', '8174771:offload/dspark/d2/generate.py'], cwd=REPO, env=env,
                         check=True, capture_output=True).stdout
    spec = importlib.util.spec_from_loader('d2.generate_8174771', loader=None)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'd2'
    sys.modules['d2.generate_8174771'] = mod
    exec(compile(src, 'generate_8174771.py', 'exec'), mod.__dict__)
    return mod


class _Capture(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'
    seen: list[bytes] = []

    def log_message(self, *a: object) -> None:
        pass

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers['Content-Length']))
        _Capture.seen.append(self.requestline.encode() + b'\n' + bytes(self.headers) + body)
        choice: dict[str, object] = {'token_ids': [7], 'prompt_token_ids': json.loads(body)['prompt'],
                                     'finish_reason': 'length'}
        if 'logprobs' in json.loads(body):
            choice['logprobs'] = wire(-0.5, [-5.0 - j * 0.1 for j in range(19)])
        out = b'data: ' + json.dumps({'choices': [choice]}).encode() + b'\n\ndata: [DONE]\n\n'
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        self.wfile.write(out)


def _wire_bytes(mod: object, top20: bool | None, port: int, prompt: list[int], max_tokens: int, extra: dict | None,
                pause: Path) -> bytes:
    server = mod.Server(f'http://127.0.0.1:{port}', 'm')  # type: ignore[attr-defined]
    kw: dict[str, object] = {} if top20 is None else {'top20': top20, 'pause_file': str(pause)}
    st = mod.Stream(server, 'p0', 2, prompt, max_tokens, extra, **kw)  # type: ignore[attr-defined]
    st.run()
    assert st.error is None and st.finish == 'length'
    return _Capture.seen.pop()


@pytest.mark.parametrize('prompt,max_tokens,extra', [([1, 2, 3], 16384, None), (list(range(300)), 77, None),
                                                    ([9] * 5, 5, {'chat_template_kwargs': {'thinking': True}})])
def test_off_mode_request_bytes_equal_8174771(prompt: list[int], max_tokens: int, extra: dict | None,
                                              tmp_path: Path) -> None:
    old = _old_generate()
    srv = http.server.HTTPServer(('127.0.0.1', 0), _Capture)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        port = srv.server_address[1]
        a = _wire_bytes(old, None, port, prompt, max_tokens, extra, tmp_path / 'PAUSE')
        b = _wire_bytes(G, None, port, prompt, max_tokens, extra, tmp_path / 'PAUSE')
        c = _wire_bytes(G, False, port, prompt, max_tokens, extra, tmp_path / 'PAUSE')
        on = _wire_bytes(G, True, port, prompt, max_tokens, extra, tmp_path / 'PAUSE')
    finally:
        srv.shutdown()
    assert a == b == c
    assert not (tmp_path / 'PAUSE').exists()
    old_body = json.loads(a.split(b'\r\n\r\n', 1)[1] if b'\r\n\r\n' in a else a.split(b'\n\n', 1)[1])
    new_body = json.loads(on.split(b'\r\n\r\n', 1)[1] if b'\r\n\r\n' in on else on.split(b'\n\n', 1)[1])
    assert new_body == {**old_body, 'logprobs': 20, 'return_tokens_as_token_ids': True}
    assert list(new_body)[-2:] == ['logprobs', 'return_tokens_as_token_ids']
