"""S26 probe: resume after an abort whose last engine step never reached the driver.

Real Capture writer (CPU tensors) -> durable FLUSH -> real decode_dataset.capture_rows, on a synthetic sequence.

The generator aborts by shutting its socket down (generate.Stream.abort) and keeps only the token ids it has read.
The engine has by then run at least one more step for the request (async output, abort noticed later), so its
validated rows - accepted drafts the driver never received - are already on file. The resumed segment samples new
tokens at those positions; first-row-wins keeps the stale rows, and assembly refuses the whole sequence.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import pytest
import torch

from d2 import decode_dataset as DD

PATH = Path(__file__).resolve().parents[2] / 'capture/decode_capture.py'
spec = importlib.util.spec_from_file_location('s26_decode_capture', PATH)
assert spec and spec.loader
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)

PROMPT = 300
R0 = PROMPT - 128


def tap(token: int, pos: int, layer: int) -> torch.Tensor:
    # Exact in bf16, never zero, depends on the (token, position) the row was computed for.
    return torch.full((5120,), float((token * 7 + pos * 3 + layer) % 200 + 1), dtype=torch.bfloat16)


def step(rid: str, toks: dict[int, int], first: int, n: int, rejected: int, capture) -> None:
    """One engine round for one tagged request: rows first..first+n-1 computed, last `rejected` rejected."""
    pos = list(range(first, first + n))
    ids = [toks.get(p, 99) for p in pos]           # rejected draft slots carry some token
    aux = [torch.stack([tap(t, p, L) for t, p in zip(ids, pos)]) for L in range(3)]
    ib = types.SimpleNamespace(req_ids=[rid], num_reqs=1, num_tokens=n,
                               query_start_loc_np=np.array([0, n], dtype=np.int32),
                               input_ids=torch.tensor(ids, dtype=torch.int32),
                               positions=torch.tensor(pos, dtype=torch.int64))
    capture.enqueue(ib, aux, torch.tensor([rejected]))


def flush(root: Path, capture) -> None:
    (root / 'FLUSH.json').write_text(json.dumps({'nonce': 's26'}))
    deadline = time.monotonic() + 10
    while not (root / 'FLUSHED.json').exists() and time.monotonic() < deadline:
        time.sleep(.02)
    capture.close()


def run(root: Path, driver_received_step2: bool) -> tuple[list[int], int]:
    old = {p: 1000 + p for p in range(PROMPT)} | {p: 5000 + p for p in range(PROMPT, 320)}
    c = C.Capture(str(root))
    rid0 = f'cmpl-d2c.pX.0.{R0}-0-1a2b3c4d'
    step(rid0, old, 0, PROMPT, 0, c)            # prefill; samples old[300]
    step(rid0, old, 300, 6, 3, c)               # rows 300..302 valid; bonus old[303]  -> driver receives
    step(rid0, old, 303, 6, 2, c)               # rows 303..306 valid; bonus old[307]  -> driver may not receive
    kept = 307 if driver_received_step2 else 303
    final = [old[p] for p in range(kept + 1)]   # driver's tokens: everything it read, through position `kept`
    # Resume: prompt = final[0:kept+1]; new samples differ from the undelivered old ones.
    new = {p: final[p] for p in range(kept + 1)} | {p: 7000 + p for p in range(kept + 1, kept + 40)}
    rid1 = f'cmpl-d2c.pX.1.{kept + 1 - 128}-0-beef0001'
    step(rid1, new, kept + 1 - 48, 48, 0, c)    # prefix-cache hit then the resume prefill tail
    step(rid1, new, kept + 1, 6, 0, c)          # all 5 drafts accepted
    step(rid1, new, kept + 7, 6, 5, c)          # only the anchor row valid; the next bonus is the final sample
    L_gen = kept + 9
    tokens = [new[p] for p in range(L_gen)]
    flush(root, c)
    return tokens, L_gen


def test_control_resume_without_lag_assembles(tmp_path: Path) -> None:
    tokens, L_gen = run(tmp_path, driver_received_step2=True)
    taps, meta = DD.capture_rows(tmp_path, 'pX', R0, L_gen - 1, tokens)
    assert taps.shape[0] == L_gen - 1 - R0 and meta['dup_rows_differing'] == 0


def test_FIX_resume_after_undelivered_step_assembles(tmp_path: Path) -> None:
    tokens, L_gen = run(tmp_path, driver_received_step2=False)
    ledger = [json.loads(x) for x in (tmp_path / 'pX/ledger.jsonl').read_text().splitlines()]
    assert sum(x['dup_rows_differing'] for x in ledger) == 3     # rows 304..306: old drafts vs new samples
    taps, _ = DD.capture_rows(tmp_path, 'pX', R0, L_gen - 1, tokens)
    # What correct capture must give: every row computed for the FINAL token at that position.
    want = torch.stack([torch.cat([tap(tokens[p], p, L) for L in range(3)]) for p in range(R0, L_gen - 1)])
    assert torch.equal(taps, want)
