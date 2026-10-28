"""S26: how long may the capture writer stall before the whole capture latches off? (real Capture, CPU tensors)

Production shape: 3 tagged D2 streams, one stage per stream per engine round. ~31 tok/s per stream (MEASURED,
first-pass gen/seqs) at 1.54-1.78 accepted drafts per round (COMPARISON.md) gives ~11-12 rounds/s (ESTIMATED).
pending=8 stages is then 2.7 rounds of slack (~0.22 s) - less on GB10, where the writer also waits for each
stage's copy event. One longer writer stall refuses every later row of the serve's lifetime; the generator is not
told and keeps generating.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time
import types

import numpy as np
import pytest
import torch

PATH = Path(__file__).resolve().parents[2] / 'capture/decode_capture.py'
spec = importlib.util.spec_from_file_location('s26_stall_capture', PATH)
assert spec and spec.loader
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)

RATE = 12.0          # rounds per second (ESTIMATED production)
STREAMS = 3


def drive(root: Path, rounds: int, stall_at: int, stall_s: float) -> tuple[C.Capture, float, int]:
    c = C.Capture(str(root))
    orig = C.SequenceFiles.write
    seen = {'n': 0}

    def slow(self, s):
        seen['n'] += 1
        if seen['n'] == stall_at * STREAMS:
            time.sleep(stall_s)          # e.g. an fdatasync / reclaim / disk hiccup in the writer thread
        return orig(self, s)

    C.SequenceFiles.write = slow
    pos = {k: 200 for k in range(STREAMS)}
    worst = 0.0
    try:
        t_next = time.perf_counter()
        for r in range(rounds):
            reqs, ids, p_all, qsl = [], [], [], [0]
            for k in range(STREAMS):
                n = 6
                reqs.append(f'cmpl-d2c.s{k}.0.72-0-{r:08x}')
                p_all += list(range(pos[k], pos[k] + n))
                ids += [7] * n
                qsl.append(len(ids))
            aux = [torch.full((len(ids), 5120), 1.0, dtype=torch.bfloat16) for _ in range(3)]
            ib = types.SimpleNamespace(req_ids=reqs, num_reqs=STREAMS, num_tokens=len(ids),
                                       query_start_loc_np=np.array(qsl, dtype=np.int32),
                                       input_ids=torch.tensor(ids, dtype=torch.int32),
                                       positions=torch.tensor(p_all, dtype=torch.int64))
            rej = torch.tensor([3] * STREAMS)       # 3 valid rows per round (anchor + ~2 accepted)
            t0 = time.perf_counter()
            c.enqueue(ib, aux, rej)
            worst = max(worst, time.perf_counter() - t0)
            for k in range(STREAMS):
                pos[k] += 3
            t_next += 1.0 / RATE
            time.sleep(max(0.0, t_next - time.perf_counter()))
        c.close()
    finally:
        C.SequenceFiles.write = orig
    rows = sum(1 for k in range(STREAMS) for _ in (root / f's{k}/ledger.jsonl').read_text().splitlines())
    return c, worst, rows


def test_short_stall_is_absorbed(tmp_path: Path) -> None:
    c, worst, rows = drive(tmp_path, 40, 10, 0.10)
    assert c.errors == 0 and rows == 40 * STREAMS


def test_one_longer_stall_is_absorbed_by_byte_budget(tmp_path: Path) -> None:
    c, worst, rows = drive(tmp_path, 60, 10, 0.40)
    print(f'S26 MEASURED: stall 0.40 s at round 10 of 60 -> errors={c.errors}, ledger rows {rows} of '
          f'{60 * STREAMS}, worst enqueue {1e3 * worst:.2f} ms')
    assert c.errors == 0 and not c.faulted.is_set()
    assert rows == 60 * STREAMS
    assert not (tmp_path / 'ERRORS.jsonl').exists()
    assert c.budget.peak < c.budget.limit
    assert worst < 0.05                              # the serving side never waited: production-safe, data-unsafe


def test_FIX_one_overflow_voids_sequences_captured_before_it(tmp_path: Path) -> None:
    """S26 FIX 2 (with the latch above): a single staging overflow, scoped to one round, makes capture_rows refuse
    EVERY sequence in the tree - including one captured completely and correctly before the fault."""
    import json
    from d2 import decode_dataset as DD

    c = C.Capture(str(tmp_path), staging_bytes=16 * 1024 * 1024)
    toks = list(range(1000, 1400))
    def one(rid: str, first: int, n: int, rejected: int, wait: bool = True) -> None:
        pos = list(range(first, first + n))
        aux = [torch.stack([torch.full((5120,), float(p % 97 + 1 + L), dtype=torch.bfloat16) for p in pos])
               for L in range(3)]
        ib = types.SimpleNamespace(req_ids=[rid], num_reqs=1, num_tokens=n,
                                   query_start_loc_np=np.array([0, n], dtype=np.int32),
                                   input_ids=torch.tensor([toks[p] for p in pos], dtype=torch.int32),
                                   positions=torch.tensor(pos, dtype=torch.int64))
        c.enqueue(ib, aux, torch.tensor([rejected]))
        deadline = time.monotonic() + 5
        while wait and c.budget.used != 0 and time.monotonic() < deadline:   # writer drains (staging_bytes=4 * 1024 * 1024)
            time.sleep(.005)
    one('cmpl-d2c.pA.0.172-0-aaaa0001', 0, 300, 0)                 # A: prompt 300 tokens
    one('cmpl-d2c.pA.0.172-0-aaaa0001', 300, 4, 0)                 # A: decode rows 300..303; final sample 304
    assert c.budget.acquire(c.budget.limit, blocking=False)                          # writer momentarily behind ...
    one('cmpl-d2c.pB.0.0-0-bbbb0001', 0, 3, 0, False)                    # ... B's stage overflows: 'staging exhausted'
    c.budget.release(c.budget.limit)
    (tmp_path / 'FLUSH.json').write_text(json.dumps({'nonce': 'n'}))
    deadline = time.monotonic() + 10
    while not (tmp_path / 'FLUSHED.json').exists() and time.monotonic() < deadline:
        time.sleep(.02)
    c.close()
    assert c.errors == 1 and 'staging exhausted' in (tmp_path / 'ERRORS.jsonl').read_text()
    # A is complete: 172..303 covered, ids exact. Today: ValueError 'capture writer recorded errors; data refused'.
    taps, _ = DD.capture_rows(tmp_path, 'pA', 172, 304, toks[:305])
    assert taps.shape[0] == 304 - 172
