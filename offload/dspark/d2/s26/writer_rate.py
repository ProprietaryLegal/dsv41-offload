"""S26 measurement: the real Capture writer on a real disk at the estimated production rate (CPU, no CUDA).

python -m d2.s26.writer_rate --dir <scratch> --seconds 300
Prints JSON: per-stage writer time distribution, fdatasync times, errors. A proxy for GB10 only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch

PATH = Path(__file__).resolve().parents[2] / 'capture/decode_capture.py'
spec = importlib.util.spec_from_file_location('s26_rate_capture', PATH)
assert spec and spec.loader
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', type=Path, required=True)
    ap.add_argument('--seconds', type=float, default=300)
    ap.add_argument('--rate', type=float, default=12.0)
    ap.add_argument('--stagger-s', type=float, default=0.0, help='stream k starts k*stagger seconds late')
    a = ap.parse_args()
    a.dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    writes: list[float] = []
    slow: list[tuple[float, float]] = []
    t_start = time.perf_counter()
    syncs: list[float] = []
    ow, osync = C.SequenceFiles.write, C.SequenceFiles.sync

    def tw(self, s):
        t = time.perf_counter()
        try:
            return ow(self, s)
        finally:
            d = time.perf_counter() - t
            writes.append(d)
            if d > 0.05:
                slow.append((round(t - t_start, 2), round(1e3 * d, 1)))

    def ts(self):
        t = time.perf_counter()
        try:
            return osync(self)
        finally:
            syncs.append(time.perf_counter() - t)
            slow.append((round(t - t_start, 2), -round(1e3 * syncs[-1], 1)))

    C.SequenceFiles.write, C.SequenceFiles.sync = tw, ts
    c = C.Capture(str(a.dir))
    pos = [200, 200, 200]
    gen = torch.Generator().manual_seed(1)
    t_end = time.perf_counter() + a.seconds
    t_next = time.perf_counter()
    r = 0
    fault_t = None
    expected_stages = 0
    while time.perf_counter() < t_end:
        if fault_t is None and c.errors:
            fault_t = round(time.perf_counter() - t_start, 2)
        n = 6
        ids, p_all, qsl = [], [], [0]
        live = [k for k in range(3) if time.perf_counter() - t_start >= k * a.stagger_s]
        for k in live:
            p_all += list(range(pos[k], pos[k] + n))
            ids += [7] * n
            qsl.append(len(ids))
        aux = [torch.randn(len(ids), 5120, generator=gen).to(torch.bfloat16) for _ in range(3)]
        ib = types.SimpleNamespace(req_ids=[f'cmpl-d2c.s{k}.0.72-0-{r:08x}' for k in live], num_reqs=len(live),
                                   num_tokens=len(ids), query_start_loc_np=np.array(qsl, dtype=np.int32),
                                   input_ids=torch.tensor(ids, dtype=torch.int32),
                                   positions=torch.tensor(p_all, dtype=torch.int64))
        expected_stages += len(live)
        c.enqueue(ib, aux, torch.tensor([3] * len(live)))
        for k in live:
            pos[k] += 3
        r += 1
        t_next += 1.0 / a.rate
        time.sleep(max(0.0, t_next - time.perf_counter()))
    c.close()
    w = sorted(writes)
    q = lambda f: round(1e3 * w[min(len(w) - 1, int(f * len(w)))], 3)  # noqa: E731
    print(json.dumps({'fault_t_s': fault_t, 'stagger_s': a.stagger_s, 'slow_events_t_ms(neg=sync)': slow, 'rounds': r, 'stages': len(w), 'expected_stages': expected_stages, 'peak_staging_bytes': c.budget.peak, 'staging_budget_bytes': c.budget.limit, 'errors': c.errors, 'write_ms_p50': q(.5), 'write_ms_p99': q(.99),
                      'write_ms_max': q(1.0), 'sync_ms': [round(1e3 * s, 1) for s in syncs],
                      'bytes_per_seq': (a.dir / 's0/taps.bin').stat().st_size}))


if __name__ == '__main__':
    main()
