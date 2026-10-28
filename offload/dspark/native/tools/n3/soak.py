""">=2000 actual protocol rounds, all B/n_out/margin graphs, dump on/off equality.

No target/GB10 traffic. Complete TP2 emulation on the development card or the
real pair, selected by DSPARK_WORKER_TP_EMULATE. Expanded emulation is refused
before allocation: two ~17GiB shards cannot fit a 24GiB card.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from ....worker import protocol as P
from ....worker.server import build_core_tp2
from ....worker.tools.server_selftest import taps_for
from .wire import Client
from ....worker.v1_dump import V1Dump, DumpTP2Backend
from .shard import memory, phase


def rounds(core: Any, taps: np.ndarray, tokens: np.ndarray, count: int, session: int) -> dict[str, Any]:
    cl = Client(core)
    exp = core.expected
    # Try the K2 identity and an arbitrary native mismatch before the valid HELLO.
    for identity in (bytes.fromhex('be2c1436fa9f9e66b398e1b1da2e140e6fddc82a11939e2ac4240a0ca79543bd'), bytes(32)):
        if identity == exp.drafter_identity:
            continue
        h, _ = cl.call(P.OP_HELLO, P.Hello(**{**exp.__dict__, 'drafter_identity': identity}).encode(), session=session)
        assert h.status == P.IDENTITY_MISMATCH
    h, _ = cl.call(P.OP_HELLO, P.Hello(**{**exp.__dict__, 'num_spec': 4}).encode(), session=session)
    assert h.status == P.CONFIG_MISMATCH
    h, _ = cl.call(P.OP_HELLO, exp.encode(), session=session)
    assert h.status == P.OK
    digest = hashlib.sha256()
    pos = [300, 500, 700, 900]
    gen = [0] * 4
    latencies = []
    cases: dict[str, int] = {}
    for i in range(count):
        B, n, margin = 1 + (i // 10) % 4, 1 + i % 5, bool((i // 5) % 2)
        entries, rowparts = [], []
        for slot in range(B):
            fresh = gen[slot] == 0 or i % 64 == 0
            nrows = 128 if fresh else 1 + (i + slot) % 6
            first = pos[slot] - nrows if fresh else pos[slot]
            anchor = pos[slot] if fresh else pos[slot] + nrows
            if fresh:
                gen[slot] += 1
            flags = P.REQ_NEED_DRAFT | (P.REQ_NEW_GEN | P.REQ_RESYNC if fresh else 0)
            entries.append(P.ReqEntry(slot, flags, n, gen[slot], first, nrows,
                                      int(tokens[anchor % len(tokens)]), anchor, n_out=n))
            rowparts.append(taps[np.arange(first, first+nrows) % len(taps)].tobytes())
            pos[slot] = anchor
        ts = time.monotonic()
        h, body = cl.call(P.OP_STEP, P.encode_step(i + 1, entries, b''.join(rowparts), exp.row_bytes,
                           P.STEP_WANT_MARGIN if margin else 0), session=session)
        assert h.status == P.OK
        r = P.decode_step_reply(body)
        assert all(s[0] == P.R_OK and s[1] == 128 for s in r.statuses), r.statuses
        assert np.isfinite(r.conf).all() and ((r.conf >= 0) & (r.conf <= 1)).all()
        assert ((r.draft_ids[:, :n] >= 0) & (r.draft_ids[:, :n] < exp.vocab_size)).all()
        assert (r.draft_ids[:, n:] == 0).all() and (r.conf[:, n:] == 0).all()
        for arr in (r.draft_ids, r.conf, r.margin):
            if arr is not None:
                digest.update(arr.tobytes())
        digest.update(json.dumps(r.statuses).encode())
        latencies.append((time.monotonic() - ts) * 1000)
        key = f'B{B}_n{n}_margin{int(margin)}'
        cases[key] = cases.get(key, 0) + 1
        if (i+1) % 500 == 0:
            print(f'{i+1}/{count} protocol rounds', flush=True)
    h, body = cl.call(P.OP_STATS, session=session)
    assert h.status == P.OK
    stats = json.loads(body)
    assert stats.get('internal_errors', 0) == 0 and stats.get('tp_comm_errors', 0) == 0, stats
    xs = sorted(latencies)
    return {'rounds': count, 'digest': digest.hexdigest(), 'cases': cases,
            'latency_ms': {'p50': xs[len(xs)//2], 'p99': xs[int(len(xs)*.99)]}, 'stats': stats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('arm')
    ap.add_argument('--rounds', type=int, default=2000)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--root', default='/opt/dsv41/x86host-data/dspark')
    a = ap.parse_args()
    if a.rounds < 2000:
        raise ValueError('at least 2000 protocol rounds per pass')
    if a.arm == 'native-expanded' and os.environ.get('DSPARK_WORKER_TP_EMULATE') == '1':
        raise RuntimeError('native-expanded full TP2 emulation needs >32GiB; use both solo shard checks and coordinator pair soak')
    os.environ['DSPARK_WORKER_ARM'] = a.arm
    if os.environ.get('DSPARK_V1_DUMP_DIR'):
        raise ValueError('soak owns its on/off phases; remove DSPARK_V1_DUMP_DIR')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    marker = a.out.with_suffix('.phase.json')
    phase(marker, 'load')
    start = time.monotonic()
    # Pre-context peaks include the full real server factory, identity verification and prewarm.
    core, model = build_core_tp2(a.root, max_model_len=1048576)
    m = cast(Any, model)
    devs = list(dict.fromkeys(m.devs))
    rep: dict[str, Any] = {'arm': a.arm, 'mode': 'emu' if m.emulated else 'pair',
        'startup_s': time.monotonic()-start, 'startup_memory': {str(d): memory(d) for d in devs},
        'drafter_identity': core.expected.drafter_identity.hex(), 'markov_identity': core.expected.markov_identity.hex()}
    phase(marker, 'serve-off')
    generator = torch.Generator(device=m.dev).manual_seed(7)
    toks = torch.randint(0, 120000, (1600,), device=m.dev, generator=generator)
    taps = taps_for(type('Embed', (), {'embed': m.ranks[0].embed})(), toks, generator)
    tokens = toks.cpu().numpy()
    for d in devs:
        torch.cuda.reset_peak_memory_stats(d)
    rep['off'] = rounds(core, taps, tokens, a.rounds, 0x4e330001)
    rep['serving_off_memory'] = {str(d): memory(d) for d in devs}
    # Install observers and recapture on the same weights. No kernel source/math changes.
    phase(marker, 'prewarm-dump')
    sink = V1Dump(m, core.backend.arm_identity, str(a.out.with_suffix('.dumps')), '1,2,3,4,5,6,7,8,9,10,40')
    m.capture(batch_sizes=(1,2,3,4), ks=(5,), n_outs=(1,2,3,4,5), insert_rows=(8,16,32,64,128))
    backend = DumpTP2Backend(m, core.expected.max_rows_per_step, core.expected.row_bytes, sink=sink)
    backend.arm_identity = core.backend.arm_identity
    core.backend = backend
    phase(marker, 'serve-on')
    rep['on'] = rounds(core, taps, tokens, a.rounds, 0x4e330002)
    assert rep['off']['digest'] == rep['on']['digest'], 'dump mode changed protocol outputs'
    rep['dump_on_off_exact'] = True
    rep['dump_count'] = len(sink.files)
    rep['serving_on_memory'] = {str(d): memory(d) for d in devs}
    phase(marker, 'done')
    a.out.write_text(json.dumps(rep, indent=2)+'\n')
    print(f'DONE {a.arm}: {a.rounds} off + {a.rounds} on, exact, {len(sink.files)} dumps', flush=True)


if __name__ == '__main__':
    main()
