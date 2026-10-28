"""S32 item 4: generator-side cost of option-ii rows on Controller CPU (synthetic rows; prints aggregates only).

Usage (from offload/dspark): python3 -I d2/s32/cost_probe.py SCRATCH_DIR
Measures the driver's own atomic_json (json.dump -> pure-Python encoder) for one full-length 16,384-row state,
json.dumps for comparison, validate_saved, and the Python heap per saved row.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import tracemalloc

sys.path.insert(0, '.')
from d2 import generate as G  # noqa: E402
from d2 import generation_top20 as T  # noqa: E402


def row(rng: random.Random) -> T.Row:
    ids = rng.sample(range(129280), 20)
    vals = [-0.1 - rng.random() * 1e-3] + sorted([-6 - rng.random() * 14 for _ in range(19)], reverse=True)
    w = {f'token_id:{k}': v for k, v in zip(ids, vals)}
    return T.row(ids[0], f'token_id:{ids[0]}', w[f'token_id:{ids[0]}'], w)


def main() -> None:
    scratch = sys.argv[1]
    rng = random.Random(32)
    tracemalloc.start()
    rows = [row(rng) for _ in range(16384)]
    heap = tracemalloc.get_traced_memory()[0] / len(rows)
    tracemalloc.stop()
    toks = [r['token'] for r in rows]
    state = {'schema': G.SCHEMA, 'prompt_ids': list(range(3000)), 'gen_ids': toks, 'top20': 'generation',
             'logprob_rows': rows, 'segments': [], 'errors': []}
    path = os.path.join(scratch, 's32-cost.json')
    dump = []
    for _ in range(3):
        t = time.perf_counter()
        G.atomic_json(path, state)
        dump.append(time.perf_counter() - t)
    t = time.perf_counter()
    json.dumps(state)
    dumps = time.perf_counter() - t
    size = os.path.getsize(path)
    t = time.perf_counter()
    G.atomic_json(path, dict(state, logprob_rows=[]))
    off = time.perf_counter() - t
    os.remove(path)
    t = time.perf_counter()
    T.validate_saved(toks, rows)
    val = time.perf_counter() - t
    print(json.dumps({'schema': 's32.cost.v1', 'classification': 'MEASURED (synthetic rows, Controller CPU)',
                      'rows': len(rows), 'atomic_json_s': [round(x, 3) for x in dump], 'json_dumps_s': round(dumps, 3),
                      'off_mode_atomic_json_s': round(off, 4), 'state_bytes': size,
                      'bytes_per_row': round(size / len(rows)), 'validate_saved_us_per_row': round(val / len(rows) * 1e6, 1),
                      'python_heap_bytes_per_row': round(heap)}))


if __name__ == '__main__':
    main()
