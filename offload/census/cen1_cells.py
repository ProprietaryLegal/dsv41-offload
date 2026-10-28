"""CEN1 request driver: build the public prompts once, then run ONE batch of a cell per invocation (coordinator only).

  python3 -m offload.census.cen1_cells                                   # plan, touches nothing
  python3 -m offload.census.cen1_cells --execute build --output cuts.json
  python3 -m offload.census.cen1_cells --execute batch --cell 32k-b4 --cuts cuts.json --seed 7 \
      --salt-prefix <tag> --deadline-utc <t> --output batch.json

offload/launch/windows/cen1_window.py calls it inside the HOLD through the owned tunnel 127.0.0.1:18001 and counts the
experiment's VLLM_EXL3_MOE_STATS lines between batches. A batch is the cell's B streams at once, MAX_TOKENS each with
ignore_eos (every stream decodes exactly MAX_TOKENS), a fresh seed per stream and ONE cache salt per cell, so the
prompt prefix is cached after the cell's first batch: CEN1 counts experts on decode calls, it never times anything.
Records numbers and hashes only: no prompt or generated text is written. Prompts are window_cells' public cuts
(offload/gb10_budget/window_cells.py: 'short' 1,024 and '32k-0..3' 32,000 tokens).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

from offload.gb10_budget import window_cells as wc
from offload.prefill_maxlen.evidence import Json, number, record
from offload.prefill_trace.window_requests import TEMPLATE, corpus

MAX_TOKENS = 300
CELLS: dict[str, tuple[str, ...]] = {'short-b1': ('short',), '32k-b1': ('32k-0',), '32k-b4': wc.B4}
ORDER = ('short-b1', '32k-b1', '32k-b4')          # 32k-b4 last: the first cell skipped when time runs short
DEPTHS: dict[str, tuple[int, ...]] = {'short-b1': tuple(range(6)), '32k-b1': tuple(range(6)), '32k-b4': tuple(range(5))}
BATCH_ESTIMATE_S = 120.0                         # one batch, eager (ESTIMATED; refused when it does not fit)


def speculative_k(batch: int) -> int:
    """serve_exp.sh SPEC: k = 5 at 1-2 streams, 4 at 3-4 streams."""
    if not 1 <= batch <= 4:
        raise ValueError('batch must be 1..4')
    return 5 if batch <= 2 else 4


def window_rows(batch: int) -> int:
    """Rows of one verify call (the MoE call's m): every stream's k drafts plus its bonus row."""
    return batch * (speculative_k(batch) + 1)


def live_rows(batch: int, depth: int) -> int:
    """Rows that route at depth d (DSparkDepthPolicy marks rows past it padding: top-k ids -1)."""
    if not 0 <= depth <= speculative_k(batch):
        raise ValueError(f'depth {depth} outside 0..k for {batch} streams')
    return batch * (depth + 1)


def body(model: str, text: str, seed: int, salt: str, max_tokens: int = MAX_TOKENS) -> dict[str, Json]:
    return {'model': model, 'messages': [{'role': 'user', 'content': text}], 'max_tokens': max_tokens,
            'ignore_eos': True, 'temperature': 1.0, 'top_p': 1.0, 'seed': seed, 'priority': 10, 'cache_salt': salt,
            'chat_template_kwargs': TEMPLATE, 'stream': True, 'stream_options': {'include_usage': True}}


def batch(api: wc.Api, cell: str, texts: dict[str, tuple[str, int]], seed: int, salt_prefix: str, deadline: float, *,
          wall: Callable[[], float] = time.time) -> dict[str, Json]:
    """One batch of `cell`: its streams at once; numbers only. Refuses a busy serve, a wrong cap or a short stream."""
    prompts = CELLS[cell]
    if wall() + BATCH_ESTIMATE_S > deadline:
        raise RuntimeError(f'window submission deadline reached: a {cell} batch does not fit')
    model = api.model()
    if model.get('max_model_len') != wc.CAP:
        raise ValueError('served cap differs from the production cap')
    if wc.busy(api.metrics()):
        raise RuntimeError('serve not idle before the batch')
    salt = wc.salt(salt_prefix, cell, 0)          # one salt per cell: the prefix is cached after the first batch

    def stream(i: int) -> dict[str, Json]:
        key = prompts[i]
        text, tokens = texts[key]
        summary = api.stream(body(str(model['id']), text, seed * 16 + i, salt), threading.Event())
        usage = record(summary['usage'])
        cached = number(record(usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0)
        if usage.get('prompt_tokens') != tokens:
            raise ValueError('usage differs from the built prompt')
        if usage.get('completion_tokens') != MAX_TOKENS or summary.get('finish') != 'length':
            raise ValueError('a stream did not decode exactly MAX_TOKENS (ignore_eos)')
        return {'stream': i, 'prompt': key, 'prompt_tokens': tokens, 'cached_tokens': cached,
                'completion_tokens': MAX_TOKENS, 'seed': seed * 16 + i,
                'prompt_sha256': hashlib.sha256(text.encode()).hexdigest()}

    started = wall()
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        streams = list(pool.map(stream, range(len(prompts))))
    return {'type': 'cen1-batch', 'cell': cell, 'batch': len(prompts), 'seed': seed, 'max_tokens': MAX_TOKENS,
            'cache_salt_sha256': hashlib.sha256(salt.encode()).hexdigest(), 'streams': cast(Json, streams),
            'wall_s': wall() - started,
            'status': 'MEASURED counts only; wall_s budgets the window and is never a speed (eager start)'}


def plan() -> dict[str, Json]:
    return {'executed': False, 'max_tokens': MAX_TOKENS, 'order': list(ORDER),
            'cells': {c: {'prompts': list(CELLS[c]), 'depths': list(DEPTHS[c]), 'window_rows': window_rows(len(CELLS[c])),
                          'live_rows': [live_rows(len(CELLS[c]), d) for d in DEPTHS[c]]} for c in ORDER}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--url', default=wc.URL)
    ap.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    sub = ap.add_subparsers(dest='command')
    b = sub.add_parser('build', help='construct every prompt once; numeric cuts and hashes only')
    b.add_argument('--output', type=Path, required=True)
    r = sub.add_parser('batch', help='one batch of one cell')
    r.add_argument('--cell', choices=ORDER, required=True)
    r.add_argument('--cuts', type=Path, required=True)
    r.add_argument('--seed', type=int, required=True)
    r.add_argument('--salt-prefix', required=True)
    r.add_argument('--deadline-utc', type=float, required=True)
    r.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(argv)
    if not args.execute or args.command is None:
        print(json.dumps(plan()))
        return 0
    if args.url != wc.URL:
        ap.error('coordinator must use the reserved experiment tunnel at 127.0.0.1:18001')
    if args.output.exists():
        ap.error('refusing an existing output; use a fresh window directory')
    materials, hashes = corpus(args.data_dir)
    api = wc.Http(args.url)
    if args.command == 'build':
        result = wc.build(api, materials, hashes)
    else:
        if args.deadline_utc <= time.time():
            ap.error('--deadline-utc is in the past')
        texts = wc.texts_from(record(cast(Json, json.loads(args.cuts.read_text()))), materials, hashes)
        result = batch(api, args.cell, texts, args.seed, args.salt_prefix, args.deadline_utc)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, allow_nan=False) + '\n')
    print(f'{args.command}: wrote {args.output.name} (numbers and hashes only)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
