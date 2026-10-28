"""WIN2-R3 smoke start (F-METHOD §3.2 rule 1): prove a candidate arm starts and serves before it costs a window.

One smoke = start the arm with start_exp.sh, prove it healthy (/health + /v1/models), one greedy prompt, one 64-token
seeded request whose server counters must move (speculative rounds > 0), then ALWAYS stop the arm. MEASURED arm start
6.8 min + ESTIMATED ~1.5 min -> ~8.5 min; the bound is SMOKE_BUDGET_S.

Two modes:
- chain-head: in a chain (production already down) the coordinator smokes every candidate of the chain back to back at
  the chain head; a failing candidate is dropped before its measurement window (`chain_head`).
- x-first: outside a chain the smoke is the window's FIRST start (candidate first); a failure ends the window before any
  control start (`x_first`; the runner turns FAIL into its stop).
Smoke windows never count as measurement windows (window_cap) and never write to the control archive.

The host side is the runner's own Live backend (it already owns start_exp.sh, scopes, fences and the HOLD): it implements
`Backend`. This module holds the shared sequence, timing and verdict only; it contacts nothing itself.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, cast

from offload.prefill_maxlen.evidence import Json, number, record

SMOKE_SCHEMA = 'dsv41-smoke-v1'
SMOKE_SEED = 990_001
SMOKE_TOKENS = 64
SMOKE_BUDGET_S = 15 * 60.0          # 6.8 min MEASURED start + ~1.5 min checks, with margin
COORDINATOR = 'Deepseek/coordinator'
STAGES = ('start', 'healthy', 'greedy', 'seeded', 'stop')


class Backend(Protocol):
    """The runner's Live adapter for one arm (start_exp.sh inside the runner's scopes and HOLD)."""
    def start(self, arm: str) -> None: ...
    def healthy(self, arm: str) -> dict[str, Json]: ...                # /health + /v1/models proof of the arm
    def greedy(self, arm: str) -> dict[str, Json]: ...                 # one registered greedy prompt: digest + tokens
    def seeded(self, arm: str, body: dict[str, Json]) -> dict[str, Json]: ...   # one chat completion, usage + digest
    def metrics(self, arm: str) -> str: ...
    def stop(self, arm: str) -> None: ...


def seeded_body(model: str) -> dict[str, Json]:
    return {'model': model, 'messages': [{'role': 'user', 'content': 'Summarize the duties of a guardian ad litem.'}],
            'max_tokens': SMOKE_TOKENS, 'temperature': 1.0, 'top_p': 1.0, 'seed': SMOKE_SEED, 'ignore_eos': True,
            'priority': 10, 'chat_template_kwargs': {'thinking': False}}


def _drafts(text: str) -> float:
    found = [float(line.split()[-1]) for line in text.splitlines()
             if line.startswith(('vllm:spec_decode_num_drafts_total{', 'vllm:spec_decode_num_drafts_total '))]
    if not found:
        raise ValueError('metrics lack vllm:spec_decode_num_drafts_total')
    return sum(found)


def smoke(backend: Backend, arm: str, *, clock: Callable[[], float] = time.monotonic,
          budget_s: float = SMOKE_BUDGET_S) -> dict[str, Json]:
    """One smoke of one arm. PASS needs every stage; the arm is ALWAYS stopped. A stop failure raises after the result
    is recorded in the exception text (the runner keeps its HOLD and restores, as for any stop)."""
    if not (math.isfinite(budget_s) and budget_s > 0) or not arm.strip():
        raise ValueError('smoke needs an arm name and a finite positive budget')
    begin = clock()
    timings: dict[str, Json] = {}
    result: dict[str, Json] = {'schema': SMOKE_SCHEMA, 'arm': arm, 'status': 'FAIL', 'stage': 'start',
                               'timings_s': timings}

    def stage(name: str) -> None:
        result['stage'] = name
        if clock() - begin > budget_s:
            raise TimeoutError(f'smoke budget {budget_s:g} s exhausted before {name}')

    started = False
    try:
        stage('start')
        started = True                      # start_exp.sh may mutate the Sparks even when its reply is lost
        backend.start(arm)
        timings['start'] = clock() - begin
        stage('healthy')
        served = backend.healthy(arm)
        model = served.get('id')
        if not isinstance(model, str) or not model:
            raise ValueError('healthy proof lacks the served model id')
        result['served'] = {k: served[k] for k in ('id', 'max_model_len') if k in served}
        timings['healthy'] = clock() - begin
        stage('greedy')
        greedy = backend.greedy(arm)
        if not isinstance(greedy.get('sha256'), str) or number(greedy.get('tokens', 0)) <= 0:
            raise ValueError('greedy prompt returned no output digest or tokens')
        result['greedy'] = {'sha256': greedy['sha256'], 'tokens': greedy['tokens']}
        timings['greedy'] = clock() - begin
        stage('seeded')
        before = _drafts(backend.metrics(arm))
        summary = backend.seeded(arm, seeded_body(model))
        rounds = _drafts(backend.metrics(arm)) - before
        usage = record(summary['usage'])
        if usage.get('completion_tokens') != SMOKE_TOKENS or rounds <= 0:
            raise ValueError(f'seeded request: {usage.get("completion_tokens")} tokens, {rounds:g} speculative rounds')
        result['seeded'] = {'seed': SMOKE_SEED, 'tokens': SMOKE_TOKENS, 'rounds': rounds,
                            'output_sha256': summary.get('output_sha256')}
        timings['seeded'] = clock() - begin
        result['status'] = 'PASS'
    except (Exception, KeyboardInterrupt) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if started:
            try:
                backend.stop(arm)
                timings['stop'] = clock() - begin
            except (Exception, KeyboardInterrupt) as exc:
                result['status'] = 'FAIL'
                result['stop_error'] = f'{type(exc).__name__}: {exc}'
                raise RuntimeError(f'smoke of {arm}: arm stop failed; HOLD stays, restore: {json.dumps(result)}') from exc
    if result['status'] == 'PASS':
        result['stage'] = 'done'
    return result


def chain_head(backend: Backend, arms: Sequence[str], *, clock: Callable[[], float] = time.monotonic,
               budget_s: float = SMOKE_BUDGET_S) -> dict[str, Json]:
    """Chain mode: smoke every candidate back to back at the chain head; failures are dropped before their windows."""
    if not arms or len(set(arms)) != len(arms):
        raise ValueError('chain-head smoke needs distinct candidate arms')
    results = [smoke(backend, arm, clock=clock, budget_s=budget_s) for arm in arms]
    return {'schema': SMOKE_SCHEMA, 'mode': 'chain-head', 'results': cast(Json, results),
            'passed': [str(r['arm']) for r in results if r['status'] == 'PASS'],
            'dropped': [str(r['arm']) for r in results if r['status'] != 'PASS']}


def x_first(backend: Backend, arm: str, *, clock: Callable[[], float] = time.monotonic,
            budget_s: float = SMOKE_BUDGET_S) -> dict[str, Json]:
    """In-window mode: the smoke is the first start (candidate first); FAIL ends the window before any control start."""
    result = smoke(backend, arm, clock=clock, budget_s=budget_s)
    return {'schema': SMOKE_SCHEMA, 'mode': 'x-first', 'result': result,
            'window': 'CONTINUE' if result['status'] == 'PASS' else 'END: smoke failed before any control start'}


def plan(experiment: str, arms: Sequence[str], mode: str) -> str:
    return (f'DRY RUN smoke start ({mode}) for {experiment}: arms {", ".join(arms)}. Per arm: start_exp.sh (MEASURED 6.8 min), '
            '/health + /v1/models, one greedy prompt, one 64-token seeded request (seed '
            f'{SMOKE_SEED}; speculative rounds must move), stop the arm ALWAYS. ~8.5 min per arm, bound {SMOKE_BUDGET_S:g} s. '
            'chain-head: failing candidates are dropped before their windows. x-first: a failure ends the window before any '
            'control start. Execution uses the runner Live backend (--backend module:factory); this tool contacts nothing.')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('experiment')
    ap.add_argument('arm', nargs='+')
    ap.add_argument('--mode', choices=('chain-head', 'x-first'), default='x-first')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--backend', help="runner adapter 'module:factory'; factory(experiment, arm_list, out) -> Backend")
    ap.add_argument('--out', type=Path)
    args = ap.parse_args(argv)
    if args.mode == 'x-first' and len(args.arm) != 1:
        ap.error('x-first smokes exactly one candidate arm')
    if not args.execute:
        print(plan(args.experiment, args.arm, args.mode))
        return 0
    if os.environ.get('WINDOW_COORDINATOR') != COORDINATOR:
        print(f'STOP: smoke start is coordinator-only (WINDOW_COORDINATOR={COORDINATOR})', file=sys.stderr)
        return 2
    if args.backend is None or ':' not in args.backend or args.out is None or args.out.exists():
        print('STOP: --execute needs --backend module:factory and a fresh --out', file=sys.stderr)
        return 2
    module, factory = args.backend.split(':', 1)
    backend = cast(Backend, getattr(importlib.import_module(module), factory)(args.experiment, list(args.arm), args.out))
    if args.mode == 'chain-head':
        result = chain_head(backend, args.arm)
        passed = cast(list[Json], result['passed'])
    else:
        result = x_first(backend, args.arm[0])
        passed = [args.arm[0]] if result['window'] == 'CONTINUE' else []
    args.out.mkdir(parents=True)
    text = json.dumps(result, indent=1, allow_nan=False) + '\n'
    (args.out / 'smoke.json').write_text(text)
    print(json.dumps({'mode': args.mode, 'passed': passed, 'sha256': hashlib.sha256(text.encode()).hexdigest()}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
