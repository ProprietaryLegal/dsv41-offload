"""CPU-only concurrent HTTP probe, sent inline to x86host by the coordinator.

Tagged requests keep their original nonstreaming bodies. Three informational
controls stream sequentially, starting only after BOTH tagged captures have
written decode rows and while both HTTP requests remain in flight. Their client
delivery timings are never used by the isolated engine-histogram gate.
This module has no import-time effects and no service/GPU operations.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import math
import re
import shlex
from statistics import median
import subprocess
import time
from typing import cast
import urllib.error
import urllib.request


def check(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


class Skip(ValueError):
    """An informational control's precondition failed BEFORE it was sent: nothing was requested, so stop collecting
    informational rows and record why (the gated checks are elsewhere and unaffected)."""


def streamed(base: str, request: dict[str, object],
             tags: list[Future[dict[str, object]]]) -> dict[str, object]:
    body = dict(cast(dict[str, object], request['body']), stream=True,
                stream_options={'include_usage': True})
    with urllib.request.urlopen(base+'/metrics', timeout=3) as response:
        start_metrics = response.read().decode()
    values: dict[str, float] = {}
    for key in ('running', 'waiting'):
        found = [float(m[1]) for line in start_metrics.splitlines()
                 if (m := re.fullmatch(r'vllm:num_requests_'+key+r'(?:\{[^}]*\})?\s+(\S+)', line))]
        check(bool(found) and all(math.isfinite(v) and v >= 0 for v in found), 'invalid active gauges')
        values[key] = sum(found)
    if values != {'running': 2, 'waiting': 0}:
        raise Skip('engine was not running exactly the two tagged requests with nothing waiting')
    if not all(not f.done() for f in tags):
        raise Skip('tagged decoding ended before the informational control')
    started = time.monotonic()
    message: dict[str, object] = {}
    ids: list[int] = []
    stamps: list[float] = []
    content = ''
    finish: str | None = None
    req = urllib.request.Request(base+str(request['endpoint']), data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                break
            chunk = cast(dict[str, object], json.loads(data))
            check(not chunk.get('error'), 'informational HTTP stream error')
            if chunk.get('prompt_token_ids') is not None:
                message['prompt_token_ids'] = chunk['prompt_token_ids']
            if chunk.get('usage') is not None:
                message['usage'] = chunk['usage']
            for c in cast(list[dict[str, object]], chunk.get('choices', [])):
                token_ids = cast(list[int], c.get('token_ids') or [])
                if token_ids:
                    stamps.append(time.monotonic())
                    ids.extend(token_ids)
                content += str(c.get('text') or cast(dict[str, object], c.get('delta') or {}).get('content') or '')
                if c.get('prompt_token_ids') is not None:
                    message['prompt_token_ids'] = c['prompt_token_ids']
                if c.get('finish_reason'):
                    finish = str(c['finish_reason'])
    check(len(stamps) >= 2 and finish in ('stop', 'length'), 'incomplete informational stream')
    choice: dict[str, object] = {'finish_reason': finish, 'token_ids': ids, 'text': content,
              'message': {'content': content}, 'prompt_token_ids': message.get('prompt_token_ids')}
    message['choices'] = [choice]
    return {'response': message, 'started_monotonic': started,
            'both_tagged_inflight_at_start': True, 'start_metrics': start_metrics,
            'latency': {'ttft_s': stamps[0]-started,
                        'delivery_chunk_mean_s': (stamps[-1]-stamps[0])/(len(stamps)-1),
                        'delivery_chunks': len(stamps)}}


def group(base: str, requests: list[dict[str, object]], remote: str) -> dict[str, object]:
    def post(request: dict[str, object]) -> dict[str, object]:
        req = urllib.request.Request(base+str(request['endpoint']),
            data=json.dumps(request['body']).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                return cast(dict[str, object], json.load(response))
        except urllib.error.HTTPError as exc:
            return {'error': 'HTTP rejected', 'http_status': exc.code}

    # Inspect metadata only, never copy/read model or capture tensors through cache.
    probe = ('import json,pathlib; p=pathlib.Path('+repr(remote)+'); '
        'print(json.dumps({s:(p/s/"ledger.jsonl").stat().st_size '
        'if (p/s/"ledger.jsonl").exists() else 0 for s in ("smoke0","smoke1")}))')
    evidence: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        tags = [pool.submit(post, r) for r in requests[:2]]
        info: list[dict[str, object]] = []
        previous: dict[str, int] | None = None
        stopped: str | None = None
        for _ in range(3):
            deadline = time.monotonic()+30
            while stopped is None:
                if not all(not f.done() for f in tags):
                    stopped = 'tagged decoding ended before active-writer evidence'
                    break
                if time.monotonic() >= deadline:
                    stopped = 'active writer evidence timeout'
                    break
                sizes = cast(dict[str, int], json.loads(subprocess.check_output(
                    ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=3', '-o', 'StrictHostKeyChecking=yes',
                     'rank0user@10.202.28.2',
                     shlex.join(['python3', '-c', probe])], text=True, timeout=5)))
                evidence.append({'t': time.monotonic(), 'ledger_bytes': sizes})
                advanced = previous is not None and all(sizes[s] > previous[s] > 0 for s in sizes)
                previous = sizes
                if advanced:
                    break
                time.sleep(.05)
            if stopped is not None:
                break
            # At most three requests occupy the engine at once (K1 has four
            # sequence slots). Never turn the informational probe into queue load.
            try:
                row = streamed(base, requests[2], tags)
            except Skip as exc:
                stopped = str(exc)
                break
            row['writer_ledger_bytes_at_start'] = sizes
            info.append(row)
        tagged = [f.result(timeout=310) for f in tags]
    return {'tagged': tagged, 'informational': info, 'writer_activity': evidence,
            'informational_status': 'complete' if len(info) == 3 else 'incomplete: '+str(stopped),
            'informational_medians': {
                k: (median(float(cast(dict[str, float], row['latency'])[k]) for row in info) if info else None)
                for k in ('ttft_s', 'delivery_chunk_mean_s')},
            'timing_source': 'client streaming token-ID chunks; NOT engine steps; informational only',
            'gated': False}
