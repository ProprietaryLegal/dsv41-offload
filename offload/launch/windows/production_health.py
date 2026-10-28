"""Bounded, read-only health evidence for the priority-20 post-close probe.

An admission timeout with active/unknown traffic is inconclusive. Only repeated
failure while independently idle (or wholly unreachable) permits a stop. The
same confirmation runs on the receiving host immediately before stop.sh.
"""
from __future__ import annotations

from collections.abc import Callable
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import cast
import urllib.request

PROBE_TIMEOUT_S = 10
CONFIRM_S = 5
PRODUCTION = 'http://HEAD_IP:8000'
MODEL_ROOT = '/srv/dsv41/legal-v2a'


class WatchError(RuntimeError):
    """Failed proof; retain HOLD."""


class BusyHealth(WatchError):
    """Healthy/active or insufficient idle evidence; never authorize a stop."""


def require(ok: bool, detail: str) -> None:
    if not ok:
        raise WatchError(detail)


def object_record(value: object, context: str) -> dict[str, object]:
    require(isinstance(value, dict) and all(isinstance(k, str) for k in value), f'{context}: expected JSON object')
    return cast(dict[str, object], value)


def requests(text: object) -> float:
    require(isinstance(text, str), 'metrics is not text')
    totals = []
    for key in ('vllm:num_requests_running', 'vllm:num_requests_waiting'):
        values = re.findall(r'^' + re.escape(key) + r'(?:\{[^\n]*\})?\s+([^\s]+)\s*$', cast(str, text), re.M)
        require(bool(values), 'metrics lacks request gauges')
        numbers = [float(value) for value in values]
        require(all(math.isfinite(value) and value >= 0 for value in numbers), 'invalid request gauges')
        totals.append(sum(numbers))
    return sum(totals)


class ProductionHealth:
    def __init__(self, url: str = PRODUCTION, *,
                 request: Callable[[str, dict[str, object] | None], object] | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.url = url.rstrip('/')
        self.request = request or self.http
        self.sleep = sleep

    def http(self, path: str, body: dict[str, object] | None) -> object:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.url + path, data=data,
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT_S) as response:
            require(response.status == 200, f'{path}: HTTP {response.status}')
            data = response.read(1024 * 1024 + 1)
            require(len(data) <= 1024 * 1024, 'oversized health response')
            return data.decode() if path == '/metrics' else cast(object, json.loads(data))

    def raw(self) -> None:
        models = object_record(self.request('/v1/models', None), 'models')
        rows = models.get('data')
        require(isinstance(rows, list), 'models lacks data array')
        found = [object_record(item, 'model') for item in cast(list[object], rows)]
        intended = [item for item in found if item.get('root') == MODEL_ROOT]
        require(len(intended) == 1 and isinstance(intended[0].get('id'), str)
                and bool(intended[0]['id']) and type(intended[0].get('max_model_len')) is int
                and intended[0]['max_model_len'] == 327680, 'production model root/id/max_model_len differs')
        reply = object_record(self.request('/v1/completions', {
            'model': intended[0]['id'], 'prompt': 'Reply OK.', 'max_tokens': 1,
            'temperature': 0, 'top_p': 1, 'seed': 42, 'priority': 20, 'stream': False,
        }), 'completion')
        choices = reply.get('choices')
        require(isinstance(choices, list) and len(choices) == 1, 'completion lacks exactly one choice')
        choice = object_record(cast(list[object], choices)[0], 'choice')
        usage = object_record(reply.get('usage'), 'usage')
        require(isinstance(choice.get('text'), str) and choice.get('finish_reason') in ('length', 'stop')
                and type(usage.get('completion_tokens')) is int and usage['completion_tokens'] == 1,
                'one-token greedy probe did not complete')

    def evidence(self) -> str:
        try:
            return 'idle' if requests(self.request('/metrics', None)) == 0 else 'busy'
        except Exception:
            # Missing/partial/invalid metrics is never evidence of idle.
            return 'unknown'

    def probe(self) -> None:
        # A successful completion is sufficient. Failed low-priority admission
        # needs independent evidence; two timeouts alone are not a fault.
        try:
            self.raw()
            return
        except Exception as exc:
            first = exc
        before = self.evidence()
        if before == 'busy':
            raise BusyHealth('active request gauges; priority-20 probe may be queued') from first
        self.sleep(CONFIRM_S)
        try:
            self.raw()
        except Exception as exc:
            second = exc
        else:
            raise BusyHealth('health recovered during bounded confirmation')
        after = self.evidence()
        if before == after == 'idle':
            raise WatchError(f'repeated failed health with idle gauges over {CONFIRM_S}s: {second}') from first
        # Total transport failure is independently distinguished from a working
        # models endpoint plus an unadmitted completion. Verify models once more.
        if before == after == 'unknown':
            try:
                self.request('/v1/models', None)
            except (OSError, TimeoutError):
                raise WatchError('models and metrics unreachable across bounded confirmation') from second
        raise BusyHealth('idle/progress evidence inconclusive; recovery refused') from second


def stop_command() -> str:
    """Receiving-side revalidation and first stop share one command lifetime."""
    return shlex.join(['python3', '-c', Path(__file__).read_text(), 'stop'])


def guarded_stop(health: ProductionHealth, stop: Callable[[], int]) -> int:
    try:
        health.probe()
    except BusyHealth:
        raise
    except WatchError:
        # Only confirmed failed idle/unreachable production reaches stop.
        return stop()
    raise BusyHealth('production is healthy immediately before stop; HOLD retained')


if __name__ == '__main__':
    require(sys.argv[1:] == ['stop'], 'unknown production health operation')
    raise SystemExit(guarded_stop(ProductionHealth(url='http://localhost:8000'),
        lambda: subprocess.call(['bash', '-c', 'cd ~/dsv41-deploy && ./stop.sh </dev/null'])))
