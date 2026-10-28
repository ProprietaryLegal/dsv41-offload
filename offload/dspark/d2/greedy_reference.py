"""Bounded idle-gated production K1 warm repeats; private, SHA-bound evidence.

R2 explicitly authorizes this collection. A durable run marker and per-attempt
markers prevent resending after a crash. No service/GPU/experiment operations.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Callable, Protocol, cast

from .decode_dataset import digest, mapping, number, obj, text
from .decode_smoke import checked_spec, choice, latency_delta
from .replay_window import IMAGE, gauges, require
from .smoke_prepare import PRIVATE, Production, historical_request, save, W5, K1_CAPTURE
from .smoke_latency import MAX_ATTEMPTS, REPEATS, medians

HTTP_CHILD = """import json,sys,urllib.request
body=json.load(sys.stdin)
r=urllib.request.Request('http://HEAD_IP:8000/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
with urllib.request.urlopen(r,timeout=60) as response:
 sys.stdout.buffer.write(response.read())
"""


class Contamination(ValueError):
    """Only traffic contamination is retryable, never output drift or bad telemetry."""


def runtime_identity(repo: Path) -> dict[str, object]:
    raw = subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',
        'sparkhead', 'docker inspect dsv41-serve'], text=True, timeout=10)
    c = mapping(json.loads(raw)[0])
    require(c['Image'] == IMAGE and mapping(c['State'])['Running'] is True, 'not running pinned K1 image')
    cfg = mapping(c['Config'])
    env = cast(list[str], cfg['Env'])
    profiles = obj(repo/'offload/operations/profiles.v1.json')
    profile = next(mapping(p) for p in cast(list[object], profiles['profiles']) if mapping(p)['name'] == 'K1-327680')
    for key, value in mapping(mapping(profile['match'])['env']).items():
        require([x.split('=',1)[1] for x in env if x.startswith(key+'=')] == ([] if value is None else [value]),
                'K1 environment differs: '+key)
    require(not any(x.startswith('VLLM_DSPARK_DECODE_CAPTURE_DIR=') for x in env), 'capture already active')
    # Container entrypoint is bash /serve.sh, so inspect the actual light launch
    # script rather than treating Config.Cmd as a vLLM argv vector.
    command = subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',
        'sparkhead', 'docker exec dsv41-serve cat /serve.sh'], text=True, timeout=10)
    require('/srv/dsv41/legal-v2a' in command and '--max-model-len 327680' in command
            and '--port 8000' in command, 'not the production K1 command')
    return {'image': c['Image'], 'container_id': c['Id'], 'config': cfg,
            'launch_script': command, 'mounts': sorted(cast(list[dict[str, object]], c['Mounts']), key=lambda m: text(m['Destination']))}


class ReferenceIO(Protocol):
    def metrics(self) -> str: ...
    def post(self, body: dict[str, object], attempt: Path) -> dict[str, object]: ...
    def pause(self) -> None: ...


class LiveReference(Production):
    def pause(self) -> None:
        time.sleep(5)

    def post(self, body: dict[str, object], attempt: Path) -> dict[str, object]:
        with (attempt/'response.json').open('xb') as out, (attempt/'http.log').open('x') as err:
            child = subprocess.Popen([sys.executable,'-c',HTTP_CHILD], stdin=subprocess.PIPE,
                                     stdout=out, stderr=err, start_new_session=True)
            try:
                require(child.stdin is not None, 'missing owned HTTP input')
                child.stdin.write(json.dumps(body).encode())
                child.stdin.close()
                deadline = time.monotonic()+65
                with (attempt/'monitor.prom').open('x') as monitor:
                    while child.poll() is None:
                        require(time.monotonic() < deadline, 'reference timeout')
                        raw = self.metrics()
                        monitor.write(f'# observed_unix {time.time()}\n'+raw+'\n')
                        monitor.flush()
                        gauges(raw)  # validates finite nonnegative gauges
                        vals = {k: [float(m[1]) for line in raw.splitlines() if
                            (m := re.fullmatch(r'vllm:num_requests_'+k+r'(?:\{[^}]*\})?\s+(\S+)',line))]
                            for k in ('running','waiting')}
                        if sum(vals['running']) > 1 or sum(vals['waiting']) != 0:
                            raise Contamination('foreign demand; stopped only owned HTTP child')
                        time.sleep(.1)
                require(child.returncode == 0, 'reference HTTP refused; inspect private http.log')
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=5)
        return obj(attempt/'response.json')


def capture(io: ReferenceIO, inputs_path: Path, runtime: dict[str, object], out: Path,
            verify_runtime: Callable[[], dict[str, object]]) -> dict[str, object]:
    spec = obj(inputs_path)
    ref = mapping(spec['greedy_reference'])
    proof_path = Path(text(ref['request_receipt']))
    require(digest(proof_path) == ref['request_receipt_sha256'] and digest(K1_CAPTURE) == ref['sha256'],
            'reference identity/capture changed')
    proof = obj(proof_path)
    body = mapping(proof['request'])
    require(body == historical_request(W5, K1_CAPTURE, number(ref['row']))
            and body.get('priority', 0) == 0, 'original priority-0 W5 request identity changed')
    # Never overwrite the original single-sample receipt or its consumed marker.
    save(PRIVATE/'r2-baseline-send-marker.json', {'started_unix':time.time(),
        'out':str(out), 'request_receipt_sha256':digest(proof_path), 'max_attempts':MAX_ATTEMPTS})
    out.mkdir(mode=0o700, exist_ok=False)
    save(out/'runtime-before.json', runtime)
    historical = json.loads(K1_CAPTURE.read_text().splitlines()[number(ref['row'])])
    warmup: dict[str, object] | None = None
    samples: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    cached: int | None = None
    for n in range(1, MAX_ATTEMPTS+1):
        attempt = out/f'attempt-{n:02d}'
        attempt.mkdir(mode=0o700)
        before = io.metrics()
        (attempt/'before.prom').write_text(before)
        row: dict[str, object] = {'attempt': n, 'accepted':False, 'sent':False, 'before':before}
        attempts.append(row)
        if not gauges(before):
            row['reason'] = 'production busy; no request sent'
        else:
            save(attempt/'send-marker.json', {'started_unix':time.time()})
            row['sent'] = True
            try:
                response = io.post(body, attempt)
            except Contamination as exc:
                row['reason'] = str(exc)
            else:
                after = io.metrics()
                (attempt/'after.prom').write_text(after)
                row['after'] = after
                content = text(mapping(choice(response)['message'])['content'])
                require(content.encode('utf-8') == historical['text'].encode('utf-8'),
                        'K1 greedy bytes differ: stop for review')
                row['output_identical'] = True
                if not gauges(after):
                    row['reason'] = 'foreign work after own request'
                else:
                    try:
                        timing = latency_delta(before, after)
                    except ValueError as exc:
                        if str(exc) != 'latency interval includes foreign requests':
                            raise
                        row['reason'] = str(exc)
                    else:
                        count = number(mapping(mapping(response['usage'])['prompt_tokens_details'])['cached_tokens'])
                        require(cached is None or count == cached, 'K1 warm repeat cache conditions changed')
                        cached = count
                        row.update(accepted=True, cached_prompt_tokens=count, latency=timing,
                                   response_sha256=digest(attempt/'response.json'))
                        if warmup is None:
                            warmup = row
                        else:
                            samples.append(row)
        save(attempt/'measurement.json', row)
        save(out/f'checkpoint-{n:02d}.json', {'attempts':n, 'sent':sum(r['sent'] is True for r in attempts),
                                              'accepted_repeats':len(samples), 'warmup':warmup is not None})
        if len(samples) == REPEATS:
            break
        if not row['accepted']:
            io.pause()  # no new Spark request until a fresh idle snapshot
    require(len(samples) == REPEATS and warmup is not None,
            'ten-attempt limit reached without five uncontaminated warm K1 repeats')
    require(gauges(io.metrics()), 'foreign work after collection; no further SSH')
    after_runtime = verify_runtime()
    save(out/'runtime-after.json', after_runtime)
    require(after_runtime == runtime, 'production runtime changed during K1 measurement')
    return qualify_saved(inputs_path, out, runtime)


def qualify_saved(inputs_path: Path, out: Path, runtime: dict[str, object],
                  recovery_note: str | None = None) -> dict[str, object]:
    """Offline validation/recovery. Never sends HTTP/SSH or replaces evidence."""
    spec = obj(inputs_path)
    ref = mapping(spec['greedy_reference'])
    proof_path = Path(text(ref['request_receipt']))
    require(digest(proof_path) == ref['request_receipt_sha256'] and digest(K1_CAPTURE) == ref['sha256'],
            'reference identity/capture changed')
    proof = obj(proof_path)
    require(mapping(proof['request']) == historical_request(W5, K1_CAPTURE, number(ref['row'])),
            'saved K1 request changed')
    expected = json.loads(K1_CAPTURE.read_text().splitlines()[number(ref['row'])])['text'].encode('utf-8')
    attempts = [obj(p) for p in sorted(out.glob('attempt-*/measurement.json'))]
    require(1 <= len(attempts) <= MAX_ATTEMPTS, 'invalid saved attempt count')
    accepted: list[dict[str, object]] = []
    for n, row in enumerate(attempts, 1):
        require(row['attempt'] == n, 'saved attempts not contiguous')
        if row.get('accepted') is not True:
            continue
        attempt = out/f'attempt-{n:02d}'
        require(digest(attempt/'response.json') == row['response_sha256'], 'saved response changed')
        response = obj(attempt/'response.json')
        require(text(mapping(choice(response)['message'])['content']).encode('utf-8') == expected,
                'saved greedy bytes differ')
        require(row['before'] == (attempt/'before.prom').read_text()
                and row['after'] == (attempt/'after.prom').read_text(), 'saved metrics changed')
        require(gauges(text(row['before'])) and gauges(text(row['after'])), 'saved interval busy')
        require(latency_delta(text(row['before']), text(row['after'])) == row['latency'], 'saved timings changed')
        require(number(mapping(mapping(response['usage'])['prompt_tokens_details'])['cached_tokens'])
                == row['cached_prompt_tokens'], 'saved cache count changed')
        accepted.append(row)
    require(len(accepted) == REPEATS+1, 'one warm-up and five saved repeats required')
    warmup, samples = accepted[0], accepted[1:]
    cached = number(warmup['cached_prompt_tokens'])
    require(all(r['cached_prompt_tokens'] == cached for r in accepted), 'saved cache conditions differ')
    timings = [cast(dict[str, float], r['latency']) for r in samples]
    receipt = dict(proof, schema='d2.k1-greedy-reference.v2', timing_qualified=True,
        warmup=warmup, samples=samples, attempts=attempts, cached_prompt_tokens=cached,
        latency=medians(timings), runtime=runtime, captured_unix=time.time(), output_identical=True,
        collector_sha256=digest(Path(__file__)), recovery_note=recovery_note)
    target = out/'k1-request-receipt.json'
    save(target, receipt)
    qualified = dict(spec, greedy_reference=dict(ref, request_receipt=str(target), request_receipt_sha256=digest(target)))
    checked_spec(qualified)
    inputs = PRIVATE/'smoke-input-r2-qualified.json'
    save(inputs, qualified)
    summary = {'qualified':True, 'greedy_identical':True, 'attempts':len(attempts),
        'requests_sent':sum(r['sent'] is True for r in attempts), 'warmups':1, 'repeats':REPEATS,
        'cached_prompt_tokens':cached, 'latency_medians':medians(timings), 'samples':timings,
        'inputs_sha256':digest(inputs), 'request_receipt_sha256':digest(target), 'receipt':str(target)}
    save(out/'summary.json', summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute-r2', action='store_true')
    a = ap.parse_args()
    require(a.execute_r2, 'explicit --execute-r2 required for authorized repeated reference')
    os.umask(0o077)
    io = LiveReference()
    require(gauges(io.metrics()), 'production busy; no SSH or generation sent')
    repo = Path(__file__).resolve().parents[3]
    runtime = runtime_identity(repo)
    result = capture(io, PRIVATE/'smoke-input.json', runtime, PRIVATE/'r2-baseline', lambda: runtime_identity(repo))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
