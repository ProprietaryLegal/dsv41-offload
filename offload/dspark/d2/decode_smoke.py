"""Coordinator's THREE-prompt :8001 smoke, offline/dry-run by default.

Does not start/stop services or touch x86host workers. --execute requires a reviewed
source pin, an already-open window with five-minute drain evidence, and private
request/reference inputs. Raw responses and copied taps stay in private output;
stdout/report contains only aggregate checks. A failure leaves HOLD/restoration
to the coordinator. Never run --execute from an implementation lane.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time
from typing import Protocol, cast

from .decode_dataset import SCHEMA, capture_rows, digest, mapping, number, obj, text
from .replay_window import CAMPAIGN, EXP, IMAGE, LiveIO, expected_hashes, gauges, require

from .smoke_latency import MAX_LATENCY_REGRESSION, REPEATS, baseline_samples, latency_gate

BASE = '/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode'


def token_ids(value: object) -> list[int]:
    require(isinstance(value, list) and bool(value), 'missing raw token IDs')
    ids = cast(list[object], value)
    require(all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x < 129280 for x in ids), 'invalid token IDs')
    return cast(list[int], ids)


def choice(response: dict[str, object]) -> dict[str, object]:
    require(not response.get('error') and not response.get('refusal'), 'HTTP-200 refusal/error')
    choices = response.get('choices')
    require(isinstance(choices, list) and len(choices) == 1, 'expected one choice')
    c = cast(dict[str, object], cast(list[object], choices)[0])
    require(c.get('finish_reason') in ('stop', 'length') and not c.get('refusal'), 'refused/unfinished smoke response')
    return c


def top20_ok(response: dict[str, object]) -> bool:
    """Full raw-token coverage, token-ID keys, a certifiable finite top20, exact committed-token LP.

    The image's chat endpoint returns exactly top_logprobs (20) entries: the SAMPLED token first, then the top
    ranks, cut at 20 (S29 2026-10-24, from the pinned image source; my 1875999 claim of a 21st entry was wrong).
    Listed order is therefore not checked; entries are ranked by value. When the sample ranks below the top 20 the
    list is the sample plus ranks 1-19 and the true 20th is lost, which is indistinguishable from a sample ranked
    exactly 20th: a row whose committed token is strictly the lowest entry cannot certify a top20 and refuses
    (R5 had 2 such rows in 128, R6 none). A collector must request top_logprobs 21 and keep the top 20 by value.
    """
    try:
        c = choice(response)
        ids = token_ids(c.get('token_ids'))
        content = mapping(c.get('logprobs')).get('content')
        require(isinstance(content, list) and len(content) == len(ids), 'incomplete generation logprobs')
        for token, raw in zip(ids, cast(list[object], content)):
            row = mapping(raw)
            mine = f'token_id:{token}'
            require(row.get('token') == mine, 'logprob token ID mismatch')
            listed = row.get('top_logprobs')
            require(isinstance(listed, list) and len(listed) == 20, 'not exactly top20')
            pairs = [(text(mapping(t)['token']), float(cast(float, mapping(t)['logprob']))) for t in cast(list[object], listed)]
            require(all(re.fullmatch(r'token_id:\d+', k) and 0 <= int(k.split(':')[1]) < 129280
                        and math.isfinite(v) and v <= 0 for k, v in pairs), 'bad logprob IDs/value')
            require(len({k for k, _ in pairs}) == 20, 'duplicate top20 ID')
            require(sum(math.exp(v) for _, v in pairs) <= 1.0001, 'top20 mass above one')
            lp = float(cast(float, row['logprob']))
            require(math.isfinite(lp) and lp <= 0, 'invalid committed LP')
            match = [v for k, v in pairs if k == mine]
            require(len(match) == 1 and abs(match[0] - lp) <= 1e-5, 'committed LP missing or mismatched')
            require(lp >= min(v for k, v in pairs if k != mine) - 1e-5, 'sample below the listed top19: top20 not certifiable')
        return True
    except (ValueError, KeyError, TypeError):
        return False


def request_drafting(response: dict[str, object], ledger: Path, prompt_length: int) -> bool:
    """Attribute drafting to chat even when per-request counters are disabled.

    The validated capture ledger removes rejected positions. A decode step
    retaining multiple positions proves this sequence accepted a draft. Prefill
    rows never count. An explicit counter (including zero) takes precedence.
    """
    stats = mapping(response.get('metrics') or {}).get('speculative_decoding')
    if stats is not None:
        return number(mapping(stats).get('num_draft_tokens', 0)) > 0
    return any(number(row['first']) >= prompt_length and number(row['rows']) > 1
               for line in ledger.read_text().splitlines() if (row := mapping(json.loads(line))))


def draft_count(metrics: str) -> float:
    values = [float(m[1]) for line in metrics.splitlines()
              if (m := re.fullmatch(r'vllm:spec_decode_num_drafts_total(?:\{[^}]*\})?\s+(\S+)', line))]
    require(bool(values) and all(math.isfinite(v) and v >= 0 for v in values), 'missing/invalid draft metrics')
    return sum(values)


def latency_delta(before: str, after: str, expected_requests: int = 1) -> dict[str, float]:
    """Image metrics: ITL observes one engine-output step, including accepted drafts.

    Source: vllm/v1/metrics/stats.py update_from_output (engine_core_timestamp -
    last_token_ts), not wall time divided by the number of accepted tokens.
    """
    def total(raw: str, name: str) -> float:
        values = [float(m[1]) for line in raw.splitlines()
                  if (m := re.fullmatch(re.escape(name) + r'(?:\{[^}]*\})?\s+(\S+)', line))]
        require(bool(values) and all(math.isfinite(v) and v >= 0 for v in values),
                'missing/invalid latency metric: ' + name)
        return sum(values)
    result: dict[str, float] = {}
    for key, metric in (('ttft_s', 'vllm:time_to_first_token_seconds'),
                        ('step_mean_s', 'vllm:inter_token_latency_seconds')):
        count = total(after, metric+'_count') - total(before, metric+'_count')
        seconds = total(after, metric+'_sum') - total(before, metric+'_sum')
        require(count > 0 and count.is_integer() and seconds > 0 and math.isfinite(seconds),
                'latency telemetry reset or no output steps')
        if key == 'ttft_s':
            require(count == expected_requests, 'latency interval includes foreign requests')
        result[key] = seconds / count
        if key == 'step_mean_s':
            result['step_samples'] = count
    return result


def window_gate(window: dict[str, object], source: str, now: float) -> None:
    require(window.get('coordinator') == 'Deepseek/coordinator' and window.get('source') == source,
            'coordinator/source window receipt mismatch')
    require(0 <= now - float(cast(float, window.get('created_unix', 0))) < 3600, 'stale window receipt')
    samples = window.get('drain_samples')
    require(isinstance(samples, list) and len(samples) >= 31, 'five-minute drain samples required')
    rows = [mapping(s) for s in cast(list[object], samples)]
    stamps = [float(cast(float, s['t'])) for s in rows]
    require(all(math.isfinite(t) for t in stamps) and stamps[-1] - stamps[0] >= 300, 'short drain interval')
    require(all(0 < b-a <= 10 for a, b in zip(stamps, stamps[1:])), 'missing/reordered drain telemetry')
    require(all(gauges(text(s['metrics'])) for s in rows), 'busy/invalid production drain')


def runtime(container: dict[str, object], repo: Path, files: dict[str, object]) -> None:
    require(container.get('image') == IMAGE and container.get('state') == 'running', 'wrong image/state')
    env = cast(list[str], container.get('env'))
    for value in ('VLLM_DSPARK_DECODE_CAPTURE_DIR=/d2-decode', 'VLLM_DSPARK_REMOTE_POLICY=capacity',
                  'VLLM_DSPARK_REMOTE_ASYNC=1', 'VLLM_DSPARK_REMOTE_ASYNC_PROXY=process',
                  'VLLM_DSV41_KV_PAIR=3', 'VLLM_DSV41_DROP_MTP_BUFFER=1',
                  'VLLM_EXL3_TF_PROMPT_TRANSIENT=1', 'VLLM_EXL3_TF_PROMPT_ALIAS=1',
                  'VLLM_INDEXER_PREFILL_BUFFER_FACTOR=4'):
        require(value in env, 'runtime env mismatch: '+value)
    mounts = cast(list[dict[str, object]], container['mounts'])
    require(any(m.get('Destination') == '/d2-decode' and m.get('RW') is True for m in mounts), 'capture mount missing')
    require(files.get('engine') == '0.31.1.dev0+g6e517b15.dsv41gb10', 'wrong engine')
    identity_files = mapping(files.get('files'))
    for path, expected_digest in expected_hashes(repo).items():
        require(identity_files.get(path) == expected_digest, 'target/renderer/overlay identity mismatch: '+path)
    actual = mapping(files['mounted_code_sha256'])
    for local, remote in (('capture/speculator_replay.py', '/dspark/speculator.py'),
                          ('capture/decode_capture.py', '/dspark/decode_capture.py'),
                          ('spark/remote_dspark/speculator.py', '/remote_dspark/speculator.py')):
        require(actual.get(BASE+remote) == digest(repo/'offload/dspark'/local), 'mounted capture code hash mismatch')
    expected = json.loads((repo/'offload/operations/profiles.v1.json').read_text())
    profile = next(p for p in expected['profiles'] if p['name'] == 'K1-327680')
    for key, expected_value in profile['match']['env'].items():
        values = [v.split('=', 1)[1] for v in env if v.startswith(key+'=')]
        require(values == ([] if expected_value is None else [expected_value]), 'K1/D profile env differs: '+key)
    # Verify every K1 file against the frozen reviewed overlay, in addition to env.
    for mount in profile['match']['mounts']:
        dest = mount['destination']
        if any(dest.endswith(path) for path in expected['shared']['k1_paths']):
            tail = dest.split('/dist-packages/')[-1]
            require(actual.get(dest) == digest(repo/'offload/kvpack/overlay'/tail), 'K1 overlay file differs')


class SmokeIO(Protocol):
    synthetic: bool
    def call(self, name: str, host: str, command: str, timeout: int = 60, supervised: bool = False) -> str: ...
    def post(self, name: str, endpoint: str, body: dict[str, object]) -> dict[str, object]: ...
    def metrics(self) -> str: ...
    def copy(self, remote: str, local: Path) -> None: ...
    def active(self, requests: list[dict[str, object]], remote: str) -> dict[str, object]: ...


class LiveSmoke(LiveIO):
    synthetic = False

    def metrics(self) -> str:
        script = "import urllib.request; print(urllib.request.urlopen('"+EXP+"/metrics',timeout=3).read().decode())"
        return self.call('experiment-metrics', 'x86host', shlex.join(['python3', '-c', script]), supervised=True)

    def post(self, name: str, endpoint: str, body: dict[str, object]) -> dict[str, object]:
        # EOF-leased child owns its HTTP connection; telemetry/STOP/timeout closes
        # only this request. No GPU imports or changes on x86host or the Sparks.
        script = ('import urllib.request,urllib.error,json; data='+repr(json.dumps(body).encode())+'; '
                  'r=urllib.request.Request('+repr(EXP+endpoint)+',data=data,headers={"Content-Type":"application/json"}); '
                  '\ntry: print(urllib.request.urlopen(r,timeout=300).read().decode())\n'
                  'except urllib.error.HTTPError as e: print(json.dumps({"error":"HTTP rejected","http_status":e.code}))')
        result = self.call(name, 'x86host', shlex.join(['python3', '-c', script]), timeout=310, supervised=True)
        return cast(dict[str, object], json.loads(result))

    def active(self, requests: list[dict[str, object]], remote: str) -> dict[str, object]:
        # One supervised, EOF-leased child owns all five HTTP connections.
        # No shared LiveIO state is accessed by background threads.
        code = Path(__file__).with_name('smoke_active.py').read_text()
        code += '\nprint(json.dumps(group('+repr(EXP)+', '+repr(requests)+', '+repr(remote)+')))'
        return mapping(json.loads(self.call('active-writer-controls', 'x86host',
            shlex.join(['python3', '-c', code]), timeout=340, supervised=True)))

    def copy(self, remote: str, local: Path) -> None:
        # O_DIRECT for taps/ids, small metadata can also use it. Never tar a Spark
        # capture through page cache. Output is a private file, never console/log.
        self.memcheck()
        argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', 'sparkhead',
                shlex.join(['dd', 'if='+remote, 'iflag=direct', 'bs=4096', 'status=none'])]
        with local.open('wb') as out:
            child = subprocess.Popen(argv, stdout=out, stderr=subprocess.PIPE, start_new_session=True)
            try:
                deadline = self.now()+60
                while child.poll() is None:
                    self.memcheck()
                    require(self.now() < deadline, 'capture-copy deadline')
                    time.sleep(.2)
                require(child.returncode == 0, 'O_DIRECT capture copy failed')
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=5)
                if child.stderr:
                    child.stderr.close()


def checked_spec(spec: dict[str, object]) -> tuple[list[dict[str, object]], str, list[dict[str, float]], int]:
    require(spec.get('schema') == 'd2.decode-smoke-input.v1', 'wrong smoke input schema')
    requests = cast(list[dict[str, object]], spec.get('requests'))
    require(isinstance(requests, list) and len(requests) == 3, 'exactly three prompts required')
    for i, r in enumerate(requests):
        body = mapping(r['body'])
        require(body.get('stream', False) is False and 1 <= number(body.get('max_tokens')) <= (512 if i == 2 else 256), 'unbounded smoke')
        require(body.get('return_token_ids') is True, 'raw token IDs required')
        # W5's exact K1 request omits top_p. The pinned image defaults it to 1;
        # preserve that original body rather than adding a sampling field to it.
        require(body.get('temperature') == (0 if i == 2 else 1)
                and body.get('top_p', 1 if i == 2 else None) == 1, 'sampling mismatch')
        require(body.get('priority', 0) in ((0, 10) if i == 2 else (20,)), 'priority mismatch')
        prompt = token_ids(r['prompt_ids'])
        require(len(prompt) <= 8192, 'smoke prompt exceeds 8192 tokens')
        if i < 2:
            require(body.get('request_id') == f'd2c.smoke{i}.0.{max(0,len(prompt)-128)}.{len(prompt)}', 'capture tag mismatch')
        else:
            rid = body.get('request_id')
            require(rid is None or (isinstance(rid, str) and 'd2c.' not in rid),
                    'greedy control must be untagged')
        require(r['endpoint'] in (('/v1/completions', '/v1/chat/completions') if i == 2 else
                                  (('/v1/chat/completions',) if i == 1 else ('/v1/completions',))), 'wrong endpoint')
    chat = mapping(requests[1]['body'])
    require(chat.get('logprobs') is True and chat.get('top_logprobs') == 20
            and chat.get('return_tokens_as_token_ids') is True, 'option ii was not actually requested')
    ref = mapping(spec['greedy_reference'])
    path = Path(text(ref['path']))
    require(digest(path) == ref['sha256'], 'K1 capture changed')
    capture = json.loads(path.read_text().splitlines()[number(ref['row'])])
    proof_path = Path(text(ref['request_receipt']))
    require(digest(proof_path) == ref['request_receipt_sha256'], 'K1 request receipt changed')
    proof = obj(proof_path)
    request = dict(mapping(requests[2]['body']))
    historical = dict(mapping(proof.get('request')))
    # Tracing/serialization fields do not alter the sampled distribution. Keep
    # every model, prompt/template and sampling parameter exact.
    for row in (request, historical):
        for key in ('request_id', 'return_token_ids', 'return_tokens_as_token_ids'):
            row.pop(key, None)
        if row.get('stream') is False:
            row.pop('stream')
    require(proof.get('endpoint', requests[2]['endpoint']) == requests[2]['endpoint'], 'K1 endpoint differs')
    require(proof.get('profile') == 'K1-327680' and proof.get('capture_sha256') == ref['sha256']
            and proof.get('row') == ref['row'] and historical == request
            and proof.get('verified') is True, 'unverified/mismatched K1 greedy request identity')
    baseline = baseline_samples(proof)
    cached = number(proof.get('cached_prompt_tokens'))
    require(cached <= len(token_ids(requests[2]['prompt_ids'])), 'invalid K1 cached-token count')
    return requests, text(capture['text']), baseline, cached


def run(io: SmokeIO, spec: dict[str, object], repo: Path, out: Path, run_id: str) -> dict[str, object]:
    requests, greedy_reference, baseline_latency, baseline_cached = checked_spec(spec)
    control_latency: dict[str, object] = {}
    require(re.fullmatch('[A-Za-z0-9_-]{1,80}', run_id) is not None, 'unsafe smoke run ID')
    probe = (repo/'offload/dspark/d2/capture_window_identity.sh').read_text()
    containers: dict[str, str] = {}

    def verify_runtime() -> None:
        # The apply plan requires fresh HOLD, both-rank file/runtime identity and
        # model checks before EACH request, including every isolated untagged control; active-group preflights precede dispatch.
        io.call('hold', 'sparkhead', 'test -s ~/dsv41-deploy/HOLD')
        for node in ('sparkhead', 'sparkworker-wired'):
            rows = [json.loads(line) for line in io.call('identity-'+node, node, shlex.join(['bash', '-c', probe]),
                                                       timeout=60, supervised=True).splitlines()]
            require(len(rows) == 2, 'incomplete both-rank runtime receipt')
            runtime(rows[0], repo, rows[1])
            cid = text(rows[0]['id'])
            require(node not in containers or containers[node] == cid, 'experiment container changed between requests')
            containers[node] = cid
        models = cast(dict[str, object], json.loads(io.call('models', 'x86host', shlex.join(['python3', '-c',
            "import urllib.request; print(urllib.request.urlopen('"+EXP+"/v1/models',timeout=3).read().decode())"]), supervised=True)))
        m = cast(list[dict[str, object]], models['data'])[0]
        require(m.get('root') == '/srv/dsv41/legal-v2a' and m.get('max_model_len') == 327680, 'wrong target/cap')
    def control_output(response: dict[str, object]) -> tuple[str, object, object]:
        c = choice(response)
        token_ids(c.get('token_ids'))
        r = requests[2]
        result = c.get('text') if r['endpoint'] == '/v1/completions' else mapping(c.get('message')).get('content')
        echo = response.get('prompt_token_ids') if r['endpoint'] == '/v1/chat/completions' else c.get('prompt_token_ids')
        cached = mapping(mapping(response.get('usage')).get('prompt_tokens_details')).get('cached_tokens')
        return text(result), echo, cached

    def control_response(response: dict[str, object]) -> None:
        result, echo, cached = control_output(response)
        require(result.encode('utf-8') == greedy_reference.encode('utf-8'),
                'untagged greedy output differs from K1 capture')
        require(echo == token_ids(requests[2]['prompt_ids']), 'server prompt token IDs differ')
        require(number(cached) == baseline_cached, 'untagged latency cache condition differs from K1')

    def informational_output(response: dict[str, object]) -> dict[str, object]:
        # Record only (coordinator 2026-10-24 after the R4 stop, NDM go): a control batched beside the two tagged
        # requests runs a different batch shape, and near-tied greedy steps can flip (window 8; R4 diverged at char
        # 52/121 while R1's control ALONE with capture mounted was byte-identical). The capture-identity check is the
        # isolated controls' hard byte check below.
        result, echo, cached = control_output(response)
        ref = greedy_reference
        first = next((i for i, (a, b) in enumerate(zip(result, ref)) if a != b), None)
        if first is None and len(result) != len(ref):
            first = min(len(result), len(ref))
        return {'text_identical': result == ref, 'first_divergence_char': first, 'chars': len(result),
                'reference_chars': len(ref), 'prompt_echo_identical': echo == token_ids(requests[2]['prompt_ids']),
                'cached_tokens': cached if isinstance(cached, int) else None}

    remote = '/home/rank0user/dsv41-offload-run/d2-decode-'+run_id
    # Preserve both-rank runtime/model/HOLD preflights for each tagged request.
    verify_runtime()
    verify_runtime()
    for _ in range(3):  # fresh identity preflights for the three informational controls
        verify_runtime()
    require(gauges(io.metrics()), 'foreign work or missing experiment gauges')
    active_before = io.metrics()
    active = io.active(requests, remote)
    (out/'active-writer-controls.json').write_text(json.dumps(active))
    informational = cast(list[dict[str, object]], active.get('informational'))
    # Informational, best-effort (coordinator 2026-10-24 after three runner stops): up to three writer-active controls;
    # a shortfall is recorded, never a refusal. Their output vs K1 is recorded, not gated (informational_output).
    # Gated checks are unchanged.
    require(isinstance(informational, list) and len(informational) <= 3, 'malformed active-writer informational controls')
    require(active.get('gated') is False and isinstance(active.get('informational_status'), str),
            'active-writer status missing')
    outputs: list[dict[str, object]] = []
    for row in informational:
        require(row.get('both_tagged_inflight_at_start') is True, 'informational control missed tagged decoding')
        outputs.append(informational_output(mapping(row['response'])))
    (out/'informational-outputs.json').write_text(json.dumps(outputs))
    replies: list[dict[str, object]] = []
    ii = False
    option_ii_http_status = None
    captured = [0, 1]
    for i, r in enumerate(requests[:2]):
        before = active_before
        require(gauges(before), 'foreign work or missing experiment gauges')
        drafts = draft_count(before)
        response = mapping(cast(list[object], active['tagged'])[i])
        (out / f'response-{i}.json').write_text(json.dumps(response))
        if i == 1 and response.get('http_status') in (400, 422, 501):
            option_ii_http_status = response['http_status']
            captured = [0]
            require(gauges(io.metrics()), 'unsupported option ii request did not drain')
            replies.append(response)
            continue
        c = choice(response)
        ids = token_ids(c.get('token_ids'))
        if i < 2:
            require(len(ids) >= 3, 'tagged smoke too short to validate decode rows')
        prompt = token_ids(r['prompt_ids'])
        echo = response.get('prompt_token_ids') if r['endpoint'] == '/v1/chat/completions' else c.get('prompt_token_ids')
        require(echo == prompt, 'server prompt token IDs differ')
        after = io.metrics()
        require(gauges(after), 'experiment did not drain after own request')
        if i == 1:
            ii = top20_ok(response) and draft_count(after) > drafts
        replies.append(response)
    active_after = io.metrics()
    require(gauges(active_after), 'active group did not drain')
    # Every informational control actually SENT completes inside this interval; a best-effort shortfall sends fewer
    # (R5 stopped here: two sent, three expected).
    latency_delta(active_before, active_after, expected_requests=len(captured)+len(informational))
    nonce = 'smoke-'+run_id
    script = ('import json,pathlib,time; p=pathlib.Path('+repr(remote)+'); '
              '(p/"FLUSH.json.tmp").write_text(json.dumps({"nonce":'+repr(nonce)+'})); '
              '(p/"FLUSH.json.tmp").replace(p/"FLUSH.json"); deadline=time.monotonic()+30\n'
              'while time.monotonic()<deadline:\n'
              ' if (p/"FLUSHED.json").exists() and json.loads((p/"FLUSHED.json").read_text()).get("nonce")=='+repr(nonce)+': break\n'
              ' time.sleep(.2)\n'
              'else: raise SystemExit("capture flush timed out")\n'
              'print((p/"FLUSHED.json").read_text())')
    flushed = cast(dict[str, object], json.loads(io.call('flush', 'sparkhead', shlex.join(['python3','-c',script]),
                                                       timeout=40, supervised=True)))
    require(flushed.get('errors') == 0 and set(mapping(flushed.get('sequences'))) == {'smoke'+str(i) for i in captured},
            'writer errors, missing tagged captures, or an untagged capture')
    # Tagged decoding has finished and the durable nonce acknowledgement proves
    # the writer queue drained. Untagged controls enqueue no further capture jobs.
    r = requests[2]
    measured: list[dict[str, float]] = []
    for repeat in range(REPEATS + 1):
        verify_runtime()
        before = io.metrics()
        require(gauges(before), 'foreign work before isolated control')
        name = 'control-warmup' if repeat == 0 else f'control-timed-{repeat}'
        response = io.post(name, text(r['endpoint']), dict(mapping(r['body'])))
        (out/(name+'-response.json')).write_text(json.dumps(response))
        after = io.metrics()
        (out/(name+'-before.prom')).write_text(before)
        (out/(name+'-after.prom')).write_text(after)
        require(gauges(after), 'experiment did not drain after own control')
        control_response(response)
        timing = latency_delta(before, after)  # also reject contaminated warm-up
        if repeat:
            measured.append(timing)
    control_latency = latency_gate(baseline_latency, measured)
    (out/'untagged-latency.json').write_text(json.dumps(control_latency))
    copy = out / 'capture'
    copy.mkdir(mode=0o700)
    io.call('no-errors', 'sparkhead', 'test ! -s '+shlex.quote(remote+'/ERRORS.jsonl'), supervised=True)
    for file in ('FLUSH.json', 'FLUSHED.json'):
        io.copy(remote+'/'+file, copy/file)
    counts: list[int] = []
    for i in captured:
        sid = 'smoke'+str(i)
        (copy/sid).mkdir(mode=0o700)
        for file in ('CAPTURE.json', 'ledger.jsonl', 'taps.bin', 'ids.bin'):
            io.copy(remote+'/'+sid+'/'+file, copy/sid/file)
        prompt = token_ids(requests[i]['prompt_ids'])
        tokens = prompt + token_ids(choice(replies[i])['token_ids'])
        _, row_report = capture_rows(copy, sid, max(0,len(prompt)-128), len(tokens)-1, tokens)
        counts.append(number(row_report['rows']))
        if i == 1:
            ii = ii and request_drafting(replies[1], copy/sid/'ledger.jsonl', len(prompt))
    # Detect a restart or capture-dir replacement before certifying this run.
    for node, cid in containers.items():
        io.call('same-container-'+node, node, 'test "$(docker inspect --format \'{{.Id}}\' dsv41-serve)" = '+shlex.quote(cid),
                supervised=True)
    report: dict[str, object] = {'schema': 'd2.decode-smoke.v1', 'synthetic': io.synthetic, 'pass': True,
        'image': IMAGE, 'capture_sha256': digest(repo/'offload/dspark/capture/decode_capture.py'),
        'overlay_sha256': digest(repo/'offload/dspark/capture/speculator_replay.py'),
        'remote_speculator_sha256': digest(repo/'offload/dspark/spark/remote_dspark/speculator.py'), 'option_ii': ii, 'option_ii_http_status': option_ii_http_status, 'top20_source': 'generation_verified' if ii else 'prefill_replay_top20_only',
        'untagged_latency': control_latency, 'informational_active_writer': {
            **{k: active[k] for k in ('informational_status', 'informational_medians', 'timing_source', 'gated')},
            'outputs_vs_k1': outputs, 'outputs_gated': False},
        'http_requests': 8 + len(informational), 'control_warmups': 1, 'control_timed_repeats': REPEATS, 'informational_repeats': len(informational), 'prompts': 3, 'capture_rows': counts, 'untagged_greedy_identical': True, 'ledger_complete': True}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--execute', action='store_true')
    ap.add_argument('--source', default='')
    ap.add_argument('--run', default='synthetic')
    ap.add_argument('--inputs', type=Path)
    ap.add_argument('--window', type=Path)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[3]
    if not a.execute:
        from .decode_smoke_dry import dry_run
        dry_run(repo)
        return
    require(os.environ.get('D2_COORDINATOR') == 'Deepseek/coordinator', 'coordinator execution only')
    require(re.fullmatch('[0-9a-f]{40}', a.source) is not None, 'full immutable source SHA required')
    require(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip() == a.source, 'wrong source HEAD')
    require(not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(), 'source is dirty')
    require(a.inputs is not None and a.window is not None and a.out is not None, 'private inputs/window/new output required')
    window_gate(obj(a.window), a.source, time.time())
    require(a.out.resolve().is_relative_to(CAMPAIGN), 'output must stay in private campaign')
    os.umask(0o077)
    a.out.mkdir(parents=True, mode=0o700, exist_ok=False)
    run(LiveSmoke(a.out, a.source), obj(a.inputs), repo, a.out, a.run)


if __name__ == '__main__':
    main()
