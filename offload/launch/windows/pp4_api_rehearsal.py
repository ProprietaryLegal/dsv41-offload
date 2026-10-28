"""Pre-window API rehearsal: every admission API probe against PRODUCTION, tiny.

Coordinator-run between windows (a queue line); never inside a HOLD and never by a
lane. It sends the admission's own request bodies (the pp4_admission_requests
builders) to production with generation capped at MAX_TOKENS and the admission's
priority 10 (background), never waits for production to be idle, and records every
response, including every HTTP error body.

Reading: production serves the same image and the same request validation as the PP4
candidate (PP4 overlays only remote_dspark, model_runner.py,
rejection_sampler_utils.py and its own admission/round-trace modules; no entrypoint
or sampling_params file), and both run speculative decoding. A 4xx here means the
PROBE is wrong for this image; a probe production accepts but the candidate refuses
is a CANDIDATE defect. The tokenize and terminator probes are already tiny and run
unmodified with the admission's exact assertions; the chat and cancel probes are
capped (request validation does not depend on max_tokens) and their exact checks run
only when the capped answer finished. One CONTROL request, the R16 stop probe exactly as
PP4-90f cmd/0132 sent it (logit_bias forcing), must come back 400 naming logit_bias and
speculative decoding; it confirms the cause on production and never enters the verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import cast

from offload.launch.windows import pp4_admission_requests as api
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, record

PRODUCTION = 'http://HEAD_IP:8000'
MAX_TOKENS = 8
TIMEOUT_S = 60.0
CHAT_NAMES = ('grammar', 'penalties', 'json', 'tool', 'reasoning')
CONTROL = 'control-r16-logit-bias'
ROOT = Path(__file__).resolve().parents[3]
Reply = tuple[int | None, dict[str, Json] | None, str | None]


def settings(body: dict[str, Json]) -> dict[str, Json]:
    return {k: v for k, v in body.items() if k not in ('prompt', 'messages')}


def capped(body: dict[str, Json]) -> dict[str, Json]:
    return {**body, 'max_tokens': min(int(str(body['max_tokens'])), MAX_TOKENS)}


class Production:
    """Tiny requests with their own timeout; HTTP errors are logged and recorded, never raised."""
    def __init__(self, url: str, timeout: float) -> None:
        self.url, self.timeout = url, timeout

    def open(self, endpoint: str, body: dict[str, Json] | None) -> Reply:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.url + endpoint, data, {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return int(response.status), record(cast(Json, json.load(response))), None
        except urllib.error.HTTPError as error:
            refused = api.refused(error, endpoint, body or {})
            return refused.status, None, refused.body
        except (OSError, ValueError) as exc:  # transport failure or a non-JSON reply
            return None, None, f'{type(exc).__name__}: {exc}'

    def stream_first(self, body: dict[str, Json]) -> tuple[int | None, bool, str | None]:
        """Open the stream and close at the first content chunk (the cancel probe's disconnect)."""
        request = urllib.request.Request(self.url + '/v1/chat/completions', json.dumps(body).encode(),
                                         {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = int(response.status)
                for line in response:
                    if line.startswith(b'data: {'):
                        choices = ck.rows(record(cast(Json, json.loads(line[6:]))).get('choices', []))
                        if choices and record(choices[0].get('delta', {})).get('content'):
                            return status, True, None
                return status, False, None
        except urllib.error.HTTPError as error:
            refused = api.refused(error, '/v1/chat/completions', body)
            return refused.status, False, refused.body
        except (OSError, ValueError) as exc:
            return None, False, f'{type(exc).__name__}: {exc}'


def row(name: str, endpoint: str, mode: str, body: dict[str, Json], reply: Reply, started: float) -> dict[str, Json]:
    status, response, error = reply
    out: dict[str, Json] = {'name': name, 'endpoint': endpoint, 'mode': mode, 'request_sha256': api.sha(cast(Json, body)),
        'settings': settings(body), 'status': status, 'error_body': error, 'seconds': round(time.monotonic()-started, 3)}
    if response is not None:
        out['response_sha256'] = api.sha(cast(Json, response))
        choices = response.get('choices')
        if isinstance(choices, list) and choices:
            choice = record(choices[0])
            out.update({'finish_reason': choice.get('finish_reason'), 'stop_reason': choice.get('stop_reason')})
        if 'usage' in response:
            out['usage'] = response['usage']
    return out


def check(out: dict[str, Json], assertion: Callable[..., object], *args: object) -> None:
    """Run the admission's own assertion; record pass or its STOP message."""
    try:
        assertion(*args)
        out['check'] = 'pass'
    except (ValueError, KeyError, TypeError) as exc:
        out['check'] = 'fail: ' + str(exc)


def served_only(served: Json) -> None:
    ck.require(isinstance(served, list) and [record(m).get('id') for m in served] == [api.MODEL],
               'production does not serve exactly '+api.MODEL)


def chat_check(name: str, body: dict[str, Json], response: dict[str, Json]) -> None:
    api.check_chat(name, api.chat_evidence(body, response)[0])


def r16_control(stop: int, eos: int) -> dict[str, Json]:
    """The R16 (90f0278) stop probe exactly as PP4-90f cmd/0132 sent it: logit_bias forcing."""
    body = dict(dict(api.terminator_bodies(stop, eos))['stop'])
    body.pop('allowed_token_ids', None)
    return {**body, 'logit_bias': {str(stop): 100}}


def spec_decode_refusal(status: Json, error: Json) -> None:
    ck.require(status == 400 and 'logit_bias' in str(error) and 'speculative decoding' in str(error),
               'control not refused for logit_bias under speculative decoding')


def plan(eos: int, stop: int | None) -> list[tuple[str, str, str, dict[str, Json]]]:
    """(name, endpoint, mode, body) in admission order; `stop` None = tokenize not yet answered."""
    rows: list[tuple[str, str, str, dict[str, Json]]] = []
    for name in CHAT_NAMES[:2]:
        rows.append((name, '/v1/chat/completions', 'capped', capped(api.chat_body(*api.CHATS[name]))))
    rows.append(('tokenize', '/tokenize', 'exact', api.TOKENIZE))
    for name, body in api.terminator_bodies(-1 if stop is None else stop, eos):
        ck.require(int(str(body['max_tokens'])) <= MAX_TOKENS, name+': terminator probe is not tiny')
        rows.append((name, '/v1/completions', 'exact', body))
    rows.append((CONTROL, '/v1/completions', 'control', r16_control(-1 if stop is None else stop, eos)))
    for name in CHAT_NAMES[2:]:
        rows.append((name, '/v1/chat/completions', 'capped', capped(api.chat_body(*api.CHATS[name]))))
    rows.append(('cancel', '/v1/chat/completions', 'capped-stream', capped(api.cancel_body())))
    return rows


def rehearse(server: Production, eos: int, responses: list[dict[str, Json]]) -> list[dict[str, Json]]:
    rows: list[dict[str, Json]] = []
    started = time.monotonic()
    reply = server.open('/v1/models', None)
    models = row('models', '/v1/models', 'exact', {}, reply, started)
    if reply[1] is not None:
        check(models, served_only, reply[1].get('data'))
    rows.append(models)
    stop: int | None = None
    for name, endpoint, mode, body in plan(eos, None):
        if name in ('stop', 'eos', CONTROL):
            body = (r16_control(stop, eos) if name == CONTROL and stop is not None
                    else dict(api.terminator_bodies(-1 if stop is None else stop, eos)).get(name, body))
            if name in ('stop', CONTROL) and stop is None:
                rows.append({'name': name, 'endpoint': endpoint, 'mode': mode, 'status': None,
                             'check': 'not-run: tokenize gave no single stop token'})
                continue
        started = time.monotonic()
        if mode == 'capped-stream':
            status, content, error = server.stream_first(body)
            out = row(name, endpoint, mode, body, (status, None, error), started)
            out['content_chunk'] = content
            if status == 200:
                check(out, ck.require, content, 'no actual streamed token before client cancel')
            rows.append(out)
            continue
        reply = server.open(endpoint, body)
        out = row(name, endpoint, mode, body, reply, started)
        response = reply[1]
        if mode == 'control':
            check(out, spec_decode_refusal, reply[0], reply[2])
        if response is not None:
            responses.append({'name': name, 'response': cast(Json, response)})
            if name == 'tokenize':
                check(out, api.stop_token, response)
                if out['check'] == 'pass':
                    stop = api.stop_token(response)
            elif mode == 'control':
                pass
            elif name in ('stop', 'eos'):
                check(out, api.check_terminator, name, eos, response)
            elif out.get('finish_reason') in ('stop', 'tool_calls'):
                check(out, chat_check, name, body, response)
            else:
                out['check'] = f'not-run: capped at {MAX_TOKENS} tokens ({out.get("finish_reason")})'
        rows.append(out)
    return rows


def verdict(rows: list[dict[str, Json]]) -> dict[str, Json]:
    controls = [r for r in rows if r.get('mode') == 'control']
    rows = [r for r in rows if r.get('mode') != 'control']
    refused = [str(r['name']) for r in rows if isinstance(r.get('status'), int) and 400 <= int(str(r['status'])) < 500]
    broken = [str(r['name']) for r in rows if (r.get('status') is None and 'not-run' not in str(r.get('check', '')))
              or (isinstance(r.get('status'), int) and int(str(r['status'])) >= 500)]
    failed = [str(r['name']) for r in rows if str(r.get('check', '')).startswith('fail')]
    blocked = [str(r['name']) for r in rows if r.get('status') is None and 'not-run' in str(r.get('check', ''))]
    reading: list[Json] = []
    if refused:
        reading.append('PROBE WRONG FOR THIS IMAGE: production refused '+', '.join(refused)+' (see error_body)')
    if broken:
        reading.append('NO ANSWER (transport or 5xx): '+', '.join(broken)+'; rerun, this is not a probe verdict')
    if failed:
        reading.append('PROBE SEMANTICS FAIL on production: '+', '.join(failed))
    if blocked:
        reading.append('NOT RUN (an earlier probe failed): '+', '.join(blocked))
    confirmed = [r.get('check') == 'pass' for r in controls if 'not-run' not in str(r.get('check', ''))]
    cause: bool | None = all(confirmed) if confirmed else None
    reading.append({True: 'CAUSE CONFIRMED on production: the R16 logit_bias probe is refused under speculative decoding',
                    False: 'CAUSE NOT REPRODUCED: production did not refuse the R16 logit_bias probe as expected (see control row)',
                    None: 'cause control not run'}[cause])
    ok = not (refused or broken or failed or blocked)
    return {'verdict': 'PASS' if ok else 'FAIL', 'cause_confirmed': cause, 'refused': cast(Json, refused), 'no_answer': cast(Json, broken),
            'semantic_failures': cast(Json, failed), 'not_run': cast(Json, blocked),
            'reading': cast(Json, reading if not ok else ['every admission API probe accepted by production; exact probes exact', *reading])}


def source() -> dict[str, Json]:
    def git(*args: str) -> str:
        return subprocess.run(['git', '-C', str(ROOT), *args], check=True, capture_output=True, text=True).stdout.strip()
    files = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
             ('offload/launch/windows/pp4_admission_requests.py', 'offload/launch/windows/pp4_api_rehearsal.py')}
    return {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain', '--', *files)),
            'sha256': cast(Json, files)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--url', default=PRODUCTION)
    parser.add_argument('--eos', type=int, required=True, help="the model's verified EOS id (admission-eos-source.json)")
    parser.add_argument('--out', type=Path, required=True, help='fresh directory for rehearsal.json + responses.jsonl')
    parser.add_argument('--timeout', type=float, default=TIMEOUT_S)
    args = parser.parse_args(argv)
    ck.require(args.eos >= 0, 'verified model EOS token required')
    if not args.execute:
        print(json.dumps({'dry': True, 'url': args.url, 'max_tokens_cap': MAX_TOKENS, 'probes': [
            {'name': n, 'endpoint': e, 'mode': m, 'settings': settings(b)} for n, e, m, b in plan(args.eos, None)]},
            sort_keys=True))
        return 0
    ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'coordinator required')
    ck.require(not args.out.exists(), 'fresh rehearsal directory required')
    provenance = source()
    args.out.mkdir(parents=True)
    responses: list[dict[str, Json]] = []
    started = time.time()
    rows = rehearse(Production(args.url.rstrip('/'), args.timeout), args.eos, responses)
    result: dict[str, Json] = {**verdict(rows), 'url': args.url, 'eos': args.eos, 'max_tokens_cap': MAX_TOKENS,
        'started_unix': started, 'seconds': round(time.time()-started, 3), 'source': provenance, 'probes': cast(Json, rows)}
    held = args.out/'responses.jsonl'
    held.touch(mode=0o600)
    held.write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in responses))
    (args.out/'rehearsal.json').write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('verdict', 'cause_confirmed', 'reading', 'refused', 'semantic_failures')}, sort_keys=True))
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
