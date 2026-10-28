"""PF1-CAUSE2 --phase match CPU rehearsal: no SSH, HTTP, Spark, x86host or GPU. Every scenario ends in the W8 restore."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import sys
import time
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest

from offload.prefill_maxlen.evidence import Json, record
from offload.launch.windows import pf1_window as w
from offload.launch.windows.tests import test_pf1_window as base
from offload.prefill_cause2 import match_requests as mr
from offload.prefill_cause2.emit_verify import render as fix_verifier
from offload.prefill_trace.emit_verify import render as verifier
from offload.prefill_trace import window_requests as wr

MATCH_TTFT = {'off-524': 91.3, 'cause-524': 69.2, 'fix-524': 69.6}
OFF, CAUSE, FIX = w.MATCH_ARMS


def cells(ttft: dict[str, tuple[float, float]]) -> dict[str, list[dict[str, Json]]]:
    return {name: [base.cold_row(524288, 150000, rep, t) for rep, t in enumerate(pair)] for name, pair in ttft.items()}


def probs(shift: float = 0.0) -> dict[str, Json]:
    p = {hashlib.sha256(f't{i}'.encode()).hexdigest(): v for i, v in enumerate((.6 - shift, .3 + shift, .05, .04))}
    return cast(dict[str, Json], p)


def check_rows(role: str, *, diverge: int = -1, shift: float = 0.0) -> dict[str, Json]:
    greedy: list[Json] = [{'type': 'greedy', 'i': i, 'completion_tokens': 300, 'finish': 'length',
                           'text_sha256': hashlib.sha256(f'answer{i}{"x" if i == diverge else ""}'.encode()).hexdigest()}
                          for i in range(6)]
    suffixes = ('a', 'aa') if role == 'off' else ('b',)
    logit: list[Json] = [{'type': 'logit', 'role': role, 'salt_suffix': s, 'prompt_sha256': 'p' * 64,
                          'probabilities': probs(shift if s == 'b' else .001 if s == 'aa' else 0.0)} for s in suffixes]
    return {'greedy': greedy, 'logit': logit}


def checks(**fix: object) -> dict[str, dict[str, Json] | None]:
    return {'off-524': check_rows('off'), 'cause-524': None, 'fix-524': check_rows('fix', **fix)}  # type: ignore[arg-type]


# ---------------------------------------------------------------- the pre-registered reading

def test_reading_outcomes() -> None:
    ok = {'off-524': (91.8, 90.8), 'cause-524': (69.2, 69.9), 'fix-524': (69.6, 70.1)}
    r = w.match_reading(cells(ok), checks())
    assert r['verdict'] == 'CONFIRMED' and r['speed'] == 'FIX_REPRODUCES_CAUSE'
    assert record(r['safety'])['status'] == 'PASS' and record(r['safety'])['greedy_identical'] == 6
    assert r['fix_vs_cause'] == pytest.approx(69.85 / 69.55 - 1)
    refuted = {**ok, 'fix-524': (90.9, 91.4)}
    assert w.match_reading(cells(refuted), checks())['verdict'] == 'REFUTED'
    state = {**ok, 'cause-524': (90.0, 90.5)}
    assert w.match_reading(cells(state), checks())['verdict'] == 'CAUSE_NOT_REPRODUCED'
    half = {**ok, 'fix-524': (80.0, 80.5)}                        # neither like cause nor like off
    assert w.match_reading(cells(half), checks())['verdict'] == 'AMBIGUOUS'
    noisy = {**ok, 'off-524': (91.8, 86.0)}
    assert w.match_reading(cells(noisy), checks())['speed'] == 'AMBIGUOUS'
    one_slow_rep = {**ok, 'fix-524': (69.6, 76.0)}                # > 5 % from cause on one rep, spread > 3 %
    assert w.match_reading(cells(one_slow_rep), checks())['verdict'] == 'AMBIGUOUS'
    weak = {'off-524': (85.0, 85.5), 'cause-524': (69.2, 69.9), 'fix-524': (69.6, 70.1)}   # cause only 18 % faster
    assert w.match_reading(cells(weak), checks())['verdict'] == 'CAUSE_NOT_REPRODUCED'


def test_unsafe_overrides_speed_and_missing_checks_never_confirm() -> None:
    ok = cells({'off-524': (91.8, 90.8), 'cause-524': (69.2, 69.9), 'fix-524': (69.6, 70.1)})
    greedy = w.match_reading(ok, checks(diverge=3))
    assert greedy['verdict'] == 'UNSAFE' and record(greedy['safety'])['greedy_identical'] == 5
    logit = w.match_reading(ok, checks(shift=.2))
    assert logit['verdict'] == 'UNSAFE' and record(record(logit['safety'])['logit'])['status'] == 'FAILED'
    missing = w.match_reading(ok, {'off-524': None, 'cause-524': None, 'fix-524': None})
    assert missing['verdict'] == 'AMBIGUOUS' and record(missing['safety'])['status'] == 'NOT RUN'


def test_reading_refuses_unpaired_or_incomplete_cells() -> None:
    ok = cells({'off-524': (91.8, 90.8), 'cause-524': (69.2, 69.9), 'fix-524': (69.6, 70.1)})
    ok['fix-524'][0]['prompt_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='paired'):
        w.match_reading(ok, checks())
    short = cells({'off-524': (91.8, 90.8), 'cause-524': (69.2, 69.9), 'fix-524': (69.6, 70.1)})
    short['off-524'].pop()
    with pytest.raises(ValueError, match='both repetitions'):
        w.match_reading(short, checks())
    bad = checks()
    record(bad['fix-524'])['logit'] = []
    with pytest.raises(ValueError, match='logit checks incomplete'):
        w.match_reading(cells({'off-524': (91.8, 90.8), 'cause-524': (69.2, 69.9), 'fix-524': (69.6, 70.1)}), bad)


# ---------------------------------------------------------------- identity, verifier, lineage, reap

def fix_identity(arm: w.Arm, run: str, host: int = 0) -> dict[str, Json]:
    live = base.live_identity(arm, run, host)
    if arm.fix:
        record(live['mounted']).update({w.DIST + p: sha for p, sha in w.overlay_pins(fix=True).items()})
        record(live['env'])['VLLM_DSV41_MOE_FENCE'] = '1'
    return live


def test_fix_identity_is_production_plus_fence_only() -> None:
    w.runtime(base.prod_identity(), fix_identity(FIX, 'r'), FIX, 'r')
    mutations: tuple[tuple[Callable[[dict[str, Json]], object], str], ...] = (
            (lambda d: record(d['env']).pop('VLLM_DSV41_MOE_FENCE'), 'VLLM_DSV41_MOE_FENCE'),
            (lambda d: record(d['env']).update({'VLLM_DSV41_PREFILL_TRACE': '1'}), 'env drift'),
            (lambda d: record(d['mounted']).update({w.DIST + 'vllm_exl3_moe/tf_prompt.py': '0' * 64}), 'mounted'),
            (lambda d: record(d['mounted']).pop(w.DIST + 'vllm/v1/worker/gpu/dsv41_moe_fence.py'), 'mounted'))
    for mutate, message in mutations:
        live = fix_identity(FIX, 'r')
        mutate(live)
        with pytest.raises(ValueError, match=message):
            w.runtime(base.prod_identity(), live, FIX, 'r')
    prod = base.prod_identity()
    record(prod['env'])['VLLM_DSV41_MOE_FENCE'] = '0'
    with pytest.raises(ValueError, match='already sets'):
        w.runtime(prod, fix_identity(FIX, 'r'), FIX, 'r')
    # the off and cause arms of the match are the existing arms, judged by the existing rules
    assert OFF == w.OFF_ARMS[1] and CAUSE == w.CAUSE_ARMS[1] and OFF.config == w.CONFIG and FIX.config == w.FIX_CONFIG


def test_fix_verifier_has_exact_14_pins() -> None:
    pins = ast.literal_eval(fix_verifier().split('pins = ', 1)[1].splitlines()[0])
    assert len(pins) == 14
    for path, sha in w.overlay_pins(fix=True).items():
        assert pins[w.DIST + path] == sha
    assert fix_verifier() not in (verifier(False), verifier(True), verifier(True, cause=True))


def test_cause_window_pins_accept_cw_and_refuse_tampering(tmp_path: Path) -> None:
    cw = w.CAMPAIGN / 'PF1-CW-20261026T024340Z'
    if not cw.is_dir():
        pytest.skip('window CW evidence not present')
    w.cause_window_pins(cw)
    copy = tmp_path / 'cw'
    copy.mkdir()
    summary = json.loads((cw / 'summary.json').read_text())
    for name in summary['artifact_sha256']:
        shutil.copy(cw / name, copy / name)
    identity = json.loads((cw / 'cause-524-identity.json').read_text())
    identity['ranks'][1]['mounted'][w.DIST + 'vllm_exl3_moe/tf_experts.py'] = '0' * 64
    (copy / 'cause-524-identity.json').write_text(json.dumps(identity))
    summary['artifact_sha256']['cause-524-identity.json'] = w.digest(copy / 'cause-524-identity.json')
    (copy / 'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='bytes window CW mounted'):
        w.cause_window_pins(copy)
    (copy / 'checkpoints.jsonl').write_text('changed\n')
    with pytest.raises(ValueError, match='evidence changed'):
        w.cause_window_pins(copy)


def test_match_reap_pattern_covers_all_three_configs_and_never_itself() -> None:
    script = w.REAP.replace('@TAG@', 'tag').replace('PF1-window[.]env', 'PF1-(window|cause|fix)[.]env')
    pattern = 'start_exp[.]sh offload/launch/configs/PF1-(window|cause|fix)[.]env'
    assert script.count(pattern) == 2
    for config in ('PF1-window.env', 'PF1-cause.env', 'PF1-fix.env'):
        assert re.search(pattern, f'timeout 900 bash offload/launch/start_exp.sh offload/launch/configs/{config}')
    assert not re.search(pattern, 'bash -c ' + script)
    assert not re.search(pattern, 'bash offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-pack.env')


# ---------------------------------------------------------------- orchestration with a Backend fake

class MatchBackend(base.FakeBackend):
    def requests(self, arm: w.Arm, contexts: tuple[int, ...], reps: int) -> list[dict[str, Json]]:
        self.step(f'requests:{arm.name}')
        assert contexts == (150000,) and reps == w.MATCH_REPS
        return [base.cold_row(arm.cap, 150000, rep, MATCH_TTFT[arm.name] * (1 + .004 * rep)) for rep in range(reps)]

    def collect(self, arm: w.Arm, rows: list[dict[str, Json]]) -> dict[str, Json]:
        self.step(f'collect:{arm.name}')
        return {} if arm.trace else {'match': check_rows('fix' if arm.fix else 'off')}


def test_match_sequence_order_reading_and_restore() -> None:
    b = MatchBackend()
    rows: list[dict[str, Json]] = []
    result = w.window(b, 'match', rows.append)
    steps = [f'{s}:{a.name}' for a in w.MATCH_ARMS for s in ('start', 'requests', 'collect')]
    assert b.calls == ['preflight', 'open_hold', 'snapshot', 'stage_trace', *steps, 'restore:True']
    assert result['status'] == 'COMPLETE' and 'stop' not in result and 'gate' not in result
    assert record(result['match'])['verdict'] == 'CONFIRMED'
    assert [r.get('arm') for r in rows[:3]] == [a.name for a in w.MATCH_ARMS] and 'match' in rows[3]


@pytest.mark.parametrize('fail', ['open_hold', 'snapshot', 'stage_trace', *[
    f'{step}:{arm.name}' for arm in w.MATCH_ARMS for step in ('start', 'requests', 'collect')]])
def test_every_match_backend_stop_ends_in_restore(fail: str) -> None:
    b = MatchBackend(fail=fail)
    result = w.window(b, 'match', lambda row: None)
    assert 'injected' in str(result['stop']) and 'match' not in result
    assert b.calls[-2] == fail and b.calls[-1] == f'restore:{fail not in ("open_hold", "snapshot", "stage_trace")}'


def test_match_restore_failure_is_reported() -> None:
    result = w.window(MatchBackend(fail='requests:fix-524', restore_fails=True), 'match', lambda row: None)
    assert record(result['restore'])['status'] == 'FAILED'


# ---------------------------------------------------------------- the real Live backend through fake commands

class MatchRemote(base.FakeRemote):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.check_fail = ''

    def identity(self, rank: int) -> dict[str, Json]:
        if self.arm is None:
            return base.prod_identity(rank)
        live = fix_identity(self.arm, f'tag-{self.arm.name}', rank)
        live['id'] = f'{self.arm.name}-{rank}-{self.serial}'
        return live

    def call(self, label: str, argv: list[str], *, stdin: str | None = None, timeout: float = 120, inflight: int = 0) -> str:
        if argv[1:3] == ['-m', 'offload.prefill_cause2.match_requests']:
            self.sup.poll()
            self.commands.append(label)
            arm = cast(w.Arm, self.arm)
            assert inflight == 1 and not arm.trace and argv[argv.index('--role') + 1] == ('fix' if arm.fix else 'off')
            if self.check_fail == arm.name:
                raise ValueError(f'STOP: {label} failed rc1; see log')
            out = Path(argv[argv.index('--output') + 1])
            rows = check_rows('fix' if arm.fix else 'off')
            out.write_text(''.join(json.dumps(r) + '\n' for r in [
                {'type': 'identity', 'role': 'fix' if arm.fix else 'off', 'cap': arm.cap, 'model': 'm'},
                *cast(list[Json], rows['greedy']), *cast(list[Json], rows['logit'])]))
            return ''
        return super().call(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)

    def requests(self, argv: list[str], inflight: int) -> str:
        out = super().requests(argv, inflight)
        arm = cast(w.Arm, self.arm)
        path = Path(argv[argv.index('--output') + 1])
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        for row in rows[2:]:
            row['ttft_s'] = MATCH_TTFT[arm.name] * (1 + .004 * int(row['rep']))
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return out

    def remote(self, label: str, host: str, command: str, stdin: str | None) -> str:
        if 'bash offload/launch/start_exp.sh' in command and w.FIX_CONFIG in command:
            self.serial += 1
            self.arm = FIX
            return 'ready'
        if stdin == fix_verifier():
            return 'PF1 verified 14 engine sources\n'
        return super().remote(label, host, command, stdin)


@pytest.fixture
def match_harness(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[tuple[w.Live, MatchRemote, base.Clock]]:
    params = cast(dict[str, object], getattr(request, 'param', {}))
    monitor = tmp_path / 'monitor.py'
    monitor.write_text(base.MONITOR)
    out = tmp_path / 'PF1-MATCH-rehearsal'
    out.mkdir()
    clock = base.Clock()
    remote: list[MatchRemote] = []
    trip = tmp_path / 'trip'
    sup = base.NoTunnel(out, clock=clock, monitor=lambda host: [sys.executable, str(monitor), str(params.get('monitor', 'ok')),
                                                                str(trip)], demand=lambda: remote[0].demand)
    fake = MatchRemote(out, sup, clock, '', trip=None)
    remote.append(fake)
    live = w.Live(out, sup, remote='dsv41-deploy-offload', expected='0' * 40, data_dir=tmp_path, tag='tag',
                  stage_bundle=False, call=fake.call, plain=fake.plain, clock=clock, match=True)
    original = wr.corpus
    wr.corpus = lambda root: ('x', {})
    try:
        yield live, fake, clock
    finally:
        wr.corpus = original
        sup.close()


def test_live_match_rehearsal(match_harness: tuple[w.Live, MatchRemote, base.Clock]) -> None:
    live, fake, _clock = match_harness
    result = w.window(live, 'match', lambda row: None)
    assert 'stop' not in result, result.get('stop')
    assert record(result['match'])['verdict'] == 'CONFIRMED'
    assert [c for c in fake.commands if c.endswith('-start')] == ['off-524-start', 'cause-524-start', 'fix-524-start']
    assert [c for c in fake.commands if c.endswith('-checks')] == ['off-524-checks', 'fix-524-checks']
    assert {'trace-copy', 'cause-copy', 'fix-copy', 'fix-build'} <= set(fake.commands)
    assert fake.commands.index('fix-524-checks') > fake.commands.index('fix-524-stats-after')
    assert fake.restores == ['start', 'check', 'close']
    assert 'PF1-(window|cause|fix)[.]env' in fake.unsupervised[0][2]
    assert set(json.loads((live.out / 'arm-memory.json').read_text())) == {a.name for a in w.MATCH_ARMS}
    arms = record(result['arms'])
    assert record(record(record(arms['cause-524'])['checks'])['trace'])['sent'] == 3
    assert 'match' not in record(record(arms['cause-524'])['checks'])
    assert len(cast(list[Json], record(record(record(arms['fix-524'])['checks'])['match'])['greedy'])) == 6
    for name in ('off-524-checks.jsonl', 'fix-524-checks.jsonl'):
        assert 'answer' not in (live.out / name).read_text()
    assert 'off/cause/fix' in fake.hold


@pytest.mark.parametrize('fault', ['checks', 'stop-file', 'interrupt', 'deadline', 'pin', 'start-floor'])
def test_live_match_fix_arm_faults_restore(match_harness: tuple[w.Live, MatchRemote, base.Clock], fault: str) -> None:
    live, fake, clock = match_harness
    original_requests, original_remote = fake.requests, fake.remote
    if fault == 'checks':
        fake.check_fail = 'fix-524'

    def requests(argv: list[str], inflight: int) -> str:
        if fake.arm == FIX:
            if fault == 'stop-file':
                (live.out / 'STOP').write_text('CPU fake\n')
            elif fault == 'interrupt':
                raise KeyboardInterrupt
            elif fault == 'deadline':
                clock.now += 81 * 60
        return original_requests(argv, inflight)

    def remote(label: str, host: str, command: str, stdin: str | None) -> str:
        if label.startswith('fix-524-'):
            if fault == 'pin' and label.endswith('-verify-sparkworker-wired'):
                return 'PF1 verified 13 engine sources'
            if fault == 'start-floor' and command == w.HOST_PARITY:
                return 'MemAvailable: 5000000 kB\n'
        return original_remote(label, host, command, stdin)
    fake.requests = requests  # type: ignore[method-assign]
    fake.remote = remote  # type: ignore[method-assign]
    result = w.window(live, 'match', lambda row: None)
    assert 'stop' in result and 'match' not in result and 'fix-524' not in record(result['arms'])
    assert list(record(result['arms'])) == ['off-524', 'cause-524']
    assert fake.restores == ['start', 'check', 'close'] and 'PF1-(window|cause|fix)[.]env' in fake.unsupervised[0][2]
    assert 'fix-524' in json.loads((live.out / 'arm-memory.json').read_text())


@pytest.mark.parametrize('match_harness', [{'monitor': 'low'}, {'monitor': 'swap'}, {'monitor': 'gpu'}], indirect=True)
def test_live_match_guard_trips_restore(match_harness: tuple[w.Live, MatchRemote, base.Clock], tmp_path: Path) -> None:
    live, fake, _clock = match_harness
    trip = tmp_path / 'trip'
    original = fake.requests

    def tripped(argv: list[str], inflight: int) -> str:
        if fake.arm == CAUSE:
            fake.trip = trip
        return original(argv, inflight)
    fake.requests = tripped  # type: ignore[method-assign]
    result = w.window(live, 'match', lambda row: None)
    assert any(s in str(result['stop']) for s in ('below 4 GiB', 'new swap use', 'GPU telemetry failing')), result.get('stop')
    assert list(record(result['arms'])) == ['off-524'] and fake.restores == ['start', 'check', 'close']


def test_match_staging_never_overwrites_different_fix_sources(match_harness: tuple[w.Live, MatchRemote, base.Clock]) -> None:
    live, fake, _clock = match_harness
    original = fake.remote

    def remote(label: str, host: str, command: str, stdin: str | None) -> str:
        return 'DIFFERENT' if label == 'fix-remote' else original(label, host, command, stdin)
    fake.remote = remote  # type: ignore[method-assign]
    result = w.window(live, 'match', lambda row: None)
    assert 'never overwrite' in str(result['stop']) and fake.restores == ['check', 'close']
    assert not any(c.endswith('-start') for c in fake.commands)


def test_dry_plan_and_cli_refusals() -> None:
    text = w.plan('match', (150000,))
    assert 'fix-524: PF1_CAP=524288 PF1_TRACE=0' in text and '14-pin verify' in text and '~42-43 min HOLD' in text
    assert w.plan('cause', (150000,)) == w.plan('cause', (150000,)) and 'fix-524' not in w.plan('cause', (150000,))
    with pytest.raises(SystemExit):
        w.main(['--phase', 'match', '--on-contexts', '300000'])


# ---------------------------------------------------------------- the check driver (fake HTTP)

class Response:
    def __init__(self, data: bytes) -> None:
        self.data = data
    def __enter__(self) -> Response: return self
    def __exit__(self, *args: object) -> None: pass
    def read(self) -> bytes: return self.data


def run_driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str, *, cached: int = 0) -> list[dict[str, Json]]:
    monkeypatch.setattr(wr, 'corpus', lambda _root: (wr.HEADER + 'public text ' * 20000, {'public': 'sha'}))
    sent: list[dict[str, Json]] = []

    def urlopen(req: str | urllib.request.Request, timeout: float = 0) -> Response:
        url = req if isinstance(req, str) else req.full_url
        if url.endswith('/metrics'):
            return Response(b'vllm:num_requests_running{} 0\nvllm:num_requests_waiting{} 0\n')
        if url.endswith('/v1/models'):
            return Response(json.dumps({'data': [{'id': 'm', 'max_model_len': 524288}]}).encode())
        assert isinstance(req, urllib.request.Request) and isinstance(req.data, bytes)
        body = record(cast(Json, json.loads(req.data)))
        text = str(record(cast(list[Json], body['messages'])[0])['content'])
        if url.endswith('/tokenize'):
            return Response(json.dumps({'count': len(text) // 4}).encode())
        sent.append(body)
        usage = {'prompt_tokens': len(text) // 4, 'completion_tokens': int(cast(int, body['max_tokens'])),
                 'prompt_tokens_details': {'cached_tokens': cached}}
        choice: dict[str, Json] = {'message': {'content': 'PRIVATE_ANSWER'}, 'finish_reason': 'length'}
        if body.get('logprobs'):
            choice['logprobs'] = {'content': [{'top_logprobs': [{'token': 'A', 'logprob': -.1}, {'token': 'B', 'logprob': -2.5}]}]}
        return Response(json.dumps({'choices': [choice], 'usage': usage}).encode())
    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    out = tmp_path / f'{role}.jsonl'
    monkeypatch.setattr(sys, 'argv', ['mr', '--execute', '--cap', '524288', '--role', role, '--cache-salt-prefix', 'tag',
                                      '--deadline-utc', str(time.time() + 60), '--output', str(out), '--data-dir', str(tmp_path),
                                      '--ledger', str(tmp_path / 'ledger.jsonl')])
    mr.main()
    greedy = [b for b in sent if not b.get('logprobs')]
    assert [b['messages'] for b in greedy] == [[{'role': 'user', 'content': p}] for p in mr.PROMPTS]
    assert all(b['temperature'] == 0 and b['max_tokens'] == 300 and b['chat_template_kwargs'] == {'thinking': False}
               for b in greedy)
    return [record(cast(Json, json.loads(x))) for x in out.read_text().splitlines()]


def test_driver_records_hashes_and_numbers_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    off = run_driver(tmp_path, monkeypatch, 'off')
    assert [r['type'] for r in off] == ['identity'] + ['greedy'] * 6 + ['logit'] * 2
    assert 'PRIVATE' not in (tmp_path / 'off.jsonl').read_text() and 'public text' not in (tmp_path / 'off.jsonl').read_text()
    assert off[1]['text_sha256'] == hashlib.sha256(b'PRIVATE_ANSWER').hexdigest()
    assert [r['salt_suffix'] for r in off[7:]] == ['a', 'aa'] and off[7]['target_ctx'] == 8192
    assert sum(cast(dict[str, float], off[7]['probabilities']).values()) == pytest.approx(0.9048 + 0.0821, abs=1e-3)
    events = [json.loads(x)['event'] for x in (tmp_path / 'ledger.jsonl').read_text().splitlines()]
    assert events == ['send_intent', 'completed'] * 8
    (tmp_path / 'f').mkdir()
    fix = run_driver(tmp_path / 'f', monkeypatch, 'fix')
    assert [r['salt_suffix'] for r in fix if r['type'] == 'logit'] == ['b']
    (tmp_path / 'c').mkdir()
    with pytest.raises(ValueError, match='exact cold'):
        run_driver(tmp_path / 'c', monkeypatch, 'fix', cached=4096)


def test_driver_prompts_equal_the_w5_greedy_check() -> None:
    source = w.CAMPAIGN / 'W5/greedy_check.py'
    if not source.is_file():
        pytest.skip('W5 greedy_check.py not present')
    tree = ast.parse(source.read_text())
    found = next(n.value for n in tree.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'PROMPTS')
    assert ast.literal_eval(found) == mr.PROMPTS
    assert '"temperature": 0, "max_tokens": 300, "chat_template_kwargs": {"thinking": False}' in source.read_text()


def test_driver_dry_run_and_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, 'argv', ['mr', '--cap', '524288', '--role', 'off', '--cache-salt-prefix', 't',
                                      '--output', str(tmp_path / 'x')])
    mr.main()
    assert json.loads(capsys.readouterr().out)['executed'] is False and not (tmp_path / 'x').exists()
    for extra in (['--deadline-utc', '1'], ['--deadline-utc', str(time.time() + 60), '--url', 'http://HEAD_IP:8000']):
        monkeypatch.setattr(sys, 'argv', ['mr', '--execute', '--cap', '524288', '--role', 'off', '--cache-salt-prefix', 't',
                                          '--output', str(tmp_path / 'x'), *extra])
        with pytest.raises(SystemExit):
            mr.main()
