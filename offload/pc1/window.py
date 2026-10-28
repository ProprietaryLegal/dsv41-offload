"""PC1 current-production OFF/arm/OFF window. Dry by default; coordinator only, one arm per HOLD."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from offload.launch import warmup as ops1_warmup
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import nbw_cells as greedy_driver
from offload.launch.windows import up2_checks as ck
from offload.launch.windows import chain, hold_admission, spark_quiet, production_health
from offload.launch.windows import post_close_watch as watchdog
from offload.launch.windows import scoped_remote as owner
from offload.launch.windows.up2_low import X86HOST_PROBE, terminate
from offload.prefill_maxlen.evidence import Json, number, record
from offload.pc1 import build as b, cells, metrics, host_gates, capacity, headroom

CONFIG = 'offload/pc1/window.env'
SUBMIT_S, ABORT_S, RESTORE_S = 73 * 60, 73 * 60, 85 * 60
# ESTIMATED from measured startup + NBW-W3 cold cells; budget.json records
# hashes, arithmetic and unmeasured mixed/host-evidence extrapolations explicitly.
STARTUP_S, WARMUP_S, TRANSITION_S, QUALIFY_S = 404, 90, 60, 30
GREEDY_S = 60  # ESTIMATED six W5 checks plus first-divergence diagnostics, not a GB10 measurement
SETUP_S, RECOVERY_S = 120, 720
PREPARE_S = STARTUP_S + WARMUP_S + TRANSITION_S + GREEDY_S
# Drafter faults (PC1-threshold512-W-20261027T181314Z: one drafter disconnect inside a CONTROL start's 150K repeat failed
# 39 rounds, and the old rule stopped the whole window). WIN2 reads the drafter counters around every repeat: a small
# fault inside one measured repeat is attributed to that repeat of that start, logged, and the repeat is re-run once with
# a fresh salt; anything larger, a third fault in one start, or any growth outside a measured repeat still STOPs.
FAULT_COUNTERS = ('rounds_failed', 'rounds_some_failed', 'disconnects', 'fallback_timeout', 'fallback_not_ready',
                  'fallback_resync_pending', 'resync_rounds')
TRANSIENT_FAILED_ROUNDS = 64  # ESTIMATED from that one event: 39 failed rounds, about 1.6 s of ~40 ms rounds
TRANSIENT_PER_START = 2
KERNEL, DRIVER = '7.0.0-1019-nvidia', '580.178.04'


def block_seconds(group: str) -> int:
    """WIN2: one start runs its group's plan (decision cells x3, reported cells x1)."""
    plan = cells.plan_for(group)
    return PREPARE_S + sum(cells.ESTIMATES[n] * plan.repeats(n) for n in plan.cells)


def arm_seconds(group: str, role: str) -> int:
    base = block_seconds(group)
    if role == 'control':
        return base
    return base + QUALIFY_S + (headroom.STARTUP_EXTRA_S if role in b.LARGE_ARMS else 0)


class LargeChunkSupervisor(pf.Supervisor):
    """Keep the shared monitor/ownership; strengthen only this window's floor."""
    def poll(self) -> None:
        super().poll()
        for host, kb in (self.arm_minima or {}).items():
            ck.require(kb >= headroom.FLOOR_KB, f'PC1 {host} MemAvailable {kb} kB below 3 GiB')


def prepare_seconds(arm: str) -> int:
    return PREPARE_S + (headroom.STARTUP_EXTRA_S if arm in b.LARGE_ARMS else 0)



@dataclass(frozen=True)
class Arm(pf.Arm):
    role: str = 'control'

    @property
    def config(self) -> str:
        return CONFIG


def arms(group: str) -> tuple[Arm, ...]:
    ck.require(group in b.ARMS[1:], 'one isolated group required')
    return tuple(Arm(f's{i}-{role}', 327680, 0, role=role)
                 for i, role in enumerate(('control', group, 'control'), 1))


def source_pins(arm: str) -> dict[str, str]:
    return b.source_pins(arm)


def runtime(prod: dict[str, Json], live: dict[str, Json], arm: Arm) -> None:
    b.admit(arm.role)
    before, after = record(prod['env']), record(live['env'])
    ck.require(after == before, 'PC1 env drift beyond registered settings')
    ck.require({k: v for k, v in record(prod['mounted']).items() if k != '/serve.sh'} ==
        {k: v for k, v in record(live['mounted']).items() if k != '/serve.sh'}, 'PC1 mounted source drift')
    a = shlex.split(str(prod['argv']).replace('\\\n', ' '))
    actual = shlex.split(str(live['argv']).replace('\\\n', ' '))
    expected_argv = list(a)
    for option, value in zip(('--port', '--max-num-batched-tokens', '--long-prefill-token-threshold'),
                             ('8001', *map(str, b.SETTINGS[arm.role]))):
        ck.require(a.count(option) == actual.count(option) == 1, 'ambiguous runtime knob ' + option)
        expected_argv[a.index(option) + 1] = value
    ck.require(actual == expected_argv, 'PC1 argv drift beyond experiment port/registered pair')
    # Only after proving exact actual equality, use PF1's other engine/target/image/Engram invariants.
    normal_prod = {**prod, 'env': {**before, 'VLLM_ENGRAM_REMOTE_REQUIRED': '0'}}
    normal_live = {**live, 'env': {**before, 'VLLM_ENGRAM_REMOTE_REQUIRED': '1'},
                   'mounted': prod['mounted'], 'argv': shlex.join([('8001' if i == a.index('--port') + 1 else v)
                                                               for i, v in enumerate(a)])}
    pf.runtime(normal_prod, normal_live, arm, arm.name)


def greedy_checks(result: dict[str, Json]) -> None:
    """Shared NBW diagnosis with the explicitly registered PC1 Round3 policy."""
    ck.require(result.get('policy') == cells.GREEDY_POLICY
        and result.get('near_tie_nats') == cells.GREEDY_NEAR_TIE_NATS,
        'PC1 greedy policy/threshold registration missing or different')
    checks = ck.rows(result.get('checks', []))
    ck.require([r.get('i') for r in checks] == list(range(6))
        and all(r.get('identical') is True or r.get('near_tie') is True for r in checks)
        and number(result.get('identical', -1)) + number(result.get('near_tie', -1)) == 6,
        'PC1 greedy divergence that is not a near-tie, or missing checks')
    ck.require(result.get('identical') == sum(r.get('identical') is True for r in checks)
        and result.get('near_tie') == sum(r.get('identical') is not True for r in checks),
        'greedy aggregate evidence differs from ordered checks')


def decision(group: str, starts: dict[str, Json]) -> dict[str, Json]:
    expected = arms(group)
    if list(starts) != [a.name for a in expected] or any(record(v).get('status') != 'COMPLETE' for v in starts.values()):
        return {'status': 'NOT A CANDIDATE', 'reason': 'all mandatory complete bracket starts required'}
    ck.require(all(record(starts[a.name]).get('role') == a.role for a in expected), 'start role mismatch')
    for arm in expected:
        greedy_checks(record(record(starts[arm.name]).get('greedy', {})))
    # WIN2 (NDM 2026-10-27 "Fairer win rule is essential"): pooled/paired repeats, at most two decision cells.
    result = metrics.decision_v2(group, [record(record(starts[a.name])['cells']) for a in expected])
    faults = {a.name: (a.role, len(ck.rows(record(record(starts[a.name]).get('checks', {})).get('drafter_faults', []))))
              for a in expected}
    attribution = metrics.fault_attribution(faults)
    result['drafter_faults'] = attribution
    if attribution['verdict'] == 'ARM-SUSPECT':
        # Reliability veto: the speed verdict stays visible in the cells; the arm is not promotable.
        result['speed_status'] = result['status']
        result['status'] = 'NOT A CANDIDATE'
        result['reason'] = str(attribution['note'])
        result['groups'] = {k: ('NOT A CANDIDATE' if v == 'CANDIDATE' else v) for k, v in record(result['groups']).items()}
    return result


ARCHIVE = pf.CAMPAIGN / 'control-archive.jsonl'


def fault_levels(stats: dict[str, Json]) -> dict[str, float]:
    """Drafter fault counters, watchdog fires, breaker trips and proxy respawns from one proxy stats snapshot."""
    counters = record(stats['counters'])
    levels = {key: number(counters.get(key, 0)) for key in FAULT_COUNTERS}
    levels['watchdog_fires'] = number(record(stats['async_watchdog'])['fires'])
    levels['breaker_trips'] = number(record(stats.get('breaker') or {}).get('trips', 0))
    levels['async_respawns'] = number(stats.get('async_respawns', 0))
    return levels


def archive_controls(result: dict[str, Json], group: str, tag: str, out: Path) -> None:
    """WIN2 checklist 4: append every repeat of every control start that COMPLETED, even when the window stopped later.

    Rows (archive schema v3, WIN2-R2) carry max_tokens, the 40-hex production commit, the start's own proven clock cap
    (win2_history.start_clock_cap: both Sparks' "clock cap N MHz" lines in that start's log), date YYYYMMDD, start and
    repeat. Window, cell and metric names match win2_history's backfill, so a later backfill of this window skips them.
    """
    from offload.launch.windows import decision_v2 as d2, win2_history
    plan = cells.plan_for(group)
    starts = record(result.get('starts', {}))
    utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    try:
        date = win2_history.window_date(out)
    except ValueError:
        date = utc[:10].replace('-', '')
    rows: list[dict[str, Json]] = []
    caps: set[int] = set()
    for arm in arms(group):
        start = record(starts.get(arm.name, {}))
        if arm.role != 'control' or start.get('status') != 'COMPLETE':
            continue
        cap, source = win2_history.start_clock_cap(out, arm.name)
        caps.add(cap)
        for name in plan.cells:
            reps = ck.rows(record(record(start['cells'])[name])['repetitions'])
            for metric in metrics.METRICS[name]:
                for repeat, row in enumerate(reps):
                    rows.append({'window': tag, 'runner': 'pc1', 'group': group, 'start': arm.name, 'cell': name,
                                 'metric': metric.name, 'value': metric.extract(row), 'repeat': repeat,
                                 'max_tokens': cells.BUDGET, 'commit': b.PRODUCTION, 'clock_cap_mhz': cap,
                                 'clock_source': source, 'date': date, 'utc': utc, 'source': 'pc1 window'})
    head_before = d2.archive_head(ARCHIVE)
    if rows:
        d2.archive_append(ARCHIVE, rows)
    spread: dict[str, Json] = {}
    for name in plan.decision:
        primary = next(m for m in metrics.METRICS[name] if m.decides)
        # One definition, one production commit, one clock cap (2150 MHz before 2026-10-27 19:23Z, 2300 after).
        spread[f'{name}/{primary.name}'] = (d2.archive_spread(ARCHIVE, name, primary.name, where={
            'max_tokens': cells.BUDGET, 'clock_cap_mhz': next(iter(caps)), 'commit': b.PRODUCTION})
            if len(caps) == 1 else None)
    result['archive'] = {'path': str(ARCHIVE), 'rows_appended': len(rows), 'head_before': head_before,
                         'head_after': d2.archive_head(ARCHIVE), 'clock_cap_mhz': cast(Json, sorted(caps)),
                         'spread_primary': spread}
    decision = record(result.get('decision', {}))
    if 'cells' in decision:
        # Reported beside the in-window control spread; the 2 % bar is never raised silently (WIN2.md).
        decision['archive_spread'] = spread


def clock_check(result: dict[str, Json], group: str, out: Path) -> None:
    """Every completed start's proven clock cap. A comparison across two caps (e.g. a thermal dial-back between starts,
    CLOCK-PERSIST.sh --now) is not a fair one: the speed verdict is kept as `speed_decision` and the window decides
    nothing."""
    from offload.launch.windows import win2_history
    found: dict[str, Json] = {}
    for arm in arms(group):
        if record(record(result.get('starts', {})).get(arm.name, {})).get('status') != 'COMPLETE':
            continue
        try:
            cap, source = win2_history.start_clock_cap(out, arm.name)
            found[arm.name] = {'clock_cap_mhz': cap, 'clock_source': source}
        except (OSError, ValueError) as exc:
            found[arm.name] = {'error': f'{type(exc).__name__}: {exc}'}
    result['clock_caps_by_start'] = cast(Json, found)
    proven = {record(v).get('clock_cap_mhz') for v in found.values() if 'clock_cap_mhz' in record(v)}
    decision = record(result.get('decision', {}))
    if len(proven) > 1 and 'cells' in decision:
        result['decision'] = {'status': 'NOT A CANDIDATE', 'speed_decision': decision,
                              'reason': f'starts ran at different clock caps {sorted(map(str, proven))}; no fair comparison'}


def finish_chain(result: dict[str, Json]) -> None:
    """WIN2.md 'restore': clear the chain marker only after this window's own strict W8 restore AND watch PASS."""
    if (chain.marker_present() and str(record(result['restore']).get('status', '')).startswith('RESTORED')
            and record(result.get('post_close_watch', {})).get('status') == 'PASS'):
        chain.clear()
        result['chain'] = 'ENDED: own strict W8 restore and watch PASS; marker cleared'


class Live(pf.Live):
    def __init__(self, out: Path, sup: pf.Supervisor, *, group: str,
                 remote: str, expected: str, data_dir: Path, tag: str, call: pf.Call | None = None,
                 plain: Callable[[str, list[str], float], tuple[int, str]] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        super().__init__(out, sup, remote=remote, expected=expected, data_dir=data_dir, tag=tag,
                         stage_bundle=False, call=call, plain=plain, clock=clock)
        host_gates.admit(group)
        headroom.admit(group)
        self.group = group
        self.restore_by = self.clock() + RESTORE_S
        self.owner_tags: list[str] = []
        self.owner_deadlines: dict[str, float] = {}
        self.recovery_serial = 0
        self.open_deadline = self.clock() + pf.DRAIN_S + 60
        self.hold_scope_deadline = self.open_deadline + RESTORE_S
        self.open_admitted = False
        self.admission_token = uuid.uuid4().hex
        self.round_baseline: dict[str, dict[str, Json]] = {}
        self.counter_seen: dict[str, dict[str, float]] = {}
        self.fault_level: dict[str, dict[str, float]] = {}
        self.drafter_faults: dict[str, list[dict[str, Json]]] = {}
        self.chain_mode = 'production'  # WIN2: 'chained' when the previous chained window left production down
        self.start_capacities: dict[str, dict[str, Json]] = {}
        self.started_at = 0.0

    @property
    def pin_files(self) -> tuple[str, ...]:
        paths = [p for p in (b.ROOT / 'offload/pc1').rglob('*') if p.is_file() and '__pycache__' not in p.parts
                 and 'tests' not in p.parts]
        paths += [b.ROOT / name for name in ('offload/pc1/window.py',
            'offload/launch/windows/nbw_cells.py', 'offload/launch/windows/pf1_window.py', 'offload/launch/windows/scoped_remote.py',
            'offload/launch/windows/hold_admission.py', 'offload/launch/windows/spark_quiet.py',
            'offload/launch/windows/post_close_watch.py', 'offload/launch/windows/production_health.py',
            'offload/launch/start_exp.sh', 'offload/launch/serve_exp.sh', 'offload/launch/dsv41_tp2_offload.sh',
            'offload/launch/warmup_serve.sh', 'offload/launch/warmup.py')]
        return tuple(sorted({str(p.relative_to(b.ROOT)) for p in paths}))

    def remember(self, tag: str, deadline: float) -> None:
        ck.require(math.isfinite(deadline), 'finite scoped deadline required')
        ck.require(self.owner_deadlines.get(tag, deadline) == deadline, 'immutable phase deadline changed')
        if tag not in self.owner_tags:
            self.owner_tags.append(tag)  # BEFORE submitting any possibly ambiguous command
            self.owner_deadlines[tag] = deadline

    def owned(self, label: str, host: str, tag: str, script: str, *, timeout: float, inflight: int = 0) -> str:
        deadline = self.hold_scope_deadline if tag.endswith('-hold') else self.sup.abort_at
        self.remember(tag, deadline)
        return self.ssh(label, host, owner.command('launch', tag, ['bash', '-c', script], deadline=deadline),
                        timeout=timeout, inflight=inflight)

    def fence_phase(self, tag: str, *, guarded: bool = False) -> None:
        def invoke(host: str, command: str) -> tuple[int, str]:
            mode = command.rsplit(' ', 2)[-2]
            label = f'owned-{mode}-{tag}-{host}'
            argv = ['bash', '-c', command] if host == 'localhost' else [*pf.SSH, host, command]
            if guarded:
                return 0, self.call(label, argv, timeout=60)
            return self.plain(label, argv, 60)
        owner.fence_all(tag, invoke, (*owner.NODES, 'localhost'))

    def recovery(self, label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        # Unique attempt identities allow a retry without ever reopening a fence.
        self.recovery_serial += 1
        tag = f'{self.tag}-restore-{label}-{self.recovery_serial}'
        deadline = self.restore_by - 90  # reserve final commit/cancel/retention
        self.remember(tag, deadline)
        ck.require(self.clock() < deadline, 'PC1 insufficient time for owned recovery; HOLD stays open')
        try:
            return self.plain(label, ['bash', '-c', owner.command('launch', tag, argv, deadline=deadline)], timeout)
        finally:
            self.fence_phase(tag)

    def preflight(self) -> None:
        host_gates.admit(self.group)
        b.verify_inputs()
        self.sup.begin()
        ck.require(pf.digest(pf.RESTORE) == pf.RESTORE_SHA256, 'W8 restore pin differs')
        greedy_driver.greedy_inputs()
        for name in ('offload/launch/warmup.py', 'offload/launch/warmup_serve.sh'):
            ck.require((b.ROOT / name).read_bytes() == b.production_file(name), 'OPS1 warm-up source drift')
        from offload.prefill_trace.window_requests import corpus
        corpus(self.data_dir)
        ck.require((pf.CAMPAIGN / 'D2-GEN.PAUSE').exists() or subprocess.run(
            ['tmux', 'has-session', '-t', '=d2-gen'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0,
            'D2 generation active without pause')
        production_files = ['dsv41.env', 'serve.sh', 'start.sh', 'stop.sh', 'firewall.sh',
                            'offload/launch/warmup.py', 'offload/launch/warmup_serve.sh', 'offload/launch/dsv41_tp2_offload.sh',
                            'offload/launch/window.sh']
        pin_lines = '\n'.join(b.digest(b.production_file(p)) + '  ' + p for p in production_files)
        self.ssh('ledger', 'sparkhead', 'cd ~/dsv41-deploy && test ! -e HOLD && '
            f'test "$(git rev-parse HEAD)" = {b.PRODUCTION} && '
            f'test "$(git rev-list -n1 {b.TAG})" = {b.PRODUCTION} && git diff --quiet HEAD -- . && '
            'printf %s ' + shlex.quote(pin_lines + '\n') + ' | sha256sum --check --status -')
        for host in pf.HOSTS:
            self.ssh('checkout-' + host, host, f'cd ~/{self.remote} && test "$(git rev-parse HEAD)" = {self.expected} && '
                     'test -z "$(git status --porcelain --untracked-files=no)"')
            pins = self.ssh('pins-' + host, host, f'cd ~/{self.remote} && sha256sum ' + shlex.join(self.pin_files))
            ck.require(pins.strip() == '\n'.join(f'{pf.digest(b.ROOT / p)}  {p}' for p in self.pin_files), 'remote reviewed source pins differ')
        # WIN2 chain mode: the ledger proved no HOLD; anything answering on :8000 counts as serving (chain.preflight
        # refuses a marker with a live server, and production mode still requires the exact K1 model below).
        answered = self.ssh('prod-probe', 'sparkhead', 'curl -s --max-time 10 http://localhost:8000/v1/models || true')
        placeholders: bool | None = None
        if chain.marker_present():
            # WIN2-R2: between chained windows only the hold placeholders keep NVO and local-law off the Sparks.
            placeholders = all('PLACEHOLDER PRESENT' in self.ssh('placeholders-' + host, host,
                f'({chain.PLACEHOLDER_PROBE}) && echo PLACEHOLDER PRESENT || echo PLACEHOLDER ABSENT').splitlines()
                for host in chain.PLACEHOLDER_HOSTS)
        self.chain_mode = chain.preflight(production_serving=bool(answered.strip()), hold_exists=False,
                                          placeholders_present=placeholders)
        if self.chain_mode == 'production':
            self.ssh('prod-health', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8000/health')
            model = ck.rows(self.json('prod-models', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8000/v1/models')['data'])
            ck.require(len(model) == 1 and model[0].get('max_model_len') == 327680
                and model[0].get('root') == '/srv/dsv41/legal-v2a', 'production cap/model differs')
        pf.worker(self.json('worker', 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)), self.clock())
        for host in pf.HOSTS:
            host_check(self.ssh('host-' + host, host, pf.HOST_PARITY))
            self.ssh('fresh-' + host, host, f'test ! -e ~/{owner.CONTROL}/{self.tag}-hold')

    def open_hold(self) -> float:
        ck.require(not self.open_admitted, 'PC1 HOLD cannot be reopened or renewed')
        self.open_admitted = True
        self.open_deadline = self.clock() + pf.DRAIN_S + 60
        self.hold_scope_deadline = self.open_deadline + RESTORE_S
        # Establish PC1 bounds as soon as HOLD is read, including if open/tunnel fails.
        try:
            self.owned('hold-open', 'sparkhead', self.tag + '-hold',
                hold_admission.open_script(self.tag + '-hold', self.admission_token,
                    f'PC1-WINDOW {self.tag}: isolated {self.group}', pf.DRAIN_S,
                    # S92 FIX 1: a chained start finds production down on purpose; only then may :8000 be unreachable.
                    allow_unreachable=chain.allow_unreachable(self.chain_mode)), timeout=pf.DRAIN_S + 60)
        finally:
            text = self.ssh('hold-read', 'sparkhead',
                hold_admission.command('check', self.tag + '-hold', self.admission_token))
            self.hold_written = f'PC1-WINDOW {self.tag}:' in text
            if self.hold_written:
                opened = pf.hold_opened(text, self.tag, self.clock())
                self.submit, self.sup.abort_at, self.restore_by = opened + SUBMIT_S, opened + ABORT_S, opened + RESTORE_S
                self.sup.bound = 'PC1 HOLD+73 abort bound'
                (self.out / 'deadlines.json').write_text(json.dumps({'opened': opened, 'submit': self.submit,
                    'abort': self.sup.abort_at, 'restore_by': self.restore_by}) + '\n')
        ck.require(self.hold_written, 'PC1 exact owned HOLD marker required')
        opened = pf.hold_opened(text, self.tag, self.clock())
        ck.require(opened <= self.open_deadline and self.restore_by <= self.hold_scope_deadline,
                   'PC1 late HOLD publication exceeds opening ceiling/placeholder lease')
        self.sup.tunnel([*pf.SSH, '-N', '-o', 'ExitOnForwardFailure=yes', '-L', '127.0.0.1:18001:127.0.0.1:8001', 'sparkhead'])
        return opened

    def capture(self, host: str, label: str) -> dict[str, Json]:
        current = super().capture(host, label)
        config = self.json(label + '-fullenv-' + host, host, "docker inspect --format '{{json .Config}}' dsv41-serve")
        env: dict[str, Json] = {}
        for value in ck.strings(config['Env']):
            key, equal, item = value.partition('=')
            ck.require(equal == '=' and key not in env, 'duplicate/malformed full container environment')
            env[key] = item
        current['env'] = env
        return current

    def verify_sources(self, host: str, arm: str) -> None:
        pins = source_pins(arm)
        got = self.json(arm + '-sources-' + host, host,
            'docker exec -i dsv41-serve python3 - ' + shlex.join(pins), pf.IMAGE_BASES)
        ck.require(got == pins, 'installed production/arm bytes differ')

    def snapshot(self) -> None:
        if self.chain_mode == 'chained':
            self.forward_snapshot()
            return
        for host in pf.HOSTS:
            current = self.capture(host, 'prod')
            host_gates.settings(str(current['argv']), pf.HOSTS.index(host), '8000', 'control')
            self.verify_sources(host, 'control')
            ck.require(current.get('image') == pf.IMAGE, 'production image pin differs')
            self.production.append(current)
            self.seen.add(str(current['id']))
        ck.require(self.production[0]['files'] == self.production[1]['files'], 'production target metadata differs by rank')
        self.production_idle()
        (self.out / 'rollback.json').write_text(json.dumps({'commit': b.PRODUCTION, 'tag': b.TAG,
            'restore_sha256': pf.RESTORE_SHA256, 'ranks': cast(Json, self.production)}, indent=1) + '\n')

    def forward_snapshot(self) -> None:
        """Chained start: production is down, so take the chain's production capture (the rollback.json of the window
        that started from serving production, whatever runner wrote it) instead of capturing a server that is not running."""
        source, data = chain.production_snapshot()  # sha256-checked against the marker (WIN2-R2)
        ck.require(data.get('commit') == b.PRODUCTION and data.get('tag') == b.TAG
                   and data.get('restore_sha256') == pf.RESTORE_SHA256, 'forwarded production snapshot is not the pinned production')
        ranks = ck.rows(data['ranks'])
        ck.require(len(ranks) == len(pf.HOSTS) and ranks[0]['files'] == ranks[1]['files']
                   and all(r.get('image') == pf.IMAGE for r in ranks), 'forwarded production identity incomplete')
        for rank, current in enumerate(ranks):
            host_gates.settings(str(current['argv']), rank, '8000', 'control')
            self.production.append(current)
            self.seen.add(str(current['id']))
        # S92 FIX 1: the gate inferred "production down" from :8000 alone; prove both Sparks quiet (no serve,
        # experiment or JIT container, no compute app) under this window's HOLD before the first start_exp.sh.
        for host in pf.HOSTS:
            text = self.ssh('chain-quiet-' + host, host,
                            spark_quiet.command('PC1 CHAIN QUIET', deadline=min(self.submit, self.clock() + 120)))
            ck.require('PC1 CHAIN QUIET' in text.splitlines(), f'{host}: Spark not quiet at the chained start')
        (self.out / 'rollback.json').write_text(json.dumps({'commit': b.PRODUCTION, 'tag': b.TAG,
            'restore_sha256': pf.RESTORE_SHA256, 'ranks': cast(Json, self.production),
            'chained_from': str(source), 'chained_from_sha256': pf.digest(source)}, indent=1) + '\n')

    def fits(self, arm: Arm) -> bool:
        planned = arms(self.group)
        ck.require(arm in planned, 'start is outside this isolated OFF/arm/OFF plan')
        remaining = sum(arm_seconds(self.group, a.role) for a in planned[planned.index(arm):])
        return self.clock() + remaining <= self.submit

    def start(self, base_arm: pf.Arm) -> None:
        ck.require(isinstance(base_arm, Arm), 'PC1 arm required')
        arm = cast(Arm, base_arm)
        headroom.admit(arm.role)
        if arm.role in b.LARGE_ARMS:
            headroom.control_reserve(arm.role, self.sup.arm_memory.get('s1-control', {}))
        ck.require(self.fits(arm), 'full mandatory start block cannot fit')
        self.sup.unready()
        for tag in list(self.owner_tags):
            if tag != self.tag + '-hold':
                self.fence_phase(tag, guarded=True)
        self.sup.arm_minima = {}
        self.sup.arm_memory[arm.name] = self.sup.arm_minima
        for host in pf.HOSTS:
            host_check(self.ssh(arm.name + '-host-' + host, host, pf.HOST_PARITY))
        self.started_at = self.clock()
        startup_limit = prepare_seconds(arm.role) - WARMUP_S - GREEDY_S + (QUALIFY_S if arm.role != 'control' else 0)
        start = (f'cd ~/{self.remote} && PC1_ARM={arm.role} '
            f'timeout --foreground {startup_limit} bash offload/launch/start_exp.sh {CONFIG}')
        start = hold_admission.command('stop', self.tag + '-hold', self.admission_token, ['bash', '-c', start])
        self.owned(arm.name + '-start', 'sparkhead', self.tag + '-' + arm.name + '-engine', start, timeout=startup_limit)
        self.ssh(arm.name + '-health', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8001/health')
        model = ck.rows(self.json(arm.name + '-model', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8001/v1/models')['data'])
        ck.require(len(model) == 1 and model[0].get('max_model_len') == 327680
            and model[0].get('root') == '/srv/dsv41/legal-v2a', 'experiment model/cap differs')
        current = [self.capture(host, arm.name) for host in pf.HOSTS]
        for rank, host in enumerate(pf.HOSTS):
            runtime(self.production[rank], current[rank], arm)
            ck.require(str(current[rank]['id']) not in self.seen, 'container reused across starts')
            self.verify_sources(host, arm.role)
            summary = self.rank_logs(arm, host, rank)
            if rank == 0:
                self.proxy_pid = int(number(cast(list[Json], summary['proxy_pids'])[0]))
        self.ids = [str(row['id']) for row in current]
        self.seen.update(self.ids)
        self.sup.mark_ready()
        (self.out / (arm.name + '-identity.json')).write_text(json.dumps({'ranks': cast(Json, current)}) + '\n')

    def capacity_report(self, arm: Arm) -> dict[str, Json]:
        return self.start_capacities.setdefault(arm.name,
            {**capacity.expectation(arm.role), 'reported_tokens': {}})

    def greedy(self, arm: Arm) -> dict[str, Json]:
        # Port of DEP1 9b72534. No alternate target comparison or near-tie implementation.
        deadline = min(self.submit, self.started_at + prepare_seconds(arm.role) + (QUALIFY_S if arm.role != 'control' else 0))
        ck.require(self.clock() < deadline, 'premeasurement greedy budget exhausted')
        output = self.out / f'{arm.name}-greedy.json'
        self.call(f'{arm.name}-greedy', self.driver('greedy', output, '--deadline', str(deadline)),
            timeout=min(GREEDY_S, deadline-self.clock()), inflight=1)
        result = ck.read(output)
        greedy_checks(result)
        ck.require(self.clock() < deadline, 'experimental greedy exceeded premeasurement allowance')
        return result

    def warmup(self, arm: Arm) -> None:
        ck.require(arm.name not in self.round_baseline, 'measurement baseline cannot be replaced')
        # DEP1 ordering: six greedy checks, then unchanged discarded OPS1 shapes1..4.
        deadline = min(self.submit, self.started_at + prepare_seconds(arm.role) + (QUALIFY_S if arm.role != 'control' else 0))
        ck.require(self.clock() < deadline, 'premeasurement warmup budget exhausted')
        # The OPS1 warm-up's own waves (up to max(SHAPES) concurrent) are owned demand, not foreign demand on :8001
        # UP3-k5-W-20261027T041627Z stopped at the concurrency-3 wave with owned0.
        self.owned(arm.name + '-warmup', 'sparkhead', self.tag + '-' + arm.name + '-warmup', f'cd ~/{self.remote} && bash offload/launch/warmup_serve.sh 8001 '
            'DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2', timeout=min(WARMUP_S, deadline-self.clock()),
            inflight=max(ops1_warmup.SHAPES))
        # DEP1 9b72534: establish one FULL baseline after greedy and OPS1.
        baseline = self.stats(f'{arm.name}-baseline')
        ck.counters(baseline, before=baseline)
        self.round_baseline[arm.name] = baseline
        self.counter_seen[arm.name] = {}
        self.fault_level[arm.name] = fault_levels(baseline)
        self.drafter_faults[arm.name] = []
        self.baseline_check(arm, baseline)

    def baseline_check(self, arm: Arm, current: dict[str, Json]) -> None:
        """DEP1 full baseline, retaining every subsequently observed sparse key too."""
        ck.require(arm.name in self.round_baseline, 'missing post-greedy/post-warmup measurement baseline')
        baseline = self.round_baseline[arm.name]
        ck.counters(current, before=baseline)
        after = record(current['counters'])
        seen = self.counter_seen[arm.name]
        for key, value in seen.items():
            ck.require(key in after and number(after[key]) >= value, f'{key}: missing/reset counter')
        for key, field in after.items():
            n = number(field)
            ck.require(math.isfinite(n) and n >= 0, f'{key}: invalid measurement counter')
        # Only attribute_faults() may raise the accepted level, and only for a fault inside one measured repeat.
        accepted = self.fault_level[arm.name]
        for key, value in fault_levels(current).items():
            name = 'watchdog' if key == 'watchdog_fires' else key
            ck.require(math.isfinite(value) and value >= 0, f'{name}: invalid drafter fault counter')
            ck.require(value <= accepted[key], f'{name} grew after warmup outside an attributed repeat')
            ck.require(value >= accepted[key], f'{name} reset after warmup')
        self.counter_seen[arm.name] = {key: number(value) for key, value in after.items()}

    def rank_logs(self, arm: Arm, host: str, rank: int) -> dict[str, Json]:
        raw = self.json(arm.name + '-logs-' + host, host, 'python3 -c ' + shlex.quote(pf.LOG_FILTER)
            + ' ' + shlex.quote(ck.STOP_LOG.pattern + '|fused SWA metadata off|warmup compile skipped|NCCL stays'))
        # Retain even a rejected capacity in this start's summary; rank1 normally
        # has no capacity line. Reporting is separate from the TTFT decision.
        record(self.capacity_report(arm)['reported_tokens'])[host] = raw['kv_tokens']
        summary = pf.logs(raw,
            rank, 327680, kv=int(number(capacity.expectation(arm.role)['expected_tokens'])))
        return summary

    def qualify(self, arm: Arm) -> dict[str, Json]:
        deadline = min(self.submit, self.started_at + prepare_seconds(arm.role) + (QUALIFY_S if arm.role != 'control' else 0))
        ck.require(self.clock() < deadline, 'startup/warmup/qualification budget exhausted')
        if arm.role == 'control':
            return {'status': 'unchanged production source/env verified'}
        host_gates.admit(arm.role)
        results: dict[str, Json] = {'policy': 'registered configuration pair; source and greedy gates separate'}
        for rank, host in enumerate(pf.HOSTS):
            ck.require(self.clock() < deadline, 'qualification deadline exhausted before submission')
            command = (f'cd ~/{self.remote} && python3 -m offload.pc1.host_gates '
                f'--arm {arm.role} --rank {rank} --container {shlex.quote(self.ids[rank])}')
            text = self.ssh(arm.name + '-host-gate-' + host, host, command,
                timeout=min(15, deadline - self.clock()))
            receipt = record(cast(Json, json.loads(text)))
            ck.require(self.clock() < deadline, 'host evidence exceeded its full block allowance')
            host_gates.validate(receipt, arm.role, rank, self.ids[rank])
            results[host] = receipt
        return results

    def driver(self, mode: str, output: Path, *extra: str) -> list[str]:
        return [sys.executable, '-m', 'offload.pc1.cells', '--execute', '--mode', mode, '--output', str(output),
            '--data-dir', str(self.data_dir), '--deadline', str(self.submit), *extra]

    def build_cuts(self) -> None:
        self.call('public-cuts', self.driver('cuts', self.out / 'cuts.json'), timeout=180)

    def cell(self, arm: Arm, name: str) -> dict[str, Json]:
        """WIN2: one driver call per repeat, drafter counters read before and after each (fault attribution)."""
        repeats = cells.plan_for(self.group).repeats(name)
        kept: list[Json] = []
        faulted: list[Json] = []
        for repeat in range(repeats):
            attempt = 0
            while True:
                row, faults = self.measure_repeat(arm, name, repeat, attempt)
                if not any(faults.values()):
                    break
                # Kept as evidence, never timed: the clean re-run with a fresh salt replaces it.
                faulted.append({**row, 'drafter_faults': cast(Json, faults), 'attempt': attempt})
                attempt += 1
                ck.require(self.replacement_fits(arm, name, repeat),
                           'faulted repeat cannot be re-run inside the submit budget')
            kept.append(row)
        result: dict[str, Json] = {'cell': name, 'batch': cells.batch_for(name), 'repetitions': kept,
                                   'faulted_repetitions': faulted}
        (self.out / (arm.name + '-' + name + '.json')).write_text(json.dumps(result, allow_nan=False) + '\n')
        return result

    def measure_repeat(self, arm: Arm, name: str, repeat: int, attempt: int) -> tuple[dict[str, Json], dict[str, int]]:
        label = f'{arm.name}-{name}-r{repeat}' + (f'-retry{attempt}' if attempt else '')
        before = self.stats(label + '-before')
        self.baseline_check(arm, before)
        output = self.out / (label + '.json')
        salt = self.tag + '-' + arm.name + (f'-retry{attempt}' if attempt else '')
        self.call(label, self.driver('cell', output, '--cell', name, '--cuts', str(self.out / 'cuts.json'),
            '--salt', salt, '--ledger', str(self.out / 'requests-ledger.jsonl'), '--repeat-index', str(repeat)),
            timeout=cells.ESTIMATES[name] * 2 + 30, inflight=3 if name == cells.CELLS[-1] else int(name[-1]))
        result = ck.read(output)
        ck.require(result.get('cell') == name, 'cell artifact identity differs')
        reps = ck.rows(result.get('repetitions', []))
        ck.require(len(reps) == 1 and reps[0].get('repeat') == repeat, 'repeat artifact identity differs from the WIN2 plan')
        after = self.stats(label + '-after')
        ck.counters(after, before=before)
        faults = self.attribute_faults(arm, name, repeat, attempt, before, after)
        self.baseline_check(arm, after)
        ck.require(number(record(after['counters']).get('rounds_ok', 0)) > number(record(before['counters']).get('rounds_ok', 0)),
            'remote drafter took no successful round')
        row = dict(reps[0])
        row['proxy_before'], row['proxy_after'] = before, after
        return row, faults

    def attribute_faults(self, arm: Arm, name: str, repeat: int, attempt: int,
                         before: dict[str, Json], after: dict[str, Json]) -> dict[str, int]:
        start, end = fault_levels(before), fault_levels(after)
        delta = {key: int(end[key] - start[key]) for key in end}
        ck.require(all(v >= 0 for v in delta.values()), 'drafter fault counter reset inside a repeat')
        if not any(delta.values()):
            return delta
        events = self.drafter_faults[arm.name]
        failed = delta['rounds_failed'] + delta['rounds_some_failed']
        transient = (delta['breaker_trips'] == 0 and delta['async_respawns'] == 0 and delta['watchdog_fires'] <= 1
                     and delta['disconnects'] <= 1 and failed <= TRANSIENT_FAILED_ROUNDS and len(events) < TRANSIENT_PER_START)
        event: dict[str, Json] = {'start': arm.name, 'role': arm.role, 'cell': name, 'repeat': repeat, 'attempt': attempt,
            'class': 'transient' if transient else 'systemic', 'delta': cast(Json, delta),
            'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        events.append(event)
        with (self.out / 'drafter-faults.jsonl').open('a') as handle:
            handle.write(json.dumps(event, allow_nan=False) + '\n')
        ck.require(transient, f'STOP: systemic drafter fault in {arm.name} {name} r{repeat}: {delta} (transient = at most '
                   f'one disconnect and one watchdog fire, <= {TRANSIENT_FAILED_ROUNDS} failed rounds, no breaker trip or '
                   f'respawn, <= {TRANSIENT_PER_START} per start)')
        self.fault_level[arm.name] = end
        return delta

    def replacement_fits(self, arm: Arm, name: str, repeat: int) -> bool:
        """The re-run plus every remaining planned repeat, cell and start must still fit before the submit deadline."""
        plan = cells.plan_for(self.group)
        this = cells.ESTIMATES[name] * (plan.repeats(name) - repeat)
        rest = sum(cells.ESTIMATES[n] * plan.repeats(n) for n in plan.cells[plan.cells.index(name) + 1:])
        planned = arms(self.group)
        later = sum(arm_seconds(self.group, a.role) for a in planned[planned.index(arm) + 1:])
        return self.clock() + this + rest + later <= self.submit

    def finish(self, arm: Arm) -> dict[str, Json]:
        result: dict[str, Json] = {}
        final = self.stats(arm.name + '-final-stats')
        self.baseline_check(arm, final)
        result['startup_baseline'] = self.round_baseline[arm.name]
        result['proxy'] = final
        result['observed_counter_keys'] = cast(Json, sorted(self.counter_seen[arm.name]))
        result['drafter_faults'] = cast(Json, list(self.drafter_faults[arm.name]))
        for rank, host in enumerate(pf.HOSTS):
            ck.require(self.ssh(arm.name + '-id-' + host, host, "docker inspect --format '{{.Id}}' dsv41-serve").strip() == self.ids[rank],
                'container changed during arm')
            result[host] = self.rank_logs(arm, host, rank)
        ck.require(self.sup.arm_minima is not None and set(self.sup.arm_minima) == set(pf.HOSTS), 'missing rank memory minima')
        result['memory_min_kb'] = cast(Json, dict(cast(dict[str, int], self.sup.arm_minima)))
        return result

    def restore(self, *, production_stopped: bool) -> dict[str, Json]:
        stopped_here = production_stopped
        # A chained start found production down: W8 must start it again even if no arm started.
        production_stopped = production_stopped or self.chain_mode == 'chained'
        self.sup.close()
        original = self.plain
        original_call = self.call
        def recovery_call(label: str, argv: list[str], *, stdin: str | None = None,
                          timeout: float = 120, inflight: int = 0) -> str:
            if stdin is not None:
                import base64
                encoded = base64.b64encode(stdin.encode()).decode()
                argv = [*argv[:-1], 'printf %s ' + shlex.quote(encoded) + ' | base64 -d | (' + argv[-1] + ')']
            rc, text = self.plain('verify-' + label, argv, timeout)
            ck.require(rc == 0, 'recovery identity command failed; HOLD stays open')
            return text
        self.call = recovery_call
        # Clamp cleanup, W8 and close transactions to ONE original HOLD deadline.
        def bounded(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
            if label in ('cancel-close', 'retain-close'):
                return original(label, argv, min(30, timeout))
            left = self.restore_by - self.clock()
            ck.require(left > 0, 'PC1 HOLD+85 exhausted; manual restore required; HOLD stays open')
            result = original(label, argv, min(left, timeout))
            ck.require(self.clock() < self.restore_by, 'PC1 recovery command exceeded original deadline')
            return result
        self.plain = bounded
        try:
            # Fence the HOLD writer BEFORE rereading it: a late opener must not
            # create a HOLD after recovery decided that production was untouched.
            for tag in list(self.owner_tags):
                self.fence_phase(tag)
            if not self.hold_written:
                rc, text = self.plain('hold-reread', [*pf.SSH, 'sparkhead',
                    hold_admission.command('check', self.tag + '-hold', self.admission_token)], 60)
                ck.require(rc == 0, 'PC1 HOLD ownership unreadable; HOLD stays open')
                self.hold_written = f'PC1-WINDOW {self.tag}:' in text
                if self.hold_written:
                    opened = pf.hold_opened(text, self.tag, self.clock())
                    self.restore_by = opened + RESTORE_S
            if not self.hold_written:
                ck.require(not stopped_here, 'PC1 production may be stopped without owned HOLD')
                if self.chain_mode == 'chained':
                    return {'status': 'NOT NEEDED HERE: no owned HOLD; production is down under the chain marker, '
                            'so the window end runs chain.abort'}
                return {'status': 'NOT NEEDED: no owned HOLD; all remote mutators fenced and absent'}
            ck.require(math.isfinite(self.restore_by), 'PC1 missing immutable restore deadline')
            if production_stopped:
                # dockerd children are outside cgroups: explicitly stop/remove
                # the same named experiment containers and require no GPU apps.
                rc, text = self.recovery('experiment-stop', [*pf.SSH, 'sparkhead',
                    f'cd ~/{self.remote} && ./stop.sh </dev/null'], 120)
                ck.require(rc == 0, 'PC1 experiment stop failed; HOLD stays open')
                for host in pf.HOSTS:
                    rc, text = self.plain('jit-wait-' + host, [*pf.SSH, host,
                        spark_quiet.command('PC1 EXPERIMENT ABSENT',
                            deadline=min(self.restore_by - 90, self.clock() + 350))], 360)
                    ck.require(rc == 0 and 'PC1 EXPERIMENT ABSENT' in text.splitlines(),
                               f'{host}: experiment/container/compute absence unproved; HOLD stays open')
            result: dict[str, Json] = {}
            steps = ('start', 'check', 'close') if production_stopped else ('check', 'close')
            for step in steps:
                # Close uses an exact suffix to install the HOLD-unlink shim on
                # every receiving host. W8 still performs its original checks.
                if step == 'close':
                    # Compare complete production identity BEFORE permitting deferred close.
                    for rank, host in enumerate(pf.HOSTS):
                        self.verify_sources(host, 'control')
                        restored = self.capture(host, 'restored')
                        if self.production:
                            prod = self.production[rank]
                            ck.require(all(restored[k] == prod[k] for k in
                                ('image', 'engine', 'files', 'mounted', 'env', 'argv', 'engram_identity')),
                                'restored production identity differs; HOLD stays open')
                    tag = f'{self.tag}-restore-close'
                    deadline = self.restore_by - 90
                    self.remember(tag, deadline)
                    ck.require(self.clock() < deadline, 'PC1 final close budget exhausted; HOLD stays open')
                    try:
                        rc, text = self.plain(step, ['bash', '-c', owner.command('launch', tag,
                            ['bash', str(pf.RESTORE), step], deadline=deadline)], 120)
                    finally:
                        self.fence_phase(tag)
                else:
                    rc, text = self.recovery(step, ['bash', str(pf.RESTORE), step], 900 if step == 'start' else 600)
                strict_restored(step, rc, text)  # requires six distinct IDENTICAL, not near ties
                result[step] = 'PASS'
            for prior in list(self.owner_tags):
                self.fence_phase(prior)
            ck.require(self.restore_by - self.clock() >= 90, 'PC1 final close needs 90s reserve; HOLD stays open')
            tag = f'{self.tag}-restore-close'
            try:
                try:
                    rc, text = self.plain('commit-close', [*pf.SSH, 'sparkhead', owner.command('commit-close', tag,
                        deadline=min(self.restore_by-5, self.clock()+25))], 30)
                    ck.require(rc == 0 and owner.CLOSED in text.splitlines(), 'PC1 final owned HOLD close unconfirmed')
                finally:
                    rc, text = self.plain('cancel-close', [*pf.SSH, 'sparkhead', owner.command('cancel-close', tag)], 30)
                    ck.require(rc == 0 and 'SCOPED CLOSE COMMIT FENCED' in text.splitlines(), 'PC1 final close fence unconfirmed')
                result['restore_deadline_met'] = self.clock() < self.restore_by
                ck.require(result['restore_deadline_met'] is True, 'PC1 restore completed late')
            except (Exception, KeyboardInterrupt):
                rc, text = self.plain('retain-close', [*pf.SSH, 'sparkhead', owner.command('retain-close', tag)], 30)
                ck.require(rc == 0 and owner.RETAINED in text.splitlines(), 'PC1 original HOLD retention unconfirmed')
                raise
            result['status'] = 'RESTORED: K1 START VERIFIED, greedy 6/6 IDENTICAL, PRODUCTION HOLD OK, window closed'
            return result
        finally:
            self.plain = original
            self.call = original_call
            (self.out / 'arm-memory.json').write_text(json.dumps(self.sup.arm_memory) + '\n')

    def chain_close(self) -> dict[str, Json]:
        """WIN2 chained, non-last, stop-free end: stop the experiment, prove both Sparks quiet, close the owned HOLD
        through the same scoped deferred-close transaction, leave production DOWN and write the chain marker.

        W8 is not used: its close step runs window.sh close, which requires production serving on :8000. The HOLD file
        alone is removed (the shared rm shim defers it to commit-close); the dsv41-hold placeholders stay so nothing
        else takes the Sparks before the next chained window, which reuses them, or the chain's final W8 close.
        """
        self.sup.close()
        original = self.plain
        def bounded(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
            if label in ('cancel-close', 'retain-close'):
                return original(label, argv, min(30, timeout))
            left = self.restore_by - self.clock()
            ck.require(left > 0, 'PC1 HOLD+85 exhausted; manual restore required; HOLD stays open')
            result = original(label, argv, min(left, timeout))
            ck.require(self.clock() < self.restore_by, 'PC1 chained close exceeded original deadline')
            return result
        self.plain = bounded
        try:
            for tag in list(self.owner_tags):
                self.fence_phase(tag)
            ck.require(self.hold_written and math.isfinite(self.restore_by), 'chained close needs the owned HOLD')
            rc, text = self.recovery('experiment-stop', [*pf.SSH, 'sparkhead', f'cd ~/{self.remote} && ./stop.sh </dev/null'], 120)
            ck.require(rc == 0, 'PC1 experiment stop failed; HOLD stays open')
            proof: dict[str, Json] = {}
            for host in pf.HOSTS:
                rc, text = self.plain('jit-wait-' + host, [*pf.SSH, host,
                    spark_quiet.command('PC1 EXPERIMENT ABSENT', deadline=min(self.restore_by - 90, self.clock() + 350))], 360)
                ck.require(rc == 0 and 'PC1 EXPERIMENT ABSENT' in text.splitlines(),
                           f'{host}: experiment/container/compute absence unproved; HOLD stays open')
                proof[host] = 'PC1 EXPERIMENT ABSENT'
            tag = f'{self.tag}-restore-close'
            deadline = self.restore_by - 90
            self.remember(tag, deadline)
            ck.require(self.clock() < deadline, 'PC1 final close budget exhausted; HOLD stays open')
            try:
                # As W8's close: a local -restore-close scope whose nested ssh runs the shared CHAINED_CLOSE on
                # sparkhead, where the rm shim only RECORDS the HOLD unlink (placeholders proved up, kept up).
                rc, text = self.plain('chain-close', ['bash', '-c', owner.command('launch', tag,
                    [*pf.SSH, 'sparkhead', chain.CHAINED_CLOSE], deadline=deadline)], 120)
            finally:
                self.fence_phase(tag)
            ck.require(rc == 0 and chain.CHAINED_CLOSE_OK in text.splitlines(),
                       'PC1 chained HOLD close (deferred unlink) failed; HOLD stays open')
            for prior in list(self.owner_tags):
                self.fence_phase(prior)
            ck.require(self.restore_by - self.clock() >= 90, 'PC1 final close needs 90s reserve; HOLD stays open')
            try:
                try:
                    rc, text = self.plain('commit-close', [*pf.SSH, 'sparkhead', owner.command('commit-close', tag,
                        deadline=min(self.restore_by-5, self.clock()+25))], 30)
                    ck.require(rc == 0 and owner.CLOSED in text.splitlines(), 'PC1 chained owned HOLD close unconfirmed')
                finally:
                    rc, text = self.plain('cancel-close', [*pf.SSH, 'sparkhead', owner.command('cancel-close', tag)], 30)
                    ck.require(rc == 0 and 'SCOPED CLOSE COMMIT FENCED' in text.splitlines(), 'PC1 final close fence unconfirmed')
            except (Exception, KeyboardInterrupt):
                rc, text = self.plain('retain-close', [*pf.SSH, 'sparkhead', owner.command('retain-close', tag)], 30)
                ck.require(rc == 0 and owner.RETAINED in text.splitlines(), 'PC1 original HOLD retention unconfirmed')
                raise
            chain.finish_chained(self.out, proof)
            return {'status': 'CHAINED: production left down', 'experiment_stop': 'PASS', 'quiet_proof': proof,
                    'close': 'PASS', 'restore_deadline_met': self.clock() < self.restore_by}
        finally:
            self.plain = original
            (self.out / 'arm-memory.json').write_text(json.dumps(self.sup.arm_memory) + '\n')


    def hold_owned(self) -> bool:
        return bool(self.hold_written)

    def chain_abort(self) -> dict[str, Json]:
        return chain.abort(self.out / 'chain-abort')


def strict_restored(step: str, rc: int, text: str) -> None:
    pf.restored(step, rc, text)
    if step == 'check':
        lines = [line for line in text.splitlines() if line.startswith('greedy K1-post vs restored:')]
        matches = [re.fullmatch(r'greedy K1-post vs restored: ([0-5]) IDENTICAL [0-9]+', line) for line in lines]
        ck.require(len(matches) == 6 and all(match is not None for match in matches)
            and sorted(match.group(1) for match in matches if match is not None) == list('012345'),
            'restore requires six distinct complete IDENTICAL lines0..5')


def host_check(text: str) -> None:
    ck.require(text.splitlines()[:2] == [KERNEL, DRIVER], 'Spark kernel/driver differs')
    memory = re.findall(r'^MemAvailable:\s+(\d+) kB$', text, re.M)
    ck.require(len(memory) == 1 and int(memory[0]) >= pf.START_FLOOR_KB, 'MemAvailable below 5GiB before start')


class Backend(Protocol):
    def preflight(self) -> None: ...
    def open_hold(self) -> float: ...
    def snapshot(self) -> None: ...
    def fits(self, arm: Arm) -> bool: ...
    def start(self, arm: Arm) -> None: ...
    def capacity_report(self, arm: Arm) -> dict[str, Json]: ...
    def greedy(self, arm: Arm) -> dict[str, Json]: ...
    def warmup(self, arm: Arm) -> None: ...
    def qualify(self, arm: Arm) -> dict[str, Json]: ...
    def build_cuts(self) -> None: ...
    def cell(self, arm: Arm, name: str) -> dict[str, Json]: ...
    def finish(self, arm: Arm) -> dict[str, Json]: ...
    def restore(self, *, production_stopped: bool) -> dict[str, Json]: ...
    def chain_close(self) -> dict[str, Json]: ...
    def hold_owned(self) -> bool: ...
    def chain_abort(self) -> dict[str, Json]: ...


def window(backend: Backend, group: str, checkpoint: Callable[[dict[str, Json]], None]) -> dict[str, Json]:
    result: dict[str, Json] = {'status': 'INCOMPLETE', 'starts': {}, 'decision': {'status': 'NOT A CANDIDATE'}}
    hold = stopped = False
    try:
        host_gates.admit(group)
        headroom.admit(group)
        backend.preflight()
        hold = True
        result['opened'] = backend.open_hold()
        backend.snapshot()
        for i, arm in enumerate(arms(group)):
            ck.require(backend.fits(arm), 'mandatory remaining OFF/arm/OFF blocks cannot fit; restore now')
            stopped = True
            row: dict[str, Json] = {'status': 'STARTED', 'role': arm.role, 'cells': {}}
            record(result['starts'])[arm.name] = row
            checkpoint({'arm': arm.name, 'status': 'STARTED'})
            row['capacity'] = backend.capacity_report(arm)
            backend.start(arm)
            row['greedy'] = backend.greedy(arm)
            backend.warmup(arm)
            row['gates'] = backend.qualify(arm)
            checkpoint({'arm': arm.name, 'gates': 'PASS'})
            if i == 0:
                backend.build_cuts()
            for name in cells.plan_for(group).cells:
                record(row['cells'])[name] = backend.cell(arm, name)
                checkpoint({'arm': arm.name, 'cell': name})
            row['checks'] = backend.finish(arm)
            row['status'] = 'COMPLETE'
            checkpoint({'arm': arm.name, 'status': 'COMPLETE'})
        result['decision'] = decision(group, record(result['starts']))
        result['status'] = 'COMPLETE'
    except (Exception, KeyboardInterrupt) as exc:
        result['stop'] = f'{type(exc).__name__}: {exc}'
    finally:
        saved = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        try:
            if hold:
                try:
                    if chain.skip_restore(stopped='stop' in result):
                        # WIN2 chain: a stop-free, non-last chained window leaves production down for the next one.
                        try:
                            result['restore'] = backend.chain_close()
                            result['chain'] = 'CHAINED'
                        except (Exception, KeyboardInterrupt) as exc:
                            # Any failure brings production back with the strict W8 restore.
                            result['chain_close_error'] = f'{type(exc).__name__}: {exc}'
                            result['restore'] = backend.restore(production_stopped=True)
                    else:
                        result['restore'] = backend.restore(production_stopped=stopped)
                except (Exception, KeyboardInterrupt) as exc:
                    result['restore'] = {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'}
            else:
                result['restore'] = {'status': 'NOT NEEDED: preflight only'}
            # S92 FIX 1 / chain.end_action: no owned HOLD but the chain marker present means production is down from
            # the previous chained window; never "not needed" then, but the shared chain abort (strict W8 + watch).
            if chain.end_action(stopped=True, hold_written=hold and backend.hold_owned()) == 'abort':
                try:
                    result['restore'] = backend.chain_abort()
                    result['chain'] = 'ABORT'
                except (Exception, KeyboardInterrupt) as exc:
                    result['restore'] = {'status': 'FAILED', 'error': f'chain abort: {type(exc).__name__}: {exc}'}
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
    if 'stop' in result or 'chain_close_error' in result or record(result['restore']).get('status') == 'FAILED':
        result['decision'] = {'status': 'NOT A CANDIDATE', 'reason': 'STOP or production restore unverified'}
    return result


def budget_minutes(group: str = 'chunk2048') -> float:
    return (SETUP_S + sum(arm_seconds(group, a.role) for a in arms(group)) + RECOVERY_S) / 60


def plan(group: str = 'chunk2048') -> str:
    b.admit(group)
    return (f'DRY PC1: no network, command or file. Coordinator Deepseek/coordinator only.\n'
        f'One isolated HOLD: OFF/{group}/OFF; all mandatory. Settings {b.SETTINGS[group]}.\n'
        + (f'Round3 headroom: {json.dumps(headroom.estimate(group))}. Round4 NDM floor 3 GiB (shared monitor floor lowered for larger-chunk arms only).\n'
           if group in b.LARGE_ARMS else '') +
        f'WIN2 plan: decision cells {cells.plan_for(group).decision} x3 each, reported {cells.plan_for(group).reported} x1; 512 outputs.\n'
        'Matched public cuts; T1/top_p1/seeds42+repeat/high/priority10; zero cache hits, unique salts.\n'
        f'{metrics.WIN2_RULE_NOTE}; verdict groups {dict(cells.plan_for(group).groups)}.\n'
        f'Each start:6 shared NBW checks under {cells.GREEDY_POLICY}, threshold{cells.GREEDY_NEAR_TIE_NATS}; OPS1 waves owned.\n'
        f'MEASURED historical startup388--404s; ESTIMATED HOLD {budget_minutes(group):.2f}min <=85.\n'
        'HOLD+73 submit/abort,+85 restoration, immutable. Shared scopes/admission/quiet/strict W8/serial logs;15min post-close watch.\n'
        'Verdict CANDIDATE, INCONCLUSIVE or NOT A CANDIDATE only; no hardware clearance or promotion from this dry plan.')



def plain_owned(out: Path) -> Callable[[str, list[str], float], tuple[int, str]]:
    # Restore fences every owned phase again after close; a serial keeps each
    # attempt's exclusive log distinct (TV1-COMPACT-W-20261027T031304Z kept HOLD
    # on a FileExistsError for the second hold-phase fence log).
    serial = itertools.count(1)
    def run(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        log = out / 'cmd' / f'restore-{next(serial):04d}-{label}.log'
        log.parent.mkdir(exist_ok=True)
        with log.open('x') as handle:
            process = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, text=True)
            try:
                rc = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                terminate(process)
                rc = 124
        return rc, log.read_text()
    return run


def post_close(result: dict[str, Json], out: Path, tag: str) -> None:
    """Watch even a stopped experiment after verified restoration; never an open HOLD."""
    if not str(record(result['restore']).get('status', '')).startswith('RESTORED'):
        return
    def log(message: str) -> None:
        print(message, flush=True)
        with (out / 'post-close-watch.log').open('a') as handle:
            handle.write(message + '\n')
    try:
        watched = watchdog.watch(production_health.ProductionHealth(),
            watchdog.ScopedRecovery(tag + '-watch', out / 'post-close-commands'), log=log)
        result['post_close_watch'] = cast(Json, dict(watched.__dict__))
    except (Exception, KeyboardInterrupt) as exc:
        print(f'PC1 post-close watch unverified: {exc}', file=sys.stderr)
        result['post_close_watch'] = {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'}
    if record(result['post_close_watch']).get('status') != 'PASS':
        result['decision'] = {'status': 'NOT A CANDIDATE', 'reason': 'post-close production health failed'}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    parser.add_argument('--group', choices=b.ARMS[1:], default='chunk2048')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--remote-repo', default='dsv41-deploy-offload')
    parser.add_argument('--expected-experiment-commit', default='')
    parser.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    args = parser.parse_args(argv)
    if not args.execute:
        print(plan(args.group))
        return 0
    try:
        host_gates.admit(args.group)
        headroom.admit(args.group)
        ck.require(budget_minutes(args.group) <= 85,
            f'PC1 matrix needs ESTIMATED {budget_minutes(args.group):.2f}min >85; redesign required before any host access')
        ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'coordinator required')
        ck.require(args.out is not None and args.out.resolve().parent == pf.CAMPAIGN and not args.out.exists(), 'fresh campaign --out required')
        ck.require(re.fullmatch('[0-9a-f]{40}', args.expected_experiment_commit) is not None, 'full reviewed commit required')
        ck.require(re.fullmatch('[A-Za-z0-9_-]+', args.remote_repo) is not None and args.remote_repo != 'dsv41-deploy', 'separate experiment checkout required')
        ck.require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=b.ROOT, text=True).strip() == args.expected_experiment_commit,
            'local HEAD differs from reviewed commit')
        ck.require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=b.ROOT, text=True).strip(), 'tracked tree dirty')
        tag = cast(Path, args.out).name
        ck.require(re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_-]{0,49}', tag) is not None, 'run name must be <=50 safe characters')
        owner.root_for(tag + '-restore-experiment-stop-999999')
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'STOP before remote action: {exc}', file=sys.stderr)
        return 1
    out = cast(Path, args.out)
    out.mkdir()
    def interrupted(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f'signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    def checkpoint(row: dict[str, Json]) -> None:
        with (out / 'checkpoints.jsonl').open('a') as handle:
            handle.write(json.dumps({'time': time.time(), **row}, allow_nan=False) + '\n')
    if args.group in b.LARGE_ARMS:
        # Round 4 (NDM 3 GiB floor): this process's shared monitor floor follows PC1's; other windows keep 4 GiB.
        pf.MEM_FLOOR_KB = headroom.FLOOR_KB
    sup = LargeChunkSupervisor(out) if args.group in b.LARGE_ARMS else pf.Supervisor(out)
    live = Live(out, sup, group=args.group,
        remote=args.remote_repo, expected=args.expected_experiment_commit, data_dir=args.data_dir, tag=tag, plain=plain_owned(out))
    result = window(live, args.group, checkpoint)
    sup.close()
    # Every attributed drafter fault, including a stopped start's systemic one (finish() never ran for it).
    result['drafter_faults'] = cast(Json, {k: cast(Json, v) for k, v in live.drafter_faults.items()})
    result['drafter_fault_attribution'] = metrics.fault_attribution(
        {a.name: (a.role, len(live.drafter_faults.get(a.name, []))) for a in arms(args.group)})
    clock_check(result, args.group, out)
    try:
        archive_controls(result, args.group, tag, out)
    except (OSError, ValueError) as exc:
        # Loud, never silent: the verdict stands, but the cross-window archive missed this window.
        print(f'PC1 control archive append FAILED: {exc}', file=sys.stderr)
        result['archive_error'] = f'{type(exc).__name__}: {exc}'
    if result.get('chain') == 'ABORT':
        # chain.abort ran its own strict W8 restore and 15-minute watch; prev-ok reads them from this summary.
        result['post_close_watch'] = record(result['restore']).get('post_close_watch', {'status': 'NOT RUN'})
    else:
        post_close(result, out, tag)
    finish_chain(result)
    result.update({'chain_mode': live.chain_mode, 'chain_enabled': chain.enabled(), 'chain_last': chain.last()})
    result.setdefault('chain', 'NONE')
    result.update({'production': b.PRODUCTION, 'experiment': args.expected_experiment_commit, 'group': args.group,
        'shared_owner_sha256': pf.digest(b.ROOT / 'offload/launch/windows/scoped_remote.py'),
        'budget_minutes_estimated': budget_minutes(args.group), 'budget_sha256': pf.digest(b.HERE / 'budget.json'),
        'headroom': headroom.estimate(args.group) if args.group in b.LARGE_ARMS else {},
        'restore_sha256': pf.RESTORE_SHA256, 'qualification_policy': 'PC1 registered configuration pairs',
        'artifact_sha256': {str(p.relative_to(out)): pf.digest(p) for p in out.rglob('*.json*') if p.is_file()}})
    (out / 'summary.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'stop': result.get('stop'), 'decision': result['decision'], 'restore': result['restore']}))
    return 2 if record(result['restore']).get('status') == 'FAILED' else (
        0 if result['status'] == 'COMPLETE' and (result['chain'] == 'CHAINED'
            or record(result.get('post_close_watch', {})).get('status') == 'PASS') else 1)


if __name__ == '__main__':
    raise SystemExit(main())
