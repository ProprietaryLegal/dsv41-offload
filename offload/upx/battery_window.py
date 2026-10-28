"""UPX-0085B attended window (MEASURE-0085B.md M1): C1 / X / C2 memory batteries for henryous 0085b. Dry by default;
coordinator only. Design data: windows.WINDOWS['UPX-0085B']; arithmetic: measure_0085b; requests: battery_cells.

Lifecycle is NBW's, by subclassing nbw_window.Live (shared HOLD admission with a private placeholder, scoped ownership
and fences on every host, quiet proof, strict six-IDENTICAL W8 restore, post-close watch), without NBW's native pair:
production's x86host drafter is never stopped, so pair_close is a no-op and W8's own check still requires PRODUCTION HOLD
OK. Arms use the production-exact base (prod.env): C = production bytes on port 8001, X = the same plus the two 0085b
files and VLLM_DSV41_RELEASE_FILE. Before every X battery the runner writes the threshold file on BOTH ranks through an
owned scoped command and reads it back.

Chain mode (WIN2.md, chain.py, WIN2-R2). Preflight takes chain.preflight's answer; while the chain marker is present it
also probes both hold placeholders (chain.PLACEHOLDER_PROBE, the view chain.gate has). 'chained' = production down by
design: the production identity comes from chain.production_snapshot() (copied into rollback.json with chained_from and
its sha256; C1 is the measurement baseline), the HOLD is admitted with chain.allow_unreachable('chained'), and both Sparks
are proven quiet before the first start. The end follows chain.end_action, after every owned phase is fenced and the
HOLD ownership reread: 'chained' stops the experiment, proves the Sparks quiet, closes this window's own HOLD with
chain.CHAINED_CLOSE in its -restore-close scope (then the fenced commit-close) and writes the marker; 'restore' is the
strict W8 restore and the post-close watch, then chain.clear(); 'abort' (no own HOLD, marker present) is chain.abort;
'none' touches nothing. A chained end first checks both placeholder views (chain_ready); under today's scoped_remote tmux
shim neither sees the hold placeholders (test_chain_placeholders), so it runs the strict W8 restore instead of leaving
production down with no placeholders or with a close that cannot succeed.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from offload.launch.windows import chain
from offload.launch.windows import decision_v2 as dv
from offload.launch.windows import hold_admission
from offload.launch.windows import nbw_cells as cells
from offload.launch.windows import nbw_window as n
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import post_close_watch as watchdog
from offload.launch.windows import scoped_remote as owner
from offload.launch.windows import spark_quiet
from offload.launch.windows import up2_checks as ck
from offload.launch.windows import win2_history as wh
from offload.launch.windows.up2_low import X86HOST_PROBE
from offload.prefill_maxlen.evidence import Json, number, record
from offload.upx import battery_cells as bc
from offload.upx import build_0085b as b
from offload.upx import configs
from offload.upx import measure_0085b as m
from offload.upx import windows as w

GROUP = w.GROUPS['UPX-0085B']
DESIGN = w.WINDOWS['UPX-0085B']
CONFIG = configs.CONFIG
ARCHIVE = pf.CAMPAIGN / 'control-archive.jsonl'
# chain.CHAINED_CLOSE without its unlink: the same placeholder test, run in its own scope before the real close (see
# Live.chain_ready). The close itself is chain.CHAINED_CLOSE verbatim; if the shared text changes shape, import refuses.
CHAINED_UNLINK = 'rm -f "$HOME/dsv41-deploy/HOLD"; echo ' + chain.CHAINED_CLOSE_OK
if not chain.CHAINED_CLOSE.endswith(CHAINED_UNLINK) or chain.CHAINED_CLOSE.count('rm -f') != 1:
    raise ImportError('chain.CHAINED_CLOSE changed shape; re-derive PLACEHOLDER_CHECK')
PLACEHOLDERS_OK = 'UPX PLACEHOLDERS PRESENT'
PLACEHOLDER_CHECK = chain.CHAINED_CLOSE.removesuffix(CHAINED_UNLINK) + 'echo ' + PLACEHOLDERS_OK
OVERLAY_FILES = tuple(str((b.OVERLAY / rel).relative_to(b.ROOT)) for rel in b.FILES)
PIN_FILES = (CONFIG, 'offload/upx/prod.env', 'offload/upx/x0085b.env', *OVERLAY_FILES, 'offload/upx/MANIFEST-0085b.json',
             'offload/launch/start_exp.sh', 'offload/launch/serve_exp.sh', 'offload/launch/dsv41_tp2_offload.sh',
             'offload/kvpack/MANIFEST.json', 'offload/upx/battery_window.py', 'offload/upx/battery_cells.py',
             'offload/upx/measure_0085b.py', 'offload/launch/windows/nbw_window.py',
             'offload/launch/windows/scoped_remote.py', 'offload/launch/windows/post_close_watch.py',
             'offload/launch/windows/spark_quiet.py', 'offload/launch/windows/production_health.py',
             'offload/launch/windows/hold_admission.py', 'offload/launch/windows/chain.py')
# Never validated or used: no start of this window has NBW's native role (probe_worker and start are overridden).
NO_NATIVE = n.Native('dsv41-offload/none/repo', '0' * 40, '1' * 64, '2' * 64)
RELEASE_MARKER = 'UPX RELEASE LOG READ'
# Streams the container log (constant memory on the Spark); prints only 0085b lines, then a completion marker.
RELEASE_FILTER = r'''
import subprocess, sys
proc = subprocess.Popen(['docker', 'logs', '--timestamps', '--since', sys.argv[1], 'dsv41-serve'],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
for line in proc.stdout:
    if 'dsv41 patch 0085b' in line:
        print(line.rstrip(), flush=True)
if proc.wait():
    raise SystemExit('docker logs failed')
print('UPX RELEASE LOG READ', flush=True)
'''


@dataclass(frozen=True)
class Arm(n.Arm):
    role: str = 'C'             # C = production bytes (control), X = 0085b mounted, file-switched; never NBW's 'I'

    @property
    def config(self) -> str:
        return CONFIG

    @property
    def env(self) -> dict[str, str]:
        return configs.ARMS['release-file-prod' if self.role == 'X' else 'C-prod']


ARMS = tuple(Arm(s.label, 327680, 0, role='X' if s.arm == GROUP.candidate else 'C') for s in DESIGN.starts)
BUDGET_S = {s.label: w.START_S + sum(step.seconds() for step in s.steps) for s in DESIGN.starts}
IMAGE_BATTERIES = next(st.repeats for st in DESIGN.starts[0].steps if st.kind == 'battery-image')
PAIRS = next(st.repeats for st in DESIGN.starts[1].steps if st.kind == 'battery-pair')


def schedule(arm: Arm) -> list[tuple[str, int]]:
    """(mode, threshold) per battery: the control starts' image batteries, the candidate's strict ON/OFF alternation."""
    return m.schedule(PAIRS) if arm.role == 'X' else [('image', 0)] * IMAGE_BATTERIES


def iso(t: float) -> str:
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def upx_runtime(base: dict[str, Json], live: dict[str, Json], arm: Arm, *, from_port: str) -> None:
    """Experiment rank == its baseline rank (production, or C1 in a chained window) except port, and for X exactly the
    two 0085b mounts and VLLM_DSV41_RELEASE_FILE."""
    for key in ('image', 'engine', 'files', 'engram_identity'):
        ck.require(live.get(key) == base.get(key), f'{key} differs from the baseline')
    ck.require(bool(record(live['engram_identity'])), 'Engram identity missing')
    ck.require(live.get('state') == 'running', 'experiment container not running')
    expected = {k: v for k, v in record(base['mounted']).items() if k != '/serve.sh'}
    if arm.role == 'X':
        manifest = json.loads(b.MANIFEST.read_text())
        for row in manifest['files']:
            ck.require(row['container_path'] not in expected, 'baseline already mounts a 0085b file')
            expected[row['container_path']] = row['overlay_sha256']
    got = {k: v for k, v in record(live['mounted']).items() if k != '/serve.sh'}
    ck.require(got == expected, 'mounted engine bytes differ from the baseline (plus the declared 0085b files)')
    env = dict(record(base['env']))
    ck.require(not any(k.startswith('VLLM_DSV41_RELEASE_') for k in env), 'baseline already sets a 0085b switch')
    if arm.role == 'X':
        env['VLLM_DSV41_RELEASE_FILE'] = configs.RELEASE_FILE_CONTAINER
    after = record(live['env'])
    for key in env.keys() | after.keys():
        ck.require(env.get(key) == after.get(key), f'env drift from the baseline: {key}')
    a = shlex.split(str(base['argv']).replace('\\\n', ' '))
    c = shlex.split(str(live['argv']).replace('\\\n', ' '))
    ck.require(a[a.index('--port') + 1] == from_port, 'baseline argv port differs')
    a[a.index('--port') + 1] = '8001'
    ck.require(a == c, 'launch argv differs from the baseline beyond the port')


def in_container(text: str, arm: Arm) -> None:
    """`docker exec dsv41-serve sha256sum <core.py> <gpu_worker.py>`: overlay bytes in X, image bytes in C."""
    manifest = json.loads(b.MANIFEST.read_text())
    want = {row['container_path']: row['overlay_sha256' if arm.role == 'X' else 'base_sha256'] for row in manifest['files']}
    got = {path: sha for sha, path in (line.split() for line in text.splitlines() if line.strip())}
    ck.require(got == want, f'{arm.name}: in-container 0085b files are not the {"overlay" if arm.role == "X" else "image"} bytes')


def decision(starts: dict[str, Json]) -> dict[str, Json]:
    """win2-v1 over the candidate's paired batteries, the gates, and the verdict (windows.arm_verdict)."""
    rows = {name: record(starts[name]) for name in (a.name for a in ARMS)}
    ck.require(all(r.get('status') == 'COMPLETE' for r in rows.values()), 'every start must be complete to decide')
    candidate = next(a.name for a in ARMS if a.role == 'X')
    results = [record(x) for x in cast(list[Json], rows[candidate]['batteries'])]
    image = [record(x) for a in ARMS if a.role == 'C' for x in cast(list[Json], rows[a.name]['batteries'])]
    decided = dv.decide_window(m.series(results), GROUP.decision_cells, groups=GROUP.groups)
    following = ARMS[[a.name for a in ARMS].index(candidate) + 1].name
    gates = {'greedy': all(record(r['greedy']).get('ok') is True for r in rows.values()),
             'release-census': record(record(rows[candidate]['checks'])['release_census']).get('ok') is True,
             'kl': record(rows[following]['kl']).get('pass') is True}
    return {'decision': decided, 'gates': cast(Json, gates), 'summary': m.summary(results, image),
            'verdict': w.arm_verdict(GROUP.name, decided, gates)}


def archive_rows(window: str, starts: dict[str, Json], cap_of: Callable[[str], tuple[int, str]]) -> list[dict[str, Json]]:
    """Every completed control start's image batteries, one row per metric and rank (WIN2 control archive, WIN2-R2 schema:
    production commit, the start's own clock cap and its source from `cap_of` (win2_history.start_clock_cap), YYYYMMDD
    date, start, repeat)."""
    out: list[dict[str, Json]] = []
    for arm in ARMS:
        row = record(starts.get(arm.name) or {})
        if arm.role != 'C' or row.get('status') != 'COMPLETE':
            continue
        cap, source = cap_of(arm.name)
        for k, value in enumerate(cast(list[Json], row['batteries'])):
            battery = record(value)
            common: dict[str, Json] = {'window': window, 'cell': 'battery-150k', 'max_tokens': bc.MAX_TOKENS,
                                       'commit': n.PROD_COMMIT, 'clock_cap_mhz': cap, 'clock_source': source,
                                       'date': str(row['date']), 'start': arm.name, 'repeat': k}
            for host in m.HOSTS:
                for metric in ('idle_kb', 'next_low_kb'):
                    out.append({**common, 'metric': f'{metric}:{host}',
                                'value': record(record(battery['hosts'])[host])[metric]})
            out.append({**common, 'metric': 'next_ttft_s', 'value': battery['next_ttft_s']})
    return out


class Live(n.Live):
    def __init__(self, out: Path, sup: pf.Supervisor, *, remote: str, expected: str, data_dir: Path, tag: str,
                 archive: Path = ARCHIVE, call: pf.Call | None = None,
                 plain: Callable[[str, list[str], float], tuple[int, str]] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        super().__init__(out, sup, native=NO_NATIVE, remote=remote, expected=expected, data_dir=data_dir, tag=tag,
                         call=call, plain=plain, clock=clock)
        self.mode = ''
        self.archive = archive
        self.baseline: list[dict[str, Json]] = []
        self.chain_snapshot: tuple[Path, dict[str, Json]] | None = None
        self.writes = 0
        self.started_at: dict[str, float] = {}

    @property
    def pin_files(self) -> tuple[str, ...]:
        return PIN_FILES

    def hold_purpose(self) -> str:
        return f'UPX-0085B {self.tag}: 0085b idle cache release batteries; production-exact base, drafter unchanged'

    def production_serving(self) -> bool:
        try:
            models = ck.rows(self.json('prod-models', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8000/v1/models')['data'])
        except ValueError:
            return False
        ck.require(len(models) == 1 and models[0].get('max_model_len') == 327680
                   and models[0].get('root') == '/srv/dsv41/legal-v2a', 'production model/cap differs')
        return True

    def preflight(self) -> None:
        self.sup.begin()
        ck.require(pf.digest(pf.RESTORE) == pf.RESTORE_SHA256, 'W8 restore changed')
        cells.greedy_inputs()
        from offload.prefill_trace.window_requests import corpus
        corpus(self.data_dir)
        ck.require((pf.CAMPAIGN / 'D2-GEN.PAUSE').exists() or subprocess.run(
            ['tmux', 'has-session', '-t', '=d2-gen'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0,
            'D2 generation active without pause')
        self.ssh('ledger', 'sparkhead', 'cd ~/dsv41-deploy && test ! -e HOLD && '
                 f'test "$(git rev-parse HEAD)" = {n.PROD_COMMIT} && '
                 f'test "$(git rev-parse HEAD)" = "$(git rev-list -n1 {n.PROD_TAG})" && '
                 'git diff --exit-code HEAD -- dsv41.env serve.sh start.sh '
                 'offload/launch/warmup.py offload/launch/warmup_serve.sh')
        checkout = (f'cd ~/{shlex.quote(self.remote)} && test "$(git rev-parse HEAD)" = {self.expected} && '
                    'test -z "$(git status --porcelain --untracked-files=no)"')
        self.ssh('checkout', 'sparkhead', checkout)
        remote = self.ssh('pins', 'sparkhead', f'cd ~/{shlex.quote(self.remote)} && sha256sum ' + shlex.join(PIN_FILES))
        ck.require(remote.strip() == '\n'.join(f'{pf.digest(pf.ROOT / p)}  {p}' for p in PIN_FILES), 'staged source pins differ')
        # The ledger proved no HOLD; chain.preflight decides between serving production and a chain left down. With the
        # marker present it also needs both hold placeholders (probed only then, as chain.gate does).
        seen = self.placeholders_seen('preflight') if chain.marker_present() else None
        self.mode = chain.preflight(production_serving=self.production_serving(), hold_exists=False,
                                    placeholders_present=None if seen is None else all(seen.values()))
        if self.mode == 'production':
            self.ssh('prod-health', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8000/health')
        else:
            path, rollback = chain.production_snapshot()       # the chain's capture, sha256 checked against the marker
            commit = rollback.get('production_commit') or rollback.get('commit')
            ck.require(commit == n.PROD_COMMIT, f'chain production snapshot {path} is not production {n.PROD_COMMIT[:7]}')
            self.chain_snapshot = (path, rollback)
        pf.worker(self.json('worker', 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)), self.clock())
        for host in pf.HOSTS:
            n.host_check(self.ssh(f'preflight-host-{host}', host, pf.HOST_PARITY))
        (self.out / 'chain.json').write_text(json.dumps({'mode': self.mode, 'chain': chain.enabled(),
                                                         'last': chain.last()}) + '\n')

    def open_hold(self) -> float:
        """nbw_window.Live.open_hold with this window's purpose; allow_unreachable only in chained mode (WIN2-R2)."""
        ck.require(not self.open_admitted, 'UPX HOLD cannot be reopened or renewed')
        ck.require(self.mode in ('production', 'chained'), 'preflight decides the chain mode first')
        self.open_admitted = True
        self.open_deadline = self.clock() + pf.DRAIN_S + 60
        self.hold_scope_deadline = self.open_deadline + n.RESTORE_S
        marker = f'UPX-0085B {self.tag}:'
        try:
            self.owned('hold-open', 'sparkhead', self.tag + '-hold',
                       hold_admission.open_script(self.tag + '-hold', self.admission_token, self.hold_purpose(),
                                                  pf.DRAIN_S, allow_unreachable=chain.allow_unreachable(self.mode)),
                       timeout=pf.DRAIN_S + 60)
        finally:
            text = self.ssh('hold-read', 'sparkhead', hold_admission.command('check', self.tag + '-hold', self.admission_token))
            self.hold_written = marker in text
            if self.hold_written:
                opened = pf.hold_opened(text, self.tag, self.clock())
                self.submit, self.sup.abort_at, self.restore_by = opened + n.SUBMIT_S, opened + n.ABORT_S, opened + n.RESTORE_S
                self.sup.bound = '75-minute UPX abort bound'
                (self.out / 'deadlines.json').write_text(json.dumps({'opened': opened, 'submit': self.submit,
                    'abort': self.sup.abort_at, 'restore_by': self.restore_by}) + '\n')
        ck.require(self.hold_written, 'UPX exact owned HOLD marker required')
        opened = pf.hold_opened(text, self.tag, self.clock())
        ck.require(opened <= self.open_deadline and self.restore_by <= self.hold_scope_deadline,
                   'UPX HOLD published after its opening ceiling; placeholder lease would not cover the window')
        self.sup.tunnel([*pf.SSH, '-N', '-o', 'ExitOnForwardFailure=yes', '-L', '127.0.0.1:18001:127.0.0.1:8001', 'sparkhead'])
        return opened

    def snapshot(self) -> None:
        if self.mode == 'production':
            super().snapshot()
            self.baseline = list(self.production)
            return
        # Chained: nothing serves; prove both Sparks quiet before the first start_exp.sh (WIN2-R2), C1 is the baseline.
        ck.require(self.chain_snapshot is not None, 'chained preflight did not take the chain production snapshot')
        path, rollback = cast(tuple[Path, dict[str, Json]], self.chain_snapshot)
        proof: dict[str, Json] = {}
        for host in pf.HOSTS:
            text = self.ssh(f'chain-quiet-{host}', host, spark_quiet.command('UPX SPARKS QUIET', deadline=self.clock() + 120),
                            timeout=150)
            ck.require('UPX SPARKS QUIET' in text.splitlines(), f'{host}: Spark not quiet at the start of a chained window')
            proof[host] = 'UPX SPARKS QUIET'
        (self.out / 'rollback.json').write_text(json.dumps({**rollback, 'chained_from': str(path),
            'chained_from_sha256': pf.digest(path), 'mode': 'chained', 'baseline': 'first control start', 'quiet': proof},
            indent=1) + '\n')

    def fits(self, arm: n.Arm) -> bool:
        return self.clock() + BUDGET_S[arm.name] <= self.submit

    def probe_worker(self, label: str, arm: n.Arm) -> dict[str, Json]:
        return pf.worker(self.json(label, 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)), self.clock())

    def pair_close(self, *, guarded: bool) -> None:
        """No native pair is ever opened here; W8's own check still requires the production worker's PRODUCTION HOLD OK."""
        self.pair_open = False

    def write_threshold(self, arm: Arm, value: int, label: str) -> None:
        """Both ranks, owned scoped writes (fenced at restore), then read back; the same value on both or STOP."""
        for host in pf.HOSTS:
            self.writes += 1
            self.owned(f'{arm.name}-release-{label}-{host}', host, f'{self.tag}-{arm.name}-release-{self.writes}',
                       configs.release_file_command(value), timeout=60)
            got = self.ssh(f'{arm.name}-release-read-{label}-{host}', host, f'cat {configs.RELEASE_FILE_NODE}').strip()
            ck.require(got == str(value), f'{host}: threshold file reads {got!r}, wrote {value}')

    def start(self, base_arm: pf.Arm) -> None:
        ck.require(isinstance(base_arm, Arm), 'UPX arm required')
        arm = cast(Arm, base_arm)
        ck.require(self.fits(arm), f'{arm.name}: full block cannot fit before the submission deadline')
        self.sup.unready()
        for host in pf.HOSTS:
            n.host_check(self.ssh(f'{arm.name}-host-{host}', host, pf.HOST_PARITY))
        self.sup.arm_minima = {}
        self.sup.arm_memory[arm.name] = self.sup.arm_minima
        if arm.role == 'X':
            self.write_threshold(arm, 0, 'init')          # never inherit a previous window's threshold
        env = ' '.join(f'{k}={v}' for k, v in arm.env.items())
        start = f'cd ~/{self.remote} && {env} timeout {pf.START_S} bash offload/launch/start_exp.sh {CONFIG}'
        start = hold_admission.command('stop', self.tag + '-hold', self.admission_token, ['bash', '-c', start])
        self.started_at[arm.name] = self.clock()
        # its cmd log (NNNN-<arm>-start.log) carries start_exp.sh's "<node> clock cap N MHz" lines: the archive's clock
        self.owned(f'{arm.name}-start', 'sparkhead', f'{self.tag}-{arm.name}-engine', start, timeout=pf.START_S + 60)
        self.ssh(f'{arm.name}-health', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8001/health')
        models = ck.rows(self.json(f'{arm.name}-models', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8001/v1/models')['data'])
        ck.require(len(models) == 1 and models[0].get('max_model_len') == 327680
                   and models[0].get('root') == '/srv/dsv41/legal-v2a', 'experiment model/cap differs')
        current = [self.capture(host, arm.name) for host in pf.HOSTS]
        first = not self.baseline
        if first:
            ck.require(self.mode == 'chained' and arm.role == 'C', 'only a chained window takes its first control as baseline')
            self.baseline = current
            (self.out / 'baseline.json').write_text(json.dumps({'start': arm.name, 'ranks': cast(Json, current)}, indent=1) + '\n')
        for rank, host in enumerate(pf.HOSTS):
            if not first:
                upx_runtime(self.baseline[rank], current[rank], arm,
                            from_port='8000' if self.mode == 'production' else '8001')
            ck.require(str(current[rank]['id']) not in self.seen, 'experiment reused a container')
            self.verify(host, arm.name, 0)
            in_container(self.ssh(f'{arm.name}-0085b-files-{host}', host, 'docker exec dsv41-serve sha256sum '
                                  + ' '.join(b.CONTAINER + rel for rel in b.FILES)), arm)
            logs = pf.logs(self.json(f'{arm.name}-logs-{host}', host, 'python3 -c ' + shlex.quote(pf.LOG_FILTER)
                                     + ' ' + shlex.quote(ck.STOP_LOG.pattern)), rank, 327680)
            if rank == 0:
                self.proxy_pid = int(number(cast(list[Json], logs['proxy_pids'])[0]))
        self.ids = [str(row['id']) for row in current]
        self.seen.update(self.ids)
        self.probe_worker(f'{arm.name}-worker', arm)
        (self.out / f'{arm.name}-identity.json').write_text(json.dumps({'ranks': cast(Json, current)}, indent=1) + '\n')
        self.sup.mark_ready()

    def driver(self, mode: str, output: Path, *extra: str) -> list[str]:
        return [sys.executable, '-m', 'offload.upx.battery_cells', '--execute', '--mode', mode,
                '--data-dir', str(self.data_dir), '--output', str(output), '--deadline-utc', str(self.submit), *extra]

    def build(self) -> None:
        self.call('public-cuts', self.driver('build', self.out / 'cuts.json'), timeout=300)

    def greedy(self, arm: n.Arm) -> dict[str, Json]:
        output = self.out / f'{arm.name}-greedy.json'
        self.call(f'{arm.name}-greedy', self.driver('greedy', output, '--salt', f'{self.tag}-{arm.name}'),
                  timeout=300, inflight=1)
        result = ck.read(output)
        checks = [record(r) for r in ck.rows(result['checks'])]
        ck.require([r.get('i') for r in checks] == list(range(6))
                   and all(r.get('identical') is True or r.get('near_tie') is True for r in checks)
                   and number(result.get('identical', -1)) + number(result.get('near_tie', -1)) == 6,
                   'greedy divergence that is not a 0.9-nat near-tie, or missing checks')
        result['ok'] = True
        return result

    def kl(self, arm: Arm, candidate: Arm) -> dict[str, Json]:
        output = self.out / f'{arm.name}-kl.json'
        self.call(f'{arm.name}-kl', self.driver('kl', output, '--candidate', str(self.out / f'{candidate.name}-greedy.json'),
                                                '--salt', f'{self.tag}-{arm.name}'), timeout=300, inflight=1)
        result = ck.read(output)
        ck.require(result.get('pass') is True, f'{candidate.name}: KL at a greedy divergence beyond {bc.KL_BOUND}')
        return result

    def telemetry(self, host: str) -> list[dict[str, Json]]:
        """Every row the Supervisor recorded from this Spark's 1 s telemetry."""
        path = self.out / f'telemetry-{host}.jsonl'
        return [record(cast(Json, json.loads(x))) for x in path.read_text().splitlines() if x.startswith('{')]

    def releases(self, label: str, host: str, since: float) -> str:
        text = self.ssh(label, host, 'python3 -c ' + shlex.quote(RELEASE_FILTER) + ' ' + shlex.quote(iso(since)))
        ck.require(text.splitlines()[-1:] == [RELEASE_MARKER], f'{host}: release log read incomplete')
        return text

    def battery(self, arm: Arm, mode: str, threshold: int, k: int) -> dict[str, Json]:
        if arm.role == 'X':
            self.write_threshold(arm, threshold, f'b{k}')
        before = self.stats(f'{arm.name}-battery{k}-before')
        self.round_baseline.setdefault(arm.name, {key: number(record(before['counters']).get(key, 0))
                                                  for key in n.FAILED_ROUNDS})
        output = self.out / f'{arm.name}-battery{k}.json'
        self.call(f'{arm.name}-battery{k}', self.driver('battery', output, '--cuts', str(self.out / 'cuts.json'),
                  '--salt', f'{self.tag}-{arm.name}-b{k}'), timeout=bc.BATTERY_S * 3, inflight=1)
        timing = ck.read(output)
        after = self.stats(f'{arm.name}-battery{k}-after')
        ck.counters(after, 0, before)
        for key in n.FAILED_ROUNDS:
            ck.require(number(record(after['counters']).get(key, 0)) == number(record(before['counters']).get(key, 0)),
                       f'{arm.name}: {key} grew during battery {k}')
        since = number(timing['a_sent']) - 60
        logs = {host: m.parse_log(self.releases(f'{arm.name}-battery{k}-releases-{host}', host, since)) for host in pf.HOSTS}
        series = {host: m.memory_series(self.telemetry(host)) for host in pf.HOSTS}
        result = m.battery_result(bc.times(timing, mode), series, logs)
        result.update(threshold=threshold, times=timing)
        (self.out / f'{arm.name}-battery{k}-result.json').write_text(json.dumps(result, allow_nan=False) + '\n')
        return result

    def finish(self, arm: n.Arm) -> dict[str, Json]:
        ck.require(isinstance(arm, Arm), 'UPX arm required')
        checks = super().finish(arm)
        if arm.role == 'X':
            # every release of the whole start: exactly two per ON battery (A's idle, after B), none failed
            on = sum(1 for mode, _ in schedule(cast(Arm, arm)) if mode == 'on')
            census: dict[str, Json] = {'expected_per_rank': 2 * on}
            for host in pf.HOSTS:
                releases, failures = m.parse_log(self.releases(f'{arm.name}-census-{host}', host,
                                                               self.started_at[arm.name] - 60))
                census[host] = {'releases': len(releases), 'failures': len(failures)}
                ck.require(len(releases) == 2 * on and not failures,
                           f'{host}: {len(releases)} releases and {len(failures)} failures in {arm.name}; expected {2 * on}, 0')
            census['ok'] = True
            checks['release_census'] = census
            self.write_threshold(cast(Arm, arm), 0, 'final')
        return checks

    def placeholders_seen(self, label: str) -> dict[str, bool]:
        """chain.PLACEHOLDER_PROBE over plain ssh on both Sparks: the view chain.gate and a chained preflight have."""
        seen: dict[str, bool] = {}
        for host in chain.PLACEHOLDER_HOSTS:
            rc, _ = self.plain(f'{label}-placeholders-{host}', [*pf.SSH, host, chain.PLACEHOLDER_PROBE], 60)
            seen[host] = rc == 0
        return seen

    def settle(self) -> bool:
        """Before chain.end_action: fence every owned phase (a late HOLD opener must not write after the decision), then
        reread this window's HOLD ownership (nbw_window.Live.restore's first step). True when it owns a HOLD."""
        for tag in list(self.owner_tags):
            self.fence_phase(tag)
        if not self.hold_written:
            rc, text = self.plain('hold-reread', [*pf.SSH, 'sparkhead',
                                                  hold_admission.command('check', self.tag + '-hold', self.admission_token)], 60)
            ck.require(rc == 0, 'UPX HOLD ownership unreadable; HOLD stays open')
            self.hold_written = f'UPX-0085B {self.tag}:' in text
            if self.hold_written:
                self.restore_by = pf.hold_opened(text, self.tag, self.clock()) + n.RESTORE_S
        return self.hold_written

    def chain_ready(self) -> dict[str, Json]:
        """The two placeholder views a clean chained end depends on: the next chained preflight's (plain-ssh probe) and
        chain.CHAINED_CLOSE's own, run as PLACEHOLDER_CHECK in an owned scope the way the close runs. Under scoped_remote's
        tmux shim the placeholders hold_admission.open_script makes are private to the hold phase, so today both fail
        (test_chain_placeholders) and the window restores production instead of leaving it down."""
        seen = self.placeholders_seen('chain-end')
        result: dict[str, Json] = {'probe': cast(Json, seen)}
        if not all(seen.values()):
            missing = ', '.join(host for host, ok in seen.items() if not ok)
            return {**result, 'ok': False, 'reason': f'hold placeholders not visible to the chain gate on {missing}'}
        rc, text = self.recovery('placeholder-check', [*pf.SSH, 'sparkhead', PLACEHOLDER_CHECK], 120)
        if rc == 0 and PLACEHOLDERS_OK in text.splitlines():
            return {**result, 'ok': True, 'scoped': PLACEHOLDERS_OK}
        return {**result, 'ok': False, 'reason': 'chain.CHAINED_CLOSE cannot see the hold placeholders from its scope'}

    def chained_close(self) -> dict[str, Json]:
        """chain.CHAINED_CLOSE in this window's own -restore-close scope on sparkhead: it proves both placeholders and its
        unlink is recorded by the scope's rm shim; commit-close then removes exactly that HOLD after every phase is fenced
        and proved absent (NBW's restore tail). The placeholders stay up."""
        tag = f'{self.tag}-restore-close'
        deadline = self.restore_by - 90
        self.remember(tag, deadline)
        ck.require(self.clock() < deadline, 'UPX chained close budget exhausted; HOLD stays open')
        try:
            rc, text = self.plain('chained-close', [*pf.SSH, 'sparkhead', owner.command('launch', tag,
                ['bash', '-c', chain.CHAINED_CLOSE], deadline=deadline)], 120)
        finally:
            self.fence_phase(tag)
        ck.require(rc == 0 and chain.CHAINED_CLOSE_OK in text.splitlines(), 'UPX chained HOLD close unrecorded; HOLD stays open')
        for prior in list(self.owner_tags):
            self.fence_phase(prior)
        ck.require(self.restore_by - self.clock() >= 90, 'UPX final close needs 90s reserve; HOLD stays open')
        try:
            try:
                rc, text = self.plain('commit-close', [*pf.SSH, 'sparkhead', owner.command('commit-close', tag,
                    deadline=min(self.restore_by - 5, self.clock() + 25))], 30)
                ck.require(rc == 0 and owner.CLOSED in text.splitlines(), 'UPX owned HOLD close unconfirmed')
            finally:
                rc, text = self.plain('cancel-close', [*pf.SSH, 'sparkhead', owner.command('cancel-close', tag)], 30)
                ck.require(rc == 0 and 'SCOPED CLOSE COMMIT FENCED' in text.splitlines(), 'UPX close fence unconfirmed')
        except (Exception, KeyboardInterrupt):
            rc, text = self.plain('retain-close', [*pf.SSH, 'sparkhead', owner.command('retain-close', tag)], 30)
            ck.require(rc == 0 and owner.RETAINED in text.splitlines(), 'UPX original HOLD retention unconfirmed')
            raise
        return {'hold': 'CLOSED (own HOLD; placeholders kept; production left down)'}

    def leave_chained(self) -> dict[str, Json]:
        """end_action 'chained': with both placeholder views present, stop the experiment, prove both Sparks quiet, close
        this window's HOLD (chained_close) and write the chain marker; production stays down. Otherwise the strict W8
        restore, so the chain ends with production serving instead of down and unguarded."""
        self.sup.close()
        for tag in list(self.owner_tags):
            self.fence_phase(tag)
        ready = self.chain_ready()
        if ready.get('ok') is not True:
            restored = self.restore(production_stopped=True)
            restored['chain_end'] = {**ready, 'action': 'strict W8 restore instead of a chained end'}
            return restored
        rc, _ = self.recovery('experiment-stop', [*pf.SSH, 'sparkhead', f'cd ~/{self.remote} && ./stop.sh </dev/null'], 120)
        ck.require(rc == 0, 'UPX experiment stop failed; HOLD stays open')
        proof: dict[str, Json] = {}
        for host in pf.HOSTS:
            rc, text = self.plain('chain-quiet-' + host, [*pf.SSH, host, spark_quiet.command(
                'UPX CHAINED QUIET', deadline=min(self.restore_by - 90, self.clock() + 350))], 360)
            ck.require(rc == 0 and 'UPX CHAINED QUIET' in text.splitlines(),
                       f'{host}: experiment/container/compute absence unproved; HOLD stays open')
            proof[host] = 'UPX CHAINED QUIET'
        result: dict[str, Json] = {**self.chained_close(), 'quiet': proof, 'chain_end': ready}
        chain.finish_chained(self.out, proof)
        result['status'] = 'CHAINED: production left down'
        return result

    def restore(self, *, production_stopped: bool) -> dict[str, Json]:
        result = super().restore(production_stopped=production_stopped)
        if 'pair_close' in result:
            result['pair_close'] = 'not opened (no native pair in UPX); W8 check proves PRODUCTION HOLD OK'
        return result

    def abort(self) -> dict[str, Json]:
        """end_action 'abort' (no own HOLD, chain marker present): chain.abort, the strict W8 under the reviewed scoped
        recovery's own HOLD plus the post-close watch; it alerts and touches nothing when any HOLD exists."""
        return chain.abort(self.out / 'chain-abort')

    def write_archive(self, starts: dict[str, Json]) -> dict[str, Json]:
        """Control rows with each start's own clock cap from its start log; the archive head before and after the append
        (the chain alone shows an edited line, not a truncated tail)."""
        rows = archive_rows(self.out.name, starts, lambda start: wh.start_clock_cap(self.out, start))
        before = dv.archive_head(self.archive)
        if rows:
            dv.archive_append(self.archive, rows)
        return {'rows': len(rows), 'head_before': before, 'head_after': dv.archive_head(self.archive)}


class Backend(Protocol):
    def preflight(self) -> None: ...
    def open_hold(self) -> float: ...
    def snapshot(self) -> None: ...
    def fits(self, arm: n.Arm) -> bool: ...
    def start(self, base_arm: pf.Arm) -> None: ...
    def build(self) -> None: ...
    def greedy(self, arm: n.Arm) -> dict[str, Json]: ...
    def kl(self, arm: Arm, candidate: Arm) -> dict[str, Json]: ...
    def battery(self, arm: Arm, mode: str, threshold: int, k: int) -> dict[str, Json]: ...
    def finish(self, arm: n.Arm) -> dict[str, Json]: ...
    def settle(self) -> bool: ...
    def leave_chained(self) -> dict[str, Json]: ...
    def restore(self, *, production_stopped: bool) -> dict[str, Json]: ...
    def abort(self) -> dict[str, Json]: ...
    def write_archive(self, starts: dict[str, Json]) -> dict[str, Json]: ...
    @property
    def mode(self) -> str: ...


def window(b: Backend, checkpoint: Callable[[dict[str, Json]], None]) -> dict[str, Json]:
    result: dict[str, Json] = {'status': 'INCOMPLETE', 'starts': {}, 'verdict': 'NOT A CANDIDATE: not measured completely'}
    starts = record(result['starts'])
    hold = stopped = False
    try:
        b.preflight()
        result['chain_mode'] = b.mode
        stopped = b.mode == 'chained'       # production is already down in a chain
        hold = True
        result['hold_opened'] = b.open_hold()
        b.snapshot()
        for i, arm in enumerate(ARMS):
            ck.require(b.fits(arm), f'{arm.name}: mandatory block cannot fit')
            stopped = True
            row: dict[str, Json] = {'status': 'STARTED', 'role': arm.role,
                                    'date': datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')}
            starts[arm.name] = row
            checkpoint({'start': arm.name, 'status': 'STARTED'})
            b.start(arm)
            if i == 0:
                b.build()
            row['greedy'] = b.greedy(arm)
            if i > 0 and ARMS[i - 1].role == 'X':
                row['kl'] = b.kl(arm, ARMS[i - 1])
            batteries: list[Json] = []
            row['batteries'] = batteries
            for k, (mode, threshold) in enumerate(schedule(arm)):
                batteries.append(b.battery(arm, mode, threshold, k))
                checkpoint({'start': arm.name, 'battery': k, 'mode': mode})
            row['checks'] = b.finish(arm)
            row['status'] = 'COMPLETE'
            checkpoint({'start': arm.name, 'status': 'COMPLETE'})
        result.update(decision(starts))
        result['status'] = 'COMPLETE'
    except (Exception, KeyboardInterrupt) as exc:
        result['stop'] = f'{type(exc).__name__}: {exc}'
    finally:
        saved = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        try:
            try:
                # end_action needs the settled ownership: an admission that was attempted is fenced and reread first
                action = chain.end_action(stopped='stop' in result, hold_written=b.settle() if hold else False)
                result['end_action'] = action
                if action == 'chained':
                    result['restore'] = b.leave_chained()
                    if str(record(result['restore']).get('status', '')).startswith('CHAINED'):
                        result['chain'] = 'CHAINED'
                elif action == 'restore':
                    result['restore'] = b.restore(production_stopped=stopped)
                elif action == 'abort':
                    aborted = b.abort()
                    result['restore'] = aborted
                    if 'post_close_watch' in aborted:
                        result['post_close_watch'] = aborted['post_close_watch']   # chain.abort ran the watch
                else:
                    result['restore'] = {'status': 'NOT NEEDED: no owned HOLD and no chain marker; production untouched'
                                                   + ('; every owned phase fenced' if hold else '')}
            except (Exception, KeyboardInterrupt) as exc:
                result['restore'] = {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'}
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
        try:
            archived = b.write_archive(starts)
            result['archived_rows'] = archived['rows']
            result['archive_head'] = {'before': archived['head_before'], 'after': archived['head_after']}
        except (Exception, KeyboardInterrupt) as exc:
            result['archive_error'] = f'{type(exc).__name__}: {exc}'
    if str(record(result['restore']).get('status', '')).startswith(('FAILED', 'ALERT')):
        result['verdict'] = 'NOT A CANDIDATE: production restore unverified'
    elif 'stop' in result:
        result['verdict'] = 'STOP: ' + str(result['stop'])
    return result


def plan() -> str:
    lines = ['DRY RUN: no command, network or file. Coordinator Deepseek/coordinator only.',
             f'Production {n.PROD_COMMIT[:7]} / {n.PROD_TAG}; production-exact base offload/upx/prod.env (UPX_BASE=prod), port 8001.',
             'Preflight: production ledger + no HOLD; reviewed experiment checkout and pins; chain.preflight decides production|chained',
             '  (marker present: both hold placeholders probed; chained: chain.production_snapshot() must be production).',
             'Owned HOLD (allow_unreachable only when chained); production snapshot, or (chained) both Sparks quiet and C1 as baseline.']
    for arm in ARMS:
        steps = ', '.join(f'{mode}{"" if mode == "image" else "@" + str(t)}' for mode, t in schedule(arm))
        lines.append(f'{arm.name} ({arm.role}, {" ".join(f"{k}={v}" for k, v in arm.env.items())}): identity vs baseline; '
                     f'six greedy at 0.9 nats{"; KL re-probe of X" if arm.name == ARMS[-1].name else ""}; '
                     f'batteries [{steps}]; ESTIMATED {BUDGET_S[arm.name] / 60:.1f} min')
    lines += ['X: threshold file written and read back on BOTH ranks before every battery (owned scoped writes); 0 at start and end.',
              'Battery: cold 150K B1 (16 tokens), 20 s idle, cold 150K, 15 s idle; 1 s telemetry; docker logs --timestamps census.',
              'Decision: win2-v1 paired ON/OFF, cells mem-r0/mem-r1 (next 150K low point; guards idle MemAvailable, next TTFT).',
              'Gates: greedy (all starts), release census (X), KL (C2); 3 GiB floor STOP during every B.',
              f'ESTIMATED HOLD {w.hold_minutes(DESIGN)} min chained, {w.hold_minutes(DESIGN, last=True)} min as the last window.',
              'End (chain.end_action, after every owned phase is fenced and the HOLD ownership reread):',
              '  chained: both placeholder views present -> stop, quiet proof, chain.CHAINED_CLOSE + commit-close, marker',
              '    (production down); either view missing (today, scoped placeholders) -> the strict W8 restore below;',
              '  restore: strict W8 (start if stopped, six IDENTICAL check, close), 15 min post-close watch, chain.clear();',
              '  abort (no own HOLD, marker present): chain.abort (strict W8 under its own HOLD + watch; alert if a HOLD exists).',
              'Archive: control rows with per-start clock_cap_mhz (start log), YYYYMMDD date, production commit; head before/after.']
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--remote-repo', default='dsv41-deploy-offload')
    ap.add_argument('--expected-experiment-commit', default='')
    ap.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    args = ap.parse_args(argv)
    if not args.execute:
        print(plan())
        return 0
    try:
        ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'WINDOW_COORDINATOR=Deepseek/coordinator required')
        ck.require(args.out is not None and args.out.resolve().parent == pf.CAMPAIGN and not args.out.exists(), 'fresh --out inside campaign required')
        ck.require(re.fullmatch('[0-9a-f]{40}', args.expected_experiment_commit) is not None, 'full reviewed experiment commit required')
        ck.require(re.fullmatch('[A-Za-z0-9_.-]+', args.remote_repo) is not None and args.remote_repo != 'dsv41-deploy', 'separate experiment checkout required')
        ck.require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pf.ROOT, text=True).strip() == args.expected_experiment_commit,
                   'local HEAD differs from reviewed commit')
        ck.require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=pf.ROOT, text=True).strip(), 'tracked source dirty')
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'STOP before any remote action: {exc}', file=sys.stderr)
        return 1
    out = cast(Path, args.out)
    out.mkdir()
    tag = re.sub('[^A-Za-z0-9_-]', '-', out.name)

    def interrupted(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f'signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)

    def checkpoint(row: dict[str, Json]) -> None:
        with (out / 'checkpoints.jsonl').open('a') as handle:
            handle.write(json.dumps({'time': time.time(), **row}, allow_nan=False) + '\n')
    sup = pf.Supervisor(out)
    live = Live(out, sup, remote=args.remote_repo, expected=args.expected_experiment_commit, data_dir=args.data_dir, tag=tag)
    result = window(live, checkpoint)
    sup.close()
    status = str(record(result['restore']).get('status', ''))
    if status.startswith('RESTORED') and result.get('end_action') != 'abort':     # chain.abort watched (and cleared) itself
        def watch_log(message: str) -> None:
            print(message, flush=True)
            with (out / 'post-close-watch.log').open('a') as handle:
                handle.write(message + '\n')
        watched = watchdog.watch(watchdog.ProductionHealth(),
                                 watchdog.ScopedRecovery(tag + '-watch', out / 'post-close-commands'), log=watch_log)
        result['post_close_watch'] = cast(Json, dict(watched.__dict__))
        if watched.status == 'PASS':
            chain.clear()                    # production verified: no chain marker may outlive it
        else:
            result['verdict'] = 'NOT A CANDIDATE: post-close production health unverified'
    result.update({'experiment_commit': args.expected_experiment_commit, 'production_commit': n.PROD_COMMIT,
                   'restore_sha256': pf.RESTORE_SHA256, 'config_sha256': pf.digest(pf.ROOT / CONFIG),
                   'artifact_sha256': {str(p.relative_to(out)): pf.digest(p) for p in sorted(out.rglob('*.json*')) if p.is_file()}})
    (out / 'summary.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'stop': result.get('stop'), 'verdict': result['verdict'],
                      'chain': result.get('chain'), 'end_action': result.get('end_action'), 'restore': result['restore']}))
    code, message = exit_status(result)
    if message:
        print(message, file=sys.stderr)
    return code


def exit_status(result: dict[str, Json]) -> tuple[int, str]:
    """0 done; 1 stopped with production restored (own W8 or chain abort) or untouched; 2 production NOT verified;
    3 production down under another HOLD (chain abort alerted and touched nothing)."""
    status = str(record(result['restore']).get('status', ''))
    if 'post_close_watch' in result and record(result['post_close_watch']).get('status') != 'PASS':
        return 2, 'POST-CLOSE PRODUCTION HEALTH FAILED; inspect watchdog log and HOLD before further work.'
    if status.startswith('ALERT'):
        return 3, 'CHAIN STOPPED: PRODUCTION DOWN and a HOLD exists; chain abort touched nothing. Coordinator action required.'
    if status.startswith('RESTORED') and 'post_close_watch' not in result:
        return 2, 'PRODUCTION RESTORED BUT NOT WATCHED; inspect before further work.'
    if status.startswith(('RESTORED', 'NOT NEEDED', 'CHAINED')):
        return (0 if 'stop' not in result else 1), ''
    return 2, 'PRODUCTION NOT VERIFIED; a HOLD may remain open. Inspect cmd/restore-*.log or chain-abort/summary.json.'


if __name__ == '__main__':
    raise SystemExit(main())
