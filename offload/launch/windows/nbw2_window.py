"""NBW v2 (WIN2): native kx2 + batch-invariant re-measure under the fairer rule, with deeper sampling. Dry by default.

NDM 2026-10-27 ~19:10Z: "We are still building this. We do not need to keep restoring production -- go window after window
after window. Do everything fable recommended. Do better deeper testing and measuring. Fairer win rule is essential."
Fable (roadmap e8e3e3a, NBW row): NBW-W3 measured native kx2 +3.3 % short / -1.1 % 32K inside a ~7 % K spread with ONE
1,500-token run per cell; "K I K I K, short + 32K B1 x3, B3 x2, pooled ms/round + acceptance; greedy 0.9".

Design (coordinator 2026-10-27 after the WIN2 history re-score: NBW-W3's 3 K vs 2 I values per cell cannot carry a
verdict; "Run K I K I K with >= 3 candidate repeats in the two B1 decision cells"). MEASURED NBW-W-20261026T231445Z
timings (start -> first finished cell 415/430 s K and 711/704 s I; 32k-b1 49-51 s, 32k-b3 107 s per cell) fit K I K I K
inside the 72-minute submission bound only with the extra repeats on the I starts:
- K starts: short-b1, 32k-b1 and 32k-b3 once each (3 controls per cell); I starts: short-b1 and 32k-b1 twice and 32k-b3
  once (4 candidates per decision cell, 2 for the reported B3). The closing K is skipped if its block no longer fits
  (then this window alone is INCONCLUSIVE; pooling with another NBW2 window decides).
- Every repeat is a fresh cache salt (a cold, comparable request); raw outputs in `repeats/`, and `<start>-<cell>.json`
  carries the per-repeat `repetitions` rows win2_history reads. Answers are 1,500 tokens: archive rows carry
  max_tokens 1500, and spreads are read at 1500 tokens, the production commit and the window's one clock cap.
- Rule win2-v1 revision 2 (decision_v2). Decision cells short-b1 and 32k-b1; primary metric pooled server ms per round,
  guards accepted/round and client tok/s (a guard loses only with its interval below zero). 32k-b3, committed tok/s
  and TTFT are reported only. Every start records its proven clock cap; a decision needs ONE cap across every pooled
  start (THERMAL-WATCH can dial back to 2150 mid-chain), else INCONCLUSIVE.
- Arms are UNPAIRED (Welch): the two repeats of one I start share that start's offset, and NBW-W3 showed a ~7 % K
  start-to-start tok/s spread; one control per K start keeps that offset in the control spread and the interval.
  The per-start-mean view of the I arm is reported beside the verdict (S94: n = 4 repeats from 2 starts).
- I starts: six greedy checks at 0.9 nats (NDM; shared nbw_cells --near-tie-nats) recording the divergence top-20; the
  following K start re-probes them (WIN2 KL, reported only). Every K repeat is appended to the shared control archive.
- Chain mode follows WIN2-R2 (chain.preflight with the placeholder probe, allow_unreachable, end_action/abort, clear only
  after the watch PASSES, chain.production_snapshot, chain.CHAINED_CLOSE). A clean chained end first proves in a
  throwaway scope that CHAINED_CLOSE's placeholder check would pass; when it cannot, the window ends with the strict
  W8 restore instead of burning its one -restore-close scope (R-NBW finding: the shared check cannot see the scoped,
  private-socket placeholders every scoped runner opens, so at offload/win2 2b48eec a chained NBW2 window always
  restores).
Everything else -- HOLD admission, fences, scopes, native pair, guards, strict W8 restore -- is the NBW runner.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from offload.launch.windows import chain
from offload.launch.windows import decision_v2 as dv
from offload.launch.windows import nbw_cells as cells
from offload.launch.windows import nbw_window as n
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import post_close_watch as watchdog
from offload.launch.windows import scoped_remote as owner
from offload.launch.windows import spark_quiet
from offload.launch.windows import up2_checks as ck
from offload.launch.windows import win2_history
from offload.prefill_maxlen.evidence import Json, number, record

REGISTRATION = 'nbw2-win2-v1'
ARMS = n.ARMS  # K I K I K
DECISION_CELLS = ('short-b1', '32k-b1')
REPEATS = {'K': {'short-b1': 1, '32k-b1': 1, '32k-b3': 1}, 'I': {'short-b1': 2, '32k-b1': 2, '32k-b3': 1}}
CELL_NAMES = tuple(REPEATS['K'])
MAX_TOKENS = 1500
PRIMARY = 'ms_per_round'
GUARDS = ('accepted_per_round', 'decode_tps')
REPORTED = ('committed_tps', 'ttft_s')
ARCHIVED = ('ms_per_round', 'accepted_per_round', 'decode_tps', 'ttft_s')  # win2_history's metric names
GREEDY_REGISTRATION = 'nbw2-near-tie-0p9'
GREEDY_NEAR_TIE_NATS = 0.9 + 1e-3  # NDM 2026-10-27 "You can loosen the nats to .9"; NBW's 1e-3 rounding allowance
CAMPAIGN = pf.CAMPAIGN
ARCHIVE = CAMPAIGN / 'control-archive.jsonl'
PIN_FILES = (*n.PIN_FILES, 'offload/launch/windows/nbw2_window.py', 'offload/launch/windows/nbw2_cells.py',
             'offload/launch/windows/decision_v2.py', 'offload/launch/windows/chain.py',
             'offload/launch/windows/greedy_kl.py')
# MEASURED (NBW-W-20261026T231445Z checkpoints.jsonl): start -> first finished cell 415/430 s (K) and 711/704 s (I) minus
# the ~40 s first cell; 32k-b1 49-51 s, 32k-b3 107 s. ESTIMATED: short-b1 ~40 s. Each block carries a 15 % margin.
MEASURED_STARTUP_S = {'K': 375.0, 'I': 670.0}
MEASURED_CELL_S = {'short-b1': 40.0, '32k-b1': 51.0, '32k-b3': 108.0}
MARGIN = 1.15
PLACEHOLDERS_PROVEN = 'NBW2 PLACEHOLDERS PROVEN'
_CLOSE_TAIL = 'rm -f "$HOME/dsv41-deploy/HOLD"; echo ' + chain.CHAINED_CLOSE_OK


def placeholder_proof() -> str:
    """chain.CHAINED_CLOSE without its HOLD unlink: the exact placeholder check, run in a throwaway scope first."""
    ck.require(chain.CHAINED_CLOSE.endswith(_CLOSE_TAIL), 'shared CHAINED_CLOSE changed; review the NBW2 proof')
    return chain.CHAINED_CLOSE.removesuffix(_CLOSE_TAIL) + 'echo ' + PLACEHOLDERS_PROVEN


def block_s(role: str) -> float:
    return MARGIN * (MEASURED_STARTUP_S[role] + sum(MEASURED_CELL_S[name] * reps for name, reps in REPEATS[role].items()))


def window_estimate_s(arms: Sequence[n.Arm] = ARMS) -> float:
    return sum(block_s(arm.role) for arm in arms)


def _ms_per_round(r: Mapping[str, Json]) -> float:
    return number(r['server_decode_ms']) / number(r['spec_request_rounds'])


METRICS: dict[str, tuple[Callable[[Mapping[str, Json]], float], bool]] = {
    'ms_per_round': (_ms_per_round, False),
    'accepted_per_round': (lambda r: number(r['acceptance_per_round']), True),
    'decode_tps': (lambda r: number(r['median_decode_tps']), True),
    'committed_tps': (lambda r: 1000 * (1 + number(r['acceptance_per_round'])) / _ms_per_round(r), True),
    'ttft_s': (lambda r: number(r['median_ttft_s']), False),
}


def repetition(result: Mapping[str, Json], rep: int, raw: Path) -> dict[str, Json]:
    """One repeat in win2_history's `repetitions` format (server_decode_ms + spec_request_rounds -> ms/round; each
    stream's prompt hash and sampling block, never its text)."""
    streams = [{'stream': s.get('stream'), 'prompt_sha256': s['prompt_sha256'], 'sampling': s['sampling']}
               for s in ck.rows(result['streams'])]
    ck.require(all(record(s['sampling']).get('max_tokens') == MAX_TOKENS for s in streams), 'NBW cells answer 1,500 tokens')
    ms, rounds = number(result['server_decode_ms_per_spec_request_round']), number(result['spec_request_rounds'])
    return {'repeat': rep, 'file': f'repeats/{raw.name}', 'sha256': pf.digest(raw), 'server_decode_ms': ms * rounds,
            'spec_request_rounds': rounds, 'server_decode_ms_per_spec_request_round': ms,
            'acceptance_per_round': result['acceptance_per_round'], 'median_decode_tps': result['median_decode_tps'],
            'median_ttft_s': result['median_ttft_s'], 'max_tokens': MAX_TOKENS, 'streams': cast(Json, streams)}


def complete_starts(window: Mapping[str, Json]) -> list[n.Arm]:
    """Every start must be COMPLETE, except a closing K skipped for time."""
    starts = record(window['starts'])
    done: list[n.Arm] = []
    for i, arm in enumerate(ARMS):
        row = record(starts.get(arm.name) or {})
        if i == len(ARMS) - 1 and row.get('status') == 'SKIPPED_TIME':
            continue
        ck.require(row.get('role') == arm.role and row.get('status') == 'COMPLETE', f'{arm.name}: complete start required')
        done.append(arm)
    return done


def start_reps(window: Mapping[str, Json], name: str) -> dict[str, list[list[dict[str, Json]]]]:
    """{'K'|'I': [per-start repeat rows]} for one cell, with the window's repeat counts, prompts and answer length."""
    starts = record(window['starts'])
    reps: dict[str, list[list[dict[str, Json]]]] = {'K': [], 'I': []}
    for arm in complete_starts(window):
        rows = ck.rows(record(record(record(starts[arm.name])['cells'])[name])['repetitions'])
        ck.require(len(rows) == REPEATS[arm.role][name], f'{arm.name}/{name}: {REPEATS[arm.role][name]} repeats required')
        reps[arm.role].append(rows)
    every = [rep for role in reps.values() for x in role for rep in x]
    hashes = [[record(s)['prompt_sha256'] for s in ck.rows(rep['streams'])] for rep in every]
    ck.require(all(h == hashes[0] for h in hashes), f'{name}: repeats/starts used different public prompts')
    ck.require(all(rep.get('max_tokens') == MAX_TOKENS for rep in every), f'{name}: not a 1,500-token NBW cell')
    return reps


def arm_values(window: Mapping[str, Json], name: str, metric: str) -> tuple[list[float], list[float]]:
    """Raw per-repeat values: control = every K repeat, candidate = every I repeat."""
    reps, get = start_reps(window, name), METRICS[metric][0]
    return [get(rep) for x in reps['K'] for rep in x], [get(rep) for x in reps['I'] for rep in x]


def clock_caps(windows: Sequence[Mapping[str, Json]]) -> list[Json]:
    """The proven clock cap of every complete start of every window (None when unproven)."""
    return [record(record(w['starts'])[arm.name]).get('clock_cap_mhz') for w in windows for arm in complete_starts(w)]


def start_view(windows: Sequence[Mapping[str, Json]], name: str) -> dict[str, Json]:
    """S94 flag: the I arm's 4 repeats come from 2 starts. Report Welch on per-start means and the within/between-start
    variance of I beside the verdict (never deciding); a one-window CANDIDATE stays provisional until pooled."""
    k_means: list[float] = []
    i_means: list[float] = []
    within: list[float] = []
    for w in windows:
        reps = start_reps(w, name)
        k_means += [statistics.fmean([_ms_per_round(r) for r in x]) for x in reps['K']]
        for x in reps['I']:
            values = [_ms_per_round(r) for r in x]
            i_means.append(statistics.fmean(values))
            if len(values) > 1:
                within.append(statistics.variance(values))
    view: dict[str, Json] = {'metric': PRIMARY, 'k_start_means': cast(Json, k_means), 'i_start_means': cast(Json, i_means),
                             'i_within_start_variance': statistics.fmean(within) if within else None,
                             'i_between_start_variance': statistics.variance(i_means) if len(i_means) > 1 else None}
    if len(k_means) > 1 and len(i_means) > 1:
        mk, mi = statistics.fmean(k_means), statistics.fmean(i_means)
        vk, vi = statistics.variance(k_means) / len(k_means), statistics.variance(i_means) / len(i_means)
        se = math.sqrt(vk + vi)
        df = (vk + vi) ** 2 / (vk ** 2 / (len(k_means) - 1) + vi ** 2 / (len(i_means) - 1)) if se > 0 else 1e9
        point, half = (mk - mi) / mk, dv.t975(df) * se / mk
        view.update(point_gain=point, interval95=[point - half, point + half], df=df)
    return view


def decide(windows: Sequence[Mapping[str, Json]]) -> dict[str, Json]:
    """win2-v1 over every supplied NBW2 window (pooled, unpaired Welch); INCONCLUSIVE below three repeats per arm or
    without one proven clock cap across every pooled start."""
    ck.require(len(windows) >= 1, 'at least one window required')
    for w in windows:
        ck.require(w.get('registration') == REGISTRATION, 'not an NBW2 window')
        for arm in ARMS:
            if arm.role == 'I':
                greedy = record(record(record(w['starts'])[arm.name])['greedy'])
                ck.require(greedy.get('greedy_registration') == GREEDY_REGISTRATION, 'greedy registration differs')
                ck.require(number(greedy.get('identical', -1)) + number(greedy.get('near_tie', 0)) == 6,
                           'six identical-or-near-tie native greedy checks required')
    base: dict[str, Json] = {'rule': dv.RULE, 'rule_revision': dv.RULE_REVISION, 'registration': REGISTRATION,
                             'windows': len(windows), 'max_tokens': MAX_TOKENS}
    caps = clock_caps(windows)
    if len(set(map(str, caps))) != 1 or not isinstance(caps[0], int):
        return {**base, 'status': 'INCONCLUSIVE', 'reason': f'mixed or unproven clock caps across pooled starts: {caps}'}
    cap = caps[0]

    def pooled(name: str, metric: str) -> dv.Series:
        control: list[float] = []
        candidate: list[float] = []
        for w in windows:
            k, i = arm_values(w, name, metric)
            control += k
            candidate += i
        return dv.Series(name, metric, METRICS[metric][1], tuple(control), tuple(candidate), paired=False)

    series = [pooled(name, metric) for name in DECISION_CELLS for metric in (PRIMARY, *GUARDS)]
    counts = {s.cell: [len(s.control), len(s.candidate)] for s in series}
    short = {cell: v for cell, v in counts.items() if min(v) < dv.MIN_REPEATS}
    if short:
        return {**base, 'status': 'INCONCLUSIVE', 'clock_cap_mhz': cap,
                'reason': f'fewer than {dv.MIN_REPEATS} repeats per arm (control, candidate): {short}'}
    series += [pooled('32k-b3', metric) for metric in METRICS]
    verdict = dv.decide_window(series, DECISION_CELLS)
    reported: dict[str, Json] = {name: {m: dv.decide_cell(pooled(name, m)) for m in REPORTED} for name in DECISION_CELLS}
    archive: dict[str, Json] = {}
    for name in DECISION_CELLS:
        try:
            archive[name] = dv.archive_spread(ARCHIVE, name, PRIMARY,
                                              where={'max_tokens': MAX_TOKENS, 'commit': n.PROD_COMMIT, 'clock_cap_mhz': cap})
        except (ValueError, OSError) as exc:
            archive[name] = f'archive unreadable: {exc}'
    kl = [record(record(w['starts']).get(arm.name) or {}).get('kl') for w in windows for arm in ARMS[2::2]]
    return {**verdict, **base, 'clock_cap_mhz': cap, 'primary_metric': PRIMARY, 'guards': list(GUARDS),
            'arms': 'unpaired (Welch): control = every K repeat (one per K start); candidate = every I repeat',
            'repeats_per_arm': cast(Json, counts),
            'per_start_view_never_decides': {name: start_view(windows, name) for name in DECISION_CELLS},
            'reported_metrics_never_decide': reported,
            'archive_day_to_day_spread_ms_per_round': archive, 'greedy_kl_reported_only': cast(Json, kl),
            'promotion': 'written package only; separate attended approval and fidelity review required'}


def archive_rows(result: Mapping[str, Json], out: Path) -> tuple[list[dict[str, Json]], list[str]]:
    """Every measured repeat of every COMPLETE K start (WIN2 checklist 4) in the archive v3 contract: per-start proven
    clock cap, window date, production commit, start and repeat. Rows a start cannot prove are skipped and named."""
    rows: list[dict[str, Json]] = []
    skipped: list[str] = []
    date = win2_history.window_date(out)
    for arm in ARMS:
        start = record(record(result['starts']).get(arm.name) or {})
        if arm.role != 'K' or start.get('status') != 'COMPLETE':
            continue
        if not isinstance(start.get('clock_cap_mhz'), int):
            skipped.append(f'{arm.name}: clock cap unproven ({start.get("clock_error")})')
            continue
        for name in CELL_NAMES:
            for rep in ck.rows(record(record(start['cells'])[name])['repetitions']):
                for metric in ARCHIVED:
                    row: dict[str, Json] = {
                        'window': out.name, 'cell': name, 'metric': metric, 'value': METRICS[metric][0](rep),
                        'max_tokens': MAX_TOKENS, 'commit': n.PROD_COMMIT, 'clock_cap_mhz': start['clock_cap_mhz'],
                        'clock_source': start.get('clock_source'), 'clock_caps_mhz': start.get('clock_caps_mhz'),
                        'date': date, 'start': arm.name, 'repeat': rep['repeat'], 'source': REGISTRATION}
                    try:
                        dv.check_row(row)
                    except ValueError as exc:
                        skipped.append(f'{arm.name}/{name}/{metric}: {exc}')
                        continue
                    rows.append(row)
    return rows, skipped


def chain_predecessor() -> tuple[Path, list[str]]:
    """The chain's production capture (chain.production_snapshot, sha-checked) and the earlier NBW2 windows to pool."""
    path, data = chain.production_snapshot()
    ck.require(data.get('production_commit') == n.PROD_COMMIT and data.get('tag', n.PROD_TAG) == n.PROD_TAG
               and len(ck.rows(data['ranks'])) == 2, 'chain production snapshot is not the current production capture')
    marker = cast(dict[str, Json], chain.read_marker())
    previous = Path(str(marker['window']))
    pooled: list[str] = []
    summary = previous / 'summary.json'
    if previous.parent == CAMPAIGN and summary.exists():
        prior = ck.read(summary)
        if prior.get('registration') == REGISTRATION and prior.get('status') == 'COMPLETE' and prior.get('chain') == 'CHAINED':
            pooled = [*(str(p) for p in cast(list[Json], prior.get('pooled_windows') or [])), str(previous)]
    return path, pooled


class Live(n.Live):
    """NBW Live with repeats, the 0.9-nat greedy registration, the KL probe and WIN2 chain mode."""

    def __init__(self, out: Path, sup: pf.Supervisor, *, native: n.Native, remote: str, expected: str, data_dir: Path,
                 tag: str, chained: bool = False, call: pf.Call | None = None,
                 plain: Callable[[str, list[str], float], tuple[int, str]] | None = None,
                 clock: Callable[[], float] = time.time,
                 chain_abort: Callable[[Path], dict[str, Json]] = chain.abort) -> None:
        super().__init__(out, sup, native=native, remote=remote, expected=expected, data_dir=data_dir, tag=tag,
                         call=call, plain=plain, clock=clock)
        self.chained = chained
        self.chain_abort = chain_abort
        self.chain_baseline: Path | None = None
        self.pooled_from: list[str] = []

    @property
    def pin_files(self) -> tuple[str, ...]:
        return PIN_FILES

    def hold_purpose(self) -> str:
        return f'NBW-WINDOW {self.tag}: K1-327680 K/I native invariant drafter re-measure (WIN2), worker and identity only'

    def fits(self, arm: n.Arm) -> bool:
        return self.clock() + block_s(arm.role) <= self.submit

    def placeholders(self) -> bool:
        """chain.placeholders_present() through this runner's supervised transport (any failure counts as absent)."""
        present = True
        for host in chain.PLACEHOLDER_HOSTS:
            text = self.ssh(f'chain-placeholder-{host}', host,
                            f'if {chain.PLACEHOLDER_PROBE}; then echo PRESENT; else echo ABSENT; fi')
            present &= text.strip() == 'PRESENT'
        return present

    def production_preflight(self) -> None:
        if not self.chained:
            super().production_preflight()
            mode = chain.preflight(production_serving=True, hold_exists=False)
        else:
            state = self.ssh('prod-down', 'sparkhead', 'if curl -fsS --max-time 10 http://localhost:8000/v1/models '
                             '>/dev/null 2>&1; then echo SERVING; else echo DOWN; fi').strip()
            ck.require(state in ('SERVING', 'DOWN'), 'production state unreadable')
            mode = chain.preflight(production_serving=state == 'SERVING', hold_exists=False,
                                   placeholders_present=self.placeholders())
            self.quiet('chain-quiet', 'NBW2 CHAIN QUIET')
            self.chain_baseline, self.pooled_from = chain_predecessor()
        ck.require(mode == ('chained' if self.chained else 'production'), f'chain preflight mode {mode} differs')
        self.admit_unreachable = chain.allow_unreachable(mode)

    def quiet(self, label: str, marker: str) -> None:
        for host in pf.HOSTS:
            text = self.ssh(f'{label}-{host}', host, spark_quiet.command(marker, deadline=self.clock() + 60), timeout=90)
            ck.require(marker in text.splitlines(), f'{host}: Spark not quiet ({label})')

    def open_hold(self) -> float:
        opened = super().open_hold()
        if self.chained:  # S94 FIX 5: the quiet proof ran before the drain; prove it again under our own HOLD
            self.quiet('chain-quiet-held', 'NBW2 CHAIN QUIET UNDER HOLD')
        return opened

    def snapshot(self) -> None:
        if not self.chained:
            super().snapshot()
            return
        # Production cannot be captured while down: inherit the chain's capture of the current production (sha-pinned).
        path = cast(Path, self.chain_baseline)
        data = ck.read(path)
        self.production = [record(r) for r in ck.rows(data['ranks'])]
        self.seen.update(str(r['id']) for r in self.production)
        (self.out / 'rollback.json').write_text(json.dumps({**data, 'production_commit': n.PROD_COMMIT, 'tag': n.PROD_TAG,
            'chained_from': str(path), 'chained_from_sha256': pf.digest(path)}, indent=1) + '\n')

    def start_clock(self, arm: n.Arm) -> dict[str, Json]:
        """This start's proven cap from its own start log (win2_history), recorded per start; never a stop."""
        try:
            cap, source = win2_history.start_clock_cap(self.out, arm.name)
            return {'clock_cap_mhz': cap, 'clock_source': source,
                    'clock_caps_mhz': cast(Json, win2_history.start_clock_caps(self.out, arm.name))}
        except (ValueError, OSError) as exc:
            return {'clock_cap_mhz': None, 'clock_error': f'{type(exc).__name__}: {exc}'}

    def greedy(self, arm: n.Arm) -> dict[str, Json]:
        output = self.out / f'{arm.name}-greedy.json'
        self.call(f'{arm.name}-greedy', self.driver('greedy', output, '--near-tie-nats', repr(GREEDY_NEAR_TIE_NATS),
                  '--record-distribution', '--probe-salt-prefix', f'{self.tag}-{arm.name}'), timeout=300, inflight=1)
        result = ck.read(output)
        checks = [record(r) for r in ck.rows(result['checks'])]
        ck.require([r.get('i') for r in checks] == list(range(6))
                   and all(r.get('identical') is True or r.get('near_tie') is True for r in checks)
                   and number(result.get('identical', -1)) + number(result.get('near_tie', -1)) == 6,
                   'native greedy divergence that is not a 0.9-nat near-tie, or missing checks')
        return {**result, 'greedy_registration': GREEDY_REGISTRATION, 'near_tie_nats': GREEDY_NEAR_TIE_NATS}

    def kl(self, arm: n.Arm, candidate: n.Arm) -> dict[str, Json]:
        (self.out / 'kl').mkdir(exist_ok=True)
        output = self.out / 'kl' / f'{arm.name}-kl.json'
        self.call(f'{arm.name}-kl', [sys.executable, '-m', 'offload.launch.windows.nbw2_cells', '--execute', '--mode', 'kl',
                  '--output', str(output), '--deadline-utc', str(self.submit), '--salt', f'{self.tag}-{arm.name}',
                  '--candidate', str(self.out / f'{candidate.name}-greedy.json')], timeout=300, inflight=1)
        return ck.read(output)

    def repeat(self, arm: n.Arm, name: str, rep: int) -> dict[str, Json]:
        """n.Live.cell with a fresh cache salt and output per repeat; returns the start's cell aggregate."""
        label = f'{arm.name}-{name}-r{rep}'
        before = self.stats(f'{label}-before')
        steps = number(self.probe_worker(f'{label}-worker-before', arm).get('steps', 0))
        (self.out / 'repeats').mkdir(exist_ok=True)
        output = self.out / 'repeats' / f'{label}.json'
        self.call(label, self.driver('cell', output, '--cell', name, '--cuts', str(self.out / 'cuts.json'),
            '--salt', f'{self.tag}-{arm.name}-{name}-r{rep}', '--ledger', str(self.out / 'requests-ledger.jsonl')),
            timeout=cells.CELLS[name].estimate_s * 3 + 60, inflight=cells.CELLS[name].batch)
        result = ck.read(output)
        ck.require(result.get('cell') == name and result.get('batch') == cells.CELLS[name].batch, 'cell artifact identity mismatch')
        after = self.stats(f'{label}-after')
        ck.counters(after, 0, before)
        self.round_baseline.setdefault(arm.name, {key: number(record(before['counters']).get(key, 0)) for key in n.FAILED_ROUNDS})
        for key in n.FAILED_ROUNDS:
            ck.require(number(record(after['counters']).get(key, 0)) == number(record(before['counters']).get(key, 0)),
                       f'{arm.name}: {key} grew during {name} repeat {rep}')
        ck.require(number(record(after['counters'])['rounds_ok']) > number(record(before['counters'])['rounds_ok']), 'no drafting during cell')
        ck.require(number(self.probe_worker(f'{label}-worker-after', arm).get('steps', 0)) > steps, 'worker took no steps')
        result['proxy_before'], result['proxy_after'], result['repeat'] = before, after, rep
        output.write_text(json.dumps(result, allow_nan=False) + '\n')
        aggregate = self.out / f'{arm.name}-{name}.json'
        fresh: dict[str, Json] = {'cell': name, 'batch': cells.CELLS[name].batch, 'repetitions': []}
        data = ck.read(aggregate) if rep else fresh
        ck.require(len(ck.rows(data['repetitions'])) == rep, 'repeat order differs')
        data['repetitions'] = [*ck.rows(data['repetitions']), repetition(result, rep, output)]
        aggregate.write_text(json.dumps(data, allow_nan=False) + '\n')
        return data

    def chained_end(self) -> dict[str, Json]:
        """WIN2 chain: pair closed, experiment stopped, Sparks quiet, placeholders proven, own HOLD closed with
        chain.CHAINED_CLOSE + the fenced commit-close, production LEFT DOWN, marker written. Raises ChainUnprovable
        before its one -restore-close scope is used when the shared placeholder check cannot pass."""
        self.sup.close()
        original = self.plain
        def bounded(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
            if label in ('cancel-close', 'retain-close'):
                return original(label, argv, min(30, timeout))
            left = self.restore_by - self.clock()
            ck.require(left > 0, 'NBW HOLD+90 exhausted; manual restore required; HOLD stays open')
            return original(label, argv, min(left, timeout))
        self.plain = bounded
        try:
            for tag in list(self.owner_tags):
                self.fence_phase(tag)
            self.pair_close(guarded=False)
            result: dict[str, Json] = {'pair_close': 'PRODUCTION HOLD OK'}
            rc, _ = self.recovery('experiment-stop', [*pf.SSH, 'sparkhead', f'cd ~/{self.remote} && ./stop.sh </dev/null'], 120)
            ck.require(rc == 0, 'NBW experiment stop failed; HOLD stays open')
            quiet: dict[str, Json] = {}
            for host in pf.HOSTS:
                rc, text = self.plain('jit-wait-' + host, [*pf.SSH, host,
                    spark_quiet.command('NBW EXPERIMENT ABSENT', deadline=min(self.restore_by - 90, self.clock() + 350))], 360)
                ck.require(rc == 0 and 'NBW EXPERIMENT ABSENT' in text.splitlines(),
                           f'{host}: experiment/container/compute absence unproved; HOLD stays open')
                quiet[host] = 'NBW EXPERIMENT ABSENT'
            result['sparks'] = quiet
            deadline = self.restore_by - 90
            ck.require(self.clock() < deadline, 'NBW chained close budget exhausted; HOLD stays open')
            proof = f'{self.tag}-chain-proof'
            self.remember(proof, deadline)
            try:
                rc, text = self.plain('chain-placeholder-proof', [*pf.SSH, 'sparkhead',
                    owner.command('launch', proof, ['bash', '-c', placeholder_proof()], deadline=deadline)], 120)
            finally:
                self.fence_phase(proof)
            if rc != 0 or PLACEHOLDERS_PROVEN not in text.splitlines():
                raise ChainUnprovable(f'hold placeholders not provable by chain.CHAINED_CLOSE (rc{rc}: {text.strip()[-200:]})')
            tag = f'{self.tag}-restore-close'
            self.remember(tag, deadline)
            try:
                rc, text = self.plain('chained-close', [*pf.SSH, 'sparkhead',
                    owner.command('launch', tag, ['bash', '-c', chain.CHAINED_CLOSE], deadline=deadline)], 120)
            finally:
                self.fence_phase(tag)
            ck.require(rc == 0 and chain.CHAINED_CLOSE_OK in text.splitlines(), 'chained HOLD close unrecorded; HOLD stays open')
            self.commit_close(result)
            chain.finish_chained(self.out, {'runner': REGISTRATION, 'sparks': quiet, 'x86host': 'PRODUCTION HOLD OK',
                                            'hold': 'owned HOLD closed (fenced commit-close)'})
            result['status'] = 'CHAINED: production left down by design (NDM 2026-10-27); Sparks quiet; own HOLD closed'
            return result
        finally:
            self.plain = original
            (self.out / 'arm-memory.json').write_text(json.dumps(self.sup.arm_memory) + '\n')


class ChainUnprovable(ValueError):
    """A clean chained end that the next chained window could not accept: end with the strict restore instead."""


def window(b: Live, checkpoint: Callable[[dict[str, Json]], None]) -> dict[str, Json]:
    """K I K I K with per-role repeats; the end follows chain.end_action (WIN2-R2)."""
    result: dict[str, Json] = {'status': 'INCOMPLETE', 'registration': REGISTRATION, 'chain': 'OFF', 'starts': {},
                               'decision': {'status': 'NOT A CANDIDATE', 'reason': 'not measured completely'}}
    starts = record(result['starts'])
    hold = False
    stopped = b.chained  # a chained window begins with production already down
    try:
        b.preflight()
        hold = True
        result['hold_opened'] = b.open_hold()
        b.snapshot()
        for i, arm in enumerate(ARMS):
            if i == len(ARMS) - 1 and not b.fits(arm):
                starts[arm.name] = {'status': 'SKIPPED_TIME', 'role': arm.role}
                checkpoint({'closing_K': 'SKIPPED_TIME'})
                break
            ck.require(b.fits(arm), f'{arm.name}: mandatory block cannot fit')
            stopped = True  # native pair operation or start_exp may mutate despite a lost reply
            row: dict[str, Json] = {'status': 'STARTED', 'role': arm.role, 'cells': {}}
            starts[arm.name] = row
            checkpoint({'start': arm.name, 'status': 'STARTED'})
            b.start(arm)
            row.update(b.start_clock(arm))
            if i == 0:
                b.build()
            if arm.role == 'I':
                row['greedy'] = b.greedy(arm)
            elif i and ARMS[i - 1].role == 'I':
                row['kl'] = b.kl(arm, ARMS[i - 1])
            for name, reps in REPEATS[arm.role].items():
                for rep in range(reps):
                    record(row['cells'])[name] = b.repeat(arm, name, rep)
                    checkpoint({'start': arm.name, 'cell': name, 'repeat': rep})
            row['checks'] = b.finish(arm)
            row['status'] = 'COMPLETE'
            checkpoint({'start': arm.name, 'status': 'COMPLETE'})
        result['status'] = 'COMPLETE'
    except (Exception, KeyboardInterrupt) as exc:
        result['stop'] = f'{type(exc).__name__}: {exc}'
    finally:
        saved = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        try:
            end(b, result, hold, stopped)
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
    rows: list[dict[str, Json]] = []
    skipped: list[str] = []
    try:
        rows, skipped = archive_rows(result, b.out)
        head = dv.archive_head(ARCHIVE)
        if rows:
            dv.archive_append(ARCHIVE, rows)
        result['archive'] = {'appended': len(rows), 'skipped': cast(Json, skipped), 'path': str(ARCHIVE),
                             'head_before': head, 'head_after': dv.archive_head(ARCHIVE)}
    except (ValueError, OSError) as exc:
        result['archive'] = {'appended': 0, 'skipped': cast(Json, skipped), 'error': f'{type(exc).__name__}: {exc}'}
    restored = str(record(result['restore']).get('status', ''))
    if result['status'] == 'COMPLETE' and restored.startswith(('RESTORED', 'CHAINED')):
        result['pooled_windows'] = cast(Json, list(b.pooled_from))
        try:
            result['decision'] = decide([result])
        except ValueError as exc:
            result['decision'] = {'status': 'NOT A CANDIDATE', 'reason': f'decision refused: {exc}'}
        if b.pooled_from:
            try:
                result['pooled_decision'] = decide([*(ck.read(Path(p) / 'summary.json') for p in b.pooled_from), result])
            except (ValueError, OSError) as exc:
                result['pooled_decision'] = {'status': 'NOT A CANDIDATE', 'reason': f'pooled decision refused: {exc}'}
    elif not restored.startswith(('RESTORED', 'CHAINED', 'NOT NEEDED')):
        result['decision'] = {'status': 'NOT A CANDIDATE', 'reason': 'production restore unverified'}
    return result


def end(b: Live, result: dict[str, Json], hold: bool, stopped: bool) -> None:
    """chain.end_action: 'chained' -> chained_end (strict restore when unprovable or failed); 'restore' -> the strict W8
    restore; a stop without an owned HOLD while the chain marker is present -> chain.abort. Signals are ignored."""
    clean = 'stop' not in result
    if hold and chain.end_action(stopped=not clean, hold_written=True) == 'chained':
        try:
            result['restore'] = b.chained_end()
            result['chain'] = 'CHAINED'
            return
        except (Exception, KeyboardInterrupt) as exc:
            # Unprovable placeholders (the -restore-close scope is still unused) or any later failure: strict restore.
            key = 'chain_end' if isinstance(exc, ChainUnprovable) else 'chained_end_error'
            result[key] = f'{type(exc).__name__}: {exc}'
    if hold:
        try:
            result['restore'] = b.restore(production_stopped=stopped)
        except (Exception, KeyboardInterrupt) as exc:
            result['restore'] = {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'}
    action = chain.end_action(stopped=True, hold_written=b.hold_written)
    if action == 'abort':
        # No owned HOLD while the chain left production down (pre-HOLD refusal, or a chained HOLD admission that never
        # published): the shared chain abort restores under its own HOLD, or alerts and touches nothing.
        if hold:
            result['restore_attempt'] = result['restore']
        aborted = b.chain_abort(b.out / 'chain-abort')
        result['restore'] = {'status': f'CHAIN ABORT: {aborted.get("status")}', 'abort': cast(Json, aborted)}
        result['chain'] = 'ABORT'
    elif not hold:
        result['restore'] = {'status': 'NOT NEEDED: preflight only'}


def after_window(result: dict[str, Json], watch: Callable[[], dict[str, Json]]) -> int:
    """Post-close watch after the window's own W8 restore; clear the chain marker ONLY on RESTORED + watch PASS (R2)."""
    restored = str(record(result['restore']).get('status', ''))
    if restored.startswith('RESTORED'):
        watched = watch()
        result['post_close_watch'] = cast(Json, watched)
        if watched.get('status') != 'PASS':
            result['decision'] = {'status': 'NOT A CANDIDATE', 'reason': 'post-close production health unverified'}
            result.pop('pooled_decision', None)
            return 2
        chain.clear()
        if chain.enabled():
            result['chain'] = 'LAST' if chain.last() and 'stop' not in result else 'ENDED (strict restore)'
        return 0 if 'stop' not in result else 1
    if restored.startswith('CHAINED') or restored.startswith('NOT NEEDED'):
        return 0 if 'stop' not in result else 1
    if restored.startswith('CHAIN ABORT: RESTORED'):
        return 1  # stopped; production restored by the abort (its own watch PASSED, marker cleared)
    return 2


def plan() -> str:
    return ('DRY RUN NBW v2 (WIN2): no command, network or file. Coordinator Deepseek/coordinator only.\n'
            f'Production {n.PROD_COMMIT[:7]} / {n.PROD_TAG}; K1 327680, KV 2502600. Same preflight/HOLD/fences/pair as NBW.\n'
            'One window = K I K I K (closing K skipped if its block no longer fits). Run as single windows; pool a second\n'
            '  one with --pool-with <first window dir>. WINDOW_CHAIN=1 is wired (WIN2-R2), but at offload/win2 2b48eec\n'
            '  the shared placeholder check cannot see scoped placeholders, so a chained NBW2 window ends with the strict\n'
            '  W8 restore; a window started inside another runner\'s chain refuses before HOLD and runs chain.abort.\n'
            'K starts: short-b1, 32k-b1, 32k-b3 x1. I starts: short-b1 x2, 32k-b1 x2 (DECISION: 3 K vs 4 I), 32k-b3 x1 (reported).\n'
            '  Fresh cache salt per repeat; T=1, 1500-token answers; every start records its proven clock cap.\n'
            f'I starts: six greedy checks at {GREEDY_NEAR_TIE_NATS} nats ({GREEDY_REGISTRATION}, nbw_cells --near-tie-nats),\n'
            '  divergence top-20 recorded; the following K re-probes them (KL, reported only).\n'
            f'Decision {REGISTRATION} = decision_v2 win2-v1 rev {dv.RULE_REVISION}, unpaired Welch: primary {PRIMARY}, guards '
            f'{", ".join(GUARDS)};\n'
            '  one clock cap across every pooled start, else INCONCLUSIVE; per-start-mean view reported beside it.\n'
            f'ESTIMATED blocks (MEASURED NBW-W3 starts/cells x{MARGIN}): K {block_s("K") / 60:.1f} min, I {block_s("I") / 60:.1f} min,\n'
            f'  K I K I {window_estimate_s(ARMS[:4]) / 60:.1f} min, with closing K {window_estimate_s() / 60:.1f} min <= 72-min bound.\n'
            '  Every K repeat appended to the control archive (v3: commit, per-start clock cap, date, start, repeat).\n'
            'End: strict W8 start/check/close + 15-min watch; the chain marker is cleared only after the watch PASSES.')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--remote-repo', default='dsv41-deploy-offload')
    ap.add_argument('--expected-experiment-commit', default='')
    ap.add_argument('--native-tree', default='')
    ap.add_argument('--native-commit', default='')
    ap.add_argument('--native-identity', default='')
    ap.add_argument('--native-markov-identity', default='')
    ap.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    ap.add_argument('--pool-with', type=Path, nargs='*', default=[],
                    help='earlier complete NBW2 window directories to pool into pooled_decision')
    ap.add_argument('--decide', type=Path, nargs='+', help='pool complete NBW2 window directories (local files only)')
    args = ap.parse_args(argv)
    if args.decide:
        print(json.dumps(decide([ck.read(p / 'summary.json') for p in args.decide]), indent=1))
        return 0
    if not args.execute:
        print(plan())
        return 0
    native = n.Native(args.native_tree, args.native_commit, args.native_identity, args.native_markov_identity)
    try:
        ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'WINDOW_COORDINATOR=Deepseek/coordinator required')
        native.validate()
        ck.require(args.out is not None and args.out.resolve().parent == pf.CAMPAIGN and not args.out.exists(), 'fresh --out inside campaign required')
        ck.require(re.fullmatch(r'NBW2-W-20[0-9]{6}T[0-9]{6}Z', args.out.name) is not None, '--out must be NBW2-W-YYYYMMDDTHHMMSSZ')
        ck.require(re.fullmatch('[0-9a-f]{40}', args.expected_experiment_commit) is not None, 'full reviewed experiment commit required')
        ck.require(re.fullmatch('[A-Za-z0-9_.-]+', args.remote_repo) is not None and args.remote_repo != 'dsv41-deploy', 'separate experiment checkout required')
        ck.require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pf.ROOT, text=True).strip() == args.expected_experiment_commit,
                   'local HEAD differs from reviewed commit')
        ck.require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=pf.ROOT, text=True).strip(), 'tracked source dirty')
        for prior in args.pool_with:
            data = ck.read(prior / 'summary.json')
            ck.require(prior.resolve().parent == pf.CAMPAIGN and data.get('registration') == REGISTRATION
                       and data.get('status') == 'COMPLETE', f'{prior}: not a complete NBW2 window')
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        if chain.marker_present():
            print(f'STOP before any remote action while the chain marker is present (production is down): {exc}; '
                  'run offload/launch/windows/chain_abort.sh', file=sys.stderr)
            return 2
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
    live = Live(out, sup, native=native, remote=args.remote_repo, expected=args.expected_experiment_commit,
                data_dir=args.data_dir, tag=tag, chained=chain.marker_present())
    live.pooled_from = [str(p.resolve()) for p in args.pool_with]
    result = window(live, checkpoint)
    sup.close()
    def watch() -> dict[str, Json]:
        def watch_log(message: str) -> None:
            print(message, flush=True)
            with (out / 'post-close-watch.log').open('a') as handle:
                handle.write(message + '\n')
        watched = watchdog.watch(watchdog.ProductionHealth(),
            watchdog.ScopedRecovery(tag + '-watch', out / 'post-close-commands'), log=watch_log)
        return cast(dict[str, Json], dict(watched.__dict__))
    code = after_window(result, watch)
    result.update({'experiment_commit': args.expected_experiment_commit, 'production_commit': n.PROD_COMMIT,
                   'restore_sha256': pf.RESTORE_SHA256, 'config_sha256': pf.digest(pf.ROOT / n.CONFIG),
                   'unavailable_traces': 'actual B, raw confidence, chosen depth, exposed draft wait not exported by unchanged production',
                   'artifact_sha256': {str(p.relative_to(out)): pf.digest(p) for p in sorted(out.rglob('*.json*')) if p.is_file()}})
    (out / 'summary.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'stop': result.get('stop'), 'chain': result['chain'],
                      'decision': record(result['decision']).get('status'),
                      'pooled_decision': record(result.get('pooled_decision') or {}).get('status'), 'restore': result['restore']}))
    if code == 2:
        print('PRODUCTION NOT VERIFIED (restore failed, watch failed, or chain abort ALERT/FAILED): inspect cmd/restore-*.log, '
              'chain-abort/, HOLD and the chain marker before further work.', file=sys.stderr)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
