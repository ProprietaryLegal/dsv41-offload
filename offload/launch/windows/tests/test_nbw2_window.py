"""NBW v2 (WIN2): real Live/Supervisor over the NBW fake transport; decision, chain and archive paths. CPU only."""
from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import chain
from offload.launch.windows import decision_v2 as dv
from offload.launch.windows import nbw2_cells as g
from offload.launch.windows import nbw2_window as w
from offload.launch.windows import nbw_cells as c
from offload.launch.windows import nbw_window as n
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import win2_history
from offload.launch.windows.tests import test_nbw_rehearsal as rh
from offload.launch.windows.tests import test_nbw_window as pure
from offload.launch.windows.tests import test_pf1_window as base
from offload.prefill_maxlen.evidence import Json, number, record

# MEASURED NBW-W-20261026T231445Z K server ms per spec request-round (s1/s3/s5) and acceptance per round.
K_MS = {'short-b1': (39.117, 39.087, 39.255), '32k-b1': (38.246, 38.189, 39.002), '32k-b3': (72.613, 73.808, 72.254)}
K_ACC = {'short-b1': (1.2124, 1.0576, 1.1052), '32k-b1': (0.9405, 0.9367, 1.1231), '32k-b3': (1.0623, 1.1067, 1.033)}
CAP = 2300


def k_ms(name: str, start: int, rep: int) -> float:
    return K_MS[name][start % 3] * (1 + .002 * (rep - 1))


def rep_row(name: str, rep: int, ms: float, acc: float, tps: float = 50.0) -> dict[str, Json]:
    streams: list[Json] = [{'stream': i, 'prompt_sha256': str(i) * 64, 'sampling': {'temperature': 1, 'max_tokens': 1500}}
                           for i in range(c.CELLS[name].batch)]
    return {'repeat': rep, 'server_decode_ms': ms * 700, 'spec_request_rounds': 700.0,
            'server_decode_ms_per_spec_request_round': ms, 'acceptance_per_round': acc, 'median_decode_tps': tps,
            'median_ttft_s': 2.0, 'max_tokens': 1500, 'streams': streams}


def summary(i_scale: float = .95, i_acc: float = 1.0, b3_scale: float = 1.0) -> dict[str, Json]:
    """A complete NBW2 window: I ms/round = K x i_scale (lower is better), I acceptance = K x i_acc."""
    starts: dict[str, Json] = {}
    for s, arm in enumerate(w.ARMS):
        cells: dict[str, Json] = {}
        for name, reps in w.REPEATS[arm.role].items():
            rows = []
            for r in range(reps):
                ms, acc = k_ms(name, s, r), K_ACC[name][s % 3]
                if arm.role == 'I':
                    ms *= b3_scale if name == '32k-b3' else i_scale
                    acc *= 1.0 if name == '32k-b3' else i_acc
                rows.append(rep_row(name, r, ms, acc, 1000 * (1 + acc) / ms))
            cells[name] = {'cell': name, 'batch': c.CELLS[name].batch, 'repetitions': cast(Json, rows)}
        row: dict[str, Json] = {'status': 'COMPLETE', 'role': arm.role, 'cells': cells, 'clock_cap_mhz': CAP}
        if arm.role == 'I':
            row['greedy'] = {'identical': 5, 'near_tie': 1, 'greedy_registration': w.GREEDY_REGISTRATION}
        starts[arm.name] = row
    return {'registration': w.REGISTRATION, 'status': 'COMPLETE', 'starts': starts}


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(w, 'ARCHIVE', tmp_path / 'control-archive.jsonl')
    monkeypatch.setattr(w, 'CAMPAIGN', tmp_path)
    monkeypatch.setattr(chain, 'MARKER', tmp_path / 'scratch' / 'CHAINED')
    monkeypatch.delenv('WINDOW_CHAIN', raising=False)
    monkeypatch.delenv('WINDOW_CHAIN_LAST', raising=False)


def test_budget_fits_the_submission_bound_from_measured_starts() -> None:
    assert w.block_s('K') == pytest.approx(1.15 * (375 + 40 + 51 + 108))
    assert w.block_s('I') == pytest.approx(1.15 * (670 + 2 * 40 + 2 * 51 + 108))
    assert w.window_estimate_s() / 60 == pytest.approx(69.8, abs=.1)
    assert w.window_estimate_s() <= n.SUBMIT_S, 'K I K I K with its closing K fits the 72-minute bound'
    assert w.window_estimate_s(w.ARMS[:4]) <= n.SUBMIT_S - 13 * 60, 'mandatory K I K I keeps >= 13 min slack'
    assert [a.role for a in w.ARMS] == ['K', 'I', 'K', 'I', 'K']
    for cell in w.DECISION_CELLS:  # coordinator 2026-10-27: >= 3 candidate repeats per B1 decision cell
        assert 2 * w.REPEATS['I'][cell] >= 3 and 3 * w.REPEATS['K'][cell] >= 3
    assert set(w.PIN_FILES) >= set(n.PIN_FILES) and 'offload/launch/windows/nbw2_window.py' in w.PIN_FILES


def test_faster_rounds_win_and_measured_w3_noise_does_not() -> None:
    # Rule revision 2 (WIN2-R2 FIX 3): spread 1.69 x SD / c4(n); a primary LOSS needs only its loss-repeat test.
    win = w.decide([summary(.95)])
    assert win['status'] == 'CANDIDATE' and win['rule'] == 'win2-v1' and win['rule_revision'] == 2
    assert record(win['repeats_per_arm'])['short-b1'] == [3, 4] and win['max_tokens'] == 1500 and win['clock_cap_mhz'] == CAP
    primary = record(record(record(record(win['cells'])['short-b1'])['metrics'])['ms_per_round'])
    assert 'Welch' in str(primary['interval_method']) and 'c4' in str(primary['spread_method'])
    assert w.decide([summary(1.0)])['status'] == 'INCONCLUSIVE'
    # 1.5 % slower in every I repeat is beyond the short-b1 control spread (0.44 % = 1.69 SD/c4 of W3's K values).
    worse = w.decide([summary(1.015)])
    assert worse['status'] == 'NOT A CANDIDATE'
    short = record(record(record(record(worse['cells'])['short-b1'])['metrics'])['ms_per_round'])
    assert short['verdict'] == 'LOSS' and number(short['control_spread']) == pytest.approx(.0044, abs=.0003)
    assert w.decide([summary(.99)])['status'] == 'INCONCLUSIVE'  # +1 % is inside the 2 % minimum


def test_guards_lose_only_with_their_interval_below_zero_and_b3_never_decides() -> None:
    # Acceptance x0.7: every I repeat is beyond the 13 % acceptance spread AND the guard's interval is below zero.
    lost = w.decide([summary(.95, i_acc=.7)])
    assert lost['status'] == 'NOT A CANDIDATE'
    guard = record(record(record(record(lost['cells'])['short-b1'])['metrics'])['accepted_per_round'])
    assert guard['role'] == 'guard' and guard['verdict'] == 'LOSS' and number(cast(list[Json], guard['interval95'])[1]) < 0
    # Acceptance x0.9 sits inside the acceptance spread: the guard does not fire and the faster rounds still win.
    assert w.decide([summary(.95, i_acc=.9)])['status'] == 'CANDIDATE'
    assert w.decide([summary(.95, i_acc=1.3)])['status'] == 'CANDIDATE'
    reported = w.decide([summary(.95, b3_scale=1.2)])
    assert reported['status'] == 'CANDIDATE'
    assert record(record(reported['cells'])['32k-b3'])['verdict'] == 'REPORTED'
    committed = record(record(reported['reported_metrics_never_decide'])['short-b1'])['committed_tps']
    assert number(record(committed)['point_gain']) > .02
    view = record(record(reported['per_start_view_never_decides'])['short-b1'])
    assert len(cast(list[Json], view['i_start_means'])) == 2 and len(cast(list[Json], view['k_start_means'])) == 3


def test_one_clock_per_decision() -> None:
    s = summary(.95)
    record(record(s['starts'])['s4-I'])['clock_cap_mhz'] = 2150  # THERMAL-WATCH dial-back between starts
    mixed = w.decide([s])
    assert mixed['status'] == 'INCONCLUSIVE' and 'mixed or unproven clock caps' in str(mixed['reason'])
    record(record(s['starts'])['s4-I'])['clock_cap_mhz'] = None
    assert w.decide([s])['status'] == 'INCONCLUSIVE'
    other = summary(.95)
    for row in record(other['starts']).values():
        record(row)['clock_cap_mhz'] = 2150
    assert w.decide([summary(.95), other])['status'] == 'INCONCLUSIVE', 'windows at different caps never pool'


def test_skipped_closing_k_is_inconclusive_alone_but_pools() -> None:
    s = summary(.95)
    record(s['starts'])['s5-K'] = {'status': 'SKIPPED_TIME', 'role': 'K'}
    alone = w.decide([s])
    assert alone['status'] == 'INCONCLUSIVE' and 'fewer than 3' in str(alone['reason'])
    both = w.decide([s, summary(.95)])
    assert both['status'] == 'CANDIDATE' and record(both['repeats_per_arm'])['short-b1'] == [5, 8]


def test_pooled_windows_double_repeats() -> None:
    pooled = w.decide([summary(.97), summary(.97)])
    assert pooled['windows'] == 2 and record(pooled['repeats_per_arm'])['32k-b1'] == [6, 8]
    assert pooled['status'] == 'CANDIDATE'


@pytest.mark.parametrize('mutate,message', [
    (lambda s: record(record(record(s['starts'])['s2-I'])['greedy']).update(greedy_registration='nbw-0p501'), 'registration'),
    (lambda s: record(record(record(s['starts'])['s2-I'])['greedy']).update(identical=4), 'six identical'),
    (lambda s: record(record(record(s['starts'])['s3-K'])).update(status='STARTED'), 's3-K: complete start'),
    (lambda s: cast(list[Json], record(record(record(record(s['starts'])['s1-K'])['cells'])['short-b1'])['repetitions']).pop(),
     'short-b1: 1 repeats required'),
    (lambda s: record(cast(list[Json], record(cast(list[Json], record(record(record(record(s['starts'])['s2-I'])['cells'])
     ['32k-b1'])['repetitions'])[0])['streams'])[0]).update(prompt_sha256='f' * 64), 'different public prompts'),
    (lambda s: record(cast(list[Json], record(record(record(record(s['starts'])['s4-I'])['cells'])['short-b1'])['repetitions'])[1])
     .update(max_tokens=512), '1,500-token'),
    (lambda s: record(record(record(s['starts'])['s4-I'])['greedy']).update(near_tie=0), 'six identical'),
    (lambda s: s.update(registration='nbw-v1'), 'not an NBW2 window'),
])
def test_decision_refuses_incomplete_or_mismatched_evidence(mutate: Callable[[dict[str, Json]], object], message: str) -> None:
    s = summary()
    mutate(s)
    with pytest.raises(ValueError, match=message):
        w.decide([s])


def test_greedy_runs_the_shared_driver_at_0p9_nats_with_salted_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """S94 probe: the registered 0.9 (+1e-3) limit reaches nbw_cells.greedy through its command line (WIN2-R2)."""
    monkeypatch.setattr(c, 'greedy_inputs', lambda: (['public prompt'] * 6, [{'i': i, 'text': 'public reference'} for i in range(6)]))
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    for margin, ok in ((0.5, True), (0.85, True), (0.9, True), (0.95, False)):
        api = pure.GreedyApi(margin)
        monkeypatch.setattr(c, 'Http', lambda url, api=api: api)
        out = tmp_path / f'greedy-{margin}.json'
        argv = ['--execute', '--mode', 'greedy', '--output', str(out), '--deadline-utc', '1e12',
                '--near-tie-nats', repr(w.GREEDY_NEAR_TIE_NATS), '--record-distribution', '--probe-salt-prefix', 'W-tag-s2-I']
        if not ok:
            with pytest.raises(ValueError, match='not a near-tie'):
                c.main(argv)
            continue
        assert c.main(argv) == 0
        result = json.loads(out.read_text())
        row = record(result['checks'][3])
        assert result['identical'] == 5 and result['near_tie'] == 1 and row['near_tie'] is True
        probe = [b for b in api.calls if b['endpoint'] == '/v1/completions']
        assert len(probe) == 1 and probe[0]['cache_salt'] == 'W-tag-s2-I:probe:3'
        assert 'top20_logprobs' in row and row['probe_prefix_token_ids'] == [1, 2, 3, 10, 11]


class Remote(rh.Remote):
    """NBW fake transport + KL driver, per-repeat server metrics, start-log clock lines, chain probes, modelled fences."""

    def __init__(self, out: Path, sup: pf.Supervisor, clock: base.Clock, fail: str, trip: Path, *,
                 i_scale: float = .95, production_down: bool = False, placeholders: bool = True,
                 provable: bool = False) -> None:
        super().__init__(out, sup, clock, fail, trip)
        self.i_scale = i_scale
        self.production_down = production_down
        self.placeholders_up = placeholders
        self.provable = provable  # True models a FIXED shared placeholder check; at offload/win2 2b48eec it is False
        self.fenced: set[str] = set()
        self.kl_starts: list[str] = []
        self.salts: list[str] = []
        self.greedy_argv: list[list[str]] = []

    def call(self, label: str, argv: list[str], *, stdin: str | None = None, timeout: float = 120, inflight: int = 0) -> str:
        if argv[1:3] == ['-m', 'offload.launch.windows.nbw2_cells']:
            self.sup.poll()
            arm = cast(n.Arm, self.arm)
            previous = w.ARMS[[a.name for a in w.ARMS].index(arm.name) - 1]
            assert arm.role == 'K' and argv[argv.index('--mode') + 1] == 'kl'
            assert Path(argv[argv.index('--candidate') + 1]).name == f'{previous.name}-greedy.json'
            self.kl_starts.append(arm.name)
            Path(argv[argv.index('--output') + 1]).write_text(json.dumps({'mode': 'kl', 'status': 'REPORTED ONLY',
                'divergences': 1, 'rows': [{'i': 3, 'kl_nats': .001, 'common_mass': .999, 'pass': True}]}))
            return ''
        if argv[1:3] == ['-m', 'offload.launch.windows.nbw_cells'] and argv[argv.index('--mode') + 1] == 'greedy':
            self.greedy_argv.append(argv)
            assert argv[argv.index('--near-tie-nats') + 1] == repr(w.GREEDY_NEAR_TIE_NATS) and '--record-distribution' in argv
            assert argv[argv.index('--probe-salt-prefix') + 1] == f'{self.tag}-{cast(n.Arm, self.arm).name}'
            if self.fail != 'greedy-divergence':
                self.fail, original = 'greedy-near-tie', self.fail
                try:
                    return super().call(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)
                finally:
                    self.fail = original
        text = super().call(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)
        if argv[1:3] == ['-m', 'offload.launch.windows.nbw_cells'] and argv[argv.index('--mode') + 1] == 'cell':
            arm, name = cast(n.Arm, self.arm), argv[argv.index('--cell') + 1]
            salt = argv[argv.index('--salt') + 1]
            self.salts.append(salt)
            rep = int(salt.rsplit('-r', 1)[1])
            start = [a.name for a in w.ARMS].index(arm.name)
            ms = k_ms(name, start, rep) * (self.i_scale if arm.role == 'I' else 1)
            path = Path(argv[argv.index('--output') + 1])
            cell = json.loads(path.read_text())
            cell.update(server_decode_ms_per_spec_request_round=ms, spec_request_rounds=700.0, median_ttft_s=2.0,
                        acceptance_per_round=K_ACC[name][start % 3])
            path.write_text(json.dumps(cell))
        return text

    def remote(self, label: str, host: str, command: str, stdin: str | None) -> str:
        if label == 'prod-down':
            return 'DOWN\n' if self.production_down else 'SERVING\n'
        if label in ('prod-health', 'prod-models'):
            assert not self.production_down, 'chained window probed a production that is down by design'
        if label.startswith('chain-placeholder-'):
            assert chain.PLACEHOLDER_PROBE in command
            return 'PRESENT\n' if self.placeholders_up else 'ABSENT\n'
        if label.startswith('chain-quiet-'):
            marker = 'NBW2 CHAIN QUIET UNDER HOLD' if label.startswith('chain-quiet-held-') else 'NBW2 CHAIN QUIET'
            assert marker in command
            if label.startswith('chain-quiet-held-'):
                assert self.hold, 'the second quiet proof runs under our own HOLD'
            return '' if self.fail == 'chain-not-quiet' else marker + '\n'
        if label == 'hold-open':  # the admission mode is the last admit argument (the inline source names both)
            _, payload = rh.scoped_argv(command)
            mode = 'allow-unreachable' if self.production_down else 'require-serving'
            assert f' {mode}; set -a;' in payload[-1]
            if self.fail == 'hold-admission':
                raise ValueError('drain metrics unavailable; HOLD not admitted')
        if label.endswith('-start') and label.split('-start')[0] in [a.name for a in w.ARMS]:
            arm_name = label[:-len('-start')]
            mhz = 2150 if self.fail == 'mixed-clock' and arm_name == 's4-I' else CAP
            log = self.out / 'cmd' / f'{9000 + len(self.started):04d}-{arm_name}-start.log'
            log.write_text(f'sparkhead clock cap {mhz} MHz 2026-10-27T21:00:00Z\nsparkworker clock cap {mhz} MHz 2026-10-27T21:00:00Z\n')
        return super().remote(label, host, command, stdin)

    def plain(self, step: str, argv: list[str], timeout: float) -> tuple[int, str]:
        if step.startswith('owned-fence-'):
            self.fenced.add(next(step[len('owned-fence-'):-len(h) - 1] for h in ('sparkworker-wired', 'sparkhead', 'x86host', 'localhost')
                                 if step.endswith('-' + h)))
        if step == 'chain-placeholder-proof':
            self.events.append(step)
            tag, payload = rh.scoped_argv(argv[-1])
            assert tag == self.tag + '-chain-proof' and payload == ['bash', '-c', w.placeholder_proof()]
            return (0, w.PLACEHOLDERS_PROVEN + '\n') if self.provable else (1, 'hold placeholder missing on sparkhead\n')
        if step == 'chained-close':
            self.events.append(step)
            tag, payload = rh.scoped_argv(argv[-1])
            assert tag == self.tag + '-restore-close' and payload == ['bash', '-c', chain.CHAINED_CLOSE]
            return (1, 'ssh refused') if self.fail == 'chained-close' else (0, chain.CHAINED_CLOSE_OK + '\n')
        if step == 'close':
            tag, _ = rh.scoped_argv(argv[-1])
            if tag in self.fenced:  # a fenced scope's guard refuses with 75 (scoped_remote main)
                self.events.append(step)
                return 75, ''
        return super().plain(step, argv, timeout)


def name(i: int) -> str:
    return f'NBW2-W-20261027T21{i:02d}00Z'


def make(tmp_path: Path, out_name: str, *, chained: bool = False, fail: str = '', i_scale: float = .95,
         monkeypatch: pytest.MonkeyPatch, **fake: bool) -> tuple[w.Live, Remote, pf.Supervisor, list[Path]]:
    monitor = tmp_path / 'monitor.py'
    monitor.write_text(base.MONITOR)
    out = tmp_path / out_name
    out.mkdir()
    clock = base.Clock()
    trip = tmp_path / f'trip-{out_name}'
    sup = base.NoTunnel(out, clock=clock, monitor=lambda host: [sys.executable, str(monitor), 'low', str(trip)], demand=lambda: 0.0)
    remote = Remote(out, sup, clock, fail, trip, i_scale=i_scale, production_down=chained, **fake)
    aborts: list[Path] = []
    def abort(directory: Path) -> dict[str, Json]:
        aborts.append(directory)
        chain.clear()
        return {'status': 'RESTORED: chain ended by abort; watch PASS; marker cleared'}
    live = w.Live(out, sup, native=pure.NATIVE, remote='dsv41-deploy-offload', expected='0' * 40, data_dir=tmp_path,
                  tag='tag', chained=chained, call=remote.call, plain=remote.plain, clock=clock, chain_abort=abort)
    import offload.prefill_trace.window_requests as wr
    monkeypatch.setattr(wr, 'corpus', lambda root: ('public fake data', {}))
    monkeypatch.setattr(c, 'greedy_inputs', lambda: (['public prompt'] * 6, [{'i': i, 'text': 'public reference'} for i in range(6)]))
    return live, remote, sup, aborts


def run(live: w.Live, sup: pf.Supervisor, watch: str = 'PASS') -> tuple[dict[str, Json], int]:
    try:
        result = w.window(live, lambda row: None)
    finally:
        sup.close()
    code = w.after_window(result, lambda: {'status': watch})
    (live.out / 'summary.json').write_text(json.dumps({**result, 'production_commit': n.PROD_COMMIT}))
    return result, code


def measured() -> list[tuple[str, str]]:
    return [(a.name, cell) for a in w.ARMS for cell, reps in w.REPEATS[a.role].items() for _ in range(reps)]


def test_single_window_measures_repeats_decides_and_archives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake, sup, aborts = make(tmp_path, name(0), monkeypatch=monkeypatch)
    result, code = run(live, sup)
    assert 'stop' not in result, result
    assert code == 0 and result['status'] == 'COMPLETE' and result['chain'] == 'OFF' and not aborts
    rh.assert_restored(result, fake)
    assert fake.measured == measured()
    assert len(set(fake.salts)) == len(fake.salts) == 3 * 3 + 2 * 5, 'every repeat needs its own cache salt'
    assert fake.greedy_starts == ['s2-I', 's4-I'] and fake.kl_starts == ['s3-K', 's5-K']
    assert fake.events.index('greedy-s2-I') > fake.events.index('start-s2-I')
    assert all(record(record(result['starts'])[a.name])['clock_cap_mhz'] == CAP for a in w.ARMS)
    decision = record(result['decision'])
    assert decision['status'] == 'CANDIDATE' and decision['clock_cap_mhz'] == CAP and 'pooled_decision' not in result
    # The window directory is readable by WIN2's history tools: per-repeat ms/round from the aggregate files.
    extracted = win2_history.extract(live.out)
    assert sorted(extracted) == sorted(w.CELL_NAMES)
    assert len(extracted['short-b1']['control']['ms_per_round']) == 3
    assert len(extracted['short-b1']['candidate']['ms_per_round']) == 4
    assert {t for *_, t in win2_history.samples(live.out)} == {1500}
    archive = dv.archive_rows(w.ARCHIVE)
    head = record(result['archive'])
    assert len(archive) == 3 * 3 * len(w.ARCHIVED) == head['appended'] and head['skipped'] == []
    assert record(head['head_before'])['rows'] == 0 and record(head['head_after'])['rows'] == len(archive)
    for row in archive:
        dv.check_row(row)
        assert row['commit'] == n.PROD_COMMIT and row['clock_cap_mhz'] == CAP and row['clock_source'] == 'start-log'
        assert row['date'] == '20261027' and row['max_tokens'] == 1500
    assert {r['start'] for r in archive} == {'s1-K', 's3-K', 's5-K'}
    assert win2_history.backfill([live.out], w.ARCHIVE) == 0, 'WIN2 backfill adds nothing on top of the runner rows'


def test_mixed_clock_inside_a_window_is_inconclusive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake, sup, _ = make(tmp_path, name(0), fail='mixed-clock', monkeypatch=monkeypatch)
    result, code = run(live, sup)
    assert 'stop' not in result and code == 0
    assert record(record(result['starts'])['s4-I'])['clock_cap_mhz'] == 2150
    assert record(result['decision'])['status'] == 'INCONCLUSIVE'


def test_closing_k_skipped_for_time_restores_and_is_inconclusive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake, sup, _ = make(tmp_path, name(0), fail='optional-no-fit', monkeypatch=monkeypatch)
    result, code = run(live, sup)
    assert 'stop' not in result, result
    assert fake.started == ['s1-K', 's2-I', 's3-K', 's4-I']
    assert record(record(result['starts'])['s5-K'])['status'] == 'SKIPPED_TIME'
    rh.assert_restored(result, fake)
    assert record(result['decision'])['status'] == 'INCONCLUSIVE'


def test_chain_with_a_fixed_placeholder_check_leaves_production_down_and_pools(tmp_path: Path,
                                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    first, fake1, sup1, _ = make(tmp_path, name(0), i_scale=.97, monkeypatch=monkeypatch, provable=True)
    one, code1 = run(first, sup1)
    assert 'stop' not in one and code1 == 0, one
    assert one['chain'] == 'CHAINED' and str(record(one['restore'])['status']).startswith('CHAINED')
    assert fake1.restores == [], 'a chained window must not run the W8 restore'
    assert not fake1.hold, 'the owned HOLD is closed through the fenced commit'
    for step in ('pair-close', 'experiment-stop', 'jit-wait-sparkhead', 'chain-placeholder-proof', 'chained-close',
                 'commit-close', 'cancel-close'):
        assert step in fake1.events, step
    assert fake1.events.index('chain-placeholder-proof') < fake1.events.index('chained-close') < fake1.events.index('commit-close')
    for host in ('sparkhead', 'sparkworker-wired', 'x86host', 'localhost'):
        assert fake1.events.index(f'owned-absent-tag-restore-close-{host}') < fake1.events.index('commit-close')
    marker = chain.read_marker()
    assert marker is not None and marker['window'] == str(first.out)
    assert record(marker['production_snapshot'])['sha256'] == pf.digest(first.out / 'rollback.json')

    monkeypatch.setenv('WINDOW_CHAIN_LAST', '1')
    second, fake2, sup2, aborts = make(tmp_path, name(1), chained=True, i_scale=.97, monkeypatch=monkeypatch)
    two, code2 = run(second, sup2)
    assert 'stop' not in two and code2 == 0, two
    assert 'prod-health' not in fake2.commands and 'prod-down' in fake2.commands
    for label in ('chain-placeholder-sparkhead', 'chain-placeholder-sparkworker-wired', 'chain-quiet-sparkhead',
                  'chain-quiet-held-sparkhead', 'chain-quiet-held-sparkworker-wired'):
        assert label in fake2.commands, label
    assert fake2.commands.index('hold-read') < fake2.commands.index('chain-quiet-held-sparkhead') < fake2.commands.index('s1-K-start')
    rollback = json.loads((second.out / 'rollback.json').read_text())
    assert rollback['chained_from'] == str(first.out / 'rollback.json') and len(rollback['ranks']) == 2
    rh.assert_restored(two, fake2)
    assert two['chain'] == 'LAST' and not chain.MARKER.exists() and not aborts
    assert two['pooled_windows'] == [str(first.out)]
    pooled = record(two['pooled_decision'])
    assert pooled['windows'] == 2 and record(pooled['repeats_per_arm'])['short-b1'] == [6, 8]
    assert pooled['status'] == 'CANDIDATE'


def test_chain_at_r2_head_ends_with_the_strict_restore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R-NBW finding: chain.CHAINED_CLOSE's placeholder check runs inside a scoped_remote scope, whose tmux shim sends
    `tmux has-session -t dsv41-hold` to that scope's private socket, so it never sees the placeholders. The proof fails
    before the one -restore-close scope is used, and the window restores instead of leaving production down."""
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    live, fake, sup, aborts = make(tmp_path, name(0), monkeypatch=monkeypatch)
    result, code = run(live, sup)
    assert 'stop' not in result and code == 0, result
    assert 'ChainUnprovable' in str(result['chain_end']) and 'chained-close' not in fake.events
    rh.assert_restored(result, fake)
    assert result['chain'] == 'ENDED (strict restore)' and not chain.MARKER.exists() and not aborts
    assert record(result['decision'])['status'] == 'CANDIDATE'


@pytest.mark.parametrize('fail', ['greedy-divergence', 'signal'])
def test_a_stop_in_a_chained_window_runs_the_strict_restore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: str) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    live, fake, sup, _ = make(tmp_path, name(0), fail=fail, monkeypatch=monkeypatch, provable=True)
    result, code = run(live, sup)
    assert result.get('stop') and code == 1, result
    rh.assert_restored(result, fake)
    assert result['chain'] == 'ENDED (strict restore)' and not chain.MARKER.exists()
    assert 'chained-close' not in fake.events
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'


def test_failed_chained_close_falls_back_and_keeps_the_hold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    live, fake, sup, _ = make(tmp_path, name(0), fail='chained-close', monkeypatch=monkeypatch, provable=True)
    result, code = run(live, sup)
    assert 'chained HOLD close unrecorded' in str(result['chained_end_error'])
    # W8 brings production back, but the single owned close scope is fenced: HOLD stays for the coordinator.
    assert fake.restores == ['start', 'check'] and fake.events[-1] != 'commit-close'
    assert record(result['restore'])['status'] == 'FAILED' and fake.hold and code == 2
    assert not chain.MARKER.exists()
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'


def chained_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> w.Live:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    first, _, sup1, _ = make(tmp_path, name(0), monkeypatch=monkeypatch, provable=True)
    assert run(first, sup1)[0]['chain'] == 'CHAINED'
    return first


@pytest.mark.parametrize('fail,fake,message', [('chain-not-quiet', {}, 'not quiet'),
                                              ('', {'placeholders': False}, 'placeholders missing')])
def test_a_pre_hold_failure_in_the_chained_state_runs_the_chain_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                      fail: str, fake: dict[str, bool], message: str) -> None:
    chained_marker(tmp_path, monkeypatch)
    second, fake2, sup2, aborts = make(tmp_path, name(1), chained=True, fail=fail, monkeypatch=monkeypatch, **fake)
    result, code = run(second, sup2)
    assert message in str(result['stop'])
    assert 'hold-open' not in fake2.commands and aborts == [second.out / 'chain-abort']
    assert str(record(result['restore'])['status']).startswith('CHAIN ABORT: RESTORED') and result['chain'] == 'ABORT'
    assert code == 1 and not chain.MARKER.exists()


def test_a_chained_hold_admission_failure_takes_the_abort_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chained_marker(tmp_path, monkeypatch)
    second, fake2, sup2, aborts = make(tmp_path, name(1), chained=True, fail='hold-admission', monkeypatch=monkeypatch)
    result, code = run(second, sup2)
    assert 'HOLD not admitted' in str(result['stop']) and not fake2.started
    assert 'without owned HOLD' in str(record(result['restore_attempt'])['error'])
    assert aborts == [second.out / 'chain-abort'] and result['chain'] == 'ABORT' and code == 1


def test_watch_failure_keeps_the_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chained_marker(tmp_path, monkeypatch)
    monkeypatch.setenv('WINDOW_CHAIN_LAST', '1')
    second, fake2, sup2, aborts = make(tmp_path, name(1), chained=True, monkeypatch=monkeypatch)
    result, code = run(second, sup2, watch='FAIL')
    rh.assert_restored(result, fake2)
    assert code == 2 and chain.MARKER.exists(), 'a failed watch must leave the marker so chain_prev_ok aborts'
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE' and not aborts


def test_chained_window_refuses_without_chain_mode_or_with_changed_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = chained_marker(tmp_path, monkeypatch)
    monkeypatch.delenv('WINDOW_CHAIN')
    second, _, sup2, aborts = make(tmp_path, name(1), chained=True, monkeypatch=monkeypatch)
    result, _ = run(second, sup2)
    assert 'WINDOW_CHAIN=1' in str(result['stop']) and aborts and result['chain'] == 'ABORT'
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    (tmp_path / 'scratch').mkdir(exist_ok=True)
    rollback = first.out / 'rollback.json'
    chain.MARKER.write_text(json.dumps({'schema': chain.SCHEMA, 'window': str(first.out),
        'production_snapshot': {'path': str(rollback), 'sha256': pf.digest(rollback)}, 'quiet_proof': {'x': 1}}))
    data = json.loads(rollback.read_text())
    data['note'] = 'edited'
    rollback.write_text(json.dumps(data))
    third, _, sup3, _ = make(tmp_path, name(2), chained=True, monkeypatch=monkeypatch)
    assert 'missing or changed' in str(run(third, sup3)[0]['stop'])


def test_dry_plan_and_offline_pooling(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert w.main([]) == 0
    assert 'K I K I K' in capsys.readouterr().out
    for i, scale in enumerate((.96, .97)):
        d = tmp_path / f'W{i}'
        d.mkdir()
        (d / 'summary.json').write_text(json.dumps(summary(scale)))
    assert w.main(['--decide', str(tmp_path / 'W0'), str(tmp_path / 'W1')]) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'CANDIDATE'
    assert w.main(['--execute', '--out', str(tmp_path / 'x')]) == 1  # no coordinator: refused before any remote action


def test_kl_driver_records_estimator_refusals_instead_of_stopping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = tmp_path / 'greedy.json'
    rows: list[Json] = [{'i': i, 'identical': i not in (2, 3)} for i in range(6)]
    candidate.write_text(json.dumps({'checks': rows}))
    from offload.launch.windows import greedy_kl
    def compare(api: object, rows: list[dict[str, Json]], salt: str, bound: float) -> list[dict[str, Json]]:
        if rows[0]['i'] == 3:
            raise ValueError('nonfinite coarse KL at the divergence')
        return [{'i': r['i'], 'kl_nats': .001, 'pass': True} for r in rows if r.get('identical') is not True]
    monkeypatch.setattr(greedy_kl, 'compare_recorded', compare)
    result = g.kl(cast(c.Http, None), candidate, 'salt')
    assert result['divergences'] == 2
    assert cast(list[Json], result['rows']) == [{'i': 2, 'kl_nats': .001, 'pass': True},
                                                {'i': 3, 'error': 'ValueError: nonfinite coarse KL at the divergence'}]


def test_placeholder_proof_is_the_shared_close_without_its_unlink() -> None:
    proof = w.placeholder_proof()
    assert proof.startswith(chain.CHAINED_CLOSE.split('rm -f')[0]) and 'rm -f' not in proof
    assert proof.endswith('echo ' + w.PLACEHOLDERS_PROVEN)


def test_runner_override_hooks_keep_nbw_behaviour(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert n.Live.admit_unreachable is False  # NBW itself still requires a serving production to admit
    live, _, sup, _ = make(tmp_path, name(0), monkeypatch=monkeypatch)
    sup.close()
    assert live.pin_files == w.PIN_FILES and live.admit_unreachable is False
