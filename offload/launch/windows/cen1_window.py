"""CEN1-WINDOW coordinator runner: distinct routed experts per verify call by live rows and depth, from one eager stats
start (documents/plans/lanes/CEN1-EXPERT-COUNT.md, CEN-RUN.plan.md). SINGLE window, never chained; dry plan by default.

One start: production verbatim + --enforce-eager + VLLM_EXL3_MOE_SMOE=0 / _STATS / DEPTH_POLICY / DEPTH_FILE + the 0047
adaptive window off (offload/launch/configs/CEN1-eager-stats.env). Function-only smoke first (fixed:5 -> decode
intervals at 6 live rows, fixed:0 -> 1; any miss STOPs), then cells short-b1 depths 0..5, 32k-b1 0..5, 32k-b4 0..4 with a
cut (both ranks' stats line counts) at every cell and depth change and >= NEED_LINES flush lines per depth. Counts only:
eager timing is never a speed. Stats files are copied with sha256 both sides and stay on Controller; after the window
offload.census.experts reads them with the cuts (cen1-experts.json). Shared lifecycle: cen_common (strict W8 restore).
"""
from __future__ import annotations

import json
import shlex
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

from offload.census import cen1_cells as cells
from offload.census import experts
from offload.launch.windows import cen_common as cc
from offload.launch.windows import nbw_window as n
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import up2_checks as ck
from offload.launch.windows.up2_low import X86HOST_PROBE
from offload.prefill_maxlen.evidence import Json, number, record

CONFIG = 'offload/launch/configs/CEN1-eager-stats.env'
ENV = {'VLLM_EXL3_MOE_SMOE': '0', 'VLLM_EXL3_MOE_STATS': '/cen1/moe-stats.jsonl', 'VLLM_DSPARK_DEPTH_POLICY': 'fixed:5',
       'VLLM_DSPARK_DEPTH_FILE': '/cen1/depth-policy', 'VLLM_DSV41_AK_MAX_REQS': '0'}
PIN_FILES = (*cc.SHARED_PINS, CONFIG, 'offload/launch/configs/NBW-native.env', 'offload/launch/windows/cen1_window.py')
RUN = 'dsv41-offload-run/cen1'
MOUNT = '/cen1'
STATS = 'moe-stats.jsonl'
NEED_LINES = 11                  # >= 10 whole flush intervals per depth on both ranks (coordinator rule: >= 10 flushes)
SMOKE_LINES = 4                  # three whole intervals after the cut: a prefill-tail interval stays a minority
DEPTH_CAP_S = 150.0              # measuring batches per depth stop here (a batch in flight completes)
SMOKE_CAP_S = 120.0
START_BUDGET_S = 15 * 60
FILE_CAP = 32 * 1024**2


def settle_batches(depth: int) -> int:
    """Batches after a depth-file write: the policy re-reads every 64 verify steps; a MAX_TOKENS stream at depth d takes
    at least MAX_TOKENS / (d + 1) steps (every draft accepted), so depths whose bound is under 80 steps settle twice."""
    return 1 if cells.MAX_TOKENS // (depth + 1) >= 80 else 2


def smoke_check(texts: list[str], from_lines: list[int], batch: int, depth: int) -> dict[str, Json]:
    """Whole decode intervals after the cut (calls of m = the verify window) on both ranks: none PURE at any row count
    but batch x (depth + 1), and the PURE ones at it carry at least half the window's calls. A request's prefill tail can
    share the window's m and leave its interval MIXED, so a minority of mixed calls is tolerated, never a wrong pure
    count. No interval: the decode path does not record (smoe still on) -> STOP. Wrong rows: the depth policy or file is
    not in force (or 0047 narrows the window) -> STOP."""
    window, expected = cells.window_rows(batch), cells.live_rows(batch, depth)
    out: dict[str, Json] = {'window_rows': window, 'expected_live_rows': expected}
    for rank, (text, start) in enumerate(zip(texts, from_lines)):
        ivs = [iv for iv in experts.intervals(experts.parse(text.splitlines()), start) if iv.m == window]
        ck.require(bool(ivs), f'smoke fixed:{depth}: no whole decode interval on rank {rank} after the cut (the decode '
                              'path does not record: VLLM_EXL3_MOE_SMOE=0 not in force?)')
        other = sorted({iv.pure_rows for iv in ivs if iv.pure_rows is not None and iv.pure_rows != expected})
        ck.require(not other, f'smoke fixed:{depth}: rank {rank} decode intervals pure at live rows {other}, expected '
                              f'{expected} (depth policy/file not in force, or 0047 still on)')
        good = sum(iv.calls for iv in ivs if iv.pure_rows == expected)
        total = sum(iv.calls for iv in ivs)
        ck.require(good > 0 and 2 * good >= total, f'smoke fixed:{depth}: rank {rank} only {good} of {total} decode '
                                                   f'calls pure at {expected} live rows')
        out[f'rank{rank}'] = {'intervals': len(ivs), 'calls': total, 'pure_calls': good, 'mixed_calls': total - good}
    return out


def mounts_check(mounts: list[dict[str, Json]], tag: str) -> dict[str, Json]:
    found = [m for m in mounts if m.get('Destination') == MOUNT]
    ck.require(len(found) == 1, f'expected exactly one {MOUNT} mount')
    m = found[0]
    ck.require(m.get('Type') == 'bind' and m.get('RW') is True and str(m.get('Source', '')).endswith(f'/{RUN}/{tag}'),
               f'{MOUNT} is not the read-write bind of ~/{RUN}/<tag>')
    return {'source_tail': f'{RUN}/{tag}', 'rw': True}


class CEN1Live(cc.CenLive):
    MARKER = 'CEN1-WINDOW'
    TIMING = cc.Timing(55 * 60, 58 * 60, 75 * 60)

    def __init__(self, out: Path, sup: pf.Supervisor, *, remote: str, expected: str, data_dir: Path, tag: str,
                 production_commit: str = n.PROD_COMMIT, production_tag: str = n.PROD_TAG,
                 call: pf.Call | None = None,
                 plain: Callable[[str, list[str], float], tuple[int, str]] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        super().__init__(out, sup, remote=remote, expected=expected, data_dir=data_dir, tag=tag,
                         production_commit=production_commit, production_tag=production_tag, call=call, plain=plain,
                         clock=clock)
        self.created: set[str] = set()
        self.collected = False
        self.serial = 0
        self.cuts_path = self.out / 'cuts.json'
        self.prompts = self.out / 'prompts.json'
        self.cut_rows: list[dict[str, Json]] = []

    @property
    def pin_files(self) -> tuple[str, ...]:
        return PIN_FILES

    @property
    def run(self) -> str:
        return f'{RUN}/{self.tag}'

    def run_dirs(self) -> tuple[str, ...]:
        return (self.run,)

    def hold_purpose(self) -> str:
        return f'{self.hold_marker()} eager distinct-expert count (counts only); no production change'

    # -- the start
    def start_cen1(self) -> list[dict[str, Json]]:
        for host in pf.HOSTS:
            self.created.add(host)        # BEFORE the command: a lost reply may still have created it
            text = self.ssh(f'cen1-dir-{host}', host, f'mkdir -p ~/{RUN} && mkdir ~/{self.run} && '
                            f"printf 'fixed:5\\n' > ~/{self.run}/depth-policy && cat ~/{self.run}/depth-policy")
            ck.require(text.strip() == 'fixed:5', f'{host} initial depth file not written')
        current = self.start_engine('cen1', f'CEN1_TAG={self.tag}', CONFIG, budget_s=START_BUDGET_S)
        for rank, host in enumerate(pf.HOSTS):
            cc.runtime_identity(self.production[rank], current[rank], env=ENV, eager=True)
            mounts_check(ck.rows(json.loads(self.ssh(f'cen1-mounts-{host}', host,
                                                     "docker inspect --format '{{json .Mounts}}' dsv41-serve"))), self.tag)
        self.accept_start('cen1', current)
        return current

    # -- requests and line counts
    def cen_driver(self, *extra: str) -> list[str]:
        return [sys.executable, '-m', 'offload.census.cen1_cells', '--execute', '--data-dir', str(self.data_dir), *extra]

    def build_prompts(self) -> dict[str, Json]:
        self.call('prompts-build', self.cen_driver('build', '--output', str(self.prompts)), timeout=600)
        return {k: record(v).get('tokens') for k, v in record(ck.read(self.prompts)['prompts']).items()}

    def lines(self, label: str) -> list[int]:
        counts: list[int] = []
        for host in pf.HOSTS:
            text = self.ssh(f'{label}-lines-{host}', host, f'f=~/{self.run}/{STATS}; '
                            'if [ -e "$f" ]; then wc -l < "$f"; else echo 0; fi')
            counts.append(int(text.split()[-1]))
        return counts

    def set_depth(self, label: str, depth: int) -> None:
        policy = f'fixed:{depth}'
        for host in pf.HOSTS:     # rank 0 reads and broadcasts; both ranks carry the same file
            d = f'~/{self.run}'
            text = self.ssh(f'{label}-depth-{host}', host, f"printf '%s\\n' {policy} > {d}/depth-policy.new && "
                            f'mv -f {d}/depth-policy.new {d}/depth-policy && cat {d}/depth-policy')
            ck.require(text.strip() == policy, f'{host} depth file readback differs from {policy}')

    def batch(self, cell: str, label: str) -> dict[str, Json]:
        self.serial += 1
        output = self.out / 'cells' / cell / f'{label}.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        size = len(cells.CELLS[cell])
        self.call(label, self.cen_driver('batch', '--cell', cell, '--cuts', str(self.prompts), '--seed', str(self.serial),
                                     '--salt-prefix', self.tag, '--deadline-utc', f'{self.submit:.3f}',
                                     '--output', str(output)), timeout=900, inflight=size)
        result = ck.read(output)
        ck.require(result.get('cell') == cell and len(cast(list[Json], result['streams'])) == size,
                   'batch artifact identity mismatch')
        return result

    def measure(self, cell: str, depth: int, *, need: int, cap_s: float, label: str) -> dict[str, Json]:
        """Write fixed:d, settle, cut, batches until `need` new lines on both ranks or the cap, cut again."""
        self.set_depth(label, depth)
        settle = [self.batch(cell, f'{label}-settle{i}') for i in range(settle_batches(depth))]
        start, began = self.lines(f'{label}-from'), self.clock()
        now, batches, status = start, 0, 'OK'
        while min(b - a for a, b in zip(start, now)) < need:
            if self.clock() - began >= cap_s:
                status = 'SHORT: per-depth cap reached'
                break
            if self.clock() + cells.BATCH_ESTIMATE_S > self.submit:
                status = 'SHORT: submission bound reached'
                break
            self.batch(cell, f'{label}-b{batches}')
            batches += 1
            now = self.lines(f'{label}-to')
        return {'cell': cell, 'depth': depth, 'policy': f'fixed:{depth}', 'settle_batches': len(settle),
                'batches': batches, 'from_lines': cast(Json, start), 'to_lines': cast(Json, now),
                'flush_lines': cast(Json, [b - a for a, b in zip(start, now)]), 'status': status}

    def read_stats(self, label: str) -> list[str]:
        return [self.ssh(f'{label}-read-{host}', host, f'cat ~/{self.run}/{STATS}') for host in pf.HOSTS]

    # -- phases
    def smoke(self) -> dict[str, Json]:
        out: dict[str, Json] = {}
        for depth in (5, 0):
            row = self.measure('short-b1', depth, need=SMOKE_LINES, cap_s=SMOKE_CAP_S, label=f'smoke-d{depth}')
            ck.require(max(cast(list[int], row['to_lines'])) > 0, 'smoke: no VLLM_EXL3_MOE_STATS line on either rank '
                       '(the decode path does not record or the stats file is not writable)')
            ck.require(row['status'] == 'OK', f'smoke fixed:{depth}: {row["status"]} before {SMOKE_LINES} new lines')
            check = smoke_check(self.read_stats(f'smoke-d{depth}'), cast(list[int], row['from_lines']), 1, depth)
            out[f'fixed:{depth}'] = {**row, 'check': check}
        (self.out / 'smoke.json').write_text(json.dumps(out, indent=1) + '\n')
        return out

    def cells(self) -> dict[str, Json]:
        rows: dict[str, Json] = {}
        for cell in cells.ORDER:
            for depth in cells.DEPTHS[cell]:
                name = f'{cell}-d{depth}'
                if self.clock() + (settle_batches(depth) + 1) * cells.BATCH_ESTIMATE_S > self.submit:
                    rows[name] = {'status': 'SKIPPED_TIME'}
                    continue
                row = self.measure(cell, depth, need=NEED_LINES, cap_s=DEPTH_CAP_S, label=name)
                rows[name] = row
                for rank in range(len(pf.HOSTS)):
                    self.cut_rows.append({'cell': name, 'requests': len(cells.CELLS[cell]), 'depth': depth, 'file': rank,
                                          'from_line': cast(list[int], row['from_lines'])[rank],
                                          'to_line': cast(list[int], row['to_lines'])[rank]})
                self.cuts_path.write_text(json.dumps({'type': 'cen1-cuts', 'cuts': cast(Json, self.cut_rows)},
                                                     indent=1) + '\n')
        return rows

    def collect_stats(self) -> list[dict[str, Json]]:
        manifest: list[dict[str, Json]] = []
        for rank, host in enumerate(pf.HOSTS):
            size = int(self.ssh(f'stats-size-{host}', host, f'stat -c %s ~/{self.run}/{STATS}').split()[-1])
            ck.require(0 < size <= FILE_CAP, f'{host} stats file size {size} outside (0, {FILE_CAP >> 20} MiB]')
            remote = self.json(f'stats-hash-{host}', host, f'cd ~/{self.run} && python3 - {STATS}', pf.IMAGE_BASES)
            local = self.out / 'stats' / host
            local.mkdir(parents=True)
            self.call(f'stats-copy-{host}', ['scp', '-q', '-o', 'BatchMode=yes', f'{host}:{self.run}/{STATS}', str(local)])
            got = pf.digest(local / STATS)
            ck.require(remote.get(STATS) == got and (local / STATS).stat().st_size == size, f'{host} stats copy differs')
            manifest.append({'rank': rank, 'host': host, 'file': str((local / STATS).relative_to(self.out)),
                             'bytes': size, 'sha256': got})
        (self.out / 'stats-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
        self.collected = True
        return manifest

    def finish_cen1(self, before: dict[str, Json]) -> dict[str, Json]:
        checks: dict[str, Json] = {}
        for rank, host in enumerate(pf.HOSTS):
            ident = self.ssh(f'final-id-{host}', host, "docker inspect --format '{{.Id}}' dsv41-serve").strip()
            ck.require(ident == self.ids[rank], 'experiment container changed during the window')
            checks[f'logs_rank{rank}'] = pf.logs(self.json(f'final-logs-{host}', host, 'python3 -c ' +
                shlex.quote(pf.LOG_FILTER) + ' ' + shlex.quote(ck.STOP_LOG.pattern)), rank, 327680)
        pf.worker(self.json('final-worker', 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)), self.clock())
        after = self.stats('final-stats')
        a, b = record(before['counters']), record(after['counters'])
        grew = sorted(k for k in ck.STOP_COUNTERS if number(b.get(k, 0)) != number(a.get(k, 0)))
        # Counts stay valid through a fallback round (it is a depth-0 round, off-depth in the curve); recorded, never a STOP.
        checks['proxy_stop_counters_changed'] = cast(Json, grew)
        checks['rounds_ok_delta'] = number(b.get('rounds_ok', 0)) - number(a.get('rounds_ok', 0))
        return checks

    def after_restore(self) -> dict[str, Json]:
        """Remove this window's run directories only when their stats were collected; otherwise they stay as evidence."""
        out: dict[str, Json] = {}
        for host in sorted(self.created):
            if not self.collected:
                out[f'run_dir_{host}'] = 'KEPT: stats not collected (evidence); remove by hand after inspection'
                continue
            d = f'~/{self.run}'
            rc, text = self.plain(f'cen1-dir-clean-{host}', [*pf.SSH, host, f'rm -f -- {d}/{STATS} {d}/depth-policy '
                                  f'{d}/depth-policy.new && rmdir {d} && echo CLEAN || ls -1 {d} | wc -l'], 60)
            out[f'run_dir_{host}'] = f'rc{rc}: {text.strip()[-80:]}'
        return out


def body(live: cc.CenLive) -> Callable[[dict[str, Json]], None]:
    w = cast(CEN1Live, live)

    def run(result: dict[str, Json]) -> None:
        w.start_cen1()
        result['clock'] = w.clock_cap('cen1')
        result['prompts'] = w.build_prompts()
        before = w.stats('cen1-stats-before')
        result['smoke'] = w.smoke()
        result['cells'] = w.cells()
        result['collected'] = cast(Json, w.collect_stats())
        result['finish'] = w.finish_cen1(before)
    return run


def analyze(out: Path) -> dict[str, Json]:
    """CPU on Controller after the window: offload.census.experts over both ranks' stats with the runner's cuts."""
    cuts = cast(list[Json], ck.read(out / 'cuts.json')['cuts'])
    manifest = [record(x) for x in cast(list[Json], json.loads((out / 'stats-manifest.json').read_text()))]
    paths = [out / str(row['file']) for row in sorted(manifest, key=lambda r: number(r['rank']))]
    for row, path in zip(sorted(manifest, key=lambda r: number(r['rank'])), paths):
        ck.require(pf.digest(path) == row['sha256'], f'{path.name}: sha256 differs from the manifest')
    result = experts.analyze(paths, [record(c) for c in cuts])
    (out / 'cen1-experts.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    return {'file': 'cen1-experts.json', 'status': result['status'], 'cross_rank': result.get('cross_rank')}


def plan() -> str:
    lines = ['DRY RUN: no command, network or file. Coordinator Deepseek/coordinator only (--execute / --precheck).',
             'SINGLE window, never chained: WINDOW_CHAIN / WINDOW_CHAIN_LAST unset and no chain marker, else STOP.',
             'Preflight (read-only): W8 restore pin; D2 pause; corpus; production ledger (--production-commit/--production-tag,',
             '  default NBW 596ba1c / prod-k1-327-ops1-20261026), clean, no HOLD; experiment checkout = reviewed commit;',
             f'  {len(PIN_FILES)} staged pins incl. {CONFIG}; :8000 K1 327680; x86host worker probe; both Sparks kernel/driver +',
             f'  MemAvailable; ~/{RUN}/<tag> absent and >= 10 GiB free home disk on both Sparks.',
             f'HOLD: owned admission "CEN1-WINDOW <tag>:" (drain); deadlines from HOLD: submit +{CEN1Live.TIMING.submit / 60:.0f},'
             f' abort +{CEN1Live.TIMING.abort / 60:.0f}, restore by +{CEN1Live.TIMING.restore / 60:.0f} min (placeholder lease +90).',
             'Snapshot production both ranks -> rollback.json; tunnel 127.0.0.1:18001 -> sparkhead :8001.',
             f'Start: mkdir ~/{RUN}/<tag> + depth-policy fixed:5 on both Sparks; CEN1_TAG=<tag> start_exp.sh {CONFIG}',
             '  (production verbatim + --enforce-eager + ' + ' '.join(f'{k}={v}' for k, v in ENV.items()) + ');',
             '  identity == production except port/eager/those env keys; /cen1 rw bind; pins, logs, proxy, worker; clock cap',
             '  from the start log (2300 registered; any other value -> INCONCLUSIVE, never a STOP).',
             'Prompts: window_cells cuts (7 public prompts, numbers + hashes only).',
             f'Smoke (function only): fixed:5 then fixed:0 at short-b1: settle, cut, batches until {SMOKE_LINES} new lines on',
             '  both ranks; no whole decode interval pure at other than 6 / 1 live rows and the pure ones carry >= half the',
             '  decode calls (a prefill tail may mix one interval), else STOP.',
             f'Cells ({cells.MAX_TOKENS}-token ignore_eos streams, fresh seeds, one cache salt per cell): '
             + '; '.join(f'{c} depths {list(cells.DEPTHS[c])}' for c in cells.ORDER) + '.',
             f'  Per depth: write fixed:d both Sparks (read back), {settle_batches(0)}-{settle_batches(5)} settle batch(es), cut,',
             f'  batches until >= {NEED_LINES} new flush lines on both ranks or {DEPTH_CAP_S:.0f} s; cut. Depths that cannot fit before',
             '  the submission bound are SKIPPED_TIME (32k-b4 last).',
             f'Collect: both stats files, sha256 both sides (<= {FILE_CAP >> 20} MiB); proxy STOP counters recorded; containers',
             '  unchanged, clean logs, worker probe.',
             'ALWAYS: fence every owned phase, experiment stop, quiet proof, W8 start/check/close (six IDENTICAL, PRODUCTION HOLD',
             '  OK), owned HOLD close, 15-minute post-close watch; run directories removed only when collected.',
             'After (CPU, Controller): offload.census.experts -> cen1-experts.json (counts only; never published).',
             'ESTIMATED HOLD ~50 min (start ~8, prompts 1, smoke ~3, cells ~25, W8 ~10); production down ~47 min.']
    return '\n'.join(lines)


ENTRY = cc.Entry(
    description=__doc__ or 'CEN1 window',
    plan=plan,
    make=lambda out, sup, args, tag: CEN1Live(out, sup, remote=args.remote_repo,
                                              expected=args.expected_experiment_commit, data_dir=args.data_dir, tag=tag,
                                              production_commit=args.production_commit,
                                              production_tag=args.production_tag),
    body=body, config=CONFIG, analyze=analyze)


def main(argv: list[str] | None = None) -> int:
    return cc.main(ENTRY, argv)


if __name__ == '__main__':
    raise SystemExit(main())
