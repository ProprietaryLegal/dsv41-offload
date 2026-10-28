#!/usr/bin/env python3
"""CPU-only parsers and owned-process supervisor for coordinator window scripts.

No remote access occurs except in explicit `supervise`, used only by --execute.
Raw evidence stays in the private campaign directory. Summaries contain aggregates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import statistics
import subprocess
import sys
import time
from typing import Any, IO, cast

MEM_FLOOR_KB = 5_242_880
PAIR = ('GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93',
        'GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc')
REQUIRED: dict[str, set[str]] = {
    'preflight': {'exclusive_window', 'production_drained', 'rollback_saved', 'd2_paused',
                  'source_pins', 'image_base_hashes', 'owned_processes'},
    'idle': {'experiment_idle', 'no_foreign_requests'},
    'runtime': {'both_rank_hashes', 'model_manifest', 'worker_identity', 'hello_identity',
                'rank_env_equal', 'capacity_async_process', 'proxy_torch_absent', 'proxy_mlock',
                'local_drafter_absent', 'engram_required', 'kv_layout', 'no_runtime_errors'},
    'clean': {'both_rank_logs', 'counters_before_after', 'no_fallback', 'no_watchdog',
              'no_nan', 'no_provenance_stops', 'no_swap_growth', 'healthy'},
    'worker': {'pair_owner_coord', 'owned_transient_only', 'both_gpu_locks', 'source_hashes',
               'arm_identity', 'ready_pid', 'four_slots', 'max_length', 'no_emulation', 'no_dumps'},
    'pair-results': {'fresh_results', 'both_arm_soaks', 'dump_identity', 'v1_classified',
                     'rx_gates', 'pair_timing', 'both_card_peaks', 'runtime_identity'},
    'recalibrate': {'pair_source_matches', 'verify_only_round_ms', 'overlap_audited',
                    'fit_heldout', 'config_staged', 'both_rank_costs'},
    'up1-prereq': {'k1_qualified', 's18_reviewed'},
    'up1-p0': {'control_k1_equivalent', 'cc_noise',
               'short_tokens_identical', 'request_accounting', 'round_within_one_percent'},
    'split-bitwise': {'rank0_rows_33_42_43_63', 'rank1_rows_33_42_43_63',
                      'per_row_oracle_43_63', 'dispatch_controls_32_64', 'no_host_count_sync'},
    'cpu-pin': {'actual_thread_affinities', 'allowed_cpuset', 'poller_verified', 'proxy_engram_tails'},
    'replay': {'successor_gt_128_both_sites', 'remainders_1_127_128_129', 'partial_budget_le_128',
               'retained_l20_kv', 'actual_suffix', 'partial_need_draft_zero', 'full_draft_window',
               'cancel_reuse', 'preempt_prefix', 'reconnect_timeout_resync', 'ragged_reject',
               'mm_exclusion', 'two_prompt_logprobs', 'final_logits_needles', 'fused_torch_equal',
               'zero_provenance_strings', 'zero_internal_error_increment'},
    'k5': {'five_proposals', 'twentyfour_target_rows', 'accepted_round_stats', 'prose_review_tool'},
    'dual-gid': {'rank0_both_ipv4_rocev2_gid3', 'rank1_both_ipv4_rocev2_gid3'},
    'dual-ring': {'collective_byte_identity', 'both_functions_traffic', 'no_nccl_fallback'},
    'vor': {'valid_lens_index_bitwise', 'padded_invalid_replay_300k', 'stock_exclusions',
            'no_exception_disable', 'eligible_once_per_step', 'copy_on_change', 'no_request_jit'},
    'rp1-preflight': {'k1_qualified', 'base20_both_images', 'checkpoint_manifest_both',
                      'cpu_topology_allowed', 'odirect', 'disk_budget_8g', 'cpu_bytes_256m'},
    'rp1-p0': {'no_connector_cli', 'no_rp1_switches', 'same_k1_pool_groups', 'same_fixed_outputs'},
    'rp1-rank': {'independent_preload_payloads', 'both_ranks_store', 'valid_ranges_recorded',
                 'rank_sha_crc_equal', 'gloo_init', 'cap_29568', 'cpu_bytes_256m'},
    'rp1-restart': {'stores_complete', 'old_engine_recorded', 'old_processes_gone',
                    'unused_shm_only', 'disk_preserved'},
    'rp1-empty': {'new_engine_empty', 'no_intervening_salts', 'new_engine_id'},
    'rp1-stale': {'new_rank1_owned_mmap', 'bounded_64k_sentinel', 'rank0_untouched',
                  'mn_verify_off', 'sync_before_gpu', 'selected_payload_equal', 'codes_logits_pass'},
    'rp1-crc': {'stopped_owned_namespace', 'aligned_odirect_flip', 'exact_crc_mismatch',
                'block_sidecar_removed', 'no_invalid_completion', 'recompute_pass'},
    'rp1-sidecar': {'stopped_owned_namespace', 'recorded_sidecar_only', 'miss_removed', 'recompute_pass'},
    'rp1-namespace': {'model_and_layout_mismatch', 'old_namespace_unread', 'cached_zero',
                     'true_identity_restored', 'no_false_namespace_writes'},
    'rp1-concurrency': {'four_distinct_150k_saved_salts', 'empty_restart', 'four_reload_lt_300s',
                       'each_load_le_29568', 'codes_six_of_six', 'no_foreign_codes',
                       'no_pending_loop', 'explicit_million_clamped', 'file_gloo_counters'},
    'rp1-reload-evidence': {'cold_fresh_cached_zero', 'reload_positive_le_cap',
                          'file_reads_and_gloo_increase', 'all_compare_pass', 'no_ambiguous_logits'},
    'depth': {'sp2_configs', 'matching_target_worker', 'max_depth_five', 'same_sampling_budget',
              'costs_measured', 'overlap_audited', 'heldout_groups', 'decisions_logged'},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def kv(text: str, expected: int) -> int:
    values = [int(x.replace(',', '')) for x in re.findall(r'GPU KV cache size: ([\d,]+)', text)]
    require(expected > 0 and bool(values), 'missing GPU KV cache size or invalid expectation')
    require(all(x == expected for x in values), f'KV token mismatch: {values}, expected {expected}')
    return values[0]


def memory(text: str) -> tuple[int, int]:
    fields = dict(re.findall(r'^(MemAvailable|SwapFree):\s*(\d+)\s+kB\s*$', text, re.M))
    require(set(fields) == {'MemAvailable', 'SwapFree'}, 'missing/malformed memory fields')
    available, swap = int(fields['MemAvailable']), int(fields['SwapFree'])
    require(available >= MEM_FLOOR_KB, f'MemAvailable {available} < {MEM_FLOOR_KB} kB')
    return available, swap


def dual(text: str, rank: int) -> None:
    prefix = f'EXL3 RDMA all-reduce: rank {rank} of 2 on rocep1s0f1,roceP2p1s0f1'
    require(any(line.startswith(prefix) and re.match(r'(?:,|$)', line[len(prefix):])
                for line in text.splitlines()), f'missing exact dual-ring rank {rank} log prefix')
    require(not re.search(r'RDMA all-reduce off.*NCCL stays', text), 'NCCL-only fallback')


def clean_logs(text: str) -> None:
    for stop in ('invalid replay tap provenance', 'replay skip would remove taps',
                 'RDMA all-reduce off', 'CUDA error', 'illegal memory access', 'Traceback (most recent call last)'):
        require(stop not in text, f'log stop: {stop}')


def counter_delta(before: dict[str, Any], after: dict[str, Any]) -> None:
    # Accept absent sparse zero counters, but refuse missing counter maps or a restarted/reset session.
    require(isinstance(before.get('counters'), dict) and isinstance(after.get('counters'), dict),
            'missing counters map')
    require(after.get('uptime_s', -1) >= before.get('uptime_s', 0), 'stats reset during arm')
    for key in set(before['counters']) | set(after['counters']):
        a, b = before['counters'].get(key, 0), after['counters'].get(key, 0)
        require(isinstance(a, (int, float)) and isinstance(b, (int, float)) and b >= a,
                f'counter reset/malformed: {key}')
        if any(s in key for s in ('fallback', 'failed', 'error', 'watchdog', 'preempted', 'skipped')):
            require(b == a, f'new failure counter: {key}: {a}->{b}')
    for stats in (before, after):
        if 'async_watchdog' in stats:
            require(stats['async_watchdog'].get('fires') == 0, 'async watchdog fired')


def receipt(path: Path, gate: str, arm: str, root: Path) -> None:
    require(gate in REQUIRED, f'unknown gate {gate}')
    data = load(path)
    require(data.get('schema') == 'wr1-gate-v1', 'wrong receipt schema')
    require(data.get('gate') == gate and data.get('arm') == arm, 'receipt gate/arm mismatch')
    require(data.get('run_dir') == str(root.resolve()), 'receipt belongs to another run')
    checks = data.get('checks', {})
    require(isinstance(checks, dict) and all(value is True for value in checks.values()), 'receipt contains a nonpassing check')
    require(all(checks.get(key) is True for key in REQUIRED[gate]),
            f'missing/failed {gate} checks: {sorted(key for key in REQUIRED[gate] if checks.get(key) is not True)}')
    require(bool(data.get('evidence')), 'receipt requires hashed local source evidence')
    for row in data['evidence']:
        source = Path(row['path']).resolve()
        require(source.is_relative_to(root.resolve()) and source != path.resolve(), 'evidence outside run or self-reference')
        require(source.is_file() and source.stat().st_mtime >= (root / '.run-start').stat().st_mtime - 2,
                f'missing/stale evidence {source}')
        require(hashlib.sha256(source.read_bytes()).hexdigest() == row['sha256'], 'evidence SHA256 mismatch')
    print(f'PASS {gate} {arm}: {len(REQUIRED[gate])} checks, source evidence hashes verified')


def timing(path: Path) -> None:
    data = load(path)
    require(data.get('classification') == 'MEASURED', 'timing must be MEASURED from pair + verify-only evidence')
    require(data.get('arm') == 'native-hybrid-kx2', 'wrong timing arm')
    require(data.get('overlap_audited') is True and data.get('verify_only') is True, 'unverified cost semantics')
    draft, rounds = data['draft_ms'], data['round_ms']
    require(isinstance(draft, (int, float)) and math.isfinite(draft) and draft >= 0, 'invalid draft_ms')
    require(isinstance(rounds, list) and len(rounds) == 6, 'round_ms must contain depths 0..5')
    require(all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in rounds), 'invalid round_ms')
    require(all(a <= b for a, b in zip(rounds, rounds[1:])), 'nonmonotonic verify-only costs')
    for row in data['sources']:
        source = Path(row['path'])
        require(hashlib.sha256(source.read_bytes()).hexdigest() == row['sha256'], 'timing source hash mismatch')
    require(bool(data['sources']), 'timing source evidence missing')
    print(f'MEASURED timing from {path}: DRAFT_MS={draft}, ROUND_MS={",".join(map(str, rounds))}')


def bench(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    runs = [r for r in records if r.get('type') == 'run']
    requests = [r for r in records if r.get('type') == 'request']
    require(bool(runs) and bool(requests), 'missing run/request measurements')
    for row in records:
        require(not row.get('error'), f'benchmark error in {path}')
    result: list[dict[str, Any]] = []
    for run in runs:
        rows = [r for r in requests if r.get('run_id') == run['run_id']]
        require(len(rows) == run['request_count'] and bool(rows), 'request accounting mismatch')
        spec = run['spec']
        require(spec.get('rounds', 0) > 0 and spec.get('mean_accepted_per_round') is not None, 'missing accepted/round')
        for row in rows:
            require(row.get('ttft_s') is not None and row.get('step_gap_ms', {}).get('p50') is not None,
                    'missing TTFT/step p50')
        require(not run.get('telemetry_errors'), 'benchmark telemetry error')
        result.append({'classification': 'MEASURED', 'source': str(path), 'label': run['config_label'],
                       'ctx': run['target_ctx'], 'concurrency': run['concurrency'], 'mixed': run['mixed'],
                       'cache': run['cache_state'], 'rep': run['rep'], 'delivered_tps': run['aggregate_tps'],
                       'accepted_per_round': spec['mean_accepted_per_round'],
                       'step_p50_ms': statistics.median(r['step_gap_ms']['p50'] for r in rows),
                       'ttft_s': statistics.median(r['ttft_s'] for r in rows),
                       'first_visible_s': [r.get('first_visible_s') for r in rows]})
    return result


def exit_status(proc: subprocess.Popen[Any]) -> int | None:
    # WNOWAIT reserves the leader PID even when it exited, so cleanup cannot signal a reused group.
    status = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if status is None:
        return None
    return status.si_status if status.si_code == os.CLD_EXITED else -status.si_status


def stop_owned(proc: subprocess.Popen[Any]) -> None:
    # The direct leader has not been reaped: its new session/group cannot belong to an unrelated process.
    if proc.returncode is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass  # Already gone. Still reap this directly owned child below.
        if sig == signal.SIGTERM:
            time.sleep(0.1)
    proc.wait(timeout=3)


def supervise(args: argparse.Namespace) -> int:
    root, name = args.out, args.name
    require(not (root / 'STOP').exists(), 'STOP file exists')
    select = selectors.DefaultSelector()
    monitors: list[subprocess.Popen[str]] = []
    outputs: list[IO[str]] = []
    command: subprocess.Popen[Any] | None = None
    baseline_path = root / 'swap-baseline.json'
    initial: dict[str, int] = load(baseline_path) if baseline_path.exists() else {}
    last: dict[str, float] = {}
    deadline = time.monotonic() + 20
    def interrupted(signum: int, _frame: Any) -> None:
        raise RuntimeError(f'supervisor interrupted by signal {signum}')
    old_handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2']
        memscript = "import time; from pathlib import Path\nwhile True:\n print(';'.join(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemAvailable:', 'SwapFree:'))), flush=True); time.sleep(1)"
        import shlex
        memcmd = 'exec python3 -u -c ' + shlex.quote(memscript)
        specs = [(host, memcmd, 'mem') for host in args.hosts]
        specs.append((args.x86host, f'nvidia-smi -i {",".join(PAIR)} --query-gpu=timestamp,uuid,memory.used,power.draw,temperature.gpu --format=csv,noheader -l 1', 'gpu'))
        for host, script, kind in specs:
            monitor_key = f'{kind}-{host}'
            log = (root / f'{name}.{monitor_key}.log').open('x')
            err = (root / f'{name}.{monitor_key}.stderr').open('x')
            outputs.extend([log, err])
            proc = subprocess.Popen([*ssh, host, script], stdout=subprocess.PIPE, stderr=err,
                                    text=True, start_new_session=True, bufsize=1)
            monitors.append(proc)
            assert proc.stdout is not None
            os.set_blocking(proc.stdout.fileno(), False)
            select.register(proc.stdout, selectors.EVENT_READ, (kind, host, log))
        pending = {key.fileobj: '' for key in select.get_map().values()}
        while True:
            now = time.monotonic()
            if (root / 'STOP').exists():
                raise RuntimeError('coordinator STOP file present')
            for key, _mask in select.select(timeout=0.1):
                stream = cast(IO[str], key.fileobj)
                chunk = os.read(stream.fileno(), 65536).decode()
                require(bool(chunk), 'telemetry stream closed')
                pending[stream] += chunk
                while '\n' in pending[stream]:
                    line, pending[stream] = pending[stream].split('\n', 1)
                    kind, host, log = key.data
                    log.write(f'{time.time():.3f} {line}\n'); log.flush()
                    if kind == 'mem':
                        available, swap = memory(line.replace(';', '\n'))
                        initial.setdefault(host, swap)
                        baseline_path.write_text(json.dumps(initial) + '\n')
                        require(swap >= initial[host], f'swap growth on {host}')
                        last[f'{kind}-{host}'] = now
                    else:
                        columns = next(csv.reader([line], skipinitialspace=True))
                        require(len(columns) == 5 and columns[1] in PAIR, 'invalid GPU telemetry/UUID')
                        require(float(columns[2]) >= 0, 'invalid GPU memory')
                        last[columns[1]] = now
            require(all(exit_status(p) is None for p in monitors), 'telemetry process exited')
            expected = [f'mem-{h}' for h in args.hosts] + list(PAIR)
            if command is None:
                require(now < deadline, 'no initial telemetry within 20 s')
                if all(k in last for k in expected):
                    cmd = args.command[1:] if args.command[0] == '--' else args.command
                    command = subprocess.Popen(cmd, start_new_session=True)
                    deadline = now + args.timeout
            else:
                require(now < deadline, f'step exceeded {args.timeout}s')
                require(all(now - last[k] < 15 for k in expected), 'telemetry stale >15s')
                rc = exit_status(command)
                if rc is not None:
                    require(rc == 0, f'command exited rc={rc}')
                    return 0
    except (Exception, KeyboardInterrupt) as exc:
        (root / 'STOP').write_text(f'{name}: {exc}\n')
        print(f'STOP {name}: {exc}', file=sys.stderr)
        return 1
    finally:
        if command is not None:
            stop_owned(command)
        for proc in monitors:
            stop_owned(proc)
        select.close()
        for output in outputs:
            output.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('kv'); p.add_argument('path', type=Path); p.add_argument('expected', type=int)
    p = sub.add_parser('dual'); p.add_argument('path', type=Path); p.add_argument('rank', type=int, choices=[0, 1])
    p = sub.add_parser('logs'); p.add_argument('path', type=Path)
    p = sub.add_parser('counters'); p.add_argument('before', type=Path); p.add_argument('after', type=Path)
    p = sub.add_parser('receipt'); p.add_argument('path', type=Path); p.add_argument('gate'); p.add_argument('arm'); p.add_argument('root', type=Path)
    p = sub.add_parser('timing'); p.add_argument('path', type=Path)
    p = sub.add_parser('rollback'); p.add_argument('path', type=Path)
    p = sub.add_parser('bench'); p.add_argument('path', type=Path)
    p = sub.add_parser('summary'); p.add_argument('root', type=Path); p.add_argument('--pair', action='store_true')
    p = sub.add_parser('supervise'); p.add_argument('--out', type=Path, required=True); p.add_argument('--name', required=True)
    p.add_argument('--hosts', nargs=2, required=True); p.add_argument('--x86host', required=True)
    p.add_argument('--timeout', type=float, default=10800); p.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == 'kv': print(f'MEASURED {args.path}: KV tokens {kv(args.path.read_text(), args.expected)}')
    elif args.mode == 'dual': dual(args.path.read_text(), args.rank)
    elif args.mode == 'logs': clean_logs(args.path.read_text())
    elif args.mode == 'counters': counter_delta(load(args.before), load(args.after))
    elif args.mode == 'receipt': receipt(args.path, args.gate, args.arm, args.root)
    elif args.mode == 'timing': timing(args.path)
    elif args.mode == 'rollback':
        text = args.path.read_text()
        require(not args.path.is_symlink() and text.startswith('#!/usr/bin/env bash\n'), 'saved coordinator-next.sh must be a local bash script')
        require(all(fragment in text for fragment in ('prod_window.sh close', 'PRODUCTION HOLD OK', './start.sh',
                                                     'window.sh close')), 'incomplete saved coordinator restoration commands')
        subprocess.run(['bash', '-n', str(args.path)], check=True)
        print(text, end='')
    elif args.mode == 'bench': print(json.dumps(bench(args.path), allow_nan=False))
    elif args.mode == 'summary':
        data = [r for path in sorted(args.root.glob('*-cold.jsonl')) + sorted(args.root.glob('*-warm.jsonl')) for r in bench(path)]
        require(bool(data) or args.pair, 'no benchmark summaries')
        (args.root / 'summary.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
        print(f'MEASURED: {len(data)} cells; sources in {args.root}/summary.json. Coordinator applies plan pass lines; no automatic promotion.')
    elif args.mode == 'supervise': return supervise(args)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f'STOP: {error}', file=sys.stderr)
        sys.exit(1)
