"""R-UPF identity gates for the UP3 runner (coordinator 2026-10-27), run as a subprocess over the experiment tunnel like
offload.up3.cells. Every gate is the shared function R-UPF's WINDOW.md names (imported, never copied); this module only
calls it and writes one JSON result. Dry by default; coordinator only.

Modes (registration upf-0p9: NDM's 0.9-nat greedy screen, offload/upfix/configs.GREEDY_NEAR_TIE_NATS):
  greedy  six K1 reference prompts, nbw_cells.greedy at 0.9+1e-3 nats, every divergence's top-20 and prefix recorded
  kl      greedy_kl.compare_recorded over ONE candidate start's recorded divergences, at a registered bound
  needle  up2_request.needle at 150,000 or 300,000 tokens (recall 6/6, no foreign codes, cached 0, finish stop);
          a gate failure is written as {"pass": false}, any other failure exits non-zero
  soak    WINDOW.md soak: four concurrent `short` streams (512 tokens, fresh salt per request) re-issued for 600 s while
          `docker exec dsv41-serve ps -eo pid,rss,args` on both Sparks and `df -B1 /dev/shm` on the head are sampled every
          10 s (read-only ssh); growth per role = median of the last three samples - median of the first three (MiB)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import greedy_kl as gk
from offload.launch.windows import nbw_cells as greedy_driver
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import up2_checks as ck
from offload.launch.windows import up2_request
from offload.prefill_maxlen.evidence import Json, number, record
from offload.prefill_trace.window_requests import corpus, TEMPLATE
from offload.up3 import cells
from offload.upfix import configs as uc
from offload.upfix import windows as uw

REGISTRATION = 'upf-0p9'
MODES = ('greedy', 'kl', 'needle', 'soak')
NEEDLE_TARGETS = (150000, 300000)
SOAK_STREAMS = 4
SOAK_INTERVAL_S = 10.0
SOAK_TOKENS = 512
SHM_ROLE = f'{pf.HOSTS[0]}:/dev/shm'
# up2_request.needle's own gate refusals (recall/foreign codes, length/cache, finish). Anything else (deadline, idle,
# transport) is not a verdict on the arm and fails the call.
NEEDLE_GATE = ('needle6/6/foreign-code failure', 'needle length/cache failure', 'needle did not complete normally')
Sample = Callable[[], dict[str, float]]


def ssh_text(argv: Sequence[str]) -> str:
    return subprocess.run(list(argv), check=True, capture_output=True, text=True, timeout=30).stdout


def process_roles(host: str, text: str) -> dict[str, float]:
    """RSS (MiB) per process role of one Spark's serving container: '<host>:<args>', '#k' for the k-th process (by pid)
    with the same args. The sampling `ps` itself is no role."""
    rows: dict[str, list[tuple[int, float]]] = {}
    for line in text.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, rss, args = parts
        ck.require(pid.isdigit() and rss.isdigit(), f'{host}: malformed ps row {line!r}')
        if args.startswith('ps '):
            continue
        rows.setdefault(' '.join(args.split()), []).append((int(pid), int(rss) / 1024))
    ck.require(bool(rows), f'{host}: no serving-container process sampled')
    out: dict[str, float] = {}
    for args, found in rows.items():
        for k, (_, mib) in enumerate(sorted(found)):
            out[f'{host}:{args}' + (f'#{k}' if len(found) > 1 else '')] = mib
    return out


def remote_sample(run: Callable[[Sequence[str]], str] = ssh_text) -> dict[str, float]:
    sample: dict[str, float] = {}
    for host in pf.HOSTS:
        sample.update(process_roles(host, run([*pf.SSH, host, 'docker exec dsv41-serve ps -eo pid=,rss=,args='])))
    used = run([*pf.SSH, pf.HOSTS[0], 'df -B1 --output=used /dev/shm']).split()
    ck.require(len(used) == 2 and used[1].isdigit(), 'df /dev/shm unreadable')
    sample[SHM_ROLE] = int(used[1]) / 2**20
    return sample


def growth(samples: Sequence[dict[str, float]]) -> tuple[dict[str, float], list[str]]:
    """(growth MiB per role present in every sample, transient roles). Needs >= 6 samples."""
    ck.require(len(samples) >= 6, f'soak took {len(samples)} samples; at least 6 required')
    roles = set(samples[0]).intersection(*samples[1:])
    transient = sorted(set().union(*samples) - roles)
    return ({r: statistics.median(s[r] for s in samples[-3:]) - statistics.median(s[r] for s in samples[:3])
             for r in sorted(roles)}, transient)


def soak(api: wc.Api, prompt: tuple[str, int], *, seconds: float, salt: str, sample: Sample,
         interval: float = SOAK_INTERVAL_S, streams: int = SOAK_STREAMS,
         clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> dict[str, Json]:
    text, n = prompt
    end = clock() + seconds
    lock = threading.Lock()
    done: list[int] = [0]
    errors: list[str] = []

    def stream(i: int) -> None:
        k = 0
        try:
            while clock() < end and not errors:
                row = api.stream({'model': api.model()['id'], 'messages': [{'role': 'user', 'content': text}],
                    'temperature': 1, 'top_p': 1, 'seed': 42 + k, 'priority': 10, 'max_tokens': SOAK_TOKENS,
                    'cache_salt': f'{salt}:soak:{i}:{k}', 'chat_template_kwargs': TEMPLATE, 'stream': True},
                    threading.Event())
                usage = record(row['usage'])
                ck.require(usage.get('completion_tokens') == SOAK_TOKENS and row.get('finish') == 'length'
                           and usage.get('prompt_tokens') == n
                           and record(usage.get('prompt_tokens_details') or {}).get('cached_tokens') == 0,
                           'soak request incomplete or cached')
                k += 1
                with lock:
                    done[0] += 1
        except Exception as exc:   # surfaced below; the sampler stops at the next tick
            errors.append(f'stream {i}: {type(exc).__name__}: {exc}')

    threads = [threading.Thread(target=stream, args=(i,), daemon=True) for i in range(streams)]
    for thread in threads:
        thread.start()
    samples: list[dict[str, float]] = []
    times: list[float] = []
    started = clock()
    try:
        while not errors:
            samples.append(sample())
            times.append(clock() - started)
            if clock() >= end:
                break
            sleep(max(0.0, min(interval, end - clock())))
    finally:
        for thread in threads:
            thread.join(timeout=900)
    ck.require(not errors, 'soak load failed: ' + '; '.join(errors))
    ck.require(not any(t.is_alive() for t in threads), 'soak stream did not finish')
    grown, transient = growth(samples)
    return {'seconds': seconds, 'streams': streams, 'interval_s': interval, 'max_tokens': SOAK_TOKENS,
            'requests': done[0], 'samples': cast(Json, samples), 'sample_times_s': cast(Json, times),
            'growth_mib': cast(Json, grown), 'transient_roles': cast(Json, transient), 'registration': REGISTRATION}


def run(mode: str, api: cells.Http, args: argparse.Namespace, sample: Sample = remote_sample) -> dict[str, Json]:
    output, salt, deadline = cast(Path, args.output), str(args.salt), float(args.deadline)
    ck.require(bool(salt.strip()), 'a cache salt prefix is required')
    if mode == 'greedy':
        result = greedy_driver.greedy(api, deadline, *greedy_driver.greedy_inputs(), output,
                                      near_tie_nats=uc.GREEDY_NEAR_TIE_NATS, record_distribution=True,
                                      probe_salt_prefix=salt)
        result.update(near_tie_nats=uc.GREEDY_NEAR_TIE_NATS, greedy_registration=REGISTRATION)
        return result
    if mode == 'kl':
        rows = ck.rows(ck.read(cast(Path, args.candidate))['checks'])
        bound = float(args.bound)
        ck.require(bound >= uw.KL_BOUND_FLOOR, 'KL bound below the registered floor')
        compared = gk.compare_recorded(api, rows, salt, bound)
        return {'rows': cast(Json, compared), 'bound': bound, 'compared': len(compared),
                'pass': all(r['pass'] is True for r in compared), 'registration': REGISTRATION}
    if mode == 'needle':
        target = int(args.target)
        recorded = output.with_name(output.stem + '-needle.json')
        try:
            up2_request.needle(target, salt, recorded, cast(Path, args.ledger), deadline, cast(Path, args.data_dir))
        except ValueError as exc:
            if not any(gate in str(exc) for gate in NEEDLE_GATE):
                raise
            return {'target': target, 'pass': False, 'error': f'{type(exc).__name__}: {exc}'}   # an arm-level STOP
        found = ck.read(recorded)
        return {'target': target, 'pass': number(found.get('recall', 0)) == 6 and not found.get('foreign_codes'),
                **{k: found.get(k) for k in ('recall', 'of', 'foreign_codes', 'context')}}
    ck.require(mode == 'soak', f'unknown mode {mode}')
    materials, hashes = corpus(cast(Path, args.data_dir))
    prompt = cells.texts(ck.read(cast(Path, args.cuts)), materials, hashes)['short']
    ck.require(time.time() + float(args.seconds) < deadline, 'soak cannot fit the submission deadline')
    return soak(api, prompt, seconds=float(args.seconds), salt=salt, sample=sample)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--salt', required=True)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--bound', type=float)
    parser.add_argument('--target', type=int, choices=NEEDLE_TARGETS)
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--cuts', type=Path)
    parser.add_argument('--seconds', type=float)
    args = parser.parse_args()
    needs = {'kl': ('candidate', 'bound'), 'needle': ('target', 'ledger'), 'soak': ('cuts', 'seconds')}
    missing = [n for n in needs.get(args.mode, ()) if getattr(args, n) is None]
    ck.require(not missing, f'{args.mode}: missing {missing}')
    ck.require(args.seconds is None or 60 <= args.seconds <= uw.SOAK_S, 'soak seconds 60..600')
    if not args.execute:
        print(f'DRY: upf {args.mode} ({REGISTRATION}); no network or file')
        return 0
    ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'coordinator required')
    ck.require(not args.output.exists() and time.time() < args.deadline, 'fresh output and live deadline required')
    result = run(args.mode, cells.Http(wc.URL), args)
    args.output.write_text(json.dumps(result, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
