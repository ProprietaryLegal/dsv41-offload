"""Emit reviewed NBW prerequisite scripts. This CLI never stages, connects to a host or starts a GPU job."""
from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from offload.launch.windows import nbw_window as n

NATIVE_COMMIT = 'dc6de03a8601acba039e972e69c8cee54560f4d7'
NATIVE_TREE = 'dsv41-offload/nbw-source/repo'
SOLO_GPU = 'GPU-bc62df2a-59cf-9828-7d43-558d1355bab7'
# S49 hashes, independently present on X86host. Stage reads these; never builds or changes shared files.
BINARIES = {
    'lib/libofl.so': '9f11b6e8d4f459c07ac35745eec95e5e54fcf705d5085e454faab7386520356d',
    'build/kx_fp8dense/kx_fp8dense.so': 'c8f9a11ad699c2ea0e6d2216ed7e2b97387a6577572b8f7893239de19ef93c52',
    'build/kx_mxfp4_regfed/kx_mxfp4_regfed.so': 'fe84ed015ebe6677bdeeeee3ad587afaab5deb7e6f50831d7816d040748f159d',
    'build/kx_fusedmoe/kx_fusedmoe.so': '60f1b82827fef1aa4c30666e5f02e3937058958a5de84dc7c524908a15d5a079',
    'p2p/08e1903b0a43feff/dspark_p2p.so': 'a5ac180b2e9b7b8f93f0dd4baefc57c2a04daebb2636f1690275870258a8b6bc',
}


def hash_check(relative: str, digest: str) -> str:
    return f'printf \'%s  %s\\n\' {shlex.quote(digest)} "$HOME/{relative}" | sha256sum -c -'


def stage_script() -> str:
    """Run ON X86host by the coordinator before HOLD, with a previously copied full source bundle as $1.

    Original W8 tune bytes/hash were not recovered. Require explicit review of the E/TP2 alternative;
    copy identical pinned bytes to NBW's new private directory, leaving E/production/N4 files alone.
    """
    lines = ['#!/usr/bin/env bash', 'set -euo pipefail', 'source ~/.config/ndm/paths.sh',
             'test "${WINDOW_COORDINATOR:-}" = Deepseek/coordinator',
             'test "${NBW_REVIEWED_CACHE_ALTERNATIVE:-}" = E-TP2-937e9633 || { '
             'echo "STOP: review the missing-W8-cache alternative in NBW-WINDOW.md first" >&2; exit 2; }',
             'bundle=${1:?absolute source bundle already copied to X86host}',
             '[[ $bundle == /* ]] && test -f "$bundle"',
             f'test ! -e "$HOME/{Path(NATIVE_TREE).parent}" && test ! -L "$HOME/{Path(NATIVE_TREE).parent}"',
             hash_check('dsv41-offload/e/run/exl3_tune_cache.bin', n.TUNE_SHA256),
             hash_check('dsv41-offload/prod/run/exl3_tune_cache_tp2.bin', n.TUNE_SHA256)]
    lines.extend(hash_check('dsv41-offload/n4/' + name, digest) for name, digest in BINARIES.items())
    lines += ['test -d "$HOME/dsv41-offload/n4/triton"', 'test -x "$HOME/dsv41-offload/e/venv/bin/python"',
              f'mkdir "$HOME/{Path(NATIVE_TREE).parent}"',
              f'git clone --no-checkout -- "$bundle" "$HOME/{NATIVE_TREE}"',
              f'cd "$HOME/{NATIVE_TREE}"', f'git checkout --detach {NATIVE_COMMIT}',
              f'test "$(git rev-parse HEAD)" = {NATIVE_COMMIT}',
              'test -z "$(git status --porcelain --untracked-files=no)"',
              f'printf \'%s\\n\' {NATIVE_COMMIT} > SOURCE_COMMIT',
              f'cp -- "$HOME/dsv41-offload/e/run/exl3_tune_cache.bin" "$HOME/{n.TUNE_CACHE}"',
              hash_check(n.TUNE_CACHE, n.TUNE_SHA256),
              f'printf \'%s\\n\' "SOURCE_COMMIT={NATIVE_COMMIT}" "TUNE_SHA256={n.TUNE_SHA256}" '
              '"TUNE_ORIGIN=E runtime; byte-identical to production TP2 cache; W8 provenance unavailable" '
              '> ../PREREQUISITES.txt',
              'echo "NBW SOURCE/CACHE STAGED; solo identity and every Triton key still need attended qualification"']
    return '\n'.join(lines) + '\n'


def solo_bootstrap() -> str:
    end = "runpy.run_module('offload.dspark.worker.server', run_name='__main__')"
    assert n.NATIVE_BOOTSTRAP.count(end) == 1
    solo = f'''from offload.dspark.worker.server import build_core
import hashlib, json
cache = Path.home() / {n.TUNE_CACHE!r}
if hashlib.sha256(cache.read_bytes()).hexdigest() != {n.TUNE_SHA256!r}:
    raise RuntimeError('NBW tune cache changed before solo startup')
core, model = build_core('/opt/dsv41/x86host-data/dspark', graphs=True, tp=2, max_model_len=1048576)
s = core.stats()
if hashlib.sha256(cache.read_bytes()).hexdigest() != {n.TUNE_SHA256!r}:
    raise RuntimeError('NBW tune cache changed during solo startup')
print('NBW_SOLO_RECEIPT ' + json.dumps({{'native_tree':{NATIVE_TREE!r},
 'native_commit':os.environ['DSPARK_SOURCE_COMMIT'], 'native_identity':s['drafter_identity'],
 'native_markov_identity':s['markov_identity'], 'solo_start':'PASS',
 'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'], 'batch_invariant':os.environ['DSPARK_NATIVE_BATCH_INVARIANT'],
 'tune_cache':{n.TUNE_CACHE!r}, 'tune_sha256':{n.TUNE_SHA256!r}}}), flush=True)
'''
    return n.NATIVE_BOOTSTRAP.replace(end, solo)


def solo_script(gpu: str) -> str:
    # Frozen dc6de03's check_visible only admits bc62df2a for a non-pair solo card.
    # Never bypass that guard or select the production pair to dodge a busy solo.
    if gpu != SOLO_GPU:
        raise ValueError('frozen dc6de03 supports only solo bc62df2a; wait for its owner to release it; '
                         'another solo UUID needs a separately reviewed source guard change')
    return '\n'.join([
        '#!/usr/bin/env bash', 'set -euo pipefail', 'source ~/.config/ndm/paths.sh',
        'test "${WINDOW_COORDINATOR:-}" = Deepseek/coordinator',
        f'cd "$HOME/{NATIVE_TREE}"', f'test "$(git rev-parse HEAD)" = {NATIVE_COMMIT}',
        f'test "$(cat SOURCE_COMMIT)" = {NATIVE_COMMIT}',
        'test -z "$(git status --porcelain --untracked-files=no)"',
        hash_check(n.TUNE_CACHE, n.TUNE_SHA256),
        # The actual descriptor must survive into Python (check_visible -> require_lock checks fd9).
        'exec 9>> "$HOME/dsv41-offload/locks/gpu-bc62df2a.lock"',
        'flock -n 9 || { echo "STOP: selected solo GPU is owned/busy; no wait, no startup" >&2; exit 73; }',
        f'export CUDA_VISIBLE_DEVICES={gpu}',
        'apps=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)',
        'test -z "$apps" || { echo "STOP: selected solo GPU has compute processes" >&2; exit 74; }',
        'export KX_FP8DENSE_LOCK_FD=9 KX_FP8DENSE_LOCKED=1 KX_FUSEDMOE_LOCKED=1',
        'export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_CACHE_DISABLE=1 OMP_NUM_THREADS=4',
        'export DSPARK_WORKER_ARM=native-hybrid-kx2 DSPARK_NATIVE_BATCH_INVARIANT=2 DSPARK_WORKER_TP_EMULATE=1',
        f'export DSPARK_SOURCE_COMMIT={NATIVE_COMMIT}',
        'export OFL_PY_DIR=$HOME/dsv41-offload/n4/lib OFL_LIB=$HOME/dsv41-offload/n4/lib/libofl.so',
        'export DSPARK_P2P_PREBUILT=$HOME/dsv41-offload/n4/p2p',
        f'export EXLLAMAV3_TUNE_CACHE=$HOME/{n.TUNE_CACHE}',
        'unset DSPARK_V1_DUMP_DIR DSPARK_D2_ADAPTER',
        # No listener, server main or request; synchronous attended process, exact prebuilt-only bootstrap.
        'exec timeout --signal=TERM --kill-after=15s 900 nice -n 19 '
        f'"$HOME/dsv41-offload/e/venv/bin/python" -u -c {shlex.quote(solo_bootstrap())}',
    ]) + '\n'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit', choices=('stage', 'solo', 'bootstrap'))
    parser.add_argument('--solo-gpu', default='', help='explicit coordinator-selected solo UUID; no default allocation')
    args = parser.parse_args(argv)
    if args.emit == 'stage':
        print(stage_script(), end='')
    elif args.emit == 'solo':
        try:
            print(solo_script(args.solo_gpu), end='')
        except ValueError as exc:
            parser.error(str(exc))
    elif args.emit == 'bootstrap':
        print(solo_bootstrap(), end='')
    else:
        print('DRY PLAN: emit scripts only; no host, GPU or file action. Coordinator only, before HOLD.\n'
              '1. Review the explicit E/TP2 cache alternative; original W8 cache bytes/hash unavailable.\n'
              '2. Copy a reviewed full source bundle to X86host; --emit stage -> attended X86host bash with bundle argument.\n'
              '3. Coordinator selects/releases a supported solo GPU (PP3 holds bc62df2a); --emit solo --solo-gpu UUID.\n'
              '4. Review complete zero-exit solo log and fresh identities before the window. No automatic promotion.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
