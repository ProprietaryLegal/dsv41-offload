"""Written-only X86host diagnostic. Preview by default; never contacts a Spark.

The outer coordinator process fences/proves its X86host-local shared scope absent
before separately scoped pair restoration, including a lost pair-open reply.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from offload.launch.windows import scoped_remote as owner
from offload.launch.windows.pf1_window import WORKER_GPUS

PYTHON = '/home/operator/miniconda3/bin/python3'
BASELINE = '1b8122992fb440edc668ee9e628ee9624ae2faf2'
SCRATCH = Path('/opt/dsv41/x86host-data/pp4')
STARTUP_S = 600.0  # existing whole-K2 qualification bound, not a new measurement
REQUEST_S = 30.0   # existing status freshness window; no request setup exemption
PROBE_S = 3600.0
RESTORE_S = 720.0
SCHEMA = 'pp4-x86host-server-probe-v1'
FILES = ('stock', 'facc', 'p2p', 'ofl', 'ofl_python', 'tune')
DRIVER = Path(__file__).resolve()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError('STOP: ' + message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text())
    require(isinstance(value, dict), 'JSON object required: ' + str(path))
    return cast(dict[str, Any], value)


def write(path: Path, data: object) -> None:
    temp = path.with_suffix(path.suffix + '.new')
    temp.write_text(json.dumps(data, allow_nan=False, sort_keys=True) + '\n')
    temp.replace(path)


def inside(path: Path) -> Path:
    require(path.is_absolute() and path.resolve().is_relative_to(SCRATCH), 'path outside X86host PP4 scratch')
    return path.resolve()


def git(tree: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(tree), *args], text=True, timeout=15).strip()


def check_memory(path: Path = Path('/proc/meminfo')) -> None:
    keys = {'MemAvailable', 'SwapTotal', 'SwapFree'}
    mem = {line.split(':', 1)[0]: int(line.split()[1]) for line in path.read_text().splitlines()
           if line.split(':', 1)[0] in keys}
    require(set(mem) == keys and mem['MemAvailable'] >= 48*1024*1024 and mem['SwapTotal'] == mem['SwapFree'],
            '48GiB available RAM and zero used swap required before opening pair')


@dataclass(frozen=True)
class Config:
    baseline: Path
    candidate: Path
    commit: str
    runtime: Path
    pair_receipt: Path
    out: Path
    run: str
    port: int
    weights: Path

    def argv(self) -> list[str]:
        return ['--baseline', str(self.baseline), '--candidate', str(self.candidate), '--commit', self.commit,
                '--runtime', str(self.runtime), '--pair-receipt', str(self.pair_receipt), '--out', str(self.out),
                '--run', self.run, '--port', str(self.port), '--weights', str(self.weights)]


def preflight(c: Config) -> dict[str, Any]:
    """No model import, window command, listener or production mutation."""
    require(platform.node().split('.')[0].lower() in ('x86host', 'x86host'), 'execute ON X86host only')
    require(sys.executable == PYTHON and os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator',
            'coordinator and qualified interpreter required')
    require(re.fullmatch('[0-9a-f]{40}', c.commit) is not None, 'full candidate commit required')
    require(re.fullmatch(r'PP4-server-[A-Za-z0-9_-]{1,40}', c.run) is not None, 'fresh PP4-server run tag required')
    require(18640 <= c.port <= 18699, 'allocated diagnostic port 18640..18699 required (never production 18620)')
    for tree, commit in ((c.baseline, BASELINE), (c.candidate, c.commit)):
        inside(tree)
        require(git(tree, 'rev-parse', 'HEAD') == commit, 'staged source revision differs')
        require(not git(tree, 'status', '--porcelain', '--untracked-files=no'), 'dirty staged source')
    require(c.baseline.resolve() != c.candidate.resolve(), 'separate source trees required')
    require(DRIVER == c.candidate.resolve()/'offload/dspark/probabilistic/pp4/server_probe.py',
            'run the driver from the reviewed candidate tree')
    inside(c.out)
    require(not c.out.exists(), 'fresh output directory required')
    # Every non-server worker byte must agree. Only the reviewed graph override
    # deletion and readiness log are allowed in the actual worker package.
    base = c.baseline/'offload/dspark/worker'
    candidate = c.candidate/'offload/dspark/worker'
    bfiles = {p.relative_to(base) for p in base.rglob('*') if p.suffix in ('.py', '.cu', '.cpp', '.h')}
    cfiles = {p.relative_to(candidate) for p in candidate.rglob('*') if p.suffix in ('.py', '.cu', '.cpp', '.h')}
    # Test additions do not participate in worker dispatch.
    bfiles = {p for p in bfiles if 'tests' not in p.parts}
    cfiles = {p for p in cfiles if 'tests' not in p.parts}
    require(bfiles == cfiles, 'worker source set differs')
    for path in bfiles - {Path('server.py')}:
        require(digest(base/path) == digest(candidate/path), 'unreviewed worker delta: ' + str(path))
    expected = (base/'server.py').read_text().replace('        graphs = False\n', '', 1)
    expected = expected.replace('%s GiB reserved per device",', '%s GiB reserved per device, graphs=%s",')
    expected = expected.replace('for d in dict.fromkeys(devs)])', 'for d in dict.fromkeys(devs)], graphs)')
    require(expected == (candidate/'server.py').read_text(), 'candidate is not the minimal reviewed service correction')
    from .pair_contract import validate
    from .pair_qualify import SOURCES
    physical = read(c.pair_receipt)
    # Reuse the preserved baseline selector/whole-model qualification, whose
    # numerical sources above must be byte-identical. It did NOT qualify the
    # service entry point. Requiring a new Spark HOLD to obtain this diagnostic
    # prerequisite would defeat the purpose of the X86host-only experiment.
    validate(physical, BASELINE, str(physical.get('run', '')))
    historical_sources = set(SOURCES) - {'offload/dspark/worker/server.py'}
    require(physical['source_sha256'] == {name: digest(c.baseline/name) for name in historical_sources},
            'preserved baseline physical qualification source hashes required')
    runtime = read(c.runtime)
    require(runtime.get('schema') == 'pp4-server-probe-runtime-v1' and runtime.get('reviewed') is True
            and runtime.get('python') == PYTHON and runtime.get('torch') == '2.12.1+cu130',
            'independently reviewed CPython3.13 runtime manifest required')
    require(set(runtime['files']) == set(FILES), 'six runtime artifacts required')
    for label in FILES:
        row = runtime['files'][label]
        path = Path(row['path'])
        require(path.is_absolute() and re.fullmatch('[0-9a-f]{64}', row['sha256']) is not None
                and digest(path) == row['sha256'], 'runtime bytes differ: ' + label)
    # Cross-check the compiled runtime against the same qualification's staging
    # receipt, rather than accepting arbitrary newly hashed binaries.
    from offload.launch.windows.pp4_prerequisites import FACC, STOCK, runtime_artifacts
    staging = read(Path(runtime['staging_receipt']))
    require(digest(Path(runtime['staging_receipt'])) == runtime['staging_sha256']
            and staging.get('source_commit') == c.commit and staging.get('staging_verdict') == 'REVIEWED',
            'runtime staging source/review differs')
    artifacts = runtime_artifacts(staging)
    require(runtime['files']['stock']['sha256'] == artifacts[STOCK]
            and runtime['files']['facc']['sha256'] == artifacts[FACC]
            and runtime['files']['p2p']['sha256'] == artifacts[next(k for k in artifacts if k.startswith('p2p/'))],
            'runtime not the qualified stock/facc/P2P build')
    require(c.weights.resolve() == Path('/opt/dsv41/x86host-data/dspark'), 'same existing production checkpoint required')
    check_memory()
    require(listener_absent(c.port), 'diagnostic port already has a listener')
    return runtime


def listener_absent(port: int) -> bool:
    for name in ('tcp', 'tcp6'):
        for line in Path('/proc/net', name).read_text().splitlines()[1:]:
            fields = line.split()
            if int(fields[1].rsplit(':', 1)[1], 16) == port and fields[3] == '0A':
                return False
    return True


def terminate(process: subprocess.Popen[bytes]) -> None:
    """Only the session/group created by this driver; never a name match."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGUSR1)  # child registered faulthandler before imports
        time.sleep(.2)
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    require(process.poll() is not None, 'owned worker did not exit')
    # A surviving child in this session is not a clean teardown.
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return
    raise ValueError('STOP: owned worker process group remains')


def child_env(c: Config, tree: Path, out: Path, runtime: dict[str, Any]) -> dict[str, str]:
    files = runtime['files']
    # Inherit toolchain locations, but refuse feature/adapter/timeout pollution.
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DSPARK_', 'TRITON_', 'CUDA_', 'EXLLAMAV3_'))}
    for name in ('HF_HOME', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR',
                 'CUDA_CACHE_PATH', 'TORCHINDUCTOR_CACHE_DIR', 'TMPDIR'):
        path = out/name.lower()
        path.mkdir()
        env[name] = str(path)
    tune = out/'tune-cache.bin'
    with tune.open('xb') as handle:
        handle.write(Path(files['tune']['path']).read_bytes())
    require(digest(tune) == files['tune']['sha256'], 'private tune copy differs')
    env.update(PYTHONPATH=os.pathsep.join((str(Path(files['stock']['path']).parent),
        str(Path(files['facc']['path']).parent), str(tree/'offload/dspark'), str(c.candidate))),
        CUDA_VISIBLE_DEVICES=WORKER_GPUS, DSPARK_WORKER_ARM='k2-tp2', DSPARK_SAMPLED_TOP32='1',
        DSPARK_P2P_PREBUILT=str(Path(files['p2p']['path']).parents[1]),
        EXLLAMAV3_TUNE_CACHE=str(tune), OFL_LIB=files['ofl']['path'],
        OFL_PY_DIR=str(Path(files['ofl_python']['path']).parent),
        PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
        OPENBLAS_NUM_THREADS='8', NUMEXPR_NUM_THREADS='8', PP4_SERVER_PROBE='1', PP4_SERVER_RUN=c.run,
        PP4_SERVER_TREE=str(tree), PP4_SERVER_OUT=str(out), PP4_SERVER_PORT=str(c.port),
        PP4_SERVER_WEIGHTS=str(c.weights), PP4_SERVER_RUNTIME=str(c.runtime))
    return env


def worker(c: Config, variant: str, tree: Path, runtime: dict[str, Any], locks: tuple[int, int]) -> dict[str, Any]:
    out = c.out/variant
    out.mkdir()
    env = child_env(c, tree, out, runtime)
    env.update(PP4_SERVER_LOCK0=str(locks[0]), PP4_SERVER_LOCK1=str(locks[1]))
    # Instrumentation comes from the reviewed candidate; worker.* comes from
    # the variant tree. The child runs real worker.server.__main__ --tp 2.
    command = ['taskset', '-c', '12-15,28-31', 'nice', '-n', '19', PYTHON, '-u',
               str(DRIVER.with_name('server_probe_child.py'))]
    started = time.monotonic()
    error = ''
    with (out/'worker.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=tree/'offload/dspark', env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=locks)
        try:
            while not (out/'complete.json').exists():
                require(process.poll() is None, f'{variant}: worker exited rc{process.returncode}')
                require(not (out/'failure.json').exists(), f'{variant}: client failed; see failure.json')
                if not (out/'listener.json').exists():
                    require(time.monotonic() < started+STARTUP_S, variant+': immutable startup bound exceeded')
                bound: dict[str, Any] = read(out/'bound.json') if (out/'bound.json').exists() else {
                    'until': started+STARTUP_S, 'phase': 'pre-import startup'}
                require(time.monotonic() < min(float(bound['until']), started+PROBE_S),
                        variant + ': bound exceeded at ' + str(bound['phase']))
                time.sleep(.05)
            result = read(out/'complete.json')
            require(result.get('complete') is True, 'incomplete worker primary')
        except (ValueError, OSError) as exc:
            error = f'{type(exc).__name__}: {exc}'
            result = {'complete': False, 'error': error}
        finally:
            terminate(process)
            require(listener_absent(c.port), 'owned diagnostic listener remains')
    result.update(variant=variant, elapsed_s=time.monotonic()-started, worker_exited=True, listener_absent=True,
                  log_sha256=digest(out/'worker.log'))
    write(out/'result.json', result)
    return result


def owned(c: Config) -> None:
    runtime = read(c.runtime)  # preflight in the outer process; rehashed before open
    for row in runtime['files'].values():
        require(digest(Path(row['path'])) == row['sha256'], 'runtime changed after admission')
    group = Path(Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1]).name
    require(group.startswith('dsv41w-'+c.run+'-probe-') and group.endswith('.scope'), 'shared owned scope required')
    subprocess.run(['systemctl', '--user', 'set-property', '--runtime', group, 'MemoryMax=48G', 'MemorySwapMax=0'],
                   check=True, timeout=15)
    marker = Path.home()/'dsv41-offload/PAIR_OK'
    require(not marker.exists() and not (Path.home()/'dsv41-offload/d2/TRAIN_OK').exists(), 'foreign pair/train window')
    # Persist intent before the first production mutation, including lost reply.
    (c.out/'pair-attempted').touch(exist_ok=False)
    with (c.out/'pair-open.log').open('x') as log:
        subprocess.run(['bash', str(Path.home()/'dsv41-offload/prod_window.sh'), 'pair', 'COORD'],
                       check=True, timeout=120, stdout=log, stderr=subprocess.STDOUT)
    require(marker.read_text().splitlines()[0] == 'COORD', 'pair owner differs')
    paths = [Path.home()/'dsv41-offload/locks'/name for name in ('gpu-c4e7a371.lock', 'gpu-7744b44b.lock')]
    handles = [p.open('a') for p in paths]
    try:
        for handle in handles:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Verify host-local placement without touching other cards or changing power.
        inventory = subprocess.check_output(['nvidia-smi', '-L'], text=True, timeout=15)
        require(all(uuid in inventory for uuid in WORKER_GPUS.split(',')), 'physical UUID inventory differs')
        write(c.out/'placement.json', {'gpus': WORKER_GPUS, 'inventory': inventory, 'locks': [str(p) for p in paths]})
        locks = (handles[0].fileno(), handles[1].fileno())
        reference = worker(c, 'reference', c.candidate, runtime, locks)
        require(reference.get('pass') is True and reference.get('graphs') is True, 'graph reference failed; stop')
        baseline = worker(c, 'baseline', c.baseline, runtime, locks)
        candidate = worker(c, 'candidate', c.candidate, runtime, locks)
        exact = (candidate.get('signatures') == reference.get('signatures'))
        require(candidate.get('pass') is True and candidate.get('graphs') is True and exact,
                'candidate failed the separately preserved qualified graph reference')
        write(c.out/'comparison.json', {'reference': reference, 'baseline': baseline, 'candidate': candidate,
            'candidate_exact_reference': exact, 'baseline_exact_reference': baseline.get('signatures') == reference.get('signatures'),
            'interpretation': 'Dispatch gap closed. Original cause supported only if baseline reproduces at a named phase '
                'and candidate clears it; without baseline reproduction the cause remains unresolved.',
            'classification': 'MEASURED X86host diagnostic only; not Spark admission or performance qualification'})
    finally:
        for handle in handles:
            handle.close()


def local(command: str, timeout: float, log: Path) -> tuple[int, str]:
    with log.open('x') as output:
        result = subprocess.run(['bash', '-c', command], stdout=output, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    return result.returncode, log.read_text()


def execute(c: Config) -> dict[str, Any]:
    runtime = preflight(c)
    c.out.mkdir(parents=True, exist_ok=False)
    write(c.out/'identity.json', {'schema': SCHEMA, 'run': c.run, 'baseline': BASELINE, 'candidate': c.commit,
        'baseline_tree': str(c.baseline.resolve()), 'candidate_tree': str(c.candidate.resolve()),
        'driver_sha256': {p.name: digest(p) for p in (DRIVER, DRIVER.with_name('server_probe_child.py'),
            DRIVER.with_name('server_probe_frames.py'), DRIVER.with_name('server_probe_trace.py'))},
        'runtime': runtime, 'runtime_sha256': digest(c.runtime), 'pair_sha256': digest(c.pair_receipt),
        'worker_source_sha256': {variant: {str(p.relative_to(tree)): digest(p)
            for p in (tree/'offload/dspark/worker').rglob('*.py') if 'tests' not in p.relative_to(tree).parts}
            for variant, tree in (('baseline', c.baseline), ('candidate', c.candidate))},
        'startup_bound_s': STARTUP_S, 'request_bound_s': REQUEST_S, 'cold_cache': 'fresh private empty dirs per process',
        'warm_cache': 'ten changed-input rounds per mask/batch in the same process; no driver cache reset'})
    deadline = time.time()+PROBE_S
    restore_by = deadline+RESTORE_S  # original ceiling, never renewed after fencing
    error = ''
    restored = False
    script = [PYTHON, str(DRIVER), '--owned', *c.argv()]
    try:
        rc, text = local(owner.command('launch', c.run+'-probe', script, deadline=deadline), PROBE_S+30, c.out/'scope.log')
        require(rc == 0, 'owned diagnostic failed: ' + text[-500:])
        comparison = read(c.out/'comparison.json')
        reference, candidate = comparison['reference'], comparison['candidate']
        for row in (reference, candidate):
            require(all(row.get(key) is True for key in ('complete', 'pass', 'graphs', 'worker_exited', 'listener_absent')),
                    'complete graph-reference/candidate primary evidence required')
        require(candidate['signatures'] == reference['signatures'] and comparison.get('candidate_exact_reference') is True,
                'candidate signature primary differs from separate reference')
    except (ValueError, OSError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        saved = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        serial = 0
        def invoke(host: str, cmd: str) -> tuple[int, str]:
            nonlocal serial
            require(host == 'localhost', 'X86host-local scope only; no SSH')
            serial += 1
            remaining = restore_by-time.time()
            require(remaining > 0, 'original restoration deadline exhausted')
            return local(cmd, min(30, remaining), c.out/f'proof-{serial:03d}.log')
        # Fencing must succeed before close: no delayed pair-open can undo it.
        try:
            owner.fence_all(c.run+'-probe', invoke, hosts=('localhost',))
            if (c.out/'pair-attempted').exists():
                try:
                    remaining = restore_by-time.time()
                    require(remaining > 90, 'reserve 90s for final close-scope absence proof')
                    close = ['bash', str(Path.home()/'dsv41-offload/prod_window.sh'), 'close']
                    rc, text = local(owner.command('launch', c.run+'-close', close, deadline=restore_by-90),
                                     min(RESTORE_S-90, remaining-90), c.out/'pair-close.log')
                    require(rc == 0 and 'PRODUCTION HOLD OK' in text.splitlines(), 'pair restoration unproved')
                    restored = True
                finally:
                    owner.fence_all(c.run+'-close', invoke, hosts=('localhost',))
            else:
                restored = True  # no production mutation was admitted
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            restored = False
            error += f'; RESTORE FAILED: {type(exc).__name__}: {exc}; coordinator inspection required'
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
    result = {'schema': SCHEMA, 'run': c.run, 'candidate': c.commit, 'stop': error,
              'pair_restored': restored, 'verdict': 'DIAGNOSTIC COMPLETE' if not error and restored else 'STOP',
              'admission': 'NOT QUALIFIED; no Spark operations; no promotion'}
    write(c.out/'summary.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--owned', action='store_true', help=argparse.SUPPRESS)
    for name in ('baseline', 'candidate', 'runtime', 'pair-receipt', 'out', 'weights'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--commit', default='')
    parser.add_argument('--run', default='')
    parser.add_argument('--port', type=int, default=18644)
    args = parser.parse_args()
    if not (args.execute or args.owned):
        print('DRY: coordinator ON X86host only; physical pair window, no Spark/HOLD/SSH. Separate reference, '
              '1b81229 baseline, candidate REAL worker.server --tp 2; private cold caches then changed-input warm '
              'B1->B4->B2->B3; direct frames then local OFL/reset/reconnect; startup600s/request30s; always scoped close.')
        return
    require(all(isinstance(getattr(args, name), Path) for name in ('baseline', 'candidate', 'runtime', 'pair_receipt', 'out', 'weights')),
            'all paths required')
    c = Config(args.baseline, args.candidate, args.commit, args.runtime, args.pair_receipt, args.out, args.run, args.port, args.weights)
    if args.owned:
        require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator' and sys.executable == PYTHON,
                'coordinator-only child')
        owned(c)
    else:
        def interrupted(signum: int, frame: object) -> None:
            raise KeyboardInterrupt(f'signal {signum}')
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, interrupted)
        result = execute(c)
        print(json.dumps(result))
        if result['verdict'] != 'DIAGNOSTIC COMPLETE':
            raise SystemExit(2)


if __name__ == '__main__':
    main()
