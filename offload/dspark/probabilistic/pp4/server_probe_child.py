"""Isolated bootstrap for the REAL worker.server CLI; no model/kernel copy.

This file is executed by path from the reviewed driver. worker.* resolves only
to the selected staged variant. Tracebacks arm before numpy/torch/model imports.
"""
from __future__ import annotations

import faulthandler
import fcntl
import hashlib
import importlib
import json
import os
import platform
import runpy
import signal
import subprocess
import sys
import threading
from pathlib import Path
from types import FrameType
from typing import Any


def guard() -> Path:
    if (os.environ.get('WINDOW_COORDINATOR') != 'Deepseek/coordinator'
            or os.environ.get('PP4_SERVER_PROBE') != '1'
            or sys.executable != '/home/operator/miniconda3/bin/python3'
            or platform.node().split('.')[0].lower() not in ('x86host', 'x86host')):
        raise ValueError('coordinator-only X86host server probe')
    out = Path(os.environ['PP4_SERVER_OUT']).resolve()
    if not out.is_relative_to('/opt/dsv41/x86host-data/pp4'):
        raise ValueError('probe output outside X86host PP4 scratch')
    identity = json.loads((out.parent/'identity.json').read_text())
    variant = 'baseline' if out.name == 'baseline' else 'candidate'
    if (out.name not in ('reference', 'baseline', 'candidate') or identity['run'] != os.environ['PP4_SERVER_RUN']
            or identity[variant+'_tree'] != os.environ['PP4_SERVER_TREE']):
        raise ValueError('child run/source directory differs from independently persisted parent identity')
    tree = Path(os.environ['PP4_SERVER_TREE'])
    for name, expected_hash in identity['worker_source_sha256'][variant].items():
        if hashlib.sha256((tree/name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError('worker source changed after parent admission: '+name)
    if hashlib.sha256(Path(os.environ['PP4_SERVER_RUNTIME']).read_bytes()).hexdigest() != identity['runtime_sha256']:
        raise ValueError('runtime manifest changed after parent admission')
    for rank, suffix in enumerate(('c4e7a371', '7744b44b')):
        fd = int(os.environ[f'PP4_SERVER_LOCK{rank}'])
        expected = (Path.home()/'dsv41-offload/locks'/f'gpu-{suffix}.lock').stat()
        actual = os.fstat(fd)
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError('wrong inherited physical GPU lock')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (Path.home()/'dsv41-offload/PAIR_OK').read_text().splitlines()[0] != 'COORD':
        raise ValueError('coordinator pair window absent')
    return out


def model_checks(model: Any, out: Path) -> list[str]:
    import torch

    from offload.dspark.probabilistic.pp4 import server_probe_frames as frames
    from offload.dspark.probabilistic.pp4.full_worker import (
        load_synthetic,
        synthetic_fixtures,
    )
    signatures = []
    inputs = load_synthetic(model, synthetic_fixtures(model.s.hidden)[0])
    # No other draft/reset between the initial positive B1 and B4 at each
    # output depth. The model API consumes preloaded KV, not wire row deltas.
    for depth in (4, 5):
        for batch in (1, 4):
            frames.bound(out, f'model/cold/depth{depth}/B{batch}')
            packed = inputs[:, :batch].contiguous().pin_memory()
            mask = torch.tensor(frames.masks(batch, 'positive'), dtype=torch.bool).pin_memory()
            keys = torch.tensor([[4104, slot+1, 0] for slot in range(batch)], dtype=torch.int64).pin_memory()
            outputs = model.draft_sampled(packed, 5, keys, mask, n_out=depth)
            signatures.append(frames.signature(*(x.detach().cpu().numpy().copy() for x in outputs)))
    for batch in frames.BATCHES:
        packed = inputs[:, :batch].contiguous().pin_memory()
        for depth in (4, 5):
            for mode in frames.MODES:
                mask = torch.tensor(frames.masks(batch, mode), dtype=torch.bool).pin_memory()
                for iteration in range(frames.ROUNDS):
                    frames.bound(out, f'model/B{batch}/depth{depth}/{mode}/{iteration}')
                    changed = packed.clone().pin_memory()
                    changed[1] += iteration
                    keys = torch.tensor([[4104, slot+1, iteration] for slot in range(batch)], dtype=torch.int64).pin_memory()
                    ids, conf, support, q = model.draft_sampled(changed, 5, keys, mask, n_out=depth)
                    actual = [x.detach().cpu().numpy().copy() for x in (ids, conf, support, q)]
                    if not bool(mask.all()):
                        if model.has_graph(batch, 5, depth):
                            greedy_ids, greedy_conf, _ = model.draft_graphed(changed, 5, n_out=depth)
                        else:
                            greedy_ids, greedy_conf, _, _ = model.draft(changed, 5, n_out=depth)
                        zero = ~mask.numpy()
                        if (not (actual[0][zero] == greedy_ids.detach().cpu().numpy()[zero]).all()
                                or not (actual[1][zero] == greedy_conf.detach().cpu().numpy()[zero]).all()
                                or actual[3][zero].any()):
                            raise RuntimeError('exact model T0 identity failed')
                    signatures.append(frames.signature(*actual))
    return signatures


def install(namespace: dict[str, Any], out: Path) -> None:
    import torch

    from offload.dspark.probabilistic.pp4 import server_probe_frames as frames
    from offload.dspark.probabilistic.pp4 import server_probe_trace as trace
    from offload.launch.windows.pf1_window import WORKER_GPUS
    expected_source = Path(os.environ['PP4_SERVER_TREE'])/'offload/dspark/worker/server.py'
    if Path(namespace['__file__']).resolve() != expected_source.resolve():
        raise ValueError('actual worker.server import is from another tree')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != WORKER_GPUS or torch.cuda.device_count() != 2:
        raise ValueError('physical pair required; emulation refused')
    for rank, uuid in enumerate(WORKER_GPUS.split(',')):
        if str(torch.cuda.get_device_properties(rank).uuid).removeprefix('GPU-') != uuid.removeprefix('GPU-'):
            raise ValueError('actual CUDA UUID differs')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    trace.install(namespace)
    builder = namespace['build_core_tp2']
    evidence: dict[str, Any] = {}
    def build(*args: Any, **kwargs: Any) -> Any:
        with trace.phase('ACTUAL-server.build_core_tp2'):
            core, model = builder(*args, **kwargs)
        runtime = json.loads(Path(os.environ['PP4_SERVER_RUNTIME']).read_text())
        if torch.__version__ != runtime['torch']:
            raise ValueError('loaded Torch version differs from reviewed runtime')
        loaded = {}
        for name, label in (('exllamav3_ext', 'stock'), ('exllamav3_ext_facc', 'facc'), ('dspark_p2p', 'p2p')):
            filename = importlib.import_module(name).__file__
            if not isinstance(filename, str):
                raise TypeError('loaded extension has no source path: '+name)
            path = Path(filename).resolve()
            row = runtime['files'][label]
            if path != Path(row['path']).resolve() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('loaded extension differs: '+name)
            loaded[name] = str(path)
        evidence['loaded_extensions'] = loaded
        evidence['graphs'] = bool(core.backend.graphs)
        evidence['source'] = str(expected_source)
        evidence['environment'] = {key: os.environ.get(key, '') for key in (
            'CUDA_VISIBLE_DEVICES', 'CUDA_MODULE_LOADING', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH',
            'TORCH_EXTENSIONS_DIR', 'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF',
            'EXLLAMAV3_TUNE_CACHE', 'DSPARK_P2P_PREBUILT', 'OFL_LIB')}
        evidence['versions'] = {'python': sys.version, 'torch': torch.__version__,
                                'triton': __import__('triton').__version__}
        # Same synthetic/checkpoint identity as the preserved qualification.
        if (core.expected.drafter_identity.hex() != 'be2c1436fa9f9e66b398e1b1da2e140e6fddc82a11939e2ac4240a0ca79543bd'
                or core.expected.markov_identity.hex() != '0b6e71e638abd5abce2db433965bb5ff523cbbe92ed7ae54324a50ccaeffa19a'):
            raise ValueError('checkpoint identity differs')
        errors: list[BaseException] = []
        def direct() -> None:
            try:
                trace.STAGE = 'direct'
                with trace.phase('ordinary-Python-thread/direct-frames'):
                    evidence['direct'] = frames.direct(core, out)
                trace.STAGE = 'model'
                with trace.phase('model-depth4-depth5'):
                    evidence['model'] = model_checks(model, out)
            except BaseException as exc:  # noqa: BLE001 -- propagate thread failure to the builder caller
                errors.append(exc)
        thread = threading.Thread(target=direct, name='pp4-direct', daemon=True)
        thread.start()
        thread.join()  # parent reads independent per-frame/model hard bounds
        if errors:
            raise errors[0]
        frames.save(out/'direct.json', evidence)
        frames.save(out/'hello.json', frames.hello_record(core.expected))
        return core, model
    namespace['build_core_tp2'] = build
    original_init = namespace['OflServer'].__init__
    def listener(server: Any, *args: Any, **kwargs: Any) -> None:
        trace.STAGE = 'ofl'
        frames.bound(out, 'real OflServer startup')
        with trace.phase('real-OflServer-start'):
            original_init(server, *args, **kwargs)
        frames.save(out/'listener.json', {'ready': True, 'port': int(os.environ['PP4_SERVER_PORT']), 'graphs': evidence['graphs']})
        def finish() -> None:
            try:
                with (out/'client.log').open('xb') as log:
                    subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('server_probe_frames.py'))],
                                   check=True, timeout=600, stdout=log, stderr=subprocess.STDOUT)
                ofl = json.loads((out/'ofl.json').read_text())
                direct_signatures = evidence['direct']
                if direct_signatures != ofl['signatures']:
                    raise RuntimeError('direct/native callback q/ID/confidence signatures differ')
                # Separate stages keep model-level setup distinct from real
                # protocol steps. Any request-time eager selector or lazy load
                # disqualifies a candidate graph dispatch.
                events = list(trace.EVENTS)
                selected = [row for row in events if row['stage'] in ('direct', 'ofl')]
                request_setup = [row for row in selected if row['event'] == 'BEGIN' and row.get('phase') in (
                    'triton.handle-load', 'triton.warmup', 'selector.select')]
                peer = [row for row in selected if row['event'] == 'peer-state-after-drain']
                replayed = [row for row in selected if row['event'] == 'graph-stream']
                evidence.update(complete=True, signatures={'direct': direct_signatures, 'model': evidence['model'],
                    'ofl': ofl['signatures']}, request_setup_events=len(request_setup),
                    exact_direct_ofl=True, peer_checks=len(peer), graph_submissions=len(replayed),
                    **{'pass': bool(evidence['graphs'] and not request_setup and peer and replayed)})
                frames.save(out/'complete.json', evidence)
            except BaseException as exc:  # noqa: BLE001 -- primary failure receipt, parent stops and restores
                trace.emit('client-failure', error=f'{type(exc).__name__}: {exc}')
                frames.save(out/'failure.json', {'error': f'{type(exc).__name__}: {exc}'})
        threading.Thread(target=finish, name='pp4-local-client', daemon=True).start()
    namespace['OflServer'].__init__ = listener


def main() -> None:
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    faulthandler.dump_traceback_later(10., repeat=True, exit=False)
    out = guard()
    # The parent deadline is independent of Python/GIL/CUDA progress.
    def profile(frame: FrameType, event: str, arg: object) -> None:
        if event == 'call' and frame.f_code.co_name == 'main' and frame.f_globals.get('__package__') == 'worker':
            sys.setprofile(None)
            install(frame.f_globals, out)
    sys.setprofile(profile)
    sys.argv = ['worker.server', '--tp', '2', '--root', os.environ['PP4_SERVER_WEIGHTS'],
                '--max-model-len', '1048576', '--no-rdma', '--ofl', '10.202.28.1:'+os.environ['PP4_SERVER_PORT'],
                '--status-file', str(out/'status.json')]
    try:
        runpy.run_module('worker.server', run_name='__main__', alter_sys=True)
    finally:
        sys.setprofile(None)
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
