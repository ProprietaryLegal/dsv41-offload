"""Read-only X86host GPU probe; stream via stdin, aggregate results to stdout.

Uses the existing N4 frozen runtime and caches. Refuses a cache miss requiring a
filesystem write. Both actual TP2 ranks run on the assigned solo card.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import cast


def readonly_runtime(root: Path) -> dict[str, str]:
    """Reuse compiled extensions and Triton artifacts without rebuilding them."""
    import torch.utils.cpp_extension as cpp
    from triton import knobs
    from triton.runtime.cache import FileCacheManager

    binaries: dict[str, str] = {}

    def extension(name: str, **_kwargs: object) -> ModuleType:
        path = root / 'build' / name / f'{name}.so'
        binaries[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f'prebuilt extension unavailable: {name}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    cpp.load = extension  # type: ignore[assignment]

    class ReadOnlyCache(FileCacheManager):  # type: ignore[misc]
        def __init__(self, key: str, override: bool = False, dump: bool = False):
            if override or dump:
                raise RuntimeError('cache override/dump is outside this probe')
            self.key = key
            self.lock_path = None
            candidates = [root / 'triton', root.parent / 'n2/triton', root.parent / 'n3/triton',
                          root.parent / 's13/triton', Path.home() / '.triton/cache']
            self.cache_dir = str(next((p / key for p in candidates
                                       if (p / key).is_dir()), candidates[0] / key))

        def put(self, data: object, filename: str, binary: bool = True) -> str:
            raise RuntimeError(f'read-only Triton cache miss: {self.key}/{filename}')

        def put_group(self, filename: str, group: dict[str, str]) -> str:
            raise RuntimeError(f'read-only Triton group miss: {filename}')

    knobs.cache.manager_class = ReadOnlyCache
    allowed_lock = str(Path.home() / 'dsv41-offload/locks/gpu-bc62df2a.lock')

    def audit(event: str, args: tuple[object, ...]) -> None:
        if event == 'open':
            path, _mode, flags = args
            if isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                if str(path) not in (allowed_lock, '/dev/null'):
                    raise RuntimeError(f'filesystem write refused: {path}')
        elif event in ('os.remove', 'os.rename', 'os.rmdir', 'os.unlink', 'os.symlink', 'os.link'):
            raise RuntimeError(f'filesystem mutation refused: {event}')
        elif event == 'os.mkdir' and not Path(cast(str, args[0])).is_dir():
            raise RuntimeError(f'filesystem mkdir refused: {args[0]}')

    sys.addaudithook(audit)
    return binaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=('k2-tp2', 'native-hybrid-kx2'))
    parser.add_argument('--runtime', type=Path, default=Path.home() / 'dsv41-offload/n4')
    parser.add_argument('--root', default='/opt/dsv41/x86host-data/dspark')
    parser.add_argument('--synthetic-only', action='store_true')
    args = parser.parse_args()
    binaries = readonly_runtime(args.runtime)
    import torch
    from offload.dspark.worker.checkpoint import Checkpoint
    from offload.dspark.worker.tp2 import check_visible
    from offload.dspark.native.tp2_native import build

    check_visible(pair=False)
    torch.set_num_threads(4)
    start = time.monotonic()
    print('PHASE load ' + args.arm, flush=True)
    dev = torch.device('cuda', 0)
    model = build(args.arm, [dev, dev], Checkpoint(args.root), slots=4)
    print(f'PHASE loaded seconds={time.monotonic()-start:.1f}', flush=True)
    model.capture(batch_sizes=(1, 2, 3), ks=(5,), insert_rows=(8, 16, 32, 64, 128), n_outs=(4, 5), margins=False)
    print('PHASE captured', flush=True)
    cells: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}
    graph_checks = 0
    graph_exact = 0
    window_checks = 0
    window_exact = 0
    prefix_checks = 0
    prefix_exact = 0
    permutation_cells: list[dict[str, object]] = []

    def compare(label: str, inputs: torch.Tensor) -> None:
        nonlocal graph_checks, graph_exact, window_checks, window_exact, prefix_checks, prefix_exact
        same_shape: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        before = [(rank.ring_pos.clone(), rank.ring_kv.clone()) for rank in model.ranks]
        alone: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for row in range(3):
            ids, conf, _, _ = model.draft(inputs[:, row:row+1].contiguous().pin_memory(), 5, n_out=5)
            alone[row] = (ids.cpu().clone(), conf.cpu().clone())
        for n in (5, 4):
            # Every request appears at every packed row offset; independent companions.
            for order in ((0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2),
                          (0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0), (1, 0, 2), (0, 2, 1)):
                inp = inputs[:, list(order)].contiguous().pin_memory()
                ids, conf, _, _ = model.draft(inp, 5, n_out=n)
                eager_ids, eager_conf = ids.cpu().clone(), conf.cpu().clone()
                gi, gc, _ = model.draft_graphed(inp, 5, n_out=n)
                gi, gc = gi.cpu().clone(), gc.cpu().clone()
                graph_checks += 1
                graph_exact += int(torch.equal(gi, eager_ids) and torch.equal(gc, eager_conf))
                if not (torch.isfinite(gc).all() and not gi[:, n:].any() and not gc[:, n:].any()):
                    raise RuntimeError('nonfinite confidence or nonzero unused tail')
                for offset, request in enumerate(order):
                    ai, ac = alone[request]
                    key = (len(order), n, request)
                    if key in same_shape:
                        previous = same_shape[key]
                        permutation_cells.append(dict(fixture=label, batch=len(order), n_out=n,
                                                      ids_exact=torch.equal(previous[0], gi[offset]),
                                                      confidence_max_abs=float((previous[1]-gc[offset]).abs().max())))
                    else:
                        same_shape[key] = (gi[offset].clone(), gc[offset].clone())
                    match = (gi[offset, :n] == ai[0, :n]).tolist()
                    delta = float((gc[offset, :n]-ac[0, :n]).abs().max())
                    cost = torch.tensor([28.79, 31.86, 34.44, 37.76, 41.56, 43.68]) + 4.4
                    cap = 4 if len(order) >= 3 else 5
                    def depth(values: torch.Tensor) -> int:
                        v = values.clone()
                        v[cap:] = 0
                        expected = torch.cat((torch.ones(1), 1 + v.cumprod(0).cumsum(0)))
                        return int((expected / cost).argmax())
                    cells.append(dict(fixture=label, batch=len(order), n_out=n, row=offset, request_index=request,
                                      agreement=match, confidence_max_abs=delta,
                                      alone_depth=depth(ac[0]), packed_depth=depth(gc[offset])))
                if n == 5:
                    summaries[str(order)] = {'ids': gi, 'conf': gc}
                else:
                    full = summaries[str(order)]
                    prefix_checks += 1
                    prefix_exact += int(torch.equal(gi[:, :n], cast(torch.Tensor, full['ids'])[:, :n]) and
                                        torch.equal(gc[:, :n], cast(torch.Tensor, full['conf'])[:, :n]))
        for rank, (pos, kv) in zip(model.ranks, before):
            window_checks += 1
            window_exact += int(torch.equal(pos, rank.ring_pos) and torch.equal(kv, rank.ring_kv))
        print('PHASE fixture complete ' + label, flush=True)

    generator = torch.Generator().manual_seed(20261024)
    # Synthetic values stay in process; no token IDs, prompts or tensors emitted.
    for case, lasts in enumerate(((0, 63, 127), (128, 300, 32768), (150000, 32000, 1000))):
        model.reset_slots([0, 1, 2])
        for slot, last in enumerate(lasts):
            pos = torch.arange(max(0, last-127), last+1)
            mx = torch.randn(len(pos), model.s.hidden, generator=generator).to(torch.bfloat16)
            model.insert_mx_eager(torch.full_like(pos, slot), pos, mx)
        inputs = torch.stack((torch.arange(3), torch.randint(100, 10000, (3,), generator=generator),
                              torch.tensor(lasts), (torch.tensor(lasts)-127).clamp_min(0)))
        compare(f'synthetic-{case}', inputs)

    insert_cells = []
    for rows_per_request in (1, 2, 5):
        positions = torch.cat([torch.arange(300, 300+rows_per_request) for _ in range(3)])
        slots = torch.arange(3).repeat_interleave(rows_per_request)
        synthetic_ids = torch.randint(100, 10000, (len(slots), 3), generator=generator)
        taps = model.ranks[0].embed[synthetic_ids.to(dev)].reshape(len(slots), -1).cpu().pin_memory()
        model.reset_slots([0, 1, 2])
        for slot in range(3):
            selected = slots == slot
            model.insert_taps_graphed(slots[selected].pin_memory(), positions[selected].pin_memory(),
                                      taps[selected].pin_memory())
        solo_windows = [(r.ring_pos.cpu().clone(), r.ring_kv.cpu().clone()) for r in model.ranks]
        model.reset_slots([0, 1, 2])
        model.insert_taps_graphed(slots.pin_memory(), positions.pin_memory(), taps)
        for rank, (p, kv) in zip(model.ranks, solo_windows):
            actual = rank.ring_kv.cpu()
            insert_cells.append(dict(rows_per_request=rows_per_request, rank=rank.r,
                                     positions_exact=torch.equal(p, rank.ring_pos.cpu()),
                                     kv_equal_elements=int((actual == kv).sum()), kv_elements=kv.numel(),
                                     kv_max_abs=float((actual-kv).abs().max())))
    print('PHASE insert comparisons complete', flush=True)

    if not args.synthetic_only:
        sys.path.insert(0, str(args.runtime / 'verify-source'))
        from verify.captures import CaptureSet
        captures = CaptureSet(args.root + '/captures/A-capture-20261022/dspark-captures')
        samples = [s for s in captures.samples if s.b == 0 and 8 <= s.round <= 19]
        if len(samples) != 12:
            raise RuntimeError('expected all twelve N4 V1 samples')
        # Pack three different captured request states, including persistent windows.
        for start_sample in range(0, 12, 3):
            group = samples[start_sample:start_sample+3]
            model.reset_slots([0, 1, 2])
            for slot, sample in enumerate(group):
                arm = captures.arm_inputs(sample)
                for source in ('engine_mx', 'taps'):
                    rows = [(p, row) for p, kind, row in arm['window'] if kind == source]
                    for a in range(0, len(rows), 128):
                        chunk = rows[a:a+128]
                        positions = torch.tensor([p for p, _ in chunk]).pin_memory()
                        slots = torch.full_like(positions, slot).pin_memory()
                        values = torch.stack([row for _, row in chunk]).pin_memory()
                        if source == 'taps':
                            model.insert_taps_graphed(slots, positions, values)
                        else:
                            model.insert_mx_eager(slots, positions, values)
                if {kind for _, kind, _ in arm['window']} - {'engine_mx', 'taps'}:
                    raise RuntimeError('unknown capture window source')
            inputs = torch.tensor([[0, 1, 2], [s.anchor for s in group], [s.P for s in group],
                                   [max(0, s.P-127) for s in group]])
            compare(f'capture-group-{start_sample//3}', inputs)

    aggregate = []
    for batch in (2, 3):
        for n in (5, 4):
            chosen = [c for c in cells if c['batch'] == batch and c['n_out'] == n]
            aggregate.append(dict(batch=batch, n_out=n, comparisons=len(chosen),
                                  per_position_equal=[sum(cast(list[bool], c['agreement'])[i] for c in chosen)
                                                      for i in range(n)],
                                  blocks_equal=sum(all(cast(list[bool], c['agreement'])) for c in chosen),
                                  confidence_max_abs=max(cast(float, c['confidence_max_abs']) for c in chosen)))
    sources = {}
    source_root = args.runtime / 'repo'
    for base in ('native', 'worker', 'kernels'):
        for path in sorted((source_root / 'offload/dspark' / base).rglob('*')):
            if path.suffix in ('.py', '.cu', '.cuh', '.json'):
                sources[str(path.relative_to(source_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(arm=args.arm, label='MEASURED', mode='both TP2 ranks emulated on assigned solo GPU',
                  graph_checks=graph_checks, graph_exact=graph_exact,
                  window_checks=window_checks, window_exact=window_exact,
                  prefix_checks=prefix_checks, prefix_exact=prefix_exact,
                  fixtures=sorted({cast(str, c['fixture']) for c in cells}), agreement=aggregate,
                  cells=cells, permutation_cells=permutation_cells, insert_cells=insert_cells, runtime_sources=sources, seconds=time.monotonic()-start,
                  prebuilt_extensions_sha256=binaries, read_only_runtime=True, torch_version=torch.__version__,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(dev))
    print('RESULT_JSON ' + json.dumps(result), flush=True)
    if graph_checks != graph_exact or window_checks != window_exact or prefix_checks != prefix_exact:
        raise RuntimeError('graph/window/prefix invariance failed; inspect aggregate result')


if __name__ == '__main__':
    main()
