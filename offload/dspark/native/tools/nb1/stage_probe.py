"""NB1 GPU probe: per-operation batch invariance of the TP2 drafter (native kx2 / K2) on the solo card.

Streamed through stdin to x86host (like N6's probe) or run from an NB1 runtime copy. Both TP2 ranks run on the
assigned solo 3090 with local collectives. Every request of a four-request fixture is drafted ALONE, then packed at
B2 (all ordered pairs), B3 and B4 (all permutations). Every top-level operation of the draft body and heads is
recorded (outputs, and inputs for replay). For each operation the probe reports

* end-to-end: the request's rows of the packed output against its alone output (identifies the FIRST divergent op);
* isolated: the operation re-run at the packed shape on the requests' ALONE inputs (its own batch dependence);
* FP64: for every isolated gap, the operation's FP64 (FP32-accumulate where the official math rounds) reference on
  the same alone input, so the gap is labelled within / beyond the operation's own numerical error.

The same is done for window insertion (main_proj -> wkv -> KV QDQ) with eager graph-bucket padding. The single
``RESULT_JSON`` line carries aggregates and sha256 digests only: no tensors, token IDs, prompts or text.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import functools
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Any, Callable, Iterable

LOCK = Path.home() / 'dsv41-offload/locks/gpu-bc62df2a.lock'
ROUND_MS = (28.79, 31.86, 34.44, 37.76, 41.56, 43.68)
DRAFT_MS = 4.4
CAP = {1: 5, 2: 5, 3: 4, 4: 4}          # production n_out cap by active batch ([0,5,5,4,4])
SMALL = 8 << 20                          # bytes: snapshots clone tensors up to this size, reference larger ones
K = 5


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bootstrap(runtime: Path, scratch: Path) -> dict[str, str]:
    """Prebuilt extensions from ``runtime/build`` (read-only); Triton artifacts looked up in the frozen caches and
    compiled only into ``scratch/triton``. A Python audit hook refuses every filesystem write outside ``scratch``,
    the assigned-card lock and /dev/null."""
    import torch.utils.cpp_extension as cpp
    from triton import knobs
    from triton.runtime.cache import FileCacheManager

    binaries: dict[str, str] = {}

    def extension(name: str, **_kwargs: object) -> ModuleType:
        path = runtime / 'build' / name / f'{name}.so'
        binaries[name] = sha(path.read_bytes())
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f'prebuilt extension unavailable: {name}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    cpp.load = extension  # type: ignore[assignment]
    writable = scratch / 'triton'
    lookup = [writable, runtime / 'triton', runtime.parent / 'n2/triton', runtime.parent / 'n3/triton',
              runtime.parent / 's13/triton', Path.home() / '.triton/cache']
    compiled: list[str] = []

    class Cache(FileCacheManager):  # type: ignore[misc]
        def __init__(self, key: str, override: bool = False, dump: bool = False):
            if override or dump:
                raise RuntimeError('cache override/dump is outside this probe')
            self.key = key
            self.lock_path = None
            found = next((p / key for p in lookup if (p / key).is_dir()), None)
            self.writable = found is None or found.parent == writable
            self.cache_dir = str(found if found is not None else writable / key)

        def put(self, data: object, filename: str, binary: bool = True) -> str:
            if not self.writable:
                raise RuntimeError(f'frozen Triton cache entry incomplete: {self.key}/{filename}')
            os.makedirs(self.cache_dir, exist_ok=True)
            compiled.append(self.key)
            return str(super().put(data, filename, binary))

        def put_group(self, filename: str, group: dict[str, str]) -> str:
            if not self.writable:
                raise RuntimeError(f'frozen Triton group incomplete: {filename}')
            os.makedirs(self.cache_dir, exist_ok=True)
            return str(super().put_group(filename, group))

    knobs.cache.manager_class = Cache
    allowed = (str(scratch.resolve()) + '/',)
    exact = (str(LOCK), '/dev/null')

    def permitted(path: object) -> bool:
        p = os.path.abspath(os.fsdecode(path)) if isinstance(path, (str, bytes, os.PathLike)) else ''
        return p in exact or p.startswith(allowed)

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event == 'open':
            path, _mode, flags = args
            if isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                if not permitted(path):
                    raise RuntimeError(f'filesystem write refused: {path}')
        elif event in ('os.remove', 'os.rmdir', 'os.unlink', 'os.symlink', 'os.link'):
            if not permitted(args[0]):
                raise RuntimeError(f'filesystem mutation refused: {event} {args[0]}')
        elif event == 'os.rename':
            if not (permitted(args[0]) and permitted(args[1])):
                raise RuntimeError(f'filesystem rename refused: {args[0]}')
        elif event == 'os.mkdir' and not Path(os.fsdecode(args[0])).is_dir() and not permitted(args[0]):
            raise RuntimeError(f'filesystem mkdir refused: {args[0]}')

    sys.addaudithook(audit)
    bootstrap.compiled = compiled  # type: ignore[attr-defined]
    return binaries


# ------------------------------------------------------------------------------------------------- tracing


@dataclass
class Entry:
    label: str
    op: str
    fn: Callable[..., Any]
    pre: Any                  # (args, kwargs) snapshot before the call
    out: Any                  # return value snapshot, or the arguments after the call for in-place operations
    obj: Any = None           # bound object for methods (dense matrix, expert bank, ...)


class Tracer:
    def __init__(self) -> None:
        self.on = False
        self.depth = 0
        self.rank = -1
        self.part = 'block'
        self.stage = -1
        self.entries: list[Entry] = []
        self.counts: Counter[str] = Counter()

    def reset(self, part: str = 'block') -> None:
        self.entries, self.counts = [], Counter()
        self.rank, self.part, self.stage, self.depth = -1, part, -1, 0


def snap(x: Any) -> Any:
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().clone() if x.numel() * x.element_size() <= SMALL else x
    if isinstance(x, dict):
        return {k: snap(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(snap(v) for v in x)
    return x


def instrument(T: Tracer, owner: Any, attr: str, op: str, *, method: bool = False,
               skip: Callable[[tuple[Any, ...]], bool] | None = None,
               namer: Callable[[tuple[Any, ...]], str] | None = None) -> None:
    orig = getattr(owner, attr)

    @functools.wraps(orig)
    def wrapper(*a: Any, **kw: Any) -> Any:
        if not T.on or T.depth or (skip is not None and skip(a)):
            return orig(*a, **kw)
        pre = snap((a, kw))
        T.depth += 1
        try:
            out = orig(*a, **kw)
        finally:
            T.depth -= 1
        post = snap(out) if out is not None else snap((a, kw))
        name = op
        obj = a[0] if method else None
        if namer is not None:
            name = f'{op}:{namer(a)}'
        elif method and getattr(obj, 'name', None):
            name = f"{op}:{getattr(obj, 'name')}"
        key = f'r{T.rank}.{T.part}{T.stage if T.stage >= 0 else ""}.{name}'
        T.entries.append(Entry(f'{key}#{T.counts[key]}', op, orig, pre, post, obj))
        T.counts[key] += 1
        return out

    setattr(owner, attr, wrapper)


def install(T: Tracer, model: Any) -> None:
    import torch
    from offload.dspark.worker import fused as FU, fused_e as FE, tp2 as TP
    from offload.dspark.native import ops as NO, tp2_native as TN, kx as KX

    for name in ('hc_mixes', 'collapse_norm', 'collapse', 'hc_post', 'rms_norm', 'rope', 'kv_rope_qdq',
                 'block_attention_ring', 'wo_a_fp8', 'kv_store', 'prescale_fp16', 'scaled_rms_norm'):
        instrument(T, FU, name, name)
    for name in ('collapse_norm2', 'route_fused', 'swiglu_fp16', 'rms_norm_fp16'):
        instrument(T, FE, name, name)
    for name in ('act_qdq', 'swiglu_bf16', 'markov_logits_fp32', 'moe_combine_official', 'ordered_slot_sum',
                 'swiglu_qdq'):
        instrument(T, NO, name, name)
    instrument(T, TN, 'attention_into', 'attention_into')
    for name in ('coop_with_shared', 'markov_local_argmax', 'markov_partials'):
        instrument(T, TP, name, name)
    instrument(T, KX.KXSelectiveDense, '__call__', 'dense', method=True)
    instrument(T, KX.KXSharedPair, '__call__', 'shared_pair', method=True)
    instrument(T, NO.NativeDense, '__call__', 'dense', method=True)
    instrument(T, KX.KXFusedBank, 'forward_slots', 'experts', method=True)
    instrument(T, KX.KXRegisterBank, 'forward_slots', 'experts', method=True)
    instrument(T, TP.Exl3Mat, 'gemm', 'exl3', method=True, skip=lambda a: a[0].fixed_bucket == 'head')
    instrument(T, TP.Exl3Pair, '__call__', 'exl3_pair', method=True)
    instrument(T, TP.EmuComm, 'all_reduce', 'all_reduce', method=True)
    instrument(T, TP.EmuComm, 'argmax', 'argmax', method=True)
    instrument(T, TP.MarkovShard, 'embed', 'markov_embed', method=True)
    instrument(T, torch, 'sigmoid', 'confidence_sigmoid', skip=lambda a: T.part != 'head')
    try:
        from offload.dspark.kernels.fp8dense import epilogue as FPE
        instrument(T, FPE, 'shared_activation_into', 'shared_activation')
    except ImportError:
        pass
    try:
        from offload.dspark.kernels.fusedmoe import epilogue as MOE
        instrument(T, MOE, 'combine_official_into', 'combine_official')
    except ImportError:
        pass

    cls = type(model)
    if hasattr(cls, '_fixed_rows'):        # NB1 level 2: one sub-batched projection = one operation
        instrument(T, cls, '_fixed_rows', 'fixed_rows', method=True, namer=lambda a: str(getattr(a[1], 'name', '')))

    def context(attr: str, part: str, stage: Callable[..., int] | None) -> None:
        orig = getattr(cls, attr)

        @functools.wraps(orig)
        def wrapper(self: Any, *a: Any, **kw: Any) -> Any:
            old = (T.part, T.stage)
            T.part = part
            if stage is not None:
                T.stage = stage(*a)
            try:
                return orig(self, *a, **kw)
            finally:
                T.part = old[0] if part == 'head' else 'mid'
                if part == 'head':
                    T.stage = old[1]
        setattr(cls, attr, wrapper)

    context('_attn_part', 'attn', lambda rk, i, *_: i)
    context('_ffn_part', 'ffn', lambda rk, st, *_: next(j for j, s in enumerate(rk.stages) if s is st))
    context('heads', 'head', lambda *_: -1)
    orig_each = cls._each

    def each(self: Any, R: Iterable[int], fn: Callable[[Any], Any]) -> dict[int, Any]:
        out = {}
        for r in R:
            old = T.rank
            T.rank = r
            try:
                out.update(orig_each(self, (r,), fn))
            finally:
                T.rank = old
        return out
    cls._each = each


# ------------------------------------------------------------------------------------------------- comparison


@dataclass
class Layout:
    """Which rows of a recorded tensor belong to which request. draft: tensors whose packed first dimension is B x
    the alone one (unit = alone rows). insert: tensors whose first dimension is the graph bucket (unit = n rows)."""
    kind: str
    B: int
    n: int = 0
    ba: int = 0
    bp: int = 0

    def unit(self, a: Any, p: Any) -> int:
        if a.ndim == 0 or p.ndim != a.ndim or tuple(a.shape[1:]) != tuple(p.shape[1:]):
            return 0
        if self.kind == 'draft':
            return int(a.shape[0]) if a.shape[0] >= 1 and p.shape[0] == self.B * a.shape[0] else 0
        return self.n if a.shape[0] == self.ba and p.shape[0] == self.bp else 0


@dataclass
class Stat:
    comparisons: int = 0
    exact: int = 0
    elements: int = 0
    differing: int = 0
    max_abs: float = 0.0
    sum_mean_abs: float = 0.0
    max_norm_rel: float = 0.0

    def add(self, a: Any, p: Any) -> bool:
        import torch
        self.comparisons += 1
        self.elements += a.numel()
        if a.dtype.is_floating_point:
            same = bool(torch.equal(a.view(torch.uint8) if a.is_contiguous() else a.contiguous().view(torch.uint8),
                                    p.contiguous().view(torch.uint8)))
        else:
            same = bool(torch.equal(a, p))
        if same:
            self.exact += 1
            return True
        d = (a.double() - p.double()).abs()
        self.differing += int((d > 0).sum())
        mx = float(d.max())
        self.max_abs = max(self.max_abs, mx)
        self.sum_mean_abs += float(d.mean())
        scale = float(a.double().abs().max())
        self.max_norm_rel = max(self.max_norm_rel, mx / scale if scale > 0 else float('inf'))
        return False

    def json(self) -> dict[str, float | int]:
        return dict(comparisons=self.comparisons, exact=self.exact, elements=self.elements,
                    differing_elements=self.differing, max_abs=self.max_abs,
                    mean_abs_over_inexact=self.sum_mean_abs / max(1, self.comparisons - self.exact),
                    max_abs_over_alone_max=self.max_norm_rel)


def leaves(x: Any, path: str = '') -> list[tuple[str, Any]]:
    import torch
    if isinstance(x, torch.Tensor):
        return [(path, x)]
    if isinstance(x, dict):
        return [item for k, v in x.items() for item in leaves(v, f'{path}.{k}')]
    if isinstance(x, (list, tuple)):
        return [item for i, v in enumerate(x) for item in leaves(v, f'{path}[{i}]')]
    return []


def compare_rows(layout: Layout, alone: Any, packed: Any, offset: int, stat: Stat) -> bool:
    """Compare every row-owned tensor leaf of ``packed`` (request at ``offset``) with ``alone``; True if exact."""
    al, pl = leaves(alone), leaves(packed)
    if [k for k, _ in al] != [k for k, _ in pl]:
        raise RuntimeError('recorded output structures differ between alone and packed calls')
    exact = True
    for (_, a), (_, p) in zip(al, pl):
        u = layout.unit(a, p)
        if u:
            exact &= stat.add(a[:u], p[offset * u:(offset + 1) * u])
    return exact


def replay_args(layout: Layout, packed: Any, alones: list[Any], order: tuple[int, ...]) -> Any:
    """Packed-call arguments with every row-owned tensor rebuilt from the requests' ALONE arguments."""
    import torch
    if isinstance(packed, torch.Tensor):
        a0 = alones[0]
        u = layout.unit(a0, packed)
        if not u:
            return packed
        out = packed.clone()
        for o, alone in enumerate(alones):
            out[o * u:(o + 1) * u] = alone[:u]
        return out
    if isinstance(packed, dict):
        return {k: replay_args(layout, v, [a[k] for a in alones], order) for k, v in packed.items()}
    if isinstance(packed, (list, tuple)):
        return type(packed)(replay_args(layout, v, [a[i] for a in alones], order) for i, v in enumerate(packed))
    return packed


def replay(T: Tracer, entry: Entry, alones: list[Entry], layout: Layout, order: tuple[int, ...]) -> Any:
    args, kwargs = replay_args(layout, entry.pre, [e.pre for e in alones], order)
    out = entry.fn(*args, **kwargs)
    return snap(out) if out is not None else snap((args, kwargs))


# ------------------------------------------------------------------------------------------------- FP64 references


def fp64_reference(model: Any, entry: Entry) -> list[Any] | None:
    """FP64 reference outputs (list aligned with leaves(entry.out)) of one ALONE call, or None when the operation
    has no reference here. Dense: exact-decoded weights, FP64 dot. Experts: official rounding points (bf16 g/u,
    SwiGLU, bf16 h, act QDQ) with FP64 dots. EXL3: reconstructed trellis, FP64 Hadamard rotations and dot."""
    import torch
    from offload.dspark.native import kx as KX, ops as NO, formats as F, reference as REF
    from offload.dspark.worker import tp2 as TP
    (args, kwargs) = entry.pre
    obj = entry.obj
    if entry.op == 'fixed_rows':             # (drafter, linear, x): reference the linear on the whole input
        obj, args = args[1], (args[1], args[2])

    def dense(w: Any, x: Any, groups: int) -> Any:
        M = x.shape[0]
        n_g, k_g = w.shape[0] // groups, w.shape[1]
        xg = x.double().view(M, groups, k_g)
        wg = w.double().view(groups, n_g, k_g)
        return torch.einsum('mgk,gnk->mgn', xg, wg).reshape(M, groups * n_g)

    if isinstance(obj, KX.KXSelectiveDense):
        return [dense(obj.baseline.w, args[1], obj.candidate.groups)]
    if isinstance(obj, NO.NativeDense):
        if obj.kind != 'bf16':
            return None
        return [dense(obj.w, args[1], 1)]
    if isinstance(obj, KX.KXSharedPair):
        return [dense(obj.gate.baseline.w, args[1], 1), dense(obj.up.baseline.w, args[1], 1)]
    if isinstance(obj, KX.KXFusedBank):
        xq, ids, rw = args[1], args[2], args[3]
        stage, rank = int(obj.name.split('.')[1]), int(obj.name.split('[r')[1][0])
        tp, I_r = 2, obj.I
        out = torch.zeros(xq.shape[0], 3, obj.H, dtype=torch.float64, device=xq.device)
        sh = model.sh
        for e in sorted(set(ids.flatten().tolist())):
            p = f'mtp.{stage}.ffn.experts.{e}.'
            rows = slice(rank * I_r, (rank + 1) * I_r)
            dev = xq.device
            w1 = F.decode_fp4(sh.get(p + 'w1.weight')[rows].to(dev), sh.get(p + 'w1.scale')[rows].to(dev)).double()
            w3 = F.decode_fp4(sh.get(p + 'w3.weight')[rows].to(dev), sh.get(p + 'w3.scale')[rows].to(dev)).double()
            w2 = F.decode_fp4(sh.get(p + 'w2.weight')[:, rank * I_r // 2:(rank + 1) * I_r // 2].contiguous().to(dev),
                              sh.get(p + 'w2.scale')[:, rank * I_r // 32:(rank + 1) * I_r // 32].contiguous().to(dev)
                              ).double()
            t, j = torch.where(ids == e)
            x = xq[t].double()
            g16 = (x @ w1.t()).float().to(torch.bfloat16)
            u16 = (x @ w3.t()).float().to(torch.bfloat16)
            h = REF.swiglu(g16, u16, rw[t, j].float().unsqueeze(1))
            hq = F.act_qdq(h)
            out[t, j] = hq.double() @ w2.t()
            del w1, w3, w2
        return [out]
    if isinstance(obj, TP.Exl3Mat):
        return [exl3_reference(obj, args[1])]
    return None


def exl3_reference(mat: Any, x16: Any) -> Any:
    import torch
    from exllamav3.util.hadamard import get_hadamard_dt
    import exllamav3_ext as stock
    w = torch.empty(mat.in_features, mat.out_features, dtype=torch.half, device=x16.device)
    stock.reconstruct(w, mat.trellis, mat.k, False, True)
    had = get_hadamard_dt(128, x16.device, torch.float64, 1.0 / 128 ** 0.5)
    M = x16.shape[0]
    xh = ((x16.double() * mat.suh.double()).view(M, -1, 128) @ had).view(M, -1)
    y = xh @ w.double()
    return ((y.view(M, -1, 128) @ had).view(M, -1)) * mat.svh.double()


def label_fp64(alone: list[Any], packed: list[Any], ref: list[Any], unit_rows: list[int], offset: int
               ) -> dict[str, float | bool]:
    gap_max = gap_mean = ea_max = ea_mean = ep_max = ep_mean = 0.0
    n = 0
    for a, p, r, u in zip(alone, packed, ref, unit_rows):
        if not u:
            continue
        a64, p64, r64 = a[:u].double(), p[offset * u:(offset + 1) * u].double(), r[:u].double()
        gap, ea, ep = (a64 - p64).abs(), (a64 - r64).abs(), (p64 - r64).abs()
        gap_max, ea_max, ep_max = max(gap_max, float(gap.max())), max(ea_max, float(ea.max())), max(ep_max, float(ep.max()))
        gap_mean += float(gap.mean())
        ea_mean += float(ea.mean())
        ep_mean += float(ep.mean())
        n += 1
    n = max(n, 1)
    return dict(gap_max=gap_max, gap_mean=gap_mean / n, alone_err_max=ea_max, alone_err_mean=ea_mean / n,
                packed_err_max=ep_max, packed_err_mean=ep_mean / n,
                within_own_error=gap_max <= ea_max, packed_not_worse=ep_mean <= ea_mean * (1 + 1e-9))


# ------------------------------------------------------------------------------------------------- main probe


def depth(conf: Any, cap: int) -> int:
    import torch
    cost = torch.tensor(ROUND_MS) + DRAFT_MS
    v = conf.double().cpu().clone()
    v[cap:] = 0
    expected = torch.cat((torch.ones(1, dtype=torch.float64), 1 + v.cumprod(0).cumsum(0)))
    return int((expected / cost.double()).argmax())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=('k2-tp2', 'native-hybrid-kx2'))
    parser.add_argument('--runtime', type=Path, default=Path.home() / 'dsv41-offload/n4')
    parser.add_argument('--repo', type=Path, default=None, help='source tree (default: RUNTIME/repo)')
    parser.add_argument('--scratch', type=Path, default=Path('/opt/dsv41/x86host-data/nb1'))
    parser.add_argument('--root', default='/opt/dsv41/x86host-data/dspark')
    parser.add_argument('--fixtures', choices=('all', 'synthetic', 'captured'), default='all')
    parser.add_argument('--batches', default='2,3,4')
    parser.add_argument('--fp64-per-cell', type=int, default=2)
    parser.add_argument('--no-insert', action='store_true')
    parser.add_argument('--insert-rows', default='1,2,5', help='rows per request (packed B with n x B <= 128)')
    parser.add_argument('--fp64-alone-insert', action='store_true',
                        help='FP64 error of every alone (B1) insert projection, by op and rows')
    args = parser.parse_args()
    repo = args.repo or args.runtime / 'repo'
    sys.path.insert(0, str(repo))
    binaries = bootstrap(args.runtime, args.scratch)
    import torch
    from offload.dspark.worker.checkpoint import Checkpoint
    from offload.dspark.worker.tp2 import check_visible
    from offload.dspark.native.tp2_native import build

    check_visible(pair=False)
    torch.set_num_threads(4)
    start = time.monotonic()
    batches = tuple(int(b) for b in args.batches.split(','))
    print('PHASE load ' + args.arm + ' invariant=' + os.environ.get('DSPARK_NATIVE_BATCH_INVARIANT', '<unset>'),
          flush=True)
    dev = torch.device('cuda', 0)
    model = build(args.arm, [dev, dev], Checkpoint(args.root), slots=4)
    print(f'PHASE loaded seconds={time.monotonic() - start:.1f}', flush=True)
    model.capture(batch_sizes=(1, 2, 3, 4), ks=(K,), insert_rows=(8, 16, 32, 64, 128), n_outs=(4, 5), margins=False)
    print('PHASE captured', flush=True)
    T = Tracer()
    install(T, model)

    fixtures: list[tuple[str, Callable[[], torch.Tensor]]] = []
    generator = torch.Generator().manual_seed(20261024)       # N6's synthetic draws for slots 0..2
    extra = torch.Generator().manual_seed(20261026)           # NB1's fourth slot
    synthetic = []
    for case, (lasts, last4) in enumerate((((0, 63, 127), 31), ((128, 300, 32768), 65535),
                                          ((150000, 32000, 1000), 262143))):
        mxs = []
        for last in lasts:
            pos = torch.arange(max(0, last - 127), last + 1)
            mxs.append((pos, torch.randn(len(pos), model.s.hidden, generator=generator).to(torch.bfloat16)))
        anchors = torch.randint(100, 10000, (3,), generator=generator)
        pos4 = torch.arange(max(0, last4 - 127), last4 + 1)
        mxs.append((pos4, torch.randn(len(pos4), model.s.hidden, generator=extra).to(torch.bfloat16)))
        anchors = torch.cat((anchors, torch.randint(100, 10000, (1,), generator=extra)))
        all_lasts = torch.tensor(list(lasts) + [last4])
        synthetic.append((case, mxs, anchors, all_lasts))

    def load_synthetic(item: tuple[int, list[Any], Any, Any]) -> Any:
        _, mxs, anchors, lasts = item
        model.reset_slots([0, 1, 2, 3])
        for slot, (pos, mx) in enumerate(mxs):
            model.insert_mx_eager(torch.full_like(pos, slot), pos, mx)
        return torch.stack((torch.arange(4), anchors, lasts, (lasts - 127).clamp_min(0)))

    if args.fixtures in ('all', 'synthetic'):
        for item in synthetic:
            fixtures.append((f'synthetic-{item[0]}', functools.partial(load_synthetic, item)))
    captures = None
    capture_taps: dict[int, list[Any]] = {}
    if args.fixtures in ('all', 'captured'):
        sys.path.insert(0, str(args.runtime / 'verify-source'))
        from verify.captures import CaptureSet
        captures = CaptureSet(args.root + '/captures/A-capture-20261022/dspark-captures')
        samples = [s for s in captures.samples if s.b == 0 and 8 <= s.round <= 19]
        if len(samples) != 12:
            raise RuntimeError('expected all twelve N4 V1 samples')

        def load_capture(group: list[Any], gi: int) -> Any:
            model.reset_slots([0, 1, 2, 3])
            taps_rows = []
            for slot, sample in enumerate(group):
                arm_in = captures.arm_inputs(sample)
                if {kind for _, kind, _ in arm_in['window']} - {'engine_mx', 'taps'}:
                    raise RuntimeError('unknown capture window source')
                for source in ('engine_mx', 'taps'):
                    rows = [(p, row) for p, kind, row in arm_in['window'] if kind == source]
                    if source == 'taps':
                        taps_rows.append([row for _, row in rows[-5:]])
                    for a in range(0, len(rows), 128):
                        chunk = rows[a:a + 128]
                        positions = torch.tensor([p for p, _ in chunk]).pin_memory()
                        slots = torch.full_like(positions, slot).pin_memory()
                        values = torch.stack([row for _, row in chunk]).pin_memory()
                        if source == 'taps':
                            model.insert_taps_graphed(slots, positions, values)
                        else:
                            model.insert_mx_eager(slots, positions, values)
            capture_taps[gi] = taps_rows
            return torch.tensor([[0, 1, 2, 3], [s.anchor for s in group], [s.P for s in group],
                                 [max(0, s.P - 127) for s in group]])

        for gi in range(3):
            fixtures.append((f'capture-group-{gi}', functools.partial(load_capture, samples[4 * gi:4 * gi + 4], gi)))

    # ---- aggregates
    e2e: dict[tuple[str, str, int], Stat] = defaultdict(Stat)
    iso: dict[tuple[str, str, int], Stat] = defaultdict(Stat)
    site_order: dict[str, int] = {}
    site_op: dict[str, str] = {}
    first_div: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    first_iso: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    fp64_cells: list[dict[str, Any]] = []
    fp64_done: Counter[tuple[str, str, int]] = Counter()
    final: dict[tuple[str, int, str], dict[str, Any]] = {}
    depth_changes: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    same_b: dict[tuple[int, int], tuple[int, int, float]] = {}
    b1_digests: dict[str, dict[str, str]] = {}
    conf_iso: dict[tuple[str, int], Stat] = defaultdict(Stat)
    head_iso: dict[tuple[str, int], Stat] = defaultdict(Stat)
    conf_fp64: list[dict[str, Any]] = []
    graph_eager = [0, 0]
    digest = hashlib.sha256()          # every eager/graph output and inserted window, in a fixed order

    def kind_of(label: str) -> str:
        return 'captured' if label.startswith('capture') else 'synthetic'

    def run_traced(inp: Any, part: str = 'block') -> tuple[list[Entry], Any, Any]:
        T.reset(part)
        T.on = True
        try:
            ids, conf, _, _ = model.draft(inp.contiguous().pin_memory(), K, n_out=K)
        finally:
            T.on = False
        torch.cuda.synchronize()
        return T.entries, ids.cpu().clone(), conf.cpu().clone()

    def head_rows(entries: list[Entry]) -> dict[int, Any]:
        rows = {}
        for e in entries:
            if '.head.' in e.label and e.op == 'rms_norm':
                rows[int(e.label[1:e.label.index('.')])] = e.out
        return rows

    def head_logits(rk: Any, rows: Any, B: int) -> Any:
        native = bool(getattr(getattr(model, 'cfg', None), 'native', False))
        x = rows.view(B, K, -1)[:, :K].reshape(B * K, -1)
        if native and rk.head_bf16 is not None:
            return rk.head_bf16(x.contiguous(), out_f32=True)
        x16 = x.to(torch.float16)
        out = torch.empty(B * K, rk.v1 - rk.v0, dtype=torch.float32, device=rk.dev)
        for c in range(0, B * K, model.HEAD_CHUNK):
            rk.head.gemm(x16[c:c + model.HEAD_CHUNK], out=out[c:c + model.HEAD_CHUNK])
        return out if native else out.to(torch.bfloat16)

    def conf_feat(entries: list[Entry]) -> Any:
        hh = [e for e in entries if e.op == 'collapse' and e.label.startswith('r0.')][-1].out
        emb = [e for e in entries if e.op == 'markov_embed'][-1].out
        B = emb.shape[0]
        return torch.cat((hh.view(B, K, -1), emb), dim=-1).float()

    for label, loader in fixtures:
        inputs = loader()
        kind = kind_of(label)
        before = [(rank.ring_pos.clone(), rank.ring_kv.clone()) for rank in model.ranks]
        alone: dict[int, tuple[list[Entry], Any, Any]] = {}
        alone_graph: dict[tuple[int, int], tuple[Any, Any]] = {}
        for r in range(4):
            col = inputs[:, r:r + 1]
            alone[r] = run_traced(col)
            for n in (5, 4):
                gi, gc, _ = model.draft_graphed(col.contiguous().pin_memory(), K, n_out=n)
                alone_graph[(r, n)] = (gi.cpu().clone(), gc.cpu().clone())
            ids_e, conf_e = alone[r][1], alone[r][2]
            for t in (ids_e, conf_e, *alone_graph[(r, 5)], *alone_graph[(r, 4)]):
                digest.update(t.numpy().tobytes())
            b1_digests[f'{label}/{r}'] = dict(
                eager_ids=sha(ids_e.numpy().tobytes()), eager_conf=sha(conf_e.numpy().tobytes()),
                graph5_ids=sha(alone_graph[(r, 5)][0].numpy().tobytes()),
                graph5_conf=sha(alone_graph[(r, 5)][1].numpy().tobytes()),
                graph4_ids=sha(alone_graph[(r, 4)][0].numpy().tobytes()),
                graph4_conf=sha(alone_graph[(r, 4)][1].numpy().tobytes()))
            graph_eager[0] += 1
            graph_eager[1] += int(torch.equal(ids_e, alone_graph[(r, 5)][0]) and
                                  torch.equal(conf_e, alone_graph[(r, 5)][1]))
        for i, e in enumerate(alone[0][0]):
            site_order.setdefault(e.label, i)
            site_op.setdefault(e.label, e.op)
        same_shape_seen: dict[tuple[int, int, int], tuple[Any, Any]] = {}
        for B in batches:
            orders = list(itertools.permutations(range(4), B))
            for order in orders:
                inp = inputs[:, list(order)]
                entries, ids_p, conf_p = run_traced(inp)
                if len(entries) != len(alone[order[0]][0]):
                    raise RuntimeError(f'{label} B{B}: {len(entries)} recorded ops vs {len(alone[order[0]][0])} alone')
                layout = Layout('draft', B)
                first_e2e = first_isolated = None
                for idx, entry in enumerate(entries):
                    al = [alone[r][0][idx] for r in order]
                    if any(a.label != entry.label for a in al):
                        raise RuntimeError(f'op sequence differs at {idx}: {entry.label} vs {al[0].label}')
                    exact_e2e = True
                    for o, a in enumerate(al):
                        exact_e2e &= compare_rows(layout, a.out, entry.out, o, e2e[(entry.label, kind, B)])
                    if not exact_e2e and first_e2e is None:
                        first_e2e = entry.label
                    rep = replay(T, entry, al, layout, order)
                    exact_iso = True
                    for o, a in enumerate(al):
                        exact_iso &= compare_rows(layout, a.out, rep, o, iso[(entry.label, kind, B)])
                    if not exact_iso:
                        if first_isolated is None:
                            first_isolated = entry.label
                        key = (entry.label, kind, B)
                        if fp64_done[key] < args.fp64_per_cell:
                            for o, a in enumerate(al):
                                ref = fp64_reference(model, a)
                                if ref is None:
                                    continue
                                a_leaves = [t for _, t in leaves(a.out)]
                                p_leaves = [t for _, t in leaves(rep)]
                                units = [layout.unit(x, y) for x, y in zip(a_leaves, p_leaves)]
                                fp64_cells.append(dict(label=entry.label, op=entry.op, kind=kind, B=B, offset=o,
                                                       **label_fp64(a_leaves, p_leaves, ref, units, o)))
                            fp64_done[key] += 1
                first_div[(kind, B)][first_e2e or '<none>'] += 1
                first_iso[(kind, B)][first_isolated or '<none>'] += 1
                # head and confidence matmul, isolated at the packed shape on alone inputs
                rows_alone = [head_rows(alone[r][0]) for r in order]
                for rk in model.ranks:
                    packed_rows = torch.cat([ra[rk.r] for ra in rows_alone])
                    lp = head_logits(rk, packed_rows, B)
                    for o, r in enumerate(order):
                        la = head_logits(rk, rows_alone[o][rk.r], 1)
                        head_iso[(kind, B)].add(la, lp[o * K:(o + 1) * K])
                feats = [conf_feat(alone[r][0]) for r in order]
                zp = torch.cat(feats) @ model.ranks[0].conf_w
                for o, f in enumerate(feats):
                    za = f @ model.ranks[0].conf_w
                    if not conf_iso[(kind, B)].add(za, zp[o:o + 1]):
                        ref = f.double() @ model.ranks[0].conf_w.double()
                        conf_fp64.append(dict(kind=kind, B=B, **label_fp64([za], [zp], [ref], [1], o)))
                # final outputs: eager and graphs, n_out 5/4, depth policy
                for n in (5, 4):
                    gi, gc, _ = model.draft_graphed(inp.contiguous().pin_memory(), K, n_out=n)
                    gi, gc = gi.cpu().clone(), gc.cpu().clone()
                    digest.update(gi.numpy().tobytes())
                    digest.update(gc.numpy().tobytes())
                    if n == 5:
                        digest.update(ids_p.numpy().tobytes())
                        digest.update(conf_p.numpy().tobytes())
                        graph_eager[0] += 1
                        graph_eager[1] += int(torch.equal(gi, ids_p) and torch.equal(gc, conf_p))
                    for o, r in enumerate(order):
                        ai, ac = alone_graph[(r, n)]
                        cell = final.setdefault((kind, B, f'graph{n}'), dict(comparisons=0, ids_exact=[0] * n,
                                                                             conf_exact=0, conf_max_abs=0.0))
                        cell['comparisons'] += 1
                        for j in range(n):
                            cell['ids_exact'][j] += int(gi[o, j] == ai[0, j])
                        cell['conf_exact'] += int(torch.equal(gc[o, :n], ac[0, :n]))
                        cell['conf_max_abs'] = max(cell['conf_max_abs'], float((gc[o, :n] - ac[0, :n]).abs().max()))
                        if n == 5:
                            cap = CAP[B]
                            da, dp = depth(ac[0], cap), depth(gc[o], cap)
                            depth_changes[(kind, B)][f'{da}->{dp}'] += 1
                        k3 = (B, n, r)
                        if k3 in same_shape_seen:
                            pi, pc = same_shape_seen[k3]
                            prev = same_b.get((B, n), (0, 0, 0.0))
                            same_b[(B, n)] = (prev[0] + 1, prev[1] + int(torch.equal(pi, gi[o]) and
                                                                      torch.equal(pc, gc[o])),
                                              max(prev[2], float((pc - gc[o]).abs().max())))
                        else:
                            same_shape_seen[k3] = (gi[o].clone(), gc[o].clone())
                    if n == 5:
                        for o, r in enumerate(order):
                            cell = final.setdefault((kind, B, 'eager5'), dict(comparisons=0, ids_exact=[0] * 5,
                                                                              conf_exact=0, conf_max_abs=0.0))
                            ai, ac = alone[r][1], alone[r][2]
                            cell['comparisons'] += 1
                            for j in range(5):
                                cell['ids_exact'][j] += int(ids_p[o, j] == ai[0, j])
                            cell['conf_exact'] += int(torch.equal(conf_p[o], ac[0]))
                            cell['conf_max_abs'] = max(cell['conf_max_abs'], float((conf_p[o] - ac[0]).abs().max()))
                del entries
            print(f'PHASE {label} B{B} done ({len(orders)} placements) t={time.monotonic() - start:.0f}s', flush=True)
        for rank, (pos, kv) in zip(model.ranks, before):
            if not (torch.equal(pos, rank.ring_pos) and torch.equal(kv, rank.ring_kv)):
                raise RuntimeError(f'{label}: window storage changed during proposals')
        del alone

    # ---- insertion: graph-bucket padded eager inserts, alone vs packed
    ins_e2e: dict[tuple[str, int, int], Stat] = defaultdict(Stat)
    ins_iso: dict[tuple[str, int, int], Stat] = defaultdict(Stat)
    ins_first: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    ins_ring: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0, 0])
    ins_graph = [0, 0]
    ins_fp64: list[dict[str, Any]] = []
    ins_fp64_done: Counter[tuple[str, int, int]] = Counter()
    alone_fp64: dict[str, list[float]] = {}
    if not args.no_insert:
        sizes = sorted(model.insert_graphs)
        hid = model.s.hidden * len(model.s.taps)
        tap_sources: list[tuple[str, list[Any]]] = []
        gen = torch.Generator().manual_seed(20261027)
        insert_rows = [int(x) for x in args.insert_rows.split(',')]
        ids = torch.randint(100, 10000, (4, max(insert_rows), len(model.s.taps)), generator=gen)
        emb = model.ranks[0].embed
        synth = [[emb[ids[r, j].to(emb.device)].reshape(-1).cpu() for j in range(max(insert_rows))]
                 for r in range(4)]
        tap_sources.append(('synthetic', synth))
        if capture_taps:
            rows = [r for g in sorted(capture_taps) for r in capture_taps[g] if len(r) >= 5]
            if len(rows) >= 4:
                tap_sources.append(('captured', [r[:5] for r in rows[:4]]))

        def insert_traced(slots: Any, pos: Any, taps: Any) -> list[Entry]:
            n = slots.shape[0]
            b = next(x for x in sizes if x >= n)
            ps = torch.full((b,), model.n_slots, dtype=torch.int64)
            pp = torch.zeros(b, dtype=torch.int64)
            pt = torch.zeros(b, hid, dtype=torch.bfloat16)
            ps[:n], pp[:n], pt[:n] = slots, pos, taps
            T.reset('insert')
            T.on = True
            try:
                model.insert_eager(ps.pin_memory(), pp.pin_memory(), pt.pin_memory())
            finally:
                T.on = False
            torch.cuda.synchronize()
            return T.entries

        for src, taps_by_req in tap_sources:
            for n in insert_rows:
                if any(len(t) < n for t in taps_by_req):
                    continue
                req_pos = [torch.arange(300 + 1000 * r, 300 + 1000 * r + n) for r in range(4)]
                req_taps = [torch.stack(taps_by_req[r][:n]).to(torch.bfloat16) for r in range(4)]
                alone_ins: dict[int, list[Entry]] = {}
                alone_ring: dict[int, list[tuple[Any, Any]]] = {}
                for r in range(4):
                    model.reset_slots([0, 1, 2, 3])
                    alone_ins[r] = insert_traced(torch.full((n,), r), req_pos[r], req_taps[r])
                    alone_ring[r] = [(rk.ring_pos[r].clone(), rk.ring_kv[r].clone()) for rk in model.ranks]
                    if args.fp64_alone_insert:
                        for e in alone_ins[r]:
                            ref = fp64_reference(model, e)
                            if ref is None:
                                continue
                            for (_, out), rf in zip(leaves(e.out), ref):
                                err = (out[:n].double() - rf[:n]).abs()
                                acc = alone_fp64.setdefault(f'{e.label.split("#")[0]}/n{n}/{src}', [0, 0.0, 0.0, 0.0])
                                acc[0] += 1
                                acc[1] += float(err.mean())
                                acc[2] = max(acc[2], float(err.max()))
                                acc[3] += float(rf[:n].abs().mean())
                ba = next(x for x in sizes if x >= n)
                for B in batches:
                    if B == 1 or n * B > sizes[-1]:
                        continue
                    for order in itertools.permutations(range(4), B):
                        slots = torch.cat([torch.full((n,), r) for r in order])
                        pos = torch.cat([req_pos[r] for r in order])
                        taps = torch.cat([req_taps[r] for r in order])
                        model.reset_slots([0, 1, 2, 3])
                        entries = insert_traced(slots, pos, taps)
                        bp = next(x for x in sizes if x >= n * B)
                        layout = Layout('insert', B, n, ba, bp)
                        if len(entries) != len(alone_ins[order[0]]):
                            raise RuntimeError('insert op sequences differ')
                        first = None
                        for idx, entry in enumerate(entries):
                            al = [alone_ins[r][idx] for r in order]
                            exact = True
                            for o, a in enumerate(al):
                                exact &= compare_rows(layout, a.out, entry.out, o, ins_e2e[(entry.label, n, B)])
                            if not exact and first is None:
                                first = entry.label
                            rep = replay(T, entry, al, layout, order)
                            exact_iso = True
                            for o, a in enumerate(al):
                                exact_iso &= compare_rows(layout, a.out, rep, o, ins_iso[(entry.label, n, B)])
                            ikey = (entry.label, n, B)
                            if not exact_iso and ins_fp64_done[ikey] < args.fp64_per_cell:
                                for o, a in enumerate(al):
                                    ref = fp64_reference(model, a)
                                    if ref is None:
                                        continue
                                    a_leaves = [t for _, t in leaves(a.out)]
                                    p_leaves = [t for _, t in leaves(rep)]
                                    units = [layout.unit(x, y) for x, y in zip(a_leaves, p_leaves)]
                                    ins_fp64.append(dict(label=entry.label, op=entry.op, source=src, n=n, B=B,
                                                         offset=o, **label_fp64(a_leaves, p_leaves, ref, units, o)))
                                ins_fp64_done[ikey] += 1
                        ins_first[(n, B)][f'{src}:{first or "<none>"}'] += 1
                        for o, r in enumerate(order):
                            for rk, (p0, kv0) in zip(model.ranks, alone_ring[r]):
                                ring_cell = ins_ring[(n, B)]
                                ring_cell[0] += 1
                                ring_cell[1] += int(torch.equal(p0, rk.ring_pos[r]) and
                                                    torch.equal(kv0, rk.ring_kv[r]))
                                ring_cell[2] = max(ring_cell[2], int((kv0 != rk.ring_kv[r]).sum()))
                        eager_rings = [(rk.ring_pos.clone(), rk.ring_kv.clone()) for rk in model.ranks]
                        model.reset_slots([0, 1, 2, 3])
                        model.insert_taps_graphed(slots.pin_memory(), pos.pin_memory(), taps.pin_memory())
                        for rk in model.ranks:
                            digest.update(rk.ring_pos.cpu().numpy().tobytes())
                            digest.update(rk.ring_kv.cpu().view(torch.int16).numpy().tobytes())
                        ins_graph[0] += 1
                        ins_graph[1] += int(all(torch.equal(p, rk.ring_pos) and torch.equal(kv, rk.ring_kv)
                                                for rk, (p, kv) in zip(model.ranks, eager_rings)))
                print(f'PHASE insert {src} n={n} done t={time.monotonic() - start:.0f}s', flush=True)

    dense_configs = {}
    for rk in model.ranks[:1]:
        objs = [('main_proj', rk.main_proj)]
        for i, st in enumerate(rk.stages):
            for nm in ('wq_a', 'wkv', 'wq_b', 'wo_b', 'wo_a_n', 'sh_w1', 'sh_w3', 'sh_w2'):
                objs.append((f'stage{i}.{nm}', getattr(st, nm, None)))
        for nm, obj in objs:
            cand = getattr(obj, 'candidate', None)
            if cand is None:
                continue
            dense_configs[nm] = dict(groups=cand.groups, outputs=cand.outputs, inputs=cand.inputs,
                                     fp8_rows={str(m): list(c) for m, c in obj.configs.items()})
    sources = {}
    for base in ('native', 'worker', 'kernels'):
        for path in sorted((repo / 'offload/dspark' / base).rglob('*')):
            if path.suffix in ('.py', '.cu', '.cuh', '.json') and '/tools/' not in str(path):
                sources[str(path.relative_to(repo))] = sha(path.read_bytes())

    def stats(d: dict[Any, Stat]) -> list[dict[str, Any]]:
        return [dict(key=list(k), **v.json()) for k, v in sorted(d.items(), key=lambda kv: (
            site_order.get(kv[0][0], 10 ** 6) if isinstance(kv[0][0], str) else 0, str(kv[0])))]

    result = dict(
        arm=args.arm, label='MEASURED', invariant_env=os.environ.get('DSPARK_NATIVE_BATCH_INVARIANT', '<unset>'),
        mode='both TP2 ranks emulated on assigned solo GPU', repo=str(repo), runtime=str(args.runtime),
        fixtures=[f for f, _ in fixtures], batches=list(batches), site_order=site_order, site_op=site_op,
        draft_e2e=stats(e2e), draft_isolated=stats(iso),
        first_divergent_e2e={f'{k}/B{b}': dict(c) for (k, b), c in first_div.items()},
        first_divergent_isolated={f'{k}/B{b}': dict(c) for (k, b), c in first_iso.items()},
        fp64=fp64_cells, head_isolated={f'{k}/B{b}': v.json() for (k, b), v in head_iso.items()},
        confidence_matmul_isolated={f'{k}/B{b}': v.json() for (k, b), v in conf_iso.items()},
        confidence_matmul_fp64=conf_fp64[:64],
        final={f'{k}/B{b}/{m}': v for (k, b, m), v in final.items()},
        depth_transitions={f'{k}/B{b}': dict(c) for (k, b), c in depth_changes.items()},
        same_batch_companion_order={f'B{b}/n{n}': dict(comparisons=v[0], exact=v[1], conf_max_abs=v[2])
                                    for (b, n), v in same_b.items()},
        graph_vs_eager=dict(checks=graph_eager[0], exact=graph_eager[1]),
        b1_digests=b1_digests, output_digest=digest.hexdigest(),
        insert_e2e=[dict(key=list(k), **v.json()) for k, v in ins_e2e.items()],
        insert_isolated=[dict(key=list(k), **v.json()) for k, v in ins_iso.items()],
        insert_first_divergent={f'n{n}/B{b}': dict(c) for (n, b), c in ins_first.items()},
        insert_ring={f'n{n}/B{b}': dict(comparisons=v[0], exact=v[1], max_differing_kv=v[2])
                     for (n, b), v in ins_ring.items()},
        insert_graph_vs_eager=dict(checks=ins_graph[0], exact=ins_graph[1]), insert_fp64=ins_fp64,
        insert_alone_fp64={k: dict(calls=int(v[0]), mean_abs_err=v[1] / v[0], max_abs_err=v[2],
                                   mean_abs_ref=v[3] / v[0]) for k, v in alone_fp64.items()},
        dense_configs=dense_configs, runtime_sources=sources, prebuilt_extensions_sha256=binaries,
        triton_compiled_keys=sorted(set(getattr(bootstrap, 'compiled', []))),
        torch_version=torch.__version__, seconds=time.monotonic() - start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(dev))
    print('RESULT_JSON ' + json.dumps(result), flush=True)
    if graph_eager[0] != graph_eager[1]:
        raise RuntimeError('graph/eager mismatch; inspect aggregate result')


if __name__ == '__main__':
    main()
