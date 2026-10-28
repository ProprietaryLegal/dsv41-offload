"""CEN3 in-image driver (parts a and c), staged to each Spark and run inside the production image with production down.

  python3 -m offload.census.cen3_bench plan                                   # CPU: what the window runs
  python3 -m offload.census.cen3_bench bw --mib 2048 --reps 20 --out /cen3-out/bw.json          # part (c), GPU
  python3 -m offload.census.cen3_bench pick-metrics --query q.txt --out m.json                 # CPU, in the container
  ncu ... python3 -m offload.census.cen3_bench ncu-target --group smoe-b1 --plan /plan.json --rank 0 --out t.json

(c) streaming bandwidth of one GB10: read (a full fp32 reduction), write (fill) and copy (read + write) of MIB-sized
buffers, `reps` timed repetitions each after 3 warm-ups, CUDA events. (a) Nsight Compute targets: each case is ONE
kernel call between cudaProfilerStart and cudaProfilerStop (ncu --profile-from-start off), after an unprofiled warm-up
call of the same case, on synthetic EXL3 banks with legal-v2a's real per-expert K plan (offload/kl1/smoe_bench.py's
construction: random trellis words, so bytes and time match real weights; no proprietary tensor is loaded). The driver
writes each profiled launch's exact LOGICAL bytes in launch order by kernel family (offload/kl1/trace_kernels.py
families smoe_gu, smoe_down, mrow, mrow_head; smoe_rot moves activations only and is left unassigned); the runner
turns them into the offload.census.bytes_model ratio manifest. Output: numbers only.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from offload.kl1 import shapes as S
from offload.kl1 import smoe_bench as KB

GROUPS = ('smoke', 'smoe-b1', 'smoe-b4', 'dense', 'head')
DENSE_ROWS = (4, 16, 20)
HEAD_ROWS = (1, 6)
SMOE_LAYER = 10                       # one plan layer's per-expert K (caches are flushed before every profiled kernel)
BW_OPS = ('read', 'write', 'copy')
# Physical byte counters, preferred first. DRAM counters when the GPU exposes them; otherwise L2 sectors (the report
# then says the physical figure is L2 traffic, not DRAM). Names are the ncu metric base names plus the .sum rollup.
READ_METRICS = ('dram__bytes_read.sum', 'dram__sectors_read.sum', 'lts__t_sectors_srcunit_tex_op_read.sum',
                'lts__t_sectors_op_read.sum')
WRITE_METRICS = ('dram__bytes_write.sum', 'dram__sectors_write.sum', 'lts__t_sectors_srcunit_tex_op_write.sum',
                 'lts__t_sectors_op_write.sum')
TIME_METRIC = 'gpu__time_duration.sum'


# ---------------------------------------------------------------- pure (CPU-tested)

def smoe_cases(group: str) -> list[KB.Case]:
    """B1 rows 3/4/6 x distinct 6/13/24 (KL1's inferred ~13 at B1 and the bracket), B4 rows 16/20 x 12/24 experts
    per request, four requests, independent and with half the experts shared (KL1 cases()' axes, fewer points: one
    call each)."""
    if group == 'smoe-b1':
        return [KB.Case(r, 1, u) for r in (3, 4, 6) for u in (6, 13, 24) if u <= S.TOP_K * r]
    if group == 'smoe-b4':
        return [KB.Case(r, 4, u, c) for r in (16, 20) for u in (12, 24) for c in (0, u // 2)]
    raise ValueError(f'{group} is not a smoe group')


def smoe_parts(ids: Sequence[Sequence[int]], bits: Sequence[tuple[float, float]], rx: int = 8,
               shared_k: float = S.SHARED_K) -> tuple[float, float]:
    """(gate|up, down) logical bytes of one smoe call: smoe_bench.layer_bytes split by launch (smoe_gu streams w1|w3 of
    every expert once per run of <= rx rows plus the shared expert's gate|up once per run of rx live rows; smoe_down
    the same for w2)."""
    rows_of: dict[int, int] = {}
    for row in ids:
        for e in row:
            rows_of[e] = rows_of.get(e, 0) + 1
    gu = down = 0.0
    for e, n in rows_of.items():
        k_gu, k_d = bits[e]
        runs = math.ceil(n / rx)
        gu += runs * S.expert_gu_bytes(k_gu)
        down += runs * S.expert_down_bytes(k_d)
    shared = S.shared_runs(len(ids), rx)
    return gu + shared * S.expert_gu_bytes(shared_k), down + shared * S.expert_down_bytes(shared_k)


@dataclass(frozen=True)
class DenseGroup:
    """One exl3_mrow launch of a decode layer: its linears as (n_in, n_out) at DENSE_K (smoe_bench.gpu_dense). With
    split_input each linear reads its own input slice (wo_a's head groups); otherwise all read the same input."""
    name: str
    linears: tuple[tuple[int, int], ...]
    split_input: bool = False

    @property
    def width(self) -> tuple[int, int]:
        n_in = sum(i for i, _ in self.linears) if self.split_input else self.linears[0][0]
        return n_in, sum(o for _, o in self.linears)

    @property
    def offsets(self) -> list[tuple[int, int]]:
        """(input column, output column) of each linear inside the group's input and output tensors."""
        out: list[tuple[int, int]] = []
        xc = oc = 0
        for i, o in self.linears:
            out.append((xc, oc))
            xc += i if self.split_input else 0
            oc += o
        return out

    @property
    def bytes(self) -> float:
        return sum(S.exl3_bytes(i, o, S.DENSE_K) for i, o in self.linears)


_WO_IN = S.HEADS * S.HEAD_DIM // S.O_GROUPS
DENSE_GROUPS = (
    DenseGroup('wq_a+wkv', ((S.H, S.Q_LORA), (S.H, S.KV_DIM))),
    DenseGroup('wq_b', ((S.Q_LORA, S.HEADS * S.HEAD_DIM // S.TP),)),
    DenseGroup('wo_a', tuple((_WO_IN, S.O_LORA) for _ in range(S.O_GROUPS // S.TP)), split_input=True),
    DenseGroup('wo_b', ((S.O_GROUPS * S.O_LORA // S.TP, S.H),)),
)
HEAD_BYTES = S.exl3_bytes(S.H, S.VOCAB_LOCAL, S.HEAD_K)


def dense_cases(group: str) -> list[tuple[str, int, str, float]]:
    """(label, rows, family, logical bytes) per profiled call of the mrow groups."""
    if group == 'smoke':
        g = DENSE_GROUPS[1]
        return [(f'smoke-{g.name}-r4', 4, 'mrow', g.bytes)]
    if group == 'dense':
        return [(f'{g.name}-r{m}', m, 'mrow', g.bytes) for m in DENSE_ROWS for g in DENSE_GROUPS]
    if group == 'head':
        return [(f'lm_head-r{m}', m, 'mrow_head', HEAD_BYTES) for m in HEAD_ROWS]
    raise ValueError(f'{group} is not an mrow group')


@dataclass(frozen=True)
class Picked:
    read: str
    write: str | None
    time: str | None

    @property
    def source(self) -> str:
        return 'DRAM counters' if self.read.startswith('dram__') else 'L2 sector counters (DRAM counters absent)'

    @property
    def metrics(self) -> list[str]:
        return [m for m in (self.read, self.write, self.time) if m is not None]

    def doc(self) -> dict[str, object]:
        return {'read': self.read, 'write': self.write, 'time': self.time, 'source': self.source,
                'metrics': self.metrics}


def pick_metrics(query: str) -> Picked:
    """The first read and write counters of the preference lists whose base name `ncu --query-metrics` lists."""
    names = {tok.strip().strip('"') for line in query.splitlines() for tok in line.replace(',', ' ').split()}

    def first(cands: Sequence[str]) -> str | None:
        return next((c for c in cands if c.rsplit('.', 1)[0] in names), None)
    read = first(READ_METRICS)
    if read is None:
        raise ValueError('ncu lists neither DRAM nor L2 read counters of the preference list')
    return Picked(read, first(WRITE_METRICS), TIME_METRIC if TIME_METRIC.rsplit('.', 1)[0] in names else None)


def bw_bytes(op: str, nbytes: int) -> int:
    if op not in BW_OPS:
        raise ValueError(op)
    return 2 * nbytes if op == 'copy' else nbytes


def bw_summary(op: str, nbytes: int, times_ms: Sequence[float]) -> dict[str, object]:
    if not times_ms or min(times_ms) <= 0:
        raise ValueError(f'{op}: no positive timing')
    moved = bw_bytes(op, nbytes)
    med = statistics.median(times_ms)
    return {'op': op, 'buffer_bytes': nbytes, 'bytes_moved': moved, 'reps': len(times_ms), 'median_ms': med,
            'min_ms': min(times_ms), 'max_ms': max(times_ms), 'gbps_median': moved / med / 1e6,
            'gbps_max': moved / min(times_ms) / 1e6}


def plan() -> dict[str, object]:
    return {'groups': {'smoke': [c[0] for c in dense_cases('smoke')],
                       'smoe-b1': [c.label for c in smoe_cases('smoe-b1')],
                       'smoe-b4': [c.label for c in smoe_cases('smoe-b4')],
                       'dense': [c[0] for c in dense_cases('dense')], 'head': [c[0] for c in dense_cases('head')]},
            'read_metrics': list(READ_METRICS), 'write_metrics': list(WRITE_METRICS), 'bw_ops': list(BW_OPS)}


# ---------------------------------------------------------------- GPU (in the image only)

def _events_ms(fn: Any, warmup: int, reps: int) -> list[float]:              # pragma: no cover - GPU
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out: list[float] = []
    cuda: Any = torch.cuda
    for _ in range(reps):
        a, b = cuda.Event(enable_timing=True), cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        out.append(float(a.elapsed_time(b)))
    return out


def bw(args: argparse.Namespace) -> int:                                    # pragma: no cover - GPU
    import torch
    if args.out.exists():
        raise SystemExit(f'{args.out} exists; refusing to overwrite')
    nbytes = args.mib * 2**20
    x = torch.ones(nbytes // 4, dtype=torch.float32, device='cuda')
    y = torch.empty_like(x)
    ops = {'read': lambda: torch.sum(x), 'write': lambda: x.fill_(0.5), 'copy': lambda: y.copy_(x)}
    rows = [bw_summary(op, nbytes, _events_ms(ops[op], 3, args.reps)) for op in BW_OPS]
    doc = {'type': 'cen3-bandwidth', 'rank': args.rank, 'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
           'time': time.time(), 'ops': rows}
    args.out.write_text(json.dumps(doc, indent=1) + '\n')
    return 0


def _profiled(fn: Any) -> None:                                              # pragma: no cover - GPU
    """One unprofiled warm-up call, then exactly one call inside a cudaProfilerStart/Stop range."""
    import torch
    profiler: Any = torch.cuda.profiler          # cudaProfilerStart / cudaProfilerStop
    fn()
    torch.cuda.synchronize()
    profiler.start()
    fn()
    torch.cuda.synchronize()
    profiler.stop()


def ncu_target(args: argparse.Namespace) -> int:                            # pragma: no cover - GPU
    import torch
    if args.out.exists():
        raise SystemExit(f'{args.out} exists; refusing to overwrite')
    dev = torch.device('cuda')
    gen = torch.Generator(device=dev).manual_seed(2027)
    cases: list[dict[str, object]] = []
    if args.group.startswith('smoe-'):
        BK = importlib.import_module('vllm_exl3_moe.bank')
        P = importlib.import_module('vllm_exl3_moe.plan')
        SM = importlib.import_module('vllm_exl3_moe.smoe')
        plan_doc = P.Exl3MoEPlan.from_json_file(args.plan)
        lp = plan_doc.layers[SMOE_LAYER]
        bank = BK.Exl3Bank(lp, S.H, S.I_EXPERT, S.TP, args.rank)
        t = bank.allocate(dev)
        for n in ('w13_trellis', 'w2_trellis'):
            t[n].random_(-32768, 32767, generator=gen)
        for n in ('w13_suh', 'w2_suh'):
            t[n].copy_(torch.randint(0, 2, t[n].shape, generator=gen, device=dev).half() * 2 - 1)
        for n in ('w13_svh', 'w2_svh'):
            t[n].normal_(0.0, 0.02, generator=gen)
        layer = SM.SmoeLayer(bank, S.TOP_K)
        layer.attach_shared(KB._rand_linear(S.H, S.I_LOCAL, S.SHARED_K, gen),
                            KB._rand_linear(S.H, S.I_LOCAL, S.SHARED_K, gen),
                            KB._rand_linear(S.I_LOCAL, S.H, S.SHARED_K, gen))
        perm = torch.tensor(lp.perm, dtype=torch.int64, device=dev)
        bits = [(b.k_gu, b.k_d) for b in lp.bits]
        for case in smoe_cases(args.group):
            ids_l = KB.routing(case, args.seed)
            ids = torch.tensor(ids_l, dtype=torch.int32, device=dev)
            w = torch.rand((case.rows, S.TOP_K), generator=gen, device=dev)
            w = (w / w.sum(-1, keepdim=True)).float()
            x = (torch.randn((case.rows, S.H), generator=gen, device=dev) * 0.1).to(torch.bfloat16)
            _profiled(lambda: layer(x, ids, w, perm, S.SWIGLU_LIMIT, shared=True))
            gu, down = smoe_parts(ids_l, bits)
            cases.append({'case': case.label, 'rows': case.rows, 'distinct': KB.distinct(ids_l),
                          'launches': [{'family': 'smoe_gu', 'bytes': gu}, {'family': 'smoe_down', 'bytes': down}]})
    else:
        MR = importlib.import_module('vllm_exl3_moe.mrow')
        groups: dict[str, Any] = {}
        widths: dict[str, tuple[int, int]] = {}
        for g in DENSE_GROUPS:
            lins = [KB._rand_linear(i, o, S.DENSE_K, gen) for i, o in g.linears]
            groups[g.name] = MR.MrowGroup([(lin.trellis, lin.suh, lin.svh, xc, oc)
                                           for lin, (xc, oc) in zip(lins, g.offsets)], S.DENSE_K)
            widths[g.name] = g.width
        head = KB._rand_linear(S.H, S.VOCAB_LOCAL, S.HEAD_K, gen)
        groups['lm_head'] = MR.MrowGroup([(head.trellis, head.suh, head.svh, 0, 0)], S.HEAD_K)
        widths['lm_head'] = (S.H, S.VOCAB_LOCAL)
        for label, m, family, nbytes in dense_cases(args.group):
            name = label.removeprefix('smoke-').rsplit('-r', 1)[0]
            x = (torch.randn((m, widths[name][0]), generator=gen, device=dev) * 0.1).to(torch.bfloat16)
            out = torch.empty((m, widths[name][1]), dtype=torch.bfloat16, device=dev)
            grp = groups[name]
            _profiled(lambda: grp(x, out))
            cases.append({'case': label, 'rows': m, 'launches': [{'family': family, 'bytes': nbytes}]})
    doc = {'type': 'cen3-ncu-target', 'group': args.group, 'rank': args.rank, 'seed': args.seed,
           'plan_layer': SMOE_LAYER if args.group.startswith('smoe-') else None,
           'device': torch.cuda.get_device_name(), 'cases': cases}
    args.out.write_text(json.dumps(doc, indent=1) + '\n')
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('plan')
    b = sub.add_parser('bw')
    b.add_argument('--mib', type=int, default=2048)
    b.add_argument('--reps', type=int, default=20)
    b.add_argument('--rank', type=int, default=0, choices=(0, 1))
    b.add_argument('--out', type=Path, required=True)
    q = sub.add_parser('pick-metrics')
    q.add_argument('--query', type=Path, required=True)
    q.add_argument('--out', type=Path, required=True)
    t = sub.add_parser('ncu-target')
    t.add_argument('--group', choices=GROUPS, required=True)
    t.add_argument('--plan', default='/plan.json')
    t.add_argument('--rank', type=int, default=0, choices=(0, 1))
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'plan':
        json.dump(plan(), sys.stdout, indent=1)
        sys.stdout.write('\n')
        return 0
    if args.cmd == 'pick-metrics':
        picked = pick_metrics(args.query.read_text(errors='replace'))
        args.out.write_text(json.dumps(picked.doc()) + '\n')
        print(','.join(picked.metrics))
        return 0
    os.environ.setdefault('EXL3_MROW32_VARIANT', 'ring')
    os.environ.setdefault('EXL3_MROW32_MIN_ROWS', '9')
    return {'bw': bw, 'ncu-target': ncu_target}[args.cmd](args)


if __name__ == '__main__':
    raise SystemExit(main())
