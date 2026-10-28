"""CPU tests for the KL1 trace and bytes tools: synthetic kineto traces only, no real trace, no host."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import cast

import pytest

from offload.kl1 import bandwidth as B
from offload.kl1 import shapes as S
from offload.kl1 import trace_kernels as T

ANN = 'execute_{n}_context_0(sq0sk0sqsq0sqsk0)_generation_1(sq{n}sk100sqsq9sqsk300)'


def _kernel(name: str, ts: float, dur: float, corr: int, grid: tuple[int, int, int] = (1, 1, 1),
            stream: int = 7) -> dict[str, object]:
    return {'ph': 'X', 'cat': 'kernel', 'name': name, 'ts': ts, 'dur': dur,
            'args': {'stream': stream, 'correlation': corr, 'grid': list(grid), 'block': [288, 1, 1],
                     'occupancy': {'activeBlocksPerMultiprocessor': 3}}}


def _launch(ts: float, corr: int, tid: int = 1) -> dict[str, object]:
    return {'ph': 'X', 'cat': 'cuda_runtime', 'name': 'cudaGraphLaunch', 'ts': ts, 'dur': 1, 'tid': tid,
            'args': {'correlation': corr}}


def _annotation(ts: float, dur: float, rows: int, tid: int = 1) -> dict[str, object]:
    return {'ph': 'X', 'cat': 'user_annotation', 'name': ANN.format(n=rows), 'ts': ts, 'dur': dur, 'tid': tid}


SMOE_GU = '(anonymous namespace)::smoe_gu((anonymous namespace)::Layer, (anonymous namespace)::Call)'
SMOE_DOWN = '(anonymous namespace)::smoe_down((anonymous namespace)::Layer, (anonymous namespace)::Call)'
SMOE_ROT = '(anonymous namespace)::smoe_rot((anonymous namespace)::Layer, (anonymous namespace)::Call)'
L2PF = '(anonymous namespace)::l2pf_kernel((anonymous namespace)::PArgs)'
MROW5 = 'void (anonymous namespace)::mrow_kernel<5, false>(__nv_bfloat16 const*, int)'
STAGE = 'void (anonymous namespace)::stage_kernel<uint4>(uint4 const*, int, char*)'
SUM = '(anonymous namespace)::sum_kernel(uint4 const*, uint4*, int, char const*)'


def _step_events(t0: float, corr: int, rows: int) -> list[dict[str, object]]:
    """One decode iteration: launch at t0, graph kernels: stage/sum, rot, gu, down (PDL overlap), l2pf on a side stream."""
    ev: list[dict[str, object]] = [_annotation(t0, 50, rows), _launch(t0 + 1, corr)]
    g = t0 + 100
    ev += [_kernel(STAGE, g, 5, corr), _kernel(SUM, g + 5, 20, corr),
           _kernel(SMOE_ROT, g + 30, 10, corr, (210, 1, 1)),
           _kernel(SMOE_GU, g + 35, 300, corr, (189, 1, 1)),
           _kernel(SMOE_DOWN, g + 40, 380, corr, (420, 1, 1)),
           _kernel(L2PF, g + 30, 200, corr, stream=9),
           _kernel(STAGE, g + 430, 5, corr), _kernel(SUM, g + 435, 10, corr)]
    return ev


def _trace(tmp: Path, steps: int = 3, rows: int = 3) -> Path:
    ev: list[dict[str, object]] = []
    for i in range(steps):
        ev += _step_events(1000.0 * i, 100 + i, rows)
    path = tmp / 'rank0.pt.trace.json.gz'
    with gzip.open(path, 'wt', encoding='utf-8') as fh:
        json.dump({'traceEvents': ev}, fh)
    return path


def test_families() -> None:
    assert T.family(SMOE_GU) == 'smoe_gu'
    assert T.family(L2PF) == 'l2pf'
    assert T.family(MROW5) == 'mrow'
    assert T.family('void (anonymous namespace)::mrow_kernel<6, false>(x)') == 'mrow_head'
    assert T.family('void (anonymous namespace)::mrow32_kernel<5, 2>(x)') == 'mrow'
    assert T.family(STAGE) == 'coll_stage' and T.family(SUM) == 'coll_sum'
    assert T.family('void at::native::vectorized_gather_kernel<16, long>(x)') != 'coll_gather'
    assert T.template_k('void reconstruct_had_kernel<5, 2, false>(__half*)') == 5


def test_steps_and_layers(tmp_path: Path) -> None:
    trace = T.load(_trace(tmp_path))
    steps = T.steps(trace)
    assert [s.index for s in steps] == [0, 1]          # the last iteration has no successor
    s = steps[0]
    assert s.phase == 'decode' and s.rows == 3
    assert s.period == pytest.approx(1000.0)
    layers = T.moe_layers(s)
    assert len(layers) == 1
    L = layers[0]
    assert L.span == pytest.approx(390.0)              # rot start 130 -> down end 520
    assert L.down_after_gu == pytest.approx(85.0)       # down end 520 - gu end 435
    segs = T.compute_segments(s)
    assert [k for k, _ in segs] == ['moe'] and segs[0][1] == pytest.approx(405.0)


def test_union_and_family_totals(tmp_path: Path) -> None:
    assert T.union([(0, 10), (5, 15), (20, 30)]) == pytest.approx(25.0)
    assert T.union([(0, 10)], 2, 8) == pytest.approx(6.0)
    s = T.steps(T.load(_trace(tmp_path)))[0]
    fam = T.family_totals(s)
    assert fam['l2pf']['sum_us'] == pytest.approx(200.0)
    assert fam['coll_sum']['n'] == 2


def test_sp1_split_moves_l2pf(tmp_path: Path) -> None:
    s = T.steps(T.load(_trace(tmp_path)))[0]
    split = B.sp1_split(s)
    # Since offload-x86host e791d46 SP1's buckets.json gives l2pf its own 'prefetch' bucket (this finding), so the SP1 view
    # and the l2pf-apart view now agree on routed_moe and the prefetch share is the l2pf share
    assert split['sp1.routed_moe'] == pytest.approx(split['l2pf_apart.routed_moe'])
    assert split['sp1.prefetch'] == pytest.approx(split['l2pf_apart.l2pf'])
    wall = B.moe_wall(s)
    assert wall['expert_union_ms'] == pytest.approx(0.390)


def test_pair_steps(tmp_path: Path) -> None:
    a = T.steps(T.load(_trace(tmp_path)))
    assert len(T.pair_steps(a, a)) == len(a)


def test_shapes() -> None:
    assert S.mean_k() == pytest.approx(2.75)
    # SP1's one-expert figure 3*5120*1152*2.75/8 plus the fp16 scales
    assert S.expert_bytes(2.75) == pytest.approx(3 * 5120 * 1152 * 2.75 / 8 + 2 * 3 * (5120 + 1152))
    assert S.expert_bytes(5) / 1e6 == pytest.approx(11.10, abs=0.01)    # smoe.cu's "~11 MB" shared expert
    assert S.LM_HEAD.bytes / 1e6 == pytest.approx(248.36, abs=0.01)
    assert S.unique_uniform(1) == pytest.approx(6.0)
    assert S.unique_uniform(4092) == pytest.approx(384.0)
    assert S.shared_runs(0) == 0 and S.shared_runs(8) == 1 and S.shared_runs(9) == 2 and S.shared_runs(20) == 3
    assert S.reconstruct_shape((128, 10, 1)) == (1280, 16384)
    with pytest.raises(ValueError):
        S.exl3_bytes(100, 128, 5)


def test_unique_at_inverts_smoe_bytes() -> None:
    span_us = 400.0
    u = B.unique_at(250.0, span_us, 4)
    assert B.smoe_bytes(u, 4) / span_us / 1e3 == pytest.approx(250.0)
    gu, dn = B.expert_mean_bytes()
    assert gu == pytest.approx(2 * dn, rel=0.01)


def test_dense_decode_rejects_partial_layers(tmp_path: Path) -> None:
    s = T.steps(T.load(_trace(tmp_path)))[0]
    extra = T.Kernel(MROW5, 'mrow', 'dense_gemm', s.lo + 1, s.lo + 2, 7, (80, 1, 1), (288, 1, 1), 3.0, None)
    s.kernels.append(extra)
    with pytest.raises(ValueError):
        B.dense_decode(s)


def test_moe_phase_parts(tmp_path: Path) -> None:
    from offload.kl1 import moe_phase as M
    s = T.steps(T.load(_trace(tmp_path)))[0]
    parts = M.layer_parts(s)
    assert len(parts) == 1
    p = parts[0]
    # attention all-reduce's sum ends at 125; gu starts 135; down ends 520
    assert p['pre_us'] == pytest.approx(10.0)
    assert p['gu_us'] == pytest.approx(300.0) and p['tail_us'] == pytest.approx(85.0)
    assert p['phase_us'] == pytest.approx(395.0)
    assert p['idle_us'] == pytest.approx(5.0)               # 125-130: nothing runs before rot and l2pf start
    assert p['idle_no_l2pf_us'] == pytest.approx(5.0)       # l2pf (130-330) sits under rot/gu here
    row = M.step_row(s)
    assert row['shared_runs'] == 1 and row['shared_reread_mb_per_step'] == 0.0
    assert row['gu_resident'] == pytest.approx(48 * 3)


def test_b4_over_b1_bytes() -> None:
    from offload.kl1 import moe_phase as M
    # one request at the same rows is the same bytes; four independent requests move more, never 4x more experts
    assert M.b4_over_b1_bytes(12, 4, 4, 1, True) == pytest.approx(1.0)
    r = M.b4_over_b1_bytes(12, 4, 16, 4, True)
    assert 3.0 < r < 4.0
    assert M.b4_over_b1_bytes(12, 4, 20, 4, False) > M.b4_over_b1_bytes(12, 4, 20, 4, True)


def _pair_trace(tmp: Path, name: str, sum_dur: float, seg: float) -> Path:
    ev: list[dict[str, object]] = []
    for i in range(3):
        t0 = 2000.0 * i
        c = 100 + i
        g = t0 + 100
        ev += [_annotation(t0, 50, 3), _launch(t0 + 1, c),
               _kernel(STAGE, g, 5, c), _kernel(SUM, g + 5, sum_dur, c),
               _kernel(SMOE_GU, g + 5 + sum_dur + 5, seg - 10, c), _kernel(SMOE_DOWN, g + 5 + sum_dur + 6, seg - 10, c),
               _kernel(STAGE, g + 5 + sum_dur + seg, 5, c), _kernel(SUM, g + 10 + sum_dur + seg, 6, c)]
    path = tmp / name
    with gzip.open(path, 'wt', encoding='utf-8') as fh:
        json.dump({'traceEvents': ev}, fh)
    return path


def test_collective_pairing(tmp_path: Path) -> None:
    from offload.kl1 import collectives as C
    a = T.steps(T.load(_pair_trace(tmp_path, 'r0.gz', 40.0, 300.0)))
    b = T.steps(T.load(_pair_trace(tmp_path, 'r1.gz', 10.0, 330.0)))
    pairs = T.pair_steps(a, b)
    assert len(pairs) == 2
    row = C.step_pair(*pairs[0])
    assert row['sum_intrinsic_ms'] == pytest.approx((10 + 6) / 1e3)
    assert row['sum_wait_ms'] == pytest.approx(30 / 1e3)
    assert row['sum_wait_r0_ms'] == pytest.approx(30 / 1e3) and row['sum_wait_r1_ms'] == 0.0
    assert row['moe_imbalance_ms'] == pytest.approx(15 / 1e3)       # |300 - 330| / 2
    assert row['moe_r1_slower_frac'] == 1.0


def test_collective_pairing_refuses_mismatch(tmp_path: Path) -> None:
    from offload.kl1 import collectives as C
    a = T.steps(T.load(_pair_trace(tmp_path, 'r0.gz', 40.0, 300.0)))[0]
    b = T.steps(T.load(_pair_trace(tmp_path, 'r1.gz', 10.0, 330.0)))[0]
    b.kernels = [k for k in b.kernels if k.family != 'coll_stage']
    with pytest.raises(ValueError):
        C.step_pair(a, b)


def _cell(stream: float, rows: float, period: float = 40.0) -> dict[str, float]:
    return {'period_ms': period, 'rows': rows, 'moe_phase_ms': stream + 1.5, 'prefix_ms': 1.5, 'stream_ms': stream,
            'shared_reread_mb': 0.0 if rows <= 8 else 780.0, 'dense_ms': 8.0, 'dense_mb': 1700.0, 'head_ms': 1.1,
            'coll_intrinsic_ms': 1.0, 'imbalance_ms': 1.4, 'unique_240': 15.0}


def test_headroom_slack_and_ranges() -> None:
    from offload.kl1 import headroom as H
    b1 = _cell(16.0, 4.0)
    # B4 taking four times B1's stream time leaves no per-byte slack; 3x leaves some
    assert H.per_byte_slack_b1(b1, _cell(64.0, 19.0, 80.0)) == 0.0
    assert H.per_byte_slack_b1(b1, _cell(48.0, 19.0, 80.0)) > 0.1
    rows = H.candidates(b1, _cell(48.0, 19.0, 80.0), (216.0, 235.0))
    by = {str(r['id']): r for r in rows}
    for r in rows:
        lo1, hi1 = cast(list[float], r['b1_ms'])
        lo4, hi4 = cast(list[float], r['b4_ms'])
        assert lo1 <= hi1 and lo4 <= hi4
    assert by['grouped-multirow']['b1_ms'] == [0.0, 0.0]
    assert by['fused-moe-launch']['b4_ms'] == [0.0, 0.0]
    assert cast(list[float], by['grouped-multirow']['b4_ms'])[1] == pytest.approx(780.0 / 216.0)
    assert H.ep_imbalance_ms(b1) > 0


def test_bench_variants_and_cases() -> None:
    from offload.kl1 import smoe_bench as SB
    for v in SB.VARIANTS:
        v.check()
    assert SB.VARIANTS[0].name == 'prod' and SB.VARIANTS[0].flags == '-DSMOE_RUNFLAGS=1'
    for bad in ('-DSMOE_PIPE=5', '-DSMOE_OBI=3', '-DSMOE_FOO=1', 'SMOE_PIPE=4'):
        with pytest.raises(ValueError):
            SB.Variant('bad', bad).check()
    labels = [c.label for c in SB.cases()]
    assert len(labels) == len(set(labels))
    assert SB.plan_summary()['n_cases'] == len(labels)


def test_bench_routing_exact_distinct() -> None:
    from offload.kl1 import smoe_bench as SB
    for case in SB.cases():
        ids = SB.routing(case, seed=3)
        assert len(ids) == case.rows
        assert all(len(set(row)) == S.TOP_K and all(0 <= e < S.E for e in row) for row in ids)
        per = SB.split_rows(case.rows, case.requests)
        start = 0
        for n in per:
            assert SB.distinct(ids[start:start + n]) == case.unique
            start += n
        if case.requests == 1:
            assert SB.distinct(ids) == case.unique
        assert SB.routing(case, seed=3) == ids                     # deterministic
    core = SB.Case(16, 4, 12, 12)
    assert SB.distinct(SB.routing(core, 0)) == 12                  # every request uses the same 12
    with pytest.raises(ValueError):
        SB.routing(SB.Case(1, 1, 7), 0)                             # one row cannot cover 7 experts
    with pytest.raises(ValueError):
        SB.split_rows(2, 3)


def test_bench_layer_bytes_and_invert() -> None:
    from offload.kl1 import smoe_bench as SB
    bits = [(2.0, 2.0)] * S.E
    ids = [[0, 1, 2, 3, 4, 5]] * 3                                  # 3 rows, the same 6 experts: one run each
    one = S.expert_gu_bytes(2) + S.expert_down_bytes(2)
    assert SB.layer_bytes(ids, bits) == pytest.approx(6 * one + S.expert_bytes(5))
    nine = [[0, 1, 2, 3, 4, 5]] * 9                                 # 9 rows: two runs of each expert and of the shared
    assert SB.layer_bytes(nine, bits) == pytest.approx(12 * one + 2 * S.expert_bytes(5))
    curve = [(6.0, 200.0), (12.0, 300.0), (18.0, 420.0)]
    u, where = SB.invert(curve, 250.0)
    assert u == pytest.approx(9.0) and where == 'inside'
    assert SB.invert(curve, 100.0) == (6.0, 'below')
    assert SB.invert(curve, 999.0) == (18.0, 'above')
    assert SB.invert([(6.0, 300.0), (12.0, 200.0)], 250.0) == (6.0, 'below')
    assert SB.invert([(6.0, 300.0), (12.0, 200.0)], 300.0) == (9.0, 'flat')


def test_bench_analyze() -> None:
    from offload.kl1 import smoe_bench as SB

    def doc(variant: str, scale: float, digest: str) -> dict[str, object]:
        res = [{'case': f'r4q1u{u}c0', 'rows': 4, 'requests': 1, 'distinct': u, 'bytes_per_layer': 11.1e6 + u * 6.12e6,
                'us_per_layer_median': scale * (100.0 + 20.0 * u), 'out_sha256': digest if u == 12 else 'same'}
               for u in (6, 12, 18)]
        return {'variant': variant, 'results': res, 'type': 'kl1-smoe-bench', 'rank': 0,
                'sample': 'first' if variant == 'prod' else 'variant', 'window': 'KL1-W-test',
                'kernel': SB.BENCH_KERNEL, 'driver': SB.BENCH_DRIVER, 'plan': 'test', 'banks': [0, 39], 'seed': 0,
                'flags': next(v.flags for v in SB.VARIANTS if v.name == variant)}

    out = SB.analyze([doc('prod', 1.0, 'a'), {**doc('prod', 1.0, 'a'), 'sample': 'last'},
                      doc('pipe4', 0.9, 'b')], [(4, 340.0), (3, 300.0)], window='SP1-W-test', rank=0,
                     production_stack=(SB.BENCH_KERNEL, SB.BENCH_DRIVER))
    assert out['oracle_mismatches'] == {'0': {'pipe4': ['r4q1u12c0']}}
    assert out['speed_vs_prod']['0']['pipe4']['r4q1u6c0'] == pytest.approx(1 / 0.9)
    assert len(out['production']) == 1                               # unsupported row is explicitly reported, never silently pooled
    p = out['production'][0]
    assert p['distinct'] == pytest.approx(12.0) and p['where'] == 'inside'
    assert out['unsupported_step_rows'] == [3]
    assert p['isolated_gbps'] == pytest.approx((11.1e6 + 12 * 6.12e6) / 340.0 / 1e3)
    with pytest.raises(ValueError):
        SB.analyze([doc('pipe4', 1.0, 'a')], [], window='SP1-W-test', rank=0,
                   production_stack=(SB.BENCH_KERNEL, SB.BENCH_DRIVER))
