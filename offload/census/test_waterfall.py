"""CPU tests for the census binner and its Nsight Systems reader: synthetic two-rank traces only, no real trace."""

from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

import pytest

from offload.census import nsys
from offload.census import rules as C
from offload.census import waterfall as W
from offload.gb10_budget import trace_budget as TB
from offload.prefill_maxlen.evidence import Json, number, record

ANN = 'execute_{n}_context_0(sq0sk0sqsq0sqsk0)_generation_1(sq{n}sk100sqsq9sqsk300)'
NCCL = 'ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>)'
HTOD = 'Memcpy HtoD (Pinned -> Device)'
HC = 'void deep_gemm::sm120_tf32_hc_prenorm_gemm_impl<24u, 20480u, 128u>(unsigned int)'
JIT = 'Kernel'
PTOPK = 'void vllm::persistent::persistent_topk_kernel<512, 4u>(vllm::persistent::PersistentTopKParams)'
MROW5 = 'void (anonymous namespace)::mrow_kernel<5, false>(__nv_bfloat16 const*, int)'
MROW6 = 'void (anonymous namespace)::mrow_kernel<6, false>(__nv_bfloat16 const*, int)'
L2PF = '(anonymous namespace)::l2pf_kernel((anonymous namespace)::PArgs)'
SMOE_GU = '(anonymous namespace)::smoe_gu((anonymous namespace)::Layer, (anonymous namespace)::Call)'
STAGE = 'void (anonymous namespace)::stage_kernel<uint4>(uint4 const*, int, char*)'
SUM = '(anonymous namespace)::sum_kernel(uint4 const*, uint4*, int, char const*)'

# (offset us, duration us, name, stream) inside one 1000-us step; rank 1 waits 95 us in the step-start broadcast
# (rank 0 idles 90 us before the drafts' H2D instead) and 50 us in the all-reduce sum
RANK0 = ((0, 10, NCCL, 7), (100, 5, HTOD, 7), (105, 45, HC, 7), (150, 10, JIT, 7), (160, 20, PTOPK, 7),
         (180, 100, MROW5, 7), (200, 100, L2PF, 9), (300, 180, SMOE_GU, 7), (480, 10, STAGE, 7), (490, 20, SUM, 7),
         (600, 100, MROW6, 7))
RANK1 = ((0, 105, NCCL, 7), (105, 45, HC, 7), (150, 10, JIT, 7), (160, 20, PTOPK, 7), (180, 100, MROW5, 7),
         (200, 100, L2PF, 9), (300, 180, SMOE_GU, 7), (480, 10, STAGE, 7), (490, 70, SUM, 7), (600, 100, MROW6, 7))
STEPS = 4                                   # the last has no successor: three complete steps


def base(i: int) -> float:
    return 1000.0 * (i + 1)


def events(kernels: tuple[tuple[int, int, str, int], ...], graph_visible: bool = True) -> list[dict[str, Json]]:
    out: list[dict[str, Json]] = []
    for i in range(STEPS):
        b = base(i)
        out.append({'ph': 'X', 'cat': 'user_annotation', 'name': ANN.format(n=3), 'ts': b - 100, 'dur': 150, 'tid': 1})
        if graph_visible:
            out.append({'ph': 'X', 'cat': 'cuda_runtime', 'name': 'cudaGraphLaunch', 'ts': b - 99, 'dur': 1, 'tid': 1,
                        'args': {'correlation': 100 + i}})
        else:                               # S42 case: an eager launch is visible, the graph's kernels are not
            out.append({'ph': 'X', 'cat': 'cuda_runtime', 'name': 'cudaGraphLaunch', 'ts': b - 99, 'dur': 1, 'tid': 1,
                        'args': {'correlation': 900 + i}})
            out.append({'ph': 'X', 'cat': 'cuda_runtime', 'name': 'cudaLaunchKernel', 'ts': b - 98, 'dur': 1,
                        'tid': 1, 'args': {'correlation': 100 + i}})
        out.append({'ph': 'X', 'cat': 'cuda_runtime', 'name': 'cuMemcpyHtoDAsync', 'ts': b - 50, 'dur': 1, 'tid': 2,
                    'args': {'correlation': 500 + i}})
        for off, dur, name, stream in kernels:
            cat = 'gpu_memcpy' if name.startswith('Memcpy') else 'kernel'
            corr = 500 + i if cat == 'gpu_memcpy' else 100 + i
            out.append({'ph': 'X', 'cat': cat, 'name': name, 'ts': b + off, 'dur': dur,
                        'args': {'stream': stream, 'correlation': corr, 'grid': [1, 1, 1], 'block': [128, 1, 1]}})
    return out


def write(path: Path, evs: list[dict[str, Json]]) -> Path:
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        json.dump({'traceEvents': evs}, f)
    return path


@pytest.fixture()
def pair(tmp_path: Path) -> dict[int, Path]:
    return {0: write(tmp_path / 'r0.pt.trace.json.gz', events(RANK0)),
            1: write(tmp_path / 'r1.pt.trace.json.gz', events(RANK1))}


def test_rules_buckets_on_recorded_names() -> None:
    expected = {
        L2PF: 'prefetch', SUM: 'allreduce', STAGE: 'allreduce', NCCL: 'bcast', HC: 'mhc',
        'mhc_pre_big_fuse_with_norm_tilelang_kernel': 'mhc', 'mhc_post_tilelang_kernel': 'mhc',
        'void (anonymous namespace)::post_collapse_exact_kernel<5>(__nv_bfloat16 const*)': 'mhc',
        MROW5: 'dense', MROW6: 'lm_head', SMOE_GU: 'smoe', JIT: 'attention', PTOPK: 'topk',
        'void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int>(int)': 'topk',
        'void deep_gemm::sm120_fp8_paged_mqa_logits<1u, 32u>(unsigned int)': 'indexer',
        '_mask_candidates_kernel': 'indexer', '_candidate_flags_kernel': 'indexer', '_block_scores_kernel': 'indexer',
        '_compute_global_topk_indices_and_lens_kernel': 'attention', '_dsv4_topk_kernel': 'moe_router',
        'void (anonymous namespace)::router_kernel<4, false>(__nv_bfloat16 const*)': 'moe_router',
        '_engram_lookup_kernel': 'engram', '_resample_kernel': 'sampler', HTOD: 'other', 'memcpy32_post': 'other',
        'void flashinfer::sparse_mla_sm120::sparse_mla_decode_dsv4_kernel<1, 32, 64>(int)': 'attention',
        'void marlin::Marlin<1, 2>(int4 const*)': 'dense', 'something_new_kernel': 'unmatched'}
    assert {n: C.bucket(n) for n in expected} == expected
    C.check()


def test_waterfall_splits_waits_and_sums_to_period(pair: dict[int, Path]) -> None:
    ranks = {k: W.load_kineto(p) for k, p in pair.items()}
    out = W.capture(ranks)
    assert out['pairs'] == 3
    r0 = record(record(out['rank_ms'])['0'])
    r1 = record(record(out['rank_ms'])['1'])
    want0 = {'bcast': .010, 'drafter_wait': .090, 'other': .005, 'mhc': .045, 'attention': .010, 'topk': .020,
             'dense': .060, 'prefetch': .060, 'smoe': .180, 'allreduce': .030, 'device_wait': .090, 'lm_head': .100,
             'host_gap': .300}
    want1 = {'bcast_wait': .095, 'bcast': .010, 'mhc': .045, 'attention': .010, 'topk': .020, 'dense': .060,
             'prefetch': .060, 'smoe': .180, 'allreduce': .030, 'rank_wait': .050, 'device_wait': .040,
             'lm_head': .100, 'host_gap': .300}
    for got, want in ((r0, want0), (r1, want1)):
        assert number(got['period_ms']) == pytest.approx(1.0)
        assert sum(number(got[b]) for b in C.REPORT) == pytest.approx(1.0)
        for b in C.REPORT:
            assert number(got[b]) == pytest.approx(want.get(b, 0.0), abs=1e-9), b
    assert number(out['drafter_wait_incl_bcast_ms']) == pytest.approx((.090 + .095) / 2)
    assert number(out['cross_rank_wait_ms']) == pytest.approx((.050 + .095) / 2)
    kl1 = record(out['kl1'])
    assert number(kl1['sum_wait_ms']) == pytest.approx(.050)
    assert number(kl1['sum_wait_r1_ms']) == pytest.approx(.050)
    assert number(kl1['nccl_r1_ms']) == pytest.approx(.105)
    assert number(kl1['allreduce_intrinsic_ms']) == pytest.approx(.020 + .010)
    assert set(record(out['by_rows'])) == {'3'}
    groups = record(out['groups_ms'])
    assert sum(number(v) for v in groups.values()) == pytest.approx(1.0)


def test_sp1_rebin_matches_trace_budget(pair: dict[int, Path]) -> None:
    r = TB.rules()
    reference = TB.capture_budget({k: TB.load(p) for k, p in pair.items()}, 'decode', r)
    out = W.capture({k: W.load_kineto(p) for k, p in pair.items()}, r, reference)
    check = record(out['sp1_rebin'])
    assert number(check['reference_max_abs_diff_ms']) < 1e-9
    assert number(check['reference_period_diff_ms']) < 1e-9


def test_invisible_graph_kernels_are_not_budgeted(tmp_path: Path) -> None:
    paths = {0: write(tmp_path / 'a.json.gz', events(RANK0, graph_visible=False)),
             1: write(tmp_path / 'b.json.gz', events(RANK1, graph_visible=False))}
    out = W.capture({k: W.load_kineto(p) for k, p in paths.items()})
    assert str(out['status']).startswith('UNBUDGETED')
    assert 'ms' not in out


def test_collective_sequences_must_match(tmp_path: Path) -> None:
    paths = {0: write(tmp_path / 'a.json.gz', events(RANK0)),
             1: write(tmp_path / 'b.json.gz', events(tuple(k for k in RANK1 if k[2] != STAGE)))}
    with pytest.raises(ValueError, match='collective sequences differ'):
        W.capture({k: W.load_kineto(p) for k, p in paths.items()})


def test_both_ranks_required(pair: dict[int, Path]) -> None:
    with pytest.raises(ValueError, match='both ranks'):
        W.capture({0: W.load_kineto(pair[0])})


def _sqlite(path: Path, kernels: tuple[tuple[int, int, str, int], ...]) -> Path:
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, end INT, streamId INT, correlationId INT,
            demangledName INT, shortName INT, gridX INT, gridY INT, gridZ INT, blockX INT, blockY INT, blockZ INT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INT, end INT, streamId INT, correlationId INT, copyKind INT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INT, end INT, globalTid INT, correlationId INT, nameId INT);
        CREATE TABLE NVTX_EVENTS (start INT, end INT, globalTid INT, text TEXT, textId INT, eventType INT);
    """)
    ids: dict[str, int] = {}

    def sid(s: str) -> int:
        if s not in ids:
            ids[s] = len(ids) + 1
            db.execute('INSERT INTO StringIds VALUES (?, ?)', (ids[s], s))
        return ids[s]

    ns = 1000
    for i in range(STEPS):
        b = int(base(i)) * ns
        db.execute('INSERT INTO NVTX_EVENTS VALUES (?, ?, 1, ?, NULL, 59)', (b - 100 * ns, b + 50 * ns,
                                                                              ANN.format(n=3)))
        db.execute('INSERT INTO NVTX_EVENTS VALUES (?, NULL, 1, ?, NULL, 34)', (b, 'a mark, not a range'))
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 1, ?, ?)',
                   (b - 99 * ns, b - 98 * ns, 100 + i, sid('cudaGraphLaunch_v10000')))
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 2, ?, ?)',
                   (b - 50 * ns, b - 49 * ns, 500 + i, sid('cuMemcpyHtoDAsync_v2')))
        for off, dur, name, stream in kernels:
            if name.startswith('Memcpy'):
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, ?, 1)',
                           (b + off * ns, b + (off + dur) * ns, stream, 500 + i))
            else:
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, 1, 1, 1, 128, 1, 1)',
                           (b + off * ns, b + (off + dur) * ns, stream, 100 + i, sid(name), sid(name[:20])))
    db.commit()
    db.close()
    return path


def test_nsys_export_gives_the_kineto_waterfall(tmp_path: Path, pair: dict[int, Path]) -> None:
    kineto = W.capture({k: W.load_kineto(p) for k, p in pair.items()})
    paths = {0: _sqlite(tmp_path / 'r0.sqlite', RANK0), 1: _sqlite(tmp_path / 'r1.sqlite', RANK1)}
    ranks = {k: nsys.load(p) for k, p in paths.items()}
    assert ranks[0].source == 'nsys'
    out = W.capture(ranks)
    for b in C.REPORT:
        assert number(record(out['ms'])[b]) == pytest.approx(number(record(kineto['ms'])[b]), abs=1e-9), b


def test_nsys_without_step_ranges_is_refused(tmp_path: Path) -> None:
    path = _sqlite(tmp_path / 'r0.sqlite', RANK0)
    db = sqlite3.connect(path)
    db.execute('DELETE FROM NVTX_EVENTS')
    db.commit()
    db.close()
    with pytest.raises(ValueError, match='execute_'):
        nsys.load(path)


def test_window_verifies_manifest_and_writes_markdown(tmp_path: Path, pair: dict[int, Path]) -> None:
    root = tmp_path / 'SP1-W-20990101T000000Z'
    (root / 'traces').mkdir(parents=True)
    rows = []
    for rank, src in pair.items():
        dst = root / 'traces' / src.name
        dst.write_bytes(src.read_bytes())
        rows.append({'cell': 'prof-short-b1', 'capture': 'decode', 'status': 'TAKEN', 'rank': rank,
                     'file': f'traces/{src.name}', 'bytes': dst.stat().st_size, 'sha256': W._sha256(dst)})
    rows.append({'cell': 'prof-32k-b1', 'capture': 'prefill-early', 'status': 'TAKEN', 'rank': 0,
                 'file': 'traces/none', 'bytes': 0, 'sha256': ''})
    (root / 'traces-manifest.json').write_text(json.dumps(rows))
    out = tmp_path / 'out.json'
    md = tmp_path / 'out.md'
    assert W.main(['--window', str(root), '--output', str(out), '--markdown', str(md)]) == 0
    res = record(json.loads(out.read_text()))
    w = record(record(res['windows'])[root.name])
    assert w['skipped_captures'] == ['prof-32k-b1:prefill-early']
    assert 'prof-short-b1' in md.read_text()
    assert W.main(['--window', str(root), '--output', str(out)]) == 1          # never overwrites
    rows[0]['sha256'] = '0' * 64
    (root / 'traces-manifest.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='sha256'):
        W.window(root)
