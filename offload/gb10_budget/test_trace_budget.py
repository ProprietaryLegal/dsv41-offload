"""trace_budget on synthetic kineto traces (and one real CPU-only local torch trace with injected GPU events). CPU only."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import pytest

from offload.gb10_budget import trace_budget as tb

R = tb.rules()
DECODE = 'execute_2_context_0(sq0sk0sqsq0sqsk0)_generation_1(sq2sk32002sqsq4sqsk64004)'
PREFILL = 'execute_4092_context_1(sq4092sk{sk}sqsq16744464sqsk{qk})_generation_0(sq0sk0sqsq0sqsk0)'


class Kineto:
    """Writes events the way kineto does: a header, then one indented object per event inside traceEvents."""

    def __init__(self, tid: int = 11) -> None:
        self.tid = tid
        self.events: list[dict[str, Any]] = [{'ph': 'M', 'name': 'process_name', 'pid': 1, 'tid': 0, 'ts': 0,
                                              'args': {'name': 'python3'}}]
        self.corr = 100

    def annotate(self, name: str, start: float, end: float) -> None:
        self.events.append({'ph': 'X', 'cat': 'user_annotation', 'name': name, 'pid': 1, 'tid': self.tid,
                            'ts': start, 'dur': end - start, 'args': {'External id': 1}})

    def launch(self, name: str, at: float, *, record: bool = True) -> int:
        self.corr += 1
        if record:
            self.events.append({'ph': 'X', 'cat': 'cuda_runtime', 'name': name, 'pid': 1, 'tid': self.tid,
                                'ts': at, 'dur': 4, 'args': {'cbid': 211, 'correlation': self.corr}})
        return self.corr

    def gpu(self, name: str, start: float, dur: float, corr: int, cat: str = 'kernel') -> None:
        self.events.append({'ph': 'X', 'cat': cat, 'name': name, 'pid': 0, 'tid': 7, 'ts': start, 'dur': dur,
                            'args': {'device': 0, 'stream': 7, 'correlation': corr}})

    def text(self) -> str:
        body = ',\n'.join('  ' + json.dumps(e, indent=4).replace('\n', '\n  ') for e in self.events)
        return ('{\n  "schemaVersion": 1,\n  "deviceProperties": [],\n  "displayTimeUnit": "ms",\n'
                f'  "traceEvents": [\n{body}\n  ],\n  "traceName": "x.pt.trace.json.gz"\n}}')

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wt', encoding='utf-8') as handle:
            handle.write(self.text())
        return path


def decode_trace(steps: int = 3, *, visible: bool = True, collective_us: float = 100, tid: int = 11) -> Kineto:
    """Graphed decode. Per step (us): smoe 300, mrow 100, ring 100 (graph); sampler 50 launched outside the
    annotation; idle 100 then the drafter's reply copy (launch not recorded); idle until the next step's graph."""
    k = Kineto(tid)
    for i in range(steps):
        base = i * 1000.0
        k.annotate(DECODE, base, base + 100)
        graph = k.launch('cudaGraphLaunch', base + 50)
        g = base + 200
        if visible:
            k.gpu('void smoe_gu(Layer, Call)', g, 300, graph)
            k.gpu('void mrow_kernel<6, true>(__nv_bfloat16 const*, int)', g + 300, 100, graph)
            k.gpu('void stage_kernel<2>(Ring)', g + 400, collective_us, graph)
        sampler = k.launch('cudaLaunchKernel', base + 150)
        k.gpu('rejection_greedy_sample_kernel', g + 400 + collective_us, 50, sampler)
        reply = k.launch('cuMemcpyHtoDAsync_v2', base + 160, record=False)
        k.gpu('Memcpy HtoD (Pinned -> Device)', g + 550 + collective_us, 10, reply, 'gpu_memcpy')
    return k


def prefill_trace(chunks: int = 3) -> Kineto:
    """Eager prefill: per chunk (us) TF grouped MoE 6000, mrow 2000, sparse attention 1500 launched in the
    annotation, then a 500 us host gap before the next chunk's first kernel."""
    k = Kineto()
    for i in range(chunks):
        base = i * 10_000.0
        k.annotate(PREFILL.format(sk=4092 * (i + 1), qk=4092 * 4092 * (i + 1)), base, base + 400)
        t = base + 500
        for name, dur in (('grouped_mma_kernel', 6000.0), ('mrow32_kernel', 2000.0), ('_sparse_attn_prefill_ragged_kernel', 1500.0)):
            k.gpu(name, t, dur, k.launch('cudaLaunchKernel', base + 10))
            t += dur
    return k


def test_sweep_splits_overlap_equally_and_reports_gaps() -> None:
    exclusive, inclusive, gaps = tb.sweep([(0, 10, 'a'), (5, 15, 'b'), (16, 18, 'a')], 0, 20)
    assert exclusive == {'a': 9.5, 'b': 7.5} and inclusive == {'a': 12, 'b': 10}
    assert gaps == [(15, 16), (18, 20)]
    assert tb.sweep([(-5, 3, 'a')], 0, 4)[0] == {'a': 3} and tb.sweep([], 0, 4)[2] == [(0, 4)]


@pytest.mark.parametrize('name,expected', [
    ('void smoe_down(Layer, Call)', 'routed_moe'), ('_sum_groups_kernel', 'routed_moe'), ('void router_kernel<6>()', 'routed_moe'),
    ('void grouped_mma3_kernel<1, 2>(...)', 'routed_moe'), ('combine_y_kernel', 'routed_moe'),
    ('void stage_kernel<2>(Ring)', 'collectives'), ('void sum_kernel(Ring)', 'collectives'), ('ncclDevKernel_AllReduce', 'collectives'),
    ('void mrow_kernel<6, true>(x)', 'dense_gemm'), ('nvjet_hsh_128x256', 'dense_gemm'),
    ('_sparse_attn_decode_partial_kernel', 'attention'), ('_fp8_paged_mqa_logits_decode_kernel', 'attention'),
    ('_indexer_k_quant_and_cache_kernel', 'attention'),
    ('_engram_lookup_kernel', 'engram'), ('_fused_engram_post_wkv_kernel', 'engram'), ('_hash_ids_kernel', 'engram'),
    ('_markov_bias_kernel', 'drafter'), ('_markov_argmax_kernel', 'drafter'),
    ('rejection_random_sample_kernel', 'sampler'), ('_gather_draft_sampled_kernel', 'sampler'), ('_bias_kernel', 'sampler'),
    ('post_collapse_exact_kernel<4>', 'other'), ('Memcpy HtoD (Pinned -> Device)', 'other'),
    ('_cp_gather_indexer_quant_cache_kernel', 'attention'), ('totally_new_kernel', 'unmatched'),
    ('mhc_post_tilelang_kernel', 'other'), ('mhc_pre_big_fuse_with_norm_tilelang_kernel', 'other'),
    ('void marlin::Marlin<1, 2, 128, 4, false>(...)', 'dense_gemm'),
    ('void reconstruct_had_kernel<5, 2, false>(__half*, unsigned short const*)', 'dense_gemm'),
    ('void vllm::deepseek_v4_fused_ops::fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelReducedGrid<c10::BFloat16, 32>', 'attention'),
    ('Kernel', 'unmatched'), ('void l2pf_kernel<4>(Layer)', 'prefetch')])
def test_bucket_patterns(name: str, expected: str) -> None:
    assert tb.bucket(name, R) == expected


def test_graphed_decode_budget_per_step() -> None:
    trace = tb.parse(io.StringIO(decode_trace().text()))
    rows = tb.steps(trace, R)
    assert [r['complete'] for r in rows] == [True, True, False]
    step = rows[0]
    assert step['gpu_period_ms'] == pytest.approx(1.0) and step['launches'] == 1 and step['activities'] == 3
    assert step['budget_ms'] == pytest.approx({'collectives': .1, 'dense_gemm': .1, 'drafter': .1, 'host_gap': .34,
                                               'other': .01, 'routed_moe': .3, 'sampler': .05})
    budget: Any = step['budget_ms']
    assert sum(budget.values()) == pytest.approx(step['gpu_period_ms'])
    assert step['unannotated_busy_ms'] == pytest.approx(.06)
    assert rows[2]['gpu_period_ms'] == pytest.approx(.5)          # tail: first to last own activity
    assert tb.graph_flag(rows)['graph_visibility'] == 'visible'
    agg = tb.aggregate(rows, 'decode', R)
    assert agg['complete_steps'] == 2 and agg['period_ms_median'] == pytest.approx(1.0)
    share: Any = agg['share']
    assert share['routed_moe'] == pytest.approx(.3) and share['host_gap'] == pytest.approx(.34)
    assert agg['idle_share'] == pytest.approx(.44) and agg['coverage'] == pytest.approx(1.0)
    assert agg['tokens_per_step_median'] == 2


def test_graphed_decode_with_no_visible_kernels_is_flagged_not_budgeted() -> None:
    rows = tb.steps(tb.parse(io.StringIO(decode_trace(visible=False).text())), R)
    assert all(r['gpu_period_ms'] is None and not r['complete'] for r in rows)
    assert all(r['graph_launches'] == 1 and r['graph_launches_with_activity'] == 0 for r in rows)
    assert tb.graph_flag(rows)['graph_visibility'] == 'invisible'
    assert tb.aggregate(rows, 'decode', R) == {'phase': 'decode', 'complete_steps': 0}
    assert all(r['cpu_period_ms'] == pytest.approx(1.0) for r in rows[:2])     # the CPU cadence is still there


def test_invisible_graph_with_eager_prep_kernels_is_not_budgeted() -> None:
    """S42 FIX 1: real decode launches eager prep kernels inside execute_model before the graph replay. With the
    graph's kernels invisible, those must not turn the step into a budgeted one with invented shares."""
    k = decode_trace(visible=False)
    for i in range(3):
        base = i * 1000.0
        k.gpu('_prepare_pos_seq_lens_kernel', base + 30, 5, k.launch('cudaLaunchKernel', base + 20))
    trace = tb.parse(io.StringIO(k.text()))
    budget = tb.capture_budget({0: trace, 1: trace}, 'decode', R)
    assert 'share' not in budget and tb.metrics({'decode': budget}) == {}
    for rank in ('0', '1'):
        info: Any = budget['ranks'][rank]       # type: ignore[index, call-overload]
        assert info['graph_visibility'] == 'invisible'
        assert info['aggregate'] == {'phase': 'decode', 'complete_steps': 0, 'unbudgeted': 'graph kernels invisible'}


def test_prefill_chunks() -> None:
    budget = tb.capture_budget({0: tb.parse(io.StringIO(prefill_trace().text())),
                                1: tb.parse(io.StringIO(prefill_trace().text()))}, 'prefill-early', R)
    chunks: Any = budget['ranks']['0']['chunks']        # type: ignore[index, call-overload]
    assert [c['complete'] for c in chunks] == [True, True, False] and [c['context_tokens'] for c in chunks] == [4092] * 3
    assert chunks[0]['gpu_period_ms'] == pytest.approx(10.0) and chunks[0]['budget_ms']['host_gap'] == pytest.approx(.5)
    assert chunks[2]['gpu_period_ms'] == pytest.approx(9.5)
    assert budget['aligned'] is True and budget['ranks']['0']['graph_visibility'] == 'no_graphs'  # type: ignore[index, call-overload]
    share: Any = budget['share']
    assert share['routed_moe'] == pytest.approx(.6) and budget['idle_share'] == pytest.approx(.05)


def test_gap_classification() -> None:
    nxt = tb.Activity('void smoe_gu(Layer, Call)', 'kernel', 100, 110, 7, 5)
    early = tb.Call('cudaLaunchKernel', 10, 14, '1', 5)
    late = tb.Call('cudaLaunchKernel', 90, 94, '1', 5)
    assert tb.classify_gap(50, nxt, early, R) == 'device_wait'
    assert tb.classify_gap(50, nxt, late, R) == 'host_gap'
    assert tb.classify_gap(50, nxt, None, R) == 'unrecorded_wait'
    copy = tb.Activity('Memcpy HtoD (Pinned -> Device)', 'memcpy', 100, 101, 7, 6)
    assert tb.classify_gap(50, copy, None, R) == 'drafter'
    assert tb.classify_gap(50, copy, tb.Call('cuMemcpyHtoDAsync_v2', 10, 12, '1', 6), R) == 'drafter'
    assert tb.classify_gap(50, copy, tb.Call('cudaMemcpyAsync', 10, 12, '1', 6), R) == 'device_wait'
    assert tb.classify_gap(50, copy, tb.Call('cudaMemcpyAsync', 60, 62, '1', 6), R) == 'host_gap'
    engram = tb.Activity('_engram_lookup_kernel', 'kernel', 100, 120, 7, 8)
    assert tb.classify_gap(50, engram, None, R) == 'engram'


def test_two_ranks_imbalance_and_alignment() -> None:
    one = tb.parse(io.StringIO(decode_trace(collective_us=100).text()))
    two = tb.parse(io.StringIO(decode_trace(collective_us=200, tid=12).text()))
    budget = tb.capture_budget({0: one, 1: two}, 'decode', R)
    assert budget['aligned'] is True and budget['collective_imbalance_ms'] == pytest.approx(.1)
    other = Kineto()
    other.annotate(DECODE.replace('generation_1', 'generation_2'), 0, 100)
    assert tb.capture_budget({0: one, 1: tb.parse(io.StringIO(other.text()))}, 'decode', R)['aligned'] is False


def test_reader_bounds_and_truncation(tmp_path: Path) -> None:
    good = decode_trace().write(tmp_path / 'a.pt.trace.json.gz')
    assert len(tb.load(good).annotations) == 3
    raw = good.read_bytes()
    (tmp_path / 'cut.gz').write_bytes(raw[:len(raw) - 9])
    with pytest.raises((EOFError, ValueError, OSError)):
        tb.load(tmp_path / 'cut.gz')
    with pytest.raises(ValueError, match='no traceEvents'):
        tb.parse(io.StringIO('{"schemaVersion": 1}'))
    with pytest.raises(ValueError, match='truncated or malformed'):
        tb.parse(io.StringIO('{"traceEvents": [ {"ph": "X", "cat": '))
    with pytest.raises(ValueError, match='decompressed bound'):
        list(tb.events(io.StringIO(decode_trace().text()), limit=100))
    with pytest.raises(ValueError, match='unrecognised iteration annotation'):
        tb.details('execute_something_else')
    assert tb.details('execute_context_1(4092)_generation_0(0)')['phase'] == 'prefill'
    assert tb.details('execute_context_1(4092)_generation_3(3)')['phase'] == 'mixed'


def test_real_cpu_kineto_trace_with_injected_gpu_events(tmp_path: Path) -> None:
    """A real trace written by the local torch profiler (CPU activity only), with runtime and kernel events inserted
    into its traceEvents text: the streaming reader must agree with json.load and find the annotations."""
    torch = pytest.importorskip('torch')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    path = tmp_path / 'real.json'
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        for _ in range(3):
            with torch.profiler.record_function(DECODE):
                torch.ones(64) @ torch.ones(64)
    prof.export_chrome_trace(str(path))
    text = path.read_text()
    loaded = json.loads(text)
    anns = sorted((e for e in loaded['traceEvents'] if e.get('cat') == 'user_annotation'), key=lambda e: e['ts'])
    assert len(anns) == 3
    k = Kineto(anns[0]['tid'])
    k.events = []
    for i, a in enumerate(anns):
        corr = k.launch('cudaGraphLaunch', a['ts'] + a['dur'] / 2)
        k.gpu('void smoe_gu(Layer, Call)', anns[-1]['ts'] + anns[-1]['dur'] + 100 + 1000 * i, 300, corr)
    injected = ''.join('\n  ' + json.dumps(e) + ',' for e in k.events)
    marker = '"traceEvents": ['
    assert text.count(marker) == 1
    patched = tmp_path / 'patched.pt.trace.json.gz'
    with gzip.open(patched, 'wt', encoding='utf-8') as handle:
        handle.write(text.replace(marker, marker + injected))
    trace = tb.load(patched)
    assert [a.name for a in trace.annotations] == [DECODE] * 3
    assert trace.events == len(loaded['traceEvents']) + len(k.events)
    rows = tb.steps(trace, R)
    assert [r['complete'] for r in rows] == [True, True, False] and rows[0]['gpu_period_ms'] == pytest.approx(1.0)
    assert rows[0]['budget_ms'] == pytest.approx({'routed_moe': .3, 'device_wait': .7})   # launched long before


def test_rank_next_maps_to_experiments() -> None:
    experiments = json.loads(tb.EXPERIMENTS.read_text())
    ids = {e['id'] for e in experiments['experiments']}
    assert {str(h['id']) for h in R.hypotheses} == ids          # every hypothesis owned, no stray rule
    values = {'prof-32k-b1:decode.share.routed_moe': .45, 'prof-150k-b1:decode.share.routed_moe': .38,
              'prof-32k-b1:decode.ms.host_gap': .9, 'prof-32k-b1:decode.share.host_gap': .07,
              'prof-32k-b1:decode.share.collectives': .03, 'prof-32k-b1:decode.collective_imbalance_ms': .2,
              'prof-32k-b1:decode.share.sampler': .01, 'prof-32k-b1:decode.share.drafter': .12,
              'prof-32k-b1:prefill-early.share.routed_moe': .35, 'prof-32k-b1:prefill-early.idle_share': .02}
    out: Any = tb.rank_next(values, R, experiments, [])
    assert out['ranking'] == 'OK' and out['next'] == ['decode-smoe-bandwidth', 'depth-policy-recalibration', 'remaining-host-gaps']
    status = {h['id']: h['status'] for h in out['hypotheses']}
    assert status['prefill-grouped-kernel'] == 'NOT_SUPPORTED' and status['expert-parallel-split'] == 'NOT_MEASURED'
    assert status['clocks-and-cooling'] == 'NOT_FROM_TRACE' and status['collective-boundary-fusion'] == 'NOT_SUPPORTED'
    assert tb.rank_next(values, R, experiments, ['x'])['ranking'] == 'LIMITED'


def stream(submitted: float, first: float, gap_ms: float, until: float, slow: tuple[float, float, float] | None = None,
           tokens: int = 32000) -> dict[str, Any]:
    times, t = [], first
    while t < until:
        times.append((t - submitted) * 1e3)
        t += (slow[2] if slow and slow[0] <= t < slow[1] else gap_ms) / 1e3
    return {'submitted_rel_s': submitted, 'ttft_s': first - submitted, 'prompt_tokens': tokens, 'sse_times_ms': times}


def window(tmp_path: Path, *, visible: bool = True) -> Path:
    root = tmp_path / 'SP1-W-test'
    rows = []
    for rank, host in enumerate(('sparkhead', 'sparkworker')):
        for kind, trace in (('prefill-early', prefill_trace()), ('decode', decode_trace(visible=visible, tid=11 + rank))):
            path = trace.write(root / 'traces' / 'prof-32k-b1' / host / f'{kind}-rank{rank}.pt.trace.json.gz')
            rows.append({'cell': 'prof-32k-b1', 'capture': kind, 'status': 'TAKEN', 'rank': rank, 'host': host,
                         'file': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (root / 'traces-manifest.json').write_text(json.dumps(rows))
    (root / 'cells').mkdir()
    control = {'cell': 'ctl-32k-b1', 'captures': [], 'streams': [stream(0, 10.0, 30, 40)]}
    profiled = {'cell': 'prof-32k-b1', 'streams': [stream(0, 10.25, 30, 40, slow=(20.0, 23.5, 31.5))],
                'captures': [{'kind': 'prefill-early', 'status': 'TAKEN', 'start_call_rel_s': -.01, 'stop_return_rel_s': 5.2},
                             {'kind': 'decode', 'status': 'TAKEN', 'start_call_rel_s': 20.0, 'stop_return_rel_s': 23.5}]}
    for cell in (control, profiled):
        (root / 'cells' / f"{cell['cell']}.json").write_text(json.dumps(cell))
    return root


def test_window_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = window(tmp_path)
    result: Any = tb.analyze_window(root)
    decode = result['captures']['prof-32k-b1:decode']
    assert decode['ranks_present'] == [0, 1] and decode['share']['routed_moe'] == pytest.approx(.3)
    cost = result['overhead']
    row = cost['cells']['prof-32k-b1']
    assert row['inside_over_outside'] == pytest.approx(1.05) and row['outside_over_control'] == pytest.approx(1.0)
    assert row['prefill-early_ttft_extra_s_per_chunk'] == pytest.approx(.25 / 3)        # 3 chunks in the trace
    assert row['prefill-early_relative'] == pytest.approx((.25 / 3) / (10.0 / 8))       # 32000 tokens = 8 chunks
    assert cost['decode_overhead'] == pytest.approx(.05)
    assert result['next_work']['ranking'] == 'OK', result['next_work']['limited_by']
    assert result['next_work']['next'][0] == 'prefill-grouped-kernel'                     # 60 % TF MoE in prefill
    assert tb.main(['--window', str(root)]) == 0 and (root / 'trace-budget.json').is_file()
    assert 'prof-32k-b1:decode' in capsys.readouterr().out
    assert tb.main(['--window', str(root)]) == 1                                            # never overwrites
    blob = (root / 'trace-budget.json').read_text()
    assert 'sse_times_ms' not in blob and '"prompt' not in blob and 'response' not in blob


def test_window_limits_and_tamper(tmp_path: Path) -> None:
    root = window(tmp_path, visible=False)
    out: Any = tb.analyze_window(root)
    assert out['next_work']['ranking'] == 'LIMITED'
    assert any('graph kernels invisible' in x for x in out['next_work']['limited_by'])
    manifest = json.loads((root / 'traces-manifest.json').read_text())
    victim = root / manifest[0]['file']
    data = bytearray(victim.read_bytes())
    data[len(data) // 2] ^= 1
    victim.write_bytes(bytes(data))
    with pytest.raises(ValueError, match='sha256 differs'):
        tb.analyze_window(root)
    manifest[0]['file'] = '../outside.gz'
    (root / 'traces-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        tb.analyze_window(root)
