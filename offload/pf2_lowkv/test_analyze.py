"""PF2-LOWKV analysis on synthetic windows (CPU only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from offload.gb10_budget import test_trace_budget as synth
from offload.pf2_lowkv import analyze as a

GIB_KB = 2**20


@pytest.mark.parametrize('ttft,want', [(70.0, 'fast'), (78.4, 'fast'), (80.0, 'ambiguous'), (84.0, 'slow'), (95.3, 'slow')])
def test_classify_against_the_pre_registered_ratios(ttft: float, want: str) -> None:
    assert a.classify(ttft, 70.0) == want


@pytest.mark.parametrize('starts,verdict', [
    ({'s1': ('H', 'slow'), 's2': ('L', 'fast'), 's3': ('H', 'slow'), 's4': ('L', 'fast')}, 'HEADROOM_IMPLICATED'),
    ({'s1': ('H', 'fast'), 's2': ('L', 'fast'), 's3': ('H', 'slow'), 's4': ('L', 'fast')}, 'HEADROOM_IMPLICATED'),
    ({'s1': ('H', 'fast'), 's2': ('L', 'fast'), 's3': ('H', 'fast'), 's4': ('L', 'fast')}, 'NO_CAUSE_SHOWN'),
    ({'s1': ('H', 'slow'), 's2': ('L', 'slow'), 's3': ('H', 'slow'), 's4': ('L', 'fast')}, 'NOT_HEADROOM'),
    ({'s1': ('H', 'fast'), 's2': ('L', 'slow'), 's3': ('H', 'fast'), 's4': ('L', 'fast')}, 'NOT_HEADROOM'),
    ({'s1': ('H', 'slow'), 's2': ('L', 'mixed'), 's3': ('H', 'slow'), 's4': ('L', 'fast')}, 'AMBIGUOUS'),
    ({'s1': ('H', 'slow'), 's2': ('L', 'fast'), 's3': ('H', 'slow'), 's4': ('L', 'incomplete')}, 'INCOMPLETE'),
    # one clearly slow L start settles it even when other starts are mixed or incomplete (S44 note 1)
    ({'s1': ('H', 'mixed'), 's2': ('L', 'slow'), 's3': ('H', 'incomplete'), 's4': ('L', 'fast')}, 'NOT_HEADROOM'),
])
def test_reading_truth_table(starts: dict[str, tuple[str, str]], verdict: str) -> None:
    assert a.reading(starts)['verdict'] == verdict


def test_start_class() -> None:
    assert a.start_class(['fast', 'fast']) == 'fast' and a.start_class(['slow', 'slow']) == 'slow'
    assert a.start_class(['fast', 'slow']) == 'mixed' and a.start_class(['fast', 'ambiguous']) == 'ambiguous'
    assert a.start_class(['fast']) == 'incomplete'


def tele_row(t: float, avail: int, stall: int, psi_total: float, swap: int = 10 * GIB_KB) -> dict[str, Any]:
    return {'kind': 'memory', 'utc': t, 'kb': {'MemAvailable': avail, 'SwapFree': swap},
            'vmstat': {**{k: 0 for k in a.COUNTERS}, 'compact_stall': stall, 'allocstall_normal': stall // 2},
            'psi': {'some': {'avg10': psi_total / 1e6, 'total': psi_total}, 'full': {'avg10': 0.0, 'total': psi_total / 2}}}


def test_memory_around_a_prefill(tmp_path: Path) -> None:
    path = tmp_path / 't.jsonl'
    rows = [tele_row(100 + i, 7 * GIB_KB - i * 1000, 10 + 3 * i, 1000.0 * i) for i in range(30)]
    rows.insert(5, {'kind': 'gpu', 'utc': 105.5, 'csv': 'x'})
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    got: Any = a.around(a.telemetry(path), 110, 120)
    assert got['samples'] == 11 and got['vmstat_delta']['compact_stall'] == 30
    assert got['vmstat_delta']['allocstall_normal'] == 15
    assert got['psi_some_total_delta_us'] == 10000.0 and got['mem_available_min_kb'] == 7 * GIB_KB - 20000
    assert a.around(a.telemetry(path), 500, 510) == {'samples': 0, 'status': 'NO_DATA'}


def test_procmem_sums_workers_and_flags_slow_reads(tmp_path: Path) -> None:
    path = tmp_path / 'p.jsonl'
    rows = [{'kind': 'procmem', 'utc': 100.0 + i, 'procs': [
        {'pid': 1, 'role': 'api_server', 'Pss': 1000, 'Rss': 2000, 'rollup_ms': 1.0},
        {'pid': 2, 'role': 'worker', 'Pss': 5000 + 100 * i, 'Rss': 6000, 'rollup_ms': 60.0 if i == 2 else 3.0},
        {'pid': 3, 'role': 'worker', 'Pss': 10, 'Rss': 20, 'rollup_ms': 1.0}]} for i in range(5)]
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    got: Any = a.procmem_delta(a.procmem(path), 100, 104)
    assert got['worker'] == {'Pss_delta_kb': 400, 'Rss_delta_kb': 0, 'rollup_ms_max': 60}
    assert a.procmem(tmp_path / 'missing.jsonl') == []


def kineto(moe_us: float, grid: list[int], extra: bool = False) -> synth.Kineto:
    k = synth.prefill_trace()
    for event in k.events:
        if event.get('cat') == 'kernel':
            event['args'].update({'grid': grid if 'grouped' in event['name'] else [64, 1, 1], 'block': [128, 1, 1],
                                  'registers per thread': 96, 'shared memory': 0})
            if 'grouped' in event['name']:
                event['dur'] = moe_us
    if extra:
        k.gpu('grouped_fallback_kernel', 40_000.0, 500.0, k.launch('cudaLaunchKernel', 39_000.0))
    return k


def test_signature_diff_lists_changed_and_only_one_side(tmp_path: Path) -> None:
    fast = a.signatures(kineto(6000.0, [96, 1, 1]).write(tmp_path / 'f.pt.trace.json.gz'))
    slow = a.signatures(kineto(8000.0, [96, 1, 1], extra=True).write(tmp_path / 's.pt.trace.json.gz'))
    got: Any = a.diff(fast, slow)
    assert got['only_second_count'] == 1 and got['only_second'][0][0] == 'grouped_fallback_kernel'
    top = got['changed'][0]
    assert top['signature'][0] == 'grouped_mma_kernel' and top['routed_moe'] is True
    assert abs(top['mean_us_ratio'] - 8000 / 6000) < 1e-9
    regrid: Any = a.diff(fast, a.signatures(kineto(6000.0, [48, 1, 1]).write(tmp_path / 'g.pt.trace.json.gz')))
    assert regrid['only_first_count'] == regrid['only_second_count'] == 1                # same name, other launch config


def write_window(root: Path, ttfts: dict[str, tuple[float, float]], *, trace: bool = True) -> Path:
    """A PF2 window directory as pf2_window writes it: summary, cells, telemetry, procmem, traces + manifest."""
    roles = {'s1-H': 'H', 's2-L': 'L', 's3-H': 'H', 's4-L': 'L'}
    root.mkdir()
    (root / 'cells').mkdir()
    starts: dict[str, Any] = {}
    manifest = []
    t0 = 1_000_000.0
    tele = {h: [] for h in a.HOSTS}           # type: dict[str, list[dict[str, Any]]]
    for i, (name, role) in enumerate(roles.items()):
        if name not in ttfts:
            starts[name] = {'status': 'SKIPPED_TIME', 'role': role}
            continue
        origin = t0 + i * 1000
        starts[name] = {'status': 'COMPLETE', 'role': role, 'position': i + 1, 'kv_bytes': 1, 'cells': {}}
        for j, cell in enumerate(('c150-a', 'c150-b', 'p150')):
            ttft = ttfts[name][min(j, 1)] * (1.2 if cell == 'p150' else 1.0)
            submitted = 200.0 * j
            stream = {'stream': 0, 'submitted_rel_s': submitted, 'ttft_s': ttft, 'total_s': ttft + 27.5,
                      'usage': {'completion_tokens': 1500, 'prompt_tokens': 149000}}
            captures = [{'kind': 'prefill-late', 'status': 'TAKEN'}] if cell == 'p150' else []
            (root / 'cells' / f'{name}-{cell}.json').write_text(json.dumps(
                {'cell': cell, 'origin_wall': origin, 'streams': [stream], 'captures': captures}))
        for s in range(0, 700):
            for h in a.HOSTS:
                tele[h].append(tele_row(origin - 20 + s, 7 * GIB_KB, s * (3 if role == 'H' else 1), 100.0 * s))
        for h in a.HOSTS:
            (root / f'procmem-{name}-{h}.jsonl').write_text(''.join(json.dumps(
                {'kind': 'procmem', 'utc': origin - 20 + s, 'procs': [{'pid': 9, 'role': 'worker', 'Pss': 1000 + s}]}) + '\n'
                for s in range(0, 700, 5)))
        if trace:
            slow = ttfts[name][0] > 84
            for rank, host in enumerate(a.HOSTS):
                path = kineto(8000.0 if slow else 6000.0, [96, 1, 1]).write(
                    root / 'traces' / f'{name}-p150' / host / f'rank{rank}.pt.trace.json.gz')
                manifest.append({'cell': f'{name}-p150', 'capture': 'prefill-late', 'rank': rank, 'host': host,
                                 'file': str(path.relative_to(root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    for h, rows in tele.items():
        (root / f'telemetry-{h}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    if trace:
        (root / 'traces-manifest.json').write_text(json.dumps(manifest))
    (root / 'summary.json').write_text(json.dumps({'status': 'COMPLETE', 'starts': starts}))
    return root


def test_window_headroom_implicated_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = write_window(tmp_path / 'PF2-W', {'s1-H': (95.0, 93.0), 's2-L': (70.0, 70.5), 's3-H': (92.0, 91.0),
                                             's4-L': (69.5, 70.1)})
    assert a.main(['--window', str(root), '--reference-ttft', '69.0']) == 0
    out = capsys.readouterr().out
    assert out.startswith('PF2-LOWKV reading: HEADROOM_IMPLICATED')
    result = json.loads((root / 'pf2-analysis.json').read_text())
    assert result['starts']['s1-H']['class'] == 'slow' and result['starts']['s2-L']['class'] == 'fast'
    mem = result['starts']['s1-H']['cells']['c150-a']['memory']['sparkworker-wired']
    assert mem['vmstat_delta']['compact_stall'] > 0 and mem['samples'] > 90
    assert result['starts']['s1-H']['cells']['c150-a']['procmem']['sparkhead']['worker']['Pss_delta_kb'] > 0
    assert result['dispatch']['diff']['label'] == 'fast vs slow' and result['dispatch']['diff']['first'] == 's2-L'
    assert result['dispatch']['diff']['ranks']['rank1']['changed'][0]['signature'][0] == 'grouped_mma_kernel'
    assert set(result['dispatch']['budgets']) == {'s1-H', 's2-L', 's3-H', 's4-L'}
    assert abs(result['decode']['l_vs_h']) < 1e-9 and result['decode']['flag'] is False
    with pytest.raises(FileExistsError):
        a.main(['--window', str(root)])                       # never overwrites an analysis


def test_window_without_contrast_and_incomplete(tmp_path: Path) -> None:
    root = write_window(tmp_path / 'W1', {'s1-H': (70.0, 70.0), 's2-L': (70.0, 70.0), 's3-H': (70.0, 70.0), 's4-L': (70.0, 70.0)})
    result: Any = a.analyze(root, 70.0, 'test')
    assert result['reading']['verdict'] == 'NO_CAUSE_SHOWN'
    assert result['dispatch']['diff']['label'] == 'H vs L (no state contrast)'
    short = write_window(tmp_path / 'W2', {'s1-H': (95.0, 95.0), 's2-L': (70.0, 70.0), 's3-H': (95.0, 95.0)}, trace=False)
    result = a.analyze(short, 70.0, 'test')
    assert result['reading']['verdict'] == 'INCOMPLETE' and result['dispatch'] == {'status': 'NO_TRACES'}


def test_tampered_trace_is_refused(tmp_path: Path) -> None:
    root = write_window(tmp_path / 'W', {'s1-H': (95.0, 95.0), 's2-L': (70.0, 70.0), 's3-H': (95.0, 95.0), 's4-L': (70.0, 70.0)})
    victim = next((root / 'traces').rglob('*.gz'))
    victim.write_bytes(victim.read_bytes() + b'x')
    with pytest.raises(ValueError, match='sha256'):
        a.analyze(root, 70.0, 'test')


def test_reference_from_an_sp1_window(tmp_path: Path) -> None:
    ref = tmp_path / 'SP1-W'
    (ref / 'cells').mkdir(parents=True)
    (ref / 'cells' / 'ctl-150k-b1.json').write_text(json.dumps({'cell': 'ctl-150k-b1', 'streams': [{'ttft_s': 68.79}]}))
    assert a.reference(ref, None) == (68.79, 'SP1-W/cells/ctl-150k-b1.json')
    assert a.reference(None, None)[0] == a.DEFAULT_REF_S and a.reference(None, 71.0) == (71.0, 'given')
