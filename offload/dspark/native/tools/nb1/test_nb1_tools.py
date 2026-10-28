"""CPU tests for the NB1 probe helpers and collector (no GPU, no checkpoint)."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import torch

from offload.dspark.native.tools.nb1 import collect
from offload.dspark.native.tools.nb1.stage_probe import Layout, Stat, compare_rows, label_fp64, replay_args


def test_draft_layout_slices_request_rows() -> None:
    layout = Layout('draft', 3)
    alone = torch.zeros(5, 7)
    assert layout.unit(alone, torch.zeros(15, 7)) == 5
    assert layout.unit(alone, torch.zeros(10, 7)) == 0          # not B x alone rows
    assert layout.unit(torch.zeros(5, 128, 4), torch.zeros(5, 128, 4)) == 0   # shared window ring


def test_insert_layout_uses_buckets() -> None:
    layout = Layout('insert', 2, n=5, ba=8, bp=16)
    assert layout.unit(torch.zeros(8, 3), torch.zeros(16, 3)) == 5
    assert layout.unit(torch.zeros(5, 3), torch.zeros(5, 3)) == 0


def test_compare_rows_detects_one_bit() -> None:
    layout = Layout('draft', 2)
    alone = (torch.ones(5, 4, dtype=torch.bfloat16),)
    packed = torch.ones(10, 4, dtype=torch.bfloat16)
    stat = Stat()
    assert compare_rows(layout, alone, (packed,), 1, stat)
    packed[7, 2] = 1.0078125                                   # the next BF16 value after 1.0
    assert not compare_rows(layout, alone, (packed,), 1, stat)
    assert stat.comparisons == 2 and stat.exact == 1 and stat.differing == 1


def test_signed_zero_is_a_difference() -> None:
    stat = Stat()
    assert not stat.add(torch.zeros(3), torch.tensor([0.0, -0.0, 0.0]))


def test_replay_args_rebuilds_rows_from_alone_inputs() -> None:
    layout = Layout('draft', 2)
    weight = torch.arange(6.0)
    packed = ((torch.full((10, 3), 9.0), weight), {'out': torch.zeros(10, 3)})
    alones = [((torch.full((5, 3), float(i)), weight), {'out': torch.zeros(5, 3)}) for i in range(2)]
    (x, w), kw = replay_args(layout, packed, alones, (0, 1))
    assert torch.equal(x[:5], torch.zeros(5, 3)) and torch.equal(x[5:], torch.ones(5, 3))
    assert w is weight and kw['out'].shape == (10, 3)


def test_label_fp64_rule_inputs() -> None:
    ref = torch.zeros(5, 4, dtype=torch.float64)
    alone = torch.full((5, 4), 1.0)
    packed = torch.cat((torch.full((5, 4), 3.0), torch.full((5, 4), 1.5)))
    cell = label_fp64([alone], [packed], [ref], [5], 1)
    assert cell['gap_max'] == 0.5 and cell['alone_err_max'] == 1.0 and cell['packed_err_mean'] == 1.5
    assert cell['within_own_error'] and not cell['packed_not_worse']


def test_collect_family_and_within_rule() -> None:
    assert collect.family('r1.ffn2.dense:mtp.2.attn.wo_b[0:5120,0:4096]#0') == 'ffn2.dense:attn.wo_b[0:5120,0:4096]'
    cells = [dict(label='r0.attn0.dense:mtp.0.attn.wo_a#0', kind='captured', B=2, gap_max=1.0, gap_mean=1.4,
                  alone_err_max=1.0, alone_err_mean=1.0, packed_err_mean=1.0, within_own_error=False),
             dict(label='r1.attn0.dense:mtp.0.attn.wo_a#0', kind='captured', B=2, gap_max=1.0, gap_mean=1.5,
                  alone_err_max=1.0, alone_err_mean=1.0, packed_err_mean=2.0, within_own_error=False)]
    row, = collect.fp64_summary(cells, ('kind', 'B'))
    assert row['cells'] == 2 and row['within_own_error'] == 1 and row['packed_err_mean_over_alone'] == 2.0


def test_bootstrap_confines_writes_to_scratch(tmp_path: Path) -> None:
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    code = '''from pathlib import Path
import sys
from offload.dspark.native.tools.nb1.stage_probe import bootstrap
bootstrap(Path(sys.argv[1]), Path(sys.argv[2]))
(Path(sys.argv[2]) / 'allowed').write_text('ok')
Path(sys.argv[3]).write_text('forbidden')
'''
    target = tmp_path / 'outside'
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path), str(scratch), str(target)],
                            text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert 'filesystem write refused' in result.stderr
    assert (scratch / 'allowed').read_text() == 'ok'
    assert not target.exists()


def test_gate_requires_inert_off_invariant_on_and_unchanged_b1() -> None:
    from offload.dspark.native.tools.nb1 import gate
    base = dict(output_digest='d', b1_digests={'f/0': {'eager_ids': 'a'}}, repo='x', seconds=1.0,
                final={'captured/B2/graph5': dict(comparisons=4, ids_exact=[4] * 5, conf_exact=4, conf_max_abs=0.0)},
                first_divergent_e2e={}, isolated_batch_dependent_ops=[], insert_ring={}, depth_transitions={})
    off = dict(base, repo='y', seconds=2.0)
    assert gate.gate(base, off, base)['passed']
    on_bad = dict(base, final={'captured/B2/graph5': dict(comparisons=4, ids_exact=[4] * 5, conf_exact=3,
                                                          conf_max_abs=1e-3)})
    assert not gate.gate(base, off, on_bad)['passed']
    assert not gate.gate(base, dict(off, output_digest='e'), base)['passed']
    assert not gate.gate(base, off, dict(base, b1_digests={'f/0': {'eager_ids': 'b'}}))['passed']


def test_level2_gate_needs_invariant_insertion_and_not_worse_accuracy() -> None:
    from offload.dspark.native.tools.nb1 import gate
    cell = {'calls': 4, 'mean_abs_err': 1.0, 'max_abs_err': 2.0, 'mean_abs_ref': 1.0}
    frozen = dict(b1_digests={'a': 1, 'b': 2}, insert_alone_fp64={'r0.insert.dense:w/n32/s': cell})
    on2 = dict(b1_digests={'a': 1, 'b': 3}, final={'k': dict(comparisons=2, ids_exact=[2], conf_exact=2)},
               insert_ring={'n5/B2': dict(comparisons=4, exact=4)},
               insert_alone_fp64={'r0.insert.fixed_rows:w/n32/s': dict(cell, mean_abs_err=1.04)})
    result = gate.gate_level2(frozen, on2)
    assert result['passed'] and result['b1_changed'] == ['b']
    worse = dict(on2, insert_alone_fp64={'r0.insert.fixed_rows:w/n32/s': dict(cell, mean_abs_err=1.06)})
    assert not gate.gate_level2(frozen, worse)['passed']
    leaky = dict(on2, insert_ring={'n5/B2': dict(comparisons=4, exact=3)})
    assert not gate.gate_level2(frozen, leaky)['passed']


def test_latency_summary_takes_the_slower_shard_and_counts_microbatching_as_b1_replays(tmp_path: Path) -> None:
    from offload.dspark.native.tools.nb1 import latency_summary
    cells = {}
    for rank, scale in (('r0', 1.0), ('r1', 1.5)):
        for arm, base in (('k2', 100.0), ('native', 200.0), ('native-invariant', 220.0)):
            for b in range(1, 5):
                for n in (4, 5):
                    v = scale * base * b
                    cells[f'{rank}/{arm}/B{b}/n{n}'] = dict(median_us=v, min_us=v, max_us=v * 1.02)
            cells[f'{rank}/{arm}/insert8'] = dict(median_us=scale * base, min_us=1.0, max_us=1.0)
        for b in range(2, 5):
            for n in (4, 5):
                cells[f'{rank}/native-microbatched/B{b}/n{n}'] = dict(median_us=scale * 210.0 * b,
                                                                      min_us=1.0, max_us=1.0)
    logs = []
    for rank in ('r0', 'r1'):                                   # one process per rank
        logs.append(tmp_path / f'latency-{rank}.log')
        logs[-1].write_text('PHASE x\nRESULT_JSON ' + __import__('json').dumps(dict(
            mode='m', reps=5, iters=200, warmup=40, cells={k: v for k, v in cells.items() if k.startswith(rank)},
            clocks=[], loads_seconds={}, torch_version='t')) + '\n')
    with pytest.raises(ValueError):
        latency_summary.load(logs[0], logs[0])
    summary = latency_summary.summarize(latency_summary.load(*logs))
    b1 = summary['draft'][0]
    assert b1['native']['slowest_rank_us'] == 300.0 and b1['native-microbatched']['slowest_rank_us'] == 300.0
    b3 = next(r for r in summary['draft'] if r['B'] == 3 and r['n_out'] == 5)
    assert b3['native-invariant']['vs_native'] == 1.1 and abs(b3['native-microbatched']['vs_native'] - 1.05) < 1e-12
    assert summary['insert'][0]['native-invariant']['slowest_rank_us'] == 330.0
    assert '|3|5|450|900|990|945|' in latency_summary.markdown(summary)
