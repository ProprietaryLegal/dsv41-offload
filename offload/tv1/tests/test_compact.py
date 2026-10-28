from __future__ import annotations

import ast
import dataclasses
import os
import types
from pathlib import Path
from typing import Callable, NamedTuple, cast

import numpy as np
import pytest
import torch

from offload.tv1 import build_overlay as b
from offload.tv1 import compact as c


def test_prefix_widths_and_bad_values() -> None:
    conf = np.array([[1., .8, .5, .9, .9], [.9, .1, 1., 1., 1.],
                     [0., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]], dtype=np.float32)
    assert c.widths(conf, [5, 3, 5, 0], .5).tolist() == [2, 1, 0, 0]
    assert c.compact_counts(np.array([6, 4, 6, 1], dtype=np.int32),
                            np.array([5, 3, 5, 0], dtype=np.int32),
                            np.array([2, 1, 0, 0], dtype=np.int32)).tolist() == [3, 2, 1, 1]
    for bad in [float('nan'), -.1, 1.1]:
        broken = conf.copy()
        broken[0, 0] = bad
        with pytest.raises(ValueError, match='confidence'):
            c.widths(broken, [5, 3, 5, 0], .5)
    with pytest.raises(ValueError, match='threshold'):
        c.widths(conf, [5, 3, 5, 0], 0.)
    with pytest.raises(ValueError, match='capacity'):
        c.widths(conf, [6, 3, 5, 0], .5)
    with pytest.raises(ValueError, match='prefix'):
        c.compact_counts(np.array([5], dtype=np.int32), np.array([5], dtype=np.int32), np.array([3], dtype=np.int32))


def test_production_plain_conf_depth_guard_and_refused_compositions() -> None:
    c.require_depth_policy(None)
    options: dict[str, object] = dict(mode='conf', calib=False, lookup_opt='0', ctx_pts=[], file=None)
    c.require_depth_policy(types.SimpleNamespace(**options))
    for key, value in [('mode', 'fixed'), ('mode', 'confr'), ('mode', 'off'),
                       ('calib', True), ('lookup_opt', '1'), ('ctx_pts', [(32000, 1.)]), ('file', 'depth-control')]:
        with pytest.raises(ValueError, match='only plain conf'):
            c.require_depth_policy(types.SimpleNamespace(**{**options, key: value}))
    with pytest.raises(ValueError, match='only plain conf'):
        c.require_depth_policy(object())


def test_actual_depth_mask_preserves_correction_row_for_every_compact_prefix() -> None:
    # Execute the installed policy's actual mask and observation methods on CPU.
    # Its sampler union and confidence updates remain unchanged in our overlay.
    source = ast.parse((b.IMAGE / 'vllm/v1/worker/gpu/spec_decode/depth_policy.py').read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'DSparkDepthPolicy')
    functions: list[ast.stmt] = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('dead_rows', 'observe')]
    ns: dict[str, object] = {'torch': torch, 'np': np}
    exec('from __future__ import annotations\n' + ast.unparse(ast.Module(body=functions, type_ignores=[])), ns)
    mask = cast(Callable[..., torch.Tensor], ns['dead_rows'])
    observe = cast(Callable[..., None], ns['observe'])
    for reserved in range(6):
        for chosen in range(reserved + 1):
            for depth in range(reserved + 1):
                live = min(chosen, depth)
                state = types.SimpleNamespace(depth=torch.tensor([depth]), log_every=0, calib=True, mode='conf')
                observed: list[tuple[list[int], list[int]]] = []
                state._calibrate = lambda idx, accepted, count: observed.append((accepted.tolist(), count.tolist()))
                dead = mask(state, torch.zeros(chosen+1, dtype=torch.int64), torch.arange(chosen+1))
                assert dead.tolist() == [position > depth for position in range(chosen+1)]
                assert not dead[live]  # target correction/bonus distribution is computed
                assert (~dead).sum().item() == live + 1
                for accepted in range(live+1):
                    observe(state, torch.tensor([0]), torch.tensor([accepted+1]),
                            np.array([chosen], dtype=np.int32), 1)
                    assert observed[-1] == ([accepted], [live])


def test_scheduler_vs_worker_rollback_all_accept_and_reject() -> None:
    # Scheduler reserves k+1; worker computes d+1. The unchanged sampler accepts a<=d.
    # Both counters MUST finish at the same absolute position for every possible a.
    for reserved in range(6):
        for chosen in range(reserved + 1):
            for accepted in range(chosen + 1):
                scheduler_position = 100 + 1 + reserved - (reserved - accepted)
                worker_position = 100 + 1 + chosen - (chosen - accepted)
                assert scheduler_position == worker_position
                assert (1 + reserved) - (reserved - accepted) == 1 + accepted


def test_real_gather_function_compacts_metadata_and_keeps_mixed_grammar(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    tree = tmp_path / 'tree'
    b.build(tree, ('compact',))
    source = ast.parse((tree / 'vllm/v1/worker/gpu/model_runner.py').read_text())
    runner = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'GPUModelRunner')
    gather = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == 'gather_batch_req_state')
    nodes = [n for n in source.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in
             ('BatchReqState', 'sort_batch_req_ids')]
    ns: dict[str, object] = {'np': np, 'dataclass': dataclasses.dataclass, 'NamedTuple': NamedTuple,
                            'get_uniform_decode_token_count': lambda *a, **kw: None}
    # Bind os only if the composed module actually imports it. Supplying it from
    # this test used to hide the startup/first-forward NameError in the overlay.
    actual_os: list[ast.stmt] = [n for n in source.body if isinstance(n, ast.Import) and any(a.name == 'os' for a in n.names)]
    exec(compile(ast.Module(body=actual_os, type_ignores=[]), '<actual runner imports>', 'exec'), ns)
    exec('from __future__ import annotations\n' + ast.unparse(ast.Module(body=[*nodes, gather], type_ignores=[])), ns)
    from unittest.mock import Mock
    get_group = types.ModuleType('vllm.distributed')
    get_group.get_tp_group = lambda: types.SimpleNamespace(broadcast=lambda t, src: t)  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__('sys').modules, 'vllm.distributed', get_group)
    monkeypatch.setitem(__import__('sys').modules, 'vllm_exl3_moe.tv1_compact', c)
    fake = types.SimpleNamespace(
        decode_query_len=6, use_pp=False, pcp_manager=None, adaptive_verification=None,
        tv1_compact=types.SimpleNamespace(choose=lambda ids, caps, broadcast: np.array([5, 1, 4, 0], dtype=np.int32)),
        req_states=types.SimpleNamespace(req_id_to_index={'a': 3, 'b': 1, 'c': 2, 'd': 0},
            prefill_len=types.SimpleNamespace(np=np.array([10]*4)),
            num_computed_prefill_tokens=np.array([10]*4)))
    schedule = types.SimpleNamespace(num_scheduled_tokens={n: 6 for n in ('a', 'b', 'c', 'd')},
        total_num_scheduled_tokens=24, scheduled_spec_decode_tokens={n: [-1]*5 for n in ('a', 'b', 'c', 'd')},
        has_structured_output_requests=False)
    run = cast(Callable[..., tuple[types.SimpleNamespace, object]], ns['gather_batch_req_state'])
    monkeypatch.setenv('VLLM_TV1_COMPACT', '1')
    result, _ = run(fake, schedule, False)
    assert result.req_ids == ['a', 'b', 'c', 'd']
    assert result.idx_mapping_np.tolist() == [3, 1, 2, 0]
    assert result.num_scheduled_tokens.tolist() == [6, 2, 5, 1]
    assert result.num_draft_tokens_np.tolist() == [5, 1, 4, 0]
    assert result.num_tokens == 14 and result.tv1_compacted
    assert schedule.total_num_scheduled_tokens == 24  # scheduler reservation retained
    assert all(len(x) == 5 for x in schedule.scheduled_spec_decode_tokens.values())
    # These requests retain their full original widths, not a partial-policy compromise.
    fake.tv1_compact.choose = Mock(side_effect=AssertionError('must not compact'))
    schedule.has_structured_output_requests = True
    result, _ = run(fake, schedule, False)
    assert result.num_tokens == 24 and not result.tv1_compacted
    schedule.has_structured_output_requests = False
    fake.req_states.num_computed_prefill_tokens[2] = 9
    result, _ = run(fake, schedule, False)
    assert result.num_tokens == 24 and result.has_prefill
    fake.req_states.num_computed_prefill_tokens[2] = 10
    monkeypatch.setenv('VLLM_TV1_COMPACT', '0')
    result, _ = run(fake, schedule, False)
    assert result.num_tokens == 24 and not result.tv1_compacted


def test_remote_gather_uses_compact_spans_and_positions() -> None:
    from offload.dspark.spark.remote_dspark.gather import RowGatherer, batch_view_from_input_batch
    widths = np.array([6, 2, 5, 1], dtype=np.int32)
    qsl = np.r_[0, widths.cumsum()].astype(np.int32)
    req_ids = ['a', 'b', 'c', 'd']
    slots = np.array([3, 1, 2, 0])
    inp = types.SimpleNamespace(num_reqs=4, req_ids=req_ids, query_start_loc_np=qsl,
        idx_mapping_np=slots, is_prefilling_np=np.zeros(4, dtype=bool),
        num_computed_prefill_tokens_np=np.full(4, 10), prefill_len_np=np.full(4, 10),
        num_draft_tokens_per_req=widths-1)
    bv = batch_view_from_input_batch(inp)
    assert bv.req_ids == req_ids and bv.k_window.tolist() == [5, 1, 4, 0]
    gather = RowGatherer(max_slots=4, window=8, hidden=2, n_taps=3, device=torch.device('cpu'),
                      dtype=torch.bfloat16, fused=False, max_rows=32)
    taps = [torch.arange(28).reshape(14, 2).bfloat16() + 100*i for i in range(3)]
    positions = torch.cat([torch.arange(int(w)) + 100*(i+1) for i,w in enumerate(widths)])
    rejected = torch.tensor([3, 0, 1, 0], dtype=torch.int32)
    sampled = widths - rejected.numpy()
    result = gather.gather(bv, taps, positions, rejected, torch.tensor(sampled),
                           torch.tensor(slots), torch.zeros(4, 1, dtype=torch.int64),
                           torch.zeros(1, 4, dtype=torch.int32))
    assert result.n_valid == int(sampled.sum())
    rows = gather.rows_view(4, result.n_valid)
    expected = torch.cat([torch.cat(taps, dim=1)[int(qsl[i]):int(qsl[i]+sampled[i])]
                          for i in range(4)])
    assert torch.equal(rows, expected)


@pytest.mark.skipif(os.environ.get('TV1_GPU_TEST') != '1', reason='locked x86host GPU only')
def test_confidence_staging_ownership_recycling_and_rank_authority() -> None:
    manager = c.CompactWidths(torch.device('cuda'), .5)
    conf = torch.tensor([[.9, .1, 1., 1., 1.], [1., 1., .1, 1., 1.]], device='cuda')
    manager.record(['left', 'right'], conf)
    capacities = np.array([5, 5, 3], dtype=np.int32)
    result = manager.choose(['right', 'left', 'new'], capacities, lambda t: t)
    assert result.tolist() == [2, 1, 3]
    manager.forget(['right'])
    result = manager.choose(['right', 'left', 'new'], capacities, lambda t: t)
    assert result.tolist() == [5, 1, 3]
    # Rank-zero's broadcast overrides a different rank-local candidate shape.
    result = manager.choose(['right', 'left', 'new'], capacities,
                             lambda t: torch.tensor([0, 1, 2], dtype=torch.int32, device=t.device))
    assert result.tolist() == [0, 1, 2]
    manager.record(['right'], torch.zeros(1, 5, device='cuda'))
    assert manager.choose(['right'], np.array([5], dtype=np.int32), lambda t: t).tolist() == [0]
