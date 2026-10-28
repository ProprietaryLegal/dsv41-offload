from __future__ import annotations

import importlib
import ast
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Iterator, cast
from unittest.mock import patch

import pytest
import torch

from offload.prefill_maxlen.evidence import record
from offload.up3 import engine_vor as v
from offload.up3.engine_common import attr, invoke


def cpu_builder(block: int = 64) -> SimpleNamespace:
    return SimpleNamespace(decode_swa_indices=torch.full((40, 1, 128), -123, dtype=torch.int32),
        decode_swa_lens=torch.zeros(40, dtype=torch.int32), is_valid_token=torch.zeros(40, dtype=torch.bool),
        no_replay_start=torch.zeros(5, dtype=torch.int32), token_to_req_indices=torch.zeros(40, dtype=torch.int32),
        block_size=block, window_size=128)


def stock_cpu(indices: torch.Tensor, lens: torch.Tensor, window: int, index_width: int,
              left: torch.Tensor, right: torch.Tensor, qsl: torch.Tensor, seq: torch.Tensor,
              mapping: torch.Tensor, valid: torch.Tensor, table: torch.Tensor, block: int,
              replay: torch.Tensor, *, num_tokens: int, token_offset: int) -> None:
    for t in range(num_tokens):
        if not valid[t]:
            lens[t] = 0
            indices[t] = -1
            continue
        req = int(mapping[t])
        pos = int(seq[req]-(qsl[req+1]-qsl[req])+t-qsl[req])
        start = max(pos-window+1, 0, int(replay[req]))
        lens[t] = pos+1-start
        row = []
        for j in range(index_width):
            p = start+j
            row.append(int(table[req, p//block])*block+p%block if p <= pos else -1)
        indices[t] = torch.tensor(row).reshape_as(indices[t])


def test_vor_real_source_plan_against_independent_cpu_stock() -> None:
    # The existing upstream CPU interpreter executes the exact transplanted Triton body.
    fixture = importlib.import_module('offload.upstream_adopt.tests.test_target_cpu')
    scope, tl = cast(tuple[dict[str, object], object], invoke(attr(fixture, 'swa_scope')))
    cls = cast(type, scope['SWADecodePlan'])
    captured = [False]
    class FakeDevicePlan(cls):  # type: ignore[misc, valid-type]
        def _launch(self, *args: object) -> bool:
            if captured[0]:
                return False
            return cast(bool, invoke(attr(super(), '_launch'), *args))

        def arm(self, entries: list[tuple[object, torch.Tensor, torch.Tensor]]) -> None:
            for builder, table, slots in entries:
                for t in (table, slots, *(v.tensor(builder, name) for name in v.BUFFERS[:3])):
                    # The production table is pitched. Preserve its underlying storage
                    # and row stride in the CPU pointer interpreter too.
                    flat = t.as_strided((t.untyped_storage().nbytes()//t.element_size(),), (1,), storage_offset=0)
                    cast(dict[int, object], attr(tl, 'memory'))[t.data_ptr()] = invoke(
                        attr(fixture, 'Pointer'), flat[t.storage_offset():].numpy())
            invoke(attr(super(), 'arm'), entries)
    @contextmanager
    def graph(*args: object) -> Iterator[None]:
        captured[0] = True
        try:
            yield
        finally:
            captured[0] = False
    fused = SimpleNamespace(SWADecodePlan=FakeDevicePlan, ReplayStartMirror=scope['ReplayStartMirror'])
    sparse = SimpleNamespace(_COMPUTE_SWA_INDICES_AND_LENS_KERNEL=stock_cpu)
    builders = [cpu_builder(64), cpu_builder(192)]
    before = builders[0].decode_swa_indices.clone()
    with patch.object(v, 'forbid_compile', lambda: nullcontext()), patch.object(torch.cuda, 'graph', graph), \
         patch.object(torch.cuda, 'CUDAGraph', lambda: SimpleNamespace(replay=lambda: None)):
        result = v.check_groups(cast(list[object], builders), fused, sparse)
    assert record(result['actual-groups'])['count'] == 2
    assert record(result['launch'])['once_per_eligible_step'] is True
    assert record(result['ctx327680'])['samples'] == 96
    assert torch.equal(builders[0].decode_swa_indices, before)


def test_vor_empty_groups_and_bad_capacity_refuse() -> None:
    with pytest.raises(ValueError, match='no groups'):
        v.check_groups([], object(), object())
    with pytest.raises(ValueError, match='padding'):
        v.metadata_inputs(cpu_builder(), 6, 6, 327680, True)


def test_compile_guard_refuses_and_restores_all_entries() -> None:
    import sys
    modules = {name: SimpleNamespace(compile=lambda: 'original')
        for name in ('triton', 'triton.compiler', 'triton.compiler.compiler')}
    with patch.dict(sys.modules, modules):
        with v.forbid_compile():
            for module in modules.values():
                with pytest.raises(ValueError, match='request-time'):
                    module.compile()
        assert all(module.compile() == 'original' for module in modules.values())


def test_installed_state_capture_and_microbatch_routing_restored() -> None:
    from pathlib import Path
    path = Path(v.__file__).parent/'overlay/vor/vllm/models/deepseek_v41/nvidia/model_state.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DeepseekV41ModelState')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_attn')
    class Base:
        def prepare_attn(self, *args: object, **kwargs: object) -> object:
            raise AssertionError('delegate must be replaced only during routing check')
    scope: dict[str, object] = {'Base': Base}
    # Keep zero-argument super() and the installed method body intact.
    new_cls = ast.ClassDef(name='State', bases=[ast.Name(id='Base', ctx=ast.Load())],
        keywords=[], body=[method], decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), new_cls],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    state = invoke(scope['State'])
    setattr(state, '_swa_plan', SimpleNamespace(armed=False))
    assert record(v.state_exclusions(state))['samples'] == 2
    with pytest.raises(AssertionError, match='replaced only'):
        Base().prepare_attn()
