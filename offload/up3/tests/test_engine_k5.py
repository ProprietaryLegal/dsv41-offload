from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch

from offload.prefill_maxlen.evidence import record
from offload.up3 import engine_k5 as k
from offload.up3.engine_common import attr, invoke
from offload.up3.tests.test_engine_vor import cpu_builder


def worker() -> SimpleNamespace:
    spec = invoke(type('RemoteDSparkCapacitySpeculator', (), {}))
    remote = SimpleNamespace(run=lambda *args, **kw: SimpleNamespace(read_status=lambda: 0, ok_mask=torch.ones(4)))
    setattr(spec, 'remote', remote)
    setattr(spec, 'num_speculative_steps', 5)
    setattr(spec, 'draft_token_confidence_probs', torch.ones(4, 5))
    def propose(*args: object, **kw: object) -> torch.Tensor:
        batch = kw['input_batch']
        if kw.get('dummy_run') is not True:
            remote.run()
        return torch.zeros(cast(int, attr(batch, 'num_reqs')), 5, dtype=torch.int64)
    setattr(spec, 'propose', propose)
    return SimpleNamespace(vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=2),
        model_config=SimpleNamespace(max_model_len=327680), scheduler_config=SimpleNamespace(max_num_seqs=4),
        speculative_config=SimpleNamespace(num_speculative_tokens_per_batch_size=[[1, 4, 5]])),
        model_runner=SimpleNamespace(speculator=spec, model_state=SimpleNamespace(prepare_attn=lambda *a, **kw: {})))


def batch(n: int) -> SimpleNamespace:
    return SimpleNamespace(num_reqs=n, num_tokens=n*6, query_start_loc=torch.arange(n+1)*6,
                           is_padding=torch.zeros(n*6, dtype=torch.bool), has_prefill=False)


def test_k5_actual_observation_and_hook_cleanup() -> None:
    w = worker()
    o = k.Observer(w)
    original = attr(o.spec, 'propose')
    try:
        o.install()
        for n in (3, 4):
            invoke(attr(o.spec, 'propose'), input_batch=batch(n), dummy_run=False)
        invoke(attr(w.model_runner.model_state, 'prepare_attn'), batch(4))
        row = o.collect({'pass': True, 'samples': 1, 'byte_identical': True})
        assert record(row['five-proposals'])['proposals'] == 5
        assert record(row['b4-24-rows'])['rows'] == 24
    finally:
        o.close()
    assert attr(o.spec, 'propose') is original


def test_k5_dummy_zeros_and_dummy_24_rows_cannot_pass() -> None:
    w = worker()
    o = k.Observer(w)
    try:
        o.install()
        for n in (3, 4):
            invoke(attr(o.spec, 'propose'), input_batch=batch(n), dummy_run=True)
        b = batch(4)
        b.is_padding.fill_(True)
        invoke(attr(w.model_runner.model_state, 'prepare_attn'), b)
        with pytest.raises(ValueError, match='real B3/B4'):
            o.collect({})
        assert o.rows == 0 and o.proposals == {}
    finally:
        o.close()


def test_k5_failed_remote_work_refuses_and_hooks_restore() -> None:
    w = worker()
    w.model_runner.speculator.remote.run = lambda: SimpleNamespace(read_status=lambda: 1, ok_mask=torch.zeros(4))
    o = k.Observer(w)
    try:
        o.install()
        with pytest.raises(ValueError, match='remote fallback'):
            invoke(attr(o.spec, 'propose'), input_batch=batch(4))
    finally:
        o.close()


def test_installed_noncausal_body_vs_independent_future_visible_oracle() -> None:
    fixture = importlib.import_module('offload.upstream_adopt.tests.test_target_cpu')
    tl = invoke(attr(fixture, 'TL'))
    path = Path(k.__file__).parent/'overlay/vor/vllm/v1/attention/backends/mla/sparse_swa.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)
        and n.name == 'ComputeDSparkNoncausalSWAIndicesKernel')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel')
    fn.decorator_list = []
    scope: dict[str, object] = {'tl': tl}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), 'exec'), scope)
    kernel = invoke(attr(fixture, 'Kernel'), scope['kernel'], tl)
    def native(indices: torch.Tensor, lens: torch.Tensor, window: int, width: int, qsl: torch.Tensor,
               seq: torch.Tensor, mapping: torch.Tensor, valid: torch.Tensor, table: torch.Tensor,
               block: int, *, num_tokens: int, token_offset: int) -> None:
        # The interpreter must preserve the pitched table's underlying storage.
        for t in (indices, lens, qsl, seq, mapping, valid, table):
            flat = t.as_strided((t.untyped_storage().nbytes()//t.element_size(),), (1,), storage_offset=0)
            cast(dict[int, object], attr(tl, 'memory'))[t.data_ptr()] = invoke(attr(fixture, 'Pointer'), flat.numpy())
        def p(t: torch.Tensor) -> object:
            return cast(dict[int, object], attr(tl, 'memory'))[t.data_ptr()]
        launcher = invoke(attr(kernel, '__getitem__'), (num_tokens, 1))
        invoke(launcher, p(indices), indices.stride(0), p(lens), window, width, p(qsl), p(seq), p(mapping), p(valid),
            p(table), table.stride(0), block, token_offset, TRITON_BLOCK_SIZE=1024)
    builder = cpu_builder()
    builder.noncausal_index_width = 192
    result = k.noncausal([builder], SimpleNamespace(_COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL=native))
    assert record(result)['samples'] == 4
    assert record(result)['future_queries_visible'] is True
    def corrupt(*args: object, **kwargs: object) -> None:
        invoke(native, *args, **kwargs)
        cast(torch.Tensor, args[0])[1, 0, 0] = -17
    with pytest.raises(ValueError, match='bytes differ'):
        k.noncausal([builder], SimpleNamespace(_COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL=corrupt))
