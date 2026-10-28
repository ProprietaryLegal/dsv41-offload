"""R-SP row layout, threshold and exactness floor (CPU only; the image's dispatch rules are lifted from its source)."""
from __future__ import annotations

import ast
import functools
import os
import types
from pathlib import Path
from typing import Any

import pytest
import torch

from offload.sp_prefill.tests import toy
from offload.sp_prefill.tests.fake_tp import FakeTP

SP = toy.load_helper()
IMAGE = toy.IMAGE / 'vllm'


@pytest.mark.parametrize('world', [2, 3, 4])
def test_layout_covers_every_row_once(world: int) -> None:
    for T in range(1, 40):
        layouts = [SP.SPRows(T, world, r) for r in range(world)]
        tp = layouts[0].padded_rows
        assert tp % world == 0 and T <= tp < T + world
        assert all(r.local_rows * world == tp for r in layouts)
        assert [r.start for r in layouts] == [i * layouts[0].local_rows for i in range(world)]
        assert sum(r.real_rows for r in layouts) == T
        x = torch.arange(T * 3, dtype=torch.float32).view(T, 3)
        parts = [r.local_slice(x) for r in layouts]
        assert torch.equal(torch.cat(parts)[:T], x)
        assert not torch.cat(parts)[T:].any(), 'pad rows are zero'


def test_layout_refuses_bad_shapes() -> None:
    for args in ((0, 2, 0), (5, 1, 0), (5, 2, 2), (5, 2, -1)):
        with pytest.raises(ValueError):
            SP.SPRows(*args)
    rows = SP.SPRows(7, 2, 1)
    with pytest.raises(ValueError, match='slice expects 7'):
        rows.local_slice(torch.zeros(8, 2))
    with pytest.raises(ValueError, match='gather expects 4'):
        rows.gather(torch.zeros(7, 2))
    with pytest.raises(ValueError, match='reduce-scatter expects 7'):
        rows.reduce_scatter(torch.zeros(8, 2))
    with pytest.raises(ValueError, match='select expects 4'):
        rows.select(torch.zeros(7, 2), torch.tensor([0]))


def test_plan_is_prompt_only() -> None:
    assert SP.plan(5000, 0, 2, 0, False) is None  # switch off
    assert SP.plan(5000, 3073, 2, 0, True) is None  # a stream is capturing
    assert SP.plan(3072, 3073, 2, 1, False) is None
    assert SP.plan(5000, 3073, 1, 0, False) is None
    assert SP.plan(3073, 3073, 2, 1, False) == SP.SPRows(3073, 2, 1)


def test_collectives_round_trip_and_select_is_exact() -> None:
    tp = FakeTP(2)
    SP.use_comm(tp)
    try:
        T = 23
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(T, 4, 5, generator=gen).to(torch.bfloat16)
        partial = [torch.randn(T, 5, generator=gen).to(torch.bfloat16) for _ in range(2)]
        index = torch.tensor([0, 11, 12, 22, 5, 17, 22], dtype=torch.int32)

        def rank(r: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            rows = SP.SPRows(T, 2, r)
            local = rows.local_slice(x)
            return rows.gather(local), rows.reduce_scatter(partial[r]), rows.select(local, index)

        out = tp.run(rank)
        empty = SP.SPRows(T, 2, 0).select(x[:12], torch.tensor([], dtype=torch.int32))
        assert empty.shape == (0, 4, 5)
        reduced = (partial[0].float() + partial[1].float()).to(torch.bfloat16)
        for r, (gathered, scattered, selected) in enumerate(out):
            assert torch.equal(gathered, x)
            rows = SP.SPRows(T, 2, r)
            assert torch.equal(scattered[: rows.real_rows], reduced[rows.start : rows.start + rows.real_rows])
            assert not scattered[rows.real_rows :].any()
            assert torch.equal(selected, x[index])
    finally:
        SP.use_comm(None)


def test_rows_above_bytes_matches_0018() -> None:
    assert SP.rows_above_bytes(2 * 1024**2, 5120 * 2, 3) == 205  # the donor's TP3 2 MiB example
    assert SP.rows_above_bytes(256 * 1024, 5120 * 2, 2) == 25  # our EXL3 RDMA default at TP2
    for limit, row, world in ((256 * 1024, 10240, 2), (2 * 1024**2, 10240, 3), (64 * 1024, 10240, 4)):
        t = SP.rows_above_bytes(limit, row, world)
        assert SP.SPRows(t, world, 0).padded_rows * row > limit
        assert SP.SPRows(t - 1, world, 0).padded_rows * row <= limit
    assert SP.rows_above_bytes(0, 10240, 2) == 1


def _image_split_rules(n_sms: int) -> tuple[Any, Any]:
    """compute_mhc_pre_num_splits and mhc_fused_post_pre_split_config, lifted from the image source."""
    props = types.SimpleNamespace(multi_processor_count=n_sms)
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_properties=lambda _: props))
    scope: dict[str, Any] = {'torch': fake_torch, 'cache': functools.cache,
                             'cdiv': lambda a, b: -(-a // b), '_FUSED_POST_PRE_N_THR': 128}
    kernels = (IMAGE / 'model_executor/kernels/mhc/tilelang_kernels.py').read_text()
    tree = ast.parse(kernels)
    consts = [n for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id.startswith('_FUSED_POST_PRE') for t in n.targets)]
    toy.lift(kernels, {'compute_num_split', 'mhc_fused_post_pre_split_config'}, scope, 'tilelang_kernels.py')
    for node in consts:
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'consts', 'exec'), scope)
    toy.lift((IMAGE / 'model_executor/kernels/mhc/warmup.py').read_text(), {'compute_mhc_pre_num_splits'}, scope,
             'warmup.py')
    return scope['compute_mhc_pre_num_splits'], scope['mhc_fused_post_pre_split_config']


def test_image_constants_are_the_helpers() -> None:
    kernels = (IMAGE / 'model_executor/kernels/mhc/tilelang_kernels.py').read_text()
    assert f'_FUSED_POST_PRE_MAX_TOKENS = {SP.FUSED_POST_PRE_MAX_TOKENS}\n' in kernels
    warmup = (IMAGE / 'model_executor/kernels/mhc/warmup.py').read_text()
    assert f'splits = compute_num_split({SP.MHC_PRE_BLOCK_ROWS}, input_size, cdiv(num_tokens, {SP.MHC_PRE_BLOCK_ROWS}))' in warmup
    csrc = Path('/opt/dsv41/data/donors/vllm/csrc/layernorm_kernels.cu')
    if csrc.exists():  # upstream vLLM source (the image ships only the built kernel)
        assert f'((num_tokens < {SP.RMS_NORM_BLOCK_ROWS}) ? 1024 : 256)' in csrc.read_text()


@pytest.mark.parametrize('n_sms', [48, 82, 132, 148])
def test_exact_floor_keeps_every_row_wise_variant(n_sms: int) -> None:
    """For every L >= the floor and any T with ceil(T / 2) = L: same mHC pre split count (both GEMM widths), no fused
    post-pre, same RMSNorm block. Below the floor some (L, T) pair differs, so the floor is not loose by a bucket."""
    splits, fused = _image_split_rules(n_sms)
    floor = SP.exact_local_floor(n_sms, True)
    for L in range(floor, floor + 600, 7):
        for T in (2 * L - 1, 2 * L, 4 * L):
            for width in (5120, 5120 * 4):
                assert splits(width, L) == splits(width, T)
            assert fused(L, 5120, 4) is None and fused(T, 5120, 4) is None
            assert (L < SP.RMS_NORM_BLOCK_ROWS) == (T < SP.RMS_NORM_BLOCK_ROWS)
    below = floor - 1
    assert any(splits(w, below) != splits(w, 2 * below) for w in (5120, 20480)) or below < SP.RMS_NORM_BLOCK_ROWS
    assert SP.exact_local_floor(n_sms, False) == SP.RMS_NORM_BLOCK_ROWS


def test_gb10_threshold() -> None:
    """GB10 (48 SMs), TP2: SP from 3,073 tokens, so every full 4,096-16,384 chunk of a 32K / 150K prompt splits."""
    assert SP.exact_local_floor(48, True) == 1537
    cfg = fake_config()
    assert SP.resolve_min_rows(cfg, **OFF_FLAGS, gpu_facts=lambda: (48, True)) == 0  # switch off by default
    rows = with_enabled(lambda: SP.resolve_min_rows(cfg, **OFF_FLAGS, gpu_facts=lambda: (48, True)))
    assert rows == 3073
    assert SP.plan(4096, rows, 2, 0, False) is not None and SP.plan(3072, rows, 2, 0, False) is None
    assert SP.memory_saved_bytes(16384, 2, 5120, 4) == 320 * 1024**2
    assert SP.memory_saved_bytes(4096, 2, 5120, 4) == 80 * 1024**2


OFF_FLAGS = dict(static_sequence_parallel=False, fuse_mhc_all_reduce=False, native_mega_moe=False, gemm_rs=False)


def fake_config(**parallel: Any) -> Any:
    par = dict(tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1,
               prefill_context_parallel_size=1, use_ubatching=False)
    par.update(parallel)
    return types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(**par),
        speculative_config=types.SimpleNamespace(num_speculative_tokens=5),
        compilation_config=types.SimpleNamespace(max_cudagraph_capture_size=64),
        scheduler_config=types.SimpleNamespace(max_num_seqs=4),
        model_config=types.SimpleNamespace(get_hidden_size=lambda: 5120, dtype=torch.bfloat16))


def with_enabled(fn: Any) -> Any:
    saved = SP.ENABLED
    SP.ENABLED = True
    try:
        return fn()
    finally:
        SP.ENABLED = saved


@pytest.mark.parametrize('change', [
    dict(tensor_parallel_size=1), dict(pipeline_parallel_size=2), dict(data_parallel_size=2),
    dict(prefill_context_parallel_size=2), dict(use_ubatching=True)])
def test_off_reasons_parallel(change: dict[str, Any]) -> None:
    cfg = fake_config(**change)
    assert with_enabled(lambda: SP.resolve_min_rows(cfg, **OFF_FLAGS, gpu_facts=lambda: (48, True))) == 0


@pytest.mark.parametrize('flag', sorted(OFF_FLAGS))
def test_off_reasons_model(flag: str) -> None:
    flags = {**OFF_FLAGS, flag: True}
    assert with_enabled(lambda: SP.resolve_min_rows(fake_config(), **flags, gpu_facts=lambda: (48, True))) == 0


def test_threshold_terms_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = fake_config()

    def resolve(facts: tuple[int, bool] = (48, False)) -> int:
        rows: int = with_enabled(lambda: SP.resolve_min_rows(cfg, **OFF_FLAGS, gpu_facts=lambda: facts))
        return rows

    assert resolve() == 2 * 255 + 1  # RMSNorm floor; decode (64 + 1) and RDMA (off) are lower
    cfg.compilation_config.max_cudagraph_capture_size = 4096
    assert resolve() == 4097  # one more than the largest decode forward
    cfg.compilation_config.max_cudagraph_capture_size = 64
    monkeypatch.setenv('EXL3_RDMA_AR', '1')
    monkeypatch.setenv('EXL3_RDMA_AR_MAX_KB', '8192')
    assert resolve() == SP.rows_above_bytes(8192 * 1024, 10240, 2) == 819
    monkeypatch.delenv('EXL3_RDMA_AR_MAX_KB')
    monkeypatch.setenv('VLLM_DSV41_PREFILL_SP_MIN_ROWS', '9000')
    assert resolve((48, True)) == 9000
    monkeypatch.setenv('VLLM_DSV41_PREFILL_SP_MIN_ROWS', '100')
    assert resolve((48, True)) == 3073, 'the override only raises'
    monkeypatch.setenv('VLLM_DSV41_PREFILL_SP_MIN_ROWS', '-5')
    with pytest.raises(ValueError, match='row count'):
        resolve()


def test_switch_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    assert os.environ.get('VLLM_DSV41_PREFILL_SP') in (None, '0') and SP.ENABLED is False
    monkeypatch.setenv('VLLM_DSV41_PREFILL_SP', 'yes')
    with pytest.raises(ValueError, match='must be 0 or 1'):
        SP._flag('VLLM_DSV41_PREFILL_SP')
    monkeypatch.setenv('VLLM_DSV41_PREFILL_SP', '1')
    assert SP._flag('VLLM_DSV41_PREFILL_SP') is True
