"""Execute checked allocator arithmetic without importing vLLM, torch or CUDA."""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, cast

import pytest

from offload.pc1 import build, capacity

OVERLAY = build.ROOT / 'offload/kvpack/overlay/vllm'
IMAGE = Path('/opt/dsv41/data/src-image/vllm')


def load(path: Path, owner: str | None, names: tuple[str, ...], scope: dict[str, object]) -> None:
    tree = ast.parse(path.read_text())
    body = tree.body if owner is None else next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    functions = [copy.deepcopy(n) for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(functions) == len(names)
    for fn in functions:
        fn.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                             *functions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)


@pytest.mark.parametrize('arm', build.ARMS)
def test_exact_checked_allocator_computes_registered_capacity(arm: str, monkeypatch: pytest.MonkeyPatch) -> None:
    lock = build.verify_inputs()
    for relative in ('v1/kv_cache_interface.py', 'v1/core/kv_cache_utils.py'):
        path = OVERLAY / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == lock['inputs']['git:offload/kvpack/overlay/vllm/' + relative]
    scope: dict[str, object] = {'cdiv': lambda x, y: -(-x // y)}
    load(IMAGE / 'config/vllm.py', 'VllmConfig', ('max_concurrent_batches', 'max_in_flight_tokens'), scope)
    cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=327680),
        scheduler_config=SimpleNamespace(async_scheduling=True, max_num_batched_tokens=build.SETTINGS[arm][0]),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, decode_context_parallel_size=1),
        use_v2_model_runner=True)
    cfg.max_concurrent_batches = cast(Callable[..., int], scope['max_concurrent_batches'])(cfg)
    cfg.max_in_flight_tokens = cast(Callable[..., int], scope['max_in_flight_tokens'])(cfg)
    load(OVERLAY / 'v1/kv_cache_interface.py', 'SlidingWindowSpec',
         ('max_admission_blocks_per_request', 'max_memory_usage_bytes'), scope)
    swa = SimpleNamespace(sliding_window=128, extra_retained_tokens=0, block_size=64,
                          page_size_bytes=37440, dcp_sharded=False)
    admission = cast(Callable[..., int], scope['max_admission_blocks_per_request'])
    swa.max_admission_blocks_per_request = lambda **kwargs: admission(swa, **kwargs)
    swa_memory = cast(Callable[..., int], scope['max_memory_usage_bytes'])
    swa.max_memory_usage_bytes = lambda config: swa_memory(swa, config)
    load(OVERLAY / 'v1/kv_cache_interface.py', 'FullAttentionSpec', ('max_memory_usage_bytes',), scope)
    full_memory = cast(Callable[..., int], scope['max_memory_usage_bytes'])
    full_specs = [SimpleNamespace(block_size=size, page_size_bytes=page, dcp_sharded=False)
                  for size, page in ((128, 46080), (192, 138240))]
    for spec in full_specs:
        spec.max_memory_usage_bytes = lambda config, spec=spec: full_memory(spec, config)
    load(OVERLAY / 'v1/kv_cache_interface.py', 'CircularBufferSpec', ('max_memory_usage_bytes',), scope)
    ring = SimpleNamespace(page_size_bytes=32768)
    ring_memory = cast(Callable[..., int], scope['max_memory_usage_bytes'])
    ring.max_memory_usage_bytes = lambda config: ring_memory(ring, config)
    groups = [SimpleNamespace(host_resident=False, kv_cache_spec=spec)
              for spec in [*full_specs, ring, *([swa] * 14)]]
    load(OVERLAY / 'v1/core/kv_cache_utils.py', None,
         ('get_max_concurrency_for_kv_cache_config', 'get_kv_cache_capacity'), scope)
    allocator = cast(Callable[..., tuple[int, float]], scope['get_kv_cache_capacity'])
    result = capacity.expectation(arm)
    tokens, concurrency = allocator(cfg, SimpleNamespace(num_blocks=6_442_450_944 // 138240, kv_cache_groups=groups))
    assert tokens == result['expected_tokens'] == {
        'control': 2502600, 'chunk2048': 2933321, 'threshold2048': 2502600,
        'threshold512': 2502600, 'chunk8192': 1934490, 'chunk16384': 1330447}[arm]
    assert int(concurrency * 327680) == tokens
    assert result['in_flight_tokens'] == cfg.max_in_flight_tokens
    assert result['swa_blocks_per_group'] == admission(swa, cfg.max_in_flight_tokens, 327680)
    # Production factor4 is unchanged and depends on max_model_len, not chunk.
    monkeypatch.setenv('VLLM_INDEXER_PREFILL_BUFFER_FACTOR', '4')
    load(IMAGE / 'v1/attention/backends/mla/indexer.py', None, ('get_max_prefill_buffer_size',), scope)
    assert cast(Callable[..., int], scope['get_max_prefill_buffer_size'])(cfg) == 4 * 327680


def test_unregistered_capacity_refuses() -> None:
    with pytest.raises(ValueError, match='unregistered arm'):
        capacity.expectation('chunk32768')
