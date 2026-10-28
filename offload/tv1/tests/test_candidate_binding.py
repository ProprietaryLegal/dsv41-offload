"""CPU execution of the installed forward boundary, with no load-time config.

Only CUDA/model operations are stand-ins. ForwardContext construction, its
context managers, the failing config accessor and forward_cuda come from the
real pinned source, rather than a fixture that always provides current config.
"""
from __future__ import annotations

import ast
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

from offload.tv1 import build_overlay as b
from offload.tv1.candidate_cases import LAYERS


class Mode(IntEnum):
    NONE = 0
    FULL = 1
    PIECEWISE = 2

    def is_valid_runtime_mode(self) -> bool:
        return True


def source_nodes(path: Path, names: set[str]) -> str:
    source = ast.parse(path.read_text())
    nodes: list[ast.stmt] = [n for n in source.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef))} == names
    return 'from __future__ import annotations\n' + ast.unparse(ast.Module(body=nodes, type_ignores=[]))


@pytest.fixture
def forward_modules(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    config = ModuleType('vllm.config')
    config.__dict__['_current_vllm_config'] = None
    exec(source_nodes(b.IMAGE / 'vllm/config/vllm.py', {'get_current_vllm_config'}), config.__dict__)
    monkeypatch.setitem(sys.modules, config.__name__, config)
    forward = ModuleType('vllm.forward_context')
    forward.__dict__.update(
        dataclass=dataclass, field=field, contextmanager=contextmanager,
        CUDAGraphMode=Mode, track_batchsize=False, _forward_context=None,
        current_platform=SimpleNamespace(set_additional_forward_context=lambda **kwargs: {}))
    monkeypatch.setitem(sys.modules, forward.__name__, forward)
    exec(source_nodes(b.IMAGE / 'vllm/forward_context.py', {
        'BatchDescriptor', 'ForwardContext', 'get_forward_context', 'create_forward_context',
        'override_forward_context', 'set_forward_context'}), forward.__dict__)
    return config, forward


def installed_metadata(tree: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    package = ModuleType('vllm_exl3_moe')
    package.__path__ = [str(tree / 'vllm_exl3_moe')]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    loaded: ModuleType | None = None
    for suffix in ('candidate', 'candidate_cases', 'candidate_metadata'):
        name = 'vllm_exl3_moe.tv1_' + suffix
        spec = importlib.util.spec_from_file_location(name, tree / 'vllm_exl3_moe' / ('tv1_' + suffix + '.py'))
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, loaded)
        spec.loader.exec_module(loaded)
    assert loaded is not None
    return loaded


def patched_forward(tree: Path) -> Callable[..., int]:
    source = ast.parse((tree / 'vllm/model_executor/layers/sparse_attn_indexer.py').read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'SparseAttnIndexer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward_cuda')
    def custom(*args: object, **kwargs: object) -> int:
        return cast(int, kwargs['tv1_layer'])
    namespace: dict[str, object] = {
        'torch': SimpleNamespace(ops=SimpleNamespace(vllm=SimpleNamespace(sparse_attn_indexer=custom))),
        '_encode_layer_name': lambda name: name}
    exec('from __future__ import annotations\n' + ast.unparse(method), namespace)
    return cast(Callable[..., int], namespace['forward_cuda'])


def op_state() -> SimpleNamespace:
    return SimpleNamespace(
        k_cache=SimpleNamespace(prefix='model.layers.20.attn.indexer.k_cache', kv_cache=object()),
        quant_block_size=128, scale_fmt='ue8m0', topk_tokens=512, head_dim=128,
        max_model_len=327680, max_total_seq_len=327680, topk_indices_buffer=object(),
        skip_k_cache_insert=True, use_pcp=False, pcp_shard_decode_requests=False,
        dense_mha_metadata_layer_name='', use_fp4_cache=False, dcp_rank=0,
        dcp_world_size=1, cp_kv_cache_interleave_size=1, candidate_blocks=object(),
        candidate_block_size=8, candidate_write=False, topk_backend='persistent')


def current_config_fails(config: ModuleType) -> None:
    with pytest.raises(AssertionError, match='Current vLLM config is not set'):
        cast(Callable[[], object], config.__dict__['get_current_vllm_config'])()


def forward_scope(forward: ModuleType, layers: dict[str, object], metadata: object,
                  mode: Mode) -> AbstractContextManager[object]:
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=layers, fast_moe_cold_start=False),
        parallel_config=SimpleNamespace(data_parallel_size=1, use_sequence_parallel_moe=False))
    return cast(Callable[..., AbstractContextManager[object]], forward.__dict__['set_forward_context'])(
        metadata, config, num_tokens=24, cudagraph_runtime_mode=mode)


@pytest.mark.parametrize('phase,mode,metadata', [
    ('profile-dummy', Mode.NONE, None), ('full-capture', Mode.FULL, {}),
    ('piecewise-capture', Mode.PIECEWISE, {}), ('prefill', Mode.NONE, {}),
    ('decode', Mode.NONE, {})])
@pytest.mark.parametrize('layer', LAYERS)
def test_real_forward_binding_without_current_config(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        forward_modules: tuple[ModuleType, ModuleType], phase: str, mode: Mode,
        metadata: object, layer: int) -> None:
    tree = tmp_path / 'overlay'
    b.build(tree, ('candidate',))
    module = installed_metadata(tree, monkeypatch)
    consumer = cast(Callable[[object], int], module.__dict__['consumer_layer'])
    call = patched_forward(tree)
    config, forward = forward_modules
    monkeypatch.setenv('VLLM_TV1_CANDIDATE', '1')
    op = op_state()
    owner = SimpleNamespace(indexer=SimpleNamespace(indexer_op=op))
    layers = {f'model.layers.{layer}.attn': owner, op.k_cache.prefix: op.k_cache}
    current_config_fails(config)
    with forward_scope(forward, layers, metadata, mode):
        current_config_fails(config)
        assert call(op, object(), object(), None, object()) == layer, phase
        assert op._tv1_layer == layer
    current_config_fails(config)
    # The existing per-op cache also avoids accessing any context on repeats.
    assert consumer(op) == layer
    with pytest.raises(AssertionError, match='Forward context is not set'):
        consumer(op_state())


@pytest.mark.parametrize('case', ['missing', 'duplicate', 'bad-name', 'unqualified', 'shared-only'])
def test_owning_layer_refusals_without_current_config(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        forward_modules: tuple[ModuleType, ModuleType], case: str) -> None:
    tree = tmp_path / 'overlay'
    b.build(tree, ('candidate',))
    consumer = cast(Callable[[object], int], installed_metadata(tree, monkeypatch).__dict__['consumer_layer'])
    op = op_state()
    owner = SimpleNamespace(indexer=SimpleNamespace(indexer_op=op))
    layers: dict[str, object] = {op.k_cache.prefix: op.k_cache}
    if case == 'duplicate':
        layers.update({'model.layers.24.attn': owner, 'model.layers.28.attn': owner})
    elif case == 'bad-name':
        layers['not-a-layer'] = owner
    elif case == 'unqualified':
        layers['model.layers.20.attn'] = owner
    elif case == 'missing':
        layers['model.layers.24.attn'] = SimpleNamespace(indexer=SimpleNamespace(indexer_op=object()))
    with forward_scope(forward_modules[1], layers, None, Mode.NONE):
        with pytest.raises(ValueError, match='uniquely|outside qualified'):
            consumer(op)
    assert not hasattr(op, '_tv1_layer')


def test_candidate_forward_context_accessors_are_explicit(tmp_path: Path) -> None:
    """Exhaustive accessor inventory: inherited helpers and our additions.

    A new accessor is a review event. Do not add an allowlist entry merely to
    pass this test; prove its lifetime in CANDIDATE-INIT.md first.
    """
    tree = tmp_path / 'overlay'
    manifest = b.build(tree, ('candidate',))
    calls: list[tuple[str, str, str]] = []
    for entry in manifest['files']:
        source = ast.parse((tree / entry['path']).read_text())
        def walk(node: ast.AST, owner: str) -> Iterator[tuple[str, str, str]]:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                owner = (owner + '.' + node.name).lstrip('.')
            if isinstance(node, ast.Call):
                name = (node.func.id if isinstance(node.func, ast.Name) else
                        node.func.attr if isinstance(node.func, ast.Attribute) else '')
                if name.startswith(('get_current_', 'get_forward_', 'get_tp_', 'get_dcp_', 'get_pcp_', 'get_pp_')):
                    yield entry['path'], owner, name
            for child in ast.iter_child_nodes(node):
                yield from walk(child, owner)
        calls.extend(walk(source, ''))
    # The complete composed source includes baseline engine accessors too.
    baseline: list[tuple[str, str, str]] = []
    for path in b.BASES:
        original = tmp_path / 'baseline' / path
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(b.BASES[path].read_bytes())
    # Compare candidate-only changes against a second walk over baseline files.
    for entry in manifest['files']:
        if entry['path'] not in b.BASES:
            continue
        source = ast.parse((tmp_path / 'baseline' / entry['path']).read_text())
        baseline.extend(walk(source, ''))
    added = list(calls)
    for site in baseline:
        added.remove(site)
    assert added == [('vllm_exl3_moe/tv1_candidate_metadata.py', 'consumer_layer', 'get_forward_context')]
    assert not any(name.startswith('get_current_') and path.startswith('vllm_exl3_moe/')
                   for path, owner, name in calls)
