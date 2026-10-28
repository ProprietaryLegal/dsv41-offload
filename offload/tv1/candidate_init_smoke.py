"""Local, isolated candidate import/binding smoke before any remote action.

Imports the composed helpers with real CPU torch and synthetic owning modules.
This is not a model load, CUDA initialization or GB10 numerical qualification.
The subprocess keeps fixture vLLM modules out of the coordinator interpreter.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import cast

from offload.prefill_maxlen.evidence import Json, record


def probe(tree: Path) -> dict[str, Json]:
    """Child only: real composed imports; no model/driver/kernel execution."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not sys.dont_write_bytecode:
        raise ValueError('TV1 candidate smoke requires hidden CUDA and disabled bytecode')
    # Deliberately no current config, just as in determine_available_memory.
    # If another candidate import tries that accessor, the smoke must refuse.
    def no_config() -> object:
        raise AssertionError('Current vLLM config is not set (TV1 candidate smoke)')
    config = ModuleType('vllm.config')
    config.__dict__['get_current_vllm_config'] = no_config
    config.__dict__['get_current_vllm_config_or_none'] = lambda: None
    vllm = ModuleType('vllm')
    vllm.__path__ = []
    sys.modules.update({'vllm': vllm, 'vllm.config': config})
    layers: dict[str, object] = {}
    lookups = 0
    def get_forward_context() -> object:
        nonlocal lookups
        lookups += 1
        return SimpleNamespace(no_compile_layers=layers, attn_metadata=None)
    forward = ModuleType('vllm.forward_context')
    forward.__dict__['get_forward_context'] = get_forward_context
    sys.modules[forward.__name__] = forward
    package = ModuleType('vllm_exl3_moe')
    package.__path__ = [str(tree / 'vllm_exl3_moe')]
    sys.modules[package.__name__] = package
    metadata = importlib.import_module('vllm_exl3_moe.tv1_candidate_metadata')
    # Require installed overlay imports, not the local-development fallback.
    imports: list[Json] = []
    for suffix in ('candidate', 'candidate_cases', 'candidate_metadata'):
        name = 'vllm_exl3_moe.tv1_' + suffix
        module = sys.modules[name]
        path = tree / 'vllm_exl3_moe' / ('tv1_' + suffix + '.py')
        if module.__file__ is None or Path(module.__file__).resolve() != path.resolve():
            raise ValueError(f'TV1 candidate smoke imported wrong helper: {name}')
        imports.append({'path': str(path.relative_to(tree)),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    consumer = cast(Callable[[object], int], metadata.__dict__['consumer_layer'])
    expected = (24, 28, 32, 36)
    shared_cache = SimpleNamespace(prefix='model.layers.20.attn.indexer.k_cache')
    layers[shared_cache.prefix] = shared_cache
    ops = [SimpleNamespace(k_cache=shared_cache) for _ in expected]
    for layer, op in zip(expected, ops, strict=True):
        layers[f'model.layers.{layer}.attn'] = SimpleNamespace(indexer=SimpleNamespace(indexer_op=op))
    bound: list[Json] = [consumer(op) for op in ops]
    if bound != list(expected) or lookups != 4:
        raise ValueError('TV1 candidate smoke owning-layer binding failed')
    # Keep the existing cache contract; later forwards need no repeated scan.
    layers.clear()
    if [consumer(op) for op in ops] != bound or lookups != 4:
        raise ValueError('TV1 candidate smoke per-op cache changed')
    import torch
    if cast(Callable[[], bool], torch.cuda.is_initialized)():
        raise ValueError('TV1 candidate smoke unexpectedly initialized CUDA')
    if 'vllm_exl3_moe.tv1_candidate_kernel' in sys.modules:
        raise ValueError('TV1 candidate smoke unexpectedly imported the CUDA kernel')
    return {'type': 'tv1-candidate-init-source-smoke-v1', 'classification': 'SOURCE/CPU ONLY',
            'imports': imports, 'bound_layers': bound, 'current_config_absent': True,
            'synthetic_forward_map': True, 'dummy_metadata_none': True,
            'cached_lookups': True, 'cuda_initialized': False,
            'cuda_or_engine_init_qualified': False}


def check(tree: Path) -> dict[str, Json]:
    """Bounded Controller child; runs before Supervisor or any transport/HOLD."""
    result = subprocess.run([sys.executable, '-m', 'offload.tv1.candidate_init_smoke',
                             '--probe', '--tree', str(tree.resolve())],
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'},
                            text=True, capture_output=True, timeout=30)
    if result.returncode:
        raise ValueError(f'TV1 candidate import/binding smoke failed rc={result.returncode}:\n'
                         f'{result.stdout}\n{result.stderr}')
    if result.stderr:
        print(result.stderr, file=sys.stderr, end='')
    receipt = record(cast(Json, json.loads(result.stdout)))
    if (receipt.get('type') != 'tv1-candidate-init-source-smoke-v1'
            or receipt.get('classification') != 'SOURCE/CPU ONLY'
            or receipt.get('bound_layers') != [24, 28, 32, 36]
            or receipt.get('cuda_or_engine_init_qualified') is not False
            or receipt.get('cuda_initialized') is not False):
        raise ValueError('TV1 candidate smoke receipt invalid')
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tree', type=Path, required=True)
    parser.add_argument('--probe', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    print(json.dumps(probe(args.tree) if args.probe else check(args.tree)))


if __name__ == '__main__':
    main()
