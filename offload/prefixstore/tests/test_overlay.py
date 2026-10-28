"""CPU regression tests of actual overlay functions (no vLLM/CUDA import).

AST extraction isolates engine methods from CUDA-only module imports. It runs
those exact methods, rather than a reimplementation of their algorithms.
"""
from __future__ import annotations

import ast
import importlib.util
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
OVL = HERE / 'overlay'
OFF = 'vllm/distributed/kv_transfer/kv_connector/v1/offloading/'


def extract(file: str, name: str, env: dict | None = None) -> object:
    tree = ast.parse((OVL / file).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    node.decorator_list = []
    code = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    scope = {'__builtins__': __builtins__, **(env or {})}
    exec(compile(code, file, 'exec'), scope)
    return scope[name]


class OverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        spec = importlib.util.spec_from_file_location('rp1_fs_io', OVL / 'vllm/v1/kv_offload/tiering/fs/io.py')
        assert spec and spec.loader
        self.io = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.io)
        self.io._DSV41_CRC = True
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = [str(Path(self.tmp.name) / f'{i}.bin') for i in range(3)]
        self.data = bytearray(bytes(range(256)) * 48)
        self.offsets = [0, 4096, 8192]

    def store(self) -> None:
        self.io.batch_store_block(self.paths, memoryview(self.data), self.offsets, 4096, False)

    def test_python_roundtrip(self) -> None:
        self.store()
        dest = bytearray(len(self.data))
        self.io.batch_load_block(self.paths, memoryview(dest), self.offsets, 4096, False)
        self.assertEqual(dest, self.data)

    def test_corrupt_block_partial_success(self) -> None:
        self.store()
        p = Path(self.paths[1])
        bad = bytearray(p.read_bytes()); bad[8] ^= 1; p.write_bytes(bad)
        with self.assertRaises(OSError) as cm:
            self.io.batch_load_block(self.paths, memoryview(bytearray(len(self.data))), self.offsets, 4096, False)
        self.assertEqual(cm.exception.num_succeeded, 1)
        self.assertFalse(p.exists())
        self.assertFalse(Path(str(p) + '.crc').exists())
        self.assertTrue(Path(self.paths[2]).exists())

    def test_missing_or_invalid_sidecar(self) -> None:
        for missing in (True, False):
            self.store()
            crc = Path(self.paths[0] + '.crc')
            if missing: crc.unlink()
            else: crc.write_text('invalid')
            with self.assertRaises(OSError) as cm:
                self.io.batch_load_block(self.paths, memoryview(bytearray(len(self.data))), self.offsets, 4096, False)
            self.assertEqual(cm.exception.num_succeeded, 0)
            self.assertFalse(Path(self.paths[0]).exists())

    def test_crc_checks_successful_prefix_of_io_failure(self) -> None:
        self.store()
        Path(self.paths[0] + '.crc').unlink()
        Path(self.paths[2]).unlink()
        with self.assertRaises(OSError) as cm:
            self.io.batch_load_block(self.paths, memoryview(bytearray(len(self.data))), self.offsets, 4096, False)
        self.assertEqual(cm.exception.num_succeeded, 0)

    def test_native_wrapper_is_checked(self) -> None:
        self.store()
        def fake_load(paths, views, direct):
            for p, view in zip(paths, views): view[:] = Path(p).read_bytes()
        self.io._HAS_FSIO_C = True
        self.io.batch_load_block_C = fake_load
        dest = bytearray(len(self.data))
        self.io.batch_load_block(self.paths, memoryview(dest), self.offsets, 4096, False)
        self.assertEqual(dest, self.data)
        Path(self.paths[1] + '.crc').write_text('00000000')
        with self.assertRaises(OSError) as cm:
            self.io.batch_load_block(self.paths, memoryview(dest), self.offsets, 4096, False)
        self.assertEqual(cm.exception.num_succeeded, 1)

    def test_expandable_gate_is_connector_specific(self) -> None:
        fn = extract('vllm/config/vllm.py', '_verify_kv_transfer_compat', {'os': os})
        obj = SimpleNamespace(model_config=None)
        with patch.dict(os.environ, {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
                                     'VLLM_DSV41_OFFLOAD_EXPANDABLE_OK': '1'}):
            obj.kv_transfer_config = SimpleNamespace(kv_connector='OffloadingConnector')
            fn(obj)
            obj.kv_transfer_config = SimpleNamespace(kv_connector='NixlConnector')
            with self.assertRaises(ValueError): fn(obj)
        with patch.dict(os.environ, {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
                                     'VLLM_DSV41_OFFLOAD_EXPANDABLE_OK': '0'}):
            obj.kv_transfer_config = SimpleNamespace(kv_connector='OffloadingConnector')
            with self.assertRaises(ValueError): fn(obj)

    def test_cpu_parser(self) -> None:
        fn = extract('vllm/v1/kv_offload/tiering/fs/thread_pool.py', '_dsv41_parse_cpus')
        self.assertEqual(fn('0-4,10-13'), {0,1,2,3,4,10,11,12,13})
        with self.assertRaises(ValueError): fn('bad')

    def test_stale_rank1_sparse_and_contiguous_broadcast(self) -> None:
        import numpy as np
        import torch
        # Fake the collective only; tensors, indexing and _mn_sync are real CPU code.
        messages = []
        class Dist:
            source = True
            def broadcast(self, buf, src, group):
                if self.source: messages.append(buf.clone())
                else: buf.copy_(messages.pop(0))
        dist = Dist()
        torch_cpu = SimpleNamespace(from_numpy=torch.from_numpy, index_select=torch.index_select, distributed=dist)
        fn = extract(OFF + 'worker.py', '_mn_sync', {'np': np, 'torch': torch_cpu,
            'LoadStoreSpec': object, '_DSV41_OFFLOAD_MN_RUN_MIN': 16, 'time': __import__('time'),
            'logger': logging.getLogger('test')})
        rows0 = torch.arange(128 * 32, dtype=torch.int32).view(128, 32).to(torch.int8)
        rows1 = torch.full_like(rows0, -53)
        ids = [1, 3, 5, 7, 9, *range(32, 64), 88, 90, 92]
        def obj(rows, source):
            return SimpleNamespace(_mn_rows=rows, _mn_stage=torch.empty((3, 32), dtype=torch.int8),
                _mn_is_src=source, _mn_group=object(), _mn_stat=[0,0,0.0,__import__('time').monotonic()])
        fn(obj(rows0, True), SimpleNamespace(block_ids=np.array(ids)))
        dist.source = False
        fn(obj(rows1, False), SimpleNamespace(block_ids=np.array(ids)))
        self.assertEqual(messages, [])
        self.assertTrue(torch.equal(rows1[ids], rows0[ids]))
        self.assertTrue(torch.equal(rows1[0], torch.full((32,), -53, dtype=torch.int8)))

    def test_multinode_file_tier_refuses_buffered_fallback(self) -> None:
        tree = ast.parse((OVL / 'vllm/v1/kv_offload/tiering/fs/manager.py').read_text())
        guard = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                     and any(isinstance(x, ast.Raise) and 'buffered fallback' in ast.unparse(x)
                             for x in n.body))
        code = compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),'direct-guard','exec')
        with patch.dict(os.environ, {'VLLM_DSV41_OFFLOAD_MULTINODE':'1'}):
            with self.assertRaises(ValueError): exec(code, {'self':SimpleNamespace(_use_o_direct=False),'os':os})
            exec(code, {'self':SimpleNamespace(_use_o_direct=True),'os':os})
        with patch.dict(os.environ, {'VLLM_DSV41_OFFLOAD_MULTINODE':'0'}):
            exec(code, {'self':SimpleNamespace(_use_o_direct=False),'os':os})

    def config_fixture(self, enabled=True, heads=1, canonical=False, identity=True):
        class Attention:
            def __init__(self, block): self.num_kv_heads = heads; self.block_size = block
        groups = [SimpleNamespace(kv_cache_spec=Attention(block), layer_names=[f'layer{i}'])
                  for i, block in enumerate((128,192))]
        extra = {'canonical_layout': canonical}
        if identity: extra['prefix_store_identity'] = {'checkpoint_sha256': 'a'*64, 'layout_sha256': 'b'*64}
        parallel = SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1,
            prefill_context_parallel_size=1, decode_context_parallel_size=1, world_size=2,
            nnodes_within_dp=2, distributed_executor_backend='mp', rank=0,
            data_parallel_index=0, data_parallel_size=1, data_parallel_rank_local=0)
        conf = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra, engine_id='test'),
            parallel_config=parallel, use_v2_model_runner=False, kv_events_config=None,
            model_config=SimpleNamespace(use_mla=True, model='/mutable/model', dtype='fp8'),
            cache_config=SimpleNamespace(cache_dtype='fp8', kv_cache_layout=None))
        cache = SimpleNamespace(kv_cache_groups=groups, hisparse_host_num_blocks=None, num_blocks=100,
            kv_cache_tensors=[SimpleNamespace(size=13824000)], kv_tp_replicas=2)
        env = {'get_offloading_group_ids': lambda c: (0,1),
            'resolve_dcp_kv_block_size': lambda spec, dcp: spec.block_size,
            'resolve_kv_cache_block_sizes': lambda c,v: (64,64),
            'OffloadingGroupConfig': SimpleNamespace, 'OffloadingConfig': SimpleNamespace,
            'OffloadingModelConfig': SimpleNamespace, 'OffloadingCacheConfig': SimpleNamespace,
            'OffloadingParallelConfig': SimpleNamespace, 'iter_layer_specs': lambda spec: [spec],
            'AttentionSpec': Attention, '_DSV41_OFFLOAD_MULTINODE': enabled,
            'logger': logging.getLogger('test')}
        return extract(OFF+'config.py', 'build_offloading_config', env), conf, cache

    def test_multinode_gate_and_namespace(self) -> None:
        fn, conf, cache = self.config_fixture()
        got = fn(conf, cache)
        self.assertTrue(got.replicated_layout)
        self.assertIn('a'*64, got.model.name)
        self.assertIn('b'*64, got.model.name)
        for options in ({'enabled':False}, {'heads':2}, {'canonical':True}, {'identity':False}):
            fn, conf, cache = self.config_fixture(**options)
            with self.assertRaises(ValueError, msg=str(options)): fn(conf, cache)
        fn, conf, cache = self.config_fixture()
        conf.parallel_config.pipeline_parallel_size = 2
        with self.assertRaises(ValueError): fn(conf, cache)

    def test_cap_is_aligned_and_bounds_request_metadata(self) -> None:
        import math
        tree = ast.parse((OVL / OFF / 'scheduler.py').read_text())
        init = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '__init__'
                    and any(isinstance(x, ast.AnnAssign) and isinstance(x.target, ast.Attribute)
                            and x.target.attr == '_dsv41_load_cap' for x in n.body))
        start = next(i for i,n in enumerate(init.body) if isinstance(n, ast.AnnAssign)
                     and isinstance(n.target, ast.Attribute) and n.target.attr == '_dsv41_load_cap')
        end = next(i for i in range(start, len(init.body)) if isinstance(init.body[i], ast.Expr)
                   and isinstance(init.body[i].value, ast.Call)
                   and isinstance(init.body[i].value.func, ast.Attribute)
                   and init.body[i].value.func.attr == 'info')
        code = compile(ast.fix_missing_locations(ast.Module(body=init.body[start:end], type_ignores=[])), 'cap', 'exec')
        groups = [SimpleNamespace(tokens_per_chunk=n, sliding_window_size_in_chunks=None) for n in (128,192)]
        for fraction, expected in ((.2, 29568), (.45, 66432), (0, None)):
            obj = SimpleNamespace(config=SimpleNamespace(kv_group_configs=groups))
            env = {'self':obj, 'spec':SimpleNamespace(num_chunks=1927), 'math':math,
                   '_DSV41_LOAD_FRACTION':fraction}
            exec(code, env)
            self.assertEqual(obj._dsv41_load_cap, expected)
        for fraction in (float('nan'), float('inf'), -1, 1.1):
            with self.assertRaises(ValueError): exec(code, {**env, '_DSV41_LOAD_FRACTION':fraction})
        clamp = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'self._dsv41_load_cap is not None')
        code = compile(ast.fix_missing_locations(ast.Module(body=[clamp], type_ignores=[])), 'clamp', 'exec')
        for requested, expected in ((None,384), (1000000,384), (0,0), (128,128)):
            status = SimpleNamespace(max_load_tokens=requested)
            exec(code, {'self':SimpleNamespace(_dsv41_load_cap=384), 'req_status':status})
            self.assertEqual(status.max_load_tokens,expected)


if __name__ == '__main__':
    unittest.main()
