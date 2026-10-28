"""CPU gates on the actual patch applied to the pinned v5 reference, never the serve."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import types

os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(os.environ.get('DSV41_V5_SOURCE', '/opt/dsv41/data/src-image'))
PKG = ROOT / 'offload/spark_leads'


@pytest.fixture(scope="module")
def patched() -> Path:
    # Exercise the committed K1 + UP1 + process-proxy composition, not X1's obsolete patch base.
    return ROOT / "offload/upstream_adopt/overlay"


def load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_scheduler_hand_port_matches_helper(patched: Path) -> None:
    helper = load_module('x1_replay', patched / 'vllm/models/deepseek_v41/replay_skip.py')
    scheduler = ast.parse((patched / 'vllm/v1/core/sched/scheduler.py').read_text())
    fn = next(n for n in scheduler.body if isinstance(n, ast.FunctionDef) and n.name == '_dsv41_replay_min_chunk')
    scope: dict[str, object] = {'_DSV41_REPLAY_SKIP': True}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<scheduler helper>', 'exec'), scope)
    rule = scope[fn.name]
    assert callable(rule)
    for enabled in [False, True]:
        for mm in [False, True]:
            for q in [0, 1, 127, 128, 129, 4096]:
                for remaining in [0, 128, 129, 4096, 150000]:
                    assert rule(q, remaining, 128, mm, enabled) == helper.min_chunk(q, remaining, 128, enabled, mm)
    assert sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == fn.name
               for n in ast.walk(scheduler)) == 2


def test_small_proposals_never_manufacture_zero(patched: Path) -> None:
    helper = load_module('x1_replay', patched / 'vllm/models/deepseek_v41/replay_skip.py')
    for q in (1, 64, 127, 128, 129):
        assert helper.min_chunk(q, 10000, 128, True) == q



def test_prompt_logprobs_and_multimodal_never_skip(patched: Path) -> None:
    h = sys.modules.get('x1_replay') or load_module('x1_replay', patched / 'vllm/models/deepseek_v41/replay_skip.py')
    skip = h.skip_mask(np.array([4096]*4), np.array([True, True, False, True]), np.array([8000]*4),
                       np.array([False, True, False, False]), np.array([False, False, False, True]), 128)
    assert skip.tolist() == [True, False, False, False]


@pytest.mark.parametrize("fused", [False, True])
def test_real_tap_suffix_with_mixed_decode_and_resync(patched: Path, fused: bool) -> None:
    import torch
    sys.path.insert(0, str(patched / 'offload/dspark/spark'))
    gmod = load_module('remote_dspark.x1_gather', patched / 'offload/dspark/spark/remote_dspark/gather.py')
    from remote_dspark.state import WindowRange
    g = gmod.RowGatherer(max_slots=2, window=128, hidden=2, n_taps=3, max_rows=256,
                        device=torch.device('cpu'), fused=fused)
    assert g.fused == fused
    rng = WindowRange(128)
    for first, q, kept, need in [(0, 4096, 1, False), (4096, 4096, 1, False),
                                  (8192, 2048, 128, False), (10240, 64, 64, True)]:
        bv = gmod.BatchView(['prefill', 'decode'], np.array([0, 1]), np.array([0, q, q+4]),
                           np.array([need, True]), np.array([0, 3]), tap_lens=np.array([kept, 4]))
        taps = [torch.zeros(q+4, 2, dtype=torch.bfloat16) for _ in range(3)]
        for tap in taps:
            tap[q-kept:q] = 7
            tap[q:] = 9
        result = g.gather(bv, taps, torch.cat([torch.arange(first, first+q), torch.arange(100, 104)]),
                          torch.tensor([0, 2]), torch.tensor([int(need), 2]), torch.tensor([0, 1]),
                          torch.tensor([[11], [12]]), torch.tensor([[13, 14]]))
        assert result.valid.tolist() == [kept, 2]
        assert result.first_pos[0] == first+q-kept
        assert (g.rows_view(2, kept+2)[:kept] == 7).all()
        rng.apply(first+q-kept, kept, first+q)
        if need:
            assert rng.rows_for_anchor(first+q) == 128
            assert g.gather_from_ring(1, [(0, rng.lo, rng.hi-rng.lo+1)]) == 128
            assert (g.rows_view(1, 128) == 7).all()
    # A stale previous prefix's rows must not remain reachable after a skipped gap.
    assert rng.lo == 10240+64-128


def test_invalid_provenance_refuses_before_gather(patched: Path) -> None:
    import torch
    sys.path.insert(0, str(patched / 'offload/dspark/spark'))
    gmod = sys.modules.get('remote_dspark.x1_gather') or load_module(
        'remote_dspark.x1_gather', patched / 'offload/dspark/spark/remote_dspark/gather.py')
    g = gmod.RowGatherer(max_slots=1, window=128, hidden=2, n_taps=3, max_rows=128,
                        device=torch.device('cpu'), fused=False)
    for lens, need in [(0, False), (130, False), (1, True)]:
        bv = gmod.BatchView(['a'], np.array([0]), np.array([0, 129]), np.array([need]), np.array([0]), tap_lens=np.array([lens]))
        with pytest.raises(ValueError, match='provenance|drafting request'):
            g.gather(bv, [torch.zeros(129, 2)]*3, torch.arange(129), torch.zeros(1), torch.zeros(1),
                     torch.zeros(1), torch.zeros(1, 1), torch.zeros(1, 1))


def test_process_proxy_batchview_positional_abi(patched: Path) -> None:
    sys.path.insert(0, str(patched / 'offload/dspark/spark'))
    from remote_dspark.roundcore import BatchView
    bv = BatchView(['a'], np.array([0]), np.array([0, 1]), np.array([False]), np.array([0]), 4096)
    assert bv.max_q == 4096 and bv.longest_query == 4096
    assert bv.tap_lens is None
    bv.tap_lens = np.array([1])
    assert bv.longest_query == 4096  # engine provenance does not rewrite the proxy phase descriptor
