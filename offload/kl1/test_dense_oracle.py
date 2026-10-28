"""Run the real dense sweep on CPU with identical fake kernels: the oracle must see one input per row count."""
from __future__ import annotations

import argparse
import contextlib
import json
import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

from offload.kl1 import smoe_bench as sb
from offload.kl1 import shapes


class Group:
    next_handle = 0

    def __init__(self, linears: object, k: int) -> None:
        self.handle = Group.next_handle
        Group.next_handle += 1

    def __call__(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        out.copy_(x.float().sum(dim=-1, keepdim=True))
        return out


def test_identical_dense_kernels_have_one_digest_per_row_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_torch = torch
    proxy = types.ModuleType('torch')
    proxy.__dict__.update(vars(real_torch))
    proxy.__dict__['device'] = lambda name: real_torch.device('cpu')
    proxy.__dict__['cuda'] = types.SimpleNamespace(synchronize=lambda: None, CUDAGraph=object,
                                                  graph=lambda graph: contextlib.nullcontext())
    monkeypatch.setitem(sys.modules, 'torch', proxy)
    mr = types.SimpleNamespace(MrowGroup=Group, ext=lambda: types.SimpleNamespace(info32=lambda handle: [2, 128, 256]))
    importer = importlib.import_module
    monkeypatch.setattr(sb, 'importlib', types.SimpleNamespace(import_module=lambda name: mr if name == 'vllm_exl3_moe.mrow' else importer(name)))
    monkeypatch.setattr(sb, '_rand_linear', lambda *args: types.SimpleNamespace(trellis=0, suh=0, svh=0))
    monkeypatch.setattr(sb, '_replay_us', lambda *args: [100.0])
    # Shapes are irrelevant to this failure mode; keep the CPU probe small while running the actual full sweep.
    for key, value in {'H': 8, 'Q_LORA': 4, 'KV_DIM': 4, 'HEADS': 4, 'HEAD_DIM': 4, 'O_GROUPS': 2, 'O_LORA': 4}.items():
        monkeypatch.setattr(shapes, key, value)
    out = tmp_path / 'dense.json'
    assert sb.gpu_dense(argparse.Namespace(out=out, banks=2, rank=0, warmup=0, reps=1)) == 0
    rows = json.loads(out.read_text())['rows']
    assert len(rows) == 18
    for m in (4, 16, 20):
        selected = [r for r in rows if r['rows'] == m]
        assert len(selected) == 6
        assert len({r['out_sha256'] for r in selected}) == 1
        for row in selected:
            assert len(row['info32']) == 2
            assert all(set(bank) == {'wq_a+wkv', 'wq_b', 'wo_a', 'wo_b'} for bank in row['info32'])
            assert all(plan == [2, 128, 256] for bank in row['info32'] for plan in bank.values())
