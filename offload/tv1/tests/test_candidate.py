from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

from offload.tv1.candidate import candidate_logits, validate

Inputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

GPU = os.environ.get('TV1_GPU_TEST') == '1'


def case(rows: int, width: int, page_size: int, device: str) -> Inputs:
    g = torch.Generator(device=device).manual_seed(41 + rows + width)
    count = (width + page_size - 1) // page_size
    # Noncontiguous page stride reproduces K1 group packing (including 192-state pages).
    storage = torch.full((count + 3, page_size * 132 + 256), 23, dtype=torch.uint8, device=device)
    cache = storage[:, :page_size * 132].view(count + 3, page_size, 1, 132)
    keys = torch.randn(count + 3, page_size, 128, generator=g, device=device).to(torch.float8_e4m3fn)
    scales = torch.exp2(torch.randint(-3, 3, (count + 3, page_size), generator=g, device=device).float())
    storage[:, :page_size * 128].copy_(keys.view(torch.uint8).reshape(count + 3, -1))
    storage[:, page_size * 128:page_size * 132].copy_(scales.view(torch.uint8).reshape(count + 3, -1))
    q = torch.randn(rows, 32, 128, generator=g, device=device).to(torch.float8_e4m3fn)
    weights = torch.randn(rows, 32, generator=g, device=device) / 64
    pages = torch.randperm(count, generator=g, device=device).expand(rows, -1).contiguous().int()
    lens = (width - torch.arange(rows, device=device) % 5).clamp_min(0).int()
    nc = min(2048, (width + 7) // 8)
    candidates = torch.stack([torch.randperm((width + 7) // 8, generator=g, device=device)[:nc]
                              for _ in range(rows)]).int()
    candidates[:, -1] = -1
    if rows > 1:
        candidates[1].fill_(-1)
    return q, cache, weights, lens, pages, candidates


def oracle(q: torch.Tensor, cache: torch.Tensor, weights: torch.Tensor, lens: torch.Tensor,
           pages: torch.Tensor, candidates: torch.Tensor, width: int) -> torch.Tensor:
    p = cache.shape[1]
    raw = torch.as_strided(cache, (cache.shape[0], p * 132), (cache.stride(0), 1))
    keys = raw[:, :p * 128].reshape(-1, p, 128).view(torch.float8_e4m3fn).float()
    scales = raw[:, p * 128:p * 132].contiguous().view(torch.float32)
    positions = torch.arange(width, device=q.device)
    out = torch.empty((q.shape[0], width), device=q.device)
    for row in range(q.shape[0]):
        phys = pages[row, positions // p].long()
        k = keys[phys, positions % p]
        logits = (q[row].float() @ k.T).clamp_min(0)
        values = (logits * weights[row, :, None]).sum(dim=0) * scales[phys, positions % p]
        keep = torch.isin(positions // 8, candidates[row]) & (positions < lens[row])
        out[row] = values.masked_fill(~keep, -float('inf'))
    return out


def selected(x: torch.Tensor, k: int) -> torch.Tensor:
    # Compare same tie rule and canonical ascending indices; no approximate overlap metric.
    ids = torch.argsort(x, dim=-1, descending=True, stable=True)[:, :k]
    return torch.where(torch.isfinite(x.gather(1, ids)), ids, -1).sort(dim=-1).values


def test_layout_validation() -> None:
    args = case(3, 193, 192, 'cpu')
    validate(*args, 193, 8)
    with pytest.raises(ValueError, match='width'):
        validate(*args, 100000, 8)
    with pytest.raises(ValueError, match='FP8'):
        validate(args[0].float(), *args[1:], 193, 8)
    with pytest.raises(ValueError, match='CUDA'):
        candidate_logits(*args, 193)


@pytest.mark.skipif(not GPU, reason='explicit TV1_GPU_TEST=1 on locked x86host UUID only')
@pytest.mark.parametrize('rows,width,page_size', [(1, 193, 64), (3, 32768, 192), (4, 150001, 192),
                                                (6, 8193, 64), (16, 32769, 192), (20, 32768, 192)])
def test_kernel_topk_oracle_and_replay(rows: int, width: int, page_size: int) -> None:
    # Install just our inert helper under the runtime module name; no engine imports.
    import types
    parent = types.ModuleType('vllm_exl3_moe')
    parent.__path__ = []
    sys.modules.setdefault('vllm_exl3_moe', parent)
    path = Path(__file__).parents[1] / 'candidate_kernel.py'
    spec = importlib.util.spec_from_file_location('vllm_exl3_moe.tv1_candidate_kernel', path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    torch.backends.cuda.matmul.allow_tf32 = False
    args = case(rows, width, page_size, 'cuda')
    want = oracle(*args, width)
    actual = candidate_logits(*args, width)
    assert torch.equal(selected(actual, min(512, width)), selected(want, min(512, width)))
    stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        candidate_logits(*args, width)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = candidate_logits(*args, width)
    # Reuse a captured graph with changed page tables, candidates, lengths and queries.
    args[0].copy_((-args[0].float()).to(args[0].dtype))
    args[3].clamp_max_(max(1, width // 2))
    args[4].copy_(args[4].flip(1))
    args[5].copy_(args[5].roll(1, 1))
    captured.fill_(float('nan'))
    graph.replay()
    want = oracle(*args, width)
    assert torch.equal(selected(captured, min(512, width)), selected(want, min(512, width)))
    args[3].zero_()
    graph.replay()
    assert torch.isneginf(captured).all()
