"""Exercise new persistent buffers, allocation boundaries and grouped fallback."""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from offload.dspark.kernels.fp8dense import DenseFP8
from offload.dspark.native import formats as F, ops as NO
from offload.dspark.native.kx import KXFusedBank, KXSelectiveDense

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires locked SM86 GPU')


def no_allocation(*args: object, **kwargs: object) -> None:
    raise AssertionError('allocation inside new wrapper call')


def test_fused_persistent_slot_buffers() -> None:
    dev = torch.device('cuda:0')
    host = NO.ExpertBank('test', 'packed', 4, 128, 64, torch.device('cpu'))
    host.w13.random_(0, 256); host.w2.random_(0, 256)
    host.s13.fill_(120); host.s2.fill_(120)
    bank = KXFusedBank(host, dev, 1)
    x = F.act_qdq(torch.randn(10, 128, dtype=torch.bfloat16)).to(dev)
    ids = (torch.arange(30).view(10, 3) % 4).to(dev, torch.int64)
    rw = torch.full((10, 3), .5, device=dev)
    storage = bank.slot_buffers[10]
    storage[:, 3].fill_(12345.)
    with patch('torch.empty', no_allocation), patch('torch.empty_like', no_allocation):
        actual = bank.forward_slots(x, ids, rw, out=storage[:, :3])
    assert actual.data_ptr() == storage.data_ptr()
    assert bool((storage[:, 3] == 12345.).all())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bank.forward_slots(x, ids, rw)
    x.mul_(.5); ids.zero_()
    graph.replay()
    replayed = bank.outputs[10].clone()
    bank.forward_slots(x, ids, rw)
    assert torch.equal(replayed, bank.outputs[10])
    bank.check()


@pytest.mark.parametrize('groups', (1, 4))
def test_dense_persistent_outputs_and_fallback(groups: int) -> None:
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dev = torch.device('cuda:0')
    codes = torch.randint(0, 0x70, (128, 128), dtype=torch.uint8)
    scales = torch.full((4, 4), 120, dtype=torch.uint8)
    decoded = F.decode_fp8_block(codes, scales).bfloat16().to(dev)
    baseline = NO.NativeDense('test', 'bf16', decoded, None)
    candidate = DenseFP8.from_checkpoint(codes, scales, dev, groups=groups)
    op = KXSelectiveDense(baseline, candidate, (5,))
    x = F.act_qdq(torch.randn(5, groups*128, dtype=torch.bfloat16)).to(dev)
    with patch('torch.empty', no_allocation), patch('torch.empty_like', no_allocation):
        output = op(x)
    assert output.data_ptr() == op.outputs[5, False].data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x)
    x.mul_(.5); graph.replay()
    replayed = output.clone()
    assert torch.equal(replayed, op(x))
    fallback = x[:4]
    want = (torch.bmm(fallback.view(4, groups, 128).transpose(0, 1),
                      decoded.view(groups, 128//groups, 128).transpose(1, 2))
            .transpose(0, 1).reshape(4, 128))
    assert torch.equal(op(fallback), want)


def test_shared_epilogue_exact_old_pipeline() -> None:
    from offload.dspark.kernels.fp8dense.epilogue import shared_activation_into

    torch.manual_seed(731)
    dev = torch.device('cuda:0')
    gate = (torch.randn(20, 1152, device=dev)*8).bfloat16()
    up = (torch.randn_like(gate)*8).bfloat16()
    ones = torch.ones(20, 1, device=dev)
    out = torch.empty_like(gate)
    expected = NO.act_qdq(NO.swiglu_bf16(gate, up))
    with patch('torch.empty', no_allocation), patch('torch.empty_like', no_allocation):
        shared_activation_into(gate, up, ones, out)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize('dtype', (torch.int32, torch.int64))
def test_post_tp_combine_exact(dtype: torch.dtype) -> None:
    from offload.dspark.kernels.fusedmoe.epilogue import combine_official_into

    dev = torch.device('cuda:0')
    torch.manual_seed(20261023)
    storage = torch.randn(20, 4*5120+17, device=dev)
    parts = storage[:, :4*5120].view(20, 4, 5120)
    parts[0, :, :4] = torch.tensor([[1.0039, -0., 0., 123.], [-1., -0., -0., -123.],
                                  [1e-4, 0., -0., 1e-5], [0., 0., 0., 0.]], device=dev)
    ids = torch.stack([torch.randperm(3, device=dev)+10 for _ in range(20)]).to(dtype)
    out = torch.empty(20, 5120, dtype=torch.bfloat16, device=dev)
    expected = NO.moe_combine_official(parts[:, :3], parts[:, 3], ids)
    with patch('torch.empty', no_allocation), patch('torch.empty_like', no_allocation):
        combine_official_into(parts, ids, out)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        combine_official_into(parts, ids, out)
    parts.mul_(.25); ids.copy_(ids.flip(-1)); graph.replay()
    expected = NO.moe_combine_official(parts[:, :3], parts[:, 3], ids)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


def test_shared_bf16_pair_buffers() -> None:
    from offload.dspark.native.kx import KXSharedPair
    from offload.dspark.kernels.fp8dense.epilogue import shared_activation_into

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dev = torch.device('cuda:0')
    codes = torch.randint(0, 0x70, (64, 128), dtype=torch.uint8)
    scale = torch.full((2, 4), 120, dtype=torch.uint8)
    w = F.decode_fp8_block(codes, scale).bfloat16().to(dev)
    ops = [KXSelectiveDense(NO.NativeDense('test', 'bf16', w, None),
                           DenseFP8.from_checkpoint(codes, scale, dev), (5,)) for _ in range(2)]
    pair = KXSharedPair(*ops)
    x = F.act_qdq(torch.randn(20, 128, dtype=torch.bfloat16)).to(dev)
    ones = torch.ones(20, 1, device=dev)
    out = torch.empty(20, 64, dtype=torch.bfloat16, device=dev)
    with patch('torch.empty', no_allocation), patch('torch.empty_like', no_allocation):
        g, u = pair(x)
        shared_activation_into(g, u, ones, out)
    assert g.untyped_storage().data_ptr() == pair.output.untyped_storage().data_ptr()
    expected = NO.act_qdq(NO.swiglu_bf16(g, u))
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g, u = pair(x)
        shared_activation_into(g, u, ones, out)
    x.mul_(.5); graph.replay()
    replayed = out.clone()
    g, u = pair(x); shared_activation_into(g, u, ones, out)
    assert torch.equal(out.view(torch.int16), replayed.view(torch.int16))
