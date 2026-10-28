"""Focused arithmetic/ABI/capture checks, executed only under the dev-card lock."""
from __future__ import annotations
import os
import pytest
import torch
from offload.dspark.kernels.mxfp4.regfed import RegisterExperts
from offload.dspark.native import formats as F, ops as NO
from . import FusedMoE, Workspace


@pytest.fixture(scope='module')
def banks() -> tuple[FusedMoE, tuple[torch.Tensor, ...]]:
    assert os.environ.get('KX_FUSEDMOE_LOCKED') == '1'
    torch.manual_seed(123)
    dev = torch.device('cuda:0')
    raw = []
    regs = []
    for n, k in ((64, 128), (64, 128), (128, 64)):
        w = torch.randint(0, 256, (4, n, k//2), dtype=torch.uint8)
        s = torch.randint(119, 127, (4, n, k//32), dtype=torch.uint8)
        raw.extend((w, s))
        regs.append(RegisterExperts.from_checkpoint(w, s, dev))
    return FusedMoE.from_register_banks(*regs), tuple(raw)


@pytest.mark.parametrize('variant', [4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536, 131072])
@pytest.mark.parametrize('rows', [1, 5, 10, 20])
@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
def test_arithmetic_and_capture(banks: tuple[FusedMoE, tuple[torch.Tensor, ...]], rows: int,
                                dtype: torch.dtype, variant: int) -> None:
    op, raw = banks
    op = FusedMoE(op.gate, op.up, op.down, op.ext, variant)
    d = op.gate.codes.device
    x = F.act_qdq(torch.randn(rows, 128).to(torch.bfloat16)).to(d)
    ids = (torch.arange(rows*3).reshape(rows, 3) % 4).to(d, dtype)
    rw = torch.rand(rows, 3, device=d)
    storage = torch.full((rows, 4, 128), 12345., device=d)
    out = storage[:, :3]
    ws = Workspace.allocate(rows, 64, d)
    status = torch.zeros(1, dtype=torch.int32, device=d)
    gu = tuple(torch.empty(rows, 3, 64, device=d) for _ in range(2))
    op.forward_into(x, ids, rw, out, status, ws, projections=gu)
    h = NO.swiglu_qdq(*gu, rw)
    assert torch.equal(h.view(torch.int16), ws.hq.view(torch.int16))
    for p, (g, u) in enumerate(zip(gu[0].flatten(0, 1), gu[1].flatten(0, 1))):
        e = int(ids.flatten()[p])
        for z, got in enumerate((g, u)):
            ref = (x[p//3].double() @ F.decode_fp4(raw[z*2][e], raw[z*2+1][e]).to(d).double().t()).float()
            torch.testing.assert_close(got, ref, atol=2e-5, rtol=1e-5)
        ref = ws.hq[p].double() @ F.decode_fp4(raw[4][e], raw[5][e]).to(d).double().t()
        torch.testing.assert_close(out[p//3, p%3], ref.float(), atol=2e-5, rtol=1e-5)
    assert int(status) == 0
    assert bool((storage[:, 3] == 12345.).all())
    if variant == 65536:
        control=FusedMoE(op.gate,op.up,op.down,op.ext,256)
        other=torch.empty(rows,3,128,device=d);other_ws=Workspace.allocate(rows,64,d)
        control.forward_into(x,ids,rw,other,status,other_ws)
        assert torch.equal(other,out)
        assert torch.equal(other_ws.hq.view(torch.int16),ws.hq.view(torch.int16))
    before = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.forward_into(x, ids, rw, out, status, ws)
    graph.replay()
    assert torch.equal(out, before)
    # Reuse >8 and changing device inputs/ids under the same captured graph.
    ids.zero_();x.mul_(0.5)
    graph.replay()
    want = out.clone()
    op.forward_into(x, ids, rw, out, status, ws)
    assert torch.equal(out, want)
    ids[0, 0] = 2**40 if dtype == torch.int64 else -1
    ids[-1, -1] = -123
    graph.replay()
    assert int(status) == 1
    assert bool(out[0, 0].isnan().all()) and bool(out[-1, -1].isnan().all())
    if rows > 1:
        assert bool(out[0, 1].isfinite().all())
    ids.zero_();graph.replay()
    assert int(status) == 1  # sticky, never cleared inside op
    assert bool(out.isfinite().all())


def test_reject_bad_output_stride(banks: tuple[FusedMoE, tuple[torch.Tensor, ...]]) -> None:
    op, _ = banks
    d = op.gate.codes.device
    with pytest.raises(ValueError, match='SLOT output'):
        op.forward_into(torch.empty(1, 128, dtype=torch.bfloat16, device=d),
            torch.zeros(1, 3, dtype=torch.int32, device=d), torch.ones(1, 3, device=d),
            torch.empty(3, 1, 128, device=d).transpose(0, 1), torch.zeros(1, dtype=torch.int32, device=d),
            Workspace.allocate(1, 64, d))


@pytest.mark.parametrize('variant', [4, 16, 64, 4096, 8192, 16384, 65536, 131072])
def test_nan_propagation(banks: tuple[FusedMoE, tuple[torch.Tensor, ...]], variant: int) -> None:
    op, _ = banks
    op=FusedMoE(op.gate,op.up,op.down,op.ext,variant)
    d=op.gate.codes.device
    x=torch.zeros(1,128,dtype=torch.bfloat16,device=d);x[0,0]=float('nan')
    ids=torch.tensor([[0,1,2]],dtype=torch.int32,device=d)
    out=torch.empty(1,3,128,device=d)
    op.forward_into(x,ids,torch.ones(1,3,device=d),out,
        torch.zeros(1,dtype=torch.int32,device=d),Workspace.allocate(1,64,d))
    assert bool(out.isnan().all())


@pytest.mark.parametrize('variant', [4096, 131072, 65536])
def test_arbitrary_row_padding(banks: tuple[FusedMoE, tuple[torch.Tensor, ...]], variant: int) -> None:
    op,_=banks
    op=FusedMoE(op.gate,op.up,op.down,op.ext,variant)
    d=op.gate.codes.device
    rows,k=5,128
    pitch=3*k+17
    storage=torch.full((rows,pitch),12345.,device=d)
    out=storage.as_strided((rows,3,k),(pitch,k,1))
    ids=torch.zeros(rows,3,dtype=torch.int32,device=d)
    x=F.act_qdq(torch.randn(rows,k).to(torch.bfloat16)).to(d)
    rw=torch.full((rows,3),.5,device=d)
    ws=Workspace.allocate(rows,64,d)
    status=torch.zeros(1,dtype=torch.int32,device=d)
    op.forward_into(x,ids,rw,out,status,ws)
    reference=torch.empty(rows,3,k,device=d)
    op.forward_into(x,ids,rw,reference,status,ws)
    assert torch.equal(out,reference)
    assert bool((storage[:,3*k:]==12345.).all())
