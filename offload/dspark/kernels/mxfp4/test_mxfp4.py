"""Decoder, shard, bounded-ID, arithmetic and graph qualification."""
from __future__ import annotations
import os
import pytest
import torch
from . import PackedExperts, activation_into, combine_into
from .kernels import decode_kernel, linear_kernel
from .reference import admit, decode_codes, unpack


def test_cpu_domain() -> None:
    c = torch.arange(16).repeat(256).to(torch.uint8)
    s = torch.arange(256).repeat_interleave(16).to(torch.uint8)
    ref = decode_codes(c, s)
    assert torch.isnan(ref[s == 255]).all()
    assert torch.isfinite(ref[s != 255]).all()
    assert ref[(c == 8) & (s != 255)].signbit().all()
    widened = ref.bfloat16().double()
    exact = torch.isfinite(widened) & (s != 255)
    assert torch.equal(widened[exact], ref[exact])
    assert int(exact.sum()) == 4068


def test_admission() -> None:
    p = torch.full((1, 2, 16), 0x21, dtype=torch.uint8)
    s = torch.full((1, 2, 1), 127, dtype=torch.uint8)
    assert admit(p, s)['values'] == 64
    with pytest.raises(ValueError, match='NaN'):
        admit(p, s.fill_(255))
    with pytest.raises(ValueError, match='subnormal'):
        admit(p, s.fill_(0))
    with pytest.raises(ValueError, match='not exactly representable'):
        admit(p.fill_(0x77), s.fill_(254))
    with pytest.raises(ValueError, match='group size'):
        unpack(torch.zeros((1, 2, 17), dtype=torch.uint8), s)


@pytest.fixture
def gpu() -> torch.device:
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != 'GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc'
            or os.environ.get('KX_MXFP4_LOCKED') != '1'):
        pytest.fail('GPU tests require run_locked.sh on the allocated development UUID')
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_capability(0) == (8, 6)
    return torch.device('cuda:0')


def bank(e: int, n: int, kk: int, d: torch.device) -> tuple[PackedExperts, torch.Tensor]:
    p = torch.randint(0, 256, (e, n, kk//2), dtype=torch.uint8)
    s = torch.randint(120, 128, (e, n, kk//32), dtype=torch.uint8)
    return PackedExperts.from_checkpoint(p, s, d), unpack(p, s).float()


def test_gpu_decode_domain(gpu: torch.device) -> None:
    c = torch.arange(16).repeat(256).to(torch.uint8)
    s = torch.arange(256).repeat_interleave(16).to(torch.uint8)
    ref = decode_codes(c, s).bfloat16()
    out = torch.empty(4096, dtype=torch.bfloat16, device=gpu)
    decode_kernel[(16,)](c.to(gpu), s.to(gpu), out, 4096, 256)
    actual = out.cpu()
    finite_scale = s != 255
    assert torch.equal(actual.view(torch.int16)[finite_scale], ref.view(torch.int16)[finite_scale])
    assert torch.isnan(actual[s == 255]).all()


@pytest.mark.parametrize('mode', ['mma', 'simt'])
@pytest.mark.parametrize('rows,top,n,kk', [(1, 1, 37, 96), (4, 3, 65, 160), (20, 3, 33, 64)])
def test_linear(gpu: torch.device, mode: str, rows: int, top: int, n: int, kk: int) -> None:
    torch.manual_seed(725)
    w, ref = bank(4, n, kk, gpu)
    x = torch.randn(rows, kk, dtype=torch.bfloat16, device=gpu)
    ids = (torch.arange(rows*top, device=gpu, dtype=torch.int32) % 4).view(rows, top)
    out = torch.empty(rows, top, n, device=gpu)
    status = torch.zeros(1, dtype=torch.int32, device=gpu)
    w.linear_into(x, ids, out, status, mode=mode)
    expected = torch.stack([x.cpu().float()[r] @ ref[int(ids[r,t])].T
                            for r in range(rows) for t in range(top)]).view(rows, top, n)
    torch.testing.assert_close(out.cpu(), expected, atol=1e-4, rtol=2e-5)
    assert status.item() == 0


def test_gpu_shards(gpu: torch.device) -> None:
    torch.manual_seed(701)
    p = torch.randint(0, 256, (3, 70, 96), dtype=torch.uint8)
    s = torch.randint(119, 129, (3, 70, 6), dtype=torch.uint8)
    full = unpack(p, s)
    for lo, hi in [(0, 96), (96, 192)]:
        w = PackedExperts.from_checkpoint(p[:, :, lo//2:hi//2].contiguous(),
                                         s[:, :, lo//32:hi//32].contiguous(), gpu)
        x = torch.eye(hi-lo, dtype=torch.bfloat16, device=gpu)
        ids = torch.full((hi-lo, 1), 2, dtype=torch.int32, device=gpu)
        out = torch.empty(hi-lo, 1, 70, device=gpu)
        status = torch.zeros(1, dtype=torch.int32, device=gpu)
        w.linear_into(x, ids, out, status)
        assert torch.equal(out[:,0].cpu(), full[2,:,lo:hi].T.float())
    for lo, hi in [(0, 35), (35, 70)]:
        w = PackedExperts.from_checkpoint(p[:,lo:hi].contiguous(), s[:,lo:hi].contiguous(), gpu)
        x = torch.eye(192, dtype=torch.bfloat16, device=gpu)
        ids = torch.ones(192, 1, dtype=torch.int32, device=gpu)
        out = torch.empty(192, 1, hi-lo, device=gpu)
        status = torch.zeros(1, dtype=torch.int32, device=gpu)
        w.linear_into(x, ids, out, status)
        assert torch.equal(out[:,0].cpu(), full[1,lo:hi].T.float())


def test_pipeline_capture_failure(gpu: torch.device) -> None:
    torch.manual_seed(39)
    rows, top, n, hidden = 4, 3, 96, 160
    gate, gr = bank(5, n, hidden, gpu)
    up, ur = bank(5, n, hidden, gpu)
    down, dr = bank(5, hidden, n, gpu)
    x = torch.randn(rows, hidden, device=gpu, dtype=torch.bfloat16) * .1
    ids = (torch.arange(rows*top, device=gpu, dtype=torch.int32) % 5).view(rows, top)
    route_weights = torch.rand(rows, top, device=gpu)
    unit_weights = torch.ones_like(route_weights)
    g = torch.empty(rows, top, n, device=gpu)
    u = torch.empty_like(g)
    a = torch.empty(rows*top, n, device=gpu, dtype=torch.bfloat16)
    parts = torch.empty(rows, top, hidden, device=gpu)
    out = torch.empty(rows, hidden, device=gpu)
    status = torch.zeros(1, dtype=torch.int32, device=gpu)

    def run(fused: bool = True) -> None:
        status.zero_()
        if fused:
            gate.linear_into(x, ids, g, status, up=up, up_out=u)
        else:
            gate.linear_into(x, ids, g, status)
            up.linear_into(x, ids, u, status)
        activation_into(g, u, route_weights, a, 10.)
        down.linear_into(a, ids, parts, status, per_slot=True)
        combine_into(parts, unit_weights, out)

    run()
    fused = out.clone()
    run(False)
    assert torch.equal(out, fused)
    expected = []
    for row in range(rows):
        total = torch.zeros(hidden)
        for slot in range(top):
            e = int(ids[row,slot])
            gg = (x[row].cpu().float() @ gr[e].T).bfloat16().float().clamp(max=10.)
            uu = (x[row].cpu().float() @ ur[e].T).bfloat16().float().clamp(-10., 10.)
            aa = (torch.nn.functional.silu(gg)*uu*route_weights[row,slot].cpu()).bfloat16().float()
            total += aa @ dr[e].T
        expected.append(total)
    torch.testing.assert_close(out.cpu(), torch.stack(expected), atol=.03, rtol=.003)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, fused)
    x.mul_(.5)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(out, fused)
    graph_result = out.clone()
    run()
    assert torch.equal(out, graph_result)
    for invalid in (5, -1):
        ids[0,0] = invalid
        graph.replay()
        torch.cuda.synchronize()
        assert status.item() == 1
        assert torch.isnan(out[0]).all()
        assert torch.isfinite(out[1:]).all()


def test_generated_sm86(gpu: torch.device) -> None:
    w, _ = bank(1, 32, 64, gpu)
    x = torch.ones(1, 64, dtype=torch.bfloat16, device=gpu)
    ids = torch.zeros(1, 1, dtype=torch.int32, device=gpu)
    out = torch.empty(1, 1, 32, device=gpu)
    status = torch.zeros(1, dtype=torch.int32, device=gpu)
    kernel = linear_kernel.warmup(x, ids, w.codes, w.scales, w.codes, w.scales, out, out, status,
                                 32, 64, 1, 1, False, False, True, 32, 64,
                                 num_warps=4, enable_fp_fusion=False, grid=(1,1))
    ptx = kernel.asm['ptx']
    assert '.target sm_86' in ptx
    assert 'mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32' in ptx
    assert 'tcgen05' not in ptx and 'e2m1' not in ptx and 'cp.async.bulk' not in ptx


@pytest.mark.parametrize('rows,top',[(1,3),(4,3),(20,3)])
def test_regfed(gpu: torch.device, rows: int, top: int) -> None:
    from .regfed import RegisterExperts
    torch.manual_seed(44)
    p = torch.randint(0,256,(5,64,96),dtype=torch.uint8)
    s = torch.randint(111,128,(5,64,6),dtype=torch.uint8)
    w = RegisterExperts.from_checkpoint(p,s,gpu)
    x = torch.randn(rows,192,device=gpu,dtype=torch.bfloat16)
    # Reuse covers groups >8; mixed order exercises the device compaction.
    ids = (torch.arange(rows*top,device=gpu,dtype=torch.int32)%3).view(rows,top)
    schedule = torch.empty(rows*top,8,device=gpu,dtype=torch.int32)
    out = torch.empty(rows,top,64,device=gpu)
    uout = torch.empty_like(out)
    status = torch.zeros(1,device=gpu,dtype=torch.int32)
    w.linear_into(x,ids,out,status,schedule,up=w,up_out=uout)
    assert torch.equal(out,uout)
    ref = unpack(p,s).float()
    expected = torch.stack([x[r].cpu().float() @ ref[int(ids[r,t])].T
                            for r in range(rows) for t in range(top)]).view(rows,top,64)
    torch.testing.assert_close(out.cpu(),expected,atol=1e-4,rtol=2e-5)
    assert status.item()==0
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        w.linear_into(x,ids,out,status,schedule)
    stream.synchronize()
    with torch.cuda.graph(graph,stream=stream):
        status.zero_()
        w.linear_into(x,ids,out,status,schedule)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.cpu(),expected,atol=1e-4,rtol=2e-5)
    ids[0,0] = -1
    graph.replay()
    torch.cuda.synchronize()
    assert status.item()==1
    assert torch.isnan(out[0,0]).all()
    assert torch.isfinite(out[0,1:]).all()
    ids.fill_(4)
    graph.replay()
    torch.cuda.synchronize()
    expected = x.cpu().float() @ ref[4].T
    torch.testing.assert_close(out[:,0].cpu(),expected,atol=1e-4,rtol=2e-5)
    assert status.item()==0


def test_regfed_domain(gpu: torch.device) -> None:
    from .regfed import extension
    codes = torch.arange(16).repeat(17).byte()
    scales = torch.arange(111,128).repeat_interleave(16).byte()
    ref = decode_codes(codes,scales).bfloat16()
    c,s = codes.to(gpu),scales.to(gpu)
    out = torch.empty_like(c,dtype=torch.bfloat16)
    extension().test_decode(c.data_ptr(),s.data_ptr(),out.data_ptr(),c.numel(),torch.cuda.current_stream().cuda_stream)
    assert torch.equal(out.cpu().view(torch.int16),ref.view(torch.int16))


def test_native_combine_order(gpu: torch.device) -> None:
    from . import combine_after_tp_into
    parts = torch.tensor([[[1.e8,1.003],[-1.e8,1.007],[1.,1.011]],
                          [[1.,2.],[-1.e8,3.],[1.e8,4.]]],device=gpu)
    ids = torch.tensor([[2,0,1],[1,0,1]],dtype=torch.int32,device=gpu)
    out = torch.empty(2,2,device=gpu)
    combine_after_tp_into(parts,ids,out)
    expected=[]
    for row in range(2):
        total=torch.zeros(2)
        for slot in sorted(range(3),key=lambda slot: (int(ids[row,slot]),slot)):
            total += parts[row,slot].cpu().bfloat16().float()
        expected.append(total)
    assert torch.equal(out.cpu(),torch.stack(expected))
