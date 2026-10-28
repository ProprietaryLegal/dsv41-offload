"""CPU admission plus focused GPU arithmetic/lifetime/capture contract tests."""
from __future__ import annotations
import os
import pytest
import torch
from .api import DenseFP8, domain_histogram, extension, require_lock
from offload.dspark.native.formats import act_qdq, decode_fp8_block


@pytest.mark.parametrize('bad', ['nan_code','nan_scale','subnormal','overflow'])
def test_admission_refuses_bad_domain(bad:str)->None:
    c=torch.full((1,32,32),56,dtype=torch.uint8)
    s=torch.full((1,1,1),127,dtype=torch.uint8)
    if bad=='nan_code': c[0,0,0]=127
    if bad=='nan_scale': s.fill_(255)
    if bad=='subnormal': c.fill_(1);s.fill_(0)
    if bad=='overflow': c.fill_(126);s.fill_(254)
    with pytest.raises(ValueError): domain_histogram(c,s)


def require_gpu()->torch.device:
    if os.environ.get('KX_FP8DENSE_LOCKED')!='1': pytest.skip('GPU tests require locked runner')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc':
        raise RuntimeError('wrong GPU pin')
    require_lock()
    return torch.device('cuda:0')


def test_decoder_full_admitted_domain()->None:
    d=require_gpu()
    codes=torch.arange(256,dtype=torch.int32).repeat(255).byte()
    scales=torch.arange(255,dtype=torch.int32).repeat_interleave(256).byte()
    finite=(codes&127)!=127
    vals=torch.ldexp(codes.view(torch.float8_e4m3fn).float(),scales.int()-127)
    good=finite&torch.isfinite(vals)&((vals==0)|(vals.abs()>=torch.finfo(torch.bfloat16).tiny))
    good&=torch.isfinite(vals.bfloat16().float())
    c,s=codes[good].to(d),scales[good].to(d)
    out=torch.empty(c.shape,device=d,dtype=torch.bfloat16)
    extension().test_decode(c.data_ptr(),s.data_ptr(),out.data_ptr(),c.numel(),torch.cuda.current_stream(d).cuda_stream)
    assert torch.equal(out.cpu().view(torch.int16),vals[good].bfloat16().view(torch.int16))


@pytest.mark.parametrize('m,groups,f32,splits', [(5,1,False,1),(10,1,False,4),(15,1,True,3),
    (20,1,True,8),(128,1,False,4),(5,4,False,2),(20,4,True,4)])
def test_mma_and_graph(m:int,groups:int,f32:bool,splits:int)->None:
    d=require_gpu()
    torch.manual_seed(1384)
    n,k=64,128
    codes=torch.randint(0,256,(groups*n,k),dtype=torch.uint8)
    codes[(codes&127)==127]=0
    scales=torch.randint(119,129,(groups*n//32,k//32),dtype=torch.uint8)
    # Exercise every non-NaN code, including source subnormals and signed zeros.
    dense=DenseFP8.from_checkpoint(codes,scales,d,groups=groups)
    wd=decode_fp8_block(codes,scales)
    x=act_qdq(torch.randn(m,groups*k,dtype=torch.bfloat16)*.03).to(d)
    backing=torch.empty((m,groups*n+32),device=d,dtype=torch.float32 if f32 else torch.bfloat16)
    out=backing[:,:groups*n]
    scratch=torch.empty((splits,m,groups,n),device=d) if splits>1 else None
    call=lambda:dense.linear_into(x,out,scratch=scratch,splits=splits)
    call()
    def check()->None:
        want=torch.cat([(x[:,g*k:(g+1)*k].cpu().double()@wd[g*n:(g+1)*n].double().t()).float() for g in range(groups)],1)
        actual=out.cpu()
        if f32: assert float((actual-want).norm()/want.norm())<=1e-6
        else: assert float((actual==want.bfloat16()).float().mean())>=.999
    check()
    torch.cuda.synchronize(d)
    allocated=torch.cuda.memory_allocated(d)
    call()
    assert torch.cuda.memory_allocated(d)==allocated
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): call()
    eager=out.clone()
    graph.replay()
    assert torch.equal(out,eager)
    x.copy_(act_qdq(torch.randn(m,groups*k,dtype=torch.bfloat16)*.03).to(d))
    graph.replay()
    check()


def test_full_grid_shard_offsets()->None:
    d=require_gpu()
    torch.manual_seed(92)
    c=torch.randint(16,120,(128,256),dtype=torch.uint8)
    s=torch.randint(113,130,(4,8),dtype=torch.uint8)
    dense=DenseFP8.from_checkpoint(c[32:96,64:192],s,d,row_offset=32,col_offset=64)
    x=act_qdq(torch.randn(5,128,dtype=torch.bfloat16)*.05).to(d)
    out=torch.empty((5,64),device=d,dtype=torch.float32)
    dense.linear_into(x,out)
    w=decode_fp8_block(c[32:96,64:192],s,rows=(32,96),cols=(64,192))
    want=(x.cpu().double()@w.double().t()).float()
    assert float((out.cpu()-want).norm()/want.norm())<=1e-6


def test_failure_contract()->None:
    d=require_gpu()
    c=torch.full((32,64),56,dtype=torch.uint8)
    s=torch.full((1,2),127,dtype=torch.uint8)
    dense=DenseFP8.from_checkpoint(c,s,d)
    x=torch.ones(5,64,device=d,dtype=torch.bfloat16)
    out=torch.empty(5,32,device=d,dtype=torch.bfloat16)
    with pytest.raises(ValueError,match='scratch'): dense.linear_into(x,out,splits=2)
    with pytest.raises(ValueError,match='alias'): dense.linear_into(x,x[:,:32])
    with pytest.raises(ValueError): dense.linear_into(x,out,row_tiles=3)
    with pytest.raises(ValueError): dense.linear_into(x.float(),out)
    with pytest.raises(ValueError): dense.linear_into(x,out[:,::2])
    with pytest.raises(ValueError): DenseFP8.from_checkpoint(c,s,d,col_offset=16)
    with pytest.raises(ValueError): DenseFP8.from_checkpoint(c,s,d,row_offset=32)
    odd=torch.empty(5,65,device=d,dtype=torch.bfloat16)[:,:64]
    with pytest.raises(ValueError): dense.linear_into(odd,out)

@pytest.mark.parametrize('m,groups,f32',[(5,1,False),(10,1,True),(15,4,False),(20,4,True)])
def test_simt_raw_control(m:int,groups:int,f32:bool)->None:
    d=require_gpu()
    torch.manual_seed(2873)
    c=torch.randint(0,256,(groups*64,128),dtype=torch.uint8)
    c[(c&127)==127]=0
    s=torch.randint(115,128,(groups*2,4),dtype=torch.uint8)
    dense=DenseFP8.from_checkpoint(c,s,d,groups=groups,layout='simt')
    x=act_qdq(torch.randn(m,groups*128,dtype=torch.bfloat16)*.1).to(d)
    out=torch.empty((m,groups*64),device=d,dtype=torch.float32 if f32 else torch.bfloat16)
    dense.linear_into(x,out)
    w=decode_fp8_block(c,s)
    ref=torch.cat([(x[:,g*128:(g+1)*128].cpu().double()@w[g*64:(g+1)*64].double().t()).float() for g in range(groups)],1)
    if f32: assert float((out.cpu()-ref).norm()/ref.norm())<=1e-6
    else: assert float((out.cpu()==ref.bfloat16()).float().mean())>=.999
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):dense.linear_into(x,out)
    x.mul_(.5)
    graph.replay();replayed=out.clone();dense.linear_into(x,out)
    assert torch.equal(replayed,out)


def test_high_scale_fallback_and_large_activation()->None:
    d=require_gpu()
    c=torch.full((32,128),25,dtype=torch.uint8)
    s=torch.full((1,4),147,dtype=torch.uint8)
    dense=DenseFP8.from_checkpoint(c,s,d)
    assert not dense.fast_scales
    x=act_qdq(torch.full((5,128),262144,dtype=torch.bfloat16)).to(d)
    out=torch.empty((5,32),device=d,dtype=torch.float32)
    dense.linear_into(x,out)
    w=decode_fp8_block(c,s)
    expected=(x.cpu().double()@w.double().t()).float()
    assert torch.equal(out.cpu(),expected)
