"""Tiled/split-K Triton control for larger M, using raw E4M3/E8M0 bytes."""
from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _decode(code, scale, FAST:tl.constexpr):
    c = code.to(tl.int32)
    s = scale.to(tl.int32)
    if FAST:
        raw=(((c&127)<<4)|((c&128)<<8)).to(tl.uint16).to(tl.bfloat16,bitcast=True)
        multiplier=((s+120)<<7).to(tl.uint16).to(tl.bfloat16,bitcast=True)
        return (raw.to(tl.float32)*multiplier.to(tl.float32)).to(tl.bfloat16)
    mag = c & 127
    normal = (mag << 4) + ((s - 7) << 7)
    low = (0x60402000c0800000 >> (mag.to(tl.uint64) * 8)) & 255
    sub = (tl.where(mag >= 4, 0x3c00, 0x3b00) | low.to(tl.int32)) + ((s - 127) << 7)
    bits = tl.where(mag >= 8, normal, sub)
    bits = tl.where(mag == 0, 0, bits) | ((c & 128) << 8)
    fallback=bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    raw=(((c&127)<<4)|((c&128)<<8)).to(tl.uint16).to(tl.bfloat16,bitcast=True)
    multiplier=((s+120)<<7).to(tl.uint16).to(tl.bfloat16,bitcast=True)
    product=(raw.to(tl.float32)*multiplier.to(tl.float32)).to(tl.bfloat16)
    return tl.where(s<=134,product,fallback)


@triton.jit
def _gemm(X,W,S,Y,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,G:tl.constexpr,XS:tl.constexpr,YS:tl.constexpr,
          SPLIT:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,FAST:tl.constexpr):
    nt=tl.program_id(0)
    mt=tl.program_id(1)
    g=tl.program_id(2)//SPLIT
    sk=tl.program_id(2)%SPLIT
    rows=mt*BM+tl.arange(0,BM)
    cols=nt*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    tiles=tl.cdiv(K,BK)
    start=tiles*sk//SPLIT
    end=tiles*(sk+1)//SPLIT
    for tile in range(start,end):
        k=tile*BK+kk
        a=tl.load(X+rows[:,None]*XS+g*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),other=0)
        b=tl.load(W+(g*N+cols[:,None])*K+k[None,:],(cols[:,None]<N)&(k[None,:]<K),other=0)
        s=tl.load(S+((g*N+cols[:,None])//32)*(K//32)+k[None,:]//32,(cols[:,None]<N)&(k[None,:]<K),other=127)
        w=_decode(b,s,FAST)
        acc=tl.dot(a,tl.trans(w),acc)
    if SPLIT>1:
        tl.store(P+((sk*M+rows[:,None])*G+g)*N+cols[None,:],acc,(rows[:,None]<M)&(cols[None,:]<N))
    else:
        tl.store(Y+rows[:,None]*YS+g*N+cols[None,:],acc,(rows[:,None]<M)&(cols[None,:]<N))


@triton.jit
def _reduce(P,Y,M:tl.constexpr,N:tl.constexpr,G:tl.constexpr,YS:tl.constexpr,SPLIT:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    count=M*N*G
    acc=tl.load(P+i,i<count,other=0)
    for s in range(1,SPLIT):
        acc+=tl.load(P+s*count+i,i<count,other=0)
    tl.store(Y+(i//(N*G))*YS+i%(N*G),acc,i<count)


def linear_into(x:torch.Tensor,w:torch.Tensor,s:torch.Tensor,out:torch.Tensor,scratch:torch.Tensor,*,
                bm:int,bn:int,bk:int,splits:int,groups:int=1,warps:int=4,stages:int=3,fast_scales:bool=False)->None:
    m=x.shape[0]
    n=w.shape[0]//groups
    k=w.shape[1]
    _gemm[(triton.cdiv(n,bn),triton.cdiv(m,bm),groups*splits)](
        x,w,s,out,scratch,m,n,k,groups,x.stride(0),out.stride(0),splits,bm,bn,bk,fast_scales,
        num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    if splits>1:
        _reduce[(triton.cdiv(m*n*groups,256),)](scratch,out,m,n,groups,out.stride(0),splits,256)
