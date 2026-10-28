"""Direct register-fed SM86 MMA adapter, isolated from all engine dependencies."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol, cast

import torch

from . import _cuda
from .reference import admit


class Extension(Protocol):
    def prepack(self, w: int, s: int, source: int, scales: int, n: int, k: int, stream: int) -> None: ...
    def schedule(self, ids: int, schedule: int, slots: int, experts: int, status: int, stream: int) -> None: ...
    def linear(self, x: int, ids: int, schedule: int, w: int, s: int, up: int, us: int, out: int,
               uout: int, n: int, k: int, slots: int, experts: int, top: int, per_slot: bool,
               fused: bool, stream: int) -> None: ...
    def test_decode(self, codes: int, scales: int, out: int, count: int, stream: int) -> None: ...


@lru_cache(maxsize=1)
def extension() -> Extension:
    from torch.utils.cpp_extension import load
    return cast(Extension, load(
        name='kx_mxfp4_regfed', sources=[str(Path(__file__).with_name('regfed.cu'))],
        extra_cuda_cflags=['-O3','-gencode=arch=compute_86,code=sm_86','--fmad=false','-Xptxas=-v'],
        extra_cflags=['-O2'], with_cuda=True, verbose=True))


@dataclass(frozen=True)
class RegisterExperts:
    codes: torch.Tensor
    scales: torch.Tensor
    experts: int
    outputs: int
    inputs: int
    ext: Extension

    @classmethod
    def from_checkpoint(cls, codes: torch.Tensor, scales: torch.Tensor,
                        device: torch.device) -> RegisterExperts:
        admit(codes,scales)
        e,n,half_k = codes.shape
        if n%16 or (half_k*2)%64:
            raise ValueError('register layout requires N%16=0 and K%64=0; use masked Triton for padding')
        if int(scales.min()) == 0:
            raise ValueError('register scale multiplier excludes E8M0 zero; use integer decoder')
        if device.type != 'cuda' or device.index is None or torch.cuda.get_device_capability(device) != (8,6):
            raise ValueError('explicit SM86 CUDA device required')
        ext = extension()  # prebuild before any capture; module has no static GPU state
        with torch.cuda.device(device):
            dst = torch.empty((e,n,half_k),dtype=torch.uint8,device=device)
            ds = torch.empty(tuple(scales.shape),dtype=torch.uint8,device=device)
            stream = torch.cuda.current_stream(device).cuda_stream
            for i in range(e):
                # One expert staging allocation only; stream directly into its final shard.
                p,s = codes[i].contiguous().to(device),scales[i].contiguous().to(device)
                ext.prepack(dst[i].data_ptr(),ds[i].data_ptr(),p.data_ptr(),s.data_ptr(),n,half_k*2,stream)
                del p,s  # same-stream allocator reuse follows the enqueued prepack
        return cls(dst,ds,e,n,half_k*2,ext)

    def linear_into(self, x: torch.Tensor, ids: torch.Tensor, out: torch.Tensor,
                    status: torch.Tensor, schedule: torch.Tensor, *, per_slot: bool=False,
                    up: RegisterExperts | None=None, up_out: torch.Tensor | None=None) -> None:
        if ids.ndim != 2 or ids.shape[0]<1 or not 1<=ids.shape[1]<=3:
            raise ValueError('expected nonempty [rows,top<=3] int32 IDs')
        rows,top = ids.shape
        if rows*top > 64:
            raise ValueError('register kernel supports at most 64 routed slots per launch')
        d = self.codes.device
        _cuda(ids,d,torch.int32,(rows,top))
        _cuda(x,d,torch.bfloat16,(rows*top if per_slot else rows,self.inputs))
        _cuda(out,d,torch.float32,(rows,top,self.outputs))
        _cuda(status,d,torch.int32,(1,))
        _cuda(schedule,d,torch.int32,(rows*top,8))
        if up is not None:
            if (up.inputs,up.outputs,up.experts,up.codes.device)!=(self.inputs,self.outputs,self.experts,d):
                raise ValueError('gate/up dimensions and devices must match')
            if up_out is None:
                raise ValueError('fused gate/up requires up_out')
            _cuda(up_out,d,torch.float32,tuple(out.shape))
        elif up_out is not None:
            raise ValueError('up_out without up bank')
        with torch.cuda.device(d):
            stream = torch.cuda.current_stream(d).cuda_stream
            self.ext.schedule(ids.data_ptr(),schedule.data_ptr(),rows*top,self.experts,status.data_ptr(),stream)
            self.ext.linear(x.data_ptr(),ids.data_ptr(),schedule.data_ptr(),self.codes.data_ptr(),self.scales.data_ptr(),
                            0 if up is None else up.codes.data_ptr(),0 if up is None else up.scales.data_ptr(),
                            out.data_ptr(),0 if up_out is None else up_out.data_ptr(),self.outputs,self.inputs,
                            rows*top,self.experts,top,per_slot,up is not None,stream)
