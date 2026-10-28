"""Load-time prepack/admission and allocation-free, current-stream dense calls."""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import os
from functools import lru_cache
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
import torch


def require_lock() -> None:
    """Qualification-only guard: verify the inherited exclusive flock, not a flag."""
    locks = {'GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc': 'gpu-7744b44b.lock',
             'GPU-bc62df2a-59cf-9828-7d43-558d1355bab7': 'gpu-bc62df2a.lock'}
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible not in locks:
        raise RuntimeError('qualification must pin the development UUID')
    lock=Path.home()/'dsv41-offload/locks'/locks[visible]
    fd=int(os.environ.get('KX_FP8DENSE_LOCK_FD','9'))
    held=os.fstat(fd)
    actual=lock.stat()
    if (held.st_ino,held.st_dev)!=(actual.st_ino,actual.st_dev):
        raise RuntimeError('inherited descriptor is not the GPU lock')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with lock.open('a') as probe:
        try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
    raise RuntimeError('exclusive GPU lock was not held')


class Extension(Protocol):
    def simt(self, x: int, w: int, s: int, y: int, m: int, n: int, k: int,
             xs: int, ys: int, groups: int, f32: bool, stream: int) -> None: ...
    def prepack(self, dst: int, src: int, n: int, k: int, stream: int) -> None: ...
    def linear(self, x: int, w: int, s: int, y: int, scratch: int, m: int, n: int, k: int,
               xs: int, ys: int, groups: int, splits: int, rows: int, f32: bool, fast: bool, stream: int) -> None: ...
    def test_decode(self, codes: int, scales: int, out: int, count: int, stream: int) -> None: ...


@lru_cache(maxsize=1)
def extension() -> Extension:
    from torch.utils.cpp_extension import load
    return cast(Extension, load(
        name='kx_fp8dense', sources=[str(Path(__file__).with_name('dense.cu'))],
        extra_cuda_cflags=['-O3', '-gencode=arch=compute_86,code=sm_86', '--fmad=false', '-Xptxas=-v'],
        extra_cflags=['-O2'], with_cuda=True, verbose=True))


def domain_histogram(codes: torch.Tensor, scales: torch.Tensor) -> NDArray[np.int64]:
    """CPU load admission; return exact present (scale,code) counts, <=32-row temporary."""
    if codes.device.type != 'cpu' or scales.device.type != 'cpu' or codes.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError('load admission requires CPU uint8 codes/scales')
    if codes.ndim != 3 or scales.shape != (codes.shape[0], codes.shape[1] // 32, codes.shape[2] // 32):
        raise ValueError('codes [G,N,K] and scales [G,N/32,K/32] required')
    c, s = codes.numpy(), scales.numpy()
    hist = np.zeros(65536, dtype=np.int64)
    for group in range(c.shape[0]):
        for row in range(0, c.shape[1], 32):
            block = c[group, row:row+32].reshape(32, -1, 32).astype(np.int32)
            pairs = (s[group, row//32].astype(np.int32)[None, :, None] << 8) | block
            hist += np.bincount(pairs.ravel(), minlength=65536)
    used = np.flatnonzero(hist)
    cc, ss = used & 255, used >> 8
    mag = cc & 127
    if np.any(mag == 127) or np.any(ss == 255):
        raise ValueError('NaN E4M3/E8M0 value in checkpoint')
    exponent, mantissa = (mag >> 3), mag & 7
    values = np.where(exponent == 0, mantissa/512.0, np.ldexp(1.0+mantissa/8.0, exponent-7))
    values = np.ldexp(values, ss-127)
    nonzero = values != 0
    # Register exponent addition is exact for every normal BF16 weight. Refuse
    # the subnormal/overflow domains rather than silently flushing or wrapping.
    if np.any(nonzero & ((values < 2.0**-126) | (values > float(torch.finfo(torch.bfloat16).max)))):
        raise ValueError('decoded weight outside finite normal BF16 domain')
    return hist


def _matrix(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    if tensor.device != device or tensor.dtype != dtype or tuple(tensor.shape) != shape or not tensor.is_contiguous():
        raise ValueError(f'expected contiguous {dtype} {shape} on {device}, got {tensor.dtype} {tuple(tensor.shape)} {tensor.device}')


@dataclass(frozen=True)
class DenseFP8:
    codes: torch.Tensor
    scales: torch.Tensor
    groups: int
    outputs: int
    inputs: int
    ext: Extension
    fast_scales: bool
    layout: str

    @classmethod
    def from_checkpoint(cls, codes: torch.Tensor, scales: torch.Tensor, device: torch.device, *,
                        groups: int = 1, row_offset: int = 0, col_offset: int = 0, layout: str = "register") -> DenseFP8:
        """codes are a [G*N,K] rank shard; scales are the FULL checkpoint grid.

        row_offset/col_offset locate the shard in that full grid. Load once,
        outside capture. G=4 implements wo_a's four block-diagonal groups.
        Source bytes need not be contiguous. Only one temporary device shard
        is retained during prepack, never a decoded-weight bank.
        """
        if codes.ndim != 2 or groups not in (1, 4) or codes.shape[0] % (32*groups) or codes.shape[1] % 32:
            raise ValueError('FP8 codes [G*N,K], G=1/4, N/K positive multiples of 32 required')
        if min(codes.shape) <= 0 or row_offset < 0 or col_offset < 0 or row_offset % 32 or col_offset % 32:
            raise ValueError('positive shapes and nonnegative block-aligned shard offsets required')
        n, k = codes.shape[0]//groups, codes.shape[1]
        if scales.ndim != 2 or scales.shape[0] < (row_offset+groups*n)//32 or scales.shape[1] < (col_offset+k)//32:
            raise ValueError('full tensor scale grid does not cover the shard')
        sc = scales[row_offset//32:(row_offset+groups*n)//32, col_offset//32:(col_offset+k)//32].contiguous()
        domain_histogram(codes.reshape(groups, n, k), sc.view(groups, n//32, k//32))
        if device.type != 'cuda' or device.index is None or torch.cuda.get_device_capability(device) != (8, 6):
            raise ValueError('explicit SM86 CUDA device required')
        if layout not in ('register', 'simt'):
            raise ValueError('layout must be register or simt')
        fast_scales=bool(int(sc.max())<=134)
        if layout=='simt' and (not fast_scales or k%128):
            raise ValueError('SIMT requires scales<=134 and K%128=0')
        ext = extension()  # compile before capture; no device-global caches/state
        with torch.cuda.device(device):
            ds = sc.to(device)
            source = codes.contiguous().to(device)
            dst = torch.empty((groups, n, k), dtype=torch.uint8, device=device) if layout=='register' else source.view(groups,n,k)
            st = torch.cuda.current_stream(device).cuda_stream
            if layout=='register':
                for group in range(groups):
                    ext.prepack(dst[group].data_ptr(), source[group*n:(group+1)*n].data_ptr(), n, k, st)
            else:
                dst=source.view(groups,n,k)
        return cls(dst, ds, groups, n, k, ext, fast_scales, layout)

    def configuration(self, rows: int) -> tuple[int, int]:
        """Offline-measured (split count, M/8 register tiles), no autotuning.

        Callers allocate [splits,M,G,N] FP32 scratch when splits>1 and pass
        both values explicitly to linear_into. Unlisted small test shapes
        use the no-scratch generic kernel.
        """
        if not 1<=rows<=128:
            raise ValueError('rows must be 1..128')
        if self.layout=='simt':
            return 1,1
        n,k=self.outputs,self.inputs
        if k==15360 and n==5120:
            return (8,2) if rows<=32 else (2,16)
        if k==5120 and n in (512,1152,1280,2304):
            return 16,1
        if (n,k)==(16384,1280):
            return 1,1
        if k in (4096,1152) and n in (1024,5120):
            return (8,1) if rows<=8 else (4,1)
        return 1,min(16,1<<((rows-1)//8).bit_length())

    def linear_into(self, x: torch.Tensor, out: torch.Tensor, *, scratch: torch.Tensor | None = None,
                    splits: int = 1, row_tiles: int | None = None) -> None:
        """x already QDQ'd, BF16 [M,G*K]; out BF16/FP32 [M,G*N].

        No allocation, sync, host tensor read, or lazy build. FP32 split-K
        scratch is [splits,M,G,N]; reduction is fixed-order, with one final RNE.
        Rows may be strided, columns must be contiguous. For the measured
        dispatch, call configuration(M) and pass both values explicitly;
        split=1 is the generic control requiring no scratch.
        """
        if x.ndim != 2 or not 1 <= x.shape[0] <= 128 or x.shape[1] != self.groups*self.inputs:
            raise ValueError('x must be [1..128,G*K]')
        m, d = x.shape[0], self.codes.device
        if x.device != d or x.dtype != torch.bfloat16 or x.stride(1) != 1 or x.stride(0) < x.shape[1] or x.stride(0) % 2 or x.data_ptr() % 4:
            raise ValueError('BF16 input on weight device with contiguous columns required')
        if out.shape != (m, self.groups*self.outputs) or out.device != d or out.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError('output shape/device/dtype mismatch')
        if out.stride(1) != 1 or out.stride(0) < out.shape[1]:
            raise ValueError('output columns must be contiguous and rows nonoverlapping')
        if splits not in range(1, 33) or splits > self.inputs//16:
            raise ValueError('splits must be 1..32 and no greater than K/16')
        rows = row_tiles if row_tiles is not None else min(16, 1 << ((m-1)//8).bit_length())
        if rows not in (1, 2, 4, 8, 16):
            raise ValueError('row_tiles must be 1,2,4,8,16')
        if splits > 1:
            if scratch is None:
                raise ValueError('split-K requires caller-owned scratch')
            _matrix(scratch, d, torch.float32, (splits, m, self.groups, self.outputs))
        for buf in (out, scratch):
            if buf is not None and buf.untyped_storage().data_ptr() in (x.untyped_storage().data_ptr(), self.codes.untyped_storage().data_ptr(), self.scales.untyped_storage().data_ptr()):
                raise ValueError('output/scratch must not alias inputs or weights')
        if scratch is not None and scratch.untyped_storage().data_ptr() == out.untyped_storage().data_ptr():
            raise ValueError('output/scratch must not alias')
        with torch.cuda.device(d):
            if self.layout=='simt':
                if splits!=1 or m>20 or x.data_ptr()%8 or x.stride(0)%4:
                    raise ValueError('SIMT needs M<=20, splits=1 and 8-byte aligned rows')
                self.ext.simt(x.data_ptr(),self.codes.data_ptr(),self.scales.data_ptr(),out.data_ptr(),m,
                              self.outputs,self.inputs,x.stride(0),out.stride(0),self.groups,
                              out.dtype==torch.float32,torch.cuda.current_stream(d).cuda_stream)
                return
            self.ext.linear(x.data_ptr(), self.codes.data_ptr(), self.scales.data_ptr(), out.data_ptr(),
                            0 if scratch is None else scratch.data_ptr(), m, self.outputs, self.inputs,
                            x.stride(0), out.stride(0), self.groups, splits, rows,
                            out.dtype == torch.float32, self.fast_scales, torch.cuda.current_stream(d).cuda_stream)
