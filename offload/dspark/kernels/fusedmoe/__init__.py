"""Allocation-free native packed expert pipeline; integration remains with N1."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol, cast

import torch
from offload.dspark.kernels.mxfp4.regfed import RegisterExperts


class Extension(Protocol):
    def schedule(self, ids: int, dst: int, status: int, slots: int, experts: int, i64: bool, group: int, stream: int) -> None: ...
    def launch(self, x: int, rw: int, sched: int, w: int, s: int, up: int, us: int, w2: int, s2: int,
               h: int, out: int, gout: int, uout: int, n: int, k: int, slots: int, out_stride: int,
               warps: int, stream: int) -> None: ...


@lru_cache(maxsize=1)
def extension() -> Extension:
    from torch.utils.cpp_extension import load
    return cast(Extension, load(name='kx_fusedmoe', sources=[str(Path(__file__).with_name('fusedmoe.cu'))],
        extra_cuda_cflags=['-O3', '-gencode=arch=compute_86,code=sm_86', '--fmad=false', '-Xptxas=-v'],
        extra_cflags=['-O2'], with_cuda=True, verbose=True))


def _tensor(t: torch.Tensor, d: torch.device, dtype: torch.dtype, shape: tuple[int, ...], *,
            contiguous: bool = True) -> None:
    if t.device != d or t.dtype != dtype or tuple(t.shape) != shape or (contiguous and not t.is_contiguous()):
        raise ValueError(f'expected {dtype} {shape} on {d}, contiguous={contiguous}; got {t.dtype} {t.shape} {t.device}')


@dataclass(frozen=True)
class Workspace:
    schedule: torch.Tensor
    hq: torch.Tensor

    @classmethod
    def allocate(cls, rows: int, intermediate: int, device: torch.device) -> Workspace:
        """Allocate before capture. The workspace is exclusively owned by its stream."""
        if not 1 <= rows <= 20 or intermediate % 32:
            raise ValueError('workspace requires 1..20 rows and 32-aligned intermediate')
        return cls(torch.empty(1 + rows*3*17, dtype=torch.int32, device=device),
                   torch.empty(rows*3, intermediate, dtype=torch.bfloat16, device=device))


VARIANTS: dict[int, str] = {
    4: '32 channels/warp, 4 warps/CTA', 8: '32 channels/warp, 8 warps/CTA',
    16: '16 channels/warp, shared amax, 4 warps/CTA',
    32: '16 channels/warp, shared amax, 8 warps/CTA',
    64: '16 channels/warp, compensated FP32, 4 warps/CTA',
    128: '16 channels/warp, compensated FP32, 8 warps/CTA',
    256: 'CTA split-K2, ordinary FP32', 512: 'CTA split-K4, ordinary FP32',
    1024: 'CTA split-K4, compensated FP32', 2048: 'CTA split-K2, compensated FP32',
    4096: 'CTA split-K2, K16 compensated FP32', 8192: 'CTA split-K4, K16 compensated FP32',
    16384: 'CTA split-K8, two output warps, K16 compensated FP32',
    65536: 'CTA split-K2, group16 reuse, ordinary FP32',
    131072: 'CTA split-K2, K64 compensated FP32 with high/low join',
}


@dataclass(frozen=True)
class FusedMoE:
    gate: RegisterExperts
    up: RegisterExperts
    down: RegisterExperts
    ext: Extension
    variant: int = 131072

    @classmethod
    def from_register_banks(cls, gate: RegisterExperts, up: RegisterExperts, down: RegisterExperts,
                            *, variant: int = 131072) -> FusedMoE:
        """Reuse existing prepacked KX banks without copies or changed weight values."""
        if (gate.experts, gate.outputs, gate.inputs, gate.codes.device) != (
                up.experts, up.outputs, up.inputs, up.codes.device):
            raise ValueError('gate/up dimensions and devices must match')
        if (down.experts, down.inputs, down.outputs, down.codes.device) != (
                gate.experts, gate.outputs, gate.inputs, gate.codes.device):
            raise ValueError('down must be the gate/up transposed shape on the same device')
        if gate.outputs % 64 or gate.inputs % 64 or variant not in (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536, 131072):
            raise ValueError('64-aligned dimensions and a supported variant required')
        if gate.codes.device.type != 'cuda' or torch.cuda.get_device_capability(gate.codes.device) != (8, 6):
            raise ValueError('SM86 CUDA weights required')
        for bank in (gate, up, down):
            _tensor(bank.codes, gate.codes.device, torch.uint8, (bank.experts, bank.outputs, bank.inputs//2))
            _tensor(bank.scales, gate.codes.device, torch.uint8, (bank.experts, bank.outputs, bank.inputs//32))
        return cls(gate, up, down, extension(), variant)

    def forward_into(self, xq: torch.Tensor, ids: torch.Tensor, rw: torch.Tensor, out: torch.Tensor,
                     status: torch.Tensor, workspace: Workspace, *,
                     projections: tuple[torch.Tensor, torch.Tensor] | None = None) -> None:
        """xq -> FP32 SLOT partials; no allocation, host synchronization, or CPU ID read.

        Optional FP32 projection buffers expose the exact MMA sums for qualification.
        They add stores and must be omitted from performance runs.
        """
        d = self.gate.codes.device
        rows = xq.shape[0]
        n, k = self.gate.outputs, self.gate.inputs
        if not 1 <= rows <= 20 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('1..20 rows and int32/int64 route IDs required')
        _tensor(xq, d, torch.bfloat16, (rows, k))
        _tensor(ids, d, ids.dtype, (rows, 3))
        _tensor(rw, d, torch.float32, (rows, 3))
        _tensor(out, d, torch.float32, (rows, 3, k), contiguous=False)
        if out.stride(1) != k or out.stride(2) != 1 or out.stride(0) < 3*k:
            raise ValueError('SLOT output needs contiguous channels/slots and nonoverlapping rows')
        _tensor(status, d, torch.int32, (1,))
        _tensor(workspace.schedule, d, torch.int32, (1+rows*3*17,))
        _tensor(workspace.hq, d, torch.bfloat16, (rows*3, n))
        if projections is not None:
            for p in projections:
                _tensor(p, d, torch.float32, (rows, 3, n))
        with torch.cuda.device(d):
            st = torch.cuda.current_stream(d).cuda_stream
            self.ext.schedule(ids.data_ptr(), workspace.schedule.data_ptr(), status.data_ptr(), rows*3,
                              self.gate.experts, ids.dtype == torch.int64, 16 if self.variant == 65536 else 8, st)
            self.ext.launch(xq.data_ptr(), rw.data_ptr(), workspace.schedule.data_ptr(), self.gate.codes.data_ptr(),
                self.gate.scales.data_ptr(), self.up.codes.data_ptr(), self.up.scales.data_ptr(),
                self.down.codes.data_ptr(), self.down.scales.data_ptr(), workspace.hq.data_ptr(), out.data_ptr(),
                0 if projections is None else projections[0].data_ptr(),
                0 if projections is None else projections[1].data_ptr(), n, k, rows*3, out.stride(0), self.variant, st)
def select_variant(stage: int, rows: int) -> int:
    """Static measured profile; choose before capture, without reading device IDs.

    CARRY is numerically qualified on every stage/rank. Plain K2 is qualified
    on stage1 and is faster at the measured ten-row shape. Other shapes retain
    CARRY. This profile does not meet all six latency targets; see RESULTS.md.
    """
    if stage not in (0, 1, 2) or not 1 <= rows <= 20:
        raise ValueError('expected stage 0..2 and rows 1..20')
    return 256 if stage == 1 and rows == 10 else 131072
