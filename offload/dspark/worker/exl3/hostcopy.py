# SPDX-License-Identifier: Apache-2.0
"""Host-to-device copy for checkpoint tensors that may be views of a safetensors mmap (torch only).

Two GB10 traps, both hit while loading v2.1 (Oct 9):
1. vLLM's lazy safetensors loader hands out views of each file mapped private and writable, and DeepSeek-V4.1's VL
   wrapper sorts (so holds) every view until the load ends. A host-to-device copy_ straight from those pages turns
   them into anonymous memory (copy-on-write) that stays until the load returns: ~30 GiB per rank, enough to swap both
   Sparks and get a worker killed. Measured with scripts/tests/loadmem.py: 20.2 GiB of rank-half expert trellis copied
   grew anonymous memory by 6.7 GiB (the contiguous pieces), by nothing when the pages were only read on the CPU.
2. A pageable host-to-device copy needs driver-side staging memory, and on GB10 the driver cannot reclaim page cache:
   with free memory eaten by the weight files' cache, one copy spun in user mode for 35 minutes (driver log: 109
   NV_ERR_NO_MEMORY from _memdescAllocInternal that day).
So every load copy goes CPU -> one pinned staging buffer (allocated on first use, while memory is still free, grown
on demand, reused) -> device: the CPU memcpy only reads the mapped pages, and the DMA source is already pinned.
"""

from __future__ import annotations

import torch

_STAGE: torch.Tensor | None = None
_STAGE_MIN = 64 << 20


def _stage(nbytes: int) -> torch.Tensor:
    global _STAGE
    if _STAGE is None or _STAGE.numel() < nbytes:
        _STAGE = None
        _STAGE = torch.empty(max(nbytes, _STAGE_MIN), dtype=torch.uint8, pin_memory=True)
    return _STAGE


def copy_from_host(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """dst.copy_(src); a CPU src goes through the pinned staging buffer (synchronous, so the buffer is free again)."""
    if src.device.type == "cpu" and dst.device.type != "cpu":
        nbytes = src.numel() * src.element_size()
        if nbytes == 0:
            return dst
        buf = _stage(nbytes)[:nbytes].view(src.dtype).view(src.shape)
        buf.copy_(src)
        return dst.copy_(buf)
    return dst.copy_(src)


def release_stage() -> None:
    """Drop the staging buffer once the load is done (it is only needed while weights stream in)."""
    global _STAGE
    _STAGE = None
