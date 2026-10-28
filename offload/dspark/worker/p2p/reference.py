"""Eager peer-copy oracle: PyTorch copies/events, no raw peer pointers or spin protocol.

Debug only, outside graph capture. FP32 addition then optional BF16 rounding matches the push collective.
"""
from __future__ import annotations

import torch


def all_reduce(parts: tuple[torch.Tensor, torch.Tensor], dtype: torch.dtype = torch.bfloat16
               ) -> tuple[torch.Tensor, torch.Tensor]:
    if any(x.dtype != torch.float32 or not x.is_cuda for x in parts) or parts[0].shape != parts[1].shape:
        raise ValueError("peer-copy reference needs two same-shaped FP32 CUDA tensors")
    ready = []
    for x in parts:
        with torch.cuda.device(x.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("peer-copy reference is eager only")
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(x.device))
            ready.append(event)
    result = []
    for r, x in enumerate(parts):
        with torch.cuda.device(x.device):
            torch.cuda.current_stream(x.device).wait_event(ready[1 - r])
            peer = parts[1 - r].to(x.device, non_blocking=True)
            result.append((x + peer).to(dtype))
    return result[0], result[1]
