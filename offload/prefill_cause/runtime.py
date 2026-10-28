"""PF1-CAUSE helper (mounted as vllm/v1/worker/gpu/dsv41_prefill_cause.py): host-only allocator counters for a span row.

Only while a PF1-R2 trace chunk is open (prefill steps of an ON arm); otherwise an empty dict and no torch call. Reads
the caching allocator's own statistics (no device read, no synchronization, no allocation). Numbers only.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the same module as reviewed in this checkout (offload/prefill_trace/runtime.py)
    from offload.prefill_trace import runtime as _pf
else:
    from vllm.v1.worker.gpu import dsv41_prefill_trace as _pf

Scalar = str | int | float | bool | None
STATS = {'alloc_retries': 'num_alloc_retries', 'device_allocs': 'num_device_alloc', 'device_frees': 'num_device_free',
         'reserved_bytes': 'reserved_bytes.all.current', 'active_bytes': 'active_bytes.all.current',
         'segments': 'segment.all.current'}


def alloc(**extra: Scalar) -> dict[str, Scalar]:
    rec = _pf._recorder
    if not _pf.ENABLED or rec is None or rec.context is None:
        return {}
    import torch
    stats = torch.cuda.memory_stats()
    return {**extra, **{name: int(stats.get(key, -1)) for name, key in STATS.items()}}
