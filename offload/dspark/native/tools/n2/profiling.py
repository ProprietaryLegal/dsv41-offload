"""Full kernel names and corrected native categories (legacy 'gemm' matched wo_a)."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable

import torch

from ....worker.tools.tp2_bench import cat2


def category(name: str) -> str:
    if '_shared_activation' in name or name == '_combine':
        return 'native_epilogues'
    if any(key in name for key in ('gate_up', 'void down', 'linear<true>', 'linear<false>', 'make_schedule')):
        return 'native_experts'
    if any(key in name for key in ('dense_kernel', 'split_reduce', 'linear_k<')) or name.startswith('void reduce_kernel<'):
        return 'native_fp8_dense'
    if 'ampere' in name or 'cutlass::' in name or ('cublasLt::' in name and '__nv_bfloat16' in name):
        return 'native_bf16_dense'
    return cat2(name)


def breakdown(fn: Callable[[], object], reps: int = 20) -> dict[str, object]:
    from torch.profiler import ProfilerActivity, profile

    fn()
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
    kernels: dict[int, Counter[str]] = defaultdict(Counter)
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernels[event.device_index][event.name] += event.device_time_total / reps
    result: dict[str, object] = {}
    for dev, times in sorted(kernels.items()):
        cats: Counter[str] = Counter()
        for name, elapsed in times.items():
            cats[category(name)] += elapsed
        result[f'device{dev}'] = {
            'kernel_us_per_step': sum(times.values()),
            'us': dict(cats.most_common()),
            'kernels_us': dict(times.most_common()),
        }
    return result
