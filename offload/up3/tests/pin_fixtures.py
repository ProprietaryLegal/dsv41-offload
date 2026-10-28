"""Synthetic startup proofs, explicitly separate from captured production shapes."""
from __future__ import annotations

from typing import cast

from offload.prefill_maxlen.evidence import Json
from offload.up3.rdma_pin import COMM

CPUS: list[Json] = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]


def receipt(pid: int, tid: int, pid_start: str = '12', tid_start: str = '13') -> dict[str, Json]:
    return {'schema': 'up3-rdma-pin-v1', 'pid': pid, 'pid_starttime': pid_start,
        'tid': tid, 'tid_starttime': tid_start, 'old_comm': 'VLLM::Worker_TP', 'new_comm': COMM,
        'affinity': [14], 'clock': 'CLOCK_MONOTONIC', 'monotonic_ns': 123456789}


def evidence(rank: int, container: str = 'fixture') -> dict[str, Json]:
    worker: dict[str, Json] = {'pid': 101, 'role': 'worker', 'status': 'OBSERVED',
        'pin_receipt': receipt(101, 1101), 'threads': [
            {'tid': 101, 'comm': 'VLLM::Worker_TP', 'starttime': '12', 'cpus': CPUS},
            {'tid': 1101, 'comm': COMM, 'starttime': '13', 'cpus': [14]}]}
    proxy: dict[str, Json] = {'pid': 102, 'role': 'proxy', 'status': 'OBSERVED',
        'threads': [{'tid': 102, 'comm': 'python3', 'starttime': '12', 'cpus': CPUS}]}
    return {'arm': 'cpu-pin', 'rank': rank, 'container': container, 'affinity': {
        'pinned_pids': [101], 'proxy_pids': [102] if rank == 0 else [],
        'processes': cast(Json, [worker, proxy] if rank == 0 else [worker])}}
