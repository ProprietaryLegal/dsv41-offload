"""CPU-pin experiment only: identify the sole native thread born at ring start.

Mounted as vllm_exl3_moe/up3_rdma_pin.py beside the patched rdma.py. No
torch/model imports. Never imported or mounted by control/production.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

COMM = 'exl3-rdma-prx'  # 13 ASCII bytes, below Linux TASK_COMM_LEN (16 incl NUL)
RECEIPTS = Path('/run/up3-rdma-pin')
CPU_MASK = '5-9,15-19'
WAIT_S = 2.0


class PinReceipt(TypedDict):
    schema: str
    pid: int
    pid_starttime: str
    tid: int
    tid_starttime: str
    old_comm: str
    new_comm: str
    affinity: list[int]
    clock: str
    monotonic_ns: int


def birth(path: Path) -> str:
    """Linux proc stat field22, allowing spaces/parentheses in comm."""
    fields = path.read_text().rpartition(') ')[2].split()
    if len(fields) < 20 or not fields[19].isdigit():
        raise RuntimeError('UP3 RDMA pin refuses malformed proc starttime')
    return fields[19]


def receipt_path(pid: int, starttime: str, root: Path = RECEIPTS) -> Path:
    if type(pid) is not int or pid <= 0 or not starttime.isdigit():
        raise ValueError('UP3 RDMA pin requires positive PID and decimal starttime')
    return root / f'{pid}-{starttime}.json'


def start_with_receipt(start: Callable[[], object], cpu: int, *, root: Path = RECEIPTS) -> PinReceipt:
    """Bracket native start, wait for its self-pin, name it, persist proof or raise.

    rdma.py:105 calls e.start; exl3_rdma.cu:566-570 creates std::thread
    synchronously. Its proxy_loop:267-272 runs sched_setaffinity asynchronously
    before polling. No inference/collective is issued here to force lazy setup.
    ring_for invokes this wrapper only when the first eligible eager call builds
    the ring. Missing initialization therefore leaves no receipt and refuses
    in the host gate after the existing greedy/OPS1 phases.
    """
    if os.environ.get('EXL3_CPU_PIN') != CPU_MASK or type(cpu) is not int or cpu != 14:
        raise RuntimeError('UP3 RDMA pin requires registered CPU-pin mask and EXL3_RDMA_CPU=14')
    pid = os.getpid()
    tasks = Path('/proc/self/task')
    before = {int(path.name) for path in tasks.iterdir()}
    if any((tasks / str(tid) / 'comm').read_text().strip() == COMM for tid in before):
        raise RuntimeError('UP3 RDMA pin refuses an existing named poller / second ring')
    pid_start = birth(Path('/proc/self/stat'))
    target = receipt_path(pid, pid_start, root)
    # Each fresh experiment container has a private writable /run (not a bind
    # mount). Never reuse a process receipt, even across a second ring attempt.
    root.mkdir(mode=0o700, parents=False, exist_ok=True)
    if root.is_symlink() or root.stat().st_uid != os.geteuid() or root.stat().st_mode & 0o077:
        raise RuntimeError('UP3 RDMA pin receipt directory is not private and owned')
    if target.exists() or target.is_symlink():
        raise RuntimeError('UP3 RDMA pin refuses an existing receipt')
    deadline = time.monotonic() + WAIT_S
    start()
    seen: set[int] = set()
    while True:
        after = {int(path.name) for path in tasks.iterdir()}
        seen.update(after - before)
        if before - after or len(seen) > 1:
            raise RuntimeError('UP3 RDMA pin requires exactly one new thread and a stable prior census')
        if time.monotonic() >= deadline:
            raise RuntimeError('UP3 RDMA pin timed out: zero candidates or native thread did not pin to CPU14')
        if len(seen) == 1:
            tid = next(iter(seen))
            if tid in after and os.sched_getaffinity(tid) == {cpu}:
                break
        time.sleep(0.005)
    task = tasks / str(tid)
    tid_start = birth(task / 'stat')
    old_comm = (task / 'comm').read_text().rstrip('\n')
    if tid == pid or not old_comm or old_comm == COMM:
        raise RuntimeError('UP3 RDMA pin refuses leader/ambiguous prior comm')
    # proc comm writes retain a supplied newline as part of the name. Reads
    # append their own newline; status Name would escape the extra one as \\n.
    (task / 'comm').write_text(COMM)
    if ((task / 'comm').read_text() != COMM + '\n' or os.sched_getaffinity(tid) != {cpu}
            or birth(task / 'stat') != tid_start or birth(Path('/proc/self/stat')) != pid_start
            or {int(path.name) for path in tasks.iterdir()} != before | {tid}):
        raise RuntimeError('UP3 RDMA pin rename/affinity/identity verification failed')
    receipt: PinReceipt = {'schema': 'up3-rdma-pin-v1', 'pid': pid, 'pid_starttime': pid_start,
        'tid': tid, 'tid_starttime': tid_start, 'old_comm': old_comm, 'new_comm': COMM,
        'affinity': [cpu], 'clock': 'CLOCK_MONOTONIC', 'monotonic_ns': time.monotonic_ns()}
    # Exclusive creation and fsync; truncated/interrupted JSON fails host parsing.
    # Any write/flush/close failure propagates through the overlay (no NCCL fallback).
    with target.open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, allow_nan=False, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    return receipt
