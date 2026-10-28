"""Supervise the dedicated D2 GPU; checkpoint on signals or RAM/swap violations."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

from .safety import assert_gpu_lease, leased_gpu, power_logger_running


def resident_and_swap(pid: int) -> tuple[int, int]:
    resident, swap = 0, 0
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            resident = int(line.split()[1])*1024
        elif line.startswith("VmSwap:"):
            swap = int(line.split()[1])*1024
    return resident, swap


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    assert_gpu_lease()
    gpu = leased_gpu()
    if not power_logger_running():
        raise SystemExit("D2 GPU jobs require live dsv41-powerlog")
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        ap.error("missing trainer command")
    child = subprocess.Popen(command, pass_fds=(int(os.environ["D2_LOCK_FD"]),), start_new_session=True)
    terminated: float | None = None
    reason = ""
    def request_stop(signum: int, frame: object) -> None:
        nonlocal reason
        reason = f"supervisor signal {signum}"
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    while child.poll() is None:
        now = time.monotonic()
        if not reason and not power_logger_running():
            reason = "power logger absent"
        try:
            rss, swap = resident_and_swap(child.pid)
            if rss > 128*(1 << 30) or swap:
                reason = f"memory budget: RSS={rss}, swap={swap}"
        except FileNotFoundError:
            break
        if reason and terminated is None:
            print(reason+"; requesting boundary checkpoint", flush=True)
            os.killpg(child.pid, signal.SIGTERM)
            terminated = now
        if terminated is not None and now-terminated >= 45:
            os.killpg(child.pid, signal.SIGKILL)
            raise SystemExit("trainer failed to checkpoint/exit in 45 seconds")
        subprocess.run(["nvidia-smi", "--id="+gpu, "--query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used", "--format=csv,noheader"], check=True)
        time.sleep(5)
    code = child.wait()
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
