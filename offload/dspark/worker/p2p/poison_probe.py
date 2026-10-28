"""Deliberately fault ONLY this offline process, then verify that TP2 refuses recovery.

Run via run_owned.sh with DSPARK_ALLOW_POISON_PROBE=RX-TP2SAFE; never run in a worker/service process.
Expected exit is 3, matching the worker's fatal restart status, after printing POISON_REFUSED.
"""
from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from ..tp2 import P2PComm, TP2Backend


@triton.jit
def _intentional_null_store():
    ptr = tl.full((1,), 0, tl.uint64).to(tl.pointer_type(tl.int32))
    tl.store(ptr, tl.full((1,), 1, tl.int32))


def main() -> None:
    if os.environ.get("DSPARK_ALLOW_POISON_PROBE") != "RX-TP2SAFE" or torch.cuda.device_count() != 2:
        raise RuntimeError("poison probe requires the explicit offline pair-test switch")
    devs = [torch.device("cuda", i) for i in (0, 1)]
    comm = P2PComm(devs, 4 * 5120, 129280)
    try:
        with torch.cuda.device(devs[0]):
            _intentional_null_store[(1,)]()
            torch.cuda.synchronize(devs[0])
    except RuntimeError as error:
        print(json.dumps({"injected_cuda_fault": str(error)}), flush=True)
    else:
        raise AssertionError("intentional null store did not fault")
    try:
        comm.pair.headers()
    except RuntimeError:
        assert comm.pair.poisoned is not None
    else:
        raise AssertionError("faulted context passed the idle/header check")
    reason = comm.pair.poisoned
    for op in (comm.pair.reset, comm.pair.headers):
        try:
            op()
        except RuntimeError as error:
            assert "restart worker" in str(error)
        else:
            raise AssertionError("poisoned context was reset or reused")
    be = TP2Backend.__new__(TP2Backend)
    be.m = SimpleNamespace(comm=comm, devs=devs)
    be.comm_errors_in_a_row = 0
    be.log = logging.getLogger("poison_probe")
    assert not be.healthy()
    assert comm.pair.poisoned == reason
    print(json.dumps({"result": "POISON_REFUSED", "healthy": False, "expected_exit": 3}), flush=True)
    raise SystemExit(3)


if __name__ == "__main__":
    main()
