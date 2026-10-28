"""Validate startup receipt against the independently captured stable proc census."""
from __future__ import annotations

import json
from typing import cast

from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, record
from offload.up3.rdma_pin import COMM, PinReceipt

PREFIX = 'proven RDMA poller identity requires '


def decode(text: str) -> dict[str, Json]:
    def unique(pairs: list[tuple[str, Json]]) -> dict[str, Json]:
        result: dict[str, Json] = {}
        for key, value in pairs:
            ck.require(key not in result, PREFIX + 'unique receipt fields')
            result[key] = value
        return result
    return record(cast(Json, json.loads(text, object_pairs_hook=unique)))


def validate(process: dict[str, Json], threads: list[dict[str, Json]]) -> int:
    """Return the sole proven poller TID; no name-only or affinity-only exception."""
    receipt = record(process.get('pin_receipt', {}))
    ck.require(set(receipt) == set(PinReceipt.__annotations__), PREFIX + 'a complete startup receipt')
    ck.require(receipt['schema'] == 'up3-rdma-pin-v1' and receipt['clock'] == 'CLOCK_MONOTONIC'
        and type(receipt['monotonic_ns']) is int and receipt['monotonic_ns'] > 0,
        PREFIX + 'versioned CLOCK_MONOTONIC evidence')
    pid, tid = receipt['pid'], receipt['tid']
    ck.require(type(pid) is int and type(tid) is int and pid == process['pid'] and pid != tid
        and pid > 0 and tid > 0, PREFIX + 'matching PID and nonleader TID')
    leader = next((row for row in threads if row.get('tid') == pid), {})
    pollers = [row for row in threads if row.get('comm') == COMM]
    ck.require(len(pollers) == 1 and pollers[0].get('tid') == tid,
        PREFIX + 'exactly one named thread matching the receipt TID')
    poller = pollers[0]
    for key, row in (('pid_starttime', leader), ('tid_starttime', poller)):
        start = receipt[key]
        ck.require(isinstance(start, str) and start.isascii() and start.isdigit()
            and start == row.get('starttime'), PREFIX + 'current process/thread starttimes (no reuse)')
    old = receipt['old_comm']
    ck.require(receipt['new_comm'] == COMM and isinstance(old, str) and bool(old.strip())
        and old != COMM and len(old.encode()) <= 15, PREFIX + 'verified old/new comm')
    for affinity in (receipt['affinity'], poller.get('cpus')):
        ck.require(isinstance(affinity, list) and len(affinity) == 1
            and type(affinity[0]) is int and affinity[0] == 14, PREFIX + 'receipt AND census affinity exactly [14]')
    return cast(int, tid)
