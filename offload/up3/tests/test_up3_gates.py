from __future__ import annotations

import copy
from typing import cast

import pytest

from offload.prefill_maxlen.evidence import Json, record
from offload.up3 import gates
from offload.up3.tests.test_up3_window import receipt


def validate(row: dict[str, Json], arm: str) -> None:
    gates.validate(row, arm=arm, rank=0, run='run', container='container', since=100, now=101, program_sha='a' * 64)


@pytest.mark.parametrize('arm', list(gates.CASES))
def test_gate_contract_rejects_stale_cpu_missing_or_failed(arm: str) -> None:
    good = receipt(arm, 0, 'run', 'container', 100, 'a' * 64)
    validate(good, arm)
    for key, value in (('scope', 'synthetic-only'), ('device_capability', [8, 6]), ('rank', 1), ('time', 99),
                       ('time', float('nan')), ('program_sha256', 'b' * 64), ('container', 'old')):
        row = copy.deepcopy(good)
        row[key] = cast(Json, value)
        with pytest.raises(ValueError):
            validate(row, arm)
    for name in gates.CASES[arm]:
        row = copy.deepcopy(good)
        record(row['cases']).pop(name)
        with pytest.raises(ValueError):
            validate(row, arm)
        row = copy.deepcopy(good)
        record(record(row['cases'])[name])['pass'] = False
        with pytest.raises(ValueError):
            validate(row, arm)


@pytest.mark.parametrize('arm,case,key,value', [
    ('split', 'm43', 'oracle', 'single-smoe'), ('split', 'm63', 'capacity', 63),
    ('split', 'm33', 'comparison', 'bf16-tolerance'), ('split', 'dispatch64', 'unchanged', False),
    ('split', 'teacher-forced-rounding', 'kl', .1),
    ('vor', 'indices', 'comparison', 'overlap'), ('vor', 'launch', 'exception_disabled', True),
    ('vor', 'launch', 'request_time_compile', True), ('vor', 'actual-groups', 'derived_from_loaded_engine', False),
    ('k5', 'five-proposals', 'proposals', 4), ('k5', 'b4-24-rows', 'rows', 20),
    ('dual-ring', 'gid-device1', 'gid_index', 0), ('dual-ring', 'traffic-device0', 'bytes_delta', 0),
    ('dual-ring', 'rank-log', 'line', 'EXL3 RDMA all-reduce: rank 1 of 2 on rocep1s0f1,roceP2p1s0f1'),
    ('cpu-pin', 'poller14', 'cpu', 5), ('cpu-pin', 'all-thread-affinities', 'every_tid', False)])
def test_exact_s17_gate_details(arm: str, case: str, key: str, value: Json) -> None:
    row = receipt(arm, 0, 'run', 'container', 100, 'a' * 64)
    record(record(row['cases'])[case])[key] = value
    with pytest.raises(ValueError):
        validate(row, arm)
