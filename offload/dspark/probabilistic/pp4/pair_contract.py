"""Physical PP4 admission contract. Solo/compile receipts can never satisfy it."""
from __future__ import annotations

import fcntl
import os
import re
from pathlib import Path

from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, record

SCHEMA = 'pp4-physical-pair-v1'
CASES = tuple((batch, depth, mode) for batch in range(1, 5) for depth in (4, 5)
              for mode in ('sampled', 'mixed-even', 'mixed-odd', 'greedy'))


def guard() -> None:
    ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator'
               and os.environ.get('PP4_PAIR_QUALIFICATION') == '1', 'coordinator-only physical pair qualification')
    # Full UUIDs come from the reviewed production constant, never guessed suffixes.
    from offload.launch.windows.pf1_window import WORKER_GPUS
    ck.require(os.environ.get('CUDA_VISIBLE_DEVICES') == WORKER_GPUS, 'physical production NVLink pair required')
    for name in ('PP4_PAIR_LOCK_FD0', 'PP4_PAIR_LOCK_FD1'):
        fcntl.flock(int(os.environ[name]), fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    ck.require(torch.cuda.device_count() == 2, 'two physical devices required; emulation refuses')
    for rank, uuid in enumerate(WORKER_GPUS.split(',')):
        ck.require(str(torch.cuda.get_device_properties(rank).uuid).removeprefix('GPU-') == uuid.removeprefix('GPU-'),
                   'physical pair UUID differs')


def validate(data: dict[str, Json], commit: str, run: str) -> None:
    from offload.launch.windows.pf1_window import WORKER_GPUS
    ck.require(data.get('schema') == SCHEMA and data.get('source_commit') == commit and data.get('run') == run
               and data.get('complete') is True and data.get('classification') == 'MEASURED physical pair',
               'physical-peer receipt identity/incomplete; solo is not qualification')
    ck.require(data.get('gpus') == WORKER_GPUS, 'physical-peer GPU receipt mismatch')
    rows = ck.rows(data['cases'])
    ck.require([(r.get('batch'), r.get('depth'), r.get('mode')) for r in rows] == list(CASES), 'physical-peer case coverage differs')
    for row in rows:
        ck.require(row.get('rank0_exact') is True and row.get('rank1_exact') is True
                   and row.get('changed_input_graph_replays') == 10
                   and re.fullmatch('[0-9a-f]{64}', str(row.get('solo_sha256'))) is not None,
                   'physical-peer q/ID/graph identity failed')
    for rank in (0, 1):
        check = record(data[f'timeout{rank}'])
        ck.require(check.get('missing_rank') == 1-rank and check.get('timeout_refused') is True
                   and check.get('sticky_retry_refused') is True and check.get('process_exited') is True,
                   'physical-peer timeout/sticky-error teardown unproved')
    worker = record(data['whole_worker'])
    ck.require(worker.get('complete') is True and worker.get('physical_pair') is True
               and worker.get('identity_checks') == 120 and worker.get('identity_exact') == 120
               and worker.get('archived_ids_exact') == 48 and worker.get('archived_confidence_exact') == 48,
               'physical whole-K2/N4 identity differs from solo receipt')
    sources = record(data['source_sha256'])
    ck.require(bool(sources) and all(re.fullmatch('[0-9a-f]{64}', str(v)) for v in sources.values()), 'physical-peer source hashes absent')


def checked_output(path: Path) -> None:
    ck.require(path.resolve().is_relative_to(Path('/opt/dsv41/x86host-data/pp4')),
               'pair artifacts must stay in PP4 scratch')
