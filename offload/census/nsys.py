"""Nsight Systems SQLite export (`nsys export --type sqlite <report>.nsys-rep`) -> the KL1 trace model, for the census
binner (offload/census/waterfall.py --nsys).

Read only (sqlite URI mode=ro). Tables used: CUPTI_ACTIVITY_KIND_KERNEL / _MEMCPY / _MEMSET (GPU activities),
CUPTI_ACTIVITY_KIND_RUNTIME (CUDA runtime and driver API calls, with correlation ids), NVTX_EVENTS (ranges), StringIds.
Times are converted from nanoseconds to microseconds (Kineto's unit). Requirements, refused loudly when absent:
  * step ranges: vLLM's `execute_*` iteration annotations as NVTX ranges on the launching thread (the binner delimits
    steps by them exactly as for Kineto);
  * node-level graph tracing (`nsys profile --cuda-graph-trace=node`): with graph-level tracing a replay is one graph
    activity and its kernels are not recorded (the binner then reports the capture UNBUDGETED).
Memcpy names follow Kineto's form ("Memcpy HtoD (...)") so buckets.json's drafter wait rule applies unchanged.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from offload.census.waterfall import CHUNK, Rank, prepare
from offload.gb10_budget.trace_budget import Rules, bucket
from offload.gb10_budget.trace_budget import rules as sp1_rules
from offload.kl1 import trace_kernels as T

# CUDA memcpy kinds (CUpti_ActivityMemcpyKind / nsys ENUM_CUDA_MEMCPY_OPER)
COPY_KIND = {1: 'HtoD', 2: 'DtoH', 3: 'HtoA', 4: 'AtoH', 5: 'AtoA', 6: 'AtoD', 7: 'DtoA', 8: 'DtoD', 9: 'HtoH',
             10: 'PtoP'}
NVTX_RANGES = (59, 60)            # NvtxPushPopRange, NvtxStartEndRange
REQUIRED = ('StringIds', 'CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_RUNTIME', 'NVTX_EVENTS')


def _tables(db: sqlite3.Connection) -> set[str]:
    return {str(r[0]) for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in db.execute(f'PRAGMA table_info({table})')}


def read(db: sqlite3.Connection, r: Rules) -> T.RankTrace:
    tables = _tables(db)
    missing = [t for t in REQUIRED if t not in tables]
    if missing:
        raise ValueError(f'nsys export lacks {missing} (needs CUDA trace and NVTX)')
    strings = {int(i): str(v) for i, v in db.execute('SELECT id, value FROM StringIds')}
    kernels: list[T.Kernel] = []
    kcols = _columns(db, 'CUPTI_ACTIVITY_KIND_KERNEL')
    name_col = 'demangledName' if 'demangledName' in kcols else 'shortName'
    for row in db.execute(f'SELECT start, end, streamId, correlationId, {name_col}, gridX, gridY, gridZ, blockX, '
                          'blockY, blockZ FROM CUPTI_ACTIVITY_KIND_KERNEL'):
        start, end, stream, corr, nid, gx, gy, gz, bx, by, bz = row
        name = strings[int(nid)]
        kernels.append(T.Kernel(name, T.family(name), bucket(name, r), start / 1e3, end / 1e3, int(stream),
                                (int(gx), int(gy), int(gz)), (int(bx), int(by), int(bz)), None,
                                None if corr is None else int(corr)))
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        for start, end, stream, corr, kind in db.execute(
                'SELECT start, end, streamId, correlationId, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY'):
            name = f'Memcpy {COPY_KIND.get(int(kind), "unknown")} (nsys)'
            kernels.append(T.Kernel(name, T.family(name), bucket(name, r), start / 1e3, end / 1e3, int(stream),
                                    (0, 0, 0), (0, 0, 0), None, None if corr is None else int(corr)))
    if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables:
        for start, end, stream, corr in db.execute(
                'SELECT start, end, streamId, correlationId FROM CUPTI_ACTIVITY_KIND_MEMSET'):
            name = 'Memset (Device)'
            kernels.append(T.Kernel(name, T.family(name), bucket(name, r), start / 1e3, end / 1e3, int(stream),
                                    (0, 0, 0), (0, 0, 0), None, None if corr is None else int(corr)))
    calls = [T.Call(strings[int(nid)], start / 1e3, end / 1e3, str(tid), None if corr is None else int(corr))
             for start, end, tid, corr, nid in db.execute(
                 'SELECT start, end, globalTid, correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME')]
    ncols = _columns(db, 'NVTX_EVENTS')
    text_id = 'textId' if 'textId' in ncols else 'NULL'
    annotations: list[T.Annotation] = []
    for start, end, tid, text, tid_text, etype in db.execute(
            f'SELECT start, end, globalTid, text, {text_id}, eventType FROM NVTX_EVENTS'):
        if int(etype) not in NVTX_RANGES or end is None:
            continue
        name = str(text) if text is not None else strings.get(int(tid_text), '') if tid_text is not None else ''
        if name.startswith('execute_'):
            annotations.append(T.Annotation(name, start / 1e3, end / 1e3, str(tid)))
    if not annotations:
        raise ValueError('no execute_* NVTX ranges: steps cannot be delimited')
    kernels.sort(key=lambda k: (k.start, k.end))
    calls.sort(key=lambda c: c.start)
    annotations.sort(key=lambda a: a.start)
    return T.RankTrace(kernels, calls, annotations)


def load(path: Path, r: Rules | None = None) -> Rank:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    db = sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True)
    try:
        trace = read(db, r or sp1_rules())
    finally:
        db.close()
    return prepare(trace, 'nsys', digest.hexdigest())
