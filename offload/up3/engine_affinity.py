"""Read actual TID affinity, including native thread birth attribution.

Install Births before normal model loading. Hooks observe normal engine calls;
they never start a proxy, change affinity, import a second model or kill a PID.
Missing early instrumentation fails closed instead of guessing thread roles.
"""
from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from offload.prefill_maxlen.evidence import Json
from offload.up3.engine_common import Cases, attr, invoke, passed, replaced, require, runner

CPUS = frozenset((*range(5, 10), *range(15, 20)))


@dataclass(frozen=True)
class Thread:
    pid: int
    tid: int
    start: str
    cpus: frozenset[int]

    def json(self) -> Json:
        return {'pid': self.pid, 'tid': self.tid, 'start_ticks': self.start, 'cpus': cast(list[Json], sorted(self.cpus))}


def stat(path: Path) -> list[str]:
    text = path.read_text()
    require(')' in text, 'malformed proc stat')
    return text[text.rindex(')')+2:].split()


def census(pid: int, *, proc: Path = Path('/proc'),
           affinity: Callable[[int], set[int]] = os.sched_getaffinity) -> dict[int, Thread]:
    root = proc/str(pid)/'task'
    tids = sorted(int(p.name) for p in root.iterdir() if p.name.isdecimal())
    result = {}
    for tid in tids:
        path = root/str(tid)/'stat'
        before = stat(path)[19]
        cpus = frozenset(affinity(tid))
        after = stat(path)[19]
        require(before == after and bool(cpus), 'TID reused or empty affinity')
        result[tid] = Thread(pid, tid, before, cpus)
    require(tids == sorted(int(p.name) for p in root.iterdir() if p.name.isdecimal()),
            'threads changed during affinity census')
    require(bool(result), 'empty thread census')
    return result


class Births:
    def __init__(self, *, snapshot: Callable[[], dict[int, Thread]] | None = None):
        self.snapshot = snapshot or (lambda: census(os.getpid()))
        self.stack = ExitStack()
        self.roles: dict[str, dict[int, Thread]] = {}
        self.watched: set[tuple[int, str]] = set()

    def watch(self, obj: object, name: str, role: str, *, exactly_one: bool = False) -> None:
        key = (id(obj), name)
        if key in self.watched:
            return
        self.watched.add(key)
        original = attr(obj, name)
        def observed(*args: object, **kwargs: object) -> object:
            before = self.snapshot()
            result = invoke(original, *args, **kwargs)
            after = self.snapshot()
            new = {tid: row for tid, row in after.items() if tid not in before}
            require(not exactly_one or len(new) == 1, role+': could not attribute exactly one actual native TID')
            self.roles.setdefault(role, {}).update(new)
            return result
        self.stack.enter_context(replaced(obj, name, observed))

    def install(self) -> None:
        rdma = importlib.import_module('vllm_exl3_moe.rdma')
        engram = importlib.import_module('vllm.models.deepseek_v41.common.engram')
        original_rdma = attr(rdma, 'ext')
        original_engram = attr(engram, '_load_engram_remote_ext')
        def ring_ext(*args: object, **kwargs: object) -> object:
            module = invoke(original_rdma, *args, **kwargs)
            self.watch(module, 'start', 'rdma-poller', exactly_one=True)
            return module
        def engram_ext(*args: object, **kwargs: object) -> object:
            module = invoke(original_engram, *args, **kwargs)
            cls = attr(module, 'EngramRemoteTable')
            self.watch(cls, 'register_staging', 'engram-reconnect')
            self.watch(cls, 'enable_spin', 'engram-spin')
            return module
        self.stack.enter_context(replaced(rdma, 'ext', ring_ext))
        self.stack.enter_context(replaced(engram, '_load_engram_remote_ext', engram_ext))

    def close(self) -> None:
        self.stack.close()


def check_threads(engine: dict[int, Thread], proxy: dict[int, Thread], births: Births,
                  *, rank: int = 0) -> Cases:
    pollers = births.roles.get('rdma-poller', {})
    require(bool(pollers), 'no actual RDMA poller TID birth evidence')
    engram = {tid: t for role in ('engram-reconnect', 'engram-spin')
              for tid, t in births.roles.get(role, {}).items()}
    require(bool(births.roles.get('engram-spin')) and bool(births.roles.get('engram-reconnect')),
            'Engram spin/reconnector TIDs were not observed at their actual API calls')
    for tid, row in {**pollers, **engram}.items():
        require(tid in engine and engine[tid].start == row.start, 'attributed native TID exited/reused')
    for tid, row in engine.items():
        require(row.cpus == (frozenset({14}) if tid in pollers else CPUS), f'worker TID{tid} affinity differs')
    require((bool(proxy) if rank == 0 else not proxy) and all(t.cpus == CPUS for t in proxy.values()),
            'async proxy thread affinity/presence differs')
    result: Cases = {'all-thread-affinities': passed(len(engine)+len(proxy), every_tid=True,
        thread_count=len(engine)+len(proxy), allowed_cpus='5-9,15-19',
        engine=cast(list[Json], [t.json() for t in engine.values()]),
        proxy=cast(list[Json], [t.json() for t in proxy.values()])),
        'poller14': passed(len(pollers), cpu=14, tids=cast(list[Json], sorted(pollers))),
        'async-proxy': passed(max(1, len(proxy)), tids=cast(list[Json], sorted(proxy)), rank=rank,
                              present=bool(proxy), rank0_only=True),
        'engram': passed(len(engram), tids=cast(list[Json], sorted(engram)),
            roles={role: cast(Json, sorted(rows)) for role, rows in births.roles.items() if role.startswith('engram')})}
    return result


def run(worker: object, births: Births, *, proc: Path = Path('/proc')) -> Cases:
    r = runner(worker)
    require(os.environ.get('EXL3_CPU_PIN') == '5-9,15-19', 'pin switch differs')
    spec = attr(r, 'speculator')
    require(type(spec).__name__ == 'RemoteDSparkCapacitySpeculator', 'actual capacity/process proxy required')
    remote = attr(spec, 'remote')
    proxy = attr(remote, 'proxy')
    rank = cast(int, attr(remote, 'tp_rank'))
    require(rank in (0, 1) and attr(attr(remote, 'cfg'), 'async_proxy') == 'process', 'not TP2 process mode')
    proxy_threads: dict[int, Thread] = {}
    if rank == 0:
        require(proxy is not None, 'rank0 actual async process proxy missing')
        pid = cast(int, attr(proxy, 'pid'))
        identity = stat(proc/str(pid)/'stat')
        require(identity[1] == str(os.getpid()), 'proxy is not this engine worker child')
        maps = (proc/str(pid)/'maps').read_text()
        require(not any(name in maps for name in ('libtorch', 'libc10', '/torch/')), 'proxy loaded torch')
        proxy_threads = census(pid, proc=proc)
        require(stat(proc/str(pid)/'stat')[19] == identity[19], 'proxy PID reused during census')
    else:
        require(proxy is None, 'rank1 must not duplicate rank0 process proxy')
    engine_threads = census(os.getpid(), proc=proc)
    tables = [module for _, module in cast(Iterable[tuple[str, object]], invoke(attr(attr(r, 'model'), 'named_modules')))
              if getattr(module, '_remote_endpoint', None)]
    require(bool(tables), 'no actual remote Engram tables')
    table_rows: list[Json] = []
    for table in tables:
        disk = attr(table, '_disk')
        stats = cast(dict[str, int], invoke(attr(disk, 'remote_stats')))
        require(stats.get('down') == 0, 'Engram is using local fallback')
        mode = str(invoke(attr(disk, 'mode')))
        require('(spin proxy)' in mode, 'Engram spin proxy absent')
        table_rows.append({'identity': str(invoke(attr(disk, 'identity'))), 'mode': mode,
            'requests': stats['requests'], 'rows': stats['rows'], 'fallbacks': stats['fallbacks']})
    result = check_threads(engine_threads, proxy_threads, births, rank=rank)
    result['engram'] = {**cast(dict[str, Json], result['engram']), 'tables': table_rows}
    return result
