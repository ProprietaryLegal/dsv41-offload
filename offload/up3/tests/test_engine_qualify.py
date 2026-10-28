from __future__ import annotations

import copy
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from offload.prefill_maxlen.evidence import Json, record
from offload.up3 import bootstrap as bs, build, gates, qualify as q, qualification_lock as lock
from offload.up3.engine_common import sha
from offload.up3.tests.test_up3_window import receipt


def request(arm: str, rank: int = 0) -> q.Request:
    return q.Request(arm, rank, 'fixture-run', 'c'*64, sha(Path(q.__file__)), 100.)


def packet(arm: str, rank: int = 0) -> dict[str, Json]:
    req = request(arm, rank)
    value = receipt(arm, rank, req.run, req.container, 100., req.program_sha256)
    value.update(worker_pid=123, qualification_package_sha256=sha(q.HERE/'qualification.lock.json'))
    return value


def identity() -> list[Json]:
    return [{'Id': 'c'*64, 'State': {'Running': True}, 'Config': {'Cmd': ['-m', 'vllm', '--port', '8001']}}]


@pytest.mark.parametrize('arm', list(gates.CASES))
@pytest.mark.parametrize('rank', [0, 1])
def test_collector_exact_contract_both_ranks(arm: str, rank: int) -> None:
    req, row = request(arm, rank), packet(arm, rank)
    log = '(worker prefix) '+q.MARKER+json.dumps(row)
    assert q.collect(req, identity(), log, now=101) == row
    for bad in ('', log+'\n'+log, log.replace('fixture-run', 'old-run')):
        with pytest.raises(ValueError, match='receipt'):
            q.collect(req, identity(), bad, now=101)
    stale = {**row, 'time': 99}
    with pytest.raises(ValueError, match='stale'):
        q.collect(req, identity(), q.MARKER+json.dumps(stale), now=101)
    prod = copy.deepcopy(identity())
    record(record(prod[0])['Config'])['Cmd'] = ['--port', '8000']
    with pytest.raises(ValueError, match='production-port'):
        q.collect(req, prod, log, now=101)


def test_collector_rejects_hash_identity_and_failed_gates() -> None:
    req, row = request('split'), packet('split')
    for key, value in (('container', 'd'*64), ('scope', 'synthetic-only'), ('program_sha256', 'a'*64),
                       ('qualification_package_sha256', 'b'*64), ('worker_pid', 0)):
        bad = {**row, key: value}
        with pytest.raises(ValueError):
            q.collect(req, identity(), q.MARKER+json.dumps(bad), now=101)
    record(record(row['cases'])['m43'])['pass'] = False
    with pytest.raises(ValueError, match='failed'):
        q.collect(req, identity(), q.MARKER+json.dumps(row), now=101)


def test_cli_defaults_and_unowned_execute_have_no_host_actions() -> None:
    with patch.object(subprocess, 'run', side_effect=AssertionError('no Docker allowed')):
        assert q.main([]) == 0
        assert q.main(['--check-package']) == 0
        with patch.dict(os.environ, {'SCOPED_REMOTE_DEADLINE': '0'}):
            assert q.main(['--execute']) == 1


def test_package_lock_cpu_rebuild_and_drift(tmp_path: Path) -> None:
    assert json.loads((q.HERE/'qualification.lock.json').read_text()) == lock.generate()
    data = record(q.package()['files'])
    for name in data:
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((build.ROOT/name).read_bytes())
    changed = tmp_path/'offload/up3/engine_split.py'
    changed.write_text(changed.read_text()+'\n# drift\n')
    with patch.object(build, 'ROOT', tmp_path):
        with pytest.raises(ValueError, match='source drift'):
            q.package()


def test_session_k5_failure_closes_real_lifecycle_hooks() -> None:
    fake_observer = SimpleNamespace(install=lambda: None, close=lambda: closed.append(True),
        collect=lambda value: (_ for _ in ()).throw(ValueError('missing actual proposals')))
    closed: list[bool] = []
    import offload.up3.engine_k5 as k5
    with patch.object(bs, 'engine_sources'), patch.object(bs, 'worker_identity'), \
         patch.object(k5, 'Observer', return_value=fake_observer), \
         patch.object(k5, 'native_gate', return_value={'pass': True, 'samples': 1}):
        session = bs.Session(object(), request('k5'))
        session.before_init()
        session.after_warmup()
        with pytest.raises(ValueError, match='missing actual'):
            session.finish(emit=False)
    assert closed


def test_session_split_cannot_emit_without_actual_logits_and_task_review() -> None:
    import offload.up3.engine_split as split
    with patch.object(bs, 'engine_sources'), patch.object(bs, 'worker_identity'), patch.object(split, 'run', return_value={}):
        session = bs.Session(object(), request('split'))
        session.before_init()
        session.after_warmup()
        with pytest.raises(ValueError, match='actual C/C/A'):
            session.finish(emit=False)


def test_session_vor_success_emits_once_and_bad_lifecycle_refuses() -> None:
    import offload.up3.engine_vor as vor
    row = packet('vor')
    with patch.object(bs, 'engine_sources'), patch.object(bs, 'worker_identity'), \
         patch.object(vor, 'run', return_value=record(row['cases'])), \
         patch.object(bs, 'receipt', return_value=row):
        session = bs.Session(object(), request('vor'))
        with pytest.raises(ValueError, match='order'):
            session.after_warmup()
        session.before_init()
        session.after_warmup()
        assert session.finish(emit=False) == row
        with pytest.raises(ValueError, match='reused'):
            session.finish(emit=False)


@pytest.mark.parametrize('capability,rank,host,valid', [
    ((12, 1), 0, 'c'*12, True), ((8, 6), 0, 'c'*12, False),
    ((12, 1), 1, 'c'*12, False), ((12, 1), 0, 'd'*12, False)])
def test_actual_worker_identity_gate_before_native_work(capability: tuple[int, int], rank: int,
                                                      host: str, valid: bool) -> None:
    import sys
    import torch
    from offload.up3.tests.test_engine_k5 import worker
    distributed = SimpleNamespace(get_tp_group=lambda: SimpleNamespace(rank_in_group=rank, world_size=2))
    original = Path.read_text
    def read(path: Path, *args: object, **kwargs: object) -> str:
        if str(path) == '/etc/hostname':
            return host
        return original(path)
    with patch.dict(sys.modules, {'vllm.distributed': distributed}), \
         patch.object(torch.cuda, 'get_device_capability', return_value=capability), patch.object(Path, 'read_text', read):
        if valid:
            assert q.worker_identity(request('vor'), worker()) == [12, 1]
        else:
            with pytest.raises(ValueError):
                q.worker_identity(request('vor'), worker())


def test_collector_uses_exact_start_instead_of_a_sliding_freshness_window(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_text
    def scoped(path: Path, *args: object, **kwargs: object) -> str:
        if path == Path('/proc/self/cgroup'):
            return '0::/user.slice/dsv41w-fixture.scope'
        # All package and source reads retain their exact real bytes.
        return original(path)
    monkeypatch.setattr(Path, 'read_text', scoped)
    monkeypatch.setattr(time, 'time', lambda: 500.)
    row = packet('k5')  # receipt100, exact run/container; older than the retired180s lookback
    def docker(argv: list[str], **kwargs: object) -> SimpleNamespace:
        assert argv[:2] in (['docker','inspect'], ['docker','logs'])
        return SimpleNamespace(stdout=json.dumps(identity()) if argv[1] == 'inspect' else q.MARKER+json.dumps(row), stderr='')
    monkeypatch.setattr(subprocess, 'run', docker)
    monkeypatch.setenv('SCOPED_REMOTE_DEADLINE', '600')
    args = ['--execute','--arm','k5','--rank','0','--run','fixture-run','--container','c'*64,
            '--program-sha256',sha(Path(q.__file__))]
    assert q.main([*args,'--since','100']) == 0
    assert q.main([*args,'--since','101']) == 1  # genuinely stale receipt still refuses
