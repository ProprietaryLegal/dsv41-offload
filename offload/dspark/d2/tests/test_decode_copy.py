"""CPU-only fake-host copy failures. No SSH, HTTP, GPU or real capture data."""
from __future__ import annotations

import builtins
import json
import shutil
import subprocess
import sys
from io import BytesIO, StringIO
from pathlib import Path

import pytest

from offload.dspark.d2 import decode_copy as m


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('test attempted live process/network')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(m.transfer.urllib.request, 'urlopen', forbidden)
    def direct_only(argv, **kwargs):
        assert argv[0] == 'dd' and 'iflag=direct' in argv
        return Path(next(a[3:] for a in argv if a.startswith('if='))).read_bytes()
    monkeypatch.setattr(subprocess, 'check_output', direct_only)


def fixture_tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path/'capture'
    seq = root/'s0'
    seq.mkdir(parents=True)
    (seq/'CAPTURE.json').write_text(json.dumps({'schema': 'd2.decode-capture.v1',
        'seq': 's0', 'taps_source': 'decode_capture'}))
    (seq/'ledger.jsonl').write_text('{"synthetic": true}\n')
    (seq/'taps.bin').write_bytes(b'x'*4096)
    (seq/'ids.bin').write_bytes(b'\1\0\0\0')
    (root/'FLUSH.json').write_text('{"nonce":"test"}')
    (root/'ERROR_COUNT.json').write_text('{"errors":0}')
    (root/'FLUSHED.json').write_text(json.dumps({'schema':'d2.decode-capture.v1',
        'nonce':'test', 'errors':0, 'sequences':{'s0':{
            'ledger_sha256':m.transfer.sha_file(seq/'ledger.jsonl'),
            'taps_bytes':4096, 'ids_bytes':4}}}))
    state = tmp_path/'state'
    state.mkdir()
    (state/'state.json').write_text('{"synthetic":true}')
    return root, state


class Fake(m.Remote):
    def __init__(self, capture: bool):
        super().__init__(['forbidden-host'], capture)

    def inventory(self, root: str, nonce: str) -> m.Inventory:
        return m.scan(root, self.capture, nonce)

    def copy(self, source: str, destination: Path, entry: m.Record) -> m.Record:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        m.require(destination.stat().st_size == entry['size'],
                  'verification failed: synthetic source size changed')
        digest = m.transfer.sha_file(destination)
        if 'sha256' in entry:
            m.require(digest == entry['sha256'], 'verification failed: synthetic source changed')
        return {'sha256':digest, 'size':destination.stat().st_size}


def perform(tmp_path: Path, capture: Fake | None = None, **kwargs) -> m.Record:
    return m.frozen_copy(capture or Fake(True), Fake(False), str(tmp_path/'capture'),
                         str(tmp_path/'state'), tmp_path/'new', 'test', **kwargs)


def test_complete_copy_hashes_every_file_and_optional_tree(tmp_path):
    fixture_tree(tmp_path)
    result = perform(tmp_path, top20=str(tmp_path/'state'))
    assert result['frozen'] is True
    receipt = json.loads((tmp_path/'new/COPY_RECEIPT.json').read_text())
    files = receipt['files']
    assert {r['tree'] for r in files} == {'capture','generator','top20'}
    assert all(m.transfer.sha_file(tmp_path/'new'/r['tree']/r['relative']) == r['sha256'] for r in files)
    assert (tmp_path/'new').stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize('failure', ['nonce','errors','ledger','size','new-sequence','symlink'])
def test_invalid_flush_refuses_before_destination(tmp_path, failure):
    root, _ = fixture_tree(tmp_path)
    if failure in ('nonce','errors'):
        p = root/'FLUSHED.json'
        d = json.loads(p.read_text())
        d['nonce' if failure == 'nonce' else 'errors'] = 'wrong' if failure == 'nonce' else 1
        p.write_text(json.dumps(d))
    elif failure == 'ledger':
        (root/'s0/ledger.jsonl').write_text('changed')
    elif failure == 'size':
        (root/'s0/taps.bin').write_bytes(b'short')
    elif failure == 'new-sequence':
        (root/'s1').mkdir()
        (root/'s1/CAPTURE.json').write_text('{}')
    else:
        (root/'escape').symlink_to(tmp_path/'state')
    with pytest.raises(RuntimeError):
        perform(tmp_path)
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('path,content,error', [
    ('s0/ledger.jsonl', b'new-ledger', 'verification failed'),
    ('s0/taps.bin', b'short', 'verification failed'),
    ('s0/taps.bin', b'y'*4096, 'snapshot changed')])
def test_mid_copy_ledger_size_or_same_size_new_write_invalidates(tmp_path, path, content, error):
    root, _ = fixture_tree(tmp_path)
    class Mutate(Fake):
        def copy(self, *args):
            result = super().copy(*args)
            (root/path).write_bytes(content)
            return result
    with pytest.raises(RuntimeError, match=error):
        perform(tmp_path, Mutate(True))
    assert not (tmp_path/'new/COPY_RECEIPT.json').exists()


def test_generator_changes_are_detected(tmp_path):
    _, state = fixture_tree(tmp_path)
    class Mutate(Fake):
        def copy(self, *args):
            result = super().copy(*args)
            (state/'state.json').write_text('changed')
            return result
    with pytest.raises(RuntimeError, match='snapshot changed'):
        m.frozen_copy(Fake(True), Mutate(False), str(tmp_path/'capture'),
                      str(state), tmp_path/'new', 'test')


def test_destination_exists_never_reused(tmp_path):
    fixture_tree(tmp_path)
    (tmp_path/'new').mkdir()
    with pytest.raises(RuntimeError, match='destination exists'):
        perform(tmp_path)


def test_destination_reread_hash_mismatch_leaves_no_receipt(tmp_path):
    fixture_tree(tmp_path)
    with pytest.raises(RuntimeError, match='re-read hash mismatch'):
        perform(tmp_path, reread=lambda p: 'bad-hash')
    assert not (tmp_path/'new/COPY_RECEIPT.json').exists()


def test_error_ledger_is_preserved_without_claiming_sequence_usability(tmp_path):
    root, _ = fixture_tree(tmp_path)
    (root/'ERRORS.jsonl').write_text('{"seq_ids":["s0"],"error":"synthetic"}\n')
    (root/'ERROR_COUNT.json').write_text('{"errors":1}')
    p = root/'FLUSHED.json'
    data = json.loads(p.read_text()); data['errors'] = 1
    p.write_text(json.dumps(data))
    perform(tmp_path)
    assert (tmp_path/'new/capture/ERRORS.jsonl').read_bytes() == (root/'ERRORS.jsonl').read_bytes()


def test_remote_scan_code_is_the_tested_contract(tmp_path, monkeypatch, capsys):
    root, _ = fixture_tree(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['scan',str(root),'capture','test'])
    exec(compile(m.SCAN_CODE, '<remote scan>', 'exec'), {})  # noqa: S102 - own source under fake-host guards
    assert json.loads(capsys.readouterr().out) == m.scan(str(root), True, 'test')


def test_idle_guard_trip_before_nic(tmp_path, monkeypatch):
    def busy():
        raise m.transfer.CopyInterrupted('serve is not idle')
    monkeypatch.setattr(m.transfer, 'idle', busy)
    with pytest.raises(m.transfer.CopyInterrupted, match='not idle'):
        m.Remote(['ssh','NEVER'], True).inventory('/never', 'test')
    with pytest.raises(m.transfer.CopyInterrupted, match='not idle'):
        m.Remote(['ssh','NEVER'], True).copy('/never', tmp_path/'out', {'size':1})


class Child:
    def __init__(self, data: bytes = b'payload'):
        self.stdout = BytesIO(data)
        self.returncode = 0
        self.terminated = False

    def wait(self):
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True


def test_reused_transport_stream_and_reread_hash_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(m.transfer, 'idle', lambda: None)
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: Child())
    calls = []
    def bad_reread(path):
        calls.append(path)
        return 'wrong'
    monkeypatch.setattr(m.transfer, 'sha_file', bad_reread)
    with pytest.raises(RuntimeError, match='disk hash mismatch'):
        m.Remote(['fake-host'], True).copy('/fake',tmp_path/'out', {'size':7})
    assert calls and not (tmp_path/'out').exists()


@pytest.mark.parametrize('samples', [[1], [5*1024**3, 1]])
def test_source_memory_guard_before_and_during_copy(monkeypatch, samples):
    child = Child()
    started = []
    def fake_popen(argv):
        assert argv[0] == 'dd' and 'iflag=direct,skip_bytes' in argv
        started.append(argv)
        return child
    monkeypatch.setattr(subprocess, 'Popen', fake_popen)
    values = iter(samples)
    def meminfo(path):
        assert path == '/proc/meminfo'
        return StringIO('MemAvailable: '+str(next(values)//1024)+' kB\n')
    monkeypatch.setattr(builtins, 'open', meminfo)
    monkeypatch.setattr(sys, 'argv', ['copy','/synthetic','0'])
    with pytest.raises(RuntimeError, match='MemAvailable below 4 GiB'):
        exec(compile(m.transfer.REMOTE, '<guarded source>', 'exec'), {})  # noqa: S102 - own source, mocked dd
    assert bool(started) == (len(samples) == 2)
    if started:
        assert child.terminated


def test_continuous_idle_guard_interrupts_owned_transfer(tmp_path, monkeypatch):
    import threading
    stopped = threading.Event()
    calls = []
    class Blocking(BytesIO):
        def read(self, size=-1):
            assert stopped.wait(3), 'idle monitor did not interrupt transfer'
            return b''
    class Running(Child):
        def __init__(self):
            super().__init__()
            self.stdout = Blocking()
        def poll(self):
            return 1 if stopped.is_set() else None
        def terminate(self):
            stopped.set()
        def wait(self):
            return 1
    def guard():
        calls.append(True)
        if len(calls) > 2:
            raise m.transfer.CopyInterrupted('foreign demand')
    monkeypatch.setattr(m.transfer, 'idle', guard)
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: Running())
    with pytest.raises(m.transfer.CopyInterrupted, match='foreign demand'):
        m.Remote(['fake'],True).copy('/synthetic', tmp_path/'out', {'size':4})
    assert stopped.is_set() and not (tmp_path/'out').exists()


def test_destination_memory_guard_before_start(tmp_path, monkeypatch):
    monkeypatch.setattr(m.transfer, 'idle', lambda: None)
    original = Path.read_text
    def read(path, *args, **kwargs):
        if str(path) == '/proc/meminfo':
            return 'MemAvailable: 1024 kB\n'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    with pytest.raises(RuntimeError, match='x86host has less than 16 GiB'):
        m.Remote(['never'],True).copy('/synthetic', tmp_path/'out', {'size':4})


def test_flush_atomic_new_nonce_wait_and_reuse_refusal(tmp_path, monkeypatch):
    import time
    root = tmp_path/'flush'
    root.mkdir()
    monkeypatch.setattr(sys, 'argv', ['flush',str(root),'new-nonce'])
    def writer_ack(delay):
        assert json.loads((root/'FLUSH.json').read_text()) == {'nonce':'new-nonce'}
        assert not (root/'FLUSH.json.tmp').exists()
        (root/'FLUSHED.json').write_text('{"nonce":"new-nonce"}')
    monkeypatch.setattr(time, 'sleep', writer_ack)
    exec(compile(m.FLUSH_CODE, '<flush request>', 'exec'), {})  # noqa: S102 - own source, synthetic tree
    with pytest.raises(SystemExit, match='choose a NEW nonce'):
        exec(compile(m.FLUSH_CODE, '<flush request>', 'exec'), {})  # noqa: S102 - own source, synthetic tree


def test_pending_flush_nonce_cannot_be_reused(tmp_path, monkeypatch):
    (tmp_path/'FLUSH.json').write_text('{"nonce":"pending"}')
    monkeypatch.setattr(sys, 'argv', ['flush',str(tmp_path),'pending'])
    with pytest.raises(SystemExit, match='choose a NEW nonce'):
        exec(compile(m.FLUSH_CODE, '<flush request>', 'exec'), {})  # noqa: S102 - own source, synthetic tree


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def read(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def pause_budget(clock: Clock, check, cap: float = 4*3600) -> m.PauseBudget:
    return m.PauseBudget(cap_seconds=cap, check=check, memory=lambda: None,
                         clock=clock.read, sleep=clock.sleep)


def test_pause_requires_thirty_continuous_idle_seconds_and_records_wait() -> None:
    clock = Clock()
    def check() -> None:
        if clock.now < 1 or 3 <= clock.now < 5:
            raise m.ProductionBusy('synthetic demand')
    budget = pause_budget(clock, check)
    budget.wait()
    assert clock.now == budget.waited == 35
    assert budget.events[0]['wait_seconds'] == 35
    assert budget.cap_seconds == 4*3600


def test_pause_cap_is_cumulative_across_interruptions() -> None:
    clock = Clock()
    budget = pause_budget(clock, lambda: None, cap=45)
    budget.wait()
    assert budget.waited == 30
    with pytest.raises(RuntimeError, match='total wait cap exhausted'):
        budget.wait()
    assert budget.waited == 45


@pytest.mark.parametrize('mutate', [False, True])
def test_busy_copy_restarts_owned_partial_or_aborts_on_paused_snapshot_change(tmp_path, monkeypatch, mutate):
    root, _ = fixture_tree(tmp_path)
    clock = Clock()
    budget = pause_budget(clock, lambda: None)
    attempts: list[Path] = []
    class ActualTransport(Fake):
        copy = m.Remote.copy
    def copy(source, target, expected, size, *, idle_check):
        idle_check()
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name+'.s2-partial')
        if target.name == 'taps.bin':
            attempts.append(target)
            assert not partial.exists(), 'interrupted file must restart at byte zero'
            if len(attempts) == 1:
                partial.write_bytes(b'partial')
                if mutate:
                    (root/'s0/taps.bin').write_bytes(b'y'*4096)
                error = m.transfer.CopyInterrupted('monitor stopped owned child')
                error.__cause__ = m.ProductionBusy('synthetic request')
                raise error
        shutil.copyfile(source, target)
        return {'sha256':m.transfer.sha_file(target), 'size':size}
    monkeypatch.setattr(m.transfer, 'copy', copy)
    if mutate:
        with pytest.raises(RuntimeError, match='snapshot changed during pause'):
            perform(tmp_path, ActualTransport(True), pause=budget)
        assert len(attempts) == 1 and not (tmp_path/'new/COPY_RECEIPT.json').exists()
    else:
        result = perform(tmp_path, ActualTransport(True), pause=budget)
        assert len(attempts) == 2
        assert result['pause_policy']['total_wait_seconds'] == 30
        assert result['pause_policy']['total_wait_cap_seconds'] == 14400
        assert (tmp_path/'new/capture/s0/taps.bin').read_bytes() == b'x'*4096


def test_wait_cap_exhaustion_leaves_no_copy_receipt(tmp_path):
    fixture_tree(tmp_path)
    class Busy(Fake):
        def copy(self, *args):
            raise m.ProductionBusy('synthetic demand')
    clock = Clock()
    def busy() -> None:
        raise m.ProductionBusy('still busy')
    budget = pause_budget(clock, busy, cap=2)
    with pytest.raises(RuntimeError, match='total wait cap exhausted'):
        perform(tmp_path, Busy(True), pause=budget)
    assert budget.waited == 2 and not (tmp_path/'new/COPY_RECEIPT.json').exists()


@pytest.mark.parametrize('failure', [RuntimeError('memory floor'), OSError('metrics unavailable'),
                                   m.transfer.CopyInterrupted('malformed metrics')])
def test_non_demand_failures_never_pause_or_retry(failure):
    clock = Clock()
    budget = pause_budget(clock, lambda: None)
    calls = []
    def action():
        calls.append(True)
        raise failure
    with pytest.raises(type(failure), match=str(failure)):
        budget.run(action)
    assert len(calls) == 1 and budget.waited == 0 and not budget.events


def test_memory_guard_remains_fatal_while_paused() -> None:
    clock = Clock()
    def memory() -> None:
        raise RuntimeError('memory floor during pause')
    budget = pause_budget(clock, lambda: None)
    budget.memory = memory
    with pytest.raises(RuntimeError, match='memory floor during pause'):
        budget.wait()
    assert clock.now == 0


@pytest.mark.parametrize('running,waiting,exception', [
    ('1','0',m.ProductionBusy), ('0','2',m.ProductionBusy),
    ('nan','0',RuntimeError), ('-1','0',RuntimeError), ('0',None,RuntimeError),
    ('1',None,RuntimeError), ('0','0',None)])
def test_only_valid_nonzero_demand_metrics_are_resumable(monkeypatch, running, waiting, exception):
    body = f'vllm:num_requests_running{{}} {running}\n'
    if waiting is not None:
        body += f'vllm:num_requests_waiting{{}} {waiting}\n'
    monkeypatch.setattr(m.urllib.request, 'urlopen', lambda *a, **k: BytesIO(body.encode()))
    if exception is None:
        m.production_idle()
    else:
        with pytest.raises(exception) as caught:
            m.production_idle()
        if exception is RuntimeError:
            assert not isinstance(caught.value, m.ProductionBusy)


def test_inventory_traffic_monitor_stops_only_its_owned_child(monkeypatch):
    import threading
    stopped = threading.Event()
    class InventoryChild:
        returncode = 1
        stdout = None
        def communicate(self):
            assert stopped.wait(3), 'inventory demand monitor failed to stop traffic'
            return '{}', None
        def terminate(self):
            stopped.set()
        def poll(self):
            return 1 if stopped.is_set() else None
        def wait(self):
            return 1
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: InventoryChild())
    calls = []
    def check():
        calls.append(True)
        if len(calls) > 1:
            raise m.ProductionBusy('demand during inventory')
    remote = m.Remote(['fake'], True)
    remote.restart_partial = True
    remote.idle_check = check
    with pytest.raises(m.ProductionBusy, match='demand during inventory'):
        remote.inventory('/synthetic', 'test')
    assert stopped.is_set()


def test_real_transfer_monitor_pause_and_byte_zero_restart(tmp_path, monkeypatch):
    import threading
    stopped = threading.Event()
    payload = b'payload'
    class InterruptedStream(BytesIO):
        def __init__(self):
            super().__init__(payload[:3])
            self.first = True
        def read(self, size=-1):
            if self.first:
                self.first = False
                return super().read(size)
            assert stopped.wait(3)
            return b''
    class InterruptedChild(Child):
        def __init__(self):
            super().__init__()
            self.stdout = InterruptedStream()
        def poll(self):
            return 1 if stopped.is_set() else None
        def terminate(self):
            stopped.set()
        def wait(self):
            return 1
    starts = []
    def spawn(argv, **kwargs):
        starts.append(argv)
        return InterruptedChild() if len(starts) == 1 else Child(payload)
    monkeypatch.setattr(subprocess, 'Popen', spawn)
    calls = []
    def check():
        calls.append(True)
        if len(calls) == 3:
            raise m.ProductionBusy('foreign demand')
    clock = Clock()
    budget = pause_budget(clock, check)
    remote = m.Remote(['fake'], True)
    remote.restart_partial = True
    remote.idle_check = check
    target = tmp_path/'out'
    receipt = budget.run(lambda: remote.copy('/synthetic', target, {'size':len(payload)}))
    assert len(starts) == 2 and stopped.is_set()
    assert target.read_bytes() == payload and receipt['resumed_from_bytes'] == 0
    assert budget.waited == 30 and not (tmp_path/'out.s2-partial').exists()
