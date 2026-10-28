"""CPU-only startup handshake; one real Linux thread test in a child process."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from offload.up3 import build as b, rdma_pin as pin


def test_real_linux_thread_rename_and_durable_receipt_in_subprocess(tmp_path: Path) -> None:
    script = tmp_path / 'linux_thread.py'
    script.write_text('''import json, os, sys, threading, time
from pathlib import Path
from offload.up3 import rdma_pin as pin
assert 14 in os.sched_getaffinity(0), 'real Linux rehearsal requires CPU14'
os.environ['EXL3_CPU_PIN']=pin.CPU_MASK
os.sched_setaffinity(0, {5,6,7,8,9,15,16,17,18,19})
stop=threading.Event()
def poller():
    time.sleep(.03)  # Native self-pin races the synchronous start return.
    os.sched_setaffinity(0,{14})
    stop.wait()
thread=threading.Thread(target=poller)
root=Path(sys.argv[1])/'receipt'
before=set(Path('/proc/self/task').iterdir())
try:
    receipt=pin.start_with_receipt(thread.start,14,root=root)
    assert receipt['tid']==thread.native_id and receipt['pid']==os.getpid()
    assert receipt['tid']!=receipt['pid']
    assert receipt['new_comm']==pin.COMM and len(pin.COMM.encode())<=15
    assert receipt['old_comm']!=pin.COMM
    assert os.sched_getaffinity(receipt['tid'])=={14}
    assert os.sched_getaffinity(0)=={5,6,7,8,9,15,16,17,18,19}
    task=Path('/proc/self/task')/str(receipt['tid'])
    assert (task/'comm').read_text()==pin.COMM+'\\n'
    assert (task/'status').read_text().splitlines()[0]=='Name:\t'+pin.COMM
    assert json.loads(pin.receipt_path(os.getpid(),receipt['pid_starttime'],root).read_text())==receipt
    assert set(Path('/proc/self/task').iterdir())-before=={Path('/proc/self/task')/str(receipt['tid'])}
    print(json.dumps(receipt))
finally:
    stop.set()
    if thread.ident is not None: thread.join(5)
    assert not thread.is_alive()
''')
    receipt = json.loads(subprocess.check_output([sys.executable, str(script), str(tmp_path)], text=True, timeout=10))
    assert receipt['affinity'] == [14]
    assert receipt['clock'] == 'CLOCK_MONOTONIC' and receipt['monotonic_ns'] > 0


@pytest.fixture
def proc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """File-backed proc fixture. Only the explicit Linux test touches real TIDs."""
    proc = tmp_path / 'proc'
    (proc / 'task/101').mkdir(parents=True)
    stat = '101 (worker) S ' + ' '.join(['0'] * 18 + ['100']) + '\n'
    (proc / 'stat').write_text(stat)
    (proc / 'task/101/stat').write_text(stat)
    (proc / 'task/101/comm').write_text('worker\n')
    monkeypatch.setenv('EXL3_CPU_PIN', pin.CPU_MASK)
    monkeypatch.setattr(os, 'getpid', lambda: 101)
    monkeypatch.setattr(os, 'sched_getaffinity', lambda tid: {14})
    monkeypatch.setattr(pin, 'Path', lambda path: proc / str(path).removeprefix('/proc/self/'))
    monkeypatch.setattr(pin, 'WAIT_S', .02)
    return proc, tmp_path / 'receipts'


def new_thread(proc: Path, tid: int = 102) -> None:
    task = proc / 'task' / str(tid)
    task.mkdir()
    (task / 'stat').write_text(f'{tid} (worker) S ' + ' '.join(['0'] * 18 + ['101']) + '\n')
    (task / 'comm').write_text('worker\n')


@pytest.mark.parametrize('fault', ['zero', 'multiple', 'wrong-affinity', 'rename-error', 'rename-ignored',
    'unwritable-receipt', 'existing-receipt', 'disappeared', 'native-failure'])
def test_refuses_every_ambiguous_or_unprovable_startup(fault: str, proc: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch) -> None:
    tree, receipts = proc
    original_write = Path.write_text
    def write(path: Path, data: str, *args: object, **kwargs: object) -> int:
        if path.name == 'comm' and data == pin.COMM:
            if fault == 'rename-error': raise PermissionError('rename denied')
            if fault == 'rename-ignored': return len(data)
            return original_write(path, data + '\n')  # proc read appends one newline
        # No caller passes optional write arguments in this fixture.
        return original_write(path, data)
    monkeypatch.setattr(Path, 'write_text', write)
    if fault == 'wrong-affinity': monkeypatch.setattr(os, 'sched_getaffinity', lambda tid: {5, 6})
    if fault == 'unwritable-receipt':
        original_open = Path.open
        def opened(path: Path, mode: str = 'r', *args: object, **kwargs: object) -> object:
            if path.suffix == '.json': raise PermissionError('receipt denied')
            return original_open(path, mode)
        monkeypatch.setattr(Path, 'open', opened)
    if fault == 'existing-receipt':
        receipts.mkdir(mode=0o700)
        (receipts / '101-100.json').write_text('prior\n')
    def start() -> None:
        if fault == 'native-failure': raise RuntimeError('native start failed')
        if fault == 'zero': return
        new_thread(tree)
        if fault == 'multiple': new_thread(tree, 103)
        if fault == 'disappeared': (tree / 'task/101/comm').unlink(); (tree / 'task/101/stat').unlink(); (tree / 'task/101').rmdir()
    with pytest.raises((RuntimeError, OSError)):
        pin.start_with_receipt(start, 14, root=receipts)
    if fault != 'existing-receipt': assert not list(receipts.glob('*.json'))


def test_registered_pin_and_receipt_path_are_mandatory(proc: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    _, root = proc
    for value in ('', '0-19', '14'):
        monkeypatch.setenv('EXL3_CPU_PIN', value)
        with pytest.raises(RuntimeError, match='registered'):
            pin.start_with_receipt(lambda: None, 14, root=root)
    for pid, start in ((0, '1'), (101, '../1'), (101, ''), (True, '1')):
        with pytest.raises(ValueError): pin.receipt_path(pid, start, root)


def test_exact_overlay_lazy_start_refuses_without_silent_nccl_fallback(tmp_path: Path) -> None:
    """Execute patched image ring_for, with only model/native dependencies stubbed."""
    script = tmp_path / 'overlay.py'
    script.write_text('''import importlib.util, logging, os, sys, types
from pathlib import Path
from offload.up3 import rdma_pin as pin
pkg=types.ModuleType('vllm_exl3_moe');pkg.__path__=[]
sys.modules['vllm_exl3_moe']=pkg
sys.modules['vllm_exl3_moe.up3_rdma_pin']=pin
torch=types.ModuleType('torch');torch.cuda=types.SimpleNamespace(is_current_stream_capturing=lambda:False)
dist=types.ModuleType('torch.distributed')
def gather(out, value, group):
    out[:]=[value,value]
dist.all_gather_object=gather
torch.distributed=dist
sys.modules['torch']=torch;sys.modules['torch.distributed']=dist
log=types.ModuleType('vllm.logger');log.init_logger=logging.getLogger
sys.modules['vllm.logger']=log
spec=importlib.util.spec_from_file_location('vllm_exl3_moe.rdma',sys.argv[1])
rdma=importlib.util.module_from_spec(spec);spec.loader.exec_module(rdma)
events=[]
rdma._ext=types.SimpleNamespace(create=lambda *a:0,local_info=lambda h:[None,None],connect=lambda *a:None,
    start=lambda h:events.append('start'),stop=lambda h:events.append('stop'))
comm=types.SimpleNamespace(unique_name='tp:fixture',world_size=2,rank_in_group=0,cpu_group=None)
os.environ['EXL3_CPU_PIN']=pin.CPU_MASK;os.environ['EXL3_RDMA_DEVICES']='rocep1s0f1';os.environ['EXL3_RDMA_CPU']='14'
def refused(start,cpu):
    start();raise RuntimeError('fixture poller receipt failed')
pin.start_with_receipt=refused
assert not events and not rdma._rings  # Import/install never starts the lazy ring.
try: rdma.ring_for(comm)
except RuntimeError as exc: assert 'refuses RDMA' in str(exc)
else: raise AssertionError('identity failure silently fell back')
assert events==['start','stop'] and 'tp:fixture' not in rdma._rings
print('lazy start -> refusal -> native stop; no fallback')
''')
    overlay = b.HERE / 'overlay/cpu-pin/vllm_exl3_moe/rdma.py'
    result = subprocess.check_output([sys.executable, str(script), str(overlay)], text=True, stderr=subprocess.STDOUT, timeout=10)
    assert 'no fallback' in result


def test_cpu_pin_is_only_arm_with_identity_overlay_and_control_pins_original() -> None:
    from offload.launch.windows.up3_window import source_pins
    name = 'vllm_exl3_moe/rdma.py'
    assert b.verify('control')['files'] == {}
    assert source_pins('control')[b.DIST + name] == b.digest((b.IMAGE / name).read_bytes())
    manifest = b.verify('cpu-pin')
    assert set(manifest['files']) == {name, 'vllm_exl3_moe/up3_rdma_pin.py'}
    assert (b.HERE / 'overlay/cpu-pin/vllm_exl3_moe/up3_rdma_pin.py').read_bytes() == Path(pin.__file__).read_bytes()
    assert source_pins('cpu-pin')[b.DIST + name] == manifest['files'][name]
    for arm in ('k5', 'dual-ring'):
        assert b.verify(arm)['files'] == {}
