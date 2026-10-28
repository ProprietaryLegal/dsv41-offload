"""Real CPU traceback/flush checks; no CUDA or physical qualification."""
from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import torch

from offload.dspark.probabilistic.pp4 import pair_diagnostics as d
from offload.dspark.probabilistic.pp4 import pair_probe as p
from offload.dspark.probabilistic.pp4 import pair_qualify as q


def test_phase_flushes_before_work_and_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, bool]] = []

    def emit(value: str, *, flush: bool) -> None:
        events.append((value, flush))

    monkeypatch.setattr('builtins.print', emit)
    error = RuntimeError('CPU fixture failure')
    with pytest.raises(RuntimeError) as raised, d.phase('capture rank=1'):
        assert events[-1][0].endswith('BEGIN capture rank=1')
        raise error
    assert raised.value is error
    assert [v.split('s ', 1)[1] for v, _ in events] == [
        'BEGIN capture rank=1', 'FAIL capture rank=1 (RuntimeError)']
    assert all(flush for _, flush in events)
    events.clear()
    with d.phase('wait_idle physical'):
        assert len(events) == 1
    assert events[-1][0].endswith('END wait_idle physical')


def test_real_watchdog_dumps_stuck_phase_then_cancels(tmp_path: Path) -> None:
    # No forced kill/timeout: this child exits naturally after a bounded CPU
    # sleep. Native faulthandler proves stderr has stacks while stdout is a file.
    source = '''
import time
from offload.dspark.probabilistic.pp4 import pair_diagnostics as d
d.DUMP_INTERVAL_S = .04
d.start_watchdog()
try:
    with d.phase('CPU fixture blocked eager position=0'):
        time.sleep(.15)
finally:
    d.stop_watchdog()
print('CANCELLED', flush=True)
time.sleep(.12)
print('EXIT', flush=True)
'''
    log = tmp_path/'watchdog.log'
    with log.open('x') as output:
        subprocess.run([sys.executable, '-c', source], stdout=output, stderr=subprocess.STDOUT, check=True)
    value = log.read_text()
    assert 'BEGIN CPU fixture blocked eager position=0' in value
    assert 'Timeout (0:00:00.040000)!' in value and 'File "<string>"' in value
    assert 'END CPU fixture blocked eager position=0' in value
    assert value.split('CANCELLED\n', 1)[1] == 'EXIT\n'


def test_cli_watchdog_precedes_torch_import_failure(tmp_path: Path) -> None:
    source = '''
import builtins, runpy, time
from offload.dspark.probabilistic.pp4 import pair_diagnostics as d
d.DUMP_INTERVAL_S = .04
real_import = builtins.__import__
def stalled_import(name, *args, **kwargs):
    if name == 'torch':
        time.sleep(.12)
        raise ImportError('CPU import sentinel')
    return real_import(name, *args, **kwargs)
builtins.__import__ = stalled_import
try:
    runpy.run_module('offload.dspark.probabilistic.pp4.pair_probe', run_name='__main__')
except ImportError as error:
    assert str(error) == 'CPU import sentinel'
finally:
    d.stop_watchdog()
'''
    log = tmp_path/'import.log'
    with log.open('x') as output:
        subprocess.run([sys.executable, '-c', source], stdout=output, stderr=subprocess.STDOUT, check=True)
    value = log.read_text()
    assert value.index('watchdog armed') < value.index('BEGIN imports torch') < value.index('Timeout (')
    assert 'stalled_import' in value and 'FAIL imports torch / selector / pair contract (ImportError)' in value
    assert 'END imports' not in value


def test_select_phase_names_last_submitted_position(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class Context:
        model = type('Model', (), {'emulated': False})()

        def select(self, ranks: tuple[int, ...], logits: object, prev: object, out: object, position: int) -> None:
            if position == 1:
                raise RuntimeError('CPU first-launch sentinel')

    ctx: Any = Context()
    output = {r: torch.zeros(1, 4) for r in (0, 1)}
    with pytest.raises(RuntimeError, match='CPU first-launch sentinel'):
        p.run(ctx, {}, {}, output, 4, (0, 1))
    value = capsys.readouterr().out
    assert 'END select emulated=False ranks=(0, 1) position=0' in value
    assert 'FAIL select emulated=False ranks=(0, 1) position=1 (RuntimeError)' in value
    assert 'position=2' not in value


def test_main_phases_wrap_guard_peer_order_and_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    events: list[object] = []

    def guard() -> None:
        assert 'BEGIN output / coordinator UUID lock guard' in capsys.readouterr().out
        events.append('guard')

    class Extension:
        def enable_peer(self, rank: int, peer: int) -> None:
            assert f'BEGIN P2P enable rank={rank} peer={peer}' in capsys.readouterr().out
            events.append((rank, peer))

    def qualify() -> list[dict[str, object]]:
        assert 'BEGIN probe mode=correctness' in capsys.readouterr().out
        events.append('qualify')
        return []

    monkeypatch.setattr(p, 'checked_output', lambda path: None)
    monkeypatch.setattr(p, 'guard', guard)
    monkeypatch.setattr(p, 'ext', Extension)
    monkeypatch.setattr(p, 'qualify', qualify)
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: None)
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda n: None)
    out = tmp_path/'result.json'
    monkeypatch.setattr(sys, 'argv', ['probe', '--mode', 'correctness', '--out', str(out)])
    p.main()
    assert events == ['guard', (0, 1), (1, 0), 'qualify']
    assert out.read_text() == '{"cases": []}\n'
    assert 'END write complete result' in capsys.readouterr().out


def test_watchdog_never_arms_on_import() -> None:
    # A normal importer must not inherit a process-wide timer or CUDA action.
    tree = ast.parse(inspect.getsource(p))
    arms = [node for node in tree.body if isinstance(node, ast.If)
            and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == 'start_watchdog' for n in ast.walk(node))]
    assert len(arms) == 1
    assert ast.unparse(arms[0].test) == "__name__ == '__main__'"
    assert 'exit=False' in inspect.getsource(d.start_watchdog)
    assert d.DUMP_INTERVAL_S < 60


def test_diagnostics_keep_numerical_and_driver_timeouts_unchanged() -> None:
    # Compare primary source bytes to the assigned git object, rather than
    # changing old receipts or accepting an approximate kernel equivalent.
    root = Path(p.__file__).resolve().parents[4]
    for name in ('offload/dspark/worker/sampled_fused.py', 'offload/dspark/worker/tp2.py'):
        original = subprocess.check_output(['git', 'show', '399d8b7:'+name], cwd=root)
        actual = (root/name).read_bytes()
        if name.endswith('sampled_fused.py'):
            # Round8's separate HYPOTHESIS changes only host handle loading;
            # every measured kernel byte preceding this class stays identical.
            boundary = b'class FusedContext:'
            assert actual.split(boundary)[0] == original.split(boundary)[0]
        else:
            assert actual == original
    # Hash the new diagnostic dependency; no other driver implementation change.
    driver = 'offload/dspark/probabilistic/pp4/pair_qualify.py'
    original_source = subprocess.check_output(['git', 'show', '399d8b7:'+driver], cwd=root, text=True)
    original_tree = ast.parse(original_source)
    for fn in (q.execute, q.main):
        original_fn = next(node for node in original_tree.body if isinstance(node, ast.FunctionDef) and node.name == fn.__name__)
        assert ast.get_source_segment(original_source, original_fn) == inspect.getsource(fn).rstrip()
    assert 'offload/dspark/probabilistic/pp4/pair_diagnostics.py' in q.SOURCES
    # Diagnostics never insert synchronization into run/capture submissions.
    for submission in (p.run, p.capture_pair):
        calls = [node for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(submission))))
                 if isinstance(node, ast.Call)]
        assert not any(isinstance(call.func, ast.Attribute) and call.func.attr == 'synchronize' for call in calls)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
