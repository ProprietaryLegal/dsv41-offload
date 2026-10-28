"""Byte staging, prefill trimming, scoped refusals and generator PAUSE."""
import json
import time
from pathlib import Path

import pytest
import torch

from d2 import decode_dataset as DD, generate as G
from d2.decode_control import ErrorMonitor, error_count
from d2.tests.test_decode_capture import C, send
from d2.tests.test_replay_overlay import _round
from d2.tests.test_decode_dataset import fixture


def test_prefill_stages_only_tail_using_cpu_offsets(tmp_path):
    c = C.Capture(str(tmp_path), staging_bytes=4*1024*1024)
    ib, aux = _round([('d2c.prefill.0.1872.2000', 1, 0, 1900)])
    ib.is_prefilling_np = [True]
    ib.num_computed_prefill_tokens_np = [0]
    c.enqueue(ib, aux, torch.tensor([0]))
    c.close()
    assert c.errors == 0
    assert c.budget.peak == 28*(C.ROW_BYTES+12)+8
    assert c.budget.used == 0
    assert (tmp_path/'prefill/taps.bin').stat().st_size == 28*C.ROW_BYTES


def test_prefill_before_tail_allocates_nothing(tmp_path):
    c = C.Capture(str(tmp_path), staging_bytes=1)
    ib, aux = _round([('d2c.prefill.0.1872.2000', 1, 0, 100)])
    ib.is_prefilling_np = [True]
    ib.num_computed_prefill_tokens_np = [0]
    c.enqueue(ib, aux, torch.tensor([0]))
    c.close()
    assert c.errors == 0 and c.budget.peak == 0 and not c.files


def test_byte_overflow_names_only_dropped_sequence_and_recovers(tmp_path):
    size = 3*(C.ROW_BYTES+12)+8
    c = C.Capture(str(tmp_path), staging_bytes=size)
    assert c.budget.acquire(size, blocking=False)
    send(c, [('d2c.dropped.0.0.3', 1, 0, 3)])
    c.budget.release(size)
    send(c, [('d2c.later.0.0.3', 1, 0, 3)])
    c.close()
    errors = [json.loads(line) for line in (tmp_path/'ERRORS.jsonl').read_text().splitlines()]
    assert errors[0]['seq_ids'] == ['dropped'] and c.errors == 1
    assert 'later' in c.files and not c.faulted.is_set()
    assert c.budget.peak == size and c.budget.used == 0
    assert error_count((tmp_path/'ERROR_COUNT.json').read_text()) == 1
    c = C.Capture(str(tmp_path), staging_bytes=size)
    c.close()
    assert c.errors == 1 and not c.faulted.is_set()


@pytest.mark.parametrize('seqs,refuses', [(['other'],False), (['pA'],True), (None,True), ([],True)])
def test_assembler_scopes_error_but_requires_flush_error_count(tmp_path, seqs, refuses):
    r, root, _, _ = fixture(tmp_path)
    (root/'ERRORS.jsonl').write_text(json.dumps({'schema':C.SCHEMA,'seq_ids':seqs,'error':'injected'})+'\n')
    receipt = json.loads((root/'FLUSHED.json').read_text())
    receipt['errors'] = 1
    (root/'FLUSHED.json').write_text(json.dumps(receipt))
    if refuses:
        with pytest.raises(ValueError,match='recorded errors'):
            DD.capture_rows(root,'pA',r.r0,r.L-1,r.tokens.tolist())
    else:
        DD.capture_rows(root,'pA',r.r0,r.L-1,r.tokens.tolist())
        receipt['errors'] = 2
        (root/'FLUSHED.json').write_text(json.dumps(receipt))
        with pytest.raises(ValueError,match='flush receipt'):
            DD.capture_rows(root,'pA',r.r0,r.L-1,r.tokens.tolist())


def test_sync_phases_are_separated_for_simultaneous_sequences(tmp_path):
    c = C.Capture(str(tmp_path))
    for i in range(3):
        send(c, [(f'd2c.s{i}.0.0.3',1,0,3)])
    deadline = time.monotonic()+5
    while (len(c.files)<3 or c.budget.used) and time.monotonic()<deadline:
        time.sleep(.01)
    phases = [c.files[f's{i}'].since_sync for i in range(3)]
    c.close()
    assert phases[1]-phases[0] == phases[2]-phases[1] == C.SYNC_BYTES//3


def test_first_error_latches_existing_pause_path(tmp_path):
    status = {'schema':'d2.decode-error-count.v1','errors':0}
    pause = tmp_path/'PAUSE'
    monitor = ErrorMonitor(lambda:json.dumps(status),pause,every_s=.01)
    monitor.start()
    assert not pause.exists()
    status['errors'] = 1
    deadline = time.monotonic()+2
    while not pause.exists() and time.monotonic()<deadline:
        time.sleep(.01)
    monitor.close()
    assert pause.read_text() == 'capture errors=1\n'
    driver = object.__new__(G.Driver)
    from types import SimpleNamespace
    driver.a = SimpleNamespace(pause_file=str(pause))
    assert driver.paused() == 'pause_file'
    assert monitor.poll() and pause.read_text() == 'capture errors=1\n'


@pytest.mark.parametrize('raw', ['{}','{"schema":"d2.decode-error-count.v1","errors":true}', 'not JSON'])
def test_bad_telemetry_pauses(tmp_path,raw):
    monitor = ErrorMonitor(lambda:raw,tmp_path/'PAUSE')
    assert monitor.poll() and 'telemetry unavailable' in (tmp_path/'PAUSE').read_text()
