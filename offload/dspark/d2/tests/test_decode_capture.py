"""CPU capture tests; GPU copy semantics are source-pinned, not GPU-qualified."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

from d2.tests.test_replay_overlay import FakeSpec, MARK, OVERLAY, _round

PATH = Path(OVERLAY).with_name('decode_capture.py')
spec = importlib.util.spec_from_file_location('d2_capture_test_module', PATH)
assert spec and spec.loader
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)


def send(capture, reqs, rejects=None):
    ib, aux = _round(reqs)
    capture.enqueue(ib, aux, torch.tensor(rejects or [0] * len(reqs)))
    return ib, aux


def test_validated_rows_first_copy_wins_resume(tmp_path):
    c = C.Capture(str(tmp_path))
    send(c, [('cmpl-d2c.pA.0.2-0-abcd', 1, 0, 7), ('unrelated', 9, 0, 10)], [2, 0])
    # Changes both the resume tag's r0 and already written bytes; first rows stay.
    ib, aux = _round([('d2c.pA.1.3', 1, 3, 6)])
    for layer in aux:
        layer[:2] += 1
    c.enqueue(ib, aux, torch.tensor([1]))
    c.close()
    d = tmp_path / 'pA'
    assert not (tmp_path / 'unrelated').exists()
    meta = json.loads((d / 'CAPTURE.json').read_text())
    assert meta['from'] == 2 and meta['taps_source'] == 'decode_capture'
    rows = np.fromfile(d / 'ids.bin', dtype='<i4')
    assert len(rows) == 6  # positions 2..7; rejected rows 8 and 5..6 of first round excluded
    ledger = [json.loads(line) for line in (d / 'ledger.jsonl').read_text().splitlines()]
    assert [(x['first'], x['rows']) for x in ledger] == [(2, 3), (3, 5)]
    assert ledger[1]['dup_rows'] == 2 and ledger[1]['dup_rows_differing'] == 2
    assert sum(x['zero_rows'] for x in ledger) == 0
    assert (d / 'taps.bin').stat().st_size == 6 * C.ROW_BYTES
    assert not (d / 'DONE.json').exists()


def test_restart_retains_original_bytes_and_origin(tmp_path):
    c = C.Capture(str(tmp_path))
    send(c, [('d2c.pA.0.2', 1, 0, 6)])
    c.close()
    before = (tmp_path / 'pA/taps.bin').read_bytes()
    c = C.Capture(str(tmp_path))
    send(c, [('d2c.pA.1.3', 1, 3, 6)])
    c.close()
    assert (tmp_path / 'pA/taps.bin').read_bytes().startswith(before)
    assert c.files['pA'].dup == 3


def test_untagged_does_not_touch_tensors(tmp_path):
    class Poison:
        def __getattr__(self, name):
            raise AssertionError('untagged tensor access: ' + name)
    c = C.Capture(str(tmp_path))
    ib = types.SimpleNamespace(req_ids=['ordinary', 'd2c.pA.0.2.trailing', '../d2c.pA.0.2'], num_reqs=3)
    c.enqueue(ib, Poison(), Poison())
    c.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ['ERROR_COUNT.json']


def test_zero_rows_and_bad_counts_are_visible(tmp_path):
    c = C.Capture(str(tmp_path))
    ib, aux = _round([('d2c.pA.0.0', 1, 0, 3)])
    aux[1][1].zero_()
    c.enqueue(ib, aux, torch.tensor([0]))
    send(c, [('d2c.pB.0.0', 1, 0, 3)], [9])
    c.close()
    assert c.files['pA'].zero == 1
    assert c.errors == 1
    assert 'rejected count' in (tmp_path / 'ERRORS.jsonl').read_text()


def test_overflow_never_waits_for_writer(tmp_path):
    c = C.Capture(str(tmp_path), staging_bytes=4 * 1024 * 1024)
    assert c.budget.acquire(c.budget.limit, blocking=False)
    send(c, [('d2c.pA.0.0', 1, 0, 3)])
    c.budget.release(c.budget.limit)
    c.close()
    assert c.errors == 1 and not (tmp_path / 'pA').exists()


def test_flush_is_durable_and_owned(tmp_path):
    import time
    c = C.Capture(str(tmp_path))
    send(c, [('d2c.pA.0.0', 1, 0, 3)])
    (tmp_path / 'FLUSH.json').write_text(json.dumps({'nonce': 'test-nonce'}))
    deadline = time.monotonic() + 10
    while not (tmp_path / 'FLUSHED.json').exists() and time.monotonic() < deadline:
        time.sleep(.02)
    receipt = json.loads((tmp_path / 'FLUSHED.json').read_text())
    assert receipt['schema'] == C.SCHEMA and receipt['nonce'] == 'test-nonce' and receipt['errors'] == 0
    assert receipt['sequences']['pA']['ledger_sha256'] == C.sha((tmp_path / 'pA/ledger.jsonl').read_bytes())
    c.close()
    assert (tmp_path / 'pA/CAPTURE.json').stat().st_mode & 0o777 == 0o600


def test_wrapper_skips_profile_and_rank1(tmp_path, monkeypatch):
    c = C.Capture(str(tmp_path))
    cls = type('Wrapped', (FakeSpec,), {})
    cls.propose = C.wrapper(c, cls.propose)
    ib, aux = _round([('d2c.pA.0.0', 1, 0, 3)])
    args = (ib, None, None, None, aux, None, torch.tensor([0]), None, None, None, None)
    assert cls().propose(*args, dummy_run=True) == 'drafts'
    assert cls().propose(*args, is_profile=True) == 'drafts'
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 1)
    assert cls().propose(*args) == 'drafts'
    c.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ['ERROR_COUNT.json']


def test_cuda_path_has_no_engine_sync_and_uses_pinned_nonblocking():
    text = PATH.read_text()
    enqueue = text[text.index('    def enqueue('):text.index('    def _owned(')]
    assert 'synchronize(' not in enqueue and '.to("cpu"' not in enqueue
    assert 'pin_memory=cuda' in enqueue and 'non_blocking=True' in enqueue
    assert 'blocking=False' in enqueue
    remote = PATH.parents[1] / 'spark/remote_dspark/speculator.py'
    assert '_d2c_wrap(RemoteDSparkCapacitySpeculator)' in remote.read_text()
    assert '_d2c_wrap(RemoteDSparkLocalSpeculator)' in remote.read_text()


def test_capture_overflow_keeps_capturing_and_staging_bounded(tmp_path):
    c = C.Capture(str(tmp_path), staging_bytes=4 * 1024 * 1024)
    assert c.budget.acquire(c.budget.limit, blocking=False)
    send(c, [('d2c.pA.0.0', 1, 0, 3)])
    c.budget.release(c.budget.limit)
    for _ in range(5):
        send(c, [('d2c.pA.0.0', 1, 0, 3)])
    c.close()
    assert c.errors == 1 and 'pA' in c.files
    assert not c.faulted.is_set() and c.budget.used == 0
    assert json.loads((tmp_path/'ERROR_COUNT.json').read_text())['errors'] == 1


def test_chat_request_id_is_tagged_and_local_nested_wrapper_captures_once(tmp_path):
    c = C.Capture(str(tmp_path))
    base = type('Base', (FakeSpec,), {})
    base.propose = C.wrapper(c, base.propose)
    class Remote(base):
        def propose(self, *args, **kwargs):
            return super().propose(*args, **kwargs)
    # Actual remote method has an explicit signature; retain it in this fake too.
    import functools
    Remote.propose = functools.wraps(FakeSpec.propose)(Remote.propose)
    Remote.propose = C.wrapper(c, Remote.propose)
    ib, aux = _round([('chatcmpl-d2c.pA.0.0-1a2b3c4d', 1, 0, 3)])
    Remote().propose(ib, None, None, None, aux, None, torch.tensor([0]), None, None, None, None)
    c.close()
    assert len((tmp_path / 'pA/ledger.jsonl').read_text().splitlines()) == 1


def test_remote_hook_prefix_is_deployed_production_source():
    import hashlib
    raw = (PATH.parents[1] / 'spark/remote_dspark/speculator.py').read_bytes()
    prefix = raw.split(b'\n# Capacity mode does not call')[0].rstrip(b'\n') + b'\n'
    assert hashlib.sha256(prefix).hexdigest() == '441be6b7d245114974c5dcb307c2efebd29516b958cd363bc70e24bcd73bdcf3'


def test_bounded_file_handles_and_flush_retains_prior_process_sequences(tmp_path):
    import time
    c = C.Capture(str(tmp_path), max_open_sequences=2)
    for i in range(4):
        send(c, [(f'd2c.p{i}.0.0', 1, 0, 3)])
        deadline = time.monotonic()+5
        while not (tmp_path/f'p{i}/ledger.jsonl').exists() and time.monotonic()<deadline:
            time.sleep(.01)
    c.close()
    assert len(c.files) == 2
    # Restart without touching completed prior sequences. Their flushed ledger
    # hashes/file sizes must still appear in the new process's durable receipt.
    c = C.Capture(str(tmp_path), max_open_sequences=2)
    (tmp_path/'FLUSH.json').write_text(json.dumps({'nonce':'after-restart'}))
    deadline = time.monotonic()+5
    while not (tmp_path/'FLUSHED.json').exists() and time.monotonic()<deadline:
        time.sleep(.01)
    flushed = json.loads((tmp_path/'FLUSHED.json').read_text())
    c.close()
    assert flushed['nonce'] == 'after-restart' and flushed['errors'] == 0
    assert sorted(flushed['sequences']) == ['p0','p1','p2','p3']
    for sid, receipt in flushed['sequences'].items():
        assert receipt['ledger_sha256'] == C.sha((tmp_path/sid/'ledger.jsonl').read_bytes())


def test_actual_image_completion_and_chat_id_forms():
    image = Path('/opt/dsv41/data/src-image/vllm')
    processor = (image/'v1/engine/input_processor.py').read_text()
    assert 'request.request_id = f"{request.external_req_id}-{random_uuid():.8}"' in processor
    assert C.TAG.fullmatch('cmpl-d2c.pA.0.72-0-1a2b3c4d')
    assert C.TAG.fullmatch('chatcmpl-d2c.pA.0.72-1a2b3c4d')


def test_adaptive_verification_refuses_capture_and_preserves_propose(tmp_path):
    c = C.Capture(str(tmp_path))
    cls = type('Adaptive', (FakeSpec,), {'enable_adaptive_verification':True})
    cls.propose = C.wrapper(c,cls.propose)
    ib,aux = _round([('d2c.adaptive.0.0.3',1,0,3)])
    assert cls().propose(ib,None,None,None,aux,None,torch.tensor([0]),None,None,None,None) == 'drafts'
    c.close()
    assert not c.files and c.errors == 1
    error = json.loads((tmp_path/'ERRORS.jsonl').read_text())
    assert error['seq_ids'] == ['adaptive'] and 'adaptive verification' in error['error']
    assert c.budget.peak == 0
