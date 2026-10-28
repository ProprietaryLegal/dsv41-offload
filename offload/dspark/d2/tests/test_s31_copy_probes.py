"""S31 independent probes of decode_copy against the REAL CPU capture writer (no host access).

Real Capture writer (07fc7d6 decode_capture.py, CPU tensors) -> decode_copy.FLUSH_CODE -> decode_copy.scan with real
`dd iflag=direct` on the scratch disk -> frozen_copy (local transfer stand-in) -> real decode_dataset.capture_rows on
the copied tree. Synthetic taps only.
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from d2 import decode_dataset as DD
from d2.s26.test_s26_resume import C, R0, step
from offload.dspark.d2 import decode_copy as m

NONCE = 's31-n1'
PROMPT = R0 + 128


def flush(root: Path, nonce: str = NONCE) -> None:
    saved = sys.argv
    sys.argv = ['flush', str(root), nonce]
    try:
        exec(compile(m.FLUSH_CODE, '<flush request>', 'exec'), {})  # noqa: S102 - reviewed module's own code
    finally:
        sys.argv = saved


def generate(root: Path, seqs: tuple[str, ...] = ('pX', 'pY'), decode_steps: int = 4) -> tuple[C.Capture, dict]:
    """Prefill + accepted-draft decode rounds, the way production writes them. Writer stays ALIVE."""
    c = C.Capture(str(root))
    final: dict[str, list[int]] = {}
    for k, seq in enumerate(seqs):
        toks = {p: 1000 + 10*k + p for p in range(PROMPT + 6*decode_steps + 8)}
        rid = f'cmpl-d2c.{seq}.0.{R0}-0-1a2b3c4d'
        step(rid, toks, 0, PROMPT, 0, c)
        pos = PROMPT
        for _ in range(decode_steps):
            step(rid, toks, pos, 6, 3, c)  # 3 valid rows, bonus at pos+3
            pos += 3
        final[seq] = [toks[p] for p in range(pos + 1)]
    deadline = time.monotonic() + 10
    while c.budget.used and time.monotonic() < deadline:
        time.sleep(0.02)
    return c, final


class Local(m.Remote):
    """Same contract as Remote: inventory = the shipped scan (real dd O_DIRECT); copy = local stand-in."""
    def __init__(self, capture: bool, during=None) -> None:
        super().__init__(['never'], capture)
        self.during = during

    def inventory(self, root: str, nonce: str) -> m.Inventory:
        return m.scan(root, self.capture, nonce)

    def copy(self, source: str, destination: Path, entry: m.Record) -> m.Record:
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = subprocess.check_output(['dd', 'if='+source, 'iflag=direct', 'bs=64M', 'status=none'])
        destination.write_bytes(data)
        if self.during:
            self.during()
            self.during = None
        m.require(len(data) == entry['size'], 'verification failed: size')
        if 'sha256' in entry:
            m.require(m.transfer.sha_file(destination) == entry['sha256'], 'verification failed: hash')
        return {'sha256': m.transfer.sha_file(destination), 'size': len(data)}


def tree(tmp_path: Path) -> tuple[Path, Path]:
    cap = tmp_path/'cap'
    cap.mkdir(mode=0o700)
    state = tmp_path/'state'
    (state/'seqs').mkdir(parents=True)
    (state/'LOCK').write_bytes(b'')  # the real generator state has an empty LOCK file
    (state/'seqs/p0.json').write_text('{"synthetic": true}')
    return cap, state


def test_real_writer_tree_flush_scan_copy_and_assembly(tmp_path: Path) -> None:
    cap, state = tree(tmp_path)
    c, final = generate(cap)
    try:
        flush(cap)
        flushed = json.loads((cap/'FLUSHED.json').read_text())
        assert flushed['nonce'] == NONCE and flushed['errors'] == 0 and set(flushed['sequences']) == {'pX', 'pY'}
        time.sleep(1.0)  # writer keeps polling FLUSH.json: must not touch the frozen tree again
        receipt = m.frozen_copy(Local(True), Local(False), str(cap), str(state), tmp_path/'new', NONCE)
    finally:
        c.close()
    assert receipt['frozen'] is True
    copied = tmp_path/'new/capture'
    for seq, tokens in final.items():
        L = len(tokens)
        taps, meta = DD.capture_rows(copied, seq, R0, L - 1, tokens)
        assert taps.shape[0] == L - 1 - R0 and meta['dup_rows_differing'] == 0
    assert {(r['tree'], r['relative']) for r in receipt['files']} >= {('generator', 'LOCK'), ('capture', 'FLUSHED.json')}


def test_tagged_write_after_flush_invalidates_copy(tmp_path: Path) -> None:
    cap, state = tree(tmp_path)
    c, final = generate(cap)
    try:
        flush(cap)
        def late_write() -> None:
            toks = {p: 4242 for p in range(PROMPT + 60)}
            step(f'cmpl-d2c.pX.1.{R0+20}-0-feed', toks, PROMPT + 20, 4, 0, c)
            time.sleep(0.6)
        with pytest.raises(RuntimeError, match='snapshot changed|verification failed'):
            m.frozen_copy(Local(True, late_write), Local(False), str(cap), str(state), tmp_path/'new', NONCE)
    finally:
        c.close()
    assert not (tmp_path/'new/COPY_RECEIPT.json').exists()


def test_new_sequence_after_flush_refused_before_copy(tmp_path: Path) -> None:
    cap, state = tree(tmp_path)
    c, _ = generate(cap)
    try:
        flush(cap)
        toks = {p: 7 for p in range(PROMPT + 8)}
        step(f'cmpl-d2c.pNEW.0.{R0}-0-feed', toks, 0, PROMPT, 0, c)
        time.sleep(0.6)
        with pytest.raises(RuntimeError, match='new or unexpected capture directory'):
            m.frozen_copy(Local(True), Local(False), str(cap), str(state), tmp_path/'new', NONCE)
    finally:
        c.close()
    assert not (tmp_path/'new').exists()


def test_named_error_after_flush_refused(tmp_path: Path) -> None:
    cap, state = tree(tmp_path)
    c, _ = generate(cap)
    try:
        flush(cap)
        c.error('synthetic overflow after flush', ('pX',))
        time.sleep(0.6)
        with pytest.raises(RuntimeError, match='error-count'):
            m.frozen_copy(Local(True), Local(False), str(cap), str(state), tmp_path/'new', NONCE)
    finally:
        c.close()


def test_preexisting_named_error_copies_and_assembly_quarantines_only_that_sequence(tmp_path: Path) -> None:
    cap, state = tree(tmp_path)
    c, final = generate(cap)
    try:
        c.error('synthetic overflow', ('pX',))
        time.sleep(0.6)
        flush(cap)
        m.frozen_copy(Local(True), Local(False), str(cap), str(state), tmp_path/'new', NONCE)
    finally:
        c.close()
    copied = tmp_path/'new/capture'
    with pytest.raises(ValueError, match='recorded errors'):
        DD.capture_rows(copied, 'pX', R0, len(final['pX']) - 1, final['pX'])
    DD.capture_rows(copied, 'pY', R0, len(final['pY']) - 1, final['pY'])


def test_second_flush_same_nonce_refused_new_nonce_acknowledged(tmp_path: Path) -> None:
    cap, _ = tree(tmp_path)
    c, _ = generate(cap)
    try:
        flush(cap)
        with pytest.raises(SystemExit, match='NEW nonce'):
            flush(cap)
        flush(cap, 's31-n2')
        assert json.loads((cap/'FLUSHED.json').read_text())['nonce'] == 's31-n2'
        # the old nonce no longer certifies anything
        with pytest.raises(RuntimeError, match='nonce'):
            m.scan(str(cap), True, NONCE)
    finally:
        c.close()


def test_receipt_transport_label_for_controller_files(tmp_path: Path) -> None:
    """Observation: transfer.copy hardcodes the Spark direct-link transport string for every file it copies."""
    import inspect
    source = inspect.getsource(m.transfer.copy)
    assert '"transport": "10.202.28.2 -> 10.202.28.1 SSH aes128-gcm"' in source
    assert 'controller' in inspect.getsource(m.main)


def ledger_bytes_per_token(tmp_path: Path) -> float:
    cap = tmp_path/'measure'
    cap.mkdir()
    c, final = generate(cap, ('pM',), decode_steps=400)
    c.close()
    rows = [json.loads(line) for line in (cap/'pM/ledger.jsonl').read_text().splitlines()]
    decode = [r for r in rows if r['first'] >= PROMPT]
    size = sum(len(json.dumps(r)) + 1 for r in decode)
    return size / sum(r['rows'] for r in decode)


def test_MEASURE_scan_memory_with_large_ledgers(tmp_path: Path) -> None:
    """MEASURED here (Controller CPU, synthetic): ledger bytes/token at 3 accepted rows per round, and the remote scan's
    peak RSS for a capture tree whose ledgers total the ESTIMATED 1M-token size. The scan keeps every ledger in RAM."""
    per_token = ledger_bytes_per_token(tmp_path)
    total = int(per_token * 1_000_000)
    cap = tmp_path/'big'
    seqs = 8
    flushed = {'schema': 'd2.decode-capture.v1', 'nonce': NONCE, 'errors': 0, 'sequences': {}}
    line = (json.dumps({'synthetic': 'x'*int(per_token*3 - 20)}) + '\n').encode()
    for k in range(seqs):
        d = cap/f's{k}'
        d.mkdir(parents=True)
        (d/'CAPTURE.json').write_text(json.dumps({'schema': 'd2.decode-capture.v1', 'seq': f's{k}',
                                                  'taps_source': 'decode_capture'}))
        with (d/'ledger.jsonl').open('wb') as f:
            for _ in range(total // seqs // len(line)):
                f.write(line)
        (d/'taps.bin').write_bytes(b'')
        (d/'ids.bin').write_bytes(b'')
        flushed['sequences'][f's{k}'] = {'ledger_sha256': m.transfer.sha_file(d/'ledger.jsonl'),
                                         'taps_bytes': 0, 'ids_bytes': 0}
    (cap/'FLUSH.json').write_text(json.dumps({'nonce': NONCE}))
    (cap/'ERROR_COUNT.json').write_text(json.dumps({'errors': 0}))
    (cap/'FLUSHED.json').write_text(json.dumps(flushed))
    ledger_total = sum((cap/f's{k}/ledger.jsonl').stat().st_size for k in range(seqs))
    code = m.SCAN_CODE + ('\nimport resource\nprint("S31_MAXRSS_KIB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'
                          ' file=sys.stderr)\n')
    start = time.monotonic()
    r = subprocess.run(['/usr/bin/python3', '-c', code, str(cap), 'capture', NONCE], capture_output=True, text=True,
                       check=True)
    elapsed = time.monotonic() - start
    rss = int(r.stderr.split('S31_MAXRSS_KIB')[1].split()[0])
    print(f'S31 MEASURED ledger_bytes_per_token={per_token:.1f} ledger_total_bytes={ledger_total} '
          f'scan_peak_rss_MiB={rss/1024:.0f} scan_seconds={elapsed:.1f}')
    shutil.rmtree(cap)
    assert rss * 1024 > ledger_total  # every ledger is held in memory
