"""S33 real-process probes of decode_copy pause/resume (CPU, local only).

The REAL frozen_copy / PauseBudget / Remote / copy_to_x86host.copy run against a synthetic capture tree that satisfies
the shipped scan contract. "ssh" is a local stub that runs the command string in bash, so the shipped SCAN_CODE and the
transport's REMOTE code (real `dd iflag=direct`, real pipe, real rate pacing) execute on Controller scratch. /metrics is a
local HTTP server whose busy flag the probe flips while a payload is mid-transfer. Nothing reaches a host.
"""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import threading
import time
from pathlib import Path

import pytest

from offload.dspark import copy_to_x86host as transfer
from offload.dspark.d2 import decode_copy as m

NONCE = 'n-s33'
PAYLOAD = 96*1024*1024
RATE = 32_000_000          # ~3 s per payload so demand can land mid-file


class Metrics:
    def __init__(self) -> None:
        self.busy = False
        self.missing = False
        probe = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a: object) -> None:
                pass

            def do_GET(self) -> None:
                n = 1.0 if probe.busy else 0.0
                body = '' if probe.missing else (f'vllm:num_requests_running{{engine="0",model_name="m"}} {n}\n'
                                                 'vllm:num_requests_waiting{engine="0",model_name="m"} 0.0\n')
                data = body.encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
        self.url = f'http://127.0.0.1:{self.httpd.server_address[1]}/metrics'
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def capture_tree(root: Path) -> None:
    seq = root/'s1'
    seq.mkdir(parents=True)
    (seq/'taps.bin').write_bytes(os.urandom(PAYLOAD))
    (seq/'ids.bin').write_bytes(os.urandom(4096))
    ledger = b'{"row": 0}\n'*100
    (seq/'ledger.jsonl').write_bytes(ledger)
    (seq/'CAPTURE.json').write_text(json.dumps({'schema': 'd2.decode-capture.v1', 'seq': 's1',
                                                'taps_source': 'decode_capture'}))
    (root/'FLUSH.json').write_text(json.dumps({'nonce': NONCE}))
    (root/'ERROR_COUNT.json').write_text(json.dumps({'errors': 0}))
    (root/'FLUSHED.json').write_text(json.dumps({
        'schema': 'd2.decode-capture.v1', 'nonce': NONCE, 'errors': 0,
        'sequences': {'s1': {'ledger_sha256': hashlib.sha256(ledger).hexdigest(), 'taps_bytes': PAYLOAD,
                             'ids_bytes': 4096}}}))


RELAY = '''#!/usr/bin/python3
import os, signal, subprocess, sys
child = subprocess.Popen(['bash', '-c', sys.argv[-1]], stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
signal.signal(signal.SIGTERM, lambda *a: os._exit(255))
out = sys.stdout.buffer
while True:
    data = child.stdout.read1(1 << 20)
    if not data:
        break
    out.write(data)
    out.flush()
sys.exit(child.wait())
'''


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # ssh-like relay: like the real client, this process alone holds the reader's pipe; the "remote" command runs
    # behind its own pipe, so terminating the client ends the stream at once and the remote dd meets EPIPE.
    stub = tmp_path/'ssh'
    stub.write_text(RELAY)
    stub.chmod(0o700)
    src = tmp_path/'src'
    capture_tree(src)
    gen = tmp_path/'gen'
    gen.mkdir()
    (gen/'budget.json').write_text('{"busy_s": 1.0}')
    metrics = Metrics()
    monkeypatch.setattr(transfer, 'METRICS', metrics.url)
    monkeypatch.setattr(transfer, 'RATE', RATE)
    calls: list[str] = []
    real_inventory = m.Remote.inventory

    def counted(self: m.Remote, root: str, nonce: str) -> m.Inventory:
        calls.append(root)
        return real_inventory(self, root, nonce)
    monkeypatch.setattr(m.Remote, 'inventory', counted)
    yield {'t': tmp_path, 'ssh': [str(stub)], 'src': src, 'gen': gen, 'metrics': metrics, 'calls': calls}
    metrics.close()


def test_demand_mid_file_stops_bytes_waits_30s_continuous_idle_and_restarts_file(rig: dict) -> None:
    t, metrics = rig['t'], rig['metrics']
    dest = t/'dest'
    partial = dest/'capture/s1/taps.bin.s2-partial'
    trace: dict[str, float] = {}

    def driver() -> None:
        # flip to busy when 24 MiB of the payload have landed; record when bytes stop growing
        while not (partial.exists() and partial.stat().st_size >= 24*1024*1024):
            time.sleep(0.005)
        metrics.busy = True
        trace['flip'] = time.monotonic()
        last_size, last_growth = partial.stat().st_size, time.monotonic()
        trace['size_at_flip'] = last_size
        while partial.exists():
            size = partial.stat().st_size
            if size != last_size:
                last_size, last_growth = size, time.monotonic()
            time.sleep(0.005)
        trace['last_growth'] = last_growth
        trace['bytes_after_flip'] = last_size-trace['size_at_flip']
        trace['partial_removed'] = time.monotonic()
        time.sleep(2.0)
        metrics.busy = False
        trace['idle1'] = time.monotonic()
        time.sleep(10.0)
        metrics.busy = True                               # a blip resets the continuous-idle clock
        time.sleep(1.0)
        metrics.busy = False
        trace['idle2'] = time.monotonic()

    th = threading.Thread(target=driver, daemon=True)
    th.start()
    capture = m.Remote(rig['ssh'], True)
    state = m.Remote(rig['ssh'], False)
    started = time.monotonic()
    receipt = m.frozen_copy(capture, state, str(rig['src']), str(rig['gen']), dest, NONCE,
                            pause=m.PauseBudget())     # production defaults: 30 s idle, 4 h cap
    th.join(timeout=5)
    total = time.monotonic()-started
    stop_latency = trace['last_growth']-trace['flip']
    policy = receipt['pause_policy']
    taps = [f for f in receipt['files'] if f['relative'] == 's1/taps.bin'][0]
    whole = hashlib.sha256((rig['src']/'s1/taps.bin').read_bytes()).hexdigest()
    print('MEASURED stop latency s', round(stop_latency, 3), 'bytes after flip', trace['bytes_after_flip'],
          'pause wait s', round(policy['total_wait_seconds'], 2), 'inventory calls', len(rig['calls']),
          'total s', round(total, 1))
    assert stop_latency <= 0.75, stop_latency                      # 0.5 s monitor + metrics GET + pipe drain
    assert policy['continuous_idle_seconds'] == 30 and policy['total_wait_cap_seconds'] == 4*3600
    assert len(policy['events']) == 1
    # resume only after 30 s of continuous idle measured from the end of the blip
    assert policy['total_wait_seconds'] >= (trace['idle2']-trace['partial_removed'])+30-0.6
    assert taps['resumed_from_bytes'] == 0 and taps['bytes_transferred'] == PAYLOAD and taps['sha256'] == whole
    assert (dest/'capture/s1/taps.bin').read_bytes() == (rig['src']/'s1/taps.bin').read_bytes()
    # 2 'before' + one revalidation of every established inventory after the pause + 2 'after'
    assert len(rig['calls']) == 2+2+2, rig['calls']
    assert (dest/'COPY_RECEIPT.json').exists()


def test_snapshot_change_during_pause_aborts_without_receipt(rig: dict) -> None:
    t, metrics, src = rig['t'], rig['metrics'], rig['src']
    dest = t/'dest'
    partial = dest/'capture/s1/taps.bin.s2-partial'

    def driver() -> None:
        while not (partial.exists() and partial.stat().st_size >= 8*1024*1024):
            time.sleep(0.005)
        metrics.busy = True
        while partial.exists():
            time.sleep(0.005)
        with (src/'s1/ids.bin').open('r+b') as fh:           # same-size tagged write while paused
            fh.write(b'\x01'*16)
        metrics.busy = False
    threading.Thread(target=driver, daemon=True).start()
    with pytest.raises(RuntimeError, match='snapshot changed during pause: capture'):
        m.frozen_copy(m.Remote(rig['ssh'], True), m.Remote(rig['ssh'], False), str(src), str(rig['gen']), dest,
                      NONCE, pause=m.PauseBudget(idle_seconds=1))
    assert not (dest/'COPY_RECEIPT.json').exists()


def test_missing_gauges_mid_file_is_a_hard_abort_not_a_pause(rig: dict) -> None:
    t, metrics = rig['t'], rig['metrics']
    dest = t/'dest'
    partial = dest/'capture/s1/taps.bin.s2-partial'

    def driver() -> None:
        while not (partial.exists() and partial.stat().st_size >= 8*1024*1024):
            time.sleep(0.005)
        metrics.missing = True
    threading.Thread(target=driver, daemon=True).start()
    budget = m.PauseBudget(idle_seconds=1)
    with pytest.raises(transfer.CopyInterrupted) as info:
        m.frozen_copy(m.Remote(rig['ssh'], True), m.Remote(rig['ssh'], False), str(rig['src']), str(rig['gen']), dest,
                      NONCE, pause=budget)
    assert not isinstance(info.value.__cause__, m.ProductionBusy) and 'missing production gauges' in str(info.value.__cause__)
    assert budget.events == [] and not (dest/'COPY_RECEIPT.json').exists()
    assert partial.exists()                                       # a hard abort keeps the partial for inspection


def test_cap_exhaustion_while_busy_aborts_without_receipt(rig: dict) -> None:
    rig['metrics'].busy = True
    dest = rig['t']/'dest'
    started = time.monotonic()
    with pytest.raises(RuntimeError, match='production pause total wait cap exhausted'):
        m.frozen_copy(m.Remote(rig['ssh'], True), m.Remote(rig['ssh'], False), str(rig['src']), str(rig['gen']), dest,
                      NONCE, pause=m.PauseBudget(cap_seconds=3))
    assert time.monotonic()-started < 10
    assert not dest.exists()                                       # demand before the first inventory: no tree
