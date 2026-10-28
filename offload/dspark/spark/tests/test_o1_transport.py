# SPDX-License-Identifier: Apache-2.0
"""The proxy over lane O1's ofl library (TCP backend on 127.0.0.1; RDMA needs the links).

Needs libofl.so + ofl.py: VLLM_DSPARK_REMOTE_OFL_DIR, else offload/dspark/spark/.build/ofl
(built from O1's offload/transport sources). Skipped when absent.
"""

import os
import socket
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from remote_dspark import wire
from remote_dspark.config import RemoteConfig
from remote_dspark.mock_worker import WorkerCore, fake_draft, serve_ofl
from remote_dspark.round import STATUS_OK, STATUS_ROUND_FAILED, RemoteRound

import sim

OFL_DIR = os.environ.get("VLLM_DSPARK_REMOTE_OFL_DIR") or str(
    Path(__file__).resolve().parents[1] / ".build/ofl")
HAVE_OFL = (Path(OFL_DIR) / "libofl.so").exists() and (Path(OFL_DIR) / "ofl.py").exists()
pytestmark = pytest.mark.skipif(not HAVE_OFL, reason=f"no libofl.so/ofl.py in {OFL_DIR}")

ID = bytes(range(32))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def hello():
    return wire.Hello(hidden=sim.H, row_bytes=sim.N_TAPS * sim.H * 2, drafter_identity=ID)


def test_round_over_ofl_tcp(monkeypatch):
    monkeypatch.setenv("VLLM_DSPARK_REMOTE_OFL_DIR", OFL_DIR)
    port = free_port()
    core = WorkerCore(hello())
    srv = serve_ofl(core, "127.0.0.1", port, OFL_DIR, rdma=False)
    try:
        cfg = RemoteConfig.from_env({
            "VLLM_DSPARK_REMOTE": f"o1tcp://127.0.0.1:{port}?bind=127.0.0.1",
            "VLLM_DSPARK_REMOTE_IDENTITY": ID.hex(),
            "VLLM_DSPARK_REMOTE_POLICY": "capacity",
            "VLLM_DSPARK_REMOTE_TIMEOUT_MS": "300",
            "VLLM_DSPARK_REMOTE_BACKOFF_MS": "2", "VLLM_DSPARK_REMOTE_BACKOFF_MAX_MS": "4",
            "VLLM_DSPARK_REMOTE_LOG_EVERY": "0", "VLLM_DSPARK_REMOTE_STATS_FILE": ""})
        rnd = RemoteRound(cfg, hello(), device=torch.device("cpu"), tp_rank=0,
                          broadcast_fn=None, background=True)
        t = time.monotonic() + 5
        while not rnd.client.ready and time.monotonic() < t:
            time.sleep(0.005)
        assert rnd.client.ready, rnd.client.last_error
        eng = sim.Engine(seed=21)
        eng.new_request(0, prompt_len=700)
        eng.new_request(1, prompt_len=60)
        exact = 0
        for step in range(60):
            if step == 30:
                core.inject.delay_ms = 600          # > 300 ms: ofl times out, breaks the QP
            if step == 31:
                core.inject.delay_ms = 0
            sd = eng.step()
            res = rnd.run(sd.batch, sd.taps, sd.num_sampled, sd.num_rejected,
                          sd.last_sampled, sd.next_prefill_tokens)
            st = res.read_status()
            if step == 30:
                assert st == STATUS_ROUND_FAILED
            ok = res.ok_mask.numpy()
            for i, r in enumerate(sd.reqs):
                if sd.need_draft[i] and ok[i]:
                    d, c = fake_draft(eng.expected_window(r), r.anchor, r.computed, sim.N,
                                      sim.VOCAB)
                    assert np.array_equal(res.drafts.numpy()[i], d)
                    exact += 1
            if st != STATUS_OK:
                time.sleep(0.05)
        c = rnd.stats.snapshot()["counters"]
        assert c.get("fallback_timeout", 0) == 1
        assert c.get("resync_rounds", 0) >= 1
        assert exact > 40
        rnd.client.close()
    finally:
        srv.stop()
