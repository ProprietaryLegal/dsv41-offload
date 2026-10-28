# SPDX-License-Identifier: Apache-2.0
"""The vLLM-facing speculators with stubbed vLLM modules (policies, validation, overlay)."""

import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import sim
import vllm_stubs

vllm_stubs.install()

from remote_dspark import wire  # noqa: E402
from remote_dspark import speculator as S  # noqa: E402
from remote_dspark.mock_worker import TcpStubServer, WorkerCore, fake_draft  # noqa: E402

ID = bytes(range(32))
OVERLAY = (Path(__file__).resolve().parents[1]
           / "overlay/vllm/v1/worker/gpu/spec_decode/__init__.py")
IMAGE = Path("/opt/dsv41/data/src-image/vllm/v1/worker/gpu/spec_decode/__init__.py")


def hello():
    return wire.Hello(hidden=sim.H, row_bytes=sim.N_TAPS * sim.H * 2, drafter_identity=ID)


@pytest.fixture
def server():
    srv = TcpStubServer(WorkerCore(hello()), "127.0.0.1", 0).start()
    yield srv
    srv.stop()


@pytest.fixture
def remote_env(monkeypatch, server):
    def setenv(policy, **kw):
        monkeypatch.setenv("VLLM_DSPARK_REMOTE", f"tcp://127.0.0.1:{server.port}")
        monkeypatch.setenv("VLLM_DSPARK_REMOTE_IDENTITY", ID.hex())
        monkeypatch.setenv("VLLM_DSPARK_REMOTE_POLICY", policy)
        monkeypatch.setenv("VLLM_DSPARK_REMOTE_TIMEOUT_MS", "200")
        monkeypatch.setenv("VLLM_DSPARK_REMOTE_BACKOFF_MS", "2")
        monkeypatch.setenv("VLLM_DSPARK_REMOTE_BACKOFF_MAX_MS", "4")
        monkeypatch.setenv("VLLM_DSPARK_REMOTE_LOG_EVERY", "0")
        monkeypatch.setenv("VLLM_DSPARK_REMOTE_STATS_FILE", "")
        monkeypatch.setenv("VLLM_DSPARK_REMOTE_STRICT_START", "1")
        for k, v in kw.items():
            monkeypatch.setenv(k, v)
        vllm_stubs.CALLS.clear()
        return server
    return setenv


def propose(spec, sd, dummy=False):
    return spec.propose(sd.batch, None, None, None, sd.taps, sd.num_sampled, sd.num_rejected,
                        sd.last_sampled, sd.next_prefill_tokens, None, None, dummy_run=dummy)


def truth_drafts(eng, sd):
    ids = np.zeros((sd.batch.num_reqs, sim.N), np.int64)
    conf = np.zeros((sd.batch.num_reqs, sim.N), np.float32)
    for i, r in enumerate(sd.reqs):
        if sd.need_draft[i]:
            d, c = fake_draft(eng.expected_window(r), r.anchor, r.computed, sim.N, sim.VOCAB)
            ids[i], conf[i] = d, c
    return ids, conf


def test_capacity_speculator(remote_env):
    remote_env("capacity")
    spec = S.init_remote_dspark_speculator(vllm_stubs.vllm_config(), torch.device("cpu"))
    assert isinstance(spec, S.RemoteDSparkCapacitySpeculator)
    assert not isinstance(spec, vllm_stubs.DSparkSpeculator)   # no local drafter at all
    assert spec.num_query_per_req == 5 and spec.draft_logits is None
    assert spec.use_confidence_head
    dummy_eng = sim.Engine(seed=99)              # dummy runs are not real engine steps
    dummy_eng.new_request(0, prompt_len=30)
    out = propose(spec, dummy_eng.step(), dummy=True)
    eng = sim.Engine(seed=2)
    eng.new_request(0, prompt_len=300)
    eng.new_request(1, prompt_len=20)
    assert not out.any()                                   # dummy runs: no network
    assert spec.remote.rounds == 0
    for _ in range(30):
        sd = eng.step()
        out = propose(spec, sd)
        ids, conf = truth_drafts(eng, sd)
        assert torch.equal(out, torch.from_numpy(ids))
        assert torch.equal(spec.draft_token_confidence_probs[:sd.batch.num_reqs],
                           torch.from_numpy(conf))
    spec.remote.client.close()


def test_capacity_on_fail_raise_needs_worker_at_start(monkeypatch, remote_env):
    remote_env("capacity", VLLM_DSPARK_REMOTE_ON_FAIL="raise",
               VLLM_DSPARK_REMOTE="tcp://127.0.0.1:1")
    with pytest.raises(RuntimeError, match="failed at startup"):
        S.init_remote_dspark_speculator(vllm_stubs.vllm_config(), torch.device("cpu"))


def test_capacity_on_fail_raise_raises_on_failed_round(remote_env):
    srv = remote_env("capacity", VLLM_DSPARK_REMOTE_ON_FAIL="raise")
    spec = S.init_remote_dspark_speculator(vllm_stubs.vllm_config(), torch.device("cpu"))
    eng = sim.Engine(seed=4)
    eng.new_request(0, prompt_len=50)
    propose(spec, eng.step())
    srv.core.inject.frame_status_once = wire.ST_INTERNAL
    with pytest.raises(RuntimeError, match="ON_FAIL=raise"):
        propose(spec, eng.step())
    spec.remote.client.close()


def test_resilience_paths(remote_env):
    srv = remote_env("resilience", VLLM_DSPARK_REMOTE_DRAFT_MS="3.5")
    spec = S.init_remote_dspark_speculator(vllm_stubs.vllm_config(), torch.device("cpu"))
    assert isinstance(spec, S.RemoteDSparkLocalSpeculator)
    eng = sim.Engine(seed=6)
    eng.new_request(0, prompt_len=200)
    eng.new_request(2, prompt_len=200)
    # warm up through prefill
    for _ in range(8):
        propose(spec, eng.step(schedule=[0, 2]))
    vllm_stubs.CALLS.clear()
    sd = eng.step(schedule=[0, 2])
    out = propose(spec, sd)
    ids, _ = truth_drafts(eng, sd)
    assert torch.equal(out, torch.from_numpy(ids))          # remote drafts
    assert vllm_stubs.CALLS.get("local_propose", 0) == 0
    assert vllm_stubs.CALLS["precompute_context_kv"] == 1   # local context kept in sync
    assert vllm_stubs.CALLS["prepare_dflash_inputs"] == 1
    assert spec.last_precompute == (0, sd.batch.num_tokens)
    # one request fails on the worker: local drafts for it, remote for the other
    srv.core.inject.req_status_once[0] = wire.RS_INTERNAL
    vllm_stubs.CALLS.clear()
    sd = eng.step(schedule=[0, 2])
    out = propose(spec, sd)
    ids, _ = truth_drafts(eng, sd)
    assert vllm_stubs.CALLS["local_propose"] == 1
    assert torch.equal(out[0], torch.full((5,), 7, dtype=torch.int64))
    assert torch.equal(out[1], torch.from_numpy(ids[1]))
    # whole round fails: local for everything
    srv.core.inject.drop_next_step = True
    sd = eng.step(schedule=[0, 2])
    out = propose(spec, sd)
    assert torch.equal(out, torch.full((2, 5), 7, dtype=torch.int64))
    assert spec.stats.snapshot()["counters"]["local_fallback_rounds"] == 2
    # recovery: remote again, exact after resync
    t = time.monotonic() + 3
    while not spec.remote.client.ready and time.monotonic() < t:
        time.sleep(0.002)
    sd = eng.step(schedule=[0, 2])
    out = propose(spec, sd)
    ids, _ = truth_drafts(eng, sd)
    assert torch.equal(out, torch.from_numpy(ids))
    assert os.environ["VLLM_DSPARK_DRAFT_MS"] == "3.5"
    spec.remote.client.close()


def test_shadow_compares_and_keeps_local(remote_env):
    remote_env("shadow")
    spec = S.init_remote_dspark_speculator(vllm_stubs.vllm_config(), torch.device("cpu"))
    eng = sim.Engine(seed=8)
    eng.new_request(1, prompt_len=150)
    eng.new_request(3, prompt_len=40)
    mismatches = 0
    for step in range(25):
        sd = eng.step(schedule=[1, 3])
        ids, conf = truth_drafts(eng, sd)
        local = torch.from_numpy(ids.copy())
        if step == 20 and sd.need_draft[0]:
            local[0, 2] += 1        # the local drafter disagrees at position 2
            mismatches += 1
        spec.next_local = local
        spec.next_local_conf = torch.from_numpy(conf)
        out = propose(spec, sd)
        assert torch.equal(out, local)                       # local drafts are used
    sh = spec.stats.snapshot()["shadow"]
    assert sh["compared_requests"] > 20
    assert sh["first_mismatch_hist"][2] == mismatches
    assert sh["first_mismatch_hist"][5] == sh["compared_requests"] - mismatches
    if mismatches:
        assert sh["examples"][0]["first_mismatch"] == 2 and "remote_margin" in sh["examples"][0]
    spec.remote.client.close()


def test_shadow_use_remote(remote_env):
    remote_env("shadow", VLLM_DSPARK_REMOTE_SHADOW_USE="remote")
    spec = S.init_remote_dspark_speculator(vllm_stubs.vllm_config(), torch.device("cpu"))
    eng = sim.Engine(seed=10)
    eng.new_request(0, prompt_len=60)
    for _ in range(10):
        sd = eng.step(schedule=[0])
        out = propose(spec, sd)
        ids, _ = truth_drafts(eng, sd)
        if sd.need_draft[0]:
            assert torch.equal(out, torch.from_numpy(ids))
    spec.remote.client.close()


@pytest.mark.parametrize("over,match", [
    ({"draft_sample": "probabilistic"}, "greedy"),
    ({"method": "eagle"}, "only dspark"),
])
def test_validation(remote_env, over, match):
    remote_env("capacity")
    with pytest.raises(ValueError, match=match):
        S.init_remote_dspark_speculator(vllm_stubs.vllm_config(**over), torch.device("cpu"))


def test_depth_policy_requires_remote_cost(remote_env):
    remote_env("capacity")
    vllm_stubs.DEPTH_POLICY_ON["v"] = True
    try:
        with pytest.raises(ValueError, match="REMOTE_DRAFT_MS"):
            S.init_remote_dspark_speculator(vllm_stubs.vllm_config(), torch.device("cpu"))
    finally:
        vllm_stubs.DEPTH_POLICY_ON["v"] = False


def test_depth_policy_requires_both_remote_costs(remote_env, monkeypatch):
    """S5 #4 at the speculator: REMOTE_DRAFT_MS alone would leave the live ROUND_MS (which
    includes the local drafter) under a remote draft cost; refused, nothing overridden."""
    remote_env("capacity", VLLM_DSPARK_DEPTH_POLICY="conf", VLLM_DSPARK_REMOTE_DRAFT_MS="4",
               VLLM_DSPARK_ROUND_MS="33.26,36.33,38.91,41.97,45.31,48.03",
               VLLM_DSPARK_DRAFT_MS="0")
    monkeypatch.delenv("VLLM_DSPARK_REMOTE_ROUND_MS", raising=False)
    vllm_stubs.DEPTH_POLICY_ON["v"] = True
    try:
        with pytest.raises(ValueError, match="missing VLLM_DSPARK_REMOTE_ROUND_MS"):
            S.init_remote_dspark_speculator(vllm_stubs.vllm_config(), torch.device("cpu"))
    finally:
        vllm_stubs.DEPTH_POLICY_ON["v"] = False
    assert os.environ["VLLM_DSPARK_ROUND_MS"] == "33.26,36.33,38.91,41.97,45.31,48.03"
    assert os.environ["VLLM_DSPARK_DRAFT_MS"] == "0"
    remote_env("capacity", VLLM_DSPARK_DEPTH_POLICY="confr:30,33,36",
               VLLM_DSPARK_REMOTE_ROUND_MS="29.66,32.73,35.31,38.63,42.43,44.55")
    vllm_stubs.DEPTH_POLICY_ON["v"] = True
    try:
        with pytest.raises(ValueError, match="confr"):
            S.init_remote_dspark_speculator(vllm_stubs.vllm_config(), torch.device("cpu"))
    finally:
        vllm_stubs.DEPTH_POLICY_ON["v"] = False


def test_n_out_from_dynamic_schedule(remote_env, monkeypatch):
    remote_env("capacity", VLLM_DSPARK_DEPTH_POLICY="conf", VLLM_DSPARK_REMOTE_DRAFT_MS="4",
               VLLM_DSPARK_REMOTE_ROUND_MS="29.66,32.73,35.31,38.63,42.43,44.55",
               VLLM_DSPARK_DRAFT_MS="9")          # restored by monkeypatch after the override
    monkeypatch.delenv("VLLM_DSPARK_DEPTH_FILE", raising=False)
    vcfg = vllm_stubs.vllm_config(schedule=[[1, 2, 5], [3, 4, 4]])
    vllm_stubs.DEPTH_POLICY_ON["v"] = True
    try:
        spec = S.init_remote_dspark_speculator(vcfg, torch.device("cpu"))
    finally:
        vllm_stubs.DEPTH_POLICY_ON["v"] = False
    assert spec.remote.n_out_by_reqs == [0, 5, 5, 4, 4]
    eng = sim.Engine(seed=5)
    for s_ in range(3):
        eng.new_request(s_, prompt_len=40 + s_)
    for _ in range(20):
        sd = eng.step()
        out = propose(spec, sd)
        ids, conf = truth_drafts(eng, sd)
        c = 4 if sd.batch.num_reqs >= 3 else 5
        assert torch.equal(out[:, :c], torch.from_numpy(ids[:, :c]))
        assert not out[:, c:].any()
    spec.remote.client.close()
    for policy, extra in (("shadow", {}), ("capacity", {"VLLM_DSPARK_DEPTH_POLICY": "fixed:4"}),
                          ("capacity", {"VLLM_DSPARK_DEPTH_FILE": "/tmp/x"})):
        remote_env(policy, **extra)
        vllm_stubs.DEPTH_POLICY_ON["v"] = True
        try:
            spec = S.init_remote_dspark_speculator(vcfg, torch.device("cpu"))
        finally:
            vllm_stubs.DEPTH_POLICY_ON["v"] = False
        assert spec.remote.n_out_by_reqs is None, (policy, extra)
        spec.remote.client.close()


def _load_overlay():
    spec = importlib.util.spec_from_file_location("overlay_spec_decode", OVERLAY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_overlay_is_image_plus_guarded_branch():
    if not IMAGE.exists():
        pytest.skip("image source copy not present")
    img = IMAGE.read_text().splitlines()
    ovl = OVERLAY.read_text().splitlines()
    added = [line for line in ovl if line not in img]
    removed = [line for line in img if line not in ovl]
    assert removed == []
    assert added == [
        "import os",
        "        # DSV41-OFFLOAD O3: remote drafter on x86host, only when VLLM_DSPARK_REMOTE is set",
        "        # (offload/dspark/CONTRACT.md); unset = the image's code path below, unchanged.",
        '        if os.environ.get("VLLM_DSPARK_REMOTE", "").strip():',
        "            from vllm.v1.worker.gpu.spec_decode.remote_dspark.speculator import (",
        "                init_remote_dspark_speculator,",
        "            return init_remote_dspark_speculator(vllm_config, device)",
    ]


def test_overlay_dispatch(monkeypatch, remote_env):
    import sys

    m = _load_overlay()
    cfg = vllm_stubs.vllm_config()
    monkeypatch.delenv("VLLM_DSPARK_REMOTE", raising=False)
    local = m.init_speculator(cfg, torch.device("cpu"))
    assert type(local) is vllm_stubs.DSparkSpeculator          # image path, unchanged
    remote_env("capacity")
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.spec_decode.remote_dspark.speculator", S)
    rem = m.init_speculator(cfg, torch.device("cpu"))
    assert isinstance(rem, S.RemoteDSparkCapacitySpeculator)
    rem.remote.client.close()


def test_async_needs_cuda_falls_back(remote_env):
    remote_env("capacity", VLLM_DSPARK_REMOTE_ASYNC="1")
    spec = S.init_remote_dspark_speculator(vllm_stubs.vllm_config(), torch.device("cpu"))
    assert not spec.remote.async_mode
    assert spec.stats.snapshot()["async_fallback_reason"] == "not cuda"
    eng = sim.Engine(seed=31)
    eng.new_request(0, prompt_len=40)
    for _ in range(5):
        sd = eng.step()
        out = propose(spec, sd)
        ids, _ = truth_drafts(eng, sd)
        assert torch.equal(out, torch.from_numpy(ids))
    spec.remote.client.close()
