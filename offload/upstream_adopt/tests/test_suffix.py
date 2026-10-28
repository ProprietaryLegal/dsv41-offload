"""Real gather, ring, RoundCore and loopback reference worker; synthetic rows only."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Iterator, cast

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'offload/upstream_adopt/overlay/offload/dspark/spark'))
sys.path.insert(0, str(ROOT / 'offload/dspark/spark/tests'))

from remote_dspark import roundcore, wire
from remote_dspark.config import RemoteConfig
from remote_dspark.mock_worker import TcpStubServer, WorkerCore, fake_draft
from remote_dspark.round import RemoteRound, RoundResult
from remote_dspark.roundcore import STATUS_OK, STATUS_ROUND_FAILED
from remote_dspark.stats import Stats
from remote_dspark.suffix_provenance import SuffixAudit, SuffixRow, agree
import sim


def hello() -> wire.Hello:
    return wire.Hello(hidden=sim.H, row_bytes=sim.N_TAPS * sim.H * 2, drafter_identity=bytes(range(32)))


def cfg_for(port: int) -> RemoteConfig:
    return RemoteConfig.from_env({"VLLM_DSPARK_REMOTE": f"tcp://127.0.0.1:{port}",
        "VLLM_DSPARK_REMOTE_IDENTITY": bytes(range(32)).hex(),
        "VLLM_DSPARK_REMOTE_POLICY": "capacity", "VLLM_DSPARK_REMOTE_TIMEOUT_MS": "150",
        "VLLM_DSPARK_REMOTE_LOG_EVERY": "0", "VLLM_DSPARK_REMOTE_STATS_FILE": ""})


@pytest.fixture
def remote(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TcpStubServer, RemoteRound]]:
    monkeypatch.setattr(roundcore, '_REPLAY_SUFFIX_REQUIRED', True)
    srv = TcpStubServer(WorkerCore(hello()), '127.0.0.1', 0).start()
    rnd = RemoteRound(cfg_for(srv.port), hello(), device=torch.device('cpu'), tp_rank=0,
                      broadcast_fn=None, background=False)
    try:
        assert rnd.client is not None
        assert rnd.client.connect_now()
        yield srv, rnd
    finally:
        rnd.close()
        srv.stop()


def run(rnd: RemoteRound, sd: sim.StepData, kept: int) -> RoundResult:
    setattr(sd.batch, "replay_tap_lens_np", np.array([kept]))
    q = sd.batch.num_tokens
    # Poison discarded rows, so certifying them gives a wrong fake draft.
    for t in sd.taps:
        t[:q-kept] = 0
    return rnd.run(sd.batch, sd.taps, sd.num_sampled, sd.num_rejected,
                   sd.last_sampled, sd.next_prefill_tokens)


@pytest.mark.parametrize('disconnect', ['before_final', 'after_final'])
@pytest.mark.parametrize('fused', [False, True])
def test_disconnect_around_final_never_resyncs_fake_rows(
    remote: tuple[TcpStubServer, RemoteRound], disconnect: str, fused: bool,
) -> None:
    srv, rnd = remote
    assert rnd.gatherer is not None and rnd.slots is not None and rnd.client is not None
    rnd.gatherer.fused = fused
    srv.core.keep_step_log = True
    eng = sim.Engine(seed=94, chunk_sizes=(4096,))
    r = eng.new_request(0, prompt_len=8192)
    first = eng.step(schedule=[0])
    assert run(rnd, first, 1).read_status() == STATUS_OK
    assert rnd.slots[0].rng.rows_for_anchor(4096) == 1
    assert not any(e['flags'] & wire.RF_NEED_DRAFT for e in srv.core.step_log)  # no NEED_DRAFT on intermediate
    if disconnect == 'before_final':
        rnd.client.new_session('CPU qualification before final suffix')
        assert rnd.client is not None
        assert rnd.client.connect_now()
    final = eng.step(schedule=[0])
    result = run(rnd, final, 128)
    assert result.read_status() == STATUS_OK and result.ok_mask[0] == 1
    want, conf = fake_draft(eng.expected_window(r), r.anchor, r.computed, sim.N, sim.VOCAB)
    assert np.array_equal(result.drafts.numpy()[0], want)
    assert np.array_equal(result.conf.numpy()[0], conf)
    assert result.reply is not None
    assert int(result.reply.window_rows[0]) == 128
    assert srv.core.step_log[-1]['n_rows'] == 128
    if disconnect == 'after_final':
        rnd.client.new_session('CPU qualification after final suffix')
        assert rnd.client is not None
        assert rnd.client.connect_now()
        decode = eng.step(schedule=[0])
        result = run(rnd, decode, decode.batch.num_tokens)
        assert result.read_status() == STATUS_OK and result.ok_mask[0] == 1
        assert result.reply is not None
        assert int(result.reply.window_rows[0]) == 128
        assert srv.core.step_log[-1]['n_rows'] == 128


def test_missing_suffix_refuses_before_worker_draft(remote: tuple[TcpStubServer, RemoteRound]) -> None:
    srv, rnd = remote
    eng = sim.Engine(seed=95, chunk_sizes=(4096,))
    r = eng.new_request(0, prompt_len=8192)
    run(rnd, eng.step(schedule=[0]), 1)
    # Deliberately violate the scheduler promise: jump to a short final chunk.
    r.computed = 8188
    sd = eng.step(schedule=[0])
    before = srv.core.counters.values.get('drafted_requests', 0)
    result = run(rnd, sd, 4)
    assert result.read_status() == STATUS_ROUND_FAILED
    assert not result.ok_mask.any() and not result.drafts.any()
    assert srv.core.counters.values.get('drafted_requests', 0) == before
    assert rnd.stats.snapshot()['counters']['fallback_replay_suffix_incomplete'] == 1


def test_cancel_slot_reuse_does_not_resync_old_owner(remote: tuple[TcpStubServer, RemoteRound]) -> None:
    _, rnd = remote
    eng = sim.Engine(seed=96, chunk_sizes=(4096,))
    eng.new_request(0, prompt_len=8192)
    run(rnd, eng.step(schedule=[0]), 1)
    assert rnd.slots is not None
    old_gen = rnd.slots[0].generation
    eng.slots[0] = None
    r = eng.new_request(0, prompt_len=256)
    result = run(rnd, eng.step(schedule=[0]), 128)
    assert result.read_status() == STATUS_OK and result.ok_mask[0] == 1
    assert rnd.slots[0].generation != old_gen
    want, _ = fake_draft(eng.expected_window(r), r.anchor, r.computed, sim.N, sim.VOCAB)
    assert np.array_equal(result.drafts.numpy()[0], want)


def observe(audit: SuffixAudit, first: int, q: int, kept: int, need: bool,
            owner: str = 'a', reject: int = 0) -> tuple[SuffixRow, ...]:
    return audit.observe([owner], [0], [0, q], [kept], list(range(first, first + q)),
                         [reject], [int(need)], [123], [456], [need])


def test_independent_rank_metadata_and_real_suffix() -> None:
    ranks = [SuffixAudit(128, 1), SuffixAudit(128, 1)]
    for first, q, kept, need in [(0, 4096, 1, False), (4096, 4096, 1, False),
                                  (8192, 129, 128, True), (8321, 4, 4, True)]:
        rows = [observe(a, first, q, kept, need) for a in ranks]
        agree([(None, x) for x in rows])
        row = rows[0][0]
        assert row.q == q and row.kept == kept and row.valid == kept
        assert row.first_pos == first + q - kept and row.anchor_pos == first + q
        assert row.anchor_token == (123 if need else 456)
        if need:
            assert row.anchor_pos - row.lo == 128
    new = [observe(a, 0, 256, 128, True, owner='b') for a in ranks]
    assert new[0][0].owner_epoch == 2
    agree([(None, x) for x in new])


@pytest.mark.parametrize('field,value', [('q', 4095), ('kept', 127), ('first_pos', 1),
                                       ('valid', 127), ('anchor_pos', 4095),
                                       ('anchor_token', 124), ('owner_epoch', 2)])
def test_rank_mismatch_stops(field: str, value: int) -> None:
    row = observe(SuffixAudit(128, 1), 0, 4096, 128, True)[0]
    with pytest.raises(RuntimeError, match='rank provenance mismatch'):
        agree([(None, (row,)), (None, (replace(row, **cast(dict[str, Any], {field: value})),))])


def test_audit_actual_round_hook_exchanges_local_failure_before_raising() -> None:
    # Execute the actual hook without constructing GPU/client dependencies.
    errors: list[str | None] = []
    def collective(snapshot: tuple[str | None, tuple[SuffixRow, ...]]) -> None:
        errors.append(snapshot[0])
        agree([snapshot, (None, ())])
    audit = SimpleNamespace(_suffix_audit=SuffixAudit(128, 1), _suffix_agree=collective,
                            stats=Stats(), tp_rank=0, broadcast_fn=None)
    eng = sim.Engine(seed=97, chunk_sizes=(4096,))
    eng.new_request(0, prompt_len=4096)
    sd = eng.step(schedule=[0])
    setattr(sd.batch, "replay_tap_lens_np", np.array([1]))
    with pytest.raises(RuntimeError, match='rank provenance invalid'):
        RemoteRound._audit_suffix(cast(RemoteRound, audit), sd.batch, sd.num_sampled, sd.num_rejected,
                                 sd.last_sampled, sd.next_prefill_tokens)
    assert errors == ['ValueError']


def test_audit_rejection_and_backtrack_do_not_fabricate_history() -> None:
    a = SuffixAudit(128, 1)
    observe(a, 0, 4096, 128, True)
    rows = observe(a, 4096, 4, 4, True, reject=2)
    assert rows[0].valid == 2 and rows[0].anchor_pos == 4098
    observe(a, 8192, 4096, 1, False)
    with pytest.raises(ValueError, match='final suffix incomplete'):
        observe(a, 12000, 4, 4, True)
    # Only genuine replay rows recover a suffix below the skipped gap.
    assert observe(a, 11876, 128, 128, True)[0].valid == 128


@pytest.mark.parametrize('incomplete', [False, True])
def test_real_process_proxy_requires_final_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, incomplete: bool,
) -> None:
    import json
    import time
    from remote_dspark.shmregion import RegionLayout, SharedRegion
    from remote_dspark.streamsync import CpuHostFlagSync
    monkeypatch.setenv('VLLM_DSV41_REPLAY_SKIP', '1')
    srv = TcpStubServer(WorkerCore(hello()), '127.0.0.1', 0).start()
    region = SharedRegion.create(RegionLayout.for_hello(hello()))
    sync = CpuHostFlagSync(timeout_s=5, region=region)
    stats_path = tmp_path / 'proxy.json'
    cfg = RemoteConfig.from_env({
        'VLLM_DSPARK_REMOTE': f'tcp://127.0.0.1:{srv.port}',
        'VLLM_DSPARK_REMOTE_IDENTITY': bytes(range(32)).hex(),
        'VLLM_DSPARK_REMOTE_POLICY': 'capacity', 'VLLM_DSPARK_REMOTE_ASYNC': '1',
        'VLLM_DSPARK_REMOTE_ASYNC_PROXY': 'process',
        'VLLM_DSPARK_REMOTE_STATS_FILE': str(stats_path),
        'VLLM_DSPARK_REMOTE_LOG_EVERY': '0'})
    rnd = RemoteRound(cfg, hello(), device=torch.device('cpu'), tp_rank=0,
                      broadcast_fn=None, background=True, stream_sync=sync)
    try:
        assert rnd.proxy is not None and rnd.async_mode
        deadline = time.monotonic() + 5
        while True:
            rnd.write_stats()
            counters = json.loads(stats_path.read_text())['counters']
            if counters.get('hellos', 0):
                break
            assert time.monotonic() < deadline, 'process did not connect to CPU stub'
            time.sleep(.01)
        eng = sim.Engine(seed=98, chunk_sizes=(4096,))
        r = eng.new_request(0, prompt_len=8192)
        assert run(rnd, eng.step(schedule=[0]), 1).read_status() == STATUS_OK
        if incomplete:
            r.computed = 8188
        sd = eng.step(schedule=[0])
        result = run(rnd, sd, 4 if incomplete else 128)
        rnd.write_stats()
        counters = json.loads(stats_path.read_text())['counters']
        if incomplete:
            assert result.read_status() == STATUS_ROUND_FAILED
            assert counters['fallback_replay_suffix_incomplete'] == 1
            assert not result.drafts.any()
            assert srv.core.counters.values.get('drafted_requests', 0) == 0
        else:
            assert result.read_status() == STATUS_OK and result.ok_mask[0] == 1
            want, _ = fake_draft(eng.expected_window(r), r.anchor, r.computed, sim.N, sim.VOCAB)
            assert np.array_equal(result.drafts.numpy()[0], want)
            assert counters.get('fallback_replay_suffix_incomplete', 0) == 0
    finally:
        rnd.close()
        srv.stop()
