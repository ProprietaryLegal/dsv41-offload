"""PP3 real async process/thread transport against a toy worker, CPU only."""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import sim
import torch
from remote_dspark.hostflag import HostFlagCore
from remote_dspark.mock_worker import TcpStubServer, fake_draft
from remote_dspark.probability_buffers import ProbabilityBuffers
from remote_dspark.round import STATUS_OK, STATUS_ROUND_FAILED, RemoteRound
from remote_dspark.shmregion import RegionLayout, SharedRegion
from remote_dspark.streamsync import CpuHostFlagSync
from test_round import cfg_for, hello

from offload.dspark.worker import protocol as wp
from offload.dspark.worker.server import DraftCore, FakeBackend


class SampleBackend(FakeBackend):
    def draft_sampled(self, requests: Any, depth: int, keys: Any, masks: Any) -> tuple[Any, Any, Any, Any]:
        ids, conf, _ = self.draft(requests, depth, False)
        support = np.arange(32,dtype=np.int32)[None,None,:].repeat(len(requests),0).repeat(depth,1)
        q = np.zeros_like(support, dtype=np.float32)
        for j, mask in enumerate(masks):
            if mask:
                ids[j] = 2
                q[j, :, 2] = 1.
        return ids, conf, support, q


@pytest.mark.parametrize('proxy', ['thread', 'process'])
def test_sampled_async_reply_and_failed_round(proxy: str, tmp_path: Any,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('DSPARK_SAMPLED_TOP32','1')
    h = replace(hello(), sample_mode=2)
    backend = SampleBackend(h.window, h.row_bytes, h.max_slots, fake_draft, h.vocab_size)
    core = DraftCore(wp.Hello.decode(h.pack()), backend)
    server = TcpStubServer(core, '127.0.0.1', 0).start()
    region = SharedRegion.create(RegionLayout.for_hello(h)) if proxy == 'process' else None
    sync = CpuHostFlagSync(timeout_s=5., region=region)
    cfg = cfg_for(server.port, VLLM_DSPARK_REMOTE_ASYNC='1', VLLM_DSPARK_REMOTE_ASYNC_PROXY=proxy,
                  VLLM_DSPARK_SAMPLED_TOP32='1', VLLM_DSPARK_REMOTE_FUSED_GATHER='0',
                  VLLM_DSPARK_REMOTE_STATS_FILE=str(tmp_path/'stats.json'))
    rnd = RemoteRound(cfg,h,device=torch.device('cpu'),tp_rank=0,broadcast_fn=None,
                      background=False,stream_sync=sync)
    eng = sim.Engine(seed=4103)
    eng.new_request(0, prompt_len=1)
    eng.new_request(1, prompt_len=1)
    temp = torch.tensor([1.,0.,1.,0.])
    seeds = torch.tensor([42,13,16,17])
    buffers = ProbabilityBuffers(4,5,h.vocab_size,torch.device('cpu'))
    try:
        if rnd.client is not None:
            assert rnd.client.connect_now()
        else:
            deadline = time.monotonic()+5
            while time.monotonic() < deadline:
                if rnd.proxy.word(576) == 1:
                    break
                time.sleep(.01)
            assert rnd.proxy.word(576) == 1
        successes = 0
        for step in range(5):
            if step == 3:
                core.inject.frame_status_once = wp.INTERNAL
            data = eng.step(schedule=[0,1])
            res = rnd.run(data.batch,data.taps,data.num_sampled,data.num_rejected,
                          data.last_sampled,data.next_prefill_tokens,temperature=temp,seeds=seeds)
            n, N, layout = res.n, h.num_spec, res.layout
            support = res.bcast[layout.o_qids:layout.o_qids+n*N*32].view(n,N,32)
            q = res.bcast[layout.o_q:layout.o_q+n*N*32].view(torch.float32).view(n,N,32)
            buffers.update(data.batch.idx_mapping,support,q,res.ok_mask)
            if step == 3:
                assert res.read_status() == STATUS_ROUND_FAILED
                assert not res.ok_mask.any() and not q.any()
                assert not buffers.valid.any()
            elif res.read_status() == STATUS_OK:
                successes += 1
                for j, slot in enumerate(data.batch.idx_mapping.tolist()):
                    if temp[slot] > 0:
                        assert res.drafts[j].tolist() == [2]*5
                        assert q[j].sum().item() == 5.
                    else:
                        assert q[j].sum().item() == 0.
        assert successes >= 3
        assert isinstance(sync, HostFlagCore)
    finally:
        rnd.close()
        server.stop()


def test_sampled_admission_needs_both_overlays_and_fixed_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    import vllm_stubs
    from test_speculator import S
    cfg = cfg_for(1,VLLM_DSPARK_SAMPLED_TOP32='1')
    config = vllm_stubs.vllm_config(draft_sample='probabilistic')
    config.speculative_config.rejection_sample_method = 'standard'
    module = types.ModuleType('vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils')
    parent = sys.modules['vllm.v1.worker.gpu.spec_decode']
    monkeypatch.setitem(sys.modules,module.__name__,module)
    monkeypatch.setitem(parent.__dict__,'rejection_sampler_utils',module)
    runner = types.ModuleType('vllm.v1.worker.gpu.model_runner')
    monkeypatch.setitem(sys.modules,runner.__name__,runner)
    with pytest.raises(ValueError,match='overlay is missing'):
        S._validate(config,cfg)
    monkeypatch.setitem(module.__dict__,'REMOTE_DRAFT_PROBS_ABI','pp3-dyadic24-standard-v1')
    with pytest.raises(ValueError,match='failure mask overlay is missing'):
        S._validate(config,cfg)
    monkeypatch.setitem(runner.__dict__,'REMOTE_DRAFT_PROBS_ABI','pp3-dyadic24-standard-v1')
    monkeypatch.setenv('VLLM_DSPARK_DEPTH_POLICY','fixed:5')
    S._validate(config,cfg)
    monkeypatch.setenv('VLLM_DSPARK_DEPTH_POLICY','confr')
    with pytest.raises(ValueError,match='fixed depth'):
        S._validate(config,cfg)
    monkeypatch.setenv('VLLM_DSPARK_DEPTH_POLICY','fixed:5')
    monkeypatch.setenv('VLLM_DSPARK_LOOKUP','lookup=2,4,.9')
    with pytest.raises(ValueError,match='prompt lookup'):
        S._validate(config,cfg)
