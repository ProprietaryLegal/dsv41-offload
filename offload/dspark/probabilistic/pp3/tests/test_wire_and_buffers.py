from __future__ import annotations

import struct
from typing import Any

import numpy as np
import pytest
import torch

from offload.dspark.spark.remote_dspark import sampled_wire as sw
from offload.dspark.spark.remote_dspark import wire
from offload.dspark.spark.remote_dspark.probability_buffers import ProbabilityBuffers
from offload.dspark.spark.remote_dspark.roundcore import ReplyLayout, collect_meta
from offload.dspark.spark.remote_dspark.shmregion import RegionLayout, SharedRegion
from offload.dspark.worker import protocol as wp
from offload.dspark.worker.server import DraftCore, FakeBackend, expected_hello


def arrays() -> tuple[Any, Any, Any, Any]:
    mask = np.array([1], dtype=np.int32)
    ids = np.arange(32, dtype=np.int32).reshape(1, 1, 32).repeat(5, 1)
    q = np.zeros_like(ids, dtype=np.float32)
    q[:, :, 1] = .25
    q[:, :, 2] = .75
    tokens = np.full((1, 5), 2, dtype=np.int32)
    return mask, ids, q, tokens


def test_q_roundtrip_checksum_and_full_target_mass() -> None:
    mask, ids, q, tokens = arrays()
    reply = wire.StepReply(12, 5, np.array([0], np.uint16), np.array([1], np.uint16),
                           np.array([2], np.uint32), tokens, np.full((1,5), .5, np.float32),
                           support_ids=ids, proposal_q=q, sampled_mask=mask)
    body = reply.pack()
    unpacked = wire.StepReply.unpack(body, expect_n_req=1, expect_num_spec=5)
    np.testing.assert_array_equal(unpacked.proposal_q, q)
    sw.validate(mask, ids, q, tokens, np.array([True]), 64, mask)
    corrupted = bytearray(body)
    corrupted[-7] ^= 1
    with pytest.raises(wire.WireError, match='CRC'):
        wire.StepReply.unpack(corrupted)
    with pytest.raises(wire.WireError, match='size'):
        wire.StepReply.unpack(body[:-1])
    dense = ProbabilityBuffers(4, 5, 64, torch.device('cpu'))
    dense.update(torch.tensor([3]), torch.from_numpy(ids), torch.from_numpy(q), torch.tensor([True]))
    assert dense.probs[3, 0, 2] == .75
    assert dense.probs[3, 0, 50] == 0
    assert dense.valid.tolist() == [False, False, False, True]
    dense.update(torch.tensor([3]), torch.zeros(1,5,32,dtype=torch.int32), torch.zeros(1,5,32),
                 torch.tensor([False]))
    assert not dense.valid.any() and not dense.probs.any()


@pytest.mark.parametrize('damage', ['duplicates', 'range', 'negative', 'nan', 'nondyadic',
                                     'sum', 'missing_token', 'mask', 'greedy_q'])
def test_q_refuses_corruption(damage: str) -> None:
    mask, ids, q, tokens = arrays()
    if damage == 'duplicates': ids[0, 0, 2] = 1
    if damage == 'range': ids[0, 0, 31] = 64
    if damage == 'negative': q[0, 0, 2] = -.75
    if damage == 'nan': q[0, 0, 2] = np.nan
    if damage == 'nondyadic': q[0, 0, 2] = .7
    if damage == 'sum': q[0, 0, 2] = .5
    if damage == 'missing_token': tokens[0, 0] = 5
    if damage == 'mask': mask[0] = 2
    if damage == 'greedy_q': mask[0] = 0
    with pytest.raises(ValueError):
        sw.validate(mask, ids, q, tokens, np.array([True]), 64)


def test_same_round_sampling_metadata_and_reply_pool() -> None:
    meta = np.zeros((6, 4), np.int64)
    meta[4] = [1, 0, 1, 0]
    meta[5] = [17, 22, -13, 19]
    result = collect_meta(meta, 4, 0)
    assert result.sample_meta == [(17,1), (22,0), (-13,1), (19,0)]
    assert ReplyLayout(4,5).size == 46
    assert ReplyLayout(4,5,True).size * 4 == 5320
    hello = wire.Hello(max_slots=4, sample_mode=wire.SAMPLE_TOP32)
    layout = RegionLayout.for_hello(hello)
    region = SharedRegion.create(layout)
    try:
        assert region.meta_np.shape == (64,6,4)
        region.meta_np[3] = meta
        assert collect_meta(region.meta_np[3],4,0).sample_meta == result.sample_meta
        assert region.meta_np[4].sum() == 0
    finally:
        region.close()


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


def test_real_worker_codec_matches_spark_and_refuses_old_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('DSPARK_SAMPLED_TOP32','1')
    def fake(_rows: Any, _token: int, _pos: int, depth: int, _vocab: int) -> tuple[Any, Any]:
        return np.full(depth,3,np.int32), np.full(depth,.5,np.float32)
    backend = SampleBackend(128, 12, 4, fake, 64)
    exp = expected_hello(b'a'*32,b'b'*32, hidden=2,row_bytes=12,vocab_size=64,sample_mode=2)
    core = DraftCore(exp,backend)
    hello_body = exp.encode()
    old = bytearray(hello_body)
    struct.pack_into('<I',old,17*4,1)
    def frame(op: int, seq: int, body: bytes) -> bytes:
        return wp.AppHeader(wp.MAGIC,1,op,seq,1,len(body),0,0).encode()+body
    refused = wp.AppHeader.decode(core.handle(frame(wp.OP_HELLO,1,bytes(old))))
    assert refused.status == wp.CONFIG_MISMATCH
    assert wp.AppHeader.decode(core.handle(frame(wp.OP_HELLO,2,hello_body))).status == wp.OK
    entry = wire.ReqEntry(slot=0,generation=1,first_pos=0,n_rows=4,anchor_token=1,anchor_pos=4,
                          flags=wire.RF_NEW_GEN|wire.RF_NEED_DRAFT)
    buf = bytearray(256)
    ro = wire.pack_step_prefix(buf,0,12,[entry],4,12,wire.SF_SAMPLED,[(42,1)])
    body = bytes(buf[:ro])+bytes(48)
    request = wp.decode_step(body,12,4,512)
    assert request.sample_meta == [(42,1)]
    response = core.handle(frame(wp.OP_STEP,3,body))
    assert response is not None
    header, data = wire.split_frame(response)
    assert header.status == wire.ST_OK
    reply = wire.StepReply.unpack(data)
    assert reply.sampled_mask.tolist() == [1]
    assert (reply.draft_ids == 2).all()
    sw.validate(reply.sampled_mask, reply.support_ids, reply.proposal_q, reply.draft_ids,
                np.array([True]),64,np.array([1],np.int32))
    # A duplicate frame cannot get fresh RNG under the same sequence.
    assert wp.AppHeader.decode(core.handle(frame(wp.OP_STEP,3,body))).status == wp.STALE_SEQ
