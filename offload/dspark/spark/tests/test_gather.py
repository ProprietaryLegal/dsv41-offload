# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

from remote_dspark import wire
from remote_dspark.gather import BatchView, RowGatherer

W, H, NT = 8, 2, 3   # small window for readable cases


def mk(max_rows=32):
    return RowGatherer(max_slots=4, window=W, hidden=H, n_taps=NT, max_rows=max_rows,
                       device=torch.device("cpu"))


def taps_for(T):
    base = torch.arange(T, dtype=torch.float32).view(T, 1)
    return [(base * 10 + t + torch.zeros(T, H)).to(torch.bfloat16) for t in range(NT)]


def row_of(pos_row_index):
    return torch.cat([(torch.full((H,), pos_row_index * 10.0 + t)).to(torch.bfloat16)
                      for t in range(NT)])


def test_trailing_window_compaction_and_meta():
    g = mk()
    # req A: prefill of 12 rows (positions 100..111) -> last 8 sent
    # req B: decode of 4 rows (positions 50..53), 2 rejected -> 2 sent
    qsl = np.array([0, 12, 16])
    positions = torch.cat([torch.arange(100, 112), torch.arange(50, 54)])
    bv = BatchView(req_ids=["a", "b"], slots=np.array([2, 0]), qsl=qsl,
                   need_draft=np.array([False, True]), k_window=np.array([0, 3]))
    last_sampled = torch.tensor([[77], [0], [0], [0]])
    next_prefill = torch.tensor([[0, 0, 555, 0]], dtype=torch.int32)
    res = g.gather(bv, taps_for(16), positions, torch.tensor([0, 2], dtype=torch.int32),
                   torch.tensor([0, 2], dtype=torch.int32), torch.tensor([2, 0], dtype=torch.int32),
                   last_sampled, next_prefill)
    assert res.valid.tolist() == [8, 2]
    assert res.first_pos.tolist() == [104, 50]
    assert res.anchor.tolist() == [555, 77]
    assert res.sampled.tolist() == [False, True]
    assert res.n_candidates == 12
    rows = g.rows_view(2, 10)
    want = torch.stack([row_of(i) for i in list(range(4, 12)) + [12, 13]])
    assert torch.equal(rows, want)
    # ring: slot 2 holds positions 104..111 at pos % 8, slot 0 holds 50, 51
    for p, src in zip(range(104, 112), range(4, 12)):
        assert torch.equal(g.ring[2 * W + p % W], row_of(src))
    assert torch.equal(g.ring[0 * W + 50 % W], row_of(12))
    assert torch.equal(g.ring[0 * W + 51 % W], row_of(13))
    # rejected rows never reach the ring
    assert not torch.equal(g.ring[0 * W + 52 % W], row_of(14))


def test_gather_from_ring_rebuilds_window():
    g = mk()
    qsl = np.array([0, 8])
    positions = torch.arange(200, 208)
    bv = BatchView(["a"], np.array([1]), qsl, np.array([True]), np.array([0]))
    g.gather(bv, taps_for(8), positions, torch.zeros(1, dtype=torch.int32),
             torch.ones(1, dtype=torch.int32), torch.tensor([1], dtype=torch.int32),
             torch.zeros(4, 1, dtype=torch.int64), torch.zeros(1, 4, dtype=torch.int32))
    n = g.gather_from_ring(1, [(1, 203, 5)])
    assert n == 5
    rows = g.rows_view(1, 5)
    assert torch.equal(rows, torch.stack([row_of(i) for i in range(3, 8)]))


def test_message_layout_offsets():
    g = mk()
    off = wire.APP_HEADER_SIZE + wire.rows_offset_for(3)
    v = g.rows_view(3, 2)
    assert v.data_ptr() - g.msg.data_ptr() == off
    assert g.row_bytes == NT * H * 2


def test_limits():
    g = mk(max_rows=8)
    qsl = np.array([0, 9, 18])
    bv = BatchView(["a", "b"], np.array([0, 1]), qsl, np.array([1, 1], bool), np.zeros(2))
    with pytest.raises(ValueError, match="max_rows"):
        g.gather(bv, taps_for(18), torch.arange(18), torch.zeros(2, dtype=torch.int32),
                 torch.ones(2, dtype=torch.int32), torch.tensor([0, 1], dtype=torch.int32),
                 torch.zeros(4, 1, dtype=torch.int64), torch.zeros(1, 4, dtype=torch.int32))
    with pytest.raises(TypeError):
        g2 = mk()
        bv1 = BatchView(["a"], np.array([0]), np.array([0, 2]), np.array([True]), np.zeros(1))
        g2.gather(bv1, [t.float() for t in taps_for(2)], torch.arange(2),
                  torch.zeros(1, dtype=torch.int32), torch.ones(1, dtype=torch.int32),
                  torch.tensor([0], dtype=torch.int32), torch.zeros(4, 1, dtype=torch.int64),
                  torch.zeros(1, 4, dtype=torch.int32))
