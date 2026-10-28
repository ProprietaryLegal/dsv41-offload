# SPDX-License-Identifier: Apache-2.0
"""The one-launch Triton gather equals the torch path bit for bit (TRITON_INTERPRET=1, CPU)."""

import numpy as np
import pytest
import torch

from remote_dspark import fused_gather
from remote_dspark.gather import RowGatherer

import sim

pytestmark = pytest.mark.skipif(not fused_gather.available(), reason="no triton")


def pair(H=sim.H, max_rows=512):
    kw = dict(max_slots=4, window=sim.W, hidden=H, n_taps=3, max_rows=max_rows,
              device=torch.device("cpu"))
    return RowGatherer(fused=False, **kw), RowGatherer(fused=True, **kw)


def test_fused_matches_torch_over_simulated_steps():
    ref, fus = pair()
    assert fus.fused and not ref.fused
    eng = sim.Engine(seed=41)
    for s in range(sim.M):
        eng.new_request(s, prompt_len=[300, 40, 129, 900][s])
    rng = np.random.default_rng(41)
    from remote_dspark.gather import batch_view_from_input_batch

    for step in range(80):
        if step % 9 == 0:
            for slot, r in enumerate(eng.slots):
                if r is not None and not r.prefilling and r.decode_left <= 0:
                    eng.slots[slot] = None
                    eng.new_request(slot, hit=int(rng.choice([0, 300])))
        sd = eng.step()
        bv = batch_view_from_input_batch(sd.batch)
        args = (bv, sd.taps, sd.batch.positions, sd.num_rejected, sd.num_sampled,
                sd.batch.idx_mapping, sd.last_sampled, sd.next_prefill_tokens)
        a = ref.gather(*args)
        b = fus.gather(*args)
        assert fus.fused, "fused path fell back"
        assert np.array_equal(a.valid, b.valid) and np.array_equal(a.first_pos, b.first_pos)
        assert np.array_equal(a.anchor, b.anchor) and np.array_equal(a.sampled, b.sampled)
        V = a.n_valid
        n = bv.n
        assert torch.equal(ref.rows_view(n, V).view(torch.int16),
                           fus.rows_view(n, V).view(torch.int16))
        # every ring row a valid candidate wrote is identical
        for i in range(n):
            s = int(bv.slots[i])
            for p in range(int(a.first_pos[i]), int(a.first_pos[i] + a.valid[i])):
                k = s * sim.W + p % sim.W
                assert torch.equal(ref.ring[k].view(torch.int16), fus.ring[k].view(torch.int16))


def test_fused_real_width_one_step():
    H = 5120
    ref, fus = pair(H=H, max_rows=64)
    T = 20
    taps = [torch.randn(T, H).to(torch.bfloat16) for _ in range(3)]
    qsl = np.array([0, 6, 12, 17, 20])
    from remote_dspark.gather import BatchView

    bv = BatchView([f"r{i}" for i in range(4)], np.array([3, 0, 2, 1]), qsl,
                   np.ones(4, bool), np.array([5, 5, 4, 2]))
    positions = torch.cat([torch.arange(100, 106), torch.arange(7, 13), torch.arange(50, 55),
                           torch.arange(0, 3)])
    nrej = torch.tensor([2, 0, 4, 1], dtype=torch.int32)
    nsamp = torch.tensor([4, 6, 1, 2], dtype=torch.int32)
    ls = torch.tensor([[11], [22], [33], [44]])
    npf = torch.tensor([[5, 6, 7, 8]], dtype=torch.int32)
    idx = torch.tensor([3, 0, 2, 1], dtype=torch.int32)
    a = ref.gather(bv, taps, positions, nrej, nsamp, idx, ls, npf)
    b = fus.gather(bv, taps, positions, nrej, nsamp, idx, ls, npf)
    assert a.valid.tolist() == b.valid.tolist() == [4, 6, 1, 2]
    assert a.anchor.tolist() == b.anchor.tolist() == [44, 11, 33, 22]
    assert torch.equal(ref.rows_view(4, 13).view(torch.int16), fus.rows_view(4, 13).view(torch.int16))


def test_self_check_passes_on_cpu_interpreter():
    g = RowGatherer(max_slots=4, window=sim.W, hidden=sim.H, n_taps=3, max_rows=512,
                    device=torch.device("cpu"), fused=True)
    ok, why = g.self_check()
    assert ok, why
    assert g.fused and not g.ring.any()
