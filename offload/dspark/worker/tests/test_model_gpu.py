"""GPU integration tests on the real drafter (needs the S2 delivery at DSPARK_ROOT, default /opt/dsv41/x86host-data/dspark,
and the 3090): n_out bit-identity (CONTRACT 1.3), the head's row-count independence, graph == eager."""

import os

import pytest
import torch

ROOT = os.environ.get("DSPARK_ROOT", "/opt/dsv41/x86host-data/dspark")
pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and os.path.exists(os.path.join(ROOT, "TENSORS.json"))),
                                reason="needs the GPU and the drafter delivery")


@pytest.fixture(scope="module")
def model():
    from worker.checkpoint import Checkpoint
    from worker.model import DSparkDrafter
    dev = torch.device("cuda", 0)
    m = DSparkDrafter(Checkpoint(ROOT), dev, slots=4)
    g = torch.Generator(device=dev).manual_seed(7)
    toks = torch.randint(0, 120000, (1100,), generator=g, device=dev)
    e = m.embed[toks]
    taps = torch.cat([e * 30, e * 50 + 0.01, e * 80 - 0.01], -1)
    for b, lp in enumerate((300, 500, 700, 900)):
        rows = torch.arange(lp - 127, lp + 1, device=dev)
        m.insert(torch.full_like(rows, b), rows, m.main_x(taps[rows]))
    m.capture(batch_sizes=(1, 2, 3, 4), ks=(5,), insert_rows=(8,), n_outs=(1, 2, 3, 4), pack_reply=True)
    return m, toks


def test_head_rows_independent(model):
    m, _ = model
    g = torch.Generator(device=m.dev).manual_seed(3)
    x = (torch.randn(16, m.s.hidden, generator=g, device=m.dev) * 0.5).to(torch.float16)
    full = m.head(x)
    for n in (1, 2, 4, 5, 8, 12, 15):
        assert torch.equal(m.head(x[:n].contiguous()), full[:n]), n


@pytest.mark.parametrize("B", [1, 2, 3, 4])
def test_n_out_bit_identical(model, B):
    m, toks = model
    dev = m.dev
    slots = torch.arange(B, device=dev)
    last = torch.tensor((300, 500, 700, 900)[:B], device=dev)
    anchors = toks[last + 1]
    ids5, conf5 = m.draft(slots, anchors, last, 5)
    for n in (1, 2, 3, 4):
        ids, conf = m.draft(slots, anchors, last, 5, n_out=n)
        assert torch.equal(ids[:, :n], ids5[:, :n]) and torch.equal(conf[:, :n], conf5[:, :n]), n
        assert int(ids[:, n:].abs().sum()) == 0 and float(conf[:, n:].abs().sum()) == 0.0
        gi, gc = m.draft_graphed(slots, anchors, last, 5, n_out=n)
        assert torch.equal(gi[:B], ids) and torch.equal(gc[:B], conf), n
    gi, gc = m.draft_graphed(slots, anchors, last, 5)
    assert torch.equal(gi[:B], ids5) and torch.equal(gc[:B], conf5)
    _, _, mg = m.draft_with_margin(slots, anchors, last, 5, n_out=3)
    _, _, mg5 = m.draft_with_margin(slots, anchors, last, 5)
    assert torch.equal(mg[:, :3], mg5[:, :3]) and float(mg[:, 3:].abs().sum()) == 0.0


def test_main_proj_fp32_accumulation(model):
    m, _ = model
    assert m.main_proj_fp32acc and m.main_proj.ext.__name__ == "exllamav3_ext_facc"


@pytest.mark.parametrize("B", [1, 2, 3, 4])
def test_captured_reply_exact_for_all_depths_and_margins(model, B):
    m, toks = model
    slots = torch.arange(B, device=m.dev)
    last = torch.tensor((300, 500, 700, 900)[:B], device=m.dev)
    # Consecutive calls must not expose a prior graph's stale outputs.
    for anchor in (toks[last + 1], toks[last + 2]):
        for n in (1, 2, 3, 4, 5):
            for margin in (False, True):
                values = m.draft_graphed(slots, anchor, last, 5, n_out=n, margin=margin)
                expected = torch.cat([v.to(torch.float64) for v in values], 1)
                (packed,) = m.draft_graphed(slots, anchor, last, 5, n_out=n, margin=margin, packed_reply=True)
                assert torch.equal(packed.view(torch.int64), expected.view(torch.int64)), (B, n, margin)


def test_missing_packed_graph_refuses_instead_of_falling_back(model):
    m, _ = model
    key = (1, 5, 5, False)
    saved = m.reply_graphs.pop(key)
    try:
        v = torch.zeros(1, device=m.dev, dtype=torch.int64)
        with pytest.raises(KeyError, match="packed True"):
            m.draft_graphed(v, v, v, 5, packed_reply=True)
    finally:
        m.reply_graphs[key] = saved


def test_backend_packed_reply_same_for_mixed_request_depths(model):
    import numpy as np
    from worker.server import GpuBackend
    from worker.window import WindowRange
    m, toks = model
    plain = GpuBackend(m, 1, 30720)
    packed = GpuBackend(m, 1, 30720, graph_reply=True)
    reqs = [(i, int(toks[lp + 1]), lp + 1, WindowRange(128, lp - 127, lp), i + 1)
            for i, lp in enumerate((300, 500, 700, 900))]
    for margin in (False, True):
        expected = plain.draft(reqs, 5, margin)
        actual = packed.draft(reqs, 5, margin)
        for a, b in zip(actual, expected):
            assert (a is None and b is None) or (
                a is not None and b is not None and a.dtype == b.dtype and a.shape == b.shape
                and np.array_equal(a.view(np.uint8), b.view(np.uint8)))


@pytest.mark.parametrize("B,n", [(1, 5), (2, 5), (4, 4), (3, 2)])
def test_margin_graph_matches_eager_and_plain_graph(model, B, n):
    m, toks = model
    dev = m.dev
    slots = torch.arange(B, device=dev)
    last = torch.tensor((300, 500, 700, 900)[:B], device=dev)
    anchors = toks[last + 1]
    ei, ec, em = m.draft_with_margin(slots, anchors, last, 5, n_out=n)
    gi, gc, gm = (t[:B].clone() for t in m.draft_graphed(slots, anchors, last, 5, n_out=n, margin=True))
    assert torch.equal(gi, ei) and torch.equal(gc, ec) and torch.equal(gm, em)
    pi, pc = (t[:B] for t in m.draft_graphed(slots, anchors, last, 5, n_out=n))
    assert torch.equal(pi, gi) and torch.equal(pc, gc)
    assert float(gm[:, n:].abs().sum()) == 0.0 and bool((gm[:, :n] >= 0).all())


def test_insert_padding_rows_zeroed(model):
    m, _ = model
    dev = m.dev
    big = torch.full((8, m.s.hidden * 3), 1000.0, dtype=torch.bfloat16, device=dev)
    sr = torch.full((8,), m.n_slots, dtype=torch.int64, device=dev)          # scratch slot only
    m.insert_taps_graphed(sr, torch.arange(8, device=dev), big)
    m.insert_taps_graphed(sr[:6], torch.arange(6, device=dev), big[:6])
    assert float(m.gi_taps[6:8].abs().sum()) == 0.0
    m.reset_slots(torch.tensor([m.n_slots], device=dev))


def test_head_rows_independent_of_their_position_in_the_tile(model):
    """n_out moves a request's rows to other places inside the 16-row head tile (B = 4: row (b1, i0) is tile row 4 at
    n_out 4, row 5 at n_out 5): each row's logits must not depend on where it sits or what shares the tile."""
    m, _ = model
    g = torch.Generator(device=m.dev).manual_seed(4)
    x = (torch.randn(16, m.s.hidden, generator=g, device=m.dev) * 0.5).to(torch.float16)
    ref = m.head(x)
    for seed in (1, 2, 3):
        perm = torch.randperm(16, generator=torch.Generator().manual_seed(seed)).to(m.dev)
        assert torch.equal(m.head(x[perm].contiguous()), ref[perm])
    other = (torch.randn(16, m.s.hidden, generator=g, device=m.dev) * 3.0).to(torch.float16)
    mixed = torch.cat((other[:7], x[3:4], other[7:15]))
    assert torch.equal(m.head(mixed)[7], ref[3])
