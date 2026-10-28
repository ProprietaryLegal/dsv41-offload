"""TP = 2 drafter (tp2.py) on the real checkpoint.

One visible GPU (GPU-7744b44b): emulation (both ranks on the card), checked against the one-card model.py.
  * every shard's decoded weights are the full tensor's decoded weights bit for bit (EXL3 trellis decode of the head,
    wq_b, wo_b, shared and routed experts per stage; FP8 wo_a bytes / scales, sinks; Markov head rows and cached bias
    rows against the one-card model's cache)
  * replicated tensors identical on both ranks after a step (stream, head hidden) and rank 1's drafts == rank 0's
  * CUDA graph == eager; n_out prefix bit-identity; WANT_MARGIN graphs give the same drafts
  * agreement with the one-card model (not bitwise: accumulation order differs; numbers printed, loose floor), also
    with windows near position 900,000 (both models built for the 1M context, --max-model-len 1048576)
Two visible GPUs (the pair): the real worker (p2p collectives) == the emulation on device 0, bit for bit.
Run: CUDA_VISIBLE_DEVICES=<uuid(s)> python -m pytest -q -s worker/tests/test_tp2.py
"""

from __future__ import annotations

import os

import pytest
import torch

from worker.checkpoint import Checkpoint
from worker.exl3.plan import PROJ_DOWN, PROJ_GATE, PROJ_UP, Exl3MoEPlan, shard_slices
from worker.tp2 import TPDrafter, vocab_split

ROOT = os.environ.get("DSPARK_ROOT", "/opt/dsv41/x86host-data/dspark")
MAX_LEN = 1048576                    # the production worker's --max-model-len (1M context); shorter sessions clamp
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.exists(ROOT), reason="GPU + checkpoint")


@pytest.fixture(scope="module")
def ck():
    return Checkpoint(ROOT)


@pytest.fixture(scope="module")
def emu(ck):
    d = torch.device("cuda", 0)
    return TPDrafter(ck, [d, d], slots=8, max_model_len=MAX_LEN)


def decode(tr: torch.Tensor, k: int, dev) -> torch.Tensor:
    import exllamav3_ext as ext
    tr = tr.contiguous().to(dev)
    w = torch.empty(tr.shape[0] * 16, tr.shape[1] * 16, dtype=torch.float16, device=dev)
    ext.reconstruct(w, tr, float(k), False, True)
    return w


def test_vocab_split():
    a, b = vocab_split(129280)
    assert a == (0, 64768) and b == (64768, 129280)


def test_shard_weights_bit_identical(ck, emu):
    dev = emu.dev
    s = emu.s
    checked = 0
    for rk in emu.ranks:
        tr, suh, svh, k = ck.exl3("head")
        w = decode(tr, k, dev)
        assert torch.equal(decode(rk.head.trellis, k, dev), w[:, rk.v0:rk.v1])
        assert torch.equal(rk.head.svh.cpu(), svh[rk.v0:rk.v1]) and torch.equal(rk.head.suh.cpu(), suh)
        del w
        checked += 1
        for i, st in enumerate(rk.stages):
            p = f"mtp.{i}."
            hd = s.head_dim
            for mat, name, cut in ((st.wq_b, "attn.wq_b", ("col", rk.h0 * hd, rk.h1 * hd)),
                                   (st.wo_b, "attn.wo_b", ("row", rk.g0 * s.o_lora, rk.g1 * s.o_lora)),
                                   (st.sh_w1, "ffn.shared_experts.w1", ("col", rk.i0, rk.i1)),
                                   (st.sh_w3, "ffn.shared_experts.w3", ("col", rk.i0, rk.i1)),
                                   (st.sh_w2, "ffn.shared_experts.w2", ("row", rk.i0, rk.i1)),
                                   (st.wq_a, "attn.wq_a", ("all",)), (st.wkv, "attn.wkv", ("all",))):
                tr, suh, svh, k = ck.exl3(p + name)
                w = decode(tr, k, dev)
                ws = decode(mat.trellis, k, dev)
                if cut[0] == "col":
                    assert torch.equal(ws, w[:, cut[1]:cut[2]]), name
                    assert torch.equal(mat.svh.cpu(), svh[cut[1]:cut[2]]) and torch.equal(mat.suh.cpu(), suh)
                elif cut[0] == "row":
                    assert torch.equal(ws, w[cut[1]:cut[2]]), name
                    assert torch.equal(mat.suh.cpu(), suh[cut[1]:cut[2]]) and torch.equal(mat.svh.cpu(), svh)
                else:
                    assert torch.equal(ws, w) and torch.equal(mat.suh.cpu(), suh) and torch.equal(mat.svh.cpu(), svh)
                checked += 1
            woa = ck.get(p + "attn.wo_a.weight").view(torch.uint8)
            sc = ck.get(p + "attn.wo_a.scale").float()
            r0, r1 = rk.g0 * s.o_lora, rk.g1 * s.o_lora
            assert torch.equal(st.wo_a8.cpu(), woa[r0:r1]) and torch.equal(st.wo_a_sc.cpu(), sc[r0 // 32:r1 // 32])
            assert torch.equal(st.attn_sink.cpu(), ck.get(p + "attn.attn_sink").float()[rk.h0:rk.h1])
            # routed experts: the bank's per-slot views against the full tensors' cut (first, middle, last expert)
            lp = Exl3MoEPlan.from_dict(ck.plan_json).drafter_layers[s.n_target_layers + i]
            for e in (0, 63, 127):
                views = st.bank.views(lp.perm[e])
                for (vt, vsuh, vsvh, vk), proj, wn in zip(views, (PROJ_GATE, PROJ_UP, PROJ_DOWN), ("w1", "w3", "w2")):
                    tr, suh, svh, k = ck.exl3(f"{p}ffn.experts.{e}.{wn}")
                    sl = shard_slices(proj, s.inter, 2, rk.r)
                    a, z = sl.trellis_range
                    want = decode(tr.narrow(sl.trellis_dim, a, z - a), k, dev)
                    full = decode(tr, k, dev)
                    if proj == PROJ_DOWN:
                        assert torch.equal(want, full[a * 16:z * 16])
                    else:
                        assert torch.equal(want, full[:, a * 16:z * 16])
                    assert torch.equal(decode(vt, vk, dev), want), (i, e, wn)
                    if sl.suh_range:
                        assert torch.equal(vsuh.cpu(), suh[sl.suh_range[0]:sl.suh_range[1]])
                    else:
                        assert torch.equal(vsuh.cpu(), suh)
                    if sl.svh_range:
                        assert torch.equal(vsvh.cpu(), svh[sl.svh_range[0]:sl.svh_range[1]])
                    else:
                        assert torch.equal(vsvh.cpu(), svh)
                    checked += 1
    print("shard matrices checked:", checked)


def test_markov_shards_match_one_card(ck, emu):
    from worker.model import DSparkDrafter
    one = DSparkDrafter(ck, emu.dev, slots=1)
    try:
        for rk in emu.ranks:
            mb = rk.markov
            assert torch.equal(mb.w2, one.markov.w2[rk.v0:rk.v1])
            assert torch.equal(mb.slot_of_token, one.markov.slot_of_token)
            assert torch.equal(mb.cache, one.markov.cache[:, rk.v0:rk.v1])
            assert torch.equal(mb.w1, one.markov.w1)
    finally:
        del one
        torch.cuda.empty_cache()


def _windows(m: TPDrafter, n_req: int = 4, seed: int = 5, off: int = 0):
    """Four 128-row windows of tapped rows (row i of the text at position i + off) and the step's input [4, B]."""
    from tokenizers import Tokenizer
    from worker.tools.parity_ref import _parity_text, make_taps
    ids = Tokenizer.from_file(os.path.join(ROOT, "tokenizer.json")).encode(_parity_text()).ids
    ids = (ids * (1 + 1200 // len(ids)))[:1200]
    toks = torch.tensor(ids, dtype=torch.int64, device=m.dev)

    class E:
        embed = m.ranks[0].embed
    taps = make_taps(E, toks, torch.Generator(device=m.dev).manual_seed(seed)).cpu()
    lasts = (300, 500, 700, 900)[:n_req]
    for b, lp in enumerate(lasts):
        rows = torch.arange(lp - 127, lp + 1)
        m.insert_eager(torch.full_like(rows, b), rows + off, taps[rows])
    last = torch.tensor(lasts)
    inp = torch.stack((torch.arange(n_req), toks.cpu()[last + 1], last + off, last - 127 + off))
    return inp, taps, toks.cpu()


def test_step_graph_eager_nout_margin(emu):
    m = emu
    inp, taps, toks = _windows(m)
    tr: dict = {}
    ids, conf, _, ids1 = m.draft(inp, 5, trace=tr)
    assert torch.equal(ids, ids1)
    m.capture(batch_sizes=(1, 4), ks=(5,), insert_rows=(8,), n_outs=(4, 5))
    for B in (1, 4):
        e_ids, e_conf, _, _ = m.draft(inp[:, :B], 5)
        g_ids, g_conf, _ = m.draft_graphed(inp[:, :B].pin_memory(), 5)
        assert torch.equal(g_ids, e_ids) and torch.equal(g_conf, e_conf), B
        g4 = [t.clone() for t in m.draft_graphed(inp[:, :B].pin_memory(), 5, n_out=4)[:2]]
        g5 = [t.clone() for t in m.draft_graphed(inp[:, :B].pin_memory(), 5, n_out=5)[:2]]
        assert torch.equal(g4[0][:, :4], g5[0][:, :4]) and torch.equal(g4[1][:, :4], g5[1][:, :4])
        assert (g4[0][:, 4] == 0).all() and (g4[1][:, 4] == 0).all()
        mg = m.draft_graphed(inp[:, :B].pin_memory(), 5, margin=True)
        assert torch.equal(mg[0], g5[0]) and torch.equal(mg[1], g5[1]) and (mg[2] >= 0).all()


@pytest.mark.parametrize("off", [0, 900000])
def test_agreement_with_one_card(ck, emu, off):
    """Not bitwise (accumulation order); a floor that only a real bug breaks, numbers printed. off = 900,000: positions
    past the 327,680 default, both RoPE tables built for the 1M context."""
    from worker.model import DSparkDrafter
    m = emu
    inp, taps, toks = _windows(m, seed=11, off=off)
    one = DSparkDrafter(ck, m.dev, slots=8, max_model_len=MAX_LEN)
    try:
        dev = m.dev
        for b in range(4):
            lp = int(inp[2, b])
            pos = torch.arange(lp - 127, lp + 1, device=dev)
            one.insert(torch.full_like(pos, b), pos, one.main_x(taps[(pos - off).cpu()].to(dev)))
        tr1: dict = {}
        tr2: dict = {}
        d1, c1 = one.draft(inp[0].to(dev), inp[1].to(dev), inp[2].to(dev), 5, trace=tr1, ctx_lo=inp[3].to(dev))
        d2, c2, _, _ = m.draft(inp, 5, trace=tr2)
        def per_row(a, b):
            a, b = a.float().flatten(0, -2), b.float().flatten(0, -2)
            return (a - b).norm(dim=-1) / b.norm(dim=-1)
        # a routing near-tie (3rd / 4th expert scores within rounding) flips an expert for a row and cascades through
        # the later stages (O2 PARITY.md D3a: head hidden up to 40 % on such rows): judge the median row, count flips
        flips = sum(int((~(tr1[f"stage{i}.experts"].sort(-1).values == tr2[f"stage{i}.experts"].sort(-1).values)
                         .all(-1)).sum()) for i in range(3))
        hh = per_row(tr2["head_hidden"], tr1["head_hidden"])
        lg = per_row(tr2["base_logits"], tr1["base_logits"])
        agree = float((d1 == d2).float().mean())
        print(f"off {off}: one-card vs TP2: drafts agree {agree:.3f}, head hidden rel per row median "
              f"{float(hh.median()):.2e} max {float(hh.max()):.2e}, base logits rel median {float(lg.median()):.2e}, "
              f"rows with an expert flip {flips}, confidence max diff {float((c1 - c2).abs().max()):.3e}")
        assert float(hh.median()) < 0.05 and float(lg.median()) < 0.05 and agree >= 0.5
    finally:
        del one
        torch.cuda.empty_cache()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs the NVLink pair")
def test_pair_equals_emulation(ck):
    """The real worker (two cards, p2p.cu) and the emulation (both ranks on one card, torch combine) compute the same
    bits: same kernels per rank, and the collectives' combine is the same arithmetic. The emulation runs on card 1
    (GPU-7744b44b): card 0 also hosts the live one-card worker during the pair window (< 14 GiB of ours there)."""
    d0, d1 = torch.device("cuda", 0), torch.device("cuda", 1)
    real = TPDrafter(ck, [d0, d1], slots=8)
    inp, taps, toks = _windows(real)
    tr_r: dict = {}
    ids_r, conf_r, _, ids_r1 = real.draft(inp, 5, trace=tr_r)
    _, _, mg_r, _ = real.draft(inp, 5, margin=True)
    real.capture(batch_sizes=(1, 4), ks=(5,), insert_rows=(8,), n_outs=(4, 5))
    g_r = [t.clone().cpu() for t in real.draft_graphed(inp.pin_memory(), 5)[:2]]
    real.comm.check()
    print("card 0 memory (ours):", round(torch.cuda.memory_reserved(d0) / 2**30, 2), "GiB")
    emu = TPDrafter(ck, [d1, d1], slots=8)
    _windows(emu)
    tr_e: dict = {}
    ids_e, conf_e, _, _ = emu.draft(inp, 5, trace=tr_e)
    _, _, mg_e, _ = emu.draft(inp, 5, margin=True)
    c = lambda t: t.cpu()
    assert torch.equal(c(ids_r), c(ids_e)) and torch.equal(c(conf_r), c(conf_e)) and torch.equal(c(ids_r1), c(ids_r))
    assert torch.equal(c(mg_r), c(mg_e))
    assert torch.equal(c(tr_r["base_logits"]), c(tr_e["base_logits"]))
    assert torch.equal(c(tr_r["head_hidden"]), c(tr_e["head_hidden"]))
    assert torch.equal(g_r[0], c(ids_r)) and torch.equal(g_r[1], c(conf_r))
