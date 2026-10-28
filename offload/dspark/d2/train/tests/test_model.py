from __future__ import annotations

import torch
import pytest
from typing import Callable, cast

from offload.dspark.d2.train.model import Drafter, Stage
from offload.dspark.d2.train.ops import frozen_linear, grouped_linear, qdq
from offload.dspark.d2.train.synthetic import SyntheticSource, inputs, tiny_config
from offload.dspark.native.formats import act_qdq, decode_fp4


def test_packed_dx_no_weight_grad() -> None:
    g = torch.Generator().manual_seed(7)
    w = torch.randint(0, 256, (32, 32), dtype=torch.uint8, generator=g)
    s = torch.full((32, 2), 120, dtype=torch.uint8)
    x = torch.randn(3, 64, generator=g, requires_grad=True)
    y = frozen_linear(x, w, s)
    decoded = decode_fp4(w, s)
    torch.testing.assert_close(y, x @ decoded.T, rtol=0, atol=0)
    dy = torch.randn(y.shape, generator=g)
    cast(Callable[..., None], y.backward)(dy)
    torch.testing.assert_close(x.grad, dy @ decoded, rtol=0, atol=0)
    assert w.grad is None and s.grad is None


def test_qdq_forward_exact_ste_backward() -> None:
    x = torch.randn(5, 64, requires_grad=True)
    y = qdq(x)
    assert torch.equal(y, act_qdq(x))
    cast(Callable[[], None], y.sum().backward)()
    assert x.grad is not None
    assert torch.equal(x.grad, torch.ones_like(x))


def test_grouped_wo_a_preserves_fp32_reference_sum_and_dx() -> None:
    torch.manual_seed(91)
    x = torch.randn(1, 5, 4, 256, requires_grad=True)
    w = torch.randn(4, 64, 256).to(torch.bfloat16)
    y = grouped_linear(x, w)
    expected = torch.einsum("bsgd,grd->bsgr", x, w.float())
    assert torch.equal(y, expected)
    grad = torch.randn_like(y)
    got = torch.autograd.grad(y, x, grad)[0]
    want = torch.autograd.grad(expected, x, grad)[0]
    assert torch.equal(got, want)


def test_checkpoint_forward_backward_and_statelessness() -> None:
    torch.set_num_threads(2)
    c = tiny_config()
    torch.manual_seed(3)
    a = Drafter(SyntheticSource(c), c)
    torch.manual_seed(3)
    b = Drafter(SyntheticSource(c), c, checkpoint_stages=False)
    inp = inputs(c)
    ha, xa = a.hidden_from(inp)
    hb, xb = b.hidden_from(inp)
    assert torch.equal(ha, hb) and torch.equal(xa, xb)
    prev = torch.tensor([[3, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    cast(Callable[[], None], a.logit_chunk(ha, prev, 0, c.vocab).square().mean().backward)()
    cast(Callable[[], None], b.logit_chunk(hb, prev, 0, c.vocab).square().mean().backward)()
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert (pa.grad is None) == (pb.grad is None)
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)
    assert all(not p.requires_grad for st in a.stages for ex in cast(Stage, st).experts for p in ex.parameters())
    assert a.t["head.weight"].dtype == torch.bfloat16
    a.eval()
    with torch.no_grad():
        h1, x1 = a.hidden_from(inp)
        a.hidden_from(inputs(c, P=27))
        h2, x2 = a.hidden_from(inp)
        assert torch.equal(h1, h2) and torch.equal(x1, x2)
        greedy, _ = a.greedy(h1, x1, inp["anchor"], chunk=17)
        previous = torch.cat((inp["anchor"][:, None], greedy[:, :-1]), -1)
        assert torch.equal(greedy, a.logit_chunk(h1, previous, 0, c.vocab).argmax(-1))


def test_reference_math_batch_matches_independent_windows() -> None:
    c = tiny_config()
    m = Drafter(SyntheticSource(c), c, acc=torch.float64, reference_math=True).eval()
    windows = [inputs(c, batch=1, P=p) for p in (12, 17)]
    windows[1]["taps"] *= 1.03
    with torch.no_grad():
        individual = [m.hidden_from(inp) for inp in windows]
        batched = {k: torch.cat([inp[k] for inp in windows], 0) for k in windows[0]}
        h, x = m.hidden_from(batched)
        torch.testing.assert_close(h, torch.cat([a for a, _ in individual]), rtol=1e-6, atol=1e-7)
        got, _ = m.greedy(h, x, batched["anchor"])
        want = torch.cat([m.greedy(a, b, inp["anchor"])[0] for (a, b), inp in zip(individual, windows)], 0)
        assert torch.equal(got, want)


def test_raw_projection_keeps_norm_gradient_and_unproven_cache_refuses() -> None:
    c = tiny_config()
    m = Drafter(SyntheticSource(c), c)
    inp = inputs(c)
    projected = m.main_proj(inp["taps"].float()).detach()
    expected = m.conditioning(inp["taps"], inp["engine_mx"], inp["has_taps"])
    norm = m.t["mtp.0.main_norm.weight"]
    grad = torch.autograd.grad(expected.square().sum(), norm)[0]
    assert bool(grad.count_nonzero())
    # A detached cache has no input/weight provenance, even when its shape fits.
    m.fast_training = True
    with pytest.raises(ValueError, match="frozen weights AND frozen inputs"):
        m.conditioning(inp["taps"], inp["engine_mx"], inp["has_taps"], projected)
    with torch.no_grad():
        norm.mul_(1.03)
    changed = m.conditioning(inp["taps"], inp["engine_mx"], inp["has_taps"])
    assert not torch.equal(changed, expected)


def test_fast_training_refuses_cpu_or_reference_model() -> None:
    c = tiny_config()
    for reference in (False, True):
        m = Drafter(SyntheticSource(c), c, reference_math=reference)
        with pytest.raises(ValueError, match="separate all-GPU"):
            m.enable_fast_training()
