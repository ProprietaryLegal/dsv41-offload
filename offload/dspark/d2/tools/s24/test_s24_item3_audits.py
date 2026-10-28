"""S24 item 3: zero-init LoRA, projection-cache refusal, analytic loss backward and whole-model STE (CPU, tiny model)."""
from __future__ import annotations

from typing import Literal, cast

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from offload.dspark.d2.train import model as model_mod, ops
from offload.dspark.d2.train.audits import one_step
from offload.dspark.d2.train.data import SyntheticBlocks, collate
from offload.dspark.d2.train.fast_loss import bucket_loss
from offload.dspark.d2.train.loss import POSITION_WEIGHTS, distillation_loss
from offload.dspark.d2.train.ops import Dense
from offload.dspark.d2.train.tests.test_loop import model
from offload.dspark.native.formats import act_qdq, decode_fp4


def lora_layers(m: torch.nn.Module) -> list[tuple[str, Dense]]:
    return [(n, mod) for n, mod in m.named_modules() if isinstance(mod, Dense) and mod.rank]


def test_s24_zero_init_signal_reaches_A_on_the_second_step() -> None:
    """Step 1: only B (and norms/Markov) get gradient; after the B update, step 2 gives EVERY LoRA A a gradient."""
    torch.manual_seed(3)
    m = model()
    block = SyntheticBlocks(m.c, 2)[1]
    first = one_step(m, block, chunk=32)
    a_names = {n+".a" for n, _ in lora_layers(m)}
    b_names = {n+".b" for n, _ in lora_layers(m)}
    assert set(first["zero_gradient_parameters"]) <= a_names
    assert b_names <= set(first["nonzero_gradient_parameters"])
    assert all(bool(layer.b.count_nonzero()) for _, layer in lora_layers(m))      # the AdamW step moved B off zero
    for p in m.parameters():
        p.grad = None
    h, _ = m.hidden_from(block["inputs"])
    loss, _ = distillation_loss(m, h, chunk=32, **block["targets"])  # type: ignore[arg-type]
    torch.autograd.backward(loss)
    dead = [n for n, layer in lora_layers(m) if layer.a.grad is None or not bool(layer.a.grad.count_nonzero())]
    assert dead == []


def test_s24_both_factors_zero_is_a_dead_adapter_and_is_refused() -> None:
    """Counterexample motivating the audit: A = B = 0 gives zero gradient to both factors (no learning signal)."""
    torch.manual_seed(3)
    m = model()
    with torch.no_grad():
        for _, layer in lora_layers(m):
            layer.a.zero_()
    block = SyntheticBlocks(m.c, 1)[0]
    h, _ = m.hidden_from(block["inputs"])
    loss, _ = distillation_loss(m, h, chunk=32, **block["targets"])  # type: ignore[arg-type]
    torch.autograd.backward(loss)
    assert all(not bool(layer.a.grad.count_nonzero()) and not bool(layer.b.grad.count_nonzero())  # type: ignore[union-attr]
               for _, layer in lora_layers(m))
    with pytest.raises(ValueError, match="nonzero A and zero B"):
        one_step(m, block, chunk=32)


def test_s24_projection_cache_has_no_remaining_entry_point() -> None:
    m = model()
    inp = dict(SyntheticBlocks(m.c, 1)[0]["inputs"])
    with torch.no_grad():
        inp["main_projection"] = torch.zeros(*inp["has_taps"].shape, m.c.dim)
    with pytest.raises(ValueError, match="projection cache refused"):
        m.hidden_from(inp)
    m.fast_training = True                                      # the old fast-only branch is gone as well
    with pytest.raises(ValueError, match="projection cache refused"):
        m.conditioning(inp["taps"], inp["engine_mx"], inp["has_taps"], inp["main_projection"])


# ---------------------------------------------------------------- loss backward vs FP64 unfused reference ------------
def unfused(m: model_mod.Drafter, h: Tensor, previous: Tensor, ids: Tensor, logp: Tensor, tail: Tensor, labels: Tensor,
            present: Tensor, mask: Tensor, divergence: str) -> Tensor:
    """Full-vocabulary FP64: z = h@head^T + E[prev]@M^T; softmax; exact bucket sums; CE fallback; weighted mean."""
    head = m.t["head.weight"].double()
    embed, markov = m.last("markov_head.embed.weight"), m.last("markov_head.head.weight")
    z = h.double()@head.T+F.embedding(previous, embed).double()@markov.double().T
    logq = z.log_softmax(-1)
    valid = ids >= 0
    safe = ids.clamp_min(0)
    q_top = logq.exp().gather(-1, safe).masked_fill(~valid, 0)
    member = torch.zeros_like(z, dtype=torch.bool).scatter(-1, safe, valid)
    q_tail = (logq.exp()*~member).sum(-1)
    p = logp.double().exp().masked_fill(~valid, 0)
    t = tail.double()
    if divergence == "kl":
        per = (p*(logp.double().masked_fill(~valid, 0)-logq.gather(-1, safe))).masked_fill(~valid, 0).sum(-1)
        per = per+torch.xlogy(t, t)-t*q_tail.log()
    else:
        per = .5*((p-q_top).abs().sum(-1)+(t-q_tail).abs())
    ce = -logq.gather(-1, labels.clamp_min(0)[..., None])[..., 0]
    per = torch.where(present, per, .1*ce)
    w = torch.tensor(POSITION_WEIGHTS, dtype=torch.float64)*mask
    return (per*w).sum()/w.sum()


def targets(m: model_mod.Drafter, B: int, seed: int) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)
    V = m.c.vocab
    teacher = (torch.randn(B, 5, V, generator=g)*3).softmax(-1)
    p, ids = teacher.topk(20, -1)
    ids = ids.clone()
    ids[0, 1, 15:] = -1                                         # padded top-k (fewer than 20 retained)
    p = p.masked_fill(ids < 0, 0)
    tail = (1-p.sum(-1)).clamp_min(0)
    previous = torch.randint(0, V, (B, 5), generator=g)
    previous[:, 1] = previous[:, 0]                             # repeated previous tokens
    labels = torch.randint(0, V, (B, 5), generator=g)
    present = torch.ones(B, 5, dtype=torch.bool)
    present[1, 3] = False                                       # sampled-CE fallback position
    mask = torch.ones(B, 5, dtype=torch.bool)
    mask[0, 4] = False                                          # masked tail position
    return {"previous": previous, "top_ids": ids, "top_logp": p.log(), "tail_mass": tail, "labels": labels,
            "present": present, "position_mask": mask}


@pytest.mark.parametrize("divergence", ["kl", "tvd"])
@pytest.mark.parametrize("chunk", [7, 40, 96, 200])
def test_s24_reference_and_analytic_losses_match_fp64_unfused(divergence: Literal["kl", "tvd"], chunk: int) -> None:
    torch.manual_seed(11)
    m = model()
    B = 2
    h = (torch.randn(B, 5, m.c.dim)*.5).requires_grad_(True)
    t = targets(m, B, 5)
    embed, markov = m.last("markov_head.embed.weight"), m.last("markov_head.head.weight")
    params = (h, embed, markov)
    want = unfused(m, h, t["previous"], t["top_ids"], t["top_logp"], t["tail_mass"], t["labels"], t["present"],
                   t["position_mask"], divergence)
    want_g = torch.autograd.grad(want, params)
    ref, _ = distillation_loss(m, h, chunk=chunk, divergence=divergence, **t)
    ref_g = torch.autograd.grad(ref, params)
    fast, _ = bucket_loss(h, m.t["head.weight"], embed, markov, t["previous"], t["top_ids"], t["top_logp"],
                          t["tail_mass"], t["labels"], t["present"], t["position_mask"], chunk, divergence)
    fast_g = torch.autograd.grad(fast, params)
    for name, value, grads in (("reference", ref, ref_g), ("analytic", fast, fast_g)):
        torch.testing.assert_close(value.double(), want, rtol=1e-6, atol=1e-6, msg=name)
        for g_got, g_want in zip(grads, want_g):
            torch.testing.assert_close(g_got.double(), g_want.double(), rtol=2e-5, atol=2e-6, msg=lambda s: f"{name}: {s}")
        assert all(bool(g.count_nonzero()) for g in grads)


# ---------------------------------------------------------------- whole-model gradient with canonical STE -----------
def canonical_qdq(x: Tensor, fast: bool = False) -> Tensor:
    assert not fast
    return act_qdq(x.detach())+(x-x.detach())                  # forward == act_qdq(x) exactly; d/dx == identity


def plain_frozen_linear(x: Tensor, w: Tensor, scale: Tensor | None = None, acc: torch.dtype = torch.float32) -> Tensor:
    decoded = w if scale is None else decode_fp4(w, scale).to(torch.bfloat16)
    return F.linear(x.to(acc), decoded.detach().to(acc)).to(x.dtype)


def plain_grouped(x: Tensor, w: Tensor) -> Tensor:
    return torch.einsum("bsgd,grd->bsgr", x, w.detach().float())


def test_s24_whole_model_gradients_equal_plain_autograd_with_canonical_ste(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every trainable's gradient through the custom _QDQ/_FrozenLinear/_GroupedLinear backwards equals plain autograd
    with the textbook straight-through estimator; forward values are identical."""
    torch.manual_seed(13)
    data = SyntheticBlocks(model().c, 3)
    block = collate([data[0], data[2]])
    def run() -> tuple[float, dict[str, Tensor]]:
        m = model()
        with torch.no_grad():
            for _, layer in lora_layers(m):
                layer.b.normal_(std=.05)                          # exercise A-gradients and LoRA forward too
        h, _ = m.hidden_from(block["inputs"])
        loss, _ = distillation_loss(m, h, chunk=40, **block["targets"])  # type: ignore[arg-type]
        torch.autograd.backward(loss)
        return float(loss), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    custom_loss, custom = run()
    monkeypatch.setattr(ops, "qdq", canonical_qdq)
    monkeypatch.setattr(model_mod, "qdq", canonical_qdq)
    monkeypatch.setattr(ops, "frozen_linear", plain_frozen_linear)
    monkeypatch.setattr(model_mod, "frozen_linear", plain_frozen_linear)
    monkeypatch.setattr(model_mod, "grouped_linear", plain_grouped)
    plain_loss, plain = run()
    assert custom_loss == plain_loss
    assert custom.keys() == plain.keys() and len(custom) > 40
    worst = max(float((custom[n]-plain[n]).abs().max()/(plain[n].abs().max()+1e-30)) for n in custom)
    assert worst < 1e-5, worst
