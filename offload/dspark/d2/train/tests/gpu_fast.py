"""Run explicitly under the dedicated GPU lease; independent forward/dX oracles."""
from __future__ import annotations

import json
from typing import Callable, Literal, cast

import torch
from torch.nn import functional as F

from ....native.formats import act_qdq, decode_fp4
from ..fast_gpu import _Grouped, fast_linear, fast_qdq
from ..safety import assert_gpu_lease
from ..fast_loss import bucket_loss
from ..loss import distillation_loss
from ..model import Drafter
from ..synthetic import SyntheticSource, tiny_config
from ..ops import head_product


def loss_oracle() -> None:
    c = tiny_config()
    m = Drafter(SyntheticSource(c), c, device="cuda")
    h = torch.randn(4, 5, c.dim, device="cuda", requires_grad=True)
    prev = torch.randint(0, c.vocab, (4, 5), device="cuda")
    p = torch.randn(4, 5, c.vocab, device="cuda").softmax(-1)
    top, ids = p.topk(20, -1)
    present = torch.ones(4, 5, device="cuda", dtype=torch.bool)
    mask = present.clone()
    divergences: tuple[Literal["kl", "tvd"], ...] = ("kl", "tvd")
    for divergence in divergences:
        loss, _ = distillation_loss(m, h, prev, ids, top.log(), 1-top.sum(-1), prev, present, mask,
                                    chunk=17, divergence=divergence)
        params = h, m.last("markov_head.embed.weight"), m.last("markov_head.head.weight")
        grads = torch.autograd.grad(loss, params)
        actual, _ = bucket_loss(h, m.t["head.weight"], params[1], params[2], prev, ids, top.log(),
                                1-top.sum(-1), prev, present, mask, 17, divergence)
        torch.testing.assert_close(actual, loss, rtol=1e-6, atol=1e-6)
        for a, b in zip(torch.autograd.grad(actual, params), grads):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-7)


def main() -> None:
    assert_gpu_lease()
    torch.manual_seed(17)
    loss_oracle()
    a = torch.randn(17, 5120, device="cuda")
    b = torch.randn(5120, 128, device="cuda").bfloat16()
    expected_head = a.double() @ b.double()
    head_error = (head_product(a, b).double() - expected_head).norm() / expected_head.norm()
    assert float(head_error) <= 1e-5
    # Include halfway values, saturation boundaries and subnormal FP8 values.
    xq = torch.cat((torch.randn(2048, device="cuda") * 1e-5,
                    torch.randn(2048, device="cuda") * 10)).reshape(2, 2048).requires_grad_()
    torch.testing.assert_close(fast_qdq(xq), act_qdq(xq), rtol=0, atol=0)
    assert torch.equal(torch.autograd.grad(fast_qdq(xq).sum(), xq)[0], torch.ones_like(xq))
    halfway = torch.tensor([448., -448., 1.0625, 1.1875, -1.0625, -1.1875, 2.**-10,
                             -2.**-10, 3.*2.**-10, -3.*2.**-10, 0., -0.] + [0.] * 20,
                            device="cuda").repeat(4, 1)
    for dtype, bits in ((torch.float32, torch.int32), (torch.bfloat16, torch.int16)):
        actual_qdq, expected_qdq = fast_qdq(halfway.to(dtype)), act_qdq(halfway.to(dtype))
        assert torch.equal(actual_qdq.view(bits), expected_qdq.view(bits))
    cases = []
    for counts, n, k in (([0, 1, 0, 33], 64, 96), ([17, 0, 19, 1], 2304, 5120),
                          ([2, 0, 7, 0], 5120, 2304)):
        w = torch.randint(0, 256, (4, n, k // 2), device="cuda", dtype=torch.uint8)
        s = torch.randint(118, 124, (4, n, k // 32), device="cuda", dtype=torch.uint8)
        off = F.pad(torch.tensor(counts, device="cuda").cumsum(0), (1, 0))
        tiles = ((torch.tensor(counts, device="cuda") + 15) // 16).cumsum(0)
        x = act_qdq(torch.randn(sum(counts), k, device="cuda")).requires_grad_()
        y = cast(Callable[..., torch.Tensor], _Grouped.apply)(x, w, s, off, tiles)
        decoded = decode_fp4(w, s)
        expected = torch.cat([x[sum(counts[:i]):sum(counts[:i+1])].double() @ decoded[i].double().T
                              for i in range(4)]).float()
        torch.testing.assert_close(y, expected, rtol=2e-5, atol=2e-5)
        dy = torch.randn_like(y)
        got = torch.autograd.grad(y, x, dy)[0]
        dx = torch.cat([dy[sum(counts[:i]):sum(counts[:i+1])].double() @ decoded[i].double()
                        for i in range(4)]).float()
        torch.testing.assert_close(got, dx, rtol=3e-5, atol=3e-5)
        cases.append({"counts": counts, "n": n, "k": k, "forward_max_abs": float((y-expected).detach().abs().max()),
                      "dx_max_abs": float((got-dx).detach().abs().max())})
    for quantized in (True, False):
        x = torch.randn(3, 5, 96, device="cuda")
        if quantized:
            x = act_qdq(x)
        x.requires_grad_()
        w = torch.randn(71, 96, device="cuda").bfloat16()
        y = fast_linear(x, w, quantized)
        torch.testing.assert_close(y, F.linear(x.double(), w.double()).float(), rtol=2e-5, atol=2e-5)
        dy = torch.randn_like(y)
        dx = torch.autograd.grad(y, x, dy)[0]
        torch.testing.assert_close(dx, dy.double().matmul(w.double()).float(), rtol=2e-5, atol=2e-5)
    print(json.dumps({"pass": True, "grouped_cases": cases, "dense_cases": 2}), flush=True)


if __name__ == "__main__":
    main()
