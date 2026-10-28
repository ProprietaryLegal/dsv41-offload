"""The deployed deterministic proposal has a different label from sampled q."""
from __future__ import annotations

import torch

from offload.dspark.d2.train.data import SyntheticBlocks
from offload.dspark.d2.train.loss import confidence_loss, distillation_loss
from offload.dspark.d2.train.model import Drafter
from offload.dspark.d2.train.synthetic import SyntheticSource, tiny_config


def test_confidence_regresses_p_of_greedy_pick_with_unknown_and_masked_rows_excluded() -> None:
    c = tiny_config()
    m = Drafter(SyntheticSource(c), c)
    for name, param in m.named_parameters():
        param.requires_grad_("confidence_head" in name)
    with torch.no_grad():
        m.last("confidence_head.proj.weight").zero_()  # known .5 prediction
    block = SyntheticBlocks(c)[0]
    h, x = m.hidden_from(block["inputs"])
    previous = block["targets"]["previous"]
    with torch.no_grad():
        picks = m.logit_chunk(h, previous, 0, c.vocab).argmax(-1)
    # Only the argmax probability matters, not normalized top-support probability or 1-TV.
    ids = picks.unsqueeze(-1)
    lp = torch.tensor([[[.8], [.4], [.1], [.2], [.3]]]).log()
    ids[0, 3, 0] = -1  # unknown p(d): skip, never invent zero or use teacher tail
    mask = torch.tensor([[True, True, False, True, True]])
    loss, known = confidence_loss(m, h, x, previous, ids, lp, mask, chunk=17)
    assert loss is not None and known == 3
    expected = ((torch.tensor([.8, .4, .3]) - .5) ** 2).mean()
    torch.testing.assert_close(loss, expected)
    # Train-loop confidence stage freezes all other parameters; loss depends on its projection.
    confidence_grad = torch.autograd.grad(loss, m.last("confidence_head.proj.weight"))[0]
    assert torch.isfinite(confidence_grad).all() and bool(confidence_grad.norm() > 0)
    assert all(not p.requires_grad for name, p in m.named_parameters() if "confidence_head" not in name)


def test_distillation_reaches_both_markov_matrices_and_freezes_confidence_projection() -> None:
    c = tiny_config()
    m = Drafter(SyntheticSource(c), c)
    for name, param in m.named_parameters():
        param.requires_grad_("confidence_head" not in name)
    block = SyntheticBlocks(c)[0]
    h, _ = m.hidden_from(block["inputs"])
    t = block["targets"]
    loss, _ = distillation_loss(m, h, t["previous"], t["top_ids"], t["top_logp"], t["tail_mass"],
                                t["labels"], t["present"], t["position_mask"], chunk=17)
    markov = (m.last("markov_head.embed.weight"), m.last("markov_head.head.weight"))
    for grad in torch.autograd.grad(loss, markov):
        assert torch.isfinite(grad).all() and bool(grad.norm() > 0)
    assert not m.last("confidence_head.proj.weight").requires_grad
