"""S25 independent dense FP64 oracle for the D2 top-k + tail objective.

Written for the S25 merge check without importing any trainer code. Values use the full
vocabulary and one global normalizer; gradients are hand-derived closed forms (checked against
central finite differences in test_s25_oracle.py), not autograd of the code under test.

Objective per scored position (teacher retained support S, teacher tail p_T, student q):
  KL  = sum_{i in S, p_i>0} p_i (log p_i - log q_i) + [p_T>0] p_T (log p_T - log q_T)
  TV  = 1/2 (sum_{i in S} |p_i - q_i| + |p_T - q_T|)
  CE  = .1 (logZ - z_label)            only where mask & ~present
  loss = sum_b w_b per_b / sum_b w_b,  w = (1,.8,.65,.5,.4) * mask
A positive teacher bucket against a zero student bucket is +inf (never floored).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

WEIGHTS = (1., .8, .65, .5, .4)


def _lse(x: Tensor) -> Tensor:
    m = x.max(-1, keepdim=True).values
    finite = torch.isfinite(m)
    m0 = torch.where(finite, m, torch.zeros_like(m))
    out = (x - m0).exp().sum(-1).log() + m0.squeeze(-1)
    return torch.where(finite.squeeze(-1), out, torch.full_like(out, -math.inf))


def logits(hidden: Tensor, head: Tensor, embed: Tensor, markov: Tensor, previous: Tensor) -> Tensor:
    return hidden.double() @ head.double().T + embed.double()[previous] @ markov.double().T


def per_position(z: Tensor, ids: Tensor, p: Tensor, logp: Tensor, tail: Tensor, labels: Tensor, present: Tensor,
                 mask: Tensor, divergence: str) -> tuple[Tensor, Tensor]:
    """Returns (per, dper/dz). z is [..., V] float64 (may hold -inf hard masks)."""
    z = z.double()
    p, logp, tail = p.double(), logp.double(), tail.double()
    V = z.shape[-1]
    valid = ids >= 0
    # Explicit loop (no scatter) so padding ids can never touch a real token's membership.
    in_s = torch.zeros(z.shape, dtype=torch.bool)
    for idx in torch.nonzero(valid):
        *b, k = idx.tolist()
        in_s[(*b, int(ids[(*b, k)]))] = True
    logz = _lse(z)
    q = (z - logz[..., None]).exp()
    logzt = _lse(z.masked_fill(in_s, -math.inf))
    logqt = logzt - logz
    qt = logqt.exp()
    safe = ids.clamp_min(0)
    logq_s = (z - logz[..., None]).gather(-1, safe)
    q_s = logq_s.exp().masked_fill(~valid, 0)
    p = p.masked_fill(~valid, 0)
    if divergence == "kl":
        pos = p > 0
        top = torch.where(pos, p * (logp.masked_fill(~pos, 0) - logq_s.masked_fill(~pos, 0)), torch.zeros_like(p))
        top = torch.where(pos & torch.isneginf(logq_s), torch.full_like(p, math.inf), top)
        tl = torch.where(tail > 0, tail * (tail.clamp_min(1e-300).log() - logqt), torch.zeros_like(tail))
        div = top.sum(-1) + tl
        mass = p.sum(-1) + tail
        # dKL/dz_j = M q_j - p_j [j in S] - p_T exp(z_j - logZ_T) [j not in S]
        g = mass[..., None] * q
        g = g - torch.zeros_like(q).scatter_add(-1, safe, p)
        tail_share = torch.where(in_s | torch.isneginf(z), torch.zeros_like(q),
                                 (z - torch.where(torch.isfinite(logzt), logzt, torch.zeros_like(logzt))[..., None]).exp())
        g = g - tail[..., None] * torch.where(tail[..., None] > 0, tail_share, torch.zeros_like(q))
    else:
        div = .5 * ((p - q_s).abs().sum(-1) + (tail - qt).abs())
        a_s = .5 * torch.sign(q_s - p).masked_fill(~valid, 0)
        a_t = .5 * torch.sign(qt - tail)
        coeff = a_t[..., None].expand_as(q).clone()
        coeff = torch.where(in_s, torch.zeros_like(q), coeff)
        coeff = coeff + torch.zeros_like(q).scatter_add(-1, safe, a_s)
        weighted = (a_s * q_s).sum(-1) + a_t * qt
        g = q * (coeff - weighted[..., None])
    missing = mask & ~present
    lab = labels.clamp(0, V - 1)
    ce = .1 * (logz - z.gather(-1, lab[..., None]).squeeze(-1))
    gce = .1 * (q - torch.nn.functional.one_hot(lab, V).double())
    per = torch.where(present, div, torch.where(missing, ce, torch.zeros_like(div)))
    grad = torch.where(present[..., None], g, torch.where(missing[..., None], gce, torch.zeros_like(g)))
    return per, grad


def objective(hidden: Tensor, head: Tensor, embed: Tensor, markov: Tensor, previous: Tensor, ids: Tensor,
              p: Tensor, logp: Tensor, tail: Tensor, labels: Tensor, present: Tensor, mask: Tensor,
              divergence: str, scale: float = 1.) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Loss, per-position values (zero where unscored) and scale*d(loss)/d{hidden,embed,markov}."""
    z = logits(hidden, head, embed, markov, previous)
    per, gz = per_position(z, ids, p, logp, tail, labels, present, mask, divergence)
    w = torch.tensor(WEIGHTS, dtype=torch.float64).expand_as(per) * mask
    loss = (per * w).sum() / w.sum()
    dz = gz * (scale * w / w.sum())[..., None]
    d_hidden = dz @ head.double()
    e = embed.double()[previous]
    d_embed = torch.zeros_like(embed, dtype=torch.float64).index_add_(
        0, previous.flatten(), (dz @ markov.double()).reshape(-1, embed.shape[-1]))
    d_markov = dz.reshape(-1, dz.shape[-1]).T @ e.reshape(-1, e.shape[-1])
    return loss, per, {"hidden": d_hidden, "embed": d_embed, "markov": d_markov}
