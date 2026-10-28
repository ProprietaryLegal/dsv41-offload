"""Sparse transport -> FP32 full-vocabulary probabilities for standard rejection.

The target vocabulary is never truncated to the sparse proposal support. State
is indexed by engine request slots, not current batch order. Failed replies have
valid=False; the probability-aware runner must invalidate every draft for them.
"""
from __future__ import annotations

import torch


class ProbabilityBuffers:
    def __init__(self, slots: int, depth: int, vocab: int, device: torch.device):
        self.probs = torch.zeros(slots, depth, vocab, dtype=torch.float32, device=device)
        self.valid = torch.zeros(slots, dtype=torch.bool, device=device)
        self.batch = torch.empty_like(self.probs)

    def update(self, mapping: torch.Tensor, support: torch.Tensor, q: torch.Tensor,
               ok: torch.Tensor) -> None:
        n = mapping.numel()
        if (support.shape != (n, self.probs.shape[1], 32) or q.shape != support.shape
                or q.dtype != torch.float32 or ok.shape != (n,)):
            raise ValueError('sparse probability GPU buffer ABI mismatch')
        dense = self.batch[:n]
        dense.zero_()
        dense.scatter_(2, support.long(), q)
        idx = mapping.long()
        self.probs.index_copy_(0, idx, dense)
        self.valid.index_copy_(0, idx, ok.bool())
