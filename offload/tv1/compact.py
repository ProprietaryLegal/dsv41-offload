"""Confidence-only per-request target widths, independent of a local draft estimator.

Only causal draft prefixes are removed. Scheduler reservations remain untouched;
the worker's actual query/logit counts and GPU rejection bookkeeping shrink together.
Initial arm leaves mixed/grammar batches intact and executes compact batches eager.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt
import torch

IntArray = npt.NDArray[np.int32]
FloatArray = npt.NDArray[np.float32]


class Broadcast(Protocol):
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor: ...


def require_depth_policy(policy: object | None) -> None:
    """Keep production's plain confidence mask/rejection policy authoritative.

    Compact only shortens its causal prefix. The original dead_rows, sampler
    invalid-draft union, record and observe hooks still run on the actual rows.
    Other policies need separate composition qualification.
    """
    if policy is None:
        return
    if (getattr(policy, 'mode', None) != 'conf'
            or getattr(policy, 'calib', None) is not False
            or getattr(policy, 'lookup_opt', None) != '0'
            or getattr(policy, 'ctx_pts', None) != []
            or getattr(policy, 'file', 'missing') is not None):
        raise ValueError('TV1 compact supports only plain conf depth policy without calibration/lookup/context/file')


def widths(confidence: FloatArray, capacities: Sequence[int], threshold: float) -> IntArray:
    if not 0. < threshold <= 1. or not math.isfinite(threshold):
        raise ValueError('TV1 compact threshold must be finite in (0,1]')
    if confidence.ndim != 2 or confidence.shape[0] != len(capacities):
        raise ValueError('TV1 confidence ownership/shape mismatch')
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError('TV1 confidence must be finite in [0,1]')
    cap = np.asarray(capacities, dtype=np.int32)
    if ((cap < 0) | (cap > confidence.shape[1])).any():
        raise ValueError('TV1 draft capacity outside remote confidence window')
    # FP64 host arithmetic is deterministic; equal-to-threshold slots stay live.
    survival = np.cumprod(confidence.astype(np.float64), axis=1)
    prefix = (survival >= threshold) & (np.arange(confidence.shape[1])[None, :] < cap[:, None])
    return np.asarray(prefix.sum(axis=1, dtype=np.int32), dtype=np.int32)


def compact_counts(scheduled: IntArray, reserved_drafts: IntArray, chosen: IntArray) -> IntArray:
    if scheduled.shape != reserved_drafts.shape or chosen.shape != scheduled.shape:
        raise ValueError('TV1 compact row vectors differ')
    if ((chosen < 0) | (chosen > reserved_drafts) | (scheduled < reserved_drafts + 1)).any():
        raise ValueError('TV1 compact prefix exceeds reserved drafts or drops the target row')
    return np.asarray(scheduled - reserved_drafts + chosen, dtype=np.int32)


class CompactWidths:
    def __init__(self, device: torch.device, threshold: float) -> None:
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError('TV1 compact threshold must be in (0,1]')
        self.device, self.threshold = device, threshold
        self.ids: tuple[str, ...] = ()
        self.invalid: set[str] = set()
        self.pending: torch.Tensor | None = None
        self.stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        self.ready = torch.cuda.Event(blocking=True)  # type: ignore[no-untyped-call]
        self.last_before = 0
        self.last_after = 0

    def forget(self, ids: Iterable[str]) -> None:
        # A recycled id/preempted generation must not consume old confidences.
        self.invalid.update(ids)

    def record(self, ids: Sequence[str], confidence: torch.Tensor) -> None:
        if len(set(ids)) != len(ids) or confidence.ndim != 2 or confidence.shape[0] != len(ids):
            raise ValueError('TV1 remote confidence rows do not match unique request ids')
        self.ids = tuple(ids)
        self.invalid.clear()
        self.pending = torch.empty(confidence.shape, dtype=torch.float32, device='cpu', pin_memory=True)
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            self.pending.copy_(confidence, non_blocking=True)
            confidence.record_stream(self.stream)
            self.ready.record()

    def choose(self, ids: Sequence[str], capacities: IntArray, broadcast: Broadcast) -> IntArray:
        proposed = capacities.copy()
        if self.pending is not None:
            # Deliberate, reported sync: current remote confidences must be complete
            # before the host chooses the target shape. No guessed/stale tensor read.
            self.ready.synchronize()
            conf = self.pending.numpy()
            if conf.dtype != np.float32:
                raise ValueError('TV1 staged confidence dtype drift')
            previous = {name: row for row, name in enumerate(self.ids) if name not in self.invalid}
            for row, name in enumerate(ids):
                if name in previous:
                    proposed[row] = widths(conf[previous[name]:previous[name]+1], [int(capacities[row])], self.threshold)[0]
        # Rank zero is the shape authority. Every rank receives the same vector;
        # there is no independent shape decision or rank-local fallback.
        packed = torch.tensor(proposed, dtype=torch.int32, device=self.device)
        agreed = broadcast(packed).cpu().numpy()
        if agreed.dtype != np.int32 or agreed.shape != proposed.shape or ((agreed < 0) | (agreed > capacities)).any():
            raise ValueError('TV1 TP compact-width agreement is invalid')
        self.last_before, self.last_after = int(capacities.sum()), int(agreed.sum())
        return np.asarray(agreed, dtype=np.int32)
