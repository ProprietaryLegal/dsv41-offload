"""Keep the model-free pair reproducer large enough for its randomized payloads (CPU only)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from worker.p2p import lifecycle
from worker.tests import test_p2p as sequences


def test_pair_lifecycle_capacity_covers_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    pair = SimpleNamespace(max_words=0)

    def make_pair(devs: list[torch.device], max_words: int) -> SimpleNamespace:
        pair.max_words = max_words
        return pair

    def run_sequence(pr: SimpleNamespace, **kwargs: int) -> None:
        assert pr.max_words >= sequences.MAXW, "reproducer must reach the collective, not fail argument validation"

    run = Mock(side_effect=run_sequence)
    monkeypatch.setattr("sys.argv", ["lifecycle", "--cycles", "1"])
    monkeypatch.setattr(lifecycle.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(lifecycle.torch.cuda, "Stream", lambda dev: object())
    monkeypatch.setattr(lifecycle.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(lifecycle, "P2PPair", make_pair)
    monkeypatch.setattr(lifecycle, "boundary", lambda label, devs, fn: fn())
    monkeypatch.setattr(sequences, "run_sequence", run)
    lifecycle.main()
    run.assert_called_once_with(pair, seed=0, n_ops=60)
