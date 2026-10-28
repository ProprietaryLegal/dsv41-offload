"""CPU checks of fail-closed lifetime rules; no CUDA context or model files needed."""
from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from worker import p2p


def pair_shell() -> p2p.P2PPair:
    pr = p2p.P2PPair.__new__(p2p.P2PPair)
    pr.poisoned = None
    pr.devs = [torch.device("cuda", 0), torch.device("cuda", 1)]
    pr.launch_streams = set()
    pr.idle_timeout_s = 1
    pr.st = []
    return pr


def test_poison_refuses_reset_and_launch_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    pr = pair_shell()
    pr.poison("illegal memory access")
    pr.poison("later error must not replace first fault")
    idle = Mock(side_effect=AssertionError("must not touch CUDA after poisoning"))
    monkeypatch.setattr(p2p, "wait_idle", idle)
    with pytest.raises(RuntimeError, match="restart worker: illegal memory access"):
        pr.reset()
    with pytest.raises(RuntimeError, match="restart worker"):
        pr._launch(0, "ar_sum")
    with pytest.raises(RuntimeError, match="restart worker"):
        pr.headers()
    idle.assert_not_called()


@pytest.mark.parametrize("failure", [RuntimeError("CUDA fault"), TimeoutError("parked stream")])
def test_failed_idle_check_latches_poison(monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    pr = pair_shell()
    monkeypatch.setattr(p2p, "wait_idle", Mock(side_effect=failure))
    with pytest.raises(type(failure), match=str(failure)):
        pr.reset()
    assert pr.poisoned == str(failure)
    with pytest.raises(RuntimeError, match="restart worker"):
        pr.reset()


def test_reset_waits_recorded_noncurrent_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    pr = pair_shell()
    stream = Mock()
    pr.launch_streams = {stream}
    observed = []
    monkeypatch.setattr(p2p, "wait_idle", lambda *a: observed.append(set(a[2]) if len(a) > 2 else set()))
    pr.reset()
    assert observed == [{stream}, set()]
    assert not pr.launch_streams


@pytest.mark.parametrize("ms", [0, -1, float("nan"), float("inf")])
def test_bad_timeout_refuses_before_cuda(ms: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        p2p.P2PPair([torch.device("cuda", 0), torch.device("cuda", 1)], 4, timeout_ms=ms)
