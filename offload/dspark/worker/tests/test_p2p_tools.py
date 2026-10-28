"""CPU refusal checks for offline GPU diagnostics; never inject a CUDA fault here."""
from __future__ import annotations

from unittest.mock import Mock

import pytest

from worker.p2p import bench, poison_probe


def test_poison_probe_requires_opt_in_before_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DSPARK_ALLOW_POISON_PROBE", raising=False)
    count = Mock(side_effect=AssertionError("must refuse before probing CUDA"))
    monkeypatch.setattr(poison_probe.torch.cuda, "device_count", count)
    with pytest.raises(RuntimeError, match="explicit offline pair-test switch"):
        poison_probe.main()
    count.assert_not_called()


def test_poison_probe_requires_pair_even_with_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSPARK_ALLOW_POISON_PROBE", "RX-TP2SAFE")
    monkeypatch.setattr(poison_probe.torch.cuda, "device_count", lambda: 1)
    comm = Mock(side_effect=AssertionError("must refuse before constructing a collective"))
    monkeypatch.setattr(poison_probe, "P2PComm", comm)
    with pytest.raises(RuntimeError, match="explicit offline pair-test switch"):
        poison_probe.main()
    comm.assert_not_called()


@pytest.mark.parametrize("devices,repeats", [(1, 100), (2, 9)])
def test_bench_refuses_invalid_run_before_allocation(monkeypatch: pytest.MonkeyPatch,
                                                    devices: int, repeats: int) -> None:
    monkeypatch.setattr(bench.torch.cuda, "device_count", lambda: devices)
    monkeypatch.setattr("sys.argv", ["bench", "--repeats", str(repeats)])
    pair = Mock(side_effect=AssertionError("must refuse before constructing a collective"))
    monkeypatch.setattr(bench, "P2PPair", pair)
    with pytest.raises(ValueError, match="authorized pair and at least 10 repeats"):
        bench.main()
    pair.assert_not_called()
