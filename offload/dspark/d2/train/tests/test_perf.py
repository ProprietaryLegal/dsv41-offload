from __future__ import annotations

import pytest

from offload.dspark.d2.train.perf import PhaseTimes, instrument
from offload.dspark.d2.train.model import Drafter


def test_phase_accounting_and_instrumentation_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter((0., 1., 3., 5.))
    monkeypatch.setattr("offload.dspark.d2.train.perf.time.perf_counter", lambda: next(clock))
    times = PhaseTimes(lambda: None)
    original = Drafter.conditioning
    with pytest.raises(RuntimeError), instrument(times), times.phase("outer"):
        assert Drafter.conditioning is not original
        with times.phase("inner"):
            raise RuntimeError("probe")
    assert Drafter.conditioning is original
    assert times.seconds == {"inner": 2., "outer": 3.}
