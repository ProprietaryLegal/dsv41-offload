"""CPU checks for fixed N4 fixture coverage and timing aggregation."""
from __future__ import annotations

from typing import Any

import pytest
import torch

from offload.dspark.probabilistic.pp4.full_worker import measured, synthetic_fixtures


def test_n4_fixture_states_and_rng_are_frozen() -> None:
    fixtures = synthetic_fixtures(8)
    replay = synthetic_fixtures(8)
    assert len(fixtures) == 3
    assert [inputs[2].tolist() for _, inputs in fixtures] == [
        [0, 63, 127, 31], [128, 300, 32768, 65535], [150000, 32000, 1000, 262143]]
    for (windows, inputs), (other_windows, other_inputs) in zip(fixtures, replay, strict=True):
        assert torch.equal(inputs, other_inputs)
        for (positions, values), (other_positions, other_values) in zip(windows, other_windows, strict=True):
            assert len(positions) <= 128
            assert values.dtype == torch.bfloat16
            assert torch.equal(positions, other_positions)
            assert torch.equal(values, other_values)


def test_timing_uses_individual_round_events(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class Event:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs == {"enable_timing": True}

        def record(self) -> None:
            calls.append("record")

        def elapsed_time(self, end: Event) -> float:
            assert isinstance(end, Event)
            return 2.5

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    result = measured(lambda: calls.append("round"), 2, 3)
    assert calls == ["round", "round", "sync", "record", "round", "record",
                     "record", "round", "record", "record", "round", "record", "sync"]
    assert result == {"median_ms": 2.5, "p95_ms": 2.5, "minimum_ms": 2.5, "maximum_ms": 2.5}
