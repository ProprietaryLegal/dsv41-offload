"""Regression for non-additive percentile timing attribution in the CPU benchmark."""

from __future__ import annotations

import importlib.util
import statistics
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "dspark/spark/tools/bench_round.py"
_SPEC = importlib.util.spec_from_file_location("r1_bench_round", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_BENCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCH)


def test_host_overhead_pairs_correlated_phases_before_taking_median() -> None:
    timing = _BENCH.RoundTimingStats()
    for total, gather, rtt in ((102., 1., 100.), (21., 10., 10.), (102., 100., 1.)):
        timing.begin_round()
        timing.observe("gather_us", gather)
        timing.observe("rtt_us", rtt)
        timing.observe("rank0_round_us", total)
    # Unpaired exact median subtraction gives 82 us; binning introduces further error.
    assert statistics.median(timing.python_pack_us) == 1.0
    assert timing.python_pack_us == [1.0, 1.0, 1.0]


def test_failed_round_cannot_reuse_previous_round_rtt() -> None:
    timing = _BENCH.RoundTimingStats()
    timing.begin_round()
    timing.observe("gather_us", 1.)
    timing.observe("rtt_us", 2.)
    timing.observe("rank0_round_us", 4.)
    timing.begin_round()
    timing.observe("gather_us", 5.)
    with pytest.raises(RuntimeError, match="lacks paired timing.*rtt_us"):
        timing.observe("rank0_round_us", 10.)
