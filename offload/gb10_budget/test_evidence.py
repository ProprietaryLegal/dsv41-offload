"""Protect accounting and aggregate-only export, including repeated rep numbers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from offload.gb10_budget.evidence import analyze
from offload.gb10_budget.model import collective_payload, expected_distinct, expert_bytes_per_rank, prefill


def test_global_counters_not_assigned_to_concurrent_requests(tmp_path: Path) -> None:
    path = tmp_path / "bench.jsonl"
    common = {"target_ctx": 32000, "rep": 0, "mixed": False, "cache_state": "cold"}
    path.write_text("\n".join(json.dumps(row) for row in [
        common | {"type": "request", "run_id": "one", "concurrency": 1, "decode_tps": 50,
                  "prompt_text": "PRIVATE_SENTINEL", "answer_text": "PRIVATE_SENTINEL"},
        common | {"type": "run", "run_id": "one", "concurrency": 1,
                  "spec": {"mean_accepted_per_round": 1}},
        common | {"type": "request", "run_id": "four", "concurrency": 4, "decode_tps": 30},
        common | {"type": "run", "run_id": "four", "concurrency": 4,
                  "spec": {"mean_accepted_per_round": 2}},
    ]))
    result = analyze([tmp_path])
    assert "PRIVATE_SENTINEL" not in json.dumps(result)
    summaries = result["summaries"]
    assert isinstance(summaries, list)
    by_batch = {int(row["batch"]): row for row in summaries
                if isinstance(row, dict) and isinstance(row["batch"], int)}
    assert by_batch[1]["acceptance_normalized_round_ms_ESTIMATED"] == 40.0
    assert by_batch[4]["acceptance_normalized_round_ms_ESTIMATED"] is None


def test_corrupt_evidence_fails_with_locator(tmp_path: Path) -> None:
    (tmp_path / "broken.jsonl").write_text("{broken\n")
    with pytest.raises(ValueError, match=r"broken.jsonl:1: malformed JSON"):
        analyze([tmp_path])


def test_routing_and_wire_units() -> None:
    assert expected_distinct(0) == 0
    assert expected_distinct(1) == 6
    assert expected_distinct(20) < 120
    assert expert_bytes_per_rank() == 6082560
    assert collective_payload(6) == 61440
    with pytest.raises(ValueError):
        expected_distinct(-1)


def test_prefill_replay_cut_and_two_independent_links() -> None:
    budget = prefill(4096)
    assert budget["layer_rows"] == 20 * (4096 + 128)
    assert budget["engram_bytes_per_rank_2layers"] == 25952256
    assert budget["reduce_wire_ms_200Gbps"] == pytest.approx(69.206016)
    assert prefill(4096, 1)["layer_rows"] == 20 * (4096 + 1)
    assert prefill(64)["layer_rows"] == 40 * 64
