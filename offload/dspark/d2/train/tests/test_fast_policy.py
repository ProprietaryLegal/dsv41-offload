from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import pytest

from offload.dspark.d2.train.fast_policy import BACKEND, numeric_sources, require_agreement


class Row(TypedDict):
    position1: bool
    exact: bool
    kl: list[float]
    loss_relative_error: float


Gradient = TypedDict("Gradient", {"pass": bool, "cosine": float})


class Population(TypedDict):
    rows: list[Row]
    gradient_groups: dict[str, Gradient]
    gradient_tensors: dict[str, Gradient]


@pytest.mark.parametrize("failure", (None, "pass", "loss", "cosine", "source", "coverage", "base", "batch", "chunk", "rank"))
def test_fast_training_refuses_failed_stale_or_incomplete_receipts(tmp_path: Path, failure: str | None) -> None:
    row: Row = {"position1": True, "exact": True, "kl": [0.] * 5, "loss_relative_error": 0.}
    pops: dict[str, Population] = {p: {"rows": [row.copy() for _ in range(n)],
                "gradient_groups": {"norm": {"pass": True, "cosine": 1.}},
                "gradient_tensors": {"norm": {"pass": True, "cosine": 1.}}} for p, n in (("n1", 4), ("captures", 50))}
    report = {"schema": "dsv41.d2.fast-agreement.v1", "backend": BACKEND, "pass": True,
              "frozen_identity": "base", "source_sha256": numeric_sources(), "populations": pops,
              "batch_size": 1, "vocab_chunk": 4096, "lora_rank": 32}
    if failure == "pass":
        report["pass"] = False
    elif failure == "loss":
        pops["captures"]["rows"][0]["loss_relative_error"] = .0051
    elif failure == "cosine":
        pops["captures"]["gradient_groups"]["norm"]["cosine"] = .9989
    elif failure == "source":
        report["source_sha256"] = {}
    elif failure == "coverage":
        pops["captures"]["rows"].pop()
    elif failure == "base":
        report["frozen_identity"] = "other"
    elif failure == "batch":
        report["batch_size"] = 32
    elif failure == "chunk":
        report["vocab_chunk"] = 8192
    elif failure == "rank":
        report["lora_rank"] = 64
    path = tmp_path / "agreement.json"
    path.write_text(json.dumps(report))
    if failure:
        with pytest.raises(ValueError):
            require_agreement(path, "base", {"norm"})
    else:
        assert len(require_agreement(path, "base", {"norm"})) == 64
