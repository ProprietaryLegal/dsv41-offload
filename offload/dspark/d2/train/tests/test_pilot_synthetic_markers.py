"""Every dataset writer's synthetic marker must reach the pilot refusal path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from offload.dspark.d2.train import fast_policy, pilot, safety
from offload.dspark.d2.train.data import sha256
from offload.dspark.d2.train.tests.test_pilot_policy import Population, receipt_fixture


MARKERS: list[dict[str, object]] = [
    {"synthetic": True},
    {"synthetic": False, "checks": {"synthetic": True}},
    {"synthetic": False, "taps_source": "synthetic"},
    {"synthetic": False, "checks": {"taps_source": "synthetic"}},
    {"synthetic": False, "generation": {"synthetic": True}},
]


@pytest.mark.parametrize("marker", MARKERS)
def test_synthetic_markers_refuse_real_prepare_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker: dict[str, object],
) -> None:
    manifest = tmp_path/"MANIFEST.json"
    manifest.write_text(json.dumps(marker))
    assert pilot.manifest_is_synthetic(manifest)
    monkeypatch.setattr(safety, "assert_gpu_lease", lambda: None)
    monkeypatch.setattr(pilot, "D2Blocks", lambda *_: Population())
    monkeypatch.setattr(pilot, "load", lambda *a, **k: pytest.fail("synthetic real plan reached model load"))
    with pytest.raises(ValueError, match="rehearsal flag"):
        pilot.prepare(argparse.Namespace(manifest=manifest, rehearsal=False))


@pytest.mark.parametrize("marker", MARKERS)
def test_production_refuses_real_receipt_for_synthetic_manifest_even_with_updated_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker: dict[str, object],
) -> None:
    receipt, manifest, parity, g1 = receipt_fixture(tmp_path, monkeypatch)
    manifest.write_text(json.dumps(marker))
    raw = json.loads(receipt.read_text())
    raw["plan"]["manifest_sha256"] = sha256(manifest)
    raw["plan"]["sample"]["manifest_sha256"] = sha256(manifest)
    receipt.write_text(json.dumps(raw))
    # This reaches synthetic classification after source, sample and input
    # hashes validate; it fails before thresholds/checkpoint verification.
    with pytest.raises(ValueError, match="pilot synthetic label mismatch"):
        fast_policy.require_pilot(receipt, "base", manifest, parity, g1, 128, 8192, 32)


@pytest.mark.parametrize("metadata", [{}, {"synthetic": False},
    {"synthetic": False, "checks": {"synthetic": False}, "generation": {"synthetic": False}, "taps_source": "prefill_replay"}])
def test_real_manifest_markers_remain_real(tmp_path: Path, metadata: dict[str, object]) -> None:
    manifest = tmp_path/"MANIFEST.json"
    manifest.write_text(json.dumps(metadata))
    assert not pilot.manifest_is_synthetic(manifest)
