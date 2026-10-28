"""Sparse scoring parity and frozen source-cluster bootstrap regression."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from offload.dspark.probabilistic.pp4 import acceptance as a
from offload.dspark.probabilistic.pp4 import acceptance_frozen as f
from offload.dspark.worker.sampled_fused import FusedContext


@pytest.fixture(scope="module")
def frozen_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("frozen")
    f.materialize(path)
    return path


@pytest.fixture(scope="module")
def numeric(frozen_dir: Path) -> ModuleType:
    return f.module(frozen_dir, "numeric")


def fixture() -> tuple[torch.Tensor, ...]:
    logits = torch.full((5, 64), -1000., dtype=torch.float32)
    logits[:, :32] = 0.
    ids = torch.arange(32).repeat(5, 1)
    q = torch.full((5, 32), 1 / 32, dtype=torch.float32)
    top_ids = torch.arange(20).repeat(5, 1)
    lp = torch.full((5, 20), torch.tensor(1 / 64).log().item(), dtype=torch.float32)
    r = torch.full((5,), 44 / 64, dtype=torch.float32)
    labels = torch.tensor([0, 31, 32, 0, 0])
    tok_lp = lp[:, 0].clone()
    return logits, ids, q, top_ids, lp, r, labels, tok_lp


def test_sparse_matches_frozen_dense_score_and_preserves_prefix(numeric: ModuleType) -> None:
    logits, ids, q, target_ids, lp, residual, labels, tok_lp = fixture()
    actual = a.score_sparse(ids, q, logits.argmax(-1), target_ids, lp, residual, labels, tok_lp,
                            vocab=64, numeric=numeric)
    previous = numeric.score(logits, target_ids, lp, residual, labels, tok_lp,
                             (numeric.Arm("k32-t0.7", .7, 32),))
    torch.testing.assert_close(actual, previous, rtol=0, atol=0)
    assert actual[0, 2].tolist() == [1., 1., 0., 0., 0.]
    assert actual[1, 2].tolist() == [1., 0., 0., 0., 0.]


@pytest.mark.parametrize("bad", ["fractional", "mass", "nan", "duplicate", "unordered", "out_of_range"])
def test_invalid_emitted_q_fails_closed(numeric: ModuleType, bad: str) -> None:
    logits, ids, q, target_ids, lp, residual, labels, tok_lp = fixture()
    if bad == "fractional":
        q[0, 0] += 2 ** -28
    elif bad == "mass":
        q[0, 0] += 2 ** -24
    elif bad == "nan":
        q[0, 0] = float("nan")
    elif bad == "duplicate":
        ids[0, 1] = ids[0, 0]
    elif bad == "unordered":
        ids[0, :2] = torch.tensor([1, 0])
    else:
        ids[0, -1] = 64
    with pytest.raises(ValueError):
        a.score_sparse(ids, q, logits.argmax(-1), target_ids, lp, residual, labels, tok_lp,
                       vocab=64, numeric=numeric)


def reduced(frozen_dir: Path) -> dict[str, Any]:
    receipt = f.value(frozen_dir, "heldout-replay.json")
    receipt.update(schema="pp4.acceptance-replay.v1", arms=[a.ARM, "greedy"], semantic="dyadic24-floor-first-v1")
    for group in receipt["groups"].values():
        group["sum"] = [group["sum"][2], group["sum"][-1]]
    return receipt


def test_frozen_bootstrap_is_identical_at_same_aggregates(frozen_dir: Path) -> None:
    result = a.summarize(reduced(frozen_dir), frozen_dir)
    previous = f.value(frozen_dir, "heldout-summary.json")
    for label, comparison in result["comparisons"].items():
        assert comparison["change_from_PP2"] == 0
        assert comparison["CI95_delta"] == previous["views"][label]["arms"]["k32-t0.7"]["paired_source_cluster_CI95_delta"]
    assert result["comparisons"]["context:>=32K"]["CI95_delta"] is None


@pytest.mark.parametrize("bad", ["cluster", "rounds", "incomplete", "smoke", "manifest", "shards", "semantic", "weights"])
def test_population_drift_refuses_summary(frozen_dir: Path, bad: str) -> None:
    receipt = reduced(frozen_dir)
    first = next(iter(receipt["groups"].values()))
    if bad == "cluster":
        first["source_cluster_sha256"] = "0" * 64
    elif bad == "rounds":
        first["rounds"] += 1
    elif bad == "incomplete":
        receipt["complete"] = False
    elif bad == "smoke":
        receipt["smoke_limit"] = 10
    elif bad == "manifest":
        receipt["manifest_sha256"] = "0" * 64
    elif bad == "shards":
        receipt["verified_shards"].reverse()
    elif bad == "semantic":
        receipt["semantic"] = "unrounded-softmax"
    else:
        receipt["weights_tensor_payload_verified"] = False
    with pytest.raises(ValueError):
        a.summarize(receipt, frozen_dir)


def test_runtime_greedy_drift_is_reported_with_paired_intervals(frozen_dir: Path) -> None:
    receipt = reduced(frozen_dir)
    first = next(iter(receipt["groups"].values()))
    first["sum"][-1][2][0] -= 1
    result = a.summarize(receipt, frozen_dir)
    assert result["greedy_baseline_drift"]["status"] == "AGGREGATES_CHANGED"
    assert result["greedy_baseline_drift"]["changed_groups"] == 1
    all_view = result["comparisons"]["all"]
    assert all_view["greedy_change_from_PP2"] == pytest.approx(-1 / 1710)
    assert all_view["sampled_delta_frozen_greedy"] == pytest.approx(all_view["PP2_delta"])
    assert all_view["change_from_PP2"] == pytest.approx(1 / 1710)
    assert all_view["CI95_greedy_change_from_PP2"] is not None


def test_frozen_identity_refuses_tampering(frozen_dir: Path, tmp_path: Path) -> None:
    path = tmp_path / "numeric.py"
    path.write_bytes(f.read(frozen_dir, "numeric.py") + b"\n")
    with pytest.raises(ValueError, match="differs"):
        f.module(tmp_path, "numeric")


def test_outside_scratch_refused() -> None:
    with pytest.raises(ValueError, match="scratch"):
        f.materialize(Path("/tmp/pp4-acceptance-never-create"))


def test_committed_gpu_evidence_matches_sources_and_frozen_population() -> None:
    package = Path(a.__file__).parent
    receipt_path = package / "heldout-replay.json"
    receipt = json.loads(receipt_path.read_text())
    summary = json.loads((package / "heldout-summary.json").read_text())
    assert receipt["complete"] and receipt["smoke_limit"] == 0
    assert receipt["rounds"] == receipt["parity_rounds"] == 1710
    assert receipt["reference_q_different_rows"] == 0
    assert summary["input_sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert summary["frozen_commit"] == f.COMMIT and summary["frozen_sha256"] == f.HASHES
    assert summary["greedy_baseline_drift"]["changed_groups"] == 0
    for name, field in (("acceptance.py", "scoring_sha256"), ("acceptance_replay.py", "replay_sha256")):
        assert hashlib.sha256((package / name).read_bytes()).hexdigest() == receipt[field]
    worker = package.parents[1] / "worker"
    # The GPU receipt is historical. Reconstruct its EXACT producer bytes by
    # removing only Round6/Round8's host capture/priming additions; never rehash the
    # receipt to imply a new physical/GPU qualification. Every other byte,
    # including the entire Triton kernel and eager launch path, must match.
    producer = (worker / "sampled_fused.py").read_text()
    deltas = (
        "from collections.abc import Iterator\nfrom contextlib import contextmanager\n",
        "        self.capture_expected: frozenset[int] = frozenset()\n",
        inspect.getsource(FusedContext.capturing)+"\n",
        '''        if self.capture_expected:
            if frozenset(active) != self.capture_expected:
                raise RuntimeError("sampled capture rank ownership differs")
            for r in active:
                with torch.cuda.device(model.devs[r]):
                    if not torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(f"sampled rank {r}: expected capture on its current device stream; refusing eager peer launch")
''',
    )
    for delta in deltas:
        assert producer.count(delta) == 1
        producer = producer.replace(delta, "", 1)
    priming = '''                        compiled = kernel.warmup(*args, grid=grid, num_warps=4, enable_fp_fusion=False)
                        # HYPOTHESIS (PP4 Round8): compile-only warmup leaves
                        # Triton's CUDA handles lazy. Resolve both devices'
                        # launchers before either peer can wait on the GPU.
                        # Reading CompiledKernel.run loads, but never launches.
                        if compiled is None or not callable(compiled.run):
                            raise RuntimeError("paired sampled warmup did not load CUDA launch handles")
'''
    assert producer.count(priming) == 1
    producer = producer.replace(priming, '                        kernel.warmup(*args, grid=grid, num_warps=4, enable_fp_fusion=False)\n', 1)
    assert hashlib.sha256(producer.encode()).hexdigest() == receipt["fused_producer_sha256"]
    assert hashlib.sha256((worker / "sampled.py").read_bytes()).hexdigest() == receipt["producer_sha256"]
    for row in summary["comparisons"].values():
        assert row["point_within_PP2_interval"] is not False
