"""PP4 whole production-K2 TP2 emulation, graphs, and N4-state T0 identity.

Both logical ranks execute on the assigned solo SM86. This measures one complete
worker round without transport or physical-pair collectives. Captures and token
IDs remain in memory; output contains aggregate timings and hashes only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import torch

from .acceptance_replay import UUID, digest, guard, module_digest, require_lock, write


def clocks() -> str:
    return subprocess.run(["nvidia-smi", "-i", UUID,
                           "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit",
                           "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip()


def tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def measured(function: Callable[[], object], warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    # Repeated replays bracketed individually include host launch gaps, as NB1.
    event = cast(Callable[..., torch.cuda.Event], torch.cuda.Event)
    events = [(event(enable_timing=True), event(enable_timing=True))
              for _ in range(iterations)]
    for start, end in events:
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    values = sorted(float(start.elapsed_time(end)) for start, end in events)
    return {"median_ms": statistics.median(values), "p95_ms": values[int(.95 * (len(values) - 1))],
            "minimum_ms": values[0], "maximum_ms": values[-1]}


def synthetic_fixtures(hidden: int) -> list[tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]]:
    generator = torch.Generator().manual_seed(20261024)
    extra = torch.Generator().manual_seed(20261026)
    result = []
    for lasts, fourth in (((0, 63, 127), 31), ((128, 300, 32768), 65535),
                           ((150000, 32000, 1000), 262143)):
        windows = []
        for last in lasts:
            pos = torch.arange(max(0, last - 127), last + 1)
            windows.append((pos, torch.randn(len(pos), hidden, generator=generator).to(torch.bfloat16)))
        anchors = torch.randint(100, 10000, (3,), generator=generator)
        pos = torch.arange(max(0, fourth - 127), fourth + 1)
        windows.append((pos, torch.randn(len(pos), hidden, generator=extra).to(torch.bfloat16)))
        anchors = torch.cat((anchors, torch.randint(100, 10000, (1,), generator=extra)))
        all_lasts = torch.tensor([*lasts, fourth])
        inputs = torch.stack((torch.arange(4), anchors, all_lasts, (all_lasts - 127).clamp_min(0)))
        result.append((windows, inputs))
    return result


def load_synthetic(model: Any, fixture: tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]) -> torch.Tensor:
    windows, inputs = fixture
    model.reset_slots([0, 1, 2, 3])
    for slot, (positions, values) in enumerate(windows):
        model.insert_mx_eager(torch.full_like(positions, slot), positions, values)
    return inputs.pin_memory()


def captured_fixtures(model: Any, runtime: Path, root: Path) -> Iterator[tuple[str, torch.Tensor]]:
    sys.path.insert(0, str(runtime / "verify-source"))
    capture_class = importlib.import_module("verify.captures").CaptureSet
    captures = capture_class(str(root / "captures/A-capture-20261022/dspark-captures"))
    samples = [s for s in captures.samples if s.b == 0 and 8 <= s.round <= 19]
    if len(samples) != 12:
        raise RuntimeError("expected exactly twelve N4 captured states")
    for first in range(0, 12, 4):
        group = samples[first:first + 4]
        model.reset_slots([0, 1, 2, 3])
        for slot, sample in enumerate(group):
            arm = captures.arm_inputs(sample)
            if {kind for _, kind, _ in arm["window"]} - {"engine_mx", "taps"}:
                raise RuntimeError("unrecognized N4 window source")
            for source in ("engine_mx", "taps"):
                rows = [(position, row) for position, kind, row in arm["window"] if kind == source]
                # Match NB1/N4 captured-state insertion chunks.
                for first_row in range(0, len(rows), 128):
                    chunk = rows[first_row:first_row + 128]
                    positions = torch.tensor([position for position, _ in chunk]).pin_memory()
                    slots = torch.full_like(positions, slot).pin_memory()
                    values = torch.stack([row for _, row in chunk]).pin_memory()
                    if source == "taps":
                        model.insert_taps_graphed(slots, positions, values)
                    else:
                        model.insert_mx_eager(slots, positions, values)
        inputs = torch.tensor([[0, 1, 2, 3], [s.anchor for s in group], [s.P for s in group],
                               [max(0, s.P - 127) for s in group]]).pin_memory()
        yield f"captured-{first // 4}", inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("/opt/dsv41/x86host-data/dspark"))
    parser.add_argument("--runtime", type=Path, default=Path.home() / "dsv41-offload/n4")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--physical-pair", action="store_true", help="coordinator only, inside fenced qualification HOLD")
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.physical_pair and not args.qualification_only:
        raise ValueError("physical mode is qualification-only; no unregistered timing run")
    if args.physical_pair:
        from .pair_contract import guard as pair_guard
        active_guard = pair_guard
    else:
        active_guard = guard
        require_lock()
    active_guard()
    if min(args.iterations, args.warmup, args.repetitions) < 1:
        raise ValueError("positive timing counts required")
    if os.environ.get("DSPARK_SAMPLED_TOP32") != "1":
        raise RuntimeError("set DSPARK_SAMPLED_TOP32=1 before capture")
    if not args.physical_pair and torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible solo GPU required")
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
    if not args.physical_pair and actual_uuid.removeprefix("GPU-") != UUID.removeprefix("GPU-"):
        raise RuntimeError("actual CUDA UUID differs from assigned solo")
    if not args.out.resolve().is_relative_to(Path("/opt/dsv41/x86host-data/pp4")):
        raise RuntimeError("output must remain inside PP4 x86host scratch")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    checkpoint = importlib.import_module("offload.dspark.worker.checkpoint").Checkpoint(str(args.weights), threads=8)
    checkpoint_identity = importlib.import_module("offload.dspark.worker.identity").compute(
        str(args.weights), manifest=checkpoint.manifest)
    if (checkpoint_identity["drafter_identity"] != "be2c1436fa9f9e66b398e1b1da2e140e6fddc82a11939e2ac4240a0ca79543bd"
            or checkpoint_identity["markov_identity"] != "0b6e71e638abd5abce2db433965bb5ff523cbbe92ed7ae54324a50ccaeffa19a"):
        raise RuntimeError("checkpoint identities differ from deployed production K2")
    tp2 = importlib.import_module("offload.dspark.worker.tp2")
    device = torch.device("cuda:0")
    started = time.monotonic()
    clock_log = [] if args.physical_pair else [clocks()]
    print("PHASE loading complete K2 TP2 model", flush=True)
    model = tp2.TPDrafter(checkpoint, [device, torch.device("cuda:1") if args.physical_pair else device], slots=4, max_rows=32,
                         max_model_len=1048576, markov_cache_rows=8192, fuse_pairs=False)
    model.capture(batch_sizes=(1, 2, 3, 4), ks=(5,), insert_rows=(8, 16, 32, 64, 128),
                  n_outs=(4, 5), margins=False)
    active_guard()
    print("PHASE complete K2 graphs captured", flush=True)
    keys = {batch: torch.tensor([[4104, slot + 1, 0] for slot in range(batch)], dtype=torch.int64).pin_memory()
            for batch in range(1, 5)}
    masks = {batch: torch.ones(batch, dtype=torch.bool).pin_memory() for batch in range(1, 5)}
    zero_masks = {batch: torch.zeros(batch, dtype=torch.bool).pin_memory() for batch in range(1, 5)}
    fixtures = synthetic_fixtures(checkpoint.shape.hidden)
    identity: list[dict[str, object]] = []
    timing: dict[str, list[dict[str, float]]] = {}
    archived_comparison: list[dict[str, object]] = []
    archive_path = Path(__file__).with_name("n4-greedy-reference.json")
    archived = json.loads(archive_path.read_text())

    def check(label: str, inputs: torch.Tensor) -> None:
        archived_label = label.replace("captured-", "capture-group-")
        for row in range(4):
            single = inputs[:, row:row + 1].contiguous().pin_memory()
            expected = archived["b1_digests"][f"{archived_label}/{row}"]
            for depth in (4, 5):
                ids, confidence, _ = model.draft_graphed(single, 5, n_out=depth)
                ids_hash, confidence_hash = tensor_digest(ids), tensor_digest(confidence)
                archived_comparison.append({"fixture": label, "row": row, "depth": depth,
                                            "ids_exact": ids_hash == expected[f"graph{depth}_ids"],
                                            "confidence_exact": confidence_hash == expected[f"graph{depth}_conf"],
                                            "ids_sha256": ids_hash, "confidence_sha256": confidence_hash})
        for batch in range(1, 5):
            packed = inputs[:, :batch].contiguous().pin_memory()
            for depth in (4, 5):
                original_ids, original_conf, _ = model.draft_graphed(packed, 5, n_out=depth)
                reference_ids, reference_conf = original_ids.clone(), original_conf.clone()
                actual_ids, actual_conf, _, _ = model.draft_sampled(packed, 5, keys[batch], zero_masks[batch], n_out=depth)
                torch.cuda.synchronize()
                exact = torch.equal(reference_ids, actual_ids) and torch.equal(reference_conf, actual_conf)
                identity.append({"fixture": label, "batch": batch, "depth": depth, "mode": "all-T0",
                                 "exact": exact, "t0_requests": batch,
                                 "proposal_sha256": tensor_digest(reference_ids)})
                if not exact:
                    raise RuntimeError(f"T0 greedy identity failed at {label}/B{batch}/depth{depth}")
                if batch >= 2:
                    for parity in (0, 1):
                        mixed = (torch.arange(batch) % 2 == parity).pin_memory()
                        ids, confidence, _, q = model.draft_sampled(packed, 5, keys[batch], mixed, n_out=depth)
                        t0 = ~mixed.to(device)
                        mixed_exact = (torch.equal(reference_ids[t0], ids[t0])
                                       and torch.equal(reference_conf[t0], confidence[t0]))
                        torch.cuda.synchronize()
                        identity.append({"fixture": label, "batch": batch, "depth": depth,
                                         "mode": f"mixed-T0/T1-parity{parity}", "exact": mixed_exact,
                                         "t0_requests": int(t0.sum().item()),
                                         "proposal_sha256": tensor_digest(reference_ids[t0])})
                        if not mixed_exact or bool(q[t0, :depth].any().item()):
                            raise RuntimeError(f"mixed T0 greedy identity failed at {label}/B{batch}/depth{depth}")
        active_guard()
        print("PHASE T0 identity " + label, flush=True)
        write(args.out, {"label": "MEASURED", "complete": False, "identity": identity, "archived_comparison": archived_comparison, "timing": timing})

    for index, fixture in enumerate(fixtures):
        check(f"synthetic-{index}", load_synthetic(model, fixture))
    timing_inputs: torch.Tensor | None = None
    for label, capture_inputs in captured_fixtures(model, args.runtime, args.weights):
        check(label, capture_inputs)
        timing_inputs = capture_inputs
    if timing_inputs is None:
        raise RuntimeError("captured timing fixture unavailable")
    # Keep the last real N4 window resident for matched ON/OFF measurements.
    inputs = timing_inputs
    plan = [(batch, depth, arm) for batch in range(1, 5) for depth in (4, 5) for arm in ("off", "on")]
    for repetition in range(0 if args.qualification_only else args.repetitions):
        clock_log.append(clocks())
        order = plan if repetition % 2 == 0 else list(reversed(plan))
        for batch, depth, arm in order:
            packed = inputs[:, :batch].contiguous().pin_memory()
            def function(packed: torch.Tensor = packed, depth: int = depth,
                         arm: str = arm, batch: int = batch) -> object:
                if arm == "off":
                    return model.draft_graphed(packed, 5, n_out=depth)
                return model.draft_sampled(packed, 5, keys[batch], masks[batch], n_out=depth)
            timing.setdefault(f"B{batch}/depth{depth}/{arm}", []).append(
                measured(function, args.warmup, args.iterations))
            active_guard()
        print(f"PHASE timing repetition {repetition + 1}/{args.repetitions}", flush=True)
        write(args.out, {"label": "MEASURED", "complete": False, "identity": identity, "archived_comparison": archived_comparison, "timing": timing})
    if not args.physical_pair:
        clock_log.append(clocks())
    if args.physical_pair and not all(row["ids_exact"] and row["confidence_exact"] for row in archived_comparison):
        raise RuntimeError("physical K2 differs from archived solo N4 receipt")
    summary = {cell: {"median_ms": statistics.median(run["median_ms"] for run in runs),
                      "p95_ms": max(run["p95_ms"] for run in runs), "runs": runs}
               for cell, runs in timing.items()}
    result = {"label": "MEASURED", "complete": True,
              "physical_pair": args.physical_pair,
              "scope": "physical K2 TP2 qualification" if args.physical_pair else "whole K2 TP2 emulation, both logical ranks, CUDA graphs; no physical-pair collectives/transport",
              "archived_comparison": archived_comparison,
              "archived_ids_exact": sum(bool(cell["ids_exact"]) for cell in archived_comparison),
              "archived_confidence_exact": sum(bool(cell["confidence_exact"]) for cell in archived_comparison),
              "archive_receipt_sha256": digest(archive_path),
              "identity": identity, "identity_checks": len(identity), "identity_exact": sum(bool(i["exact"]) for i in identity),
              "request_identity_checks": sum(int(str(i["t0_requests"])) for i in identity),
              "n4_states": 24, "timing_fixture": "captured-2 (four real N4 request states)", "query_block_size": 5, "output_depths": [4, 5], "timing": summary, "iterations": args.iterations, "warmup": args.warmup,
              "repetitions": args.repetitions, "seconds": time.monotonic() - started,
              "torch_version": torch.__version__, "clocks": clock_log, "checkpoint_identity": checkpoint_identity,
              "source_sha256": {name: module_digest(name) for name in (
                  "offload.dspark.worker.tp2", "offload.dspark.worker.sampled_fused",
                  "exllamav3_ext", "exllamav3_ext_facc")},
              "runner_sha256": digest(Path(__file__)), "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    write(args.out, result)
    print("RESULT_JSON " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
