"""Isolated current-worker profile; no listener and no production-worker access.

Run on x86host with the O2 environment and W1_GPU_OK. Records paired CPU timestamps /
CUDA events for each complete insert + draft + reply-copy sample. No percentile
subtraction. Eager traces and optional CUPTI runs are separate diagnostics, never
subtracted from these critical-path measurements. Synthetic inputs and raw outputs
stay in the local output directory. A single model is retained (<20 GiB W1b process budget).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

from .profile_support import PROCESS_LIMIT_MIB, expert_counts, require_gpu_window, summarize


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--cupti", action="store_true")
    p.add_argument("--cadence-ms", type=float, default=0)
    p.add_argument("--revision", required=True, help="exact git archive revision synced into this W1 snapshot")
    args = p.parse_args()
    if args.iters < 1 or args.cadence_ms < 0:
        p.error("iters must be positive and cadence-ms nonnegative")
    if re.fullmatch(r"[0-9a-f]{40}", args.revision) is None:
        p.error("revision must be the full 40-character source commit")
    require_gpu_window()  # no torch import / CUDA context before ownership check
    import torch
    from ..checkpoint import Checkpoint
    from ..model import DSparkDrafter
    from .bench import breakdown

    args.out.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()
    telemetry_error: list[str] = []

    def telemetry() -> None:
        try:
            with (args.out / "telemetry.jsonl").open("w") as f:
                while not stop.is_set():
                    q = "uuid,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used,utilization.gpu"
                    data = subprocess.run(["nvidia-smi", "--id=" + os.environ["CUDA_VISIBLE_DEVICES"],
                                           "--query-gpu=" + q, "--format=csv,noheader,nounits"],
                                          text=True, capture_output=True, check=True).stdout.strip()
                    f.write(json.dumps({"cpu_ns": time.perf_counter_ns(), "query": q, "values": data}) + "\n")
                    f.flush()
                    stop.wait(.1)
        except Exception as exc:
            telemetry_error.append(str(exc))

    thread = threading.Thread(target=telemetry, daemon=True)
    thread.start()
    try:
        torch.set_num_threads(2)
        dev = torch.device("cuda", 0)
        require_gpu_window()
        m = DSparkDrafter(Checkpoint(str(args.root)), dev, slots=8)
        rng = torch.Generator().manual_seed(41)
        tokens = torch.randint(m.s.vocab, (1100,), generator=rng)
        # Fixed synthetic taps; save the exact inputs before any measured work.
        taps_cpu = (torch.randn(1100, m.s.hidden * 3, generator=rng) * 3).to(torch.bfloat16)
        torch.save({"tokens": tokens, "taps": taps_cpu}, args.out / "synthetic.pt")
        taps = taps_cpu.to(dev)
        last_values = (300, 500, 700, 900)
        for slot, last in enumerate(last_values):
            require_gpu_window()
            positions = torch.arange(last - 127, last + 1, device=dev)
            m.insert(torch.full_like(positions, slot), positions, m.main_x(taps[positions]))
        # capture() owns shared input buffers: calling it repeatedly would invalidate earlier graphs.
        require_gpu_window()
        m.capture(batch_sizes=(1, 2, 3, 4), ks=(5,), insert_rows=(8, 16, 32, 64, 128),
                  n_outs=(1, 2, 3, 4, 5), margins=True)
        require_gpu_window()
        # Allocator is not process total; include measured context/module overhead plus safety.
        process = subprocess.run(["nvidia-smi", "--id=" + os.environ["CUDA_VISIBLE_DEVICES"],
                                  "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                                 text=True, capture_output=True, check=True).stdout
        own_mib = next(float(line.split(",")[1]) for line in process.splitlines()
                       if int(line.split(",")[0]) == os.getpid())
        if own_mib >= PROCESS_LIMIT_MIB:
            raise RuntimeError(f"W1b process exceeds 20-GiB ceiling: {own_mib} MiB")
        provenance = {"pid": os.getpid(), "process_mib": own_mib, "device": torch.cuda.get_device_name(),
                      "torch": torch.__version__, "base": args.revision,
                      "sources": {name: hashlib.sha256((Path(__file__).parents[1] / name).read_bytes()).hexdigest()
                                  for name in ("model.py", "fused.py", "server.py")}, "memory": m.memory_table()}
        (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2))
        summary: dict[str, object] = {}
        with (args.out / "samples.jsonl").open("w") as raw, (args.out / "replies.jsonl").open("w") as replies:
            for batch in (1, 2, 3, 4):
                slots = torch.arange(batch, device=dev)
                last = torch.tensor(last_values[:batch], device=dev)
                # Deliberately exercise cached and uncached first Markov inputs.
                for cache_kind in ("hit", "miss"):
                    require_gpu_window()
                    candidates = torch.where((m.markov.slot_of_token >= 0) if cache_kind == "hit"
                                             else (m.markov.slot_of_token < 0))[0]
                    if candidates.numel() < batch:
                        raise RuntimeError(f"not enough Markov {cache_kind} candidates")
                    anchors = candidates[:batch]
                    for n_out in range(1, 6):
                        for rows in (batch, 6 * batch):
                            # Keep inserted context identical across repeats, modes and depths.
                            sr = torch.arange(batch).repeat_interleave(rows // batch)
                            pr = torch.cat([torch.arange(lp - rows // batch + 1, lp + 1)
                                            for lp in last_values[:batch]])
                            host_taps = taps_cpu[pr].pin_memory()
                            sr_host, pr_host = sr.pin_memory(), pr.pin_memory()
                            sr_dev, pr_dev = sr.to(dev), pr.to(dev)
                            taps_dev = host_taps.to(dev)
                            case = f"B{batch}_n{n_out}_rows{rows}_{cache_kind}"
                            case_report: dict[str, object] = {}
                            for mode in ("graph", "eager"):
                                samples: list[dict[str, float]] = []
                                for iteration in range(args.iters + 3):
                                    require_gpu_window()
                                    if telemetry_error:
                                        raise RuntimeError("telemetry failed: " + telemetry_error[0])
                                    if args.cadence_ms:
                                        stop.wait(args.cadence_ms / 1000)
                                        require_gpu_window()
                                    torch.cuda.synchronize()
                                    events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
                                    cpu = [time.perf_counter_ns()]
                                    events[0].record()
                                    taps_dev.copy_(host_taps, non_blocking=True)
                                    sr_dev.copy_(sr_host, non_blocking=True)
                                    pr_dev.copy_(pr_host, non_blocking=True)
                                    events[1].record()
                                    cpu.append(time.perf_counter_ns())
                                    m.insert_taps_graphed(sr_dev, pr_dev, taps_dev)
                                    events[2].record()
                                    cpu.append(time.perf_counter_ns())
                                    ids, conf = (m.draft_graphed(slots, anchors, last, 5, n_out=n_out) if mode == "graph"
                                                 else m.draft(slots, anchors, last, 5, n_out=n_out))
                                    events[3].record()
                                    cpu.append(time.perf_counter_ns())
                                    packed = torch.cat((ids.to(torch.float64), conf.to(torch.float64)), 1)
                                    events[4].record()
                                    cpu.append(time.perf_counter_ns())
                                    reply = packed.cpu().numpy()  # exact production D2H + wait
                                    events[5].record()
                                    cpu.append(time.perf_counter_ns())
                                    events[5].synchronize()
                                    cpu.append(time.perf_counter_ns())
                                    sample = {"wall_us": (cpu[-1] - cpu[0]) / 1000,
                                              "gpu_us": events[0].elapsed_time(events[-1]) * 1000}
                                    for j, phase in enumerate(("h2d", "insert", "draft", "reply_pack", "d2h")):
                                        sample[phase + "_gpu_us"] = events[j].elapsed_time(events[j + 1]) * 1000
                                        sample[phase + "_cpu_us"] = (cpu[j + 1] - cpu[j]) / 1000
                                    sample["final_event_wait_us"] = (cpu[-1] - cpu[-2]) / 1000
                                    if iteration >= 3:
                                        samples.append(sample)
                                        raw.write(json.dumps({"case": case, "mode": mode, "iteration": iteration - 3,
                                                              "cpu_ns": cpu, **sample}) + "\n")
                                        raw.flush()
                                case_report[mode] = summarize(samples)
                                # This is synthetic output, saved only on x86host.
                                if mode == "graph":
                                    graph_reply = reply.copy()
                                    replies.write(json.dumps({"case": case, "anchors": anchors.cpu().tolist(),
                                                              "packed_reply": graph_reply.tolist()}) + "\n")
                                    replies.flush()
                                elif not (reply == graph_reply).all():
                                    raise RuntimeError(f"graph/eager output parity failed for {case}")
                            require_gpu_window()
                            tr: dict[str, torch.Tensor] = {}
                            ids, conf = m.draft(slots, anchors, last, 5, trace=tr, n_out=n_out)
                            prev = torch.cat((anchors[:, None], ids[:, :n_out - 1]), 1)
                            case_report["markov_hits"] = (m.markov.slot_of_token[prev] >= 0).cpu().tolist()
                            case_report["routed_rows"] = {
                                f"stage{i}": expert_counts(tr[f"stage{i}.experts"].reshape(-1).cpu().tolist(), m.s.n_experts)
                                for i in range(m.s.n_stages)}
                            if args.cupti:
                                require_gpu_window()
                                case_report["separate_kernel_diagnostic"] = breakdown(
                                    lambda: m.draft_graphed(slots, anchors, last, 5, n_out=n_out), reps=2)
                            summary[case] = case_report
                            (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
                            print(case, "complete", flush=True)
        require_gpu_window()
    finally:
        stop.set()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
