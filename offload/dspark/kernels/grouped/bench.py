"""Locked native expanded benchmarks; reports only local operation boundaries.

All captures/tensors remain on x86host. Timings are graph CUDA events, 3 trials,
20 replays/trial. No per-replay synchronization. Telemetry samples only assigned GPU.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Callable

import torch

from api import Dense, ExpandedExperts, GroupedWoA, HeadEpilogue
from build import UUID, load_extension, require_lock
from native_fixture import NativeFixture
from verify import qdq_ref


def timed(fn: Callable[[], object], replays: int = 20) -> dict[str, object]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    gpu, wall = [], []
    for _ in range(3):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter_ns()
        a.record()
        for _ in range(replays):
            g.replay()
        b.record()
        b.synchronize()
        gpu.append(a.elapsed_time(b)*1000/replays)
        wall.append((time.perf_counter_ns()-start)/1000/replays)
    return {"gpu_us": gpu, "gpu_median_us": median(gpu), "wall_us_per_replay": wall,
            "trials": 3, "replays_per_trial": replays, "boundary": "captured operation only"}


class Reporter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[dict[str, object]] = []
        self.error: BaseException | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.telemetry, daemon=True)
        self.thread.start()

    def telemetry(self) -> None:
        try:
            with self.path.with_suffix(".telemetry.jsonl").open("w") as f:
                while not self.stop.is_set():
                    proc = subprocess.run(["nvidia-smi", "-i", UUID,
                        "--query-gpu=uuid,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
                    f.write(json.dumps({"time_ns": time.time_ns(), "csv": proc.stdout.strip()})+"\n")
                    f.flush()
                    self.stop.wait(.5)
        except BaseException as exc:
            self.error = exc

    def log(self, row: dict[str, object]) -> None:
        if self.error is not None:
            raise RuntimeError("telemetry failed") from self.error
        row["time_ns"] = time.time_ns()
        self.records.append(row)
        self.path.write_text(json.dumps(self.records, indent=2)+"\n")
        print(row, flush=True)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("telemetry did not stop")
        if self.error is not None:
            raise RuntimeError("telemetry failed") from self.error


def experts(ext: ModuleType, native: NativeFixture, routes: Path, log: Reporter) -> None:
    cases = torch.load(routes, weights_only=False, map_location="cpu")
    live = [c for c in cases if c["stage"] == 0 and c["x"].shape[0] == 5
            and c.get("category") == "live"]
    if not live:
        raise RuntimeError("no live (non-warmup, sampled) recorded inputs")
    case = max(live, key=lambda c: int(torch.unique(c["ids"]).numel()))
    dev = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    wg, wu, wd = native.expert_shard(0, torch.bfloat16, dev)
    log.log({"test": "native_load", "source": "official 1017 mtp.0", "rank": 0,
             "decoded_weight_bytes": sum(t.numel()*t.element_size() for t in (wg, wu, wd)),
             "load_peak_allocated": torch.cuda.max_memory_allocated(),
             "load_peak_reserved": torch.cuda.max_memory_reserved(),
             "all_384_matrices_bit_exact_bf16": True})
    native_receipt = log.path.with_suffix(".native.json")
    native_receipt.write_text(json.dumps(native.records, indent=2)+"\n")
    for dtype in (torch.bfloat16, torch.float16):
        if dtype == torch.float16:
            converted = []
            for w in (wg, wu, wd):
                narrowed = w.to(dtype)
                if not torch.equal(w.float(), narrowed.float()):
                    log.log({"test": "dtype_admission", "dtype": str(dtype), "admitted": False,
                             "reason": "native values change on FP16 narrowing"})
                    return
                converted.append(narrowed)
            wg, wu, wd = converted
            log.log({"test": "dtype_admission", "dtype": str(dtype), "admitted": True,
                     "all_weight_values_exact": True})
        for rows in (1, 2, 3, 4, 5, 10, 15, 20, 40, 80):
            repeats = (rows+4)//5
            x = case["x"].repeat(repeats, 1)[:rows].to(device=dev, dtype=dtype).contiguous()
            ids = case["ids"].repeat(repeats, 1)[:rows].to(dev).contiguous()
            rw = case["weights"].repeat(repeats, 1)[:rows].to(dev).contiguous()
            counts = torch.bincount(ids.flatten().long(), minlength=128)
            outputs = {}
            for mode in ("grouped", "small"):
                m = ExpandedExperts(ext, wg, wu, wd, rows, mode=mode)
                m.run(x, ids, rw)
                outputs[mode] = m.output.clone()
                assert int(m.status.item()) == 0 and bool(m.output.isfinite().all())
                if dtype == torch.float16:
                    original = case["x"].repeat(repeats, 1)[:rows].to(dev)
                    original_qdq = qdq_ref(original)
                    narrowed_qdq = qdq_ref(x).bfloat16()
                    if not torch.equal(original_qdq, narrowed_qdq):
                        raise RuntimeError("FP16 execution changes native input QDQ values")
                    del original, original_qdq, narrowed_qdq
                result = timed(lambda: m.run(x, ids, rw))
                slot_result = timed(lambda: m.run_slots(x, ids, rw))
                log.log({"test": "expert_slots", "dtype": str(dtype), "rows": rows, "mode": mode,
                         "output_shape": list(m.slot_output.shape), **slot_result})
                log.log({"test": "experts", "dtype": str(dtype), "rows": rows, "mode": mode,
                         "source_round": case["round"], "routing_source": "K2 captured round control",
                         "input_pattern": "actual capture" if rows==5 else "capture slice or repeated capture",
                         "active_experts": int((counts>0).sum()), "max_rows_per_expert": int(counts.max()),
                         "workspace_bytes": m.workspace_bytes, "native_qdq": True,
                         "tp_collective_included": False, **result})
                if mode == "grouped" and rows in (1, 5):
                    # Independent FP32 dense dots of precisely the same native values/QDQ.
                    qx = qdq_ref(x)
                    refg, refu = torch.empty_like(m.state[12]), torch.empty_like(m.state[13])
                    inverse = m.state[1].cpu().tolist()
                    for pos, route in enumerate(inverse):
                        e = int(ids.flatten()[route])
                        row = route//3
                        refg[pos] = wg[e].float() @ qx[row].float()
                        refu[pos] = wu[e].float() @ qx[row].float()
                    for got, ref in ((m.state[12], refg), (m.state[13], refu)):
                        torch.testing.assert_close(got, ref, rtol=2e-4, atol=2e-4)
                    # FP64 corner dots, using CPU independent accumulation.
                    pos = 0
                    route = inverse[pos]
                    e = int(ids.flatten()[route])
                    ref64 = wg[e, :64].cpu().double() @ qx[route//3].cpu().double()
                    got64 = m.state[12][pos, :64].cpu().double()
                    rel64 = float((got64-ref64).norm()/ref64.norm())
                    if rel64 > 2e-6:
                        raise RuntimeError(f"native GEMM relative error vs FP64 {rel64}")
                    log.log({"test": "native_dot_reference", "rows": rows, "dtype": str(dtype),
                             "gate_fp64_corner_rel": rel64,
                             "gate_bf16_match_fraction": float((m.state[12].bfloat16()==refg.bfloat16()).float().mean()),
                             "up_bf16_match_fraction": float((m.state[13].bfloat16()==refu.bfloat16()).float().mean()),
                             "gate_fp32_max_abs": float((m.state[12]-refg).abs().max()),
                             "up_fp32_max_abs": float((m.state[13]-refu).abs().max())})
                del m
            rel = float((outputs["grouped"]-outputs["small"]).norm()/outputs["grouped"].norm())
            log.log({"test": "grouped_small_agreement", "rows": rows, "dtype": str(dtype),
                     "relative_frobenius": rel,
                     "note": "FP32 reduction order can cross native BF16/QDQ boundaries"})
            del outputs, x, ids, rw
    # Requested low-reuse controls: real input activations, deliberately synthetic IDs.
    for rows in (5, 10, 15, 20):
        repeats = (rows+4)//5
        x = case["x"].repeat(repeats, 1)[:rows].to(device=dev, dtype=wg.dtype).contiguous()
        rw = case["weights"].repeat(repeats, 1)[:rows].to(dev).contiguous()
        for reuse, unique in (("all_distinct", rows*3), ("70_percent_unique", max(3, round(rows*3*.7)))):
            ids = (torch.arange(rows*3, device=dev)%unique).reshape(rows,3).int()
            for mode in ("grouped", "small"):
                m = ExpandedExperts(ext, wg, wu, wd, rows, mode=mode)
                log.log({"test": "low_reuse_experts", "rows": rows, "dtype": str(wg.dtype),
                         "mode": mode, "pattern": reuse, "unique_experts": unique,
                         "activation_source": "repeated recorded live K2 input",
                         "routing_source": "synthetic device IDs", **timed(lambda: m.run_slots(x, ids, rw))})
                del m
    log.log({"test": "expert_peak", "allocated": torch.cuda.max_memory_allocated(),
             "reserved": torch.cuda.max_memory_reserved()})


def reuse(ext: ModuleType, native: NativeFixture, routes: Path, log: Reporter) -> None:
    cases = torch.load(routes, weights_only=False, map_location="cpu")
    dev = torch.device("cuda:0")
    for stage in range(3):
        wg, wu, wd = native.expert_shard(stage, torch.bfloat16, dev)
        for rows in (5, 10):
            eligible = [c for c in cases if c["category"]=="live" and c["stage"]==stage and c["x"].shape[0]==rows]
            eligible.sort(key=lambda c: int(torch.unique(c["ids"]).numel()))
            if not eligible:
                raise RuntimeError("no real case for requested reuse histogram")
            for label, case in (("median_reuse", eligible[len(eligible)//2]), ("max_unique", eligible[-1])):
                x, ids, rw = (case[k].to(dev) for k in ("x", "ids", "weights"))
                unique = int(torch.unique(ids).numel())
                for mode in ("grouped", "small"):
                    m = ExpandedExperts(ext, wg, wu, wd, rows, mode=mode)
                    log.log({"test": "real_reuse", "stage": stage, "rows": rows, "mode": mode,
                             "case": label, "source_round": case["round"], "unique_experts": unique,
                             "native_stage_weights": stage, "precision": "BF16/FP32",
                             "output": "SLOT", "tp_collective_included": False,
                             **timed(lambda: m.run_slots(x, ids, rw))})
                    del m
        del wg, wu, wd, x, ids, rw
        gc.collect()
        torch.cuda.empty_cache()
    log.log({"test": "reuse_peak", "allocated": torch.cuda.max_memory_allocated()})


def dense(ext: ModuleType, native: NativeFixture, worker: Path, checkpoint: Path, log: Reporter) -> None:
    sys.path.insert(0, str(worker))
    from worker.checkpoint import Checkpoint
    from worker.model import Exl3Linear, facc_ext
    from worker import fused as FU
    from worker.tp2 import Exl3Mat, KernelPlan, col_shard, row_shard
    ck = Checkpoint(str(checkpoint))
    dev = torch.device("cuda:0")
    # Same native tensors and actual TP2 shard shapes, selected at setup.
    for name in ("mtp.0.main_proj", "mtp.0.attn.wq_a", "mtp.0.attn.wkv",
                 "mtp.0.attn.wq_b", "mtp.0.attn.wo_b", "mtp.0.attn.wo_a", "head"):
        raw = native.narrowed(name, torch.bfloat16)
        if name.endswith("wq_b"):
            raw = raw[:raw.shape[0]//2]
        if name.endswith("wo_b"):
            raw = raw[:, :raw.shape[1]//2]
        weights = raw.reshape(8, 1024, 4096)[:4] if name.endswith("wo_a") else raw.unsqueeze(0)
        widths = (64640, 64768, 64512) if name=="head" else (weights.shape[1],)
        for width in widths:
            w = weights[:, :width].contiguous().to(dev)
            groups, n, k = w.shape
            old = None
            if name.endswith("wo_a"):
                w8, _ = native.raw(name+".weight")
                sc, _ = native.raw(name+".scale")
                old_w8 = w8[:groups*1024].contiguous().to(dev)
                old_sc = torch.exp2(sc[:groups*32].float()-127).contiguous().to(dev)
            elif name.endswith("main_proj"):
                tr, sh, sv, bits = ck.exl3(name)
                old = Exl3Linear(tr, sh, sv, bits, dev, prescale=True, ext=facc_ext(), force_shape=0)
            else:
                tr, sh, sv, bits = ck.exl3(name)
                if name.endswith("wq_b"):
                    tr, sh, sv = col_shard(tr, sh, sv, 0, n)
                elif name.endswith("wo_b"):
                    tr, sh, sv = row_shard(tr, sh, sv, 0, k)
                elif name=="head":
                    if width%128:
                        # 64640 is 128-aligned, so all current tested widths are valid.
                        raise RuntimeError("K6 width is not Hadamard aligned")
                    tr, sh, sv = col_shard(tr, sh, sv, 0, width)
                old = Exl3Mat(tr, sh, sv, bits, dev, KernelPlan(), name=name)
            row_grid = (8, 32, 128) if name.endswith("main_proj") else ((1, 4, 20) if name=="head" else (5, 10, 15, 20))
            for rows in row_grid:
                torch.manual_seed(321)
                x = torch.randn((groups, rows, k), device=dev, dtype=torch.bfloat16)
                is_woa = name.endswith("wo_a")
                if is_woa:
                    x = x.transpose(0, 1).reshape(rows, groups*k).contiguous()
                    m = GroupedWoA(ext, w, rows, backend="cutlass")
                else:
                    m = Dense(ext, w, rows, backend="cutlass")
                # Dense trained FP8 projections require QDQ; BF16 head has no QDQ.
                qx = torch.empty_like(x)
                use_qdq = name!="head"
                def expanded() -> torch.Tensor:
                    if use_qdq:
                        ext.qdq(x, qx)
                    return m.run(qx if use_qdq else x)
                expanded()
                reference_x = qx if use_qdq else x
                if is_woa:
                    reference_x = reference_x.view(rows, groups, k).transpose(0, 1)
                ref = torch.bmm(reference_x.float(), w.float().transpose(1, 2))
                if is_woa:
                    ref = ref.transpose(0,1).reshape(rows,groups*n)
                error = float((m.output-ref).norm()/ref.norm())
                if error > 2e-5:
                    raise RuntimeError(f"dense native GEMM error {error} at {name}")
                result = timed(expanded)
                log.log({"test": "dense", "name": name, "shape": [groups, rows, n, k],
                         "native_dtype": "BF16", "native_qdq": use_qdq,
                         "relative_frobenius_fp32": error, "decoded_weight_bytes": w.numel()*2, **result})
                fast = (GroupedWoA(ext, w, rows, backend="cublas") if is_woa else
                        Dense(ext, w, rows, backend="cublas"))
                def expanded_fast() -> torch.Tensor:
                    if use_qdq:
                        ext.qdq(x, qx)
                    return fast.run(qx if use_qdq else x)
                expanded_fast()
                fast_error = float((fast.output-ref).norm()/ref.norm())
                if fast_error > 2e-5:
                    raise RuntimeError(f"cuBLAS FP32 native GEMM error {fast_error} at {name}")
                log.log({"test": "dense_cublas", "name": name, "shape": [groups, rows, n, k],
                         "native_dtype": "BF16", "native_qdq": use_qdq,
                         "relative_frobenius_fp32": fast_error, "fp32_accumulation": True,
                         **timed(expanded_fast)})
                del fast
                if name.endswith("wo_a"):
                    old_input = x
                    log.log({"test": "wo_a_current_fp8_control", "shape": [groups, rows, n, k],
                             "includes_native_act_qdq": False,
                             **timed(lambda: FU.wo_a_fp8(old_input, old_w8, old_sc, groups))})
                if old is not None:
                    old_x = x[0].contiguous()
                    old_out = torch.empty((rows, n), dtype=torch.float32, device=dev)
                    ctrl = timed(lambda: old(old_x, out=old_out))
                    log.log({"test": "exl3_control", "name": name, "bits": 6 if name=="head" else 5,
                             "shape": [rows, n, k], "same_input_before_native_qdq": True,
                             "semantic_identity_claimed": False, **ctrl})
                if name=="head":
                    ep = HeadEpilogue(ext, rows, n, dev)
                    bias = torch.randn((rows, n), device=dev)
                    log.log({"test": "head_epilogue", "shape": [rows, n],
                             "bias_source": "synthetic computed/cached row boundary",
                             **timed(lambda: ep.run(m.output[0], bias))})
                    del ep, bias
                del m, ref, x, qx
            del old, w
            gc.collect()
            torch.cuda.empty_cache()
    log.path.with_suffix(".native.json").write_text(json.dumps(native.records, indent=2)+"\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("experts", "dense", "reuse"))
    p.add_argument("donor", type=Path)
    p.add_argument("build", type=Path)
    p.add_argument("native", type=Path)
    p.add_argument("routes", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--worker", type=Path)
    p.add_argument("--checkpoint", type=Path)
    a = p.parse_args()
    require_lock()
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    ext = load_extension(a.donor, a.build)
    native = NativeFixture(a.native)
    log = Reporter(a.output)
    try:
        log.log({"test": "conditions", "phase": a.phase, "uuid": UUID, "torch": torch.__version__,
                 "cuda": torch.version.cuda, "arch": torch.cuda.get_device_capability(),
                 "tf32": False, "source_commit": "see build receipt source_sha256",
                 "boundary": "offline op/shard; excludes network, TP collective and engine"})
        if a.phase == "experts":
            experts(ext, native, a.routes, log)
        elif a.phase == "reuse":
            reuse(ext, native, a.routes, log)
        else:
            if a.worker is None or a.checkpoint is None:
                raise ValueError("dense phase requires --worker and --checkpoint")
            dense(ext, native, a.worker, a.checkpoint, log)
        log.log({"test": "complete", "peak_allocated": torch.cuda.max_memory_allocated(),
                 "peak_reserved": torch.cuda.max_memory_reserved()})
    finally:
        log.close()


if __name__ == "__main__":
    main()
