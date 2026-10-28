"""Same-sample exact-path versus real TP2 worker evaluation through process proxy.

CPU host-flag stand-in drives the unchanged process proxy and TCP contract;
worker compute uses serving precision/graphs/Markov cache. No Spark endpoint.
All predictions stay in memory. Output contains aggregate comparisons only.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
import gc
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict, cast

import numpy as np
import torch
from torch import Tensor

from ...worker.server import DraftCore, TcpStubServer, build_adapted_evaluation_core
from ...spark.remote_dspark.config import RemoteConfig
from ...spark.remote_dspark.round import RemoteRound
from ...spark.remote_dspark.shmregion import RegionLayout, SharedRegion
from ...spark.remote_dspark.streamsync import CpuHostFlagSync, release_hand_off
from ...spark.remote_dspark import wire
from .data import D2Blocks, sha256
from .evaluate import match_length
from .model import Drafter, load
from .pilot import read_spec, write_json
from .pilot_eval import verify_sample_draw
from .pilot_sample import SPLITS, RoundKey, complete_rounds
from .replay import apply_adapter
from .worker_eval_lease import require_evaluation_lease


class Comparison(TypedDict):
    rounds: int
    exact_blocks: int
    position_matches: list[int]
    exact_path_EA: float
    worker_EA: float
    bonus_excluded: bool


class ProcessPredictor:
    def __init__(self, core: DraftCore, bind: str):
        # TcpStubServer applies the existing direct-link-only bind policy.
        self.server = TcpStubServer(core, bind, 0)
        hello = cast(Callable[[bytes], wire.Hello], wire.Hello.unpack)(core.expected.encode())
        cfg = RemoteConfig.from_env({"VLLM_DSPARK_REMOTE": f"tcp://{bind}:{self.server.port}",
            "VLLM_DSPARK_REMOTE_IDENTITY": hello.drafter_identity.hex(),
            "VLLM_DSPARK_REMOTE_MARKOV_IDENTITY": hello.markov_identity.hex(),
            "VLLM_DSPARK_REMOTE_POLICY": "capacity", "VLLM_DSPARK_REMOTE_ASYNC": "1",
            "VLLM_DSPARK_REMOTE_ASYNC_PROXY": "process", "VLLM_DSPARK_REMOTE_STRICT_START": "1",
            "VLLM_DSPARK_REMOTE_FUSED_GATHER": "0", "VLLM_DSPARK_REMOTE_TIMEOUT_MS": "30000",
            "VLLM_DSPARK_REMOTE_LOG_EVERY": "0", "VLLM_DSPARK_REMOTE_STATS_FILE": ""})
        layout = RegionLayout.for_hello(hello)
        self.region = cast(Callable[[RegionLayout], SharedRegion], SharedRegion.create)(layout)
        self.sync = CpuHostFlagSync(timeout_s=35., region=self.region)
        try:
            self.round = RemoteRound(cfg, hello, device=torch.device("cpu"), tp_rank=0,
                                     broadcast_fn=None, stream_sync=self.sync, background=False)
            if self.round.proxy is None:
                raise RuntimeError("real process proxy required; no thread/synchronous fallback")
        except BaseException:
            self.server.stop()
            release_hand_off(self.sync)
            raise
        self.hello = hello

    def __call__(self, inp: dict[str, Tensor], request: str) -> Tensor:
        if not bool(inp["has_taps"].all()) or not bool(inp["mask"].all()):
            raise ValueError("real-worker sample requires complete raw tap rows")
        positions = inp["positions"][0].cpu()
        rows, absolute = positions.numel(), int(inp["P"][0])+1
        if rows < 1 or int(positions[-1]) != absolute-1 or not torch.equal(positions, torch.arange(int(positions[0]), absolute)):
            raise ValueError("real-worker sample window is not contiguous/aligned")
        batch = SimpleNamespace(req_ids=[request], num_reqs=1,
            idx_mapping_np=np.array([0], dtype=np.int64), idx_mapping=torch.tensor([0], dtype=torch.int32),
            query_start_loc_np=np.array([0, rows], dtype=np.int64), positions=positions,
            is_prefilling_np=np.array([True]), num_computed_prefill_tokens_np=np.array([absolute-rows]),
            prefill_len_np=np.array([absolute]), num_draft_tokens_per_req=np.array([0], dtype=np.int32), num_tokens=rows)
        full_taps = inp["taps"][0].cpu()
        taps = [full_taps[:, i*self.hello.hidden:(i+1)*self.hello.hidden].contiguous() for i in range(self.hello.n_taps)]
        anchor = torch.zeros(self.hello.max_slots, 1, dtype=torch.int64)
        anchor[0, 0] = inp["anchor"][0]
        result = self.round.run(batch, taps, torch.tensor([1], dtype=torch.int32), torch.tensor([0], dtype=torch.int32),
                                anchor, torch.zeros(1, self.hello.max_slots, dtype=torch.int32))
        if result.read_status() != 0 or not bool(result.ok_mask[0]):
            raise RuntimeError("adapted real-worker process-proxy round failed; no fallback accepted")
        return result.drafts.detach().cpu().clone()

    def close(self) -> None:
        self.round.close()
        del self.round  # release all shared-region tensor/numpy views before unmap
        gc.collect()
        self.server.stop()
        error = release_hand_off(self.sync)
        if error:
            raise RuntimeError(f"process proxy cleanup failed: {error}")


@torch.no_grad()
def reference_predictions(model: Drafter, data: D2Blocks, k2: Path, keys: list[RoundKey]) -> list[Tensor]:
    selected = selected_blocks(data, k2, keys)
    result = []
    for index, _ in selected:
        block = data[index]
        inp = {k: t.to(model.activation_device) for k, t in block["inputs"].items()}
        h, hx = model.hidden_from(inp)
        result.append(model.greedy(h, hx, inp["anchor"])[0].cpu())
    return result


def selected_blocks(data: D2Blocks, k2: Path, keys: list[RoundKey]) -> list[tuple[int, RoundKey]]:
    wanted = {(k["request"], k["P"]) for k in keys}
    selected = [(i, k) for i, k in complete_rounds(data, k2) if (k["request"], k["P"]) in wanted]
    if not keys or len(wanted) != len(keys) or [k for _, k in selected] != keys:
        raise ValueError("worker evaluation requires the exact frozen pilot sample")
    return selected


def compare(data: D2Blocks, k2: Path, keys: list[RoundKey], reference: list[Tensor],
            predictor: Callable[[dict[str, Tensor], str], Tensor]) -> Comparison:
    selected = selected_blocks(data, k2, keys)
    if len(reference) != len(selected):
        raise ValueError("reference coverage differs from fixed worker sample")
    matches = torch.zeros(5, dtype=torch.int64)
    exact, reference_sum, worker_sum = 0, 0, 0
    for (index, key), ref in zip(selected, reference):
        block = data[index]
        if not bool(block["targets"]["position_mask"].all()):
            raise ValueError("worker sample has partial supervised targets")
        actual = predictor(block["inputs"], f"worker-eval/{key['request']}/{key['P']}")
        if actual.shape != (1, 5) or ref.shape != (1, 5) or actual.dtype not in (torch.int32, torch.int64):
            raise ValueError("worker/reference draft shape/dtype mismatch")
        if bool(((actual < 0) | (actual >= 129280)).any()):
            raise ValueError("worker produced out-of-vocabulary draft")
        same = (actual == ref).squeeze(0)
        matches += same.long()
        exact += int(same.all())
        truth = block["targets"]["labels"][0].tolist()
        a, b = match_length(ref[0].tolist(), truth), match_length(actual[0].tolist(), truth)
        if a is None or b is None:
            raise ValueError("indeterminate worker accepted prefix")
        reference_sum += a
        worker_sum += b
    return {"rounds": len(keys), "exact_blocks": exact, "position_matches": matches.tolist(),
            "exact_path_EA": reference_sum/len(keys), "worker_EA": worker_sum/len(keys), "bonus_excluded": True}


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    for name in ("plan", "manifest", "adapter", "k2-acap", "k2-legal", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--root", required=True, help="verified worker checkpoint tree")
    p.add_argument("--native-root", required=True)
    p.add_argument("--max-model-len", type=int, default=327680)
    p.add_argument("--bind", default="10.202.28.1")
    p.add_argument("--two-solos", action="store_true", help="requires BOTH individual nonblocking flocks")
    a = p.parse_args()
    require_evaluation_lease(not a.two_solos)
    started = time.perf_counter()
    spec = read_spec(a.plan, sampled=True)
    k2 = {"acap": a.k2_acap, "heldout_legal": a.k2_legal}
    data = verify_sample_draw(spec, a.manifest, k2)
    if not 1 <= a.max_model_len <= 524288 or any(k["P"]+7 > a.max_model_len for keys in spec["sample"]["keys"].values() for k in keys):
        raise ValueError("sample does not fit declared <=524288 total-context qualification profile")
    torch.set_num_threads(8)
    reference_started = time.perf_counter()
    ref = load(a.native_root, "cuda", reference_math=True)
    adapter_hash = apply_adapter(ref, str(a.adapter))
    predictions = {s: reference_predictions(ref, data[s], k2[s], spec["sample"]["keys"][s]) for s in SPLITS}
    reference_seconds = time.perf_counter()-reference_started
    reference_peak = torch.cuda.max_memory_reserved(0)
    # Worker and training model are never resident concurrently on a solo card.
    del ref
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    worker_started = time.perf_counter()
    core, worker = build_adapted_evaluation_core(a.root, str(a.adapter), a.native_root, emulated=not a.two_solos, max_model_len=a.max_model_len)
    if adapter_hash != sha256(a.adapter):
        raise ValueError("adapter changed between reference and serving load")
    worker_load_capture_seconds = time.perf_counter()-worker_started
    predictor = ProcessPredictor(core, a.bind)
    try:
        results = {s: compare(data[s], k2[s], spec["sample"]["keys"][s], predictions[s], predictor) for s in SPLITS}
        output = {"schema": "dsv41.d2.worker-evaluation.v1", "label": "MEASURED; isolated adapted native TP2",
                  "reference_seconds": reference_seconds, "reference_peak_reserved_bytes": reference_peak,
                  "worker_load_capture_seconds": worker_load_capture_seconds,
                  "elapsed_seconds": time.perf_counter()-started,
                  "leased_gpu_hours": (time.perf_counter()-started)*(2 if a.two_solos else 1)/3600,
                  "worker_peak_reserved_bytes": [torch.cuda.max_memory_reserved(i) for i in range(2 if a.two_solos else 1)],
                  "plan_sha256": sha256(a.plan), "sample_sha256": spec["sample"]["sha256"], "adapter_sha256": adapter_hash,
                  "identity": core.backend.arm_identity, "graphs": True, "process_proxy": True,
                  "host_handoff": "CPU stand-in; no Spark CUDA stream/gather qualification claimed",
                  "tp2_mode": "two-solo PCIe" if a.two_solos else "two ranks emulated on one solo", "splits": results}
        write_json(a.out, output)
    finally:
        predictor.close()
        del worker, core


if __name__ == "__main__":
    main()
