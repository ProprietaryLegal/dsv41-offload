"""FIX1 GPU regression on native head shards and V1 captured head_hidden.

Run under run_locked.sh. Inputs/results are proprietary: keep on X86host/Controller,
never commit captures, logits, previous tokens or model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import ModuleType

import torch
from safetensors import safe_open

from api import Dense, HeadEpilogue
from build import UUID, load_extension, require_lock


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(ext: ModuleType, base: torch.Tensor, bias: torch.Tensor, offset: int) -> None:
    """Check both explicit policies, including graph replay."""
    ep = HeadEpilogue(ext, base.shape[0], base.shape[1], base.device, offset)
    for engine in (False, True):
        run = ep.run_engine_bf16_control if engine else ep.run
        run(base, bias)  # warmup before capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(base, bias)
        for replay in (False, True):
            if replay:
                graph.replay()
            values, ids = ((base.bfloat16() + bias.bfloat16()).float() if engine else base + bias).max(-1)
            torch.testing.assert_close(ep.value, values, rtol=0, atol=0)
            torch.testing.assert_close(ep.winner, ids + offset, rtol=0, atol=0)


def run(donor: Path, build: Path, checkpoint: Path, captures: list[Path], output: Path) -> None:
    require_lock()
    started = time.time()
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    ext = load_extension(donor, build)
    index_path = checkpoint / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]

    def read(name: str, start: int, end: int) -> torch.Tensor:
        with safe_open(str(checkpoint / weight_map[name]), framework="pt", device="cpu") as f:
            return f.get_slice(name)[start:end].contiguous()

    # The dump's own prefix is [anchor] + drafts[:-1], the exact preceding tokens
    # for these head rows. No synthetic previous-token substitution.
    hidden_rows, previous_rows = [], []
    for path in captures:
        capture = torch.load(path, map_location="cpu", weights_only=True)
        hidden = capture["head_hidden"].reshape(-1, 5120).float()
        prev = torch.tensor(capture["variants"]["own"]["prev"], dtype=torch.long)
        if len(prev) != len(hidden) or not torch.isfinite(hidden).all():
            raise ValueError(f"invalid head_hidden/prefix in {path}")
        hidden_rows.append(hidden)
        previous_rows.append(prev)
    hidden = torch.cat(hidden_rows).cuda()
    previous = torch.cat(previous_rows)
    embed = read("mtp.2.markov_head.embed.weight", 0, 129280)[previous].float().cuda()
    records: list[dict[str, int | str | bool]] = []
    for offset, width in ((0, 64640), (64640, 64640), (0, 64768), (64768, 64512)):
        head = read("head.weight", offset, offset + width).cuda()
        markov = read("mtp.2.markov_head.head.weight", offset, offset + width).float().cuda()
        bias = torch.nn.functional.linear(embed, markov).contiguous()
        if bias.dtype != torch.float32 or not torch.isfinite(bias).all():
            raise AssertionError("Markov row must be finite FP32")
        # The required native head oracle keeps captured FP32 hidden unchanged
        # and widens the released BF16 weight, as official ParallelHead does.
        # Also exercise Dense with its explicit BF16-input boundary as an extra
        # component check; this is not an assertion of official GEMM equality.
        dense = Dense(ext, head.unsqueeze(0), len(hidden), backend="cutlass")
        native_base = dense.run(hidden.bfloat16().unsqueeze(0).contiguous())[0]
        check(ext, native_base, bias, offset)
        official_base = torch.nn.functional.linear(hidden, head.float()).contiguous()
        check(ext, official_base, bias, offset)
        records.append({"offset": offset, "vocab": width, "rows": len(hidden),
                        "native_dense_fp32_epilogue_exact": True, "official_linear_fp32_epilogue_exact": True,
                        "engine_control_exact": True, "graph_replay_exact": True})
        del dense, native_base, official_base, head, markov, bias

    # Across chunk boundaries and at each shard tail: BF16 collapses a genuine
    # FP32 difference, while the equal-value row must select the lowest ID.
    for width in (64640, 64768, 64512):
        base = torch.zeros((2, width), device="cuda", dtype=torch.float32)
        bias = torch.zeros_like(base)
        base[:, 2] = base[:, -1] = 100.0
        bias[0, -1] = 0.01
        check(ext, base, bias, 64768)
        ep = HeadEpilogue(ext, 2, width, base.device, 64768)
        value, winner = ep.run(base, bias)
        torch.testing.assert_close(value, (base + bias).max(-1).values, rtol=0, atol=0)
        assert winner.tolist() == [64768 + width - 1, 64770]
        value, winner = ep.run_engine_bf16_control(base, bias)
        assert winner.tolist() == [64770, 64770]
        assert value.tolist() == [100.0, 100.0]
        records.append({"vocab": width, "constructed_policy_disagreement": True,
                        "lowest_id_exact_tie": True})
    torch.cuda.synchronize()
    result = {"passed": True, "gpu_uuid": UUID, "torch": torch.__version__,
              "cuda": torch.version.cuda, "tf32": False, "bf16_reduced_precision_reduction": False,
              "capture_sources": [{"path": str(p), "sha256": digest(p)} for p in captures],
              "checkpoint": str(checkpoint), "index_sha256": digest(index_path),
              "build_receipt": json.loads((build / "receipt.json").read_text()),
              "records": records, "seconds": time.time() - started,
              "cuda_peak_allocated": torch.cuda.max_memory_allocated()}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("donor", type=Path)
    p.add_argument("build", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("captures", type=Path, nargs="+")
    a = p.parse_args()
    run(a.donor, a.build, a.checkpoint, a.captures, a.output)


if __name__ == "__main__":
    main()
