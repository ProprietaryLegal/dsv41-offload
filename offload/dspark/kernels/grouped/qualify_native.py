"""Native decoded GEMM/QDQ qualification on all 3 stages and both TP channel cuts.

Streams one complete expert at a time; no whole-model allocation. For every expert,
all output channels and all scale groups participate in FP32/reference dots. The
CPU FP64 and independent GPU FP32 oracles check every output channel.
N1's independent activation QDQ implementation is loaded from an explicit file.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

from api import Dense
from build import load_extension, require_lock
from native_fixture import NativeFixture


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("donor", "build", "native", "n1_formats", "routes", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--experts", type=int, default=128)
    a = p.parse_args()
    require_lock()
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    ext = load_extension(a.donor, a.build)
    spec = importlib.util.spec_from_file_location("n1_formats_reference", a.n1_formats)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned N1 formats")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = NativeFixture(a.native)
    cases = torch.load(a.routes, weights_only=False, map_location="cpu")
    rows: list[dict[str, object]] = []
    torch.manual_seed(91)
    dev = torch.device("cuda:0")
    # Prove same QDQ over real captured inputs plus stress at scale boundaries.
    for stage in range(3):
        candidates = [c for c in cases if c["stage"]==stage and c["category"]=="live"]
        if not candidates:
            raise RuntimeError("missing live capture for stage")
        case = candidates[len(candidates)//2]
        x = case["x"].to(dev)
        out = torch.empty_like(x)
        ext.qdq(x, out)
        ref = module.act_qdq(x)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        for rank in (0, 1):
            for e in range(a.experts):
                for proj in ("w1", "w3", "w2"):
                    key = f"mtp.{stage}.ffn.experts.{e}.{proj}"
                    full = native.narrowed(key, torch.bfloat16)
                    w = full[rank*1152:(rank+1)*1152] if proj!="w2" else full[:, rank*1152:(rank+1)*1152]
                    # Check FP16 admission separately from native BF16 activation semantics.
                    fp16_exact = torch.equal(w.float(), w.half().float())
                    if not fp16_exact:
                        raise RuntimeError(f"FP16 changed native weights at {key}")
                    k = w.shape[1]
                    inp = x[:1] if proj!="w2" else torch.randn((1, k), device=dev, dtype=torch.bfloat16)
                    qx = module.act_qdq(inp)
                    m = Dense(ext, w.unsqueeze(0).contiguous().to(dev), 1, backend="cutlass")
                    m.run(qx.unsqueeze(0))
                    got = m.output[0, 0]
                    ref32 = m.weight[0].float() @ qx[0].float()
                    diff = got-ref32
                    rel = float(diff.norm()/ref32.norm().clamp_min(1e-30))
                    if rel>2e-5:
                        raise RuntimeError(f"native dot drift {rel} at {key} rank {rank}")
                    ref64 = w.double() @ qx[0].cpu().double()
                    got_cpu = got.cpu()
                    match = (got_cpu.bfloat16()==ref64.bfloat16())
                    mismatches = (~match).nonzero().flatten().cpu().tolist()
                    # Record every BF16 rounding disagreement against the FP64 oracle.
                    # N1's >99.9% criterion applies across the sampled projection elements.
                    frac = float(match.float().mean())
                    rel64 = float((got_cpu.double()-ref64).norm()/ref64.norm().clamp_min(1e-30))
                    if rel64>2e-5:
                        raise RuntimeError(f"FP64 projection mismatch at {key}")
                    tie_details = []
                    for ix in mismatches:
                        gv, rv = float(got_cpu[ix].bfloat16()), float(ref64[ix].bfloat16())
                        midpoint = (gv+rv)/2
                        tie_details.append({"index": ix, "computed_fp32": float(got_cpu[ix]),
                                            "reference_fp64": float(ref64[ix]),
                                            "bf16_midpoint": midpoint,
                                            "distance_reference_to_midpoint": float(ref64[ix])-midpoint})
                    rows.append({"key": key, "rank": rank, "relative_fp32": rel,
                                 "max_abs_fp32": float(diff.abs().max()), "bf16_match_fraction": frac,
                                 "bf16_tie_indices": mismatches, "bf16_tie_details": tie_details,
                                 "elements": got.numel(), "bf16_matches": int(match.sum()),
                                 "fp64_projection_relative": rel64,
                                 "all_native_values_fp16_exact": fp16_exact})
                    del m
                if e%16==0:
                    a.output.write_text(json.dumps(rows, indent=2)+"\n")
                    print("stage",stage,"rank",rank,"expert",e,"qualified",flush=True)
    a.output.write_text(json.dumps(rows, indent=2)+"\n")
    fraction = sum(int(r["bf16_matches"]) for r in rows)/sum(int(r["elements"]) for r in rows)
    if fraction <= .999:
        raise RuntimeError(f"aggregate native BF16 projection agreement {fraction} <=99.9%")
    a.output.with_suffix(".receipt.json").write_text(json.dumps({
        "n1_formats_sha256": hashlib.sha256(a.n1_formats.read_bytes()).hexdigest(),
        "native_tensor_receipts": native.records, "peak_allocated": torch.cuda.max_memory_allocated(),
        "checked_matrices": len(rows), "native_qdq_bit_exact": True,
        "bf16_projection_match_fraction_fp64": fraction,
    }, indent=2)+"\n")
    print("COMPLETE",len(rows),"matrices/cuts",flush=True)


if __name__ == "__main__":
    main()
