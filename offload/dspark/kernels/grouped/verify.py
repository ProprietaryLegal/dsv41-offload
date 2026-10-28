"""Focused GPU correctness/capture tests against independent CPU FP64 and torch FP32.

Run under run_locked.sh. Synthetic fixtures are not native checkpoint validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from api import Dense, ExpandedExperts, GroupedWoA, HeadEpilogue
from build import load_extension, require_lock


def qdq_ref(x: torch.Tensor) -> torch.Tensor:
    dtype, shape = x.dtype, x.shape
    v = x.float().reshape(-1, 32)
    scale = torch.exp2(torch.ceil(torch.log2(v.abs().amax(-1, keepdim=True).clamp_min(1e-4)/448)))
    return ((v/scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()*scale).reshape(shape).to(dtype)


def reference(x: torch.Tensor, ids: torch.Tensor, rw: torch.Tensor, gate: torch.Tensor,
              up: torch.Tensor, down: torch.Tensor, native: bool, round_down: bool) -> torch.Tensor:
    """CPU FP64 dots; no shared kernel code. FP32 boundaries are explicit."""
    x, ids, rw, gate, up, down = [t.cpu() for t in (x, ids, rw, gate, up, down)]
    if native:
        x = qdq_ref(x)
    y = torch.zeros(x.shape, dtype=torch.float32)
    for row in range(x.shape[0]):
        order = sorted(range(ids.shape[1]), key=lambda s: int(ids[row, s])) if native else range(ids.shape[1])
        for slot in order:
            e = int(ids[row, slot])
            g = (gate[e].double() @ x[row].double()).float()
            u = (up[e].double() @ x[row].double()).float()
            if native:
                g, u = g.bfloat16().float(), u.bfloat16().float()
            a = torch.nn.functional.silu(g.clamp(max=10))*u.clamp(-10, 10)
            if native:
                a = (a*rw[row, slot]).bfloat16().to(x.dtype)
                a = qdq_ref(a)
            else:
                a = a.to(x.dtype)
            d = (down[e].double() @ a.double()).float()
            if native and round_down:
                d = d.bfloat16().float()
            y[row] += d if native else d*rw[row, slot]
    return y


def run(donor: Path, build: Path, output: Path) -> None:
    require_lock()
    ext = load_extension(donor, build)
    torch.manual_seed(923)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_num_threads(8)
    dev = torch.device("cuda:0")
    records: list[dict[str, object]] = []
    def log(record: dict[str, object]) -> None:
        records.append(record)
        output.write_text(json.dumps(records, indent=2)+"\n")
        print(record, flush=True)
    # Complete FP8 midpoint neighbourhoods, zero, tiny and high exponents in each block.
    domain = torch.arange(0, 127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    domain = torch.cat([domain, -domain, (domain[:-1]+domain[1:])/2,
                        -(domain[:-1]+domain[1:])/2, torch.zeros(1)])
    for dtype in (torch.bfloat16, torch.float16):
        v = torch.cat([domain*2.0**e for e in (-12, -3, 0, 4) if dtype == torch.bfloat16 or e<=4])
        v = torch.cat([v, torch.randn(4096)*1e-6])
        v = torch.nn.functional.pad(v, (0, (-v.numel())%32)).to(dtype).reshape(-1, 32).to(dev)
        got = torch.empty_like(v)
        ext.qdq(v, got)
        ref = qdq_ref(v.cpu()).to(dev)
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
        bad = v.clone()
        bad.flatten()[0] = float("nan")
        bad.flatten()[32] = float("inf")
        ext.qdq(bad, got)
        assert bool(got.flatten()[0].isnan()) and bool(got.flatten()[32].isnan())
        log({"test": "qdq", "dtype": str(dtype), "elements": v.numel(), "bit_exact": True})
    # Integer-valued fixtures make all dots exactly representable FP32 and test scheduling,
    # empty experts, repeated experts, dynamic graph replay, and stable segment mappings.
    for dtype in (torch.bfloat16, torch.float16):
        for native in (False, True):
            for rows in (1, 4, 20):
                e, h, j = 8, 64, 32
                wg = torch.randint(-2, 3, (e, j, h), device=dev).to(dtype)/8
                wu = torch.randint(-2, 3, wg.shape, device=dev).to(dtype)/8
                wd = torch.randint(-2, 3, (e, h, j), device=dev).to(dtype)/8
                x = torch.randint(-4, 5, (rows, h), device=dev).to(dtype)/4
                ids = torch.stack([(torch.arange(rows, device=dev)+s)%e for s in (0, 2, 5)], -1).int()
                rw = torch.rand((rows, 3), device=dev)
                rw *= 1.5/rw.sum(-1, keepdim=True)
                for mode in ("grouped", "small"):
                    m = ExpandedExperts(ext, wg, wu, wd, rows, mode=mode, trained_qdq=native)
                    m.run(x, ids, rw)
                    got = m.combine_after_tp(ids).clone() if native else m.output.clone()
                    ref = reference(x, ids, rw, wg, wu, wd, native, native).to(dev)
                    torch.testing.assert_close(got, ref, rtol=2e-4, atol=2e-5)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        m.run(x, ids, rw)
                        if native:
                            m.combine_after_tp(ids)
                    graph.replay()
                    torch.testing.assert_close(m.output, got, rtol=0, atol=0)
                    # Changed routing must be read on device, including zero-count groups.
                    ids.copy_(torch.tensor([7, 1, 4], device=dev).expand_as(ids))
                    graph.replay()
                    ref = reference(x, ids, rw, wg, wu, wd, native, native).to(dev)
                    torch.testing.assert_close(m.output, ref, rtol=2e-4, atol=2e-5)
                    assert int(m.status.item()) == 0
                    slots = m.run_slots(x, ids, rw)
                    expected = m.state[15].index_select(0, m.state[2].long()).reshape_as(slots)
                    torch.testing.assert_close(slots, expected, rtol=0, atol=0)
                    slot_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(slot_graph):
                        m.run_slots(x, ids, rw)
                    slot_graph.replay()
                    torch.testing.assert_close(m.slot_output, expected, rtol=0, atol=0)
                    ids[0, 0] = -1
                    graph.replay()
                    assert int(m.status.item()) == 1 and bool(m.output.isnan().all())
                    ids[0, 0] = e
                    graph.replay()
                    assert int(m.status.item()) == 1 and bool(m.output.isnan().all())
                    ids[0, 0] = 7
                    graph.replay()
                    assert int(m.status.item()) == 0 and bool(m.output.isfinite().all())
                    saved = x[0, 0].clone()
                    x[0, 0] = float("nan")
                    graph.replay()
                    assert bool(m.output[0].isnan().all())
                    x[0, 0] = saved
                    graph.replay()
                    assert bool(m.output.isfinite().all())
                    log({"test": "experts", "dtype": str(dtype), "native_qdq": native,
                         "rows": rows, "mode": mode, "graph_dynamic_routing": True,
                         "invalid_id_nan": True, "max_abs_error": float((m.output-ref).abs().max()),
                         "workspace_bytes": m.workspace_bytes})
    # Dense groups include non-tile tails and a wide vocabulary-like tail.
    for dtype in (torch.bfloat16, torch.float16):
        for groups, rows, n, k in ((1, 1, 136, 64), (4, 20, 1024, 128), (1, 4, 64640, 32)):
            w = (torch.randint(-4, 5, (groups, n, k), device=dev)/8).to(dtype)
            x = (torch.randint(-4, 5, (groups, rows, k), device=dev)/8).to(dtype)
            m = Dense(ext, w, rows)
            m.run(x)
            ref = torch.bmm(x.double(), w.double().transpose(1, 2)).float()
            torch.testing.assert_close(m.output, ref, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                m.run(x)
            graph.replay()
            torch.testing.assert_close(m.output, ref, rtol=0, atol=0)
            log({"test": "dense", "dtype": str(dtype), "shape": [groups, rows, n, k], "bit_exact_fp64": True})
    for dtype in (torch.bfloat16, torch.float16):
        w = torch.randint(-4, 5, (4, 136, 64), device=dev).to(dtype)/8
        x = torch.randint(-4, 5, (20, 4*64), device=dev).to(dtype)/8
        m = GroupedWoA(ext, w, 20)
        m.run(x)
        ref = torch.bmm(x.view(20,4,64).transpose(0,1).double(),w.double().transpose(1,2))
        ref = ref.transpose(0,1).reshape(20,4*136).float()
        torch.testing.assert_close(m.output, ref, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            m.run(x)
        graph.replay()
        torch.testing.assert_close(m.output, ref, rtol=0, atol=0)
        log({"test": "wo_a_flat", "dtype": str(dtype), "bit_exact_fp64": True})
    for dtype in (torch.bfloat16, torch.float16):
        for flat in (False, True):
            w = torch.randint(-4, 5, (4, 136, 64), device=dev).to(dtype)/8
            x = torch.randint(-4, 5, (4, 20, 64), device=dev).to(dtype)/8
            ref = torch.bmm(x.double(), w.double().transpose(1,2)).float()
            if flat:
                x = x.transpose(0,1).reshape(20,4*64).contiguous()
                ref = ref.transpose(0,1).reshape(20,4*136)
                m = GroupedWoA(ext, w, 20, backend="cublas")
            else:
                m = Dense(ext, w, 20, backend="cublas")
            m.run(x)
            torch.testing.assert_close(m.output, ref, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                m.run(x)
            graph.replay()
            torch.testing.assert_close(m.output, ref, rtol=0, atol=0)
            log({"test": "cublas_fp32", "dtype": str(dtype), "flat": flat, "bit_exact_fp64": True})
    w = torch.randint(-4, 5, (1,512,5120), device=dev).bfloat16()/8
    x = torch.randint(-4, 5, (1,5,5120), device=dev).bfloat16()/8
    m = Dense(ext, w, 5)
    m.run(x)
    ref = torch.bmm(x.double(),w.double().transpose(1,2)).float()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        m.run(x)
    graph.replay()
    torch.testing.assert_close(m.output, ref, rtol=0, atol=0)
    log({"test": "auto_native_wkv", "backend": m.backend, "capture_exact_fp64": True})
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    try:
        Dense(ext, w, 5)
    except ValueError as exc:
        if "reduced_precision_reduction" not in str(exc):
            raise
    else:
        raise AssertionError("unsafe cuBLAS reduction flags were admitted")
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    log({"test": "unsafe_reduction_flags_refused", "passed": True})
    for vocab in (17, 64640, 64768, 64512):
        base = torch.randn((4, vocab), device=dev)*3
        bias = torch.randn_like(base)/4
        # Exact ties spanning blocks, plus a high-precision near tie rounded to BF16.
        base[:, 2] = 100
        base[:, -1] = 100.01
        bias[:, 2] = bias[:, -1] = 0
        m = HeadEpilogue(ext, 4, vocab, dev, offset=64768)
        vals, ids = m.run(base, bias)
        ref = base + bias
        rv, ri = ref.max(-1)
        torch.testing.assert_close(vals, rv, rtol=0, atol=0)
        torch.testing.assert_close(ids, ri+64768, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            m.run(base, bias)
        graph.replay()
        torch.testing.assert_close(m.winner, ri+64768, rtol=0, atol=0)
        base[0, 1] = float("nan")
        graph.replay()
        assert bool(m.value[0].isnan()) and int(m.winner[0]) == -1
        base[0, 1] = float("inf")
        graph.replay()
        assert bool(m.value[0].isnan()) and int(m.winner[0]) == -1
        log({"test": "head", "vocab": vocab, "bit_exact": True, "nan_propagates": True})
    log({"test": "complete", "checks": len(records), "cuda_peak_allocated": torch.cuda.max_memory_allocated()})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("donor", type=Path)
    p.add_argument("build", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    run(a.donor, a.build, a.output)


if __name__ == "__main__":
    main()
