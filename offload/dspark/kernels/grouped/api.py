"""Preallocated native-expanded components; setup is separate from graph-safe execution."""
from __future__ import annotations

from types import ModuleType
from typing import Literal

import torch

Mode = Literal["grouped", "small", "auto"]
DenseBackend = Literal["cutlass", "cublas", "auto"]


def dense_backend(groups: int, rows: int, n: int, k: int, dtype: torch.dtype) -> Literal["cutlass", "cublas"]:
    """Frozen measured dispatch, no startup tuning; unmeasured shapes use CUTLASS."""
    if dtype == torch.bfloat16:
        if groups == 1 and (n, k) == (5120, 15360) and rows in (8, 32, 128):
            return "cublas"
        if rows in (5, 10, 15, 20):
            if groups == 1 and (n, k) in ((1280, 5120), (512, 5120), (5120, 4096)):
                return "cublas"
            if groups == 4 and (n, k) == (1024, 4096):
                return "cublas"
    return "cutlass"


def _weight(w: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> None:
    if tuple(w.shape) != shape or w.dtype != dtype or w.device != device or not w.is_contiguous():
        raise ValueError(f"expected contiguous {shape} {dtype} on {device}; got {w.shape} {w.dtype} {w.device}")


def _input(x: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> None:
    _weight(x, shape, dtype, device)


class ExpandedExperts:
    """A single TP channel-shard partial. Output/status/scratch are overwritten on next call.

    Inputs/weights must remain alive until all captured graphs using this object are destroyed.
    One instance per in-flight stream. Router weights are never narrowed to FP16.
    """

    def __init__(self, ext: ModuleType, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor,
                 rows: int, top_k: int = 3, limit: float = 10.0, mode: Mode = "auto", trained_qdq: bool = True, input_is_qdq: bool = False) -> None:
        if gate.ndim != 3 or gate.dtype not in (torch.float16, torch.bfloat16) or not gate.is_cuda:
            raise ValueError("gate must be CUDA BF16/FP16 [E,J,H]")
        e, j, h = gate.shape
        if not 1 <= e <= 128 or min(h, j) <= 0 or h % 8 or j % 8 or rows <= 0 or top_k <= 0:
            raise ValueError("E must be 1..128; positive H/J must be multiples of 8; rows/top_k positive")
        if mode == "auto":
            mode = "small" if (h, j, top_k) == (5120, 1152, 3) and rows <= 3 else "grouped"
        if mode not in ("grouped", "small") or not 0 < limit < float("inf"):
            raise ValueError("unknown mode or invalid SwiGLU limit")
        self.ext, self.dtype, self.device = ext, gate.dtype, gate.device
        _weight(gate, (e, j, h), self.dtype, self.device)
        _weight(up, (e, j, h), self.dtype, self.device)
        _weight(down, (e, h, j), self.dtype, self.device)
        self.gate, self.up, self.down = gate, up, down
        self.rows, self.top_k, self.limit, self.mode = rows, top_k, limit, mode
        if trained_qdq and (top_k != 3 or h % 32 or j % 32):
            raise ValueError("native QDQ requires top-3 and 32-aligned H/J")
        self.trained_qdq, self.input_is_qdq = trained_qdq, input_is_qdq
        self.h, self.j = h, j
        r, g = rows * top_k, min(e, rows * top_k)
        def new(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(shape, dtype=dtype, device=self.device)
        self.state = [new((r, h), self.dtype), new((r,), torch.int32), new((r,), torch.int32),
                      new((e,), torch.int32), new((e,), torch.int32), new((1,), torch.int32),
                      new((2*g, 3), torch.int32), new((g, 3), torch.int32),
                      new((4, 2*g), torch.int64), new((4, g), torch.int64),
                      new((4, 2*g), torch.int64), new((4, g), torch.int64),
                      new((r, j), torch.float32), new((r, j), torch.float32),
                      new((r, j), self.dtype), new((r, h), torch.float32)]
        self.qx = new((rows, h), self.dtype)
        self.qa = new((r, j), self.dtype)
        self.output = new((rows, h), torch.float32)
        self.slot_output = new((rows, top_k, h), torch.float32)
        self.status = self.state[5]
        self.nblocks = ext.blocks(gate)

    def project(self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        _input(x, (self.rows, self.h), self.dtype, self.device)
        _input(ids, (self.rows, self.top_k), torch.int32, self.device)
        _input(weights, (self.rows, self.top_k), torch.float32, self.device)
        s = self.state
        if self.trained_qdq and not self.input_is_qdq:
            self.ext.qdq(x, self.qx)
            x = self.qx
        if self.mode == "grouped":
            self.ext.prepare(x, ids, self.gate, self.up, self.down, s)
            self.ext.gemm(x, s[6], s[8], s[10], self.nblocks)
            self._activate(weights)
            self.ext.gemm(x, s[7], s[9], s[11], self.nblocks)
        else:
            self.ext.small_prepare(ids, s[2], s[1], s[5], self.gate.shape[0])
            self.ext.small(x, self.gate, self.up, ids, s[12], s[13], True)
            self._activate(weights)
            self.ext.small(s[14], self.down, self.down, ids, s[15], s[15], False)
        return s[15]

    def run(self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        self.project(x, ids, weights)
        s = self.state
        if self.trained_qdq:
            self.ext.native_combine(s[15], ids, s[2], s[5], self.output, False)
        else:
            self.ext.combine(s[15], s[2], weights, s[5], self.output)
        return self.output

    def run_slots(self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """FP32 [T,3,H] in original route-slot order, ready for N1's exact TP2 epilogue."""
        self.project(x, ids, weights)
        self.ext.slot_scatter(self.state[15], self.state[2], self.status, self.slot_output)
        return self.slot_output

    def _activate(self, weights: torch.Tensor) -> None:
        s = self.state
        if self.trained_qdq:
            self.ext.native_activate(s[12], s[13], s[1], weights, self.qa, self.limit)
            self.ext.qdq(self.qa, s[14])
        else:
            self.ext.activate(s[12], s[13], s[14], self.limit)

    def combine_after_tp(self, ids: torch.Tensor) -> torch.Tensor:
        """Call after the integrator sums BOTH ranks' per-route state[15] FP32 partials.

        This is the native BF16 down-output boundary, before ascending-expert FP32 sum.
        run() alone returns an unrounded shard partial for diagnostics.
        """
        if not self.trained_qdq:
            raise RuntimeError("combine_after_tp requires the native arithmetic profile")
        _input(ids, (self.rows, self.top_k), torch.int32, self.device)
        self.ext.native_combine(self.state[15], ids, self.state[2], self.status, self.output, True)
        return self.output

    @property
    def workspace_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (*self.state, self.qx, self.qa, self.output, self.slot_output))


class Dense:
    """Uniform grouped GEMM, [G,M,K] @ [G,N,K]^T -> FP32 [G,M,N]."""

    def __init__(self, ext: ModuleType, weight: torch.Tensor, rows: int,
                 backend: DenseBackend = "auto") -> None:
        if weight.ndim != 3 or not weight.is_cuda or weight.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("weight must be CUDA BF16/FP16 [groups,N,K]")
        g, n, k = weight.shape
        if min(g, n, k, rows) <= 0 or n % 8 or k % 8:
            raise ValueError("positive dimensions and 8-aligned K/N required")
        _weight(weight, (g, n, k), weight.dtype, weight.device)
        if backend == "auto":
            backend = dense_backend(g, rows, n, k, weight.dtype)
        if backend not in ("cutlass", "cublas"):
            raise ValueError("unknown dense backend")
        if backend == "cublas":
            reduced = (torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                       if weight.dtype == torch.bfloat16 else
                       torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
            if reduced:
                raise ValueError("cuBLAS FP32 profile requires reduced_precision_reduction=False at setup")
        self.backend = backend
        self.ext, self.weight, self.rows = ext, weight, rows
        self.input = torch.empty((g, rows, k), dtype=weight.dtype, device=weight.device)
        self.output = torch.empty((g, rows, n), dtype=torch.float32, device=weight.device)
        self.sizes = torch.empty((g, 3), dtype=torch.int32, device=weight.device)
        self.ptr = torch.empty((4, g), dtype=torch.int64, device=weight.device)
        self.ld = torch.empty_like(self.ptr)
        self.nblocks = ext.blocks(weight)
        ext.dense_prepare(self.input, weight, self.output, self.sizes, self.ptr, self.ld, False)
        self.wt = weight.transpose(1, 2)
        self.a0, self.b0, self.c0 = self.input[0], self.wt[0], self.output[0]

    def run(self, x: torch.Tensor) -> torch.Tensor:
        _input(x, tuple(self.input.shape), self.weight.dtype, self.weight.device)
        self.input.copy_(x)
        if self.backend == "cublas":
            if self.input.shape[0] == 1:
                torch.mm(self.a0, self.b0, out_dtype=torch.float32, out=self.c0)
            else:
                torch.bmm(self.input, self.wt, out_dtype=torch.float32, out=self.output)
        else:
            self.ext.gemm(self.input, self.sizes, self.ptr, self.ld, self.nblocks)
        return self.output


class GroupedWoA:
    """Grouped attention output projection in the model's flat [T,G*C] layout.

    Avoids the transpose/contiguous passes needed by grouped bmm. Weights [G,N,K]
    are output-major, output [T,G*N] is FP32, with native BF16 rounding caller-owned.
    """

    def __init__(self, ext: ModuleType, weight: torch.Tensor, rows: int,
                 backend: DenseBackend = "auto") -> None:
        if weight.ndim != 3 or not weight.is_cuda or weight.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("weight must be CUDA BF16/FP16 [groups,N,K]")
        g, n, k = weight.shape
        if min(g, n, k, rows)<=0 or n%8 or k%8:
            raise ValueError("positive dimensions and 8-aligned K/N required")
        _weight(weight, (g, n, k), weight.dtype, weight.device)
        if backend == "auto":
            backend = dense_backend(g, rows, n, k, weight.dtype)
        if backend not in ("cutlass", "cublas"):
            raise ValueError("unknown dense backend")
        if backend == "cublas":
            reduced = (torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                       if weight.dtype == torch.bfloat16 else
                       torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
            if reduced:
                raise ValueError("cuBLAS FP32 profile requires reduced_precision_reduction=False at setup")
        self.backend = backend
        self.ext, self.weight, self.rows = ext, weight, rows
        self.input = torch.empty((rows, g*k), dtype=weight.dtype, device=weight.device)
        self.output = torch.empty((rows, g*n), dtype=torch.float32, device=weight.device)
        self.sizes = torch.empty((g, 3), dtype=torch.int32, device=weight.device)
        self.ptr = torch.empty((4, g), dtype=torch.int64, device=weight.device)
        self.ld = torch.empty_like(self.ptr)
        self.nblocks = ext.blocks(weight)
        ext.dense_prepare(self.input, weight, self.output, self.sizes, self.ptr, self.ld, True)
        self.av = self.input.view(rows, g, k).transpose(0, 1)
        self.cv = self.output.view(rows, g, n).transpose(0, 1)
        self.wt = weight.transpose(1, 2)

    def run(self, x: torch.Tensor) -> torch.Tensor:
        _input(x, tuple(self.input.shape), self.weight.dtype, self.weight.device)
        self.input.copy_(x)
        if self.backend == "cublas":
            torch.bmm(self.av, self.wt, out_dtype=torch.float32, out=self.cv)
        else:
            self.ext.gemm(self.input, self.sizes, self.ptr, self.ld, self.nblocks)
        return self.output


class HeadEpilogue:
    """Local vocab winner for an existing TP2 winner collective. Does not implement confidence.

    Native run() adds FP32 base and exact FP32 Markov bias without BF16 rounding.
    Equal logits select the lowest global ID. BF16 engine arithmetic is an explicit
    run_engine_bf16_control() operation only. No bias is dropped.
    Nonfinite candidates propagate a NaN winner value, requiring caller failure.
    """

    def __init__(self, ext: ModuleType, rows: int, vocab: int, device: torch.device, offset: int = 0) -> None:
        if rows <= 0 or vocab <= 0 or offset < 0:
            raise ValueError("positive rows/vocab and nonnegative offset required")
        self.ext, self.rows, self.vocab, self.device, self.offset = ext, rows, vocab, device, offset
        chunks = (vocab+1023)//1024
        self.values = torch.empty((rows, chunks), dtype=torch.float32, device=device)
        self.ids = torch.empty((rows, chunks), dtype=torch.int64, device=device)
        self.winner = torch.empty((rows,), dtype=torch.int64, device=device)
        self.value = torch.empty((rows,), dtype=torch.float32, device=device)

    def run(self, base: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._run(base, bias, engine_bf16_control=False)

    def run_engine_bf16_control(self, base: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Engine comparison only: BF16(base), BF16(bias), BF16(sum), then argmax."""
        return self._run(base, bias, engine_bf16_control=True)

    def _run(self, base: torch.Tensor, bias: torch.Tensor, *, engine_bf16_control: bool) -> tuple[torch.Tensor, torch.Tensor]:
        for t in (base, bias):
            _input(t, (self.rows, self.vocab), torch.float32, self.device)
        head = self.ext.head_engine_bf16_control if engine_bf16_control else self.ext.head
        head(base, bias, self.values, self.ids, self.offset)
        self.ext.head_reduce(self.values, self.ids, self.value, self.winner)
        return self.value, self.winner
