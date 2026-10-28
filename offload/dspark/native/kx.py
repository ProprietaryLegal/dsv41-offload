"""KX lanes' kernels behind N1's ExpertBank interface (lane branches offload/kx-mxfp4, offload/kx-grouped; imported
from offload/dspark/kernels/** when that tree is present, refused loudly when not).

KXRegisterBank: KX-MXFP4's register-fed SM86 W4A16 kernel (regfed.RegisterExperts, fused gate/up, device expert
schedule) for the projections; N1's swiglu_qdq for the official epilogue between them (full-precision expf, bf16
rounding points, per-32 QDQ); per-slot fp32 down outputs (SLOT mode), so the official combine stays N1's.
The kernel's sticky status word (bit 0: an invalid expert id, NaN outputs) is zeroed at build and checked by check().
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import ops as NO

if TYPE_CHECKING:
    from ..kernels.fp8dense import DenseFP8


class KXRegisterBank:
    def __init__(self, bank: NO.ExpertBank, dev: torch.device, max_rows: int = 21):
        """``bank``: the rank's packed shards on the HOST (KX admission validates on the CPU before any GPU
        allocation); RegisterExperts streams them one expert at a time into its fragment layout on ``dev``."""
        try:
            from ..kernels.mxfp4.regfed import RegisterExperts
        except ImportError as e:
            raise RuntimeError("KX-MXFP4 kernels (offload/dspark/kernels/mxfp4, branch offload/kx-mxfp4) not "
                               "present in this tree") from e
        if bank.kind != "packed" or bank.dev.type != "cpu":
            raise ValueError("KXRegisterBank takes a packed bank on the host")
        self.name, self.kind, self.E, self.H, self.I, self.dev = bank.name + "/kx", "packed", bank.E, bank.H, bank.I, dev
        with torch.cuda.device(dev):
            self.gate = RegisterExperts.from_checkpoint(bank.w13[:, 0], bank.s13[:, 0], dev)
            self.up = RegisterExperts.from_checkpoint(bank.w13[:, 1], bank.s13[:, 1], dev)
            bank.w13 = bank.s13 = None
            self.down = RegisterExperts.from_checkpoint(bank.w2, bank.s2, dev)
            bank.w2 = bank.s2 = None
            self.status = torch.zeros(1, dtype=torch.int32, device=dev)
            self.schedule = torch.empty(max_rows * 3, 8, dtype=torch.int32, device=dev)
        self.max_rows = max_rows

    def tensors(self) -> list[torch.Tensor]:
        return [t for b in (self.gate, self.up, self.down) for t in (b.codes, b.scales)]

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors())

    def forward_slots(self, xq: torch.Tensor, ids: torch.Tensor, rw: torch.Tensor, h: torch.Tensor | None = None,
                      out: torch.Tensor | None = None) -> torch.Tensor:
        T = xq.shape[0]
        if T > self.max_rows:
            raise ValueError(f"{self.name}: {T} rows > {self.max_rows} (the kernel admits 64 routed slots)")
        ids32 = ids.to(torch.int32).contiguous()
        sched = self.schedule[:T * 3]
        g = torch.empty(T, 3, self.I, dtype=torch.float32, device=xq.device)
        u = torch.empty_like(g)
        self.gate.linear_into(xq.contiguous(), ids32, g, self.status, sched, up=self.up, up_out=u)
        hq = NO.swiglu_qdq(g, u, rw)
        y = torch.empty(T, 3, self.H, dtype=torch.float32, device=xq.device)
        self.down.linear_into(hq, ids32, y, self.status, sched, per_slot=True)
        if out is not None:
            out.copy_(y)
            return out
        return y

    def check(self) -> None:
        s = int(self.status.item())
        if s:
            raise RuntimeError(f"{self.name}: KX kernel status {s:#x} (invalid expert ids: NaN outputs)")


class KXFusedBank(KXRegisterBank):
    """KX delivery selector with caller-owned, pre-capture SLOT workspaces.

    A bank belongs to one rank/capture stream, like the parent drafter. Different
    simultaneous streams must use separate banks and workspaces.
    ``variant_rows``: select every row count's kernel variant as for that many rows
    (batch-invariant mode: the B1 block); None keeps the measured per-row profile.
    """

    def __init__(self, bank: NO.ExpertBank, dev: torch.device, stage: int, variant_rows: int | None = None):
        from ..kernels.fusedmoe import FusedMoE, Workspace, select_variant

        super().__init__(bank, dev, max_rows=20)
        self.pipelines = {
            rows: FusedMoE.from_register_banks(self.gate, self.up, self.down,
                                              variant=select_variant(stage, rows if variant_rows is None
                                                                     else variant_rows))
            for rows in range(1, 21)
        }
        self.workspaces = {rows: Workspace.allocate(rows, self.I, dev) for rows in self.pipelines}
        self.slot_buffers = {rows: torch.empty(rows, 4, self.H, dtype=torch.float32, device=dev)
                             for rows in self.pipelines}
        self.outputs = {rows: buf[:, :3] for rows, buf in self.slot_buffers.items()}
        self.combined_outputs = {rows: torch.empty(rows, self.H, dtype=torch.bfloat16, device=dev)
                                 for rows in self.pipelines}

    def forward_slots(self, xq: torch.Tensor, ids: torch.Tensor, rw: torch.Tensor,
                      h: torch.Tensor | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
        rows = xq.shape[0]
        if rows not in self.pipelines:
            raise ValueError(f"{self.name}: fused pipeline admits 1..20 rows, got {rows}")
        result = self.outputs[rows] if out is None else out
        self.pipelines[rows].forward_into(xq, ids, rw, result, self.status, self.workspaces[rows])
        return result


class KXSelectiveDense:
    """Keep BF16 as baseline; use measured FP8 only at explicitly selected M.

    Selected FP8 calls allocate no tensor storage and make no GPU-dependent choice.
    The legacy BF16 fallback retains its original graph-captured allocation policy.
    """

    def __init__(self, baseline: NO.NativeDense, candidate: DenseFP8, rows: tuple[int, ...],
                 splits: int | None = None):
        """``splits``: one split-K count for every selected M (batch-invariant mode: B1's); None keeps the
        measured per-M count. Row tiles stay per M: they partition rows, not the K reduction."""
        from ..kernels.fp8dense import DenseFP8

        if not isinstance(candidate, DenseFP8):
            raise TypeError("candidate must be DenseFP8")
        self.baseline, self.candidate = baseline, candidate
        self.name, self.kind = baseline.name, baseline.kind
        self.w, self.sc = baseline.w, baseline.sc
        self.in_features = candidate.groups * candidate.inputs
        self.out_features = candidate.groups * candidate.outputs
        self.configs = {m: candidate.configuration(m) for m in rows}
        if splits is not None:
            self.configs = {m: (splits, tiles) for m, (_, tiles) in self.configs.items()}
        self.scratch = {
            m: (torch.empty(splits, m, candidate.groups, candidate.outputs,
                            dtype=torch.float32, device=self.w.device) if splits > 1 else None)
            for m, (splits, _) in self.configs.items()
        }
        self.outputs = {(m, f32): torch.empty(m, self.out_features,
                                            dtype=torch.float32 if f32 else torch.bfloat16, device=self.w.device)
                        for m in rows for f32 in (False, True)}

    def nbytes(self) -> int:
        return self.baseline.nbytes() + sum(t.numel() * t.element_size()
                                            for t in (self.candidate.codes, self.candidate.scales))

    def __call__(self, xq: torch.Tensor, out_f32: bool = False,
                 out: torch.Tensor | None = None) -> torch.Tensor:
        rows = xq.shape[0]
        if out is not None and out.dtype != (torch.float32 if out_f32 else torch.bfloat16):
            raise ValueError(f"{self.name}: caller output dtype disagrees with out_f32")
        if rows not in self.configs:
            if self.candidate.groups != 1:
                if out_f32 or out is not None:
                    raise ValueError("grouped wo_a baseline returns BF16 without caller output")
                groups, inputs, outputs = self.candidate.groups, self.candidate.inputs, self.candidate.outputs
                z = torch.bmm(xq.view(rows, groups, inputs).transpose(0, 1),
                              self.w.view(groups, outputs, inputs).transpose(1, 2))
                return z.transpose(0, 1).reshape(rows, -1)
            return self.baseline(xq, out_f32=out_f32, out=out)
        result = self.outputs[rows, out_f32] if out is None else out
        splits, tiles = self.configs[rows]
        self.candidate.linear_into(xq, result, scratch=self.scratch[rows], splits=splits, row_tiles=tiles)
        return result


class KXSharedPair:
    """Measured T20 BF16 shared gate/up pair; smaller shapes retain FP8 calls."""

    def __init__(self, gate: KXSelectiveDense, up: KXSelectiveDense):
        if gate.kind != "bf16" or up.kind != "bf16" or gate.w.shape != up.w.shape or gate.w.device != up.w.device:
            raise ValueError("shared pair requires matching BF16 baseline matrices")
        self.gate, self.up = gate, up
        self.weight = torch.cat((gate.w, up.w), dim=0)
        self.output = torch.empty(20, self.weight.shape[0], dtype=torch.bfloat16, device=self.weight.device)
        self.projections = (self.output[:, :gate.out_features], self.output[:, gate.out_features:])

    def nbytes(self) -> int:
        return self.weight.numel() * self.weight.element_size()

    def __call__(self, xq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if xq.shape[0] != 20:
            return self.gate(xq), self.up(xq)
        if torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
            raise RuntimeError("paired shared BF16 reduced-precision reduction is enabled")
        torch.mm(xq, self.weight.t(), out=self.output)
        return self.projections
