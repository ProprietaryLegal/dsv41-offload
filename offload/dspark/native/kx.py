"""KX lanes' kernels behind N1's ExpertBank interface (lane branches offload/kx-mxfp4, offload/kx-grouped; imported
from offload/dspark/kernels/** when that tree is present, refused loudly when not).

KXRegisterBank: KX-MXFP4's register-fed SM86 W4A16 kernel (regfed.RegisterExperts, fused gate/up, device expert
schedule) for the projections; N1's swiglu_qdq for the official epilogue between them (full-precision expf, bf16
rounding points, per-32 QDQ); per-slot fp32 down outputs (SLOT mode), so the official combine stays N1's.
The kernel's sticky status word (bit 0: an invalid expert id, NaN outputs) is zeroed at build and checked by check().
"""
from __future__ import annotations

import torch

from . import ops as NO


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
