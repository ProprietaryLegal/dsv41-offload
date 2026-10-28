# SPDX-License-Identifier: Apache-2.0
"""One layer's EXL3 expert bank on one TP rank (torch only: no vLLM, no exllamav3 import).

Shared by the vLLM method (moe_method.py, where the tensors are registered Parameters) and the
kernel tests (tests/test_kernels_gpu.py, where the bank owns its tensors).

Layout: experts sit in K-sorted physical slots (plan.LayerPlan.perm). Trellis words of all experts
share two flat int16 banks (w13 = [w1 | w3] per slot, w2), sign/scale vectors are [E, ...] fp16:
    w13_suh [E, 2, H]   (w1, w3: full input width, not cut)
    w13_svh [E, 2, I_loc] (cut along the intermediate dim)
    w2_suh  [E, I_loc]  (cut)
    w2_svh  [E, H]      (not cut)
Each rank keeps intermediate channels [rank * I_loc, (rank + 1) * I_loc) of every expert, cut from
the full exllamav3 tensors at load time (plan.shard_slices; exact because the cut is 128-aligned).
"""

from __future__ import annotations

import torch

from .hostcopy import copy_from_host
from .kernels import GroupTables, ProjTables
from .plan import (
    PROJ_DOWN,
    PROJ_GATE,
    PROJ_UP,
    BankLayout,
    LayerPlan,
    shard_slices,
    tp_shard,
    trellis_shape,
)

TENSOR_NAMES = ("w13_trellis", "w2_trellis", "w13_suh", "w13_svh", "w2_suh", "w2_svh")


class Exl3Bank:

    def __init__(self, lp: LayerPlan, hidden: int, inter_full: int, tp_size: int, tp_rank: int):
        self.lp = lp
        self.hidden = hidden
        self.inter_full = inter_full
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        i0, i1 = tp_shard(inter_full, tp_size, tp_rank)
        self.inter_local = i1 - i0
        self.bl = BankLayout.build(lp, hidden, self.inter_local)
        self.t: dict[str, torch.Tensor] = {}

    @property
    def num_experts(self) -> int:
        return self.lp.num_experts

    def allocate(self, device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
        """Empty tensors (device None = torch's current default device, which is what vLLM's
        create_weights relies on)."""
        E, H, I = self.num_experts, self.hidden, self.inter_local
        kw = {} if device is None else {"device": device}
        f16 = torch.float16
        self.t = {
            "w13_trellis": torch.empty(self.bl.total13, dtype=torch.int16, **kw),
            "w2_trellis": torch.empty(self.bl.total2, dtype=torch.int16, **kw),
            "w13_suh": torch.empty(E, 2, H, dtype=f16, **kw),
            "w13_svh": torch.empty(E, 2, I, dtype=f16, **kw),
            "w2_suh": torch.empty(E, I, dtype=f16, **kw),
            "w2_svh": torch.empty(E, H, dtype=f16, **kw),
        }
        return self.t

    def bind(self, tensors: dict[str, torch.Tensor]) -> None:
        """Point the bank at externally owned tensors (vLLM Parameters' .data after loading)."""
        for n in TENSOR_NAMES:
            self.t[n] = tensors[n]

    @staticmethod
    def dest_name(kind: str, proj: str) -> str:
        side = "w2" if proj == PROJ_DOWN else "w13"
        return f"{side}_{kind}"

    def load_piece(self, kind: str, proj: str, expert_id: int, w: torch.Tensor,
                   dst: torch.Tensor | None = None) -> None:
        """Copy this rank's part of one full exllamav3 tensor (trellis / suh / svh) of logical expert
        `expert_id` into its slot. Raises on any shape or dtype the plan does not predict."""
        if proj not in (PROJ_GATE, PROJ_UP, PROJ_DOWN):
            raise ValueError(f"bad projection {proj!r}")
        e = int(expert_id)
        p = self.lp.perm[e]
        b = self.lp.bits[e]
        bl = self.bl
        gate_up = proj in (PROJ_GATE, PROJ_UP)
        sl = shard_slices(proj, self.inter_full, self.tp_size, self.tp_rank)
        if dst is None:
            dst = self.t[self.dest_name(kind, proj)]

        if kind == "trellis":
            k = b.k_gu if gate_up else b.k_d
            rows, cols = (self.hidden, self.inter_full) if gate_up else (self.inter_full, self.hidden)
            want = trellis_shape(rows, cols, k)
            if tuple(w.shape) != want or w.dtype != torch.int16:
                raise ValueError(f"expert {e} {proj}.trellis: got {tuple(w.shape)} {w.dtype}; the plan's "
                                 f"K={k} means {want} int16 (plan and checkpoint disagree)")
            a, z = sl.trellis_range
            src = w.narrow(sl.trellis_dim, a, z - a)
            if proj == PROJ_GATE:
                off, n = bl.off_w1[p], bl.words_w1[p]
            elif proj == PROJ_UP:
                off, n = bl.off_w3[p], bl.words_w1[p]
            else:
                off, n = bl.off_w2[p], bl.words_w2[p]
            copy_from_host(dst[off:off + n].view(src.shape), src)     # w1/w3 source is a strided column cut
        elif kind in ("suh", "svh"):
            if w.dtype != torch.float16:
                raise ValueError(f"expert {e} {proj}.{kind}: dtype {w.dtype}, want float16")
            rng = sl.suh_range if kind == "suh" else sl.svh_range
            src = w if rng is None else w[rng[0]:rng[1]]
            d = dst[p, 0 if proj == PROJ_GATE else 1] if gate_up else dst[p]
            if tuple(d.shape) != tuple(src.shape):
                raise ValueError(f"expert {e} {proj}.{kind}: {tuple(src.shape)} vs slot {tuple(d.shape)}")
            copy_from_host(d, src)
        else:
            raise ValueError(f"bad tensor kind {kind!r}")

    def views(self, p: int):
        """((trellis view, suh, svh, K) for w1, w3, w2) of physical slot p."""
        bl, t = self.bl, self.t
        b = self.lp.bits[self.lp.inv_perm[p]]
        s1 = trellis_shape(self.hidden, self.inter_local, b.k_gu)
        s2 = trellis_shape(self.inter_local, self.hidden, b.k_d)
        w13, w2 = t["w13_trellis"], t["w2_trellis"]
        return (
            (w13[bl.off_w1[p]:bl.off_w1[p] + bl.words_w1[p]].view(s1), t["w13_suh"][p, 0], t["w13_svh"][p, 0], b.k_gu),
            (w13[bl.off_w3[p]:bl.off_w3[p] + bl.words_w1[p]].view(s1), t["w13_suh"][p, 1], t["w13_svh"][p, 1], b.k_gu),
            (w2[bl.off_w2[p]:bl.off_w2[p] + bl.words_w2[p]].view(s2), t["w2_suh"][p], t["w2_svh"][p], b.k_d),
        )

    def tables(self) -> tuple[list[GroupTables], tuple[ProjTables, ProjTables, ProjTables]]:
        """Device pointer tables (int64, physical slot order) and one GroupTables per K group.
        Pointers are raw addresses: rebuild them if the tensors ever move."""
        bl, t = self.bl, self.t
        E, H, I = self.num_experts, self.hidden, self.inter_local
        dev = t["w13_trellis"].device
        b13, b2 = t["w13_trellis"].data_ptr(), t["w2_trellis"].data_ptr()
        s13, v13 = t["w13_suh"].data_ptr(), t["w13_svh"].data_ptr()
        s2, v2 = t["w2_suh"].data_ptr(), t["w2_svh"].data_ptr()
        fb = 2   # int16 / fp16 bytes
        for n in TENSOR_NAMES:
            if not t[n].is_contiguous():
                raise ValueError(f"{n} must be contiguous for raw pointer tables")

        def tab(vals):
            return torch.tensor(vals, dtype=torch.int64, device=dev)

        g = ProjTables(tab([b13 + fb * o for o in bl.off_w1]),
                       tab([s13 + fb * (p * 2 + 0) * H for p in range(E)]),
                       tab([v13 + fb * (p * 2 + 0) * I for p in range(E)]))
        u = ProjTables(tab([b13 + fb * o for o in bl.off_w3]),
                       tab([s13 + fb * (p * 2 + 1) * H for p in range(E)]),
                       tab([v13 + fb * (p * 2 + 1) * I for p in range(E)]))
        d = ProjTables(tab([b2 + fb * o for o in bl.off_w2]),
                       tab([s2 + fb * p * I for p in range(E)]),
                       tab([v2 + fb * p * H for p in range(E)]))
        groups = [GroupTables(grp.k_gu, grp.k_d, grp.lo, grp.hi,
                              g.slice(grp.lo, grp.hi), u.slice(grp.lo, grp.hi), d.slice(grp.lo, grp.hi))
                  for grp in self.lp.groups]
        return groups, (g, u, d)
