"""DW1 item 3 (F-DWAIT option A2): one graph per STEP on the one-card worker. Default OFF (DSPARK_WORKER_LEAN=1).

Today a decode STEP on GpuBackend makes, after the frame is decoded: a pinned staging copy of the rows, three pageable
host->device copies (slot ids, positions, draft inputs; each blocks the host until the stream reaches it), the insert
graph (with six copy/fill kernels in front of it), the draft graph (with four copy kernels in front of it), three
dtype kernels to pack the reply, one device->host copy into a fresh tensor, and the dtype conversions on the host.

LeanGpuBackend runs the same insert and draft kernels, in the same order on the same inputs, inside ONE captured graph
per (requests B, depth k, n_out, insert bucket): the graph's first nodes copy the pinned host input block and the
pinned row staging into the graph's own device inputs, and its last nodes copy the draft ids and confidences into
pinned host buffers. The host fills the pinned blocks with numpy (padding rows exactly as insert_taps_graphed pads:
slot = scratch slot, position 0, zero taps), replays the graph once, and waits on one event. Draft ids and
confidences are byte-identical to GpuBackend's by construction (same kernels, same inputs, same order); the DW1
identity harness (tools/dw1_identity.py) checks that on synthetic windows and on the V1 replay set.

Shapes without a combined graph (no rows, more rows than the largest combined bucket, margins, a batch or n_out that
was not captured) fall back to GpuBackend's flush + draft unchanged.
"""

from __future__ import annotations

import numpy as np
import torch

from .server import GpuBackend

LEAN_BUCKETS = (8, 16, 32)          # decode STEPs: 1 + accepted rows per request (<= 6 at k = 5), 4 requests <= 24


class LeanGpuBackend(GpuBackend):
    """GpuBackend plus flush_draft(): rows and drafts of one STEP in one graph replay (see module docstring)."""

    def __init__(self, model, max_rows: int, row_bytes: int, num_spec: int, max_batch: int,
                 n_outs: tuple[int, ...] = (4, 5), buckets: tuple[int, ...] = LEAN_BUCKETS):
        super().__init__(model, max_rows, row_bytes, graphs=True)
        m = self.m
        if not m.insert_graphs or not m.graphs:
            raise RuntimeError("LeanGpuBackend needs the model's graphs captured first (model.capture)")
        self.N = num_spec
        self.bmax = max_batch
        self.rmax = max(buckets)
        # pinned host inputs: [slot_rows rmax | pos_rows rmax | slots bmax | anchors bmax | last bmax | lo bmax]
        self.h_in = torch.zeros(2 * self.rmax + 4 * self.bmax, dtype=torch.int64, pin_memory=True)
        self.h_in_np = self.h_in.numpy()
        self.h_taps = torch.zeros(self.rmax, self.hidden3, dtype=torch.bfloat16, pin_memory=True)
        self.h_taps_u8 = self.h_taps.view(torch.uint8).numpy().reshape(self.rmax, row_bytes)
        self.h_ids = torch.zeros(self.bmax, num_spec, dtype=torch.int64, pin_memory=True)
        self.h_conf = torch.zeros(self.bmax, num_spec, dtype=torch.float32, pin_memory=True)
        self.done = torch.cuda.Event(enable_timing=True)
        self._t0 = torch.cuda.Event(enable_timing=True)
        self.time_graph = False             # DW1 timing only: GPU time of the last combined replay
        self.last_graph_us = 0.0
        self.lean_graphs: dict[tuple[int, int, int, int], torch.cuda.CUDAGraph] = {}
        self.n_lean = 0
        self.n_fallback = 0
        self.buckets = tuple(sorted(buckets))
        if self.buckets[-1] > max(self.m.insert_graphs):
            raise ValueError(f"lean bucket {self.buckets[-1]} > the model's largest insert bucket")
        self._capture(n_outs, self.buckets)

    def _capture(self, n_outs: tuple[int, ...], buckets: tuple[int, ...]) -> None:
        m, dev, R, Bm, N = self.m, self.dev, self.rmax, self.bmax, self.N
        o_pos, o_sl, o_an, o_la, o_lo = R, 2 * R, 2 * R + Bm, 2 * R + 2 * Bm, 2 * R + 3 * Bm
        # the warm-up and capture inputs address only the scratch slot, as model.capture does
        self.h_in_np[:R] = m.n_slots
        self.h_in_np[o_pos:o_pos + R] = 0
        self.h_in_np[o_sl:o_sl + Bm] = m.n_slots
        self.h_in_np[o_an:] = 0
        s1 = torch.cuda.Stream(dev)
        for B in range(1, Bm + 1):
            for n in n_outs:
                if (B, N, n) not in m.graphs:
                    continue
                for b in buckets:
                    def body(B=B, n=n, b=b) -> None:
                        m.gi_slot[:b].copy_(self.h_in[:b], non_blocking=True)
                        m.gi_pos[:b].copy_(self.h_in[o_pos:o_pos + b], non_blocking=True)
                        m.gi_taps[:b].copy_(self.h_taps[:b], non_blocking=True)
                        m.insert(m.gi_slot[:b], m.gi_pos[:b], m.main_x(m.gi_taps[:b]))
                        m.g_slots[:B].copy_(self.h_in[o_sl:o_sl + B], non_blocking=True)
                        m.g_anchor[:B].copy_(self.h_in[o_an:o_an + B], non_blocking=True)
                        m.g_last[:B].copy_(self.h_in[o_la:o_la + B], non_blocking=True)
                        m.g_lo[:B].copy_(self.h_in[o_lo:o_lo + B], non_blocking=True)
                        ids, conf = m.draft(m.g_slots[:B], m.g_anchor[:B], m.g_last[:B], N, ctx_lo=m.g_lo[:B],
                                            n_out=n)
                        self.h_ids[:B].copy_(ids, non_blocking=True)
                        self.h_conf[:B].copy_(conf, non_blocking=True)
                    s1.wait_stream(torch.cuda.current_stream(dev))
                    with torch.cuda.stream(s1):
                        for _ in range(2):
                            body()
                    torch.cuda.current_stream(dev).wait_stream(s1)
                    torch.cuda.synchronize(dev)
                    gr = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gr, pool=m.pool, stream=s1):
                        body()
                    self.lean_graphs[(B, N, n, b)] = gr
        m.reset_slots(torch.tensor([m.n_slots], device=dev))
        torch.cuda.synchronize(dev)

    def flush_draft(self, reqs, N: int, want_margin: bool):
        """flush() + draft() of one STEP; one graph replay when a combined graph fits, else the two calls."""
        B = len(reqs)
        n = max(k for *_r, k in reqs)
        rows = sum(r.shape[0] for _, _, r in self.pending)
        b = next((x for x in self.buckets if x >= rows), None) if rows else None
        g = self.lean_graphs.get((B, N, n, b)) if (b is not None and not want_margin) else None
        if g is None:
            self.n_fallback += 1
            self.flush()
            return self.draft(reqs, N, want_margin)
        R, Bm = self.rmax, self.bmax
        o_pos, o_sl = R, 2 * R
        hin, ht = self.h_in_np, self.h_taps_u8
        k = 0
        for slot, p0, r in self.pending:
            c = r.shape[0]
            ht[k:k + c] = r
            hin[k:k + c] = slot
            hin[o_pos + k:o_pos + k + c] = np.arange(p0, p0 + c)
            k += c
        self.pending = []
        ht[rows:b] = 0                      # padding rows: zero taps, scratch slot, position 0 (insert_taps_graphed)
        hin[rows:b] = self.m.n_slots
        hin[o_pos + rows:o_pos + b] = 0
        for j, (slot, tok, apos, rng, _k) in enumerate(reqs):
            hin[o_sl + j] = slot
            hin[o_sl + Bm + j] = tok
            hin[o_sl + 2 * Bm + j] = apos - 1
            hin[o_sl + 3 * Bm + j] = rng.read_lo(apos)
        if self.time_graph:
            self._t0.record()
        g.replay()
        self.done.record()
        self.done.synchronize()
        if self.time_graph:
            self.last_graph_us = self._t0.elapsed_time(self.done) * 1e3
        self.n_flush_rows += rows
        self.n_lean += 1
        return (self.h_ids.numpy()[:B].astype(np.int32), self.h_conf.numpy()[:B].copy(), None)

    def stats(self) -> dict:
        out = super().stats()
        out["lean_steps"] = self.n_lean
        out["lean_fallback_steps"] = self.n_fallback
        return out
