"""Opt-in V1 diagnostics. Imported/installed only with DSPARK_V1_DUMP_DIR.

Selected graph replies are checked against a traced eager replay on the SAME
state. The reply is never replaced. A graph/eager disagreement fails the round.
Window KV is read from the live ring; main_x is the insert graph's actual output,
not a differently batched reconstruction. This mode deliberately synchronizes and
must be disabled for latency/engine measurements. Dumps contain private data.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import torch

from . import fused as FU
from .server import DraftCore
from .tp2 import TP2Backend


def chosen_rounds(value: str) -> frozenset[int]:
    rounds = frozenset(int(x) for x in value.split(","))
    if not rounds or min(rounds) < 1 or len(rounds) > 1024:
        raise ValueError("DSPARK_V1_ROUNDS requires 1..1024 positive round numbers")
    return rounds


class V1Dump:
    def __init__(self, model: Any, ident: dict[str, Any], directory: str, rounds: str = "1"):
        self.m = model
        self.ident = ident
        self.directory = Path(directory).resolve()
        source = Path(__file__).resolve().parents[3]
        if self.directory == source or source in self.directory.parents:
            raise ValueError("V1 dumps must be outside the repository (private capture data)")
        self.rounds = chosen_rounds(rounds)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.main_x: dict[tuple[int, int], torch.Tensor] = {}
        self.sources: dict[tuple[int, int], str] = {}
        self.call = 0
        self.wire: dict[str, int] = {}
        self.samples: list[dict[str, Any]] | None = None
        self.files: list[str] = []
        self._install_insert_observers()

    def _remember(self, slots: torch.Tensor, pos: torch.Tensor, mx: torch.Tensor, source: str) -> None:
        slot_values, positions = slots.cpu().tolist(), pos.cpu().tolist()
        saved_mx = mx.detach().cpu().clone()
        for j, (slot, p) in enumerate(zip(slot_values, positions)):
            if slot >= 4:                    # graph warmup/padding is outside the contract
                continue
            self.main_x[slot, p] = saved_mx[j]
            self.sources[slot, p] = source
            old = [(s, q) for s, q in self.main_x if s == slot and q < p - self.m.s.window + 1]
            for key in old:
                self.main_x.pop(key)
                self.sources.pop(key)

    def _install_insert_observers(self) -> None:
        m = self.m
        m._dump_insert_mx = {}
        original_mx = cast(Callable[[Any, torch.Tensor], torch.Tensor], m._main_x)

        def traced_mx(rk: Any, taps: torch.Tensor) -> torch.Tensor:
            y = original_mx(rk, taps)
            m._dump_insert_mx[rk.r] = y
            return y
        m._main_x = traced_mx
        original_insert = m.insert_taps_graphed

        def insert(slots: torch.Tensor, positions: torch.Tensor, taps: torch.Tensor) -> None:
            original_insert(slots, positions, taps)
            bucket = min(n for n in m.insert_graphs if n >= slots.numel())
            self._remember(slots, positions, m.insert_graphs[bucket]["dump_main_x"][0][:slots.numel()], "taps")
        m.insert_taps_graphed = insert
        original_eager = m.insert_eager

        def eager(slots: torch.Tensor, positions: torch.Tensor, taps: torch.Tensor) -> None:
            m._dump_insert_mx.clear()
            original_eager(slots, positions, taps)
            self._remember(slots, positions, m._dump_insert_mx[0], "taps")
        m.insert_eager = eager
        original_inject = m.insert_mx_eager

        def inject(slots: torch.Tensor, positions: torch.Tensor, mx: torch.Tensor) -> None:
            original_inject(slots, positions, mx)
            self._remember(slots, positions, mx, "engine_mx")
        m.insert_mx_eager = inject
        original_reset = m.reset_slots

        def reset(slots: list[int]) -> None:
            original_reset(slots)
            for key in [key for key in self.main_x if key[0] in slots]:
                self.main_x.pop(key)
                self.sources.pop(key)
        m.reset_slots = reset

    def emit(self, host: torch.Tensor, N: int, n: int, want_margin: bool,
             reply: tuple[np.ndarray, np.ndarray, np.ndarray | None]) -> None:
        self.call += 1
        if self.call not in self.rounds:
            return
        if tuple(self.m.ALL) != (0, 1):
            raise RuntimeError("V1 dumps require complete TP2; a solo shard is not a reference sample")
        trace: dict[str, Any] = {}
        ids, conf, margin, peer_ids = self.m.draft(host, N, n_out=n, margin=want_margin, trace=trace)
        if not (np.array_equal(ids.cpu().numpy(), reply[0]) and
                np.array_equal(conf.cpu().numpy(), reply[1]) and torch.equal(ids, peer_ids.to(ids.device))):
            raise RuntimeError("V1 graph/eager dump replay differs from the actual reply")
        if want_margin:
            if margin is None or reply[2] is None or not np.array_equal(margin.cpu().numpy(), reply[2]):
                raise RuntimeError("V1 graph/eager margins differ")
        rk = self.m.ranks[0]
        B = host.shape[1]
        hh = trace["head_hidden"]
        head_norm = FU.rms_norm(hh, rk.norm, self.m.eps).cpu().view(B, N, -1)
        for j in range(B):
            slot, anchor, P, lo = (int(t) for t in host[:, j])
            positions = list(range(max(lo, P - self.m.s.window + 1), P + 1))
            keys = [(slot, p) for p in positions]
            if not keys or any(key not in self.main_x for key in keys):
                raise RuntimeError("V1 dump lacks inserted main_x provenance for its window")
            src = [self.sources[key] for key in keys]
            sample = self.samples[j] if self.samples else {}
            key = sample.get("key", f"r{self.call:06d}_b{j}")
            record: dict[str, Any] = {
                "schema": "dspark.v1.arm-dump.v1", "P": P, "anchor": anchor, "n_out": n,
                "drafts": reply[0][j].tolist(), "conf": torch.from_numpy(reply[1][j].copy()),
                "base_logits": trace["base_logits"][j].detach().float().cpu(),
                "main_x_P": self.main_x[slot, P].float(),
                "kv_window": rk.ring_kv[slot, :, [p % self.m.s.window for p in positions]].transpose(0, 1).float().cpu(),
                "window": {"n": len(positions), "taps_rows": src.count("taps"),
                           "engine_mx_rows": src.count("engine_mx"), "P_source": src[-1],
                           "first": positions[0], "last": positions[-1]},
                "head_x": hh.view(B, N, -1)[j].float().cpu(), "head_hidden": head_norm[j],
                "sample": sample, "wire": dict(self.wire),
                "meta": {"arm": self.ident.get("arm", "k2-tp2"), "draft_head": self.ident.get("draft_head", "K6"),
                         "drafter_identity": self.ident["drafter_identity"], "weights": "official 1017 below K6 head",
                         "git_sha": os.environ.get("DSPARK_SOURCE_COMMIT", "unrecorded"),
                         "devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "dtype_policy": "arm config/identity recipe",
                         "trace": "eager replay checked bit-for-bit against graph reply"},
            }
            for i in range(self.m.s.n_stages):
                tag = f"stage{i}"
                st = rk.stages[i]
                xin = trace[tag + ".ffn_in"].view(B, N, -1)[j]
                biased = torch.nn.functional.softplus(xin.float() @ st.gate_w.t()).sqrt() + st.gate_bias
                top = biased.topk(self.m.s.top_k + 1, dim=-1)
                record[f"s{i}"] = {
                    "attn_out": trace[tag + ".attn_out"].view(B, N, -1)[j].float().cpu(),
                    "ffn_in": xin.float().cpu(),
                    "ffn_out": trace[tag + ".ffn_out"].view(B, N, -1)[j].float().cpu(),
                    "h_out": trace[tag + ".stream"].view(B, N, self.m.s.hc_mult, -1)[j].float().cpu(),
                    "route": {"idx": trace[tag + ".experts"].view(B, N, -1)[j].long().cpu(),
                              "w": trace[tag + ".route_w"].view(B, N, -1)[j].float().cpu(),
                              "biased": biased.cpu(), "top_vals": top.values.cpu(), "top_ids": top.indices.cpu(),
                              "margin": (top.values[:, -2] - top.values[:, -1]).cpu()},
                }
            # Exclusive creation prevents silently mixing identities or reusing stale samples.
            path = self.directory / f"{key}.pt"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                torch.save(record, f)
            self.files.append(str(path))


class DumpTP2Backend(TP2Backend):
    arm_identity: dict[str, Any]

    def __init__(self, *args: Any, sink: V1Dump, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.sink = sink

    def draft(self, reqs: Any, N: int, want_margin: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        reply = super().draft(reqs, N, want_margin)
        self.sink.emit(self.inp_buf[len(reqs)], N, max(k for *_r, k in reqs), want_margin, reply)
        return cast(tuple[np.ndarray, np.ndarray, np.ndarray | None], reply)


class DumpDraftCore(DraftCore):
    def _step(self, h: Any, body: Any, t_in: float) -> bytes | None:
        self.backend.sink.wire = {"session": h.session, "seq": h.seq}
        return super()._step(h, body, t_in)
