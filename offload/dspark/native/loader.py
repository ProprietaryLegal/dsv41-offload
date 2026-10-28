"""Streaming loader: official 1017 tensors -> each rank's final native shards on its own device.

Nothing is assembled whole on a GPU: every expert matrix / dense tensor is read from the memory-mapped official shard,
cut on the host to the rank's block-aligned window, copied once to that rank's device and (expanded arms) decoded there
straight into its slot of the preallocated bank. Peak extra device memory while loading = one matrix being decoded
(~0.1 GB); peak extra host memory = one matrix (~6 MB) plus the page cache of the mapped shards.

Verification (fail loudly, never serve unverified bytes): every official tensor is re-hashed against
NATIVE_MANIFEST.json before use; packed shards are compared byte for byte with the host slice after the copy;
expanded shards are compared with formats.decode_fp4 on the host for a sample of experts per stage (all three matrices,
both ranks) and checked finite everywhere. FP8 dense scales become fp32 values (exact powers of two).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import formats as F
from .official import OfficialShards
from .ops import ExpertBank, NativeDense

N_EXPERTS = 128
HIDDEN = 5120
INTER = 2304


@dataclass
class LoadStats:
    seconds: float = 0.0
    bytes_read: int = 0
    tensors: int = 0
    verified_full: int = 0
    verified_sample: int = 0
    peak_alloc: dict = field(default_factory=dict)        # device -> torch max_memory_allocated during the load
    peak_reserved: dict = field(default_factory=dict)


def _np(sh: OfficialShards, name: str, stats: LoadStats, verify: bool) -> np.ndarray:
    if verify:
        sh.verify(name)
    info = sh.info(name)
    raw = sh.raw(name)
    stats.bytes_read += raw.nbytes
    stats.tensors += 1
    shape = info["shape"]
    if info["dtype"] in ("I8", "F8_E4M3", "F8_E8M0"):
        return np.asarray(raw).view(np.uint8).reshape(shape)
    if info["dtype"] == "BF16":
        return np.asarray(raw).view(np.uint16).reshape(shape)
    if info["dtype"] == "F32":
        return np.asarray(raw).view(np.float32).reshape(shape)
    raise ValueError(f"{name}: dtype {info['dtype']}")


def _to(dev: torch.device, a: np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(a))
    if dtype is torch.bfloat16:
        t = t.view(torch.bfloat16)
    return t.to(dev)


def _check_scales(name: str, s: np.ndarray) -> None:
    if (s == 255).any() or (s == 0).any():
        raise RuntimeError(f"{name}: E8M0 scale bytes 0/255 present (not served: the kernels assume 1..254)")


def load_expert_banks(sh: OfficialShards, stage: int, ranks: list[tuple[int, torch.device]], tp: int, kind: str,
                      stats: LoadStats, verify_sample: int = 4, n_experts: int = N_EXPERTS) -> dict[int, ExpertBank]:
    """Routed experts of mtp.<stage> for each (rank, device): rank r holds intermediate channels
    [r*I/tp, (r+1)*I/tp) of w1/w3 (output rows) and of w2 (inputs, = packed bytes and 32-groups of the same window)."""
    I_r = INTER // tp
    if I_r % 64:
        raise ValueError(f"tp {tp}: {I_r} channels per rank not a multiple of 64")
    banks = {r: ExpertBank(f"mtp.{stage}.experts[r{r}]", kind, n_experts, HIDDEN, I_r, dev) for r, dev in ranks}
    sample = set(np.linspace(0, n_experts - 1, verify_sample).astype(int).tolist()) if verify_sample else set()
    for e in range(n_experts):
        p = f"mtp.{stage}.ffn.experts.{e}."
        w1, s1 = _np(sh, p + "w1.weight", stats, True), _np(sh, p + "w1.scale", stats, True)
        w3, s3 = _np(sh, p + "w3.weight", stats, True), _np(sh, p + "w3.scale", stats, True)
        w2, s2 = _np(sh, p + "w2.weight", stats, True), _np(sh, p + "w2.scale", stats, True)
        for nm, s in (("w1", s1), ("w3", s3), ("w2", s2)):
            _check_scales(p + nm, s)
        for r, dev in ranks:
            bk = banks[r]
            rows = slice(r * I_r, (r + 1) * I_r)
            pk = slice(r * I_r // 2, (r + 1) * I_r // 2)
            sg = slice(r * I_r // 32, (r + 1) * I_r // 32)
            parts = {"w1": (w1[rows], s1[rows]), "w3": (w3[rows], s3[rows]), "w2": (w2[:, pk], s2[:, sg])}
            for j, nm in enumerate(("w1", "w3")):
                wd, sd = _to(dev, parts[nm][0]), _to(dev, parts[nm][1])
                if kind == "packed":
                    bk.w13[e, j].copy_(wd)
                    bk.s13[e, j].copy_(sd)
                else:
                    bk.w13[e, j].copy_(F.decode_fp4(wd, sd).to(torch.bfloat16))
            wd, sd = _to(dev, parts["w2"][0]), _to(dev, parts["w2"][1])
            if kind == "packed":
                bk.w2[e].copy_(wd)
                bk.s2[e].copy_(sd)
            else:
                bk.w2[e].copy_(F.decode_fp4(wd, sd).to(torch.bfloat16))
            # verification
            if kind == "packed":
                for got, want in ((bk.w13[e, 0], parts["w1"][0]), (bk.s13[e, 0], parts["w1"][1]),
                                  (bk.w13[e, 1], parts["w3"][0]), (bk.s13[e, 1], parts["w3"][1]),
                                  (bk.w2[e], parts["w2"][0]), (bk.s2[e], parts["w2"][1])):
                    if not np.array_equal(got.cpu().numpy(), np.ascontiguousarray(want)):
                        raise RuntimeError(f"{p}: rank {r} packed shard differs from the official bytes")
                stats.verified_full += 6
            elif e in sample:
                for got, (wa, sa) in ((bk.w13[e, 0], parts["w1"]), (bk.w13[e, 1], parts["w3"]), (bk.w2[e], parts["w2"])):
                    want = F.decode_fp4(torch.from_numpy(np.ascontiguousarray(wa)),
                                        torch.from_numpy(np.ascontiguousarray(sa)))
                    if not torch.equal(got.cpu().float(), want):
                        raise RuntimeError(f"{p}: rank {r} expanded shard is not the exact decode")
                stats.verified_sample += 3
    for r, dev in ranks:
        if kind == "expanded":                 # one expert at a time: a whole-bank isfinite needs ~4 GiB of temporaries
            for e in range(n_experts):
                for t in (banks[r].w13[e], banks[r].w2[e]):
                    if not bool(torch.isfinite(t).all()):
                        raise RuntimeError(f"mtp.{stage} expert {e} rank {r}: non-finite expanded weights")
    return banks


def load_dense(sh: OfficialShards, name: str, dev: torch.device, kind: str, stats: LoadStats,
               rows: tuple[int, int] | None = None, cols: tuple[int, int] | None = None) -> NativeDense:
    """FP8 + E8M0 32x32 tensor ``name`` (``.weight`` / ``.scale``), the block-aligned window rows x cols, as ``fp8``
    (bytes + fp32 scale values) or ``bf16`` (exact decode). Verified: bytes / exact decode against the host."""
    w = _np(sh, name + ".weight", stats, True)
    s = _np(sh, name + ".scale", stats, True)
    _check_scales(name, s)
    N, K = w.shape
    r0, r1 = rows or (0, N)
    c0, c1 = cols or (0, K)
    if r0 % 32 or c0 % 32 or (r1 - r0) % 32 or (c1 - c0) % 32:
        raise ValueError(f"{name}: window rows {r0}:{r1} cols {c0}:{c1} not on 32x32 scale blocks")
    ws = np.ascontiguousarray(w[r0:r1, c0:c1])
    ss = np.ascontiguousarray(s[r0 // 32:r1 // 32, c0 // 32:c1 // 32])
    tag = f"{name}[{r0}:{r1},{c0}:{c1}]"
    if kind == "fp8":
        if ((ws == 0x7F) | (ws == 0xFF)).any():
            raise RuntimeError(f"{name}: NaN e4m3 codes")
        sv = F.e8m0_to_f32(torch.from_numpy(ss))
        d = NativeDense(tag, "fp8", _to(dev, ws), sv.to(dev))
        if not np.array_equal(d.w.cpu().numpy(), ws):
            raise RuntimeError(f"{tag}: device bytes differ from the official bytes")
    else:
        full = F.decode_fp8_block(torch.from_numpy(ws), torch.from_numpy(ss))
        d = NativeDense(tag, "bf16", full.to(torch.bfloat16).to(dev), None)
        if not torch.equal(d.w.cpu().float(), full):
            raise RuntimeError(f"{tag}: expanded weights are not the exact decode")
    stats.verified_full += 1
    return d


def load_bf16(sh: OfficialShards, name: str, dev: torch.device, stats: LoadStats,
              rows: tuple[int, int] | None = None) -> torch.Tensor:
    """An official BF16 tensor (e.g. head.weight), optionally a row window, verified by sha256 of the whole tensor."""
    a = _np(sh, name, stats, True)
    if rows:
        a = a[rows[0]:rows[1]]
    t = _to(dev, a, torch.bfloat16)
    stats.verified_full += 1
    return t


def track_peaks(devs: list[torch.device]):
    """Context helper: reset and later read torch's peak counters on each device."""
    class _P:
        def __enter__(self):
            self.t0 = time.time()
            for d in devs:
                torch.cuda.synchronize(d)
                torch.cuda.reset_peak_memory_stats(d)
            return self

        def __exit__(self, *a):
            self.seconds = time.time() - self.t0
            self.alloc = {str(d): torch.cuda.max_memory_allocated(d) for d in devs}
            self.reserved = {str(d): torch.cuda.max_memory_reserved(d) for d in devs}
            return False
    return _P()
