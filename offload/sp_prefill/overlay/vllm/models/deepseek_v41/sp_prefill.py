# SPDX-License-Identifier: Apache-2.0
"""dsv41 R-SP (27.10.2026): prompt-only sequence parallelism (SP) for the DeepSeek V4.1 decoder at TP 2.

A port of the idea in spark-ds41f patches 0014-0018 (UPS1 #3) onto our composed K1 model.py; the donor's code targets a
different model tree (deepseek_v4_1, CED, b12x) and does not apply, so this is a re-implementation. Switch:
VLLM_DSV41_PREFILL_SP=1 (default 0 = the model runs exactly as K1). VLLM_DSV41_PREFILL_SP_MIN_ROWS raises the threshold.

Mode. At TP 2 both ranks repeat the row-wise work of every decoder layer (the mHC residual carry, mHC pre / post /
post_pre, both norms, the Engram gate) on every token. A prompt forward running SP keeps L = Tp / N rows on each rank
between collectives (Tp = N * ceil(T / N); rank r owns padded rows [r * L, (r + 1) * L)) and gathers the rows back to T
before the ops that need all of them:

- attention: all-gather the normed input to T rows, run every projection, the indexer, compressor, KV writes and sparse
  MLA on all T rows exactly as K1 does, leave wo_b's TP partial unreduced and reduce-scatter it (instead of the
  all-reduce);
- MoE: all-gather to T rows with the full input_ids; gate, top-k, routed and shared experts run as K1; the runner skips
  its final all-reduce (``moe_config.skip_final_all_reduce``, the hook vLLM's Inkling model uses) and the partial
  shared + routed sum is reduce-scattered;
- Engram: the lookup and ``wkv`` run on all T rows as K1; the gate runs on this rank's real rows, pad rows pass through;
- after the last layer the collapse and final norm run on L rows and are gathered; aux hidden states and the MTP buffer
  are gathered where K1's static SP gathers them;
- decoder SWA bounded replay (patch 0034): a trimming step leaves SP before the replay layers. The cut layer's KV write
  gathers its attention input; the replay batch's rows are fetched exactly from their owning ranks; the replay layers
  run unsplit, as in K1.

Exactness. A reduce-scatter adds the same two partials as the all-reduce (a + b is commutative), every gathered tensor
is a copy, and attention and the MoE see the same T rows. What remains is the row-wise kernels running on L instead of
T rows: they must pick the same variant for both counts. ``exact_local_floor`` sizes L so that they do on the image's
dispatch rules (the fused mHC post + pre-norm GEMM below 33 rows, the RMSNorm block switch at 256 rows, the mHC pre
GEMM's split-k count from the SM count). DeepGEMM's mega mHC kernel, if present, is not covered by that argument; the
first window's greedy check covers it.

Threshold (0018's rule plus the floor above): a forward runs SP only when T exceeds every decode forward (largest CUDA
graph capture size, max_num_seqs * (1 + k), N), its padded hidden-state message exceeds the EXL3 RDMA one-shot
all-reduce limit, its local rows clear the exactness floor, and no stream is capturing. Every rank shares T, so no
agreement is needed.

No vLLM import at module load (the CPU tests load this file without vLLM); collectives, the forward-context override
and the Engram kernel are resolved on first use.
"""
from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import torch

logger = logging.getLogger(__name__)


def _flag(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


ENABLED = _flag("VLLM_DSV41_PREFILL_SP")

# Image dispatch rules the exactness floor follows (pinned against the image source by the CPU tests):
FUSED_POST_PRE_MAX_TOKENS = 32  # kernels/mhc/tilelang_kernels.py _FUSED_POST_PRE_MAX_TOKENS: fused post + pre-norm GEMM
RMS_NORM_BLOCK_ROWS = 256  # csrc/layernorm_kernels.cu: 1024-thread blocks below 256 rows, 256 from there
MHC_PRE_BLOCK_ROWS = 64  # kernels/mhc/warmup.py compute_mhc_pre_num_splits: compute_num_split(64, K, cdiv(rows, 64))


class Comm(Protocol):
    """Dim-0 collectives of the TP group."""

    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...

    def reduce_scatter(self, x: torch.Tensor) -> torch.Tensor: ...


class _TPComm:
    """The image's static-SP ops (models/common/ops/sequence_parallel.py): NCCL for these message sizes."""

    def __init__(self) -> None:
        from vllm.models.common.ops.sequence_parallel import (  # type: ignore[import-not-found]
            sp_all_gather,
            sp_reduce_scatter,
        )

        self._gather = sp_all_gather
        self._reduce_scatter = sp_reduce_scatter

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self._gather(x)
        return out

    def reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self._reduce_scatter(x)
        return out


_COMM: Comm | None = None


def comm() -> Comm:
    global _COMM
    if _COMM is None:
        _COMM = _TPComm()
    return _COMM


def use_comm(new: Comm | None) -> None:
    """Tests only: route the collectives through ``new`` (None: back to the TP group)."""
    global _COMM
    _COMM = new


@dataclass(frozen=True)
class SPRows:
    """Row layout of one sequence-parallel forward on one TP rank."""

    num_tokens: int
    world_size: int
    rank: int

    def __post_init__(self) -> None:
        if self.num_tokens <= 0 or self.world_size <= 1 or not 0 <= self.rank < self.world_size:
            raise ValueError(f"invalid SP rows {self}")

    @property
    def padded_rows(self) -> int:
        """Tp: T rounded up to a multiple of N; the rows every collective sees."""
        return -(-self.num_tokens // self.world_size) * self.world_size

    @property
    def local_rows(self) -> int:
        """L: rows this rank carries between collectives."""
        return self.padded_rows // self.world_size

    @property
    def start(self) -> int:
        return self.rank * self.local_rows

    @property
    def real_rows(self) -> int:
        """Leading local rows that are tokens; the rest (last rank only) are zero padding."""
        return max(0, min(self.local_rows, self.num_tokens - self.start))

    def local_slice(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's L rows of a T-row tensor; pad rows are zero."""
        if x.shape[0] != self.num_tokens:
            raise ValueError(f"SP slice expects {self.num_tokens} rows, got {tuple(x.shape)}")
        rows = x[self.start : self.start + self.real_rows]
        if self.real_rows == self.local_rows:
            return rows
        local = x.new_zeros((self.local_rows, *x.shape[1:]))
        local[: self.real_rows].copy_(rows)
        return local

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        """All ranks' L rows, in rank order, cut to the first T."""
        if x.shape[0] != self.local_rows:
            raise ValueError(f"SP gather expects {self.local_rows} rows, got {tuple(x.shape)}")
        return comm().all_gather(x)[: self.num_tokens]

    def reduce_scatter(self, partial: torch.Tensor) -> torch.Tensor:
        """This rank's L rows of the sum over ranks of a T-row partial (zero-padded to Tp)."""
        if partial.shape[0] != self.num_tokens:
            raise ValueError(f"SP reduce-scatter expects {self.num_tokens} rows, got {tuple(partial.shape)}")
        if self.padded_rows != self.num_tokens:
            partial = torch.nn.functional.pad(partial, (0, 0) * (partial.dim() - 1) + (0, self.padded_rows - self.num_tokens))
        out = comm().reduce_scatter(partial)
        if out.shape[0] != self.local_rows:
            raise RuntimeError(f"SP reduce-scatter returned {tuple(out.shape)}, expected {self.local_rows} rows")
        return out

    def select(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Rows ``index`` (global, < T, the same on every rank) of a tensor whose rows are spread over the ranks:
        each rank contributes the rows it owns (zeros elsewhere), one all-gather, then each row is taken from its
        owner. Copies only, so the values are bit-identical to indexing the T-row tensor."""
        if x.shape[0] != self.local_rows:
            raise ValueError(f"SP select expects {self.local_rows} rows, got {tuple(x.shape)}")
        if index.numel() == 0:  # every rank holds the same index, so every rank skips the collective
            return x.new_empty((0, *x.shape[1:]))
        index = index.long()
        owner = torch.div(index, self.local_rows, rounding_mode="floor")
        mine = owner == self.rank
        local = torch.where(mine, index - owner * self.local_rows, torch.zeros_like(index))
        picked = x.index_select(0, local)
        keep = mine.view(-1, *([1] * (x.dim() - 1)))
        picked = torch.where(keep, picked, torch.zeros_like(picked))
        gathered = comm().all_gather(picked).view(self.world_size, index.shape[0], *x.shape[1:])
        return gathered[owner, torch.arange(index.shape[0], device=index.device)]


def exact_local_floor(n_sms: int, deep_gemm_mhc: bool) -> int:
    """The fewest local rows L for which every row-wise kernel picks the variant it picks for any T >= N * L - N + 1.

    The fused mHC post + pre-norm GEMM serves <= 32 rows only; the RMSNorm kernel switches block size at 256 rows; with
    DeepGEMM the mHC pre-norm GEMM splits K max(1, min(n_sms // cdiv(rows, 64), K_blocks // 4)) ways (buckets 1/4/16),
    which is 1 for every rows with cdiv(rows, 64) > n_sms // 2."""
    if n_sms <= 0:
        raise ValueError(f"SM count must be positive, got {n_sms}")
    floor = max(FUSED_POST_PRE_MAX_TOKENS + 1, RMS_NORM_BLOCK_ROWS)
    if deep_gemm_mhc:
        floor = max(floor, MHC_PRE_BLOCK_ROWS * (n_sms // 2) + 1)
    return floor


def rows_above_bytes(limit: int, row_bytes: int, world_size: int) -> int:
    """The fewest tokens T whose Tp-row message exceeds ``limit`` bytes (0018)."""
    if limit <= 0:
        return 1
    padded = -(-(limit // row_bytes + 1) // world_size) * world_size
    return padded - world_size + 1


def _rdma_all_reduce_max_bytes() -> int:
    """EXL3 RDMA one-shot all-reduce limit (vllm_exl3_moe/rdma.py: EXL3_RDMA_AR=1, EXL3_RDMA_AR_MAX_KB, default 256)."""
    if os.environ.get("EXL3_RDMA_AR", "0") != "1":
        return 0
    return int(os.environ.get("EXL3_RDMA_AR_MAX_KB", "256")) * 1024


def _gpu_facts() -> tuple[int, bool]:
    """(SM count, whether the mHC pre-norm GEMM takes DeepGEMM's split-k path) on this worker."""
    device = torch.cuda.current_device()
    n_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    try:
        from vllm.utils.deep_gemm import is_deep_gemm_supported  # type: ignore[import-not-found]

        deep_gemm = bool(is_deep_gemm_supported())
    except ImportError:
        deep_gemm = True  # the larger floor is always exact
    return n_sms, deep_gemm


def resolve_min_rows(
    vllm_config: Any,
    *,
    static_sequence_parallel: bool,
    fuse_mhc_all_reduce: bool,
    native_mega_moe: bool,
    gemm_rs: bool,
    gpu_facts: Callable[[], tuple[int, bool]] = _gpu_facts,
) -> int:
    """The smallest forward, in tokens, that runs SP; 0 means never (with the reason logged once)."""
    if not ENABLED:
        return 0
    parallel = vllm_config.parallel_config
    reason = None
    if parallel.tensor_parallel_size <= 1:
        reason = "tensor parallel size is 1"
    elif parallel.pipeline_parallel_size != 1:
        reason = "pipeline parallelism keeps full-row intermediate tensors"
    elif parallel.data_parallel_size != 1:
        reason = "data parallelism is not covered"
    elif getattr(parallel, "prefill_context_parallel_size", 1) != 1:
        reason = "prefill context parallelism shards the rows already"
    elif getattr(parallel, "use_ubatching", False):
        reason = "microbatching splits the forward"
    elif static_sequence_parallel:
        reason = "static sequence parallelism is enabled"
    elif native_mega_moe:
        reason = "the native mega MoE takes token-sharded input itself"
    elif fuse_mhc_all_reduce:
        reason = "the fused mHC all-reduce owns the layer boundary"
    elif gemm_rs:
        reason = "GEMM-RS is bound"
    if reason is not None:
        logger.warning("dsv41 R-SP: prefill sequence parallelism off: %s", reason)
        return 0
    spec = vllm_config.speculative_config
    world = int(parallel.tensor_parallel_size)
    decode_rows = max(
        int(vllm_config.compilation_config.max_cudagraph_capture_size or 0),
        int(vllm_config.scheduler_config.max_num_seqs) * (1 + (int(spec.num_speculative_tokens) if spec is not None else 0)),
        world,
    )
    model = vllm_config.model_config
    row_bytes = int(model.get_hidden_size()) * int(model.dtype.itemsize)
    n_sms, deep_gemm = gpu_facts()
    exact_rows = world * (exact_local_floor(n_sms, deep_gemm) - 1) + 1
    rows = max(decode_rows + 1, rows_above_bytes(_rdma_all_reduce_max_bytes(), row_bytes, world), exact_rows)
    override = os.environ.get("VLLM_DSV41_PREFILL_SP_MIN_ROWS")
    if override is not None:
        if not override.isdigit():
            raise ValueError(f"VLLM_DSV41_PREFILL_SP_MIN_ROWS must be a row count, got {override!r}")
        rows = max(rows, int(override))
    logger.warning(
        "dsv41 R-SP: prefill sequence parallelism from %d tokens (decode %d, %d SMs, DeepGEMM mHC %s)",
        rows, decode_rows, n_sms, deep_gemm,
    )
    return rows


def plan(num_tokens: int, min_rows: int, world_size: int, rank: int, capturing: bool) -> SPRows | None:
    """The SP row layout of a forward of ``num_tokens`` tokens, or None for the K1 path."""
    if min_rows <= 0 or capturing or num_tokens < min_rows or world_size <= 1:
        return None
    return SPRows(num_tokens, world_size, rank)


class _Linear(Protocol):
    reduce_results: bool


@contextmanager
def unreduced(linear: _Linear) -> Iterator[None]:
    """Run a RowParallelLinear without its all-reduce (it reads ``reduce_results`` at call time)."""
    if not linear.reduce_results:
        raise RuntimeError("dsv41 R-SP: wo_b is already unreduced (static SP or the fused mHC all-reduce)")
    linear.reduce_results = False
    try:
        yield
    finally:
        linear.reduce_results = True


def moe(ffn: Any, rows: SPRows, x: torch.Tensor, input_ids: torch.Tensor | None) -> torch.Tensor:
    """The MoE on every row (K1's runner path for prompt-size batches) and this rank's rows of its sum.

    K1 reaches the same runner call through ``DeepseekV4MoE._forward_fused_moe``: the EXL3 one-launch path there takes
    at most 8 rows, far below any SP forward."""
    if ffn.use_native_mega_moe or ffn.use_sequence_parallel:
        raise RuntimeError("dsv41 R-SP: the MoE is token-sharded already")
    if (ffn.gate.tid2eid is not None or getattr(ffn.gate, "bias_vl", None) is not None) and input_ids is None:
        raise ValueError("dsv41 R-SP: hash / vision MoE routing requires input_ids")
    config = ffn.experts.moe_config
    if config.skip_final_all_reduce or config.is_sequence_parallel:
        raise RuntimeError("dsv41 R-SP: the MoE runner does not end in its own all-reduce")
    full = rows.gather(x)
    config.skip_final_all_reduce = True
    try:
        partial = ffn.experts(hidden_states=full, router_logits=full, input_ids=input_ids)
    finally:
        config.skip_final_all_reduce = False
    out: torch.Tensor = rows.reduce_scatter(partial.view(full.shape))
    return out


def engram_local(
    engram: Any, rows: SPRows, hidden_states: torch.Tensor, hash_ids: torch.Tensor, token_mask: torch.Tensor | None
) -> torch.Tensor:
    """``Engram.forward`` for this rank's rows: the lookup and ``wkv`` on all T rows (column-parallel wkv needs the
    same rows on every rank), the gate on the local real rows; pad rows pass through (the kernel zeroes key and value
    past ``num_kv_tokens``)."""
    if engram.use_sequence_parallel:
        raise RuntimeError("dsv41 R-SP: Engram is built for static SP")
    if hidden_states.shape[0] != rows.local_rows or hash_ids.shape[0] != rows.num_tokens:
        raise ValueError(f"dsv41 R-SP: Engram rows {tuple(hidden_states.shape)} / {tuple(hash_ids.shape)} for {rows}")
    kv = engram.wkv(engram.embed(hash_ids).flatten(-2))
    start, real = rows.start, rows.real_rows
    if token_mask is not None:
        if token_mask.shape != (rows.num_tokens,):
            raise ValueError(f"dsv41 R-SP: Engram mask {tuple(token_mask.shape)} for {rows.num_tokens} tokens")
        token_mask = token_mask[start : start + real]
    out: torch.Tensor = engram_post(engram, hidden_states, kv[start : start + real], token_mask, real)
    return out


def _triton_engram_post(
    engram: Any, hidden_states: torch.Tensor, kv: torch.Tensor, token_mask: torch.Tensor | None, num_kv_tokens: int
) -> torch.Tensor:
    """Engram.forward's own kernel launch (the CPU tests compare this call with Engram.forward's, argument by argument)."""
    module = sys.modules[type(engram).__module__]
    triton = module.triton
    num_tokens, hc_mult, dim = hidden_states.shape
    assert hc_mult == engram.hc_mult and dim == engram.dim
    assert kv.ndim == 2 and kv.shape[1] == (hc_mult + 1) * dim
    output = torch.empty_like(hidden_states)
    if num_tokens == 0:
        return output

    block_size = triton.next_power_of_2(dim)
    num_warps = 8 if block_size >= 2048 else 4
    mask = token_mask if token_mask is not None else hidden_states
    module._fused_engram_post_wkv_kernel[(num_tokens * hc_mult,)](
        hidden_states,
        kv,
        engram.q_weight,
        engram.k_weight,
        mask,
        output,
        num_kv_tokens,
        hidden_states.stride(0),
        hidden_states.stride(1),
        hidden_states.stride(2),
        kv.stride(0),
        kv.stride(1),
        engram.q_weight.stride(0),
        engram.q_weight.stride(1),
        engram.k_weight.stride(0),
        engram.k_weight.stride(1),
        token_mask.stride(0) if token_mask is not None else 0,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        engram.eps,
        engram.clamp_value,
        DIM=dim,
        HC_MULT=hc_mult,
        BLOCK_SIZE=block_size,
        HAS_MASK=token_mask is not None,
        num_warps=num_warps,
    )
    return output


# The CPU tests swap in a torch reference of the kernel.
engram_post: Callable[[Any, torch.Tensor, torch.Tensor, torch.Tensor | None, int], torch.Tensor] = _triton_engram_post


def _override_forward_context(context: Any) -> Any:
    from vllm.forward_context import override_forward_context  # type: ignore[import-not-found]

    return override_forward_context(context)


forward_context = _override_forward_context


def replay(
    layers: Any,
    run_split: Callable[..., tuple[torch.Tensor, ...]],
    rows: SPRows,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    input_ids: torch.Tensor | None,
    pre_mix: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[SPRows | None, tuple[torch.Tensor, ...]]:
    """``DecoderReplayLayers.__call__`` for a forward running SP; returns the rows still split (None once left).

    Without a replay batch the replay layers continue on the split rows. A trimming step writes the cut layer's KV for
    every row (its attention input gathered), fetches the replay rows from their owners and runs the replay layers
    unsplit under the replay batch's forward context, scattering their outputs back to all T rows like K1."""
    batch_rows = layers.rows
    if batch_rows is None:
        return rows, run_split(hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual, sp_rows=rows)
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("dsv41 R-SP: a trimming step's last full layer finalizes its own MoE")
    if layers.write_batch_kv is not None:
        layers.write_batch_kv(hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual, sp_rows=rows)
    num_rows = batch_rows.shape[0]
    for buf in layers.row_buffers:
        buf[:num_rows].copy_(buf.index_select(0, batch_rows))
    with forward_context(layers.forward_context):
        row_outputs = layers.run_layers(
            rows.select(hidden_states, batch_rows),
            positions.index_select(0, batch_rows),
            None if input_ids is None else input_ids.index_select(0, batch_rows),
            rows.select(pre_mix, batch_rows),
            rows.select(post_mix, batch_rows),
            rows.select(res_mix, batch_rows),
            rows.select(residual, batch_rows),
        )
    num_tokens = rows.num_tokens
    return None, tuple(out.new_zeros((num_tokens, *out.shape[1:])).index_copy_(0, batch_rows, out) for out in row_outputs)


def memory_saved_bytes(num_tokens: int, world_size: int, hidden_size: int, hc_mult: int, itemsize: int = 2) -> int:
    """Bytes per rank the split residual carry saves (the [rows, hc_mult, hidden] stream; the mixes are small)."""
    rows = SPRows(num_tokens, world_size, 0)
    return (num_tokens - rows.local_rows) * hc_mult * hidden_size * itemsize

