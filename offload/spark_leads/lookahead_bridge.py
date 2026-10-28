# SPDX-License-Identifier: Apache-2.0
"""0078 remote adapter draft. Hash arithmetic ported from HeNryous afe4d5e; no reader/thread created."""
from __future__ import annotations
import os
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import TYPE_CHECKING, Sequence
if TYPE_CHECKING:
    import torch
    from vllm.models.deepseek_v41.common.engram import EngramLayout
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState
IMAGE_SENTINEL_ID = 129264
TOKENS = int(os.environ.get("VLLM_X1_ENGRAM_LOOKAHEAD", "0"))
if not 0 <= TOKENS <= 512:
    raise ValueError("VLLM_X1_ENGRAM_LOOKAHEAD must be 0..512")

class HashTables:
    """Host copies of NgramHashState's hash parameters (built once at model init when the lookahead is on)."""

    def __init__(self, token_map: ArrayLike, multipliers: ArrayLike, primes: ArrayLike, offsets: ArrayLike, pad_id: int, dead_token_id: int = IMAGE_SENTINEL_ID) -> None:
        # own copies: never a view on the model's buffers
        self.token_map = np.array(token_map, dtype=np.int64)  # [vocab]
        self.multipliers = np.array(multipliers, dtype=np.int64)  # [layers, max_ngram]
        self.primes = np.array(primes, dtype=np.int64)  # [layers, max_ngram - 1, heads]
        self.offsets = np.array(offsets, dtype=np.int64)  # [layers, (max_ngram - 1) * heads]
        self.pad_id = int(pad_id)
        self.dead_token_id = int(dead_token_id)
        self.num_layers, self.max_ngram = self.multipliers.shape
        self.num_heads = self.primes.shape[-1]

    @classmethod
    def build(cls, token_map: Sequence[int], layout: EngramLayout, multipliers: torch.Tensor, pad_id: int) -> "HashTables":
        """From NgramHashState's inputs: the token map (list), the EngramLayout and the multiplier tensor."""
        primes = np.asarray(layout.primes, dtype=np.int64)
        flat = primes.reshape(primes.shape[0], -1)
        # EngramLayout.offsets: per layer np.cumsum([0, *sizes[:-1]]) over the flattened (n-gram, head) primes
        offsets = np.concatenate(
            [np.zeros((flat.shape[0], 1), np.int64), np.cumsum(flat, axis=1)[:, :-1]], axis=1
        )
        mult = multipliers.detach().to("cpu").numpy() if hasattr(multipliers, "detach") else multipliers
        return cls(token_map, mult, primes, offsets, pad_id)


def segment_start(tables: HashTables, start: int) -> int:
    """First history position the hashes of positions >= start read."""
    return max(0, start - (tables.max_ngram - 1))


def host_hashes(tables: HashTables, seg: NDArray[np.int64], start: int, end: int) -> NDArray[np.int64]:
    """Hash ids [end - start, layers, (max_ngram - 1) * heads] (int64) of positions start..end-1 of one request.

    `seg` holds the request's token ids of positions segment_start(start) .. end-1. Same arithmetic as engram.py's
    _hash_ids_kernel with the V2 runner's inputs (no slot cache; the chunk and the lookback window both come from the
    token history): position p hashes ids p, p-1, .., p-max_ngram+1; a negative position or an image-span token blocks
    that n-gram and every longer one (pad id); n-gram n of head h is
    (xor over s < n of compressed_id(p - s) * multiplier[layer, s]) mod prime[layer, n-2, h] + offset[layer, n-2, h].
    """
    t = tables
    m, heads = t.max_ngram, t.num_heads
    n = end - start
    lo = segment_start(t, start)
    seg = np.asarray(seg, dtype=np.int64)
    assert seg.shape == (end - lo,), "token segment does not cover the hashed positions"
    comp = t.token_map[seg]
    dead = seg == t.dead_token_id
    pos = np.arange(start, end, dtype=np.int64)
    out = np.empty((n, t.num_layers, (m - 1) * heads), dtype=np.int64)
    for layer in range(t.num_layers):
        rolling = np.zeros(n, dtype=np.int64)
        blocked = np.zeros(n, dtype=bool)
        for shift in range(m):
            q = pos - shift
            inside = q >= 0
            idx = np.where(inside, q - lo, 0)
            blocked |= ~inside | dead[idx]
            value = np.where(blocked, t.pad_id, comp[idx])
            rolling ^= value * t.multipliers[layer, shift]
            if shift > 0:
                cols = slice((shift - 1) * heads, shift * heads)
                out[:, layer, cols] = rolling[:, None] % t.primes[layer, shift - 1][None, :] + t.offsets[layer, cols][None, :]
    return out



def after_forward(model: torch.nn.Module, input_batch: InputBatch, req_states: RequestState) -> None:
    # Called on every non-dummy step, including decode-only/finish, so empty snapshots cancel hints.
    if not TOKENS:
        return
    ht = next((getattr(mod, "_x1_host") for mod in model.modules() if hasattr(mod, "_x1_host")), None)
    if ht is None:
        raise RuntimeError("X1 lookahead enabled without host hash tables")
    buf = getattr(getattr(req_states, "all_token_ids", None), "_uva_buf", None)
    if buf is None:
        raise RuntimeError("X1 lookahead requires host-visible prompt token history")
    # Copy prompt-only segments NOW, while this engine step owns request slots. No deferred UVA view.
    work = []
    for i in range(input_batch.num_reqs):
        if not input_batch.is_prefilling_np[i]:
            continue
        slot = int(input_batch.idx_mapping_np[i])
        start = int(input_batch.num_computed_prefill_tokens_np[i]) + int(input_batch.num_scheduled_tokens[i])
        end = min(int(input_batch.prefill_len_np[i]), start + TOKENS)
        if end <= start:
            continue
        seg = np.array(buf.np[slot, segment_start(ht, start):end], dtype=np.int64, copy=True)
        if (seg < 0).any() or (seg >= len(ht.token_map)).any():
            raise ValueError("X1 lookahead token outside vocabulary")
        work.append(host_hashes(ht, seg, start, end))
    for mod in model.modules():
        emb = getattr(mod, "embed_tokens", None)
        if not hasattr(mod, "layer_hash_index") or getattr(emb, "_remote_endpoint", None) is None:
            continue
        if emb.dp_size != 1:
            raise RuntimeError("X1 lookahead is TP2-only; DP table sharing unsupported")
        layer = int(mod.layer_hash_index)
        arrays = [h[:, layer, emb.head_start:min(emb.head_start+emb.part_n_hash_cols,emb.n_hash_cols)].reshape(-1)
                  - emb.vocab_start_idx for h in work]
        rows = np.concatenate(arrays) if arrays else np.empty(0,dtype=np.int64)
        # Bound before queue publication. Dead/padded heads are left to the exact foreground lookup.
        rows = rows[(rows >= 0) & (rows < emb.part_num_embeddings)]
        rows = np.unique(rows)[:8192]
        emb._disk.replace_lookahead(rows.tolist(), layer)
