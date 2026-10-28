"""Small ops of the DSpark drafter forward, written to the engine's semantics (NOTES.md).

Everything here is plain torch except `markov_bias_rows`, a copy of the serve's Triton kernel
(vllm/models/deepseek_v41/nvidia/ops/markov_cache.py, Apache-2.0) so cached and computed Markov rows have the same
bits as on the Spark's code path. Float formats follow the engine: the hyper-connection stream and every activation
handed between modules are bf16; norms, mixes, RoPE, attention scores and routing are computed in fp32.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------------------------------- RoPE


def rope_cos_sin(max_pos: int, rope_dim: int = 64, base: float = 10000.0, beta_fast: int = 32, beta_slow: int = 1,
                 orig_max_pos: int = 1048576) -> torch.Tensor:
    """fp32 [max_pos, rope_dim] = cat(cos, sin) exactly as vLLM's DeepseekV4ScalingRotaryEmbedding builds it for the
    pure sliding-window layers (common/rope.py: rope_type deepseek_yarn, factor 1.0, mscale 1, original max position =
    max_position_embeddings): the YaRN blend of two equal frequency sets, then an fp32 outer product with the
    positions and cos / sin of each row. Built on the CPU like vLLM's (at module init, before .to(device))."""
    pos_freqs = base ** (torch.arange(0, rope_dim, 2, dtype=torch.float) / rope_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (1.0 * pos_freqs)

    def corr_dim(num_rot: float) -> float:
        return (rope_dim * math.log(orig_max_pos / (num_rot * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(corr_dim(beta_fast)), 0)
    high = min(math.ceil(corr_dim(beta_slow)), rope_dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(rope_dim // 2, dtype=torch.float) - low) / (high - low), 0, 1)
    mask = 1 - ramp
    inv_freq = inv_freq_interpolation * (1 - mask) + inv_freq_extrapolation * mask
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos() * 1.0, freqs.sin() * 1.0), dim=-1)


def apply_rope(x: torch.Tensor, pos: torch.Tensor, cos_sin: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate the last rope_dim channels of x (interleaved pairs, as is_neox_style=False), in fp32, at positions pos.

    x: [N, ..., D] (any float dtype), pos: [N] int64. Returns a new tensor in x's dtype; the NoPE channels are copied."""
    rd = cos_sin.shape[-1]
    cs = cos_sin[pos]                                   # [N, rd]
    cos, sin = cs[:, : rd // 2], cs[:, rd // 2:]
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [rd // 2]
    cos, sin = cos.view(shape), sin.view(shape)
    if inverse:
        sin = -sin
    r = x[..., -rd:].float().unflatten(-1, (rd // 2, 2))
    x0, x1 = r[..., 0], r[..., 1]
    y = torch.stack((x0 * cos - x1 * sin, x1 * cos + x0 * sin), dim=-1).flatten(-2)
    out = x.clone()
    out[..., -rd:] = y.to(x.dtype)
    return out


# --------------------------------------------------------------------------------------------- KV record


def kv_record_qdq(kv: torch.Tensor, nope: int = 448, group: int = 64) -> torch.Tensor:
    """Values of the V4 fp8_ds_mla record (SM12x, 584 B): the NoPE channels as e4m3 with one power-of-two (UE8M0) scale
    per `group` channels, scale = 2^ceil(log2(amax / 448)); the RoPE channels kept bf16. Returns bf16 [.., 512] holding
    exactly the values the engine's attention reads back (e4m3 x 2^k is exact in bf16).

    The ceil(log2) is taken from the IEEE bits of amax * (1/448) (DeepSeek's fast_round_scale, the plugin's
    _qdq_ue8m0 kernel): exact at powers of two, where log2f-then-ceil can round the wrong way."""
    n = kv[..., :nope].float().unflatten(-1, (nope // group, group))
    a = (n.abs().amax(-1, keepdim=True) * (1.0 / 448.0)).clamp_min(2.0 ** -126)
    bits = a.view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale = ((e + 127) << 23).view(torch.float32)
    q = (n / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale
    out = kv.to(torch.bfloat16).clone()
    out[..., :nope] = q.flatten(-2).to(torch.bfloat16)
    return out


# --------------------------------------------------------------------------------------------- norms / mHC


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """DeepSeek reference RMSNorm: fp32 math, weight applied in fp32, one rounding to x's dtype (bf16)."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (w.float() * xf).to(x.dtype)


def hc_mixes(stream: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, rms_eps: float,
             hc_eps: float, iters: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """pre [T, hc], post [T, hc], comb [T, hc, hc] (fp32) from the bf16 stream [T, hc, H] (mhc_pre_delayed_torch)."""
    hc = stream.shape[1]
    x = stream.flatten(1).float()
    mixes = (x @ fn.t()) * torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + hc_eps
    post = torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc]) * 2.0
    comb = mixes[:, 2 * hc:].view(-1, hc, hc) * scale[2] + base[2 * hc:].view(1, hc, hc)
    comb = torch.softmax(comb, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    return pre, post, comb


def hc_collapse(stream: torch.Tensor, pre_mix: torch.Tensor | None) -> torch.Tensor:
    """[T, hc, H] -> [T, H]: copy 0 when pre_mix is None (the first stage), else sum_i pre_i * stream_i in fp32 -> bf16."""
    if pre_mix is None:
        return stream[:, 0].contiguous()
    return (pre_mix.unsqueeze(-1) * stream.float()).sum(dim=1).to(stream.dtype)


def hc_post(y: torch.Tensor, stream: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """new stream_j = post_j * y + sum_i comb[i, j] * stream_i, fp32 -> bf16 (mhc_post_torch)."""
    mixed = torch.einsum("tij,tih->tjh", comb, stream.float())
    return (mixed + post.unsqueeze(-1) * y.float().unsqueeze(1)).to(stream.dtype)


# --------------------------------------------------------------------------------------------- routing


def route(x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor, top_k: int, route_scale: float
          ) -> tuple[torch.Tensor, torch.Tensor]:
    """sqrtsoftplus / noaux_tc: experts by top-k of score + bias, weights = the unbiased scores renormalized x scale.
    gate_w fp32 [E, H]. Returns (ids int64 [T, k], weights fp32 [T, k])."""
    logits = x.float() @ gate_w.t()
    scores = torch.nn.functional.softplus(logits).sqrt()
    ids = (scores + bias).topk(top_k, dim=-1)[1]
    w = scores.gather(1, ids)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20) * route_scale
    return ids, w


# --------------------------------------------------------------------------------------------- attention


def block_attention(q: torch.Tensor, ctx_kv: torch.Tensor, ctx_valid: torch.Tensor, blk_kv: torch.Tensor,
                    sink: torch.Tensor, scale: float) -> torch.Tensor:
    """The DSpark block attention: every block row sees the request's valid context window and the whole block.

    q [B, k, Hh, D] bf16; ctx_kv [B, W, D] bf16 (keys = values, MQA); ctx_valid [B, W] bool; blk_kv [B, k, D] bf16;
    sink [Hh] fp32 (joins the softmax denominator only). fp32 scores; the probabilities are rounded to bf16 for the
    weighted sum (as DeepSeek's sparse_attn kernel and FlashInfer do), the denominator stays fp32; bf16 out."""
    B, k, Hh, D = q.shape
    keys = torch.cat((ctx_kv, blk_kv), dim=1).float()                          # [B, W + k, D] (exact bf16 values)
    s = torch.bmm(q.reshape(B, k * Hh, D).float(), keys.transpose(1, 2)) * scale   # [B, k*Hh, W+k]
    valid = torch.cat((ctx_valid, torch.ones(B, k, dtype=torch.bool, device=q.device)), dim=1)
    s = s.masked_fill(~valid.unsqueeze(1), float("-inf"))
    sk = sink.view(1, 1, Hh).expand(B, k, Hh).reshape(B, k * Hh, 1)
    m = torch.maximum(s.amax(-1, keepdim=True), sk)
    p = torch.exp(s - m)
    denom = p.sum(-1, keepdim=True) + torch.exp(sk - m)
    o = torch.bmm(p.to(torch.bfloat16).float(), keys) / denom
    return o.view(B, k, Hh, D).to(torch.bfloat16)


# --------------------------------------------------------------------------------------------- Markov bias
# Copied from vllm/models/deepseek_v41/nvidia/ops/markov_cache.py (the running image, Apache-2.0): one program per
# (row, 64 vocab columns); cached rows are read back, others computed as an fp32 dot over the rank, stored in the
# output dtype. The cache is filled by this same kernel, so a token's bias has the same bits either way.

_BLOCK_V = 64


@triton.jit
def _markov_bias_kernel(ids_ptr, slot_ptr, cache_ptr, emb_ptr, w2_ptr, out_ptr, V, cache_stride, w2_stride,
                        emb_stride, out_stride, R: tl.constexpr, BLOCK_V: tl.constexpr):
    b = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = cols < V
    tok = tl.load(ids_ptr + b).to(tl.int64)
    slot = tl.load(slot_ptr + tok)
    if slot >= 0:
        row = tl.load(cache_ptr + slot.to(tl.int64) * cache_stride + cols, mask=mask)
        tl.store(out_ptr + b * out_stride + cols, row, mask=mask)
    else:
        r = tl.arange(0, R)
        e = tl.load(emb_ptr + b * emb_stride + r).to(tl.float32)
        w = tl.load(w2_ptr + cols[:, None].to(tl.int64) * w2_stride + r[None, :], mask=mask[:, None], other=0.0)
        acc = tl.sum(w.to(tl.float32) * e[None, :], axis=1)
        tl.store(out_ptr + b * out_stride + cols, acc.to(out_ptr.dtype.element_ty), mask=mask)


def markov_bias_rows(token_ids: torch.Tensor, slot_of_token: torch.Tensor, cache: torch.Tensor,
                     markov_embed: torch.Tensor, w2: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """out[b] = cache[slot_of_token[t_b]] if cached, else w2 @ markov_embed[b] (fp32 dot, one rounding)."""
    num_rows, rank = markov_embed.shape
    vocab = w2.shape[0]
    assert w2.stride(1) == 1 and markov_embed.stride(1) == 1 and out.stride(1) == 1
    if num_rows == 0:
        return out
    grid = (num_rows, triton.cdiv(vocab, _BLOCK_V))
    _markov_bias_kernel[grid](token_ids, slot_of_token, cache, markov_embed, w2, out, vocab,
                              cache.stride(0) if cache.numel() else 0, w2.stride(0), markov_embed.stride(0),
                              out.stride(0), R=rank, BLOCK_V=_BLOCK_V, num_warps=4)
    return out


class MarkovBias:
    """The Markov bias with cached rows for the serve's most frequent tokens (VLLM_DSPARK_MARKOV_CACHE list, first
    `rows` distinct ids, default 4096 like VLLM_DSPARK_MARKOV_CACHE_ROWS)."""

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, token_list: list[int] | None, rows: int = 4096):
        self.w1, self.w2 = w1, w2
        vocab, dev = w1.shape[0], w2.device
        ids: list[int] = []
        seen: set[int] = set()
        for t in token_list or []:
            t = int(t)
            if 0 <= t < vocab and t not in seen:
                seen.add(t)
                ids.append(t)
        ids = ids[:rows]
        self.slot_of_token = torch.full((vocab,), -1, dtype=torch.int32, device=dev)
        self.cache = torch.empty((max(len(ids), 1), w2.shape[0]), dtype=torch.bfloat16, device=dev)
        if ids:
            tok = torch.tensor(ids, dtype=torch.int64, device=dev)
            for s in range(0, len(ids), 256):
                t = tok[s: s + 256]
                markov_bias_rows(t, self.slot_of_token, self.cache, w1[t].contiguous(), w2, self.cache[s: s + 256])
            self.slot_of_token[tok] = torch.arange(len(ids), dtype=torch.int32, device=dev)
        self.cached_rows = len(ids)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.w1[token_ids]

    def __call__(self, token_ids: torch.Tensor, markov_embed: torch.Tensor, out: torch.Tensor | None = None
                 ) -> torch.Tensor:
        if out is None:
            out = torch.empty((markov_embed.shape[0], self.w2.shape[0]), dtype=torch.bfloat16,
                              device=markov_embed.device)
        return markov_bias_rows(token_ids, self.slot_of_token, self.cache, markov_embed.contiguous(), self.w2, out)
