"""Exact decoders for the official 1017 checkpoint formats and the official activation quantize-dequantize (QDQ).

Formats (inference/convert.py, kernel.py @ 1bae0e10):
  MXFP4 E2M1 weight  safetensors I8 [out, in/2]; byte j holds input 2j in the LOW nibble and 2j+1 in the HIGH nibble;
                     code -> value FP4_TABLE (sign bit 3; 0b1000 is -0.0)
  E8M0 scale         safetensors F8_E8M0, one byte b -> 2^(b-127); b = 255 is NaN (refused)
                     FP4 weights: [out, in/32] (one scale per 32 inputs of one output row)
                     FP8 weights: [ceil(out/32), ceil(in/32)] (one scale per 32x32 block)
  FP8 E4M3 weight    safetensors F8_E4M3 (e4m3fn: bias 7, no inf, 0x7F/0xFF NaN, max 448)
Every decoded value code * 2^(b-127) is computed exactly in float32 (codes have <= 4 significant bits, scales are
powers of two); decode_* refuse a NaN code or scale instead of returning one.

act_qdq reproduces kernel.act_quant(x, 32, "ue8m0", e8m0[, inplace=True]) bit for bit: per 32 consecutive elements of
the last dim, amax = max(max|x|, 1e-4) in fp32, s = 2^ceil(log2(fp32(amax * fp32(1/448)))) by exponent bits
(fast_round_scale), q = e4m3(clamp(x / s, -448, 448)) (round to nearest even), value = fp32(q) * s, stored in x's dtype
(bf16). The QDQ'd bf16 value is exactly q * s (4 significant bits, power-of-two scale), so a bf16 GEMM on QDQ'd
activations and exactly decoded weights forms the same products the official FP8 x FP8 / FP8 x FP4 kernels form.
"""
from __future__ import annotations

import torch

FP4_TABLE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
FP8_MAX = 448.0
BLOCK = 32
E8M0_NAN = 255
E4M3_NAN = (0x7F, 0xFF)


def _fp4_lut(device) -> torch.Tensor:
    return torch.tensor(FP4_TABLE, dtype=torch.float32, device=device)


def e8m0_exponents(scale_u8: torch.Tensor) -> torch.Tensor:
    """int32 exponents b - 127; raises on a NaN (255) scale."""
    s = scale_u8.view(torch.uint8)
    if bool((s == E8M0_NAN).any()):
        raise ValueError(f"E8M0 scale holds {(s == E8M0_NAN).sum().item()} NaN bytes (255)")
    return s.to(torch.int32) - 127


def e8m0_to_f32(scale_u8: torch.Tensor) -> torch.Tensor:
    """2^(b-127) as float32 (exact for b in 1..254; b = 0 is 2^-127, an fp32 subnormal, still exact)."""
    return torch.ldexp(torch.ones((), dtype=torch.float32, device=scale_u8.device), e8m0_exponents(scale_u8))


def fp4_codes(packed: torch.Tensor) -> torch.Tensor:
    """uint8 codes [out, in] from packed bytes [out, in/2] (low nibble = even input)."""
    p = packed.view(torch.uint8)
    return torch.stack((p & 0x0F, p >> 4), dim=-1).flatten(-2)


def decode_fp4(packed: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    """float32 [out, in] = FP4_TABLE[code] * 2^(scale-127), scale [out, in/32]; exact."""
    codes = fp4_codes(packed)
    out, inn = codes.shape[-2], codes.shape[-1]
    if scale_u8.shape[-2:] != (out, inn // BLOCK) or inn % BLOCK:
        raise ValueError(f"fp4 {tuple(codes.shape)} vs scale {tuple(scale_u8.shape)}")
    v = _fp4_lut(packed.device)[codes.long()].view(*codes.shape[:-1], inn // BLOCK, BLOCK)
    return (v * e8m0_to_f32(scale_u8).unsqueeze(-1)).flatten(-2)


def e4m3_codes_to_f32(w_u8: torch.Tensor) -> torch.Tensor:
    w = w_u8.view(torch.uint8)
    if bool(((w == E4M3_NAN[0]) | (w == E4M3_NAN[1])).any()):
        raise ValueError("FP8 E4M3 weight holds NaN codes")
    return w.view(torch.float8_e4m3fn).to(torch.float32)


def decode_fp8_block(w_u8: torch.Tensor, scale_u8: torch.Tensor, rows: tuple[int, int] | None = None,
                     cols: tuple[int, int] | None = None) -> torch.Tensor:
    """float32 [out, in] = e4m3(code) * 2^(scale-127) per 32x32 block; exact. ``rows`` / ``cols``: the block-aligned
    window of the FULL tensor that ``w_u8`` holds (shards keep the full tensor's scale blocks)."""
    out, inn = w_u8.shape
    r0 = rows[0] if rows else 0
    c0 = cols[0] if cols else 0
    if r0 % BLOCK or c0 % BLOCK:
        raise ValueError(f"fp8 shard at ({r0}, {c0}) is not on a 32x32 scale block")
    s = e8m0_to_f32(scale_u8)
    s = s[r0 // BLOCK:(r0 + out + BLOCK - 1) // BLOCK, c0 // BLOCK:(c0 + inn + BLOCK - 1) // BLOCK]
    s = s.repeat_interleave(BLOCK, 0)[:out].repeat_interleave(BLOCK, 1)[:, :inn]
    return e4m3_codes_to_f32(w_u8) * s


def _log2_ceil_pow2(t: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(t)) for positive normal fp32 t by exponent bits (kernel.fast_log2_ceil + fast_pow2)."""
    bits = t.view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return ((e + 127) << 23).view(torch.float32)


def act_scale(x: torch.Tensor) -> torch.Tensor:
    """Official per-32 power-of-two activation scales, float32 [..., K/32]."""
    K = x.shape[-1]
    if K % BLOCK:
        raise ValueError(f"act_qdq: last dim {K} not a multiple of {BLOCK}")
    xb = x.float().unflatten(-1, (K // BLOCK, BLOCK))
    amax = xb.abs().amax(-1).clamp_min(1e-4)
    inv = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32, device=x.device)       # fp32(1/448), as the kernel
    return _log2_ceil_pow2((amax * inv).contiguous())


def act_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(e4m3 codes as float32 [..., K], scales float32 [..., K/32]) of the official activation quantization."""
    s = act_scale(x)
    xb = x.float().unflatten(-1, (x.shape[-1] // BLOCK, BLOCK))
    q = (xb / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(torch.float32)
    return q.flatten(-2), s


def act_qdq(x: torch.Tensor) -> torch.Tensor:
    """kernel.act_quant(x, 32, 'ue8m0', e8m0, inplace=True) as a new tensor of x's dtype."""
    q, s = act_quant(x)
    v = q.unflatten(-1, (x.shape[-1] // BLOCK, BLOCK)) * s.unsqueeze(-1)
    return v.flatten(-2).to(x.dtype)


def exact_in(v: torch.Tensor, dtype: torch.dtype) -> bool:
    """Every float32 value of ``v`` survives a round trip through ``dtype`` unchanged (no rounding, no overflow)."""
    r = v.to(dtype).to(torch.float32)
    return bool(torch.equal(r, v)) and bool(torch.isfinite(r).all())
