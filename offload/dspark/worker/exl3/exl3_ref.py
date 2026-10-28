# SPDX-License-Identifier: Apache-2.0
"""Plain-torch reference of an EXL3 linear (mul1 codebook), written from exllamav3 v1.6.0's CUDA (MIT, (c) 2025
Turboderp): exl3_dq.cuh (bitstream windows), codebook.cuh (mul1 decode), reconstruct.cu (tile layout). The custom
decode-size kernel is checked against this, and this is checked bit-exact against exllamav3's own reconstruct.

Layout: trellis int16 [in/16, out/16, 16*K], one 16x16 tile per (k_tile, n_tile). A tile is a cyclic bitstream of 256*K
bits held as 8*K uint32 words (uint16 pairs, little-endian: word j = u16[2j+1] << 16 | u16[2j]), bits numbered from the
MSB of word 0. Weight t (0..255) reads the 16-bit window that ends at bit (t+1)*K (mod 256*K). mul1: x = window *
0x83DCD12D (mod 2^32), s = sum of x's four bytes, w = fp16(fp16(1024 + s) * fp16(0x1eee) + fp16(0xc931)) (one fused
multiply-add, so one rounding: a 1021-entry table computed in fp64). Inside the tile, weight t lands at
    l = t // 8, i = t % 8, g = l // 8, h = (l % 8) // 4, q = l % 4
    row (input)  = 2q + i % 2 + 8 * ((i // 2) % 2)
    col (output) = 2g + h + 8 * (i // 4)
The linear: y = H(H(x * suh) @ W) * svh, H = right-multiplication by the normalised 128x128 Hadamard per 128 block.
"""

from __future__ import annotations

import math

import torch

MUL1 = 0x83DCD12D


def mul1_table(device=None) -> torch.Tensor:
    """fp16 weight for every byte sum 0..1020 (exllamav3's decode_mul1_product_2, one rounding like its hfma2)."""
    k_inv = torch.tensor([0x1EEE], dtype=torch.int16).view(torch.float16).double()
    k_bias = torch.tensor([0xC931 - 65536], dtype=torch.int16).view(torch.float16).double()
    s = torch.arange(1021, dtype=torch.float64)
    return ((1024.0 + s) * k_inv + k_bias).to(torch.float16).to(device)


def tile_perm() -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.arange(256)
    l, i = t // 8, t % 8
    g, h, q = l // 8, (l % 8) // 4, l % 4
    return 2 * q + i % 2 + 8 * ((i // 2) % 2), 2 * g + h + 8 * (i // 4)


def decode(trellis: torch.Tensor, K: int) -> torch.Tensor:
    """Rotated-basis fp16 weight [in, out] from an int16 trellis [in/16, out/16, 16*K]."""
    kt, nt, words = trellis.shape
    assert words == 16 * K
    u16 = trellis.reshape(kt * nt, 8 * K, 2).to(torch.int64) & 0xFFFF
    w32 = (u16[..., 1] << 16) | u16[..., 0]                          # [tiles, 8K] uint32 values in int64
    t = torch.arange(256, device=trellis.device)
    b0 = t * K + K - 16 + 256 * K
    b1 = b0 + 16
    i0, i1 = b0 // 32, (b1 - 1) // 32
    s0 = (i1 + 1) * 32 - b1
    a = w32[:, i0 % (8 * K)]
    b = w32[:, i1 % (8 * K)]
    state = (((a << 32) | b) >> s0) & 0xFFFF                          # [tiles, 256]
    x = (state * MUL1) & 0xFFFFFFFF
    bsum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
    vals = mul1_table(trellis.device)[bsum]                           # [tiles, 256] fp16
    r, c = tile_perm()
    tile = torch.empty((kt * nt, 16, 16), dtype=torch.float16, device=trellis.device)
    tile[:, r.to(trellis.device), c.to(trellis.device)] = vals
    return tile.view(kt, nt, 16, 16).permute(0, 2, 1, 3).reshape(kt * 16, nt * 16)


def hadamard128(device=None, dtype=torch.float32) -> torch.Tensor:
    h = torch.ones((1, 1), dtype=torch.float64)
    while h.shape[0] < 128:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(128)).to(device=device, dtype=dtype)


def had_r(x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """x [m, n] (n % 128 == 0) times the block-diagonal Hadamard, per 128-wide block."""
    m, n = x.shape
    return (x.view(m, n // 128, 128) @ H).view(m, n)


def linear(x: torch.Tensor, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, K: int) -> torch.Tensor:
    """fp32 reference of y = H(H(x * suh) @ W) * svh."""
    H = hadamard128(x.device)
    xh = had_r(x.float() * suh.float(), H)
    y = xh @ decode(trellis, K).float()
    return had_r(y, H) * svh.float()
