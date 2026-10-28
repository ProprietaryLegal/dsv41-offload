"""PP3 bounded sparse-q codec; importing it makes no CUDA call.

Negotiated sample_mode=2/capability8, STEP flag2, reply flag2. Semantic1 is
dyadic24-floor-first-v1 with top32/tau.7 and fixed depth4/5. Old peers refuse
HELLO. The original frame version and all OFF bytes remain unchanged.
"""
from __future__ import annotations

import struct
import zlib
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

SAMPLE_TOP32 = 2
CAP_TOP32 = 8
FLAG = 2
META = struct.Struct('<qI4x')  # user seed and target-temperature-positive mask
HEADER = struct.Struct('<4sIII')
MASS = 1 << 24
K = 32
I32: TypeAlias = NDArray[np.int32]
F32: TypeAlias = NDArray[np.float32]


def append_q(body: bytes, mask: I32, ids: I32, q: F32) -> bytes:
    n = len(mask)
    if ids.shape != q.shape or ids.ndim != 3 or ids.shape[0] != n or ids.shape[2] != K:
        raise ValueError('sampled reply tensor shapes')
    data = (HEADER.pack(b'PP3Q', 1, n, ids.shape[1]) + np.asarray(mask, dtype='<i4').tobytes()
            + np.asarray(ids, dtype='<i4').tobytes() + np.asarray(q, dtype='<f4').tobytes())
    return bytes(body + data + struct.pack('<I', zlib.crc32(body + data)))


def unpack_q(body: Any, offset: int, n: int, depth: int) -> tuple[I32, I32, F32]:
    size = HEADER.size + n * 4 + n * depth * K * 8 + 4
    if len(body) != offset + size:
        raise ValueError('sampled reply size mismatch')
    if HEADER.unpack_from(body, offset) != (b'PP3Q', 1, n, depth):
        raise ValueError('sampled reply semantic/shape mismatch')
    if zlib.crc32(body[:-4]) != struct.unpack_from('<I', body, len(body)-4)[0]:
        raise ValueError('sampled reply CRC mismatch')
    off = offset + HEADER.size
    mask = np.frombuffer(body, dtype='<i4', count=n, offset=off).copy()
    off += n * 4
    ids = np.frombuffer(body, dtype='<i4', count=n*depth*K, offset=off).reshape(n, depth, K).copy()
    off += n * depth * K * 4
    q = np.frombuffer(body, dtype='<f4', count=n*depth*K, offset=off).reshape(n, depth, K).copy()
    return mask, ids, q


def validate(mask: I32, ids: I32, q: F32, tokens: I32, needed: NDArray[np.bool_],
             vocab: int, expected_mask: I32 | None = None) -> None:
    n, depth = tokens.shape
    if (mask.shape != (n,) or ids.shape != (n, depth, K) or q.shape != ids.shape
            or depth not in (4, 5) or not np.isin(mask, (0, 1)).all()):
        raise ValueError('invalid sampled q shape/mask')
    if expected_mask is not None and not np.array_equal(mask[needed], expected_mask[needed]):
        raise ValueError('sampled mask does not match target temperature')
    # Every row is scattered on GPU, including zero-q greedy/failed rows.
    if np.any(ids < 0) or np.any(ids >= vocab):
        raise ValueError('q support ID outside target vocabulary')
    for row in range(n):
        if not needed[row] or mask[row] == 0:
            if np.any(q[row] != 0):
                raise ValueError('q present for a greedy/unused row')
            continue
        rids, rq = ids[row], q[row]
        if (np.any(rids < 0) or np.any(rids >= vocab) or np.any(np.diff(rids.astype(np.int64), axis=1) <= 0)
                or not np.isfinite(rq).all() or np.any(rq < 0)):
            raise ValueError('invalid q support or values')
        counts = rq.astype(np.float64) * MASS
        if not np.equal(counts, np.floor(counts)).all() or not np.equal(counts.sum(1), MASS).all():
            raise ValueError('q must contain dyadic24 counts summing exactly to 2^24')
        if not ((rids == tokens[row, :, None]) & (rq > 0)).any(1).all():
            raise ValueError('sampled token missing positive proposal mass')
