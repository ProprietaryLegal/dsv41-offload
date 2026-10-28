"""Fixed production K1 pool, with arm-specific SWA admission accounting.

Transcribed from the checked allocator, not fitted to the observed token count.
See CAPACITY.md for source locators and CPU execution of the actual functions.
"""
from __future__ import annotations

from offload.pc1 import build
from offload.prefill_maxlen.evidence import Json


def cdiv(value: int, divisor: int) -> int:
    return -(-value // divisor)


def expectation(arm: str) -> dict[str, Json]:
    build.admit(arm)
    chunk, _ = build.SETTINGS[arm]
    kv_bytes, max_len = 6_442_450_944, 327_680
    # K1 ratio-2 group: three (64-state MLA + indexer) pages; ratio-1
    # is packed into three kernel pages (192 tokens), and does not widen it.
    stride = 3 * (37_440 + 8_640)
    pool_blocks = kv_bytes // stride
    # Async scheduling, PP1: two batches can hold unsettled SWA blocks.
    in_flight = 2 * chunk
    swa_blocks = cdiv(min(128 - 1 + in_flight, max_len), 64) + 1
    swa_groups = cdiv(40, 3)
    required = cdiv(max_len, 128) + cdiv(max_len, 192) + 1 + swa_groups * swa_blocks
    # Match get_kv_cache_capacity's float division then int truncation exactly.
    tokens = int(pool_blocks / required * max_len)
    return {'expected_tokens': tokens, 'kv_bytes': kv_bytes, 'max_model_len': max_len,
            'max_num_batched_tokens': chunk, 'pool_stride_bytes': stride,
            'pool_blocks': pool_blocks, 'in_flight_tokens': in_flight,
            'swa_blocks_per_group': swa_blocks, 'swa_groups': swa_groups,
            'blocks_per_request': required}
