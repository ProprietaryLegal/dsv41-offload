"""Round 3 admission estimate, deliberately separate from measured KV identity.

HEADROOM.md cites every observation/allocation. Estimates are not hardware proof;
the first control must reproduce the reserve and a 3 GiB live guard still applies (NDM 2026-10-27:
"Put safety floor to 3gb and promote 16384"; Round 4 replaced the 5 GiB floor).
"""
from __future__ import annotations

from offload.pc1 import build, capacity
from offload.prefill_maxlen.evidence import Json, number

FLOOR_KB = 3 * 1024**2  # NDM 2026-10-27 "Put safety floor to 3gb" (stricter GiB reading)
# Worst unprofiled 150K request minimum across the cited PC1 and SP1 captures.
MEASURED_MIN_KB = {'sparkhead': 6_749_280, 'sparkworker-wired': 7_872_596}
# K1 R1p max measured FFN peak-over-entry (MiB), including startup4096.
# Real4092-row prefills peak875MiB;885 is the larger observed value.
PEAK_MIB = 885
CONTINGENCY_KB = 256 * 1024
STARTUP_EXTRA_S = 180  # ESTIMATED allowance above measured404s, never a measured larger-chunk start


def estimate(arm: str) -> dict[str, Json]:
    result = capacity.expectation(arm)
    chunk = build.SETTINGS[arm][0]
    # Charge an entire measured phase peak per additional4096 rows, plus
    # explicit allocator/metadata contingency; no credit for reclaimed KV.
    extra_kb = max(0, chunk - 4096) * PEAK_MIB * 1024 // 4096
    if arm in build.LARGE_ARMS:
        extra_kb += CONTINGENCY_KB
    predicted: dict[str, Json] = {host: kb - extra_kb for host, kb in MEASURED_MIN_KB.items()}
    return {'classification': 'ESTIMATED from MEASURED 4096/4092-row peaks; not native qualification',
            'expected_kv_tokens': result['expected_tokens'], 'floor_kb': FLOOR_KB,
            'extra_activation_reserve_kb': extra_kb, 'predicted_min_kb': predicted,
            'admissible': number(result['expected_tokens']) >= 327680
                and all(number(kb) >= FLOOR_KB for kb in predicted.values())}


def admit(arm: str) -> None:
    build.admit(arm)
    if arm in build.LARGE_ARMS and estimate(arm)['admissible'] is not True:
        raise ValueError(f'PC1 {arm} refused: derived KV capacity or estimated activation peak breaks 3 GiB floor; see HEADROOM.md')


def control_reserve(arm: str, minima: dict[str, int]) -> None:
    """Use the entire fresh control start, including mixed load/prompt building."""
    admit(arm)
    required = FLOOR_KB + int(number(estimate(arm)['extra_activation_reserve_kb']))
    if set(minima) != set(MEASURED_MIN_KB) or any(kb < required for kb in minima.values()):
        raise ValueError(f'PC1 {arm} fresh control lacks {required} kB activation/floor reserve on both ranks')
