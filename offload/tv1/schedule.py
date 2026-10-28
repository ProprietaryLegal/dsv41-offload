"""Round 4 fixed C/arm/C estimates and admission; no host work on import.

ESTIMATED seconds from MEASURED historical timings, not a TV1 runtime result.
WINDOW.md gives source locators, extrapolations and the three-start reading rule.
"""
from __future__ import annotations

from offload.tv1 import cells, qualification

ROLES = ('off', 'feature', 'off')
WINDOW_S = 85 * 60  # recovery INCLUDED; original 85/86/98 hard ceilings remain
RECOVERY_S = 12 * 60
SETUP_S = 30  # snapshot/staging before first begin
BUILD_S = 30  # prompt cuts built AFTER first begin; reserve them before launch
STARTUP_S = 420  # measured 388–404s plus allowance
IDENTITY_S = 60  # six unchanged K1 references, ESTIMATED
SUPPORT_S = 75  # source/mount/env/stats/trace collection/finish allowance per start
ORACLE_S = 240  # two existing <=120s installed SM121 calls; time is unmeasured


def block_s(feature: str, role: str) -> float:
    if role not in ('off', feature):
        raise ValueError('TV1 role does not belong to feature window')
    traffic = sum(cells.CELLS[n].estimate_s * cells.repetitions(cells.CELLS[n])
                  for n in cells.order(feature))
    extra = qualification.ESTIMATE_S + ORACLE_S if role == 'candidate' else 0
    return STARTUP_S + IDENTITY_S + SUPPORT_S + traffic + extra


def roles(feature: str) -> tuple[str, ...]:
    cells.controls(feature)  # validate before making a schedule
    return tuple(feature if r == 'feature' else r for r in ROLES)


def remaining_s(feature: str, index: int) -> float:
    sequence = roles(feature)
    if not 0 <= index < len(sequence):
        raise ValueError('TV1 start index outside fixed C/arm/C schedule')
    return sum(block_s(feature, role) for role in sequence[index:]) + RECOVERY_S + (BUILD_S if index == 0 else 0)


def total_s(feature: str, *, five: bool = False) -> float:
    sequence = ('off', feature, 'off', feature, 'off') if five else roles(feature)
    return SETUP_S + BUILD_S + sum(block_s(feature, role) for role in sequence) + RECOVERY_S


def admit(feature: str, index: int, *, opened: float, now: float) -> None:
    import math
    if (not math.isfinite(opened) or not math.isfinite(now) or now < opened
            or total_s(feature) > WINDOW_S
            or now + remaining_s(feature, index) > opened + WINDOW_S):
        raise ValueError('TV1 remaining starts plus recovery do not fit HOLD+85; refuse over-budget start')
