"""R-UPF windows (offload/upfix 42fe86a) as UP3 runner groups (coordinator 2026-10-27). The design is imported, never
copied: arms are configs.ARMS environments of configs.CONFIG (UPF.env), starts and steps are windows.WINDOWS, the verdict is
windows.arm_verdict over decision_v2.decide_window, soak growth is windows.soak_ok and archive rows are windows.archive_rows.

Coordinator decisions 2026-10-27 (rule 27): no combined UPF-W1 window. NDM's standing ceiling is production restored by
HOLD+132 and UPF-W1 needs about 165 min, so admit() refuses it before any host access. UPF runs as its two fallback designs,
each a SINGLE window (no chain until WIN2-R3): upf-topk first, then upf-0091, off-hours only. Each group's HOLD bounds are
sized from its own budget (bounds()).

Runner mapping (the UP3 per-start order is start, greedy, OPS1 warm-up, gates, cells, finish):
- warm-up: UP3's OPS1 production warm-up stands in for the catalogue's 64-token warm-up cell (same position, after start).
- greedy: offload.up3.upf_driver greedy (nbw_cells.greedy at 0.9+1e-3 nats, distributions recorded, registration upf-0p9).
  C1 must be 6/6 IDENTICAL to the K1 reference or the window STOPs (the base drifted). C2's greedy step is the KL re-probe
  of each candidate's recorded divergences (greedy_kl.compare_recorded) at max(0.005, 2 x the C1/C2 A/A floor): C1 is
  6/6 IDENTICAL by gate, so no A/A rows exist and the bound is the 0.005 floor.
- needles: the gates step (qualify); soak: after the cells (finish), where the design puts it.
- T0-T2: after the HOLD and the production snapshot, before C1 (Live.upf_pre).
- Identity is relative to C1 (coordinator 2026-10-27): UPF.env starts from D-kv6-async-pack.env, not production's
  dsv41.env, so C1-vs-production is RECORDED as a diff (same image, model target files and engine required); every
  later start must equal C1 plus exactly its arm's documented mounts and environment.
- Arm-level stops (a candidate's failed gate, a failed or timed-out T1/T2, a refused verify-topk) skip or stop that arm
  only; a control's failure, or any infrastructure failure, stops the window.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import decision_v2 as dv2
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, record
from offload.up3 import build as b
from offload.up3 import cells
from offload.up3 import win2_plan
from offload.upfix import configs as uc
from offload.upfix import windows as uw

GROUPS = {'upf-topk': 'UPF-TOPK', 'upf-0091': 'UPF-0091', 'upf-w1': 'UPF-W1'}
QUEUE = ('upf-topk', 'upf-0091')   # coordinator order, each a single window; upf-w1 is refused (ceiling)
REGISTRATION = 'upf-0p9'
CONTROL = 'C-k1'
TOPK = 'topk-fix'
CELLS = uw.DECISION_CELLS
GATE_KINDS = ('needle-150k', 'needle-300k')       # qualify, in design order
CEILING_S = 132 * 60       # NDM's standing ceiling: production restored by HOLD+132 (coordinator 2026-10-27)
RECOVERY_S = 12 * 60       # strict W8 restore reserve after the submit bound (>= 10 min; MEASURED W8 7.4 min)
MARGIN = 1.15              # submit/abort = ceil(1.15 x the design's in-HOLD estimate) minutes: 108/120 for UPF-TOPK
FINISH_S = win2_plan.FINISH_S
T_TIMEOUT_S = uw.T2_TIMEOUT_S + 30     # timeout -k 30 600 around T1's and T2's docker run
assert set(cells.WC_CELLS) == set(CELLS), 'UP3 cells.py must carry the R-UPF decision cells'
assert all(wc.CELLS[c].batch == cells.batch_for(c) for c in CELLS)


def window_for(group: str) -> uw.Window:
    ck.require(group in GROUPS, f'{group}: not an R-UPF group')
    window = uw.WINDOWS[GROUPS[group]]
    uw.validate(window)
    return window


def starts(group: str) -> tuple[uw.Start, ...]:
    return window_for(group).starts


def roles(group: str) -> tuple[str, ...]:
    return tuple(s.arm for s in starts(group))


def kinds(group: str, position: int) -> tuple[str, ...]:
    return tuple(step.kind for step in starts(group)[position].steps)


def pre(group: str) -> tuple[str, ...]:
    return tuple(step.kind for step in window_for(group).pre)


def gates(group: str, position: int) -> tuple[str, ...]:
    return tuple(k for k in kinds(group, position) if k in GATE_KINDS)


def repeats(group: str, position: int, cell: str) -> int:
    found = [s.repeats for s in starts(group)[position].steps if isinstance(s, uw.CellStep) and s.cell == cell]
    ck.require(len(found) == 1, f'{group}/{position}: {cell} not exactly once in the design')
    return found[0]


def is_control(role: str) -> bool:
    return role == CONTROL


def hold_s(group: str) -> float:
    """The design's in-HOLD estimate (fences, T0-T2, every start), without the W8 restore or the watch."""
    return window_for(group).hold_s(last=False)


def bounds(group: str) -> tuple[int, int, int]:
    """(submit, abort, restore) seconds after the HOLD publication, sized from this group's own budget."""
    submit = math.ceil(MARGIN * hold_s(group) / 60) * 60
    return submit, submit, submit + RECOVERY_S


def admit(group: str) -> None:
    """Refused BEFORE any host access: unknown group, or a window that cannot restore production by HOLD+132."""
    restore = bounds(group)[2]
    ck.require(restore <= CEILING_S,
               f'{GROUPS[group]} needs production restored by HOLD+{restore // 60} > the HOLD+{CEILING_S // 60} ceiling '
               f'(NDM); run {" then ".join(QUEUE)} as single windows (coordinator 2026-10-27)')


def block_s(group: str, position: int) -> int:
    """One start: the design's MEASURED arm start + its ESTIMATED steps (catalogue cell bounds) + finish."""
    start = starts(group)[position]
    return math.ceil(uw.START_S + sum(step.seconds() for step in start.steps) + FINISH_S)


def pre_s(group: str) -> int:
    return math.ceil(sum(step.seconds() for step in window_for(group).pre))


def budget_s(group: str) -> int:
    """Fences + pre-steps + every start + the W8 recovery reserve (what the restore bound must cover)."""
    return math.ceil(uw.FENCES_S + pre_s(group) + sum(block_s(group, i) for i in range(len(starts(group)))) + RECOVERY_S)


def gate_s(group: str, position: int) -> int:
    return math.ceil(sum(uw.GATE_S[k] for k in gates(group, position)))


def env_delta(role: str) -> dict[str, str]:
    """Container environment an arm adds to C1 (UPF.env: only UPF_TOPK=1 adds environment)."""
    return dict(uc.TOPK_ENV) if uc.ARMS[role]['UPF_TOPK'] == '1' else {}


def mount_delta(role: str) -> dict[str, str]:
    """dist-packages mounts an arm adds to C1, with the sha256 of the reviewed source (the library is pinned separately
    from its build receipt; capture_window_identity hashes only dist-packages, /opt/dsv41-ofl and /serve.sh)."""
    return {target: hashlib.sha256((b.ROOT / source).read_bytes()).hexdigest()
            for source, target in uc.added(uc.ARMS[role]) if target.startswith(b.DIST)}


def image_bases(group: str) -> dict[str, str]:
    """Container path -> image base sha256 of every overlay this group's arms mount (MANIFEST-0091 image variant,
    MANIFEST-topk overlay): the S93 runner note's in-image base check."""
    used = [uc.ARMS[r] for r in set(roles(group))]
    out: dict[str, str] = {}
    if any(arm['UPF_0091'] == '1' for arm in used):
        for row in json.loads((b.ROOT / 'offload/upfix/MANIFEST-0091.json').read_text())['files']:
            if row['variant'] == 'image':
                out[str(row['container_path'])] = str(row['base_sha256'])
    if any(arm['UPF_TOPK'] == '1' for arm in used):
        overlay = json.loads((b.ROOT / 'offload/upfix/topk/MANIFEST-topk.json').read_text())['overlay']
        out[str(overlay['container_path'])] = str(overlay['base_sha256'])
    ck.require(bool(out) and all(k.startswith(b.DIST) for k in out), f'{group}: no UPF overlay bases')
    return out


def start_env(role: str) -> str:
    return ' '.join(f'{k}={v}' for k, v in uc.ARMS[role].items())


def control_values(row: Mapping[str, Json]) -> dict[tuple[str, str], list[float]]:
    """Every repeat of each decision-cell metric of one COMPLETE start (pooled per repeat, win2_plan.repeat_values)."""
    measured = record(row['cells'])
    return {(cell, metric): win2_plan.repeat_values(record(measured[cell]), metric)
            for cell in CELLS for metric, _ in uw.METRICS[cell]}


def _gates(label: str, role: str, starts_: Mapping[str, Json], pre_: Mapping[str, Json],
           c2: Mapping[str, Json]) -> dict[str, bool | None]:
    """Each required gate of a candidate: True / False, or None when it never ran. An arm-level STOP names its gate."""
    row = record(starts_.get(label, {}))
    out: dict[str, bool | None] = {}
    for gate in uw.REQUIRED_GATES[role]:
        if row.get('failed_gate') == gate:
            out[gate] = False
        elif gate == 'bitcompare':
            out[gate] = record(pre_.get('topk', {})).get('status') == 'READY'
        elif gate == 'greedy':
            greedy = record(row.get('greedy', {}))
            kl = record(record(c2.get('kl', {})).get(label, {}))
            out[gate] = None if not greedy or not kl else (greedy.get('pass') is True and kl.get('pass') is True)
        elif gate == 'soak':
            soak = record(record(row.get('checks', {})).get('soak', {}))
            out[gate] = None if 'ok' not in soak else soak['ok'] is True
        else:   # a needle STOP carries the needles run so far as its gate evidence
            needles = record(row.get('gates', {})).get('needles') or record(row.get('gate_evidence', {})).get('needles', {})
            needle = record(record(needles).get(gate, {}))
            out[gate] = None if 'pass' not in needle else needle['pass'] is True
    return out


def decide(group: str, starts_: Mapping[str, Json], pre_: Mapping[str, Json], archive: Path,
           production: str) -> dict[str, Json]:
    """Identity / no-loss verdict per candidate (windows.arm_verdict, rule win2-v1, decide_window per candidate with the
    controls C1 + C2 pooled, unpaired)."""
    design = starts(group)
    labels = [s.label for s in design]
    controls = [s.label for s in design if is_control(s.arm)]
    base: dict[str, Json] = {'rule': dv2.RULE, 'design': GROUPS[group], 'registration': REGISTRATION,
                             'promotion': 'none; a PROMOTABLE (no loss) arm goes to independent review'}
    if list(starts_) != labels or any(record(starts_[c]).get('status') != 'COMPLETE' for c in controls):
        return {**base, 'status': 'NOT A CANDIDATE', 'reason': 'both control starts must be COMPLETE'}
    done = [label for label in labels if record(starts_[label]).get('status') == 'COMPLETE']
    caps = {label: win2_plan.clock_caps(starts_, label) for label in done}
    if any(len(c) != 1 for c in caps.values()) or len({c[0] for c in caps.values()}) != 1:
        return {**base, 'status': 'NOT A CANDIDATE', 'reason': 'clock cap unproved or changed between starts',
                'clock_caps_mhz': cast(Json, caps)}
    cap = caps[controls[0]][0]
    control = [control_values(record(starts_[c])) for c in controls]
    c2 = record(record(starts_[controls[-1]]).get('greedy', {}))
    arms: dict[str, Json] = {}
    for start in design:
        if is_control(start.arm):
            continue
        row = record(starts_[start.label])
        gates_ = _gates(start.label, start.arm, starts_, pre_, c2)
        entry: dict[str, Json] = {'arm': start.arm, 'status': row.get('status'), 'gates': cast(Json, gates_)}
        if row.get('status') == 'SKIPPED':
            entry['verdict'] = f'SKIPPED: {row.get("reason")}'
        elif row.get('status') != 'COMPLETE' or any(v is not True for v in gates_.values()):
            failed = sorted(g for g, v in gates_.items() if v is False)
            unrun = sorted(g for g, v in gates_.items() if v is None)
            entry['verdict'] = 'STOP: ' + (', '.join(failed) or str(row.get('stop', 'incomplete'))) + (
                f' (not run: {", ".join(unrun)})' if unrun else '')
        else:
            candidate = control_values(row)
            series = [uw.series(cell, metric, control[0][(cell, metric)] + control[1][(cell, metric)],
                                candidate[(cell, metric)]) for cell in CELLS for metric, _ in uw.METRICS[cell]]
            decision = dv2.decide_window(series, CELLS, groups=uw.GROUPS)
            entry['decision'] = decision
            if decision.get('rule_revision') != uw.RULE_REVISION:
                entry['verdict'] = (f'UNJUDGED: decision_v2 rule_revision {decision.get("rule_revision")} but '
                                    f'offload/upfix judges revision {uw.RULE_REVISION}')
            else:
                entry['verdict'] = uw.arm_verdict(start.arm, decision, cast(dict[str, bool], gates_))
        arms[start.label] = entry
    spreads: dict[str, Json] = {}
    where: dict[str, Json] = {'max_tokens': uw.MAX_TOKENS, 'clock_cap_mhz': cap, 'commit': production}
    for cell in CELLS:
        for name in (cell, uw.HISTORY_NAMES[cell]):     # the ctl- cells' history is filed under the old names (S93)
            for metric, _ in uw.METRICS[cell]:
                try:
                    spreads[f'{name}/{metric}'] = dv2.archive_spread(archive, name, metric, where=where)
                except ValueError as exc:
                    spreads[f'{name}/{metric}'] = f'UNREADABLE: {exc}'
    promotable = [label for label, entry in arms.items() if record(entry).get('verdict') == 'PROMOTABLE (no loss)']
    return {**base, 'status': 'PROMOTABLE: ' + ', '.join(promotable) if promotable else 'NO PROMOTABLE ARM',
            'rule_revision': uw.RULE_REVISION, 'arms': arms, 'clock_cap_mhz': cap, 'archive_spread': spreads,
            'archive_where': where}
