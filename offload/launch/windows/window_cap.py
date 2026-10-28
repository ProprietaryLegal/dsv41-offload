"""WIN2-R3 two-window cap (F-METHOD §3.2 rule 2): a lane gets two measurement windows since the rule's adoption; a third
(and every later one) needs a new hypothesis dated after the lane's latest counted window.

Counted: window directories `<C>/<PREFIX>...-W-<YYYYMMDDTHHMMSSZ>` of the lane at or after adoption whose summary
`status` is COMPLETE, INCONCLUSIVE, NOT A CANDIDATE or CANDIDATE. Windows that stopped (status INCOMPLETE, whatever
their pre-filled decision says), admission windows and smoke starts (name contains 'admission' or 'smoke', or summary
kind 'admission'/'smoke') never count.

A new hypothesis is any of, dated (an ISO UTC timestamp within its first lines) after the latest counted window:
- `<experiments>/<LANE>/hypothesis-<n>.md` (default experiments root: the campaign's `experiments/`);
- a line of the lane register (LANES.md) containing `HYPOTHESIS <LANE>`;
- NDM's written override: a register line containing `WINDOW-CAP OVERRIDE <LANE>`.
Launchers call `python -m offload.launch.windows.window_cap <LANE>`: exit 0 ALLOW, 2 REFUSE. Read only.
"""
from __future__ import annotations

import argparse
import calendar
import json
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from offload.prefill_maxlen.evidence import Json, record

CAMPAIGN = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD')
CAP = 2
COUNTED = ('COMPLETE', 'INCONCLUSIVE', 'NOT A CANDIDATE', 'CANDIDATE')
WINDOW = re.compile(r'-W-(20[0-9]{6}T[0-9]{6}Z)$')
STAMP = re.compile(r'(20[0-9]{2}-[0-9]{2}-[0-9]{2})[T ]([0-9]{2}:[0-9]{2})(?::([0-9]{2}))?Z?')
ADOPTION_FILE = 'WINDOW-CAP.json'          # {"adopted_utc": "YYYY-MM-DDTHH:MM:SSZ"} written by the coordinator


def utc(text: str) -> float:
    for fmt in ('%Y%m%dT%H%M%SZ', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%MZ'):
        try:
            return float(calendar.timegm(time.strptime(text, fmt)))
        except ValueError:
            continue
    raise ValueError(f'not a UTC timestamp: {text!r}')


def _stamp(line: str) -> float | None:
    m = STAMP.search(line)
    if m is None:
        return None
    return float(calendar.timegm(time.strptime(f'{m.group(1)}T{m.group(2)}:{m.group(3) or "00"}', '%Y-%m-%dT%H:%M:%S')))


def counted_windows(campaign: Path, prefixes: Sequence[str], since: float) -> list[dict[str, Json]]:
    found: list[dict[str, Json]] = []
    for path in sorted(campaign.iterdir()):
        m = WINDOW.search(path.name)
        if not any(path.name.startswith(prefix) for prefix in prefixes) or m is None:
            continue
        if not path.is_dir() or not (path / 'summary.json').is_file():
            continue
        when = utc(m.group(1))
        if when < since or 'admission' in path.name.lower() or 'smoke' in path.name.lower():
            continue
        summary = record(cast(Json, json.loads((path / 'summary.json').read_text())))
        if str(summary.get('kind', '')).lower() in ('admission', 'smoke'):
            continue
        # Only the summary status counts: runners pre-fill decision NOT A CANDIDATE ("not measured completely") on
        # windows that stopped, and those are not measurement windows.
        status = str(summary.get('status', ''))
        if status in COUNTED:
            decision = summary.get('decision')
            decided = str(record(decision).get('status', '')) if isinstance(decision, dict) else ''
            found.append({'window': path.name, 'utc': when, 'status': status, 'decision': decided})
    return sorted(found, key=lambda w: float(cast(float, w['utc'])))


def hypotheses(lane: str, experiments: Path, register: Path) -> list[dict[str, Json]]:
    """Every dated hypothesis or override for the lane: files experiments/<LANE>/hypothesis-<n>.md and register lines."""
    out: list[dict[str, Json]] = []
    for path in sorted((experiments / lane).glob('hypothesis-*.md')) if (experiments / lane).is_dir() else []:
        stamps = [s for s in (_stamp(line) for line in path.read_text().splitlines()[:8]) if s is not None]
        if stamps:
            out.append({'source': str(path), 'utc': stamps[0]})
    if register.is_file():
        for n, line in enumerate(register.read_text().splitlines(), 1):
            if re.search(rf'\b(HYPOTHESIS|WINDOW-CAP OVERRIDE) {re.escape(lane)}\b', line):
                stamp = _stamp(line)
                if stamp is not None:
                    out.append({'source': f'{register}:{n}', 'utc': stamp, 'override': 'OVERRIDE' in line})
    return out


def check(lane: str, *, prefixes: Sequence[str] | None = None, since: float, campaign: Path = CAMPAIGN,
          experiments: Path | None = None, register: Path | None = None) -> dict[str, Json]:
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*', lane):
        raise ValueError(f'invalid lane id {lane!r}')
    windows = counted_windows(campaign, list(prefixes or [lane]), since)
    result: dict[str, Json] = {'lane': lane, 'cap': CAP, 'since_utc': since, 'counted': cast(Json, windows)}
    if len(windows) < CAP:
        result['verdict'] = 'ALLOW'
        result['reason'] = f'{len(windows)} of {CAP} measurement windows used since adoption'
        return result
    latest = float(cast(float, windows[-1]['utc']))
    later = [h for h in hypotheses(lane, experiments or campaign / 'experiments', register or campaign / 'LANES.md')
             if float(cast(float, h['utc'])) > latest]
    result['hypotheses_after_latest_window'] = cast(Json, later)
    if later:
        result['verdict'] = 'ALLOW'
        result['reason'] = f'{len(windows)} windows used; new hypothesis dated after {windows[-1]["window"]}'
    else:
        result['verdict'] = 'REFUSE'
        result['reason'] = (f'{len(windows)} measurement windows since adoption (cap {CAP}); write '
                            f'experiments/{lane}/hypothesis-{len(windows)}.md (dated) or a register line '
                            f'"HYPOTHESIS {lane} <UTC>" after {windows[-1]["window"]}, or NDM "WINDOW-CAP OVERRIDE {lane} <UTC>"')
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('lane')
    ap.add_argument('--prefix', action='append', help='window name prefix of the lane (default: the lane id)')
    ap.add_argument('--since', help='adoption UTC (YYYY-MM-DDTHH:MM:SSZ); default: <campaign>/WINDOW-CAP.json')
    ap.add_argument('--campaign', type=Path, default=CAMPAIGN)
    ap.add_argument('--experiments', type=Path)
    ap.add_argument('--register', type=Path)
    args = ap.parse_args(argv)
    if args.since is None:
        adoption = args.campaign / ADOPTION_FILE
        if not adoption.is_file():
            print(f'STOP: the two-window cap is not adopted ({adoption} missing); pass --since', file=sys.stderr)
            return 2
        args.since = str(record(cast(Json, json.loads(adoption.read_text())))['adopted_utc'])
    result = check(args.lane, prefixes=args.prefix, since=utc(args.since), campaign=args.campaign,
                   experiments=args.experiments, register=args.register)
    print(json.dumps(result, allow_nan=False))
    return 0 if result['verdict'] == 'ALLOW' else 2


if __name__ == '__main__':
    raise SystemExit(main())
