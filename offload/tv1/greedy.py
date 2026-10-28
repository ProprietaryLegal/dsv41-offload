"""Round 10 registration and receipts; experimental checks delegate to the NBW checker at NDM's 0.9-nat limit."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import cast

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import nbw_cells as nbw
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record

# NDM 2026-10-27 ~18:20Z: "You can loosen the nats to .9". Only the threshold changes (0.501 -> 0.901, the same
# 1e-3 rounding allowance); NBW's own default and every other condition stay as registered in Round 7.
REGISTRATION_ID = 'tv1-r10-near-tie-0p9'
NEAR_TIE_NATS = 0.9 + 1e-3
RULE = ('NBW first divergence: both tokens in target top-20; gap <=0.901 nats; '
        'better token <=0.901 nats from top; length/prefix difference STOP; W8 restore strictly IDENTICAL')
REFERENCE_PINS = Path(__file__).with_name('greedy-references.json')


class Failure(ValueError):
    """Carry the numeric partial receipt through STOP into the window summary."""

    def __init__(self, receipt: dict[str, Json], cause: BaseException) -> None:
        super().__init__(str(cause))
        self.receipt = receipt


def registered(result: dict[str, Json]) -> dict[str, Json]:
    return {**result, 'registration_id': REGISTRATION_ID, 'greedy_rule': RULE, 'near_tie_nats': NEAR_TIE_NATS}


def validate(result: dict[str, Json]) -> None:
    ck.require(result.get('registration_id') == REGISTRATION_ID and result.get('greedy_rule') == RULE
               and result.get('near_tie_nats') == NEAR_TIE_NATS,
               'TV1 experimental greedy registration missing/different')
    rows = ck.rows(result['checks'])
    ck.require([r.get('i') for r in rows] == list(range(6)), 'TV1 six ordered NBW greedy checks required')
    for row in rows:
        if row.get('identical') is True:
            continue
        ck.require(row.get('identical') is False and row.get('near_tie') is True,
                   'TV1 greedy divergence is not an NBW near-tie')
        index = row.get('token_index')
        ck.require(isinstance(index, int) and not isinstance(index, bool) and index >= 0,
                   'TV1 first-divergence position missing')
        for key in ('margin_nats', 'gap_to_top_nats'):
            gap = number(row[key])
            ck.require(math.isfinite(gap) and 0 <= gap <= NEAR_TIE_NATS,
                       'TV1 NBW divergence diagnosis missing/outside registered rule')
    identical = sum(r.get('identical') is True for r in rows)
    ck.require(result.get('identical') == identical and result.get('near_tie') == 6-identical,
               'TV1 NBW greedy counts mismatch')


def run(api: wc.Http, deadline: float, output: Path) -> dict[str, Json]:
    # The existing TV1 manifest still pins the exact K1 text bytes. Text is read
    # locally for NBW's first-divergence probe and never enters a receipt.
    pins = ck.read(REFERENCE_PINS)
    ck.require(hashlib.sha256(nbw.REFERENCE.read_bytes()).hexdigest() == pins['source_sha256'],
               'TV1 K1 greedy reference bytes changed')
    prompts, references = nbw.greedy_inputs()
    ck.require([hashlib.sha256(str(r['text']).encode()).hexdigest() for r in references]
               == [r['text_sha256'] for r in ck.rows(pins['references'])], 'TV1 K1 reference hashes changed')
    try:
        result = registered(nbw.greedy(api, deadline, prompts, references, output, near_tie_nats=NEAR_TIE_NATS))
    except (Exception, KeyboardInterrupt) as exc:
        # NBW saves a numeric partial receipt before rejecting a diagnosed flip.
        # Prefix/length differences or failed diagnostic RPCs cannot qualify.
        partial = ck.read(output) if output.exists() else {'checks': cast(Json, [])}
        result = registered({**partial, 'diagnostic_error': f'{type(exc).__name__}: {exc}'})
        output.write_text(json.dumps(result, allow_nan=False) + '\n')
        raise
    output.write_text(json.dumps(result, allow_nan=False) + '\n')
    validate(result)
    return result
