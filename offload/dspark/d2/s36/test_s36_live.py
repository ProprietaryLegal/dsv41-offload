"""S36 probes of the real G1 live reference (amendment 7/7b, G1-LIVE.md). Read only; skipped when artifacts are absent.

The coordinator attested paused_at 06:18:49Z (generator session gone) instead of decode_close's 06:19:00Z pause, and
rollback_at 06:22:45Z (rollback window open). These probes check that both are supported by the recorded evidence and
that the receipt's counters and rates recompute from byte-identical copies of the W12 snapshots.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from d2.live_reference import build_reference, digest, parse_metrics, timestamp

CAMPAIGN = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD')
INPUTS = CAMPAIGN/'S36-inputs'
LIVE = CAMPAIGN/'W12/private/live-acceptance'
ROLLBACK_LOG = CAMPAIGN/'W12/private/rollback-d2pass2-r1.log'
EVENTS = Path('/opt/dsv41/data/d2/gen-decode-07fc7d6-d2pass2-r1/events.jsonl')
FIRST, FINAL, RATE = 'metrics-20261025T042122Z.prom', 'metrics-20261025T061849Z.prom', 'rate-20261025T061849Z.json'
needs_real = pytest.mark.skipif(not all(p.exists() for p in (INPUTS/'live-reference.json', LIVE, ROLLBACK_LOG, EVENTS)),
                                reason='real live-reference artifacts absent')
SKIP = {'sources', 'snapshots', 'rates', 'sha256'}


def local() -> tuple[list[Path], list[Path]]:
    return [INPUTS/'live-sources'/FIRST, INPUTS/'live-sources'/FINAL], [INPUTS/'live-sources'/RATE]


@needs_real
def test_sources_are_the_w12_snapshots_and_receipt_recomputes() -> None:
    real = json.loads((INPUTS/'live-reference.json').read_text())
    for name in (FIRST, FINAL, RATE):
        assert (INPUTS/'live-sources'/name).read_bytes() == (LIVE/name).read_bytes()
        assert hashlib.sha256((LIVE/name).read_bytes()).hexdigest() in real['sources'].values()
    mine = build_reference(*local(), real['paused_at'], real['rollback_at'])
    assert {k: v for k, v in mine.items() if k not in SKIP} == {k: v for k, v in real.items() if k not in SKIP}
    assert digest({k: v for k, v in real.items() if k != 'sha256'}) == real['sha256'] == (
        'af6ad0d3154467a8f339bceca072ff974bb9236e7508c50d07324f202e297079')
    c, p = real['counters'], 'vllm:spec_decode_'
    assert real['status'] == 'VALID' and real['generator_tokens'] == 974303
    assert c['vllm:generation_tokens_total'] - 974303 == 1215
    assert round(real['position0_acceptance'], 4) == 0.6764 and round(real['accepted_per_round'], 4) == 1.4319
    assert c[p + 'num_accepted_tokens_per_pos_total:0'] / c[p + 'num_drafts_total'] == real['position0_acceptance']
    # Counters belong to the 02:55Z engine start (created 02:55:12Z) in both snapshots.
    starts = {v for path in local()[0] for v in parse_metrics(path)['starts'].values()}
    assert all(timestamp('2026-10-25T02:55:12Z') <= v < timestamp('2026-10-25T02:55:13Z') for v in starts)


@needs_real
def test_decode_close_pause_time_cannot_be_used_and_generator_exit_supports_0618() -> None:
    with pytest.raises(ValueError, match='after pause and before rollback'):
        build_reference(*local(), '2026-10-25T06:19:00Z', '2026-10-25T06:22:45Z')
    events = [json.loads(line) for line in EVENTS.read_text().splitlines()]
    tail = [(e['t'], e['event']) for e in events[-6:]]
    # The generator stopped itself at the Spark-h budget at 06:18:32Z (abort of its 3 own streams, finished, stop).
    assert tail[0] == ('2026-10-25T06:18:32Z', 'abort_all') and tail[-2:] == [
        ('2026-10-25T06:18:32Z', 'finished'), ('2026-10-25T06:18:32Z', 'stop')]
    assert events[-6]['why'] == 'spark-hours budget'
    marker = json.loads((LIVE/'FINAL-20261025T061849Z.json').read_text())
    assert marker == {'generator_session_gone_utc': '2026-10-25T06:18:49Z', 'final_snapshot': FINAL, 'rate_row': RATE}
    rate = json.loads((LIVE/RATE).read_text())
    assert rate['own_streams'] == 0 and rate['gen_tokens'] == 974303
    assert timestamp('2026-10-25T06:18:49Z') - timestamp(rate['t']) == 17  # within the 120 s freshness bound
    # Any attested pause in [generator stop, final snapshot] yields the same measurement.
    alt = build_reference(*local(), '2026-10-25T06:18:32Z', '2026-10-25T06:22:45Z')
    real = json.loads((INPUTS/'live-reference.json').read_text())
    assert {k: v for k, v in alt.items() if k not in SKIP | {'paused_at'}} == {
        k: v for k, v in real.items() if k not in SKIP | {'paused_at'}}


@needs_real
def test_rollback_at_is_the_window_open_before_the_engine_stop() -> None:
    lines = ROLLBACK_LOG.read_text().splitlines()
    opened = [i for i, line in enumerate(lines) if line.startswith('window open: 2026-10-25T06:22:45Z')]
    down = [i for i, line in enumerate(lines) if line == 'spark down']
    # The counters reset only at the engine stop, which the log records after the window opened; 06:22:45Z is
    # therefore a conservative (earliest) rollback bound, and the final snapshot precedes it by ~4 minutes.
    assert opened and down and opened[0] < down[0]
    assert 'ROLLBACK-EXIT=0' in lines
