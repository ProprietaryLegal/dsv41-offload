"""WIN2-R3 process rules as code: two-window cap and 2x resolution filter (CPU only, tmp campaign and archive)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from offload.launch.windows import decision_v2 as d
from offload.launch.windows import resolution as rs
from offload.launch.windows import window_cap as cap
from offload.prefill_maxlen.evidence import Json, number, record

PROD = '596ba1cf4e3b11be29b41a57e7f69a5d8efa4799'
SINCE = cap.utc('2026-10-27T00:00:00Z')


def _window(camp: Path, name: str, status: str, **summary: Json) -> Path:
    w = camp / name
    w.mkdir(parents=True)
    (w / 'summary.json').write_text(json.dumps({'status': status, 'decision': {'status': 'NOT A CANDIDATE'}, **summary}))
    return w


def test_only_completed_measurement_windows_count(tmp_path: Path) -> None:
    _window(tmp_path, 'PC1-a-W-20261027T100000Z', 'COMPLETE')
    _window(tmp_path, 'PC1-b-W-20261027T110000Z', 'INCOMPLETE')                    # stopped: never counts
    _window(tmp_path, 'PC1-admission-W-20261027T120000Z', 'COMPLETE')
    _window(tmp_path, 'PC1-c-W-20261027T130000Z', 'COMPLETE', kind='smoke')
    _window(tmp_path, 'PC1-d-W-20261026T130000Z', 'COMPLETE')                      # before adoption
    _window(tmp_path, 'UP3-e-W-20261027T140000Z', 'COMPLETE')                      # another lane
    r = cap.check('PC1', since=SINCE, campaign=tmp_path)
    counted = r['counted']
    assert isinstance(counted, list)
    assert r['verdict'] == 'ALLOW' and [record(w)['window'] for w in counted] == ['PC1-a-W-20261027T100000Z']


def test_third_window_needs_a_hypothesis_dated_after_the_second(tmp_path: Path) -> None:
    _window(tmp_path, 'PC1-a-W-20261027T100000Z', 'COMPLETE')
    _window(tmp_path, 'PC1-b-W-20261027T150000Z', 'INCONCLUSIVE')
    r = cap.check('PC1', since=SINCE, campaign=tmp_path)
    assert r['verdict'] == 'REFUSE' and 'hypothesis-2.md' in str(r['reason'])
    hyp = tmp_path / 'experiments/PC1/hypothesis-2.md'
    hyp.parent.mkdir(parents=True)
    hyp.write_text('# PC1 hypothesis 2\ndate: 2026-10-27T14:00Z\n')                # dated BEFORE the second window
    assert cap.check('PC1', since=SINCE, campaign=tmp_path)['verdict'] == 'REFUSE'
    hyp.write_text('# PC1 hypothesis 2\ndate: 2026-10-27T16:30Z\nchunk 8192 changes ...\n')
    assert cap.check('PC1', since=SINCE, campaign=tmp_path)['verdict'] == 'ALLOW'
    _window(tmp_path, 'PC1-c-W-20261027T170000Z', 'COMPLETE')                      # the 4th needs another one
    assert cap.check('PC1', since=SINCE, campaign=tmp_path)['verdict'] == 'REFUSE'


@pytest.mark.parametrize('line,verdict', [('| HYPOTHESIS PC1 3 | 2026-10-27 18:10 | depth 1 |', 'ALLOW'),
                                          ('2026-10-27T18:00Z WINDOW-CAP OVERRIDE PC1 (NDM: one more)', 'ALLOW'),
                                          ('2026-10-27T18:00Z HYPOTHESIS PC10 3', 'REFUSE'),
                                          ('HYPOTHESIS PC1 3 (undated)', 'REFUSE')])
def test_register_line_or_override(tmp_path: Path, line: str, verdict: str) -> None:
    _window(tmp_path, 'PC1-a-W-20261027T100000Z', 'COMPLETE')
    _window(tmp_path, 'PC1-b-W-20261027T150000Z', 'COMPLETE')
    (tmp_path / 'LANES.md').write_text('# register\n' + line + '\n')
    assert cap.check('PC1', since=SINCE, campaign=tmp_path)['verdict'] == verdict


def test_cli_needs_adoption_and_prefixes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _window(tmp_path, 'NBW2-W-20261027T100000Z', 'COMPLETE')
    _window(tmp_path, 'NBW-W-20261027T110000Z', 'COMPLETE')
    assert cap.main(['NBW', '--campaign', str(tmp_path)]) == 2                    # not adopted: refuses
    (tmp_path / cap.ADOPTION_FILE).write_text(json.dumps({'adopted_utc': '2026-10-27T00:00:00Z'}))
    assert cap.main(['NBW', '--campaign', str(tmp_path), '--prefix', 'NBW-W', '--prefix', 'NBW2-W']) == 2
    capsys.readouterr()
    assert cap.main(['NBW', '--campaign', str(tmp_path), '--prefix', 'NBW2-W']) == 0
    assert json.loads(capsys.readouterr().out)['verdict'] == 'ALLOW'
    with pytest.raises(ValueError):
        cap.check('../x', since=SINCE, campaign=tmp_path)


# ---------------------------------------------------------------- 2x resolution filter

def _archive(tmp_path: Path, cv: float, windows: int = 3, n: int = 6, metric: str = 'ms_per_round') -> Path:
    path = tmp_path / 'a.jsonl'
    rows: list[dict[str, Json]] = []
    for w in range(windows):
        base = 40.0 * (1 + 0.01 * w)
        # n values with exactly sample SD = cv * mean: alternate +-k around the mean
        k = cv * base * math.sqrt((n - 1) / n)
        for i in range(n):
            rows.append({'window': f'W{w}', 'cell': '32k-b1', 'metric': metric, 'value': base + (k if i % 2 else -k),
                         'max_tokens': 512, 'commit': PROD, 'clock_cap_mhz': 2150, 'date': '20261027',
                         'start': 's1-control', 'repeat': i})
    d.archive_append(path, rows)
    return path


def test_pooled_cv_and_half_widths_match_the_formulas(tmp_path: Path) -> None:
    path = _archive(tmp_path, .02)
    got = rs.pooled_cv(path, '32k-b1', 'ms_per_round', max_tokens=512, clock_cap_mhz=2150, commit=PROD)
    assert number(got['cv']) == pytest.approx(.02) and got['windows'] == 3 and got['df'] == 15
    assert number(rs.half_width('unpaired:3v3', .021)['half_width']) == pytest.approx(2.776 * .021 * math.sqrt(2 / 3))
    assert number(rs.half_width('pairs:6', .021)['half_width']) == pytest.approx(2.571 * .021 * math.sqrt(2) / math.sqrt(6))
    est = rs.half_width('seeded:6', 0.0, ms_cv=.01)
    assert number(est['half_width']) == pytest.approx(2.571 * .01 * math.sqrt(2) / math.sqrt(6)) and 'ESTIMATED' in str(est['basis'])
    measured = rs.half_width('seeded:12', 0.0, diff_cv=.02)
    assert number(measured['half_width']) == pytest.approx(2.201 * .02 / math.sqrt(12)) and 'MEASURED' in str(measured['basis'])
    for bad in ('3v3', 'unpaired:1v3', 'seeded:4', 'pairs:x'):
        with pytest.raises(ValueError):
            rs.half_width(bad, .02, ms_cv=.01)


def test_filter_refuses_below_twice_the_half_width(tmp_path: Path) -> None:
    path = _archive(tmp_path, .01)
    def verdict(metric: str, design: str, gain: float) -> dict[str, Json]:
        return rs.check('32k-b1', metric, design, gain, max_tokens=512, clock_cap_mhz=2150, commit=PROD, archive=path)
    seeded6 = verdict('decode_tps', 'seeded:6', .03)
    assert seeded6['verdict'] == 'QUEUE' and number(seeded6['required_gain']) < .03          # +-1.48 % (F-METHOD)
    assert verdict('ms_per_round', 'unpaired:3v3', .05)['verdict'] == 'QUEUE'
    assert verdict('ms_per_round', 'unpaired:3v3', .04)['verdict'] == 'REFUSE'                # +-2.27 %
    with pytest.raises(ValueError, match='no archive history'):
        rs.check('32k-b1', 'ms_per_round', 'unpaired:3v3', .03, max_tokens=512, clock_cap_mhz=2300, commit=PROD,
                 archive=path)


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _archive(tmp_path, .01)
    base = ['--cell', '32k-b1', '--metric', 'decode_tps', '--max-tokens', '512', '--clock', '2150', '--commit', PROD,
            '--archive', str(path)]
    assert rs.main([*base, '--design', 'seeded:6', '--expected-gain', '0.03']) == 0
    assert record(json.loads(capsys.readouterr().out))['verdict'] == 'QUEUE'
    assert rs.main([*base, '--design', 'seeded:6', '--expected-gain', '0.01']) == 2
    assert rs.main([*base, '--design', 'seeded:6', '--expected-gain', '0.03', '--diff-cv', '0.05']) == 2
    assert rs.main([*base, '--design', 'seeded:3', '--expected-gain', '0.03']) == 2
