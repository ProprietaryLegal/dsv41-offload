"""CEN4 rank-imbalance attribution, CPU only: pre-registered rules over a synthetic CEN3-W directory and SP1 window."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from offload.census import cen3_analyze as A
from offload.census import cen4
from offload.census import test_waterfall as TW
from offload.census.test_cen3_analyze import window
from offload.prefill_maxlen.evidence import Json, record


def telemetry(path: Path, clock: int, reasons: str = '0x0000000000000000', t0: float = 1.9e9) -> None:
    rows = [{'kind': 'memory', 'utc': t0, 'kb': {'MemAvailable': 1}}]
    rows += [{'kind': 'gpu', 'utc': t0 + i, 'csv': f'GPU-x, {clock}, [N/A], 40.5, [N/A], 60, {90 if i % 2 else 0}, '
                                                    f'{reasons}'} for i in range(20)]
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def capture(w0: float, w1: float) -> dict[str, Json]:
    return {'rank_ms': {'0': {'rank_wait': w0}, '1': {'rank_wait': w1}}}


def test_cen4_rules(tmp_path: Path) -> None:
    out = window(tmp_path, slow_rank=1)
    telemetry(out / 'telemetry-sparkhead.jsonl', 2281)
    telemetry(out / 'telemetry-sparkworker-wired.jsonl', 2190, '0x0000000000000004')
    thermal = tmp_path / 'THERMAL.log'
    thermal.write_text('2030-03-17T17:46:45Z sparkworker 70, 2190, 40.0, Not Active, Active\n'
                       '2030-03-17T17:46:45Z sparkhead 60, 2281, 40.0, Not Active, Not Active\n'
                       '1999-01-01T00:00:00Z sparkhead 99, 1, 1.0, Active, Active\n')
    base = A.analyze(out)
    record(base['parts'])['b'] = {'captures': {'cen3-short-b1:decode': capture(3.0, 0.5),
                                               'cen3-32k-b4:decode': capture(2.5, 0.4),
                                               'cen3-target-only-short-b1:decode': capture(0.0, 9.0)}}
    res = cen4.analyze(out, thermal_log=thermal, sp1=None, analysis=base)
    assert record(res['imbalance'])['slower_rank'] == 1
    found = record(res['causes'])
    assert {k for k, v in found.items() if record(v)['fires'] is True} == {'clock', 'throttle', 'thermal', 'bandwidth',
                                                                           'kernel'}
    assert str(res['verdict']).startswith('ATTRIBUTED (rank 1 = sparkworker-wired slower)')
    record(base['parts'])['b'] = {'captures': {'cen3-short-b1:decode': capture(0.6, 0.5)}}
    assert str(cen4.analyze(out, thermal_log=thermal, sp1=None, analysis=base)['verdict']).startswith('SYMMETRIC')
    record(base['parts'])['b'] = {'captures': {'cen3-short-b1:decode': capture(0.2, 2.2)}}
    res = cen4.analyze(out, thermal_log=thermal, sp1=None, analysis=base)
    assert str(res['verdict']).startswith('UNATTRIBUTED: rank 0 = sparkhead slower'), 'every cause points at rank 1'
    (out / 'cmd' / '0007-cen3-start.log').write_text('spark clock cap 2150 MHz\nsparkworker clock cap 2150 MHz\n')
    assert str(cen4.analyze(out, thermal_log=thermal, sp1=None, analysis=base)['verdict']).startswith('INCONCLUSIVE')


def test_cen4_compares_with_the_sp1_window(tmp_path: Path) -> None:
    out = window(tmp_path)
    sp1 = tmp_path / 'SP1-W-20990101T000000Z'
    (sp1 / 'traces').mkdir(parents=True)
    rows: list[Json] = []
    for rank, kernels in ((0, TW.RANK0), (1, TW.RANK1)):
        p = TW.write(sp1 / 'traces' / f'r{rank}.pt.trace.json.gz', TW.events(kernels))
        rows.append({'cell': 'prof-short-b1', 'capture': 'decode', 'status': 'TAKEN', 'rank': rank,
                     'file': f'traces/{p.name}', 'bytes': p.stat().st_size,
                     'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    (sp1 / 'traces-manifest.json').write_text(json.dumps(rows))
    res = cen4.analyze(out, thermal_log=tmp_path / 'none.log', sp1=sp1)
    cmp_ = record(res['sp1_comparison'])
    assert cmp_['window'] == sp1.name and str(cmp_['statement']).startswith('unchanged'), cmp_
    json.dumps(res, allow_nan=False)
