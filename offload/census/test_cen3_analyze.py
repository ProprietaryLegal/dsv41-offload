"""CEN3 analysis, CPU only, over a synthetic CEN3-W directory (no real trace, count or text)."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from offload.census import cen3_analyze as A
from offload.census import cen3_bench as B
from offload.census import test_waterfall as TW
from offload.prefill_maxlen.evidence import Json, number, record

ROT = '(anonymous namespace)::smoe_rot((anonymous namespace)::Layer, (anonymous namespace)::Call)'
NAME = {'smoe_gu': TW.SMOE_GU, 'smoe_down': '(anonymous namespace)::smoe_down((anonymous namespace)::Layer, '
        '(anonymous namespace)::Call)', 'mrow': TW.MROW5, 'mrow_head': TW.MROW6}
HEAD = ('"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size",'
        '"Device","CC","dram__bytes_read.sum","dram__bytes_write.sum","gpu__time_duration.sum"')


def target(group: str) -> dict[str, Json]:
    if group == 'smoe-b1':
        cases: list[Json] = [{'case': f'c{i}', 'launches': [{'family': 'smoe_gu', 'bytes': 30e6 + i},
                                                           {'family': 'smoe_down', 'bytes': 15e6 + i}]} for i in range(3)]
    else:
        cases = [{'case': label, 'launches': [{'family': fam, 'bytes': nbytes}]}
                 for label, _, fam, nbytes in B.dense_cases(group)]
    return {'type': 'cen3-ncu-target', 'group': group, 'cases': cases}


def csv(doc: Mapping[str, Json], factor: float, us: float, *, extra: int = 0) -> str:
    rows, i = [HEAD, '"","","","","","","","","","","","byte","byte","nsecond"'], 0
    for case in cast(list[Json], doc['cases']):
        launches = cast(list[Json], record(case)['launches'])
        if any(record(x)['family'] == 'smoe_gu' for x in launches):
            rows.append(f'"{i}","1","p","h","{ROT}","1","7","(1, 1, 1)","(1, 1, 1)","0","12.1","4096","4096","1000"')
            i += 1
        for x in launches + launches[:extra]:
            x = record(x)
            rows.append(f'"{i}","1","p","h","{NAME[str(x["family"])]}","1","7","(1, 1, 1)","(1, 1, 1)","0","12.1",'
                        f'"{number(x["bytes"]) * factor:.0f}","1000","{us * 1000:.0f}"')
            i += 1
    return '==PROF== Connected\n' + '\n'.join(rows) + '\n'



def window(tmp_path: Path, *, slow_rank: int | None = None, log_file: bool = True, cap: int = 2300) -> Path:
    out = tmp_path / 'CEN3-W-20990101T000000Z'
    rows: list[Json] = []
    for rank, host in enumerate(A.HOSTS):
        d = out / 'bench' / host
        d.mkdir(parents=True)
        files: dict[str, str] = {
            'bw.json': json.dumps({'ops': [B.bw_summary(op, 2**31, [10.0 if rank != slow_rank else 11.0])
                                           for op in B.BW_OPS]}),
            'ncu-metrics.json': json.dumps(B.pick_metrics('dram__bytes_read dram__bytes_write gpu__time_duration').doc())}
        for group in ('smoe-b1', 'head'):
            doc = target(group)
            files[f'ncu-{group}-target.json'] = json.dumps(doc)
            text = csv(doc, 1.02 if group == 'smoe-b1' else 0.5, 100.0 * (1.1 if rank == slow_rank else 1.0))
            files[f'ncu-{group}.csv' if log_file else f'ncu-{group}.app.txt'] = text
            files[f'ncu-{group}.app.txt' if log_file else f'ncu-{group}.csv'] = ''
        for name, text in files.items():
            (d / name).write_text(text)
            rows.append({'host': host, 'file': f'bench/{host}/{name}', 'bytes': (d / name).stat().st_size,
                         'sha256': hashlib.sha256((d / name).read_bytes()).hexdigest()})
    (out / 'bench-manifest.json').write_text(json.dumps(rows))
    (out / 'traces').mkdir()
    pair = {0: TW.write(out / 'traces' / 'r0.pt.trace.json.gz', TW.events(TW.RANK0)),
            1: TW.write(out / 'traces' / 'r1.pt.trace.json.gz', TW.events(TW.RANK1))}
    manifest = [{'cell': cell, 'capture': 'decode', 'status': status, 'rank': r, 'host': A.HOSTS[r],
                 'file': f'traces/{p.name}', 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                for cell, status in (('cen3-short-b1', 'TAKEN'), ('cen3-32k-b4', 'ENDED')) for r, p in pair.items()]
    (out / 'traces-manifest.json').write_text(json.dumps(manifest))
    (out / 'cmd').mkdir()
    (out / 'cmd' / '0007-cen3-start.log').write_text(f'spark clock cap {cap} MHz\nsparkworker clock cap {cap} MHz\n')
    return out


@pytest.mark.parametrize('log_file', [True, False])
def test_analysis_reads_every_part(tmp_path: Path, log_file: bool) -> None:
    out = window(tmp_path, log_file=log_file)
    res = A.analyze(out)
    assert res['clock_cap_mhz'] == 2300
    parts = record(res['parts'])
    rates = record(record(parts['c'])['bandwidth'])
    assert number(record(record(rates['sparkhead'])['copy'])['gbps_median']) == pytest.approx(2**32 / 10 / 1e6)
    ratio = record(record(record(parts['a'])['ratio'])['sparkhead'])
    fams = record(ratio['families'])
    assert str(record(fams['smoe_gu'])['verdict']).startswith('PHYSICAL ~ LOGICAL')
    assert str(record(fams['mrow_head'])['verdict']).startswith('BELOW LOGICAL')
    assert ratio['physical_source'] == 'DRAM counters'
    run = record(cast(list[Json], ratio['runs'])[0])
    assert run['unassigned_launches'] == 3, 'smoe_rot moves activations only'
    times = record(record(record(parts['a'])['kernel_time'])['sparkworker-wired'])
    assert record(times['smoe-b1:smoe_gu'])['launches'] == 3
    b = record(parts['b'])
    assert set(record(b['captures'])) == {'cen3-short-b1:decode'}
    assert b['skipped_captures'] == ['cen3-32k-b4:decode (ENDED)']
    manifest = json.loads((out / 'bench' / 'sparkhead' / 'ncu-manifest.json').read_text())
    gu = next(x for x in manifest['runs'][0]['launches'] if x['match'] == 'smoe_gu')
    assert gu['logical'] == {'bytes_by_index': [30e6, 30e6 + 1, 30e6 + 2]}


def test_a_launch_count_off_the_driver_refuses(tmp_path: Path) -> None:
    out = window(tmp_path)
    d = out / 'bench' / 'sparkhead'
    (d / 'ncu-smoe-b1.csv').write_text(csv(target('smoe-b1'), 1.0, 100.0, extra=1))
    rows = json.loads((out / 'bench-manifest.json').read_text())
    for row in rows:
        if row['file'] == 'bench/sparkhead/ncu-smoe-b1.csv':
            row['bytes'] = (d / 'ncu-smoe-b1.csv').stat().st_size
            row['sha256'] = hashlib.sha256((d / 'ncu-smoe-b1.csv').read_bytes()).hexdigest()
    (out / 'bench-manifest.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='driver profiled|do not match'):
        A.analyze(out)


def test_a_changed_bench_file_refuses(tmp_path: Path) -> None:
    out = window(tmp_path)
    (out / 'bench' / 'sparkworker-wired' / 'bw.json').write_text('{}')
    with pytest.raises(ValueError, match='bench manifest'):
        A.analyze(out)
