"""CPU test for the census table builder on a synthetic campaign folder (one SP1-style window and an archive)."""

from __future__ import annotations

import json
from pathlib import Path

from offload.census import table as TB
from offload.census import test_costtable as TC
from offload.census import test_waterfall as TW
from offload.census import waterfall as W
from offload.prefill_maxlen.evidence import Json, number, record


def campaign(tmp_path: Path) -> Path:
    root = tmp_path / 'camp'
    win = root / 'SP1-W-20990101T000000Z'
    (win / 'traces').mkdir(parents=True)
    (win / 'cmd').mkdir()
    (win / 'cmd' / '0037-sp1-327-start.log').write_text('start\nok\nspark clock cap 2300 MHz\nsparkworker clock cap 2300 MHz\n')
    rows: list[dict[str, Json]] = []
    for rank, spec in ((0, TW.RANK0), (1, TW.RANK1)):
        path = TW.write(win / 'traces' / f'r{rank}.pt.trace.json.gz', TW.events(spec))
        rows.append({'cell': 'prof-short-b1', 'capture': 'decode', 'status': 'TAKEN', 'rank': rank,
                     'file': f'traces/{path.name}', 'bytes': path.stat().st_size, 'sha256': W._sha256(path)})
    (win / 'traces-manifest.json').write_text(json.dumps(rows))
    archive = [TC.row('short-b1', 'control', 'decode_tps', 50.0 + w, 0, 2300, s, f'W{w}')
               for w in range(3) for s in ('s1-control', 's3-control')]
    (root / 'control-archive.jsonl').write_text('\n'.join(TC.lines(archive)) + '\n')
    return root


def test_build_and_markdown(tmp_path: Path) -> None:
    root = campaign(tmp_path)
    res = TB.build(root, ['SP1-W-20990101T000000Z'], None, root / 'control-archive.jsonl')
    meta = record(record(res['window_meta'])['SP1-W-20990101T000000Z'])
    assert meta['clock_caps_mhz'] == [2300]
    floors = res['round_floors']
    assert isinstance(floors, list) and len(floors) == 1
    first = record(floors[0])
    assert first['median_rows'] == 3 and number(first['distinct']) == 13.0
    res2300 = record(record(res['resolution'])['decode_tps@2300'])
    assert record(record(res2300['cells'])['short-b1'])['pairs'] == 3
    md = TB.markdown(res)
    for heading in ('## Waterfall per verify round', '## Verify-round logical bytes', '## Target-only token floor',
                    '## Control-archive resolution', 'wall (union, incl. overlap): mhc'):
        assert heading in md
    assert TB.kl1_reproduction(root, root / 'KL1-W-x')['status'] == 'SKIPPED: KL1-bandwidth.json absent'


def test_main_never_overwrites(tmp_path: Path) -> None:
    root = campaign(tmp_path)
    args = ['--campaign', str(root), '--sp1', 'SP1-W-20990101T000000Z', '--output', 'o.json', '--markdown', 'o.md']
    assert TB.main(args) == 0
    assert (root / 'o.md').read_text().startswith('# CEN0 census table')
    assert TB.main(args) == 1
