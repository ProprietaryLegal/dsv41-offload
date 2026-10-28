"""Aggregate N3 receipts only, never token/capture values. CPU-only."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any


def sampled_peaks(smi: Path, phase_log: Path) -> dict[str,dict[str,float]]:
    if not smi.is_file() or not phase_log.is_file():
        raise ValueError(f'missing telemetry/phase log: {smi} {phase_log}')
    phases = [json.loads(line) for line in phase_log.read_text().splitlines() if line.strip()]
    if not phases or phases[-1]['phase'] != 'done':
        raise ValueError('job phase log is incomplete')
    phases.sort(key=lambda row: row['time'])
    peaks: dict[str,dict[str,float]] = {}
    for values in csv.reader(smi.read_text().splitlines()):
        if len(values) < 3:
            raise ValueError('malformed nvidia-smi sample')
        stamp = dt.datetime.fromisoformat(values[0].strip().replace('/','-')).replace(tzinfo=dt.timezone.utc).timestamp()
        active = [row for row in phases if row['time'] <= stamp]
        if not active or active[-1]['phase'] == 'done':
            continue
        phase = active[-1]['phase']
        gpu, memory = values[1].strip(),float(values[2].strip().split()[0])
        peaks.setdefault(phase,{})[gpu] = max(peaks.get(phase,{}).get(gpu,0),memory)
    return peaks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    a = ap.parse_args()
    shards = []
    for p in sorted(a.results.glob('shard-native-*-r[01].json')):
        r = json.loads(p.read_text())
        if r['rounds'] < 2000 or len(r['latency_ms']) != 40 or r['mode'] != 'solo-shard-local-collectives':
            raise ValueError(f'incomplete shard qualification: {p}')
        if r['devices'] != 'GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc':
            raise ValueError('unauthorized GPU receipt')
        if sum(x['n'] for x in r['latency_ms'].values()) != r['rounds']:
            raise ValueError('missing latency cases')
        r['observed_smi_peaks_mib_1s'] = sampled_peaks(p.with_suffix('.smi.csv'),p.with_suffix('.phase.jsonl'))
        r['latency_boundary'] = 'host graph replay + synchronization + rank0 validation; no settled performance claim'
        shards.append(r)
    soaks = []
    for p in sorted(a.results.glob('soak-native-*.json')):
        r = json.loads(p.read_text())
        if not r['dump_on_off_exact'] or r['off']['digest'] != r['on']['digest']:
            raise ValueError('dump equality did not pass')
        if min(r['off']['rounds'],r['on']['rounds']) < 2000 or min(len(r['off']['cases']),len(r['on']['cases'])) != 40:
            raise ValueError('incomplete protocol soak')
        r['observed_smi_peaks_mib_1s'] = sampled_peaks(p.with_suffix('.smi.csv'),p.with_suffix('.phase.jsonl'))
        # Reply digests are not needed in the public/committed aggregate.
        for mode in ('off','on'):
            r[mode].pop('digest')
        soaks.append(r)
    if not shards:
        raise ValueError('no completed shard receipts')
    out: dict[str,Any] = {'schema':'dspark.n3.aggregate.v1','shards':shards,'protocol_soaks':soaks,
        'pair_measured':False,'expanded_full_emulation':'not possible on 24GiB; coordinator pair required'}
    a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(f'{len(shards)} shard receipts, {len(soaks)} complete protocol soaks; pair unmeasured')


if __name__ == '__main__':
    main()
