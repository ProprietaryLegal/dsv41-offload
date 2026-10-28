"""CPU-only report extraction. Never copies routing fixtures or activation data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out: dict[str,object]={'qualification':[],'timing':[],'baseline':[]}
    qs=[];bs=[];controls=[]
    for path in sorted(a.results.glob('qual*v*-final.json')):
        r=json.loads(path.read_text())
        if 'gates' not in r:continue
        qs.append({'file':path.name,'stage':r['stage'],'rank':r['rank'],'variant':r['variant'],
            'elements':r['elements'],'n1_bf16_mismatch_count':len(r['n1_bf16_mismatches']),
            'n1_bf16_agreement':r['n1_bf16_agreement'],
            'n1_bf16_mismatches':r['n1_bf16_mismatches'],'f64_bf16_mismatches':r['f64_bf16_mismatches'],
            'hq_self_mismatches':r['hq_self_mismatches'],'hq_n1_mismatches':r['hq_n1_mismatches'],
            'max_full_f64_relative':max(v['full_f64_relative_frobenius'] for v in r['slot_batches']),
            'max_down_relative':max(v['down_from_own_hq_relative_frobenius'] for v in r['slot_batches']),
            'slot_batches':r['slot_batches'],'gates':r['gates'],'elapsed_s':r['elapsed_s'],
            'before':r['before'],'after':r['after'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    for path in sorted(a.results.glob('bench*final.json')):
        r=json.loads(path.read_text())
        points=[]
        for rec in r['records']:
            for when in ('before','after'):
                vals=rec[when].split(',')
                points.append([float(vals[i].strip().split()[0]) for i in (2,3,4,5,6)])
        summaries=[]
        for s in r['summary']:
            records=[v for v in r['records'] if v['rows']==s['rows'] and v['warps']==s['warps'] and v['round']!=-1]
            us=[v['unique'] for v in records]
            summaries.append({**s,'unique_min':min(us) if us else None,'unique_median':statistics.median(us) if us else None,
                'unique_max':max(us) if us else None})
        bs.append({'file':path.name,'stage':r['stage'],'rank':r['rank'],'summary':summaries,
            'telemetry_min':[min(v[i] for v in points) for i in range(5)],
            'telemetry_max':[max(v[i] for v in points) for i in range(5)],
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    for path in sorted(a.results.glob('baseline*final.json')):
        controls.append({'file':path.name,'records':json.loads(path.read_text()),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    out['qualification']=qs;out['timing']=bs;out['baseline']=controls
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(f'completed qualifications={len(qs)}, full timing sweeps={len(bs)}, baseline screens={len(controls)}')

if __name__=='__main__':main()
