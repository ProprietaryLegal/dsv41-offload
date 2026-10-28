#!/usr/bin/env python3
"""Coordinator-only diagnostic: preserve exact needle prompts/salts across starts.

prepare writes local prompt packets; run streams a saved packet; compare checks
codes, first-token common-support-plus-OTHER KL, and fresh-control cache misses.
Top-20 API logprobs cannot establish full-vocabulary KL. This diagnostic fixes
thinking=False; production speed gates still use the unchanged reasoning suite.

MC1: the persistent tier is matter-gated (default deny). A saved case's salt is 64 hex
and doubles as its synthetic matter namespace: ordinary runs send
kv_transfer_params {matter_ns: salt, persist: true}, so they store and restore through the
file tier. --fresh controls use a new salt and no tag (never persisted, never read).
--matter-ns overrides the namespace (e.g. a synthetic excluded namespace for the
zero-bytes test); --no-persist sends the salt without the tag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
from pathlib import Path
import sys
import time
from typing import Any
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from offload.kvpack.needle import Builder, score


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Prompt/response evidence belongs outside git and is private by default.
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def ask(url: str, model: str, case: dict[str, Any], fresh: bool, timeout: float,
        persist: bool = True, matter_ns: str | None = None) -> dict[str, Any]:
    salt = secrets.token_hex(32) if fresh else case['salt']
    body: dict[str, Any] = {'model': model, 'messages': [{'role': 'user', 'content': case['prompt']}],
        'cache_salt': salt, 'temperature': 0, 'top_p': 1, 'max_tokens': 300, 'priority': 10,
        'chat_template_kwargs': {'thinking': False}, 'logprobs': True, 'top_logprobs': 20,
        'stream': True, 'stream_options': {'include_usage': True}}
    tagged = persist and not fresh
    if tagged:
        ns = matter_ns or salt
        if len(ns) != 64 or any(c not in '0123456789abcdef' for c in ns):
            raise ValueError('matter namespace must be 64 lowercase hex (re-prepare packets made before MC1)')
        body['kv_transfer_params'] = {'matter_ns': ns, 'persist': True}
    req = urllib.request.Request(url.rstrip('/')+'/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    first_token = None
    first_visible = None
    usage = None
    text = ''
    tokens: list[dict[str, Any]] = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.startswith(b'data: '): continue
            data = line[6:].strip()
            if data == b'[DONE]': break
            event = json.loads(data)
            if event.get('error'): raise RuntimeError(event['error'])
            if event.get('usage'): usage = event['usage']
            for choice in event.get('choices', []):
                delta = choice.get('delta') or {}
                visible = delta.get('content') or ''
                lps = (choice.get('logprobs') or {}).get('content') or []
                if first_token is None and (lps or visible or delta.get('reasoning_content')):
                    first_token = time.monotonic()-start
                if first_visible is None and visible: first_visible = time.monotonic()-start
                text += visible
                tokens.extend(lps)
    if not usage or not tokens or first_token is None:
        raise ValueError('stream missing usage, token logprobs or first-token timing')
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    if not isinstance(cached,int): raise ValueError('missing cached_tokens: enable prompt token details')
    return {'case_sha256': digest(case), 'text':text, 'first': tokens[0], 'persisted_tag': tagged,
        'cached_tokens':cached, 'prompt_tokens':usage['prompt_tokens'], 'ttft_s':first_token,
        'first_visible_s':first_visible, 'elapsed_s':time.monotonic()-start}


def bucket_kl(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, float]:
    def distribution(token: dict[str, Any]) -> dict[tuple[int, ...],float]:
        out = {}
        for item in token['top_logprobs']:
            # Bytes distinguish display-equivalent tokens where supplied by the API.
            key = tuple(item['bytes']) if item.get('bytes') is not None else tuple(item['token'].encode())
            if key in out: raise ValueError('ambiguous duplicate token bytes in top-K distribution')
            out[key] = math.exp(item['logprob'])
        if not out or sum(out.values()) > 1.0001: raise ValueError('invalid top-K distribution')
        return out
    p, q = distribution(a), distribution(b)
    keys = p.keys() & q.keys()
    mass = min(sum(p[k] for k in keys), sum(q[k] for k in keys))
    ps = [p[k] for k in keys]; qs = [q[k] for k in keys]
    ps.append(max(0.0, 1-sum(ps))); qs.append(max(0.0, 1-sum(qs)))
    kl = sum(x*math.log(x/max(y,1e-300)) for x,y in zip(ps,qs) if x > 0)
    return max(kl,0.0), mass


def compare(packet: dict[str, Any], cold: dict[str, Any], reload: dict[str, Any], fresh: dict[str, Any],
            min_cached: int) -> list[dict[str, Any]]:
    for name, run in (('cold',cold),('reload',reload),('fresh',fresh)):
        if run['packet_sha256'] != digest(packet): raise ValueError(f'{name} packet mismatch')
        if len(run['results']) != len(packet['cases']): raise ValueError(f'{name} incomplete run')
    results = []
    all_codes = {code for case in packet['cases'] for code in case['codes'].values()}
    for case, a, b, f in zip(packet['cases'], cold['results'], reload['results'], fresh['results']):
        if any(r['case_sha256'] != digest(case) for r in (a,b,f)): raise ValueError('case identity mismatch')
        score_rows = [score(r['text'],case['codes'],all_codes-set(case['codes'].values())) for r in (a,b,f)]
        kl, mass = bucket_kl(f['first'], b['first'])
        floor, floor_mass = bucket_kl(f['first'],a['first'])
        limit = max(.005,2*floor)
        codes_ok = all(s['recall']==s['of'] and not s['foreign_codes'] for s in score_rows)
        cache_ok = a['cached_tokens']==0 and f['cached_tokens']==0 and b['cached_tokens']>=min_cached
        first_equal = (b['first'].get('bytes'),b['first']['token']) == (f['first'].get('bytes'),f['first']['token'])
        ok = codes_ok and cache_ok and first_equal and min(mass,floor_mass)>=.99 and kl <= limit
        results.append({'case_sha256':digest(case),'pass':ok,'codes':score_rows,
            'cached_tokens':b['cached_tokens'],'first_token_equal':first_equal,
            'coarse_kl':kl,'coarse_kl_cold_floor':floor,'coarse_kl_limit':limit,'common_mass':mass,
            'ttft_cold_s':a['ttft_s'],'ttft_reload_s':b['ttft_s'],
            'note':'coarse KL is a lower bound; use full-logit capture for full-vocabulary claims'})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action',required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--url',required=True); p.add_argument('--model',required=True)
    p.add_argument('--ctx',type=int,nargs='+',default=[24000,75000,150000])
    p.add_argument('--data-dir',type=Path,default=Path('/opt/dsv41/data/bench-data'))
    p.add_argument('--out',type=Path,required=True)
    p = sub.add_parser('run')
    p.add_argument('--packet',type=Path,required=True); p.add_argument('--url',required=True)
    p.add_argument('--fresh',action='store_true'); p.add_argument('--timeout',type=float,default=300)
    p.add_argument('--no-persist',action='store_true'); p.add_argument('--matter-ns')
    p.add_argument('--out',type=Path,required=True)
    p = sub.add_parser('compare')
    for name in ('packet','cold','reload','fresh','out'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--min-cached',type=int,default=384)
    args = parser.parse_args()
    if args.action == 'prepare':
        builder = Builder(args.url,args.model,args.data_dir)
        cases = []
        for n in args.ctx:
            prompt,codes = builder.build(n,6,231023+n)
            cases.append({'ctx_requested':n,'prompt':prompt,'codes':codes,'salt':secrets.token_hex(32)})
        save(args.out,{'model':args.model,'cases':cases})
        print(f'prepared {len(cases)} saved needle prompts/salts (tokenization only)')
    elif args.action == 'run':
        packet = json.loads(args.packet.read_text())
        results = []
        partial = args.out.with_name(args.out.name+'.partial.jsonl')
        partial.parent.mkdir(parents=True,exist_ok=True)
        with os.fdopen(os.open(partial,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as progress:
            for case in packet['cases']:
                results.append(ask(args.url,packet['model'],case,args.fresh,args.timeout,
                                   not args.no_persist,args.matter_ns))
                progress.write(json.dumps(results[-1])+'\n'); progress.flush()
                print(json.dumps({k:v for k,v in results[-1].items() if k not in ('text','first')}),flush=True)
        save(args.out,{'packet_sha256':digest(packet),'results':results})
    else:
        inputs = [json.loads(p.read_text()) for p in (args.packet,args.cold,args.reload,args.fresh)]
        results = compare(inputs[0],inputs[1],inputs[2],inputs[3],args.min_cached)
        save(args.out,results)
        for result in results: print(json.dumps(result))
        if not all(r['pass'] for r in results): raise SystemExit(1)


if __name__ == '__main__': main()
