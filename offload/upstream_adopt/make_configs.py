"""Render isolated opt-in UP1 arms on K1's async process/nap/lazy D-kv6 base; no launcher execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import cast

from offload.upstream_adopt.build_overlay import HERE, ROOT, Manifest, mounts, verify_inputs

CFG = ROOT/'offload/launch/configs'
PROFILES: dict[str,dict[str,str]] = {
    'control':{}, 'split':{'SPLIT':'63'}, 'vor':{'VOR':'1'}, 'replay':{'REPLAY':'1'},
    'k5':{'K5':'1'}, 'dual-ring':{'DUAL_RING':'1'},
    'cpu-pin':{'CPU_PIN':'1'},
    'core':{'SPLIT':'63','VOR':'1','REPLAY':'1'},
    'all':{'SPLIT':'63','VOR':'1','REPLAY':'1','K5':'1','DUAL_RING':'1','CPU_PIN':'1'},
}
FLAGS = {'SPLIT':'0|63','VOR':'0|1|swa|copy','REPLAY':'0|1',
         'K5':'0|1','DUAL_RING':'0|1','CPU_PIN':'0|1'}


def render(profile: str) -> str:
    verify_inputs()
    base = (CFG/'D-kv6-async-pack.env').read_text()
    manifest = cast(Manifest,json.loads((HERE/'MANIFEST.json').read_text()))
    composed = mounts(manifest)
    match = re.search(r'^OVERLAYS="(.*)"$',base,re.M)
    assert match is not None
    existing = [tuple(pair.split('=',1)) for pair in match.group(1).split()]
    replacing = {dst for _,dst in composed}
    pairs = [(src,dst) for src,dst in existing if dst not in replacing]+composed
    assert len({dst for _,dst in pairs}) == len(pairs)
    base = base[:match.start()]+f'OVERLAYS="{" ".join(f"{src}={dst}" for src,dst in pairs)}"'+base[match.end():]
    base = re.sub(r'^LABEL=.*$',f'LABEL=D-kv6-async-up1-{profile}',base,flags=re.M)
    lines = [f'# UP1 {profile}: composed K1 + 0064 + 0067 + tap-aware0075; CPU qualified only.',
             '# Generated, no services started. Runtime UP1_* variables override defaults; 0 disables a feature.',base.rstrip()]
    for flag,choices in FLAGS.items():
        value = PROFILES[profile].get(flag,'0')
        lines += [f'UP1_{flag}=${{UP1_{flag}-{value}}}',
                  f'case "$UP1_{flag}" in {choices}) ;; *) echo "invalid UP1_{flag}=$UP1_{flag}" >&2; return 1 ;; esac']
    lines += [
        'EXP_DOCKER_ENV="$EXP_DOCKER_ENV VLLM_EXL3_MOE_SPLIT_MAX_TOKENS=$UP1_SPLIT VLLM_DSV41_VOR_FUSED=$UP1_VOR VLLM_DSV41_REPLAY_SKIP=$UP1_REPLAY"',
        'if [ "$UP1_K5" = 1 ]; then',
        '  [ -z "$K1_NODROP" ] || { echo "UP1_K5 with K1_NODROP needs an explicit combined spec" >&2; return 1; }',
        '''  SPEC_OVERRIDE='{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,4,5]]}' ''',
        'fi',
        'if [ "$UP1_DUAL_RING" = 1 ]; then EXP_DOCKER_ENV="$EXP_DOCKER_ENV EXL3_RDMA_DEVICES=rocep1s0f1,roceP2p1s0f1"; fi',
        'if [ "$UP1_CPU_PIN" = 1 ]; then EXP_DOCKER_ENV="$EXP_DOCKER_ENV EXL3_CPU_PIN=5-9,15-19"; fi',
    ]
    return '\n'.join(line.rstrip() for line in lines)+'\n'



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    bodies = {profile:render(profile) for profile in PROFILES}
    expected = {CFG/f'D-kv6-async-up1-{profile}.env' for profile in bodies}
    extra = set(CFG.glob('D-kv6-async-up1-*.env')) - expected
    if extra:
        raise ValueError(f'obsolete or held UP1 configs: {sorted(p.name for p in extra)}')
    for profile,body in bodies.items():
        path = CFG/f'D-kv6-async-up1-{profile}.env'
        if args.check:
            if path.read_text() != body:
                raise ValueError(f'config drift: {path.name}')
        else:
            path.write_text(body)
    print(f'{len(bodies)} configs {"verified" if args.check else "written"}')


if __name__ == '__main__':
    main()
