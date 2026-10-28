"""Coordinator-only oracle between experiment stop/staging and server startup.

No model, listener or second CUDA process beside a server. Nested SSH inherits
the existing scoped owner. The sole container name is dsv41-serve, so the
existing stop/quiet recovery also covers a lost docker-run acknowledgement.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import cast

from offload.prefill_maxlen.evidence import Json, record
from offload.tv1 import build_overlay as overlay
from offload.tv1.candidate_cases import validate_oracle_receipt

IMAGE = 'sha256:435a5236ce3a7db9d45f5a2c483817d60ea629c4ea0a0b3af33da2ef8122334b'


def receipt_path(run: str) -> Path:
    if re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]{0,80}', run) is None:
        raise ValueError('TV1 invalid prestart oracle run')
    return Path.home() / 'dsv41-offload-run' / f'tv1-oracle-{run}.json'


def remote_script(extra: str, env: str) -> str:
    """Only pinned, explicit mounts; expand the receiving host's HOME, never eval."""
    mounts = shlex.split(extra)
    if len(mounts) % 2 or any(mounts[i] != '-v' for i in range(0, len(mounts), 2)):
        raise ValueError('TV1 oracle requires explicit volume pairs')
    for spec in mounts[1::2]:
        parts = spec.split(':')
        if len(parts) != 3 or parts[2] not in ('ro', 'rw'):
            raise ValueError('TV1 malformed oracle mount')
        if parts[2] == 'rw' and parts[1] != '/sp1-profile':
            raise ValueError('TV1 unexpected writable oracle mount')
    variables = shlex.split(env)
    if ('VLLM_TV1_CANDIDATE=1' not in variables or 'VLLM_TV1_COMPACT=0' not in variables
            or any('=' not in v for v in variables)):
        raise ValueError('TV1 prestart requires candidate-only environment')
    return f'''import json,os,pathlib,subprocess,sys
def checked(argv):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=15)
 if p.returncode: raise ValueError('TV1 prestart command failed: '+p.stderr)
 return p.stdout.strip()
if checked(['docker','ps','-aq','--filter','name=^/dsv41-serve$']):
 raise ValueError('TV1 oracle refuses an existing server/container')
if checked(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']):
 raise ValueError('TV1 oracle refuses competing GPU processes')
image=checked(['docker','image','inspect','--format','{{{{.Id}}}}',{IMAGE!r}])
if image!={IMAGE!r}: raise ValueError('TV1 prestart image pin differs')
mounts={mounts!r}
for i in range(1,len(mounts),2):
 source,sep,rest=mounts[i].partition(':')
 if source.startswith('$HOME/'): source=str(pathlib.Path.home())+source[5:]
 if '$' in source or not source.startswith('/'): raise ValueError('TV1 unsafe oracle source')
 mounts[i]=source+sep+rest
argv=['docker','run','--rm','--name','dsv41-serve','--gpus','all','--ipc','host',
      '--network','none','-i',*mounts]
for value in {variables!r}: argv+=['-e',value]
argv+=['--entrypoint','python3',image,'-']
p=subprocess.run(argv,input=sys.stdin.read(),capture_output=True,text=True,timeout=115)
if p.returncode:
 sys.stderr.write(p.stdout+p.stderr)
 raise ValueError('TV1 isolated installed oracle failed rc'+str(p.returncode))
rows=[json.loads(line) for line in p.stdout.splitlines() if line.startswith('{{')]
if len(rows)!=1: raise ValueError('TV1 isolated oracle receipt missing/ambiguous')
if checked(['docker','ps','-aq','--filter','name=^/dsv41-serve$']):
 raise ValueError('TV1 oracle container remained after exit')
if checked(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']):
 raise ValueError('TV1 oracle compute remained after exit')
print(json.dumps({{'image':image,'server_absent_before':True,'compute_absent_after':True,
                  'oracle':rows[0]}}))
'''


def validate(receipt: object, run: str) -> dict[str, Json]:
    data = record(cast(Json, receipt))
    if data.get('type') != 'tv1-prestart-oracles-v1' or data.get('run') != run:
        raise ValueError('TV1 prestart receipt run/type differs')
    ranks = record(data.get('ranks'))
    if set(ranks) != {'sparkhead', 'sparkworker-wired'}:
        raise ValueError('TV1 prestart requires both SM121 ranks')
    for value in ranks.values():
        rank = record(value)
        if (rank.get('image') != IMAGE or rank.get('server_absent_before') is not True
                or rank.get('compute_absent_after') is not True):
            raise ValueError('TV1 oracle isolation/image proof incomplete')
        validate_oracle_receipt(rank.get('oracle'))
    return data


def main() -> None:
    # This entry is reachable only through the scoped experiment launcher hook.
    run = os.environ['TV1_ORACLE_RUN']
    destination = receipt_path(run)
    if destination.exists():
        raise ValueError('TV1 oracle refuses stale/repeated receipt')
    if not os.environ.get('SCOPED_REMOTE_DEADLINE'):
        raise ValueError('TV1 oracle requires scoped receiving-hop ownership')
    if os.environ.get('TV1_ROLE') != 'candidate':
        raise ValueError('TV1 prestart oracle only belongs to candidate')
    tree = overlay.safe(overlay.ROOT, os.environ['TV1_TREE'])
    manifest = cast(overlay.Manifest, json.loads(tree.with_suffix('.manifest.json').read_text()))
    if manifest['switches'] != ['candidate']:
        raise ValueError('TV1 prestart candidate manifest differs')
    overlay.verify(tree, manifest)
    # Hash actual installed K1+candidate files BEFORE importing torch or helpers.
    from offload.launch.windows.tv1_window import source_verifier
    program = source_verifier(manifest) + (overlay.HERE / 'gb10_oracle.py').read_text().replace(
        'from __future__ import annotations\n', '')
    script = remote_script(os.environ['EXTRA_DOCKER_ARGS'], os.environ['EXP_DOCKER_ENV'])
    ranks: dict[str, Json] = {}
    for host in ('sparkhead', 'sparkworker-wired'):
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', host,
                                 'python3 -c ' + shlex.quote(script)], input=program,
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise ValueError(f'TV1 isolated oracle {host} rc{result.returncode}: '
                             + result.stdout + result.stderr)
        ranks[host] = cast(Json, json.loads(result.stdout))
    receipt = validate({'type': 'tv1-prestart-oracles-v1', 'run': run, 'ranks': ranks}, run)
    with destination.open('x') as handle:
        handle.write(json.dumps(receipt) + '\n')
    print('TV1 BOTH ISOLATED SM121 ORACLES IDENTICAL', flush=True)


if __name__ == '__main__':
    main()
