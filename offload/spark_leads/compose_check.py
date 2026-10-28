#!/usr/bin/env python3
"""Read K1's pinned overlay from git; compose X1 patches in scratch and report hashes. Never stages or deploys."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import cast

ROOT=Path(__file__).resolve().parents[2]
PKG=Path(__file__).resolve().parent
K1='87b08b5c486e68458223ea38dca6bb911b678400'


def git_file(ref: str, name: str) -> bytes:
    return subprocess.check_output(['git','-C',str(ROOT),'show',f'{ref}:{name}'])


def check(ref: str, base: Path) -> dict[str,object]:
    if re.fullmatch('[0-9a-f]{40}',ref) is None:
        raise ValueError('K1 must be pinned by full commit SHA')
    raw=git_file(ref,'offload/kvpack/MANIFEST.json')
    manifest=json.loads(raw)
    with tempfile.TemporaryDirectory(prefix='x1-compose-') as tmp:
        tree=Path(tmp)
        for entry in manifest['files']:
            name=entry['path']; path=tree/name
            if not path.resolve().is_relative_to(tree.resolve()):
                raise ValueError('invalid K1 manifest path')
            data=git_file(ref,'offload/kvpack/overlay/'+name)
            if hashlib.sha256(data).hexdigest()!=entry['overlay_sha256']:
                raise ValueError(f'K1 overlay hash differs: {name}')
            path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
        patches=['0075-v5.patch','0078-remote.patch']
        for stem,patch in zip(['0075','0078'],patches,strict=True):
            pins=json.loads((PKG/f'patches/{stem}-bases.json').read_text())['bases']
            for name,sha in pins.items():
                if sha is None:
                    continue
                src=(ROOT if name.startswith('offload/') else base)/name
                data=src.read_bytes()
                if hashlib.sha256(data).hexdigest()!=sha:
                    raise ValueError(f'X1 base changed: {name}')
                path=tree/name
                if not path.exists():
                    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
            subprocess.run(['patch','--batch','--fuzz=0','--no-backup-if-mismatch','-p1','-d',tmp,
                            '-i',str(PKG/'patches'/patch)],check=True,capture_output=True,text=True)
        if list(tree.rglob('*.rej')) or list(tree.rglob('*.orig')):
            raise ValueError('patch left rejects/backups')
        files={}
        for path in sorted(tree.rglob('*')):
            if path.is_file():
                if path.suffix=='.py': compile(path.read_text(),str(path),'exec')
                if path.suffix=='.sh': subprocess.run(['bash','-n',str(path)],check=True)
                files[str(path.relative_to(tree))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return {'k1_ref':ref,'k1_manifest_sha256':hashlib.sha256(raw).hexdigest(),'patches':patches,'fuzz':0,
            'patch_sha256':{p:hashlib.sha256((PKG/'patches'/p).read_bytes()).hexdigest() for p in patches},
            'combined_files':files,'qualification':'text composition and Python/shell syntax only; no JIT/GB10/service test'}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k1-ref',default=K1); parser.add_argument('--base',type=Path,default=Path('/opt/dsv41/data/src-image'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); result=check(args.k1_ref,args.base)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(f"K1 {args.k1_ref} + X1: zero fuzz, {len(cast(dict[str,str],result['combined_files']))} files, Python/shell syntax passed")
    return 0


if __name__=='__main__': raise SystemExit(main())
