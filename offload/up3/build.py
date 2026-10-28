"""CPU-only exact production/arm rebuild. No SSH, imports from donor, or deployment."""
from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
import subprocess
import tempfile
from pathlib import Path
from typing import TypedDict, cast

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
IMAGE = Path('/opt/dsv41/data/src-image')
PRODUCTION = '596ba1cf4e3b11be29b41a57e7f69a5d8efa4799'
TAG = 'prod-k1-327-ops1-20261026'
DIST = '/usr/local/lib/python3.12/dist-packages/'
ARMS = ('control', 'split', 'cpu-pin', 'vor', 'k5', 'dual-ring', 'fixed2', 'defer', 'ring-defer')
FILES = {
    'cpu-pin': ('vllm_exl3_moe/rdma.py', 'vllm_exl3_moe/up3_rdma_pin.py'),
    'split': ('vllm_exl3_moe/forward.py', 'vllm_exl3_moe/smoe.py'),
    'vor': ('vllm/v1/attention/backends/mla/sparse_swa.py',
            'vllm/models/deepseek_v41/nvidia/model_state.py',
            'vllm/models/deepseek_v41/common/ops/swa_fused.py'),
}
PATCHES = {'split': '0064.patch', 'vor': '0067.patch', 'cpu-pin': 'rdma-pin.patch'}
ENV = {'control': {}, 'split': {'VLLM_EXL3_MOE_SPLIT_MAX_TOKENS': '63'},
       'cpu-pin': {'EXL3_CPU_PIN': '5-9,15-19'}, 'vor': {'VLLM_DSV41_VOR_FUSED': '1'},
       'k5': {}, 'dual-ring': {'EXL3_RDMA_DEVICES': 'rocep1s0f1,roceP2p1s0f1'},
       # DEP1 9b72534 fixed depth 2 (offload/launch/configs/DEP1-depth.env): adaptive verification off, K5 schedule (SPEC).
       'fixed2': {'VLLM_DSPARK_DEPTH_POLICY': 'fixed:2', 'VLLM_DSV41_ADAPTIVE_K': '0'},
       # 0051 indexer defer: production K1 attention.py carries it, default off (set nowhere in 596ba1c); bit-identical.
       'defer': {'VLLM_DSV41_INDEXER_DEFER': '1'},
       'ring-defer': {'EXL3_RDMA_DEVICES': 'rocep1s0f1,roceP2p1s0f1', 'VLLM_DSV41_INDEXER_DEFER': '1'}}
SPEC = '{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,4,5]]}'


class Lock(TypedDict):
    production: str
    inputs: dict[str, str]


class Manifest(TypedDict):
    production: str
    arm: str
    inputs_sha256: str
    files: dict[str, str]
    fuzz: int


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def production_file(name: str) -> bytes:
    return subprocess.check_output(['git', 'show', f'{PRODUCTION}:{name}'], cwd=ROOT)


def inputs() -> dict[str, bytes]:
    values = {'git:' + name: production_file(name) for name in (
        'dsv41.env', 'serve.sh', 'start.sh', 'stop.sh', 'firewall.sh',
        'offload/launch/warmup.py', 'offload/launch/warmup_serve.sh',
        'offload/launch/dsv41_tp2_offload.sh', 'offload/launch/window.sh',
        'offload/kvpack/MANIFEST.json')}
    # Freeze ALL current production K1 and remote package bytes, not the stale UP1 copy.
    k1 = json.loads(values['git:offload/kvpack/MANIFEST.json'])
    for row in k1['files']:
        name = 'offload/kvpack/overlay/' + row['path']
        values['git:' + name] = production_file(name)
        if digest(values['git:' + name]) != row['overlay_sha256']:
            raise ValueError('production K1 manifest mismatch: ' + name)
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', PRODUCTION,
        'offload/dspark/spark/remote_dspark', 'offload/dspark/spark/overlay'], cwd=ROOT, text=True).splitlines()
    for name in names:
        values['git:' + name] = production_file(name)
    for arm, names_tuple in FILES.items():
        for name in names_tuple:
            if (IMAGE / name).exists():
                values['image:' + name] = (IMAGE / name).read_bytes()
        values['patch:' + arm] = (HERE / 'patches' / PATCHES[arm]).read_bytes()
    values['source:rdma_pin.py'] = (HERE / 'rdma_pin.py').read_bytes()
    return values


def verify_inputs() -> Lock:
    lock = cast(Lock, json.loads((HERE / 'inputs.json').read_text()))
    actual = {key: digest(value) for key, value in inputs().items()}
    if lock != {'production': PRODUCTION, 'inputs': actual}:
        raise ValueError('UP3 production/image/patch input drift')
    # The checked-out K1 package must also match the production pin. Never edit/rebuild K1 here.
    for key, sha in actual.items():
        if key.startswith(('git:offload/kvpack/',)):
            if digest((ROOT / key[4:]).read_bytes()) != sha:
                raise ValueError('UP3 checkout differs from production input: ' + key)
    return lock


def build(arm: str, tree: Path) -> Manifest:
    if arm not in ARMS:
        raise ValueError('unknown isolated arm')
    verify_inputs()
    if tree.exists() and any(tree.iterdir()):
        raise ValueError('build destination must be empty')
    tree.mkdir(parents=True, exist_ok=True)
    if arm in FILES:
        for name in FILES[arm]:
            if (IMAGE / name).exists():
                target = tree / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((IMAGE / name).read_bytes())
        if arm == 'cpu-pin':
            (tree / 'vllm_exl3_moe/up3_rdma_pin.py').write_bytes((HERE / 'rdma_pin.py').read_bytes())
        patch = HERE / 'patches' / PATCHES[arm]
        result = subprocess.run(['patch', '--batch', '--forward', '--fuzz=0', '--no-backup-if-mismatch',
            '-p1', '-d', str(tree), '-i', str(patch)], text=True, capture_output=True)
        if result.returncode or any(word in result.stdout.lower() for word in ('fuzz', 'offset', 'failed')):
            raise ValueError('UP3 patch not exact: ' + result.stdout + result.stderr)
    files = {str(p.relative_to(tree)): digest(p.read_bytes()) for p in tree.rglob('*') if p.is_file()}
    if set(files) != set(FILES.get(arm, ())):
        raise ValueError('unexpected overlay path set')
    with tempfile.TemporaryDirectory(prefix='up3-compile-') as tmp:
        for name in files:
            py_compile.compile(str(tree / name), cfile=str(Path(tmp) / 'check.pyc'), doraise=True)
    return {'production': PRODUCTION, 'arm': arm, 'inputs_sha256': digest((HERE / 'inputs.json').read_bytes()),
            'files': files, 'fuzz': 0}


def verify(arm: str) -> Manifest:
    manifest = cast(Manifest, json.loads((HERE / 'manifests' / (arm + '.json')).read_text()))
    with tempfile.TemporaryDirectory(prefix='up3-verify-') as tmp:
        if build(arm, Path(tmp) / 'tree') != manifest:
            raise ValueError('UP3 manifest drift')
    tree = HERE / 'overlay' / arm
    actual = {str(p.relative_to(tree)): digest(p.read_bytes()) for p in tree.rglob('*') if p.is_file()}
    if actual != manifest['files']:
        raise ValueError('UP3 overlay drift')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output:
        if args.output.exists():
            raise ValueError('fresh output directory required')
        args.output.mkdir(parents=True)
        for arm in ARMS:
            manifest = build(arm, args.output / arm)
            (args.output / (arm + '.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    else:
        for arm in ARMS:
            verify(arm)
        print(f'UP3: {len(ARMS)} isolated arms verified; production inputs pinned, fuzz=0, offset=0; OFF has no replacement bytes')


if __name__ == '__main__':
    main()
