"""Dry-default full FAST / full G2 coordinator stager. Local x86host only on execute."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import tarfile
import tempfile

from .data import sha256
from .full_admission import pilot_recipe, require_full_pilot
from .pilot_stage import ASSEMBLY, COORDINATOR, Settings, check_inputs, check_settings, git, obj
from .safety import GPU, SOLO_B

PHASES = ('stage', 'full-train', 'full-g2', 'finish')


@dataclass(frozen=True)
class Full:
    settings: Settings
    pilot_source: Path
    pilot_receipt: Path
    k2_general: Path


def validate(s: Full) -> None:
    check_settings(s.settings)
    for path in (s.pilot_source, s.pilot_receipt, s.k2_general):
        if not path.is_absolute() or any(c in str(path) for c in '\n\r'):
            raise ValueError('absolute single-line full-run paths required')
    for target in (s.settings.snapshot, s.settings.output):
        for protected in (s.pilot_source.parent, s.pilot_receipt.parent, s.k2_general):
            if target.resolve().is_relative_to(protected.resolve()) or protected.resolve().is_relative_to(target.resolve()):
                raise ValueError('full namespace overlaps pilot inputs')


def admission(s: Full) -> dict[str, str]:
    validate(s)
    inputs = check_inputs(s.settings)
    receipt = obj(s.pilot_receipt)
    from typing import cast
    spec = cast(dict[str, object], receipt['plan'])
    inputs['pilot'] = require_full_pilot(s.pilot_receipt, cast(str, spec['frozen_identity']), s.settings.manifest,
        s.settings.parity, s.settings.g1, 128, 8192, 32, s.pilot_source, {'dev': s.settings.k2_dev, 'heldout_legal': s.settings.k2_legal})
    inputs['recipe'] = pilot_recipe(s.pilot_receipt, inputs['pilot'])
    if (not s.pilot_receipt.parent.name.endswith('-p2-'+inputs['recipe'])
            or s.pilot_source.parent.name != s.pilot_receipt.parent.name):
        raise ValueError('pilot namespace does not match admitted recipe/source')
    if s.settings.output.name != s.pilot_receipt.parent.name+'-full' or s.settings.snapshot.name != s.settings.output.name:
        raise ValueError('full namespace must be pilot recipe namespace plus -full')
    from .pilot_sample import complete_rounds
    from .data import D2Blocks
    if not complete_rounds(D2Blocks(s.settings.manifest, 'heldout_general'), s.k2_general):
        raise ValueError('full G2 heldout_general split requires complete K2 coverage')
    inputs['k2_general'] = sha256(s.k2_general)
    return inputs


def commands(s: Full, resume: bool = False) -> dict[str, list[tuple[str, str]]]:
    validate(s)
    b = s.settings
    repo = b.snapshot/'repo'
    q = shlex.quote
    setup = '\n'.join(['set -euo pipefail', 'source ~/.config/ndm/paths.sh', 'umask 077',
        'export D2_TRAIN_ROOT='+q(str(b.snapshot)), 'export D2_GIT_SHA='+q(b.commit),
        'export D2_PYTHON='+q(str(b.python)),
        'export PYTHONPATH='+q(str(repo)+':'+str(repo/'offload/dspark')),
        'export OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=', 'cd '+q(str(repo))])+'\n'
    common = ['--manifest', str(b.manifest), '--ckpt', str(b.ckpt), '--pilot-receipt', str(s.pilot_receipt),
        '--pilot-source', str(s.pilot_source), '--k2-dev', str(b.k2_dev), '--k2-legal', str(b.k2_legal), '--parity-receipt', str(b.parity), '--g1-receipt', str(b.g1)]
    locked = 'bash '+q(str(repo/'offload/dspark/d2/train/run_pilot_locked.sh'))
    def gpu(uuid: str, name: str, module: str, args: list[str]) -> str:
        return 'D2_PILOT_GPU='+q(uuid)+' '+locked+' '+name+' -m '+module+' '+shlex.join(args)
    train = gpu(GPU, 'full-fast', 'offload.dspark.d2.train.full_train', common+['--device', 'cuda', '--fast-training',
        '--epochs', '3', '--batch-size', '128', '--chunk', '8192', '--rank', '32',
        '--output', str(b.output/'full-fast.pt')]+(['--resume'] if resume else []))
    def ev(split: str, k2: Path, uuid: str) -> str:
        return gpu(uuid, 'full-g2-'+split, 'offload.dspark.d2.train.full_eval', ['score', *common,
            '--adapter', str(b.output/'full-fast.pt'), '--k2', str(k2), '--split', split,
            '--out', str(b.output/(split+'.json'))]+(['--resume'] if resume else []))
    merge = shlex.join([str(b.python), '-m', 'offload.dspark.d2.train.full_eval', 'merge', '--parts',
        *[str(b.output/(split+'.json')) for split in ('dev', 'heldout_legal', 'heldout_general')],
        '--out', str(b.output/'g2.json')])
    return {'full-train': [('full-fast', setup+train)],
        'full-g2': [('g2-dev', setup+ev('dev', b.k2_dev, GPU)),
                    ('g2-legal', setup+ev('heldout_legal', b.k2_legal, SOLO_B)),
                    ('g2-general', setup+ev('heldout_general', s.k2_general, SOLO_B))],
        'finish': [('g2-finish', setup+merge)]}


def execute(s: Full, phase: str, job: str | None, resume: bool, attempt: int = 1) -> None:
    b = s.settings
    if attempt < 1:
        raise ValueError('attempt must be positive')
    if os.environ.get('D2_COORDINATOR') != COORDINATOR or socket.gethostname().split('.')[0] not in ('x86host', 'x86host'):
        raise ValueError('--execute requires the local x86host coordinator')
    if b.assembly != ASSEMBLY:
        raise ValueError('execution requires the reviewed -c2 assembly')
    inputs = admission(s)
    jobs = commands(s)
    if git(b, 'rev-parse', 'HEAD') != b.commit or git(b, 'status', '--porcelain'):
        raise ValueError('clean reviewed source HEAD required')
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    os.umask(0o077)
    record = {'schema': 'dsv41.d2.full-stage.v1', 'commit': b.commit, 'inputs': inputs,
              'output': str(b.output), 'commands': {k: [list(v) for v in rows] for k, rows in jobs.items()}}
    if phase == 'stage':
        if resume or job is not None or b.snapshot.exists() or b.output.exists():
            raise FileExistsError('stage requires NEW full namespaces and no resume/job')
        b.snapshot.mkdir(parents=True)
        (b.snapshot/'repo').mkdir()
        with tempfile.TemporaryFile(dir=b.snapshot) as archive:
            subprocess.run(['git', '-C', str(b.source), 'archive', b.commit], env=env, check=True, stdout=archive)
            archive.seek(0)
            with tarfile.open(fileobj=archive) as tar:
                tar.extractall(b.snapshot/'repo', filter='data')
        (b.snapshot/'SOURCE_COMMIT').write_text(b.commit+'\n')
        from .pilot import write_json
        write_json(b.snapshot/'STAGED.json', record)
        (b.snapshot/'logs').mkdir()
        b.output.mkdir(parents=True)
        return
    if obj(b.snapshot/'STAGED.json') != record or (b.snapshot/'SOURCE_COMMIT').read_text().strip() != b.commit:
        raise ValueError('full staged source/input/command contract changed')
    tracked = set(git(b, 'ls-tree', '-r', '--name-only', b.commit).splitlines())
    repo = b.snapshot/'repo'
    if {str(p.relative_to(repo)) for p in repo.rglob('*') if p.is_file()} != tracked:
        raise ValueError('full staged source file set changed')
    if any(sha256(b.source/p) != sha256(repo/p) for p in tracked):
        raise ValueError('full staged source bytes changed')
    if phase == 'full-g2':
        from .full_eval import completed
        completed(b.output/'full-fast.pt', b.manifest, inputs['pilot'], pilot_recipe(s.pilot_receipt, inputs['pilot']))
    if phase == 'finish':
        from .full_eval import verdict
        from .full_eval import completed
        adapter_hash = completed(b.output/'full-fast.pt', b.manifest, inputs['pilot'], pilot_recipe(s.pilot_receipt, inputs['pilot']))
        result = verdict([obj(b.output/(split+'.json')) for split in ('dev', 'heldout_legal', 'heldout_general')])
        from .pilot import sources
        if (result['source_sha256'] != sources() or result['adapter_sha256'] != adapter_hash or result['manifest_sha256'] != inputs['manifest']
                or result['pilot_sha256'] != inputs['pilot'] or result['recipe'] != inputs['recipe']):
            raise ValueError('G2 parts differ from staged full candidate/inputs')
    available = dict(commands(s, resume)[phase])
    if job is None or job not in available:
        raise ValueError('choose exactly one --job from '+str(list(available)))
    # General and legal share SOLO_B; launch them sequentially, never race a lease.
    name = job+('-resume-'+str(attempt) if resume else '')
    session = 'd2full-'+b.output.name+'-'+name
    probe = subprocess.run(['tmux', 'has-session', '-t', '='+session], capture_output=True)
    if probe.returncode == 0:
        raise ValueError('session already exists: '+session)
    if probe.returncode != 1:
        raise RuntimeError('tmux preflight failed: '+probe.stderr.decode())
    script = b.snapshot/(name+'.sh')
    with script.open('x') as f:
        f.write(available[job]+'\n')
    log = b.snapshot/'logs'/(name+'.coordinator.log')
    subprocess.run(['tmux', 'new-session', '-d', '-s', session,
        'bash '+shlex.quote(str(script))+' >'+shlex.quote(str(log))+' 2>&1'], check=True)


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--check-inputs', action='store_true')
    p.add_argument('--phase', choices=PHASES, default='stage')
    p.add_argument('--job')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--attempt', type=int, default=1)
    p.add_argument('--commit', required=True)
    p.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[4])
    p.add_argument('--snapshot', type=Path, default=Path.home()/'dsv41-offload/d2/full/decode-d2pass2-r1-p2-ace-full')
    p.add_argument('--output', type=Path, default=Path('/opt/dsv41/x86host-data/d2/full/decode-d2pass2-r1-p2-ace-full'))
    p.add_argument('--assembly', type=Path, default=ASSEMBLY)
    p.add_argument('--ckpt', type=Path, default=Path('/opt/dsv41/x86host-data/native-1017'))
    p.add_argument('--python', type=Path, default=Path.home()/'dsv41-offload/e/venv/bin/python')
    p.add_argument('--parity', type=Path, default=Path('/opt/dsv41/x86host-data/d2/train/parity50-solo225/parity.json'))
    for name in ('g1', 'k2-dev', 'k2-legal', 'k2-general'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--pilot-source', type=Path, default=Path.home()/'dsv41-offload/d2/pilot/decode-d2pass2-r1-p2-ace/repo')
    p.add_argument('--pilot-receipt', type=Path, default=Path('/opt/dsv41/x86host-data/d2/pilot/decode-d2pass2-r1-p2-ace/receipt.json'))
    a = p.parse_args()
    s = Full(Settings(**{k: getattr(a, k) for k in Settings.__dataclass_fields__ if k not in ('ref_gpu', 'fast_gpu')}),
             a.pilot_source, a.pilot_receipt, a.k2_general)
    validate(s)
    if a.execute:
        execute(s, a.phase, a.job, a.resume, a.attempt)
    else:
        if a.check_inputs:
            print(json.dumps({'validated_inputs': admission(s)}, indent=2))
        print(json.dumps({'dry': True, 'order': PHASES, 'commands': commands(s, a.resume),
            'result': str(s.settings.output/'g2.json'), 'pending': 'genuine pilot PASS; no commands executed'}, indent=2))


if __name__ == '__main__':
    main()
