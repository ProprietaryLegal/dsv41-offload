"""Dry-default LOCAL x86host staging. Never connects to a host or controls services."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import tarfile
import tempfile
from typing import cast

from ..copy_binding import verify_copy
from .data import D2Blocks, sha256
from .pilot import RECIPES, Recipe, manifest_is_synthetic, validate_live_g1
from .safety import GPU, SOLO_B

COORDINATOR = 'Deepseek/coordinator'
# The verified copy retry (COPY VERIFIED 2026-10-25T06:41:07Z, receipt 7c7f4944...08b7). The first copy attempt's
# decode-d2pass2-r1 tree stopped at the Spark memory guard, has no receipt and is never used.
ASSEMBLY = Path('/opt/dsv41/x86host-data/d2/decode-d2pass2-r1-c2')
FIRST_NAME = 'decode-d2pass2-r1'
FIRST_OUTPUT = Path('/opt/dsv41/x86host-data/d2/pilot')/FIRST_NAME
FIRST_PLAN_SHA256 = 'c4732bc224944af7df7de1d1fe67dbf6efd2d6bcd9d49bdc94af584085074402'
PHASES = ('stage', 'prepare', 'train', 'evaluate', 'finish')


@dataclass(frozen=True)
class Settings:
    source: Path
    commit: str
    snapshot: Path
    output: Path
    assembly: Path
    parity: Path
    g1: Path
    k2_dev: Path
    k2_legal: Path
    ckpt: Path
    python: Path
    ref_gpu: str = GPU
    fast_gpu: str = SOLO_B

    @property
    def manifest(self) -> Path:
        return self.assembly/'shards/MANIFEST.json'


@dataclass(frozen=True)
class PilotSettings(Settings):
    """Pilot-only fields; preserve full_stage's shared Settings CLI contract."""
    recipe: Recipe = "kl"
    run_tag: str = FIRST_NAME
    previous_plan: Path = FIRST_OUTPUT/"plan.json"


def namespace(recipe: Recipe, run_tag: str) -> str:
    if recipe not in RECIPES or not re.fullmatch('[a-z0-9][a-z0-9-]{0,39}', run_tag):
        raise ValueError('recipe/run tag must be a supported recipe and short lowercase identifier')
    return run_tag+'-p2-'+recipe


def session(s: PilotSettings, name: str) -> str:
    namespace(s.recipe, s.run_tag)
    return 'd2pilot2-'+s.recipe+'-'+s.run_tag+'-'+name


def obj(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('expected JSON object: '+str(path))
    return cast(dict[str, object], value)


def inside(root: Path, relative: str) -> Path:
    path = (root/relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError('receipt path escapes assembly: '+relative)
    return path


def check_settings(s: Settings) -> None:
    if isinstance(s, PilotSettings):
        namespace(s.recipe, s.run_tag)
    if not re.fullmatch('[0-9a-f]{40}', s.commit):
        raise ValueError('reviewed commit must be a full lowercase SHA1')
    if (s.ref_gpu, s.fast_gpu) != (GPU, SOLO_B):
        raise ValueError('REF/FAST must pin the two prescribed solo UUIDs; production pair forbidden')
    for path in (s.source, s.snapshot, s.output, s.assembly, s.python, s.ckpt, s.parity, s.g1, s.k2_dev, s.k2_legal):
        if not path.is_absolute() or any(c in str(path) for c in '\n\r'):
            raise ValueError('absolute single-line paths required')
    if isinstance(s, PilotSettings) and (not s.previous_plan.is_absolute() or any(c in str(s.previous_plan) for c in '\n\r')):
        raise ValueError('absolute single-line previous-plan path required')
    # Stage is a NEW sibling tree, never an in-place checkout or dataset change.
    for target in (s.snapshot, s.output):
        for protected in (FIRST_OUTPUT, Path.home()/'dsv41-offload/d2/pilot'/FIRST_NAME):
            if target.resolve().is_relative_to(protected.resolve()) or protected.resolve().is_relative_to(target.resolve()):
                raise ValueError('second pilot overlaps first pilot namespace')
        for protected in (s.assembly, s.source, s.ckpt):
            if target.resolve().is_relative_to(protected.resolve()) or protected.resolve().is_relative_to(target.resolve()):
                raise ValueError('stage/output overlaps a protected input tree')
    if s.snapshot.resolve().is_relative_to(s.output.resolve()) or s.output.resolve().is_relative_to(s.snapshot.resolve()):
        raise ValueError('snapshot and pilot output must be separate trees')


def check_inputs(s: Settings) -> dict[str, str]:
    """Revalidate local receipt bytes and CPU gates; no models, GPU or network."""
    from .. import dataset, decode_dataset
    from ..g1 import score, Sequence
    from .pilot_sample import complete_rounds
    from .replay import g1_drafts

    check_settings(s)
    receipt_path = s.assembly/'ASSEMBLED.json'
    receipt = obj(receipt_path)
    if (receipt.get('schema') != 'd2.decode-assembly.v1' or receipt.get('g1a_pass') is not True
            or receipt.get('synthetic') is not False or manifest_is_synthetic(receipt_path)):
        raise ValueError('requires genuine nonsynthetic ASSEMBLED.json G1(a) pass')
    if manifest_is_synthetic(s.manifest):
        raise ValueError('synthetic manifest cannot stage a real pilot')
    manifest = obj(s.manifest)
    if manifest.get('taps_source') != 'decode_capture':
        raise ValueError('expected decode_capture shards')
    splits = cast(dict[str, object], manifest.get('splits', {}))
    if not {'train', 'dev', 'heldout_legal'} <= set(splits):
        raise ValueError('missing train/dev/heldout_legal split; amendment-7 population required')
    gate = obj(s.g1)
    live = validate_live_g1(s.g1, s.manifest)
    # Fail on missing/mismatched shards before the more expensive provenance
    # recheck. D2Blocks validates every split, including cross-split leakage.
    populations = {name: D2Blocks(s.manifest, name) for name in ('train', 'dev', 'heldout_legal')}
    required = {'export/VERIFIED.json', 'decode-plan/DECODE.json', 'decode-plan/plan.jsonl',
                'shards/MANIFEST.json', 'shards/SHA256SUMS', 'G1A.log', 'AGGREGATE.json'}
    outputs = cast(dict[str, str], receipt.get('outputs', {}))
    if set(outputs) != required:
        raise ValueError('assembly output receipt coverage differs')
    inputs = cast(dict[str, str], receipt.get('inputs', {}))
    if not inputs:
        raise ValueError('assembly input receipt is empty')
    for relative, expected in {**outputs, **inputs}.items():
        if sha256(inside(s.assembly, relative)) != expected:
            raise ValueError('assembly receipt hash mismatch: '+relative)
    if verify_copy(s.assembly) != receipt.get('copy_receipt_sha256'):
        raise ValueError('assembly copy receipt mismatch')
    decode = decode_dataset.decode_identity(s.assembly/'decode-plan')
    replay = cast(dict[str, object], manifest.get('replay', {}))
    if (decode.get('top20_source') != 'generation_verified'
            or replay.get('decode_sha256') != sha256(s.assembly/'decode-plan/DECODE.json')
            or replay.get('copy_receipt_sha256') != receipt['copy_receipt_sha256']):
        raise ValueError('generation-time top20/decode provenance mismatch')
    if dataset.verify_dir(str(s.manifest.parent))['errors'] != 0:
        raise ValueError('shard structure/hash/probability check failed')
    if any(row['plumbing_errors'] for row in dataset.plumbing(str(s.manifest.parent)).values()):
        raise ValueError('G1(a) plumbing recheck failed')
    if len(populations['train']) < 20000:
        raise ValueError('fewer than 20000 unique training blocks')
    for split, path in (('dev', s.k2_dev), ('heldout_legal', s.k2_legal)):
        if not complete_rounds(populations[split], path):
            raise ValueError('no complete fixed-K2 rounds in '+split)
    # Recompute the hash-bound G1 score from
    # these shards and the exact K2 file, rather than trusting a borrowed PASS.
    from safetensors import safe_open
    seqs: dict[str, Sequence] = {}
    for path, row, req in populations['dev'].sequences:
        offset, length, gen = row[:3]
        with safe_open(path, framework='pt', device='cpu') as f:
            tokens = f.get_slice('tokens')[offset:offset+length].tolist()
        seqs[req] = {'tokens': tokens, 'L': length, 'gen_start': gen,
                     'family': populations['dev'].request_tasks[req]}
    actual = {'split': 'dev', **score(seqs, g1_drafts(s.k2_dev, populations['dev']), 'dev', live_reference=live),
              'manifest_sha256': sha256(s.manifest), 'k2_sha256': sha256(s.k2_dev),
              'live_reference_path': gate['live_reference_path']}
    if gate != actual or cast(dict[str, object], gate.get('g1', {})).get('pass') is not True:
        raise ValueError('binding dev G1(b) PASS missing or differs from exact shard/K2 score')
    parity = obj(s.parity)
    if (manifest_is_synthetic(s.parity) or parity.get('pass') is not True
            or cast(int, parity.get('samples', 0)) < 50 or parity.get('reference_math') is not True):
        raise ValueError('genuine 50-sample reference parity required')
    if not s.ckpt.is_dir() or not s.python.is_file():
        raise ValueError('existing native checkpoint and x86host training interpreter required')
    prior: dict[str, str] = {}
    if isinstance(s, PilotSettings):
        if sha256(s.previous_plan) != FIRST_PLAN_SHA256:
            raise ValueError('first pilot plan identity differs from recorded pilot')
        prior['previous_plan'] = sha256(s.previous_plan)
    return prior | {name: sha256(path) for name, path in {
        'assembled': receipt_path, 'manifest': s.manifest, 'g1': s.g1,
        'live_reference': Path(cast(str, gate['live_reference_path'])), 'parity': s.parity, 'k2_dev': s.k2_dev, 'k2_legal': s.k2_legal}.items()}


def commands(s: PilotSettings) -> dict[str, list[tuple[str, str]]]:
    """Shell bodies run ONLY in coordinator-created local detached sessions."""
    check_settings(s)
    q = shlex.quote
    repo = s.snapshot/'repo'
    output = s.output
    common = shlex.join(['--manifest', str(s.manifest), '--ckpt', str(s.ckpt),
        '--parity-receipt', str(s.parity), '--g1-receipt', str(s.g1), '--plan', str(output/'plan.json')])
    ev = shlex.join(['--plan', str(output/'plan.json'), '--root', str(output/'arms'),
        '--manifest', str(s.manifest), '--ckpt', str(s.ckpt), '--k2-dev', str(s.k2_dev),
        '--k2-legal', str(s.k2_legal)])
    cpu = shlex.join([str(s.python), '-m'])
    setup = '\n'.join(['set -euo pipefail', 'source ~/.config/ndm/paths.sh', 'umask 077',
        'export D2_TRAIN_ROOT='+q(str(s.snapshot)), 'export D2_GIT_SHA='+q(s.commit),
        'export D2_PYTHON='+q(str(s.python)), 'export PYTHONPATH='+q(str(repo)+':'+str(repo/'offload/dspark')),
        'export OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1',
        'export CUDA_VISIBLE_DEVICES=' + q(''), 'cd '+q(str(repo))])+'\n'
    locked = 'bash '+q(str(repo/'offload/dspark/d2/train/run_pilot_locked.sh'))
    def gpu(uuid: str, name: str, module: str, args: str) -> str:
        return 'D2_PILOT_GPU='+q(uuid)+' '+locked+' '+q(name)+' -m '+module+' '+args
    prepare = gpu(s.ref_gpu, 'prepare', 'offload.dspark.d2.train.pilot', 'prepare '+common+' --recipe '+s.recipe)
    sample = cpu+' offload.dspark.d2.train.pilot sample '+shlex.join([
        '--plan', str(output/'plan.json'), '--manifest', str(s.manifest), '--k2-dev', str(s.k2_dev), '--k2-legal', str(s.k2_legal), '--previous-plan', str(s.previous_plan)])
    def arm(name: str, uuid: str) -> str:
        return gpu(uuid, name, 'offload.dspark.d2.train.pilot', 'arm '+common+' --arm '+name+' --output '+q(str(output/'arms')))
    def evaluation(split: str, uuid: str, name: str) -> str:
        return gpu(uuid, 'eval-'+name, 'offload.dspark.d2.train.pilot_eval', ev+' --split '+split+' --out '+q(str(output/(name+'.part.json'))))
    merge = cpu+' offload.dspark.d2.train.pilot_eval '+shlex.join([
        '--plan', str(output/'plan.json'), '--merge-parts', str(output/'dev.part.json'), str(output/'legal.part.json'), '--out', str(output/'evaluation.json')])
    verdict = cpu+' offload.dspark.d2.train.pilot_verdict '+shlex.join([
        '--plan', str(output/'plan.json'), '--evaluation', str(output/'evaluation.json'), '--root', str(output/'arms'),
        '--manifest', str(s.manifest), '--parity-receipt', str(s.parity), '--g1-receipt', str(s.g1), '--out', str(output/'receipt.json')])
    return {'prepare': [('prepare', setup+prepare+'\n'+sample)],
        'train': [('ref', setup+arm('REF', s.ref_gpu)), ('fast', setup+arm('FAST', s.fast_gpu)+'\n'+arm('FAST-B', s.fast_gpu))],
        'evaluate': [('eval-dev', setup+evaluation('dev', s.ref_gpu, 'dev')), ('eval-legal', setup+evaluation('heldout_legal', s.fast_gpu, 'legal'))],
        'finish': [('finish', setup+merge+'\n'+verdict)]}


def git(s: Settings, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    return subprocess.run(['git', '-C', str(s.source), *args], env=env, check=True,
                          capture_output=True, text=True).stdout.strip()


def execute(s: PilotSettings, phase: str) -> None:
    # All admissions precede writes, subprocess launch, GPU checks or host work.
    if os.environ.get('D2_COORDINATOR') != COORDINATOR:
        raise ValueError('--execute requires D2_COORDINATOR='+COORDINATOR)
    if socket.gethostname().split('.')[0] not in ('x86host', 'x86host'):
        raise ValueError('--execute is local x86host coordinator only')
    if s.assembly != ASSEMBLY:
        raise ValueError('execution requires the reviewed decode-d2pass2-r1-c2 assembly path')
    identities = check_inputs(s)
    jobs = commands(s)
    os.umask(0o077)
    if phase == 'stage':
        if git(s, 'rev-parse', 'HEAD') != s.commit or git(s, 'status', '--porcelain'):
            raise ValueError('stage requires clean reviewed HEAD at the exact commit')
        if s.snapshot.exists() or s.output.exists():
            raise FileExistsError('snapshot and output namespaces must both be NEW')
        s.snapshot.mkdir(parents=True)
        repo = s.snapshot/'repo'
        repo.mkdir()
        env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
        with tempfile.TemporaryFile(dir=s.snapshot) as archive:
            subprocess.run(['git', '-C', str(s.source), 'archive', s.commit], env=env, stdout=archive, check=True)
            archive.seek(0)
            with tarfile.open(fileobj=archive) as tar:
                tar.extractall(repo, filter='data')
        (s.snapshot/'SOURCE_COMMIT').write_text(s.commit+'\n')
        (s.snapshot/'STAGED.json').write_text(json.dumps({'schema': 'd2.pilot-stage.v1',
            'source_commit': s.commit, 'recipe': s.recipe, 'run_tag': s.run_tag, 'inputs': identities, 'output': str(s.output),
            'commands': jobs}, indent=2)+'\n')
        s.output.mkdir(parents=True)
        (s.snapshot/'logs').mkdir()
        return
    stage = obj(s.snapshot/'STAGED.json')
    if (stage.get('run_tag') != s.run_tag or stage.get('recipe') != s.recipe or stage.get('source_commit') != s.commit or stage.get('inputs') != identities
            or stage.get('commands') != {k: [list(job) for job in v] for k, v in jobs.items()}
            or stage.get('output') != str(s.output)
            or (s.snapshot/'SOURCE_COMMIT').read_text().strip() != s.commit):
        raise ValueError('staged source/input/command contract changed')
    # Validate staged bytes against the reviewed archive, including additions.
    if git(s, 'rev-parse', 'HEAD') != s.commit or git(s, 'status', '--porcelain'):
        raise ValueError('review checkout changed since staging')
    tracked = set(git(s, 'ls-tree', '-r', '--name-only', s.commit).splitlines())
    repo = s.snapshot/'repo'
    if {str(p.relative_to(repo)) for p in repo.rglob('*') if p.is_file()} != tracked:
        raise ValueError('staged source file set changed')
    for relative in tracked:
        if sha256(s.source/relative) != sha256(s.snapshot/'repo'/relative):
            raise ValueError('staged source changed: '+relative)
    if phase == 'prepare' and (s.output/'plan.json').exists():
        raise FileExistsError('prepare is once-only; preserve existing sampled plan')
    if phase in ('train', 'evaluate', 'finish'):
        from .pilot import assert_same_sample, read_spec
        if read_spec(s.output/'plan.json', sampled=True).get('recipe', 'kl') != s.recipe:
            raise ValueError('staged recipe differs from prepared plan')
        assert_same_sample(s.output/'plan.json', s.previous_plan)
    if phase == 'evaluate':
        for arm in ('REF', 'FAST', 'FAST-B'):
            if obj(s.output/'arms'/arm/'arm.json').get('complete') is not True:
                raise ValueError('evaluation waits for all three complete arms')
    if phase == 'finish':
        for part in ('dev', 'legal'):
            obj(s.output/(part+'.part.json'))
    # Preflight ALL sessions before creating any; tmux also refuses name races.
    for name, _ in jobs[phase]:
        job_session = session(s, name)
        probe = subprocess.run(['tmux', 'has-session', '-t', '='+job_session], capture_output=True)
        if probe.returncode == 0:
            raise ValueError('session already exists: '+job_session)
        if probe.returncode != 1:
            raise RuntimeError('tmux preflight failed: '+probe.stderr.decode())
        if (s.snapshot/(name+'.sh')).exists():
            raise FileExistsError('job already staged; inspect its log and use documented resume: '+name)
    for name, body in jobs[phase]:
        script = s.snapshot/(name+'.sh')
        log = s.snapshot/'logs'/(name+'.coordinator.log')
        script.write_text(body+'\n')
        command = 'bash '+shlex.quote(str(script))+' >'+shlex.quote(str(log))+' 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', session(s, name), command], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--recipe', choices=RECIPES, default='kl')
    ap.add_argument('--run-tag', default=FIRST_NAME)
    ap.add_argument('--previous-plan', type=Path, default=FIRST_OUTPUT/'plan.json')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--check-inputs', action='store_true', help='CPU-only validation of LOCAL receipt tree')
    ap.add_argument('--phase', choices=PHASES, default='stage')
    ap.add_argument('--commit', required=True)
    ap.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[4])
    ap.add_argument('--snapshot', type=Path, default=None)
    ap.add_argument('--output', type=Path, default=None)
    ap.add_argument('--assembly', type=Path, default=ASSEMBLY)
    ap.add_argument('--parity', type=Path, default=Path('/opt/dsv41/x86host-data/d2/train/parity50-solo225/parity.json'))
    ap.add_argument('--g1', type=Path, required=True)
    ap.add_argument('--k2-dev', type=Path, required=True)
    ap.add_argument('--k2-legal', type=Path, required=True)
    ap.add_argument('--ckpt', type=Path, default=Path('/opt/dsv41/x86host-data/native-1017'))
    ap.add_argument('--python', type=Path, default=Path.home()/'dsv41-offload/e/venv/bin/python')
    ap.add_argument('--ref-gpu', default=GPU)
    ap.add_argument('--fast-gpu', default=SOLO_B)
    a = ap.parse_args()
    pilot_name = namespace(a.recipe, a.run_tag)
    if a.snapshot is None:
        a.snapshot = Path.home()/'dsv41-offload/d2/pilot'/pilot_name
    if a.output is None:
        a.output = Path('/opt/dsv41/x86host-data/d2/pilot')/pilot_name
    s = PilotSettings(**{name: getattr(a, name) for name in PilotSettings.__dataclass_fields__})
    check_settings(s)
    if a.execute:
        execute(s, a.phase)
    else:
        if a.check_inputs:
            print(json.dumps({'validated_inputs': check_inputs(s)}, indent=2))
        print(json.dumps({'dry': True, 'recipe': s.recipe, 'run_tag': s.run_tag, 'sessions': {phase: [session(s, name) for name, _ in jobs] for phase, jobs in commands(s).items()}, 'source_commit': s.commit, 'snapshot': str(s.snapshot),
            'assembly': str(s.assembly), 'order': PHASES, 'commands': commands(s),
            'checkpoint_blocks': [2500, 5000, 10000, 20000], 'checkpoint_seconds': 300,
            'result_receipt': str(s.output/'receipt.json'),
            'estimates': {'label': 'ESTIMATED from MEASURED first pilot: REF 6999 s, FAST 206 s/arm, evaluation 104.9 min/card',
                'training_gpu_hours': (6999+2*206)/3600,
                'evaluation_gpu_hours_upper': 2*104.9/60,
                'training_plus_evaluation_wall_hours_ideal': 6999/3600+104.9/60},
            'pending': 'local genuine G1(a), amendment-7b dev G1(b) tripwire, VALID live reference, parity, exact K2, clean source; no commands executed'}, indent=2))


if __name__ == '__main__':
    main()
