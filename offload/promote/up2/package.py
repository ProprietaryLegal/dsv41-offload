"""UP2-PROMOTE immutable production package for 0075 replay skip. CPU only; Controller, before any HOLD.

The package is the 0075 bytes production mounts once the switch is applied: four vLLM files (mounted per file) and a
complete remote DSpark package directory (mounted in place of production's directory). It is content-addressed: its id
is the first 12 hex digits of the sha256 of its SHA256SUMS text, so the same bytes always stage to the same directory.

Composition order (explicit; anything else is refused): image v5 -> K1 overlay (13 files, kvpack 559c5867ba42) ->
PROMO1 low profile (dsv41.env only) -> PP4 (remote_dspark package + model_runner/rejection_sampler_utils/round trace,
dsv41.env) -> 0075 (this package, dsv41.env). Shared files are listed in COMPOSITION and checked by the tests.

* `qualified`: production bases ops1 (596ba1c) and promo1-low (PROMO1). The bytes are EXACTLY the UP3 `replay` arm the
  UP2-ADOPT qualification window measures (offload/up3/overlay/replay, manifests/replay.json); `arm` names it. The
  promotion runner still requires that window's ADOPT CANDIDATE summary for these very file hashes.
* `pp4-composed`: production base pp4 (941a3c3's remote package). 0075-process/-successor/-ready-waiter apply to PP4's
  files with fuzz 0; 0075-suffix conflicts in PP4's `round.py` run() (PP4 added the sampled temperature/seed parameters),
  so `patches/0075-suffix-on-pp4-941a3c3.patch` carries the SAME added/removed lines re-anchored to PP4's context. These
  bytes were never measured: `arm` is null and the switch refuses them until a window measures this exact id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import TypedDict, cast

from offload.up3 import build as b

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SCHEMA = 'up2-promote-package-v1'
REGISTRATION = 'up2-0075-adopt-0p9'
PP4_COMMIT = '941a3c3fbe6fec06a05e925a1e7d77569b0d12aa'
VLLM_FILES = ('vllm/v1/core/sched/scheduler.py', 'vllm/v1/worker/gpu/input_batch.py',
              'vllm/models/deepseek_v41/nvidia/model_state.py', 'vllm/models/deepseek_v41/replay_skip.py')
REMOTE = 'remote_dspark/'
PP4_SUFFIX_PORT = 'patches/0075-suffix-on-pp4-941a3c3.patch'
PACKAGES = {'qualified': ('ops1', 'promo1-low'), 'pp4-composed': ('pp4',)}
# Every production file each layer replaces (container paths under dist-packages). 0075 shares NO file with K1 or with
# PROMO1 (env only); with PP4 it shares the remote DSpark package, and four of its files carry both changes.
COMPOSITION: dict[str, tuple[str, ...]] = {
    'k1': ('vllm/models/deepseek_v41/attention.py', 'vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py',
           'vllm/models/deepseek_v41/common/ops/nvfp4_fi_cache.py', 'vllm/models/deepseek_v41/nvidia/dspark.py',
           'vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py', 'vllm/models/deepseek_v41/nvidia/model.py',
           'vllm/v1/core/kv_cache_utils.py', 'vllm/v1/kv_cache_interface.py', 'vllm/v1/worker/utils.py',
           'vllm/v1/worker/gpu/model_runner.py', 'vllm/v1/worker/gpu/dsv41_memprobe.py', 'vllm_exl3_moe/tf_prompt.py',
           'vllm_exl3_moe/tf_experts.py'),
    'promo1-low': (),
    'pp4': ('vllm/v1/worker/gpu/model_runner.py', 'vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py',
            'vllm/v1/worker/gpu/dsv41_pp4_round_trace.py', 'vllm/v1/worker/gpu/spec_decode/remote_dspark/'),
    '0075': (*VLLM_FILES, 'vllm/v1/worker/gpu/spec_decode/remote_dspark/'),
}
# Remote files both PP4 and 0075 change (the composed package carries both); suffix_provenance.py is byte-identical.
SHARED_WITH_PP4 = ('gather.py', 'round.py', 'roundcore.py', 'speculator.py', 'suffix_provenance.py')


class Package(TypedDict):
    schema: str
    name: str
    id: str
    bases: list[str]
    arm: str | None
    evidence: str
    files: dict[str, str]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sums(files: dict[str, bytes]) -> str:
    """`sha256sum -c` text in byte order of the paths; the package id hashes exactly this text."""
    return ''.join(f'{digest(files[name])}  {name}\n' for name in sorted(files))


def package_id(files: dict[str, bytes]) -> str:
    return digest(sums(files).encode())[:12]


def qualified_files() -> dict[str, bytes]:
    """The UP3 replay arm's verified bytes, renamed to the staged layout."""
    manifest = b.verify('replay')
    tree = b.HERE / 'overlay/replay'
    files: dict[str, bytes] = {}
    for name in manifest['files']:
        staged = REMOTE + name.removeprefix(b.REMOTE) if name.startswith(b.REMOTE) else name
        files[staged] = (tree / name).read_bytes()
    if set(files) != {*VLLM_FILES, *(REMOTE + n for n in (*b.REMOTE_PACKAGE, 'suffix_provenance.py'))}:
        raise ValueError('qualified replay file set differs from the promotion layout')
    return files


def pp4_files() -> dict[str, bytes]:
    """PP4's remote package with 0075 applied (suffix step re-anchored), plus the same four vLLM files."""
    remote = sorted(Path(p).name for p in subprocess.check_output(['git', 'ls-tree', '--name-only', PP4_COMMIT,
        'offload/dspark/spark/remote_dspark/'], cwd=ROOT, text=True).splitlines() if p.endswith('.py'))
    patches = [ROOT / 'offload/upstream_adopt/patches' / p for p in ('0075-process.patch', '0075-successor.patch')]
    patches += [HERE / PP4_SUFFIX_PORT, ROOT / 'offload/upstream_adopt/patches/0075-ready-waiter.patch']
    with tempfile.TemporaryDirectory(prefix='up2-pp4-') as tmp:
        work = Path(tmp)
        for name in b.REPLAY_IMAGE:
            (work / name).parent.mkdir(parents=True, exist_ok=True)
            (work / name).write_bytes((b.IMAGE / name).read_bytes())
        package = work / 'offload/dspark/spark/remote_dspark'
        package.mkdir(parents=True)
        for name in remote:
            (package / name).write_bytes(subprocess.check_output(['git', 'show',
                f'{PP4_COMMIT}:offload/dspark/spark/remote_dspark/{name}'], cwd=ROOT))
        for patch in patches:
            result = subprocess.run(['patch', '--batch', '--forward', '--fuzz=0', '--no-backup-if-mismatch', '-p1',
                '-d', str(work), '-i', str(patch)], text=True, capture_output=True)
            if result.returncode or any(word in result.stdout.lower() for word in ('fuzz', 'failed', 'reject')):
                raise ValueError(f'0075 on PP4 not exact: {patch.name}\n{result.stdout}{result.stderr}')
        if list(work.rglob('*.rej')) or list(work.rglob('*.orig')):
            raise ValueError('0075 on PP4 left rejects/backups')
        provenance = (ROOT / b.REPLAY_PROVENANCE).read_bytes()
        if (package / 'suffix_provenance.py').read_bytes() != provenance:
            raise ValueError("PP4's suffix_provenance.py differs from 0075's; re-review the composition")
        files = {name: (work / name).read_bytes() for name in VLLM_FILES}
        files.update({REMOTE + p.name: p.read_bytes() for p in sorted(package.glob('*.py'))})
    qualified = qualified_files()
    if any(files[name] != qualified[name] for name in VLLM_FILES):
        raise ValueError('PP4 composition changed a 0075 vLLM file; PP4 must not touch them')
    return files


def files_for(name: str) -> dict[str, bytes]:
    if name == 'qualified':
        return qualified_files()
    if name == 'pp4-composed':
        return pp4_files()
    raise ValueError(f'unknown UP2 promotion package {name!r}')


def describe(name: str) -> Package:
    files = files_for(name)
    measured = name == 'qualified'
    return {'schema': SCHEMA, 'name': name, 'id': package_id(files), 'bases': list(PACKAGES[name]),
            'arm': 'up3-replay' if measured else None,
            'evidence': (f'the UP2-ADOPT window ({REGISTRATION}) measures exactly these bytes as the UP3 replay arm; '
                         'promotion requires its ADOPT CANDIDATE summary' if measured else
                         'NONE: composed bytes never measured; a window for this id is required before any switch'),
            'files': {path: digest(data) for path, data in sorted(files.items())}}


def manifest_path(name: str) -> Path:
    return HERE / 'packages' / (name + '.json')


def verify() -> dict[str, Package]:
    """Rebuild every package and compare with the committed manifests (the switch reads only those)."""
    result: dict[str, Package] = {}
    for name in PACKAGES:
        committed = cast(Package, json.loads(manifest_path(name).read_text()))
        if committed != describe(name):
            raise ValueError(f'{manifest_path(name)} drift: rebuild/review the UP2 promotion package')
        result[name] = committed
    return result


def materialize(name: str, out: Path) -> Package:
    """Write the package directory (files + SHA256SUMS + package.json) for stage.sh; never into an existing path."""
    manifest = verify()[name]
    if out.exists():
        raise ValueError(f'{out} exists; fresh directory required')
    files = files_for(name)
    for path, data in files.items():
        (out / path).parent.mkdir(parents=True, exist_ok=True)
        (out / path).write_bytes(data)
    (out / 'SHA256SUMS').write_text(sums(files))
    (out / 'package.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'write', 'materialize'))
    parser.add_argument('--name', choices=tuple(PACKAGES), default='qualified')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.action == 'write':
        for name in PACKAGES:
            manifest_path(name).parent.mkdir(exist_ok=True)
            manifest_path(name).write_text(json.dumps(describe(name), indent=2) + '\n')
        print('UP2 promotion package manifests written')
    elif args.action == 'verify':
        packages = verify()
        print('UP2 promotion packages verified: ' + ', '.join(f"{n} {p['id']} arm={p['arm']}"
                                                              for n, p in packages.items()))
    else:
        if args.out is None:
            parser.error('--out required')
        manifest = materialize(args.name, args.out)
        print(f"UP2 package {args.name} {manifest['id']} materialized at {args.out}")


if __name__ == '__main__':
    main()
