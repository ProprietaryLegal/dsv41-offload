"""Offline coordinator assembly. Invoked by assemble.sh; never uses hosts or GPUs."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
from collections.abc import Mapping

from .assembly_paths import AssemblyPaths, parse_mappings
from .copy_binding import digest, verify_copy

COORDINATOR = 'Deepseek/coordinator'


def run(root: Path, smoke: Path, smoke_sha256: str, prompts: Path, context: Path,
        mappings: Mapping[str, str], *, quarantine_named_errors: bool = False,
        expect_nonce: str | None = None) -> dict[str, object]:
    if os.environ.get('D2_COORDINATOR') != COORDINATOR:
        raise ValueError('coordinator execution only: D2_COORDINATOR=' + COORDINATOR)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('offline assembly requires CUDA_VISIBLE_DEVICES empty')
    from . import dataset as D, decode_dataset as DD, generation_export as E, g1 as G
    from .post_replay import aggregate

    paths = AssemblyPaths(root, mappings)
    root = paths.root
    smoke, prompts, context = (paths.inside(p) for p in (smoke, prompts, context))
    if digest(smoke) != smoke_sha256:
        raise ValueError('copied R6 smoke hash differs from coordinator-pinned original')
    # Refuse before creating outputs. A failed attempt is retained for inspection;
    # never overwrite, resume or remove a previously assembled tree implicitly.
    for name in ('export', 'decode-plan', 'shards', 'shards.partial', 'G1A.log', 'AGGREGATE.json', 'ASSEMBLED.json'):
        if (root / name).exists():
            raise ValueError('assembly requires NEW outputs: ' + str(root / name))
    copy_sha = verify_copy(root)
    if expect_nonce is not None and DD.obj(root / 'COPY_RECEIPT.json').get('nonce') != expect_nonce:
        raise ValueError('copy receipt differs from coordinator expected nonce')
    inputs = {str(p.relative_to(root)): digest(p) for p in (smoke, prompts, context)}
    E.export(root / 'generator', root / 'export', smoke)
    DD.prepare(root / 'export', root / 'capture', root / 'decode-plan', 'generation_verified',
               root / 'export/VERIFIED.json', assembly_root=root, path_maps=mappings, quarantine_named_errors=quarantine_named_errors)
    D.cmd_build(argparse.Namespace(replay=str(root / 'decode-plan'), out=str(root / 'shards'),
        prompt_manifest=str(prompts), sequence_context=str(context), taps_source='decode_capture',
        splits='', allow_skips=False, equiv_gate='', override_equiv_gate='', override_plumbing=''))
    # Real G1(a), not the GPU-dependent G1(b) and never a pilot clearance.
    log = io.StringIO()
    try:
        with redirect_stdout(log):
            G.cmd_plumbing(argparse.Namespace(shards=str(root / 'shards'), stride=1))
    finally:
        (root / 'G1A.log').write_text(log.getvalue())
        (root / 'G1A.log').chmod(0o600)
    (root / 'AGGREGATE.json').write_text(json.dumps(aggregate(root / 'shards'), indent=2, allow_nan=False))
    if verify_copy(root) != copy_sha or any(digest(root / p) != sha for p, sha in inputs.items()):
        raise ValueError('assembly inputs changed during build; no assembly receipt published')
    manifest = DD.obj(root / 'shards/MANIFEST.json')
    result: dict[str, object] = {
        'schema': 'd2.decode-assembly.v1', 'classification': 'MEASURED',
        'g1a_pass': True, 'pilot_cleared': False,
        'synthetic': DD.mapping(manifest['generation'])['synthetic'],
        'copy_receipt_sha256': copy_sha, 'inputs': inputs,
        'expected_nonce': expect_nonce,
        'quarantine': DD.obj(root / 'decode-plan/DECODE.json').get('quarantine'),
        'relocation': DD.obj(root / 'decode-plan/DECODE.json')['relocation'],
        'outputs': {p: digest(root / p) for p in ('export/VERIFIED.json', 'decode-plan/DECODE.json',
            'decode-plan/plan.jsonl', 'shards/MANIFEST.json', 'shards/SHA256SUMS', 'G1A.log', 'AGGREGATE.json')}}
    (root / 'ASSEMBLED.json').write_text(json.dumps(result, indent=2))
    for name in ('G1A.log', 'AGGREGATE.json', 'ASSEMBLED.json'):
        (root / name).chmod(0o600)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--image-smoke', type=Path, required=True)
    ap.add_argument('--smoke-sha256', required=True)
    ap.add_argument('--prompt-manifest', type=Path, required=True)
    ap.add_argument('--sequence-context', type=Path, required=True)
    ap.add_argument('--expect-nonce', required=True)
    ap.add_argument('--quarantine-named-errors', action='store_true')
    ap.add_argument('--path-map', action='append', default=[])
    a = ap.parse_args()
    os.umask(0o077)
    print(json.dumps(run(a.root, a.image_smoke, a.smoke_sha256, a.prompt_manifest,
                         a.sequence_context, parse_mappings(a.path_map),
                         quarantine_named_errors=a.quarantine_named_errors, expect_nonce=a.expect_nonce), indent=2))


if __name__ == '__main__':
    main()
