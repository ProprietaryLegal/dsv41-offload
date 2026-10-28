"""Assemble disk evidence and an explicitly INCOMPLETE private context scaffold.

This never calls a model or blesses a live identity. Run on Controller after input staging.
Only aggregate settings, file identities and a context scaffold are emitted.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import cast

CAMPAIGN = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD')
REMOTE = '/opt/dsv41/x86host-data/d2'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text()))


def assemble(repo: Path, data: Path, out: Path) -> None:
    from tokenizers import Tokenizer
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    manifest = read(data/'prompts/manifest.json')
    inputs = cast(dict[str, str], manifest['inputs'])
    tokenizer, encoder = Path(inputs['tokenizer']), Path(inputs['encoder'])
    if sha(tokenizer) != inputs['tokenizer_sha256'] or sha(encoder) != inputs['encoder_sha256']:
        raise ValueError('frozen prompt manifest renderer hashes differ')
    tok = Tokenizer.from_file(str(tokenizer))
    strings = ('<think>', '</think>', '<｜Assistant｜>', '<｜User｜>', '<｜end▁of▁sentence｜>', '｜DSML｜')
    markers = dict(zip(('think_start', 'think_end', 'assistant', 'user', 'eos', 'dsml'),
                       (tok.token_to_id(s) for s in strings)))
    if None in markers.values():
        raise ValueError('missing tokenizer marker')
    captures = [repo/'offload/results/capture'/n for n in
                ('cap-short.jsonl', 'cap-8k.jsonl', 'cap-8k-c2.jsonl')]
    # Files were appended later. The original four source locators are fixed by ACAP-SAMPLING.md.
    raw = [json.loads(line)['settings'] for p, count in zip(captures, (1, 1, 2))
           for line in p.read_text().splitlines()[:count]]
    settings = [{k: s[k] for k in ('temperature', 'top_p', 'seed', 'priority', 'chat_template_kwargs')}
                for s in raw]
    if len(settings) != 4 or any(s != settings[0] for s in settings):
        raise ValueError('original capture settings are not uniform')
    if any(settings[0][k] != v for k, v in {'temperature': 1, 'top_p': 1, 'seed': 42}.items()):
        raise ValueError('capture sampling contradicts recorded contract')
    paths = [tokenizer, encoder, data/'prompts/manifest.json',
             repo/'offload/operations/profiles.v1.json',
             repo/'offload/dspark/d2/GENERATOR-REVISION.md',
             repo/'offload/dspark/d2/ACAP-SAMPLING.md',
             CAMPAIGN/'D2-FIX1-R2.acap-settings.json', CAMPAIGN/'OPS1.live-profile.final.json',
             CAMPAIGN/'D2-REPLAY-PREP.live-container.json', CAMPAIGN/'D2-REPLAY-PREP.live-files.txt',
             *captures]
    states = [read(p) for p in (data/'gen/seqs').glob('*.json')]
    segments = [cast(dict[str, object], s) for state in states for s in cast(list[object], state['segments'])]
    inventory = {'schema': 'd2.replay-prep-evidence.v1', 'classification': 'MEASURED',
                 'qualification': 'disk evidence only; NOT a complete sequence-context or window attestation',
                 'evidence': [{'path': str(p), 'sha256': sha(p)} for p in paths],
                 'capture_settings': settings[0], 'generation_segments': len(segments),
                 'segments_without_sampling': sum('sampling' not in s for s in segments),
                 'generation_contract': 'historical e2b2186 priority5; file hashes and window identity need coordinator attestation',
                 'missing': ['generation target/quantization/runtime evidence linked to all source segments',
                             'original acap effective defaults/processors and target/runtime evidence',
                             'replay both-rank image/mount/runtime/renderer/defaults evidence',
                             'coordinator-verified final d2.sequence-context.v1 with per_sequence acap overrides']}
    target = {'model_root': '/srv/dsv41/legal-v2a', 'quantization': None, 'engine_revision': None,
              'weights_manifest_sha256': None, 'quantization_manifest_sha256': None, 'receipt_sha256': None}
    context: dict[str, object] = {
        'schema': 'd2.sequence-context.v1', 'status': 'INCOMPLETE_DO_NOT_BUILD',
        'target': target, 'replay_target': deepcopy(target),
        'tokenizer_path': REMOTE+'/tokenizer/tokenizer.json', 'tokenizer_sha256': sha(tokenizer),
        'encoder_path': REMOTE+'/encoder/deepseek_v41_encoding.py', 'encoder_sha256': sha(encoder),
        'chat_template_path': REMOTE+'/encoder/deepseek_v41_encoding.py', 'chat_template_sha256': sha(encoder),
        'chat_template_qualification': 'pinned local renderer; confirm served renderer in window',
        'markers': markers,
        'sampling': {'temperature': 1.0, 'top_p': 1.0, 'seed': None,
                     'top_k': 'UNVERIFIED_EFFECTIVE_DEFAULT', 'min_p': 'UNVERIFIED_EFFECTIVE_DEFAULT',
                     'structured_outputs': None, 'stop': None, 'stop_token_ids': None},
        'formatting': {'chat_template_kwargs': manifest['chat_template_kwargs'], 'tools': None, 'tool_choice': None},
        'generation_config': 'ABSENT on current rank0 file read; historical and replay attestation pending',
        'per_sequence': {}}
    per_sequence: dict[str, object] = {}
    for p in sorted((data/'acap').glob('*.json')):
        acap = read(p)
        ctx = deepcopy(context)
        del ctx['per_sequence']
        cast(dict[str, object], ctx['sampling'])['seed'] = 42
        ctx['original_capture_priority'] = settings[0]['priority']
        per_sequence[str(acap['seq_id'])] = ctx
    context['per_sequence'] = per_sequence
    for name, value in [('evidence.json', inventory), ('sequence-context.INCOMPLETE.json', context)]:
        with (out/name).open('x') as f:
            os.fchmod(f.fileno(), 0o600)
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    assemble(Path(__file__).resolve().parents[3], Path('/opt/dsv41/data/d2'), a.out)


if __name__ == '__main__':
    main()
