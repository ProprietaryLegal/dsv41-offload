"""S35 integrated CPU rehearsal on synthetic data: copy receipt -> pass2_attest draft -> pass2_context (decode-capture)
-> export/prepare/shards/G1(a) (assemble.run) -> amendment-7 live reference -> K2 file -> G1(b) score -> pilot_stage
admission, then tampering after the PASS. No hosts, GPUs, W12 files or real receipts.

Stubs, each named: the external image smoke (both module copies, exactly as the lane rehearsals stub it), the
>= 20,000 training-block floor (synthetic trees hold a handful of blocks) and a hand-written parity receipt. The K2
drafts are synthetic (the realized tokens), not the K2 worker's.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from d2 import assemble as ASM, generation_export as E, pass2_context as P, g1, shards as S
from d2.tests.test_pass2_attest import fixture, run
from offload.dspark.d2 import generation_export as E2
from offload.dspark.d2.live_reference import PREFIX, RESTART, build_reference, timestamp
from offload.dspark.d2.train import pilot_stage as PS
from offload.dspark.d2.train.data import D2Blocks


def prom(rounds: int, per_pos: list[int], total: int) -> str:
    labels = 'engine="0",model_name="DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2"'
    created = timestamp(RESTART) + 12.3
    lines = []
    for name, value in (('num_drafts', rounds), ('num_draft_tokens', 5 * rounds), ('num_accepted_tokens', sum(per_pos))):
        lines += [f'{PREFIX}{name}_total{{{labels}}} {value:.1f}', f'{PREFIX}{name}_created{{{labels}}} {created:.8e}']
    for pos, v in enumerate(per_pos):
        lines += [f'{PREFIX}num_accepted_tokens_per_pos_total{{{labels},position="{pos}"}} {v:.1f}',
                  f'{PREFIX}num_accepted_tokens_per_pos_created{{{labels},position="{pos}"}} {created:.8e}']
    return '\n'.join(lines + [f'vllm:generation_tokens_total{{{labels}}} {total:.1f}']) + '\n'


def k2_file(root: Path, split: str, path: Path) -> None:
    man = json.loads((root / 'shards/MANIFEST.json').read_text())
    with path.open('w') as fh:
        for f in man['splits'][split]['files']:
            sh = S.Shard(str(root / 'shards' / f['file']))
            for i in range(len(sh)):
                s, tok = sh.seq(i), sh.tokens(i).tolist()
                d = {str(p): [tok[t] if t < s['L'] else 0 for t in range(p + 2, p + 7)] for p in sh.blocks(i)}
                fh.write(json.dumps({'seq_id': s['seq_id'], 'drafts': d}) + '\n')


def live_reference(live: Path, *, gen: int = 990) -> Path:
    live.mkdir()
    (live / 'metrics-20261025T043300Z.prom').write_text(prom(80, [80] * 5, 900))
    (live / 'metrics-20261025T060000Z.prom').write_text(prom(100, [100] * 5, 1000))
    (live / 'rate-20261025T060000Z.json').write_text(json.dumps({'t': '2026-10-25T05:59:30Z', 'gen_tokens': gen}))
    ref = build_reference([live / 'metrics-20261025T043300Z.prom', live / 'metrics-20261025T060000Z.prom'],
                          [live / 'rate-20261025T060000Z.json'], '2026-10-25T05:59:00Z', '2026-10-25T06:30:00Z')
    out = live / 'live-reference.json'
    out.write_text(json.dumps(ref))
    return out


def chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, gen: int = 990, mode: str = 'live-reference',
          split: str = 'dev') -> tuple[PS.Settings, Path, Path]:
    root, pm, emap, res, out = fixture(tmp_path, monkeypatch)
    draft = run(pm, emap, res, out)
    staged = P.stage(root, Path(draft['attestation']), draft['sha256'], pm)
    smoke = root / 'inputs/R6-smoke.json'
    with patch.object(E, 'smoke'):
        result = ASM.run(root, smoke, E.digest(smoke), pm, Path(staged['sequence_context']), staged['path_maps'],
                         expect_nonce='s34-synthetic')
    assert result['g1a_pass'] is True and result['synthetic'] is False and result['pilot_cleared'] is False
    work = tmp_path / 'g1'
    work.mkdir()
    for s in ('dev', 'heldout_legal'):
        k2_file(root, s, work / f'{s}.k2_drafts.jsonl')
    ref = live_reference(tmp_path / 'live', gen=gen)
    g1.cmd_score(argparse.Namespace(shards=str(root / 'shards'), split=split, work=str(work), out=str(work / 'gate.json'),
                                    mode=mode, live_reference=ref if mode == 'live-reference' else None))
    parity = tmp_path / 'parity.json'
    parity.write_text(json.dumps({'pass': True, 'samples': 50, 'reference_math': True}))
    (tmp_path / 'ckpt').mkdir()
    settings = PS.Settings(source=tmp_path / 'src', commit='a' * 40, snapshot=tmp_path / 'stage', output=tmp_path / 'out',
                           assembly=root, parity=parity, g1=work / 'gate.json', k2_dev=work / 'dev.k2_drafts.jsonl',
                           k2_legal=work / 'heldout_legal.k2_drafts.jsonl', ckpt=tmp_path / 'ckpt', python=Path(sys.executable))

    class Floor(D2Blocks):
        def __init__(self, manifest: Path, name: str) -> None:
            super().__init__(manifest, name)
            self._s35_split = name

        def __len__(self) -> int:
            n = super().__len__()
            return max(n, 20000) if self._s35_split == 'train' else n
    monkeypatch.setattr(PS, 'D2Blocks', Floor)
    return settings, ref, root


def admit(settings: PS.Settings) -> dict[str, str]:
    with patch.object(E2, 'smoke'):
        return PS.check_inputs(settings)


def test_integrated_chain_admits_only_with_every_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings, ref, root = chain(tmp_path, monkeypatch)
    gate = json.loads(settings.g1.read_text())
    assert gate['g1']['status'] == 'PASS' and gate['g1']['binding'] is True and gate['g1']['amendment'] == '7b'
    bound = admit(settings)
    assert set(bound) == {'assembled', 'manifest', 'g1', 'live_reference', 'parity', 'k2_dev', 'k2_legal'}
    assert bound['live_reference'] == E.digest(ref)


@pytest.mark.parametrize('defect', ['snapshot_byte', 'k2_after_score', 'manifest_synthetic', 'assembled_synthetic',
                                    'assembled_no_g1a', 'shard_byte', 'copy_receipt'])
def test_integrated_tamper_after_pass_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    settings, ref, root = chain(tmp_path, monkeypatch)
    if defect == 'snapshot_byte':
        final = Path(json.loads(ref.read_text())['snapshots'][-1])
        final.write_text(final.read_text() + '# s35\n')
    elif defect == 'k2_after_score':
        lines = settings.k2_dev.read_text().splitlines()
        row = json.loads(lines[0])
        p = sorted(row['drafts'], key=int)[0]
        row['drafts'][p] = [-1] * 5
        settings.k2_dev.write_text('\n'.join([json.dumps(row), *lines[1:]]) + '\n')
    elif defect in ('manifest_synthetic', 'assembled_synthetic', 'assembled_no_g1a'):
        path = root / ('shards/MANIFEST.json' if defect == 'manifest_synthetic' else 'ASSEMBLED.json')
        value = json.loads(path.read_text())
        value.update({'manifest_synthetic': {'synthetic': True}, 'assembled_synthetic': {'synthetic': True},
                      'assembled_no_g1a': {'g1a_pass': False}}[defect])
        path.write_text(json.dumps(value))
    elif defect == 'shard_byte':
        shard = sorted((root / 'shards/dev').glob('*.safetensors'))[0]
        data = bytearray(shard.read_bytes())
        data[-1] ^= 1
        shard.write_bytes(bytes(data))
    elif defect == 'copy_receipt':
        receipt = root / 'COPY_RECEIPT.json'
        receipt.write_text(receipt.read_text().replace('}', ' }', 1))
    with pytest.raises((ValueError, KeyError)):
        admit(settings)


@pytest.mark.parametrize('variant', ['inconclusive_foreign', 'historical_acap', 'wrong_split'])
def test_integrated_non_binding_g1_never_admits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str) -> None:
    kwargs = {'inconclusive_foreign': {'gen': 970}, 'historical_acap': {'mode': 'historical-acap'},
              'wrong_split': {'split': 'heldout_legal'}}[variant]
    settings, ref, root = chain(tmp_path, monkeypatch, **kwargs)  # type: ignore[arg-type]
    gate = json.loads(settings.g1.read_text())
    assert gate['g1']['pass'] is not True
    with pytest.raises(ValueError):
        admit(settings)
