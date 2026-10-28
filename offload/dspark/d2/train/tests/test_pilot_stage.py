"""CPU-only dry planning and fail-closed coordinator admissions. No execute jobs."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import socket
import sys
from unittest.mock import patch

import pytest
import torch

from offload.dspark.d2 import assemble as A, generation_export as E
from offload.dspark.d2.tests.test_assemble_rehearsal import make_copy
from offload.dspark.d2.train import pilot_stage as S
from offload.dspark.d2.train.data import sha256
from offload.dspark.d2.train.safety import GPU, SOLO_B
from offload.dspark.d2.train.tests.test_g1_live import gate_fixture


def settings(root: Path) -> S.PilotSettings:
    return S.PilotSettings(source=root/'source', commit='a'*40, snapshot=root/'snapshot',
        output=root/'output', assembly=root/'assembly', parity=root/'parity.json',
        g1=root/'dev.json', k2_dev=root/'dev.k2.jsonl', k2_legal=root/'legal.k2.jsonl',
        ckpt=root/'native', python=root/'venv/bin/python')


def test_dry_main_never_reads_receipts_or_launches_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                          capsys: pytest.CaptureFixture[str]) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('dry mode attempted a read or execution')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    monkeypatch.setattr(Path, 'read_text', forbidden)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(sys, 'argv', ['pilot_stage', '--commit', 'a'*40,
        '--g1', str(tmp_path/'absent-g1'), '--k2-dev', str(tmp_path/'absent-acap'),
        '--k2-legal', str(tmp_path/'absent-legal')])
    S.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan['dry'] is True
    assert plan['assembly'] == str(S.ASSEMBLY)
    assert plan['order'] == list(S.PHASES)
    assert plan['checkpoint_blocks'] == [2500, 5000, 10000, 20000]
    assert plan['checkpoint_seconds'] == 300
    assert plan['estimates']['training_gpu_hours'] == pytest.approx((6999+2*206)/3600)
    assert plan['estimates']['evaluation_gpu_hours_upper'] == pytest.approx(2*104.9/60)
    assert plan['result_receipt'].endswith('/decode-d2pass2-r1-p2-kl/receipt.json')
    assert list(tmp_path.iterdir()) == []


def test_rendered_order_uuid_leases_and_shell_syntax(tmp_path: Path) -> None:
    s = settings(tmp_path/'space and $literal')
    jobs = S.commands(s)
    prep = jobs['prepare'][0][1]
    assert prep.index('pilot prepare') < prep.index('pilot sample')
    ref = jobs['train'][0][1]
    fast = jobs['train'][1][1]
    assert GPU in ref and SOLO_B not in ref
    assert SOLO_B in fast and GPU not in fast
    assert fast.index('--arm FAST --') < fast.index('--arm FAST-B --')
    assert 'set -euo pipefail' in fast
    assert '--split dev' in jobs['evaluate'][0][1]
    assert '--split heldout_legal' in jobs['evaluate'][1][1]
    assert '--merge-parts' in jobs['finish'][0][1] and 'pilot_verdict' in jobs['finish'][0][1]
    for group in jobs.values():
        for _, body in group:
            subprocess.run(['bash', '-n'], input=body, text=True, check=True)
            assert 'run_pilot_locked.sh' in body or 'pilot_verdict' in body
            assert 'ssh ' not in body and 'nvidia-smi' not in body
    assert not s.snapshot.exists() and not s.output.exists()


@pytest.mark.parametrize('gpu', ['GPU-c4e7a371-production-pair', 'GPU-7744b44b-production-pair', '0', SOLO_B])
def test_production_pair_or_wrong_ref_allocation_refused_even_dry(tmp_path: Path, gpu: str) -> None:
    with pytest.raises(ValueError, match='prescribed solo UUIDs'):
        S.commands(replace(settings(tmp_path), ref_gpu=gpu))


def test_no_assembly_receipt_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match='ASSEMBLED.json'):
        S.check_inputs(settings(tmp_path))


@pytest.mark.parametrize('defect', ['missing_g1a', 'failed_g1a', 'synthetic', 'nested_synthetic'])
def test_false_or_synthetic_assembly_receipts_refuse(tmp_path: Path, defect: str) -> None:
    s = settings(tmp_path)
    s.assembly.mkdir()
    receipt: dict[str, object] = {'schema': 'd2.decode-assembly.v1', 'synthetic': False, 'g1a_pass': True}
    if defect == 'missing_g1a':
        del receipt['g1a_pass']
    elif defect == 'failed_g1a':
        receipt['g1a_pass'] = False
    elif defect == 'synthetic':
        receipt['synthetic'] = True
    else:
        receipt['checks'] = {'synthetic': True}
    (s.assembly/'ASSEMBLED.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='genuine nonsynthetic.*G1'):
        S.check_inputs(s)


@pytest.mark.parametrize('defect', ['missing_shard', 'wrong_split', 'synthetic_manifest',
    'honest_synthetic_receipt', 'missing_g1b', 'synthetic_g1b', 'wrong_g1b_split', 'forged_real'])
def test_real_assembly_helper_artifacts_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    root, smoke, prompts, context = make_copy(tmp_path, monkeypatch, include_synthetic_acap=True)
    with patch.object(E, 'smoke'):
        A.run(root, smoke, E.digest(smoke), prompts, context, {})
    s = replace(settings(tmp_path), assembly=root)
    if defect != 'honest_synthetic_receipt':
        # Adversarial falsification for refusal tests ONLY: never a passing fixture.
        rp = root/'ASSEMBLED.json'
        receipt = json.loads(rp.read_text())
        receipt['synthetic'] = False
        mp = s.manifest
        metadata = json.loads(mp.read_text())
        if defect != 'synthetic_manifest':
            metadata['generation']['synthetic'] = False
        if defect == 'missing_shard':
            (mp.parent/metadata['splits']['train']['files'][0]['file']).unlink()
        if defect == 'wrong_split':
            metadata['splits']['renamed_dev'] = metadata['splits'].pop('dev')
        mp.write_text(json.dumps(metadata))
        receipt['outputs']['shards/MANIFEST.json'] = sha256(mp)
        rp.write_text(json.dumps(receipt))
    # Invented counters only to exercise later admissions; this is never a
    # passing stage fixture (the synthetic copy provenance still refuses).
    if defect != 'missing_g1b':
        fixture = gate_fixture(tmp_path/'gate-fixture', s.manifest)
        gate = json.loads(fixture.read_text())
        if defect == 'synthetic_g1b':
            gate['synthetic'] = True
        if defect == 'wrong_g1b_split':
            gate['split'] = 'heldout_legal'
        s.g1.write_text(json.dumps(gate))
    if defect in ('missing_shard', 'missing_g1b'):
        with pytest.raises(FileNotFoundError, match='train' if defect == 'missing_shard' else 'dev.json'):
            S.check_inputs(s)
    else:
        pattern = {'wrong_split': 'amendment-7 population', 'synthetic_g1b': 'binding dev G1',
            'wrong_g1b_split': 'binding dev G1', 'forged_real': 'real image-bound smoke'}.get(defect, 'synthetic')
        with pytest.raises(ValueError, match=pattern):
            S.check_inputs(s)
    assert not s.snapshot.exists() and not s.output.exists()


def test_execute_identity_and_host_refuse_before_reads_or_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('refused execution reached input read or subprocess')
    monkeypatch.setattr(S, 'check_inputs', forbidden)
    monkeypatch.setattr(subprocess, 'run', forbidden)
    monkeypatch.delenv('D2_COORDINATOR', raising=False)
    with pytest.raises(ValueError, match='D2_COORDINATOR'):
        S.execute(settings(tmp_path), 'stage')
    monkeypatch.setenv('D2_COORDINATOR', S.COORDINATOR)
    monkeypatch.setattr(socket, 'gethostname', lambda: 'controller')
    with pytest.raises(ValueError, match='local x86host'):
        S.execute(settings(tmp_path), 'stage')
    # Only the verified copy retry is admitted; the first copy attempt's tree has no receipt.
    monkeypatch.setattr(socket, 'gethostname', lambda: 'x86host')
    assert S.ASSEMBLY == Path('/opt/dsv41/x86host-data/d2/decode-d2pass2-r1-c2')
    partial = replace(settings(tmp_path), assembly=Path('/opt/dsv41/x86host-data/d2/decode-d2pass2-r1'))
    with pytest.raises(ValueError, match='decode-d2pass2-r1-c2 assembly path'):
        S.execute(partial, 'stage')
    assert not (tmp_path/'snapshot').exists()


@pytest.mark.parametrize('field', ['snapshot', 'output'])
def test_stage_never_overlaps_source_or_assembly(tmp_path: Path, field: str) -> None:
    s = settings(tmp_path)
    with pytest.raises(ValueError, match='overlaps'):
        S.check_settings(replace(s, snapshot=s.assembly/'nested') if field == 'snapshot' else replace(s, output=s.assembly/'nested'))
