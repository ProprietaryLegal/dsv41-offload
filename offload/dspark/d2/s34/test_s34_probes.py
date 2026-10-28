"""S34 independent probes of offload/d2-assemble 38168e0. CPU only, synthetic data only.

No hosts, sockets, GPUs or real receipts. Only the external image-smoke gate
(E.smoke) is replaced, exactly as the lane's own rehearsal does; every copy,
receipt, relocation, context, shard and G1(a) check is the real code.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from d2 import assemble as A, dataset as D, decode_dataset as DD, generation_export as E
from d2 import generation_top20 as T
from d2.assembly_paths import AssemblyPaths
from d2.copy_binding import verify_copy
from d2.s26.test_s26_resume import C, step
from d2.tests.test_generation_top20 import wire
from d2.tests.test_sequence_audit import context_fixture
from offload.dspark.d2 import decode_copy as COPY

REPO = Path(__file__).resolve().parents[4]
R2_REVISION = 'eb55f82'
NONCE = 's34-synthetic'


def forbid_hosts(monkeypatch, module: ModuleType) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError('S34 probe attempted host access')
    monkeypatch.setattr(module.transfer, 'idle', forbidden)
    monkeypatch.setattr(module.transfer, 'copy', forbidden)


def source_tree(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    """Real CPU capture writer + option-ii generator states (lane rehearsal technique)."""
    source = tmp_path / 'source'
    capture, state = source / 'capture', source / 'generator'
    capture.mkdir(parents=True)
    (state / 'seqs').mkdir(parents=True)
    (state / 'LOCK').touch()
    writer = C.Capture(str(capture))
    prompts: list[dict[str, object]] = []
    families = {'train': 'holding_summary', 'dev': 'issue_spotting', 'heldout_legal': 'chronology',
                'heldout_general': 'general_book'}
    for i, split in enumerate(families):
        sid = 'p' + str(i)
        prompt = [10 + i] * 200
        generated = [100, 999, 100, 999, 100, 999, 100, 999]
        tokens = prompt + generated
        raw = {'schema': 'd2.genstate.v1', 'pid': sid, 'status': 'partial', 'split': split,
               'family': families[split], 'prompt_sha256': 'hash_' + sid, 'prompt_ids': prompt,
               'gen_ids': generated, 'segments': [], 'top20': 'generation',
               'logprob_rows': [T.rows([t], wire(t, t == 999))[0] for t in generated]}
        (state / 'seqs' / (sid + '.json')).write_text(json.dumps(raw))
        rid = f'cmpl-d2c.{sid}.0.72.200-0-feed'
        toks = dict(enumerate(tokens))
        step(rid, toks, 0, 200, 0, writer)
        step(rid, toks, 200, 8, 0, writer)
        prompts.append({'pid': sid, 'split': split, 'family': families[split],
                        'prompt_sha256': raw['prompt_sha256'], 'source_docs': ['doc-' + sid]})
    old_argv = sys.argv
    try:
        sys.argv = ['flush', str(capture), NONCE]
        exec(compile(COPY.FLUSH_CODE, '<reviewed synthetic flush>', 'exec'), {})
    finally:
        sys.argv = old_argv
        writer.close()
    return capture, state, prompts


def local_remote(module: ModuleType):
    class Local(module.Remote):  # type: ignore[name-defined, misc]
        def __init__(self, capture: bool) -> None:
            super().__init__(['NEVER-A-HOST'], capture)

        def inventory(self, root: str, nonce: str):
            return module.scan(root, self.capture, nonce)
    return Local


def fake_transfer(interrupt: set[str] | None = None, busy: type[Exception] | None = None):
    """Stands in for copy_to_x86host.copy: stream copy + digest; optionally one demand interruption."""
    pending = set(interrupt or ())

    def copy(source: str, destination: Path, expected: str | None, size: int | None, **_: object):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.name in pending:
            pending.discard(destination.name)
            destination.with_name(destination.name + '.s2-partial').write_bytes(b'partial')
            assert busy is not None
            raise busy('production busy; transfer paused')
        shutil.copyfile(source, destination)
        sha = E.digest(destination)
        assert destination.stat().st_size == size and (expected is None or expected == sha)
        return {'file': str(destination), 'source': source, 'size': size, 'sha256': sha}
    return copy


def r2_module(tmp_path: Path) -> ModuleType:
    """decode_copy exactly as committed on offload/d2-close-r2 (under S33 check)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    text = subprocess.run(['git', '-C', str(REPO), 'show', R2_REVISION + ':offload/dspark/d2/decode_copy.py'],
                          env=env, check=True, capture_output=True, text=True).stdout
    path = tmp_path / 'r2_decode_copy.py'
    path.write_text(text)
    spec = importlib.util.spec_from_file_location('s34_r2_decode_copy', path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert 'pause_policy' in text and hasattr(module, 'PauseBudget')
    return module


def served_inputs(tmp_path: Path, root: Path, prompts: list[dict[str, object]]) -> tuple[Path, Path, dict[str, str]]:
    """Non-synthetic prompt manifest + sequence context whose identity/evidence files sit at ORIGINAL absolute
    paths (as on Controller), copied into root/inputs and then removed, so only explicit maps can resolve them."""
    original = tmp_path / 'controller-original'
    identity, evidence = original / 'identity', original / 'evidence'
    identity.mkdir(parents=True)
    evidence.mkdir()
    context, manifest_inputs = context_fixture(identity)
    receipt = evidence / 'target-receipt.json'
    context['mode'] = 'decode_capture'
    context['decode_capture_target'] = context.pop('replay_target')
    target = {k: v for k, v in context['target'].items() if k != 'receipt_sha256'}
    context['generator'] = {'synthetic_probe': True}
    receipt.write_text(json.dumps({'schema': 'd2.pass2-attestation.v1',
        'target': target, 'decode_capture_target': target, 'sampling': context['sampling'],
        'formatting': context['formatting'], 'generator': context['generator'],
        'identity': {label: {'sha256': context[label + '_sha256']}
                     for label in ('tokenizer', 'encoder', 'chat_template')}}))
    sha = E.digest(receipt)
    for key in ('target', 'decode_capture_target'):
        context[key]['receipt_sha256'] = sha
    context['evidence_receipts'] = [{'path': str(receipt), 'sha256': sha}]
    counts: dict[str, dict[str, int]] = {}
    for p in prompts:
        counts.setdefault(str(p['split']), {})[str(p['family'])] = 1
    manifest = {'schema': 'd2.prompts.v1', 'n_prompts': len(prompts), 'counts': counts, 'excluded_pids': {},
                'prompts': [dict(p, source='courtlistener') for p in prompts], **manifest_inputs}
    inputs = root / 'inputs'
    inputs.mkdir(exist_ok=True)
    shutil.copytree(identity, inputs / 'identity')
    shutil.copytree(evidence, inputs / 'evidence')
    (inputs / 'prompt-manifest.json').write_text(json.dumps(manifest))
    (inputs / 'sequence-context.json').write_text(json.dumps(context))
    shutil.rmtree(original)
    maps = {str(identity): 'inputs/identity', str(evidence): 'inputs/evidence'}
    return inputs / 'prompt-manifest.json', inputs / 'sequence-context.json', maps


def copied(tmp_path: Path, monkeypatch, module: ModuleType = COPY, transfer_copy=None,
           **copy_kwargs) -> tuple[Path, list[dict[str, object]]]:
    forbid_hosts(monkeypatch, module)
    monkeypatch.setattr(module.transfer, 'copy', transfer_copy or fake_transfer())
    capture, state, prompts = source_tree(tmp_path)
    Local = local_remote(module)
    root = tmp_path / 'decode-s34'
    module.frozen_copy(Local(True), Local(False), str(capture), str(state), root, NONCE, **copy_kwargs)
    shutil.rmtree(tmp_path / 'source')
    (root / 'inputs').mkdir()
    (root / 'inputs/R6-smoke.json').write_text('{"synthetic":true}')
    monkeypatch.setenv('D2_COORDINATOR', A.COORDINATOR)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    return root, prompts


def run_served(root: Path, pm: Path, ctx: Path, maps: dict[str, str]) -> dict[str, object]:
    smoke = root / 'inputs/R6-smoke.json'
    with patch.object(E, 'smoke'):
        return A.run(root, smoke, E.digest(smoke), pm, ctx, maps)


# ------------------------------------------------------------------------------------- item 1: served path end to end
def test_served_path_end_to_end_binds_receipt_context_and_relocation(tmp_path, monkeypatch):
    """The lane rehearsal uses a SYNTHETIC prompt manifest, which skips context_ready/validate_context and the
    'served decode requires assembly' gate. This drives the non-synthetic branch the real data takes."""
    root, prompts = copied(tmp_path, monkeypatch)
    pm, ctx, maps = served_inputs(tmp_path, root, prompts)
    result = run_served(root, pm, ctx, maps)
    copy_sha = E.digest(root / 'COPY_RECEIPT.json')
    manifest = DD.obj(root / 'shards/MANIFEST.json')
    decode = DD.obj(root / 'decode-plan/DECODE.json')
    assert result['synthetic'] is False and result['g1a_pass'] is True and result['pilot_cleared'] is False
    assert result['copy_receipt_sha256'] == decode['copy_receipt_sha256'] == copy_sha
    assert manifest['replay']['copy_receipt_sha256'] == copy_sha
    assert manifest['replay']['decode_sha256'] == E.digest(root / 'decode-plan/DECODE.json')
    assert manifest['generation']['sequence_context_sha256'] == E.digest(ctx)
    assert decode['relocation']['mappings'] == maps == DD.mapping(result['relocation'])['mappings']
    assert manifest['checks']['sequences'] == 4 and manifest['checks']['skipped_sequences'] == 0
    log = (root / 'G1A.log').read_text()
    verify, end = json.JSONDecoder().raw_decode(log)
    assert verify['errors'] == 0 and json.loads(log[end:])['pass'] is True  # cmd_verify, then stride-1 plumbing
    # Every shard sidecar carries the served sequence audit (not the synthetic placeholder).
    for split in manifest['splits'].values():
        for f in split['files']:
            side = json.loads((root / 'shards' / f['sidecar']).read_text())
            for row in side['sequences'] if 'sequences' in side else [side]:
                audit = row.get('sequence_audit', {})
                assert audit.get('schema') == 'd2.sequence-audit.v1', audit.get('schema')


def test_served_path_refuses_unmapped_identity_and_changed_evidence(tmp_path, monkeypatch):
    root, prompts = copied(tmp_path, monkeypatch)
    pm, ctx, maps = served_inputs(tmp_path, root, prompts)
    only_identity = {k: v for k, v in maps.items() if v == 'inputs/identity'}
    with pytest.raises(SystemExit, match='sequence provenance refused'):
        run_served(root, pm, ctx, only_identity)
    assert not (root / 'shards').exists() and not (root / 'ASSEMBLED.json').exists()


def test_served_path_refuses_changed_evidence(tmp_path, monkeypatch):
    root, prompts = copied(tmp_path, monkeypatch)
    pm, ctx, maps = served_inputs(tmp_path, root, prompts)
    (root / 'inputs/evidence/target-receipt.json').write_text('{"changed": true}')
    with pytest.raises(SystemExit, match='evidence receipt hash differs'):
        run_served(root, pm, ctx, maps)
    assert not (root / 'ASSEMBLED.json').exists()


def test_served_build_without_assembly_root_refused(tmp_path, monkeypatch):
    """A decode plan prepared without --assembly-root (no COPY_RECEIPT binding) cannot become served shards."""
    root, prompts = copied(tmp_path, monkeypatch)
    pm, ctx, maps = served_inputs(tmp_path, root, prompts)
    smoke = root / 'inputs/R6-smoke.json'
    with patch.object(E, 'smoke'):
        E.export(root / 'generator', root / 'export', smoke)
        DD.prepare(root / 'export', root / 'capture', root / 'decode-plan', 'generation_verified',
                   root / 'export/VERIFIED.json')
        with pytest.raises((SystemExit, FileNotFoundError, ValueError)):
            D.cmd_build(argparse.Namespace(replay=str(root / 'decode-plan'), out=str(root / 'shards'),
                prompt_manifest=str(pm), sequence_context=str(ctx), taps_source='decode_capture', splits='',
                allow_skips=False, equiv_gate='', override_equiv_gate='', override_plumbing=''))
    assert not (root / 'shards').exists()


def test_real_smoke_gate_not_bypassed_by_assembly(tmp_path, monkeypatch):
    """Without the test seam, the copied synthetic smoke is refused before any output is created."""
    root, prompts = copied(tmp_path, monkeypatch)
    pm, ctx, maps = served_inputs(tmp_path, root, prompts)
    smoke = root / 'inputs/R6-smoke.json'
    with pytest.raises(ValueError, match='real image-bound smoke'):
        A.run(root, smoke, E.digest(smoke), pm, ctx, maps)
    assert not (root / 'export').exists() and not (root / 'ASSEMBLED.json').exists()


def test_execute_identity_rechecked_inside_python(tmp_path, monkeypatch):
    root, prompts = copied(tmp_path, monkeypatch)
    pm, ctx, maps = served_inputs(tmp_path, root, prompts)
    monkeypatch.setenv('D2_COORDINATOR', 'Deepseek/someone-else')
    with pytest.raises(ValueError, match='coordinator execution only'):
        run_served(root, pm, ctx, maps)
    monkeypatch.setenv('D2_COORDINATOR', A.COORDINATOR)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError, match='CUDA_VISIBLE_DEVICES'):
        run_served(root, pm, ctx, maps)


# ------------------------------------------------------------------------------------ item 1: copy-binding refusals
def prepared(tmp_path, monkeypatch):
    root, prompts = copied(tmp_path, monkeypatch)
    smoke = root / 'inputs/R6-smoke.json'
    with patch.object(E, 'smoke'):
        E.export(root / 'generator', root / 'export', smoke)
        DD.prepare(root / 'export', root / 'capture', root / 'decode-plan', 'generation_verified',
                   root / 'export/VERIFIED.json', assembly_root=root)
    return root


@pytest.mark.parametrize('bad', ['extra_capture_file', 'generator_changed', 'receipt_replaced',
                                 'flush_nonce', 'stray_partial'])
def test_build_refuses_after_prepare(tmp_path, monkeypatch, bad):
    root = prepared(tmp_path, monkeypatch)
    receipt = root / 'COPY_RECEIPT.json'
    if bad == 'extra_capture_file':
        (root / 'capture/p1/extra.bin').write_bytes(b'x')
    elif bad == 'generator_changed':
        p = root / 'generator/seqs/p2.json'
        p.write_bytes(p.read_bytes().replace(b'"partial"', b'"stopped"'))
    elif bad == 'receipt_replaced':
        raw = json.loads(receipt.read_text())
        raw['coordinator_note'] = 'a different but internally valid receipt'
        receipt.write_text(json.dumps(raw))
        assert verify_copy(root)  # still a valid receipt for the tree, but not the one DECODE.json bound
    elif bad == 'flush_nonce':
        # Consistent receipt (hashes updated) whose nonce differs from FLUSH/FLUSHED.
        raw = json.loads(receipt.read_text())
        raw['nonce'] = 'another-nonce'
        receipt.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match='flush nonce'):
            verify_copy(root)
    else:
        (root / 'capture/p0/taps.bin.s2-partial').write_bytes(b'partial')
    with patch.object(E, 'smoke'), pytest.raises((ValueError, FileNotFoundError), match='copy|receipt'):
        D.tree_taps_source(str(root / 'decode-plan'))


def test_wrong_flush_nonce_refused_at_prepare(tmp_path, monkeypatch):
    root, _ = copied(tmp_path, monkeypatch)
    receipt = root / 'COPY_RECEIPT.json'
    raw = json.loads(receipt.read_text())
    raw['nonce'] = 'another-nonce'
    receipt.write_text(json.dumps(raw))
    smoke = root / 'inputs/R6-smoke.json'
    with patch.object(E, 'smoke'):
        E.export(root / 'generator', root / 'export', smoke)
        with pytest.raises(ValueError, match='flush nonce'):
            DD.prepare(root / 'export', root / 'capture', root / 'decode-plan', 'generation_verified',
                       root / 'export/VERIFIED.json', assembly_root=root)


# ------------------------------------------------------------------------------- item 1: relocation edge cases (obs)
def test_obs_decode_json_captures_not_pinned_to_receipt_tree(tmp_path, monkeypatch):
    """S34 O5 regression: reject an unbound capture tree substituted after prepare."""
    root = prepared(tmp_path, monkeypatch)
    shutil.copytree(root / 'capture', root / 'unbound-capture')
    decode = root / 'decode-plan/DECODE.json'
    meta = json.loads(decode.read_text())
    meta['captures'] = str(root / 'unbound-capture')
    decode.write_text(json.dumps(meta))
    with patch.object(E, 'smoke'), pytest.raises(ValueError, match='receipt-bound copy tree'):
        DD.decode_identity(root / 'decode-plan', full_copy=True)


def test_obs_root_mapping_and_relative_alias(tmp_path):
    """S34 O6 regression: root maps and relative in-root symlink aliases refuse."""
    root = tmp_path / 'root'
    (root / 'real').mkdir(parents=True)
    (root / 'real/file').write_text('x')
    with pytest.raises(ValueError, match='source prefix'):
        AssemblyPaths(root, {'/': 'real'})
    (root / 'alias').symlink_to(root / 'real')
    paths = AssemblyPaths(root, {})
    with pytest.raises(ValueError, match='symlink'):
        paths.inside(root / 'alias/file')
    relative = Path(os.path.relpath(root / 'alias/file'))
    with pytest.raises(ValueError, match='symlink'):
        paths.inside(relative)


def test_obs_g1a_failure_report_is_discarded(tmp_path, monkeypatch, capsys):
    """S34 O4 regression: preserve the failed stride-1 G1(a) report in G1A.log."""
    from d2 import g1 as G
    root, prompts = copied(tmp_path, monkeypatch)
    pm, ctx, maps = served_inputs(tmp_path, root, prompts)
    real = G.S.ShardCaptureSet

    class Failing(real):  # type: ignore[misc, valid-type]
        def __init__(self, files, stride=16):
            super().__init__(files, stride=stride)
            if stride == 1:
                self.plumbing = ['probe-injected stride-1 plumbing error']
    monkeypatch.setattr(G.S, 'ShardCaptureSet', Failing)
    with pytest.raises(SystemExit) as info:
        run_served(root, pm, ctx, maps)
    assert info.value.code == 1
    assert 'probe-injected' in (root / 'G1A.log').read_text()
    assert not (root / 'ASSEMBLED.json').exists()
    assert 'probe-injected' not in capsys.readouterr().out


# ------------------------------------------------------------------------------- item 2: R2 decode_copy receipt format
def test_r2_receipt_with_pause_events_assembles(tmp_path, monkeypatch):
    """A copy made by eb55f82's decode_copy, including one demand pause and restart, verifies and assembles."""
    R2 = r2_module(tmp_path)
    clock = iter(float(i) for i in range(10_000))
    budget = R2.PauseBudget(cap_seconds=3600, idle_seconds=1, check=lambda: None, memory=lambda: None,
                            clock=lambda: next(clock), sleep=lambda s: None)
    root, prompts = copied(tmp_path, monkeypatch, R2, fake_transfer({'taps.bin'}, R2.ProductionBusy), pause=budget)
    raw = json.loads((root / 'COPY_RECEIPT.json').read_text())
    assert raw['schema'] == 'd2.decode-copy.v1' and raw['pause_policy']['events'], raw.get('pause_policy')
    assert not list(root.rglob('*.s2-partial'))
    pm, ctx, maps = served_inputs(tmp_path, root, prompts)
    result = run_served(root, pm, ctx, maps)
    assert result['copy_receipt_sha256'] == E.digest(root / 'COPY_RECEIPT.json')
    assert result['g1a_pass'] is True and result['synthetic'] is False


def test_r2_receipt_is_not_weakened(tmp_path, monkeypatch):
    """The additive R2 fields do not relax any v1 requirement: removing a required field still refuses."""
    R2 = r2_module(tmp_path)
    budget = R2.PauseBudget(check=lambda: None, memory=lambda: None)
    root, _ = copied(tmp_path, monkeypatch, R2, pause=budget)
    receipt = root / 'COPY_RECEIPT.json'
    good = receipt.read_text()
    assert verify_copy(root) == hashlib.sha256(good.encode()).hexdigest()
    for key, value in [('frozen', False), ('schema', 'd2.decode-copy.v2'), ('nonce', None)]:
        raw = json.loads(good)
        raw[key] = value
        receipt.write_text(json.dumps(raw))
        with pytest.raises(ValueError):
            verify_copy(root)
    raw = json.loads(good)
    raw['files'] = [f for f in raw['files'] if f['relative'] != 'p0/ids.bin']
    receipt.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        verify_copy(root)


# ----------------------------------------------------------------------------------------------- item 4: operational
SCRIPT = Path(__file__).resolve().parents[1] / 'assemble.sh'


def fake_python(tmp_path: Path) -> tuple[Path, Path]:
    record = tmp_path / 'argv-env.json'
    fake = tmp_path / 'python'
    fake.write_text('#!/bin/bash\n' + sys.executable + ' -c "import json,os,sys; json.dump({\'argv\': sys.argv[2:], '
                    '\'cuda\': os.environ.get(\'CUDA_VISIBLE_DEVICES\'), \'umask\': oct(os.umask(0)), '
                    '\'pythonpath\': os.environ.get(\'PYTHONPATH\'), \'coordinator\': os.environ.get(\'D2_COORDINATOR\')}, '
                    'open(sys.argv[1], \'w\'))" ' + str(record) + ' "$@"\n')
    fake.chmod(0o700)
    return fake, record


@pytest.mark.parametrize('identity', [None, '', 'Deepseek/coordinato', 'deepseek/coordinator', 'Deepseek/coordinator '])
def test_wrapper_execute_refuses_other_identities(tmp_path, identity):
    fake, record = fake_python(tmp_path)
    env = {k: v for k, v in os.environ.items() if k != 'D2_COORDINATOR'}
    env['D2_ASSEMBLY_PYTHON'] = str(fake)
    if identity is not None:
        env['D2_COORDINATOR'] = identity
    result = subprocess.run(['bash', str(SCRIPT), '--execute', '--root', str(tmp_path)], env=env,
                            text=True, capture_output=True)
    assert result.returncode == 2 and not record.exists()


def test_wrapper_execute_pins_cpu_umask_and_module(tmp_path):
    fake, record = fake_python(tmp_path)
    env = dict(os.environ, D2_ASSEMBLY_PYTHON=str(fake), D2_COORDINATOR=A.COORDINATOR, CUDA_VISIBLE_DEVICES='0')
    subprocess.run(['bash', str(SCRIPT), '--execute', '--root', str(tmp_path / 'r'), '--smoke-sha256', 'x'],
                   env=env, text=True, capture_output=True, check=True)
    seen = json.loads(record.read_text())
    assert seen['argv'] == ['-m', 'd2.assemble', '--root', str(tmp_path / 'r'), '--smoke-sha256', 'x']
    assert seen['cuda'] == '' and seen['umask'] == '0o77' and seen['coordinator'] == A.COORDINATOR
    assert seen['pythonpath'].split(':')[0].endswith('offload/dspark')


def test_wrapper_execute_only_as_first_argument(tmp_path):
    """--execute elsewhere stays dry (prints only) even for the coordinator."""
    fake, record = fake_python(tmp_path)
    env = dict(os.environ, D2_ASSEMBLY_PYTHON=str(fake), D2_COORDINATOR=A.COORDINATOR)
    result = subprocess.run(['bash', str(SCRIPT), '--root', str(tmp_path), '--execute'], env=env,
                            text=True, capture_output=True, check=True)
    assert result.stdout.startswith('DRY CPU:') and not record.exists()


def test_assembly_writes_only_inside_root(tmp_path, monkeypatch):
    """Run the served assembly in a child with empty HOME/TMPDIR/XDG cache/cwd; nothing appears outside root."""
    root, prompts = copied(tmp_path, monkeypatch)
    pm, ctx, maps = served_inputs(tmp_path, root, prompts)
    sandbox = {name: tmp_path / name for name in ('home', 'tmp', 'cache', 'cwd')}
    for path in sandbox.values():
        path.mkdir()
    before = {p for p in tmp_path.rglob('*') if not p.is_relative_to(root)}
    in_root = {p for p in root.rglob('*')}
    child = ('import json,sys\nfrom pathlib import Path\nfrom unittest.mock import patch\n'
             'from d2 import assemble as A, generation_export as E\n'
             'root, pm, ctx, maps = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), json.loads(sys.argv[4])\n'
             'smoke = root / "inputs/R6-smoke.json"\n'
             'with patch.object(E, "smoke"):\n'
             '    A.run(root, smoke, E.digest(smoke), pm, ctx, maps)\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(HOME=str(sandbox['home']), TMPDIR=str(sandbox['tmp']), XDG_CACHE_HOME=str(sandbox['cache']),
               PYTHONPATH=f'{REPO / "offload/dspark"}:{REPO}', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
               D2_COORDINATOR=A.COORDINATOR, OMP_NUM_THREADS='2')
    subprocess.run([sys.executable, '-c', child, str(root), str(pm), str(ctx), json.dumps(maps)], env=env,
                   cwd=sandbox['cwd'], check=True, capture_output=True)
    assert {p for p in tmp_path.rglob('*') if not p.is_relative_to(root)} == before
    assert all(not any(path.iterdir()) for path in sandbox.values())
    created = {str(p.relative_to(root)).split('/')[0] for p in set(root.rglob('*')) - in_root}
    assert created == {'export', 'decode-plan', 'shards', 'G1A.log', 'AGGREGATE.json', 'ASSEMBLED.json'}
    assert not (root / 'shards.partial').exists()


def test_obs_map_prefix_containing_root_fails_closed(tmp_path, monkeypatch):
    """S34 O3 regression: reject an ancestor source map before export or path resolution."""
    root, prompts = copied(tmp_path, monkeypatch)
    pm, ctx, maps = served_inputs(tmp_path, root, prompts)
    (root / 'inputs/parent').mkdir()
    broad = dict(maps, **{str(tmp_path): 'inputs/parent'})
    with pytest.raises(ValueError, match='source prefix'):
        run_served(root, pm, ctx, broad)
    assert not (root / 'ASSEMBLED.json').exists()
